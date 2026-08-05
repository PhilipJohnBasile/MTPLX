"""Fast-path AR optimizations for the LiquidAI LFM2 architecture.

LFM2 (``model_type == "lfm2"``) is a dense hybrid: most layers are a
double-gated *short convolution* token mixer (``conv_L_cache == 3``), the rest
are GQA attention.  At batch-1 decode the model is GPU-bound, and the dispatch
census shows the ShortConv decode path spends a disproportionate number of
kernels on the sliding-window bookkeeping — ``concatenate([state, Bx])`` +
``pad`` + ``conv1d`` + a slice to re-store the window — per conv layer, per
token.  For the ``L_cache == 3`` decode step (sequence length 1) that whole
sequence collapses to a fused 3-tap FIR:

    conv_out = s0*w0 + s1*w1 + Bx*w2        # w_k = conv.weight[:, k, 0]
    new_state = stack([s1, Bx])             # the last L_cache-1 taps

which is bit-exact with the stock ``nn.Conv1d`` window and removes the
concat/pad/conv1d/slice dispatches on the decode hot path.  Prefill (sequence
length > 1) and any masked step fall back to the stock implementation, so the
optimization is decode-only and changes no numerics.

Install with :func:`install_lfm2_fast` after the model is loaded.  It is a
no-op for non-LFM2 models.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def is_lfm2_config(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    mt = str(config.get("model_type", "")).lower()
    if mt == "lfm2":
        return True
    # LFM2-VL and future wrappers nest the text config.
    text = config.get("text_config")
    if isinstance(text, dict) and str(text.get("model_type", "")).lower() == "lfm2":
        return True
    archs = config.get("architectures") or []
    return any("lfm2" in str(a).lower() for a in archs)


def _bind_fast_shortconv(short_conv: Any) -> bool:
    """Bind the fused 3-tap decode path onto one ShortConv instance.

    Returns True if the fast path was installed (kernel_size == 3, depthwise).
    """
    conv = getattr(short_conv, "conv", None)
    if conv is None or not hasattr(conv, "weight"):
        return False
    w = conv.weight  # depthwise Conv1d weight: [channels, kernel_size, 1]
    if w.ndim != 3 or w.shape[-1] != 1:
        return False
    l_cache = int(getattr(short_conv, "L_cache", w.shape[1]))
    if l_cache != 3 or w.shape[1] != 3:
        return False  # only the standard LFM2 3-tap window is fused

    taps = w[:, :, 0]  # [channels, 3]
    short_conv._fast_w0 = taps[:, 0][None, :]  # [1, channels]
    short_conv._fast_w1 = taps[:, 1][None, :]
    short_conv._fast_w2 = taps[:, 2][None, :]
    bias = None
    if getattr(short_conv, "bias", False) and getattr(conv, "bias", None) is not None:
        bias = conv.bias[None, :]
    short_conv._fast_bias = bias
    mx.eval(short_conv._fast_w0, short_conv._fast_w1, short_conv._fast_w2)

    stock_call = type(short_conv).__call__

    def fast_call(self, x, mask=None, cache=None):
        # Decode-only fast path: single token, no mask, live cache.
        if cache is None or mask is not None or x.shape[1] != 1:
            return stock_call(self, x, mask, cache)
        BCx = self.in_proj(x)
        d = BCx.shape[-1] // 3
        B = BCx[..., :d]
        C = BCx[..., d : 2 * d]
        xx = BCx[..., 2 * d :]
        bx = (B * xx)[:, 0, :]  # [1, D]
        state = cache[0]
        if state is None:
            s0 = mx.zeros_like(bx)
            s1 = mx.zeros_like(bx)
        else:
            s0 = state[:, 0, :]
            s1 = state[:, 1, :]
        conv_out = s0 * self._fast_w0 + s1 * self._fast_w1 + bx * self._fast_w2
        if self._fast_bias is not None:
            conv_out = conv_out + self._fast_bias
        cache[0] = mx.stack([s1, bx], axis=1)  # keep last L_cache-1 taps
        cache.advance(1)
        y = (C[:, 0, :] * conv_out)[:, None, :]
        return self.out_proj(y)

    # Bind as an instance method so only patched modules take the fast path.
    short_conv.__class__ = _fast_shortconv_subclass(type(short_conv), fast_call)
    return True


_SUBCLASS_CACHE: dict[type, type] = {}


def _fast_shortconv_subclass(base: type, fast_call) -> type:
    cached = _SUBCLASS_CACHE.get(base)
    if cached is not None:
        return cached
    sub = type(f"Fast{base.__name__}", (base,), {"__call__": fast_call})
    _SUBCLASS_CACHE[base] = sub
    return sub


def install_lfm2_fast(model: Any) -> dict[str, Any]:
    """Apply LFM2 decode fast-paths to a loaded model (bit-exact, decode-only).

    Safe to call on any model; returns a report dict. No-op unless the model
    exposes LFM2-style ShortConv layers.
    """
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        return {"applied": False, "reason": "no layers"}
    patched = 0
    conv_total = 0
    for layer in layers:
        sc = getattr(layer, "conv", None)
        if sc is None:
            continue
        conv_total += 1
        if _bind_fast_shortconv(sc):
            patched += 1
    return {
        "applied": patched > 0,
        "shortconv_layers": conv_total,
        "shortconv_fast": patched,
    }
