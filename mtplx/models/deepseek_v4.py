"""Native MLX loader/backend for DeepSeek-V4-Flash (``model_type: deepseek_v4``).

This is a from-scratch port, not a config tweak: no ``deepseek_v4`` loader exists
in mlx-lm.  The scaffold reuses the DeepSeek-V3/V3.2 MoE + shared-expert shape and
the ``noaux_tc`` routing idea, but V4 adds four pieces of genuinely new math over
V3.2, all transcribed here from the authoritative reference
``deepseek-ai/DeepSeek-V4-Flash/inference/model.py`` + ``inference/kernel.py``:

1. **Hyper-Connections (HCA)** — the residual stream is replaced by ``hc_mult=4``
   parallel copies.  Each block runs ``hc_pre`` (collapse 4->1 via Sinkhorn-derived
   pre-weights) around attn/ffn, then ``hc_post`` (expand 1->4 and re-mix with the
   residual copies through a doubly-stochastic ``comb`` matrix).  The ``comb`` matrix
   is produced by ``hc_split_sinkhorn``: one row-softmax + one column-normalise, then
   ``hc_sinkhorn_iters-1`` (=19) further row/column normalisation passes.  ``hc_eps``
   (1e-6) guards every division.  Reference: ``Block.hc_pre/hc_post`` (model.py
   L673-699) and ``hc_split_sinkhorn_kernel`` (kernel.py L371-427).

2. **Compressed Sparse Attention (CSA)** — for layers with ``compress_ratio != 0``,
   a ``Compressor`` builds a second, compressed KV cache by learned gated pooling of
   ``compress_ratio`` consecutive tokens (softmax over a learned gate + absolute
   position embedding ``ape``).  ``compress_ratio==4`` layers pool overlapping
   windows and own an ``Indexer`` that scores compressed positions and returns the
   top-``index_topk`` (512) to attend; ``compress_ratio==128`` layers pool
   non-overlapping windows and use a deterministic strided index.  These layers rope
   with ``compress_rope_theta`` (160000) under YaRN; ``compress_ratio==0`` layers are
   pure sliding-window (``window_size=128``) with base ``rope_theta`` and no YaRN.
   Reference: ``Compressor`` (L279-377), ``Indexer`` (L380-433), ``Attention`` (L436-543).

3. **Output-LoRA (o-LoRA)** — where V3 low-ranks only q and kv, V4 also low-ranks the
   output projection *in groups*.  The ``n_heads*head_dim = 32768`` attention output
   is split into ``o_groups=8`` chunks of 4096; each chunk is projected to
   ``o_lora_rank=1024`` by its own matrix (grouped/block matmul ``wo_a``), the 8
   results concatenate to 8192, then ``wo_b`` maps 8192->dim.  Reference:
   ``Attention.forward`` L536-542.

4. **Hash layers** — the first ``num_hash_layers=3`` layers route each token to a
   fixed expert set determined by token id (``gate.tid2eid`` lookup) instead of
   score-based top-k.  Reference: ``Gate`` (L546-584).

Attention itself is MQA-shaped MLA: ``num_key_value_heads=1``, a single 512-dim KV
latent (``head_dim=512``, ``rope_head_dim=64`` on its tail) shared across all 64
query heads, each head carrying a learned ``attn_sink`` logit.  Routing uses
``scoring_func="sqrtsoftplus"`` (softplus then sqrt) with ``routed_scaling_factor=1.5``.

Weight names mirror the reference module tree, which is exactly what the
``mlx-community/DeepSeek-V4-Flash-4bit`` checkpoint ships:
``model.layers.{i}.attn.{wq_a,wq_b,wkv,wo_a,wo_b,q_norm,kv_norm,attn_sink}``,
``...attn.compressor.{wkv,wgate,norm,ape}``, ``...attn.indexer.{wq_b,weights_proj,compressor.*}``,
``...ffn.{gate,switch_mlp,shared_experts}``, ``...{attn_hc,ffn_hc}.{fn,base,scale}``,
``model.hc_head.{fn,base,scale}``, ``model.{embed_tokens,norm}``, ``lm_head``.
Quantisation in that checkpoint is mixed: routed experts (``ffn.switch_mlp.*``) are
**mxfp4 group_size 32** (scales, no biases); everything else is **affine 4-bit
group_size 64** (weight/scales/biases).  The MTP block is dropped by the conversion.

Status:
  * The four new-math components are numerically gated against the reference
    (tests/test_deepseek_v4_new_math.py) and the WHOLE forward is gated layer-by-layer
    against a reference golden covering every layer type
    (tests/test_deepseek_v4_parity.py) — this is the prefill path.
  * The attention integrates the compressor's compressed KV (overlap ratio-4 and
    non-overlap ratio-128), the window+compressed causal mask, compress-YaRN rope and
    per-head attn_sink; it is the dense-mask equivalent of the reference sparse_attn
    + topk_idxs gather.
  * The ratio-4 :class:`Indexer` top-k filter is wired in both directions
    (tests/test_deepseek_v4_indexer.py): it scores every compressed row against the
    query and masks all but the top ``index_topk``, so the backend is correct past
    ~``index_topk*ratio`` tokens of context, where dense-over-compressed stops being
    the reference's computation.  Below that threshold the filter provably selects
    every causal row, and the scoring path is skipped outright, leaving the short
    regime bit-identical.
  * Streaming decode runs off ``DeepseekV4Cache`` (``make_cache``): a sliding-window
    per-position KV buffer, the growing compressed-KV rows, the compressor's
    in-progress window frontier, and the same pair again for the indexer's own
    compressor lane.  Prompt-prefill + token-by-token decode reproduces the one-shot
    forward (tests/test_deepseek_v4_decode.py), including partial prompt windows,
    both compress ratios, context past ``window_size``, and crossing ``index_topk``
    mid-generation.  The state machine is adapted from ds4.c (antirez/DwarfStar4,
    MIT), which carries ``index_state_kv``/``index_comp_kv`` beside the attention
    lane's for exactly this reason.
  * Dropped on purpose: the reference's inference-time QAT emulation (FP8 on the
    attention compressor's rows, FP4 on the indexer's q and rows).  It is noise
    injection, not model math — except that in the indexer it perturbs a *discrete*
    top-k boundary, so selections near the cut can differ from the reference.  The
    Hadamard rotation that precedes the FP4 step is implemented (it is graph, not
    noise), and is a no-op for selection on its own; see :class:`Indexer`.
  * ``deepseek-v4`` is registered in ``mtplx/backends/registry.py`` so ``mtplx serve``
    resolves the load path.

Provenance: reference files fetched read-only from
``https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash`` (inference/model.py,
inference/kernel.py, config.json) and
``https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit`` (config.json,
model.safetensors.index.json).  The reference GPU kernels require CUDA/tilelang and
cannot run on this box; the M2 oracle is a faithful transcription of their documented
elementwise math (verified elementwise, not by running the shipped kernel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.switch_layers import SwitchGLU


# Default per-layer compress ratios for DeepSeek-V4-Flash (43 body layers; the
# 44th entry is the dropped MTP layer).  0 = pure sliding-window; 4 = overlapping
# compressor + indexer; 128 = non-overlapping compressor + strided index.
_DEFAULT_COMPRESS_RATIOS = (
    [0, 0]
    + [4, 128] * 20
    + [4, 0]
)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_hash_layers: int = 3
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    # moe
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    topk_method: str = "noaux_tc"
    swiglu_limit: float = 10.0
    # attention (MQA-shaped MLA)
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    window_size: int = 128
    sliding_window: int = 128
    # index / compressed-sparse attention
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    compress_ratios: List[int] = field(default_factory=lambda: list(_DEFAULT_COMPRESS_RATIOS))
    compress_rope_theta: float = 160000.0
    # hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # norm / rope / yarn
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 1048576
    rope_scaling: Optional[dict] = None
    # yarn (flattened from rope_scaling for convenience; overridden in __post_init__)
    original_seq_len: int = 65536
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        # Accept the HF rope_scaling block and mirror it into the flat YaRN fields
        # the reference precompute uses.
        rs = self.rope_scaling or {}
        if rs:
            self.original_seq_len = int(
                rs.get("original_max_position_embeddings", self.original_seq_len)
            )
            self.rope_factor = float(rs.get("factor", self.rope_factor))
            self.beta_fast = int(rs.get("beta_fast", self.beta_fast))
            self.beta_slow = int(rs.get("beta_slow", self.beta_slow))
        # window_size / sliding_window are the same knob under two names.
        self.window_size = int(self.sliding_window or self.window_size)


# ---------------------------------------------------------------------------
# RoPE (YaRN, interleaved / "traditional") — matches reference precompute_freqs_cis
# + apply_rotary_emb, which rope only the last ``rope_head_dim`` dims of q/kv as
# complex pairs (x0+ix1, x2+ix3, ...).
# ---------------------------------------------------------------------------
def _yarn_inv_freq(
    dim: int,
    base: float,
    original_seq_len: int,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> mx.array:
    """Per-(pair) inverse frequencies with the reference's YaRN interpolation ramp.

    Mirrors ``precompute_freqs_cis`` (model.py L199-229): standard inv-freq, then when
    ``original_seq_len > 0`` a smooth linear ramp blends the ``/factor`` (interpolated)
    and un-interpolated frequencies between the beta_fast/beta_slow correction dims.
    """
    half = dim // 2
    freqs = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    if original_seq_len and original_seq_len > 0:
        def correction_dim(num_rot):
            return dim * math.log(original_seq_len / (num_rot * 2 * math.pi)) / (
                2 * math.log(base)
            )

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = (mx.arange(half, dtype=mx.float32) - low) / (high - low)
        ramp = mx.clip(ramp, 0.0, 1.0)
        smooth = 1.0 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs  # [half]


def _hadamard_rotate(x: mx.array) -> mx.array:
    """Normalised Walsh-Hadamard rotation of the last axis (power-of-two width).

    Reference ``rotate_activation`` (model.py L247-251) calls
    ``fast_hadamard_transform.hadamard_transform(x, scale=d**-0.5)``; this is the same
    map written as the in-place butterfly ds4.c uses
    (``dsv4_hadamard128_inplace_cpu``, antirez/DwarfStar4, MIT), which is bit-for-bit
    the reference's own accumulation order.

    ``H/sqrt(d)`` is **orthogonal**, so it leaves every ``q·k`` the indexer forms
    invariant — see :class:`Indexer` for why it is applied anyway.
    """
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"Hadamard rotation needs a power-of-two width, got {n}")
    y = x.reshape(-1, n)
    stride = 1
    while stride < n:
        y = y.reshape(-1, n // (2 * stride), 2, stride)
        a = y[:, :, 0]
        b = y[:, :, 1]
        y = mx.stack([a + b, a - b], axis=2).reshape(-1, n)
        stride *= 2
    return (y * (n ** -0.5)).reshape(x.shape)


def _topk_mask(key: mx.array, k_row: mx.array, k_max: int) -> mx.array:
    """``True`` for the ``k_row`` largest entries of ``key`` along the last axis.

    ``key`` is ``[..., n]``; ``k_row`` is a *per-row* count broadcastable to
    ``[..., 1]``; ``k_max`` is any upper bound on it (only used to shrink the sort).

    Ties are broken toward the **lowest index**, which is exactly what ds4.c's
    selection does (``indexer_allowed_decode_one``: scan ascending, take over on a
    strict ``>``).  That matters here: the score of a compressed row is a sum of
    ReLU'd dot products, so rows whose every head is negative all score an exact 0
    and collide.  Without a fixed tie-break the one-shot and streaming paths — whose
    score rows have different lengths — could resolve such a collision differently
    and select different rows.

    ``k_row == 0`` selects nothing; ``k_row >= n`` selects everything.
    """
    n = key.shape[-1]
    if k_max <= 0:
        return mx.zeros(key.shape, dtype=mx.bool_)
    if k_max >= n:
        ranked = mx.sort(key, axis=-1)[..., ::-1]
    else:
        ranked = mx.sort(mx.topk(key, k_max, axis=-1), axis=-1)[..., ::-1]
    kth = mx.clip(k_row - 1, 0, ranked.shape[-1] - 1)
    thr = mx.take_along_axis(ranked, kth, axis=-1)          # k_row-th largest
    gt = key > thr
    eq = key == thr
    n_gt = mx.sum(gt.astype(mx.int32), axis=-1, keepdims=True)
    tie_rank = mx.cumsum(eq.astype(mx.int32), axis=-1) - 1  # rank among equals, index order
    return gt | (eq & (tie_rank < (k_row - n_gt)))


def _apply_interleaved_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the last dim of ``x`` (size 2*half) as interleaved complex pairs.

    ``x`` is ``[..., 2*half]``; ``cos``/``sin`` are ``[..., half]`` (broadcastable).
    Pair p = (x[2p], x[2p+1]) -> (x0*cos - x1*sin, x0*sin + x1*cos), matching
    ``apply_rotary_emb`` (model.py L232-244, forward direction).  The inverse
    (de-rotation applied to the attention output) uses cos, -sin.
    """
    shape = x.shape
    x = x.reshape(*shape[:-1], shape[-1] // 2, 2)
    x0 = x[..., 0]
    x1 = x[..., 1]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    out = mx.stack([r0, r1], axis=-1)
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# Hyper-Connections
# ---------------------------------------------------------------------------
def hc_split_sinkhorn(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc: int,
    iters: int,
    eps: float,
):
    """Transcription of ``hc_split_sinkhorn_kernel`` (kernel.py L371-427).

    ``mixes`` is ``[..., (2+hc)*hc]``.  Returns ``(pre, post, comb)`` with shapes
    ``[..., hc]``, ``[..., hc]``, ``[..., hc, hc]``.  ``comb`` is made (approximately)
    doubly-stochastic by one row-softmax + column-normalise, then ``iters-1`` more
    row/column normalisation passes.
    """
    pre = mx.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * mx.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]
    comb = comb.reshape(*comb.shape[:-1], hc, hc)  # [..., j, k]

    # comb = softmax(comb, dim=-1) + eps
    comb = mx.softmax(comb, axis=-1) + eps
    # comb = comb / (comb.sum(dim=-2) + eps)   (column normalise)
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)  # row normalise
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)  # column normalise
    return pre, post, comb


class HyperConnection(nn.Module):
    """Holds a block's ``{fn, base, scale}`` HC parameters and applies pre/post.

    ``fn``: ``[(2+hc)*hc, hc*dim]``   ``base``: ``[(2+hc)*hc]``   ``scale``: ``[3]``.
    Checkpoint keys: ``model.layers.{i}.{attn_hc,ffn_hc}.{fn,base,scale}``.
    """

    def __init__(self, dim: int, hc: int, eps: float):
        super().__init__()
        self.dim = dim
        self.hc = hc
        self.eps = eps
        mix_hc = (2 + hc) * hc
        self.fn = mx.zeros((mix_hc, hc * dim))
        self.base = mx.zeros((mix_hc,))
        self.scale = mx.zeros((3,))

    def _mixes(self, x: mx.array) -> mx.array:
        # x: [..., hc, dim]
        x_flat = x.reshape(*x.shape[:-2], self.hc * self.dim).astype(mx.float32)
        rsqrt = mx.rsqrt(mx.mean(mx.square(x_flat), axis=-1, keepdims=True) + self.eps)
        return (x_flat @ self.fn.astype(mx.float32).T) * rsqrt

    def pre(self, x: mx.array):
        """Collapse the ``hc`` copies to one; return (y[..., dim], post, comb)."""
        dtype = x.dtype
        xf = x.astype(mx.float32)
        mixes = self._mixes(xf)
        pre, post, comb = hc_split_sinkhorn(
            mixes, self.scale.astype(mx.float32), self.base.astype(mx.float32),
            self.hc, self._iters, self.eps,
        )
        y = mx.sum(pre[..., None] * xf, axis=-2)  # [..., dim]
        return y.astype(dtype), post, comb

    def post(self, x: mx.array, residual: mx.array, post: mx.array, comb: mx.array):
        """Expand one -> ``hc`` copies and re-mix with the residual copies.

        ``x``: ``[..., dim]``  ``residual``: ``[..., hc, dim]``
        ``post``: ``[..., hc]``  ``comb``: ``[..., hc, hc]``  ->  ``[..., hc, dim]``.
        """
        xf = x.astype(mx.float32)
        rf = residual.astype(mx.float32)
        term = post[..., None] * xf[..., None, :]  # [..., hc, dim]
        mixed = mx.einsum("...jk,...jd->...kd", comb, rf)  # sum_j comb[j,k] res[j]
        return (term + mixed).astype(x.dtype)

    # iterations set at construction from args
    _iters: int = 20


class HeadHC(nn.Module):
    """Final head hyper-connection collapse (``ParallelHead.hc_head``, model.py L728).

    Simpler than a block HC: ``pre = sigmoid(mixes*scale + base) + eps`` (no Sinkhorn,
    no post/comb), then weighted sum over the ``hc`` copies.  ``fn``: ``[hc, hc*dim]``.
    Checkpoint keys: ``model.hc_head.{fn,base,scale}``.
    """

    def __init__(self, dim: int, hc: int, eps: float):
        super().__init__()
        self.dim = dim
        self.hc = hc
        self.eps = eps
        self.fn = mx.zeros((hc, hc * dim))
        self.base = mx.zeros((hc,))
        self.scale = mx.zeros((1,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: [..., hc, dim]
        dtype = x.dtype
        xf = x.astype(mx.float32)
        x_flat = xf.reshape(*xf.shape[:-2], self.hc * self.dim)
        rsqrt = mx.rsqrt(mx.mean(mx.square(x_flat), axis=-1, keepdims=True) + self.eps)
        mixes = (x_flat @ self.fn.astype(mx.float32).T) * rsqrt
        pre = mx.sigmoid(mixes * self.scale.astype(mx.float32) + self.base.astype(mx.float32)) + self.eps
        y = mx.sum(pre[..., None] * xf, axis=-2)
        return y.astype(dtype)


# ---------------------------------------------------------------------------
# Compressed KV pooling (CSA)
# ---------------------------------------------------------------------------
class Compressor(nn.Module):
    """Learned gated pooling of ``compress_ratio`` consecutive tokens into one
    compressed KV row (reference ``Compressor``, model.py L279-377).

    Full-prefill math (``start_pos == 0``, the path the M2/M3 gates exercise):
        kv    = wkv(x_fp32)                                  # [b,s,coff*head_dim]
        score = wgate(x_fp32)
        (drop the trailing ``s % ratio`` remainder), reshape to windows of ``ratio``,
        add the per-window absolute-position embedding ``ape``, softmax the gate over
        the window and take the gated sum; overlapping windows (ratio==4) additionally
        fold in the previous window's second half.  Then RMSNorm, rope the tail
        ``rope_head_dim`` dims with the compressor's (YaRN) frequencies.

    NOTE: the reference simulates FP8/FP4 on the pooled KV at inference (``act_quant``
    /``fp4_act_quant`` in-place).  That QAT noise is intentionally dropped in this clean
    MLX path; the divergence it introduces is quantified in M3, not hidden here.

    Two entry points share one pooling core (:meth:`_pool`): :meth:`__call__` pools a
    whole sequence from position 0 (the parity-gated path), :meth:`step` pools
    incrementally against a :class:`CompressorState` frontier for streaming decode.
    """

    def __init__(
        self, args: ModelArgs, compress_ratio: int, head_dim: int, rotate: bool = False
    ):
        super().__init__()
        self.dim = args.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        # rotate=True is the indexer's copy: the reference Hadamard-rotates its pooled
        # rows before FP4-quantising them (model.py L368-370).  Applied in _pool, so
        # the prefill and streaming paths get it from the same place.
        self.rotate = rotate
        coff = 1 + self.overlap
        self.ape = mx.zeros((compress_ratio, coff * head_dim))
        self.wkv = nn.Linear(self.dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(self.dim, coff * head_dim, bias=False)
        self.norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        # Compressor rope uses the compress theta + YaRN (reference passes the
        # compressor its own freqs_cis; window w gets position w*ratio).
        self._inv_freq = _yarn_inv_freq(
            self.rope_head_dim, args.compress_rope_theta, args.original_seq_len,
            args.rope_factor, args.beta_fast, args.beta_slow,
        )

    def _overlap_transform(
        self, t: mx.array, value: float, prev: Optional[mx.array] = None
    ) -> mx.array:
        """Reference ``overlap_transform`` (model.py L307-314).

        ``t``: ``[b, nwin, ratio, 2*d]`` -> ``[b, nwin, 2*ratio, d]``.  The first
        ``ratio`` slots of window w hold the previous window's tokens under the
        first-half (``:d``) projection (``value`` for w==0); the last ``ratio``
        slots hold the current window's tokens under the second-half (``d:``)
        projection.

        ``prev`` seeds window 0's first half from a window that was pooled in an
        earlier call (streaming decode); ``None`` is the fresh-sequence pad.
        """
        b, nwin, r, _ = t.shape
        d = self.head_dim
        cur = t[..., d:]                       # [b, nwin, ratio, d]  (current, d: half)
        prev_half = t[..., :d]                 # [b, nwin, ratio, d]  (:d half)
        if prev is None:
            seed = mx.full((b, 1, r, d), value, dtype=t.dtype)
        else:
            seed = prev[..., :d][:, None]      # [b, 1, ratio, d]
        prev_shift = mx.concatenate([seed, prev_half[:, :-1]], axis=1)  # w -> window w-1
        return mx.concatenate([prev_shift, cur], axis=2)          # [b, nwin, 2*ratio, d]

    def _pool(self, kv: mx.array, score: mx.array, first_window: int) -> mx.array:
        """Gated pool + norm + compress-YaRN rope of already-formed windows.

        ``kv``/``score``: ``[b, nwin, slots, d]`` (``slots`` is ``ratio``, or
        ``2*ratio`` once ``_overlap_transform`` has folded the previous window in).
        Window ``first_window + i`` ropes at absolute position ``(first_window+i)*ratio``
        — its own first token — for both the overlap and non-overlap lanes.
        """
        nwin = kv.shape[1]
        rd = self.rope_head_dim
        pooled = mx.sum(kv * mx.softmax(score, axis=2), axis=2)            # [b, nwin, d]
        pooled = self.norm(pooled)
        win_pos = (mx.arange(nwin, dtype=mx.float32) + float(first_window)) * self.compress_ratio
        ang = win_pos[:, None] * self._inv_freq[None, :]
        cos, sin = mx.cos(ang), mx.sin(ang)
        head = pooled[..., :-rd]
        tail = _apply_interleaved_rope(pooled[..., -rd:], cos[None], sin[None])
        out = mx.concatenate([head, tail], axis=-1)
        return _hadamard_rotate(out) if self.rotate else out

    def __call__(self, x: mx.array) -> mx.array:
        """Whole-sequence pooling from ``start_pos == 0`` (the parity-gated path).

        The incremental equivalent is :meth:`step`; both funnel into :meth:`_pool`
        so the two paths cannot drift.
        """
        b, s, _ = x.shape
        ratio = self.compress_ratio
        d = self.head_dim
        cutoff = s - (s % ratio)
        nwin = cutoff // ratio
        if nwin == 0:
            return mx.zeros((b, 0, d), dtype=x.dtype)
        xf = x.astype(mx.float32)
        kv = self.wkv(xf)[:, :cutoff].reshape(b, nwin, ratio, -1)          # [b,nwin,ratio,coff*d]
        score = self.wgate(xf)[:, :cutoff].reshape(b, nwin, ratio, -1) + self.ape
        if self.overlap:
            kv = self._overlap_transform(kv, 0.0)                          # [b,nwin,2*ratio,d]
            score = self._overlap_transform(score, float("-inf"))
        return self._pool(kv, score, 0)

    def step(self, x: mx.array, state: "CompressorState", offset: int) -> mx.array:
        """Incremental pooling: consume ``x`` (positions ``offset..offset+s-1``) and
        emit the compressed rows whose windows *complete* inside that span.

        State machine adapted from ``ds4.c``'s ``compressor_decode_one`` (antirez/
        DwarfStar4, MIT): a token at position ``p`` lands in slot ``p % ratio`` of the
        in-progress window, and a row is emitted exactly when ``(p+1) % ratio == 0``.
        Window ``w`` therefore becomes attendable by query ``p == (w+1)*ratio - 1``,
        which is precisely what the prefill mask's ``c < (i+1)//ratio`` allows, so a
        decode step needs no compressed-column mask at all.

        The buffered frontier is the window's *projected* rows (post-``ape`` for the
        gate), not the raw hidden states, so the emit does the same arithmetic on the
        same values ``__call__`` would have.  For the overlap lane ``state.prev_*``
        keeps the last completed window's full-width rows, which
        :meth:`_overlap_transform` folds in under the ``:d`` projection.
        """
        b, s, _ = x.shape
        ratio = self.compress_ratio
        d = self.head_dim
        xf = x.astype(mx.float32)
        kv_rows = self.wkv(xf)                                   # [b, s, coff*d]
        ape_idx = (mx.arange(s) + offset) % ratio                # slot of each token
        score_rows = self.wgate(xf) + self.ape[ape_idx]
        if state.cur_kv is not None:
            kv_rows = mx.concatenate([state.cur_kv, kv_rows], axis=1)
            score_rows = mx.concatenate([state.cur_score, score_rows], axis=1)
        # kv_rows[:, 0] is at position offset - (offset % ratio), a window boundary.
        total = kv_rows.shape[1]
        nwin = total // ratio
        filled = nwin * ratio
        if nwin:
            kv_w = kv_rows[:, :filled].reshape(b, nwin, ratio, -1)
            score_w = score_rows[:, :filled].reshape(b, nwin, ratio, -1)
            if self.overlap:
                kv_slots = self._overlap_transform(kv_w, 0.0, state.prev_kv)
                score_slots = self._overlap_transform(
                    score_w, float("-inf"), state.prev_score
                )
                state.prev_kv = kv_w[:, -1]                      # [b, ratio, coff*d]
                state.prev_score = score_w[:, -1]
            else:
                kv_slots, score_slots = kv_w, score_w
            out = self._pool(kv_slots, score_slots, state.n_emitted)
            state.n_emitted += nwin
        else:
            out = mx.zeros((b, 0, d), dtype=mx.float32)
        state.cur_kv = kv_rows[:, filled:] if filled < total else None
        state.cur_score = score_rows[:, filled:] if filled < total else None
        return out


class Indexer(nn.Module):
    """Sparse-position selector for ``compress_ratio==4`` layers (reference
    ``Indexer``, model.py L380-433).

    It owns a second, narrower :class:`Compressor` (``index_head_dim`` wide, Hadamard
    rotated) that pools the *same* token windows as the attention compressor, plus
    ``wq_b``/``weights_proj``.  For each query it scores every compressed row

        ``score[q, c] = sum_h relu(q_h · row_c) * weights[q, h]``
        ``weights     = weights_proj(x) / sqrt(index_head_dim * index_n_heads)``

    and keeps the top ``index_topk`` of the rows that are causally available to that
    query.  :meth:`__call__` returns that decision as a boolean ``[b, s, n_comp]``
    mask (True = attend), which is what attention needs — the reference instead
    returns gathered indices for its sparse kernel and marks unusable slots ``-1``
    (``sparse_attn``, kernel.py L323-327, zeroes those rows and scores them ``-inf``),
    which is the same thing expressed for a gather.

    Per-query ``k``: the reference prefill takes one global
    ``k = min(index_topk, end_pos // ratio)`` over ``-inf``-masked scores and then
    re-invalidates any non-causal pick (L424-430), which is equivalent to taking
    ``k = min(index_topk, n_causal(q))`` per query — the form ds4.c evaluates
    directly (``indexer_allowed_decode_one``) and the form used here, because it also
    covers the chunked-prefill case the reference has no branch for.

    QAT: the reference FP4-quantises both ``q`` and the indexer's compressed rows
    (``fp4_act_quant``, L370/L416).  That emulation is dropped here, consistently with
    the attention compressor's dropped FP8 (see :class:`Compressor`).  The Hadamard
    rotation that precedes it *is* kept, because it is part of the model graph — but
    note it is an orthogonal map applied to both sides of the same dot product, so it
    cancels exactly; with FP4 dropped it cannot change a selection, and it is retained
    as the (tested) slot the quantiser would occupy.  ds4.c keeps both
    (``dsv4_indexer_qat_row_inplace_cpu``) and warns that without the pair "the top-k
    compressed-row selection is not the model's graph" — the divergence that warning
    is about is the FP4 step, not the rotation.
    """

    def __init__(self, args: ModelArgs, compress_ratio: int):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.compress_ratio = compress_ratio
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5
        self.compressor = Compressor(args, compress_ratio, self.head_dim, rotate=True)
        # The reference hands the indexer the *attention layer's* freqs_cis
        # (model.py L494); on a ratio-4 layer that is compress_rope_theta + YaRN.
        self._inv_freq = _yarn_inv_freq(
            self.rope_head_dim, args.compress_rope_theta, args.original_seq_len,
            args.rope_factor, args.beta_fast, args.beta_slow,
        )

    def scores(
        self, x: mx.array, qr: mx.array, positions: mx.array, rows: mx.array
    ) -> mx.array:
        """Per-query relevance of every compressed row (reference L411-421).

        ``x``: ``[b, s, dim]`` (the attention input) — ``weights_proj`` reads it.
        ``qr``: ``[b, s, q_lora_rank]`` — ``q_norm(wq_a(x))``, shared with attention.
        ``positions``: ``[s]`` absolute query positions.
        ``rows``: ``[b, n_comp, index_head_dim]`` every compressed row emitted so far.
        Returns ``[b, s, n_comp]`` fp32.  No causality applied — that is
        :meth:`__call__`'s job.
        """
        b, s, _ = x.shape
        rd = self.rope_head_dim
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim)
        ang = positions[:, None].astype(mx.float32) * self._inv_freq[None, :]
        cos, sin = mx.cos(ang), mx.sin(ang)
        q = mx.concatenate(
            [
                q[..., :-rd],
                _apply_interleaved_rope(
                    q[..., -rd:], cos[None, :, None, :], sin[None, :, None, :]
                ),
            ],
            axis=-1,
        )
        q = _hadamard_rotate(q.astype(mx.float32))
        weights = self.weights_proj(x).astype(mx.float32) * (
            self.softmax_scale * self.n_heads ** -0.5
        )                                                        # [b, s, n_heads]
        score = mx.einsum("bshd,btd->bsht", q, rows.astype(mx.float32))
        return mx.sum(mx.maximum(score, 0.0) * weights[..., None], axis=2)  # [b,s,t]

    def __call__(
        self, x: mx.array, qr: mx.array, positions: mx.array, rows: mx.array
    ) -> mx.array:
        """Select compressed rows for each query; returns ``[b, s, n_comp]`` bool."""
        b, s, _ = x.shape
        n_comp = int(rows.shape[1])
        ratio = self.compress_ratio
        score = self.scores(x, qr, positions, rows)

        # Causality: window c holds tokens [c*ratio, (c+1)*ratio), so query p may use
        # it once p has completed it — the same rule the dense mask uses.
        causal = (mx.arange(n_comp)[None, :] < ((positions[:, None] + 1) // ratio))[None]
        key = mx.where(causal, score, mx.array(-float("inf"), mx.float32))
        k_row = mx.minimum(
            mx.sum(causal.astype(mx.int32), axis=-1, keepdims=True), self.index_topk
        )
        k_row = mx.broadcast_to(k_row, (b, s, 1))
        return _topk_mask(key, k_row, min(self.index_topk, n_comp)) & causal


# ---------------------------------------------------------------------------
# Streaming decode state (sliding-window KV + compressed KV + compressor frontier)
# ---------------------------------------------------------------------------
class CompressorState:
    """Rolling frontier of one compressor lane.

    Mirrors ``ds4.c``'s ``attn_state_kv`` / ``attn_state_score`` row block
    (antirez/DwarfStar4, MIT).  ds4 keeps a fixed ``coff*ratio`` block and clears the
    unfilled tail after prefill (``compressor_finish_prefill_state_cpu``); here the
    filled rows are simply buffered, which is the same state without the -inf padding.
    """

    def __init__(self) -> None:
        self.cur_kv: Optional[mx.array] = None      # [b, offset % ratio, coff*head_dim]
        self.cur_score: Optional[mx.array] = None   # same, post-``ape``
        self.prev_kv: Optional[mx.array] = None     # [b, ratio, coff*head_dim] (overlap)
        self.prev_score: Optional[mx.array] = None
        self.n_emitted = 0

    def reset(self) -> None:
        self.cur_kv = None
        self.cur_score = None
        self.prev_kv = None
        self.prev_score = None
        self.n_emitted = 0


class DeepseekV4Cache:
    """Per-layer streaming cache.

    Five pieces, following ``ds4_layer_cache`` (ds4.c, MIT):
      * ``window``  — the rotated per-position KV rows still inside the sliding
        window, sliding by one row once full (``kv_cache_push_raw``).
      * ``compressed`` — every compressed KV row emitted so far
        (``kv_cache_push_comp``, ds4's ``attn_comp_kv``).
      * ``comp`` — the compressor's in-progress window (:class:`CompressorState`).
      * ``index_compressed`` / ``index_comp`` — the same two things for the ratio-4
        indexer's own, narrower compressor: ds4 carries ``index_comp_kv`` beside
        ``attn_comp_kv`` and ``index_state_kv``/``index_state_score`` beside
        ``attn_state_*``.  Maintained on every ratio-4 step regardless of whether the
        filter is currently active, because a row can only be built when its window's
        tokens go past — a context that crosses ``index_topk`` mid-decode would
        otherwise have no rows to score.

    Neither compressed lane is evicted, matching both the reference (a
    ``max_seq_len // ratio`` cache written at ``start_pos // ratio``, model.py L376)
    and ds4 (``comp_cap = ctx/ratio + 2``): the top-k filter bounds how many rows are
    *attended*, not how many are *kept*.  Row storage therefore still grows at
    ``head_dim/ratio`` bytes per token per compressed layer.

    ``offset`` is the absolute position of the next token, i.e. the standard
    mlx-lm cache contract the generate/serve path reads.
    """

    _META_VERSION = "mtplx-deepseek-v4-cache-v2"

    def __init__(self, window_size: int, compress_ratio: int, head_dim: int) -> None:
        self.window_size = int(window_size)
        self.compress_ratio = int(compress_ratio)
        self.head_dim = int(head_dim)
        self.offset = 0
        self.window: Optional[mx.array] = None      # [b, L, head_dim]
        self.window_start = 0                       # abs position of window[:, 0]
        self.compressed: Optional[mx.array] = None  # [b, n_comp, head_dim]
        self.comp = CompressorState()
        self.index_compressed: Optional[mx.array] = None  # [b, n_comp, index_head_dim]
        self.index_comp = CompressorState()

    # -- streaming updates -------------------------------------------------
    @property
    def n_compressed(self) -> int:
        return 0 if self.compressed is None else int(self.compressed.shape[1])

    @property
    def n_index_compressed(self) -> int:
        return 0 if self.index_compressed is None else int(self.index_compressed.shape[1])

    def update_window(self, kv: mx.array):
        """Append ``kv`` (positions ``offset..offset+s-1``) and return the rows this
        call can still see, as ``(rows, first_position)``.

        A query at ``p`` attends ``(p - window_size, p]``, so once the newest query is
        ``offset+s-1`` nothing older than ``offset+s-window_size`` can ever matter:
        rows below that are dropped here rather than masked.  For ``s == 1`` that
        leaves exactly the attendable set, so the decode step needs no mask.
        """
        s = int(kv.shape[1])
        if self.window is None:
            rows, start = kv, self.offset
        else:
            rows = mx.concatenate([self.window, kv], axis=1)
            start = self.window_start
        keep = self.window_size + s - 1
        if rows.shape[1] > keep:
            rows = rows[:, -keep:]
            start = self.offset + s - keep
        held = min(int(rows.shape[1]), self.window_size)
        self.window = rows if held == rows.shape[1] else rows[:, -held:]
        self.window_start = start + int(rows.shape[1]) - held
        return rows, start

    @staticmethod
    def _grow(rows: Optional[mx.array], new: mx.array) -> Optional[mx.array]:
        if new.shape[1] == 0:
            return rows
        return new if rows is None else mx.concatenate([rows, new], axis=1)

    def update_compressed(self, compressor: Compressor, x: mx.array) -> None:
        """Run the attention compressor's frontier over ``x`` and append its rows."""
        self.compressed = self._grow(
            self.compressed, compressor.step(x, self.comp, self.offset)
        )

    def update_index_compressed(self, compressor: Compressor, x: mx.array) -> None:
        """Same, for the ratio-4 indexer's own compressor lane."""
        self.index_compressed = self._grow(
            self.index_compressed, compressor.step(x, self.index_comp, self.offset)
        )

    def advance(self, s: int) -> None:
        self.offset += int(s)

    # -- mlx-lm cache contract --------------------------------------------
    @property
    def state(self):
        return (
            self.window,
            self.compressed,
            self.comp.cur_kv,
            self.comp.cur_score,
            self.comp.prev_kv,
            self.comp.prev_score,
            self.index_compressed,
            self.index_comp.cur_kv,
            self.index_comp.cur_score,
            self.index_comp.prev_kv,
            self.index_comp.prev_score,
        )

    @state.setter
    def state(self, value) -> None:
        if value is None:
            self.window = None
            self.compressed = None
            self.index_compressed = None
            self.comp.reset()
            self.index_comp.reset()
            self.offset = 0
            self.window_start = 0
            return
        if not isinstance(value, (tuple, list)) or len(value) != 11:
            raise ValueError("DeepSeek-V4 cache state must contain eleven entries")
        (
            self.window,
            self.compressed,
            self.comp.cur_kv,
            self.comp.cur_score,
            self.comp.prev_kv,
            self.comp.prev_score,
            self.index_compressed,
            self.index_comp.cur_kv,
            self.index_comp.cur_score,
            self.index_comp.prev_kv,
            self.index_comp.prev_score,
        ) = value

    def replace_state(self, value) -> None:
        self.state = value

    @property
    def meta_state(self):
        return (
            self._META_VERSION,
            str(self.offset),
            str(self.window_start),
            str(self.comp.n_emitted),
            str(self.index_comp.n_emitted),
        )

    @meta_state.setter
    def meta_state(self, value) -> None:
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 5
            or value[0] != self._META_VERSION
        ):
            raise ValueError(f"unsupported DeepSeek-V4 cache meta state: {value!r}")
        self.offset = int(value[1])
        self.window_start = int(value[2])
        self.comp.n_emitted = int(value[3])
        self.index_comp.n_emitted = int(value[4])

    def is_trimmable(self) -> bool:
        # Trimming would have to rewind the compressor frontier and the emitted
        # compressed rows together; not supported (ds4 snapshots both or neither).
        return False

    def size(self) -> int:
        return int(self.offset)

    def empty(self) -> bool:
        return self.offset == 0


# ---------------------------------------------------------------------------
# Attention (MQA-shaped MLA + sliding window + optional CSA + o-LoRA)
# ---------------------------------------------------------------------------
class DeepseekV4Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = args.head_dim - args.qk_rope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.eps = args.rms_norm_eps
        self.compress_ratio = args.compress_ratios[layer_id]
        self.softmax_scale = self.head_dim ** -0.5

        self.attn_sink = mx.zeros((self.n_heads,))
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
        # o-LoRA: grouped down-projection (block matmul) then a dense up-projection.
        # wo_a stores one [n_groups*o_lora_rank, n_heads*head_dim//n_groups] matrix
        # applied group-wise; see GroupedLoRA / __call__.
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)

        # rope frequencies: compressor layers use compress_rope_theta + YaRN;
        # ratio==0 layers use base rope_theta with no YaRN.
        if self.compress_ratio:
            inv = _yarn_inv_freq(
                self.rope_head_dim, args.compress_rope_theta, args.original_seq_len,
                args.rope_factor, args.beta_fast, args.beta_slow,
            )
        else:
            inv = _yarn_inv_freq(self.rope_head_dim, args.rope_theta, 0, 1.0, 32, 1)
        self._inv_freq = inv  # [rope_head_dim//2]

    def _rope_tables(self, positions: mx.array):
        # positions: [L] -> cos/sin [L, rope_head_dim//2]
        ang = positions[:, None].astype(mx.float32) * self._inv_freq[None, :]
        return mx.cos(ang), mx.sin(ang)

    def _o_lora(self, o: mx.array) -> mx.array:
        """Grouped output-LoRA (reference model.py L536-542).

        ``o``: ``[b, s, n_heads*head_dim]`` -> reshape ``[b, s, n_groups, per]``;
        each group projects ``per -> o_lora_rank`` by its own slice of ``wo_a``;
        concat to ``n_groups*o_lora_rank`` then ``wo_b`` -> dim.
        """
        b, s, _ = o.shape
        g = self.n_groups
        per = self.n_heads * self.head_dim // g
        r = self.o_lora_rank
        og = o.reshape(b, s, g, per)
        # wo_a stores one [g*r, per] matrix applied group-wise.  When the module
        # was quantised at load (real 4-bit checkpoint), dequantise to a dense
        # [g*r, per] before the grouped reshape; on the unquantised M2 path
        # wo_a is a plain nn.Linear.
        if isinstance(self.wo_a, nn.QuantizedLinear):
            w = mx.dequantize(
                self.wo_a.weight, self.wo_a.scales, self.wo_a.biases,
                group_size=self.wo_a.group_size, bits=self.wo_a.bits,
            )
        else:
            w = self.wo_a.weight
        w = w.reshape(g, r, per)  # grouped [g, r, per]
        # out[b,s,g,r] = sum_p og[...,g,p] * w[g,r,p]
        out = mx.einsum("bsgp,grp->bsgr", og, w)
        out = out.reshape(b, s, g * r)
        return self.wo_b(out)

    def _attn_mask(
        self,
        q_pos: mx.array,
        kv_pos: Optional[mx.array],
        n_win: int,
        n_comp: int,
        ratio: int,
        dtype,
        comp_sel: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        """Additive ``[b, 1, s, n_win + n_comp]`` mask reproducing the reference
        sparse gather: a query attends the causal sliding window over the per-position
        KV, plus the compressed windows selected for it.

        ``q_pos``/``kv_pos`` are *absolute* positions, so the same rule covers the
        one-shot prefill (both ``arange(s)``) and a cached chunk whose KV rows start
        before the queries.  ``kv_pos is None`` means the caller already dropped every
        unattendable window row (the ``s == 1`` decode step), so that half needs no
        mask.  ``comp_sel`` is the indexer's ``[b, s, n_comp]`` decision; without it
        the compressed half falls back to plain causality, i.e. every compressed row
        the query has completed — which is what the indexer itself returns whenever
        ``n_comp <= index_topk``.

        Returns ``None`` when there is nothing to mask.
        """
        if kv_pos is None and comp_sel is None:
            return None
        s = int(q_pos.shape[0])
        parts = []
        if kv_pos is not None:
            i = q_pos[:, None]
            j = kv_pos[None, :]
            parts.append(((j <= i) & (j > i - self.window_size))[None])   # [1, s, n_win]
        elif n_win:
            parts.append(mx.ones((1, s, n_win), dtype=mx.bool_))
        if n_comp:
            if comp_sel is None:
                c = mx.arange(n_comp)[None, :]
                parts.append((c < ((q_pos[:, None] + 1) // ratio))[None])
            else:
                parts.append(comp_sel)
        b = max(int(p.shape[0]) for p in parts)
        if len(parts) == 1:
            ok = parts[0]
        else:
            ok = mx.concatenate(
                [mx.broadcast_to(p, (b, s, p.shape[2])) for p in parts], axis=-1
            )
        neg = mx.array(mx.finfo(dtype).min, dtype)
        return mx.where(ok, mx.array(0.0, dtype), neg)[:, None]

    def _indexer_active(self, n_comp: int) -> bool:
        """Is the top-k filter load-bearing for this call?

        Below the threshold ``min(index_topk, n_comp) == n_comp``, so the indexer would
        select every causally-available row and return exactly the dense causal mask.
        Skipping the whole scoring path there is not just an optimisation: it keeps the
        short-context regime bit-identical to the pre-filter backend (ds4.c takes the
        same early-out — ``if (top_k == n_comp) { all allowed }``).
        """
        return self.compress_ratio == 4 and n_comp > self.indexer.index_topk

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        # Attends the causal sliding window over per-position KV plus the compressed
        # KV rows the ratio-4 indexer selects (every causal row on ratio-128 layers,
        # and on ratio-4 layers below index_topk) — the dense-mask equivalent of the
        # reference sparse_attn + topk_idxs gather.
        # `cache is None` runs the whole sequence in one shot (the parity-gated path);
        # otherwise the same math runs incrementally off DeepseekV4Cache.  `mask` is
        # built internally either way — it needs the compressed-position columns.
        b, s, _ = x.shape
        rd = self.rope_head_dim
        ratio = self.compress_ratio
        offset = 0 if cache is None else cache.offset
        positions = mx.arange(offset, offset + s)
        cos, sin = self._rope_tables(positions)

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim)
        # per-head RMS-like normalisation (no learned weight), reference L498
        q = q * mx.rsqrt(mx.mean(mx.square(q.astype(mx.float32)), axis=-1, keepdims=True) + self.eps)
        q = q.astype(x.dtype)
        q = mx.concatenate(
            [q[..., :-rd], _apply_interleaved_rope(q[..., -rd:], cos[None, :, None, :], sin[None, :, None, :])],
            axis=-1,
        )

        kv = self.kv_norm(self.wkv(x))  # [b, s, head_dim] (single shared KV — MQA)
        kv = mx.concatenate(
            [kv[..., :-rd], _apply_interleaved_rope(kv[..., -rd:], cos[None, :, :], sin[None, :, :])],
            axis=-1,
        )

        # concat the compressor's compressed KV (reference cats kv + kv_compress)
        comp_sel = None
        if cache is None:
            full_kv = kv
            n_comp = 0
            n_win = s
            if ratio:
                kvc = self.compressor(x)  # [b, n_comp, head_dim]
                n_comp = kvc.shape[1]
                if n_comp:
                    full_kv = mx.concatenate([kv, kvc], axis=1)  # [b, s+n_comp, head_dim]
                    if self._indexer_active(n_comp):
                        # No cache to keep, so the indexer's compressor only runs when
                        # its rows are actually about to be scored.
                        comp_sel = self.indexer(
                            x, qr, positions, self.indexer.compressor(x)
                        )
            kv_pos = positions
        else:
            # Compressor first: the window a token *completes* is attendable by that
            # same token (mask rule `c < (i+1)//ratio`), so it must land in the cache
            # before this step's scores are formed.  Order copied from ds4.c's decode
            # layer (push raw KV, compressor_decode_one, index compressor_decode_one,
            # indexer selection, then mixed attention).
            if ratio:
                cache.update_compressed(self.compressor, x)
                if ratio == 4:
                    cache.update_index_compressed(self.indexer.compressor, x)
            win_kv, win_start = cache.update_window(kv)
            n_comp = cache.n_compressed
            n_win = int(win_kv.shape[1])
            full_kv = win_kv if not n_comp else mx.concatenate(
                [win_kv, cache.compressed], axis=1
            )
            if self._indexer_active(n_comp):
                assert cache.n_index_compressed == n_comp, (
                    "indexer compressor lane desynced from the attention lane: "
                    f"{cache.n_index_compressed} vs {n_comp}"
                )
                comp_sel = self.indexer(x, qr, positions, cache.index_compressed)
            # s == 1: update_window already dropped every row outside the query's
            # window, so that half needs no mask (the compressed half still does once
            # the indexer is filtering).
            kv_pos = None if s == 1 else mx.arange(
                win_start, win_start + win_kv.shape[1]
            )
            cache.advance(s)

        q_t = q.transpose(0, 2, 1, 3)          # [b, h, s, head_dim]
        kt = full_kv[:, None]                  # [b, 1, s+n_comp, head_dim] (shared over heads)
        scores = (q_t * self.softmax_scale) @ mx.swapaxes(kt, -1, -2)  # [b, h, s, s+n_comp]
        add = self._attn_mask(
            positions, kv_pos, n_win, n_comp, ratio, scores.dtype, comp_sel=comp_sel
        )
        if add is not None:
            scores = scores + add
        # attn_sink: per-head learned logit in the softmax denominator
        sink = self.attn_sink.reshape(1, self.n_heads, 1, 1)
        m = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink)
        ex = mx.exp(scores - m)
        denom = mx.sum(ex, axis=-1, keepdims=True) + mx.exp(sink - m)
        o = (ex / denom) @ kt                  # [b, h, s, head_dim]
        o = o.transpose(0, 2, 1, 3)            # [b, s, h, head_dim]
        # de-rotate the tail dims (reference L534, inverse rope)
        o = mx.concatenate(
            [o[..., :-rd], _apply_interleaved_rope(o[..., -rd:], cos[None, :, None, :], -sin[None, :, None, :])],
            axis=-1,
        )
        o = o.reshape(b, s, self.n_heads * self.head_dim)
        return self._o_lora(o)


# ---------------------------------------------------------------------------
# MoE (gate: sqrtsoftplus / hash / noaux bias  +  SwitchGLU + shared expert)
# ---------------------------------------------------------------------------
class DeepseekV4MLP(nn.Module):
    """Shared-expert / dense MLP with the reference's swiglu clamp (limit=10)."""

    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.limit = args.swiglu_limit
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.limit and self.limit > 0:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        return self.down_proj(nn.silu(gate) * up)


class MoEGate(nn.Module):
    """Reference ``Gate`` (model.py L546-584): sqrtsoftplus scoring, bias-corrected
    (noaux_tc) top-k for score layers, or fixed tid2eid lookup for hash layers.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.dim = args.hidden_size
        self.topk = args.num_experts_per_tok
        self.score_func = args.scoring_func
        self.route_scale = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.n_routed = args.n_routed_experts
        self.hash = layer_id < args.num_hash_layers
        self.weight = mx.zeros((self.n_routed, self.dim))
        if self.hash:
            self.tid2eid = mx.zeros((args.vocab_size, self.topk), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros((self.n_routed,))

    def _score(self, x: mx.array) -> mx.array:
        s = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        if self.score_func == "softmax":
            return mx.softmax(s, axis=-1)
        if self.score_func == "sigmoid":
            return mx.sigmoid(s)
        # sqrtsoftplus
        return mx.sqrt(nn.softplus(s))

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        scores = self._score(x)  # [n, n_routed]
        if self.hash:
            assert input_ids is not None
            indices = self.tid2eid[input_ids.reshape(-1)]  # [n, topk]
        else:
            biased = scores + self.e_score_correction_bias
            indices = mx.argpartition(-biased, kth=self.topk - 1, axis=-1)[..., : self.topk]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.score_func != "softmax":
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True))
        weights = weights * self.route_scale
        return indices, weights


class DeepseekV4MoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.gate = MoEGate(args, layer_id)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts
        )
        self.shared_experts = DeepseekV4MLP(
            args, args.moe_intermediate_size * args.n_shared_experts
        )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None) -> mx.array:
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        ids = input_ids.reshape(-1) if input_ids is not None else None
        indices, weights = self.gate(xf, ids)
        y = self.switch_mlp(xf, indices)
        y = (y * weights[..., None].astype(y.dtype)).sum(axis=-2)
        y = y + self.shared_experts(xf)
        return y.reshape(shape)


# ---------------------------------------------------------------------------
# Decoder block (Hyper-Connections around attn + MoE)
# ---------------------------------------------------------------------------
class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.attn = DeepseekV4Attention(args, layer_id)
        self.ffn = DeepseekV4MoE(args, layer_id)
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn_hc = HyperConnection(args.hidden_size, args.hc_mult, args.hc_eps)
        self.ffn_hc = HyperConnection(args.hidden_size, args.hc_mult, args.hc_eps)
        self.attn_hc._iters = args.hc_sinkhorn_iters
        self.ffn_hc._iters = args.hc_sinkhorn_iters

    def __call__(self, h: mx.array, mask=None, cache=None, input_ids=None) -> mx.array:
        # h: [b, s, hc, dim]
        residual = h
        x, post, comb = self.attn_hc.pre(h)
        x = self.attn_norm(x)
        x = self.attn(x, mask=mask, cache=cache)
        h = self.attn_hc.post(x, residual, post, comb)

        residual = h
        x, post, comb = self.ffn_hc.pre(h)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids=input_ids)
        h = self.ffn_hc.post(x, residual, post, comb)
        return h


class DeepseekV4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hc_mult = args.hc_mult
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DeepseekV4DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hc_head = HeadHC(args.hidden_size, args.hc_mult, args.hc_eps)

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        h = self.embed_tokens(input_ids)  # [b, s, dim]
        # expand to hc_mult residual copies
        h = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], self.hc_mult, h.shape[-1]))
        # The attention builds its own window + compressed-KV causal mask internally
        # (it needs the compressed-position columns), so no mask is threaded here.
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=None, cache=c, input_ids=input_ids)
        # collapse hc copies then final norm
        h = self.hc_head(h)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV4Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        out = self.model(inputs, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights: dict) -> dict:
        """Adapt checkpoint tensors to this module tree.

        NOTE(M3): this is the placeholder from M1.  Confirmed remapping work for M3:
          * ``ffn.switch_mlp.*`` ships pre-stacked (already ``[n_experts, ...]``) with
            mxfp4 scales and no biases — feed straight into ``SwitchGLU``'s quantised
            path (mode override supplied via config["quantization"]).
          * ``attn.wo_a`` is a single ``[g*r, per]`` matrix; the grouped einsum in
            ``_o_lora`` consumes it as-is (reshaped to ``[g, r, per]``) — no split
            needed once quantised grouped matmul is wired.
          * ``ffn.gate.tid2eid`` (hash layers) loads as int32.
        For M1 the identity map keeps the module importable and unit-testable on
        synthetic weights.
        """
        return weights

    def make_cache(self):
        """One :class:`DeepseekV4Cache` per layer (sliding-window KV + compressed KV
        + compressor frontier).  Shapes come off the built attention modules so the
        cache cannot drift from the layer's own compress ratio."""
        return [
            DeepseekV4Cache(
                window_size=layer.attn.window_size,
                compress_ratio=layer.attn.compress_ratio,
                head_dim=layer.attn.head_dim,
            )
            for layer in self.layers
        ]
