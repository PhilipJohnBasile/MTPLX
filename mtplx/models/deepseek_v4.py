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
group_size 64** (weight/scales/biases).  The MTP block is dropped by the conversion
— see :class:`DeepseekV4MTP` and ``scripts/deepseek_v4_build_mtp_model.py``, which
restores it from the upstream FP8/FP4 checkpoint into a merged model directory.

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
  * That cache is **rewindable** (``DeepseekV4Cache.trim``), which is what the
    speculative lane needs on a rejected draft: emitted compressed rows truncate,
    both compressor frontiers rebuild from a bounded journal of their own projected
    rows, and the sliding window retains ``rollback_capacity`` extra rows because
    eviction cannot be undone.  Exactness is gated bit-for-bit against a
    never-speculated arm, on every lane and across every boundary that can break a
    rewind (tests/test_deepseek_v4_spec.py), with four rollback mutations caught.
    Making it exact is also what lets the *engine's* generic all-trimmable
    rejection repair serve this backend
    (``mtplx.cache_state.trim_verified_window_without_snapshot``) instead of a
    bespoke snapshot/restore path.
  * Dropped on purpose: the reference's inference-time QAT emulation (FP8 on the
    attention compressor's rows, FP4 on the indexer's q and rows).  It is noise
    injection, not model math — except that in the indexer it perturbs a *discrete*
    top-k boundary, so selections near the cut can differ from the reference.  The
    Hadamard rotation that precedes the FP4 step is implemented (it is graph, not
    noise), and is a no-op for selection on its own; see :class:`Indexer`.
  * The MTP draft block (:class:`DeepseekV4MTP`) is implemented and gated against a
    NumPy transcription of the reference ``MTPBlock`` (tests/test_deepseek_v4_mtp.py,
    max_rel ~2e-7 at a shrunk config; nine implementation mutations all caught).
    It binds through the ordinary load path from a checkpoint that ships ``mtp.0.*``
    — no sidecar, no env var — and :meth:`Model.sanitize` drops it from the tree
    when the weights are absent, which is the published mlx-community case and
    keeps the runtime's degrade-to-autoregressive branch reachable unchanged.
  * The speculative lane is wired: :class:`Model` carries the uniform runtime draft
    surface (``__call__(return_hidden=...)``, :meth:`Model.mtp_forward`,
    :meth:`Model.mtp_update_cache`, :meth:`Model.make_mtp_cache`) and
    :func:`inject_deepseek_v4_mtp_support` publishes it, so ``mtplx.generation``
    drives draft/verify/accept/reject/rollback here exactly as it does for every
    other native MTP backend — no parallel loop.  Greedy speculative decode at K =
    1, 2, 3 emits the identical committed sequence as pure AR through the real
    engine (tests/test_deepseek_v4_spec.py); acceptance counters are the engine's
    and come with it.  Not owned here: draft/verify are batch-shaped forwards, so
    the committed row's KV is projected inside a K+1-wide GEMM rather than alone —
    the invariance is committed-sequence exactness, not bitwise-identical logits.
  * The ``swiglu_limit`` clamp (10.0 in the shipped config) is applied in every
    expert, routed and shared, as the reference does (``Expert.forward``, model.py
    L600-602, handed the limit at L624/L627).  The shared expert carries it in
    :class:`DeepseekV4MLP`; the routed experts get it from :class:`ClampedSwiGLU`
    plugged into ``SwitchGLU``'s ``activation`` seam, so the batched expert kernels
    are untouched and one constructor covers trunk, hash and MTP layers alike.
    The clamp is asymmetric — ``up`` clipped to ``[-limit, +limit]``, ``gate`` cut
    only at ``+limit`` — and is gated against a NumPy oracle with the branches
    driven into saturation, with the branch-flip and clamp-removal mutations
    caught (tests/test_deepseek_v4_swiglu_clamp.py).  At ``swiglu_limit=0`` the
    routed path defers to the stock fused ``swiglu``, bit-identically, which is
    where both parity goldens were captured.
    Not yet measured: the activation ranges real V4-Flash weights actually reach,
    i.e. how often the clamp binds in practice.  That needs a checkpoint load and
    is deferred to a GPU window.
  * ``deepseek-v4`` is registered in ``mtplx/backends/registry.py`` so ``mtplx serve``
    resolves the load path.  That arch_id is what BOTH the AR-only mlx-community
    conversions and an MTP-bearing merged directory detect as; the separate
    ``deepseek-v4-mtp`` entry describes vLLM's *split* checkpoint layout, which is a
    different artifact shape MTPLX still has no loader for.

Decode-path bytes (tests/test_deepseek_v4_o_lora.py):
  * **o-LoRA weight handling.**  ``wo_a`` is static — ``[8192, 4096]`` on
    DeepSeek-V4-Flash — and the first cut ran ``mx.dequantize`` on it inside every
    ``_o_lora`` call, i.e. 64 MiB of dense bytes written and re-read per layer per
    decoded token, 43 layers deep.  It is now dequantised once and kept
    (``MTPLX_DSV4_O_LORA=cached``, the default, bit-identical to the old path and
    gated as such), which is what the reference does — it holds ``wo_a`` dense and
    just ``view``\\s it (model.py L537).  ``dequant`` restores the per-call
    behaviour as an A/B control; ``gather_qmm`` skips the dense tensor entirely and
    runs the 8 LoRA groups as one quantised block-diagonal matmul — the
    optimisation the reference explicitly leaves on the table (L538-539) — and is
    off by default because it is not bit-identical.
    What it is worth: ``cached`` vs ``dequant`` on the real checkpoint measured
    +2.1% AR (4.534 -> 4.627 tok/s) with fp32 activation storage, which is inside
    this box's cross-window drift — i.e. not distinguishable from zero, because at
    fp32 the einsum promotes ``wo_a`` anyway and caching removes the dequantize
    but not the cast that followed it.  It is kept because it is bit-identical and
    removes real redundant work, not because it is the speed win; the speed win is
    the activation-dtype fix below.  ``cached`` costs +2.69 GiB resident, which
    ``gather_qmm`` gives back in full.

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
import os
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU


# Default per-layer compress ratios for DeepSeek-V4-Flash (43 body layers; the
# 44th entry is the dropped MTP layer).  0 = pure sliding-window; 4 = overlapping
# compressor + indexer; 128 = non-overlapping compressor + strided index.
_DEFAULT_COMPRESS_RATIOS = (
    [0, 0]
    + [4, 128] * 20
    + [4, 0]
)

# How many token positions a :class:`DeepseekV4Cache` can un-decode (``trim``).
# Speculative decode only ever rewinds the rejected tail of one verify batch, so
# the real requirement is ``speculative_depth + 1`` (<= 9 for every depth MTPLX
# runs).  The default is set an order of magnitude above that because the cost is
# a handful of retained rows per layer, while the alternative -- discovering the
# bound is too small mid-request -- is a hard failure.  See
# :meth:`DeepseekV4Cache.trim` for what the capacity buys on each lane.
_DEFAULT_ROLLBACK_CAPACITY = 64


# ---------------------------------------------------------------------------
# Decode-path knobs
# ---------------------------------------------------------------------------
#: How :meth:`DeepseekV4Attention._o_lora` gets ``wo_a``.
#:
#: ``cached`` (default)
#:     Dequantise the static ``[o_groups*o_lora_rank, n_heads*head_dim/o_groups]``
#:     matrix once and keep the dense result.  Bit-identical to ``dequant``.
#: ``dequant``
#:     Re-run ``mx.dequantize`` on every call — the pre-cache behaviour, kept as
#:     the A/B control and as the oracle the bit-identity gate compares against.
#: ``gather_qmm``
#:     Skip the dense materialisation entirely and run the ``o_groups`` LoRA groups
#:     as one quantised block-diagonal matmul.  *Not* bit-identical (different
#:     accumulation order); off by default until a GPU window says it wins.
_O_LORA_MODES = ("cached", "dequant", "gather_qmm")


def _o_lora_mode_from_env() -> str:
    raw = (os.environ.get("MTPLX_DSV4_O_LORA") or "").strip().lower()
    if not raw:
        return "cached"
    if raw not in _O_LORA_MODES:
        raise ValueError(
            "MTPLX_DSV4_O_LORA must be one of "
            f"{', '.join(_O_LORA_MODES)}; got {raw!r}"
        )
    return raw


class _DerivedCache:
    """Holder for a tensor derived from parameters (e.g. a one-time dequant).

    A plain object rather than a bare ``mx.array`` attribute on purpose:
    ``nn.Module.__setattr__`` routes every ``mx.array``/``dict``/``list``/``tuple``
    into the module's own dict, and only the leading-underscore filter keeps it out
    of ``parameters()``.  Hanging the cache off a plain object keeps it out of the
    module dict altogether, so ``load_weights(strict=True)``, ``save_weights``,
    ``set_dtype`` and ``mx.eval(model)`` cannot see it at all.

    ``src`` holds the parameters the value was derived from, so a later
    ``load_weights``/``update``/``set_dtype`` (which rebinds those arrays)
    invalidates the cache by identity instead of serving a stale copy.
    """

    __slots__ = ("src", "value")

    def __init__(self) -> None:
        self.src: Optional[tuple] = None
        self.value: Optional[mx.array] = None

    def get(self, src: tuple) -> Optional[mx.array]:
        if self.value is None or self.src is None or len(self.src) != len(src):
            return None
        return self.value if all(a is b for a, b in zip(self.src, src)) else None

    def put(self, src: tuple, value: mx.array) -> mx.array:
        self.src = src
        self.value = value
        return value


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
    # multi-token prediction (draft head).  DeepSeek-V4-Flash ships one MTP block
    # upstream as ``mtp.0.*``; a conversion that drops it leaves this field at 1
    # while shipping no weights, which :meth:`Model.sanitize` detects and honours.
    num_nextn_predict_layers: int = 0

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
        # Rollback journal: the projected rows are per-position pure functions, so
        # keeping the most recent few is all a rewind needs to rebuild the frontier
        # (:meth:`CompressorState.rollback`).  Pushed BEFORE the frontier concat so
        # the journal stores each row exactly once, as the very array the emit path
        # consumes — a rebuilt frontier is then bit-identical, not merely equal.
        state.push_rollback_rows(kv_rows, score_rows)
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
    """Rolling frontier of one compressor lane, plus its rollback journal.

    Mirrors ``ds4.c``'s ``attn_state_kv`` / ``attn_state_score`` row block
    (antirez/DwarfStar4, MIT).  ds4 keeps a fixed ``coff*ratio`` block and clears the
    unfilled tail after prefill (``compressor_finish_prefill_state_cpu``); here the
    filled rows are simply buffered, which is the same state without the -inf padding.

    **Rollback.**  Speculative decode has to un-decode the rejected tail of a verify
    batch, and on this lane that means rewinding the frontier *and* the rows already
    emitted from it.  The emitted rows are trivial — a compressed row is a pure
    function of one completed window, so dropping the rows past the rewind point is
    exact.  The frontier is not: after a window completes, ``cur_*`` is reset to the
    remainder, so a rewind that crosses an emission boundary needs rows the frontier
    no longer holds.  They are also not recomputable without the hidden states, which
    the cache does not keep.

    So this state carries a bounded **journal** of the last ``rollback_rows`` projected
    rows (``tail_kv``/``tail_score``, the same post-``wkv``/post-``ape`` values the emit
    path consumes).  :meth:`rollback` slices the frontier back out of it.  The journal
    is sized to always cover the deepest legal rewind:

        ``rollback_rows = (2 if overlap else 1) * ratio + rollback_capacity``

    — ``ratio`` rows for ``prev_*`` (the last completed window, overlap lane only),
    up to ``ratio - 1`` for ``cur_*``, and ``rollback_capacity`` for the rewind itself.
    """

    def __init__(
        self,
        ratio: int = 0,
        overlap: bool = False,
        rollback_capacity: int = 0,
    ) -> None:
        self.ratio = int(ratio)
        self.overlap = bool(overlap)
        self.rollback_capacity = max(0, int(rollback_capacity))
        self.rollback_rows = (
            0
            if self.ratio <= 0
            else (2 if self.overlap else 1) * self.ratio + self.rollback_capacity
        )
        self.cur_kv: Optional[mx.array] = None      # [b, offset % ratio, coff*head_dim]
        self.cur_score: Optional[mx.array] = None   # same, post-``ape``
        self.prev_kv: Optional[mx.array] = None     # [b, ratio, coff*head_dim] (overlap)
        self.prev_score: Optional[mx.array] = None
        self.tail_kv: Optional[mx.array] = None     # [b, <=rollback_rows, coff*head_dim]
        self.tail_score: Optional[mx.array] = None
        self.n_emitted = 0

    def reset(self) -> None:
        self.cur_kv = None
        self.cur_score = None
        self.prev_kv = None
        self.prev_score = None
        self.tail_kv = None
        self.tail_score = None
        self.n_emitted = 0

    # -- rollback journal --------------------------------------------------
    def push_rollback_rows(self, kv: mx.array, score: mx.array) -> None:
        """Append this step's freshly projected rows to the bounded journal."""
        if self.rollback_rows <= 0 or kv.shape[1] == 0:
            return
        self.tail_kv = kv if self.tail_kv is None else mx.concatenate(
            [self.tail_kv, kv], axis=1
        )
        self.tail_score = score if self.tail_score is None else mx.concatenate(
            [self.tail_score, score], axis=1
        )
        if self.tail_kv.shape[1] > self.rollback_rows:
            self.tail_kv = self.tail_kv[:, -self.rollback_rows:]
            self.tail_score = self.tail_score[:, -self.rollback_rows:]

    def rollback(self, n: int, new_offset: int) -> None:
        """Rewind ``n`` token positions; ``new_offset`` is the resulting offset.

        Rebuilds ``cur_*``/``prev_*``/``n_emitted`` from the journal so the state is
        the one this lane would hold had those ``n`` tokens never been stepped —
        bit-identical, because every row it installs is a slice of the same array
        the forward pass produced for that position.
        """
        if self.ratio <= 0:
            return
        held = 0 if self.tail_kv is None else int(self.tail_kv.shape[1])
        kept = held - int(n)
        if kept < 0:
            raise ValueError(
                f"compressor rollback of {n} exceeds the journal ({held} rows held)"
            )
        r = int(new_offset) % self.ratio
        need = r + (self.ratio if (self.overlap and new_offset >= self.ratio) else 0)
        if kept < need:
            raise ValueError(
                f"compressor rollback of {n} leaves {kept} journal rows, "
                f"{need} needed to rebuild the frontier at offset {new_offset}"
            )
        if kept == 0:
            self.tail_kv = None
            self.tail_score = None
        else:
            self.tail_kv = self.tail_kv[:, :kept]
            self.tail_score = self.tail_score[:, :kept]
        self.n_emitted = int(new_offset) // self.ratio
        self.cur_kv = None if r == 0 else self.tail_kv[:, kept - r:]
        self.cur_score = None if r == 0 else self.tail_score[:, kept - r:]
        if self.overlap and self.n_emitted > 0:
            lo = kept - r - self.ratio
            self.prev_kv = self.tail_kv[:, lo: lo + self.ratio]
            self.prev_score = self.tail_score[:, lo: lo + self.ratio]
        else:
            self.prev_kv = None
            self.prev_score = None


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

    **Rollback (``trim``).**  The speculative lane verifies ``K+1`` tokens in one
    forward and then has to un-decode the rejected tail.  All three lanes here are
    rewindable, by three different mechanisms, each chosen because it is *exact*:

      * emitted compressed rows (both lanes) — **truncate**.  A row is a pure
        function of one completed window, so the rows a shorter context would have
        produced are a prefix of the rows this one did.
      * compressor / indexer frontier — **journal**.  ``cur_*``/``prev_*`` are
        rebuilt from :class:`CompressorState`'s bounded row journal, because a
        rewind across an emission boundary needs rows the frontier itself dropped
        and the cache keeps no hidden states to recompute them from.
      * sliding-window KV — **retention**.  Evicted rows are gone for good, so the
        window simply holds ``rollback_capacity`` rows more than it needs and
        returns only the attendable prefix to attention (which is why the retention
        change is invisible to the forward).  ``trim`` past that bound raises rather
        than silently half-rewinding.

    ``rollback_capacity`` is therefore a hard bound on rewind depth, uniform across
    the three lanes.  It is not a bound on how far back the model can *attend*.
    """

    _META_VERSION = "mtplx-deepseek-v4-cache-v3"

    def __init__(
        self,
        window_size: int,
        compress_ratio: int,
        head_dim: int,
        rollback_capacity: int = _DEFAULT_ROLLBACK_CAPACITY,
    ) -> None:
        self.window_size = int(window_size)
        self.compress_ratio = int(compress_ratio)
        self.head_dim = int(head_dim)
        self.rollback_capacity = max(0, int(rollback_capacity))
        self.offset = 0
        self.window: Optional[mx.array] = None      # [b, L, head_dim]
        self.window_start = 0                       # abs position of window[:, 0]
        self.compressed: Optional[mx.array] = None  # [b, n_comp, head_dim]
        overlap = self.compress_ratio == 4
        self.comp = CompressorState(
            ratio=self.compress_ratio,
            overlap=overlap,
            rollback_capacity=self.rollback_capacity,
        )
        self.index_compressed: Optional[mx.array] = None  # [b, n_comp, index_head_dim]
        self.index_comp = CompressorState(
            ratio=self.compress_ratio,
            overlap=overlap,
            rollback_capacity=self.rollback_capacity,
        )

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

        A query at ``p`` attends ``(p - window_size, p]``, so once the oldest query is
        ``offset`` nothing older than ``offset - window_size`` can matter to this
        call: those rows are excluded from the returned slice rather than masked.
        For ``s == 1`` that leaves exactly the attendable set, so the decode step
        needs no mask.

        What is *returned* and what is *retained* are two different sets.  The buffer
        keeps ``window_size + rollback_capacity`` rows so a rewind still has the rows
        the shorter context would have been holding (eviction is irreversible — see
        the class docstring); the extra rows never reach attention, so retention
        depth cannot change the forward.
        """
        s = int(kv.shape[1])
        if self.window is None:
            buf, buf_start = kv, self.offset
        else:
            buf = mx.concatenate([self.window, kv], axis=1)
            buf_start = self.window_start
        # rows visible to this call's oldest query (position ``offset``)
        first_visible = max(0, self.offset - self.window_size + 1)
        lo = max(0, first_visible - buf_start)
        rows = buf[:, lo:] if lo else buf
        start = buf_start + lo
        keep = self.window_size + self.rollback_capacity
        if buf.shape[1] > keep:
            buf = buf[:, -keep:]
            buf_start = self.offset + s - keep
        self.window = buf
        self.window_start = buf_start
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

    # -- rollback ----------------------------------------------------------
    @property
    def max_rollback(self) -> int:
        """Deepest legal :meth:`trim`, in token positions."""
        return min(self.rollback_capacity, int(self.offset))

    def trim(self, n: int) -> int:
        """Un-decode the last ``n`` token positions; returns ``n``.

        The mlx-lm cache trim contract (``rollback_after_verify`` /
        ``trim_verified_window_to_prefix`` in ``mtplx.cache_state``), implemented
        exactly: afterwards every field holds what it would hold had those ``n``
        tokens never been passed to the model, so the next forward is bit-identical
        to the one the shorter context would have run.

        Unlike a plain KV cache this trim is *bounded* (:attr:`max_rollback`) — the
        sliding window physically discards evicted rows.  Exceeding the bound raises
        instead of clamping: ``rollback_after_verify`` ignores the return value, so a
        clamped rewind would leave a silently desynced cache decoding on.

        The speculative lane never approaches the bound (it rewinds at most the
        verify width, ``K+1``).  The one caller that can is the session bank's
        near-prefix restore, which trims a restored snapshot down to an arbitrary
        matched prefix (``generation._trim_cache_to_offset``); on this backend that
        depth is not recoverable at all — the rows are gone — so it raises rather
        than returning the False that would let the caller fall back to a cold
        prefill.  Serving V4 behind a session bank therefore needs either a
        ``rollback_capacity`` sized for it or a ``max_rollback`` pre-check in that
        caller.
        """
        n = int(n)
        if n <= 0:
            return 0
        if n > int(self.offset):
            raise ValueError(
                f"cannot trim {n} tokens from a DeepSeek-V4 cache at offset "
                f"{self.offset}"
            )
        if n > self.rollback_capacity:
            raise ValueError(
                f"DeepSeek-V4 cache rollback of {n} exceeds rollback_capacity="
                f"{self.rollback_capacity}: the sliding window has already evicted "
                "the rows that depth would need"
            )
        new_offset = int(self.offset) - n
        if self.window is not None:
            kept = int(self.window.shape[1]) - n
            if kept <= 0:
                self.window = None
                self.window_start = new_offset
            else:
                self.window = self.window[:, :kept]
        if self.compress_ratio:
            n_rows = new_offset // self.compress_ratio
            if self.compressed is not None:
                self.compressed = None if n_rows == 0 else self.compressed[:, :n_rows]
            self.comp.rollback(n, new_offset)
            # The indexer lane only exists on ratio-4 layers; on ratio-128 its
            # state is constructed but never stepped, so there is nothing to rewind.
            if self.compress_ratio == 4:
                if self.index_compressed is not None:
                    self.index_compressed = (
                        None if n_rows == 0 else self.index_compressed[:, :n_rows]
                    )
                self.index_comp.rollback(n, new_offset)
        self.offset = new_offset
        return n

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
            self.comp.tail_kv,
            self.comp.tail_score,
            self.index_compressed,
            self.index_comp.cur_kv,
            self.index_comp.cur_score,
            self.index_comp.prev_kv,
            self.index_comp.prev_score,
            self.index_comp.tail_kv,
            self.index_comp.tail_score,
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
        if not isinstance(value, (tuple, list)) or len(value) != 15:
            raise ValueError("DeepSeek-V4 cache state must contain fifteen entries")
        (
            self.window,
            self.compressed,
            self.comp.cur_kv,
            self.comp.cur_score,
            self.comp.prev_kv,
            self.comp.prev_score,
            self.comp.tail_kv,
            self.comp.tail_score,
            self.index_compressed,
            self.index_comp.cur_kv,
            self.index_comp.cur_score,
            self.index_comp.prev_kv,
            self.index_comp.prev_score,
            self.index_comp.tail_kv,
            self.index_comp.tail_score,
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
        # :meth:`trim` rewinds all three lanes exactly, which is what lets the
        # engine's snapshot-free rejection repair
        # (``mtplx.cache_state.trim_verified_window_without_snapshot``) serve this
        # backend instead of a bespoke restore path.
        return True

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
        # How _o_lora consumes wo_a, and where the one-time dequant lives.  Both
        # are plain (non-array) attributes, so neither reaches the weight tree.
        self.o_lora_mode = _o_lora_mode_from_env()
        self._wo_a_cache = _DerivedCache()

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

    def _wo_a_quant(self):
        """``wo_a``'s quantised tensors + format, or ``None`` when it is dense.

        ``(weight, scales, biases, group_size, bits, mode)``.  ``biases`` is
        ``None`` for the bias-free modes (mxfp4); ``mode`` is carried through
        rather than assumed so the affine path stays byte-for-byte what it was.
        """
        wo = self.wo_a
        if not isinstance(wo, nn.QuantizedLinear):
            return None
        return (
            wo.weight,
            wo.scales,
            getattr(wo, "biases", None),
            wo.group_size,
            wo.bits,
            getattr(wo, "mode", "affine"),
        )

    def _wo_a_grouped(self) -> mx.array:
        """``wo_a`` as a dense ``[n_groups, o_lora_rank, per]`` tensor.

        ``wo_a`` is **static**: one ``[g*r, per]`` matrix, the same on every token
        of every step.  On the real checkpoint it is 4-bit, and the pre-cache code
        ran ``mx.dequantize`` on it inside every ``_o_lora`` call — on
        DeepSeek-V4-Flash that is a 64 MiB dense tensor written and re-read per
        layer per decoded token, 43 layers deep, for a value that never changes.
        The reference does the dequant once at load and keeps the dense matrix
        (``wo_a = self.wo_a.weight.view(...)``, model.py L537).

        ``cached`` therefore stores exactly what ``mx.dequantize`` returned, so the
        consuming einsum sees the identical values and the path stays bit-identical
        to ``dequant`` (gated by tests/test_deepseek_v4_o_lora.py).  Resident cost
        is one dense copy per layer beside the quantised one it is derived from —
        2.69 GiB across 43 layers on DeepSeek-V4-Flash.

        Do not read the byte count above as a speed claim: measured on the real
        checkpoint at fp32 activation storage, ``cached`` vs ``dequant`` is +2.1%
        AR, inside cross-window drift (bench/deepseek-v4/goal-ab-20260731).  At
        fp32 the einsum promotes ``wo_a`` regardless, so caching removes the
        dequantize and not the cast behind it.  The measured decode win in this
        lane is the activation-dtype fix, not this.
        """
        g = self.n_groups
        r = self.o_lora_rank
        per = self.n_heads * self.head_dim // g
        q = self._wo_a_quant()
        if q is None:
            # Unquantised (the M2/parity path): wo_a is a plain nn.Linear.
            return self.wo_a.weight.reshape(g, r, per)
        w, scales, biases, group_size, bits, mode = q
        src = (w, scales, biases)
        if self.o_lora_mode != "dequant":
            hit = self._wo_a_cache.get(src)
            if hit is not None:
                return hit
        dense = mx.dequantize(
            w, scales, biases, group_size=group_size, bits=bits, mode=mode
        ).reshape(g, r, per)
        if self.o_lora_mode == "dequant":
            return dense
        return self._wo_a_cache.put(src, dense)

    def _o_lora_gather_qmm(self, o: mx.array) -> mx.array:
        """Grouped o-LoRA as a quantised block-diagonal matmul (arm b).

        The ``o_groups`` LoRA groups are ``o_groups`` independent ``[r, per]``
        matrices, so the projection is one :func:`mx.gather_qmm` over a leading
        group axis — every row visits every group, and nothing dense is ever
        materialised.  The reference flags exactly this as the optimisation it did
        not take ("wo_a is FP8 in checkpoint; could do FP8 einsum here for better
        perf, but using BF16 for simplicity", model.py L538-539).

        **Calling convention.**  ``x`` must carry the row axis in the *batch* dims
        with the matmul rows in the last two, i.e. ``[g, rows, per] -> [g, rows,
        r]``; a flat ``[rows, per]`` broadcasts instead and silently does ``g``
        times the work while still producing usable-looking numbers.  This box's
        ledger has been bitten by that twice, which is why the output shape is
        checked here rather than assumed.

        **Not bit-identical** to :meth:`_wo_a_grouped` + einsum: the quantised
        kernel dequantises inside the accumulation, so the products are summed in
        a different order.  Gated on tolerance + argmax stability, default off.
        """
        b, s, _ = o.shape
        g = self.n_groups
        r = self.o_lora_rank
        per = self.n_heads * self.head_dim // g
        w, scales, biases, group_size, bits, mode = self._wo_a_quant()
        rows = b * s
        # [b, s, g*per] -> [g, rows, per]: group g owns o's g-th per-wide chunk.
        x = o.reshape(rows, g, per).swapaxes(0, 1)
        out = mx.gather_qmm(
            x,
            w.reshape(g, r, -1),
            scales.reshape(g, r, -1),
            None if biases is None else biases.reshape(g, r, -1),
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        if tuple(out.shape) != (g, rows, r):
            raise AssertionError(
                "gather_qmm o-LoRA shape contract broken: expected "
                f"{(g, rows, r)}, got {tuple(out.shape)} — an x of shape "
                f"{tuple(x.shape)} was broadcast instead of batched"
            )
        return self.wo_b(out.swapaxes(0, 1).reshape(b, s, g * r))

    def _o_lora(self, o: mx.array) -> mx.array:
        """Grouped output-LoRA (reference model.py L536-542).

        ``o``: ``[b, s, n_heads*head_dim]`` -> reshape ``[b, s, n_groups, per]``;
        each group projects ``per -> o_lora_rank`` by its own slice of ``wo_a``;
        concat to ``n_groups*o_lora_rank`` then ``wo_b`` -> dim.

        See :data:`_O_LORA_MODES` for the three ways ``wo_a`` gets there.
        """
        if self.o_lora_mode == "gather_qmm" and self._wo_a_quant() is not None:
            return self._o_lora_gather_qmm(o)
        b, s, _ = o.shape
        g = self.n_groups
        per = self.n_heads * self.head_dim // g
        r = self.o_lora_rank
        og = o.reshape(b, s, g, per)
        w = self._wo_a_grouped()  # [g, r, per]
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
    """Shared-expert / dense MLP with the reference's swiglu clamp (limit=10).

    Reference ``Expert.forward`` (model.py L596-606), verbatim::

        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up

    Note the asymmetry, which is easy to get wrong in both directions: the *up*
    branch (``w3`` = ``up_proj``) is clipped to ``[-limit, +limit]``, the *gate*
    branch (``w1`` = ``gate_proj``) only has its upper tail cut at ``+limit`` and
    keeps its whole negative range.  Both cuts land on the pre-activation
    projections, before ``silu``.  ``limit <= 0`` disables the clamp entirely,
    which is what both parity goldens were captured at.
    """

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


class ClampedSwiGLU(SwiGLU):
    """``SwitchGLU`` activation carrying the reference's ``swiglu_limit`` clamp.

    The reference applies the clamp inside *every* expert, routed ones included
    (``MoE.__init__`` L624 passes ``swiglu_limit=args.swiglu_limit`` to each
    routed :class:`Expert`, exactly as L627 does for the shared one).  Routed
    experts here run through mlx-lm's :class:`SwitchGLU`, whose only seam is the
    ``activation`` module it calls between the ``up``/``gate`` projections and
    ``down_proj`` — which is precisely where the reference's clamp sits.  So the
    faithful port is an activation, not a fork of ``SwitchGLU``: the batched
    ``gather_mm``/``gather_qmm`` expert kernels are untouched.

    ``SwitchGLU.__call__`` invokes ``self.activation(x_up, x_gate)``, so the
    first argument is the *up* branch and the second is the *gate* branch — the
    opposite of the reading the names suggest.  The clamp is asymmetric between
    them; see :class:`DeepseekV4MLP` for the quoted reference lines.

    At ``limit <= 0`` this defers to :class:`SwiGLU` untouched, so the disabled
    path is the stock fused ``swiglu`` kernel and stays bit-identical to a model
    built without this class at all (both parity goldens were captured there).
    Holds no parameters, so the load path and the weight tree are unchanged.
    """

    def __init__(self, limit: float = 0.0):
        super().__init__()
        self.limit = float(limit or 0.0)

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        if self.limit > 0:
            x = mx.clip(x, -self.limit, self.limit)      # up:   two-sided
            gate = mx.minimum(gate, self.limit)          # gate: upper tail only
        return super().__call__(x, gate)


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
    """Routed experts + one shared expert (reference ``MoE``, model.py L609-644).

    ``swiglu_limit`` reaches both halves: the routed experts through
    :class:`ClampedSwiGLU` (the ``SwitchGLU`` activation seam) and the shared one
    through :class:`DeepseekV4MLP`, matching L624/L627 where the reference hands
    the same limit to both.  This constructor is the *only* place the backend
    builds routed experts, so trunk score layers, trunk hash layers and the
    :class:`DeepseekV4MTP` draft block (a :class:`DeepseekV4DecoderLayer`
    subclass) are all covered by construction rather than by three call sites
    kept in sync.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.gate = MoEGate(args, layer_id)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=ClampedSwiGLU(args.swiglu_limit),
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


# ---------------------------------------------------------------------------
# Multi-token-prediction draft block
# ---------------------------------------------------------------------------
class DeepseekV4MTP(DeepseekV4DecoderLayer):
    """Speculative-decode draft block (reference ``MTPBlock``, model.py L738-766).

    **What it owns.**  ``MTPBlock`` subclasses ``Block``, so the draft head is a
    full decoder layer in its own right: its own attention, its own 256-expert
    MoE, its own ``attn_norm``/``ffn_norm`` and its own two Hyper-Connection
    blocks — none of it shared with the trunk.  On top of a body block it adds
    six pieces (L742-752): ``enorm``/``hnorm`` normalise the two inputs,
    ``e_proj``/``h_proj`` project and sum them, and ``norm`` + ``hc_head`` do the
    final collapse that the trunk does with ``model.norm`` + ``model.hc_head``.
    Every one of those ships upstream under ``mtp.0.*``.

    **What it shares.**  Exactly two things, and it holds no copy of either:
    the token embedding and the output projection.  ``Transformer.__init__``
    L792-793 assigns ``mtp[i].embed = self.embed`` and ``mtp[i].head = self.head``
    after constructing the block, so the draft's logits land in the same
    vocabulary space as the target's — which is what makes accept/reject a
    comparison of like with like.  Both are therefore passed *in* to
    :meth:`__call__` rather than stored, so the 129280-row embedding and lm_head
    are never duplicated in memory.

    **Which layer it is.**  ``layer_id = n_layers + i`` (L791) — 43 on
    DeepSeek-V4-Flash — and that index is what the inherited ``Attention`` and
    ``Gate`` read.  ``compress_ratios[43] == 0`` in the shipped config, so the
    draft block is a **pure sliding-window** attention layer: base ``rope_theta``,
    no YaRN, no :class:`Compressor`, no :class:`Indexer`.  ``43 >=
    num_hash_layers`` (3), so its gate is score-routed (``noaux_tc`` bias), not
    hash-routed.  Both fall out of the inherited constructor rather than being
    re-decided here.

    **Forward** (L757-766).  ``h`` is the trunk's pre-head Hyper-Connection state
    ``[b, s, hc, dim]`` (:meth:`DeepseekV4Model.hc_hidden`), ``input_ids`` are the
    tokens whose *embeddings* get fused in — the caller aligns them, and for
    speculative decode that means position ``i`` of ``input_ids`` is the token the
    trunk predicted *at* ``h[:, i]``, i.e. shifted one ahead of the ids that
    produced ``h``.  The block does not shift anything itself; the reference
    does not either.
    """

    def __init__(self, args: ModelArgs, layer_id: Optional[int] = None):
        layer_id = args.num_hidden_layers if layer_id is None else int(layer_id)
        ratios = list(args.compress_ratios)
        if len(ratios) <= layer_id:
            # The shipped config carries the MTP layer's entry (44 ratios for 43
            # layers, trailing 0).  A config trimmed to the trunk length gets the
            # same value rather than an IndexError out of Attention.__init__.
            ratios = ratios + [0] * (layer_id + 1 - len(ratios))
            args = replace(args, compress_ratios=ratios)
        super().__init__(args, layer_id)
        dim = args.hidden_size
        eps = args.rms_norm_eps
        self.enorm = nn.RMSNorm(dim, eps=eps)
        self.hnorm = nn.RMSNorm(dim, eps=eps)
        self.e_proj = nn.Linear(dim, dim, bias=False)
        self.h_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.hc_head = HeadHC(dim, args.hc_mult, args.hc_eps)

    def __call__(
        self,
        h: mx.array,
        input_ids: mx.array,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        cache=None,
        return_hidden: bool = False,
    ) -> mx.array:
        """``h``: ``[b, s, hc, dim]`` -> draft logits ``[b, s, vocab]``.

        ``embed_tokens``/``lm_head`` are the trunk's, per the sharing above.  The
        reference's ``ParallelHead.get_logits`` slices ``x[:, -1]`` before the
        matmul because its caller only ever wants the last row; the full sequence
        is returned here (mlx-lm's convention) and that slice is the caller's.

        ``return_hidden`` additionally returns the block's own pre-head
        Hyper-Connection state ``[b, s, hc, dim]``.  That is the tensor a
        multi-step draft chain feeds back in as ``h``: it occupies exactly the
        position the trunk's :meth:`DeepseekV4Model.hc_hidden` output does, which
        is what makes step ``i+1`` of the chain the same computation step ``i``
        ran.  Depth > 1 is an MTPLX extension either way — the reference ships one
        block and defines only the depth-1 call — and it is the same extension the
        sibling appended-layer backends make (GLM's modulo-into-layers, Hy3's
        single NextN layer reused at every depth).
        """
        e = self.enorm(embed_tokens(input_ids))          # [b, s, dim]
        x = self.hnorm(h)                                # [b, s, hc, dim]
        x = self.e_proj(e)[:, :, None, :] + self.h_proj(x)
        x = super().__call__(x, mask=None, cache=cache, input_ids=input_ids)
        logits = lm_head(self.norm(self.hc_head(x)))
        return (logits, x) if return_hidden else logits


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

    def hc_hidden(self, input_ids: mx.array, cache=None) -> mx.array:
        """Run the body and stop at the Hyper-Connection state ``[b, s, hc, dim]``.

        This is the split point the MTP block needs: the reference keeps ``h`` in
        hc form all the way out of the body and hands *that* tensor to both the
        output head and ``MTPBlock.forward`` (``Transformer.forward`` L806-808 vs
        model.py L757-763).  Collapsing to ``[b, s, dim]`` first — which is what
        :meth:`__call__` returns — would destroy the copies the draft block's own
        ``hnorm``/``h_proj`` read.
        """
        h = self.embed_tokens(input_ids)  # [b, s, dim]
        # expand to hc_mult residual copies
        h = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], self.hc_mult, h.shape[-1]))
        # The attention builds its own window + compressed-KV causal mask internally
        # (it needs the compressed-position columns), so no mask is threaded here.
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=None, cache=c, input_ids=input_ids)
        return h

    def collapse(self, h: mx.array) -> mx.array:
        """Head-side collapse of the hc copies + final norm (``ParallelHead.forward``
        L718-721, minus the ``lm_head`` matmul the caller owns)."""
        return self.norm(self.hc_head(h))

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        return self.collapse(self.hc_hidden(input_ids, cache))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV4Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        # Reference ``Transformer.mtp`` (model.py L789-793): a top-level list, so
        # the parameter paths are ``mtp.{i}.*`` — exactly the upstream checkpoint's
        # names.  Dropped again by :meth:`sanitize` if the weights are not there.
        self.mtp = [
            DeepseekV4MTP(args, args.num_hidden_layers + i)
            for i in range(max(int(args.num_nextn_predict_layers or 0), 0))
        ]

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        return_hidden: bool = False,
        input_embeddings=None,
        hidden_variant: Optional[str] = None,
        emit_logits: bool = True,
        logits_keep: Optional[int] = None,
        **kwargs,
    ):
        """Target forward; also the MTPLX runtime's ``forward_ar`` surface.

        Plain ``model(ids)`` / ``model(ids, cache=cache)`` is unchanged.  The extra
        keywords are the contract ``mtplx.runtime.MTPLXRuntime.forward_ar`` drives
        every MTP backend through:

        * ``return_hidden`` — also return the state the draft block consumes.  For
          this architecture that is the pre-head Hyper-Connection tensor
          ``[b, s, hc, dim]`` (:meth:`hc_hidden`), NOT a ``[b, s, dim]`` hidden:
          collapsing first would destroy the copies ``DeepseekV4MTP.hnorm`` /
          ``h_proj`` read.  Engine code only ever slices axis 1 of this tensor and
          hands it back to :meth:`mtp_forward`, so the extra axis is transparent.
        * ``hidden_variant`` — accepted and ignored.  The variant knob picks
          between a Qwen-style draft's pre-norm/post-norm/fc taps; V4's draft input
          is defined by the reference as exactly one tensor, so there is nothing to
          pick.  Raising instead would break every draft call, since
          ``runtime.draft_mtp`` always resolves the contract default.  Same
          decision as the sibling appended-layer backends (glm_mtp, step3p5, hy3).
        * ``emit_logits`` / ``logits_keep`` — skip, or restrict to the last ``k``
          rows, the ``lm_head`` matmul.  Over a 129280-row vocabulary that matmul
          dominates a prefill chunk, and prefill only needs the final row.
        """
        if input_embeddings is not None:
            raise ValueError(
                "the DeepSeek-V4 backend does not support input_embeddings "
                "(no vision splice path)"
            )
        h = self.model.hc_hidden(inputs, cache)
        logits = None
        if emit_logits:
            source = h
            if logits_keep is not None:
                source = h[:, -max(1, int(logits_keep)):]
            logits = self.logits_from_hc_hidden(source)
        if not return_hidden:
            return logits
        return logits, h

    @property
    def layers(self):
        return self.model.layers

    # -- MTP (speculative draft head) --------------------------------------
    @property
    def mtp_blocks(self) -> list:
        """The draft blocks, however ``mtp`` is currently bound.

        :meth:`__init__` binds ``self.mtp`` to a plain list so the parameter paths
        are the checkpoint's ``mtp.{i}.*``.  ``inject_deepseek_v4_mtp_support``
        rebinds it (post-load) to a container that also answers ``.layers``, which
        is what ``mtplx.mtp_patch.validate_mtp_support`` probes for.  Everything
        else goes through this property so neither binding is load-bearing.
        """
        blocks = getattr(self, "mtp", None)
        if blocks is None:
            return []
        return list(getattr(blocks, "layers", blocks))

    @property
    def has_mtp(self) -> bool:
        return bool(self.mtp_blocks)

    def hc_hidden(self, inputs: mx.array, cache=None) -> mx.array:
        """Trunk forward stopping at the pre-head state the MTP block consumes."""
        return self.model.hc_hidden(inputs, cache)

    def logits_from_hc_hidden(self, h: mx.array) -> mx.array:
        """``[b, s, hc, dim]`` -> target logits; the other half of :meth:`hc_hidden`.

        ``logits_from_hc_hidden(hc_hidden(x)) == self(x)`` — a speculative step
        gets the target's logits and the draft's input from one trunk pass.
        """
        return self.lm_head(self.model.collapse(h))

    def mtp_forward(
        self,
        h: mx.array,
        input_ids: mx.array,
        index: int = 0,
        cache=None,
        *,
        mtp_cache=None,
        concat_order: Optional[str] = None,
        return_hidden: bool = False,
        mtp_hidden_variant: Optional[str] = None,
        position_offset: Optional[int] = None,
        mtp_depth: Optional[int] = None,
    ):
        """Draft logits from the trunk's ``h`` and the next tokens' ids.

        Supplies the two modules the reference assigns onto the block (the trunk
        embedding and lm_head) instead of duplicating them; see
        :class:`DeepseekV4MTP` for the ``input_ids`` alignment contract.

        Two ways to hand it a cache, because it answers to two callers:

        * ``cache`` — the **one** :class:`DeepseekV4Cache` belonging to block
          ``index`` (``make_mtp_cache()[index]``, not the list).  The trunk takes a
          list because it has one entry per layer; a draft block is a single layer
          and takes its own.
        * ``mtp_cache`` — the whole list, which is what
          ``MTPLXRuntime.draft_mtp`` passes; ``index`` selects from it.

        The remaining keywords are the runtime's uniform draft signature.
        ``concat_order`` and ``mtp_hidden_variant`` are Qwen-shaped knobs with no
        V4 counterpart (see :meth:`__call__`) and are accepted and ignored;
        ``mtp_depth`` is informational, as it is for every single-block draft head
        (the one block is reused at every depth); ``position_offset`` is rejected
        rather than ignored, because silently dropping it would put the draft's
        RoPE at the wrong absolute position instead of failing.
        """
        blocks = self.mtp_blocks
        if not blocks:
            raise RuntimeError("this checkpoint ships no MTP block")
        if isinstance(cache, (list, tuple)):
            raise TypeError(
                "mtp_forward takes the MTP block's own cache, not the list: "
                f"pass make_mtp_cache()[{index}]"
            )
        if position_offset is not None:
            raise ValueError(
                "the DeepSeek-V4 draft block takes its RoPE offset from its own "
                "cache; explicit position_offset is not supported"
            )
        if mtp_cache is not None:
            if not isinstance(mtp_cache, (list, tuple)):
                raise TypeError("mtp_cache must be the make_mtp_cache() list")
            if cache is not None:
                raise TypeError("pass either cache= or mtp_cache=, not both")
            cache = mtp_cache[index] if mtp_cache else None
        return blocks[index](
            h,
            input_ids,
            self.model.embed_tokens,
            self.lm_head,
            cache=cache,
            return_hidden=return_hidden,
        )

    def mtp_update_cache(
        self,
        h: mx.array,
        input_ids: mx.array,
        index: int = 0,
        *,
        mtp_cache=None,
        concat_order: Optional[str] = None,
        mtp_hidden_variant: Optional[str] = None,
        position_offset: Optional[int] = None,
        mtp_depth: Optional[int] = None,
        input_embeddings=None,
    ) -> mx.array:
        """Append committed history to the draft cache; returns the draft hidden.

        ``MTPLXRuntime.update_mtp_cache`` drives this to keep the draft block's KV
        in step with the tokens the target committed.  The ``lm_head`` matmul still
        runs — the draft head shares the trunk's 129280-row projection and this
        call is off the hot path (history append, not per-step drafting).
        """
        if input_embeddings is not None:
            raise ValueError(
                "the DeepSeek-V4 backend does not support input_embeddings "
                "(no vision splice path)"
            )
        _logits, hidden = self.mtp_forward(
            h,
            input_ids,
            index,
            mtp_cache=mtp_cache,
            concat_order=concat_order,
            return_hidden=True,
            mtp_hidden_variant=mtp_hidden_variant,
            position_offset=position_offset,
            mtp_depth=mtp_depth,
        )
        return hidden

    def make_mtp_cache(self):
        """One :class:`DeepseekV4Cache` per MTP block.

        Separate from :meth:`make_cache`: the draft block's attention is its own
        module with its own KV (reference ``Attention.__init__`` L474 registers a
        per-instance ``kv_cache``), so it must not share the trunk's.  Its
        ``compress_ratio`` is 0, which makes the cache a plain sliding window —
        no compressed rows, no compressor frontier, so ``trim`` there rewinds only
        the window and ``offset``.
        """
        return [
            DeepseekV4Cache(
                window_size=block.attn.window_size,
                compress_ratio=block.attn.compress_ratio,
                head_dim=block.attn.head_dim,
            )
            for block in self.mtp_blocks
        ]

    def sanitize(self, weights: dict) -> dict:
        """Adapt this module tree to the checkpoint's tensors.

        Confirmed no-ops (the checkpoint already matches the tree):
          * ``ffn.switch_mlp.*`` ships pre-stacked (already ``[n_experts, ...]``) with
            mxfp4 scales and no biases — feed straight into ``SwitchGLU``'s quantised
            path (mode override supplied via config["quantization"]).
          * ``attn.wo_a`` is a single ``[g*r, per]`` matrix; the grouped einsum in
            ``_o_lora`` consumes it as-is (reshaped to ``[g, r, per]``) — no split
            needed once quantised grouped matmul is wired.
          * ``ffn.gate.tid2eid`` (hash layers) loads as int32.

        The one real adaptation is the MTP block.  ``num_nextn_predict_layers`` is
        not trustworthy on its own: the published MLX conversions declare 1 while
        shipping no ``mtp.*`` tensor at all (which is what
        ``mtplx.artifacts.mtp_weights_present_on_disk`` and the runtime's
        degrade-to-autoregressive branch exist for).  So the *weights* decide —
        a checkpoint that ships the draft head keeps it and binds through the
        ordinary load path, and one that does not drops it from the tree here so
        ``load_weights(strict=True)`` still sees an exact match instead of 58
        spurious "missing" keys.
        """
        if self.mtp_blocks and not any(str(k).startswith("mtp.") for k in weights):
            self.mtp = []
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


# ---------------------------------------------------------------------------
# MTPLX runtime binding (speculative lane)
# ---------------------------------------------------------------------------
class MTPHead(nn.Module):
    """Post-load container so ``model.mtp`` answers ``.layers``.

    Every other MTP backend is an mlx-lm model that MTPLX *grafts* a draft head
    onto, and ``mtplx.mtp_patch.validate_mtp_support`` probes that graft with
    ``model.mtp.layers``.  This backend owns its draft head natively and binds it
    from the checkpoint's own ``mtp.{i}.*`` paths, which means ``Model.mtp`` has to
    be a plain list at load time — a container would rename every tensor.  So the
    list is wrapped here *after* the weights are bound, holding the very same block
    objects (no copy, no re-load), and :attr:`Model.mtp_blocks` reads through either
    binding.  Same move ``hy_v3_mtp_patch`` makes when it aliases
    ``model.mtp.layers = [model.mtp.layer]``.
    """

    def __init__(self, blocks):
        super().__init__()
        self.layers = list(blocks)


def is_deepseek_v4_mtp_config(config: dict) -> bool:
    """Does this artifact declare a DeepSeek-V4 draft head?

    Weight presence is decided later by :meth:`Model.sanitize` (the published
    mlx-community conversions declare the layer and ship no tensors, which is what
    the runtime's degrade-to-autoregressive branch exists for).
    """
    model_type = str((config or {}).get("model_type") or "").lower()
    architectures = [str(a) for a in (config or {}).get("architectures") or []]
    if model_type != "deepseek_v4" and not any(
        a.lower() == "deepseekv4forcausallm" for a in architectures
    ):
        return False
    return int((config or {}).get("num_nextn_predict_layers") or 0) > 0


def inject_deepseek_v4_mtp_support(
    model,
    path=None,
    config: Optional[dict] = None,
    contract=None,
) -> bool:
    """Enable the speculative lane on an already-loaded DeepSeek-V4 model.

    There is nothing to graft: :class:`DeepseekV4MTP` binds through the ordinary
    load path from the checkpoint's ``mtp.0.*`` tensors, and :class:`Model` already
    carries the runtime's draft surface (``__call__(return_hidden=...)``,
    :meth:`Model.mtp_forward`, :meth:`Model.mtp_update_cache`,
    :meth:`Model.make_mtp_cache`).  All this does is publish that fact in the shape
    ``mtplx.mtp_patch.validate_mtp_support`` checks, and report False — the
    degrade-to-autoregressive signal — for a checkpoint whose draft head
    :meth:`Model.sanitize` dropped.

    Returns True when the model can speculate.  The ``path``/``config``/``contract``
    parameters exist to match the sibling ``inject_*_mtp_support`` signature the
    runtime dispatches on; a bare :class:`~mtplx.mtp_patch.MTPContract` needs no
    adaptation here, because the V4 draft input is a single defined tensor with no
    hidden-variant or concat-order choice to make.
    """
    if not is_deepseek_v4_mtp_config(config or {}):
        return False
    blocks = getattr(model, "mtp_blocks", None)
    if not blocks:
        return False
    if getattr(getattr(model, "mtp", None), "layers", None) is None:
        model.mtp = MTPHead(blocks)
    return True
