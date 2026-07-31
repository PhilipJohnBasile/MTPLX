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

Milestone status (see the port's job card / report):
  * M1 (this commit): architecture study + importable skeleton; MoE/gate/shared-expert
    + HC + RoPE + o-LoRA math transcribed.  The four new-math components are
    implemented but NOT yet numerically gated.
  * M2: unit-gate HCA/CSA/o-LoRA/hash against the reference on small synthetic tensors.
  * M3: load the real 4-bit weights and gate first-token logits against the reference.
  * M4: register ``deepseek-v4`` in ``mtplx/backends/registry.py`` so ``mtplx serve``
    can resolve the load path.

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


def _causal_window_mask(seqlen: int, window: int, dtype=mx.float32) -> mx.array:
    """Additive ``[s, s]`` mask: 0 where token j is attendable from query i (causal,
    within ``window`` positions), else a large negative.  Sliding window matches the
    reference's ``window_size`` sparse gather at the dense-scaffold level.
    """
    i = mx.arange(seqlen)[:, None]
    j = mx.arange(seqlen)[None, :]
    allowed = (j <= i) & (j > i - window)
    neg = mx.array(mx.finfo(dtype).min, dtype)
    return mx.where(allowed, mx.array(0.0, dtype), neg)


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
    The incremental single-token decode state machine (kv_state/score_state buffers) is
    a separate M3 deliverable; this class implements the prefill pooling only.
    """

    def __init__(self, args: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.dim = args.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
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

    def __call__(self, x: mx.array) -> mx.array:
        """Non-overlap (ratio != 4) prefill pooling, ``start_pos == 0``.

        NOTE(M3): overlapping windows (ratio==4) fold in the previous window's
        second half, and the single-token decode state machine (kv_state /
        score_state) is separate.  This path is the one the M2 gate verifies.
        """
        b, s, _ = x.shape
        ratio = self.compress_ratio
        d = self.head_dim
        rd = self.rope_head_dim
        xf = x.astype(mx.float32)
        kv = self.wkv(xf)
        score = self.wgate(xf)
        cutoff = s - (s % ratio)
        nwin = cutoff // ratio
        kv = kv[:, :cutoff].reshape(b, nwin, ratio, -1)
        score = score[:, :cutoff].reshape(b, nwin, ratio, -1) + self.ape
        pooled = mx.sum(kv * mx.softmax(score, axis=2), axis=2)  # [b, nwin, coff*d]
        if self.overlap:
            pooled = pooled[..., :d]
        pooled = self.norm(pooled)
        # rope tail at window positions [0, ratio, 2*ratio, ...]
        win_pos = mx.arange(nwin, dtype=mx.float32) * ratio
        ang = win_pos[:, None] * self._inv_freq[None, :]
        cos, sin = mx.cos(ang), mx.sin(ang)
        head = pooled[..., :-rd]
        tail = _apply_interleaved_rope(pooled[..., -rd:], cos[None], sin[None])
        return mx.concatenate([head, tail], axis=-1)


class Indexer(nn.Module):
    """Sparse-position selector for ``compress_ratio==4`` layers (reference
    ``Indexer``, model.py L380-433).  Has its own compressor (Hadamard-rotated in the
    reference) plus ``wq_b``/``weights_proj``; scores compressed positions and returns
    the top-``index_topk`` to attend.

    NOTE(M3): the full top-k selection + Hadamard rotation + FP4 QAT are integrated in
    M3.  The submodule tree (wq_b, weights_proj, compressor) is defined here so the
    checkpoint loads; a dense fallback is used until the sparse path is gated.
    """

    def __init__(self, args: ModelArgs, compress_ratio: int):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5
        self.compressor = Compressor(args, compress_ratio, self.head_dim)


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
        # wo_a.weight: [g*r, per]  ->  grouped [g, r, per]; batched matmul over g.
        w = self.wo_a.weight.reshape(g, r, per)
        # og: [b,s,g,per] ; want out[b,s,g,r] = sum_p og[...,g,p]*w[g,r,p]
        out = mx.einsum("bsgp,grp->bsgr", og, w)
        out = out.reshape(b, s, g * r)
        return self.wo_b(out)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        b, s, _ = x.shape
        rd = self.rope_head_dim
        positions = mx.arange(s)  # NOTE(M3): + cache.offset for decode
        cos, sin = self._rope_tables(positions)

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim)
        # per-head RMS-like normalisation (no learned weight), reference L498
        q = q * mx.rsqrt(mx.mean(mx.square(q.astype(mx.float32)), axis=-1, keepdims=True) + self.eps)
        q = q.astype(x.dtype)
        q_head = q[..., :-rd]
        q_tail = _apply_interleaved_rope(q[..., -rd:], cos[None, :, None, :], sin[None, :, None, :])
        q = mx.concatenate([q_head, q_tail], axis=-1)

        kv = self.kv_norm(self.wkv(x)).reshape(b, s, 1, self.head_dim)
        kv_head = kv[..., :-rd]
        kv_tail = _apply_interleaved_rope(kv[..., -rd:], cos[None, :, None, :], sin[None, :, None, :])
        kv = mx.concatenate([kv_head, kv_tail], axis=-1)  # [b,s,1,head_dim]

        # NOTE(M3): dense (non-sparse) sliding-window attention as the scaffold path.
        # The sparse top-k gather (window + compressed KV via Indexer/strided idx),
        # the compressor's second cache, and the streaming decode cache are M3.
        q_t = q.transpose(0, 2, 1, 3)  # [b,h,s,hd]
        k_t = mx.broadcast_to(kv.transpose(0, 2, 1, 3), (b, self.n_heads, s, self.head_dim))
        scores = (q_t * self.softmax_scale) @ k_t.transpose(0, 1, 3, 2)  # [b,h,s,s]
        if mask is not None:
            scores = scores + mask
        # attn_sink: per-head learned logit appended to the softmax denominator.
        sink = self.attn_sink.reshape(1, self.n_heads, 1, 1)
        m = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink)
        ex = mx.exp(scores - m)
        denom = mx.sum(ex, axis=-1, keepdims=True) + mx.exp(sink - m)
        attn = ex / denom
        o = attn @ k_t  # [b,h,s,head_dim]
        o = o.transpose(0, 2, 1, 3)  # [b,s,h,head_dim]
        # de-rotate the tail dims (reference L534, inverse rope)
        o_head = o[..., :-rd]
        o_tail = _apply_interleaved_rope(o[..., -rd:], cos[None, :, None, :], -sin[None, :, None, :])
        o = mx.concatenate([o_head, o_tail], axis=-1)
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
        # NOTE(M3): dense sliding-window causal mask stands in for the reference's
        # window + compressed-KV sparse top-k gather.
        mask = _causal_window_mask(h.shape[1], self.args.window_size, h.dtype)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c, input_ids=input_ids)
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
        # NOTE(M3): real sliding-window + compressed KV caches. Placeholder for now.
        return [None] * len(self.layers)
