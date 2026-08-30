"""Fused preparation kernels for the Qwen sparse-attention indexer.

The QSA indexer has two small but dispatch-heavy preparation chains around its
large score/top-k operation:

* projected queries: RMSNorm -> partial non-interleaved RoPE;
* completed raw-key blocks: fp32 mean -> input-dtype round -> RMSNorm ->
  partial non-interleaved RoPE.

Each public helper below expresses its complete chain as one Metal dispatch.
They deliberately accept already-projected rows.  This is the same boundary as
vLLM's fused indexer-Q kernel and, importantly for MTPLX, also works when the
indexer projection is packed into the attention layer's shared quantized QKV
projection.

There is no silent eager fallback in this module.  The model owns the fail-
closed eligibility check and keeps its stock MLX implementation as the oracle.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx

_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16, mx.float32)
_SIMD = 32
_RMS_READS = 4
_MAX_EXACT_HEAD_DIM = _SIMD * _RMS_READS

__all__ = [
    "qsa_indexer_pool_keys_metal",
    "qsa_indexer_prepare_queries_metal",
    "qsa_indexer_prepare_supported",
]


def _require_metal() -> None:
    if not mx.metal.is_available():
        raise RuntimeError("QSA indexer preparation requires an available Metal GPU")
    if mx.default_device() != mx.gpu:
        raise RuntimeError(
            "QSA indexer preparation requires the MLX default device to be the GPU"
        )


def _as_i32_scalar(value: int | mx.array, name: str) -> mx.array:
    """Normalize a host or traced scalar without synchronizing an array."""

    if isinstance(value, mx.array):
        if value.dtype != mx.int32 or int(value.size) != 1:
            raise TypeError(f"{name} tensor must be one int32 value")
        return value.reshape((1,))
    return mx.array([int(value)], dtype=mx.int32)


def _dtype_tag(dtype: mx.Dtype) -> str:
    return {
        mx.float16: "f16",
        mx.bfloat16: "bf16",
        mx.float32: "f32",
    }[dtype]


def _attention_scaling(value: float) -> float:
    scaling = float(value)
    if not math.isfinite(scaling) or scaling <= 0.0:
        raise ValueError(
            "attention_scaling must be finite and positive; "
            f"got {scaling}"
        )
    return scaling


def _float_tag(value: float) -> str:
    return (
        format(float(value), ".9g")
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "d")
    )


def _validate_common(
    values: mx.array,
    norm_weight: mx.array,
    inv_freq: mx.array,
    *,
    expected_ndim: int,
) -> tuple[int, int]:
    _require_metal()
    if values.ndim != expected_ndim or int(values.shape[0]) != 1:
        shape = "[1,S,H,D]" if expected_ndim == 4 else "[1,T,D]"
        raise ValueError(f"values must have shape {shape}, got {tuple(values.shape)}")
    if values.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"values must be float16, bfloat16, or float32; got {values.dtype}"
        )
    if norm_weight.ndim != 1 or int(norm_weight.shape[0]) != int(values.shape[-1]):
        raise ValueError(
            "norm_weight must be one head-dimension vector; got "
            f"{tuple(norm_weight.shape)} for D={int(values.shape[-1])}"
        )
    if norm_weight.dtype != values.dtype:
        raise TypeError(
            "exact fused RMSNorm requires values and norm_weight to share a dtype; "
            f"got {values.dtype} and {norm_weight.dtype}"
        )
    if inv_freq.ndim != 1 or inv_freq.dtype != mx.float32:
        raise TypeError(
            "inv_freq must be a one-dimensional float32 array; "
            f"got shape={tuple(inv_freq.shape)}, dtype={inv_freq.dtype}"
        )

    head_dim = int(values.shape[-1])
    rotary_dim = 2 * int(inv_freq.shape[0])
    if not (0 < head_dim <= _MAX_EXACT_HEAD_DIM):
        raise ValueError(
            f"head_dim must be in [1,{_MAX_EXACT_HEAD_DIM}], got {head_dim}"
        )
    if not (0 < rotary_dim <= head_dim) or rotary_dim % 2:
        raise ValueError(
            "rotary_dim=2*len(inv_freq) must be positive, even, and no larger "
            f"than head_dim; got rotary_dim={rotary_dim}, head_dim={head_dim}"
        )
    return head_dim, rotary_dim


def qsa_indexer_prepare_supported(
    values: mx.array,
    norm_weight: mx.array,
    inv_freq: mx.array,
    *,
    expected_ndim: int,
) -> bool:
    """Return the static eligibility result without hiding kernel failures."""

    try:
        _validate_common(
            values,
            norm_weight,
            inv_freq,
            expected_ndim=expected_ndim,
        )
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


@lru_cache(maxsize=128)
def _prepare_queries_kernel(
    heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    attention_scaling: float,
    dtype: mx.Dtype,
):
    half_rotary = rotary_dim // 2
    header = f"""
        #include <metal_stdlib>
        using namespace metal;
        constant constexpr uint HEADS = {heads};
        constant constexpr uint HEAD_DIM = {head_dim};
        constant constexpr uint ROTARY_DIM = {rotary_dim};
        constant constexpr uint HALF_ROTARY = {half_rotary};
        constant constexpr float RMS_EPS = {float(eps)!r}f;
        constant constexpr float ROPE_ATTENTION_SCALE = {float(attention_scaling)!r}f;
    """
    source = r"""
        const uint row_head = threadgroup_position_in_grid.x;
        const uint lane = thread_index_in_simdgroup;
        const uint row = row_head / HEADS;
        const uint head = row_head - row * HEADS;
        const size_t src_base =
            (size_t)row * raw_q_strides[1] +
            (size_t)head * raw_q_strides[2];
        const size_t out_base =
            ((size_t)row * HEADS + head) * HEAD_DIM;

        // MLX rms_single_row uses 32 lanes and four contiguous values per
        // lane for every axis up through 128.  Reproduce that reduction and
        // its cast-before-weight ordering exactly.
        float square_sum = 0.0f;
        const uint lane_base = lane * 4u;
        for (uint i = 0; i < 4u; ++i) {
            const uint dim = lane_base + i;
            if (dim < HEAD_DIM) {
                const float value = float(raw_q[
                    src_base + (size_t)dim * raw_q_strides[3]]);
                square_sum += value * value;
            }
        }
        square_sum = simd_sum(square_sum);
        const float inverse_rms = metal::precise::rsqrt(
            square_sum / float(HEAD_DIM) + RMS_EPS);
        const float position = float(pos_start[0] + int(row));

        // Non-interleaved/rotate-half partial RoPE, matching
        // _rope_cos_sin + _apply_partial_rope.  mx.cos/mx.sin use the Metal
        // precise variants (unlike mx.fast.rope, which intentionally uses
        // fast trig), so this kernel does too.
        for (uint pair = lane; pair < HALF_ROTARY; pair += 32u) {
            const size_t first_at =
                src_base + (size_t)pair * raw_q_strides[3];
            const size_t second_at = src_base +
                (size_t)(pair + HALF_ROTARY) * raw_q_strides[3];
            const T first_norm = norm_weight[
                (size_t)pair * norm_weight_strides[0]] *
                static_cast<T>(float(raw_q[first_at]) * inverse_rms);
            const T second_norm = norm_weight[
                (size_t)(pair + HALF_ROTARY) * norm_weight_strides[0]] *
                static_cast<T>(float(raw_q[second_at]) * inverse_rms);
            const float theta = position * float(inv_freq[
                (size_t)pair * inv_freq_strides[0]]);
            const float cosine =
                metal::precise::cos(theta) * ROPE_ATTENTION_SCALE;
            const float sine =
                metal::precise::sin(theta) * ROPE_ATTENTION_SCALE;
            const float first = float(first_norm);
            const float second = float(second_norm);
            // Keep the products as distinct fp32 operations.  The stock
            // _apply_partial_rope graph rounds both multiplies before its
            // add/subtract; allowing Metal to contract this expression into
            // an FMA changes a handful of bf16 cutoff values at large
            // positions (and can therefore perturb an exact top-k set).
            const float first_cosine = first * cosine;
            const float second_sine = second * sine;
            const float second_cosine = second * cosine;
            const float first_sine = first * sine;
            prepared_q[out_base + pair] =
                static_cast<T>(first_cosine - second_sine);
            prepared_q[out_base + pair + HALF_ROTARY] =
                static_cast<T>(second_cosine + first_sine);
        }

        for (uint dim = ROTARY_DIM + lane; dim < HEAD_DIM; dim += 32u) {
            const T normalized = norm_weight[
                (size_t)dim * norm_weight_strides[0]] *
                static_cast<T>(float(raw_q[
                    src_base + (size_t)dim * raw_q_strides[3]]) * inverse_rms);
            prepared_q[out_base + dim] = normalized;
        }
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_qsa_prepare_q_h{heads}_d{head_dim}_r{rotary_dim}_"
            f"s{_float_tag(attention_scaling)}_{_dtype_tag(dtype)}"
        ),
        input_names=["raw_q", "norm_weight", "inv_freq", "pos_start"],
        output_names=["prepared_q"],
        header=header,
        source=source,
        ensure_row_contiguous=False,
    )


def qsa_indexer_prepare_queries_metal(
    raw_q: mx.array,
    norm_weight: mx.array,
    inv_freq: mx.array,
    *,
    pos_start: int | mx.array,
    eps: float,
    attention_scaling: float = 1.0,
) -> mx.array:
    """Fuse query RMSNorm and partial RoPE into one Metal dispatch.

    ``raw_q`` and the result have shape ``[1,S,H,D]``.  ``pos_start`` may be
    a one-element int32 array so a compiled graph can replay at new absolute
    positions without baking an offset into its trace.
    """

    head_dim, rotary_dim = _validate_common(
        raw_q,
        norm_weight,
        inv_freq,
        expected_ndim=4,
    )
    rows = int(raw_q.shape[1])
    heads = int(raw_q.shape[2])
    if rows <= 0 or heads <= 0:
        raise ValueError(f"raw_q dimensions must be positive, got {tuple(raw_q.shape)}")
    start = _as_i32_scalar(pos_start, "pos_start")
    rope_scale = _attention_scaling(attention_scaling)
    kernel = _prepare_queries_kernel(
        heads,
        head_dim,
        rotary_dim,
        float(eps),
        rope_scale,
        raw_q.dtype,
    )
    return kernel(
        inputs=[raw_q, norm_weight, inv_freq, start],
        template=[("T", raw_q.dtype)],
        grid=(rows * heads * _SIMD, 1, 1),
        threadgroup=(_SIMD, 1, 1),
        output_shapes=[tuple(raw_q.shape)],
        output_dtypes=[raw_q.dtype],
    )[0]


@lru_cache(maxsize=128)
def _pool_keys_kernel(
    head_dim: int,
    rotary_dim: int,
    ratio: int,
    eps: float,
    attention_scaling: float,
    dtype: mx.Dtype,
):
    half_rotary = rotary_dim // 2
    header = f"""
        #include <metal_stdlib>
        using namespace metal;
        constant constexpr uint HEAD_DIM = {head_dim};
        constant constexpr uint ROTARY_DIM = {rotary_dim};
        constant constexpr uint HALF_ROTARY = {half_rotary};
        constant constexpr uint RATIO = {ratio};
        constant constexpr float RMS_EPS = {float(eps)!r}f;
        constant constexpr float ROPE_ATTENTION_SCALE = {float(attention_scaling)!r}f;
    """
    source = r"""
        const uint block = threadgroup_position_in_grid.x;
        const uint lane = thread_index_in_simdgroup;
        const uint lane_base = lane * 4u;
        threadgroup float rounded_means[HEAD_DIM];

        // The stock path reduces raw keys in float32, divides by the block
        // width, then casts back to the raw-key dtype before RMSNorm.
        float square_sum = 0.0f;
        for (uint i = 0; i < 4u; ++i) {
            const uint dim = lane_base + i;
            if (dim < HEAD_DIM) {
                float sum = 0.0f;
                for (uint within = 0; within < RATIO; ++within) {
                    const uint token = block * RATIO + within;
                    sum += float(raw_keys[
                        (size_t)token * raw_keys_strides[1] +
                        (size_t)dim * raw_keys_strides[2]]);
                }
                const T rounded = static_cast<T>(sum / float(RATIO));
                const float mean_value = float(rounded);
                rounded_means[dim] = mean_value;
                square_sum += mean_value * mean_value;
            }
        }
        square_sum = simd_sum(square_sum);
        const float inverse_rms = metal::precise::rsqrt(
            square_sum / float(HEAD_DIM) + RMS_EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const size_t out_base = (size_t)block * HEAD_DIM;
        const float position = float(
            (block_start[0] + int(block)) * int(RATIO));
        for (uint pair = lane; pair < HALF_ROTARY; pair += 32u) {
            const T first_norm = norm_weight[
                (size_t)pair * norm_weight_strides[0]] *
                static_cast<T>(rounded_means[pair] * inverse_rms);
            const T second_norm = norm_weight[
                (size_t)(pair + HALF_ROTARY) * norm_weight_strides[0]] *
                static_cast<T>(
                    rounded_means[pair + HALF_ROTARY] * inverse_rms);
            const float theta = position * float(inv_freq[
                (size_t)pair * inv_freq_strides[0]]);
            const float cosine =
                metal::precise::cos(theta) * ROPE_ATTENTION_SCALE;
            const float sine =
                metal::precise::sin(theta) * ROPE_ATTENTION_SCALE;
            const float first = float(first_norm);
            const float second = float(second_norm);
            const float first_cosine = first * cosine;
            const float second_sine = second * sine;
            const float second_cosine = second * cosine;
            const float first_sine = first * sine;
            pooled[out_base + pair] =
                static_cast<T>(first_cosine - second_sine);
            pooled[out_base + pair + HALF_ROTARY] =
                static_cast<T>(second_cosine + first_sine);
        }
        for (uint dim = ROTARY_DIM + lane; dim < HEAD_DIM; dim += 32u) {
            pooled[out_base + dim] = norm_weight[
                (size_t)dim * norm_weight_strides[0]] *
                static_cast<T>(rounded_means[dim] * inverse_rms);
        }
    """
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_qsa_pool_k_d{head_dim}_r{rotary_dim}_c{ratio}_"
            f"s{_float_tag(attention_scaling)}_{_dtype_tag(dtype)}"
        ),
        input_names=["raw_keys", "norm_weight", "inv_freq", "block_start"],
        output_names=["pooled"],
        header=header,
        source=source,
        ensure_row_contiguous=False,
    )


def qsa_indexer_pool_keys_metal(
    raw_keys: mx.array,
    norm_weight: mx.array,
    inv_freq: mx.array,
    *,
    block_start: int | mx.array,
    compress_ratio: int,
    eps: float,
    attention_scaling: float = 1.0,
) -> mx.array:
    """Fuse completed-block mean, RMSNorm, and partial RoPE.

    ``raw_keys`` is the exact completed range ``[1,N*ratio,D]`` and the result
    is ``[1,N,D]``.  Output row zero represents absolute block
    ``block_start``.  A tensor block start keeps the absolute position dynamic
    under ``mx.compile``.
    """

    head_dim, rotary_dim = _validate_common(
        raw_keys,
        norm_weight,
        inv_freq,
        expected_ndim=3,
    )
    ratio = int(compress_ratio)
    if ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {ratio}")
    token_count = int(raw_keys.shape[1])
    if token_count <= 0 or token_count % ratio:
        raise ValueError(
            "raw_keys token count must be a positive multiple of "
            f"compress_ratio={ratio}; got {token_count}"
        )
    blocks = token_count // ratio
    start = _as_i32_scalar(block_start, "block_start")
    rope_scale = _attention_scaling(attention_scaling)
    kernel = _pool_keys_kernel(
        head_dim,
        rotary_dim,
        ratio,
        float(eps),
        rope_scale,
        raw_keys.dtype,
    )
    return kernel(
        inputs=[raw_keys, norm_weight, inv_freq, start],
        template=[("T", raw_keys.dtype)],
        grid=(blocks * _SIMD, 1, 1),
        threadgroup=(_SIMD, 1, 1),
        output_shapes=[(1, blocks, head_dim)],
        output_dtypes=[raw_keys.dtype],
    )[0]
