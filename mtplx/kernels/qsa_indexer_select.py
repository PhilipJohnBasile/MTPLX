"""Exact fused Metal selector for the Qwen sparse-attention indexer.

The v2.10 QSA indexer scores a pooled key block by summing the positive
per-head query/key dot products, masks blocks which are not complete at the
query position, and keeps the highest ``block_topk`` adjusted scores.  The
adjustment ``score - block_id * 1e-12`` is part of the model contract: it
makes the common all-zero ReLU tie prefer lower block ids.

This module performs that complete score/mask/tie-break/select sequence in
one custom-kernel dispatch per query chunk (one threadgroup per query row).
The lanes stream an arbitrary logical history into a float32 scratch output,
then run an eight-pass byte-radix selection over a strict 64-bit composite
key.  Work is ``O(N)`` for the scores plus ``O(8N)`` integer selection, not
an insertion-style ``O(N * K)`` scan.  A small bitonic network orders only
the at-most-512 winners by block id for deterministic epilogues.

There is deliberately no eager or CPU fallback here.  Callers decide whether
an unsupported geometry/device should use the stock MLX path; silently taking
a different selector would make graph-capture and performance receipts
ambiguous.
"""

from __future__ import annotations

import math
import os
import platform
import re
from functools import lru_cache
from typing import Literal

import mlx.core as mx

QSASelectMode = Literal["blocks", "dense_mask", "row_tokens"]

# MLX's NAX Metal float32 GEMM uses TF32-style operands, while its GEMV route is
# full fp32. The Metal source below mirrors that runtime split; float16/bfloat16
# conversions already have the truncated low bits either way.
_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16, mx.float32)
_MAX_TOPK = 512
_MAX_WIDTH = 1024

__all__ = [
    "qsa_indexer_select_blocks_metal",
    "qsa_indexer_select_dense_mask_metal",
    "qsa_indexer_select_metal",
    "qsa_indexer_select_nax_available",
    "qsa_indexer_select_row_tokens_metal",
]


def _next_power_of_two(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


def _dtype_tag(dtype: mx.Dtype) -> str:
    return {
        mx.float16: "f16",
        mx.bfloat16: "bf16",
        mx.float32: "f32",
    }[dtype]


@lru_cache(maxsize=1)
def _mlx_nax_available() -> bool:
    """Conservatively mirror MLX Metal's cached ``is_nax_available``."""

    try:
        macos_version = platform.mac_ver()[0]
    except (OSError, RuntimeError, TypeError, ValueError):
        return False

    info = None
    device_info = getattr(mx, "device_info", None)
    if callable(device_info):
        try:
            # Capability is about the Metal GPU even if a CPU parity test has
            # temporarily changed MLX's default device.
            info = device_info(mx.gpu)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            info = None
    if not isinstance(info, dict):
        # Compatibility fallback for an MLX release without mx.device_info.
        metal_device_info = getattr(mx.metal, "device_info", None)
        if callable(metal_device_info):
            try:
                info = metal_device_info()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                info = None
    architecture = info.get("architecture") if isinstance(info, dict) else None
    return _nax_available_for_platform(macos_version, architecture)


def _nax_available_for_platform(
    macos_version: str | None, architecture: str | None
) -> bool:
    """Pure parser for MLX's macOS/GPU-generation NAX availability gate.

    MLX 0.31.2 and 0.32.2 require macOS 26.2 or newer. Their architecture
    parser reads the last two digits as the generation and the final suffix as
    the device class; phone-class ``p`` needs generation 18 while base/pro,
    max, and ultra classes need generation 17. Unknown formats fail closed.
    """

    if not isinstance(macos_version, str) or not isinstance(architecture, str):
        return False
    version_match = re.match(r"^\s*(\d+)\.(\d+)(?:\.\d+)?(?:\D.*)?$", macos_version)
    if version_match is None:
        return False
    version = (int(version_match.group(1)), int(version_match.group(2)))
    if version < (26, 2):
        return False

    arch_match = re.search(r"(\d{2})([pgsd])$", architecture.lower())
    if arch_match is None:
        return False
    generation = int(arch_match.group(1))
    suffix = arch_match.group(2)
    return generation >= (18 if suffix == "p" else 17)


@lru_cache(maxsize=1)
def _mlx_tf32_enabled() -> bool:
    """Mirror MLX's NAX + ``enable_tf32`` conjunction for float32 GEMM."""

    if not _mlx_nax_available():
        return False

    raw = os.environ.get("MLX_ENABLE_TF32")
    if raw is None:
        return True
    match = re.match(r"^\s*([+-]?\d+)", raw)
    # C atoi returns zero when no numeric prefix exists. Environment mutation
    # after MLX's first matmul is unsupported by MLX itself.
    return bool(match and int(match.group(1)) != 0)


def qsa_indexer_select_nax_available() -> bool:
    """Whether MLX can use its NAX float32-matmul numerics on this host.

    Integration should fail closed to the eager selector for float32 inputs
    when this is false. Half and bfloat inputs do not need this restriction.
    """

    return _mlx_nax_available()


def _require_metal() -> None:
    if not mx.metal.is_available():
        raise RuntimeError("QSA fused selection requires an available Metal GPU")
    if mx.default_device() != mx.gpu:
        raise RuntimeError(
            "QSA fused selection requires the MLX default device to be the GPU"
        )


def _as_i32_scalar(value: int | mx.array, name: str) -> mx.array:
    """Return an int32 device scalar without synchronizing a traced value."""

    if isinstance(value, mx.array):
        if value.dtype != mx.int32 or int(value.size) != 1:
            raise TypeError(f"{name} tensor must be one int32 value")
        # MLX generates a 0-D custom-kernel input as a constant reference,
        # while the Metal body intentionally reads all dynamic scalars through
        # device pointers. Normalize both 0-D and size-one views to [1].
        return value.reshape((1,))
    return mx.array([int(value)], dtype=mx.int32)


def _static_int(value: int | mx.array | None) -> int | None:
    return None if value is None or isinstance(value, mx.array) else int(value)


def _validate(
    q: mx.array,
    pooled: mx.array,
    *,
    pos_start: int | mx.array,
    total_tokens: int | mx.array,
    logical_blocks: int | mx.array | None,
    block_topk: int,
    compress_ratio: int,
) -> tuple[int, int, int, int, int, int]:
    """Return ``(S, H, D, backing_N, K, WIDTH)`` after static validation.

    The kernel indexes through MLX-provided strides, so a cache-capacity slice
    or query chunk remains correct without an implicit contiguous-copy launch.
    """

    _require_metal()
    if q.ndim != 4 or int(q.shape[0]) != 1:
        raise ValueError(f"q must have shape [1,S,H,D], got {tuple(q.shape)}")
    if pooled.ndim != 3 or int(pooled.shape[0]) != 1:
        raise ValueError(f"pooled must have shape [1,N,D], got {tuple(pooled.shape)}")
    if q.dtype not in _SUPPORTED_DTYPES or pooled.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "q and pooled must be float16, bfloat16, or float32; "
            f"got {q.dtype} and {pooled.dtype}"
        )

    rows, heads, head_dim = map(int, q.shape[1:])
    backing_blocks, pooled_dim = map(int, pooled.shape[1:])
    if rows <= 0 or heads <= 0 or head_dim <= 0:
        raise ValueError(f"q dimensions must be positive, got {tuple(q.shape)}")
    if pooled_dim != head_dim:
        raise ValueError(
            f"q head dim {head_dim} does not match pooled dim {pooled_dim}"
        )

    topk = int(block_topk)
    ratio = int(compress_ratio)
    if not (1 <= topk <= _MAX_TOPK):
        raise ValueError(f"block_topk must be in [1,{_MAX_TOPK}], got {topk}")
    if ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {ratio}")
    start = _static_int(pos_start)
    total = _static_int(total_tokens)
    logical = backing_blocks if logical_blocks is None else _static_int(logical_blocks)
    if start is not None and start < 0:
        raise ValueError(f"pos_start must be non-negative, got {start}")
    if total is not None and total < 0:
        raise ValueError(f"total_tokens must be non-negative, got {total}")
    if start is not None and total is not None and total < start + rows:
        raise ValueError(
            "total_tokens must include every query row: "
            f"got total_tokens={total}, pos_start={start}, S={rows}"
        )
    if logical is not None:
        if not (0 <= logical <= backing_blocks):
            raise ValueError(
                f"logical_blocks must be in [0,{backing_blocks}], got {logical}"
            )
        if total is not None and logical != total // ratio:
            raise ValueError(
                "logical_blocks must equal total_tokens // compress_ratio: "
                f"got logical_blocks={logical}, T={total}, ratio={ratio}"
            )

    # WIDTH is independent of history length. A 256-bin radix histogram needs
    # at least 256 lanes; the final selected-set sort needs at least K lanes.
    width = _next_power_of_two(max(256, topk))
    if width > _MAX_WIDTH:
        raise ValueError(
            f"selector WIDTH={width} exceeds the Metal guard {_MAX_WIDTH} (K={topk})"
        )
    return rows, heads, head_dim, backing_blocks, topk, width


_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

// Total order for the block/dense output network: selected blocks by
// ascending id, followed by invalid/padded lanes.
inline bool qsa_index_before(
    uint a_index, bool a_valid, uint b_index, bool b_valid) {
    if (a_valid != b_valid) {
        return a_valid;
    }
    return a_index < b_index;
}

// MLX's Metal ArgPartition currently delegates to its stable ascending merge
// sort.  The v2.10 rows-gather lane consumes the final K indices directly, so
// its selected valid blocks appear in ascending adjusted-score order.  Equal
// values preserve input order, which is ascending block id.  Keep invalid
// lanes after valid lanes inside the network; the epilogue places the exact
// number of selected masked fillers before the winners.
inline bool qsa_row_score_before(
    float a_score,
    uint a_index,
    bool a_valid,
    float b_score,
    uint b_index,
    bool b_valid) {
    if (a_valid != b_valid) {
        return a_valid;
    }
    if (!a_valid) {
        return a_index < b_index;
    }
    bool a_nan = metal::isnan(a_score);
    bool b_nan = metal::isnan(b_score);
    if (a_nan || b_nan) {
        if (a_nan != b_nan) {
            return !a_nan;
        }
        return a_index < b_index;
    }
    if (a_score < b_score) {
        return true;
    }
    if (b_score < a_score) {
        return false;
    }
    return a_index < b_index;
}

// Monotonic IEEE-754 mapping: numerically larger finite floats produce larger
// unsigned keys. Appending the block id makes the 64-bit key unique and mirrors
// v2.10's stable ascending GPU sort followed by a final-K slice: if the
// 1e-12 adjustment itself rounds away, the later/higher id wins the cutoff.
inline uint qsa_float_order_key(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
}

inline ulong qsa_composite_key(float adjusted_score, uint block_id) {
    return (ulong(qsa_float_order_key(adjusted_score)) << 32) |
           ulong(block_id);
}

// MLX's NAX float32 GEMM path truncates each operand to a 10-bit mantissa
// before fp32 accumulation, but the legacy GEMV route remains full fp32.
// A uniform runtime mask mirrors that eager-matmul contract without adding a
// query-row or logical-history specialization to the Python kernel cache.
inline float qsa_mlx_gemm_operand(float value, uint operand_mask) {
    return as_type<float>(as_type<uint>(value) & operand_mask);
}
"""


def _epilogue(mode: QSASelectMode) -> tuple[list[str], str]:
    if mode == "blocks":
        return (
            ["block_ids", "block_valid", "adjusted_scores"],
            r"""
        if (lane < TOP_K) {
            const bool ok = my_valid;
            const size_t out_at = (size_t)row * TOP_K + lane;
            block_ids[out_at] = ok ? int(my_index) : 0;
            block_valid[out_at] = ok;
            adjusted_scores[out_at] = ok ? my_score : -INFINITY;
        }
        """,
        )
    if mode == "dense_mask":
        return (
            ["dense_mask"],
            r"""
        // First establish the visible incomplete tail (and clear both skipped
        // blocks and capacity beyond runtime T) in O(T/WIDTH) work per lane.
        for (int token = int(lane); token < int(OUTPUT_TOKENS); token += WIDTH) {
            const bool in_tail = token >= int(complete * RATIO);
            const bool causal = token < total && token <= qpos;
            dense_mask[(size_t)row * OUTPUT_TOKENS + token] =
                in_tail && causal;
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Then mark each selected complete block. Sorted selected lanes own
        // disjoint blocks, so these writes cannot race with one another.
        if (lane < TOP_K && my_valid) {
            const uint token0 = my_index * RATIO;
            for (uint within = 0; within < RATIO; ++within) {
                const uint token = token0 + within;
                if (token < OUTPUT_TOKENS && int(token) < total &&
                    int(token) <= qpos) {
                    dense_mask[(size_t)row * OUTPUT_TOKENS + token] = true;
                }
            }
        }
        """,
        )
    if mode == "row_tokens":
        return (
            ["token_ids", "token_valid"],
            r"""
        exchange_indices[lane] = my_index;
        exchange_valid[lane] = my_valid ? 1 : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        constexpr uint BLOCK_TOKEN_SLOTS = TOP_K * RATIO;
        constexpr uint ROW_TOKEN_SLOTS = BLOCK_TOKEN_SLOTS + RATIO;
        const size_t out_base = (size_t)row * ROW_TOKEN_SLOTS;
        for (uint slot = lane; slot < ROW_TOKEN_SLOTS; slot += WIDTH) {
            int token = 0;
            bool ok = false;
            if (slot < BLOCK_TOKEN_SLOTS) {
                const uint block_slot = slot / RATIO;
                const uint in_block = slot % RATIO;
                // The eager GPU argpartition is a full stable ascending sort,
                // sliced to its final min(K, logical) entries.  When a row
                // sees fewer than K complete blocks, selected -inf fillers
                // therefore precede the finite winners.  If logical<K, the
                // Python path appends the remaining padding after them.
                const uint selected_slots = metal::min(TOP_K, logical);
                const uint invalid_prefix = selected_slots - k_eff;
                const bool winner_slot =
                    block_slot >= invalid_prefix &&
                    block_slot < invalid_prefix + k_eff;
                const uint winner = block_slot >= invalid_prefix
                    ? block_slot - invalid_prefix
                    : 0u;
                ok = winner_slot && exchange_valid[winner] != 0;
                if (ok) {
                    token = int(exchange_indices[winner] * RATIO + in_block);
                }
            } else {
                const uint in_tail = slot - BLOCK_TOKEN_SLOTS;
                token = int(complete * RATIO + in_tail);
                ok = token <= qpos;
            }
            token_ids[out_base + slot] = ok ? token : 0;
            token_valid[out_base + slot] = ok;
        }
        """,
        )
    raise AssertionError(f"unknown QSA selector mode {mode!r}")


@lru_cache(maxsize=256)
def _selector_kernel(
    mode: QSASelectMode,
    heads: int,
    head_dim: int,
    blocks: int,
    topk: int,
    ratio: int,
    width: int,
    output_tokens: int,
    enable_tf32: bool,
    q_dtype: mx.Dtype,
    pooled_dtype: mx.Dtype,
):
    output_names, epilogue = _epilogue(mode)
    output_names = [*output_names, "score_scratch"]
    sqrt_dim = math.sqrt(head_dim)
    header = (
        _HEADER
        + f"""
constant constexpr uint HEADS = {heads};
constant constexpr uint HEAD_DIM = {head_dim};
constant constexpr uint BACKING_BLOCKS = {blocks};
constant constexpr uint TOP_K = {topk};
constant constexpr uint RATIO = {ratio};
constant constexpr uint WIDTH = {width};
constant constexpr uint RADIX_BINS = 256;
constant constexpr uint OUTPUT_TOKENS = {output_tokens};
constant constexpr bool ENABLE_TF32 = {str(enable_tf32).lower()};
constant constexpr bool ROW_TOKEN_MODE = {str(mode == "row_tokens").lower()};
constant constexpr bool Q_INPUT_IS_FLOAT32 = {str(q_dtype == mx.float32).lower()};
constant constexpr bool POOLED_INPUT_IS_FLOAT32 = {str(pooled_dtype == mx.float32).lower()};
constant constexpr float SQRT_HEAD_DIM = {sqrt_dim!r}f;
"""
    )

    source = (
        r"""
        const uint row = threadgroup_position_in_grid.x;
        const uint lane = thread_position_in_threadgroup.x;
        const int qpos = pos_start[0] + int(row);
        const int total = total_tokens[0];
        const int logical_value = logical_blocks[0];
        const uint logical = logical_value > 0
            ? metal::min(uint(logical_value), BACKING_BLOCKS)
            : 0u;
        // Mirror Matmul::eval_gpu's check_transpose + batch-collapse route.
        // A non-f32 astype materializes contiguous storage. For an existing
        // f32 view, check_transpose preserves recognized row/transposed
        // layouts and copies anything else. A copied operand is contiguous;
        // a copied broadcast B, however, no longer has a zero S-batch stride
        // and prevents folding S into M. When folding does not happen, M is
        // HEADS. MLX sends min(M,N)==1 to full-fp32 GEMV before NAX/TF32.
        const bool q_is_vector = HEADS == 1u;
        const bool q_kept_untransposed = Q_INPUT_IS_FLOAT32 &&
            q_strides[3] == 1u &&
            (!q_is_vector || q_strides[2] == HEAD_DIM);
        const bool q_kept_transposed = Q_INPUT_IS_FLOAT32 &&
            !q_kept_untransposed && q_strides[2] == 1u &&
            (!q_is_vector || q_strides[3] == HEADS);
        const bool q_copied =
            !Q_INPUT_IS_FLOAT32 ||
            (!q_kept_untransposed && !q_kept_transposed);
        const bool q_batch_contiguous = q_copied ||
            (q_kept_untransposed && q_strides[2] == HEAD_DIM &&
             q_strides[1] == size_t(HEADS) * HEAD_DIM);

        // pooled:[1,N,D] is swapped to B:[1,1,D,N]. For N>1 its two
        // recognized matrix layouts correspond to either original stride
        // being one. Otherwise check_transpose copies the broadcast view.
        const bool pooled_kept = !POOLED_INPUT_IS_FLOAT32 ||
            pooled_strides[1] == 1u || pooled_strides[2] == 1u;
        const bool collapse_s_into_m = q_shape[1] > 1u &&
            !q_kept_transposed && q_batch_contiguous && pooled_kept;
        const bool effective_m_gt_one =
            HEADS > 1u || collapse_s_into_m;
        const bool use_tf32_operands = ENABLE_TF32 &&
            effective_m_gt_one && logical > 1u;
        const uint gemm_operand_mask =
            use_tf32_operands ? 0xffffe000u : 0xffffffffu;
        const int complete_value = (qpos + 1) / int(RATIO);
        const uint complete = complete_value > 0 ? uint(complete_value) : 0u;
        const uint valid_count = metal::min(logical, complete);

        threadgroup float exchange_scores[WIDTH];
        threadgroup uint exchange_indices[WIDTH];
        threadgroup uchar exchange_valid[WIDTH];
        threadgroup atomic_uint radix_histogram[RADIX_BINS];
        threadgroup atomic_uint selected_count;
        threadgroup ulong radix_prefix;
        threadgroup uint radix_rank;
        threadgroup ulong threshold_key;

        // Score every visible complete block once. The scratch output keeps
        // those fp32 adjusted scores available to all eight radix passes
        // without recomputing HEADS*HEAD_DIM dot products. Only the visible
        // prefix is written/read; backing capacity does not tax an early row.
        const size_t scratch_base = (size_t)row * BACKING_BLOCKS;
        for (uint block = lane; block < valid_count; block += WIDTH) {
            float score_sum = 0.0f;
            for (uint head = 0; head < HEADS; ++head) {
                float dot = 0.0f;
                const size_t q_base =
                    (size_t)row * q_strides[1] +
                    (size_t)head * q_strides[2];
                const size_t pooled_base =
                    (size_t)block * pooled_strides[1];
                for (uint dim = 0; dim < HEAD_DIM; ++dim) {
                    const float q_value = qsa_mlx_gemm_operand(
                        float(q[q_base + (size_t)dim * q_strides[3]]),
                        gemm_operand_mask);
                    const float k_value = qsa_mlx_gemm_operand(
                        float(pooled[
                            pooled_base + (size_t)dim * pooled_strides[2]]),
                        gemm_operand_mask);
                    dot += q_value * k_value;
                }
                score_sum += metal::max(dot, 0.0f);
            }
            const float score = score_sum / SQRT_HEAD_DIM;
            const float adjusted = score - float(block) * 1.0e-12f;
            score_scratch[scratch_base + block] = adjusted;
        }
        threadgroup_barrier(
            mem_flags::mem_threadgroup | mem_flags::mem_device);

        // Select the Kth largest strict composite key a byte at a time. Each
        // pass scans only candidates matching the already-selected high-byte
        // prefix. The rank is relative to that prefix bucket.
        const uint k_eff = metal::min(TOP_K, valid_count);
        if (lane == 0) {
            radix_prefix = 0ul;
            radix_rank = k_eff > 0 ? k_eff - 1 : 0;
            threshold_key = 0xfffffffffffffffful;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (k_eff > 0) {
            for (uint pass = 0; pass < 8; ++pass) {
                if (lane < RADIX_BINS) {
                    atomic_store_explicit(
                        &radix_histogram[lane], 0u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint shift = 56u - pass * 8u;
                const ulong prefix = radix_prefix;
                for (uint block = lane; block < valid_count; block += WIDTH) {
                    const float adjusted = score_scratch[scratch_base + block];
                    const ulong key = qsa_composite_key(adjusted, block);
                    bool prefix_matches = true;
                    if (pass > 0) {
                        prefix_matches = (key >> (shift + 8u)) == prefix;
                    }
                    if (prefix_matches) {
                        const uint digit = uint((key >> shift) & 0xfful);
                        atomic_fetch_add_explicit(
                            &radix_histogram[digit], 1u, memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lane == 0) {
                    uint rank = radix_rank;
                    uint chosen = 0;
                    for (int digit = 255; digit >= 0; --digit) {
                        const uint count = atomic_load_explicit(
                            &radix_histogram[uint(digit)], memory_order_relaxed);
                        if (rank < count) {
                            chosen = uint(digit);
                            break;
                        }
                        rank -= count;
                    }
                    radix_prefix = (radix_prefix << 8) | ulong(chosen);
                    radix_rank = rank;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) {
                threshold_key = radix_prefix;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compact the exact winners into WIDTH-sized threadgroup storage. The
        // atomic arrival order is irrelevant: the following bitonic network
        // establishes the mode's deterministic output order.
        exchange_scores[lane] = -INFINITY;
        exchange_indices[lane] = 0xffffffffu;
        exchange_valid[lane] = 0;
        if (lane == 0) {
            atomic_store_explicit(
                &selected_count, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_eff > 0) {
            const ulong threshold = threshold_key;
            for (uint block = lane; block < valid_count; block += WIDTH) {
                const float adjusted = score_scratch[scratch_base + block];
                if (qsa_composite_key(adjusted, block) >= threshold) {
                    const uint slot = atomic_fetch_add_explicit(
                        &selected_count, 1u, memory_order_relaxed);
                    if (slot < TOP_K) {
                        exchange_scores[slot] = adjusted;
                        exchange_indices[slot] = block;
                        exchange_valid[slot] = 1;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint my_index = exchange_indices[lane];
        bool my_valid = exchange_valid[lane] != 0;
        float my_score = exchange_scores[lane];
        for (uint sequence = 2; sequence <= WIDTH; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                exchange_scores[lane] = my_score;
                exchange_indices[lane] = my_index;
                exchange_valid[lane] = my_valid ? 1 : 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint partner = lane ^ stride;
                const float other_score = exchange_scores[partner];
                const uint other_index = exchange_indices[partner];
                const bool other_valid = exchange_valid[partner] != 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const bool is_lower = (lane & stride) == 0;
                const float a_score = is_lower ? my_score : other_score;
                const uint a_index = is_lower ? my_index : other_index;
                const bool a_valid = is_lower ? my_valid : other_valid;
                const float b_score = is_lower ? other_score : my_score;
                const uint b_index = is_lower ? other_index : my_index;
                const bool b_valid = is_lower ? other_valid : my_valid;

                const bool lower_wants_before = (lane & sequence) == 0;
                const bool b_before_a = ROW_TOKEN_MODE
                    ? qsa_row_score_before(
                        b_score, b_index, b_valid,
                        a_score, a_index, a_valid)
                    : qsa_index_before(
                        b_index, b_valid, a_index, a_valid);
                const bool a_before_b = ROW_TOKEN_MODE
                    ? qsa_row_score_before(
                        a_score, a_index, a_valid,
                        b_score, b_index, b_valid)
                    : qsa_index_before(
                        a_index, a_valid, b_index, b_valid);
                const bool swap = lower_wants_before ? b_before_a : a_before_b;
                if (swap) {
                    my_score = is_lower ? b_score : a_score;
                    my_index = is_lower ? b_index : a_index;
                    my_valid = is_lower ? b_valid : a_valid;
                }
            }
        }
    """
        + epilogue
    )

    name = (
        f"mtplx_qsa_select_{mode}_h{heads}_d{head_dim}_n{blocks}_"
        f"k{topk}_r{ratio}_w{width}_t{output_tokens}_tf{int(enable_tf32)}_"
        f"{_dtype_tag(q_dtype)}_"
        f"{_dtype_tag(pooled_dtype)}"
    )
    return mx.fast.metal_kernel(
        name=name,
        input_names=[
            "q",
            "pooled",
            "pos_start",
            "total_tokens",
            "logical_blocks",
        ],
        output_names=output_names,
        header=header,
        source=source,
        # q_strides/pooled_strides are consumed in the source. Avoid allowing
        # the wrapper to insert a hidden contiguous-copy dispatch.
        ensure_row_contiguous=False,
    )


def qsa_indexer_select_metal(
    q: mx.array,
    pooled: mx.array,
    *,
    pos_start: int | mx.array,
    total_tokens: int | mx.array,
    block_topk: int,
    compress_ratio: int,
    logical_blocks: int | mx.array | None = None,
    output_total_tokens: int | None = None,
    mode: QSASelectMode = "blocks",
):
    """Run one exact fused selector dispatch in the requested output mode.

    Args:
        q: Post-normalization/post-RoPE queries ``[1,S,H,D]``.
        pooled: Pooled indexer backing keys ``[1,capacity,D]``.
        pos_start: Absolute position of query row zero, either a Python int or
            a one-element int32 MLX array. ``total_tokens`` may extend beyond
            this query chunk, which preserves chunked operation.
        total_tokens: Logical KV length, with the same scalar forms.
        block_topk: Fixed complete-block budget, guarded at ``<= 512``.
        compress_ratio: Tokens represented by one pooled block.
        logical_blocks: Valid pooled prefix. Defaults to the full backing
            shape. A one-element int32 array keeps this frontier dynamic under
            ``mx.compile`` without a host synchronization.
        output_total_tokens: Static dense-mask output width when
            ``total_tokens`` is a tensor. It is a capacity: runtime positions
            at or beyond ``total_tokens`` are written false. Omit it when a
            Python ``total_tokens`` should define the exact output width.
        mode: ``"blocks"`` returns ascending selected block ids, validity,
            and adjusted scores; ``"dense_mask"`` returns
            ``[1,1,S,output_total_tokens]``;
            ``"row_tokens"`` returns fixed ``[S,K*r+r]`` token ids/validity.

    The three modes all fuse scoring, ReLU, causal complete-block masking,
    tie adjustment, and exact top-k selection into their single dispatch.
    """

    if mode not in ("blocks", "dense_mask", "row_tokens"):
        raise ValueError(f"unknown QSA selector mode {mode!r}")
    rows, heads, head_dim, blocks, topk, width = _validate(
        q,
        pooled,
        pos_start=pos_start,
        total_tokens=total_tokens,
        logical_blocks=logical_blocks,
        block_topk=block_topk,
        compress_ratio=compress_ratio,
    )
    ratio = int(compress_ratio)
    kernel_output_tokens = 0

    if mode == "blocks":
        output_shapes = [(rows, topk), (rows, topk), (rows, topk)]
        output_dtypes = [mx.int32, mx.bool_, mx.float32]
    elif mode == "dense_mask":
        static_total = _static_int(total_tokens)
        dense_width = (
            static_total if output_total_tokens is None else int(output_total_tokens)
        )
        if dense_width is None:
            raise ValueError(
                "output_total_tokens is required for a tensor total_tokens "
                "in dense_mask mode"
            )
        if dense_width < 0:
            raise ValueError(
                f"output_total_tokens must be non-negative, got {dense_width}"
            )
        if static_total is not None and dense_width < static_total:
            raise ValueError(
                f"output_total_tokens={dense_width} is smaller than "
                f"total_tokens={static_total}"
            )
        kernel_output_tokens = dense_width
        output_shapes = [(1, 1, rows, dense_width)]
        output_dtypes = [mx.bool_]
    else:
        token_width = topk * ratio + ratio
        output_shapes = [(rows, token_width), (rows, token_width)]
        output_dtypes = [mx.int32, mx.bool_]

    kernel = _selector_kernel(
        mode,
        heads,
        head_dim,
        blocks,
        topk,
        ratio,
        width,
        kernel_output_tokens,
        _mlx_tf32_enabled(),
        q.dtype,
        pooled.dtype,
    )
    output_shapes.append((rows, blocks))
    output_dtypes.append(mx.float32)
    pos_scalar = _as_i32_scalar(pos_start, "pos_start")
    total_scalar = _as_i32_scalar(total_tokens, "total_tokens")
    logical_scalar = _as_i32_scalar(
        blocks if logical_blocks is None else logical_blocks,
        "logical_blocks",
    )
    outputs = kernel(
        inputs=[q, pooled, pos_scalar, total_scalar, logical_scalar],
        grid=(rows * width, 1, 1),
        threadgroup=(width, 1, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
    # The final output is device scratch owned by this single dispatch. It is
    # intentionally not part of the public selection contract.
    return outputs[0] if mode == "dense_mask" else tuple(outputs[:-1])


def qsa_indexer_select_blocks_metal(
    q: mx.array,
    pooled: mx.array,
    *,
    pos_start: int | mx.array,
    total_tokens: int | mx.array,
    block_topk: int,
    compress_ratio: int,
    logical_blocks: int | mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return ascending block ids, validity, and adjusted scores ``[S,K]``."""

    return qsa_indexer_select_metal(
        q,
        pooled,
        pos_start=pos_start,
        total_tokens=total_tokens,
        block_topk=block_topk,
        compress_ratio=compress_ratio,
        logical_blocks=logical_blocks,
        mode="blocks",
    )


def qsa_indexer_select_dense_mask_metal(
    q: mx.array,
    pooled: mx.array,
    *,
    pos_start: int | mx.array,
    total_tokens: int | mx.array,
    block_topk: int,
    compress_ratio: int,
    logical_blocks: int | mx.array | None = None,
    output_total_tokens: int | None = None,
) -> mx.array:
    """Return the exact sparse-plus-tail causal mask ``[1,1,S,T]``."""

    return qsa_indexer_select_metal(
        q,
        pooled,
        pos_start=pos_start,
        total_tokens=total_tokens,
        block_topk=block_topk,
        compress_ratio=compress_ratio,
        logical_blocks=logical_blocks,
        output_total_tokens=output_total_tokens,
        mode="dense_mask",
    )


def qsa_indexer_select_row_tokens_metal(
    q: mx.array,
    pooled: mx.array,
    *,
    pos_start: int | mx.array,
    total_tokens: int | mx.array,
    block_topk: int,
    compress_ratio: int,
    logical_blocks: int | mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Return fixed-width per-row token ids/validity ``[S,K*r+r]``."""

    return qsa_indexer_select_metal(
        q,
        pooled,
        pos_start=pos_start,
        total_tokens=total_tokens,
        block_topk=block_topk,
        compress_ratio=compress_ratio,
        logical_blocks=logical_blocks,
        mode="row_tokens",
    )
