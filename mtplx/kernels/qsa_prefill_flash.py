"""TensorOps block-sparse attention for Qwen4Exp QSA prefill.

This is the attention consumer for the indexer's per-query ``block_ids`` and
``block_valid`` outputs.  It reads the full K/V cache backing arrays in place,
streams only the selected complete four-token blocks, and then includes the
query row's zero-to-three-token visible causal tail.  It never constructs a
dense ``[S, T]`` mask and never gathers a per-row K/V tensor.

The production geometry has an especially useful reuse axis: all twelve query
heads in one GQA group share both a KV head and the indexer's selection.  One
simdgroup therefore owns ``(query row, KV head)`` and treats those heads as the
live rows of a padded M=16 matrix. Each selected N=32 token tile is gathered
once into one 16 KiB threadgroup allocation. Metal 4 MPP ``matmul2d`` computes
QK for all twelve heads, the same allocation is overwritten with V-transpose,
and a second ``matmul2d`` computes PV. Scores, online-softmax statistics, and
the output accumulator remain fp32. A final padded tile handles the row's
zero-to-three-token causal tail.

The entry point deliberately has no fallback.  It covers only B=1, Hq=24,
Hkv=2, D=256, four-token blocks, K=512, fp16/bfloat16 inputs, a host-static
prefill suffix, and a context that has crossed the dense/sparse boundary.
Callers must ask :func:`qsa_prefill_flash_supported` before routing here; a
direct unsupported call raises instead of silently changing the attention
algorithm or allocating a dense fallback.
"""

from __future__ import annotations

import math
import operator
from functools import lru_cache

import mlx.core as mx

from mtplx.kernels.qsa_indexer_select import qsa_indexer_select_nax_available

_BATCH = 1
_Q_HEADS = 24
_KV_HEADS = 2
_GQA = 12
_HEAD_DIM = 256
_MAX_CONTEXT = 1_048_576
_COMPRESS_RATIO = 4
_TOP_K_BLOCKS = 512
_M_ROWS = 16
_TILE_BLOCKS = 8
_TOKENS_PER_TILE = _TILE_BLOCKS * _COMPRESS_RATIO
_SIMD_WIDTH = 32
_THREADS = _SIMD_WIDTH
_EXPECTED_SCALE = 0.0625  # 1 / sqrt(256), exactly representable
_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16)

__all__ = ["qsa_prefill_flash", "qsa_prefill_flash_supported"]


_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant constexpr short QSA_ELEMS_PER_FRAG = 8;
constant constexpr short QSA_ELEM_COLS = 4;
constant constexpr short QSA_ELEM_ROWS_JUMP = 8;

// NAX fragment lane -> (column, row) mapping used by sdpa_nax_tile.py and the
// proven M5/G17G MetalPerformancePrimitives 16x32x16 descriptor.
inline short2 qsa_nax_coord(ushort lane) {
    short quad = short(lane >> 2);
    short row = ((quad & 4) | ((short(lane) >> 1) & 3));
    short col = ((quad & 2) | (short(lane) & 1)) * 4;
    return short2{col, row};
}
"""


_SOURCE = r"""
    constexpr int HEAD_DIM = 256;
    constexpr int KV_HEADS = 2;
    constexpr int GQA = 12;
    constexpr int BLOCK_TOKENS = 4;
    constexpr int TOP_K_BLOCKS = 512;
    constexpr int M_ROWS = 16;
    constexpr int TILE_BLOCKS = 8;
    constexpr int TOKENS_PER_TILE = TILE_BLOCKS * BLOCK_TOKENS;
    constexpr int MAX_SELECTED_TILES = TOP_K_BLOCKS / TILE_BLOCKS;
    constexpr int D_FRAGS = HEAD_DIM / 16;
    constexpr int OUT_GROUPS = HEAD_DIM / 32;

    // MLX Steel's wide-load idiom: a byte aggregate aligned only to one T.
    // Unlike vec<T,4>, this remains defined for unit-stride views whose base
    // begins at an odd fp16/bf16 element offset.
    struct alignas(sizeof(T)) QSAReadVector8 {
        uchar bytes[sizeof(T) * 8];
    };

    const ushort lane = ushort(thread_index_in_simdgroup);
    const int work = int(threadgroup_position_in_grid.x);
    const int row = work / KV_HEADS;
    const int kv_head = work - row * KV_HEADS;
    const int pos_start = int(params[0]);
    const int total_tokens = int(params[1]);
    const int query_pos = pos_start + row;
    const int complete_blocks = (query_pos + 1) / BLOCK_TOKENS;
    const int tail_start = complete_blocks * BLOCK_TOKENS;
    const int tail_count = query_pos + 1 - tail_start;
    // Find the highest selection slot that can actually contribute.  The
    // production selector emits a chronological valid prefix, but deriving
    // this from the slots themselves preserves the public kernel's existing
    // semantics for arbitrary validity holes as well.  Encoding slot+1 lets
    // zero mean "no selected tile" in the simd-wide max reduction.
    uint local_active_slots = 0u;
    for (uint block_slot = uint(lane);
         block_slot < uint(TOP_K_BLOCKS);
         block_slot += 32u) {
        const size_t id_at = size_t(row) * block_ids_strides[0] +
            size_t(block_slot) * block_ids_strides[1];
        const size_t valid_at = size_t(row) * block_valid_strides[0] +
            size_t(block_slot) * block_valid_strides[1];
        const int block_id = int(block_ids[id_at]);
        if (bool(block_valid[valid_at]) &&
            block_id >= 0 && block_id < complete_blocks) {
            local_active_slots = metal::max(local_active_slots, block_slot + 1u);
        }
    }
    const uint active_blocks = simd_max(local_active_slots);
    const int active_selected_tiles =
        (int(active_blocks) + TILE_BLOCKS - 1) / TILE_BLOCKS;
    const int active_tiles = active_selected_tiles + (tail_count > 0 ? 1 : 0);
    const short2 sc = qsa_nax_coord(lane);

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> mm;
    auto ct_a = mm.get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = mm.get_right_input_cooperative_tensor<T, T, float>();
    auto ct_c =
        mm.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

    // Exactly 32 * 256 * sizeof(T) == 16 KiB. The allocation first holds K
    // row-major and is then overwritten with V-transpose for the PV multiply.
    threadgroup T tg_tile[TOKENS_PER_TILE * HEAD_DIM];

    // Each lane carries two padded M rows. Four lanes share a logical row;
    // xor(1),xor(8) reduce the four score fragments for that row.
    float row_max[2] = {-1.0e38f, -1.0e38f};
    float row_sum[2] = {0.0f, 0.0f};
    float out_frag[OUT_GROUPS][2][QSA_ELEMS_PER_FRAG];
    for (int group = 0; group < OUT_GROUPS; ++group) {
        for (short row_part = 0; row_part < 2; ++row_part) {
            for (short elem = 0; elem < QSA_ELEMS_PER_FRAG; ++elem) {
                out_frag[group][row_part][elem] = 0.0f;
            }
        }
    }

    for (int tile_index = 0; tile_index < active_tiles; ++tile_index) {
        const bool tail_tile =
            tail_count > 0 && tile_index == active_selected_tiles;
        const int block_base = tile_index * TILE_BLOCKS;

        // Gather K into tg_tile[token,dim]. Metadata is deliberately recomputed
        // by the one simdgroup so the kernel owns only one 16 KiB TG allocation.
        for (int token_slot = 0; token_slot < TOKENS_PER_TILE; ++token_slot) {
            int token = 0;
            bool token_valid = false;
            if (tail_tile) {
                token = tail_start + token_slot;
                token_valid = token_slot < tail_count && token < total_tokens;
            } else {
                const int local_block = token_slot / BLOCK_TOKENS;
                const int within = token_slot - local_block * BLOCK_TOKENS;
                const int block_slot = block_base + local_block;
                const size_t id_at = size_t(row) * block_ids_strides[0] +
                    size_t(block_slot) * block_ids_strides[1];
                const size_t valid_at = size_t(row) * block_valid_strides[0] +
                    size_t(block_slot) * block_valid_strides[1];
                const int block_id = int(block_ids[id_at]);
                token_valid = bool(block_valid[valid_at]) &&
                    block_id >= 0 && block_id < complete_blocks;
                token = token_valid ? block_id * BLOCK_TOKENS + within : 0;
            }
            if (k_strides[3] == 1) {
                // Each lane exposes one compiler-visible 16-byte copy. Across
                // the simdgroup the full 256-wide cache row is contiguous.
                const int dim0 = int(lane) * 8;
                const int destination = token_slot * HEAD_DIM + dim0;
                if (token_valid) {
                    const size_t k_at = size_t(kv_head) * k_strides[1] +
                        size_t(token) * k_strides[2] + size_t(dim0);
                    *reinterpret_cast<threadgroup QSAReadVector8*>(
                        &tg_tile[destination]) =
                        *reinterpret_cast<const device QSAReadVector8*>(&k[k_at]);
                } else {
                    for (short elem = 0; elem < 8; ++elem) {
                        tg_tile[destination + int(elem)] = T(0);
                    }
                }
            } else {
                // Fail-correct for unusual cache views whose feature axis is
                // not contiguous; the production backing takes the wide-copy lane.
                for (int dim = int(lane); dim < HEAD_DIM; dim += 32) {
                    const size_t k_at = size_t(kv_head) * k_strides[1] +
                        size_t(token) * k_strides[2] +
                        size_t(dim) * k_strides[3];
                    tg_tile[token_slot * HEAD_DIM + dim] =
                        token_valid ? k[k_at] : T(0);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // QK: padded [16,256] x gathered [32,256]^T -> [16,32].
        for (short elem = 0; elem < 2 * QSA_ELEMS_PER_FRAG; ++elem) {
            ct_c[elem] = 0.0f;
        }
        for (int frag = 0; frag < D_FRAGS; ++frag) {
            for (short row_part = 0; row_part < 2; ++row_part) {
                const int m_row = int(sc.y) + int(row_part) * QSA_ELEM_ROWS_JUMP;
                if (m_row < GQA) {
                    const int q_head = kv_head * GQA + m_row;
                    const size_t q_base = size_t(q_head) * q_strides[1] +
                        size_t(row) * q_strides[2] +
                        size_t(frag * 16 + int(sc.x)) * q_strides[3];
                    for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                        ct_a[row_part * QSA_ELEM_COLS + col] =
                            q[q_base + size_t(col) * q_strides[3]];
                    }
                } else {
                    for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                        ct_a[row_part * QSA_ELEM_COLS + col] = T(0);
                    }
                }
            }
            for (short n_half = 0; n_half < 2; ++n_half) {
                for (short row_half = 0; row_half < 2; ++row_half) {
                    const int token_slot = int(n_half) * 16 + int(sc.y) +
                        int(row_half) * QSA_ELEM_ROWS_JUMP;
                    const int tile_base = token_slot * HEAD_DIM +
                        frag * 16 + int(sc.x);
                    for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                        ct_b[n_half * QSA_ELEMS_PER_FRAG +
                             row_half * QSA_ELEM_COLS + col] =
                            tg_tile[tile_base + int(col)];
                    }
                }
            }
            mm.run(ct_a, ct_b, ct_c);
        }

        // Mask, scale, and exponentiate the score fragment in registers.
        float probabilities[2][QSA_ELEMS_PER_FRAG];
        float correction[2];
        for (short row_part = 0; row_part < 2; ++row_part) {
            const int m_row = int(sc.y) + int(row_part) * QSA_ELEM_ROWS_JUMP;
            const bool live_row = m_row < GQA;
            float tile_max = -1.0e38f;
            for (short n_half = 0; n_half < 2; ++n_half) {
                for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                    const int token_slot = int(n_half) * 16 + int(sc.x) + int(col);
                    bool token_valid = false;
                    if (tail_tile) {
                        const int token = tail_start + token_slot;
                        token_valid = token_slot < tail_count && token < total_tokens;
                    } else {
                        const int local_block = token_slot / BLOCK_TOKENS;
                        const int block_slot = block_base + local_block;
                        const size_t id_at = size_t(row) * block_ids_strides[0] +
                            size_t(block_slot) * block_ids_strides[1];
                        const size_t valid_at = size_t(row) * block_valid_strides[0] +
                            size_t(block_slot) * block_valid_strides[1];
                        const int block_id = int(block_ids[id_at]);
                        token_valid = bool(block_valid[valid_at]) &&
                            block_id >= 0 && block_id < complete_blocks;
                    }
                    const short at = row_part * QSA_ELEM_COLS + col;
                    const float score = live_row && token_valid
                        ? ct_c[n_half * QSA_ELEMS_PER_FRAG + at] * scale[0]
                        : -1.0e38f;
                    probabilities[n_half][at] = score;
                    tile_max = metal::max(tile_max, score);
                }
            }
            tile_max = metal::max(tile_max, simd_shuffle_xor(tile_max, ushort(1)));
            tile_max = metal::max(tile_max, simd_shuffle_xor(tile_max, ushort(8)));
            const float new_max = metal::max(row_max[row_part], tile_max);
            correction[row_part] = metal::exp(row_max[row_part] - new_max);
            float tile_sum = 0.0f;
            for (short n_half = 0; n_half < 2; ++n_half) {
                for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                    const short at = row_part * QSA_ELEM_COLS + col;
                    const float score = probabilities[n_half][at];
                    const float probability = score > -1.0e37f
                        ? metal::exp(score - new_max)
                        : 0.0f;
                    probabilities[n_half][at] = probability;
                    tile_sum += probability;
                }
            }
            tile_sum += simd_shuffle_xor(tile_sum, ushort(1));
            tile_sum += simd_shuffle_xor(tile_sum, ushort(8));
            row_max[row_part] = new_max;
            row_sum[row_part] = row_sum[row_part] * correction[row_part] + tile_sum;
        }

        for (short row_part = 0; row_part < 2; ++row_part) {
            const float factor = correction[row_part];
            for (int group = 0; group < OUT_GROUPS; ++group) {
                for (short dim_half = 0; dim_half < 2; ++dim_half) {
                    for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                        out_frag[group][dim_half]
                                [row_part * QSA_ELEM_COLS + col] *= factor;
                    }
                }
            }
        }

        // QK is finished: overwrite the same 16 KiB with V^T[dim,token].
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int token_slot = 0; token_slot < TOKENS_PER_TILE; ++token_slot) {
            int token = 0;
            bool token_valid = false;
            if (tail_tile) {
                token = tail_start + token_slot;
                token_valid = token_slot < tail_count && token < total_tokens;
            } else {
                const int local_block = token_slot / BLOCK_TOKENS;
                const int within = token_slot - local_block * BLOCK_TOKENS;
                const int block_slot = block_base + local_block;
                const size_t id_at = size_t(row) * block_ids_strides[0] +
                    size_t(block_slot) * block_ids_strides[1];
                const size_t valid_at = size_t(row) * block_valid_strides[0] +
                    size_t(block_slot) * block_valid_strides[1];
                const int block_id = int(block_ids[id_at]);
                token_valid = bool(block_valid[valid_at]) &&
                    block_id >= 0 && block_id < complete_blocks;
                token = token_valid ? block_id * BLOCK_TOKENS + within : 0;
            }
            if (v_strides[3] == 1) {
                const int dim0 = int(lane) * 8;
                thread T v_lane[8];
                if (token_valid) {
                    const size_t v_at = size_t(kv_head) * v_strides[1] +
                        size_t(token) * v_strides[2] + size_t(dim0);
                    *reinterpret_cast<thread QSAReadVector8*>(&v_lane[0]) =
                        *reinterpret_cast<const device QSAReadVector8*>(&v[v_at]);
                } else {
                    for (short elem = 0; elem < 8; ++elem) {
                        v_lane[elem] = T(0);
                    }
                }
                for (short elem = 0; elem < 8; ++elem) {
                    tg_tile[(dim0 + int(elem)) * TOKENS_PER_TILE + token_slot] =
                        v_lane[elem];
                }
            } else {
                for (int dim = int(lane); dim < HEAD_DIM; dim += 32) {
                    const size_t v_at = size_t(kv_head) * v_strides[1] +
                        size_t(token) * v_strides[2] +
                        size_t(dim) * v_strides[3];
                    tg_tile[dim * TOKENS_PER_TILE + token_slot] =
                        token_valid ? v[v_at] : T(0);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // PV: [16,32] x [256,32]^T -> [16,256]. MPP consumes probabilities
        // in T, while accumulation and all persistent online state remain fp32.
        for (int group = 0; group < OUT_GROUPS; ++group) {
            for (short elem = 0; elem < QSA_ELEMS_PER_FRAG; ++elem) {
                ct_c[elem] = out_frag[group][0][elem];
                ct_c[QSA_ELEMS_PER_FRAG + elem] = out_frag[group][1][elem];
            }
            for (short token_half = 0; token_half < 2; ++token_half) {
                for (short row_part = 0; row_part < 2; ++row_part) {
                    for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                        ct_a[row_part * QSA_ELEM_COLS + col] = T(
                            probabilities[token_half][row_part * QSA_ELEM_COLS + col]);
                    }
                }
                for (short dim_half = 0; dim_half < 2; ++dim_half) {
                    for (short row_half = 0; row_half < 2; ++row_half) {
                        const int dim = group * 32 + int(dim_half) * 16 +
                            int(sc.y) + int(row_half) * QSA_ELEM_ROWS_JUMP;
                        const int tile_base = dim * TOKENS_PER_TILE +
                            int(token_half) * 16 + int(sc.x);
                        for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                            ct_b[dim_half * QSA_ELEMS_PER_FRAG +
                                 row_half * QSA_ELEM_COLS + col] =
                                tg_tile[tile_base + int(col)];
                        }
                    }
                }
                mm.run(ct_a, ct_b, ct_c);
            }
            for (short elem = 0; elem < QSA_ELEMS_PER_FRAG; ++elem) {
                out_frag[group][0][elem] = ct_c[elem];
                out_frag[group][1][elem] = ct_c[QSA_ELEMS_PER_FRAG + elem];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and store the twelve live rows into contiguous [1,24,S,256].
    for (short row_part = 0; row_part < 2; ++row_part) {
        const int m_row = int(sc.y) + int(row_part) * QSA_ELEM_ROWS_JUMP;
        if (m_row >= GQA) continue;
        const int q_head = kv_head * GQA + m_row;
        const float inv_sum = row_sum[row_part] > 0.0f ? 1.0f / row_sum[row_part] : 0.0f;
        const size_t out_base =
            (size_t(q_head) * size_t(params[2]) + size_t(row)) * HEAD_DIM;
        for (int group = 0; group < OUT_GROUPS; ++group) {
            for (short dim_half = 0; dim_half < 2; ++dim_half) {
                for (short col = 0; col < QSA_ELEM_COLS; ++col) {
                    const int dim = group * 32 + int(dim_half) * 16 +
                        int(sc.x) + int(col);
                    out[out_base + size_t(dim)] = T(
                        out_frag[group][dim_half][row_part * QSA_ELEM_COLS + col] *
                        inv_sum);
                }
            }
        }
    }
"""


def _on_metal_device() -> bool:
    """Metal availability is insufficient when MLX currently targets CPU."""

    try:
        return mx.metal.is_available() and mx.default_device() == mx.gpu
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _unsupported_reason(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    block_valid: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    scale: float,
) -> str | None:
    if not _on_metal_device():
        return "the active MLX device is not an available Metal GPU"
    if not qsa_indexer_select_nax_available():
        return "Metal 4 TensorOps are unavailable on this macOS/GPU generation"
    arrays = (queries, keys, values, block_ids, block_valid)
    if any(not isinstance(array, mx.array) for array in arrays):
        return "all tensor inputs must be MLX arrays"
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return "Q, K, and V must be rank four"
    if block_ids.ndim != 2 or block_valid.ndim != 2:
        return "block ids and validity must be rank two"

    batch, query_heads, rows, head_dim = (int(x) for x in queries.shape)
    if (batch, query_heads, head_dim) != (_BATCH, _Q_HEADS, _HEAD_DIM):
        return "Q must have production shape [1, 24, S, 256]"
    if rows <= 1:
        return "the prefill kernel requires at least two query rows"

    key_batch, kv_heads, capacity, key_dim = (int(x) for x in keys.shape)
    if (key_batch, kv_heads, key_dim) != (_BATCH, _KV_HEADS, _HEAD_DIM):
        return "K must have production shape [1, 2, capacity, 256]"
    if tuple(int(x) for x in values.shape) != tuple(int(x) for x in keys.shape):
        return "V must have the same full-backing shape as K"

    if queries.dtype not in _SUPPORTED_DTYPES:
        return "Q must be float16 or bfloat16"
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return "Q, K, and V dtypes must match"
    if block_ids.dtype != mx.int32 or block_valid.dtype != mx.bool_:
        return "block ids must be int32 and validity must be bool"
    if tuple(int(x) for x in block_ids.shape) != (rows, _TOP_K_BLOCKS):
        return "block ids must have shape [S, 512]"
    if tuple(int(x) for x in block_valid.shape) != (rows, _TOP_K_BLOCKS):
        return "block validity must have shape [S, 512]"

    # Tensor scalars would make the host comparisons below synchronize a
    # traced value.  This isolated lane accepts only the static suffix contract
    # used by the current Attention call; future graph wiring needs an explicit
    # dynamic-scalar contract rather than an accidental .item()/int conversion.
    if isinstance(pos_start, mx.array) or isinstance(total_tokens, mx.array):
        return "pos_start and total_tokens must be host integers"
    if isinstance(scale, mx.array):
        return "scale must be a host float"
    if isinstance(pos_start, bool) or isinstance(total_tokens, bool):
        return "pos_start and total_tokens cannot be bool"
    try:
        pos_start_i = operator.index(pos_start)
        total_tokens_i = operator.index(total_tokens)
    except TypeError:
        return "pos_start and total_tokens must be exact host integers"
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        return "scale must be a numeric host scalar"
    scale_f = float(scale)

    if pos_start_i < 0 or total_tokens_i <= 0:
        return "positions must describe a non-empty non-negative suffix"
    if pos_start_i + rows != total_tokens_i:
        return "Q must be the suffix ending exactly at total_tokens"
    if total_tokens_i > capacity:
        return "the logical token count exceeds the full K/V backing capacity"
    if total_tokens_i > _MAX_CONTEXT:
        return "the logical token count exceeds the production context limit"
    if total_tokens_i // _COMPRESS_RATIO <= _TOP_K_BLOCKS:
        return "the context has not crossed the dense/sparse boundary"
    if not math.isfinite(scale_f) or scale_f != _EXPECTED_SCALE:
        return "scale must equal the production 1/sqrt(256) value"
    return None


def qsa_prefill_flash_supported(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    block_valid: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    scale: float,
) -> bool:
    """Return whether the exact production-only kernel contract is met."""

    return (
        _unsupported_reason(
            queries,
            keys,
            values,
            block_ids,
            block_valid,
            pos_start=pos_start,
            total_tokens=total_tokens,
            scale=scale,
        )
        is None
    )


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtplx_qsa_prefill_flash_nax_m16_n32_h24_kv2_d256_b4_k512",
        input_names=[
            "q",
            "k",
            "v",
            "block_ids",
            "block_valid",
            "params",
            "scale",
        ],
        output_names=["out"],
        header=_HEADER,
        source=_SOURCE,
        # Every tensor is indexed with its injected stride array.  In
        # particular, the transposed Q view and full-capacity K/V backing must
        # not acquire hidden contiguous-copy dispatches.
        ensure_row_contiguous=False,
    )


def qsa_prefill_flash(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    block_valid: mx.array,
    *,
    pos_start: int,
    total_tokens: int,
    scale: float,
) -> mx.array:
    """Compute sparse-plus-tail QSA prefill attention as one Metal dispatch.

    ``queries`` is ``[1,24,S,256]``. ``keys`` and ``values`` are the unsliced
    full cache backings ``[1,2,capacity,256]``.  Selection arrays are
    ``[S,512]`` and belong to each query row.  The output is
    ``[1,24,S,256]`` in the Q/K/V dtype.

    Unsupported calls raise.  This function does not allocate a gathered K/V
    tensor and does not provide a dense-attention fallback.
    """

    reason = _unsupported_reason(
        queries,
        keys,
        values,
        block_ids,
        block_valid,
        pos_start=pos_start,
        total_tokens=total_tokens,
        scale=scale,
    )
    if reason is not None:
        raise ValueError(f"unsupported QSA prefill flash call: {reason}")

    rows = int(queries.shape[2])
    params = mx.array([int(pos_start), int(total_tokens), rows], dtype=mx.int32)
    scale_arg = mx.array([float(scale)], dtype=mx.float32)
    kernel = _kernel()
    (out,) = kernel(
        inputs=[
            queries,
            keys,
            values,
            block_ids,
            block_valid,
            params,
            scale_arg,
        ],
        template=[("T", queries.dtype)],
        grid=(rows * _KV_HEADS * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(_BATCH, _Q_HEADS, rows, _HEAD_DIM)],
        output_dtypes=[queries.dtype],
    )
    return out
