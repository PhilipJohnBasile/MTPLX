"""Static gates for the production QSA prefill Metal module.

These tests intentionally read and parse the module instead of importing it.
They are safe on hosts without MLX/Metal and cannot compile or dispatch a
kernel.  Numeric Metal parity belongs to the later operator-controlled gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mtplx" / "kernels" / "qsa_prefill_flash.py"
MODULE_TEXT = MODULE_PATH.read_text(encoding="utf-8")
MODULE_TREE = ast.parse(MODULE_TEXT, filename=str(MODULE_PATH))


def _assignment(name: str) -> ast.expr:
    for node in MODULE_TREE.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id == name for target in targets
            ):
                assert node.value is not None
                return node.value
    raise AssertionError(f"missing module assignment {name}")


def _literal(name: str):
    return ast.literal_eval(_assignment(name))


def _function_source(name: str) -> str:
    for node in MODULE_TREE.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return ast.get_source_segment(MODULE_TEXT, node) or ""
    raise AssertionError(f"missing function {name}")


def test_production_geometry_is_narrow_and_explicit():
    assert _literal("_BATCH") == 1
    assert _literal("_Q_HEADS") == 24
    assert _literal("_KV_HEADS") == 2
    assert _literal("_GQA") == 12
    assert _literal("_HEAD_DIM") == 256
    assert _literal("_MAX_CONTEXT") == 1_048_576
    assert _literal("_COMPRESS_RATIO") == 4
    assert _literal("_TOP_K_BLOCKS") == 512
    assert _literal("_M_ROWS") == 16
    assert _literal("_TILE_BLOCKS") == 8
    assert _literal("_SIMD_WIDTH") == 32
    assert _literal("_EXPECTED_SCALE") == 0.0625

    assert ast.unparse(_assignment("_TOKENS_PER_TILE")) == (
        "_TILE_BLOCKS * _COMPRESS_RATIO"
    )
    assert ast.unparse(_assignment("_THREADS")) == "_SIMD_WIDTH"


def test_kernel_is_one_m16_n32_tensorops_simdgroup_per_row_and_kv_head():
    header = _literal("_HEADER")
    source = _literal("_SOURCE")
    assert "MetalPerformancePrimitives/MetalPerformancePrimitives.h" in header
    assert "qsa_nax_coord" in header
    assert "constexpr int M_ROWS = 16" in source
    assert "constexpr int TILE_BLOCKS = 8" in source
    assert "constexpr int TOKENS_PER_TILE = TILE_BLOCKS * BLOCK_TOKENS" in source
    assert "constexpr int MAX_SELECTED_TILES = TOP_K_BLOCKS / TILE_BLOCKS" in source
    assert "uint local_active_slots = 0u" in source
    assert "const uint active_blocks = simd_max(local_active_slots)" in source
    assert "block_id >= 0 && block_id < complete_blocks" in source
    assert "const int active_selected_tiles" in source
    assert "const int active_tiles" in source
    assert "const int row = work / KV_HEADS" in source
    assert "const int kv_head = work - row * KV_HEADS" in source
    assert "mpp::tensor_ops::matmul2d_descriptor(" in source
    assert "16, 32, 16, false, true, true" in source
    assert "mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>" in source
    assert source.count("mm.run(ct_a, ct_b, ct_c)") == 2

    # Exactly one 32x256 T tile exists. It is K row-major first and then the
    # same allocation is overwritten as V^T for PV.
    assert source.count("threadgroup T tg_tile[TOKENS_PER_TILE * HEAD_DIM]") == 1
    assert "threadgroup T tile_k" not in source
    assert "threadgroup T tile_v" not in source
    k_store = "tg_tile[token_slot * HEAD_DIM + dim]"
    vt_store = "tg_tile[dim * TOKENS_PER_TILE + token_slot]"
    assert k_store in source
    assert vt_store in source
    assert source.index(k_store) < source.index(vt_store)
    assert "simdgroup_barrier(mem_flags::mem_threadgroup)" in source

    # All potentially non-contiguous inputs use the strides injected by MLX.
    for stride_name in (
        "q_strides",
        "k_strides",
        "v_strides",
        "block_ids_strides",
        "block_valid_strides",
    ):
        assert stride_name in source

    kernel_factory = _function_source("_kernel")
    for input_name in ('"q"', '"k"', '"v"', '"block_ids"', '"block_valid"'):
        assert input_name in kernel_factory
    assert "ensure_row_contiguous=False" in kernel_factory

    forbidden = (
        "mx.take",
        "scaled_dot_product_attention",
        "dense_mask",
        "concatenate",
    )
    implementation = _function_source("qsa_prefill_flash")
    for token in forbidden:
        assert token not in source
        assert token not in implementation


def test_kernel_rechecks_validity_and_adds_only_the_visible_causal_tail():
    source = _literal("_SOURCE")
    assert "bool(block_valid[valid_at])" in source
    assert "block_id >= 0 && block_id < complete_blocks" in source
    assert "const int complete_blocks = (query_pos + 1) / BLOCK_TOKENS" in source
    assert "const int tail_start = complete_blocks * BLOCK_TOKENS" in source
    assert "const int tail_count = query_pos + 1 - tail_start" in source
    assert "tile_index == active_selected_tiles" in source
    assert "token_valid = token_slot < tail_count && token < total_tokens" in source
    assert "tile_index < active_tiles" in source


def test_cache_gathers_use_unaligned_safe_wide_copies_with_stride_fallbacks():
    source = _literal("_SOURCE")
    assert "if (k_strides[3] == 1)" in source
    assert "if (v_strides[3] == 1)" in source
    assert "struct alignas(sizeof(T)) QSAReadVector8" in source
    assert "uchar bytes[sizeof(T) * 8]" in source
    assert "threadgroup QSAReadVector8" in source
    assert "thread QSAReadVector8" in source
    assert source.count("const device QSAReadVector8") == 2
    assert "const device vec<T, 4>" not in source
    assert "const int dim0 = int(lane) * 8" in source
    assert "for (int dim = int(lane); dim < HEAD_DIM; dim += 32)" in source

    # The selection itself never substitutes for causality: a producer bug
    # cannot make an incomplete/future block visible.
    assert source.index("block_id < complete_blocks") < source.index("k[k_at]")


def test_kernel_uses_fp32_online_softmax_and_tensorops_pv():
    source = _literal("_SOURCE")
    assert "float row_max[2]" in source
    assert "float row_sum[2]" in source
    assert "float out_frag[OUT_GROUPS][2][QSA_ELEMS_PER_FRAG]" in source
    assert "float probabilities[2][QSA_ELEMS_PER_FRAG]" in source
    assert "const float new_max = metal::max(row_max[row_part], tile_max)" in source
    assert "metal::exp(row_max[row_part] - new_max)" in source
    assert "metal::exp(score - new_max)" in source
    assert (
        "row_sum[row_part] = row_sum[row_part] * correction[row_part] + tile_sum"
        in source
    )
    assert "const float factor = correction[row_part]" in source
    assert (
        "out_frag[group][dim_half]"
        "\n                                [row_part * QSA_ELEM_COLS + col] *= factor"
        in source
    )
    assert (
        "row_sum[row_part] > 0.0f ? 1.0f / row_sum[row_part] : 0.0f" in source
    )
    assert "ct_a[row_part * QSA_ELEM_COLS + col] = T(" in source

    # No decode-style scalar token loop remains in the active source.
    assert "dot_partial" not in source
    assert "simd_sum" not in source
    assert "TOP_K_BLOCKS * HEAD_DIM" not in source
    assert "Q_HEADS * TOP_K_BLOCKS" not in source


def test_wrapper_launch_and_output_contract_match_one_tg_per_row_and_kv_head():
    implementation = _function_source("qsa_prefill_flash")
    assert "grid=(rows * _KV_HEADS * _THREADS, 1, 1)" in implementation
    assert "threadgroup=(_THREADS, 1, 1)" in implementation
    assert "output_shapes=[(_BATCH, _Q_HEADS, rows, _HEAD_DIM)]" in implementation
    assert "output_dtypes=[queries.dtype]" in implementation
    assert 'template=[("T", queries.dtype)]' in implementation
    assert ".reshape(" not in implementation


def test_validator_and_entry_point_fail_closed_outside_the_contract():
    validator = _function_source("_unsupported_reason")
    required_guards = (
        "_on_metal_device()",
        "qsa_indexer_select_nax_available()",
        "queries.ndim != 4",
        "block_ids.ndim != 2",
        "(_BATCH, _Q_HEADS, _HEAD_DIM)",
        "(_BATCH, _KV_HEADS, _HEAD_DIM)",
        "queries.dtype not in _SUPPORTED_DTYPES",
        "block_ids.dtype != mx.int32",
        "block_valid.dtype != mx.bool_",
        "(rows, _TOP_K_BLOCKS)",
        "operator.index(pos_start)",
        "operator.index(total_tokens)",
        "pos_start_i + rows != total_tokens_i",
        "total_tokens_i > capacity",
        "total_tokens_i > _MAX_CONTEXT",
        "total_tokens_i // _COMPRESS_RATIO <= _TOP_K_BLOCKS",
        "scale_f != _EXPECTED_SCALE",
    )
    for guard in required_guards:
        assert guard in validator

    # Dynamic tensor scalars are rejected rather than host-synchronized from
    # inside a future mx.compile region.
    assert "isinstance(pos_start, mx.array)" in validator
    assert "isinstance(total_tokens, mx.array)" in validator

    implementation = _function_source("qsa_prefill_flash")
    assert "reason = _unsupported_reason(" in implementation
    assert "if reason is not None:" in implementation
    assert 'raise ValueError(f"unsupported QSA prefill flash call: {reason}")' in (
        implementation
    )


def test_active_module_contains_no_retained_scalar_kernel_source():
    assert "_SCALAR_SOURCE" not in MODULE_TEXT
