"""Small-fixture runtime parity gates for the fused QSA MPP score stage.

This module loads no model or artifact. It is intentionally separate from the
MLX-free structural suite so source-only work can be gated without importing
MLX or dispatching Metal. Run it only when the operator has released the GPU.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.kernels.qsa_indexer_prefill import (
    qsa_indexer_prefill_scores,
    qsa_indexer_prefill_scores_mpp,
    qsa_indexer_prefill_scores_mpp_supported,
    qsa_indexer_prefill_topk_metal,
)
from mtplx.kernels.qsa_indexer_select import qsa_indexer_select_nax_available

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available()
    or mx.default_device() != mx.gpu
    or not qsa_indexer_select_nax_available(),
    reason="QSA MPP score parity requires a Metal 4/NAX GPU",
)


def _fixture(
    seed: int,
    rows: int,
    blocks: int,
    dtype: mx.Dtype,
) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    q = mx.contiguous(mx.random.normal((1, rows, 4, 128)).astype(dtype))
    pooled = mx.contiguous(mx.random.normal((1, blocks, 128)).astype(dtype))
    return q, pooled


def _oracle(q: mx.array, pooled: mx.array) -> mx.array:
    pooled_t = mx.swapaxes(pooled.astype(mx.float32), 1, 2)[:, None]
    return qsa_indexer_prefill_scores(q, pooled_t, head_dim=128)


def _maximum_abs(actual: mx.array, expected: mx.array) -> float:
    mx.eval(actual, expected)
    return float(mx.max(mx.abs(actual - expected)).item())


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "seed,rows,blocks",
    [
        (101, 1, 1),
        (102, 3, 31),
        (103, 4, 32),
        (104, 5, 33),
        (105, 15, 63),
        (106, 16, 64),
        (107, 17, 97),
        (108, 32, 257),
    ],
)
def test_mpp_scores_match_vectorized_mlx_oracle(
    dtype: mx.Dtype,
    seed: int,
    rows: int,
    blocks: int,
):
    q, pooled = _fixture(seed, rows, blocks, dtype)
    assert qsa_indexer_prefill_scores_mpp_supported(q, pooled)
    actual = qsa_indexer_prefill_scores_mpp(q, pooled)
    expected = _oracle(q, pooled)
    assert actual.dtype == mx.float32
    assert tuple(actual.shape) == (rows, blocks)
    assert _maximum_abs(actual, expected) <= 1e-4


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("seed,rows,blocks", [(201, 5, 33), (202, 17, 97)])
def test_mpp_scores_preserve_exact_selected_indices(
    dtype: mx.Dtype,
    seed: int,
    rows: int,
    blocks: int,
):
    q, pooled = _fixture(seed, rows, blocks, dtype)
    assert qsa_indexer_prefill_scores_mpp_supported(q, pooled)
    fused_scores = qsa_indexer_prefill_scores_mpp(q, pooled)
    oracle_scores = _oracle(q, pooled)
    total_tokens = blocks * 4
    pos_start = total_tokens - rows
    common = {
        "pos_start": pos_start,
        "total_tokens": total_tokens,
        "logical_blocks": blocks,
        "block_topk": min(16, blocks),
        "compress_ratio": 4,
        "mode": "blocks",
    }
    actual_ids, actual_valid, actual_adjusted = qsa_indexer_prefill_topk_metal(
        fused_scores,
        **common,
    )
    expected_ids, expected_valid, expected_adjusted = qsa_indexer_prefill_topk_metal(
        oracle_scores,
        **common,
    )
    mx.eval(
        actual_ids,
        actual_valid,
        actual_adjusted,
        expected_ids,
        expected_valid,
        expected_adjusted,
    )
    assert bool(mx.array_equal(actual_ids, expected_ids).item())
    assert bool(mx.array_equal(actual_valid, expected_valid).item())
    assert _maximum_abs(actual_adjusted, expected_adjusted) <= 1e-4


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_mpp_zero_relu_ties_keep_the_exact_selector_contract(dtype: mx.Dtype):
    rows, blocks = 17, 97
    q = mx.zeros((1, rows, 4, 128), dtype=dtype)
    pooled = mx.ones((1, blocks, 128), dtype=dtype)
    assert qsa_indexer_prefill_scores_mpp_supported(q, pooled)
    fused_scores = qsa_indexer_prefill_scores_mpp(q, pooled)
    oracle_scores = _oracle(q, pooled)
    mx.eval(fused_scores, oracle_scores)
    assert bool(mx.array_equal(fused_scores, oracle_scores).item())
    common = {
        "pos_start": blocks * 4 - rows,
        "total_tokens": blocks * 4,
        "logical_blocks": blocks,
        "block_topk": 16,
        "compress_ratio": 4,
        "mode": "blocks",
    }
    fused_ids, fused_valid, _ = qsa_indexer_prefill_topk_metal(
        fused_scores,
        **common,
    )
    oracle_ids, oracle_valid, _ = qsa_indexer_prefill_topk_metal(
        oracle_scores,
        **common,
    )
    mx.eval(fused_ids, fused_valid, oracle_ids, oracle_valid)
    assert bool(mx.array_equal(fused_ids, oracle_ids).item())
    assert bool(mx.array_equal(fused_valid, oracle_valid).item())
    assert fused_ids.tolist() == [list(range(16)) for _ in range(rows)]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("layout", ["row_slice", "strided_axes"])
def test_mpp_scores_consume_runtime_strides_without_a_hidden_copy(
    dtype: mx.Dtype,
    layout: str,
):
    rows, blocks = 17, 97
    mx.random.seed(301 if layout == "row_slice" else 302)
    if layout == "row_slice":
        q_storage = mx.random.normal((1, rows + 5, 4, 128)).astype(dtype)
        pooled_storage = mx.random.normal((1, blocks + 7, 128)).astype(dtype)
        q = q_storage[:, 3 : 3 + rows]
        pooled = pooled_storage[:, 5 : 5 + blocks]
    else:
        q_storage = mx.random.normal((1, rows * 2, 4, 256)).astype(dtype)
        pooled_storage = mx.random.normal((1, blocks * 2, 256)).astype(dtype)
        q = q_storage[:, ::2, :, ::2]
        pooled = pooled_storage[:, ::2, ::2]

    # The host support decision is shape/dtype based because MLX 0.32 does
    # not expose strides to Python.  ensure_row_contiguous=False plus scalar
    # source gathers makes the actual custom-kernel dispatch layout-general.
    assert qsa_indexer_prefill_scores_mpp_supported(q, pooled)
    actual = qsa_indexer_prefill_scores_mpp(q, pooled)
    expected = _oracle(q, pooled)
    assert _maximum_abs(actual, expected) <= 1e-4


def test_non_production_geometry_stays_on_the_vectorized_oracle():
    q = mx.zeros((1, 8, 3, 128), dtype=mx.bfloat16)
    pooled = mx.zeros((1, 64, 128), dtype=mx.bfloat16)
    assert not qsa_indexer_prefill_scores_mpp_supported(q, pooled)
    with pytest.raises(ValueError, match="unsupported signature"):
        qsa_indexer_prefill_scores_mpp(q, pooled)

    pooled_t = mx.swapaxes(pooled.astype(mx.float32), 1, 2)[:, None]
    fallback = qsa_indexer_prefill_scores(q, pooled_t, head_dim=128)
    mx.eval(fallback)
    assert tuple(fallback.shape) == (8, 64)
    assert fallback.dtype == mx.float32
