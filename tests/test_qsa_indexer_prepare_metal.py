"""Small-array exactness gates for the QSA preparation Metal kernels.

These tests never construct a model or load checkpoint weights.  They compare
the fused query and pooled-key chains directly with the v2.10 eager MLX math,
including the input-dtype rounding between block mean and RMSNorm.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.kernels.qsa_indexer_prepare import (
    qsa_indexer_pool_keys_metal,
    qsa_indexer_prepare_queries_metal,
)
from mtplx.models.qwen4_exp import TextArgs, _rope_inv_freq_and_scaling

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or mx.default_device() != mx.gpu,
    reason="QSA preparation kernels require the Metal GPU",
)


def _inv_freq(rotary_dim: int) -> mx.array:
    return 10000.0 ** (-mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim)


def _partial_rope(
    values: mx.array,
    positions: mx.array,
    inv_freq: mx.array,
    attention_scaling: float = 1.0,
) -> mx.array:
    angles = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    angles = mx.concatenate([angles, angles], axis=-1)
    cosine = mx.cos(angles) * float(attention_scaling)
    sine = mx.sin(angles) * float(attention_scaling)
    rotary_dim = int(cosine.shape[-1])
    half = rotary_dim // 2
    rotary = values[..., :rotary_dim]
    passthrough = values[..., rotary_dim:]
    rotated = mx.concatenate([-rotary[..., half:], rotary[..., :half]], axis=-1)
    rotary = (
        rotary.astype(mx.float32) * cosine[None, :, None, :]
        + rotated.astype(mx.float32) * sine[None, :, None, :]
    ).astype(values.dtype)
    return mx.concatenate([rotary, passthrough], axis=-1)


def _query_oracle(
    raw: mx.array,
    weight: mx.array,
    inv_freq: mx.array,
    pos_start: int,
    eps: float,
    attention_scaling: float = 1.0,
) -> mx.array:
    normalized = mx.fast.rms_norm(raw, weight, eps)
    positions = mx.arange(
        pos_start,
        pos_start + int(raw.shape[1]),
        dtype=mx.int32,
    )
    return _partial_rope(
        normalized,
        positions,
        inv_freq,
        attention_scaling,
    )


def _pool_oracle(
    raw: mx.array,
    weight: mx.array,
    inv_freq: mx.array,
    block_start: int,
    ratio: int,
    eps: float,
    attention_scaling: float = 1.0,
) -> mx.array:
    blocks = int(raw.shape[1]) // ratio
    values = raw.reshape(1, blocks, ratio, int(raw.shape[-1]))
    pooled = mx.mean(values.astype(mx.float32), axis=2).astype(raw.dtype)
    normalized = mx.fast.rms_norm(pooled, weight, eps)
    positions = mx.arange(block_start, block_start + blocks, dtype=mx.int32) * ratio
    # _partial_rope expects [1,S,H,D]; the indexer pooled stream has one
    # implicit head and drops it after rotation.
    return _partial_rope(
        normalized[:, :, None, :],
        positions,
        inv_freq,
        attention_scaling,
    )[:, :, 0, :]


def _assert_numeric_equal(
    actual: mx.array,
    expected: mx.array,
    *,
    exact: bool,
) -> None:
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert tuple(actual.shape) == tuple(expected.shape)
    if exact:
        assert bool(mx.array_equal(actual, expected).item())
    else:
        maximum = float(
            mx.max(
                mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
            ).item()
        )
        assert maximum <= 5e-7


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize(
    "seed,rows,heads,head_dim,rotary_dim,pos_start",
    [
        (1, 1, 4, 128, 64, 0),
        (2, 1, 4, 128, 64, 2051),
        (3, 7, 4, 128, 64, 8191),
        (4, 3, 2, 32, 16, 7),
        (5, 2, 3, 7, 6, 1),
        (6, 2, 4, 128, 64, 999_998),
        (7, 2, 4, 128, 64, 1_048_574),
    ],
)
def test_query_prepare_matches_eager(
    dtype,
    seed: int,
    rows: int,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    pos_start: int,
):
    mx.random.seed(seed)
    raw = mx.random.normal((1, rows, heads, head_dim)).astype(dtype)
    weight = mx.random.uniform(
        low=0.5,
        high=1.5,
        shape=(head_dim,),
    ).astype(dtype)
    inv_freq = _inv_freq(rotary_dim)
    expected = _query_oracle(raw, weight, inv_freq, pos_start, 1e-6)
    actual = qsa_indexer_prepare_queries_metal(
        raw,
        weight,
        inv_freq,
        pos_start=pos_start,
        eps=1e-6,
    )
    _assert_numeric_equal(actual, expected, exact=dtype != mx.float32)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize(
    "seed,blocks,ratio,head_dim,rotary_dim,block_start",
    [
        (11, 1, 4, 128, 64, 0),
        (12, 1, 4, 128, 64, 513),
        (13, 5, 4, 128, 64, 1021),
        (16, 5, 4, 128, 64, 16_383),
        (14, 3, 2, 32, 16, 7),
        (15, 2, 3, 7, 6, 2),
        (17, 2, 4, 128, 64, 249_999),
        (18, 2, 4, 128, 64, 262_142),
    ],
)
def test_pool_prepare_matches_eager(
    dtype,
    seed: int,
    blocks: int,
    ratio: int,
    head_dim: int,
    rotary_dim: int,
    block_start: int,
):
    mx.random.seed(seed)
    raw = mx.random.normal((1, blocks * ratio, head_dim)).astype(dtype)
    weight = mx.random.uniform(
        low=0.5,
        high=1.5,
        shape=(head_dim,),
    ).astype(dtype)
    inv_freq = _inv_freq(rotary_dim)
    expected = _pool_oracle(
        raw,
        weight,
        inv_freq,
        block_start,
        ratio,
        1e-6,
    )
    actual = qsa_indexer_pool_keys_metal(
        raw,
        weight,
        inv_freq,
        block_start=block_start,
        compress_ratio=ratio,
        eps=1e-6,
    )
    _assert_numeric_equal(actual, expected, exact=dtype != mx.float32)


def test_prepare_handles_split_and_cache_slice_strides():
    """The real inputs are views from qk split and positional cache slices."""

    dtype = mx.bfloat16
    mx.random.seed(31)
    rows, heads, head_dim, rotary_dim = 3, 4, 128, 64
    projected = mx.random.normal((1, rows, (heads + 1) * head_dim)).astype(dtype)
    raw_q, raw_k = mx.split(projected, [heads * head_dim], axis=-1)
    raw_q = raw_q.reshape(1, rows, heads, head_dim)
    raw_backing = mx.zeros((1, 32, head_dim), dtype=dtype)
    raw_backing[:, 9 : 9 + rows, :] = raw_k
    raw_slice = raw_backing[:, 8:12, :]
    weight = mx.random.uniform(low=0.5, high=1.5, shape=(head_dim,)).astype(dtype)
    inv_freq = _inv_freq(rotary_dim)

    expected_q = _query_oracle(raw_q, weight, inv_freq, 4093, 1e-6)
    actual_q = qsa_indexer_prepare_queries_metal(
        raw_q,
        weight,
        inv_freq,
        pos_start=mx.array([4093], dtype=mx.int32),
        eps=1e-6,
    )
    expected_pool = _pool_oracle(raw_slice, weight, inv_freq, 2, 4, 1e-6)
    actual_pool = qsa_indexer_pool_keys_metal(
        raw_slice,
        weight,
        inv_freq,
        block_start=mx.array([2], dtype=mx.int32),
        compress_ratio=4,
        eps=1e-6,
    )
    _assert_numeric_equal(actual_q, expected_q, exact=True)
    _assert_numeric_equal(actual_pool, expected_pool, exact=True)


def test_real_prefill_sized_query_preparation_is_bit_identical():
    """The production 2,048-row prefill chunk remains one exact dispatch."""

    dtype = mx.bfloat16
    rows, heads, head_dim, rotary_dim = 2_048, 4, 128, 64
    mx.random.seed(37)
    raw = mx.random.normal((1, rows, heads, head_dim)).astype(dtype)
    weight = mx.random.uniform(
        low=0.5,
        high=1.5,
        shape=(head_dim,),
    ).astype(dtype)
    inv_freq = _inv_freq(rotary_dim)
    expected = _query_oracle(raw, weight, inv_freq, 65_533, 1e-6)
    actual = qsa_indexer_prepare_queries_metal(
        raw,
        weight,
        inv_freq,
        pos_start=65_533,
        eps=1e-6,
    )
    _assert_numeric_equal(actual, expected, exact=True)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "factor,max_position,pos_start,block_start",
    [
        (2.0, 524_288, 524_286, 131_070),
        (4.0, 1_048_576, 1_048_574, 262_142),
    ],
)
def test_static_yarn_query_and_pool_preparation_match_eager(
    dtype,
    factor: float,
    max_position: int,
    pos_start: int,
    block_start: int,
):
    """The fused indexer applies the same static-YaRN scale as attention."""

    args = TextArgs(
        max_position_embeddings=max_position,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
            "factor": factor,
            "original_max_position_embeddings": 262_144,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
    )
    inv_freq, attention_scaling = _rope_inv_freq_and_scaling(args)
    mx.random.seed(int(factor * 101))
    raw_q = mx.random.normal((1, 2, 4, 128)).astype(dtype)
    raw_k = mx.random.normal((1, 8, 128)).astype(dtype)
    weight = mx.random.uniform(low=0.5, high=1.5, shape=(128,)).astype(dtype)

    expected_q = _query_oracle(
        raw_q,
        weight,
        inv_freq,
        pos_start,
        1e-6,
        attention_scaling,
    )
    actual_q = qsa_indexer_prepare_queries_metal(
        raw_q,
        weight,
        inv_freq,
        pos_start=pos_start,
        eps=1e-6,
        attention_scaling=attention_scaling,
    )
    expected_k = _pool_oracle(
        raw_k,
        weight,
        inv_freq,
        block_start,
        4,
        1e-6,
        attention_scaling,
    )
    actual_k = qsa_indexer_pool_keys_metal(
        raw_k,
        weight,
        inv_freq,
        block_start=block_start,
        compress_ratio=4,
        eps=1e-6,
        attention_scaling=attention_scaling,
    )
    _assert_numeric_equal(actual_q, expected_q, exact=True)
    _assert_numeric_equal(actual_k, expected_k, exact=True)


def test_query_prepare_compiles_once_with_dynamic_position():
    """A tensor offset must remain a replay input rather than a trace constant."""

    dtype = mx.bfloat16
    mx.random.seed(41)
    raw = mx.random.normal((1, 1, 4, 128)).astype(dtype)
    weight = mx.random.uniform(low=0.5, high=1.5, shape=(128,)).astype(dtype)
    inv_freq = _inv_freq(64)
    # Compiled closures in production capture evaluated model parameters.
    # Materialize the synthetic equivalents before tracing so the test is not
    # also exercising lazy-random closure capture.
    mx.eval(raw, weight, inv_freq)

    @mx.compile
    def compiled(values, position):
        return qsa_indexer_prepare_queries_metal(
            values,
            weight,
            inv_freq,
            pos_start=position,
            eps=1e-6,
        )

    for position in (2047, 2048, 4097):
        dynamic = mx.array([position], dtype=mx.int32)
        actual = compiled(raw, dynamic)
        expected = _query_oracle(raw, weight, inv_freq, position, 1e-6)
        _assert_numeric_equal(actual, expected, exact=True)


def test_pool_rejects_partial_completed_block():
    raw = mx.zeros((1, 5, 8), dtype=mx.bfloat16)
    weight = mx.ones((8,), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="positive multiple"):
        qsa_indexer_pool_keys_metal(
            raw,
            weight,
            _inv_freq(4),
            block_start=0,
            compress_ratio=4,
            eps=1e-6,
        )
