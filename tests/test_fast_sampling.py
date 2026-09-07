import math

import mlx.core as mx
import numpy as np
import pytest

import mtplx.fast_sampling as fs
from mtplx.fast_sampling import (
    batched_sparse_distributions_from_mlx_logits,
    sparse_distribution_from_mlx_logits,
    sparse_distribution_from_mlx_logits_relaxed_ties,
    sparse_distributions_from_mlx_logits,
)
from mtplx.sampling import SamplerConfig, distribution_from_logits
from mtplx.sampling import sample_from_distribution


def test_sparse_distribution_nan_mass_falls_back_to_one_hot():
    dist = sparse_distribution_from_mlx_logits(
        mx.array([math.nan, math.nan], dtype=mx.float32),
        SamplerConfig(temperature=0.6, top_p=0.95, top_k=2),
    )

    assert dist is not None
    assert np.isfinite(dist.probs).all()
    assert dist.probs.sum() == 1.0


def test_sparse_distribution_batch_nan_mass_falls_back_to_one_hot():
    dists = sparse_distributions_from_mlx_logits(
        mx.array([[math.nan, math.nan], [1.0, 0.0]], dtype=mx.float32),
        SamplerConfig(temperature=0.6, top_p=0.95, top_k=2),
    )

    assert dists is not None
    assert len(dists) == 2
    assert np.isfinite(dists[0].probs).all()
    assert dists[0].probs.sum() == 1.0


def test_batched_sparse_distribution_nan_mass_falls_back_to_one_hot():
    batch = batched_sparse_distributions_from_mlx_logits(
        mx.array([[math.nan, math.nan], [1.0, 0.0]], dtype=mx.float32),
        SamplerConfig(temperature=0.6, top_p=0.95, top_k=2),
    )

    assert batch is not None
    assert np.isfinite(batch.probs).all()
    assert np.allclose(batch.probs.sum(axis=1), 1.0)


def test_top_p_one_sparse_distribution_matches_top_k_filtered_sampler():
    logits = np.array([1.0, 4.0, 3.0, 2.0], dtype=np.float32)
    config = SamplerConfig(temperature=0.6, top_p=1.0, top_k=2)

    sparse = sparse_distribution_from_mlx_logits(mx.array(logits), config)
    dense = distribution_from_logits(logits, config)

    assert sparse is not None
    assert set(sparse.token_ids.tolist()) == {1, 2}
    assert np.allclose(sparse.to_dense(), dense)


def test_top_p_one_batched_sparse_distribution_matches_top_k_filtered_sampler():
    logits = np.array(
        [[1.0, 4.0, 3.0, 2.0], [5.0, 2.0, 4.0, 1.0]],
        dtype=np.float32,
    )
    config = SamplerConfig(temperature=0.6, top_p=1.0, top_k=2)

    batch = batched_sparse_distributions_from_mlx_logits(mx.array(logits), config)

    assert batch is not None
    for row in range(logits.shape[0]):
        dense = distribution_from_logits(logits[row], config)
        assert np.allclose(batch.to_distribution(row).to_dense(), dense)


def test_batched_sparse_distribution_matches_default_top_p_top_k_sampler():
    logits = np.random.default_rng(44).normal(size=(8, 64)).astype(np.float32)
    config = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)

    batch = batched_sparse_distributions_from_mlx_logits(mx.array(logits), config)

    assert batch is not None
    for row in range(logits.shape[0]):
        dense = distribution_from_logits(logits[row], config)
        assert np.allclose(
            batch.to_distribution(row).to_dense(), dense, rtol=1e-5, atol=1e-7
        )


def test_batched_sparse_sampling_preserves_dense_vocab_order_and_rng():
    logits = np.array([[0.0, 2.0, -1.0, -2.0, 3.0]], dtype=np.float32)
    config = SamplerConfig(temperature=1.0, top_p=1.0, top_k=2)
    dense = distribution_from_logits(logits[0], config)
    batch = batched_sparse_distributions_from_mlx_logits(mx.array(logits), config)
    dense_rng = np.random.default_rng(2)
    sparse_rng = np.random.default_rng(2)

    assert batch is not None
    assert batch.sample(0, sparse_rng) == sample_from_distribution(dense, dense_rng)
    assert sparse_rng.random() == dense_rng.random()


def test_single_sparse_sampling_preserves_dense_vocab_order_and_rng():
    logits = np.array([0.0, 2.0, -1.0, -2.0, 3.0], dtype=np.float32)
    config = SamplerConfig(temperature=1.0, top_p=1.0, top_k=2)
    dense = distribution_from_logits(logits, config)
    sparse = sparse_distribution_from_mlx_logits(mx.array(logits), config)
    dense_rng = np.random.default_rng(2)
    sparse_rng = np.random.default_rng(2)

    assert sparse is not None
    assert sample_from_distribution(sparse, sparse_rng) == sample_from_distribution(
        dense, dense_rng
    )
    assert sparse_rng.random() == dense_rng.random()


def test_relaxed_ties_match_exact_support_probabilities_and_rng_without_ties():
    logits = np.random.default_rng(91).normal(size=256).astype(np.float32)
    config = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
    exact = sparse_distribution_from_mlx_logits(mx.array(logits), config)
    relaxed = sparse_distribution_from_mlx_logits_relaxed_ties(
        mx.array(logits), config
    )
    exact_rng = np.random.default_rng(20260829)
    relaxed_rng = np.random.default_rng(20260829)

    assert exact is not None
    assert relaxed is not None
    assert np.array_equal(relaxed.token_ids, exact.token_ids)
    assert np.array_equal(relaxed.probs, exact.probs)
    assert sample_from_distribution(relaxed, relaxed_rng) == sample_from_distribution(
        exact, exact_rng
    )
    assert relaxed_rng.random() == exact_rng.random()


def test_sparse_row_list_sampling_preserves_dense_vocab_order_and_rng():
    logits = np.array([[0.0, 2.0, -1.0, -2.0, 3.0]], dtype=np.float32)
    config = SamplerConfig(temperature=1.0, top_p=1.0, top_k=2)
    dense = distribution_from_logits(logits[0], config)
    sparse_rows = sparse_distributions_from_mlx_logits(mx.array(logits), config)
    dense_rng = np.random.default_rng(2)
    sparse_rng = np.random.default_rng(2)

    assert sparse_rows is not None
    assert sample_from_distribution(
        sparse_rows[0], sparse_rng
    ) == sample_from_distribution(dense, dense_rng)
    assert sparse_rng.random() == dense_rng.random()


def test_bound_batched_top_k_route_bypasses_generic_checks_and_fails_nan(
    monkeypatch,
):
    config = SamplerConfig(temperature=0.6, top_p=0.95, top_k=2)
    execute = fs.bind_batched_top_k_distributions(config, vocab_size=5)

    monkeypatch.setattr(
        fs,
        "batched_sparse_distributions_from_mlx_logits",
        lambda *_args, **_kwargs: pytest.fail("bound route used generic helper"),
    )
    batch = execute(mx.array([[0.0, 2.0, -1.0, -2.0, 3.0]]))

    assert batch.token_ids.shape == (1, 2)
    with pytest.raises(FloatingPointError, match="finite positive mass"):
        execute(mx.array([[math.nan] * 5]))


def test_bound_top_p_route_matches_dense_float64_nucleus_boundary_and_rng():
    logits = np.array([[0.0, 2.0, -1.0, -2.0, 3.0]], dtype=np.float32)
    config = SamplerConfig(
        temperature=0.6,
        top_p=0.8353335822095811,
        top_k=2,
    )
    dense = distribution_from_logits(logits[0], config)
    execute = fs.bind_batched_top_k_distributions(config, vocab_size=5)
    batch = execute(mx.array(logits))
    dense_rng = np.random.default_rng(2)
    sparse_rng = np.random.default_rng(2)

    assert set(batch.to_distribution(0).token_ids.tolist()) == set(
        np.flatnonzero(dense).tolist()
    )
    assert batch.sample(0, sparse_rng) == sample_from_distribution(dense, dense_rng)
    assert sparse_rng.random() == dense_rng.random()


@pytest.mark.parametrize("top_p", [0.95, 1.0])
def test_bound_route_and_dense_sampler_share_bf16_top_k_tie_break(top_p):
    logits = np.zeros(32, dtype=np.float32)
    logits[2] = 2.0
    config = SamplerConfig(temperature=0.6, top_p=top_p, top_k=20)
    dense = distribution_from_logits(logits, config)
    execute = fs.bind_batched_top_k_distributions(config, vocab_size=32)
    mlx_logits = mx.array(logits[None, :], dtype=mx.bfloat16)
    batch = execute(mlx_logits)
    single = sparse_distribution_from_mlx_logits(mlx_logits[0], config)
    row_list = sparse_distributions_from_mlx_logits(mlx_logits, config)
    generic_batch = batched_sparse_distributions_from_mlx_logits(mlx_logits, config)
    dense_rng = np.random.default_rng(5)
    sparse_rng = np.random.default_rng(5)

    # Equal scores at the cutoff keep the lower vocabulary ids.
    assert set(np.flatnonzero(dense).tolist()) == set(range(20))
    assert set(batch.to_distribution(0).token_ids.tolist()) == set(range(20))
    assert single is not None
    assert set(single.token_ids.tolist()) == set(range(20))
    assert row_list is not None
    assert set(row_list[0].token_ids.tolist()) == set(range(20))
    assert generic_batch is not None
    assert set(generic_batch.to_distribution(0).token_ids.tolist()) == set(range(20))
    assert np.allclose(batch.to_distribution(0).to_dense(), dense)
    assert batch.sample(0, sparse_rng) == sample_from_distribution(dense, dense_rng)
    assert sparse_rng.random() == dense_rng.random()


def test_large_top_k_distribution_routes_do_not_use_quadratic_device_order(
    monkeypatch,
):
    config = SamplerConfig(temperature=0.6, top_p=0.95, top_k=128)
    logits = mx.zeros((1, 128), dtype=mx.bfloat16)
    monkeypatch.setattr(
        fs,
        "_order_bounded_mlx_top_k_support",
        lambda *_args: pytest.fail("distribution route used quadratic ordering"),
    )

    single = sparse_distribution_from_mlx_logits(logits[0], config)
    batch = fs.bind_batched_top_k_distributions(config, vocab_size=128)(logits)

    assert single is not None
    assert single.token_ids.size > 0
    assert batch.token_ids.shape == (1, 128)


@pytest.mark.parametrize("top_p", [0.95, 1.0])
def test_reproduced_bf16_cutoff_tie_has_exact_serial_rng_parity(top_p):
    raw = np.random.default_rng(8).normal(size=(1000, 128)).astype(np.float32)
    bf16_rows = mx.array(raw).astype(mx.bfloat16).astype(mx.float32)
    mx.eval(bf16_rows)
    logits = np.asarray(bf16_rows, dtype=np.float32)[406]
    config = SamplerConfig(temperature=0.6, top_p=top_p, top_k=20)
    dense = distribution_from_logits(logits, config)
    batch = fs.bind_batched_top_k_distributions(config, vocab_size=128)(
        mx.array(logits[None, :], dtype=mx.bfloat16)
    )
    dense_rng = np.random.default_rng(5)
    batch_rng = np.random.default_rng(5)

    assert logits[50] == logits[122] == 0.9921875
    assert dense[50] > 0.0
    assert dense[122] == 0.0
    assert batch.probability(0, 50) > 0.0
    assert batch.probability(0, 122) == 0.0
    assert sample_from_distribution(dense, dense_rng) == 104
    assert batch.sample(0, batch_rng) == 104
    assert batch_rng.random() == dense_rng.random()
