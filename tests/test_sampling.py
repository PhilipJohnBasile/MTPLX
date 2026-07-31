from __future__ import annotations

import numpy as np
import mlx.core as mx

from mtplx.fast_sampling import sample_token_ids_from_mlx_logits
from mtplx.sampling import (
    SamplerConfig,
    SparseDistribution,
    acceptance_probability,
    distribution_from_logits,
    residual_distribution,
    sample_from_distribution,
    speculative_output_marginal,
    verify_one_token,
)


def test_distribution_from_logits_normalizes_after_filtering():
    logits = np.array([4.0, 3.0, 2.0, 1.0])
    probs = distribution_from_logits(logits, SamplerConfig(temperature=0.6, top_p=0.9, top_k=2))
    assert np.isclose(probs.sum(), 1.0)
    assert np.count_nonzero(probs) <= 2


def test_acceptance_probability_caps_at_one():
    target = np.array([0.8, 0.2])
    draft = np.array([0.4, 0.6])
    assert acceptance_probability(target, draft, 0) == 1.0
    assert np.isclose(acceptance_probability(target, draft, 1), 1.0 / 3.0)


def test_residual_distribution_uses_positive_target_minus_draft_mass():
    target = np.array([0.6, 0.3, 0.1])
    draft = np.array([0.2, 0.5, 0.3])
    residual = residual_distribution(target, draft)
    assert np.isclose(residual.sum(), 1.0)
    assert residual[0] == 1.0
    assert residual[1] == 0.0


def test_verify_one_token_rejects_into_residual_when_random_is_high():
    target = np.array([0.2, 0.8])
    draft = np.array([0.8, 0.2])
    rng = np.random.default_rng(1)
    decision = verify_one_token(target, draft, 0, rng)
    assert decision.accept_probability == 0.25
    if not decision.accepted:
        assert decision.token_id == 1


def test_zero_uniform_draw_cannot_accept_zero_probability_draft():
    class ZeroRng:
        def random(self):
            return 0.0

        def choice(self, values, p):
            return np.asarray(values)[int(np.argmax(p))]

    decision = verify_one_token(
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        0,
        ZeroRng(),
    )

    assert decision.accept_probability == 0.0
    assert decision.accepted is False
    assert decision.token_id == 1


def test_speculative_output_marginal_recovers_target_distribution():
    target = np.array([0.55, 0.25, 0.15, 0.05])
    draft = np.array([0.10, 0.55, 0.20, 0.15])
    marginal = speculative_output_marginal(target, draft)
    assert np.allclose(marginal, target)


def test_sparse_distribution_acceptance_and_residual():
    target = SparseDistribution(
        token_ids=np.array([2, 5, 9]),
        probs=np.array([0.5, 0.3, 0.2]),
        vocab_size=12,
    )
    draft = SparseDistribution.one_hot(5, vocab_size=12)

    assert np.isclose(acceptance_probability(target, draft, 5), 0.3)
    residual = residual_distribution(target, draft)
    assert isinstance(residual, SparseDistribution)
    assert residual.token_ids.tolist() == [2, 9]
    assert np.allclose(residual.probs, [5 / 7, 2 / 7])


def test_sparse_distribution_sampling_returns_original_token_ids():
    dist = SparseDistribution(
        token_ids=np.array([7, 11]),
        probs=np.array([0.0, 1.0]),
        vocab_size=12,
    )
    assert sample_from_distribution(dist, np.random.default_rng(0)) == 11


def test_sparse_distribution_zero_or_nan_mass_falls_back_to_safe_token():
    dist = SparseDistribution(
        token_ids=np.array([7, 11]),
        probs=np.array([0.0, np.nan]),
        vocab_size=12,
    )

    assert dist.token_ids.tolist() == [7]
    assert np.allclose(dist.probs, [1.0])


def test_residual_distribution_nan_mass_falls_back_to_target():
    target = np.array([0.25, 0.75])
    draft = np.array([np.nan, np.nan])

    residual = residual_distribution(target, draft)

    assert np.allclose(residual, target)


def test_sample_token_ids_from_mlx_logits_keeps_top_p_top1_on_device():
    logits = mx.array([[4.0, 3.0, 1.0], [0.0, 2.0, 5.0]], dtype=mx.float32)
    sampled = sample_token_ids_from_mlx_logits(
        logits,
        SamplerConfig(temperature=0.6, top_p=0.01, top_k=2),
    )
    mx.eval(sampled)

    assert np.asarray(sampled).tolist() == [0, 2]
