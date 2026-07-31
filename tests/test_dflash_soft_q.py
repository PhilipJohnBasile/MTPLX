from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.dflash_source import DFlashDraftSource
from mtplx.fast_sampling import sparse_distributions_from_mlx_logits
from mtplx.sampling import SamplerConfig, distribution_from_logits


@pytest.fixture(autouse=True)
def _cpu_device():
    """Keep each test on CPU without leaking global device state."""

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _IdentityLayer:
    def __call__(self, values):
        return values


class _Target:
    def __init__(self) -> None:
        self.layers = [_IdentityLayer(), _IdentityLayer()]


class _SoftDraft:
    def __init__(self, *, block_size: int = 4) -> None:
        self.config = SimpleNamespace(
            target_layer_ids=(0, 1),
            block_size=block_size,
            mask_token_id=7,
        )
        self.calls = 0

    def bind(self, _model) -> None:
        pass

    def make_cache(self):
        return []

    def __call__(self, block, target_context, cache, *, logits_start):
        del block, target_context, cache
        assert logits_start == 1
        self.calls += 1
        # Each row has a non-degenerate, distinct proposal distribution.
        return mx.array(
            [[
                [2.0, 1.0, 0.0, -1.0, -2.0],
                [-2.0, 2.0, 1.0, 0.0, -1.0],
                [-1.0, -2.0, 2.0, 1.0, 0.0],
            ]],
            dtype=mx.float32,
        )


def _prepared_source(*, staged_k1: bool = False):
    target = _Target()
    draft = _SoftDraft()
    source = DFlashDraftSource(
        "soft-test",
        block_size=4,
        staged_k1=staged_k1,
        draft_model=draft,
    )
    source.prepare(SimpleNamespace(model=target))
    source.begin_request()
    for layer in target.layers:
        layer(mx.ones((1, 1, 2)))
    return source, target, draft


def _soft_sampler() -> SamplerConfig:
    return SamplerConfig(temperature=0.6, top_p=0.95, top_k=3)


def test_product_sampler_sparse_q_matches_reference_distribution() -> None:
    rng = np.random.default_rng(17)
    logits = rng.normal(size=(4, 64)).astype(np.float32)
    sampler = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)

    rows = sparse_distributions_from_mlx_logits(mx.array(logits), sampler)

    assert rows is not None
    assert len(rows) == len(logits)
    for row_logits, sparse in zip(logits, rows, strict=True):
        expected = distribution_from_logits(row_logits, sampler)
        assert np.allclose(sparse.to_dense(), expected, rtol=2e-6, atol=1e-8)


def test_soft_q_proposal_declares_each_sampled_distribution() -> None:
    source, _target, draft = _prepared_source()
    proposal = source.propose(
        primary_token=1,
        max_draft_tokens=3,
        committed_tokens=(),
        target_hidden=None,
        draft_sampler=_soft_sampler(),
        rng=np.random.default_rng(7),
    )

    assert proposal.source == "dflash-soft-q"
    assert proposal.metadata["declaration"] == "sampled_q"
    assert proposal.metadata["sampler"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 3,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    assert (
        proposal.metadata["sampling_policy"]
        == "host_request_rng_independent_rows"
    )
    assert proposal.metadata["draft_forward_calls"] == 1
    assert proposal.metadata["proposal_block_from_single_forward"] is True
    assert proposal.metadata["q_support_sizes"] == [2, 2, 2]
    assert draft.calls == 1
    assert proposal.draft_qs is not None
    assert len(proposal.draft_qs) == len(proposal.tokens) == 3
    assert [q.token_ids.tolist() for q in proposal.draft_qs] == [
        [0, 1],
        [1, 2],
        [2, 3],
    ]
    assert all(
        distribution.probability(token) > 0.0
        for token, distribution in zip(
            proposal.tokens, proposal.draft_qs, strict=True
        )
    )


def test_greedy_proposal_explicitly_declares_point_mass_q() -> None:
    source, _target, _draft = _prepared_source()
    proposal = source.propose(
        primary_token=1,
        max_draft_tokens=3,
        committed_tokens=(),
        target_hidden=None,
        draft_sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        rng=np.random.default_rng(3),
    )

    assert proposal.source == "dflash-one-hot"
    assert proposal.metadata["declaration"] == "one_hot"
    assert proposal.draft_qs is not None
    assert all(
        distribution.probability(token) == 1.0
        and len(distribution.token_ids) == 1
        for token, distribution in zip(
            proposal.tokens, proposal.draft_qs, strict=True
        )
    )


def test_soft_q_fails_closed_without_bounded_support_or_request_rng() -> None:
    source, _target, draft = _prepared_source()

    with pytest.raises(ValueError, match="top_k > 0"):
        source.propose(
            primary_token=1,
            max_draft_tokens=3,
            committed_tokens=(),
            target_hidden=None,
            draft_sampler=SamplerConfig(
                temperature=0.6, top_p=0.95, top_k=0
            ),
            rng=np.random.default_rng(7),
        )
    assert draft.calls == 0

    with pytest.raises(ValueError, match="request RNG"):
        source.propose(
            primary_token=1,
            max_draft_tokens=3,
            committed_tokens=(),
            target_hidden=None,
            draft_sampler=_soft_sampler(),
            rng=None,
        )
    assert draft.calls == 0


def test_staged_soft_q_token_and_distribution_queues_clear_together() -> None:
    source, target, draft = _prepared_source(staged_k1=True)
    proposal = source.propose(
        primary_token=1,
        max_draft_tokens=1,
        committed_tokens=(),
        target_hidden=None,
        draft_sampler=_soft_sampler(),
        rng=np.random.default_rng(11),
    )

    assert proposal.draft_qs is not None
    assert len(source._queued_tokens) == len(source._queued_distributions) == 2
    assert draft.calls == 1

    source.begin_target_verify()
    for layer in target.layers:
        layer(mx.ones((1, 2, 2)))
    source.commit_target_prefix(0)
    assert source._queued_tokens == []
    assert source._queued_distributions == []


def test_staged_rejection_cannot_treat_residual_row_as_an_accepted_draft() -> None:
    source, target, _draft = _prepared_source(staged_k1=True)
    source.propose(
        primary_token=1,
        max_draft_tokens=1,
        committed_tokens=(),
        target_hidden=None,
        draft_sampler=_soft_sampler(),
        rng=np.random.default_rng(19),
    )
    assert source._queued_tokens

    source.begin_target_verify()
    for layer in target.layers:
        layer(mx.ones((1, 2, 2)))
    source.commit_target_prefix(
        0,
        committed_target_rows=2,
        residual_correction_rows=1,
    )

    assert source._queued_tokens == []
    assert source._queued_distributions == []
