from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.dflash_source import BlockDraftProposal, DFlashDraftSource
from mtplx.generation import generate_ar, generate_mtpk
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig


@pytest.fixture(autouse=True)
def _force_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return " ".join(str(int(token)) for token in tokens)


def test_block_draft_proposal_preserves_legacy_positional_fields() -> None:
    proposal = BlockDraftProposal((7,), "legacy", 0.25, {"route": "old"})

    assert proposal.elapsed_s == 0.25
    assert proposal.metadata == {"route": "old"}
    assert proposal.draft_qs is None


class _ScriptedModel:
    def __init__(self, vocab: int = 64) -> None:
        self.vocab = vocab
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        return hidden_states

    def _logits(self, tokens):
        rows = []
        for token in tokens:
            row = [0.0] * self.vocab
            row[(int(token) + 1) % self.vocab] = 10.0
            rows.append(row)
        return mx.array([rows], dtype=mx.float32)

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden=False,
        hidden_variant=None,
        emit_logits=True,
        logits_keep=None,
    ):
        del cache, hidden_variant
        tokens = [int(token) for token in np.asarray(input_ids).reshape(-1)]
        keep = (
            len(tokens)
            if logits_keep is None
            else min(len(tokens), max(1, int(logits_keep)))
        )
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        logits = self._logits(tokens[-keep:])
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        return_hidden=False,
        mtp_hidden_variant=None,
        position_offset=None,
    ):
        del hidden_states, mtp_cache, concat_order, mtp_hidden_variant, position_offset
        tokens = [int(token) for token in np.asarray(next_token_ids).reshape(-1)]
        logits = self._logits(tokens)
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        return (logits, hidden) if return_hidden else logits


class _ExactBlockSource:
    def __init__(self) -> None:
        self.prepared = 0
        self.requests = 0
        self.commits: list[int] = []
        self.verify_epochs = 0

    def prepare(self, runtime) -> None:
        assert runtime.mtp_enabled
        self.prepared += 1

    def begin_request(self) -> None:
        self.requests += 1

    def propose(
        self,
        *,
        primary_token,
        max_draft_tokens,
        committed_tokens,
        target_hidden,
    ):
        del committed_tokens, target_hidden
        tokens = tuple(
            (int(primary_token) + offset + 1) % 64
            for offset in range(max_draft_tokens)
        )
        return BlockDraftProposal(
            tokens=tokens,
            source="fake-dflash-one-hot",
            metadata={"declaration": "one_hot"},
        )

    def begin_target_verify(self) -> None:
        self.verify_epochs += 1

    def commit_target_prefix(self, accepted_draft_tokens) -> None:
        self.commits.append(int(accepted_draft_tokens))


class _IdentityLayer:
    def __call__(self, values):
        return values


class _TapModel:
    def __init__(self) -> None:
        self.layers = [_IdentityLayer(), _IdentityLayer()]


class _ScriptedDFlashDraft:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            target_layer_ids=(0, 1),
            block_size=8,
            mask_token_id=63,
        )
        self.bound_model = None
        self.calls = 0

    def bind(self, model) -> None:
        self.bound_model = model

    def make_cache(self):
        return []

    def __call__(self, block, target_context, cache, *, logits_start):
        del block, target_context, cache
        assert logits_start == 1
        self.calls += 1
        logits = np.zeros((1, 7, 32), dtype=np.float32)
        for row, token in enumerate(range(10, 17)):
            logits[0, row, token] = 10.0
        return mx.array(logits)


def _runtime() -> MTPLXRuntime:
    return MTPLXRuntime(
        model=_ScriptedModel(),
        tokenizer=_Tokenizer(),
        model_path=Path("tiny-dflash"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _sampler() -> SamplerConfig:
    return SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)


def test_block_source_is_kill_switched_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_DFLASH_DRAFT", raising=False)
    with pytest.raises(RuntimeError, match="MTPLX_DFLASH_DRAFT=1"):
        generate_mtpk(
            _runtime(),
            [0],
            max_tokens=12,
            sampler=_sampler(),
            speculative_depth=3,
            stop_token_ids=set(),
            verify_strategy="capture_commit",
            block_draft_source=_ExactBlockSource(),
        )


def test_one_hot_block_source_matches_ar_and_commits_only_accepted_rows(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MTPLX_DFLASH_DRAFT", "1")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    source = _ExactBlockSource()

    speculative = generate_mtpk(
        _runtime(),
        [0],
        max_tokens=24,
        sampler=_sampler(),
        speculative_depth=3,
        stop_token_ids=set(),
        verify_strategy="capture_commit",
        block_draft_source=source,
    )
    baseline = generate_ar(
        _runtime(),
        [0],
        max_tokens=24,
        sampler=_sampler(),
        stop_token_ids=set(),
    )

    assert speculative.tokens == baseline.tokens
    assert source.prepared == 1
    assert source.requests == 1
    assert source.verify_epochs == len(source.commits)
    assert source.commits
    assert all(0 <= accepted <= 3 for accepted in source.commits)
    block_events = [
        event for event in speculative.stats.events if "block_draft_source" in event
    ]
    assert block_events
    assert all(
        event["block_draft_source"]["source"] == "fake-dflash-one-hot"
        for event in block_events
    )


def test_staged_source_reuses_one_block_and_discards_queue_on_rejection() -> None:
    target = _TapModel()
    original_layers = tuple(target.layers)
    draft = _ScriptedDFlashDraft()
    source = DFlashDraftSource(
        "scripted",
        block_size=8,
        staged_k1=True,
        draft_model=draft,
    )
    source.prepare(SimpleNamespace(model=target))
    source.begin_request()

    assert draft.bound_model is target
    assert source.target_model is target
    assert source.target_layer_count == 2

    for layer in target.layers:
        layer(mx.ones((1, 1, 2)))
    first = source.propose(
        primary_token=1,
        max_draft_tokens=1,
        committed_tokens=(),
        target_hidden=None,
    )
    assert first.tokens == (10,)
    assert first.metadata["queued_tokens_remaining"] == 6
    assert draft.calls == 1

    source.begin_target_verify()
    for layer in target.layers:
        layer(mx.ones((1, 2, 2)))
    source.commit_target_prefix(1)
    second = source.propose(
        primary_token=11,
        max_draft_tokens=1,
        committed_tokens=(10,),
        target_hidden=None,
    )
    assert second.tokens == (12,)
    assert second.metadata["queued_tokens_remaining"] == 4
    assert draft.calls == 1

    source.begin_target_verify()
    for layer in target.layers:
        layer(mx.ones((1, 2, 2)))
    source.commit_target_prefix(0)
    third = source.propose(
        primary_token=99,
        max_draft_tokens=1,
        committed_tokens=(10, 99),
        target_hidden=None,
    )
    assert third.tokens == (10,)
    assert draft.calls == 2

    source.begin_target_verify()
    for layer in target.layers:
        layer(mx.ones((1, 2, 2)))
    source.commit_target_prefix(0)
    source.close()
    assert tuple(target.layers) == original_layers
