from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.dflash_source import BlockDraftProposal
from mtplx.generation import generate_ar, generate_mtp1, generate_mtpk
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig, SparseDistribution
from mtplx.server.openai import _server_runtime_env_overrides


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


def test_mtp1_public_sequential_strategy_reaches_prefill(monkeypatch) -> None:
    class _PrefillReached(Exception):
        pass

    runtime = SimpleNamespace(
        mtp_enabled=True,
        a3b_whole_moe_installed=False,
        tokenizer=_Tokenizer(),
    )
    monkeypatch.setattr(
        "mtplx.generation._runtime_counter_snapshot", lambda _rt: {}
    )

    def stop_at_prefill(*_args, **_kwargs):
        raise _PrefillReached

    monkeypatch.setattr("mtplx.generation._prefill", stop_at_prefill)
    with pytest.raises(_PrefillReached):
        generate_mtp1(
            runtime,
            [0],
            max_tokens=1,
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
            verify_strategy="sequential",
        )


class _TrimmableHistory:
    def __init__(self) -> None:
        self.values: list[int] = []

    @property
    def offset(self) -> int:
        return len(self.values)

    @property
    def state(self):
        return self.values

    def is_trimmable(self) -> bool:
        return True

    def trim(self, count: int) -> int:
        count = min(max(0, int(count)), len(self.values))
        if count:
            del self.values[-count:]
        return count


class _RecurrentHistory:
    def __init__(self) -> None:
        self.state: list[int] = []
        self.meta_state = ("scalar-oracle",)

    def is_trimmable(self) -> bool:
        return False

    def replace_state(self, state) -> None:
        self.state = list(state)


class _StatefulScalarModel:
    """Target whose decode log records every input width and committed token."""

    def __init__(self, vocab: int = 32) -> None:
        self.vocab = int(vocab)
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])
        self.decode_widths: list[int] = []
        self.latest_cache = None
        self.tap = None

    def make_cache(self):
        self.latest_cache = [_TrimmableHistory(), _RecurrentHistory()]
        return self.latest_cache

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        return hidden_states

    def _rows(self, tokens):
        rows = []
        for token in tokens:
            row = [-mx.inf] * self.vocab
            row[(int(token) + 1) % self.vocab] = 0.0
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
        del hidden_variant
        tokens = [int(token) for token in np.asarray(input_ids).reshape(-1)]
        if cache is not None:
            self.decode_widths.append(len(tokens))
            cache[0].values.extend(tokens)
            cache[1].state.extend(tokens)
        if self.tap is not None:
            self.tap.capture(tokens)
        hidden = mx.array(
            [[[float(token), 1.0] for token in tokens]], dtype=mx.float32
        )
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        keep = len(tokens) if logits_keep is None else min(len(tokens), int(logits_keep))
        logits = self._rows(tokens[-keep:])
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden_states, next_token_ids, **_kwargs):
        del hidden_states, _kwargs
        tokens = [int(token) for token in np.asarray(next_token_ids).reshape(-1)]
        return self._rows(tokens)


class _TrackingBlockSource:
    def __init__(self, *, mode: str, vocab: int = 32) -> None:
        self.mode = mode
        self.vocab = int(vocab)
        self.capture_active = False
        self.tap_rows: list[int] = []
        self.committed_rows: list[tuple[int, ...]] = []
        self.commit_arguments: list[int] = []

    def prepare(self, runtime) -> None:
        runtime.model.tap = self

    def begin_request(self) -> None:
        self.capture_active = False
        self.tap_rows.clear()

    def capture(self, tokens) -> None:
        if self.capture_active:
            self.tap_rows.extend(int(token) for token in tokens)

    def propose(
        self,
        *,
        primary_token,
        max_draft_tokens,
        committed_tokens,
        target_hidden,
        draft_sampler,
        rng,
    ):
        del committed_tokens, target_hidden, draft_sampler, rng
        delta = 1 if self.mode == "accept" else 2
        tokens = tuple(
            (int(primary_token) + delta + index) % self.vocab
            for index in range(int(max_draft_tokens))
        )
        draft_qs = None
        if self.mode == "soft-reject":
            draft_qs = tuple(
                SparseDistribution.one_hot(token, self.vocab) for token in tokens
            )
        return BlockDraftProposal(
            tokens=tokens,
            source=f"tracking-{self.mode}",
            draft_qs=draft_qs,
        )

    def begin_target_verify(self) -> None:
        self.tap_rows.clear()
        self.capture_active = True

    def commit_target_prefix(
        self,
        accepted_draft_tokens,
        *,
        committed_target_rows=None,
        residual_correction_rows=0,
    ) -> None:
        keep = (
            int(accepted_draft_tokens) + 1
            if committed_target_rows is None
            else int(committed_target_rows)
        )
        assert keep == int(accepted_draft_tokens) + 1 + int(residual_correction_rows)
        assert 1 <= keep <= len(self.tap_rows)
        self.commit_arguments.append(int(accepted_draft_tokens))
        self.committed_rows.append(tuple(self.tap_rows[:keep]))
        self.capture_active = False


class _FailingClosingSource(_TrackingBlockSource):
    def __init__(self) -> None:
        super().__init__(mode="accept")
        self.close_calls = 0
        self.model = None

    def prepare(self, runtime) -> None:
        super().prepare(runtime)
        self.model = runtime.model

    def propose(self, **_kwargs):
        raise RuntimeError("proposal failed")

    def close(self) -> None:
        self.close_calls += 1
        if self.model is not None:
            self.model.tap = None


def _runtime(model: _StatefulScalarModel) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        model_path=Path("scalar-oracle"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _configure(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_DFLASH_DRAFT", "1")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "0")


def test_external_sequential_oracle_matches_ar_and_repairs_one_row_at_a_time(
    monkeypatch,
) -> None:
    _configure(monkeypatch)
    # Direct callers can inherit the server fast-path value.  The oracle must
    # keep its recurrent snapshot regardless.
    monkeypatch.setenv("MTPLX_SKIP_VERIFY_SNAPSHOT", "1")
    source = _TrackingBlockSource(mode="reject")
    speculative_model = _StatefulScalarModel()
    speculative = generate_mtpk(
        _runtime(speculative_model),
        [0],
        max_tokens=12,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        speculative_depth=3,
        seed=4,
        stop_token_ids=set(),
        verify_strategy="sequential",
        block_draft_source=source,
    )
    baseline_model = _StatefulScalarModel()
    baseline = generate_ar(
        _runtime(baseline_model),
        [0],
        max_tokens=12,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        seed=4,
        stop_token_ids=set(),
    )

    assert speculative.tokens == baseline.tokens
    # Width one prompt keeps every model call observable as a decode call.
    assert speculative_model.decode_widths
    assert set(speculative_model.decode_widths) == {1}
    assert speculative_model.latest_cache[0].values == baseline_model.latest_cache[0].values
    assert speculative_model.latest_cache[1].state == baseline_model.latest_cache[1].state
    events = [
        event for event in speculative.stats.events if "block_draft_source" in event
    ]
    assert events
    assert all(event["sequential_verify"]["authoritative"] for event in events)
    assert all("rollback" in event.get("timing_s", {}) for event in events)
    expected_verify_forward_calls = sum(
        int(event["sequential_verify"]["forward_calls"]) for event in events
    )
    expected_repair_forward_calls = sum(
        int(event["sequential_repair"]["forward_calls"]) for event in events
    )
    assert speculative.stats.verify_calls == len(events)
    assert (
        speculative.stats.verify_forward_calls == expected_verify_forward_calls
    )
    assert (
        speculative.stats.repair_forward_calls == expected_repair_forward_calls
    )
    assert speculative.stats.verify_forward_calls > speculative.stats.verify_calls
    # Every wrong draft is excluded from the committed companion context.
    assert all(len(rows) == 1 for rows in source.committed_rows)


def test_sequential_soft_rejection_commits_residual_correction_tap(monkeypatch) -> None:
    _configure(monkeypatch)
    source = _TrackingBlockSource(mode="soft-reject")
    output = generate_mtpk(
        _runtime(_StatefulScalarModel()),
        [0],
        max_tokens=8,
        sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=1),
        draft_sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=1),
        speculative_depth=2,
        seed=9,
        stop_token_ids=set(),
        verify_strategy="sequential",
        block_draft_source=source,
    )

    event = next(
        event
        for event in output.stats.events
        if event.get("block_draft_source", {}).get("residual_correction_rows") == 1
    )
    committed = source.committed_rows[0]
    primary = int(event["primary"])
    wrong_draft = int(event["drafts"][0]["token"])
    correction = int(event["drafts"][0]["correction"])
    assert committed == (primary, correction)
    assert wrong_draft not in committed
    assert event["block_draft_source"]["accepted_draft_tokens"] == 0
    assert event["block_draft_source"]["committed_target_rows"] == 2


def test_external_oracle_disables_lazy_bonus_and_keeps_batched_denied(
    monkeypatch,
) -> None:
    _configure(monkeypatch)
    monkeypatch.setenv("MTPLX_LAZY_BONUS_VERIFY", "1")
    monkeypatch.setenv("MTPLX_LAZY_TARGET_DISTRIBUTIONS", "0")
    source = _TrackingBlockSource(mode="accept")
    speculative_model = _StatefulScalarModel()
    output = generate_mtpk(
        _runtime(speculative_model),
        [0],
        max_tokens=10,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        speculative_depth=3,
        stop_token_ids=set(),
        verify_strategy="sequential",
        block_draft_source=source,
    )
    baseline_model = _StatefulScalarModel()
    baseline = generate_ar(
        _runtime(baseline_model),
        [0],
        max_tokens=10,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        stop_token_ids=set(),
    )
    block_events = [
        event for event in output.stats.events if "block_draft_source" in event
    ]
    assert output.tokens == baseline.tokens
    # An all-accept verify owns the final draft row as well; generate_ar stops
    # at max_tokens before forwarding that last emitted token.  Both histories
    # are therefore exact at their documented boundaries.
    assert speculative_model.latest_cache[0].values == [0, *output.tokens]
    assert speculative_model.latest_cache[1].state == [0, *output.tokens]
    assert baseline_model.latest_cache[0].values == [0, *baseline.tokens[:-1]]
    assert baseline_model.latest_cache[1].state == [0, *baseline.tokens[:-1]]
    assert block_events
    assert all(not event["lazy_bonus_verify"]["enabled"] for event in block_events)
    assert all(len(rows) == argument + 1 for rows, argument in zip(
        source.committed_rows, source.commit_arguments, strict=True
    ))

    with pytest.raises(ValueError, match="requires sequential"):
        generate_mtpk(
            _runtime(_StatefulScalarModel()),
            [0],
            max_tokens=4,
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
            speculative_depth=2,
            stop_token_ids=set(),
            verify_strategy="batched",
            block_draft_source=_TrackingBlockSource(mode="accept"),
        )


@pytest.mark.parametrize("strategy", ["capture_commit", "target_prefix"])
def test_sampled_external_source_requires_sequential_before_prepare(
    monkeypatch, strategy
) -> None:
    _configure(monkeypatch)
    model = _StatefulScalarModel()
    source = _TrackingBlockSource(mode="soft-reject")

    with pytest.raises(ValueError, match="requires verify_strategy='sequential'"):
        generate_mtpk(
            _runtime(model),
            [0],
            max_tokens=4,
            sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=1),
            draft_sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=1),
            speculative_depth=2,
            stop_token_ids=set(),
            verify_strategy=strategy,
            block_draft_source=source,
        )

    # Preflight rejection happens before the source can install taps or mutate
    # companion state.
    assert model.tap is None


def test_server_fast_path_value_is_safe_because_oracle_forces_snapshot() -> None:
    args = argparse.Namespace(verify_strategy="sequential", generation_mode="mtp")
    overrides = _server_runtime_env_overrides(
        args, {"MTPLX_SKIP_VERIFY_SNAPSHOT": "1"}
    )
    # Keep the established server contract; generate_mtpk ignores this skip for
    # the sequential lane, as the forced-rejection test above proves.
    assert overrides["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "1"


def test_sequential_strategy_is_not_exposed_to_internal_mtp(monkeypatch) -> None:
    _configure(monkeypatch)
    with pytest.raises(ValueError, match="restricted to external block"):
        generate_mtpk(
            _runtime(_StatefulScalarModel()),
            [0],
            max_tokens=4,
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
            speculative_depth=2,
            stop_token_ids=set(),
            verify_strategy="sequential",
        )


def test_external_source_taps_are_closed_when_generation_raises(monkeypatch) -> None:
    _configure(monkeypatch)
    model = _StatefulScalarModel()
    source = _FailingClosingSource()

    with pytest.raises(RuntimeError, match="proposal failed"):
        generate_mtpk(
            _runtime(model),
            [0],
            max_tokens=4,
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
            speculative_depth=2,
            stop_token_ids=set(),
            verify_strategy="sequential",
            block_draft_source=source,
        )

    assert source.close_calls == 1
    assert model.tap is None
