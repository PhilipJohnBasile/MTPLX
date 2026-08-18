"""Runtime observability F35 → 2.8.3 candidate gating: armed-stream wire
semantics.

Deterministic no-model harness (pattern from test_loop_guard): a scripted
model walks distinct tokens then enters a fixed cycle, so the uncapped
repetition stop fires at a known step. The stream callback records every
wire batch.

Contract by mode (MTPLX_REPETITION_STREAM_HOLDBACK):
- "strict" (the 2.8.0-2.8.2 behavior): wire == final tokens byte for byte —
  the holdback is unconditional while armed.
- "candidate" (default since the 2.8.3 streaming-freeze fix): non-looping
  armed streams are LIVE (byte-identical to the disarmed wire — this is the
  product regression test for the 2.8.x every-chat freeze); when a real
  loop trims, wire divergence is bounded by the candidate engagement span
  (a short prefix of the trimmed suffix may have streamed).
- Disarmed requests keep the exact historical per-call emit pattern in
  every mode.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mtplx.generation import (
    RepetitionStopConfig,
    _repetition_stream_emit_limit,
    _repetition_stream_holdback_tokens,
    generate_ar,
    generate_mtpk,
)
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig

VOCAB = 64
LOOP_START = 40
PERIOD = 8
MARGIN = 10.0


def _next_token(token: int) -> int:
    nxt = int(token) + 1
    if nxt >= LOOP_START + PERIOD:
        return LOOP_START
    return nxt


def _next_token_fresh(token: int) -> int:
    return (int(token) + 1) % VOCAB


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(f"<{int(token)}>" for token in tokens)


class _ScriptedModel:
    """After token t the model deterministically wants script(t)."""

    def __init__(self, script) -> None:
        self._script = script

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        return hidden_states

    def _logits_for(self, last_tokens: list[int]) -> mx.array:
        rows = []
        for token in last_tokens:
            row = [0.0] * VOCAB
            row[self._script(int(token))] = MARGIN
            rows.append(row)
        return mx.array([rows], dtype=mx.float32)

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        tokens = [int(token) for token in np.asarray(input_ids).reshape(-1)]
        keep = (
            len(tokens)
            if logits_keep is None
            else min(len(tokens), max(1, int(logits_keep)))
        )
        logits = self._logits_for(tokens[-keep:]) if emit_logits else None
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        if return_hidden:
            return logits, hidden
        return logits


class _ScriptedMTPModel(_ScriptedModel):
    def __init__(self, script) -> None:
        super().__init__(script)
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        position_offset=None,
    ):
        tokens = [int(token) for token in np.asarray(next_token_ids).reshape(-1)]
        logits = self._logits_for(tokens)
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


def _runtime(model) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        model_path=Path("tiny-scripted"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _set_repetition_env(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MIN_TOKENS", "48")
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MIN_REPEATED_TOKENS", "16")
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MIN_REPEATS", "2")
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MIN_BLOCK_TOKENS", "1")
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MAX_BLOCK_TOKENS", "8")


GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)


# ---------------------------------------------------------------------------
# Emit-limit math.
# ---------------------------------------------------------------------------


def test_holdback_zero_when_disarmed() -> None:
    assert _repetition_stream_holdback_tokens(RepetitionStopConfig(enabled=False)) == 0
    assert _repetition_stream_emit_limit(37, RepetitionStopConfig(enabled=False), 0) == 37


def test_holdback_window_covers_default_first_fire_bound() -> None:
    config = RepetitionStopConfig(enabled=True)  # product defaults
    holdback = _repetition_stream_holdback_tokens(config)
    # Steady-state first-fire trim bound: max(min_repeats*max_block,
    # min_repeated+max_block) = max(384, 288); +64 multi-commit margin.
    assert holdback == 448
    # Short armed responses stream untouched (safe prefix 768-448=320).
    assert _repetition_stream_emit_limit(100, config, holdback) == 100
    assert _repetition_stream_emit_limit(320, config, holdback) == 320
    # Past the safe prefix, exactly `holdback` trailing tokens are held.
    assert _repetition_stream_emit_limit(400, config, holdback) == 320
    assert _repetition_stream_emit_limit(1000, config, holdback) == 552
    # Monotone: the wire cursor never moves backwards.
    limits = [
        _repetition_stream_emit_limit(total, config, holdback)
        for total in range(0, 1200, 7)
    ]
    assert limits == sorted(limits)


def test_holdback_covers_steady_state_trim() -> None:
    config = RepetitionStopConfig(
        enabled=True,
        min_tokens=48,
        min_repeated_tokens=16,
        min_repeats=2,
        min_block_tokens=1,
        max_block_tokens=8,
    )
    holdback = _repetition_stream_holdback_tokens(config)
    # Any steady-state fire trims at most max(2*8, 16+8) = 24 (+commits);
    # the emitted prefix at fire time is total-holdback, so the trimmed
    # region always stays inside the held tail.
    for total in (48, 56, 90, 200):
        emitted = _repetition_stream_emit_limit(total, config, holdback)
        assert emitted <= max(0, total - min(holdback, total))


# ---------------------------------------------------------------------------
# generate_ar: serial loop wire semantics.
# ---------------------------------------------------------------------------


def _collecting_callback():
    calls: list[list[int]] = []

    def callback(tokens: list[int]) -> None:
        calls.append([int(token) for token in tokens])

    return calls, callback


def test_ar_armed_stream_never_shows_trimmed_tokens_strict(monkeypatch):
    _set_repetition_env(monkeypatch)
    monkeypatch.setenv("MTPLX_REPETITION_STREAM_HOLDBACK", "strict")
    calls, callback = _collecting_callback()
    out = generate_ar(
        _runtime(_ScriptedModel(_next_token)),
        [0],
        max_tokens=200,
        sampler=GREEDY,
        seed=7,
        stop_token_ids=set(),
        token_callback=callback,
        repetition_stop=True,
    )
    # The cycle fires the trimmer at len 55 (rotated block [40,41..47]
    # aligns one step before the [41..47,40] alignment): 16 repeated
    # tokens are retracted, keeping values 1..39.
    assert list(out.tokens) == list(range(1, LOOP_START))
    assert out.finish_reason == "stop"
    assert any("repetition_stop" in event for event in out.stats.events)
    wire = [token for call in calls for token in call]
    # THE invariant: the wire shows exactly the post-trim tokens.
    assert wire == list(out.tokens)


def test_ar_armed_stream_flushes_full_tail_when_no_trim(monkeypatch):
    _set_repetition_env(monkeypatch)
    calls, callback = _collecting_callback()
    out = generate_ar(
        _runtime(_ScriptedModel(_next_token_fresh)),
        [0],
        max_tokens=30,  # below min_tokens: detector never fires
        sampler=GREEDY,
        seed=7,
        stop_token_ids=set(),
        token_callback=callback,
        repetition_stop=True,
    )
    wire = [token for call in calls for token in call]
    assert wire == list(out.tokens)
    assert len(out.tokens) == 30
    # The held tail flushed in full once the no-trim decision was known.
    assert calls, "armed stream must still deliver the response"


def test_ar_disarmed_stream_is_byte_identical_per_token(monkeypatch):
    _set_repetition_env(monkeypatch)
    calls, callback = _collecting_callback()
    out = generate_ar(
        _runtime(_ScriptedModel(_next_token)),
        [0],
        max_tokens=60,
        sampler=GREEDY,
        seed=7,
        stop_token_ids=set(),
        token_callback=callback,
        repetition_stop=False,
    )
    # Historical contract, unchanged: one callback per token, singletons.
    assert len(out.tokens) == 60
    assert calls == [[token] for token in out.tokens]


# ---------------------------------------------------------------------------
# generate_mtpk: speculative serial loop wire semantics.
# ---------------------------------------------------------------------------


def test_mtpk_armed_stream_never_shows_trimmed_tokens_strict(monkeypatch):
    _set_repetition_env(monkeypatch)
    monkeypatch.setenv("MTPLX_REPETITION_STREAM_HOLDBACK", "strict")
    calls, callback = _collecting_callback()
    out = generate_mtpk(
        _runtime(_ScriptedMTPModel(_next_token)),
        [0],
        max_tokens=200,
        sampler=GREEDY,
        speculative_depth=2,
        seed=7,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
        token_callback=callback,
        repetition_stop=True,
    )
    assert any("repetition_stop" in event for event in out.stats.events)
    wire = [token for call in calls for token in call]
    assert wire == list(out.tokens)
    # The trimmer retracted the repeated suffix before it hit the wire.
    assert len(out.tokens) < 200


def test_ar_armed_nonlooping_stream_is_live_per_token(monkeypatch):
    """THE 2.8.x product regression test.

    An armed uncapped stream with healthy (non-looping) output must be
    byte-identical to the disarmed wire: one singleton callback per token,
    no holdback, no freeze window, no end-of-response flush burst. The
    2.8.0-2.8.2 unconditional holdback fails exactly this — it silenced
    every desktop chat for a detector-window of tokens (8-11 s at chat
    rates) and dumped the tail as one final burst.
    """
    _set_repetition_env(monkeypatch)
    calls, callback = _collecting_callback()
    out = generate_ar(
        _runtime(_ScriptedModel(_next_token_fresh)),
        [0],
        max_tokens=120,  # far past min_tokens=48: detector armed and active
        sampler=GREEDY,
        seed=7,
        stop_token_ids=set(),
        token_callback=callback,
        repetition_stop=True,
    )
    assert len(out.tokens) == 120
    # Live wire: the exact historical per-token singleton pattern.
    assert calls == [[token] for token in out.tokens]


def test_ar_armed_looping_stream_divergence_is_bounded(monkeypatch):
    """Candidate mode on a real loop: the wire may briefly stream a prefix
    of the suffix the trimmer later retracts — bounded by the candidate
    engagement span — and never anything else."""
    _set_repetition_env(monkeypatch)
    # Fire threshold 4 copies/32 tokens; candidate engages at 2 copies/16.
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MIN_REPEATED_TOKENS", "32")
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MIN_REPEATS", "4")
    calls, callback = _collecting_callback()
    out = generate_ar(
        _runtime(_ScriptedModel(_next_token)),
        [0],
        max_tokens=200,
        sampler=GREEDY,
        seed=7,
        stop_token_ids=set(),
        token_callback=callback,
        repetition_stop=True,
    )
    assert any("repetition_stop" in event for event in out.stats.events)
    final = list(out.tokens)
    wire = [token for call in calls for token in call]
    # The final answer streamed in order and in full.
    assert wire[: len(final)] == final
    extra = wire[len(final) :]
    # Anything beyond it is a prefix of the loop the trimmer retracted...
    trimmed_span = 200  # loop values cycle LOOP_START..LOOP_START+PERIOD-1
    for token in extra:
        assert LOOP_START <= token < LOOP_START + PERIOD
    # ...bounded by the engagement span (min_repeated//2) plus one commit.
    assert len(extra) <= 16 + 2


def test_mtpk_armed_nonlooping_stream_is_live(monkeypatch):
    """Speculative lane: armed healthy streams deliver committed tokens as
    they commit (multiple wire batches), byte-identical to disarmed."""
    _set_repetition_env(monkeypatch)
    calls, callback = _collecting_callback()
    out = generate_mtpk(
        _runtime(_ScriptedMTPModel(_next_token_fresh)),
        [0],
        max_tokens=120,
        sampler=GREEDY,
        speculative_depth=2,
        seed=7,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
        token_callback=callback,
        repetition_stop=True,
    )
    wire = [token for call in calls for token in call]
    assert wire == list(out.tokens)
    assert len(out.tokens) == 120
    # Liveness: the response arrived across many wire batches, not one
    # end-of-response flush.
    assert len(calls) >= 5


def test_mtpk_disarmed_stream_matches_committed_tokens(monkeypatch):
    _set_repetition_env(monkeypatch)
    calls, callback = _collecting_callback()
    out = generate_mtpk(
        _runtime(_ScriptedMTPModel(_next_token)),
        [0],
        max_tokens=60,
        sampler=GREEDY,
        speculative_depth=2,
        seed=7,
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
        token_callback=callback,
        repetition_stop=False,
    )
    wire = [token for call in calls for token in call]
    assert len(out.tokens) == 60  # no trim when disarmed
    assert wire == list(out.tokens)
