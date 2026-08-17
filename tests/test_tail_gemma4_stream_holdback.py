"""Gemma4 F35: the armed repetition stop must trim BEFORE the wire.

Both gemma4 loops (target-only AR and the exact-speculative assistant
loop) had the same emit-then-trim divergence wave 3 fixed in
mtplx.generation's serial loops: every token hit the stream callback
before ``_trim_repeated_suffix`` ran, so a fired trim left already
streamed garbage on the wire. The fix wires the SAME holdback helpers
(``_repetition_stream_holdback_tokens`` / ``_repetition_stream_emit_limit``)
through ``_Gemma4RepetitionAwareWire``.

Deterministic no-model harness (pattern from
test_runtime_obs_stream_holdback): a scripted target walks distinct
tokens then enters a fixed cycle, so the uncapped repetition stop fires
at a known step. Invariant when armed: wire == final tokens, byte for
byte. Disarmed requests keep the historical per-token emit pattern.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

import mtplx.backends.gemma4_assistant as gemma4
from mtplx.sampling import SamplerConfig

VOCAB = 64
LOOP_START = 40
PERIOD = 8
MARGIN = 10.0

GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)


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


def _logits_for(token: int) -> mx.array:
    row = [0.0] * VOCAB
    row[int(token)] = MARGIN
    return mx.array([[row]], dtype=mx.float32)


class _ScriptedGemmaRuntime:
    """Target adapter double: after token t the target wants script(t)."""

    def __init__(self, script) -> None:
        self._script = script
        self.tokenizer = _Tokenizer()
        self.telemetry = SimpleNamespace(to_dict=lambda: {})
        self.config = SimpleNamespace(
            draft_block_size=2,
            assistant_model_path="scripted-assistant",
            target_distribution_mode="exact",
        )
        self.distribution_compile_stats = {}

    def forward_target(self, input_ids, *, cache=None, phase=None):
        del cache, phase
        token = int(np.asarray(input_ids).reshape(-1)[-1])
        return SimpleNamespace(
            logits=_logits_for(self._script(token)),
            hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
            shared_kv_states={},
            cache_offset=0,
        )


def _prompt_state(script, prompt_ids):
    last = int(prompt_ids[-1])
    return SimpleNamespace(
        cache=[],
        logits=_logits_for(script(last))[:, -1, :],
        hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
        shared_kv_states={},
        kv_offset=0,
        prompt_eval_time_s=0.0,
        cached_tokens=0,
        suffix_tokens=len(prompt_ids),
        cache_hit=False,
        cache_source="none",
        cache_miss_reason=None,
        restore_mode="cold",
    )


def _patch_prefill(monkeypatch, script) -> None:
    monkeypatch.setattr(
        gemma4,
        "_restore_or_prefill_gemma4_prompt",
        lambda runtime, prompt_ids, **_kwargs: _prompt_state(script, prompt_ids),
    )


def _set_repetition_env(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MIN_TOKENS", "48")
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MIN_REPEATED_TOKENS", "16")
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MIN_REPEATS", "2")
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MIN_BLOCK_TOKENS", "1")
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MAX_BLOCK_TOKENS", "8")


def _collecting_callback():
    calls: list[list[int]] = []

    def callback(tokens: list[int]) -> None:
        calls.append([int(token) for token in tokens])

    return calls, callback


# ---------------------------------------------------------------------------
# generate_gemma4_ar: target-only loop wire semantics.
# ---------------------------------------------------------------------------


def test_gemma4_ar_armed_stream_never_shows_trimmed_tokens(monkeypatch):
    _set_repetition_env(monkeypatch)
    _patch_prefill(monkeypatch, _next_token)
    calls, callback = _collecting_callback()
    out = gemma4.generate_gemma4_ar(
        _ScriptedGemmaRuntime(_next_token),
        [0],
        max_tokens=200,
        sampler=GREEDY,
        seed=7,
        stop_token_ids=set(),
        token_callback=callback,
        repetition_stop=True,
    )
    # The cycle fires the trimmer; the repeated suffix is retracted from
    # the final tokens, keeping only the distinct walk 1..39.
    assert list(out.tokens) == list(range(1, LOOP_START))
    assert out.stats.repetition_stop_triggered
    assert any("repetition_stop" in event for event in out.stats.events)
    wire = [token for call in calls for token in call]
    # THE invariant: the wire shows exactly the post-trim tokens.
    assert wire == list(out.tokens)


def test_gemma4_ar_armed_stream_flushes_full_tail_when_no_trim(monkeypatch):
    _set_repetition_env(monkeypatch)
    _patch_prefill(monkeypatch, _next_token_fresh)
    calls, callback = _collecting_callback()
    out = gemma4.generate_gemma4_ar(
        _ScriptedGemmaRuntime(_next_token_fresh),
        [0],
        max_tokens=30,  # below min_tokens: the detector never fires
        sampler=GREEDY,
        seed=7,
        stop_token_ids=set(),
        token_callback=callback,
        repetition_stop=True,
    )
    wire = [token for call in calls for token in call]
    assert len(out.tokens) == 30
    assert wire == list(out.tokens)
    assert calls, "armed stream must still deliver the response"


def test_gemma4_ar_disarmed_stream_is_byte_identical_per_token(monkeypatch):
    _set_repetition_env(monkeypatch)
    _patch_prefill(monkeypatch, _next_token)
    calls, callback = _collecting_callback()
    out = gemma4.generate_gemma4_ar(
        _ScriptedGemmaRuntime(_next_token),
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
# generate_gemma4_assistant: exact-speculative loop wire semantics.
# ---------------------------------------------------------------------------


def _patch_speculative_round(monkeypatch, script) -> None:
    def fake_round(
        runtime,
        *,
        primary_token_id,
        hidden,
        shared_kv_states,
        kv_offset,
        cache,
        sampler,
        draft_sampler,
        rng,
        draft_block_size,
    ):
        del runtime, hidden, shared_kv_states, kv_offset, cache
        del sampler, draft_sampler, rng
        accepted = []
        token = int(primary_token_id)
        for _ in range(max(1, int(draft_block_size) - 1)):
            token = script(token)
            accepted.append(int(token))
        return SimpleNamespace(
            accepted_token_ids=accepted,
            accepted_count=len(accepted),
            corrected_token_id=None,
            bonus_token_id=None,
            next_primary_token_id=script(token),
            next_hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
            next_shared_kv_states={},
            next_kv_offset=0,
            metadata={},
        )

    monkeypatch.setattr(gemma4, "gemma4_exact_speculative_round", fake_round)


def test_gemma4_assistant_armed_stream_never_shows_trimmed_tokens(monkeypatch):
    _set_repetition_env(monkeypatch)
    _patch_prefill(monkeypatch, _next_token)
    _patch_speculative_round(monkeypatch, _next_token)
    calls, callback = _collecting_callback()
    out = gemma4.generate_gemma4_assistant(
        _ScriptedGemmaRuntime(_next_token),
        [0],
        max_tokens=200,
        sampler=GREEDY,
        speculative_depth=2,
        seed=7,
        stop_token_ids=set(),
        token_callback=callback,
        repetition_stop=True,
    )
    assert out.stats.repetition_stop_triggered
    assert any("repetition_stop" in event for event in out.stats.events)
    assert len(out.tokens) < 200
    wire = [token for call in calls for token in call]
    assert wire == list(out.tokens)


def test_gemma4_assistant_disarmed_stream_matches_committed_tokens(monkeypatch):
    _set_repetition_env(monkeypatch)
    _patch_prefill(monkeypatch, _next_token)
    _patch_speculative_round(monkeypatch, _next_token)
    calls, callback = _collecting_callback()
    out = gemma4.generate_gemma4_assistant(
        _ScriptedGemmaRuntime(_next_token),
        [0],
        max_tokens=60,
        sampler=GREEDY,
        speculative_depth=2,
        seed=7,
        stop_token_ids=set(),
        token_callback=callback,
        repetition_stop=False,
    )
    wire = [token for call in calls for token in call]
    assert len(out.tokens) == 60  # no trim when disarmed
    assert wire == list(out.tokens)
