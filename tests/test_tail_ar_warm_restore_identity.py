"""AR warm-restore token identity (#246 follow-up, tail sweep 2026-08-16).

The AR lane routes warm turns through ``restore_or_prefill_prompt_state``
(generation.generate_ar). The existing #246 tests pin the TELEMETRY of
that path (cached_tokens/cache_hit) with a fake bank and a cache-blind
model; nothing pinned the actual decode: a warm-restored AR session must
produce byte-identical tokens to a cold run of the same request.

Harness: a deterministic tiny model whose logits at every step depend on
the FULL cached history (a real ``mlx_lm`` KVCache holding one-hot key
history; logits are integer-valued f32 sums, so equality is exact, not
approximate) plus a real ``SessionBank`` round-trip
(``snapshot_cache`` -> ``put_snapshot`` -> in-loop restore). If the
restore drops, duplicates, or corrupts any prefix state, every following
argmax moves — the corruption control below proves that sensitivity, so
the identity assertion cannot pass vacuously.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from mtplx.cache_state import snapshot_cache
from mtplx.generation import (
    _resolve_runtime_base_hidden_variant,
    generate_ar,
    restore_or_prefill_prompt_state,
)
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig
from mtplx.session_bank import SessionBank

VOCAB = 32
PROMPT = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]
PREFIX_LEN = 8
MAX_TOKENS = 8
GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)

# Fixed integer mixing matrix: history counts @ MIX gives integer-valued
# f32 logits, so float accumulation is exact and argmax is deterministic.
_MIX = mx.array(
    [[((i * 7 + j * 13) % 31) - 15 for j in range(VOCAB)] for i in range(VOCAB)],
    dtype=mx.float32,
)


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(f"<{int(token)}>" for token in tokens)


class HistoryCountModel:
    """Causal toy model: position t's logits depend on tokens[0..t].

    Keys stored in the KVCache are one-hot token embeddings; the logits
    for each new position are the cumulative one-hot counts of the whole
    history up to that position, mixed through a fixed integer matrix.
    Any divergence in restored prefix state changes the counts and
    therefore the argmax of every subsequent step.
    """

    def __init__(self):
        self.calls: list[int] = []

    def make_cache(self):
        return [KVCache()]

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        return hidden_states

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
        del hidden_variant
        batch, length = int(input_ids.shape[0]), int(input_ids.shape[1])
        self.calls.append(length)
        onehot = mx.eye(VOCAB, dtype=mx.float32)[input_ids]  # (B, S, V)
        entry = cache[0]
        keys, _values = entry.update_and_fetch(
            onehot[:, None, :, :], onehot[:, None, :, :]
        )  # (B, 1, T, V) full trimmed history
        hidden = mx.zeros((batch, length, 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        counts = mx.cumsum(keys[:, 0, :, :], axis=1)  # (B, T, V)
        counts = counts[:, -length:, :]  # causal rows for the new positions
        logits = counts @ _MIX  # integer-valued f32
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = logits[:, -keep:, :]
        if return_hidden:
            return logits, hidden[:, -keep:, :]
        return logits


def _runtime() -> MTPLXRuntime:
    return MTPLXRuntime(
        model=HistoryCountModel(),
        tokenizer=_Tokenizer(),
        model_path=Path("models/tail-warm-identity"),
        mtp_enabled=False,
        contract=MTPContract(),
    )


def _bank() -> SessionBank:
    return SessionBank(
        max_entries=8, max_bytes=1 << 24, per_session_max_bytes=1 << 24
    )


def _bank_with_prefix(prefix_ids: list[int], *, corrupt: bool = False) -> SessionBank:
    """Prefill ``prefix_ids`` on a fresh runtime and bank the real snapshot.

    ``corrupt=True`` prefills a DIFFERENT token sequence but banks it under
    the true prefix ids — the shape every restore-fidelity bug takes.
    """

    producer = _runtime()
    filled = list(prefix_ids)
    if corrupt:
        filled[1] = (filled[1] + 1) % VOCAB
    state = restore_or_prefill_prompt_state(
        producer,
        filled,
        base_hidden_variant=None,
        mtp_history_policy="cycle",
    )
    bank = _bank()
    entry = bank.put_snapshot(
        runtime=producer,
        token_ids=tuple(prefix_ids),
        cache_snapshot=snapshot_cache(state.trunk_cache),
        logits=state.logits,
        # Identity fields exactly as the production store path stamps them
        # (generation._maybe_store_prefix_snapshot): the restore gate
        # rejects on hidden_variant mismatch (policy_mismatch) otherwise.
        hidden_variant=_resolve_runtime_base_hidden_variant(producer, None),
        session_id="warm-identity",
        mtp_history_policy="cycle",
        snapshot_epoch=len(prefix_ids),
    )
    assert entry is not None, "prefix snapshot must be admitted to the bank"
    return bank


def _generate(runtime: MTPLXRuntime, *, session_bank=None) -> object:
    return generate_ar(
        runtime,
        list(PROMPT),
        max_tokens=MAX_TOKENS,
        sampler=GREEDY,
        seed=0,
        stop_token_ids=set(),
        session_bank=session_bank,
        session_id="warm-identity" if session_bank is not None else None,
    )


def test_warm_restored_ar_tokens_are_byte_identical_to_cold():
    cold = _generate(_runtime())
    assert len(cold.tokens) == MAX_TOKENS

    warm_runtime = _runtime()
    warm = _generate(
        warm_runtime, session_bank=_bank_with_prefix(PROMPT[:PREFIX_LEN])
    )

    # The warm lane really restored: telemetry says so, and the model never
    # saw a full-prompt prefill call.
    assert warm.stats.session_cache_hit is True
    assert warm.stats.cached_tokens > 0
    assert warm.stats.new_prefill_tokens < len(PROMPT)
    assert len(PROMPT) not in warm_runtime.model.calls

    # THE invariant: byte-identical tokens, cold vs warm-restored.
    assert list(warm.tokens) == list(cold.tokens)


def test_corrupted_prefix_state_changes_tokens_proving_sensitivity():
    """Teeth check: the harness must be able to FAIL. A banked snapshot
    whose state came from a different prefix must change the decode —
    otherwise the identity assertion above proves nothing."""

    cold = _generate(_runtime())
    corrupted = _generate(
        _runtime(),
        session_bank=_bank_with_prefix(PROMPT[:PREFIX_LEN], corrupt=True),
    )
    assert corrupted.stats.session_cache_hit is True
    assert list(corrupted.tokens) != list(cold.tokens)


def test_cold_run_is_deterministic_baseline():
    first = _generate(_runtime())
    second = _generate(_runtime())
    assert list(first.tokens) == list(second.tokens)
