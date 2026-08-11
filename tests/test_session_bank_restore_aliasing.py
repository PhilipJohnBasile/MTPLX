"""Issue #247: restores must never install the bank entry's own state objects.

druide67's report: a byte-identical temperature-0 call was answered according
to a *different* request's system prompt after an interleaved near-prefix
request. Root cause: lazy (kvcache-v2) snapshots restored with
``clone_states=False`` installed the entry's stored array objects directly
into the borrower's cache. The borrower's suffix prefill then wrote into those
very objects (setitem mutates the installed object; a rebind write may donate
its buffer because the lone shared object counts as uniquely referenced), so
the lender's banked span was silently rewritten and every later match served
the poisoned pages.

These tests are the regression pin built from that reproducer's geometry:
commit A -> near-prefix borrow for B -> B's suffix write -> A's entry (and
every later restore from it) must be byte-identical to a cache that was never
borrowed from. They run on real ``mx.array`` state through the mlx-lm-style
setitem write path, the exact container class the report reproduced on.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mtplx.cache_state import restore_cache, snapshot_cache_lazy_hybrid
from mtplx.session_bank import SessionBank

HEADS = 2
DIM = 4
CAPACITY = 32
PREFIX = 10
MATCH = 8  # near-prefix borrow point; gap 2 stays under the tiny-gap limit


class MxKVCache:
    """mlx-lm-style KV container: preallocated buffers, in-place setitem
    appends, trimmed-slice ``state``. The write pattern #247 hinges on."""

    def __init__(self, capacity: int = CAPACITY):
        self.keys = mx.zeros((1, HEADS, capacity, DIM), dtype=mx.float16)
        self.values = mx.zeros((1, HEADS, capacity, DIM), dtype=mx.float16)
        self.offset = 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(int(n), self.offset)
        self.offset -= n
        return n

    def append(self, k: mx.array, v: mx.array) -> None:
        steps = int(k.shape[2])
        if self.offset + steps > int(self.keys.shape[2]):
            raise RuntimeError("test cache capacity exceeded")
        self.keys[..., self.offset : self.offset + steps, :] = k
        self.values[..., self.offset : self.offset + steps, :] = v
        self.offset += steps

    @property
    def state(self):
        if self.offset == int(self.keys.shape[2]):
            return self.keys, self.values
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = int(self.keys.shape[2])

    @property
    def meta_state(self):
        return (str(self.offset),)

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])


class RuntimeWithMxCaches:
    model_path = Path("models/example")
    mtp_enabled = True

    def make_cache(self):
        return [MxKVCache()]

    def make_mtp_cache(self):
        return [MxKVCache()]


def _step(seed: int, steps: int = 1) -> tuple[mx.array, mx.array]:
    k = mx.random.normal((1, HEADS, steps, DIM), key=mx.random.key(seed)).astype(
        mx.float16
    )
    v = mx.random.normal(
        (1, HEADS, steps, DIM), key=mx.random.key(seed + 10_000)
    ).astype(mx.float16)
    return k, v


def _fill(cache: MxKVCache, tokens: int, *, base_seed: int = 0) -> None:
    for i in range(tokens):
        cache.append(*_step(base_seed + i))


def _keys_prefix(cache: MxKVCache, tokens: int) -> mx.array:
    return cache.keys[..., :tokens, :]


def test_restore_cache_lazy_install_is_immune_to_borrower_writes() -> None:
    """Primitive pin: clone_states=False must not hand out the stored objects."""
    src = MxKVCache()
    control = MxKVCache()
    _fill(src, PREFIX)
    _fill(control, PREFIX)

    snapshot = snapshot_cache_lazy_hybrid([src])

    borrower = MxKVCache()
    restore_cache([borrower], snapshot, clone_states=False)
    assert borrower.offset == PREFIX

    # Near-prefix borrow: trim back, then setitem-write a divergent suffix
    # inside the span the snapshot still covers.
    borrower.trim(PREFIX - MATCH)
    borrower.append(*_step(500, steps=2))
    mx.eval(borrower.keys, borrower.values)

    snap_keys, snap_values = snapshot.states[0]
    assert mx.array_equal(
        snap_keys, _keys_prefix(control, PREFIX)
    ).item(), "borrower setitem write reached the snapshot's stored keys (#247)"
    assert mx.array_equal(
        snap_values, control.values[..., :PREFIX, :]
    ).item(), "borrower setitem write reached the snapshot's stored values (#247)"

    # A second restore from the same snapshot must serve pristine pages.
    second = MxKVCache()
    restore_cache([second], snapshot, clone_states=False)
    assert mx.array_equal(
        _keys_prefix(second, PREFIX), _keys_prefix(control, PREFIX)
    ).item(), "restore after a borrower write served poisoned pages (#247)"


def test_bank_near_prefix_borrow_does_not_poison_the_lender_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full bank flow from the reproducer: put A, near-prefix borrow, write,
    then exact restore of A — byte-identical to a never-borrowed cache."""
    monkeypatch.setenv("MTPLX_SESSION_LAZY_SNAPSHOT", "1")
    runtime = RuntimeWithMxCaches()
    bank = SessionBank()
    tokens = list(range(PREFIX))

    src = MxKVCache()
    control = MxKVCache()
    _fill(src, PREFIX)
    _fill(control, PREFIX)

    entry = bank.put(
        runtime=runtime,
        token_ids=tokens,
        cache=[src],
        logits=None,
        hidden=None,
        session_id="lender",
    )
    assert entry is not None
    assert entry.lazy_kv is True, "test requires the lazy-snapshot geometry"

    # Interleaved request B: borrows A's entry at a near-prefix boundary and
    # prefills its own divergent suffix. B stores nothing — in the report B's
    # uncovered suffix was under the store minimum, and A was poisoned anyway.
    borrowed = bank.restore_entry_prefix_cache(runtime, entry, MATCH)
    assert borrowed is not None
    borrower_cache = borrowed[0]
    borrower_cache[0].append(*_step(700, steps=2))
    mx.eval(borrower_cache[0].keys, borrower_cache[0].values)

    # The entry's stored span must be untouched by B's writes...
    snap_keys, _ = entry.cache_snapshot.states[0]
    assert mx.array_equal(
        snap_keys, _keys_prefix(control, PREFIX)
    ).item(), "near-prefix borrower poisoned the lender's banked span (#247)"

    # ...and A's own exact restore must serve the original bytes.
    restored = bank.restore(runtime, tokens, session_id="lender")
    assert restored is not None
    assert mx.array_equal(
        _keys_prefix(restored.cache[0], PREFIX), _keys_prefix(control, PREFIX)
    ).item(), "exact restore after an interleaved borrow served poisoned pages (#247)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
