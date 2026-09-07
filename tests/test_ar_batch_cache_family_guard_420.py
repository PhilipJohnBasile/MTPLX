"""#420: the batched AR lane refuses itself for cache families it cannot batch.

mlx-lm's BatchGenerator merges every batched prompt's caches and needs
merge() on each entry. Flash-Next's QSACache has none, so two concurrent
requests under --scheduler-mode ar_batch raised an unhandled ValueError
(HTTP 500) while sequential requests were fine. The lane now probes the
runtime's cache family at startup and, when it cannot batch, every request
rides the solo lane one at a time.
"""

from __future__ import annotations

from types import SimpleNamespace

import mtplx.server.openai as srv
from mtplx.batching.state import SchedulerMode


class _Mergeable:
    def merge(self, others):
        return self


class _QSALike:
    pass


def _runtime(*entries):
    # mlx-lm's make_prompt_cache asks the model for make_cache() first; the
    # probe must see the same entries the batch generator would.
    return SimpleNamespace(model=SimpleNamespace(make_cache=lambda: list(entries)))


def test_mergeable_family_is_batchable():
    assert srv._ar_batch_unavailable_reason(_runtime(_Mergeable(), _Mergeable())) is None


def test_no_cache_is_batchable():
    runtime = SimpleNamespace(model=SimpleNamespace(make_cache=lambda: None))
    assert srv._ar_batch_unavailable_reason(runtime) is None


def test_probe_uses_the_batch_generator_factory_not_the_runtime_factory():
    """The 27B's runtime.make_cache() returns paged classes without merge()
    while its outer model has no make_cache, so mlx-lm builds stock KVCache
    entries and batches it fine. The probe must follow mlx-lm."""

    class _Layer:
        pass

    class _Attn:
        pass

    outer = SimpleNamespace(layers=[SimpleNamespace(self_attn=_Attn())])
    runtime = SimpleNamespace(model=outer, make_cache=lambda: [_QSALike()])
    assert srv._ar_batch_unavailable_reason(runtime) is None


def test_unmergeable_entry_names_the_class():
    reason = srv._ar_batch_unavailable_reason(_runtime(_Mergeable(), _QSALike()))
    assert reason is not None
    assert "_QSALike" in reason
    assert "merge()" in reason


def test_probe_failure_is_a_reason_not_a_crash():
    def boom():
        raise RuntimeError("no model")

    runtime = SimpleNamespace(model=SimpleNamespace(make_cache=boom))
    reason = srv._ar_batch_unavailable_reason(runtime)
    assert reason is not None and "probe failed" in reason


def test_live_ar_batch_yields_to_the_solo_lane_when_unavailable(monkeypatch):
    config = SimpleNamespace(mode=SchedulerMode.AR_BATCH)
    monkeypatch.setattr(srv, "_scheduler_config_from_args", lambda args: config)
    monkeypatch.setattr(srv, "_ar_batch_mtp_fallback_reason", lambda state: "concurrent")
    available = SimpleNamespace(args=None, ar_batch_unavailable_reason=None)
    refused = SimpleNamespace(args=None, ar_batch_unavailable_reason="QSACache has no merge()")

    assert srv._use_live_ar_batch(available, effective_mode="mtp") == (True, "concurrent")
    assert srv._use_live_ar_batch(available, effective_mode="ar") == (True, "generation_mode_ar")
    assert srv._use_live_ar_batch(refused, effective_mode="mtp") == (False, None)
    assert srv._use_live_ar_batch(refused, effective_mode="ar") == (False, None)
