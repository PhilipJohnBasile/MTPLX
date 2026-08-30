"""Pure host-side gates for QSA MTP replay metadata.

These tests intentionally import neither MLX nor the qwen4_exp model.  They
pin the geometry and speculative-cache rules that are staged outside a future
compiled indexer graph.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from mtplx.qsa_mtp_precompute import (
    precompute_and_stage_qsa_replay,
    precompute_and_stage_qsa_replay_caches,
    precompute_mtp_indexer_replay,
    precompute_qsa_replay_capacity,
    qsa_indexer_capacity_bucket,
    qsa_indexer_is_bucket_capacity,
    qsa_mtp_outer_device_core_supported,
    qsa_mtp_precompute_enabled,
    stage_qsa_replay_capacity,
)


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "typo"])
def test_phase3_gate_is_default_off_and_fail_closed(value):
    environ = {} if value is None else {"MTPLX_QSA_MTP_PRECOMPUTE": value}

    assert qsa_mtp_precompute_enabled(environ) is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_phase3_gate_requires_explicit_truthy_value(value):
    assert qsa_mtp_precompute_enabled({"MTPLX_QSA_MTP_PRECOMPUTE": value}) is True


@pytest.mark.parametrize(
    ("required", "expected"),
    [(0, 0), (1, 256), (255, 256), (256, 256), (257, 512), (768, 1024)],
)
def test_shared_compiled_capacity_bucket(required, expected):
    capacity = qsa_indexer_capacity_bucket(required)

    assert capacity == expected
    assert qsa_indexer_is_bucket_capacity(capacity)


@pytest.mark.parametrize("capacity", [1, 255, 257, 768, 1536])
def test_non_bucket_capacities_are_rejected_by_signature_guard(capacity):
    assert not qsa_indexer_is_bucket_capacity(capacity)


def test_shared_bucket_validates_inputs():
    with pytest.raises(ValueError, match="required"):
        qsa_indexer_capacity_bucket(-1)
    with pytest.raises(ValueError, match="power of two"):
        qsa_indexer_capacity_bucket(1, minimum=192)
    with pytest.raises(ValueError, match="capacity"):
        qsa_indexer_is_bucket_capacity(-1)


@pytest.mark.parametrize(
    ("start", "width", "ratio", "end", "blocks"),
    [
        (0, 0, 4, 0, 0),
        (0, 1, 4, 1, 0),
        (0, 3, 4, 3, 0),
        (0, 4, 4, 4, 1),
        (3, 1, 4, 4, 1),
        (4, 1, 4, 5, 1),
        (255, 4, 4, 259, 64),
    ],
)
def test_capacity_uses_only_complete_blocks(start, width, ratio, end, blocks):
    plan = precompute_qsa_replay_capacity(
        start_offset=start,
        window_tokens=width,
        compress_ratio=ratio,
    )

    assert plan.end_offset == end
    assert plan.complete_blocks == blocks


def test_raw_and_pooled_capacities_use_independent_power_of_two_buckets():
    plan = precompute_qsa_replay_capacity(
        start_offset=1023,
        window_tokens=2,
        compress_ratio=4,
        allocation_step=256,
    )

    assert plan.raw_capacity == 2048
    assert plan.pooled_capacity == 256
    assert plan.graph_key == (2048, 256)


def test_unaligned_prefill_buckets_the_fixed_pool_staging_window():
    plan = precompute_qsa_replay_capacity(
        start_offset=0,
        window_tokens=1025,
        compress_ratio=4,
        allocation_step=256,
    )

    assert plan.complete_blocks == 256
    staging_blocks = (
        plan.window_tokens + plan.compress_ratio - 1
    ) // plan.compress_ratio
    assert staging_blocks == 257
    assert plan.raw_capacity == 2048
    assert plan.pooled_capacity == 512


def test_non_power_of_two_ratio_buckets_the_full_raw_staging_window():
    plan = precompute_qsa_replay_capacity(
        start_offset=0,
        window_tokens=1024,
        compress_ratio=3,
        allocation_step=256,
    )

    assert plan.complete_blocks == 341
    assert plan.raw_capacity == 2048
    assert plan.pooled_capacity == 512


def test_existing_backing_never_shrinks():
    plan = precompute_qsa_replay_capacity(
        start_offset=17,
        window_tokens=5,
        compress_ratio=4,
        current_raw_capacity=4096,
        current_pooled_capacity=2048,
    )

    assert plan.raw_capacity == 4096
    assert plan.pooled_capacity == 2048


def test_additive_v210_backing_is_promoted_to_bounded_graph_bucket():
    plan = precompute_qsa_replay_capacity(
        start_offset=600,
        window_tokens=4,
        compress_ratio=4,
        current_raw_capacity=768,
        current_pooled_capacity=256,
    )

    assert plan.raw_capacity == 1024
    assert plan.pooled_capacity == 256


def test_graph_key_stays_stable_inside_capacity_bucket():
    keys = {
        precompute_qsa_replay_capacity(
            start_offset=offset,
            window_tokens=4,
            compress_ratio=4,
            current_raw_capacity=512,
            current_pooled_capacity=256,
        ).graph_key
        for offset in range(257, 509)
    }

    assert keys == {(512, 256)}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start_offset": -1, "window_tokens": 1, "compress_ratio": 4}, "start_offset"),
        (
            {"start_offset": 0, "window_tokens": -1, "compress_ratio": 4},
            "window_tokens",
        ),
        (
            {"start_offset": 0, "window_tokens": 1, "compress_ratio": 0},
            "compress_ratio",
        ),
        (
            {
                "start_offset": 0,
                "window_tokens": 1,
                "compress_ratio": 4,
                "allocation_step": 0,
            },
            "allocation_step",
        ),
        (
            {
                "start_offset": 0,
                "window_tokens": 1,
                "compress_ratio": 4,
                "allocation_step": 192,
            },
            "power of two",
        ),
        (
            {
                "start_offset": 0,
                "window_tokens": 1,
                "compress_ratio": 4,
                "current_raw_capacity": -1,
            },
            "current_raw_capacity",
        ),
    ],
)
def test_invalid_capacity_geometry_fails_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        precompute_qsa_replay_capacity(**kwargs)


def test_capacity_plan_is_immutable():
    plan = precompute_qsa_replay_capacity(
        start_offset=0,
        window_tokens=4,
        compress_ratio=4,
    )

    with pytest.raises(FrozenInstanceError):
        plan.raw_capacity = 0


class _Reservable:
    def __init__(
        self,
        *,
        offset=0,
        ratio=4,
        step=256,
        raw_capacity=0,
        pooled_capacity=0,
    ):
        self.calls = []
        self.offset = offset
        self.ratio = ratio
        self.step = step
        self.raw_keys = None if raw_capacity == 0 else _ShapeOnly((1, raw_capacity, 16))
        self.pooled = (
            None if pooled_capacity == 0 else _ShapeOnly((1, pooled_capacity, 16))
        )

    def reserve_indexer_capacity(self, *, raw_capacity, pooled_capacity):
        self.calls.append((raw_capacity, pooled_capacity))


class _ShapeOnly:
    def __init__(self, shape):
        self.shape = shape


def test_outer_device_core_accepts_only_cache_stacks_without_qsa_state():
    assert qsa_mtp_outer_device_core_supported(None) is True
    assert qsa_mtp_outer_device_core_supported([]) is True
    assert qsa_mtp_outer_device_core_supported([object()]) is True


def test_outer_device_core_fails_closed_for_any_qsa_cache_entry():
    assert qsa_mtp_outer_device_core_supported([_Reservable()]) is False
    assert qsa_mtp_outer_device_core_supported([object(), _Reservable()]) is False


def test_stage_applies_precomputed_buckets_once():
    cache = _Reservable()
    plan = precompute_qsa_replay_capacity(
        start_offset=255,
        window_tokens=4,
        compress_ratio=4,
    )

    assert stage_qsa_replay_capacity(cache, plan) is True
    assert cache.calls == [(512, 256)]


def test_stage_is_a_noop_for_non_qsa_cache():
    plan = precompute_qsa_replay_capacity(
        start_offset=255,
        window_tokens=4,
        compress_ratio=4,
    )

    assert stage_qsa_replay_capacity(object(), plan) is False


def test_stage_does_not_hide_reservation_failure():
    class Broken:
        def reserve_indexer_capacity(self, **_kwargs):
            raise RuntimeError("allocation failed")

    plan = precompute_qsa_replay_capacity(
        start_offset=0,
        window_tokens=4,
        compress_ratio=4,
    )

    with pytest.raises(RuntimeError, match="allocation failed"):
        stage_qsa_replay_capacity(Broken(), plan)


def test_plan_and_stage_reads_only_cache_metadata():
    cache = _Reservable(
        offset=510,
        ratio=4,
        raw_capacity=512,
        pooled_capacity=256,
    )

    plan = precompute_and_stage_qsa_replay(cache, window_tokens=4)

    assert plan is not None
    assert plan.end_offset == 514
    assert plan.graph_key == (1024, 256)
    assert cache.calls == [(1024, 256)]


def test_plan_and_stage_skips_heterogeneous_non_qsa_entries():
    first = _Reservable(offset=255, ratio=4)
    second = object()
    third = _Reservable(offset=1023, ratio=8, raw_capacity=1280)

    plans = precompute_and_stage_qsa_replay_caches(
        [first, second, third],
        window_tokens=4,
    )

    assert [plan.graph_key for plan in plans] == [(512, 256), (2048, 256)]
    assert first.calls == [(512, 256)]
    assert third.calls == [(2048, 256)]


def test_normal_mtp_draft_retains_only_authoritative_primary_row():
    plan = precompute_mtp_indexer_replay(cycle_offset=100, observed_offset=104)

    assert plan.speculative_rows == 4
    assert plan.primary_staged is True
    assert plan.reusable_rows == 1
    assert plan.rollback_offset == 101
    assert plan.reappend_start == 1
    assert plan.reappend_tokens([11, 12, 13, 14]) == (12, 13, 14)
    assert plan.authoritative_hidden_rows(4) == 3


def test_non_mtp_draft_source_reappends_primary_too():
    plan = precompute_mtp_indexer_replay(cycle_offset=100, observed_offset=100)

    assert plan.speculative_rows == 0
    assert plan.primary_staged is False
    assert plan.reusable_rows == 0
    assert plan.rollback_offset == 100
    assert plan.reappend_start == 0
    assert plan.reappend_tokens([11, 12]) == (11, 12)
    assert plan.authoritative_hidden_rows(2) == 2


def test_one_staged_row_needs_no_reappend_for_primary_only_commit():
    plan = precompute_mtp_indexer_replay(cycle_offset=7, observed_offset=8)

    assert plan.reappend_tokens([91]) == ()
    assert plan.authoritative_hidden_rows(1) == 0


def test_replay_plan_rejects_backwards_or_short_geometry():
    with pytest.raises(ValueError, match="observed_offset"):
        precompute_mtp_indexer_replay(cycle_offset=8, observed_offset=7)

    plan = precompute_mtp_indexer_replay(cycle_offset=8, observed_offset=9)
    with pytest.raises(ValueError, match="retained prefix"):
        plan.authoritative_hidden_rows(0)
