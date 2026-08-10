from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

import mtplx.batched_decode as bd
import mtplx.fast_sampling as fs
from mtplx.a3b_mtp_batch import (
    A3BMTPBatchRequest,
    _BatchedSparseMTPK1SamplingRoute,
    _DenseMTPK1SamplingRoute,
    _merge_qwen35b_mtp_caches,
    _merge_qwen35b_target_caches,
    generate_a3b_mtp_batch,
)
from mtplx.ragged_kv_cache import RaggedBatchKVCache
from mtplx.sampling import SamplerConfig


VOCAB = 16
LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
    for index in range(40)
)


def _logits(token: int) -> np.ndarray:
    row = np.full((VOCAB,), -8.0, dtype=np.float32)
    row[int(token) % VOCAB] = 8.0
    return row


class _FakeLane:
    def __init__(
        self,
        *,
        fail_verify: bool = False,
        logits_dtype=mx.float32,
        cohort_slots: int = 8,
    ):
        self.geometry = SimpleNamespace(
            cohort_slots=int(cohort_slots),
            verify_tokens=2,
            max_context_tokens=131072,
            num_kv_heads=2,
            head_dim=256,
        )
        self.route_id = f"fake_qwen35b_b{int(cohort_slots)}_t2"
        self.fail_verify = fail_verify
        self.logits_dtype = logits_dtype
        self.last_cache = None
        self.last_mtp_cache = None
        self.prefill_calls = 0

    merge_target_caches = staticmethod(_merge_qwen35b_target_caches)
    merge_mtp_caches = staticmethod(_merge_qwen35b_mtp_caches)

    def prefill_request(self, prompt, *, abort_check=None):
        self.prefill_calls += 1
        if abort_check is not None and abort_check():
            from mtplx.generation import PostcommitAbort

            raise PostcommitAbort("cancelled")
        length = len(prompt)
        values = mx.broadcast_to(
            mx.array(np.asarray(prompt, dtype=np.float32)).reshape(1, 1, length, 1),
            (1, 2, length, 256),
        )
        cache = []
        for layer_type in LAYER_TYPES:
            if layer_type == "full_attention":
                entry = KVCache()
                entry.update_and_fetch(values, values)
            else:
                entry = ArraysCache(2)
                entry[0] = mx.array([[[float(prompt[-1])]]])
                entry[1] = mx.array([[[[float(prompt[-1])]]]])
            cache.append(entry)
        logits = mx.array(_logits(prompt[-1] + 1))[None, :].astype(self.logits_dtype)
        hidden = mx.array([[[float(prompt[-1])]]])
        mtp = KVCache()
        history = list(prompt[1:])
        if history:
            history_values = mx.broadcast_to(
                mx.array(np.asarray(history, dtype=np.float32)).reshape(
                    1, 1, len(history), 1
                ),
                (1, 2, len(history), 256),
            )
            mtp.update_and_fetch(history_values, history_values)
        return cache, logits, hidden, [mtp], 0.0

    def draft_forward(self, hidden, primary, **kwargs):
        del hidden
        mtp_cache = kwargs["mtp_cache"]
        self.last_mtp_cache = mtp_cache
        ids = np.asarray(primary).reshape(-1)
        values = mx.broadcast_to(
            mx.array(ids.astype(np.float32)).reshape(len(ids), 1, 1, 1),
            (len(ids), 2, 1, 256),
        )
        mtp_cache[0].update_and_fetch(values, values)
        rows = []
        for row, token in enumerate(ids):
            target = int(token) + 1
            if row % 2:
                target += 3
            rows.append(_logits(target))
        return mx.array(np.stack(rows)).astype(self.logits_dtype)[:, None, :]

    def update_mtp_cache(self, hidden, token_ids, *, mtp_cache):
        del hidden
        ids = np.asarray(token_ids).reshape(-1)
        values = mx.broadcast_to(
            mx.array(ids.astype(np.float32)).reshape(len(ids), 1, 1, 1),
            (len(ids), 2, 1, 256),
        )
        mtp_cache[0].update_and_fetch(values, values)

    def commit_rows(self, cache, captures, keeps, base_recurrent):
        del base_recurrent
        from mtplx.gdn_capture import commit_captured_rows

        safe_keeps = [max(1, int(value)) for value in keeps]
        assert commit_captured_rows(
            cache,
            captures,
            keep_tokens_by_row=safe_keeps,
            verified_tokens=2,
        )
        inactive = mx.array(
            [1 if int(value) == 0 else 0 for value in keeps], dtype=mx.int32
        )
        for entry in cache:
            if isinstance(entry, RaggedBatchKVCache):
                entry.offsets = entry.offsets - inactive

    def capture_forward(self, verify_input, *, cache):
        if self.fail_verify:
            raise RuntimeError("verify failed")
        self.last_cache = cache
        ids = np.asarray(verify_input)
        logits = np.stack(
            [
                np.stack((_logits(primary + 1), _logits(draft + 1)))
                for primary, draft in ids
            ]
        )
        hidden = ids.astype(np.float32)[:, :, None]
        for entry in cache:
            if isinstance(entry, RaggedBatchKVCache):
                entry.offsets = entry.offsets + 2
                entry._capacity_bound += 2
        conv = ids.astype(np.float32)[:, :, None, None]
        states = ids.astype(np.float32)[:, :, None, None, None]
        captures = {
            layer_idx: {
                "conv_states": mx.array(conv),
                "states": mx.array(states),
            }
            for layer_idx, layer_type in enumerate(LAYER_TYPES)
            if layer_type == "linear_attention"
        }
        return mx.array(logits).astype(self.logits_dtype), mx.array(hidden), captures


def _request(
    request_id: str,
    prompt,
    *,
    max_tokens=4,
    seed=7,
    callback=None,
    cancelled=lambda: False,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
):
    return A3BMTPBatchRequest(
        request_id=request_id,
        prompt_ids=tuple(prompt),
        sampler=SamplerConfig(temperature=temperature, top_p=top_p, top_k=top_k),
        draft_sampler=SamplerConfig(temperature=temperature, top_p=top_p, top_k=top_k),
        seed=seed,
        max_tokens=max_tokens,
        on_token=callback,
        cancelled=cancelled,
    )


def test_driver_runs_fixed_b8_t2_and_commits_one_or_two_positions_per_row():
    lane = _FakeLane()
    streamed = {"a": [], "b": []}
    result = generate_a3b_mtp_batch(
        lane,
        [
            _request("a", [1, 2, 3], max_tokens=2, callback=streamed["a"].append),
            _request("b", [7], max_tokens=2, callback=streamed["b"].append),
        ],
    )

    assert [stream.tokens for stream in result.streams] == [(4, 5), (8, 9)]
    assert streamed == {"a": [4, 5], "b": [8, 9]}
    assert result.accepted_drafts == 1
    assert result.rejected_drafts == 1
    assert dict(result.width_histogram) == {8: 1}
    ragged = next(
        entry for entry in lane.last_cache if isinstance(entry, RaggedBatchKVCache)
    )
    assert np.asarray(ragged.offsets)[:2].tolist() == [5, 2]
    assert isinstance(lane.last_mtp_cache[0], RaggedBatchKVCache)
    assert np.asarray(lane.last_mtp_cache[0].offsets)[:2].tolist() == [4, 1]


def test_driver_reads_real_bfloat16_logits_without_numpy_buffer_errors():
    result = generate_a3b_mtp_batch(
        _FakeLane(logits_dtype=mx.bfloat16),
        [_request(f"row-{row}", [row + 1], max_tokens=2) for row in range(8)],
    )

    assert len(result.streams) == 8
    assert all(len(stream.tokens) == 2 for stream in result.streams)


def test_driver_keeps_greedy_draft_and_verify_logits_on_device(monkeypatch):
    original_asarray = np.asarray

    def reject_full_vocab_transfer(value, *args, **kwargs):
        shape = tuple(int(item) for item in getattr(value, "shape", ()))
        if len(shape) >= 2 and shape[-1] == VOCAB:
            raise AssertionError(
                f"greedy B8 transferred full-vocabulary logits to host: {shape}"
            )
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", reject_full_vocab_transfer)
    result = generate_a3b_mtp_batch(
        _FakeLane(),
        [_request(f"row-{row}", [row + 1], max_tokens=2) for row in range(8)],
    )

    assert len(result.streams) == 8
    assert all(len(stream.tokens) == 2 for stream in result.streams)


def test_driver_keeps_request_rng_and_output_independent_of_neighbor():
    sampler_runs = []
    for neighbor in ([4], [11, 12, 13, 14]):
        result = generate_a3b_mtp_batch(
            _FakeLane(),
            [
                _request("stable", [1, 2, 3], max_tokens=8, seed=91, temperature=0.8),
                _request("neighbor", neighbor, max_tokens=8, seed=123, temperature=0.8),
            ],
        )
        sampler_runs.append(result.streams[0].tokens)

    assert sampler_runs[0] == sampler_runs[1]


def test_driver_uses_batched_sparse_route_for_default_stochastic_sampler(
    monkeypatch,
):
    def fail_dense_distribution(*_args, **_kwargs):
        raise AssertionError("default top-k sampling must stay sparse and batched")

    monkeypatch.setattr(bd, "distribution_from_logits", fail_dense_distribution)
    monkeypatch.setattr(
        fs,
        "batched_sparse_distributions_from_mlx_logits",
        fail_dense_distribution,
    )
    result = generate_a3b_mtp_batch(
        _FakeLane(),
        [
            _request(
                f"row-{row}",
                [row + 1],
                max_tokens=4,
                seed=500 + row,
                temperature=0.6,
                top_p=0.95,
                top_k=4,
            )
            for row in range(8)
        ],
    )

    assert len(result.streams) == 8
    assert all(len(stream.tokens) == 4 for stream in result.streams)


@pytest.mark.parametrize("accepted", [True, False])
def test_sparse_route_matches_dense_fixed_seed_for_every_sampling_phase(accepted):
    request = _request(
        "row-0",
        [1],
        seed=2,
        temperature=1.0,
        top_p=1.0,
        top_k=2,
    )
    dense = _DenseMTPK1SamplingRoute()
    sparse = _BatchedSparseMTPK1SamplingRoute(
        request.sampler,
        request.draft_sampler,
        vocab_size=5,
    )
    dense_rng = np.random.default_rng(2)
    sparse_rng = np.random.default_rng(2)
    primary_logits = mx.array(
        np.tile([0.0, 2.0, -1.0, -2.0, 3.0], (8, 1)),
        dtype=mx.float32,
    )
    draft_logits = primary_logits[:, None, :]

    dense_primary = dense.sample_primary(
        dense.primary_source(primary_logits),
        0,
        request,
        dense_rng,
        [],
        None,
    )
    sparse_primary = sparse.sample_primary(
        sparse.primary_source(primary_logits),
        0,
        request,
        sparse_rng,
        [],
        None,
    )
    dense_proposal = dense.sample_draft(
        dense.draft_source(draft_logits),
        0,
        dense_primary,
        request,
        dense_rng,
    )
    sparse_proposal = sparse.sample_draft(
        sparse.draft_source(draft_logits),
        0,
        sparse_primary,
        request,
        sparse_rng,
    )
    target_row = (
        [0.0, 2.0, -1.0, -2.0, 3.0] if accepted else [3.0, -2.0, 2.0, -1.0, 0.0]
    )
    bonus_row = [0.0, 3.0, -1.0, -2.0, 2.0]
    verify_logits = mx.array(
        np.tile([target_row, bonus_row], (8, 1, 1)),
        dtype=mx.float32,
    )
    dense_result = dense.finish(
        dense.verify_source(verify_logits),
        0,
        dense_proposal,
        request,
        dense_rng,
        [dense_primary],
        True,
    )
    sparse_result = sparse.finish(
        sparse.verify_source(verify_logits),
        0,
        sparse_proposal,
        request,
        sparse_rng,
        [sparse_primary],
        True,
    )

    assert sparse_primary == dense_primary
    assert sparse_proposal.draft_token == dense_proposal.draft_token
    assert sparse_result == dense_result
    assert sparse_result.accepted is accepted
    assert sparse_rng.random() == dense_rng.random()


def test_driver_resets_host_capacity_bounds_to_logical_progress():
    lane = _FakeLane()
    generate_a3b_mtp_batch(
        lane,
        [
            _request("accept", [1, 2, 3], max_tokens=32),
            _request("reject", [7], max_tokens=32),
        ],
    )

    target_ragged = next(
        entry for entry in lane.last_cache if isinstance(entry, RaggedBatchKVCache)
    )
    assert target_ragged._capacity_bound == max(
        np.asarray(target_ragged.offsets).tolist()
    )
    assert lane.last_mtp_cache[0]._capacity_bound == max(
        np.asarray(lane.last_mtp_cache[0].offsets).tolist()
    )


def test_finished_long_prompt_row_stays_frozen_while_short_peer_decodes():
    lane = _FakeLane()
    generate_a3b_mtp_batch(
        lane,
        [
            _request("long-finished", list(range(100)), max_tokens=1),
            _request("short-running", [7], max_tokens=32),
        ],
    )

    target_ragged = next(
        entry for entry in lane.last_cache if isinstance(entry, RaggedBatchKVCache)
    )
    assert int(np.asarray(target_ragged.offsets)[0]) == 101
    assert int(np.asarray(lane.last_mtp_cache[0].offsets)[0]) == 100


def test_merge_prefilled_caches_materializes_and_releases_scalar_sources():
    caches = []
    for length in (2, 5, 1, 1, 1, 1, 1, 1):
        entry = KVCache()
        values = mx.arange(length, dtype=mx.float32).reshape(1, 1, length, 1)
        entry.update_and_fetch(values, values)
        caches.append([entry])

    merged = _merge_qwen35b_mtp_caches(caches)

    assert isinstance(merged[0], RaggedBatchKVCache)
    assert np.asarray(merged[0].offsets).tolist() == [2, 5, 1, 1, 1, 1, 1, 1]
    assert all(cache[0] is None for cache in caches)
    assert np.asarray(merged[0].keys[:, :, :2, :]).shape == (8, 1, 2, 1)


def test_empty_mtp_history_merge_reserves_matching_first_draft_mask():
    caches = [[KVCache()] for _ in range(8)]

    merged = _merge_qwen35b_mtp_caches(caches)[0]
    merged._capacity_bound = 0
    merged.reserve(1)
    mask = merged.make_mask(1)
    keys = mx.zeros((8, 2, 1, 256), dtype=mx.bfloat16)
    values = mx.zeros((8, 2, 1, 256), dtype=mx.bfloat16)
    written_keys, _written_values = merged.update_and_fetch(keys, values)

    assert tuple(mask.shape) == (8, 1, 1, int(written_keys.shape[2]))
    assert np.asarray(merged.offsets).tolist() == [1] * 8


def test_driver_cancellation_stops_future_streaming_without_affecting_peer():
    cancelled = {"value": False}
    first = []

    def on_first(token):
        first.append(token)
        cancelled["value"] = True

    peer = []
    result = generate_a3b_mtp_batch(
        _FakeLane(),
        [
            _request(
                "cancel",
                [1, 2],
                max_tokens=8,
                callback=on_first,
                cancelled=lambda: cancelled["value"],
            ),
            _request("peer", [4], max_tokens=4, callback=peer.append),
        ],
    )

    assert first == [3]
    assert result.streams[0].finish_reason == "cancelled"
    assert result.streams[1].tokens == tuple(peer)
    assert len(peer) == 4


def test_driver_verify_failure_emits_nothing_for_any_request():
    emitted = []
    with pytest.raises(RuntimeError, match="verify failed"):
        generate_a3b_mtp_batch(
            _FakeLane(fail_verify=True),
            [
                _request("a", [1], callback=emitted.append),
                _request("b", [2], callback=emitted.append),
            ],
        )

    assert emitted == []


def test_driver_rejects_capacity_before_any_prefill_allocation():
    from mtplx.a3b_mtp_batch import A3BMTPBatchCapacityError

    lane = _FakeLane()
    lane.geometry.max_context_tokens = 4

    with pytest.raises(A3BMTPBatchCapacityError, match=r"prompt_tokens \+ max_tokens"):
        generate_a3b_mtp_batch(
            lane,
            [_request("a", [1, 2, 3, 4]), _request("b", [5])],
        )

    assert lane.prefill_calls == 0


def test_driver_interrupts_cancelled_prefill_and_keeps_peer_alive():
    cancelled = {"value": False}
    terminals = []

    long_prompt = list(range(100))

    class CancellingLane(_FakeLane):
        def prefill_request(self, prompt, *, abort_check=None):
            if prompt == long_prompt and not cancelled["value"]:
                cancelled["value"] = True
            return super().prefill_request(prompt, abort_check=abort_check)

    first = _request(
        "cancel",
        long_prompt,
        cancelled=lambda: cancelled["value"],
    )
    first = A3BMTPBatchRequest(
        **{
            **first.__dict__,
            "on_terminal": lambda reason, cycles: terminals.append((reason, cycles)),
        }
    )
    lane = CancellingLane()
    result = generate_a3b_mtp_batch(
        lane,
        [first, _request("peer", [4], max_tokens=3)],
    )

    assert terminals == [("cancelled", 0)]
    assert result.streams[0].finish_reason == "cancelled"
    assert len(result.streams[1].tokens) == 3
    target = next(
        entry for entry in lane.last_cache if isinstance(entry, RaggedBatchKVCache)
    )
    assert int(np.asarray(target.offsets)[0]) == 1
    assert int(np.asarray(lane.last_mtp_cache[0].offsets)[0]) == 0
    assert target._capacity_bound == max(np.asarray(target.offsets).tolist())
    assert lane.last_mtp_cache[0]._capacity_bound == max(
        np.asarray(lane.last_mtp_cache[0].offsets).tolist()
    )


def test_later_prefill_poll_closes_an_already_prefilled_cancelled_peer():
    cancelled = {"value": False}
    terminals = []

    class PollingLane(_FakeLane):
        def prefill_request(self, prompt, *, abort_check=None):
            if prompt == [9, 10]:
                cancelled["value"] = True
                assert abort_check is not None
                assert abort_check() is False
                assert terminals == [("cancelled", 0)]
            return super().prefill_request(prompt, abort_check=abort_check)

    long_prompt = list(range(100))
    first = _request(
        "first",
        long_prompt,
        cancelled=lambda: cancelled["value"],
    )
    first = A3BMTPBatchRequest(
        **{
            **first.__dict__,
            "on_terminal": lambda reason, cycles: terminals.append((reason, cycles)),
        }
    )

    lane = PollingLane()
    result = generate_a3b_mtp_batch(
        lane,
        [first, _request("second", [9, 10], max_tokens=2)],
    )

    assert terminals == [("cancelled", 0)]
    assert result.streams[0].finish_reason == "cancelled"
    assert result.streams[1].finish_reason == "length"
    target = next(
        entry for entry in lane.last_cache if isinstance(entry, RaggedBatchKVCache)
    )
    assert int(np.asarray(target.offsets)[0]) == 1
    assert int(np.asarray(lane.last_mtp_cache[0].offsets)[0]) == 0
    assert target._capacity_bound == max(np.asarray(target.offsets).tolist())
    assert lane.last_mtp_cache[0]._capacity_bound == max(
        np.asarray(lane.last_mtp_cache[0].offsets).tolist()
    )


def test_final_prefill_boundary_replaces_newly_cancelled_long_row():
    cancelled = {"value": False}
    terminals = []

    class FinalBoundaryLane(_FakeLane):
        def prefill_request(self, prompt, *, abort_check=None):
            result = super().prefill_request(prompt, abort_check=abort_check)
            if self.prefill_calls == self.geometry.cohort_slots:
                cancelled["value"] = True
            return result

    long_prompt = list(range(100))
    first = _request(
        "first",
        long_prompt,
        cancelled=lambda: cancelled["value"],
    )
    first = A3BMTPBatchRequest(
        **{
            **first.__dict__,
            "on_terminal": lambda reason, cycles: terminals.append((reason, cycles)),
        }
    )
    lane = FinalBoundaryLane()

    result = generate_a3b_mtp_batch(
        lane,
        [first, _request("second", [9, 10], max_tokens=2)],
    )

    assert terminals == [("cancelled", 0)]
    assert result.streams[0].finish_reason == "cancelled"
    target = next(
        entry for entry in lane.last_cache if isinstance(entry, RaggedBatchKVCache)
    )
    assert int(np.asarray(target.offsets)[0]) == 1
    assert int(np.asarray(lane.last_mtp_cache[0].offsets)[0]) == 0
    assert target._capacity_bound == max(np.asarray(target.offsets).tolist())
    assert lane.last_mtp_cache[0]._capacity_bound == max(
        np.asarray(lane.last_mtp_cache[0].offsets).tolist()
    )


# ---------------------------------------------------------------------------
# Session-bank hooks (composite lane: restore at admission, commit pre-merge)
# ---------------------------------------------------------------------------


class _HookLane(_FakeLane):
    """FakeLane that records the session kwargs the driver passes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prefill_kwargs: list[dict] = []

    def prefill_request(self, prompt, *, abort_check=None, restored=None, boundary_sink=None):
        self.prefill_kwargs.append(
            {"prompt": list(prompt), "restored": restored, "boundary_sink": boundary_sink}
        )
        if boundary_sink is not None:
            boundary_sink.append((len(prompt) - 1, object(), None))
        return super().prefill_request(prompt, abort_check=abort_check)


def test_session_hooks_restore_state_reaches_prefill_and_commit_runs_pre_merge():
    lane = _HookLane()
    restored_state = SimpleNamespace(
        cache=None,
        mtp_cache=None,
        restore_point=2,
        boundary_hidden=object(),
        inherited_boundaries=[(2, "snap-2", None)],
    )
    commits: list[dict] = []
    restored_request = _request("warm", [1, 2, 3, 4], max_tokens=2)
    cold_request = _request("cold", [7, 8], max_tokens=2)
    object.__setattr__(restored_request, "session_restore", lambda: restored_state)
    object.__setattr__(
        restored_request, "session_commit", lambda **kw: commits.append(kw)
    )
    object.__setattr__(cold_request, "session_restore", lambda: None)
    object.__setattr__(cold_request, "session_commit", lambda **kw: commits.append(kw))

    result = generate_a3b_mtp_batch(lane, [restored_request, cold_request])

    assert len(result.streams) == 2
    warm_call = lane.prefill_kwargs[0]
    assert warm_call["restored"] is restored_state
    # Inherited boundaries seed the sink; the (fake) prefill appended one more.
    assert warm_call["boundary_sink"][0] == (2, "snap-2", None)
    cold_call = lane.prefill_kwargs[1]
    assert cold_call["restored"] is None
    assert cold_call["boundary_sink"] == [(1, cold_call["boundary_sink"][0][1], None)]
    assert [c["restored"] for c in commits] == [restored_state, None]
    assert [len(c["gdn_boundaries"]) for c in commits] == [2, 1]
    # Padding rows never see session kwargs (old call shape preserved).
    assert all(
        call["restored"] is None and call["boundary_sink"] is None
        for call in lane.prefill_kwargs[2:]
    )


def test_session_restore_failure_degrades_to_cold():
    lane = _HookLane()
    request = _request("a", [1, 2, 3], max_tokens=2)

    def _boom():
        raise RuntimeError("bank offline")

    object.__setattr__(request, "session_restore", _boom)
    result = generate_a3b_mtp_batch(lane, [request, _request("b", [7], max_tokens=2)])
    assert len(result.streams) == 2
    assert lane.prefill_kwargs[0]["restored"] is None


def test_session_commit_failure_never_breaks_the_cohort():
    lane = _HookLane()
    request = _request("a", [1, 2, 3], max_tokens=2)

    def _boom(**_kw):
        raise RuntimeError("bank full")

    object.__setattr__(request, "session_commit", _boom)
    result = generate_a3b_mtp_batch(lane, [request, _request("b", [7], max_tokens=2)])
    assert [stream.finish_reason for stream in result.streams] == ["length", "length"]


def test_prefill_restored_path_runs_suffix_only_with_exact_history_transitions():
    from mtplx.a3b_mtp_batch import _prefill_qwen35b_batch_request

    forwards: list[list[int]] = []
    history_updates: list[tuple[int, list[int]]] = []

    def target_forward(tokens, *, cache, return_hidden, hidden_variant):
        ids = np.asarray(tokens).reshape(-1).tolist()
        forwards.append(ids)
        t = len(ids)
        return mx.zeros((1, t, VOCAB)), mx.arange(t, dtype=mx.float32).reshape(1, t, 1)

    def update_mtp_cache(hidden, ids, *, mtp_cache, position_offset):
        id_list = np.asarray(ids).reshape(-1).tolist()
        history_updates.append((int(hidden.shape[1]), id_list))
        return hidden

    prompt = list(range(100, 112))  # N=12: body 100..110, final 111
    restored = SimpleNamespace(
        cache=[ArraysCache(2)],
        mtp_cache=[KVCache()],
        restore_point=5,
        boundary_hidden=mx.ones((1, 1, 1)),
    )
    sink: list = []
    cache, logits, hidden, mtp_cache, *_ = _prefill_qwen35b_batch_request(
        prompt,
        target_forward=target_forward,
        target_cache_factory=lambda: pytest.fail("restored row must not build a cold cache"),
        mtp_cache_factory=lambda: pytest.fail("restored row must not build a cold MTP cache"),
        update_mtp_cache=update_mtp_cache,
        chunk_size=4,
        cleanup_every=0,
        restored=restored,
        boundary_sink=sink,
    )

    # Target forwards: suffix chunks [5..8], [9..10], then the final token —
    # token 4 (restore_point-1) is never re-run.
    assert forwards == [[105, 106, 107, 108], [109, 110], [111]]
    # History transitions: pre-step (restore_point-1 -> restore_point) from the
    # boundary hidden, then chunk pairs — ids are absolute and complete.
    assert history_updates[0] == (1, [105])
    assert history_updates[1] == (4, [106, 107, 108, 109])
    assert history_updates[2] == (2, [110, 111])
    # Boundary captures at absolute chunk edges (9 and 11 = len(body)).
    assert [record[0] for record in sink] == [9, 11]
    assert cache is restored.cache and mtp_cache is restored.mtp_cache


def test_prefill_cold_path_is_byte_identical_in_shape_and_captures_edges():
    from mtplx.a3b_mtp_batch import _prefill_qwen35b_batch_request

    forwards: list[list[int]] = []

    def target_forward(tokens, *, cache, return_hidden, hidden_variant):
        ids = np.asarray(tokens).reshape(-1).tolist()
        forwards.append(ids)
        t = len(ids)
        return mx.zeros((1, t, VOCAB)), mx.zeros((1, t, 1))

    sink: list = []
    _prefill_qwen35b_batch_request(
        list(range(9)),  # N=9: body 0..7, final 8
        target_forward=target_forward,
        target_cache_factory=lambda: [ArraysCache(2)],
        mtp_cache_factory=lambda: [KVCache()],
        update_mtp_cache=lambda hidden, ids, *, mtp_cache, position_offset: hidden,
        chunk_size=4,
        cleanup_every=0,
        boundary_sink=sink,
    )
    assert forwards == [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    assert [record[0] for record in sink] == [4, 8]


def test_prefill_restored_requires_boundary_hidden():
    from mtplx.a3b_mtp_batch import _prefill_qwen35b_batch_request

    with pytest.raises(ValueError):
        _prefill_qwen35b_batch_request(
            [1, 2, 3, 4],
            target_forward=lambda *a, **k: pytest.fail("must not forward"),
            target_cache_factory=lambda: [],
            mtp_cache_factory=lambda: [],
            update_mtp_cache=lambda *a, **k: None,
            chunk_size=4,
            cleanup_every=0,
            restored=SimpleNamespace(
                cache=[], mtp_cache=[], restore_point=2, boundary_hidden=None
            ),
        )


def test_per_row_stats_reflect_each_rows_own_active_window():
    """Streams carry the ROW's truth, not cohort totals (Pro flaw 4).

    An early finisher must report fewer active cycles and an earlier
    terminal stamp than a peer that keeps decoding; per-row draft outcomes
    must partition the cohort totals.
    """
    lane = _FakeLane()
    result = generate_a3b_mtp_batch(
        lane,
        [
            _request("early", list(range(100)), max_tokens=1),
            _request("late", [7], max_tokens=32),
        ],
    )

    early, late = result.streams
    assert early.finish_reason == "length"
    assert late.finish_reason == "length"
    assert 0 < early.cycles < late.cycles
    assert late.cycles == result.cycles
    assert early.terminal_perf_s is not None
    assert late.terminal_perf_s is not None
    assert early.terminal_perf_s <= late.terminal_perf_s
    assert early.accepted_drafts + late.accepted_drafts == result.accepted_drafts
    assert early.rejected_drafts + late.rejected_drafts == result.rejected_drafts
    assert early.accepted_drafts + early.rejected_drafts <= early.cycles


def test_driver_runs_native_width_three_cohort_end_to_end():
    """A B3 lane runs 2-3 real rows with no padded slots beyond width 3."""
    lane = _FakeLane(cohort_slots=3)
    streamed = {"a": [], "b": [], "c": []}
    result = generate_a3b_mtp_batch(
        lane,
        [
            _request("a", [1, 2, 3], max_tokens=2, callback=streamed["a"].append),
            _request("b", [7], max_tokens=2, callback=streamed["b"].append),
            _request("c", [11], max_tokens=2, callback=streamed["c"].append),
        ],
    )

    assert [stream.tokens for stream in result.streams] == [(4, 5), (8, 9), (12, 13)]
    assert streamed == {"a": [4, 5], "b": [8, 9], "c": [12, 13]}
    assert dict(result.width_histogram) == {3: result.cycles}
    assert result.route_id == "fake_qwen35b_b3_t2"
    ragged = next(
        entry for entry in lane.last_cache if isinstance(entry, RaggedBatchKVCache)
    )
    # Physical batch is exactly three rows: no inert padding beyond width.
    assert int(ragged.offsets.shape[0]) == 3
    assert int(lane.last_mtp_cache[0].offsets.shape[0]) == 3


def test_width_three_lane_rejects_more_requests_than_slots():
    lane = _FakeLane(cohort_slots=3)
    with pytest.raises(ValueError, match="2-3 requests"):
        generate_a3b_mtp_batch(
            lane,
            [_request(str(row), [row + 1], max_tokens=1) for row in range(4)],
        )


def test_width_three_pads_single_free_slot_with_inert_row():
    """A 2-request B3 cohort pads exactly one slot and keeps rows truthful."""
    lane = _FakeLane(cohort_slots=3)
    result = generate_a3b_mtp_batch(
        lane,
        [
            _request("a", [1, 2, 3], max_tokens=2),
            _request("b", [7], max_tokens=2),
        ],
    )
    assert [stream.tokens for stream in result.streams] == [(4, 5), (8, 9)]
    assert dict(result.width_histogram) == {3: result.cycles}
    # 2 real prefills + 1 padded slot.
    assert lane.prefill_calls == 3


def test_per_row_stats_stay_row_truthful_in_width_three_cohort():
    """8e1cc55 semantics survive the native width: rows report their own truth."""
    lane = _FakeLane(cohort_slots=3)
    result = generate_a3b_mtp_batch(
        lane,
        [
            _request("early", list(range(100)), max_tokens=1),
            _request("late", [7], max_tokens=32),
        ],
    )

    early, late = result.streams
    assert early.finish_reason == "length"
    assert late.finish_reason == "length"
    assert 0 < early.cycles < late.cycles
    assert late.cycles == result.cycles
    assert early.terminal_perf_s is not None
    assert late.terminal_perf_s is not None
    assert early.terminal_perf_s <= late.terminal_perf_s
    assert early.accepted_drafts + late.accepted_drafts == result.accepted_drafts
    assert early.rejected_drafts + late.rejected_drafts == result.rejected_drafts


class _WidthThreeHookLane(_HookLane):
    def __init__(self, **kwargs):
        super().__init__(cohort_slots=3, **kwargs)


def test_session_hooks_fire_in_width_three_cohorts():
    """0e6fc5e restore/commit machinery is width-agnostic per-row scalar work."""
    lane = _WidthThreeHookLane()
    restored_state = SimpleNamespace(
        cache=None,
        mtp_cache=None,
        restore_point=2,
        boundary_hidden=object(),
        inherited_boundaries=[(2, "snap-2", None)],
    )
    commits: list[dict] = []
    restored_request = _request("warm", [1, 2, 3, 4], max_tokens=2)
    cold_request = _request("cold", [7, 8], max_tokens=2)
    object.__setattr__(restored_request, "session_restore", lambda: restored_state)
    object.__setattr__(
        restored_request, "session_commit", lambda **kw: commits.append(kw)
    )
    object.__setattr__(cold_request, "session_restore", lambda: None)
    object.__setattr__(cold_request, "session_commit", lambda **kw: commits.append(kw))

    result = generate_a3b_mtp_batch(lane, [restored_request, cold_request])

    assert len(result.streams) == 2
    warm_call = lane.prefill_kwargs[0]
    assert warm_call["restored"] is restored_state
    assert warm_call["boundary_sink"][0] == (2, "snap-2", None)
    cold_call = lane.prefill_kwargs[1]
    assert cold_call["restored"] is None
    assert [c["restored"] for c in commits] == [restored_state, None]
    assert [len(c["gdn_boundaries"]) for c in commits] == [2, 1]
    # Exactly one padded slot behind the two real rows, with no session kwargs.
    assert len(lane.prefill_kwargs) == 3
    assert (
        lane.prefill_kwargs[2]["restored"] is None
        and lane.prefill_kwargs[2]["boundary_sink"] is None
    )


def test_commit_rows_and_merge_infer_width_from_arguments():
    """The shared commit/merge helpers scale to three rows by inference."""
    from mtplx.a3b_mtp_batch import _commit_qwen35b_b8_t2_rows

    width = 3
    cache = []
    base_recurrent = {}
    for layer_idx, layer_type in enumerate(LAYER_TYPES):
        if layer_type == "full_attention":
            entry = RaggedBatchKVCache(batch_size=width, step=256)
            entry.offsets = mx.array([4, 4, 4], dtype=mx.int32)
            cache.append(entry)
        else:
            entry = ArraysCache(2)
            entry[0] = mx.zeros((width, 1))
            entry[1] = mx.zeros((width, 1, 1))
            cache.append(entry)
            base_recurrent[layer_idx] = (entry[0], entry[1])
    captures = {
        layer_idx: {
            "conv_states": mx.broadcast_to(
                mx.arange(2, dtype=mx.float32).reshape(1, 2, 1), (width, 2, 1)
            ),
            "states": mx.broadcast_to(
                mx.arange(2, dtype=mx.float32).reshape(1, 2, 1, 1), (width, 2, 1, 1)
            ),
        }
        for layer_idx, layer_type in enumerate(LAYER_TYPES)
        if layer_type == "linear_attention"
    }

    _commit_qwen35b_b8_t2_rows(cache, captures, [0, 1, 2], base_recurrent)

    ragged = next(e for e in cache if isinstance(e, RaggedBatchKVCache))
    assert np.asarray(ragged.offsets).tolist() == [2, 3, 4]
    recurrent = next(e for e in cache if isinstance(e, ArraysCache))
    # Row 0 inactive (keep=0) -> base value; rows 1/2 take position keep-1.
    assert np.asarray(recurrent[0]).reshape(-1).tolist() == [0.0, 0.0, 1.0]

    merged = _merge_qwen35b_mtp_caches(
        [[_kv_with_tokens(5)], [_kv_with_tokens(3)], [_kv_with_tokens(7)]]
    )[0]
    assert int(merged.keys.shape[0]) == 3
    assert np.asarray(merged.offsets).tolist() == [5, 3, 7]


def _kv_with_tokens(count: int) -> KVCache:
    entry = KVCache()
    values = mx.ones((1, 1, count, 1), dtype=mx.float32)
    entry.update_and_fetch(values, values)
    return entry


def test_merge_capacity_follows_logical_offsets_not_stale_allocation():
    """A restored clone's oversized allocation must not inflate the cohort.

    Destination capacity previously came from max physical keys.shape[2]:
    one boundary-trimmed restore keeping its bank entry's 2048-slot
    allocation padded every row of the merged B8 cache to 2048.
    """
    big = KVCache()
    grown = mx.arange(2000, dtype=mx.float32).reshape(1, 1, 2000, 1)
    big.update_and_fetch(grown, grown)
    big.offset = 100  # boundary-trimmed restore: committed 100, allocated 2048

    caches = [[big]]
    for _ in range(7):
        entry = KVCache()
        small = mx.ones((1, 1, 5, 1), dtype=mx.float32)
        entry.update_and_fetch(small, small)
        caches.append([entry])

    merged = _merge_qwen35b_mtp_caches(caches)[0]

    assert int(merged.keys.shape[2]) == 256
    assert np.asarray(merged.offsets).tolist() == [100, 5, 5, 5, 5, 5, 5, 5]
    assert (
        np.asarray(merged.keys[0, :, :100, :]).tolist()
        == np.asarray(grown[0, :, :100, :]).tolist()
    )
