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
    def __init__(self, *, fail_verify: bool = False, logits_dtype=mx.float32):
        self.geometry = SimpleNamespace(
            cohort_slots=8,
            verify_tokens=2,
            max_context_tokens=131072,
            num_kv_heads=2,
            head_dim=256,
        )
        self.route_id = "fake_qwen35b_b8_t2"
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
        logits = mx.array(_logits(prompt[-1] + 1))[None, :].astype(
            self.logits_dtype
        )
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
        draft_sampler=SamplerConfig(
            temperature=temperature, top_p=top_p, top_k=top_k
        ),
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
    ragged = next(entry for entry in lane.last_cache if isinstance(entry, RaggedBatchKVCache))
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
        [0.0, 2.0, -1.0, -2.0, 3.0]
        if accepted
        else [3.0, -2.0, 2.0, -1.0, 0.0]
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
