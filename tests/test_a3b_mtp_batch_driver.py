from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from mtplx.a3b_mtp_batch import A3BMTPBatchRequest, generate_a3b_mtp_batch
from mtplx.ragged_kv_cache import RaggedBatchKVCache
from mtplx.sampling import SamplerConfig


VOCAB = 16


def _logits(token: int) -> np.ndarray:
    row = np.full((VOCAB,), -8.0, dtype=np.float32)
    row[int(token) % VOCAB] = 8.0
    return row


class _FakeLane:
    def __init__(self, *, fail_verify: bool = False):
        self.geometry = SimpleNamespace(
            cohort_slots=8,
            verify_tokens=2,
        )
        self.route_id = "fake_qwen35b_b8_t2"
        self.fail_verify = fail_verify
        self.last_cache = None

    def prefill_request(self, prompt):
        kv = KVCache()
        length = len(prompt)
        values = mx.array(np.asarray(prompt, dtype=np.float32)).reshape(1, 1, length, 1)
        kv.update_and_fetch(values, values)
        recurrent = ArraysCache(2)
        recurrent[0] = mx.array([[[float(prompt[-1])]]])
        recurrent[1] = mx.array([[[[float(prompt[-1])]]]])
        logits = mx.array(_logits(prompt[-1] + 1))[None, :]
        hidden = mx.array([[[float(prompt[-1])]]])
        return [kv, recurrent], logits, hidden, 0.0

    def make_mtp_cache(self):
        return []

    def draft_forward(self, hidden, primary, **kwargs):
        del hidden, kwargs
        ids = np.asarray(primary).reshape(-1)
        rows = []
        for row, token in enumerate(ids):
            target = int(token) + 1
            if row % 2:
                target += 3
            rows.append(_logits(target))
        return mx.array(np.stack(rows))[:, None, :]

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
        conv = ids.astype(np.float32)[:, :, None, None]
        states = ids.astype(np.float32)[:, :, None, None, None]
        captures = {1: {"conv_states": mx.array(conv), "states": mx.array(states)}}
        return mx.array(logits), mx.array(hidden), captures


def _request(
    request_id: str,
    prompt,
    *,
    max_tokens=4,
    seed=7,
    callback=None,
    cancelled=lambda: False,
    temperature=0.0,
):
    return A3BMTPBatchRequest(
        request_id=request_id,
        prompt_ids=tuple(prompt),
        sampler=SamplerConfig(temperature=temperature, top_p=1.0, top_k=0),
        draft_sampler=SamplerConfig(temperature=temperature, top_p=1.0, top_k=0),
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
