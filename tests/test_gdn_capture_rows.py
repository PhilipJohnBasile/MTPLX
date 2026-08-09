from __future__ import annotations

import mlx.core as mx
import numpy as np

from mtplx.cache_state import OwnedRecurrentStateCache
from mtplx.gdn_capture import commit_captured_rows
from mtplx.ragged_kv_cache import RaggedBatchKVCache


def _captured(batch: int = 4) -> dict[int, dict[str, mx.array]]:
    conv = np.arange(batch * 2 * 2 * 3, dtype=np.float32).reshape(
        batch, 2, 2, 3
    )
    state = np.arange(batch * 2 * 2 * 2 * 3, dtype=np.float32).reshape(
        batch, 2, 2, 2, 3
    )
    return {
        0: {
            "conv_states": mx.array(conv),
            "states": mx.array(state),
        }
    }


def test_commit_captured_rows_selects_each_rows_authoritative_position() -> None:
    captures = _captured()
    recurrent = OwnedRecurrentStateCache(size=2)

    assert commit_captured_rows(
        [recurrent], captures, keep_tokens_by_row=[1, 2, 1, 2], verified_tokens=2
    )

    expected_conv = np.stack(
        [
            np.array(captures[0]["conv_states"])[row, keep - 1]
            for row, keep in enumerate([1, 2, 1, 2])
        ]
    )
    expected_state = np.stack(
        [
            np.array(captures[0]["states"])[row, keep - 1]
            for row, keep in enumerate([1, 2, 1, 2])
        ]
    )
    np.testing.assert_array_equal(np.array(recurrent.state[0]), expected_conv)
    np.testing.assert_array_equal(np.array(recurrent.state[1]), expected_state)


def test_commit_captured_rows_rewinds_only_rejecting_ragged_offsets() -> None:
    ragged = RaggedBatchKVCache(
        batch_size=4,
        offsets=mx.array([12, 22, 32, 42], dtype=mx.int32),
    )

    assert commit_captured_rows(
        [ragged], {}, keep_tokens_by_row=[1, 2, 1, 2], verified_tokens=2
    )

    np.testing.assert_array_equal(
        np.array(ragged.offsets), np.array([11, 22, 31, 42], dtype=np.int32)
    )


def test_commit_captured_rows_fails_closed_for_unsupported_capture() -> None:
    recurrent = OwnedRecurrentStateCache(size=2)
    captures = _captured()
    captures[0]["tape"] = mx.array([1])

    assert not commit_captured_rows(
        [recurrent], captures, keep_tokens_by_row=[1, 2, 1, 2], verified_tokens=2
    )
    assert not commit_captured_rows(
        [recurrent], captures, keep_tokens_by_row=[0, 2, 1, 2], verified_tokens=2
    )
    assert not commit_captured_rows(
        [recurrent], {}, keep_tokens_by_row=[1, 2, 1, 2], verified_tokens=2
    )
