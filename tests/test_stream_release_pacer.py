"""Release pacer for block-sized token batches (2026-09-02).

A context-copy round commits up to 25 tokens at once; without pacing the
stream hands the client one frame per round. These tests pin the knob
parsing and the interval law: a batch drains in about 90% of the expected
gap, never slower than 30 ms per piece, and batches below the floor are
left alone.
"""

from __future__ import annotations

import pytest

from mtplx.server.openai import (
    _stream_pacer_enabled,
    _stream_pacer_interval,
    _stream_pacer_min_tokens,
)


def test_pacer_is_on_by_default_and_honors_the_kill_switch(monkeypatch):
    monkeypatch.delenv("MTPLX_STREAM_PACER", raising=False)
    assert _stream_pacer_enabled() is True
    for raw in ("0", "false", "off", "no", ""):
        monkeypatch.setenv("MTPLX_STREAM_PACER", raw)
        assert _stream_pacer_enabled() is False
    monkeypatch.setenv("MTPLX_STREAM_PACER", "1")
    assert _stream_pacer_enabled() is True


def test_pacer_floor_parses_and_never_drops_below_two(monkeypatch):
    monkeypatch.delenv("MTPLX_STREAM_PACER_MIN_TOKENS", raising=False)
    assert _stream_pacer_min_tokens() == 6
    monkeypatch.setenv("MTPLX_STREAM_PACER_MIN_TOKENS", "10")
    assert _stream_pacer_min_tokens() == 10
    monkeypatch.setenv("MTPLX_STREAM_PACER_MIN_TOKENS", "1")
    assert _stream_pacer_min_tokens() == 2
    monkeypatch.setenv("MTPLX_STREAM_PACER_MIN_TOKENS", "banana")
    assert _stream_pacer_min_tokens() == 6


@pytest.mark.parametrize(
    ("pieces", "gap", "expected"),
    [
        (25, 0.113, pytest.approx(0.113 * 0.9 / 25)),  # a Flash copy block drains inside its round
        (25, 0.0, pytest.approx(0.1 * 0.9 / 25)),  # no history yet: assume a 100 ms round
        (2, 1.0, 0.030),  # never slower than 30 ms per piece
        (1, 0.5, 0.0),  # a single piece is released at once
        (1000, 0.05, 0.001),  # never faster than 1 ms per piece
    ],
)
def test_pacer_interval_law(pieces, gap, expected):
    assert _stream_pacer_interval(pieces, gap) == expected


def test_pacer_drains_a_block_before_the_next_round():
    # 24 tokens plus the bonus every 113 ms: the schedule must finish before the next batch lands.
    pieces = 25
    dt = _stream_pacer_interval(pieces, 0.113)
    assert (pieces - 1) * dt < 0.113
