"""#436: a client-capped 1-token answer gets one WARNING line.

Pi caps max_tokens at its model contextWindow minus a chars/4 estimate of
the transcript; when the estimate overflows it sends max_tokens=1. The
server honors it, the answer stops after one token (finish_reason
"length", often inside a tool call) and the client reports a truncated
response. The serve log now says who capped it.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import mtplx.server.openai as srv


def _state(window: int = 16384):
    return SimpleNamespace(
        context_window=window,
        memory_plan=None,
        allow_swap=False,
        args=SimpleNamespace(
            max_response_tokens=None,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            default_presence_penalty=0.0,
            default_frequency_penalty=0.0,
        ),
    )


def _warnings(caplog):
    return [r for r in caplog.records if r.getMessage() == "max_tokens leaves no room to answer"]


def test_tiny_cap_with_room_warns_and_names_the_client(caplog):
    caplog.set_level(logging.WARNING, logger=srv.LOGGER.name)
    lease, _sampler, limits = srv._generation_params(
        _state(), prompt_token_count=13612, max_tokens=1, temperature=None, top_p=None, top_k=None
    )
    assert lease == 1
    assert limits["request_max_tokens"] == 1
    records = _warnings(caplog)
    assert len(records) == 1
    assert records[0].requested_max_tokens == 1
    assert records[0].remaining_context_tokens == 16384 - 13612
    assert "client" in records[0].hint


def test_no_warning_when_the_window_itself_is_the_limit(caplog):
    caplog.set_level(logging.WARNING, logger=srv.LOGGER.name)
    srv._generation_params(
        _state(), prompt_token_count=16380, max_tokens=1, temperature=None, top_p=None, top_k=None
    )
    assert _warnings(caplog) == []


def test_no_warning_for_a_normal_cap(caplog):
    caplog.set_level(logging.WARNING, logger=srv.LOGGER.name)
    srv._generation_params(
        _state(), prompt_token_count=2948, max_tokens=6972, temperature=None, top_p=None, top_k=None
    )
    srv._generation_params(
        _state(), prompt_token_count=2948, max_tokens=None, temperature=None, top_p=None, top_k=None
    )
    assert _warnings(caplog) == []
