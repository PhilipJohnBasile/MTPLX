"""Busy-session forking semantics for parallel clients without session ids (#94)."""

import threading
import time

import pytest

from mtplx.engine_session import (
    EngineSessionBusy,
    EngineSessionManager,
    _new_anon_session_id,
)


def test_prefix_receipt_belongs_to_the_request_even_after_another_resolution():
    manager = EngineSessionManager()
    anonymous = {}
    manager.resolve_session_id(prompt_ids=[1, 2, 3], diagnostic_out=anonymous)
    assert anonymous["prompt_len"] == 3
    explicit = {}
    manager.resolve_session_id(headers={"x-mtplx-session-id": "opencode"},
                               diagnostic_out=explicit)
    assert explicit == {}
    assert manager.last_prefix_diagnostic is None
    # The awaiting anonymous request keeps its own original evidence.
    manager.resolve_session_id(prompt_ids=[4, 5], diagnostic_out={})
    assert anonymous["prompt_len"] == 3


def test_busy_implicit_session_forks_to_fresh_anonymous_session():
    manager = EngineSessionManager()
    shared = manager.get_or_create("anon-shared")
    assert shared.try_begin_generation()
    try:
        with manager.generation_slot(shared, source="longest_prefix") as acquired:
            assert acquired.session_id != shared.session_id
            assert acquired.session_id.startswith("anon-")
            assert acquired.in_flight
        assert not acquired.in_flight
    finally:
        shared.end_generation()


def test_busy_pending_postcommit_source_also_forks():
    manager = EngineSessionManager()
    shared = manager.get_or_create("anon-shared")
    assert shared.try_begin_generation()
    try:
        with manager.generation_slot(
            shared, source="pending_postcommit_near_prefix"
        ) as acquired:
            assert acquired.session_id != shared.session_id
    finally:
        shared.end_generation()


def test_busy_explicit_session_still_raises():
    manager = EngineSessionManager()
    named = manager.get_or_create("user-42")
    assert named.try_begin_generation()
    try:
        with pytest.raises(EngineSessionBusy):
            with manager.generation_slot(named, source="header.x-mtplx-session-id"):
                pass
    finally:
        named.end_generation()


def test_idle_session_is_acquired_directly_without_forking():
    manager = EngineSessionManager()
    shared = manager.get_or_create("anon-shared")

    with manager.generation_slot(shared, source="longest_prefix") as acquired:
        assert acquired is shared
        assert shared.in_flight
    assert not shared.in_flight


def test_parallel_prefix_matched_generations_all_complete():
    manager = EngineSessionManager()
    shared = manager.get_or_create("anon-shared")
    acquired_ids: list[str] = []
    errors: list[Exception] = []
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            with manager.generation_slot(shared, source="longest_prefix") as acquired:
                with lock:
                    acquired_ids.append(acquired.session_id)
                time.sleep(0.05)
        except Exception as error:  # pragma: no cover - failure detail
            with lock:
                errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # The pre-fix behavior failed three of four with EngineSessionBusy.
    assert errors == []
    assert len(acquired_ids) == 4
    assert not shared.in_flight


def test_anonymous_session_ids_survive_burst_allocation():
    # The clock-derived scheme could collide inside one timer tick under
    # parallel bursts; random ids cannot.
    ids = {_new_anon_session_id() for _ in range(4096)}
    assert len(ids) == 4096
