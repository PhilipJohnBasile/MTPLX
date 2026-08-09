from __future__ import annotations

from threading import Event, Thread, get_ident
from types import MappingProxyType, SimpleNamespace

import pytest

from mtplx.a3b_mtp_batch import (
    A3BMTPBatchResult,
    A3BMTPBatchStreamResult,
)
from mtplx.sampling import SamplerConfig
from mtplx.server.mtp_batch import (
    MTPBatchFinalizeOwnership,
    MTPBatchGenerationService,
    MTPBatchJob,
)


class _Driver:
    def __init__(self):
        self.widths = []
        self.fail_next = False

    def __call__(self, lane, requests):
        del lane
        self.widths.append(len(requests))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("cohort failed")
        streams = []
        for request in requests:
            marker = int(request.prompt_ids[0])
            output = (marker, marker + 1000)
            for token in output:
                if request.cancelled():
                    break
                if request.on_token is not None:
                    request.on_token(token)
            streams.append(
                A3BMTPBatchStreamResult(
                    request_id=request.request_id,
                    tokens=output if not request.cancelled() else (),
                    finish_reason="length" if not request.cancelled() else "cancelled",
                    cycles=2,
                    accepted_drafts=1,
                    rejected_drafts=0,
                )
            )
        return A3BMTPBatchResult(
            streams=tuple(streams),
            cycles=2,
            accepted_drafts=len(requests),
            rejected_drafts=0,
            route_id="fake-b8-t2",
            width_histogram=MappingProxyType({8: 2}),
        )


def _job(index: int, *, compatibility_key=("default",), solo_runner=None):
    emitted = []
    job = MTPBatchJob(
        request_id=f"request-{index}",
        prompt_ids=[index + 10],
        max_tokens=2,
        sampler=SamplerConfig(temperature=0.0),
        draft_sampler=SamplerConfig(temperature=0.0),
        seed=100 + index,
        stop_token_ids=set(),
        token_callback=emitted.extend,
        compatibility_key=compatibility_key,
        generation_limits={},
        solo_runner=solo_runner,
        cancel_error=lambda job: RuntimeError(f"cancelled {job.request_id}"),
    )
    job.test_emitted = emitted
    return job


def _service(driver):
    state = SimpleNamespace(runtime=SimpleNamespace(tokenizer=None))
    return MTPBatchGenerationService(
        state,
        lane=object(),
        driver=driver,
        batch_wait_s=0.0,
        auto_schedule=False,
    )


def test_b1_exact_profile_runs_each_sealed_request_through_unchanged_solo_runner():
    driver = _Driver()
    lane = SimpleNamespace(
        numerics_profile="b1-exact",
        route_id="qwen35b_mtp_batch_b1_exact_serial",
    )
    service = MTPBatchGenerationService(
        SimpleNamespace(runtime=SimpleNamespace(tokenizer=None)),
        lane=lane,
        driver=driver,
        batch_wait_s=0.0,
        auto_schedule=False,
    )
    calls = []
    jobs = []
    for index in range(8):
        expected = {
            "request_id": f"request-{index}",
            "tokens": [index, index + 100],
            "stats": {"mode": "mtp", "server_seed": 100 + index},
        }

        def solo(job, *, expected=expected):
            calls.append(job.request_id)
            return expected

        jobs.append(_job(index, solo_runner=solo))
    futures = [service.submit(job) for job in jobs]

    assert service.pump_once()
    results = [future.result(timeout=1) for future in futures]

    assert driver.widths == []
    assert calls == [job.request_id for job in jobs]
    assert [result["tokens"] for result in results] == [
        [index, index + 100] for index in range(8)
    ]
    assert all(result["_mtp_batch_solo"] is True for result in results)
    snapshot = service.snapshot()
    assert snapshot["last_route_id"] == "qwen35b_mtp_batch_b1_exact_serial"
    assert snapshot["solo_runs"] == 8
    assert snapshot["fixed_width_histogram"] == {}


def test_b1_exact_profile_keeps_one_solo_failure_request_local():
    driver = _Driver()
    lane = SimpleNamespace(
        numerics_profile="b1-exact",
        route_id="qwen35b_mtp_batch_b1_exact_serial",
    )
    owner_finalize_calls = []
    service = MTPBatchGenerationService(
        SimpleNamespace(runtime=SimpleNamespace(tokenizer=None)),
        lane=lane,
        driver=driver,
        batch_wait_s=0.0,
        auto_schedule=False,
        owner_finalize=lambda jobs: owner_finalize_calls.append(
            [job.request_id for job in jobs]
        ),
    )
    jobs = []
    for index in range(3):

        def solo(job, *, index=index):
            if index == 1:
                raise RuntimeError("row-local exact failure")
            return {
                "request_id": job.request_id,
                "tokens": [index],
                "stats": {"mode": "mtp"},
            }

        jobs.append(_job(index, solo_runner=solo))
    futures = [service.submit(job) for job in jobs]

    assert service.pump_once()

    assert futures[0].result(timeout=1)["tokens"] == [0]
    with pytest.raises(RuntimeError, match="row-local exact failure"):
        futures[1].result(timeout=1)
    assert futures[2].result(timeout=1)["tokens"] == [2]
    assert owner_finalize_calls == [["request-1"]]
    assert driver.widths == []


def test_b1_exact_owner_finalize_failure_stops_remaining_serial_rows():
    driver = _Driver()
    calls = []
    service = MTPBatchGenerationService(
        SimpleNamespace(runtime=SimpleNamespace(tokenizer=None)),
        lane=SimpleNamespace(
            numerics_profile="b1-exact",
            route_id="qwen35b_mtp_batch_b1_exact_serial",
        ),
        driver=driver,
        batch_wait_s=0.0,
        auto_schedule=False,
        owner_finalize=lambda _jobs: (_ for _ in ()).throw(
            RuntimeError("clear exploded")
        ),
    )
    jobs = []
    for index in range(3):

        def solo(job, *, index=index):
            calls.append(job.request_id)
            if index == 0:
                raise RuntimeError("generation failed")
            return {"tokens": [index], "stats": {"mode": "mtp"}}

        jobs.append(_job(index, solo_runner=solo))
    futures = [service.submit(job) for job in jobs]

    assert service.pump_once()

    assert calls == ["request-0"]
    for future in futures:
        with pytest.raises(RuntimeError, match="owner finalize failed"):
            future.result(timeout=1)
    assert "owner finalize failed" in str(service.snapshot()["last_error"])


def test_eight_requests_stream_only_their_own_tokens_and_close_once():
    driver = _Driver()
    service = _service(driver)
    jobs = [_job(index) for index in range(8)]
    futures = [service.submit(job) for job in jobs]

    assert service.pump_once()
    results = [future.result(timeout=1) for future in futures]

    assert [result["request_id"] for result in results] == [
        job.request_id for job in jobs
    ]
    for index, (job, result) in enumerate(zip(jobs, results)):
        expected = [index + 10, index + 1010]
        assert job.test_emitted == expected
        assert result["tokens"] == expected
        assert result["stats"]["generation_mode"] == "mtp"
        assert result["stats"]["target_verify_cycles"] == 2
    assert service.snapshot()["batch_histogram"] == {"8": 1}
    assert service.snapshot()["fixed_width_histogram"] == {"8": 2}


def test_cancelled_rows_close_as_errors_without_changing_survivors():
    driver = _Driver()
    service = _service(driver)
    jobs = [_job(index) for index in range(8)]
    futures = [service.submit(job) for job in jobs]
    jobs[1].cancel_event.set()
    jobs[6].cancel_event.set()

    service.pump_once()

    assert "cancelled request-1" in str(futures[1].exception(timeout=1))
    assert "cancelled request-6" in str(futures[6].exception(timeout=1))
    for index in (0, 2, 3, 4, 5, 7):
        assert futures[index].result(timeout=1)["tokens"] == [
            index + 10,
            index + 1010,
        ]


def test_callback_stop_error_closes_only_its_request():
    service = _service(_Driver())
    jobs = [_job(index) for index in range(8)]

    def stop_row(_tokens):
        raise RuntimeError("row-local stop")

    jobs[3].token_callback = stop_row
    futures = [service.submit(job) for job in jobs]

    service.pump_once()

    assert "row-local stop" in str(futures[3].exception(timeout=1))
    for index in (0, 1, 2, 4, 5, 6, 7):
        assert futures[index].result(timeout=1)["tokens"] == [
            index + 10,
            index + 1010,
        ]


def test_one_request_uses_unchanged_solo_runner():
    driver = _Driver()
    service = _service(driver)
    calls = []

    def solo(job):
        calls.append(job.request_id)
        return {"request_id": job.request_id, "tokens": [77], "stats": {"mode": "mtp"}}

    job = _job(0, solo_runner=solo)
    future = service.submit(job)

    service.pump_once()

    result = future.result(timeout=1)
    assert result["tokens"] == [77]
    assert result["_mtp_batch_solo"] is True
    assert calls == [job.request_id]
    assert driver.widths == []
    assert service.snapshot()["solo_runs"] == 1


def test_cancellation_before_admission_keeps_finalize_ownership_local():
    ownership = MTPBatchFinalizeOwnership()
    service = _service(_Driver())
    job = _job(0)
    job.finalize_ownership = ownership

    assert ownership.claim_cancellation_finalize() == "not_required_before_admission"
    future = service.submit(job)

    with pytest.raises(RuntimeError, match="cancelled request-0"):
        future.result(timeout=1)
    assert service.snapshot()["pending"] == 0


def test_admission_transfers_cancellation_finalize_ownership_to_model_owner():
    ownership = MTPBatchFinalizeOwnership()
    assert ownership.accept_owner() is True
    ownership.mark_admitted()

    assert ownership.claim_cancellation_finalize() == "cohort_owner_after_decode"


def test_cancellation_claim_wins_selection_to_admission_race():
    ownership = MTPBatchFinalizeOwnership()
    entered_mark = Event()
    release_mark = Event()
    owner_finalize_calls = []
    solo_calls = []
    original_mark_admitted = ownership.mark_admitted

    def blocked_mark_admitted():
        entered_mark.set()
        assert release_mark.wait(timeout=2)
        return original_mark_admitted()

    ownership.mark_admitted = blocked_mark_admitted
    service = MTPBatchGenerationService(
        SimpleNamespace(runtime=SimpleNamespace(tokenizer=None)),
        lane=object(),
        driver=_Driver(),
        batch_wait_s=0.0,
        auto_schedule=False,
        owner_finalize=lambda jobs: owner_finalize_calls.append(list(jobs)),
    )
    job = _job(
        0,
        solo_runner=lambda job: solo_calls.append(job.request_id) or {},
    )
    job.finalize_ownership = ownership
    service.submit(job)
    pump = Thread(target=service.pump_once)
    pump.start()
    assert entered_mark.wait(timeout=1)

    job.cancel_event.set()
    assert ownership.claim_cancellation_finalize() == "not_required_before_admission"
    release_mark.set()
    pump.join(timeout=2)

    assert not pump.is_alive()
    assert solo_calls == []
    assert owner_finalize_calls == []
    with pytest.raises(RuntimeError, match="cancelled request-0"):
        job.future.result(timeout=1)


@pytest.mark.parametrize("cancel_before_start", [True, False])
def test_cancelled_solo_request_uses_truthful_finalize_ownership(
    cancel_before_start,
):
    events = []

    def solo(job):
        events.append("solo")
        job.cancel_event.set()
        raise job.cancel_error(job)

    state = SimpleNamespace(runtime=SimpleNamespace(tokenizer=None))
    service = MTPBatchGenerationService(
        state,
        lane=object(),
        driver=_Driver(),
        batch_wait_s=0.0,
        auto_schedule=False,
        owner_finalize=lambda jobs: events.append(
            ("owner_finalize", [job.request_id for job in jobs])
        ),
    )
    job = _job(0, solo_runner=solo)
    if cancel_before_start:
        job.cancel_event.set()
    service.submit(job)

    service.pump_once()

    with pytest.raises(RuntimeError, match="cancelled request-0"):
        job.future.result(timeout=1)
    if cancel_before_start:
        assert events == []
        assert (
            job.finalize_ownership.claim_cancellation_finalize()
            == "not_required_before_admission"
        )
    else:
        assert events == [
            "solo",
            ("owner_finalize", [job.request_id]),
        ]


def test_cohort_text_strips_terminal_stop_tokens():
    service = _service(_Driver())
    service.state.runtime.tokenizer = SimpleNamespace(
        decode=lambda tokens: ",".join(str(token) for token in tokens)
    )
    jobs = [_job(0), _job(1)]
    jobs[0].stop_token_ids = {1010}
    for job in jobs:
        service.submit(job)

    service.pump_once()

    result = jobs[0].future.result(timeout=1)
    assert "text" not in result
    assert result["_mtp_batch_decode_on_request"] is True
    assert result["_mtp_batch_stop_token_ids"] == [1010]


def test_cohort_seals_at_eight_and_later_request_waits_for_next_pump():
    driver = _Driver()
    service = _service(driver)
    jobs = [_job(index) for index in range(9)]
    futures = [service.submit(job) for job in jobs]

    service.pump_once()

    assert all(future.done() for future in futures[:8])
    assert not futures[8].done()
    assert service.snapshot()["pending"] == 1
    service.pump_once()
    assert futures[8].done()
    assert driver.widths == [8]


def test_compatibility_key_seals_separate_cohorts():
    driver = _Driver()
    service = _service(driver)
    first = _job(0, compatibility_key=("a",))
    second = _job(1, compatibility_key=("b",))
    service.submit(first)
    service.submit(second)

    service.pump_once()

    assert first.future.done()
    assert not second.future.done()
    assert driver.widths == []


def test_driver_error_fails_only_sealed_cohort_and_fresh_cohort_can_run():
    driver = _Driver()
    driver.fail_next = True
    service = _service(driver)
    first = [_job(index) for index in range(8)]
    later = _job(
        20,
        solo_runner=lambda job: {
            "request_id": job.request_id,
            "tokens": [99],
            "stats": {"mode": "mtp"},
        },
    )
    for job in [*first, later]:
        service.submit(job)

    service.pump_once()

    assert all("cohort failed" in str(job.future.exception(timeout=1)) for job in first)
    assert not later.future.done()
    service.pump_once()
    assert later.future.done()
    assert service.snapshot()["last_error"] == "RuntimeError: cohort failed"


def test_shutdown_closes_queued_requests():
    service = _service(_Driver())
    job = _job(0)
    service.submit(job)

    service.shutdown()

    with pytest.raises(RuntimeError, match="shut down"):
        job.future.result(timeout=1)


def test_shutdown_closes_active_requests_before_scheduler_cancellation():
    started = Event()
    owner_finalize = []

    def solo(job):
        started.set()
        assert job.cancel_event.wait(timeout=2)
        raise job.cancel_error(job)

    state = SimpleNamespace(runtime=SimpleNamespace(tokenizer=None))
    service = MTPBatchGenerationService(
        state,
        lane=object(),
        driver=_Driver(),
        batch_wait_s=0.0,
        auto_schedule=False,
        owner_finalize=lambda jobs: owner_finalize.append(
            (get_ident(), [item.request_id for item in jobs])
        ),
    )
    job = _job(0, solo_runner=solo)
    service.submit(job)
    pump = Thread(target=service.pump_once)
    pump.start()
    assert started.wait(timeout=1)
    shutdown_thread = get_ident()

    service.shutdown(timeout_s=2)
    pump.join(timeout=1)

    assert job.cancel_requested()
    with pytest.raises(RuntimeError, match="cancelled request-0"):
        job.future.result(timeout=1)
    assert owner_finalize == [(pump.ident, [job.request_id])]
    assert owner_finalize[0][0] != shutdown_thread


def test_duplicate_public_request_ids_keep_distinct_cohort_rows():
    def driver(_lane, requests):
        requests[0].on_token(10)
        return A3BMTPBatchResult(
            streams=(
                A3BMTPBatchStreamResult(requests[0].request_id, (10,), "length"),
                A3BMTPBatchStreamResult(requests[1].request_id, (), "cancelled"),
            ),
            cycles=1,
            accepted_drafts=0,
            rejected_drafts=1,
            route_id="fake-b8-t2",
            width_histogram=MappingProxyType({8: 1}),
        )

    service = _service(driver)
    first = _job(0)
    second = _job(1)
    first.request_id = second.request_id = "client-duplicate"
    service.submit(first)
    service.submit(second)

    service.pump_once()

    assert first.future.result(timeout=1)["tokens"] == [10]
    with pytest.raises(RuntimeError, match="cancelled client-duplicate"):
        second.future.result(timeout=1)


def test_cancelled_terminal_future_closes_before_long_peer_finishes():
    peer_blocked = Event()
    release_peer = Event()

    def blocking_driver(_lane, requests):
        requests[0].on_terminal("cancelled", 0)
        peer_blocked.set()
        assert release_peer.wait(timeout=2)
        return A3BMTPBatchResult(
            streams=(
                A3BMTPBatchStreamResult("0", (), "cancelled"),
                A3BMTPBatchStreamResult("1", (11,), "length"),
            ),
            cycles=1,
            accepted_drafts=0,
            rejected_drafts=1,
            route_id="fake-b8-t2",
            width_histogram=MappingProxyType({8: 1}),
        )

    service = _service(blocking_driver)
    first = _job(0)
    second = _job(1)
    service.submit(first)
    service.submit(second)
    pump = Thread(target=service.pump_once)
    pump.start()
    try:
        assert peer_blocked.wait(timeout=1)
        with pytest.raises(RuntimeError, match="cancelled request-0"):
            first.future.result(timeout=0.1)
        assert not second.future.done()
    finally:
        release_peer.set()
        pump.join(timeout=2)


def test_successful_future_waits_for_owner_finalize_after_every_row_stops():
    first_terminal = Event()
    release_peer = Event()
    owner_finalize_calls = []

    def blocking_driver(_lane, requests):
        requests[0].on_terminal("length", 1)
        first_terminal.set()
        assert release_peer.wait(timeout=2)
        return A3BMTPBatchResult(
            streams=(
                A3BMTPBatchStreamResult("0", (), "length"),
                A3BMTPBatchStreamResult("1", (), "length"),
            ),
            cycles=1,
            accepted_drafts=0,
            rejected_drafts=1,
            route_id="fake-b8-t2",
            width_histogram=MappingProxyType({8: 1}),
        )

    state = SimpleNamespace(runtime=SimpleNamespace(tokenizer=None))
    service = MTPBatchGenerationService(
        state,
        lane=SimpleNamespace(route_id="fake-b8-t2"),
        driver=blocking_driver,
        batch_wait_s=0.0,
        auto_schedule=False,
        owner_finalize=lambda jobs: owner_finalize_calls.append(list(jobs)),
    )
    first = _job(0)
    second = _job(1)
    service.submit(first)
    service.submit(second)
    pump = Thread(target=service.pump_once)
    pump.start()
    try:
        assert first_terminal.wait(timeout=1)
        assert not first.future.done()
        assert owner_finalize_calls == []
    finally:
        release_peer.set()
        pump.join(timeout=2)

    assert owner_finalize_calls == [[first, second]]
    assert first.future.result(timeout=1)["_mtp_batch_defer_mlx_finalize"] is True


@pytest.mark.parametrize("failure", ["exception", "uncleared"])
def test_owner_finalize_failure_poisons_service_without_claiming_success(failure):
    def owner_finalize(_jobs):
        if failure == "exception":
            raise RuntimeError("clear exploded")
        return {
            "mlx_cache_cleanup": {
                "cleared": False,
                "reason": "clear_cache_error",
            }
        }

    state = SimpleNamespace(runtime=SimpleNamespace(tokenizer=None))
    service = MTPBatchGenerationService(
        state,
        lane=SimpleNamespace(route_id="fake-b8-t2"),
        driver=_Driver(),
        batch_wait_s=0.0,
        auto_schedule=False,
        owner_finalize=owner_finalize,
    )
    first = _job(0)
    second = _job(1)
    service.submit(first)
    service.submit(second)

    service.pump_once()

    assert "finalize" in str(service.snapshot()["last_error"]).lower()
    with pytest.raises(RuntimeError, match="owner finalize failed"):
        first.future.result(timeout=1)
    with pytest.raises(RuntimeError, match="owner finalize failed"):
        second.future.result(timeout=1)
    assert (
        first.finalize_ownership.claim_cancellation_finalize()
        == "cohort_owner_finalize_failed"
    )
    later = _job(2)
    with pytest.raises(RuntimeError, match="shut down"):
        service.submit(later).result(timeout=1)


def test_owner_finalize_failure_drains_requests_queued_behind_active_cohort():
    entered = Event()
    release = Event()
    solo_calls = []

    def blocking_driver(lane, requests):
        entered.set()
        assert release.wait(timeout=2)
        return _Driver()(lane, requests)

    def owner_finalize(_jobs):
        raise RuntimeError("clear exploded")

    service = MTPBatchGenerationService(
        SimpleNamespace(runtime=SimpleNamespace(tokenizer=None)),
        lane=SimpleNamespace(route_id="fake-b8-t2"),
        driver=blocking_driver,
        batch_wait_s=0.0,
        auto_schedule=False,
        owner_finalize=owner_finalize,
    )
    first = _job(0)
    second = _job(1)
    service.submit(first)
    service.submit(second)
    pump = Thread(target=service.pump_once)
    pump.start()
    assert entered.wait(timeout=1)

    queued = _job(
        2,
        solo_runner=lambda job: solo_calls.append(job.request_id) or {},
    )
    service.submit(queued)
    release.set()
    pump.join(timeout=2)

    assert not pump.is_alive()
    assert solo_calls == []
    assert service.snapshot()["pending"] == 0
    with pytest.raises(RuntimeError, match="owner finalize failed"):
        queued.future.result(timeout=1)


def test_real_model_owner_scheduler_gathers_eight_requests():
    from mtplx.model_scheduler import ModelWorkScheduler

    scheduler = ModelWorkScheduler(name="test-mtp-batch-owner", idle_grace_s=0.0)
    driver = _Driver()
    state = SimpleNamespace(
        runtime=SimpleNamespace(tokenizer=None), model_scheduler=scheduler
    )
    service = MTPBatchGenerationService(
        state,
        lane=SimpleNamespace(route_id="fake-b8-t2"),
        driver=driver,
        batch_wait_s=0.05,
    )
    try:
        futures = [service.submit(_job(index)) for index in range(8)]
        results = [future.result(timeout=2) for future in futures]

        assert driver.widths == [8]
        assert len(results) == 8
        assert service.snapshot()["batch_histogram"] == {"8": 1}
    finally:
        service.shutdown()
        scheduler.shutdown(wait=True, cancel_futures=True)
