from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

from mtplx.a3b_mtp_batch import (
    A3BMTPBatchResult,
    A3BMTPBatchStreamResult,
)
from mtplx.sampling import SamplerConfig
from mtplx.server.mtp_batch import MTPBatchGenerationService, MTPBatchJob


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


def test_eight_requests_stream_only_their_own_tokens_and_close_once():
    driver = _Driver()
    service = _service(driver)
    jobs = [_job(index) for index in range(8)]
    futures = [service.submit(job) for job in jobs]

    assert service.pump_once()
    results = [future.result(timeout=1) for future in futures]

    assert [result["request_id"] for result in results] == [job.request_id for job in jobs]
    for index, (job, result) in enumerate(zip(jobs, results)):
        expected = [index + 10, index + 1010]
        assert job.test_emitted == expected
        assert result["tokens"] == expected
        assert result["stats"]["generation_mode"] == "mtp"
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

    assert future.result(timeout=1)["tokens"] == [77]
    assert calls == [job.request_id]
    assert driver.widths == []
    assert service.snapshot()["solo_runs"] == 1


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
