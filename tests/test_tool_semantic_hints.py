"""CPU-only contracts for the bounded target-side OoO-Spec lane."""

from __future__ import annotations

from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
from pathlib import Path
from threading import Event, Thread
import time
from types import SimpleNamespace

import pytest

from mtplx import tool_semantic_hints as semantic_hints
from mtplx.tool_semantic_hints import (
    CallableSemanticHintProvider,
    HTTPSemanticHintProvider,
    K1SemanticHintDraftSource,
    DisabledSemanticHintProvider,
    SEMANTIC_HINT_TIMEOUT_MAX_S,
    SemanticHintMailbox,
    SemanticHintMailboxOwner,
    SemanticHintProviderConfig,
    SemanticHint,
    SemanticHintRequest,
    normalize_semantic_hint_timeout_s,
    semantic_hint_provider_from_config,
    validate_semantic_hint,
    suffix_alignment_length,
    target_tokenized_hint_bank,
)

try:
    import mlx.core as mx
    import numpy as np

    from mtplx.generation import generate_mtpk
    from mtplx.mtp_patch import MTPContract
    from mtplx.runtime import MTPLXRuntime
    from mtplx.sampling import SamplerConfig
    _MLX_SKIP_REASON = ""
except ModuleNotFoundError as exc:
    mx = None
    np = None
    _MLX_SKIP_REASON = f"MLX runtime dependency is unavailable: {exc.name}"


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    for name in (
        "MTPLX_COMPILED_VERIFY",
        "MTPLX_COMPILED_TARGET_PREFIX",
        "MTPLX_CONTEXT_COPY_TARGET_PREFIX",
    ):
        monkeypatch.delenv(name, raising=False)
    if mx is None:
        yield
        return
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    }
]


def _callable_hint_provider(_request):
    return _hint(arguments={"path": "provider.txt"})


def _callable_blocking_provider(_request):
    time.sleep(30.0)
    return None


def _callable_mutating_provider(request):
    request.tools[0]["function"]["name"] = "provider_mutated"
    request.tools[0]["function"]["parameters"]["properties"].clear()
    request.messages[0]["content"]["nested"] = "provider_mutated"
    return _hint(arguments={"path": "provider.txt"})


def _mailbox(
    payload=None,
    *,
    pending: bool = False,
    timeout_s: float = 10.0,
    clock=None,
    deadline_scheduler=None,
) -> tuple[SemanticHintMailbox, Future]:
    future: Future = Future()
    if not pending:
        future.set_result(payload)
    return (
        SemanticHintMailbox(
            future,
            timeout_s=timeout_s,
            clock=clock or (lambda: 0.0),
            **(
                {"deadline_scheduler": deadline_scheduler}
                if deadline_scheduler is not None
                else {}
            ),
        ),
        future,
    )


class _ManualDeadline:
    def __init__(self, delay_s, callback):
        self.delay_s = delay_s
        self.callback = callback
        self.cancelled = False
        self.fired = False

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if self.cancelled:
            return
        self.fired = True
        self.callback()


class _ManualDeadlineScheduler:
    def __init__(self):
        self.handles = []

    def __call__(self, delay_s, callback):
        handle = _ManualDeadline(delay_s, callback)
        self.handles.append(handle)
        return handle

    def fire(self):
        assert self.handles
        self.handles[-1].fire()


class _CountingPipeEndpoint:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _NeverStartedProcess:
    def __init__(self):
        self.pid = None
        self.start_calls = 0
        self.is_alive_calls = 0
        self.join_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    def start(self):
        self.start_calls += 1
        raise RuntimeError("synthetic process start failure")

    def is_alive(self):
        self.is_alive_calls += 1
        raise AssertionError("can only test a child process")

    def join(self, timeout=None):
        del timeout
        self.join_calls += 1
        raise AssertionError("can only join a started process")

    def terminate(self):
        self.terminate_calls += 1
        raise AssertionError("can only terminate a started process")

    def kill(self):
        self.kill_calls += 1
        raise AssertionError("can only kill a started process")


class _StartFailureContext:
    def __init__(self):
        self.receive_connection = _CountingPipeEndpoint()
        self.send_connection = _CountingPipeEndpoint()
        self.process = _NeverStartedProcess()

    def Pipe(self, *, duplex):
        assert duplex is False
        return self.receive_connection, self.send_connection

    def Process(self, **_kwargs):
        return self.process

    def get_start_method(self):
        return "spawn"


class _StartedBlockingProcess:
    def __init__(self):
        self.pid = 4242
        self.started = False
        self.alive = False
        self.join_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    def start(self):
        self.started = True
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        del timeout
        self.join_calls += 1

    def terminate(self):
        self.terminate_calls += 1
        self.alive = False

    def kill(self):
        self.kill_calls += 1
        self.alive = False


class _StartedBlockingContext:
    def __init__(self):
        self.receive_connection = _CountingPipeEndpoint()
        self.send_connection = _CountingPipeEndpoint()
        self.process = _StartedBlockingProcess()

    def Pipe(self, *, duplex):
        assert duplex is False
        return self.receive_connection, self.send_connection

    def Process(self, **_kwargs):
        return self.process

    def get_start_method(self):
        return "spawn"


class _FailingMonitorThread:
    def __init__(self, **_kwargs):
        self.join_calls = 0

    def start(self):
        raise RuntimeError("synthetic monitor thread start failure with private data")

    def join(self, timeout=None):
        del timeout
        self.join_calls += 1
        raise AssertionError("an unstarted monitor thread cannot be joined")


class _FailingStartDeadlineTimer:
    instances = []

    def __init__(self, delay_s, callback):
        self.delay_s = delay_s
        self.callback = callback
        self.daemon = False
        self.name = ""
        self.cancel_calls = 0
        self.__class__.instances.append(self)

    def start(self):
        raise RuntimeError("synthetic deadline start failure with private data")

    def cancel(self):
        self.cancel_calls += 1


def _source(mailbox: SemanticHintMailbox, *, rendered=(1, 2, 3, 4)):
    return K1SemanticHintDraftSource(
        mailbox,
        tools=_TOOLS,
        render=lambda _hint: list(rendered),
    )


def _hint(*, tool_index: int = 0, arguments=None):
    return {
        "tool_index": tool_index,
        "arguments": {"path": "private.txt"} if arguments is None else arguments,
    }


@pytest.mark.skipif(mx is None, reason=_MLX_SKIP_REASON)
def test_disabled_and_never_ready_sources_leave_cpu_generation_byte_identical():
    baseline = _generate()
    mailbox, _future = _mailbox(pending=True)
    pending = _generate(source=_source(mailbox))

    assert pending.tokens == baseline.tokens
    assert pending.text == baseline.text
    assert pending.stats.tool_semantic_hints["pending"] > 0
    assert pending.stats.tool_semantic_hints["drafts"] == 0


@pytest.mark.skipif(mx is None, reason=_MLX_SKIP_REASON)
def test_ready_hint_is_fully_target_verified_and_never_logs_arguments():
    baseline = _generate()
    mailbox, _future = _mailbox(_hint())
    hinted = _generate(source=_source(mailbox, rendered=tuple(range(1, 17))))

    assert hinted.tokens == baseline.tokens
    assert hinted.stats.tool_semantic_hints["accepted"] > 0
    assert hinted.stats.tool_semantic_hints["drafts"] > 0
    assert "private.txt" not in json.dumps(hinted.stats.tool_semantic_hints)


def test_delayed_readiness_polls_without_blocking_then_suffix_aligns():
    mailbox, future = _mailbox(pending=True)
    source = _source(mailbox)

    assert source.propose(primary_token=1, committed_tokens=(1,)) is None
    future.set_result(_hint())
    proposal = source.propose(primary_token=1, committed_tokens=(1,))

    assert proposal is not None and proposal.token == 2
    assert source.telemetry()["pending"] == 1
    assert source.telemetry()["suffix_matches"] == 1


def test_ready_hint_retries_alignment_after_target_preamble_tokens():
    mailbox, _future = _mailbox(_hint())
    source = _source(mailbox, rendered=(10, 11, 12, 13))

    # The ready sidecar result must not be discarded while the target emits
    # unrelated reasoning/preamble tokens before it reaches the tool call.
    assert source.propose(primary_token=7, committed_tokens=(4, 5, 6)) is None
    assert source.propose(primary_token=8, committed_tokens=(4, 5, 6, 7)) is None
    proposal = source.propose(primary_token=10, committed_tokens=(4, 5, 6, 7, 10))

    assert proposal is not None and proposal.token == 11
    telemetry = source.telemetry()
    assert telemetry["alignment_waits"] == 2
    assert telemetry["suffix_matches"] == 1


def test_disabled_null_provider_never_proposes():
    source = _source(
        SemanticHintMailbox.submit(
            DisabledSemanticHintProvider(),
            SemanticHintRequest(messages=(), tools=()),
            timeout_s=1.0,
        )
    )

    assert source.propose(primary_token=1, committed_tokens=(1,)) is None
    assert source.telemetry()["status"] == "disabled"


def test_configured_endpoint_without_opt_in_keeps_the_provider_disabled():
    provider = semantic_hint_provider_from_config(
        SemanticHintProviderConfig(
            enabled=False,
            endpoint="https://hints.example.test/v1/hint",
        )
    )

    assert isinstance(provider, DisabledSemanticHintProvider)


@pytest.mark.parametrize(
    "timeout_s",
    [float("nan"), float("inf"), float("-inf"), 0.0, -0.1, 30.0001],
)
def test_timeout_validation_rejects_nonfinite_nonpositive_and_over_max(timeout_s):
    with pytest.raises(ValueError, match="finite"):
        normalize_semantic_hint_timeout_s(timeout_s)


def test_timeout_validation_accepts_the_bounded_maximum():
    assert (
        normalize_semantic_hint_timeout_s(SEMANTIC_HINT_TIMEOUT_MAX_S)
        == SEMANTIC_HINT_TIMEOUT_MAX_S
    )


@pytest.mark.parametrize(
    "timeout_s",
    [float("nan"), float("inf"), float("-inf"), 0.0, -0.1, 30.0001],
)
def test_direct_mailbox_invalid_timeout_cancels_future_without_scheduling(timeout_s):
    future = Future()
    scheduler = _ManualDeadlineScheduler()

    with pytest.raises(ValueError, match="finite"):
        SemanticHintMailbox(
            future,
            timeout_s=timeout_s,
            deadline_scheduler=scheduler,
        )

    assert future.cancelled()
    assert scheduler.handles == []


@pytest.mark.parametrize(
    "timeout_s",
    [float("nan"), float("inf"), float("-inf"), 0.0, -0.1, 30.0001],
)
def test_mailbox_submit_rejects_invalid_timeout_before_provider_start(timeout_s):
    class RecordingProvider:
        calls = 0

        def submit(self, _request):
            self.calls += 1
            return Future()

    provider = RecordingProvider()

    with pytest.raises(ValueError, match="finite"):
        SemanticHintMailbox.submit(
            provider,
            SemanticHintRequest(messages=(), tools=()),
            timeout_s=timeout_s,
        )

    assert provider.calls == 0


@pytest.mark.parametrize(
    "timeout_s",
    [float("nan"), float("inf"), float("-inf"), 0.0, -0.1, 30.0001],
)
def test_enabled_provider_config_rejects_invalid_timeout(timeout_s):
    with pytest.raises(ValueError, match="finite"):
        semantic_hint_provider_from_config(
            SemanticHintProviderConfig(
                enabled=True,
                endpoint="https://hints.example.test/v1/hint",
                timeout_s=timeout_s,
            )
        )


def test_disabled_provider_config_does_not_start_or_validate_transport():
    provider = semantic_hint_provider_from_config(
        SemanticHintProviderConfig(
            enabled=False,
            endpoint="http://non-loopback.invalid/hint",
            timeout_s=float("inf"),
        )
    )

    assert isinstance(provider, DisabledSemanticHintProvider)


def test_semantic_hint_request_is_a_deep_independent_json_snapshot():
    messages = [{"role": "user", "content": {"nested": "target"}}]
    tools = json.loads(json.dumps(_TOOLS))
    request = SemanticHintRequest(messages=tuple(messages), tools=tuple(tools))

    request.messages[0]["content"]["nested"] = "provider_mutated"
    request.tools[0]["function"]["name"] = "provider_mutated"
    request.tools[0]["function"]["parameters"]["properties"].clear()
    payload = request.to_payload()
    payload["tools"][0]["function"]["name"] = "transport_mutated"

    assert messages[0]["content"]["nested"] == "target"
    assert tools == _TOOLS
    assert request.tools[0]["function"]["name"] == "provider_mutated"


def test_callable_provider_mutation_cannot_change_target_render_or_schema():
    messages = [{"role": "user", "content": {"nested": "target"}}]
    target_tools = json.loads(json.dumps(_TOOLS))
    request = SemanticHintRequest(messages=tuple(messages), tools=tuple(target_tools))
    provider = CallableSemanticHintProvider(_callable_mutating_provider)
    mailbox = SemanticHintMailbox(
        provider.submit(request),
        timeout_s=5.0,
    )
    rendered = []

    def render(hint):
        rendered.append((hint.tool_name, hint.arguments))
        return [1, 2]

    source = K1SemanticHintDraftSource(
        mailbox,
        tools=target_tools,
        render=render,
    )
    try:
        deadline = time.monotonic() + 5.0
        proposal = None
        while proposal is None and time.monotonic() < deadline:
            proposal = source.propose(primary_token=1, committed_tokens=(1,))
            if proposal is None:
                time.sleep(0.01)

        assert proposal is not None and proposal.token == 2
        assert rendered == [("read_file", {"path": "provider.txt"})]
        assert target_tools == _TOOLS
        assert messages[0]["content"]["nested"] == "target"
        assert request.tools[0]["function"]["name"] == "read_file"
    finally:
        source.close()


def test_full_accept_advances_only_after_target_acceptance():
    mailbox, _ = _mailbox(_hint())
    source = _source(mailbox)

    assert source.propose(primary_token=1, committed_tokens=(1,)).token == 2
    # Merely proposing does not advance the cursor.
    assert source.telemetry()["accepted"] == 0
    source.commit_target_prefix(1)
    # Target-prefix's target-only bonus is consumed only after it appears in
    # the next committed suffix; it is never trusted from the sidecar.
    assert source.propose(primary_token=3, committed_tokens=(1, 2, 3)).token == 4
    source.commit_target_prefix(1)

    telemetry = source.telemetry()
    assert telemetry["accepted"] == 2
    assert telemetry["rejected"] == 0


def test_first_rejection_drops_the_remaining_hint():
    mailbox, _ = _mailbox(_hint())
    source = _source(mailbox)

    assert source.propose(primary_token=1, committed_tokens=(1,)).token == 2
    source.commit_target_prefix(0)

    assert source.propose(primary_token=9, committed_tokens=(1, 9)) is None
    assert source.telemetry()["status"] == "rejected"
    assert source.telemetry()["rejected"] == 1


def test_mid_hint_rejection_drops_the_remaining_hint():
    mailbox, _ = _mailbox(_hint())
    source = _source(mailbox)

    assert source.propose(primary_token=1, committed_tokens=(1,)).token == 2
    source.commit_target_prefix(1)
    assert source.propose(primary_token=2, committed_tokens=(1, 2)).token == 3
    source.commit_target_prefix(0)

    assert source.propose(primary_token=8, committed_tokens=(1, 2, 8)) is None
    assert source.telemetry()["rejected"] == 1


@pytest.mark.parametrize(
    "payload,status",
    [
        (_hint(tool_index=4), "invalid_tool_index"),
        (_hint(arguments={"unknown": "x"}), "invalid_arguments"),
        ({"tool_index": 0, "arguments": {"path": "x"}, "extra": 1}, "invalid_payload"),
    ],
)
def test_invalid_index_and_schema_never_propose(payload, status):
    mailbox, _ = _mailbox(payload)
    source = _source(mailbox)

    assert source.propose(primary_token=1, committed_tokens=(1,)) is None
    assert source.telemetry()["status"] == status
    assert source.telemetry()["drafts"] == 0


@pytest.mark.parametrize(
    ("tool_index", "arguments"),
    [(True, {}), (0, {"value": float("nan")})],
)
def test_semantic_hint_constructor_enforces_the_strict_wire_shape(
    tool_index,
    arguments,
):
    with pytest.raises(ValueError, match="invalid_payload"):
        SemanticHint(tool_index=tool_index, arguments=arguments)


def test_template_mismatch_never_proposes():
    mailbox, _ = _mailbox(_hint())
    source = K1SemanticHintDraftSource(
        mailbox,
        tools=_TOOLS,
        render=lambda _hint: None,
    )

    assert source.propose(primary_token=1, committed_tokens=(1,)) is None
    assert source.telemetry()["status"] == "template_mismatch"


def test_target_tokenized_bank_requires_an_exact_baseline_prefix():
    bank = target_tokenized_hint_bank((10, 11), (10, 11, 12, 13))

    assert bank is not None and bank.tokens == (12, 13)
    assert target_tokenized_hint_bank((10, 11), (10, 99, 12)) is None


def test_froggeric_style_assistant_prefix_rewrite_fails_closed():
    # Froggeric profiles can rewrite the open assistant scaffold when a
    # completed tool call is appended. The one-view MVP must not treat the
    # resulting token stream as a continuation of the generation prompt.
    baseline = (151644, 872, 198, 151645, 198, 151644, 77091, 198)
    rewritten = (151644, 872, 198, 151645, 198, 151644, 77091, 271, 151667)

    assert target_tokenized_hint_bank(baseline, rewritten) is None


def test_timeout_and_cancel_are_nonblocking_terminal_states():
    now = [0.0]
    timeout_mailbox, _ = _mailbox(
        pending=True,
        timeout_s=0.25,
        clock=lambda: now[0],
    )
    timeout_source = _source(timeout_mailbox)
    assert timeout_source.propose(primary_token=1, committed_tokens=(1,)) is None
    now[0] = 0.25
    assert timeout_source.propose(primary_token=1, committed_tokens=(1,)) is None
    assert timeout_source.telemetry()["status"] == "timeout"

    cancelled_mailbox, future = _mailbox(pending=True)
    cancelled_source = _source(cancelled_mailbox)
    cancelled_source.close()
    assert future.cancelled()
    assert cancelled_source.propose(primary_token=1, committed_tokens=(1,)) is None
    assert cancelled_source.telemetry()["status"] == "cancelled"


def test_deadline_cancels_pending_future_without_any_mailbox_poll():
    now = [0.0]
    scheduler = _ManualDeadlineScheduler()
    mailbox, future = _mailbox(
        pending=True,
        timeout_s=0.25,
        clock=lambda: now[0],
        deadline_scheduler=scheduler,
    )

    now[0] = 0.25
    scheduler.fire()

    assert future.cancelled()
    assert mailbox.poll().status == "timeout"
    assert scheduler.handles[0].fired


def test_default_deadline_timer_cancels_without_polling():
    future = Future()
    completed = Event()
    future.add_done_callback(lambda _future: completed.set())
    mailbox = SemanticHintMailbox(future, timeout_s=0.05)

    assert completed.wait(2.0)
    assert future.cancelled()
    assert mailbox.poll().status == "timeout"


def test_result_completed_after_deadline_is_rejected_when_polled_later():
    now = [0.0]
    scheduler = _ManualDeadlineScheduler()
    mailbox, future = _mailbox(
        pending=True,
        timeout_s=0.25,
        clock=lambda: now[0],
        deadline_scheduler=scheduler,
    )

    now[0] = 0.5
    future.set_result(_hint())
    now[0] = 10.0

    assert mailbox.completion_s == 0.5
    assert mailbox.poll().status == "timeout"
    assert scheduler.handles[0].cancelled


def test_on_time_completion_survives_late_poll_and_cleans_deadline_handle():
    now = [0.0]
    scheduler = _ManualDeadlineScheduler()
    mailbox, future = _mailbox(
        pending=True,
        timeout_s=1.0,
        clock=lambda: now[0],
        deadline_scheduler=scheduler,
    )

    now[0] = 0.5
    future.set_result(_hint())
    now[0] = 10.0
    polled = mailbox.poll()

    assert polled.status == "ready"
    assert polled.hint is not None
    assert mailbox.completion_s == 0.5
    assert scheduler.handles[0].cancelled


def test_pre_generation_failure_cancels_an_untransferred_mailbox():
    mailbox, future = _mailbox(pending=True)
    owner = SemanticHintMailboxOwner()
    owner.adopt(mailbox)

    with pytest.raises(RuntimeError, match="template failure"):
        try:
            raise RuntimeError("template failure")
        finally:
            owner.cancel_untransferred()

    assert future.cancelled()


def test_owner_retains_source_until_generation_atomically_adopts_it():
    mailbox, future = _mailbox(pending=True)
    source = _source(mailbox)
    owner = SemanticHintMailboxOwner()
    owner.adopt(mailbox)
    owner.stage_source(source)

    owner.cancel_untransferred()

    assert future.cancelled()
    assert not source.adopt_for_generation()


def test_atomic_generation_adoption_is_the_only_ownership_transfer():
    mailbox, future = _mailbox(pending=True)
    source = _source(mailbox)
    owner = SemanticHintMailboxOwner()
    owner.adopt(mailbox)
    owner.stage_source(source)

    assert source.adopt_for_generation()
    owner.cancel_untransferred(force=True)
    assert not future.cancelled()

    source.close()
    assert future.cancelled()


def test_stream_defer_requires_forced_cleanup_when_generation_never_starts():
    mailbox, future = _mailbox(pending=True)
    source = _source(mailbox)
    owner = SemanticHintMailboxOwner()
    owner.adopt(mailbox)
    owner.stage_source(source)
    owner.defer_to_stream()

    owner.cancel_untransferred()
    assert not future.cancelled()
    owner.cancel_untransferred(force=True)

    assert future.cancelled()


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://hints.example.test/v1/hint",
        "http://localhost:9000/hint",
        "http://127.0.0.1:9000/hint",
        "http://[::1]:9000/hint",
    ],
)
def test_http_provider_transport_accepts_https_and_exact_loopback(endpoint):
    HTTPSemanticHintProvider(endpoint, timeout_s=0.1)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://hints.example.test/hint",
        "http://localhost.example.test/hint",
        "http://127.0.0.2/hint",
        "ftp://localhost/hint",
        "https://:443/hint",
    ],
)
def test_http_provider_transport_rejects_non_loopback_cleartext(endpoint):
    with pytest.raises(ValueError, match="HTTPS"):
        HTTPSemanticHintProvider(endpoint, timeout_s=0.1)


@pytest.mark.parametrize(
    "timeout_s",
    [float("nan"), float("inf"), float("-inf"), 0.0, -0.1, 30.0001],
)
def test_http_provider_rejects_invalid_timeout_before_submit(timeout_s):
    with pytest.raises(ValueError, match="finite"):
        HTTPSemanticHintProvider(
            "https://hints.example.test/v1/hint",
            timeout_s=timeout_s,
        )


def test_http_provider_process_returns_decoded_semantics_and_exits():
    class HintHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            response = b'{"hint":{"tool_index":0,"arguments":{"path":"x"}}}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), HintHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    provider = HTTPSemanticHintProvider(
        f"http://127.0.0.1:{server.server_port}/hint",
        timeout_s=5.0,
    )
    try:
        future = provider.submit(SemanticHintRequest(messages=(), tools=()))
        result = future.result(timeout=5.0)

        assert result == {"tool_index": 0, "arguments": {"path": "x"}}
        assert future.wait_stopped(2.0)
        assert not future.worker_alive()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)


def test_callable_provider_uses_spawn_process_and_exits_after_result():
    provider = CallableSemanticHintProvider(_callable_hint_provider)
    future = provider.submit(SemanticHintRequest(messages=(), tools=()))

    assert future.start_method == "spawn"
    assert future.result(timeout=5.0) == _hint(arguments={"path": "provider.txt"})
    assert future.wait_stopped(2.0)
    assert not future.worker_alive()


def test_callable_provider_cancel_terminates_running_worker():
    provider = CallableSemanticHintProvider(_callable_blocking_provider)
    future = provider.submit(SemanticHintRequest(messages=(), tools=()))
    pid = future.worker_pid

    assert future.wait_started(2.0)
    assert future.worker_alive()
    future.cancel()

    assert future.cancelled()
    assert future.wait_stopped(2.0)
    assert not future.worker_alive()
    assert pid not in {child.pid for child in multiprocessing.active_children()}


def test_callable_provider_immediate_prework_cancel_leaves_no_worker():
    provider = CallableSemanticHintProvider(_callable_blocking_provider)
    future = provider.submit(SemanticHintRequest(messages=(), tools=()))
    pid = future.worker_pid

    future.cancel()

    assert future.cancelled()
    assert future.wait_stopped(2.0)
    assert not future.worker_alive()
    assert pid not in {child.pid for child in multiprocessing.active_children()}


def test_process_start_failure_owner_cancel_before_poll_is_idempotent():
    context = _StartFailureContext()
    future = semantic_hints._ProcessSemanticHintFuture(
        time.sleep,
        (30.0,),
        process_context=context,
    )
    scheduler = _ManualDeadlineScheduler()
    mailbox = SemanticHintMailbox(
        future,
        timeout_s=1.0,
        deadline_scheduler=scheduler,
    )
    owner = SemanticHintMailboxOwner()
    owner.adopt(mailbox)

    owner.cancel_untransferred()
    owner.cancel_untransferred()

    assert future.done()
    assert future.cancelled()
    assert future.wait_stopped(0.0)
    assert not future.worker_alive()
    with pytest.raises(RuntimeError, match="worker start failed"):
        future.result()
    assert scheduler.handles == []
    assert context.receive_connection.close_calls == 1
    assert context.send_connection.close_calls == 1
    assert context.process.is_alive_calls == 0
    assert context.process.join_calls == 0


def test_process_start_failure_repeated_direct_cancel_never_touches_process():
    context = _StartFailureContext()
    future = semantic_hints._ProcessSemanticHintFuture(
        time.sleep,
        (30.0,),
        process_context=context,
    )

    assert future.cancel()
    assert future.cancel()

    assert future.done()
    assert future.cancelled()
    assert future.wait_started(0.0)
    assert future.wait_stopped(0.0)
    assert future.worker_pid is None
    assert context.receive_connection.close_calls == 1
    assert context.send_connection.close_calls == 1
    assert context.process.is_alive_calls == 0
    assert context.process.join_calls == 0
    assert context.process.terminate_calls == 0
    assert context.process.kill_calls == 0


def test_process_start_failure_mailbox_poll_and_deadline_cleanup_are_safe():
    context = _StartFailureContext()
    future = semantic_hints._ProcessSemanticHintFuture(
        time.sleep,
        (30.0,),
        process_context=context,
    )
    scheduler = _ManualDeadlineScheduler()
    mailbox = SemanticHintMailbox(
        future,
        timeout_s=1.0,
        deadline_scheduler=scheduler,
    )

    assert mailbox.poll().status == "provider_error"
    mailbox.cancel()
    mailbox.cancel()

    assert scheduler.handles == []
    assert future.done()
    assert not future.cancelled()
    with pytest.raises(RuntimeError, match="worker start failed"):
        future.result()
    assert future.wait_stopped(0.0)
    assert context.receive_connection.close_calls == 1
    assert context.send_connection.close_calls == 1


@pytest.mark.parametrize("cleanup", ["owner", "direct", "mailbox"])
def test_monitor_thread_start_failure_stops_child_and_is_idempotent(
    monkeypatch,
    cleanup,
):
    context = _StartedBlockingContext()
    monkeypatch.setattr(semantic_hints, "Thread", _FailingMonitorThread)

    future = semantic_hints._ProcessSemanticHintFuture(
        time.sleep,
        (30.0,),
        process_context=context,
    )

    assert future.done()
    assert future.wait_started(0.0)
    assert future.wait_stopped(0.0)
    assert not future.worker_alive()
    if cleanup == "owner":
        mailbox = SemanticHintMailbox(future, timeout_s=1.0)
        owner = SemanticHintMailboxOwner()
        owner.adopt(mailbox)
        owner.cancel_untransferred()
        owner.cancel_untransferred()
    elif cleanup == "mailbox":
        mailbox = SemanticHintMailbox(future, timeout_s=1.0)
        assert mailbox.poll().status == "provider_error"
        mailbox.cancel()
        mailbox.cancel()
        future.cancel()
        future.cancel()
    else:
        future.cancel()
        future.cancel()

    assert future.cancelled()
    assert future.wait_stopped(0.0)
    with pytest.raises(RuntimeError, match="monitor start failed") as error:
        future.result()
    assert "private data" not in str(error.value)
    assert context.process.terminate_calls == 1
    assert context.process.kill_calls == 0
    assert context.process.join_calls >= 1
    assert context.receive_connection.close_calls == 1
    assert context.send_connection.close_calls == 1


def test_monitor_thread_start_failure_reaps_real_spawn_child(monkeypatch):
    monkeypatch.setattr(semantic_hints, "Thread", _FailingMonitorThread)

    future = semantic_hints._ProcessSemanticHintFuture(time.sleep, (30.0,))
    pid = future.worker_pid

    assert pid is not None
    assert future.done()
    assert future.wait_stopped(2.0)
    assert not future.worker_alive()
    assert pid not in {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(RuntimeError, match="monitor start failed"):
        future.result()
    future.cancel()
    future.cancel()
    assert future.cancelled()
    assert future.wait_stopped(0.0)


def test_mailbox_scheduler_failure_cancels_spawn_worker_before_returning():
    future = semantic_hints._ProcessSemanticHintFuture(time.sleep, (30.0,))
    pid = future.worker_pid

    def fail_scheduler(_delay_s, _callback):
        raise RuntimeError("synthetic scheduler failure with private data")

    try:
        mailbox = SemanticHintMailbox(
            future,
            timeout_s=1.0,
            deadline_scheduler=fail_scheduler,
        )
        assert mailbox.poll().status == "provider_error"
        mailbox.cancel()
        mailbox.cancel()
    finally:
        future.cancel()

    assert future.cancelled()
    assert future.wait_stopped(2.0)
    assert not future.worker_alive()
    assert pid not in {child.pid for child in multiprocessing.active_children()}


def test_default_scheduler_cancels_partial_timer_and_spawn_worker(monkeypatch):
    _FailingStartDeadlineTimer.instances = []
    monkeypatch.setattr(semantic_hints, "Timer", _FailingStartDeadlineTimer)
    future = semantic_hints._ProcessSemanticHintFuture(time.sleep, (30.0,))
    pid = future.worker_pid

    try:
        mailbox = SemanticHintMailbox(future, timeout_s=1.0)
        assert mailbox.poll().status == "provider_error"
        mailbox.cancel()
        mailbox.cancel()
    finally:
        future.cancel()

    assert len(_FailingStartDeadlineTimer.instances) == 1
    assert _FailingStartDeadlineTimer.instances[0].cancel_calls == 1
    assert future.cancelled()
    assert future.wait_stopped(2.0)
    assert not future.worker_alive()
    assert pid not in {child.pid for child in multiprocessing.active_children()}


def test_callable_provider_deadline_stops_worker_without_polling():
    provider = CallableSemanticHintProvider(_callable_blocking_provider)
    future = provider.submit(SemanticHintRequest(messages=(), tools=()))
    mailbox = SemanticHintMailbox(future, timeout_s=0.05)
    pid = future.worker_pid

    assert future.wait_stopped(3.0)
    assert future.cancelled()
    assert mailbox.poll().status == "timeout"
    assert not future.worker_alive()
    assert pid not in {child.pid for child in multiprocessing.active_children()}


def test_repeated_callable_provider_cancellation_leaves_no_orphans():
    provider = CallableSemanticHintProvider(_callable_blocking_provider)
    worker_pids = set()
    for _ in range(5):
        future = provider.submit(SemanticHintRequest(messages=(), tools=()))
        assert future.wait_started(2.0)
        worker_pids.add(future.worker_pid)
        future.cancel()
        assert future.wait_stopped(2.0)

    active_pids = {child.pid for child in multiprocessing.active_children()}
    assert None not in worker_pids
    assert worker_pids.isdisjoint(active_pids)


@pytest.mark.parametrize("terminal", ["close", "timeout"])
def test_http_provider_cancellation_stops_running_network_worker(terminal):
    request_started = Event()
    release_response = Event()

    class BlockingHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            request_started.set()
            release_response.wait(5.0)
            try:
                response = b'{"hint":null}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), BlockingHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    now = [0.0]
    provider = HTTPSemanticHintProvider(
        f"http://127.0.0.1:{server.server_port}/hint",
        timeout_s=30.0,
    )
    future = provider.submit(SemanticHintRequest(messages=(), tools=()))
    mailbox = SemanticHintMailbox(
        future,
        timeout_s=5.0,
        clock=lambda: now[0],
    )
    source = _source(mailbox)
    try:
        assert request_started.wait(2.0)
        if terminal == "close":
            source.close()
            assert source.telemetry()["status"] == "cancelled"
        else:
            now[0] = 10.0
            assert source.propose(primary_token=1, committed_tokens=(1,)) is None
            assert source.telemetry()["status"] == "timeout"

        assert future.cancelled()
        assert future.wait_stopped(2.0)
        assert future.done()
    finally:
        release_response.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)


def test_spawn_process_boundary_cancels_synthetic_preconnect_work():
    future = semantic_hints._ProcessSemanticHintFuture(time.sleep, (30.0,))
    pid = future.worker_pid

    assert future.start_method == "spawn"
    assert future.wait_started(2.0)
    assert pid is not None
    assert future.worker_alive()

    future.cancel()

    assert future.cancelled()
    assert future.wait_stopped(2.0)
    assert future.done()
    assert not future.worker_alive()
    assert pid not in {child.pid for child in multiprocessing.active_children()}


def test_active_deadline_terminates_synthetic_preconnect_worker_without_polling():
    now = [0.0]
    scheduler = _ManualDeadlineScheduler()
    future = semantic_hints._ProcessSemanticHintFuture(time.sleep, (30.0,))
    mailbox = SemanticHintMailbox(
        future,
        timeout_s=0.25,
        clock=lambda: now[0],
        deadline_scheduler=scheduler,
    )
    pid = future.worker_pid
    assert future.wait_started(2.0)
    assert future.worker_alive()

    now[0] = 0.25
    scheduler.fire()

    assert future.wait_stopped(2.0)
    assert future.cancelled()
    assert mailbox.poll().status == "timeout"
    assert not future.worker_alive()
    assert pid not in {child.pid for child in multiprocessing.active_children()}


def test_default_deadline_terminates_spawn_worker_without_polling():
    future = semantic_hints._ProcessSemanticHintFuture(time.sleep, (30.0,))
    mailbox = SemanticHintMailbox(future, timeout_s=0.05)
    pid = future.worker_pid

    assert future.wait_stopped(3.0)
    assert future.cancelled()
    assert mailbox.poll().status == "timeout"
    assert not future.worker_alive()
    assert pid not in {child.pid for child in multiprocessing.active_children()}


def test_repeated_process_cancellation_leaves_no_orphan_workers():
    worker_pids = set()
    for _ in range(5):
        future = semantic_hints._ProcessSemanticHintFuture(time.sleep, (30.0,))
        assert future.wait_started(2.0)
        worker_pids.add(future.worker_pid)
        future.cancel()
        assert future.wait_stopped(2.0)

    active_pids = {child.pid for child in multiprocessing.active_children()}
    assert None not in worker_pids
    assert worker_pids.isdisjoint(active_pids)


def test_additional_properties_false_without_properties_rejects_arguments():
    validated, status = validate_semantic_hint(
        SemanticHint(tool_index=0, arguments={"secret": "must-not-pass"}),
        [
            {
                "type": "function",
                "function": {
                    "name": "no_args",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                    },
                },
            }
        ],
    )

    assert validated is None
    assert status == "invalid_arguments"


@pytest.mark.parametrize(
    ("schema", "accepted", "rejected"),
    [
        (
            {
                "type": "object",
                "properties": {
                    "value": {
                        "oneOf": [{"type": "number"}, {"type": "integer"}]
                    }
                },
                "required": ["value"],
                "additionalProperties": False,
            },
            {"value": 1.5},
            {"value": 1.0},
        ),
        (
            {
                "type": "object",
                "properties": {"mode": {"const": "safe"}},
                "required": ["mode"],
                "additionalProperties": False,
            },
            {"mode": "safe"},
            {"mode": "unsafe"},
        ),
        (
            {
                "type": "object",
                "properties": {"name": {"type": "string", "minLength": 3}},
                "required": ["name"],
                "additionalProperties": False,
            },
            {"name": "abc"},
            {"name": "ab"},
        ),
        (
            {
                "type": "object",
                "additionalProperties": {"type": "integer"},
            },
            {"count": 2},
            {"count": "2"},
        ),
        (
            {
                "$defs": {"name": {"type": "string", "minLength": 2}},
                "type": "object",
                "properties": {"name": {"$ref": "#/$defs/name"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            {"name": "ok"},
            {"name": "x"},
        ),
    ],
)
def test_bounded_json_schema_validator_enforces_draft_semantics(
    schema,
    accepted,
    rejected,
):
    tool = {
        "type": "function",
        "function": {"name": "validate", "parameters": schema},
    }

    validated, status = validate_semantic_hint(
        SemanticHint(tool_index=0, arguments=accepted),
        [tool],
    )
    assert validated is not None
    assert status == "ready"

    validated, status = validate_semantic_hint(
        SemanticHint(tool_index=0, arguments=rejected),
        [tool],
    )
    assert validated is None
    assert status == "invalid_arguments"


@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "object",
            "properties": {
                "nested": {
                    "$id": "https://foreign.example/resource",
                    "type": "string",
                }
            },
        },
        {
            "allOf": [
                {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                }
            ]
        },
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "contains": {"type": "string"},
                    "minContains": 0,
                }
            },
        },
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "contains": {"type": "string"},
                    "maxContains": 1,
                }
            },
        },
        {
            "definitions": {"value": {"type": "string"}},
            "type": "object",
        },
        {
            "type": "object",
            "properties": {"nested": {"$anchor": "foreign-resource"}},
        },
        {
            "type": "object",
            "properties": {"nested": {"$dynamicRef": "#foreign-resource"}},
        },
    ],
)
def test_schema_resources_and_draft_sensitive_keywords_fail_closed_recursively(
    schema,
):
    validated, status = validate_semantic_hint(
        SemanticHint(tool_index=0, arguments={}),
        [
            {
                "type": "function",
                "function": {"name": "unsafe", "parameters": schema},
            }
        ],
    )

    assert validated is None
    assert status == "invalid_tool_schema"


@pytest.mark.parametrize(
    "reference",
    [
        "#/$defs/value~2name",
        "#/$defs/value~",
        "#/%24defs/value",
        "https://foreign.example/schema#/$defs/value",
    ],
)
def test_malformed_or_foreign_schema_references_fail_closed(reference):
    schema = {
        "$defs": {"value~2name": {"type": "string"}},
        "type": "object",
        "properties": {"value": {"$ref": reference}},
    }
    validated, status = validate_semantic_hint(
        SemanticHint(tool_index=0, arguments={"value": "safe"}),
        [
            {
                "type": "function",
                "function": {"name": "unsafe", "parameters": schema},
            }
        ],
    )

    assert validated is None
    assert status == "invalid_tool_schema"


def test_local_schema_references_decode_only_valid_json_pointer_escapes():
    schema = {
        "$defs": {
            "slash/name": {"type": "string", "minLength": 2},
            "tilde~name": {"type": "integer"},
        },
        "type": "object",
        "properties": {
            "slash": {"$ref": "#/$defs/slash~1name"},
            "tilde": {"$ref": "#/$defs/tilde~0name"},
        },
        "required": ["slash", "tilde"],
        "additionalProperties": False,
    }
    tool = {
        "type": "function",
        "function": {"name": "safe", "parameters": schema},
    }

    validated, status = validate_semantic_hint(
        SemanticHint(tool_index=0, arguments={"slash": "ok", "tilde": 1}),
        [tool],
    )

    assert validated is not None
    assert status == "ready"


@pytest.mark.parametrize(
    "schema",
    [
        None,
        {"type": None},
        {"additionalProperties": None},
        {"oneOf": []},
        {"prefixItems": []},
        {"pattern": "^[a-z]+$"},
        {"unknownAssertion": True},
        {"const": float("nan")},
    ],
)
def test_unsupported_or_malformed_schema_fails_closed(schema):
    validated, status = validate_semantic_hint(
        SemanticHint(tool_index=0, arguments={}),
        [
            {
                "type": "function",
                "function": {"name": "unsafe", "parameters": schema},
            }
        ],
    )

    assert validated is None
    assert status == "invalid_tool_schema"


def test_valid_unsatisfiable_and_duplicate_enum_schemas_keep_standard_semantics():
    duplicate_enum_tool = {
        "type": "function",
        "function": {
            "name": "duplicate_enum",
            "parameters": {
                "type": "object",
                "properties": {"value": {"enum": [1, 1.0]}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
    validated, status = validate_semantic_hint(
        SemanticHint(tool_index=0, arguments={"value": 1}),
        [duplicate_enum_tool],
    )
    assert validated is not None
    assert status == "ready"

    unsatisfiable_tool = {
        "type": "function",
        "function": {
            "name": "unsatisfiable",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "minLength": 3, "maxLength": 2}
                },
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
    validated, status = validate_semantic_hint(
        SemanticHint(tool_index=0, arguments={"value": "abc"}),
        [unsatisfiable_tool],
    )
    assert validated is None
    assert status == "invalid_arguments"


def test_suffix_alignment_uses_the_paper_descending_seven_to_one_window():
    assert suffix_alignment_length((9, 1, 2, 3), (1, 2, 3, 4)) == 3
    assert suffix_alignment_length(tuple(range(10)), tuple(range(3, 11))) == 7
    assert suffix_alignment_length((4, 5), (1, 2, 3)) == 0


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return " ".join(str(int(token)) for token in tokens)


class _ScriptedModel:
    def __init__(self, vocab: int = 32) -> None:
        self.vocab = vocab
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        return hidden_states

    def _logits(self, tokens):
        rows = []
        for token in tokens:
            row = [0.0] * self.vocab
            row[(int(token) + 1) % self.vocab] = 10.0
            rows.append(row)
        return mx.array([rows], dtype=mx.float32)

    def __call__(self, input_ids, *, return_hidden=False, emit_logits=True, logits_keep=None, **_kwargs):
        tokens = [int(token) for token in np.asarray(input_ids).reshape(-1)]
        keep = len(tokens) if logits_keep is None else min(len(tokens), max(1, int(logits_keep)))
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        logits = self._logits(tokens[-keep:])
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden_states, next_token_ids, *, return_hidden=False, **_kwargs):
        del hidden_states
        tokens = [int(token) for token in np.asarray(next_token_ids).reshape(-1)]
        logits = self._logits(tokens)
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        return (logits, hidden) if return_hidden else logits


class _OffsetHistoryCache:
    def __init__(self, tokens=()) -> None:
        self.tokens = [int(token) for token in tokens]
        self.offset = len(self.tokens)

    def is_trimmable(self):
        return True

    def trim(self, count):
        trimmed = min(self.offset, max(0, int(count)))
        if trimmed:
            del self.tokens[-trimmed:]
            self.offset -= trimmed
        return trimmed

    def append(self, tokens) -> None:
        values = [int(token) for token in tokens]
        self.tokens.extend(values)
        self.offset += len(values)


class _OffsetScriptedModel(_ScriptedModel):
    def __init__(self, vocab: int = 32) -> None:
        super().__init__(vocab=vocab)
        self.mtp_caches: list[list[_OffsetHistoryCache]] = []

    def make_cache(self):
        return [_OffsetHistoryCache()]

    def make_mtp_cache(self):
        cache = [_OffsetHistoryCache()]
        self.mtp_caches.append(cache)
        return cache

    def __call__(self, input_ids, *, cache=None, **kwargs):
        if cache:
            values = [int(token) for token in np.asarray(input_ids).reshape(-1)]
            for entry in cache:
                entry.append(values)
        return super().__call__(input_ids, cache=cache, **kwargs)

    def mtp_update_cache(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        **_kwargs,
    ):
        values = [int(token) for token in np.asarray(next_token_ids).reshape(-1)]
        for entry in mtp_cache or ():
            entry.append(values)
        return hidden_states

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        return_hidden=False,
        **kwargs,
    ):
        values = [int(token) for token in np.asarray(next_token_ids).reshape(-1)]
        for entry in mtp_cache or ():
            entry.append(values)
        return super().mtp_forward(
            hidden_states,
            next_token_ids,
            return_hidden=return_hidden,
            **kwargs,
        )


def _generate(
    *,
    source=None,
    model=None,
    prompt_ids=None,
    repetition_stop=False,
):
    runtime = MTPLXRuntime(
        model=model or _ScriptedModel(),
        tokenizer=_Tokenizer(),
        model_path=Path("tiny-semantic-hint"),
        mtp_enabled=True,
        contract=MTPContract(),
    )
    return generate_mtpk(
        runtime,
        prompt_ids or [0],
        max_tokens=12,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        speculative_depth=1,
        seed=0,
        stop_token_ids=set(),
        verify_strategy="target_prefix",
        mtp_history_policy="committed",
        capture_final_state=True,
        semantic_hint_draft_source=source,
        repetition_stop=repetition_stop,
    )


@pytest.mark.skipif(mx is None, reason=_MLX_SKIP_REASON)
def test_context_copy_exclusively_owns_the_draft_arbiter(monkeypatch):
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_TARGET_PREFIX", "1")
    mailbox, _future = _mailbox(_hint())
    semantic_source = _source(mailbox, rendered=tuple(range(1, 24)))

    generated = _generate(source=semantic_source)

    assert semantic_source.telemetry()["polls"] == 0
    assert generated.stats.tool_semantic_hints["disabled_reason"] == "context_copy"


@pytest.mark.skipif(mx is None, reason=_MLX_SKIP_REASON)
def test_repetition_stop_keeps_native_mtp_and_never_polls_semantic_source():
    baseline = _generate()
    mailbox, _future = _mailbox(_hint())
    semantic_source = _source(mailbox, rendered=tuple(range(1, 24)))

    generated = _generate(source=semantic_source, repetition_stop=True)

    assert generated.tokens == baseline.tokens
    assert semantic_source.telemetry()["polls"] == 0
    assert semantic_source.telemetry()["status"] == "cancelled"
    assert generated.stats.tool_semantic_hints["disabled_reason"] == "repetition_stop"


@pytest.mark.skipif(mx is None, reason=_MLX_SKIP_REASON)
@pytest.mark.parametrize("hint_mode", ["accepted", "rejected", "pending", "timeout"])
def test_offset_bearing_session_restore_cache_matches_native_mtp(
    monkeypatch,
    hint_mode,
):
    from mtplx.generation import PromptState

    prompt = [20, 21, 22, 0]

    def restored_state(runtime, prompt_ids, **_kwargs):
        target_cache = [_OffsetHistoryCache(prompt_ids)]
        mtp_cache = [_OffsetHistoryCache(prompt_ids[1:])]
        runtime.model.mtp_caches.append(mtp_cache)
        return PromptState(
            trunk_cache=target_cache,
            logits=runtime.model._logits([prompt_ids[-1]])[:, -1, :],
            hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
            committed_mtp_cache=mtp_cache,
            token_prefix=tuple(prompt_ids),
            prompt_eval_time_s=0.0,
            mtp_history_policy="committed",
            cached_tokens=len(prompt_ids),
            cache_hit=True,
            restore_mode="clone",
        )

    monkeypatch.setattr(
        "mtplx.generation.restore_or_prefill_prompt_state",
        restored_state,
    )
    baseline = _generate(model=_OffsetScriptedModel(), prompt_ids=prompt)

    if hint_mode == "accepted":
        mailbox, _future = _mailbox(_hint())
        source = _source(mailbox, rendered=tuple(range(1, 24)))
    elif hint_mode == "rejected":
        mailbox, _future = _mailbox(_hint())
        source = _source(mailbox, rendered=(1, 9, 10, 11))
    elif hint_mode == "pending":
        mailbox, _future = _mailbox(pending=True)
        source = _source(mailbox)
    else:
        now = [0.0]
        scheduler = _ManualDeadlineScheduler()
        mailbox, _future = _mailbox(
            pending=True,
            timeout_s=0.25,
            clock=lambda: now[0],
            deadline_scheduler=scheduler,
        )
        now[0] = 0.25
        scheduler.fire()
        source = _source(mailbox)
    hinted = _generate(
        source=source,
        model=_OffsetScriptedModel(),
        prompt_ids=prompt,
    )

    baseline_cache = baseline.final_state.final_committed_mtp_cache[0]
    hinted_cache = hinted.final_state.final_committed_mtp_cache[0]
    assert baseline.tokens == hinted.tokens
    assert baseline_cache.offset > len(prompt[1:])
    assert baseline_cache.tokens == [*prompt[1:], *baseline.tokens]
    assert hinted_cache.offset == baseline_cache.offset
    assert hinted_cache.tokens == baseline_cache.tokens
