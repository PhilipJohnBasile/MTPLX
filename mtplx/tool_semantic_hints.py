"""Target-side semantic tool hints for the bounded OoO-Spec MVP.

The sidecar returns *tool semantics* (a filtered tool index plus JSON
arguments), never target token ids.  The target server validates those
semantics, re-renders them through its own chat template, tokenizes the
rendered continuation, and remains the only authority that verifies or
commits a token.  This module intentionally knows nothing about a model,
device, or a specific sidecar implementation.

This bounded MVP has exactly one canonical target-rendered view and substitutes
at K=1 only.  It deliberately does not implement the paper's multi-view hint
bank.

No diagnostic in this module includes a prompt, tool arguments, or provider
exception text.  Callers receive only stable status labels and counters.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import (
    Future,
    InvalidStateError,
)
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import http.client
import json
import math
import multiprocessing
from multiprocessing.context import BaseContext
from multiprocessing.connection import Connection
from threading import Event, Lock, Thread, Timer, current_thread
import time
from typing import Any, Protocol
from urllib.parse import urlparse


SEMANTIC_HINT_TIMEOUT_DEFAULT_S = 1.0
SEMANTIC_HINT_TIMEOUT_MAX_S = 30.0


def normalize_semantic_hint_timeout_s(value: Any) -> float:
    """Return a finite, positive, bounded sidecar deadline."""

    if isinstance(value, bool):
        raise ValueError("semantic hint timeout must be a number")
    try:
        timeout_s = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("semantic hint timeout must be a number") from exc
    if (
        not math.isfinite(timeout_s)
        or timeout_s <= 0.0
        or timeout_s > SEMANTIC_HINT_TIMEOUT_MAX_S
    ):
        raise ValueError(
            "semantic hint timeout must be finite and in the range "
            f"(0, {SEMANTIC_HINT_TIMEOUT_MAX_S:g}] seconds"
        )
    return timeout_s


class SemanticHintFuture(Protocol):
    """Small future surface shared by injected and interruptible providers."""

    def cancel(self) -> bool: ...

    def cancelled(self) -> bool: ...

    def done(self) -> bool: ...

    def result(self, timeout: float | None = None) -> Any: ...

    def add_done_callback(
        self,
        callback: Callable[["SemanticHintFuture"], Any],
    ) -> None: ...


@dataclass(frozen=True)
class SemanticHint:
    """Sidecar output.  ``tool_index`` indexes the final server tool list."""

    tool_index: int
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.tool_index, bool) or not isinstance(self.tool_index, int):
            raise ValueError("invalid_payload")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("invalid_payload")
        try:
            canonical_arguments = json.loads(
                json.dumps(
                    self.arguments,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid_payload") from exc
        if not isinstance(canonical_arguments, dict):
            raise ValueError("invalid_payload")
        object.__setattr__(self, "arguments", canonical_arguments)

    @classmethod
    def from_payload(cls, payload: Any) -> "SemanticHint":
        if not isinstance(payload, Mapping) or set(payload) != {
            "tool_index",
            "arguments",
        }:
            raise ValueError("invalid_payload")
        tool_index = payload.get("tool_index")
        arguments = payload.get("arguments")
        if isinstance(tool_index, bool) or not isinstance(tool_index, int):
            raise ValueError("invalid_payload")
        if not isinstance(arguments, Mapping):
            raise ValueError("invalid_payload")
        # Construction performs the strict JSON round-trip and canonicalizes
        # nested mappings before the target template turns them back into text.
        return cls(tool_index=tool_index, arguments=dict(arguments))


@dataclass(frozen=True)
class SemanticHintRequest:
    """Provider-neutral request containing only server-owned request state."""

    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        try:
            payload = json.loads(
                json.dumps(
                    {"messages": self.messages, "tools": self.tools},
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid_request") from exc
        messages = payload.get("messages")
        tools = payload.get("tools")
        if (
            not isinstance(messages, list)
            or not all(isinstance(message, dict) for message in messages)
            or not isinstance(tools, list)
            or not all(isinstance(tool, dict) for tool in tools)
        ):
            raise ValueError("invalid_request")
        # The provider owns this canonical JSON snapshot. It shares no nested
        # mappings/lists with the target's filtered tools or messages.
        object.__setattr__(self, "messages", tuple(messages))
        object.__setattr__(self, "tools", tuple(tools))

    def to_payload(self) -> dict[str, Any]:
        # Return another independent JSON view so transport adapters cannot
        # mutate even the provider request object through this method.
        return json.loads(
            json.dumps(
                {"messages": self.messages, "tools": self.tools},
                allow_nan=False,
                separators=(",", ":"),
            )
        )


@dataclass(frozen=True)
class ValidatedSemanticHint:
    tool_index: int
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class SemanticHintProviderConfig:
    """Server-owned transport configuration.  Disabled is the safe default."""

    enabled: bool = False
    endpoint: str | None = None
    timeout_s: float = SEMANTIC_HINT_TIMEOUT_DEFAULT_S


class SemanticHintProvider(Protocol):
    def submit(self, request: SemanticHintRequest) -> SemanticHintFuture: ...


class DisabledSemanticHintProvider:
    """Null object used when the feature is disabled or not configured."""

    def submit(self, request: SemanticHintRequest) -> Future[Any]:
        del request
        future: Future[Any] = Future()
        future.set_result(None)
        return future


def _semantic_hint_process_entry(
    operation: Callable[..., Any],
    operation_args: tuple[Any, ...],
    result_connection: Connection,
) -> None:
    """Run one provider operation without leaking exception or payload text."""

    try:
        result = operation(*operation_args)
        result_connection.send(("result", result))
    except BaseException:
        try:
            result_connection.send(("error", None))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        result_connection.close()


class _ProcessSemanticHintFuture:
    """Spawn-isolated provider work that cancel can terminate before connect."""

    _JOIN_TIMEOUT_S = 1.0

    def __init__(
        self,
        operation: Callable[..., Any],
        operation_args: tuple[Any, ...],
        *,
        process_context: BaseContext | None = None,
    ) -> None:
        self._cancel_requested = Event()
        self._worker_started = Event()
        self._worker_stopped = Event()
        self._monitor_stopped = Event()
        self._process_lock = Lock()
        self._connection_lock = Lock()
        self._future: Future[Any] = Future()
        self._context = process_context or multiprocessing.get_context("spawn")
        receive_connection, send_connection = self._context.Pipe(duplex=False)
        self._receive_connection = receive_connection
        self._receive_connection_closed = False
        self._send_connection = send_connection
        self._send_connection_closed = False
        self._process_started = False
        self._monitor = None
        self._monitor_started = False
        self._process = self._context.Process(
            target=_semantic_hint_process_entry,
            args=(operation, operation_args, send_connection),
            daemon=True,
            name="mtplx-semantic-hint-provider",
        )
        try:
            self._process.start()
        except BaseException:
            self._close_receive_connection()
            self._close_send_connection()
            self._worker_started.set()
            self._worker_stopped.set()
            self._monitor_stopped.set()
            self._future.set_exception(RuntimeError("semantic hint worker start failed"))
            return
        self._process_started = True
        self._close_send_connection()
        self._worker_started.set()
        self._monitor = Thread(
            target=self._monitor_process,
            daemon=True,
            name="mtplx-semantic-hint-monitor",
        )
        try:
            self._monitor.start()
        except BaseException:
            # A started child must never outlive failure to create its IPC
            # monitor. Close the parent endpoint first, then stop and reap the
            # child before publishing a redacted terminal error.
            self._close_receive_connection()
            self._join_process(terminate=True)
            self._monitor_stopped.set()
            self._future.set_exception(
                RuntimeError("semantic hint monitor start failed")
            )
            return
        self._monitor_started = True

    def _close_receive_connection(self) -> None:
        with self._connection_lock:
            if self._receive_connection_closed:
                return
            self._receive_connection_closed = True
            connection = self._receive_connection
        try:
            connection.close()
        except (OSError, ValueError):
            pass

    def _close_send_connection(self) -> None:
        with self._connection_lock:
            if self._send_connection_closed:
                return
            self._send_connection_closed = True
            connection = self._send_connection
        try:
            connection.close()
        except (OSError, ValueError):
            pass

    def _finish_result(self, result: Any) -> None:
        if self._cancel_requested.is_set():
            return
        try:
            self._future.set_result(result)
        except InvalidStateError:
            pass

    def _finish_error(self) -> None:
        if self._cancel_requested.is_set():
            return
        try:
            self._future.set_exception(RuntimeError("semantic hint provider error"))
        except InvalidStateError:
            pass

    def _join_process(self, *, terminate: bool) -> None:
        with self._process_lock:
            process = self._process
            if not self._process_started:
                self._worker_stopped.set()
                return
            if terminate and process.is_alive():
                process.terminate()
            process.join(timeout=self._JOIN_TIMEOUT_S)
            if process.is_alive():
                process.kill()
                process.join(timeout=self._JOIN_TIMEOUT_S)
            if not process.is_alive():
                self._worker_stopped.set()

    def _monitor_process(self) -> None:
        try:
            while not self._cancel_requested.is_set():
                if self._receive_connection.poll(0.05):
                    break
                if not self._process.is_alive():
                    self._finish_error()
                    return
            if self._cancel_requested.is_set():
                return
            kind, payload = self._receive_connection.recv()
            if kind == "result":
                self._finish_result(payload)
            else:
                self._finish_error()
        except (EOFError, OSError, ValueError):
            self._finish_error()
        finally:
            self._close_receive_connection()
            self._join_process(terminate=True)
            self._monitor_stopped.set()

    def cancel(self) -> bool:
        self._cancel_requested.set()
        self._join_process(terminate=True)
        self._future.cancel()
        monitor = self._monitor
        if (
            self._monitor_started
            and monitor is not None
            and monitor is not current_thread()
        ):
            monitor.join(timeout=self._JOIN_TIMEOUT_S)
        return True

    def cancelled(self) -> bool:
        return self._cancel_requested.is_set()

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> Any:
        return self._future.result(timeout=timeout)

    def add_done_callback(
        self,
        callback: Callable[[SemanticHintFuture], Any],
    ) -> None:
        self._future.add_done_callback(lambda _future: callback(self))

    @property
    def start_method(self) -> str:
        return self._context.get_start_method()

    @property
    def worker_pid(self) -> int | None:
        return self._process.pid if self._process_started else None

    def worker_alive(self) -> bool:
        if not self._process_started:
            return False
        return self._process.is_alive()

    def wait_started(self, timeout: float | None = None) -> bool:
        return self._worker_started.wait(timeout)

    def wait_stopped(self, timeout: float | None = None) -> bool:
        """Test/teardown hook proving both worker and IPC monitor have exited."""

        started = time.monotonic()
        if not self._worker_stopped.wait(timeout):
            return False
        remaining = None
        if timeout is not None:
            remaining = max(0.0, timeout - (time.monotonic() - started))
        return self._monitor_stopped.wait(remaining)


class CallableSemanticHintProvider:
    """Spawn-isolated injection seam for a top-level picklable callback.

    This is deliberately process-backed just like the HTTP transport. Closing
    a request or reaching its deadline can therefore terminate provider work;
    no production callback runs in an in-process thread.
    """

    def __init__(
        self,
        callback: Callable[[SemanticHintRequest], Any],
        *,
        process_context: BaseContext | None = None,
    ) -> None:
        self._callback = callback
        self._process_context = process_context

    def submit(self, request: SemanticHintRequest) -> SemanticHintFuture:
        return _ProcessSemanticHintFuture(
            self._callback,
            (request,),
            process_context=self._process_context,
        )


_MAX_PROVIDER_RESPONSE_BYTES = 1 << 20


def _validated_semantic_hint_endpoint(endpoint: str):
    parsed = urlparse(str(endpoint))
    hostname = str(parsed.hostname or "").casefold()
    loopback_http = parsed.scheme == "http" and hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }
    if (
        not parsed.netloc
        or not hostname
        or (parsed.scheme != "https" and not loopback_http)
    ):
        raise ValueError(
            "semantic hint endpoint must use HTTPS; HTTP is allowed only "
            "for localhost, 127.0.0.1, or [::1]"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("semantic hint endpoint must not contain credentials")
    return parsed


def _http_semantic_hint_process_operation(
    endpoint: str,
    timeout_s: float,
    body: bytes,
) -> Any:
    """Perform one HTTP request inside the disposable provider process."""

    parsed = _validated_semantic_hint_endpoint(endpoint)
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(
        str(parsed.hostname),
        port=parsed.port,
        timeout=timeout_s,
    )
    path = parsed.path or "/"
    if parsed.params:
        path = f"{path};{parsed.params}"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        response_body = response.read(_MAX_PROVIDER_RESPONSE_BYTES + 1)
        if not 200 <= response.status < 300:
            raise RuntimeError("semantic hint provider HTTP error")
        if len(response_body) > _MAX_PROVIDER_RESPONSE_BYTES:
            raise RuntimeError("semantic hint provider response too large")
        decoded = json.loads(response_body.decode("utf-8"))
    finally:
        connection.close()
    if isinstance(decoded, Mapping) and set(decoded) == {"hint"}:
        return decoded["hint"]
    return decoded


class HTTPSemanticHintProvider:
    """Small HTTP-only sidecar client; it never loads a sidecar in-process."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float,
        process_context: BaseContext | None = None,
    ) -> None:
        _validated_semantic_hint_endpoint(str(endpoint))
        self._endpoint = str(endpoint)
        self._timeout_s = normalize_semantic_hint_timeout_s(timeout_s)
        self._process_context = process_context

    def submit(self, request: SemanticHintRequest) -> SemanticHintFuture:
        body = json.dumps(
            request.to_payload(),
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return _ProcessSemanticHintFuture(
            _http_semantic_hint_process_operation,
            (self._endpoint, self._timeout_s, body),
            process_context=self._process_context,
        )


def semantic_hint_provider_from_config(
    config: SemanticHintProviderConfig,
) -> SemanticHintProvider:
    if not config.enabled or not str(config.endpoint or "").strip():
        return DisabledSemanticHintProvider()
    timeout_s = normalize_semantic_hint_timeout_s(config.timeout_s)
    return HTTPSemanticHintProvider(
        str(config.endpoint),
        timeout_s=timeout_s,
    )


@dataclass(frozen=True)
class MailboxPoll:
    status: str
    hint: SemanticHint | None = None


class _DeadlineHandle(Protocol):
    def cancel(self) -> Any: ...


def _schedule_deadline_timer(
    delay_s: float,
    callback: Callable[[], None],
) -> _DeadlineHandle:
    timer = Timer(max(0.0, float(delay_s)), callback)
    timer.daemon = True
    timer.name = "mtplx-semantic-hint-deadline"
    try:
        timer.start()
    except BaseException:
        # The scheduler owns a timer until start returns it. If startup fails,
        # cancel that partial handle here so the mailbox can safely cancel the
        # already-started provider future.
        try:
            timer.cancel()
        except BaseException:
            pass
        raise
    return timer


class SemanticHintMailbox:
    """One nonblocking provider future with an autonomous bounded deadline."""

    def __init__(
        self,
        future: SemanticHintFuture,
        *,
        timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
        disabled: bool = False,
        submitted_s: float | None = None,
        deadline_scheduler: Callable[
            [float, Callable[[], None]],
            _DeadlineHandle,
        ] = _schedule_deadline_timer,
    ) -> None:
        try:
            timeout_s = normalize_semantic_hint_timeout_s(timeout_s)
        except ValueError:
            # Direct construction is a supported injection seam. Invalid
            # deadlines must not leave already-started provider work orphaned.
            try:
                future.cancel()
            except BaseException:
                pass
            raise
        self._future = future
        self._clock = clock
        self._state_lock = Lock()
        self._submitted_s = clock() if submitted_s is None else float(submitted_s)
        self._deadline_s = self._submitted_s + timeout_s
        self._completion_s: float | None = None
        self._terminal_status: str | None = "disabled" if disabled else None
        self._deadline_handle: _DeadlineHandle | None = None
        if not disabled:
            if future.done():
                future.add_done_callback(self._future_completed)
            else:
                try:
                    deadline_handle = deadline_scheduler(
                        max(0.0, self._deadline_s - clock()),
                        self._deadline_expired,
                    )
                except BaseException:
                    # Provider work has already started. Treat scheduler
                    # startup failure as a private provider error and stop the
                    # future synchronously before construction returns.
                    with self._state_lock:
                        self._terminal_status = "provider_error"
                    try:
                        future.cancel()
                    except BaseException:
                        pass
                    return
                cancel_handle = False
                with self._state_lock:
                    if self._terminal_status is None:
                        self._deadline_handle = deadline_handle
                    else:
                        cancel_handle = True
                if cancel_handle:
                    try:
                        deadline_handle.cancel()
                    except BaseException:
                        pass
                future.add_done_callback(self._future_completed)

    @property
    def completion_s(self) -> float | None:
        with self._state_lock:
            return self._completion_s

    def _cancel_deadline(self) -> None:
        handle: _DeadlineHandle | None = None
        with self._state_lock:
            handle = self._deadline_handle
            self._deadline_handle = None
        if handle is not None:
            try:
                handle.cancel()
            except BaseException:
                pass

    def _future_completed(self, _future: SemanticHintFuture) -> None:
        completed_s = self._clock()
        cancel_deadline = False
        late = False
        with self._state_lock:
            if self._completion_s is None:
                self._completion_s = completed_s
            completed_s = self._completion_s
            if self._terminal_status is not None:
                return
            if completed_s > self._deadline_s:
                self._terminal_status = "timeout"
                late = True
            else:
                cancel_deadline = True
        if cancel_deadline:
            self._cancel_deadline()
        if late:
            self._future.cancel()
            self._cancel_deadline()

    def _deadline_expired(self) -> None:
        with self._state_lock:
            self._deadline_handle = None
            if self._terminal_status is not None:
                return
            if (
                self._completion_s is not None
                and self._completion_s <= self._deadline_s
            ):
                return
            self._terminal_status = "timeout"
        self._future.cancel()

    @classmethod
    def submit(
        cls,
        provider: SemanticHintProvider,
        request: SemanticHintRequest,
        *,
        timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> "SemanticHintMailbox":
        timeout_s = normalize_semantic_hint_timeout_s(timeout_s)
        submitted_s = clock()
        disabled = isinstance(provider, DisabledSemanticHintProvider)
        return cls(
            provider.submit(request),
            timeout_s=timeout_s,
            clock=clock,
            disabled=disabled,
            submitted_s=submitted_s,
        )

    def poll(self) -> MailboxPoll:
        with self._state_lock:
            terminal_status = self._terminal_status
            completion_s = self._completion_s
        if terminal_status is not None:
            return MailboxPoll(terminal_status)
        if self._future.cancelled():
            with self._state_lock:
                if self._terminal_status is None:
                    self._terminal_status = "cancelled"
                terminal_status = self._terminal_status
            self._cancel_deadline()
            if terminal_status == "timeout":
                return MailboxPoll("timeout")
            return MailboxPoll("cancelled")
        if not self._future.done():
            if self._clock() >= self._deadline_s:
                self._deadline_expired()
                return MailboxPoll("timeout")
            return MailboxPoll("pending")
        if completion_s is None:
            self._future_completed(self._future)
            with self._state_lock:
                terminal_status = self._terminal_status
            if terminal_status is not None:
                return MailboxPoll(terminal_status)
        try:
            result = self._future.result()
        except BaseException:
            with self._state_lock:
                self._terminal_status = "provider_error"
            self._cancel_deadline()
            return MailboxPoll("provider_error")
        if result is None:
            with self._state_lock:
                self._terminal_status = "empty"
            self._cancel_deadline()
            return MailboxPoll("empty")
        try:
            hint = SemanticHint.from_payload(
                {
                    "tool_index": result.tool_index,
                    "arguments": result.arguments,
                }
                if isinstance(result, SemanticHint)
                else result
            )
        except ValueError:
            with self._state_lock:
                self._terminal_status = "invalid_payload"
            self._cancel_deadline()
            return MailboxPoll("invalid_payload")
        with self._state_lock:
            self._terminal_status = "ready"
        self._cancel_deadline()
        return MailboxPoll("ready", hint)

    def cancel(self) -> None:
        with self._state_lock:
            if self._terminal_status is not None:
                return
            self._terminal_status = "cancelled"
        self._cancel_deadline()
        self._future.cancel()


class SemanticHintMailboxOwner:
    """Idempotent request owner until generation atomically adopts a source."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._mailbox: SemanticHintMailbox | None = None
        self._source: K1SemanticHintDraftSource | None = None
        self._deferred_to_stream = False

    def adopt(self, mailbox: SemanticHintMailbox | None) -> None:
        if mailbox is None:
            return
        previous: SemanticHintMailbox | None = None
        with self._lock:
            if self._mailbox is not None and self._mailbox is not mailbox:
                previous = self._mailbox
            self._mailbox = mailbox
        if previous is not None:
            previous.cancel()

    def stage_source(self, source: K1SemanticHintDraftSource) -> None:
        with self._lock:
            if self._mailbox is not source.mailbox:
                raise RuntimeError("semantic hint source does not own the request mailbox")
            self._source = source
        source.bind_owner(self)

    def adopt_for_generation(self, source: K1SemanticHintDraftSource) -> bool:
        """Transfer ownership only at the target generation call boundary."""

        with self._lock:
            if self._source is not source or self._mailbox is not source.mailbox:
                return False
            self._source = None
            self._mailbox = None
            return True

    def release_cancelled(self, mailbox: SemanticHintMailbox) -> None:
        with self._lock:
            if self._mailbox is mailbox:
                self._mailbox = None
            if self._source is not None and self._source.mailbox is mailbox:
                self._source = None

    def defer_to_stream(self) -> None:
        with self._lock:
            self._deferred_to_stream = True

    def cancel_untransferred(self, *, force: bool = False) -> None:
        source: K1SemanticHintDraftSource | None = None
        mailbox: SemanticHintMailbox | None = None
        with self._lock:
            if self._deferred_to_stream and not force:
                return
            source = self._source
            mailbox = self._mailbox
            self._source = None
            self._mailbox = None
            self._deferred_to_stream = False
        if source is not None:
            source.close()
        elif mailbox is not None:
            mailbox.cancel()


def _tool_name(tool: Mapping[str, Any]) -> str | None:
    function = tool.get("function")
    source = function if isinstance(function, Mapping) else tool
    name = source.get("name") if isinstance(source, Mapping) else None
    return str(name).strip() if isinstance(name, str) and name.strip() else None


def _tool_schema(tool: Mapping[str, Any]) -> Any:
    function = tool.get("function")
    source = function if isinstance(function, Mapping) else tool
    if isinstance(source, Mapping) and "parameters" in source:
        return source["parameters"]
    return {}


_SCHEMA_MAX_DEPTH = 32
_SCHEMA_TYPES = {
    "null",
    "boolean",
    "object",
    "array",
    "number",
    "integer",
    "string",
}
_SCHEMA_ANNOTATION_KEYS = {
    "$comment",
    "contentEncoding",
    "contentMediaType",
    "default",
    "deprecated",
    "description",
    "examples",
    "format",
    "readOnly",
    "title",
    "writeOnly",
}
_SCHEMA_ASSERTION_KEYS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "allOf",
    "anyOf",
    "const",
    "contains",
    "dependentRequired",
    "else",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "if",
    "items",
    "maxItems",
    "maxLength",
    "maxProperties",
    "maximum",
    "minItems",
    "minLength",
    "minProperties",
    "minimum",
    "multipleOf",
    "not",
    "oneOf",
    "prefixItems",
    "properties",
    "propertyNames",
    "required",
    "then",
    "type",
    "uniqueItems",
}
_SCHEMA_KEYS = _SCHEMA_ANNOTATION_KEYS | _SCHEMA_ASSERTION_KEYS


def _is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_json_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if _is_json_number(left) and _is_json_number(right):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _local_schema_ref(root: Any, reference: str) -> Any | None:
    if reference == "#":
        return root
    if not reference.startswith("#/") or "%" in reference:
        return None
    current = root
    for raw_part in reference[2:].split("/"):
        decoded: list[str] = []
        index = 0
        while index < len(raw_part):
            character = raw_part[index]
            if character != "~":
                decoded.append(character)
                index += 1
                continue
            if index + 1 >= len(raw_part) or raw_part[index + 1] not in {"0", "1"}:
                return None
            decoded.append("~" if raw_part[index + 1] == "0" else "/")
            index += 2
        part = "".join(decoded)
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
            continue
        if isinstance(current, list):
            if not part.isdigit() or (len(part) > 1 and part.startswith("0")):
                return None
            item_index = int(part)
            if item_index >= len(current):
                return None
            current = current[item_index]
            continue
        return None
    return current


def _schema_supported(
    schema: Any,
    *,
    root: Any,
    depth: int = 0,
    ref_stack: frozenset[str] = frozenset(),
) -> bool:
    """Validate the bounded Draft 2020-12 subset; unknown assertions fail closed."""

    if depth > _SCHEMA_MAX_DEPTH:
        return False
    if isinstance(schema, bool):
        return True
    if not isinstance(schema, Mapping):
        return False
    # This evaluator has one root document and one fixed dialect. Resource
    # boundaries and dialect switches are deliberately unsupported anywhere
    # in the tree, so recursive descent rejects them before provider submit.
    if "$id" in schema or "$schema" in schema:
        return False
    if any(
        not isinstance(key, str) or key not in _SCHEMA_KEYS for key in schema
    ):
        return False

    if "$ref" in schema:
        reference = schema["$ref"]
        if not isinstance(reference, str):
            return False
        target = _local_schema_ref(root, reference)
        if target is None:
            return False
        if reference in ref_stack:
            # Recursive schemas are valid JSON Schema but outside this
            # deliberately bounded evaluator. Reject them before disclosure.
            return False
        if not _schema_supported(
            target,
            root=root,
            depth=depth + 1,
            ref_stack=ref_stack | {reference},
        ):
            return False

    if "type" in schema:
        raw_type = schema["type"]
        declared_types = [raw_type] if isinstance(raw_type, str) else raw_type
        if (
            not isinstance(declared_types, list)
            or not declared_types
            or any(item not in _SCHEMA_TYPES for item in declared_types)
            or len(set(declared_types)) != len(declared_types)
        ):
            return False

    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list):
            return False

    for key in ("allOf", "anyOf", "oneOf"):
        if key not in schema:
            continue
        children = schema[key]
        if not isinstance(children, list) or not children:
            return False
        if any(
            not _schema_supported(child, root=root, depth=depth + 1, ref_stack=ref_stack)
            for child in children
        ):
            return False

    if "prefixItems" in schema:
        children = schema["prefixItems"]
        if not isinstance(children, list) or not children or any(
            not _schema_supported(
                child,
                root=root,
                depth=depth + 1,
                ref_stack=ref_stack,
            )
            for child in children
        ):
            return False

    for key in (
        "additionalProperties",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
    ):
        if key in schema:
            child = schema[key]
            if not _schema_supported(
                child,
                root=root,
                depth=depth + 1,
                ref_stack=ref_stack,
            ):
                return False

    for key in ("$defs", "properties"):
        if key not in schema:
            continue
        children = schema[key]
        if not isinstance(children, Mapping) or any(
            not isinstance(name, str)
            or not _schema_supported(
                child,
                root=root,
                depth=depth + 1,
                ref_stack=ref_stack,
            )
            for name, child in children.items()
        ):
            return False

    if "required" in schema:
        required = schema["required"]
        if (
            not isinstance(required, list)
            or any(not isinstance(name, str) for name in required)
            or len(set(required)) != len(required)
        ):
            return False
    if "dependentRequired" in schema:
        dependent_required = schema["dependentRequired"]
        if (
            not isinstance(dependent_required, Mapping)
            or any(
                not isinstance(name, str)
                or not isinstance(dependencies, list)
                or any(not isinstance(item, str) for item in dependencies)
                or len(set(dependencies)) != len(dependencies)
                for name, dependencies in dependent_required.items()
            )
        ):
            return False

    for key in (
        "maxItems",
        "maxLength",
        "maxProperties",
        "minItems",
        "minLength",
        "minProperties",
    ):
        if key in schema and not _is_nonnegative_integer(schema[key]):
            return False
    for key in ("exclusiveMaximum", "exclusiveMinimum", "maximum", "minimum"):
        if key in schema and not _is_json_number(schema[key]):
            return False
    if "multipleOf" in schema and (
        not _is_json_number(schema["multipleOf"]) or schema["multipleOf"] <= 0
    ):
        return False
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        return False
    if "format" in schema and not isinstance(schema["format"], str):
        return False
    for key in (
        "$comment",
        "contentEncoding",
        "contentMediaType",
        "description",
        "title",
    ):
        if key in schema and not isinstance(schema[key], str):
            return False
    for key in ("deprecated", "readOnly", "writeOnly"):
        if key in schema and not isinstance(schema[key], bool):
            return False
    if "examples" in schema and not isinstance(schema["examples"], list):
        return False
    return True


def _schema_document_supported(schema: Any) -> bool:
    try:
        json.dumps(schema, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return False
    return _schema_supported(schema, root=schema)


def semantic_hint_tool_schemas_supported(
    tools: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether every final tool has a valid bounded JSON Schema."""

    for tool in tools:
        if _tool_name(tool) is None:
            return False
        schema = _tool_schema(tool)
        if not _schema_document_supported(schema):
            return False
    return True


def _value_has_type(value: Any, value_type: str) -> bool:
    if value_type == "null":
        return value is None
    if value_type == "boolean":
        return isinstance(value, bool)
    if value_type == "object":
        return isinstance(value, dict)
    if value_type == "array":
        return isinstance(value, list)
    if value_type == "number":
        return _is_json_number(value)
    if value_type == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return True
        return isinstance(value, float) and math.isfinite(value) and value.is_integer()
    if value_type == "string":
        return isinstance(value, str)
    return False


def _schema_validates(
    value: Any,
    schema: Any,
    *,
    root: Any,
    depth: int = 0,
) -> bool:
    if depth > _SCHEMA_MAX_DEPTH:
        return False
    if isinstance(schema, bool):
        return schema
    if not isinstance(schema, Mapping):
        return False

    reference = schema.get("$ref")
    if reference is not None:
        target = _local_schema_ref(root, reference)
        if target is None or not _schema_validates(
            value,
            target,
            root=root,
            depth=depth + 1,
        ):
            return False

    raw_type = schema.get("type")
    if raw_type is not None:
        declared_types = [raw_type] if isinstance(raw_type, str) else raw_type
        if not any(_value_has_type(value, item) for item in declared_types):
            return False
    if "const" in schema and not _json_equal(value, schema["const"]):
        return False
    if "enum" in schema and not any(
        _json_equal(value, candidate) for candidate in schema["enum"]
    ):
        return False

    if "allOf" in schema and not all(
        _schema_validates(value, child, root=root, depth=depth + 1)
        for child in schema["allOf"]
    ):
        return False
    if "anyOf" in schema and not any(
        _schema_validates(value, child, root=root, depth=depth + 1)
        for child in schema["anyOf"]
    ):
        return False
    if "oneOf" in schema and sum(
        _schema_validates(value, child, root=root, depth=depth + 1)
        for child in schema["oneOf"]
    ) != 1:
        return False
    if "not" in schema and _schema_validates(
        value,
        schema["not"],
        root=root,
        depth=depth + 1,
    ):
        return False
    if "if" in schema:
        condition = _schema_validates(
            value,
            schema["if"],
            root=root,
            depth=depth + 1,
        )
        branch = schema.get("then" if condition else "else")
        if branch is not None and not _schema_validates(
            value,
            branch,
            root=root,
            depth=depth + 1,
        ):
            return False

    if isinstance(value, dict):
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            return False
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            return False
        if any(name not in value for name in schema.get("required", ())):
            return False
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for name, item in value.items():
            if "propertyNames" in schema and not _schema_validates(
                name,
                schema["propertyNames"],
                root=root,
                depth=depth + 1,
            ):
                return False
            matched = False
            if name in properties:
                matched = True
                if not _schema_validates(
                    item,
                    properties[name],
                    root=root,
                    depth=depth + 1,
                ):
                    return False
            if not matched and not _schema_validates(
                item,
                additional,
                root=root,
                depth=depth + 1,
            ):
                return False
        for name, dependencies in schema.get("dependentRequired", {}).items():
            if name in value and any(item not in value for item in dependencies):
                return False

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            return False
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return False
        if schema.get("uniqueItems") and any(
            _json_equal(item, previous)
            for index, item in enumerate(value)
            for previous in value[:index]
        ):
            return False
        prefix_items = schema.get("prefixItems", ())
        for index, item in enumerate(value):
            child = (
                prefix_items[index]
                if index < len(prefix_items)
                else schema.get("items", True)
            )
            if not _schema_validates(item, child, root=root, depth=depth + 1):
                return False
        if "contains" in schema:
            if not any(
                _schema_validates(
                    item,
                    schema["contains"],
                    root=root,
                    depth=depth + 1,
                )
                for item in value
            ):
                return False

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return False
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return False

    if _is_json_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return False
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            return False
        if "multipleOf" in schema:
            try:
                if Decimal(str(value)) % Decimal(str(schema["multipleOf"])) != 0:
                    return False
            except (InvalidOperation, ValueError):
                return False
    return True


def validate_semantic_hint(
    hint: SemanticHint,
    tools: Sequence[Mapping[str, Any]],
    *,
    allowed_tool_indices: frozenset[int] | None = None,
) -> tuple[ValidatedSemanticHint | None, str]:
    """Validate against the final filtered tool list without exposing arguments."""

    if hint.tool_index < 0 or hint.tool_index >= len(tools):
        return None, "invalid_tool_index"
    if allowed_tool_indices is not None and hint.tool_index not in allowed_tool_indices:
        return None, "tool_choice"
    tool = tools[hint.tool_index]
    name = _tool_name(tool)
    if name is None:
        return None, "invalid_tool_schema"
    schema = _tool_schema(tool)
    if not _schema_document_supported(schema):
        return None, "invalid_tool_schema"
    if not _schema_validates(hint.arguments, schema, root=schema):
        return None, "invalid_arguments"
    return (
        ValidatedSemanticHint(
            tool_index=hint.tool_index,
            tool_name=name,
            arguments=hint.arguments,
        ),
        "ready",
    )


@dataclass(frozen=True)
class TargetTokenizedHintBank:
    """A target-template continuation, never a sidecar token sequence."""

    tokens: tuple[int, ...]


def target_tokenized_hint_bank(
    baseline_tokens: Sequence[int],
    rendered_tokens: Sequence[int],
) -> TargetTokenizedHintBank | None:
    """Keep only a continuation with an exact target-token prefix relation."""

    prefix_len = len(baseline_tokens)
    if (
        len(rendered_tokens) <= prefix_len
        or tuple(rendered_tokens[:prefix_len]) != tuple(baseline_tokens)
    ):
        return None
    return TargetTokenizedHintBank(
        tuple(int(token) for token in rendered_tokens[prefix_len:])
    )


def suffix_alignment_length(
    committed_tokens: Sequence[int],
    hint_tokens: Sequence[int],
    *,
    max_suffix: int = 7,
) -> int:
    """Return the paper's longest 7..1 suffix-to-hint-prefix alignment."""

    upper = min(max(0, int(max_suffix)), len(committed_tokens), len(hint_tokens))
    for length in range(upper, 0, -1):
        if tuple(committed_tokens[-length:]) == tuple(hint_tokens[:length]):
            return length
    return 0


@dataclass(frozen=True)
class SemanticHintDraftProposal:
    token: int


class K1SemanticHintDraftSource:
    """Nonblocking K=1 target-prefix substitution.

    The cursor is moved only by ``commit_target_prefix(accepted_draft_tokens)``.
    A rejected draft clears the remainder of the bank; the normal native MTP
    source owns every later cycle.  An unmatched ready bank instead remains
    available for later candidate-construction boundaries: target reasoning or
    preamble can precede the rendered tool-call prefix.  Its lifetime is
    bounded by the enclosing request, which calls ``close`` on completion or
    cancellation.
    """

    def __init__(
        self,
        mailbox: SemanticHintMailbox,
        *,
        tools: Sequence[Mapping[str, Any]],
        render: Callable[[ValidatedSemanticHint], Sequence[int] | None],
        allowed_tool_indices: frozenset[int] | None = None,
    ) -> None:
        self._mailbox = mailbox
        self._owner: SemanticHintMailboxOwner | None = None
        self._adopted_for_generation = False
        self._tools = tuple(tools)
        self._render = render
        self._allowed_tool_indices = allowed_tool_indices
        self._bank: TargetTokenizedHintBank | None = None
        self._cursor: int | None = None
        self._outstanding = False
        self._known_committed_tokens = 0
        self._proposal_committed_tokens = 0
        self._seen_ready = False
        self._closed = False
        self._status = "pending"
        self._counts: dict[str, int] = {
            "polls": 0,
            "pending": 0,
            "ready": 0,
            "invalid": 0,
            "template_mismatch": 0,
            "alignment_waits": 0,
            "suffix_matches": 0,
            "drafts": 0,
            "accepted": 0,
            "rejected": 0,
            "timeouts": 0,
            "cancelled": 0,
        }

    @property
    def mailbox(self) -> SemanticHintMailbox:
        return self._mailbox

    def bind_owner(self, owner: SemanticHintMailboxOwner) -> None:
        self._owner = owner

    def adopt_for_generation(self) -> bool:
        if self._closed:
            return False
        if self._adopted_for_generation:
            return True
        owner = self._owner
        if owner is not None and not owner.adopt_for_generation(self):
            return False
        self._owner = None
        self._adopted_for_generation = True
        return True

    def propose(
        self,
        *,
        primary_token: int,
        committed_tokens: Sequence[int],
    ) -> SemanticHintDraftProposal | None:
        del primary_token  # The committed target token is the alignment authority.
        if self._closed:
            return None
        self._counts["polls"] += 1
        if self._bank is None and not self._seen_ready:
            polled = self._mailbox.poll()
            if polled.status == "pending":
                self._counts["pending"] += 1
                self._status = "pending"
                return None
            if polled.status == "timeout":
                self._counts["timeouts"] += 1
                self._status = "timeout"
                self._seen_ready = True
                return None
            if polled.status == "cancelled":
                self._counts["cancelled"] += 1
                self._status = "cancelled"
                self._seen_ready = True
                return None
            if polled.status != "ready" or polled.hint is None:
                self._counts["invalid"] += 1
                self._status = polled.status
                self._seen_ready = True
                return None
            validated, status = validate_semantic_hint(
                polled.hint,
                self._tools,
                allowed_tool_indices=self._allowed_tool_indices,
            )
            if validated is None:
                self._counts["invalid"] += 1
                self._status = status
                self._seen_ready = True
                return None
            rendered = self._render(validated)
            self._seen_ready = True
            if not rendered:
                self._counts["template_mismatch"] += 1
                self._status = "template_mismatch"
                return None
            self._bank = TargetTokenizedHintBank(tuple(int(token) for token in rendered))
            self._counts["ready"] += 1
            self._status = "ready"
        if self._bank is None:
            return None
        if self._cursor is None:
            aligned = suffix_alignment_length(committed_tokens, self._bank.tokens)
            if aligned <= 0:
                # A ready semantic result can arrive while the target emits
                # reasoning or another preamble. Keep the one canonical view
                # for the next target boundary; native MTP handles this cycle.
                self._counts["alignment_waits"] += 1
                self._status = "late_pending_alignment"
                return None
            self._cursor = aligned
            self._known_committed_tokens = len(committed_tokens)
            self._counts["suffix_matches"] += 1
        elif len(committed_tokens) < self._known_committed_tokens:
            self._bank = None
            self._cursor = None
            self._status = "late_unaligned"
            return None
        else:
            # A fully accepted K=1 cycle can also commit a target-only bonus
            # token.  It is not a sidecar acceptance, but it is authoritative
            # target progress, so consume it only when it exactly continues
            # the rendered bank.  This keeps the cursor target-owned without
            # accidentally proposing a token the target already committed.
            for token in committed_tokens[self._known_committed_tokens:]:
                if self._cursor >= len(self._bank.tokens):
                    self._status = "consumed"
                    return None
                if int(token) != int(self._bank.tokens[self._cursor]):
                    self._bank = None
                    self._cursor = None
                    self._status = "late_unaligned"
                    return None
                self._cursor += 1
            self._known_committed_tokens = len(committed_tokens)
        if self._cursor >= len(self._bank.tokens):
            self._status = "consumed"
            return None
        self._outstanding = True
        self._proposal_committed_tokens = len(committed_tokens)
        self._counts["drafts"] += 1
        self._status = "drafted"
        return SemanticHintDraftProposal(int(self._bank.tokens[self._cursor]))

    def commit_target_prefix(self, accepted_draft_tokens: int) -> None:
        if not self._outstanding or self._bank is None or self._cursor is None:
            return
        self._outstanding = False
        if int(accepted_draft_tokens) <= 0:
            self._counts["rejected"] += 1
            self._bank = None
            self._cursor = None
            self._status = "rejected"
            return
        self._cursor += 1
        self._counts["accepted"] += 1
        self._known_committed_tokens = (
            self._proposal_committed_tokens + int(accepted_draft_tokens)
        )
        self._status = "consumed" if self._cursor >= len(self._bank.tokens) else "ready"

    def telemetry(self) -> dict[str, int | str | bool]:
        return {
            "enabled": True,
            "status": self._status,
            **{name: int(value) for name, value in self._counts.items()},
        }

    def close(self) -> None:
        if self._closed:
            return
        self._mailbox.cancel()
        self._bank = None
        self._cursor = None
        self._outstanding = False
        self._closed = True
        self._status = "cancelled"
