from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server import responses as responses_api
from mtplx.server.openai import create_app
from tests.test_server_openai import (
    CaptureTokenizer,
    _fake_generation,
    _fake_state,
    _fake_streaming_generation,
)


FIXTURES = Path(__file__).parent / "fixtures" / "responses"
DEFAULT_CODEX_FIXTURE = FIXTURES / "codex_0.146.0_responses_default.sanitized.json"
XHIGH_CODEX_FIXTURE = FIXTURES / "codex_0.146.0_responses_xhigh.sanitized.json"


def _fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _without_hosted_tools(body: dict) -> dict:
    derived = copy.deepcopy(body)
    derived["tools"] = [
        tool
        for tool in derived.get("tools", [])
        if tool.get("type") not in {"web_search", "tool_search"}
    ]
    return derived


def _responses_events(response_text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for frame in response_text.split("\n\n"):
        event = None
        data = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        if event is not None and data is not None:
            events.append((event, data))
    return events


def test_representative_codex_default_converts_client_tools_then_rejects_web_only():
    body = _fixture(DEFAULT_CODEX_FIXTURE)

    tools = responses_api.convert_tools(body["tools"])

    assert len(tools.chat_tools) == 3
    assert [tool["function"]["name"] for tool in tools.chat_tools[:1]] == ["view_image"]
    assert {
        (identity.namespace, identity.name)
        for identity in tools.namespace_functions.values()
    } == {
        ("multi_agent_v1", "close_agent"),
        ("multi_agent_v1", "wait_agent"),
    }
    assert tools.hosted_tool_types == ["web_search"]

    response = TestClient(create_app(_fake_state())).post("/v1/responses", json=body)

    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "hosted tool type(s) unavailable on local MTPLX: web_search" in message
    assert "namespace" not in message.split(": web_search", 1)[0]


def test_representative_codex_without_hosted_tool_reaches_inference_and_sdk_union(
    monkeypatch,
):
    openai_sdk = pytest.importorskip("openai", minversion="2.52.0")
    from pydantic import TypeAdapter
    from openai.types.responses import Response, ResponseStreamEvent

    assert openai_sdk.__version__ == "2.52.0"
    body = _without_hosted_tools(_fixture(DEFAULT_CODEX_FIXTURE))
    state = _fake_state()
    state.args.stats_footer = False
    state.args.enable_thinking = False
    state.runtime.tokenizer = CaptureTokenizer()
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation("OK"))

    response = TestClient(create_app(state)).post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json=body,
    )

    assert response.status_code == 200
    assert "[DONE]" not in response.text
    events = _responses_events(response.text)
    assert [name for name, _ in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    event_adapter = TypeAdapter(ResponseStreamEvent)
    for _name, payload in events:
        event_adapter.validate_python(payload)
    final = Response.model_validate(events[-1][1]["response"])
    assert final.output_text == "OK"
    assert final.tools[1].type == "namespace"
    assert final.reasoning.summary == "auto"
    rendered_messages, _template_kwargs = state.runtime.tokenizer.calls[0]
    assert [message["role"] for message in rendered_messages] == ["system", "user"]
    assert rendered_messages[0]["content"].startswith(body["instructions"])
    assert "<skills_instructions>" in rendered_messages[0]["content"]
    assert "# AGENTS.md instructions" in rendered_messages[-1]["content"]
    assert rendered_messages[-1]["content"].endswith("Reply with exactly OK.")


def test_representative_codex_xhigh_downgrades_to_local_high_and_records_it(
    monkeypatch,
):
    body = _without_hosted_tools(_fixture(XHIGH_CODEX_FIXTURE))
    conversion = responses_api.translate_request(
        responses_api.ResponsesRequest.model_validate(body)
    )

    assert conversion.chat["reasoning_effort"] == "high"
    assert conversion.chat["metadata"]["responses_reasoning_effort_requested"] == (
        "xhigh"
    )
    assert conversion.chat["metadata"]["responses_reasoning_effort_effective"] == (
        "high"
    )
    assert conversion.chat["metadata"]["responses_reasoning_effort_downgraded"] is True

    state = _fake_state()
    state.args.stats_footer = False
    state.runtime.tokenizer = CaptureTokenizer()
    seen: dict = {}
    fake_stream = _fake_streaming_generation("OK")

    def capture_generation(*args, **kwargs):
        seen.update(kwargs)
        return fake_stream(*args, **kwargs)

    monkeypatch.setattr(openai, "_run_generation", capture_generation)
    response = TestClient(create_app(state)).post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json=body,
    )

    assert response.status_code == 200
    final = _responses_events(response.text)[-1][1]["response"]
    assert final["status"] == "completed"
    assert final["reasoning"] == {"effort": "xhigh", "summary": "auto"}
    observability = seen["request_observability"]
    assert observability["request_responses_reasoning_effort_requested"] == "xhigh"
    assert observability["request_responses_reasoning_effort_effective"] == "high"
    assert observability["request_responses_reasoning_effort_downgraded"] is True


def test_namespace_flat_names_are_collision_safe_and_reversible():
    tools = [
        {
            "type": "function",
            "name": "foo__bar",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "namespace",
            "name": "foo",
            "description": "Namespaced foo tools.",
            "tools": [
                {
                    "type": "function",
                    "name": "bar",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
    ]

    converted = responses_api.convert_tools(tools)
    flat_name = converted.namespace_flat_names[("foo", "bar")]

    assert flat_name != "foo__bar"
    assert len({tool["function"]["name"] for tool in converted.chat_tools}) == 2
    assert converted.namespace_functions[flat_name] == responses_api.NamespaceFunction(
        namespace="foo", name="bar"
    )
    output = responses_api.output_from_chat(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_nested",
                                "type": "function",
                                "function": {
                                    "name": flat_name,
                                    "arguments": '{"value":1}',
                                },
                            }
                        ],
                    },
                }
            ]
        },
        namespace_functions=converted.namespace_functions,
    )
    assert output[0]["type"] == "function_call"
    assert output[0]["name"] == "bar"
    assert output[0]["namespace"] == "foo"
    assert "foo__bar" not in json.dumps(output)


def test_parallel_namespace_continuation_uses_original_names_and_matching_outputs():
    request = responses_api.ResponsesRequest.model_validate(
        {
            "tools": [
                {
                    "type": "namespace",
                    "name": "agents",
                    "description": "Agent lifecycle tools.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "wait",
                            "parameters": {"type": "object", "properties": {}},
                        },
                        {
                            "type": "function",
                            "name": "close",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    ],
                }
            ],
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_wait",
                    "namespace": "agents",
                    "name": "wait",
                    "arguments": {"target": "a"},
                },
                {
                    "type": "function_call",
                    "call_id": "call_close",
                    "namespace": "agents",
                    "name": "close",
                    "arguments": {"target": "b"},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_wait",
                    "output": "done",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_close",
                    "output": "closed",
                },
            ],
        }
    )

    converted = responses_api.translate_request(request)
    messages = converted.chat["messages"]

    assert [message["role"] for message in messages] == ["assistant", "tool", "tool"]
    assert [call["id"] for call in messages[0]["tool_calls"]] == [
        "call_wait",
        "call_close",
    ]
    assert [
        converted.namespace_functions[call["function"]["name"]]
        for call in messages[0]["tool_calls"]
    ] == [
        responses_api.NamespaceFunction("agents", "wait"),
        responses_api.NamespaceFunction("agents", "close"),
    ]
    assert [message["tool_call_id"] for message in messages[1:]] == [
        "call_wait",
        "call_close",
    ]


def test_namespace_stream_returns_official_namespace_qualified_output_item(
    monkeypatch,
):
    openai_sdk = pytest.importorskip("openai", minversion="2.52.0")
    from pydantic import TypeAdapter
    from openai.types.responses import (
        Response,
        ResponseFunctionCallArgumentsDoneEvent,
        ResponseFunctionToolCall,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseStreamEvent,
    )

    assert openai_sdk.__version__ == "2.52.0"
    tools = [
        {
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Client-executed multi-agent tools.",
            "tools": [
                {
                    "type": "function",
                    "name": "close_agent",
                    "parameters": {
                        "type": "object",
                        "properties": {"target": {"type": "string"}},
                    },
                }
            ],
        }
    ]
    flat_name = responses_api.convert_tools(tools).namespace_flat_names[
        ("multi_agent_v1", "close_agent")
    ]
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    monkeypatch.setattr(openai, "_encode_messages", lambda *_a, **_k: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            f"<tool_call>\n<function={flat_name}>\n"
            "<parameter=target>agent-1</parameter>\n</function>\n</tool_call>"
        ),
    )

    response = TestClient(create_app(state)).post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={"input": "Close it.", "stream": True, "tools": tools},
    )

    assert response.status_code == 200
    events = _responses_events(response.text)
    event_adapter = TypeAdapter(ResponseStreamEvent)
    for _name, payload in events:
        event_adapter.validate_python(payload)
    added_payload = next(
        payload for name, payload in events if name == "response.output_item.added"
    )
    done_payload = next(
        payload for name, payload in events if name == "response.output_item.done"
    )
    arguments_done = next(
        payload
        for name, payload in events
        if name == "response.function_call_arguments.done"
    )
    ResponseOutputItemAddedEvent.model_validate(added_payload)
    ResponseOutputItemDoneEvent.model_validate(done_payload)
    ResponseFunctionCallArgumentsDoneEvent.model_validate(arguments_done)
    call = ResponseFunctionToolCall.model_validate(done_payload["item"])
    assert call.name == "close_agent"
    assert call.namespace == "multi_agent_v1"
    assert arguments_done["name"] == "close_agent"
    final = Response.model_validate(events[-1][1]["response"])
    assert final.output[0].namespace == "multi_agent_v1"
    assert flat_name not in response.text


def test_namespace_nonstream_restores_official_namespace_qualified_output_item(
    monkeypatch,
):
    openai_sdk = pytest.importorskip("openai", minversion="2.52.0")
    from openai.types.responses import Response

    assert openai_sdk.__version__ == "2.52.0"
    tools = [
        {
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Client-executed multi-agent tools.",
            "tools": [
                {
                    "type": "function",
                    "name": "wait_agent",
                    "parameters": {
                        "type": "object",
                        "properties": {"targets": {"type": "array"}},
                    },
                }
            ],
        }
    ]
    flat_name = responses_api.convert_tools(tools).namespace_flat_names[
        ("multi_agent_v1", "wait_agent")
    ]
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_a, **_k: _fake_generation(
            f"<tool_call>\n<function={flat_name}>\n"
            '<parameter=targets>["agent-1"]</parameter>\n'
            "</function>\n</tool_call>"
        ),
    )

    response = TestClient(create_app(state)).post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={"input": "Wait.", "stream": False, "tools": tools},
    )

    assert response.status_code == 200
    parsed = Response.model_validate(response.json())
    call = parsed.output[0]
    assert call.type == "function_call"
    assert call.name == "wait_agent"
    assert call.namespace == "multi_agent_v1"
    assert flat_name not in response.text


def test_custom_tool_stream_preserves_freeform_responses_events(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    monkeypatch.setattr(openai, "_encode_messages", lambda *_a, **_k: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "<tool_call>\n<function=shell>\n"
            "<parameter=input>git status</parameter>\n</function>\n</tool_call>"
        ),
    )

    response = TestClient(create_app(state)).post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "input": "Check status.",
            "stream": True,
            "tools": [{"type": "custom", "name": "shell"}],
        },
    )

    assert response.status_code == 200
    events = _responses_events(response.text)
    names = [name for name, _ in events]
    assert "response.custom_tool_call_input.delta" in names
    assert "response.custom_tool_call_input.done" in names
    assert "response.function_call_arguments.delta" not in names
    call = events[-1][1]["response"]["output"][0]
    assert call["type"] == "custom_tool_call"
    assert call["name"] == "shell"
    assert call["input"] == "git status"


def test_nonstream_text_response_reuses_chat_generation(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    monkeypatch.setattr(
        openai, "_run_generation", lambda *_a, **_k: _fake_generation("Hello")
    )

    response = TestClient(create_app(state)).post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "instructions": "Be concise.",
            "input": [{"role": "user", "content": "Say hi."}],
            "max_output_tokens": 12,
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["output"][0]["content"][0]["text"] == "Hello"
    messages, _kwargs = state.runtime.tokenizer.calls[0]
    assert messages[0] == {"role": "system", "content": "Be concise."}


def test_namespace_function_tool_choice_is_rejected_until_schema_can_qualify_it():
    base = {
        "input": "Wait.",
        "tools": [
            {
                "type": "namespace",
                "name": "agents",
                "description": "Agent lifecycle tools.",
                "tools": [
                    {
                        "type": "function",
                        "name": "wait",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        ],
    }
    client = TestClient(create_app(_fake_state()))

    unqualified = client.post(
        "/v1/responses",
        json={**base, "tool_choice": {"type": "function", "name": "wait"}},
    )
    invented_qualifier = client.post(
        "/v1/responses",
        json={
            **base,
            "tool_choice": {
                "type": "function",
                "namespace": "agents",
                "name": "wait",
            },
        },
    )

    assert unqualified.status_code == 400
    assert (
        "current Responses selector has no namespace field"
        in (unqualified.json()["error"]["message"])
    )
    assert invented_qualifier.status_code == 400
    assert (
        "schema has only type and name"
        in (invented_qualifier.json()["error"]["message"])
    )


def test_json_schema_format_reaches_existing_constrained_decode(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    seen: dict = {}

    def fake_run_generation(*_args, **kwargs):
        seen.update(kwargs)
        return _fake_generation('{"ok":true}')

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    response = TestClient(create_app(state)).post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "input": "Return JSON.",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                }
            },
        },
    )

    assert response.status_code == 200
    assert seen["constraint_spec"].source_type == "json_schema"
    assert response.json()["text"]["format"]["name"] == "answer"


def test_stream_length_and_failure_have_responses_terminal_events(monkeypatch):
    state = _fake_state()
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_a, **_k: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation("cut", finish_reason="length"),
    )
    incomplete = client.post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={"input": "long", "stream": True, "max_output_tokens": 2},
    )
    incomplete_events = _responses_events(incomplete.text)
    assert incomplete_events[-1][0] == "response.incomplete"
    assert incomplete_events[-1][1]["response"]["incomplete_details"] == {
        "reason": "max_output_tokens"
    }

    def fail_generation(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(openai, "_run_generation", fail_generation)
    failed = client.post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={"input": "fail", "stream": True},
    )
    failed_events = _responses_events(failed.text)
    assert [name for name, _ in failed_events[-2:]] == ["error", "response.failed"]
    assert failed_events[-1][1]["response"]["status"] == "failed"


def _duplicate_tool(tool_type: str) -> dict:
    if tool_type == "custom":
        return {"type": "custom", "name": "duplicate"}
    if tool_type == "namespace":
        return {
            "type": "namespace",
            "name": "duplicate",
            "description": "Duplicate namespace.",
            "tools": [
                {
                    "type": "function",
                    "name": "nested",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
    return {
        "type": "function",
        "name": "duplicate",
        "parameters": {"type": "object", "properties": {}},
    }


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize(
    ("first_type", "second_type"),
    [
        ("function", "custom"),
        ("function", "function"),
        ("custom", "custom"),
        ("function", "namespace"),
        ("namespace", "namespace"),
    ],
)
def test_duplicate_top_level_tool_names_fail_before_inference_or_reclassification(
    monkeypatch,
    stream,
    first_type,
    second_type,
):
    inference_called = False

    def unexpected_generation(*_args, **_kwargs):
        nonlocal inference_called
        inference_called = True
        return _fake_generation("must not run")

    monkeypatch.setattr(openai, "_run_generation", unexpected_generation)
    response = TestClient(create_app(_fake_state())).post(
        "/v1/responses",
        json={
            "input": "Use the duplicate tool.",
            "stream": stream,
            "tools": [
                _duplicate_tool(first_type),
                _duplicate_tool(second_type),
            ],
        },
    )

    assert response.status_code == 400
    assert inference_called is False
    message = response.json()["error"]["message"]
    assert "tools[1].name='duplicate' duplicates tools[0].name" in message
    assert f"({second_type} vs {first_type})" in message
    assert "top-level tool names must be unique" in message


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"input": "hi", "previous_response_id": "resp_old"}, "previous_response_id"),
        ({"input": "hi", "background": True}, "background=true"),
        ({"input": "hi", "store": True}, "store=true"),
        ({"input": "hi", "conversation": "conv_123"}, "unsupported field(s)"),
        (
            {"input": "hi", "tools": [{"type": "tool_search"}]},
            "hosted tool type(s) unavailable",
        ),
        (
            {"input": "hi", "max_output_tokens": 4, "max_tokens": 5},
            "disagree",
        ),
        (
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_old",
                        "output": "x",
                    }
                ]
            },
            "matching function_call",
        ),
    ],
)
def test_unsupported_or_stateful_requests_fail_precisely(body, message):
    response = TestClient(create_app(_fake_state())).post("/v1/responses", json=body)

    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
