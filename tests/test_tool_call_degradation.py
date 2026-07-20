"""Silent tool-argument loss is now visible (issue #170's failure class).

Covers: faithful roundtrip of nested edit-style arguments through the output
parser, degradation counters when the model's emission carries no/unusable
arguments, the private marker never leaking to clients, the adapter tripwire
for history arguments, and the strict input path staying strict.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mtplx.server.omlx_bridge.adapter import _tool_call_for_template
from mtplx.server.omlx_bridge.tool_calling import (
    extract_tool_calls_with_thinking,
    parse_tool_calls,
)

EDIT_ARGS = {
    "filePath": "a.py",
    "edits": [
        {"search": "def f(x):", "replace": "def f(x, y):"},
        {"search": "return x", "replace": "return x + y"},
    ],
}


def _xml(payload: dict) -> str:
    return f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"


def test_nested_arguments_roundtrip_unmangled():
    text = "I'll fix that.\n" + _xml({"name": "edit", "arguments": EDIT_ARGS})
    extraction = parse_tool_calls(text, None, None)
    assert extraction.tool_calls and len(extraction.tool_calls) == 1
    call = extraction.tool_calls[0]
    assert json.loads(call["function"]["arguments"]) == EDIT_ARGS
    assert extraction.arguments_degraded == 0
    assert extraction.arguments_degraded_reasons == ()


def test_missing_arguments_flagged_not_silent():
    extraction = parse_tool_calls(_xml({"name": "edit"}), None, None)
    call = extraction.tool_calls[0]
    assert call["function"]["arguments"] == "{}"
    assert extraction.arguments_degraded == 1
    assert extraction.arguments_degraded_reasons == ("missing_arguments",)
    # The private degradation marker never reaches the client payload.
    assert all(not key.startswith("_mtplx") for key in call)


def test_explicit_empty_arguments_are_not_degraded():
    extraction = parse_tool_calls(
        _xml({"name": "edit", "arguments": {}}), None, None
    )
    assert extraction.tool_calls[0]["function"]["arguments"] == "{}"
    assert extraction.arguments_degraded == 0


def test_unparseable_bracket_arguments_flagged():
    extraction = parse_tool_calls(
        "[Calling tool: edit({not valid json})]", None, None
    )
    call = extraction.tool_calls[0]
    assert call["function"]["arguments"] == "{}"
    assert extraction.arguments_degraded == 1
    assert extraction.arguments_degraded_reasons == ("unparseable_arguments",)


def test_native_parser_exception_counted():
    calls_payload = [
        {"name": "edit", "arguments": EDIT_ARGS},
        None,  # second envelope makes the parser raise
    ]

    def parser(match: str, _tools):
        payload = calls_payload.pop(0)
        if payload is None:
            raise ValueError("boom")
        return payload

    tokenizer = SimpleNamespace(
        has_tool_calling=True,
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
        tool_parser=parser,
    )
    text = (
        "<tool_call>one</tool_call><tool_call>two</tool_call>"
    )
    extraction = parse_tool_calls(text, tokenizer, None)
    assert extraction.parser_source == "native"
    assert len(extraction.tool_calls) == 1
    assert extraction.parser_exceptions == 1
    assert json.loads(
        extraction.tool_calls[0]["function"]["arguments"]
    ) == EDIT_ARGS


def test_extract_with_thinking_carries_degradation():
    extraction = extract_tool_calls_with_thinking(
        "some reasoning",
        _xml({"name": "edit"}),
        None,
        None,
    )
    assert extraction.arguments_degraded == 1
    assert extraction.arguments_degraded_reasons == ("missing_arguments",)


# --- history (input) side ---------------------------------------------------


def test_adapter_roundtrips_valid_history_arguments():
    for arguments in (EDIT_ARGS, json.dumps(EDIT_ARGS)):
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "edit", "arguments": arguments},
        }
        normalized = _tool_call_for_template(call)
        assert normalized["function"]["arguments"] == EDIT_ARGS


def test_adapter_treats_absent_arguments_as_empty():
    call = {"function": {"name": "edit", "arguments": None}}
    assert _tool_call_for_template(call)["function"]["arguments"] == {}
    call = {"function": {"name": "edit", "arguments": ""}}
    assert _tool_call_for_template(call)["function"]["arguments"] == {}


@pytest.mark.parametrize("bad", ["{broken", "[1, 2]", '"a string"', 42])
def test_adapter_raises_on_malformed_history_arguments(bad):
    call = {"function": {"name": "edit", "arguments": bad}}
    with pytest.raises(ValueError):
        _tool_call_for_template(call)


def test_strict_input_path_rejects_malformed_arguments():
    from mtplx.server.openai import _template_tool_call

    with pytest.raises(Exception) as excinfo:
        _template_tool_call(
            {"function": {"name": "edit", "arguments": "{broken"}}
        )
    assert "arguments" in str(getattr(excinfo.value, "detail", excinfo.value))
