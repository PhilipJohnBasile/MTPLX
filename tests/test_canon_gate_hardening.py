"""Committed-reasoning gate hardening (audit F11 #1/#2/#4/P2, 2.8 wave).

The 2.6-era gate compared visible text only and mapped turns positionally,
so three exactness holes remained: a tool-call branch switch inherited the
stale committed reasoning (#1), server-side transcript drops shifted the
positional mapping and substituted the WRONG turn's reasoning (#2), and
client-side launderers (OpenCode preamble strip, <mtplx_final_answer>,
inline <think>) made every gate comparison mismatch so the whole
canonicalization silently no-opped for the headline clients (#4).

Exactness law: refusing to substitute is always safe (cold prefill, correct
output); substituting wrongly is never acceptable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.server import openai as oa


def _committed_two_tool_turns() -> str:
    return (
        "<|im_start|>system\nsys<|im_end|>\n"
        "<|im_start|>user\nfix the bug<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
        "Turn A reasoning: scan the Python files.\n"
        "</think>\n\n"
        '<tool_call>\n{"name": "glob", "arguments": {"pattern": "*.py"}}\n'
        "</tool_call>"
        "<|im_end|>\n"
        "<|im_start|>user\nresult A<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
        "Turn B reasoning: now scan the docs.\n"
        "</think>\n\n"
        '<tool_call>\n{"name": "glob", "arguments": {"pattern": "*.md"}}\n'
        "</tool_call>"
        "<|im_end|>\n"
        "<|im_start|>user\nresult B<|im_end|>\n"
    )


def _fake_state(committed_ids, committed_text, session_id="s1"):
    session = SimpleNamespace(committed_token_ids=tuple(committed_ids))
    sessions = SimpleNamespace(
        resolve_session_id=lambda **kw: (session_id, "header.x-mtplx-session-id"),
        peek=lambda sid: session if sid == session_id else None,
    )
    tokenizer = SimpleNamespace(decode=lambda ids: committed_text)
    return SimpleNamespace(
        args=SimpleNamespace(strip_assistant_reasoning_history=False),
        sessions=sessions,
        runtime=SimpleNamespace(tokenizer=tokenizer),
    )


def _tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _run_gate(
    state,
    messages,
    prompt_ids,
    monkeypatch,
    canon_ids,
    **extra,
):
    captured: dict[str, object] = {}

    def _fake_encode(tokenizer, msgs, **kwargs):
        captured["messages"] = msgs
        captured["allow_committed_reasoning"] = kwargs.get(
            "allow_committed_reasoning"
        )
        return list(canon_ids)

    monkeypatch.setattr(oa, "_encode_messages", _fake_encode)
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    request = oa.ChatCompletionRequest(model="m", messages=messages)
    observability: dict[str, object] = {}
    result = oa._maybe_canonicalize_committed_reasoning(
        state,
        messages=request.messages,
        prompt_ids=list(prompt_ids),
        headers={},
        metadata={},
        request=request,
        thinking_enabled=True,
        reasoning_effort="medium",
        tools=None,
        tool_choice=None,
        tool_prompt_mode="hybrid",
        template_observability={},
        request_observability=observability,
        **extra,
    )
    return result, captured, observability


def _substituted_fields(canon_messages):
    return [
        oa._message_extra(m, oa._COMMITTED_REASONING_FIELD)
        for m in canon_messages
        if m.role == "assistant"
    ]


# --- gate test: byte-identical resend is a no-op -------------------------


def test_byte_identical_resend_is_a_noop(monkeypatch):
    committed = list(range(100, 200))
    state = _fake_state(committed, _committed_two_tool_turns())
    messages = [{"role": "user", "content": "fix the bug"}]
    # Raw encode already contained in the committed stream: the gate must
    # decline before decoding or substituting anything.
    result, _captured, observability = _run_gate(
        state, messages, committed[:40], monkeypatch, committed[:40]
    )
    assert result is None
    assert "committed_reasoning_canonicalization" not in observability


# --- gate test: differing tool_calls refuse (#1) -------------------------


def test_differing_tool_calls_refuse_substitution(monkeypatch):
    committed = list(range(100, 260))
    state = _fake_state(committed, _committed_two_tool_turns())
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the bug"},
        {
            "role": "assistant",
            "content": "",
            # Branch switch: same (empty) visible text, DIFFERENT arguments
            # from the committed glob *.py call.
            "tool_calls": [_tool_call("call_a", "glob", '{"pattern": "*.js"}')],
        },
        {"role": "tool", "content": "result A", "tool_call_id": "call_a"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_b", "glob", '{"pattern": "*.md"}')],
        },
        {"role": "tool", "content": "result B", "tool_call_id": "call_b"},
    ]
    raw_ids = committed[:10] + [1, 2, 3]
    canon_ids = committed[:120] + [7]
    result, _captured, _obs = _run_gate(
        state, messages, raw_ids, monkeypatch, canon_ids
    )
    # Turn A mismatches on tool identity; the prefix rule must also close
    # substitution for turn B even though B matches its committed twin.
    assert result is None, (
        "no substitution may survive a tool-call branch switch at turn A"
    )


def test_matching_tool_calls_substitute_both_turns(monkeypatch):
    committed = list(range(100, 260))
    state = _fake_state(committed, _committed_two_tool_turns())
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the bug"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_a", "glob", '{"pattern": "*.py"}')],
        },
        {"role": "tool", "content": "result A", "tool_call_id": "call_a"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_b", "glob", '{"pattern": "*.md"}')],
        },
        {"role": "tool", "content": "result B", "tool_call_id": "call_b"},
    ]
    raw_ids = committed[:10] + [1, 2, 3]
    canon_ids = committed[:120] + [7]
    result, captured, _obs = _run_gate(
        state, messages, raw_ids, monkeypatch, canon_ids
    )
    assert result is not None
    canon_messages, _ids = result
    assert _substituted_fields(canon_messages) == [
        "Turn A reasoning: scan the Python files.",
        "Turn B reasoning: now scan the docs.",
    ]
    assert captured["allow_committed_reasoning"] is True


# --- gate test: ordinal-drift refuse (#2) --------------------------------


def test_ordinal_drift_refuses_substitution(monkeypatch):
    committed = list(range(100, 260))
    state = _fake_state(committed, _committed_two_tool_turns())
    # Turn A is missing from the incoming transcript (canonicalization
    # dropped it); positional mapping would hand turn B turn A's reasoning.
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the bug"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_b", "glob", '{"pattern": "*.md"}')],
        },
        {"role": "tool", "content": "result B", "tool_call_id": "call_b"},
    ]
    raw_ids = committed[:10] + [1, 2]
    canon_ids = committed[:120] + [7]
    result, _captured, observability = _run_gate(
        state,
        messages,
        raw_ids,
        monkeypatch,
        canon_ids,
        transcript_stats=SimpleNamespace(skipped_aborted_assistant_messages=1),
    )
    assert result is None
    outcome = observability["committed_reasoning_canonicalization"]
    assert outcome["refused_reason"] == "transcript_assistant_turns_dropped"
    assert outcome["dropped_assistant_turns"] == 1
    assert outcome["applied"] is False


def test_ordinal_drift_without_stats_still_blocked_by_tool_identity(monkeypatch):
    """Second fence for the same hole: even with no drop telemetry, the
    positional mismatch lands on a turn whose tool identity differs, so the
    stale-substitution path stays closed."""
    committed = list(range(100, 260))
    state = _fake_state(committed, _committed_two_tool_turns())
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the bug"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_b", "glob", '{"pattern": "*.md"}')],
        },
        {"role": "tool", "content": "result B", "tool_call_id": "call_b"},
    ]
    raw_ids = committed[:10] + [1, 2]
    canon_ids = committed[:120] + [7]
    result, _captured, _obs = _run_gate(
        state, messages, raw_ids, monkeypatch, canon_ids
    )
    assert result is None, (
        "turn B must not inherit turn A's reasoning through positional drift"
    )


# --- gate test: OpenCode-stripped apply (#4) -----------------------------


def test_opencode_stripped_preamble_still_applies(monkeypatch):
    committed_text = (
        "<|im_start|>system\nsys<|im_end|>\n"
        "<|im_start|>user\nfix the bug<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
        "Preamble-turn reasoning.\n"
        "</think>\n\n"
        "Let me inspect the sources first."
        '<tool_call>\n{"name": "glob", "arguments": {"pattern": "*.py"}}\n'
        "</tool_call>"
        "<|im_end|>\n"
        "<|im_start|>user\nresult A<|im_end|>\n"
    )
    committed = list(range(100, 200))
    state = _fake_state(committed, committed_text)
    # OpenCode's canonicalized transcript strips the tool-call preamble to
    # empty content; the committed side generated it. The normalized gate
    # must view both through the same choke point and still substitute.
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the bug"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_a", "glob", '{"pattern": "*.py"}')],
        },
        {"role": "tool", "content": "result A", "tool_call_id": "call_a"},
    ]
    raw_ids = committed[:10] + [1, 2]
    canon_ids = committed[:80] + [7]
    result, _captured, _obs = _run_gate(
        state,
        messages,
        raw_ids,
        monkeypatch,
        canon_ids,
        strip_tool_call_preamble_text=True,
    )
    assert result is not None, (
        "the stripped-preamble turn must still receive its committed think"
    )
    canon_messages, _ids = result
    assert _substituted_fields(canon_messages) == ["Preamble-turn reasoning."]


def test_final_answer_marker_and_inline_think_normalize_equal():
    committed_gate = (
        "<mtplx_final_answer>The fix is a one-line change.</mtplx_final_answer>"
    )
    incoming = (
        "<think>leftover inline reasoning</think>\n"
        "The fix is a one-line change."
    )
    normalized_committed = oa._canonical_turn_gate_text(
        committed_gate, has_tool_calls=False, strip_tool_call_preamble=False
    )
    normalized_incoming = oa._canonical_turn_gate_text(
        incoming, has_tool_calls=False, strip_tool_call_preamble=False
    )
    assert normalized_committed == normalized_incoming == (
        "The fix is a one-line change."
    )


# --- gate test: kill-switch inert ----------------------------------------


def test_kill_switch_produces_zero_canonicalization(monkeypatch):
    monkeypatch.setenv("MTPLX_COMMITTED_THINK_CANONICALIZATION", "off")
    committed = list(range(100, 260))
    state = _fake_state(committed, _committed_two_tool_turns())
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the bug"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_a", "glob", '{"pattern": "*.py"}')],
        },
        {"role": "tool", "content": "result A", "tool_call_id": "call_a"},
    ]
    encode_calls: list[int] = []
    monkeypatch.setattr(
        oa,
        "_encode_messages",
        lambda *a, **kw: encode_calls.append(1) or committed[:80],
    )
    request = oa.ChatCompletionRequest(model="m", messages=messages)
    observability: dict[str, object] = {}
    result = oa._maybe_canonicalize_committed_reasoning(
        state,
        messages=request.messages,
        prompt_ids=committed[:10] + [1],
        headers={},
        metadata={},
        request=request,
        thinking_enabled=True,
        reasoning_effort="medium",
        tools=None,
        tool_choice=None,
        tool_prompt_mode="hybrid",
        template_observability={},
        request_observability=observability,
    )
    assert result is None
    assert encode_calls == [], "kill-switch must not even re-encode"
    assert "committed_reasoning_canonicalization" not in observability


# --- gate test: Gemma4 (non-Qwen family) inert ---------------------------


def test_gemma4_family_committed_stream_is_inert(monkeypatch):
    committed_text = (
        "<start_of_turn>user\nfix the bug<end_of_turn>\n"
        "<start_of_turn>model\nSome gemma answer.<end_of_turn>\n"
    )
    committed = list(range(100, 200))
    state = _fake_state(committed, committed_text)
    messages = [
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "Some gemma answer."},
        {"role": "user", "content": "next"},
    ]
    result, _captured, observability = _run_gate(
        state, messages, committed[:10] + [1], monkeypatch, committed[:80]
    )
    assert result is None, (
        "non-Qwen template markers must make the gate an explicit no-op"
    )
    assert "committed_reasoning_canonicalization" not in observability


# --- gate test: committed-reasoning scrub on gate-out --------------------


def test_client_planted_field_scrubbed_inside_substitution(monkeypatch):
    committed = list(range(100, 260))
    state = _fake_state(committed, _committed_two_tool_turns())
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the bug"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_a", "glob", '{"pattern": "*.py"}')],
            oa._COMMITTED_REASONING_FIELD: "client-planted lie",
        },
        {"role": "tool", "content": "result A", "tool_call_id": "call_a"},
    ]
    raw_ids = committed[:10] + [1]
    canon_ids = committed[:80] + [7]
    result, _captured, _obs = _run_gate(
        state, messages, raw_ids, monkeypatch, canon_ids
    )
    assert result is not None
    canon_messages, _ids = result
    values = _substituted_fields(canon_messages)
    assert values == ["Turn A reasoning: scan the Python files."], values


def test_committed_turn_tool_keys_parse_both_markup_dialects():
    json_form = (
        '<tool_call>\n{"name": "glob", "arguments": {"pattern": "*.py"}}\n'
        "</tool_call>"
    )
    xml_form = (
        "<tool_call>\n<function=glob>\n<parameter=pattern>\n*.py\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    json_keys = oa._committed_turn_tool_keys(json_form)
    xml_keys = oa._committed_turn_tool_keys(xml_form)
    incoming = oa._incoming_tool_loop_keys(
        [_tool_call("c1", "glob", '{"pattern": "*.py"}')]
    )
    assert json_keys == incoming
    assert xml_keys == incoming
    assert oa._committed_turn_tool_keys("") == []
    assert oa._committed_turn_tool_keys("<tool_call>garbage soup") is None


def test_unparseable_committed_markup_refuses(monkeypatch):
    committed_text = (
        "<|im_start|>user\nfix<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nSecret reasoning.\n</think>\n\n"
        "<tool_call>not json not xml<|im_end|>\n"
    )
    committed = list(range(100, 200))
    state = _fake_state(committed, committed_text)
    messages = [
        {"role": "user", "content": "fix"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("c1", "glob", '{"pattern": "*.py"}')],
        },
        {"role": "tool", "content": "r", "tool_call_id": "c1"},
    ]
    result, _captured, _obs = _run_gate(
        state, messages, committed[:5] + [1], monkeypatch, committed[:80]
    )
    assert result is None, "unparseable committed markup must refuse, not guess"
