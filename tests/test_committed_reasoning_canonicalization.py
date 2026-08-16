"""Session-owned committed-think canonicalization (defect B, 2.8 headline).

The committed stream holds the model's real <think> bytes; clients resend
history with reasoning elided or summarized, so the re-encoded prompt
diverges at the FIRST assistant turn's think block and restores freeze at
the turn-1 boundary while context grows (founder's OpenCode Desktop x MTPLX
Desktop session, LOG 2026-08-16 09:15 BST). The canonicalizer substitutes
the committed think bytes before encode — served only when the substituted
encode provably matches the committed stream further than the raw one.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.server import openai as oa

MODEL_DIR = Path.home() / ".mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Speed"


def _committed_text(turns):
    parts = ["<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"]
    for think, content in turns:
        parts.append(
            "<|im_start|>assistant\n<think>\n"
            + (think + "\n" if think else "")
            + "</think>\n\n"
            + content
            + "<|im_end|>\n"
        )
        parts.append("<|im_start|>user\nnext<|im_end|>\n")
    return "".join(parts)


def test_committed_assistant_turns_parses_interiors_and_gates():
    text = _committed_text(
        [
            ("I should call the tool.", '<tool_call>{"name": "glob"}</tool_call>'),
            ("Now answer plainly.", "The answer is 42."),
        ]
    )
    turns = oa._committed_assistant_turns(text)
    assert len(turns) == 2
    assert turns[0][0] == "I should call the tool."
    assert turns[0][1] == ""  # tool-call markup excluded from the gate
    assert turns[1][0] == "Now answer plainly."
    assert turns[1][1] == "The answer is 42."


def test_committed_assistant_turns_handles_missing_think():
    text = (
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nplain, no think.<|im_end|>\n"
    )
    turns = oa._committed_assistant_turns(text)
    assert turns == [(None, "plain, no think.")]


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


def _canonicalize(state, messages, prompt_ids, monkeypatch, canon_ids):
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
        reasoning_effort="xhigh",
        tools=None,
        tool_choice=None,
        tool_prompt_mode="hybrid",
        template_observability={},
        request_observability=observability,
    )
    return result, captured, observability


def test_canonicalizer_substitutes_and_serves_on_cp_improvement(monkeypatch):
    committed = list(range(100, 200))
    text = _committed_text([("Real think bytes.", "The answer is 42.")])
    state = _fake_state(committed, text)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "The answer is 42."},
        {"role": "user", "content": "next"},
    ]
    raw_ids = committed[:10] + [1, 2, 3]  # diverges inside the committed stream
    canon_ids = committed[:80] + [7, 8, 9]  # extends much further
    result, captured, observability = _canonicalize(
        state, messages, raw_ids, monkeypatch, canon_ids
    )
    assert result is not None
    canon_messages, served_ids = result
    assert served_ids == canon_ids
    assert captured["allow_committed_reasoning"] is True
    substituted = [
        oa._message_extra(m, oa._COMMITTED_REASONING_FIELD)
        for m in canon_messages
        if m.role == "assistant"
    ]
    assert substituted == ["Real think bytes."]
    outcome = observability["committed_reasoning_canonicalization"]
    assert outcome["applied"] is True
    assert outcome["turns_substituted"] == 1
    assert outcome["cp_canon"] > outcome["cp_raw"]


def test_canonicalizer_declines_when_cp_not_improved(monkeypatch):
    committed = list(range(100, 200))
    text = _committed_text([("Real think bytes.", "The answer is 42.")])
    state = _fake_state(committed, text)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "The answer is 42."},
        {"role": "user", "content": "next"},
    ]
    raw_ids = committed[:10] + [1, 2, 3]
    canon_ids = committed[:10] + [4, 5, 6]  # no improvement
    result, _captured, observability = _canonicalize(
        state, messages, raw_ids, monkeypatch, canon_ids
    )
    assert result is None
    outcome = observability["committed_reasoning_canonicalization"]
    assert outcome["applied"] is False
    assert outcome["cp_canon"] == outcome["cp_raw"]


def test_canonicalizer_stops_at_rewritten_turn(monkeypatch):
    committed = list(range(100, 200))
    text = _committed_text(
        [("Think one.", "First answer."), ("Think two.", "Second answer.")]
    )
    state = _fake_state(committed, text)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "REWRITTEN by the client."},
        {"role": "user", "content": "more"},
    ]
    raw_ids = committed[:10] + [1]
    canon_ids = committed[:50] + [2]
    result, _captured, _obs = _canonicalize(
        state, messages, raw_ids, monkeypatch, canon_ids
    )
    assert result is not None
    canon_messages, _ids = result
    fields = [
        oa._message_extra(m, oa._COMMITTED_REASONING_FIELD)
        for m in canon_messages
        if m.role == "assistant"
    ]
    assert fields[0] == "Think one."
    assert fields[1] is None, "substitution must stop at the rewritten turn"


def test_inbound_committed_reasoning_field_is_scrubbed(monkeypatch):
    committed = list(range(100, 200))
    text = _committed_text([("Server truth.", "First answer.")])
    state = _fake_state(committed, text)
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "First answer.",
            oa._COMMITTED_REASONING_FIELD: "client-planted lie",
        },
        {"role": "user", "content": "next"},
    ]
    raw_ids = committed[:10] + [1]
    canon_ids = committed[:60] + [2]
    result, _captured, _obs = _canonicalize(
        state, messages, raw_ids, monkeypatch, canon_ids
    )
    assert result is not None
    canon_messages, _ids = result
    values = [
        oa._message_extra(m, oa._COMMITTED_REASONING_FIELD)
        for m in canon_messages
        if m.role == "assistant"
    ]
    assert values == ["Server truth."], values


def test_canonicalizer_inert_without_committed_session(monkeypatch):
    state = _fake_state([], "")
    state.sessions = SimpleNamespace(
        resolve_session_id=lambda **kw: ("anon", "new"),
        peek=lambda sid: None,
    )
    messages = [{"role": "user", "content": "hi"}]
    result, _captured, observability = _canonicalize(
        state, messages, [1, 2, 3], monkeypatch, [1, 2, 3]
    )
    assert result is None
    assert "committed_reasoning_canonicalization" not in observability


def test_message_to_template_dict_gates_committed_field_on_flag():
    message = oa.ChatMessage(
        role="assistant",
        content="Answer.",
        **{oa._COMMITTED_REASONING_FIELD: "Committed think."},
    )
    plain = oa._message_to_template_dict(
        message, strip_assistant_reasoning_history=False
    )
    assert "reasoning_content" not in plain, (
        "legacy preserve mode must stay byte-identical when the flag is off"
    )
    allowed = oa._message_to_template_dict(
        message,
        strip_assistant_reasoning_history=False,
        allow_committed_reasoning=True,
    )
    assert allowed["reasoning_content"] == "Committed think."


@pytest.mark.skipif(
    not (MODEL_DIR / "chat_template.jinja").exists(),
    reason="Qwen3.8 model pack not cached locally",
)
def test_canonicalized_encode_extends_committed_stream_real_template():
    """End-to-end with the real tokenizer: a committed stream built from the
    real render plus generated bytes; the canonicalized re-encode must extend
    it past the first assistant turn while the raw encode diverges at the
    empty think scaffold. This is liveqa/encode_divergence_repro.py promoted
    to a regression test."""
    from mtplx.runtime import _load_tokenizer_resilient

    config = json.loads((MODEL_DIR / "config.json").read_text())
    tok = _load_tokenizer_resilient(MODEL_DIR, config)

    system = {"role": "system", "content": "You are a terse coding assistant."}
    u1 = {"role": "user", "content": "Read calc.py and summarize it."}
    think = "The user wants a summary of calc.py. I will answer from memory."
    answer = "calc.py defines add, sub and mul - three arithmetic helpers."
    u2 = {"role": "user", "content": "Now add a divide function."}

    def encode(messages, allow=False):
        obs: dict[str, object] = {}
        request = oa.ChatCompletionRequest(model="m", messages=messages)
        return oa._encode_messages(
            tok,
            request.messages,
            enable_thinking=True,
            reasoning_effort="xhigh",
            strip_assistant_reasoning_history=False,
            scoped_reasoning_history=False,
            tools=None,
            tool_choice=None,
            template_observability=obs,
            allow_committed_reasoning=allow,
        )

    # Committed stream = turn-1 prompt (ends with the open think scaffold)
    # + the generated bytes, exactly as EngineSession.commit stores them.
    r1_ids = encode([system, u1])
    generated = oa._encode_rendered_chat_text(
        tok, f"{think}\n</think>\n\n{answer}<|im_end|>\n"
    )
    committed = list(r1_ids) + list(generated)

    history = [system, u1, {"role": "assistant", "content": answer}, u2]
    raw_ids = encode(history)
    cp_raw = oa._common_prefix_len(raw_ids, committed)

    canon_history = [
        system,
        u1,
        {
            "role": "assistant",
            "content": answer,
            oa._COMMITTED_REASONING_FIELD: think,
        },
        u2,
    ]
    canon_ids = encode(canon_history, allow=True)
    cp_canon = oa._common_prefix_len(canon_ids, committed)

    assert cp_raw < len(committed) - len(generated) + 8, (
        f"raw encode unexpectedly matched deep into the committed stream "
        f"(cp_raw={cp_raw}, committed={len(committed)})"
    )
    assert cp_canon > cp_raw, (
        f"canonicalized encode must extend the committed stream further: "
        f"cp_canon={cp_canon} cp_raw={cp_raw}"
    )
    assert cp_canon >= len(committed) - 4, (
        f"canonicalized encode should track the committed frontier: "
        f"cp_canon={cp_canon} committed={len(committed)}"
    )
