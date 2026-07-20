# SPDX-License-Identifier: Apache-2.0
"""Message normalization adapted from oMLX's OpenAI adapter.

Source inspiration: oMLX ``omlx/api/utils.py`` on origin/main, Apache-2.0.
MTPLX keeps this as a separate bridge layer so OpenCode/Qwen history is adapted
once, without prompt-injected tool contracts or transcript-stripping heuristics.
"""

from __future__ import annotations

import json
from typing import Any

_MERGEABLE_ROLES = {"user", "assistant"}
_PRESERVE_BOUNDARY_KEY = "_preserve_role_boundary"


def _message_extra(message: Any, key: str, default: Any = None) -> Any:
    value = getattr(message, key, None)
    if value is not None:
        return value
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(key, default)
    if isinstance(message, dict):
        return message.get(key, default)
    return default


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    return str(getattr(message, "role", "") or "")


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def _extract_text_from_content_list(content: list[Any]) -> str:
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            if item.get("type") == "text":
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif "text" in item:
                parts.append(str(item["text"]))
    return "".join(parts)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _extract_text_from_content_list(content)
    return str(content)


def _try_parse_json(value: Any) -> dict[str, Any]:
    """Parse history tool-call arguments to a dict, or fail loudly.

    The chat path strict-validates arguments before this adapter runs
    (_template_tool_call), so malformed input here means a caller bypassed
    that contract. This used to fall back to {} silently, which erased
    tool-call arguments from rendered history with no trace (issue #170's
    class of failure); an invariant violation must raise, not degrade.
    """
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "tool-call arguments in history are not valid JSON"
            ) from exc
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("tool-call arguments in history are not a JSON object")
    raise ValueError("tool-call arguments in history are not a JSON object")


def _drop_void_assistant_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        msg
        for msg in messages
        if not (
            msg.get("role") == "assistant"
            and not msg.get("content")
            and not msg.get("tool_calls")
            and not msg.get("tool_responses")
            and not msg.get("reasoning_content")
        )
    ]


def _consolidate_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_parts: list[str] = []
    non_system: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = _extract_text_from_content_list(content)
            if content:
                system_parts.append(str(content))
        else:
            non_system.append(msg)
    if not system_parts:
        return messages
    return [{"role": "system", "content": "\n\n".join(system_parts)}] + non_system


def _merge_consecutive_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        return messages
    merged: list[dict[str, Any]] = [dict(messages[0])]
    for msg in messages[1:]:
        prev = merged[-1]
        if (
            msg.get("role") == prev.get("role")
            and msg.get("role") in _MERGEABLE_ROLES
            and not prev.get(_PRESERVE_BOUNDARY_KEY)
            and not msg.get(_PRESERVE_BOUNDARY_KEY)
        ):
            prev_content = prev.get("content", "")
            next_content = msg.get("content", "")
            if prev_content and next_content:
                prev["content"] = f"{prev_content}\n\n{next_content}"
            elif next_content:
                prev["content"] = next_content
        else:
            merged.append(dict(msg))
    return merged


def _tool_call_for_template(tool_call: Any) -> dict[str, Any] | None:
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = str(function.get("name") or "").strip()
            arguments = _try_parse_json(function.get("arguments", {}))
        else:
            name = str(tool_call.get("name") or "").strip()
            arguments = _try_parse_json(tool_call.get("arguments", {}))
        call_id = tool_call.get("id")
    else:
        function = getattr(tool_call, "function", None)
        name = str(getattr(function, "name", "") or "").strip()
        arguments = _try_parse_json(getattr(function, "arguments", {}))
        call_id = getattr(tool_call, "id", None)
    if not name:
        return None
    item: dict[str, Any] = {
        "function": {
            "name": name,
            "arguments": arguments,
        }
    }
    if call_id:
        item["id"] = str(call_id)
    return item


def _reasoning_for_message(message: Any) -> str | None:
    reasoning = _message_extra(message, "reasoning_content")
    if reasoning is None:
        reasoning = _message_extra(message, "reasoning")
    return str(reasoning) if reasoning else None


def _apply_reasoning_reconstruction(
    *,
    role: str,
    content: Any,
    reasoning: str | None,
    native_reasoning_content: bool,
) -> tuple[str, str | None]:
    text = _content_to_text(content)
    if role != "assistant" or not reasoning:
        return text, None
    if native_reasoning_content:
        return text, reasoning
    return f"<think>\n{reasoning}\n</think>\n\n{text}", None


def normalize_messages_for_template(
    messages: list[Any],
    *,
    tokenizer: Any | None = None,
    native_reasoning_content: bool = True,
    max_tool_result_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Convert OpenAI messages into Qwen-template-safe dictionaries.

    The important contract is preservation, not steering: assistant text stays
    text, structured tool calls stay structured when the tokenizer supports
    native tool calling, tool results retain their ``tool_call_id``, and Qwen
    reasoning stays in ``reasoning_content`` when the template supports it.
    """

    del max_tool_result_tokens  # MTPLX currently does not truncate OpenCode tools here.
    native_tools = bool(getattr(tokenizer, "has_tool_calling", True))
    processed: list[dict[str, Any]] = []
    for message in messages:
        role = _message_role(message)
        if role == "developer":
            role = "system"
        content = _message_content(message)
        reasoning = _reasoning_for_message(message)
        content_text, reasoning_out = _apply_reasoning_reconstruction(
            role=role,
            content=content,
            reasoning=reasoning,
            native_reasoning_content=native_reasoning_content,
        )

        if role == "tool":
            tool_call_id = _message_extra(message, "tool_call_id", "") or ""
            if native_tools:
                processed.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call_id),
                        "content": content_text,
                    }
                )
            else:
                processed.append(
                    {
                        "role": "user",
                        "content": f"[Tool Result ({tool_call_id})]: {content_text}",
                        _PRESERVE_BOUNDARY_KEY: True,
                    }
                )
            continue

        tool_calls = _message_extra(message, "tool_calls")
        if role == "assistant" and tool_calls:
            item: dict[str, Any] = {"role": "assistant", "content": content_text}
            if reasoning_out is not None:
                item["reasoning_content"] = reasoning_out
            name = _message_extra(message, "name")
            if name:
                item["name"] = str(name)
            partial = _message_extra(message, "partial")
            if partial:
                item["partial"] = True
            if native_tools:
                normalized_calls = [
                    normalized
                    for call in tool_calls
                    if (normalized := _tool_call_for_template(call)) is not None
                ]
                if normalized_calls:
                    item["tool_calls"] = normalized_calls
            else:
                lines = [content_text] if content_text else []
                for call in tool_calls:
                    normalized = _tool_call_for_template(call)
                    if normalized is None:
                        continue
                    function = normalized["function"]
                    lines.append(
                        f"[Calling tool: {function['name']}("
                        f"{json.dumps(function['arguments'], ensure_ascii=False)})]"
                    )
                item["content"] = "\n".join(lines)
            item[_PRESERVE_BOUNDARY_KEY] = True
            processed.append(item)
            continue

        item = {"role": role, "content": content_text}
        if reasoning_out is not None:
            item["reasoning_content"] = reasoning_out
        name = _message_extra(message, "name")
        if name:
            item["name"] = str(name)
        partial = _message_extra(message, "partial")
        if partial:
            item["partial"] = True
        processed.append(item)

    normalized = _merge_consecutive_roles(
        _drop_void_assistant_messages(_consolidate_system_messages(processed))
    )
    return normalized or [{"role": "user", "content": ""}]
