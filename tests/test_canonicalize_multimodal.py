"""Consecutive-user canonicalization must never eat image parts (#327).

DSH-style agent presets send `user(image + text) -> user(runtime context)`;
the retry-pollution canonicalizer used to merge that pair into plain text
before the vision extractor ran, so the model silently answered blind.
"""

from __future__ import annotations

import mtplx.server.openai as oa

IMAGE_PART = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,Zm9v"}}


def _canonicalize(messages):
    stats = oa.AgentTranscriptCanonicalization()
    return oa._canonicalize_user_retry_pollution(messages, stats), stats


def test_image_survives_consecutive_user_merge():
    messages = [
        oa.ChatMessage(role="system", content="sys"),
        oa.ChatMessage(
            role="user",
            content=[IMAGE_PART, {"type": "text", "text": "what is in this image?"}],
        ),
        oa.ChatMessage(role="user", content="runtime-context snapshot"),
    ]
    canonical, stats = _canonicalize(messages)
    assert [m.role for m in canonical] == ["system", "user", "user"]
    assert canonical[1].content[0] == IMAGE_PART
    assert stats.merged_consecutive_user_messages == 0


def test_image_in_second_message_also_blocks_pooling():
    messages = [
        oa.ChatMessage(role="user", content="look at this"),
        oa.ChatMessage(role="user", content=[IMAGE_PART]),
    ]
    canonical, _stats = _canonicalize(messages)
    assert len(canonical) == 2
    assert canonical[1].content[0] == IMAGE_PART


def test_plain_text_consecutive_users_still_merge():
    messages = [
        oa.ChatMessage(role="user", content="first chunk of context"),
        oa.ChatMessage(role="user", content="second chunk of context"),
    ]
    canonical, stats = _canonicalize(messages)
    assert len(canonical) == 1
    assert stats.merged_consecutive_user_messages == 1
    text = oa._content_to_text(canonical[0].content)
    assert "first chunk" in text and "second chunk" in text


def test_tandem_repeat_collapse_skips_multimodal():
    repeated = "please describe the attached image, thanks a lot friend"
    messages = [
        oa.ChatMessage(
            role="user",
            content=[IMAGE_PART, {"type": "text", "text": repeated + repeated}],
        ),
    ]
    canonical, stats = _canonicalize(messages)
    assert canonical[0].content[0] == IMAGE_PART
    assert stats.collapsed_repeated_user_messages == 0
