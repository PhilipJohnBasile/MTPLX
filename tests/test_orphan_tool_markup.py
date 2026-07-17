"""Issue #160: raw tool-call protocol markup must not reach no-tools chats.

Small models answer "look it up" prompts by emitting their trained tool
format even when the request declares no tools; the app chat then rendered
raw `<tool_call><function=web_search>...` XML (often UNCLOSED — the
reporter's transcript opened `<tool_call>` twice and never closed it).
"""

import mtplx.server.openai as srv


def test_strip_well_formed_block():
    text = "Let me check.\n<tool_call>\n{\"name\": \"web_search\"}\n</tool_call>\nDone."
    cleaned, count = srv._strip_orphan_tool_markup(text)
    assert count == 1
    assert "<tool_call>" not in cleaned
    assert "Let me check." in cleaned and "Done." in cleaned


def test_strip_unclosed_block_reporter_shape():
    # The exact #160 shape: two openers, function/parameter body, no closer.
    text = (
        "Let me search Wikipedia directly for accurate specs.\n\n"
        "<tool_call>\n<function=web_search>\n<parameter=query>\n"
        "site:wikipedia.org Apple M1 M2 memory bandwidth\n"
        "</parameter>\n</function>\n<tool_call>"
    )
    cleaned, count = srv._strip_orphan_tool_markup(text)
    assert count >= 1
    assert "<tool_call" not in cleaned
    assert "<function=" not in cleaned
    assert cleaned.startswith("Let me search Wikipedia")


def test_strip_preserves_code_fences():
    text = (
        "Here is the syntax you asked about:\n"
        "```xml\n<tool_call>\n{\"name\": \"demo\"}\n</tool_call>\n```\n"
        "That is how agents call tools."
    )
    cleaned, count = srv._strip_orphan_tool_markup(text)
    assert count == 0
    assert cleaned == text


def test_strip_no_markup_untouched():
    text = "Plain answer with a < comparison and no protocol tags."
    cleaned, count = srv._strip_orphan_tool_markup(text)
    assert count == 0
    assert cleaned == text


def test_stream_splitter_suppresses_orphan_spans():
    splitter = srv._ThinkingContentStreamSplitter(
        thinking_enabled=False,
        suppress_orphan_tool_markup=True,
    )
    chunks = []
    for piece in (
        "Checking now.\n",
        "<tool_call>\n<function=web_search>q</function>\n",
        "</tool_call>",
        "\nAll done.",
    ):
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish())
    content = "".join(text for field, text in chunks if field == "content")
    assert "<tool_call" not in content
    assert "web_search" not in content
    assert "Checking now." in content
    assert "All done." in content
    assert splitter.suppressed_tool_markup_chars > 0


def test_stream_splitter_passthrough_when_tools_active():
    splitter = srv._ThinkingContentStreamSplitter(
        thinking_enabled=False,
        suppress_orphan_tool_markup=False,
    )
    chunks = []
    for piece in ("Hi ", "<tool_call>x</tool_call>", " bye"):
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish())
    content = "".join(text for field, text in chunks if field == "content")
    assert "<tool_call>x</tool_call>" in content
