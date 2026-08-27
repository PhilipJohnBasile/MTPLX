from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "render_release_notes.py"
SPEC = importlib.util.spec_from_file_location("render_release_notes", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
render_release_notes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render_release_notes)


def test_release_notes_inline_code_has_explicit_light_and_dark_contrast() -> None:
    document = render_release_notes.release_notes_document(
        "<p><code>quantize: false</code></p>",
        "2.9.2",
    )

    assert '<meta name="color-scheme" content="light dark">' in document
    assert "@media (prefers-color-scheme: dark)" in document
    assert "--mtplx-notes-code-bg: #f2f2f4" in document
    assert "--mtplx-notes-code-text: #1d1d1f" in document
    assert "--mtplx-notes-code-bg: #343437" in document
    assert "--mtplx-notes-code-text: #f5f5f7" in document
    assert "color: var(--mtplx-notes-code-text) !important" in document
    assert "background: var(--mtplx-notes-code-bg) !important" in document
    assert "<code>quantize: false</code>" in document


def test_release_notes_title_escapes_untrusted_version_text() -> None:
    document = render_release_notes.release_notes_document("<p>Notes</p>", '2.9.3<script>')

    assert "<title>MTPLX 2.9.3&lt;script&gt;</title>" in document
    assert "<title>MTPLX 2.9.3<script></title>" not in document
