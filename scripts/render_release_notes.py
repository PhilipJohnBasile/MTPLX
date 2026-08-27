#!/usr/bin/env python3
"""Render MTPLX Markdown release notes for the hosted page and Sparkle."""

from __future__ import annotations

import html
from pathlib import Path
import sys


RELEASE_NOTES_CSS = """
:root {
  color-scheme: light dark;
  --mtplx-notes-text: #1d1d1f;
  --mtplx-notes-code-bg: #f2f2f4;
  --mtplx-notes-code-text: #1d1d1f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --mtplx-notes-text: #f5f5f7;
    --mtplx-notes-code-bg: #343437;
    --mtplx-notes-code-text: #f5f5f7;
  }
}
body {
  color: var(--mtplx-notes-text);
  background: transparent;
  font-family: -apple-system, system-ui, sans-serif;
  max-width: 42em;
  margin: 2em auto;
  padding: 0 1em;
  line-height: 1.55;
}
h1, h2 { line-height: 1.2; }
code {
  color: var(--mtplx-notes-code-text) !important;
  background: var(--mtplx-notes-code-bg) !important;
  padding: 0 .25em;
  border-radius: 4px;
}
pre code {
  display: block;
  padding: .75em;
  overflow-x: auto;
}
""".strip()


def release_notes_document(body: str, version: str) -> str:
    title = html.escape(f"MTPLX {version}")
    return (
        "<!doctype html>\n"
        '<meta charset="utf-8">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"<title>{title}</title>\n"
        f"<style>\n{RELEASE_NOTES_CSS}\n</style>\n"
        f"{body}\n"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        raise SystemExit("usage: render_release_notes.py SOURCE DESTINATION VERSION")

    import markdown

    source_arg, destination_arg, version = argv[1:]
    source = Path(source_arg)
    destination = Path(destination_arg)
    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["extra"],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(release_notes_document(body, version), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
