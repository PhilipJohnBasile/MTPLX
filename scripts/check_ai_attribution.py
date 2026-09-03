#!/usr/bin/env python3
"""Fail if any AI identity appears anywhere in commit attribution.

Blocks AI coding tools from polluting the contributor graph. GitHub credits
three surfaces, and every one of them is scanned:

  1. commit author  (name + email)
  2. commit committer (name + email)
  3. trailer lines in the message body: Co-authored-by is the sneaky one,
     GitHub counts it toward contribution stats exactly like authorship.
     This is how "claude" reached rank #3 on this repo's contributor graph
     on 2026-09-01 with only 3 authored commits: ~47 more rode
     "Co-Authored-By: Claude <noreply@anthropic.com>" trailers.

Known stamps (verified against vendor docs and live incidents, 2026-09):
  - Claude Code:  "Co-Authored-By: Claude <noreply@anthropic.com>" with
                  model-name variants (Claude Opus 4.8, Claude Sonnet 4.6,
                  Claude Fable 5, ...), plus a
                  "Generated with [Claude Code](...)" marker line.
  - Cursor:       "Co-authored-by: Cursor <cursoragent@cursor.com>"
                  (CLI and cloud agents add it unconditionally).
  - VS Code:      "Co-authored-by: Copilot <copilot@github.com>"
                  (git.addAICoAuthor shipped enabled for a while in 2026).
  - Copilot coding agent: "175728472+Copilot@users.noreply.github.com".
  - Aider:        "(aider)" appended to the author name, model as co-author.
  - Codex/ChatGPT and Devin/Jules identities, belt and suspenders.

Message PROSE is deliberately not scanned: this repo legitimately discusses
"Claude Code" as a client (the Anthropic bridge), so only identity fields,
trailer lines, and line-anchored generated-with markers count.

Escape hatch for a real human whose name collides (for example a
contributor actually named Claude): put full 40-char commit shas, one per
line, in .github/ai-attribution-allowlist.

Usage:
  scripts/check_ai_attribution.py            # scan full history of HEAD
  scripts/check_ai_attribution.py --range A..B
Exit 0 clean, exit 1 with a violation table otherwise.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

EMAIL_PATTERNS = [
    r"@anthropic\.com$",
    r"@cursor\.com$",
    r"@cursor\.sh$",
    r"@openai\.com$",
    r"^copilot@github\.com$",
    r"\+copilot@users\.noreply\.github\.com$",
    r"devin-ai",
    r"google-labs-jules",
]

NAME_PATTERNS = [
    r"\bclaude\b",
    r"^cursor(\s+agent)?$",
    r"\bcursoragent\b",
    r"\bcodex\b",
    r"\bcopilot\b",
    r"\bchatgpt\b",
    r"\baider\b|\(aider\)",
    r"\bdevin\s*ai\b",
    r"jules\[bot\]",
]

TRAILER_KEYS = r"(?:co-authored-by|signed-off-by|reviewed-by|suggested-by|generated-by)"
TRAILER_LINE = re.compile(rf"^\s*{TRAILER_KEYS}\s*:\s*(.+)$", re.IGNORECASE)
MARKER_LINE = re.compile(
    r"^\s*(?:\U0001F916\s*)?generated (?:with|by) \[?(?:claude(?: code)?|codex|cursor|copilot|chatgpt)\b",
    re.IGNORECASE,
)

EMAIL_RE = [re.compile(p, re.IGNORECASE) for p in EMAIL_PATTERNS]
NAME_RE = [re.compile(p, re.IGNORECASE) for p in NAME_PATTERNS]

FIELD_SEP = "\x1f"
COMMIT_SEP = "\x1e"


def identity_hit(name: str, email: str) -> str | None:
    for rx in EMAIL_RE:
        if rx.search(email):
            return f"email matches {rx.pattern!r}"
    for rx in NAME_RE:
        if rx.search(name):
            return f"name matches {rx.pattern!r}"
    return None


def scan(range_spec: str, allow: set[str]) -> list[str]:
    fmt = FIELD_SEP.join(["%H", "%an", "%ae", "%cn", "%ce", "%B"]) + COMMIT_SEP
    out = subprocess.run(
        ["git", "log", range_spec, f"--format={fmt}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    violations: list[str] = []
    for record in out.split(COMMIT_SEP):
        record = record.strip("\n")
        if not record.strip():
            continue
        sha, an, ae, cn, ce, body = record.split(FIELD_SEP, 5)
        if sha in allow:
            continue
        hit = identity_hit(an, ae)
        if hit:
            violations.append(f"{sha[:12]}  author     {an} <{ae}>  ({hit})")
        hit = identity_hit(cn, ce)
        if hit:
            violations.append(f"{sha[:12]}  committer  {cn} <{ce}>  ({hit})")
        for line in body.splitlines():
            m = TRAILER_LINE.match(line)
            if m:
                payload = m.group(1)
                for rx in NAME_RE + EMAIL_RE:
                    if rx.search(payload):
                        violations.append(
                            f"{sha[:12]}  trailer    {line.strip()}  (matches {rx.pattern!r})"
                        )
                        break
            elif MARKER_LINE.match(line):
                violations.append(f"{sha[:12]}  marker     {line.strip()}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", default="HEAD", help="commit range (default: full HEAD history)")
    args = parser.parse_args()

    allow: set[str] = set()
    allow_file = pathlib.Path(".github/ai-attribution-allowlist")
    if allow_file.exists():
        allow = {
            line.strip()
            for line in allow_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }

    violations = scan(args.range, allow)
    if violations:
        print(f"AI attribution found in {args.range} ({len(violations)} violations):\n")
        for v in violations:
            print(f"  {v}")
        print(
            "\nNo AI identity is allowed as author, committer, or co-author on this"
            "\nrepository. Strip the attribution (amend or rebase) and push again."
            "\nA human whose name genuinely collides can be allowlisted by full sha"
            "\nin .github/ai-attribution-allowlist."
        )
        return 1
    print(f"clean: no AI attribution in {args.range}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
