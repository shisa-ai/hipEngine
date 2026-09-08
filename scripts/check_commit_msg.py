#!/usr/bin/env python3
"""Reject commit messages with bylines, agent attribution, or session IDs.

hipEngine policy (AGENTS.md "Commit Messages"): no bylines, no
Co-authored-by, no agent attribution, no generated-by footers. Session
URLs (claude.ai/code/session_...) additionally leak a live session
identifier and are treated as a security issue, not just style.

Usage:
    python3 scripts/check_commit_msg.py <commit-message-file>
    git log -1 --format=%B | python3 scripts/check_commit_msg.py -

Exit 0 when clean, 1 with diagnostics when a banned pattern matches.
Installed as .git/hooks/commit-msg by scripts/install_commit_msg_hook.py.
"""

from __future__ import annotations

import re
import sys

# (label, pattern, flags). Keep each pattern tight; every entry must name
# a trailer/attribution form, never ordinary prose.
BANNED: list[tuple[str, str, int]] = [
    (
        "Co-Authored-By trailer (no bylines, AGENTS.md)",
        r"(?im)^Co-Authored-By\s*:",
        0,
    ),
    (
        "Anthropic noreply attribution address",
        r"(?i)noreply@anthropic\.com",
        0,
    ),
    (
        "Claude-Session trailer (leaks a live session id)",
        r"(?im)^Claude-Session\s*:",
        0,
    ),
    (
        "claude.ai session URL (leaks a live session id)",
        r"(?i)claude\.ai/code/session_",
        0,
    ),
    (
        "generated-by footer (no generated-by footers, AGENTS.md)",
        r"(?i)(?:\U0001f916\s*)?generated[ -]with\s+claude",
        0,
    ),
    (
        "Generated-By trailer (no generated-by footers, AGENTS.md)",
        r"(?im)^Generated-By\s*:",
        0,
    ),
]


def check(text: str) -> list[str]:
    """Return a list of violation labels found in the message."""
    hits: list[str] = []
    for label, pattern, flags in BANNED:
        if re.search(pattern, text, flags):
            hits.append(label)
    return hits


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[1] == "-":
        text = sys.stdin.read()
    else:
        with open(argv[1], encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    hits = check(text)
    if not hits:
        return 0
    print("commit-msg check: banned attribution/session content found:", file=sys.stderr)
    for label in hits:
        print(f"  - {label}", file=sys.stderr)
    print(
        "Remove the trailer(s) and recommit. See AGENTS.md 'Commit Messages'.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
