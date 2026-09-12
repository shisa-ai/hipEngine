"""Tests for the commit-message byline guard (scripts/check_commit_msg.py).

The guard backs the commit-msg hook installed by
scripts/install_commit_msg_hook.py. AGENTS.md 'Commit Messages' bans
bylines, agent attribution, and generated-by footers; session URLs are
additionally treated as leaked credentials.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_commit_msg.py"

spec = importlib.util.spec_from_file_location("check_commit_msg", SCRIPT)
check_commit_msg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_commit_msg)


def _run(message: str) -> int:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "-"],
        input=message,
        capture_output=True,
        text=True,
    )
    return proc.returncode


CLEAN_MESSAGES = [
    "fix: repair the allocator fall-through",
    "docs: no bylines here\n\nBody mentions co-authors in prose but never as a\ntrailer. Session-related identifiers like session_id stay clean.\n\nRefs: #12",
    "port: upstream lineage from llama.cpp\n\nUpstream commit 1234567 noreply@example.com reference is fine.",
]

BANNED_MESSAGES = [
    "test: x\n\nCo-authored-by: Someone <noreply@anthropic.com>\n",
    "test: x\n\nCo-Authored-By: Someone <someone@example.com>\n",
    "test: x\n\nAssisted-By: Foo Agent\n",
    "test: x\n\nClaude-Session: https://claude.ai/code/session_01CYcKGm\n",
    "test: x\n\nSee https://claude.ai/code/session_01CYcKGm for details.\n",
    "test: x\n\nGenerated-By: copilot\n",
    "test: x\n\nGenerated with Claude Code\n",
    "test: x\n\nGenerated-with some other tool\n",
    "test: x\n\n\U0001f916 Generated with Claude Code\n",
    "test: x\n\ncontact noreply@anthropic.com for details\n",
]


def test_check_function_clean() -> None:
    for message in CLEAN_MESSAGES:
        assert check_commit_msg.check(message) == [], message


def test_check_function_banned() -> None:
    for message in BANNED_MESSAGES:
        assert check_commit_msg.check(message) != [], message


def test_cli_clean_exit_zero() -> None:
    assert _run("fix: a clean message\n\nBody text.\n") == 0


def test_cli_banned_exit_one() -> None:
    assert _run("test: x\n\nClaude-Session: https://claude.ai/code/session_abc\n") == 1


def test_cli_banned_reports_label() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "-"],
        input="test: x\n\nCo-authored-by: A <a@b.c>\n",
        capture_output=True,
        text=True,
    )
    assert "Co-Authored-By" in proc.stderr


def test_guard_files_reject_known_offenders() -> None:
    """The hook path used by .git/hooks/commit-msg rejects offenders."""
    for message in BANNED_MESSAGES[:3]:
        assert _run(message) == 1, message
