#!/usr/bin/env python3
"""Install the hipEngine commit-msg guard into .git/hooks/commit-msg.

Mirrors the worklog install-hook convention: does not set core.hooksPath,
does not touch the Git LFS post-checkout/post-commit/post-merge/pre-push
hooks, and refuses to overwrite an unrelated existing commit-msg hook.
Re-running updates a hook previously installed by this script.

    python3 scripts/install_commit_msg_hook.py
"""

from __future__ import annotations

import os
import subprocess
import sys

MARKER = "hipengine-commit-msg-guard"

HOOK = f"""#!/bin/sh
# {MARKER}: installed by scripts/install_commit_msg_hook.py
# Rejects bylines, agent attribution, and session-id trailers (AGENTS.md).
exec python3 "$(git rev-parse --show-toplevel)/scripts/check_commit_msg.py" "$1"
"""


def main() -> int:
    git_dir = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    hook_path = os.path.join(git_dir, "hooks", "commit-msg")

    if os.path.exists(hook_path):
        with open(hook_path, encoding="utf-8", errors="replace") as fh:
            existing = fh.read()
        if MARKER not in existing:
            print(
                f"refusing to overwrite unrelated existing hook: {hook_path}",
                file=sys.stderr,
            )
            return 1

    os.makedirs(os.path.dirname(hook_path), exist_ok=True)
    with open(hook_path, "w", encoding="utf-8") as fh:
        fh.write(HOOK)
    os.chmod(hook_path, 0o755)
    print(f"installed commit-msg guard: {hook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
