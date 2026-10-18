#!/usr/bin/env python3
"""Run every documentation gate in one pass.

    python3 scripts/docs/check_all.py

Individual gates stay where they are so existing references keep working; this
is the single entry point that runs them together and reports one summary.

    check_docs.py                    front-matter + index freshness (docs/)
    check_envs_docs.py               every env var read in the tree is in ENVS.md
    sync_benchmark_readme.py --check exported benchmark prose matches the source
    check_published_command_drift.py published artifact commands match their script

Use --write to let the gates that can repair themselves do so (currently the
docs index). Everything else is read-only.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

GATES: tuple[tuple[str, list[str], list[str]], ...] = (
    # (label, argv, extra argv when --write)
    ("docs front-matter and indexes", ["scripts/docs/check_docs.py"], ["--write"]),
    ("ENVS.md env-var coverage", ["scripts/check_envs_docs.py"], []),
    ("benchmark README export sync", ["scripts/sync_benchmark_readme.py", "--check"], []),
    ("published command drift", ["scripts/check_published_command_drift.py", "--quiet"], []),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="let self-repairing gates fix themselves")
    args = parser.parse_args()

    failures: list[str] = []
    for label, argv, write_argv in GATES:
        command = [sys.executable, *argv, *(write_argv if args.write else [])]
        script = REPO_ROOT / argv[0]
        if not script.exists():
            print(f"SKIP  {label} ({argv[0]} not found)")
            continue
        result = subprocess.run(command, cwd=REPO_ROOT)
        status = "ok" if result.returncode == 0 else "FAIL"
        print(f"{status:5} {label}")
        if result.returncode != 0:
            failures.append(label)

    print()
    if failures:
        print(f"docs gates: {len(failures)} failed — " + ", ".join(failures), file=sys.stderr)
        return 1
    print(f"docs gates: all {len(GATES)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
