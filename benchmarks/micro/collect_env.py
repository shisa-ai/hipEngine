#!/usr/bin/env python3
"""Collect environment metadata for hipEngine microbenchmarks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MAX_OUTPUT_CHARS = 20000


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def run_command(args: list[str], *, timeout_s: float, max_output_chars: int, cwd: Path | None = None) -> dict[str, Any]:
    executable = shutil.which(args[0])
    record: dict[str, Any] = {
        "args": args,
        "available": executable is not None,
        "path": executable,
        "returncode": None,
        "timed_out": False,
        "stdout": "",
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
    }
    if executable is None:
        return record
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        record["timed_out"] = True
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        record["stdout"], record["stdout_truncated"] = _truncate(stdout, max_output_chars)
        record["stderr"], record["stderr_truncated"] = _truncate(stderr, max_output_chars)
        return record
    record["returncode"] = completed.returncode
    record["stdout"], record["stdout_truncated"] = _truncate(completed.stdout, max_output_chars)
    record["stderr"], record["stderr_truncated"] = _truncate(completed.stderr, max_output_chars)
    return record


def _git_text(repo_root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip()


def collect_git(repo_root: Path) -> dict[str, Any]:
    status_short = _git_text(repo_root, ["status", "--short"]).splitlines()
    return {
        "root": str(repo_root),
        "branch": _git_text(repo_root, ["branch", "--show-current"]),
        "commit": _git_text(repo_root, ["rev-parse", "HEAD"]),
        "commit_short": _git_text(repo_root, ["rev-parse", "--short", "HEAD"]),
        "dirty": bool(status_short),
        "status_short": status_short,
    }


def _first_matching_lines(text: str, needles: tuple[str, ...], *, limit: int = 80) -> list[str]:
    matches: list[str] = []
    for line in text.splitlines():
        if any(needle in line for needle in needles):
            matches.append(line.strip())
            if len(matches) >= limit:
                break
    return matches


def parse_device_summaries(commands: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rocminfo = commands.get("rocminfo", {})
    vulkaninfo = commands.get("vulkaninfo_summary", {})
    lspci = commands.get("lspci", {})
    rocm_smi = commands.get("rocm_smi", {})
    return {
        "rocminfo_name_gfx_lines": _first_matching_lines(
            str(rocminfo.get("stdout", "")),
            ("Name:", "gfx"),
        ),
        "vulkan_summary_lines": _first_matching_lines(
            str(vulkaninfo.get("stdout", "")),
            ("GPU", "deviceName", "driver", "apiVersion", "Mesa", "RADV"),
        ),
        "lspci_display_lines": _first_matching_lines(
            str(lspci.get("stdout", "")),
            ("VGA", "Display", "3D controller", "AMD", "ATI"),
        ),
        "rocm_smi_lines": _first_matching_lines(
            str(rocm_smi.get("stdout", "")),
            ("GPU", "Card", "Driver", "VBIOS", "Device", "ASIC"),
        ),
    }


def collect_environment(
    *,
    repo_root: Path,
    include_device_probes: bool,
    timeout_s: float,
    max_output_chars: int,
) -> dict[str, Any]:
    commands: dict[str, dict[str, Any]] = {
        "uname": run_command(["uname", "-a"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "git_version": run_command(["git", "--version"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "hipcc_version": run_command(["hipcc", "--version"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "amdclang_version": run_command(["amdclang++", "--version"], timeout_s=timeout_s, max_output_chars=max_output_chars),
    }
    if include_device_probes:
        commands.update(
            {
                "rocminfo": run_command(["rocminfo"], timeout_s=timeout_s, max_output_chars=max_output_chars),
                "rocm_smi": run_command(
                    ["rocm-smi", "--showproductname", "--showdriverversion"],
                    timeout_s=timeout_s,
                    max_output_chars=max_output_chars,
                ),
                "vulkaninfo_summary": run_command(
                    ["vulkaninfo", "--summary"],
                    timeout_s=timeout_s,
                    max_output_chars=max_output_chars,
                ),
                "lspci": run_command(["lspci", "-nn"], timeout_s=timeout_s, max_output_chars=max_output_chars),
            }
        )

    return {
        "schema_version": 1,
        "kind": "hipengine_micro_environment",
        "collected_at_unix": time.time(),
        "collector": {
            "path": str(Path(__file__).resolve()),
            "argv": sys.argv,
            "max_output_chars": max_output_chars,
            "timeout_s": timeout_s,
            "include_device_probes": include_device_probes,
        },
        "repo": collect_git(repo_root),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "uname": platform.uname()._asdict(),
            "env": {
                "PATH": os.environ.get("PATH", ""),
                "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
                "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
                "VK_ICD_FILENAMES": os.environ.get("VK_ICD_FILENAMES"),
                "RADV_PERFTEST": os.environ.get("RADV_PERFTEST"),
            },
        },
        "commands": commands,
        "devices": parse_device_summaries(commands),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root to record git metadata from")
    parser.add_argument("--out", type=Path, help="Write JSON to this path instead of stdout")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON with sorted keys")
    parser.add_argument("--skip-device-probes", action="store_true", help="Skip rocminfo, rocm-smi, vulkaninfo, and lspci")
    parser.add_argument("--timeout-s", type=float, default=8.0, help="Per-command timeout")
    parser.add_argument("--max-output-chars", type=int, default=DEFAULT_MAX_OUTPUT_CHARS, help="Truncate each command stream to this many chars")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    data = collect_environment(
        repo_root=repo_root,
        include_device_probes=not args.skip_device_probes,
        timeout_s=args.timeout_s,
        max_output_chars=args.max_output_chars,
    )
    indent = 2 if args.pretty else None
    text = json.dumps(data, indent=indent, sort_keys=args.pretty)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
