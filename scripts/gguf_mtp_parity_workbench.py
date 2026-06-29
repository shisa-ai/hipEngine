#!/usr/bin/env python3
"""Run the GGUF-MTP parity workbench for E2E and per-piece analysis.

This script intentionally orchestrates existing leaf tools instead of embedding
benchmark logic. It gives the llama.cpp parity work one repeatable command for:

* full B3/C5 end-to-end GGUF-MTP smokes across candidate runtime routes;
* selected-MoE q8_1+sudot4 microbench slices for gate/up and down paths;
* optional rocprofv3 kernel-trace capture plus GGUF bucket summary.

Artifacts are diagnostic unless a later caller promotes them through the full
benchmark protocol in docs/BENCHMARK.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_PROMPT = "Write a Python function that implements merge sort:"
DEFAULT_STAGES = "e2e,pieces"
DEFAULT_CANDIDATES = "default,resident-serial-fallback"
DEFAULT_RAW_ROOT = Path("/tmp/hipengine-gguf-mtp-parity-workbench")
SCHEMA = "hipengine.gguf_mtp_parity_workbench.v1"


@dataclass(frozen=True)
class Candidate:
    name: str
    env: dict[str, str]
    extra_args: tuple[str, ...] = ()
    description: str = ""


CANDIDATES: dict[str, Candidate] = {
    "default": Candidate(
        name="default",
        env={},
        description="Production T16 decode-repack route.",
    ),
    "x8-both": Candidate(
        name="x8-both",
        env={"HIPENGINE_GGUF_SELECTED_X8_REPACK": "both"},
        description="Sidecar-free X8 selected-down Q5_K and Q6_K diagnostic route.",
    ),
    "x8-q5": Candidate(
        name="x8-q5",
        env={"HIPENGINE_GGUF_SELECTED_X8_REPACK": "q5"},
        description="Sidecar-free X8 selected-down Q5_K diagnostic route.",
    ),
    "x8-q6": Candidate(
        name="x8-q6",
        env={"HIPENGINE_GGUF_SELECTED_X8_REPACK": "q6"},
        description="Sidecar-free X8 selected-down Q6_K diagnostic route.",
    ),
    "resident-draft": Candidate(
        name="resident-draft",
        env={},
        extra_args=("--resident-mtp-draft",),
        description="Production-shaped resident GGUF MTP draft chain with resident MTP KV commit.",
    ),
    "resident-b5-adaptive": Candidate(
        name="resident-b5-adaptive",
        env={},
        extra_args=("--resident-mtp-draft", "--adaptive-draft-window"),
        description="Resident GGUF MTP draft with adaptive window; use with --draft-n-max 5 for B5 diagnostics.",
    ),
    "resident-serial-fallback": Candidate(
        name="resident-serial-fallback",
        env={},
        extra_args=("--resident-mtp-draft", "--adaptive-ar-fallback", "--no-target-block-verify"),
        description="Production-safe resident GGUF MTP draft with serial graph probe and AR fallback after zero-accept cycles.",
    ),
    "resident-strict-context": Candidate(
        name="resident-strict-context",
        env={},
        extra_args=(
            "--resident-mtp-draft",
            "--root-topk-accept",
            "1",
            "--sibling-topk-accept",
            "1",
            "--mtp-context-replay",
            "--mtp-device-kv-cache",
            "--no-target-block-verify",
        ),
        description="llama.cpp-style strict top-1 route with prompt catch-up replay and device-resident MTP KV.",
    ),
    "resident-production": Candidate(
        name="resident-production",
        env={},
        extra_args=(
            "--resident-mtp-draft",
            "--adaptive-block-after-full-accept",
            "--adaptive-probe-draft-n-max",
            "3",
            "--adaptive-ar-fallback",
        ),
        description="Production selector: serial B3 probe, B5 block verify after full accept, AR fallback after zero accept.",
    ),
    "q4-x8": Candidate(
        name="q4-x8",
        env={"HIPENGINE_GGUF_SELECTED_GATE_UP_X8": "1"},
        description="Sidecar-free X8 selected gate/up Q4_K q8_1+sudot4 route.",
    ),
    "q4-x8-q6": Candidate(
        name="q4-x8-q6",
        env={
            "HIPENGINE_GGUF_SELECTED_GATE_UP_X8": "1",
            "HIPENGINE_GGUF_SELECTED_X8_REPACK": "q6",
        },
        description="Sidecar-free X8 selected gate/up Q4_K plus selected-down Q6_K route.",
    ),
    "t16-dp4a": Candidate(
        name="t16-dp4a",
        env={"HIPENGINE_GGUF_T16_SELECTED_DP4A": "1"},
        description="T16 selected-MoE q8_1+sudot4 diagnostic route.",
    ),
    "q4-t16-dp4a": Candidate(
        name="q4-t16-dp4a",
        env={"HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A": "1"},
        description="Q4_K selected-dual T16/raw q8_1+sudot4 diagnostic gate.",
    ),
    "raw-dp4a": Candidate(
        name="raw-dp4a",
        env={"HIPENGINE_GGUF_RAW_SELECTED_DP4A": "1"},
        extra_args=("--no-decode-repack",),
        description="Raw no-decode-repack selected-MoE q8_1+sudot4 diagnostic route.",
    ),
    "selected-down-raw-dp4a": Candidate(
        name="selected-down-raw-dp4a",
        env={
            "HIPENGINE_GGUF_SELECTED_DOWN_RAW": "both",
            "HIPENGINE_GGUF_RAW_SELECTED_DP4A": "1",
        },
        description="Production decode-repack route with selected-down Q5_K/Q6_K kept raw and routed through q8_1+sudot4.",
    ),
    "block-wmma": Candidate(
        name="block-wmma",
        env={},
        extra_args=("--target-block-wmma-prefill",),
        description="Target block verifier with WMMA/selected-prefill kernels enabled.",
    ),
    "block-native": Candidate(
        name="block-native",
        env={},
        extra_args=("--target-block-verify-mode", "native"),
        description="Target block verifier through the native row loop attention scheduler.",
    ),
    "row-compact-gemv": Candidate(
        name="row-compact-gemv",
        env={"HIPENGINE_GGUF_ROW_COMPACT_GEMV": "1"},
        description="Target block verifier with compact grouped selected-MoE GEMV routing.",
    ),
    "row-lm-head": Candidate(
        name="row-lm-head",
        env={"HIPENGINE_GGUF_VERIFY_ROW_LM_HEAD": "1"},
        description="Target block verifier with row-batched lm-head/argmax sampling.",
    ),
}

ALL_CANDIDATE_NAMES = (
    "default",
    "x8-q6",
    "resident-draft",
    "resident-b5-adaptive",
    "resident-serial-fallback",
    "resident-strict-context",
    "resident-production",
    "x8-both",
    "q4-x8",
    "q4-x8-q6",
    "t16-dp4a",
    "q4-t16-dp4a",
    "raw-dp4a",
    "selected-down-raw-dp4a",
    "block-wmma",
    "block-native",
    "row-compact-gemv",
    "row-lm-head",
)
STAGE_NAMES = ("e2e", "pieces", "rocprof", "true-ar", "category")


def parse_csv_set(text: str, *, valid: set[str], aliases: dict[str, tuple[str, ...]] | None = None, label: str) -> list[str]:
    aliases = aliases or {}
    result: list[str] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"{label} contains an empty entry")
        expanded = aliases.get(part, (part,))
        for item in expanded:
            if item not in valid:
                raise ValueError(f"unknown {label} entry {item!r}; expected one of {sorted(valid | set(aliases))}")
            if item not in result:
                result.append(item)
    return result


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def repo_provenance() -> dict[str, Any]:
    def git(args: list[str]) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    diff = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--quiet"], check=False)
    staged = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--cached", "--quiet"], check=False)
    tracked_dirty = None
    if diff.returncode in {0, 1} and staged.returncode in {0, 1}:
        tracked_dirty = diff.returncode == 1 or staged.returncode == 1
    untracked = git(["ls-files", "--others", "--exclude-standard"])
    return {
        "repo_root": str(REPO_ROOT),
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_tracked_dirty": tracked_dirty,
        "git_untracked_count": None if untracked is None else len([line for line in untracked.splitlines() if line]),
    }


def base_env(extra: dict[str, str], *, hip_arch: str, compiler_version_file: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["HIPENGINE_HIP_ARCH"] = hip_arch
    if compiler_version_file is not None:
        env["HIPENGINE_COMPILER_VERSION_FILE"] = str(compiler_version_file)
    env.update(extra)
    return env


def command_env_delta(candidate: Candidate, *, hip_arch: str, compiler_version_file: Path | None) -> dict[str, str]:
    env = {"PYTHONPATH": str(REPO_ROOT), "HIPENGINE_HIP_ARCH": hip_arch}
    if compiler_version_file is not None:
        env["HIPENGINE_COMPILER_VERSION_FILE"] = str(compiler_version_file)
    env.update(candidate.env)
    return env


def run_command(
    *,
    cmd: list[str],
    env: dict[str, str],
    cwd: Path,
    log_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "command": quote_cmd(cmd),
        "env": {k: env[k] for k in sorted(env) if k.startswith("HIPENGINE_") or k == "PYTHONPATH"},
        "log": str(log_path),
    }
    if dry_run:
        record.update({"status": "dry_run", "returncode": None, "wall_seconds": 0.0})
        return record
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=log_file, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    record.update({"status": "passed" if completed.returncode == 0 else "failed", "returncode": completed.returncode, "wall_seconds": wall})
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(f"command failed with {completed.returncode}: {quote_cmd(cmd)}\n" + "\n".join(tail))
    return record


def build_e2e_cmd(args: argparse.Namespace, candidate: Candidate, output: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/gguf_mtp_bench.py",
        "--model",
        str(args.model),
        "--prompt",
        args.prompt,
        "--prompt-reasoning",
        args.prompt_reasoning,
        "--cycles",
        str(args.cycles),
        "--draft-n-max",
        str(args.draft_n_max),
        "--root-topk-accept",
        str(args.root_topk_accept),
        "--mtp-draft-vocab-cap",
        str(args.mtp_draft_vocab_cap),
        "--output",
        str(output),
        "--mtp-context-replay",
        "--mtp-device-kv-cache",
        "--target-block-verify",
        *candidate.extra_args,
    ]


def summarize_e2e_artifact(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    warm = metrics.get("warm_excluding_cycle0") or {}
    return {
        "artifact": str(path),
        "status": data.get("status"),
        "tokens_per_sec": metrics.get("tokens_per_sec"),
        "ar_baseline_tokens_per_sec": metrics.get("ar_baseline_tokens_per_sec"),
        "speedup_vs_ar_visible": metrics.get("speedup_vs_ar_visible"),
        "avg_cycle_ms": metrics.get("avg_cycle_ms"),
        "avg_ar_decode_ms": metrics.get("avg_ar_decode_ms"),
        "avg_mtp_draft_ms": metrics.get("avg_mtp_draft_ms"),
        "total_accepted": metrics.get("total_accepted"),
        "total_drafts": metrics.get("total_drafts"),
        "total_output_tokens": metrics.get("total_output_tokens"),
        "warm_tokens_per_sec": warm.get("tokens_per_sec"),
        "warm_avg_cycle_ms": warm.get("avg_cycle_ms"),
        "warm_avg_ar_decode_ms": warm.get("avg_ar_decode_ms"),
        "warm_avg_mtp_draft_ms": warm.get("avg_mtp_draft_ms"),
    }


def run_e2e(args: argparse.Namespace, candidates: list[Candidate], root: Path, *, dry_run: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        output = root / "e2e" / f"{args.tag}-{candidate.name}-b{args.draft_n_max}-c{args.cycles}.json"
        log = root / "logs" / f"e2e-{candidate.name}.log"
        env = base_env(candidate.env, hip_arch=args.hip_arch, compiler_version_file=args.compiler_version_file)
        cmd = build_e2e_cmd(args, candidate, output)
        row = {
            "candidate": candidate.name,
            "description": candidate.description,
            "env": command_env_delta(candidate, hip_arch=args.hip_arch, compiler_version_file=args.compiler_version_file),
            "extra_args": list(candidate.extra_args),
            "artifact": str(output),
        }
        try:
            command = run_command(cmd=cmd, env=env, cwd=REPO_ROOT, log_path=log, dry_run=dry_run)
        except RuntimeError as exc:
            row["command"] = {
                "command": quote_cmd(cmd),
                "env": command_env_delta(candidate, hip_arch=args.hip_arch, compiler_version_file=args.compiler_version_file),
                "log": str(log),
                "status": "failed",
            }
            row["status"] = "failed"
            row["error"] = str(exc).splitlines()[-1] if str(exc) else "command failed"
            rows.append(row)
            continue
        row["command"] = command
        if not dry_run:
            row["metrics"] = summarize_e2e_artifact(output)
        rows.append(row)
    return rows


def piece_commands(args: argparse.Namespace, root: Path) -> list[tuple[str, list[str], Path]]:
    pieces = [
        (
            "q4_selected_dual",
            [
                sys.executable,
                "scripts/gguf_q4_k_selected_dual_dp4a_microbench.py",
                "--iters",
                str(args.piece_iters),
                "--warmup",
                str(args.piece_warmup),
                "--json",
                str(root / "pieces" / f"{args.tag}-q4-selected-dual-dp4a.json"),
            ],
            root / "logs" / "piece-q4-selected-dual.log",
        ),
        (
            "raw_selected_down_q5_q6",
            [
                sys.executable,
                "scripts/gguf_k_selected_pack8_dp4a_microbench.py",
                "--iters",
                str(args.piece_iters),
                "--warmup",
                str(args.piece_warmup),
                "--json",
                str(root / "pieces" / f"{args.tag}-raw-selected-down-q5-q6-dp4a.json"),
            ],
            root / "logs" / "piece-raw-selected-down.log",
        ),
        (
            "x8_selected_down_q5_q6",
            [
                sys.executable,
                "scripts/gguf_x8_selected_down_dp4a_microbench.py",
                "--iters",
                str(args.piece_iters),
                "--warmup",
                str(args.piece_warmup),
                "--json",
                str(root / "pieces" / f"{args.tag}-x8-selected-down-q5-q6-dp4a.json"),
            ],
            root / "logs" / "piece-x8-selected-down.log",
        ),
    ]
    if args.compiler_version_file is not None:
        with_compiler: list[tuple[str, list[str], Path]] = []
        for name, cmd, log in pieces:
            cmd = [*cmd[:-2], "--compiler-version-file", str(args.compiler_version_file), *cmd[-2:]]
            with_compiler.append((name, cmd, log))
        pieces = with_compiler
    return pieces


def summarize_piece_artifact(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {"artifact": str(path), "schema": data.get("schema"), "shape": data.get("shape")}
    if "timing_ms" in data:
        summary["timing_ms"] = data.get("timing_ms")
        summary["speedup"] = data.get("speedup")
        summary["correctness"] = data.get("correctness_vs_raw")
        return summary
    rows = []
    for result in data.get("results", []):
        rows.append(
            {
                "quant": result.get("quant"),
                "timing_ms": result.get("timing_ms"),
                "speedup": result.get("speedup"),
                "correctness": result.get("correctness_vs_float") or result.get("correctness_vs_production_t16_float"),
            }
        )
    summary["results"] = rows
    return summary


def run_pieces(args: argparse.Namespace, root: Path, *, dry_run: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env = base_env({}, hip_arch=args.hip_arch, compiler_version_file=args.compiler_version_file)
    for name, cmd, log in piece_commands(args, root):
        command = run_command(cmd=cmd, env=env, cwd=REPO_ROOT, log_path=log, dry_run=dry_run)
        output = Path(cmd[-1])
        row = {"piece": name, "command": command, "artifact": str(output)}
        if not dry_run:
            row["metrics"] = summarize_piece_artifact(output)
        rows.append(row)
    return rows


def run_category(args: argparse.Namespace, candidates: list[Candidate], root: Path, *, dry_run: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    true_ar_baseline = args.true_ar_baseline_json or getattr(args, "_generated_true_ar_baseline_json", None)
    for candidate in candidates:
        output = root / "category" / f"{args.tag}-{candidate.name}-summary.json"
        raw_root = root / "category" / candidate.name
        log = root / "logs" / f"category-{candidate.name}.log"
        extra_args = [
            "--prompt-reasoning",
            args.prompt_reasoning,
            "--root-topk-accept",
            str(args.root_topk_accept),
            "--mtp-context-replay",
            "--mtp-device-kv-cache",
            "--target-block-verify",
            "--mtp-draft-vocab-cap",
            str(args.mtp_draft_vocab_cap),
            *candidate.extra_args,
        ]
        cmd = [
            sys.executable,
            "scripts/gguf_mtp_category_bench.py",
            "--model",
            str(args.model),
            "--prompts",
            str(args.prompts),
            "--budgets",
            str(args.draft_n_max),
            "--cycles",
            str(args.cycles),
            "--raw-root",
            str(raw_root),
            "--output",
            str(output),
            *(("--limit", str(args.category_limit)) if args.category_limit is not None else ()),
            *(("--true-ar-baseline-json", str(true_ar_baseline)) if true_ar_baseline is not None else ()),
            *(("--reuse-existing",) if args.category_reuse_existing else ()),
            *(f"--extra-arg={extra}" for extra in extra_args),
            *(("--dry-run",) if dry_run else ()),
        ]
        env = base_env(candidate.env, hip_arch=args.hip_arch, compiler_version_file=args.compiler_version_file)
        row = {
            "candidate": candidate.name,
            "description": candidate.description,
            "env": command_env_delta(candidate, hip_arch=args.hip_arch, compiler_version_file=args.compiler_version_file),
            "extra_args": list(candidate.extra_args),
            "artifact": str(output),
        }
        try:
            command = run_command(cmd=cmd, env=env, cwd=REPO_ROOT, log_path=log, dry_run=dry_run)
        except RuntimeError as exc:
            row["command"] = {
                "command": quote_cmd(cmd),
                "env": command_env_delta(candidate, hip_arch=args.hip_arch, compiler_version_file=args.compiler_version_file),
                "log": str(log),
                "status": "failed",
            }
            row["status"] = "failed"
            row["error"] = str(exc).splitlines()[-1] if str(exc) else "command failed"
            rows.append(row)
            continue
        row["command"] = command
        if not dry_run and output.exists():
            data = json.loads(output.read_text(encoding="utf-8"))
            row["summary"] = {
                "status": data.get("status"),
                "best": data.get("best"),
                "totals": data.get("totals"),
                "true_ar_comparison_available": data.get("true_ar_comparison_available"),
            }
        rows.append(row)
    return rows


def run_true_ar(args: argparse.Namespace, root: Path, *, dry_run: bool) -> dict[str, Any]:
    output = root / "true_ar" / f"{args.tag}-true-ar-baseline.json"
    raw_root = root / "true_ar" / "raw"
    log = root / "logs" / "true-ar.log"
    decode_tokens = int(args.true_ar_decode_tokens or (args.cycles * (args.draft_n_max + 1)))
    cmd = [
        sys.executable,
        "scripts/gguf_true_ar_category_bench.py",
        "--model",
        str(args.model),
        "--prompts",
        str(args.prompts),
        "--decode-tokens",
        str(decode_tokens),
        "--warmup-decode-tokens",
        str(args.true_ar_warmup_decode_tokens),
        "--raw-root",
        str(raw_root),
        "--output",
        str(output),
        *(("--limit", str(args.category_limit)) if args.category_limit is not None else ()),
        *(("--compiler-version-file", str(args.compiler_version_file)) if args.compiler_version_file is not None else ()),
        *(("--dry-run",) if dry_run else ()),
    ]
    env = base_env({}, hip_arch=args.hip_arch, compiler_version_file=args.compiler_version_file)
    command = run_command(cmd=cmd, env=env, cwd=REPO_ROOT, log_path=log, dry_run=dry_run)
    row: dict[str, Any] = {
        "artifact": str(output),
        "raw_root": str(raw_root),
        "decode_tokens": decode_tokens,
        "warmup_decode_tokens": int(args.true_ar_warmup_decode_tokens),
        "command": command,
    }
    if not dry_run and output.exists():
        data = json.loads(output.read_text(encoding="utf-8"))
        row["summary"] = {
            "status": data.get("status"),
            "totals": data.get("totals"),
            "categories": data.get("categories"),
        }
    return row


def find_rocprof_csv(directory: Path) -> Path:
    matches = sorted(directory.glob("*_kernel_trace.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one rocprof kernel trace CSV in {directory}, found {len(matches)}")
    return matches[0]


def run_rocprof(args: argparse.Namespace, candidates: list[Candidate], root: Path, *, dry_run: bool) -> list[dict[str, Any]]:
    rocprofv3 = shutil.which(args.rocprofv3) or args.rocprofv3
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        prof_dir = root / "rocprof" / candidate.name
        prof_dir.mkdir(parents=True, exist_ok=True)
        bench_output = prof_dir / f"{args.tag}-{candidate.name}-profiled-e2e.json"
        bench_cmd = build_e2e_cmd(args, candidate, bench_output)
        cmd = [
            rocprofv3,
            "--kernel-trace",
            "--output-format",
            "csv",
            "--output-directory",
            str(prof_dir),
            "--output-file",
            "trace",
            "--",
            *bench_cmd,
        ]
        env = base_env(candidate.env, hip_arch=args.hip_arch, compiler_version_file=args.compiler_version_file)
        command = run_command(cmd=cmd, env=env, cwd=REPO_ROOT, log_path=root / "logs" / f"rocprof-{candidate.name}.log", dry_run=dry_run)
        row: dict[str, Any] = {
            "candidate": candidate.name,
            "command": command,
            "profiled_e2e_artifact": str(bench_output),
            "rocprof_directory": str(prof_dir),
        }
        if not dry_run:
            csv_path = find_rocprof_csv(prof_dir)
            summary_json = prof_dir / "trace-summary.json"
            summary_cmd = [
                sys.executable,
                "scripts/qwen35_gguf_rocprof_summary.py",
                "--csv",
                str(csv_path),
                "--tokens-decode",
                str(args.cycles * (args.draft_n_max + 1)),
                "--json",
                str(summary_json),
                "--quiet",
            ]
            summary_command = run_command(
                cmd=summary_cmd,
                env=base_env({}, hip_arch=args.hip_arch, compiler_version_file=args.compiler_version_file),
                cwd=REPO_ROOT,
                log_path=root / "logs" / f"rocprof-summary-{candidate.name}.log",
                dry_run=False,
            )
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            phase = summary.get("phases", {}).get("prefill", {})
            row.update(
                {
                    "kernel_trace_csv": str(csv_path),
                    "summary_json": str(summary_json),
                    "summary_command": summary_command,
                    "top_buckets": phase.get("buckets", [])[:10],
                    "top_kernels": phase.get("top_kernels", [])[:10],
                }
            )
        rows.append(row)
    return rows


def choose_best_e2e(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    measured = [row for row in rows if isinstance(row.get("metrics"), dict)]
    if not measured:
        return None
    return max(measured, key=lambda row: float(row["metrics"].get("tokens_per_sec") or 0.0))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-reasoning", choices=("off", "open", "none"), default="off")
    parser.add_argument("--prompts", type=Path, default=REPO_ROOT / "benchmarks" / "prompts" / "mtpbench-code-general-ja.jsonl")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--draft-n-max", type=int, default=3)
    parser.add_argument("--root-topk-accept", type=int, default=1)
    parser.add_argument("--mtp-draft-vocab-cap", type=int, default=32768)
    parser.add_argument("--piece-iters", type=int, default=80)
    parser.add_argument("--piece-warmup", type=int, default=20)
    parser.add_argument("--category-limit", type=int, default=None)
    parser.add_argument(
        "--category-reuse-existing",
        action="store_true",
        help="Pass --reuse-existing to the category stage so completed per-prompt child JSONs are aggregated without rerunning.",
    )
    parser.add_argument(
        "--true-ar-baseline-json",
        type=Path,
        default=None,
        help="Optional true no-MTP AR baseline artifact to attach to category summaries.",
    )
    parser.add_argument(
        "--true-ar-decode-tokens",
        type=int,
        default=None,
        help="Decode tokens for the true-ar stage; default cycles * (draft_n_max + 1).",
    )
    parser.add_argument("--true-ar-warmup-decode-tokens", type=int, default=1)
    parser.add_argument("--hip-arch", default=os.environ.get("HIPENGINE_HIP_ARCH", "gfx1151"))
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y-%m-%d-gguf-mtp-parity"))
    parser.add_argument("--stages", default=DEFAULT_STAGES, help=f"Comma-separated subset of {STAGE_NAMES}; aliases: all")
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES, help=f"Comma-separated candidates; aliases: all. Known: {tuple(CANDIDATES)}")
    parser.add_argument("--rocprofv3", default="rocprofv3")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.category_limit is not None and args.category_limit < 0:
        raise ValueError("--category-limit must be non-negative")
    if args.true_ar_decode_tokens is not None and args.true_ar_decode_tokens <= 0:
        raise ValueError("--true-ar-decode-tokens must be positive")
    if args.true_ar_warmup_decode_tokens < 0:
        raise ValueError("--true-ar-warmup-decode-tokens must be non-negative")
    stages = parse_csv_set(args.stages, valid=set(STAGE_NAMES), aliases={"all": STAGE_NAMES}, label="stage")
    candidate_names = parse_csv_set(args.candidates, valid=set(CANDIDATES), aliases={"all": ALL_CANDIDATE_NAMES}, label="candidate")
    candidates = [CANDIDATES[name] for name in candidate_names]
    root = args.raw_root / args.tag
    root.mkdir(parents=True, exist_ok=True)
    output = args.output or (root / "summary.json")

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "dry_run" if args.dry_run else "complete",
        "performance_claim": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "repo": repo_provenance(),
        "protocol": {
            "model": str(args.model),
            "prompt": args.prompt,
            "prompt_reasoning": args.prompt_reasoning,
            "cycles": args.cycles,
            "draft_n_max": args.draft_n_max,
            "root_topk_accept": args.root_topk_accept,
            "mtp_draft_vocab_cap": args.mtp_draft_vocab_cap,
            "category_limit": args.category_limit,
            "category_reuse_existing": args.category_reuse_existing,
            "true_ar_baseline_json": None if args.true_ar_baseline_json is None else str(args.true_ar_baseline_json),
            "true_ar_decode_tokens": args.true_ar_decode_tokens,
            "true_ar_warmup_decode_tokens": args.true_ar_warmup_decode_tokens,
            "hip_arch": args.hip_arch,
            "raw_root": str(root),
            "stages": stages,
            "candidates": candidate_names,
        },
        "notes": [
            "Diagnostic workbench only; not a retained speed claim.",
            "E2E commands use the B3/C5 llama.cpp-parity smoke shape with context replay, device MTP KV, block verify, and 32k draft vocab cap.",
            "Per-piece commands measure selected-MoE q8_1+sudot4 slices outside the production runtime.",
        ],
        "e2e": [],
        "pieces": [],
        "rocprof": [],
        "true_ar": None,
        "category": [],
        "decision": {
            "promote_candidate": None,
            "promotion_blocker": "A candidate must beat default on same-protocol E2E and clear full-suite validation before promotion.",
        },
    }

    if "e2e" in stages:
        artifact["e2e"] = run_e2e(args, candidates, root, dry_run=args.dry_run)
        best = choose_best_e2e(artifact["e2e"])
        if best is not None:
            artifact["decision"]["best_e2e_candidate"] = {
                "candidate": best["candidate"],
                "tokens_per_sec": best["metrics"].get("tokens_per_sec"),
                "artifact": best["metrics"].get("artifact"),
            }
    if "pieces" in stages:
        artifact["pieces"] = run_pieces(args, root, dry_run=args.dry_run)
    if "true-ar" in stages:
        artifact["true_ar"] = run_true_ar(args, root, dry_run=args.dry_run)
        args._generated_true_ar_baseline_json = Path(artifact["true_ar"]["artifact"])
    if "category" in stages:
        artifact["category"] = run_category(args, candidates, root, dry_run=args.dry_run)
    if "rocprof" in stages:
        artifact["rocprof"] = run_rocprof(args, candidates, root, dry_run=args.dry_run)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    if artifact["decision"].get("best_e2e_candidate"):
        best = artifact["decision"]["best_e2e_candidate"]
        print(f"best e2e: {best['candidate']} {best['tokens_per_sec']} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
