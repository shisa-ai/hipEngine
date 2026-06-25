#!/usr/bin/env python3
"""llama.cpp F16-vs-Q8_0 KV quality sweep using llama-perplexity KL mode.

This is an external comparison diagnostic.  It does *not* run hipEngine and it is
not a performance-claim harness.  It automates the llama.cpp workflow documented
in tools/perplexity/README.md:

1. run a reference pass with F16 KV and ``--kl-divergence-base`` to save logits;
2. run a candidate pass with Q8_0 KV, the same logit file, and
   ``--kl-divergence``;
3. parse mean/max KL and "same top p" so the row can be compared to hipEngine's
   KL/top-1 quality guard.

Large contexts produce very large temporary logit files: roughly
``ctx * vocab * 2`` bytes.  The script deletes them by default after each row;
pass ``--keep-logits`` only when you explicitly need to inspect them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LLAMA_PERPLEXITY = Path("/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-perplexity")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_WORK_DIR = Path("/tmp/hipengine-llamacpp-q8-kv-quality-sweep")
_FLOAT_RE = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"

_METRIC_PATTERNS = {
    "kl_mean": [
        re.compile(rf"\bMean\s+(?:KLD|KL(?:\s+divergence)?)\b\s*[:=]?\s*{_FLOAT_RE}", re.I),
        re.compile(rf"\bmean\s+kld\b.*?{_FLOAT_RE}", re.I),
    ],
    "kl_max": [
        re.compile(rf"\bMaximum\s+(?:KLD|KL(?:\s+divergence)?)\b\s*[:=]?\s*{_FLOAT_RE}", re.I),
        re.compile(rf"\bmax(?:imum)?\s+kld\b.*?{_FLOAT_RE}", re.I),
    ],
    "same_top_p": [
        re.compile(rf"^\s*Same\s+top\s+p\s*[:=]\s*{_FLOAT_RE}\s*%?", re.I | re.M),
        re.compile(rf"^\s*same[-_\s]+top[-_\s]+p\s*[:=]\s*{_FLOAT_RE}\s*%?", re.I | re.M),
    ],
    "ppl_ratio": [
        re.compile(rf"\bMean\s+PPL\(Q\)\s*/\s*PPL\(base\)\b\s*[:=]?\s*{_FLOAT_RE}", re.I),
        re.compile(rf"\bMean\s+PPL\s+ratio\b\s*[:=]?\s*{_FLOAT_RE}", re.I),
    ],
}


def _parse_count(text: str) -> int:
    value = text.strip().lower()
    if not value:
        raise ValueError("empty count")
    if value.endswith("k"):
        return int(float(value[:-1]) * 1024)
    if value.endswith("m"):
        return int(float(value[:-1]) * 1024 * 1024)
    return int(value)


def _parse_count_list(text: str) -> list[int]:
    values = [_parse_count(item) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one context size")
    return sorted(set(values))


def _format_count(value: int) -> str:
    if value % 1024 == 0:
        return f"{value // 1024}K"
    return str(value)


def _command_string(command: list[str], env_prefix: dict[str, str] | None = None) -> str:
    parts: list[str] = []
    for key, value in sorted((env_prefix or {}).items()):
        parts.append(f"{key}={shlex.quote(value)}")
    parts.extend(shlex.quote(part) for part in command)
    return " ".join(parts)


def _tail(text: str, max_lines: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, patterns in _METRIC_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                value = float(match.group(1))
                if name == "same_top_p" and value > 1.0:
                    value /= 100.0
                metrics[name] = value
                break
    return metrics


def _run(command: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    start = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, env=merged_env, check=False)
    elapsed = time.perf_counter() - start
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    return {
        "command": command,
        "command_string": _command_string(command, env),
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(elapsed),
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
        "combined_tail": _tail(stdout + "\n" + stderr),
        "metrics": _parse_metrics(stdout + "\n" + stderr),
    }


def _write_generated_corpus(path: Path, *, contexts: list[int], chunks: int) -> None:
    target_units = max(contexts) * max(chunks, 1) * 8
    sentence = (
        "In a long-context cache quality test, the model must preserve attention "
        "decisions over repeated factual and code-like text. "
    )
    with path.open("w", encoding="utf-8") as handle:
        written_words = 0
        while written_words < target_units:
            handle.write(sentence)
            handle.write("\n")
            written_words += len(sentence.split())


def _common_llama_args(args: argparse.Namespace, *, ctx: int, cache_k: str, cache_v: str) -> list[str]:
    command = [
        str(args.llama_perplexity),
        "-m",
        str(args.model),
        "-ngl",
        str(args.n_gpu_layers),
        "-fa",
        str(args.flash_attn),
        "-ctk",
        cache_k,
        "-ctv",
        cache_v,
        "-c",
        str(ctx),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "--chunks",
        str(args.chunks),
        "-f",
        str(args.prompt_file),
    ]
    if args.no_warmup:
        command.append("--no-warmup")
    if args.extra_args:
        command.extend(shlex.split(args.extra_args))
    return command


def _run_row(args: argparse.Namespace, *, ctx: int, work_dir: Path) -> dict[str, Any]:
    label = _format_count(ctx)
    logits_path = work_dir / f"llamacpp-f16-kv-base-{label}.kld"
    reference_cmd = [
        *_common_llama_args(
            args,
            ctx=ctx,
            cache_k=args.reference_cache_type_k,
            cache_v=args.reference_cache_type_v,
        ),
        "--kl-divergence-base",
        str(logits_path),
    ]
    candidate_cmd = [
        *_common_llama_args(
            args,
            ctx=ctx,
            cache_k=args.candidate_cache_type_k,
            cache_v=args.candidate_cache_type_v,
        ),
        "--kl-divergence-base",
        str(logits_path),
        "--kl-divergence",
    ]
    row_start = time.perf_counter()
    reference = _run(reference_cmd, env=args.env)
    candidate: dict[str, Any] | None = None
    if reference["returncode"] == 0:
        candidate = _run(candidate_cmd, env=args.env)
    logits_size_bytes = logits_path.stat().st_size if logits_path.exists() else 0
    if logits_path.exists() and not args.keep_logits:
        logits_path.unlink()
    elapsed = time.perf_counter() - row_start
    metrics = dict(candidate.get("metrics", {}) if candidate is not None else {})
    kl_mean = metrics.get("kl_mean")
    top1 = metrics.get("same_top_p")
    parsed_guard = kl_mean is not None and top1 is not None
    passed = bool(
        parsed_guard
        and float(kl_mean) <= float(args.kl_threshold)
        and float(top1) >= float(args.top1_threshold)
        and reference["returncode"] == 0
        and candidate is not None
        and candidate["returncode"] == 0
    )
    return {
        "workload": f"ctx{label}_chunks{args.chunks}",
        "ctx_size": int(ctx),
        "chunks": int(args.chunks),
        "passed_hipengine_like_quality_guard": passed,
        "guard_metric_note": "uses llama.cpp mean KL and same-top-p; max KL is retained as an outlier diagnostic",
        "quality_thresholds": {
            "kl_mean_max": float(args.kl_threshold),
            "same_top_p_min": float(args.top1_threshold),
        },
        "metrics": metrics,
        "parsed_guard_metrics": bool(parsed_guard),
        "reference": reference,
        "candidate": candidate,
        "logits_file": {
            "path": str(logits_path),
            "size_bytes_before_cleanup": int(logits_size_bytes),
            "kept": bool(args.keep_logits),
        },
        "elapsed_seconds": float(elapsed),
    }


def _git_rev_parse(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _repo_from_binary(binary: Path) -> Path | None:
    current = binary.resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _parse_env(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--env entry must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"--env entry has empty key: {item!r}")
        env[key] = value
    return env


def _top_command(args: argparse.Namespace) -> str:
    parts = [
        "python3 scripts/llamacpp_q8_kv_quality_sweep.py",
        f"--llama-perplexity {args.llama_perplexity}",
        f"--model {args.model}",
        f"--contexts {args.contexts}",
        f"--chunks {args.chunks}",
        f"--prompt-file {args.prompt_file}",
        f"--work-dir {args.work_dir}",
        f"--reference-cache-type-k {args.reference_cache_type_k}",
        f"--reference-cache-type-v {args.reference_cache_type_v}",
        f"--candidate-cache-type-k {args.candidate_cache_type_k}",
        f"--candidate-cache-type-v {args.candidate_cache_type_v}",
        f"--kl-threshold {args.kl_threshold}",
        f"--top1-threshold {args.top1_threshold}",
    ]
    if args.keep_logits:
        parts.append("--keep-logits")
    if args.no_warmup:
        parts.append("--no-warmup")
    if args.extra_args:
        parts.append(f"--extra-args {shlex.quote(args.extra_args)}")
    for key, value in sorted(args.env.items()):
        parts.append(f"--env {key}={shlex.quote(value)}")
    if args.json is not None:
        parts.append(f"--json {args.json}")
    return " ".join(parts)


def run(args: argparse.Namespace) -> dict[str, Any]:
    contexts = _parse_count_list(args.contexts)
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.prompt_file is None:
        args.prompt_file = work_dir / "generated-corpus.txt"
        _write_generated_corpus(args.prompt_file, contexts=contexts, chunks=args.chunks)
    llama_repo = _repo_from_binary(args.llama_perplexity)
    started = time.perf_counter()
    rows = [_run_row(args, ctx=ctx, work_dir=work_dir) for ctx in contexts]
    elapsed = time.perf_counter() - started
    all_completed = all(
        row["reference"]["returncode"] == 0
        and row["candidate"] is not None
        and row["candidate"]["returncode"] == 0
        for row in rows
    )
    all_parsed = all(bool(row["parsed_guard_metrics"]) for row in rows)
    all_passed = all(bool(row["passed_hipengine_like_quality_guard"]) for row in rows)
    status = "accepted" if all_completed and all_parsed and all_passed else "diagnostic_retained"
    return {
        "schema": 1,
        "status": status,
        "performance_claim": False,
        "mode": "llamacpp_q8_kv_quality_sweep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": _top_command(args),
        "model": str(args.model),
        "backend": "llamacpp_external",
        "llama_perplexity": str(args.llama_perplexity),
        "llama_cpp_repo": None if llama_repo is None else str(llama_repo),
        "llama_cpp_commit": None if llama_repo is None else _git_rev_parse(llama_repo),
        "contexts": [int(item) for item in contexts],
        "chunks": int(args.chunks),
        "prompt_file": str(args.prompt_file),
        "work_dir": str(work_dir),
        "reference_cache": {
            "type_k": args.reference_cache_type_k,
            "type_v": args.reference_cache_type_v,
        },
        "candidate_cache": {
            "type_k": args.candidate_cache_type_k,
            "type_v": args.candidate_cache_type_v,
        },
        "quality_thresholds": {
            "kl_mean_max": float(args.kl_threshold),
            "same_top_p_min": float(args.top1_threshold),
        },
        "rows": rows,
        "summary": {
            "all_processes_completed": bool(all_completed),
            "all_guard_metrics_parsed": bool(all_parsed),
            "all_rows_passed_hipengine_like_guard": bool(all_passed),
            "first_non_passing_row": next(
                (row["workload"] for row in rows if not row["passed_hipengine_like_quality_guard"]),
                None,
            ),
            "metrics_by_workload": {row["workload"]: row["metrics"] for row in rows},
        },
        "elapsed_seconds": float(elapsed),
        "caveats": [
            "External llama.cpp diagnostic, not a hipEngine correctness artifact or performance claim.",
            "llama-perplexity compares corpus logits and reports same-top-p, not hipEngine's fixed-prompt generated-token equality.",
            "The hipEngine-like pass bit uses mean KL plus same-top-p >= threshold; max KL is retained only as an outlier diagnostic.",
            "Large contexts create large temporary .kld logit files; this script deletes them unless --keep-logits is set.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-perplexity", type=Path, default=DEFAULT_LLAMA_PERPLEXITY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--contexts", default="4K", help="Comma-separated context sizes, e.g. 4K,8K,16K")
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--reference-cache-type-k", default="f16")
    parser.add_argument("--reference-cache-type-v", default="f16")
    parser.add_argument("--candidate-cache-type-k", default="q8_0")
    parser.add_argument("--candidate-cache-type-v", default="q8_0")
    parser.add_argument("--n-gpu-layers", default="99")
    parser.add_argument("--flash-attn", default="on")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--keep-logits", action="store_true")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--env", action="append", default=[], help="Extra process env as KEY=VALUE")
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.chunks <= 0:
        raise ValueError("--chunks must be positive")
    if args.batch_size <= 0 or args.ubatch_size <= 0:
        raise ValueError("--batch-size and --ubatch-size must be positive")
    args.env = _parse_env(args.env)
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0 if payload["summary"]["all_processes_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
