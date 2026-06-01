#!/usr/bin/env python3
"""Audit hipfire DFlash token exactness against hipfire AR baseline.

This is a local diagnostic helper for comparing hipfire's
``dflash_spec_demo --ar-baseline`` output to its default DFlash path under the
same target, prompt bytes, tokenizer path, temperature, and max-token setting.
It does not judge task quality; it answers the narrower exact-greedy question:
did DFlash emit the same token IDs as target AR?
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_TOKEN_RE_TEMPLATE = r"{kind} tokens:\s*(\[[^\n]*\])"
_ROW_START_RE = re.compile(r"^@@@ ROW (\d+):\s*(.*?)\s*@@@$")
_ROW_END_RE = re.compile(r"^@@@ ROW (\d+) END @@@$")
_METRIC_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$")


@dataclass(frozen=True)
class PromptRow:
    index: int
    label: str
    prompt: str
    max_tokens: int
    source: dict[str, Any]


def load_prompt_rows(path: Path, *, default_max_tokens: int, max_prompts: int | None = None) -> list[PromptRow]:
    rows: list[PromptRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line or raw_line.startswith("#"):
                continue
            raw = json.loads(raw_line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            prompt = raw.get("prompt") or raw.get("prompt_text") or raw.get("text")
            if not isinstance(prompt, str):
                raise ValueError(f"{path}:{line_no}: missing prompt/prompt_text string")
            label = str(raw.get("label") or raw.get("id") or raw.get("name") or f"row{len(rows)}")
            if "@@@" in label or "\n" in label:
                raise ValueError(f"{path}:{line_no}: label/id must not contain '@@@' or newline")
            max_tokens = int(raw.get("max", raw.get("max_tokens", default_max_tokens)))
            if max_tokens <= 0:
                raise ValueError(f"{path}:{line_no}: max/max_tokens must be positive")
            rows.append(PromptRow(index=len(rows), label=label, prompt=prompt, max_tokens=max_tokens, source=raw))
            if max_prompts is not None and len(rows) >= max_prompts:
                break
    if not rows:
        raise ValueError(f"no prompt rows loaded from {path}")
    return rows


def write_hipfire_prompts(rows: Sequence[PromptRow], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"label": row.label, "prompt": row.prompt, "max": row.max_tokens}, ensure_ascii=False) + "\n")


def parse_token_rows(stderr_text: str, *, kind: str) -> tuple[dict[int, list[int]], dict[int, dict[str, float]], dict[int, str]]:
    """Parse AR/DFlash token vectors and simple metrics from dflash_spec_demo stderr."""

    token_re = re.compile(_TOKEN_RE_TEMPLATE.format(kind=re.escape(kind)))
    current_idx = 0
    current_label = ""
    tokens: dict[int, list[int]] = {}
    metrics: dict[int, dict[str, float]] = {}
    labels: dict[int, str] = {}
    for line in stderr_text.splitlines():
        start = _ROW_START_RE.match(line)
        if start:
            current_idx = int(start.group(1))
            current_label = start.group(2)
            labels[current_idx] = current_label
            metrics.setdefault(current_idx, {})
            continue
        end = _ROW_END_RE.match(line)
        if end:
            current_idx = 0
            current_label = ""
            continue
        match = token_re.search(line)
        if match:
            parsed = ast.literal_eval(match.group(1))
            if not isinstance(parsed, list):
                raise ValueError(f"{kind} token line did not parse to a list: {line[:200]}")
            tokens[current_idx] = [int(x) for x in parsed]
            if current_label:
                labels[current_idx] = current_label
            continue
        metric = _METRIC_RE.match(line)
        if metric:
            key = metric.group(1)
            value = float(metric.group(2))
            metrics.setdefault(current_idx, {})[key] = value
    return tokens, metrics, labels


def first_mismatch(left: Sequence[int], right: Sequence[int], *, limit: int | None = None) -> int | None:
    n = min(len(left), len(right)) if limit is None else min(len(left), len(right), int(limit))
    for idx in range(n):
        if int(left[idx]) != int(right[idx]):
            return idx
    return None


def compare_token_rows(rows: Sequence[PromptRow], ar_tokens: Mapping[int, Sequence[int]], dflash_tokens: Mapping[int, Sequence[int]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        ar = [int(x) for x in ar_tokens.get(row.index, ())]
        spec = [int(x) for x in dflash_tokens.get(row.index, ())]
        shared_len = min(len(ar), len(spec), row.max_tokens)
        mismatch = first_mismatch(ar, spec, limit=shared_len)
        strict_exact = bool(ar) and ar == spec
        prefix_equal_to_shared = mismatch is None
        dflash_prefix_matches_ar = len(spec) >= len(ar) and spec[: len(ar)] == ar
        ar_prefix_matches_dflash = len(ar) >= len(spec) and ar[: len(spec)] == spec
        comparison: dict[str, Any] = {
            "row_index": row.index,
            "label": row.label,
            "max_tokens": row.max_tokens,
            "ar_len": len(ar),
            "dflash_len": len(spec),
            "shared_len": shared_len,
            "strict_exact": strict_exact,
            "prefix_equal_to_shared_len": prefix_equal_to_shared,
            "dflash_prefix_matches_full_ar": dflash_prefix_matches_ar,
            "ar_prefix_matches_full_dflash": ar_prefix_matches_dflash,
            "hard_mismatch_before_shared_len": mismatch is not None,
            "first_mismatch_index": mismatch,
            "over_emitted_vs_max": len(spec) > row.max_tokens,
            "over_emitted_vs_ar": len(spec) > len(ar),
            "under_emitted_vs_ar": len(spec) < len(ar),
        }
        if mismatch is not None:
            comparison["ar_token_at_mismatch"] = ar[mismatch]
            comparison["dflash_token_at_mismatch"] = spec[mismatch]
        comparisons.append(comparison)

    total = len(comparisons)
    aggregate = {
        "rows": total,
        "strict_exact_rows": sum(1 for row in comparisons if row["strict_exact"]),
        "prefix_equal_to_shared_len_rows": sum(1 for row in comparisons if row["prefix_equal_to_shared_len"]),
        "dflash_prefix_matches_full_ar_rows": sum(1 for row in comparisons if row["dflash_prefix_matches_full_ar"]),
        "hard_mismatch_before_shared_len_rows": sum(1 for row in comparisons if row["hard_mismatch_before_shared_len"]),
        "over_emitted_vs_max_rows": sum(1 for row in comparisons if row["over_emitted_vs_max"]),
        "over_emitted_vs_ar_rows": sum(1 for row in comparisons if row["over_emitted_vs_ar"]),
        "missing_ar_rows": sum(1 for row in comparisons if row["ar_len"] == 0),
        "missing_dflash_rows": sum(1 for row in comparisons if row["dflash_len"] == 0),
    }
    aggregate["all_strict_exact"] = aggregate["strict_exact_rows"] == total
    aggregate["all_prefix_equal_to_shared_len"] = aggregate["prefix_equal_to_shared_len_rows"] == total
    aggregate["all_dflash_prefix_matches_full_ar"] = aggregate["dflash_prefix_matches_full_ar_rows"] == total
    aggregate["all_exact_speculative_decode"] = aggregate["all_strict_exact"]
    return comparisons, aggregate


def parse_env(values: Iterable[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError("--env key must be non-empty")
        env[key] = value
    return env


def run_demo(
    *,
    demo: Path,
    target: Path,
    draft: Path,
    prompts_file: Path,
    ctx: int,
    kv_mode: str,
    temp: float,
    seed: int,
    no_chatml: bool,
    ar_baseline: bool,
    extra_args: Sequence[str],
    env: Mapping[str, str],
    timeout_s: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(demo),
        "--target",
        str(target),
        "--draft",
        str(draft),
        "--prompts-file",
        str(prompts_file),
        "--ctx",
        str(ctx),
        "--kv-mode",
        kv_mode,
        "--temp",
        str(temp),
        "--seed",
        str(seed),
    ]
    if no_chatml:
        cmd.append("--no-chatml")
    if ar_baseline:
        cmd.append("--ar-baseline")
    cmd.extend(str(arg) for arg in extra_args)
    proc_env = os.environ.copy()
    proc_env.setdefault("HIPFIRE_DFLASH_LOOP_BREAK", "off")
    proc_env.update(env)
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        env=proc_env,
        timeout=timeout_s,
    )


def _write_raw_outputs(save_dir: Path, *, mode: str, proc: subprocess.CompletedProcess[str]) -> dict[str, str]:
    stdout_path = save_dir / f"{mode}.stdout.txt"
    stderr_path = save_dir / f"{mode}.stderr.txt"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return {"stdout": str(stdout_path), "stderr": str(stderr_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, default=Path("/tmp/hipfire-target/release/examples/dflash_spec_demo"), help="Path to hipfire dflash_spec_demo binary")
    parser.add_argument("--target", type=Path, required=True, help="hipfire target model path")
    parser.add_argument("--draft", type=Path, required=True, help="hipfire DFlash draft model path")
    parser.add_argument("--prompts", type=Path, default=Path("fixtures/dflash/stable_prompts.jsonl"), help="JSONL prompts with prompt_text/prompt and id/label")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit rows loaded from --prompts")
    parser.add_argument("--max", type=int, default=128, help="Default max tokens for prompt rows that do not set max/max_tokens")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--kv-mode", default="q8")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-chatml", action="store_true", help="Pass --no-chatml to hipfire for both AR and DFlash runs")
    parser.add_argument("--env", action="append", default=[], help="Extra environment KEY=VALUE; may be repeated")
    parser.add_argument("--ar-extra", action="append", default=[], help="Extra argument for AR-baseline run; may be repeated")
    parser.add_argument("--dflash-extra", action="append", default=[], help="Extra argument for DFlash run; may be repeated")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional directory for normalized prompts and raw stdout/stderr. Raw logs are omitted from the artifact by default.")
    parser.add_argument("--json", type=Path, required=True, help="Output audit JSON")
    args = parser.parse_args(argv)

    if args.max <= 0:
        raise ValueError("--max must be positive")
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive when set")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")

    rows = load_prompt_rows(args.prompts, default_max_tokens=args.max, max_prompts=args.max_prompts)
    extra_env = parse_env(args.env)

    retain_raw = args.save_dir is not None
    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if retain_raw:
        save_dir = args.save_dir
        assert save_dir is not None
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="hipfire-dflash-exactness-")
        save_dir = Path(temp_dir_obj.name)
    normalized_prompts = save_dir / "hipfire_prompts.jsonl"
    write_hipfire_prompts(rows, normalized_prompts)

    ar_proc = run_demo(
        demo=args.demo,
        target=args.target,
        draft=args.draft,
        prompts_file=normalized_prompts,
        ctx=args.ctx,
        kv_mode=args.kv_mode,
        temp=args.temp,
        seed=args.seed,
        no_chatml=args.no_chatml,
        ar_baseline=True,
        extra_args=args.ar_extra,
        env=extra_env,
        timeout_s=args.timeout_s,
    )
    dflash_proc = run_demo(
        demo=args.demo,
        target=args.target,
        draft=args.draft,
        prompts_file=normalized_prompts,
        ctx=args.ctx,
        kv_mode=args.kv_mode,
        temp=args.temp,
        seed=args.seed,
        no_chatml=args.no_chatml,
        ar_baseline=False,
        extra_args=args.dflash_extra,
        env=extra_env,
        timeout_s=args.timeout_s,
    )

    raw_output_paths = {
        "ar": _write_raw_outputs(save_dir, mode="ar", proc=ar_proc),
        "dflash": _write_raw_outputs(save_dir, mode="dflash", proc=dflash_proc),
        "normalized_prompts": str(normalized_prompts),
    }
    raw_outputs: dict[str, Any] = (
        {"retained": True, **raw_output_paths}
        if retain_raw
        else {"retained": False, "note": "raw stdout/stderr omitted; rerun with --save-dir to retain them"}
    )

    if ar_proc.returncode != 0 or dflash_proc.returncode != 0:
        artifact = {
            "status": "failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "returncodes": {"ar": ar_proc.returncode, "dflash": dflash_proc.returncode},
            "raw_outputs": raw_outputs,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "failed", "returncodes": artifact["returncodes"], "json": str(args.json)}, sort_keys=True))
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()
        return 1

    ar_tokens, ar_metrics, ar_labels = parse_token_rows(ar_proc.stderr, kind="AR")
    dflash_tokens, dflash_metrics, dflash_labels = parse_token_rows(dflash_proc.stderr, kind="DFlash")
    comparisons, aggregate = compare_token_rows(rows, ar_tokens, dflash_tokens)

    command_prompts_file = str(normalized_prompts) if retain_raw else "<internal normalized prompts jsonl>"
    command_base = [
        str(args.demo),
        "--target",
        str(args.target),
        "--draft",
        str(args.draft),
        "--prompts-file",
        command_prompts_file,
        "--ctx",
        str(args.ctx),
        "--kv-mode",
        args.kv_mode,
        "--temp",
        str(args.temp),
        "--seed",
        str(args.seed),
    ]
    if args.no_chatml:
        command_base.append("--no-chatml")
    artifact = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": "hipfire DFlash token-id exactness vs hipfire target AR baseline",
        "models": {"target": str(args.target), "draft": str(args.draft)},
        "config": {
            "ctx": args.ctx,
            "kv_mode": args.kv_mode,
            "temp": args.temp,
            "seed": args.seed,
            "no_chatml": bool(args.no_chatml),
            "default_max_tokens": args.max,
            "env": {"HIPFIRE_DFLASH_LOOP_BREAK": os.environ.get("HIPFIRE_DFLASH_LOOP_BREAK", "off"), **extra_env},
            "ar_extra": list(args.ar_extra),
            "dflash_extra": list(args.dflash_extra),
        },
        "commands": {
            "audit": " ".join(shlex.quote(x) for x in ["python3", "scripts/hipfire_dflash_exactness_audit.py", *(argv if argv is not None else sys.argv[1:])]),
            "ar": " ".join(shlex.quote(x) for x in [*command_base, "--ar-baseline", *args.ar_extra]),
            "dflash": " ".join(shlex.quote(x) for x in [*command_base, *args.dflash_extra]),
        },
        "raw_outputs": raw_outputs,
        "aggregate": aggregate,
        "rows": [
            {
                **comparison,
                "prompt_source": {
                    "id": row.source.get("id"),
                    "benchmark_group": row.source.get("benchmark_group"),
                    "category": row.source.get("category"),
                    "prompt_text_sha256": row.source.get("prompt_text_sha256"),
                    "prompt_ids_sha256": row.source.get("prompt_ids_sha256"),
                },
                "ar_metrics": ar_metrics.get(row.index, {}),
                "dflash_metrics": dflash_metrics.get(row.index, {}),
                "ar_label": ar_labels.get(row.index),
                "dflash_label": dflash_labels.get(row.index),
            }
            for row, comparison in zip(rows, comparisons, strict=True)
        ],
        "notes": [
            "strict_exact requires identical AR and DFlash token-id vectors",
            "hard_mismatch_before_shared_len means DFlash emitted a token different from target AR before either output ended",
            "prefix equality with over-emission is not an exact-greedy pass, but can be reported separately after truncation",
        ],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", **aggregate, "json": str(args.json)}, sort_keys=True))

    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()
    return 0 if aggregate["all_strict_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
