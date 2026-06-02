#!/usr/bin/env python3
"""Run Qwen/PARO generated-token equality checks for multiple c>N sizes.

This is a correctness-only orchestration helper for the concurrency punchlist. It
runs ``qwen35_batch_retained_bench.py`` for each requested batch size, extracts
its generated-token equality section, and writes a compact summary artifact. The
retained bench may mark per-row diagnostic fallback runs as ``blocked`` for
performance claims; this helper intentionally treats only generated-token
agreement vs independent c=1 as the pass/fail criterion and always records
``performance_claim=false``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RETAINED_BENCH = Path("scripts/qwen35_batch_retained_bench.py")
DEFAULT_BATCH_SIZES = (2, 4, 8)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    sizes: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise argparse.ArgumentTypeError("--batch-sizes must not contain blank entries")
        try:
            size = int(part, 10)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--batch-sizes entries must be integers") from exc
        if size <= 0:
            raise argparse.ArgumentTypeError("--batch-sizes entries must be positive")
        if size in sizes:
            raise argparse.ArgumentTypeError("--batch-sizes entries must be unique")
        sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("--batch-sizes must not be empty")
    return tuple(sizes)


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _format_batch_template(value: str | Path, *, batch_size: int, option: str) -> str:
    text = str(value)
    try:
        return text.format(batch_size=batch_size, c=batch_size)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"{option} supports only {{batch_size}}/{{c}} placeholders") from exc


def _command_env_prefix_parts() -> tuple[str, ...]:
    value = os.environ.get("HIP_VISIBLE_DEVICES")
    if value is None or not value.strip():
        return ()
    return ("env", f"HIP_VISIBLE_DEVICES={value}")


def _display_command(argv: Sequence[str]) -> str:
    return shlex.join((*_command_env_prefix_parts(), *argv))


def _prefix_lengths_from_sequences(eq: dict[str, Any]) -> list[int]:
    batch = eq.get("batch_sequences")
    c1 = eq.get("c1_sequences")
    if not isinstance(batch, list) or not isinstance(c1, list) or len(batch) != len(c1):
        return []
    prefixes: list[int] = []
    for batch_row, c1_row in zip(batch, c1):
        if not isinstance(batch_row, list) or not isinstance(c1_row, list):
            return []
        prefix = 0
        for batch_token, c1_token in zip(batch_row, c1_row):
            if batch_token != c1_token:
                break
            prefix += 1
        prefixes.append(prefix)
    return prefixes


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _append_unique(strings: list[str], value: str) -> None:
    if value and value not in strings:
        strings.append(value)


def _equality_summary(artifact_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact exception is platform/input dependent.
        return {
            "passed": False,
            "artifact_load_error": str(exc),
            "artifact_path": str(artifact_path),
            "retained_ready": False,
        }
    eq = payload.get("correctness", {}).get("generated_token_equality", {})
    if not isinstance(eq, dict):
        eq = {}
    workload = payload.get("workload", {})
    if not isinstance(workload, dict):
        workload = {}
    batch_execution = payload.get("execution", {}).get("batch_execution", {})
    if not isinstance(batch_execution, dict):
        batch_execution = {}
    decode_execution = batch_execution.get("decode_execution", {})
    if not isinstance(decode_execution, dict):
        decode_execution = {}
    sampler_execution = decode_execution.get("sampler_execution", {})
    if not isinstance(sampler_execution, dict):
        sampler_execution = {}
    blockers: list[str] = []
    for blocker in _string_list(batch_execution.get("blockers")):
        _append_unique(blockers, blocker)
    for blocker in _string_list(decode_execution.get("blockers")):
        _append_unique(blockers, blocker)
    decision = payload.get("decision")
    if isinstance(decision, dict) and decision.get("accepted") is not True and isinstance(decision.get("reason"), str):
        _append_unique(blockers, decision["reason"])
    prefix_lengths = eq.get("prefix_lengths")
    if not (
        isinstance(prefix_lengths, list)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in prefix_lengths)
    ):
        prefix_lengths = _prefix_lengths_from_sequences(eq)
    first_mismatch_indices = eq.get("first_mismatch_indices")
    if not isinstance(first_mismatch_indices, list):
        first_mismatch_indices = []
    passed = bool(eq.get("passed") is True)
    retained_ready = bool(payload.get("status") == "accepted" and payload.get("performance_claim") is True)
    return {
        "passed": passed,
        "retained_ready": retained_ready,
        "retained_artifact_status": payload.get("status"),
        "performance_claim": payload.get("performance_claim"),
        "artifact_path": str(artifact_path),
        "min_equal_prefix_tokens": min(prefix_lengths) if prefix_lengths else 0,
        "prefix_lengths": prefix_lengths,
        "first_mismatch_indices": first_mismatch_indices,
        "retained_blockers": blockers,
        "workload": {
            "batch_decode_linear_path": workload.get("batch_decode_linear_path"),
            "batch_decode_full_attention_path": workload.get("batch_decode_full_attention_path"),
            "batch_decode_linear_output_path": workload.get("batch_decode_linear_output_path"),
            "batch_decode_full_attention_output_path": workload.get("batch_decode_full_attention_output_path"),
            "batch_decode_full_attention_layer_copy": workload.get("batch_decode_full_attention_layer_copy"),
            "batch_decode_full_attention_moe_path": workload.get("batch_decode_full_attention_moe_path"),
            "batch_decode_post_attention_path": workload.get("batch_decode_post_attention_path"),
            "batch_sample_mode": workload.get("batch_sample_mode"),
            "batch_sample_eq_ok": workload.get("batch_sample_eq_ok"),
            "batch_sample_eq_artifact": workload.get("batch_sample_eq_artifact"),
            "batch_sample_eq_rows": workload.get("batch_sample_eq_rows"),
            "native_caware_decode": workload.get("native_caware_decode"),
        },
        "decode_execution": {
            "native_caware_decode": decode_execution.get("native_caware_decode"),
            "linear_attention_decode_path": decode_execution.get("linear_attention_decode_path"),
            "linear_attention_projection_path": decode_execution.get("linear_attention_projection_path"),
            "linear_attention_state_path": decode_execution.get("linear_attention_state_path"),
            "linear_attention_output_path": decode_execution.get("linear_attention_output_path"),
            "full_attention_decode_path": decode_execution.get("full_attention_decode_path"),
            "full_attention_output_path": decode_execution.get("full_attention_output_path"),
            "post_attention_decode_path": decode_execution.get("post_attention_decode_path"),
            "moe_decode_path": decode_execution.get("moe_decode_path"),
        },
        "sampler_execution": {
            "rows": sampler_execution.get("rows"),
            "requested_mode": sampler_execution.get("requested_mode"),
            "mode": sampler_execution.get("mode"),
            "native_row_aware_lm_head": sampler_execution.get("native_row_aware_lm_head"),
            "equality_artifact": sampler_execution.get("equality_artifact"),
            "equality_rows": sampler_execution.get("equality_rows"),
            "blockers": sampler_execution.get("blockers"),
        },
    }


def _retained_command(args: argparse.Namespace, batch_size: int, artifact_path: Path) -> list[str]:
    argv = [
        sys.executable,
        str(RETAINED_BENCH),
        "--model",
        str(args.model),
        "--fixture",
        str(args.fixture),
        "--prompt-length",
        str(args.prompt_length),
        "--batch-size",
        str(batch_size),
        "--decode-tokens",
        str(args.decode_tokens),
        "--warmup-decode-tokens",
        str(args.warmup_decode_tokens),
        "--max-layers",
        str(args.max_layers),
        "--json",
        str(artifact_path),
    ]
    if args.compiler_version_file is not None:
        argv.extend(("--compiler-version-file", str(args.compiler_version_file)))
    if args.require_cached_build:
        argv.append("--require-cached-build")
    if args.batch_decode_moe_path != "default":
        argv.extend(("--batch-decode-moe-path", str(args.batch_decode_moe_path)))
    if args.batch_decode_linear_path != "default":
        argv.extend(("--batch-decode-linear-path", str(args.batch_decode_linear_path)))
    if args.batch_decode_linear_projection_path != "default":
        argv.extend(("--batch-decode-linear-projection-path", str(args.batch_decode_linear_projection_path)))
    if args.batch_decode_linear_state_path != "default":
        argv.extend(("--batch-decode-linear-state-path", str(args.batch_decode_linear_state_path)))
    if args.batch_decode_linear_moe_path != "default":
        argv.extend(("--batch-decode-linear-moe-path", str(args.batch_decode_linear_moe_path)))
    if args.batch_decode_linear_output_path != "default":
        argv.extend(("--batch-decode-linear-output-path", str(args.batch_decode_linear_output_path)))
    if args.batch_decode_full_attn_path != "default":
        argv.extend(("--batch-decode-full-attn-path", str(args.batch_decode_full_attn_path)))
    if args.batch_decode_full_attn_output_path != "default":
        argv.extend(("--batch-decode-full-attn-output-path", str(args.batch_decode_full_attn_output_path)))
    if args.batch_decode_full_attn_layer_copy != "default":
        argv.extend(("--batch-decode-full-attn-layer-copy", str(args.batch_decode_full_attn_layer_copy)))
    if args.batch_decode_full_attn_moe_path != "default":
        argv.extend(("--batch-decode-full-attn-moe-path", str(args.batch_decode_full_attn_moe_path)))
    if args.batch_decode_post_attn_path != "default":
        argv.extend(("--batch-decode-post-attn-path", str(args.batch_decode_post_attn_path)))
    if args.batch_sample_mode != "default":
        argv.extend(("--batch-sample-mode", str(args.batch_sample_mode)))
    if args.batch_sample_eq_ok:
        argv.append("--batch-sample-eq-ok")
    if args.batch_sample_eq_artifact_template is not None:
        argv.extend(
            (
                "--batch-sample-eq-artifact",
                _format_batch_template(
                    args.batch_sample_eq_artifact_template,
                    batch_size=batch_size,
                    option="--batch-sample-eq-artifact-template",
                ),
            )
        )
    if args.batch_sample_eq_rows is not None:
        argv.extend(
            (
                "--batch-sample-eq-rows",
                _format_batch_template(
                    args.batch_sample_eq_rows,
                    batch_size=batch_size,
                    option="--batch-sample-eq-rows",
                ),
            )
        )
    elif args.batch_sample_eq_ok:
        argv.extend(("--batch-sample-eq-rows", str(batch_size)))
    return argv


def _run_one(args: argparse.Namespace, batch_size: int, output_dir: Path, *, repeat_index: int) -> dict[str, Any]:
    repeat_suffix = f"-r{repeat_index}" if args.repeat_runs > 1 else ""
    artifact_path = output_dir / f"native-equality-c{batch_size}-p{args.prompt_length}-d{args.decode_tokens}{repeat_suffix}.json"
    log_path = artifact_path.with_suffix(".log")
    argv = _retained_command(args, batch_size, artifact_path)
    command = _display_command(argv)
    row: dict[str, Any] = {
        "batch_size": batch_size,
        "repeat_index": repeat_index,
        "artifact_path": str(artifact_path),
        "log_path": str(log_path),
        "command": command,
    }
    if args.dry_run:
        row.update({"status": "planned", "returncode": None, "duration_seconds": 0.0})
        return row

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(argv, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    row["duration_seconds"] = time.monotonic() - started
    row["returncode"] = completed.returncode
    equality = _equality_summary(artifact_path)
    row["generated_token_equality"] = equality
    row["status"] = "passed" if equality.get("passed") is True else "failed"
    return row


def _build_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if args.dry_run:
        status = "planned"
        equality_passed = False
    else:
        equality_passed = bool(rows) and all(row.get("generated_token_equality", {}).get("passed") is True for row in rows)
        status = "passed" if equality_passed else "failed"
    retained_ready = bool(rows) and all(
        row.get("generated_token_equality", {}).get("retained_ready") is True for row in rows
    )
    return {
        "schema": 1,
        "created_at": _utc_now(),
        "status": status,
        "performance_claim": False,
        "generated_equality_passed": equality_passed,
        "retained_ready": retained_ready,
        "device": {"env": {"HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES")}},
        "workload": {
            "batch_sizes": list(args.batch_sizes),
            "prompt_length": args.prompt_length,
            "decode_tokens": args.decode_tokens,
            "warmup_decode_tokens": args.warmup_decode_tokens,
            "max_layers": args.max_layers,
            "repeat_runs": args.repeat_runs,
            "batch_decode_moe_path": args.batch_decode_moe_path,
            "batch_decode_linear_path": args.batch_decode_linear_path,
            "batch_decode_linear_projection_path": args.batch_decode_linear_projection_path,
            "batch_decode_linear_state_path": args.batch_decode_linear_state_path,
            "batch_decode_linear_moe_path": args.batch_decode_linear_moe_path,
            "batch_decode_linear_output_path": args.batch_decode_linear_output_path,
            "batch_decode_full_attn_path": args.batch_decode_full_attn_path,
            "batch_decode_full_attn_output_path": args.batch_decode_full_attn_output_path,
            "batch_decode_full_attn_layer_copy": args.batch_decode_full_attn_layer_copy,
            "batch_decode_full_attn_moe_path": args.batch_decode_full_attn_moe_path,
            "batch_decode_post_attn_path": args.batch_decode_post_attn_path,
            "batch_sample_mode": args.batch_sample_mode,
            "batch_sample_eq_ok": args.batch_sample_eq_ok,
            "batch_sample_eq_artifact_template": str(args.batch_sample_eq_artifact_template or ""),
            "batch_sample_eq_rows": args.batch_sample_eq_rows,
        },
        "commands": rows,
        "notes": (
            "Correctness-only c>N generated-token equality matrix vs independent c=1. "
            "Retained artifacts may be blocked for performance claims when diagnostic per-row fallbacks are active."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")
    parser.add_argument("--fixture", default="/tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-decode-tokens", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--repeat-runs", type=_parse_positive_int, default=1)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/hipengine-e2e-native-equality-matrix"))
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-decode-moe-path", choices=("default", "grouped_compact", "selected_c1"), default="default")
    parser.add_argument("--batch-decode-linear-path", choices=("default", "batch_segments", "per_row"), default="default")
    parser.add_argument(
        "--batch-decode-linear-projection-path",
        choices=(
            "default",
            "auto",
            "batch",
            "batch_gemv",
            "selected_c1",
            "selected_qkv_z",
            "selected_qkv_z_input",
            "selected_qkv",
            "selected_z",
            "selected_ab",
            "batch_gemv_selected_ab",
        ),
        default="default",
    )
    parser.add_argument(
        "--batch-decode-linear-state-path",
        choices=("default", "batch_segments", "selected_c1"),
        default="default",
    )
    parser.add_argument(
        "--batch-decode-linear-moe-path",
        choices=("default", "grouped_compact", "per_row_c1"),
        default="default",
    )
    parser.add_argument(
        "--batch-decode-linear-output-path",
        choices=("default", "auto", "batch", "batch_gemv", "selected_c1"),
        default="default",
    )
    parser.add_argument("--batch-decode-full-attn-path", choices=("default", "native_batch", "per_row"), default="default")
    parser.add_argument(
        "--batch-decode-full-attn-output-path",
        choices=("default", "batch", "batch_gemv", "per_row"),
        default="default",
    )
    parser.add_argument(
        "--batch-decode-full-attn-layer-copy",
        choices=("default", "batch", "per_row"),
        default="default",
    )
    parser.add_argument(
        "--batch-decode-full-attn-moe-path",
        choices=("default", "grouped_compact", "per_row_c1"),
        default="default",
    )
    parser.add_argument("--batch-decode-post-attn-path", choices=("default", "batch", "per_row"), default="default")
    parser.add_argument(
        "--batch-sample-mode",
        choices=("default", "serial_lm_head", "batched_lm_head"),
        default="default",
        help="Retained-bench sampler/LM-head path to pass through; default leaves retained-bench defaults unchanged.",
    )
    parser.add_argument("--batch-sample-eq-ok", action="store_true")
    parser.add_argument(
        "--batch-sample-eq-artifact-template",
        help="Template for --batch-sample-eq-artifact; supports {batch_size} or {c} placeholders.",
    )
    parser.add_argument(
        "--batch-sample-eq-rows",
        help="Template/value for --batch-sample-eq-rows; defaults to the current batch size when --batch-sample-eq-ok is set.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        _run_one(args, batch_size, args.output_dir, repeat_index=repeat_index)
        for repeat_index in range(args.repeat_runs)
        for batch_size in args.batch_sizes
    ]
    summary = _build_summary(args, rows)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
        return 0
    return 0 if summary["generated_equality_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
