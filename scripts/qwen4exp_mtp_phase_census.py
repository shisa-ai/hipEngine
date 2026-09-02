#!/usr/bin/env python3
"""Measure Qwen4Exp MTP host/device phase boundaries across prompt categories."""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata

DEFAULT_MODEL = Path("/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL")
DEFAULT_SIDECAR = Path(
    "/models/gguf/Qwen3.8-Flash-Next-MTP-Q8_0/"
    "mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
)
DEFAULT_PROMPTS = ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--candidate-budget", type=int, default=2)
    parser.add_argument(
        "--draft-output-mode", choices=("compact", "debug", "both"), default="compact"
    )
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = []
    for category in CATEGORIES:
        matches = [row for row in rows if row.get("category") == category]
        if not matches:
            raise ValueError(f"prompt suite has no {category!r} row")
        selected.append(matches[0])
    return selected


def _prompt_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"prompt {row.get('id')!r} has no messages")
    return "\n".join(str(message["content"]) for message in messages)


def _summary(values: Sequence[float]) -> dict[str, float]:
    samples = [float(value) for value in values]
    return {
        "sum_ms": sum(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if args.max_tokens < 2:
        raise ValueError("max-tokens must be at least 2")
    if not 1 <= args.candidate_budget <= 4:
        raise ValueError("candidate-budget must be in 1..4")
    rows = _load_rows(args.prompts)
    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")
    if args.compiler_version_file is not None:
        os.environ.setdefault(
            "HIPENGINE_COMPILER_VERSION_FILE", str(args.compiler_version_file)
        )
    if args.require_cached_build:
        os.environ.setdefault("HIPENGINE_REQUIRE_CACHED_BUILD", "1")

    from hipengine import LLM, SamplingParams
    from hipengine.core.memory import memory_stats, reset_memory_stats

    reset_memory_stats()
    llm = LLM(
        str(args.model_root),
        backend="hip_gfx1151",
        quant="gguf_ud_q4_k_xl",
        execution_profile="production",
        max_sequence_length=1_024,
        speculative_provider="qwen4_exp_mtp",
        draft_model=str(args.sidecar),
        speculative_candidate_budget=args.candidate_budget,
    )
    output_rows: list[dict[str, Any]] = []
    try:
        modes = (
            ("debug", "compact")
            if args.draft_output_mode == "both"
            else (args.draft_output_mode,)
        )
        if args.draft_output_mode == "both":
            for mode in modes:
                os.environ["HIPENGINE_QWEN4_EXP_MTP_COMPACT_OUTPUT"] = (
                    "1" if mode == "compact" else "0"
                )
                llm.generate_speculative_mtp_detailed(
                    [_prompt_text(rows[0])],
                    SamplingParams(
                        max_tokens=2,
                        temperature=0.0,
                        top_k=1,
                        ignore_eos=True,
                    ),
                )
        for row_index, row in enumerate(rows):
            row_modes = modes if row_index % 2 == 0 else tuple(reversed(modes))
            if args.draft_output_mode == "both":
                for mode in row_modes:
                    os.environ["HIPENGINE_QWEN4_EXP_MTP_COMPACT_OUTPUT"] = (
                        "1" if mode == "compact" else "0"
                    )
                    llm.generate_speculative_mtp_detailed(
                        [_prompt_text(row)],
                        SamplingParams(
                            max_tokens=2,
                            temperature=0.0,
                            top_k=1,
                            ignore_eos=True,
                        ),
                    )
            for mode in row_modes:
                os.environ["HIPENGINE_QWEN4_EXP_MTP_COMPACT_OUTPUT"] = (
                    "1" if mode == "compact" else "0"
                )
                started = time.perf_counter()
                output = llm.generate_speculative_mtp_detailed(
                    [_prompt_text(row)],
                    SamplingParams(
                        max_tokens=args.max_tokens,
                        temperature=0.0,
                        top_k=1,
                        ignore_eos=True,
                    ),
                )[0]
                request_wall_ms = (time.perf_counter() - started) * 1_000.0
                if output.telemetry is None:
                    raise RuntimeError("Qwen4Exp MTP output has no telemetry")
                diagnostics = dict(output.telemetry.diagnostics)
                census = diagnostics.get("phase_census")
                if not isinstance(census, dict):
                    raise RuntimeError("Qwen4Exp MTP output has no phase census")
                output_rows.append(
                    {
                        "id": row["id"],
                        "category": row["category"],
                        "draft_output_mode": mode,
                        "request_wall_ms": request_wall_ms,
                        "generated_token_ids": list(output.generated_token_ids),
                        "proposed_draft_tokens": diagnostics["proposed_draft_tokens"],
                        "accepted_draft_tokens": diagnostics["accepted_draft_tokens"],
                        "draft_acceptance": diagnostics["draft_acceptance"],
                        "target_hidden_handoff": diagnostics.get(
                            "target_hidden_handoff", "host"
                        ),
                        "phase_census": census,
                    }
                )
    finally:
        llm.close()
    after_close = memory_stats()

    stage_names = sorted(
        {
            name
            for row in output_rows
            for name in row["phase_census"]["draft_stages_ms"]
        }
    )
    return {
        "schema": 1,
        "date": date.today().isoformat(),
        "kind": "qwen4exp_mtp_phase_census",
        "status": "passed" if after_close["current_allocated_bytes"] == 0 else "failed",
        "performance_claim": False,
        "command": list(command),
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "configuration": {
            "model_root": str(args.model_root),
            "sidecar": str(args.sidecar),
            "execution_profile": "production",
            "candidate_budget": args.candidate_budget,
            "max_tokens": args.max_tokens,
            "draft_output_mode": args.draft_output_mode,
            "mode_warmups": "one global and one per prompt/mode" if args.draft_output_mode == "both" else 0,
            "pair_order": "alternating AB/BA by category" if args.draft_output_mode == "both" else None,
            "categories": list(CATEGORIES),
        },
        "rows": output_rows,
        "aggregate": {
            "draft_stages": {
                name: _summary(
                    [row["phase_census"]["draft_stages_ms"].get(name, 0.0) for row in output_rows]
                )
                for name in stage_names
            },
            "proposal": _summary([row["phase_census"]["proposal"]["ms"] for row in output_rows]),
            "target_verify": _summary([row["phase_census"]["target_verify"]["ms"] for row in output_rows]),
            "acceptance_control": _summary([row["phase_census"]["acceptance_control"]["ms"] for row in output_rows]),
            "draft_commit_or_rollback": _summary([row["phase_census"]["draft_commit_or_rollback"]["ms"] for row in output_rows]),
            "request_wall_by_mode": {
                mode: _summary(
                    [
                        row["request_wall_ms"]
                        for row in output_rows
                        if row["draft_output_mode"] == mode
                    ]
                )
                for mode in sorted({row["draft_output_mode"] for row in output_rows})
            },
            "paired_output_ids_exact": all(
                next(
                    row["generated_token_ids"]
                    for row in output_rows
                    if row["id"] == prompt["id"]
                    and row["draft_output_mode"] == modes[0]
                )
                == next(
                    row["generated_token_ids"]
                    for row in output_rows
                    if row["id"] == prompt["id"]
                    and row["draft_output_mode"] == modes[-1]
                )
                for prompt in rows
            ),
        },
        "lifecycle": {"after_close": after_close, "passed": after_close["current_allocated_bytes"] == 0},
        "notes": [
            "Host wall stage timings are diagnostic and may overlap nested draft stages.",
            "Target verify includes hidden export; draft device synchronize precedes separate logits and hidden D2H timings.",
            "Current public provider uses no MTP graph and performs serial exact target verification plus cursor trim.",
        ],
    }


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args, command=[Path(sys.argv[0]).name, *sys.argv[1:]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
