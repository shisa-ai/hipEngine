#!/usr/bin/env python3
"""Frozen multi-sequence G0/G1/G2 distribution evaluator for the Qwen3.8 DMS campaign.

Phase A campaign evaluator (docs/campaigns/DMS-SELECTOR-IMPROVEMENT.md
Section 4).  Loads one sealed data manifest, filters by split and category,
requires the declared number of sequences per category, and evaluates every
selected sequence separately: one dense BF16-KV teacher trajectory per
sequence, then ``no_evict`` and/or ``sidecar`` candidate modes replaying the
exact dense-teacher input tokens.  Full-vocabulary prefill and decode rows are
scored against the frozen G0/G1/G2 gates in
``hipengine.benchmark.dms_campaign``.  The evaluator is independent of any
candidate implementation; it consumes logits and decides pass/fail.

This tool is offline evaluation only; it is not a production routing surface
and never conditions runtime routing on prompt identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.benchmark.dms_campaign import (
    CATEGORIES,
    GATE_THRESHOLDS,
    compare_row,
    dms_digest,
    evaluate_gate,
    load_manifest,
    sequence_correlations,
)

SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    return {
        "commit": commit,
        "scoped_dirty_diff": sorted(
            line[3:] for line in dirty.splitlines() if line[3:].strip()
        ),
    }


def _evaluator_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    library = root / "hipengine" / "benchmark" / "dms_campaign.py"
    script = Path(__file__)
    return {
        "library_path": str(library),
        "library_sha256": _sha256(library),
        "script_path": str(script),
        "script_sha256": _sha256(script),
    }


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--split", required=True,
                        help="explicit manifest split filter, e.g. qualification or final-32k")
    parser.add_argument("--categories", default=",".join(CATEGORIES))
    parser.add_argument("--expected-sequences-per-category", type=int, required=True,
                        help="declared sequence count each selected category must supply exactly")
    parser.add_argument("--prompt-tokens", type=int,
                        help="optional per-sequence prompt truncation; default uses full manifest tokens")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--modes", default="no_evict,sidecar")
    parser.add_argument("--diagnostic-injection-dir", type=Path,
                        help="sealed evaluator-only injection directory; files are <sequence_id>.json")
    parser.add_argument("--codec", choices=("bf16", "int8_evaluation"), default="bf16",
                        help="Offline candidate codec; INT8 evaluation does not qualify serving.")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--gate", choices=tuple(GATE_THRESHOLDS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-fail", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Heavy runtime imports stay inside run() so tests and help text never load
    # the model stack.
    from hipengine.core.memory import memory_stats
    from hipengine.kvcache.dms import (
        create_dms_bf16_backend,
        create_dms_int8_evaluation_backend,
    )
    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFFullStackRunner,
        Qwen35GGUFResidentSession,
    )

    decode_steps = int(args.decode_steps)
    if decode_steps <= 0:
        raise ValueError("decode-steps must be positive")
    if int(args.expected_sequences_per_category) <= 0:
        raise ValueError("expected-sequences-per-category must be positive")
    categories = _parse_csv(args.categories)
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("categories must be a non-empty unique list")
    if any(category not in CATEGORIES for category in categories):
        raise ValueError("categories contains an unsupported DMS category")
    modes = _parse_csv(args.modes)
    if not modes or any(mode not in {"no_evict", "sidecar", "diagnostic"} for mode in modes):
        raise ValueError("modes must be a comma-separated subset of no_evict,sidecar,diagnostic")
    if "diagnostic" in modes and args.diagnostic_injection_dir is None:
        raise ValueError("diagnostic mode requires --diagnostic-injection-dir")
    if "diagnostic" not in modes and args.diagnostic_injection_dir is not None:
        raise ValueError("--diagnostic-injection-dir requires diagnostic mode")
    codec = str(args.codec)
    backend_factory = {
        "bf16": create_dms_bf16_backend,
        "int8_evaluation": create_dms_int8_evaluation_backend,
    }[codec]

    sequences = load_manifest(
        args.data_manifest,
        split=str(args.split),
        categories=categories,
        expected_sequences_per_category=int(args.expected_sequences_per_category),
        prompt_tokens=int(args.prompt_tokens) if args.prompt_tokens else None,
    )
    sequences.sort(key=lambda record: str(record["sequence_id"]))
    correlations = sequence_correlations(sequences)

    started = time.perf_counter()
    memory_baseline = memory_stats()
    runner = Qwen35GGUFFullStackRunner(args.model, backend=str(args.backend))
    loaded_at = time.perf_counter()
    sequence_results: dict[str, dict[str, Any]] = {}
    try:
        for record in sequences:
            prompt = record["token_ids"]
            prompt_digest = hashlib.sha256(
                np.asarray(prompt, dtype=np.int64).tobytes()
            ).hexdigest()
            max_positions = len(prompt) + decode_steps
            teacher_logits: list[np.ndarray] = []
            teacher_inputs: list[int] = []
            teacher_rows: list[dict[str, Any]] = []
            dense_started = time.perf_counter()
            with Qwen35GGUFResidentSession(
                args.model,
                backend=str(args.backend),
                shared_runner=runner,
                max_sequence_length=max_positions,
                use_wmma_prefill=True,
                use_gemv_decode=True,
            ) as dense:
                seed = dense.prefill(
                    prompt,
                    use_bulk=True,
                    bulk_attention_mode="bulk",
                    return_logits=True,
                )
                teacher_prefill = seed.logits.copy()
                current = int(seed.token_id)
                for step in range(decode_steps):
                    teacher_inputs.append(current)
                    step_started = time.perf_counter()
                    result = dense.step(current, return_logits=True)
                    teacher_logits.append(result.logits.copy())
                    teacher_rows.append(
                        {
                            "step": step,
                            "input_token": current,
                            "output_token": int(result.token_id),
                            "seconds": time.perf_counter() - step_started,
                        }
                    )
                    current = int(result.token_id)
            dense_seconds = time.perf_counter() - dense_started

            candidates: dict[str, dict[str, Any]] = {}
            for mode in modes:
                mode_started = time.perf_counter()
                diagnostic_injection_path = (
                    Path(args.diagnostic_injection_dir) / f"{record['sequence_id']}.json"
                    if mode == "diagnostic" else None
                )
                with Qwen35GGUFResidentSession(
                    args.model,
                    backend=str(args.backend),
                    shared_runner=runner,
                    max_sequence_length=max_positions,
                    dms_metadata_path=args.metadata,
                    dms_max_new_tokens=decode_steps,
                    dms_decision_mode=mode,
                    dms_diagnostic_injection_path=diagnostic_injection_path,
                    dms_backend_factory=backend_factory,
                    use_wmma_prefill=True,
                    use_gemv_decode=True,
                ) as candidate:
                    seed = candidate.prefill(
                        prompt,
                        use_bulk=True,
                        bulk_attention_mode="bulk",
                        return_logits=True,
                    )
                    if candidate._dms_dense_prefill_pool is not None:
                        raise AssertionError(
                            f"DMS candidate ({mode}) retained dense prefill pool"
                        )
                    prefill_row = compare_row(teacher_prefill, seed.logits)
                    prefill_row.update(
                        {
                            "sequence_id": record["sequence_id"],
                            "category": record["category"],
                            "phase": "prefill",
                            "step": None,
                        }
                    )
                    rows: list[dict[str, Any]] = []
                    for step, input_token in enumerate(teacher_inputs):
                        step_started = time.perf_counter()
                        result = candidate.step(input_token, return_logits=True)
                        row = compare_row(teacher_logits[step], result.logits)
                        row.update(
                            {
                                "sequence_id": record["sequence_id"],
                                "category": record["category"],
                                "phase": "decode",
                                "step": step,
                                "input_token": input_token,
                                "teacher_output_token": teacher_rows[step]["output_token"],
                                "candidate_output_token": int(result.token_id),
                                "seconds": time.perf_counter() - step_started,
                            }
                        )
                        rows.append(row)
                    snapshot = candidate._dms_backend.observability_snapshot()
                    if not snapshot["backend"]["device_payloads"]:
                        raise AssertionError(
                            f"DMS quality ({mode}) requires device payloads"
                        )
                candidates[mode] = {
                    "decision_mode": mode,
                    "prefill_row": prefill_row,
                    "decode_rows": rows,
                    "dms_digest": dms_digest(snapshot),
                    "diagnostic_observability": getattr(candidate, "_dms_diagnostic_observability", None),
                    "diagnostic_injection": (
                        {
                            "path": str(diagnostic_injection_path.resolve()),
                            "sha256": _sha256(diagnostic_injection_path),
                        }
                        if diagnostic_injection_path is not None else None
                    ),
                    "timing_seconds": time.perf_counter() - mode_started,
                }
            sequence_results[record["sequence_id"]] = {
                "sequence": {
                    "sequence_id": record["sequence_id"],
                    "category": record["category"],
                    "split": record["split"],
                    "source_id": record["source_id"],
                    "normalized_text_sha256": record["normalized_text_sha256"],
                    "prompt_tokens": len(prompt),
                    "token_ids_sha256": prompt_digest,
                    "decode_steps": decode_steps,
                },
                "teacher": {
                    "rows": teacher_rows,
                    "timing_seconds": dense_seconds,
                },
                "candidates": candidates,
            }
    finally:
        runner.close()
    memory_after_close = memory_stats()
    teardown_ok = (
        memory_after_close["current_allocated_bytes"]
        == memory_baseline["current_allocated_bytes"]
        and memory_after_close["active_allocations"] == memory_baseline["active_allocations"]
    )
    if not teardown_ok:
        raise AssertionError(
            "tracked device allocations did not return to baseline after teardown"
        )

    gate_verdicts: dict[str, dict[str, Any]] = {}
    for mode in modes:
        prefill_rows = [
            sequence_results[sid]["candidates"][mode]["prefill_row"]
            for sid in sequence_results
        ]
        decode_rows = [
            row
            for sid in sequence_results
            for row in sequence_results[sid]["candidates"][mode]["decode_rows"]
        ]
        gate_verdicts[mode] = evaluate_gate(
            str(args.gate),
            prefill_rows=prefill_rows,
            decode_rows=decode_rows,
            categories=categories,
            correlations=correlations,
        )
    passed = all(verdict["passed"] for verdict in gate_verdicts.values())
    ended = time.perf_counter()
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "hipengine_qwen38_dms_selector_campaign_gate_evaluation",
        "status": "passed" if passed else "rejected_quality",
        "performance_claim": False,
        "host": socket.gethostname(),
        "backend": str(args.backend),
        "codec": codec,
        "gate": str(args.gate),
        "model": {"path": str(Path(args.model).resolve()), "sha256": _sha256(args.model)},
        "metadata": {"path": str(Path(args.metadata).resolve()), "sha256": _sha256(args.metadata)},
        "data_manifest": {
            "path": str(Path(args.data_manifest).resolve()),
            "sha256": _sha256(args.data_manifest),
            "split": str(args.split),
        },
        "evaluator": _evaluator_hashes(),
        "protocol": {
            "teacher": "dense BF16 KV exact-Q4 resident session; one trajectory per sequence",
            "trajectory": "strict dense-teacher input tokens for every candidate step",
            "candidate_owner": "integrated compact device route; dense prefill pool released before decode",
            "serving_qualification": False,
            "modes": list(modes),
            "diagnostic_injection_dir": (
                str(Path(args.diagnostic_injection_dir).resolve())
                if args.diagnostic_injection_dir is not None else None
            ),
            "decode_steps": decode_steps,
            "categories": list(categories),
            "expected_sequences_per_category": int(args.expected_sequences_per_category),
            "prompt_tokens": int(args.prompt_tokens) if args.prompt_tokens else None,
            "correlation_note": correlations["note"],
        },
        "sequences": sequence_results,
        "correlations": correlations,
        "gate_verdicts": gate_verdicts,
        "resource_checks": {
            "dense_prefill_pool_released": True,
            "device_payloads_present": True,
            "teardown_to_baseline": teardown_ok,
            "memory_baseline": memory_baseline,
            "memory_after_close": memory_after_close,
        },
        "timing": {
            "load_seconds": loaded_at - started,
            "total_seconds": ended - started,
        },
        "provenance": _git(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "gate": result["gate"],
                "gate_verdicts": {
                    mode: {
                        "passed": verdict["passed"],
                        "failures": verdict["failures"],
                        "row_counts": verdict["row_counts"],
                        "rows_above_kl_0_02": verdict["rows_above_kl_0_02"],
                        "top1_mismatches": verdict["top1_mismatches"],
                    }
                    for mode, verdict in result["gate_verdicts"].items()
                },
                "timing": result["timing"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_fail and result["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
