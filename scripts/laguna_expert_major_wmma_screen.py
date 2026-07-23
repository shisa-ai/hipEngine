#!/usr/bin/env python3
"""Screen explicit Laguna expert-major compensated WMMA at production row shapes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import host_array_ptr, memory_stats
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_matrix_chunk_bench import _kv_digest
from scripts.laguna_prefill_profile import _profile_token_stream
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_PROMPTS,
    _compiler_version,
    _load_prompts,
    _progress,
    _repo_state,
    _sha256_bytes,
    _sha256_json,
)

ROOT = Path(__file__).resolve().parents[1]
ROWS = (32, 55, 64, 122, 128, 256, 512)
MODES = ("retained", "expert_major_comp")
MODE_ROUTES = {
    "retained": "adaptive_grouped_smallm_fused",
    "expert_major_comp": "expert_major_wmma_comp",
}
EXACT_STATE_FIELDS = (
    "logits_sha256",
    "final_hidden_sha256",
    "post_layer_hidden_sha256",
    "kv_sha256",
    "next_token_id",
    "next_token_logit_hex",
    "final_position",
)
DEFAULT_LEAF_SCREEN = (
    ROOT
    / "benchmarks/results/2026-07-24-gfx1151-laguna-expert-major-wmma-leaf-screen.json"
)
DEFAULT_OUTPUT = Path("/tmp/laguna-expert-major-wmma-full-model-screen.raw.json")


def _parse_rows(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if not parsed or tuple(sorted(set(parsed))) != parsed or any(item <= 1 for item in parsed):
        raise argparse.ArgumentTypeError("rows must be sorted distinct integers above one")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=max(ROWS))
    parser.add_argument("--rows", type=_parse_rows, default=ROWS)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--leaf-screen", type=Path, default=DEFAULT_LEAF_SCREEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _mode_order(shape_index: int, repetition: int) -> tuple[str, str]:
    return MODES if (int(shape_index) + int(repetition)) % 2 == 0 else tuple(reversed(MODES))


def _device_array(runtime, buffer, dtype: np.dtype) -> np.ndarray:
    values = np.empty(buffer.nbytes // np.dtype(dtype).itemsize, dtype=dtype)
    runtime.memcpy(
        host_array_ptr(values),
        buffer.ptr,
        values.nbytes,
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return values


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _quality(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    finite = bool(np.isfinite(ref).all() and np.isfinite(cand).all())
    if not finite:
        return {
            "finite": False,
            "kl_divergence": math.inf,
            "top1_agreement": False,
            "reference_top1": None,
            "candidate_top1": None,
        }
    ref_shift = ref - np.max(ref)
    cand_shift = cand - np.max(cand)
    ref_logp = ref_shift - np.log(np.exp(ref_shift).sum())
    cand_logp = cand_shift - np.log(np.exp(cand_shift).sum())
    ref_top1 = int(np.argmax(ref))
    cand_top1 = int(np.argmax(cand))
    return {
        "finite": True,
        "kl_divergence": float(
            np.sum(np.exp(ref_logp) * (ref_logp - cand_logp))
        ),
        "top1_agreement": ref_top1 == cand_top1,
        "reference_top1": ref_top1,
        "candidate_top1": cand_top1,
    }


def _load_leaf_screen(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("kind") != "hipengine_laguna_expert_major_wmma_leaf_screen"
        or payload.get("status") != "leaf_screen_passed"
        or not payload.get("pass")
        or payload.get("candidate", {}).get("registry_variant")
        != "selected_t16_expert_major_wmma_comp_bf16_bf16_out"
    ):
        raise ValueError("expert-major full-model screen requires the passing leaf artifact")
    return payload


def _summarize(
    records: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    *,
    rows: Sequence[int] = ROWS,
) -> dict[str, Any]:
    parsed_rows = tuple(int(value) for value in rows)
    if parsed_rows != ROWS:
        raise ValueError(f"expert-major screen requires exact rows {ROWS}")
    expected = len(ROWS) * len(MODES)
    by_shape_mode: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_shape_mode[(int(record["rows"]), str(record["mode"]))].append(record)
    sample_counts = {len(value) for value in by_shape_mode.values()}
    if len(by_shape_mode) != expected or len(sample_counts) != 1 or sample_counts == {0}:
        raise ValueError("every shape/mode requires equal non-empty samples")

    shapes: dict[str, Any] = {}
    deterministic = True
    all_cursor_exact = True
    threshold_candidates: list[int] = []
    for row in ROWS:
        mode_summaries: dict[str, Any] = {}
        for mode in MODES:
            selected = by_shape_mode[(row, mode)]
            timings = [float(item["prefill_seconds"]) for item in selected]
            if any(not math.isfinite(value) or value <= 0.0 for value in timings):
                raise ValueError("all timing samples must be finite and positive")
            mode_deterministic = all(
                len({item[field] for item in selected}) == 1
                for field in EXACT_STATE_FIELDS
            )
            deterministic = bool(deterministic and mode_deterministic)
            median = statistics.median(timings)
            mode_summaries[mode] = {
                "samples_seconds": timings,
                "median_seconds": median,
                "median_tok_s": row / median,
                "same_mode_deterministic": mode_deterministic,
                "state": {
                    field: selected[0][field] for field in EXACT_STATE_FIELDS
                },
            }
        speedup = (
            mode_summaries["retained"]["median_seconds"]
            / mode_summaries["expert_major_comp"]["median_seconds"]
        )
        shape_comparisons = [
            item for item in comparisons if int(item["rows"]) == row
        ]
        if len(shape_comparisons) != next(iter(sample_counts)):
            raise ValueError(f"rows={row} requires one quality comparison per repetition")
        shape_cursor_exact = all(bool(item["cursor_exact"]) for item in shape_comparisons)
        all_cursor_exact = bool(all_cursor_exact and shape_cursor_exact)
        shapes[str(row)] = {
            "rows": row,
            **mode_summaries,
            "expert_major_comp_vs_retained_speedup": speedup,
            "quality": {
                "max_kl": max(float(item["kl_divergence"]) for item in shape_comparisons),
                "top1_agreement": sum(
                    bool(item["top1_agreement"]) for item in shape_comparisons
                )
                / len(shape_comparisons),
                "finite": all(bool(item["finite"]) for item in shape_comparisons),
                "cursor_exact": shape_cursor_exact,
            },
        }

    finite = all(bool(item["finite"]) for item in comparisons)
    max_kl = max(float(item["kl_divergence"]) for item in comparisons)
    top1 = sum(bool(item["top1_agreement"]) for item in comparisons) / len(comparisons)
    quality_pass = bool(
        finite
        and max_kl <= 0.05
        and top1 >= 0.9
        and all_cursor_exact
        and deterministic
    )

    baseline_sum = sum(
        float(shapes[str(row)]["retained"]["median_seconds"]) for row in ROWS
    )
    policies: dict[str, Any] = {}
    for threshold in ROWS:
        candidate_rows = tuple(row for row in ROWS if row >= threshold)
        every_selected_shape_positive = all(
            float(shapes[str(row)]["expert_major_comp_vs_retained_speedup"]) > 1.0
            for row in candidate_rows
        )
        adaptive_sum = sum(
            float(
                shapes[str(row)][
                    "expert_major_comp" if row >= threshold else "retained"
                ]["median_seconds"]
            )
            for row in ROWS
        )
        speedup = baseline_sum / adaptive_sum
        eligible = bool(every_selected_shape_positive and speedup > 1.0)
        if eligible:
            threshold_candidates.append(threshold)
        policies[str(threshold)] = {
            "threshold_rows": threshold,
            "candidate_rows": list(candidate_rows),
            "every_selected_shape_positive": every_selected_shape_positive,
            "adaptive_median_sum_seconds": adaptive_sum,
            "speedup_vs_retained": speedup,
            "eligible": eligible,
        }
    selected_threshold = min(threshold_candidates) if threshold_candidates else None
    failed: list[str] = []
    if not quality_pass:
        failed.append("full_model_quality_or_determinism_failed")
    if selected_threshold is None:
        failed.append("no_nonregressive_adaptive_threshold")
    return {
        "pass": not failed,
        "failed_checks": failed,
        "shapes": shapes,
        "quality": {
            "pass": quality_pass,
            "finite": finite,
            "max_kl": max_kl,
            "top1_agreement": top1,
            "cursor_exact": all_cursor_exact,
            "same_mode_state_deterministic": deterministic,
            "policy": "finite logits; KL<=0.05; top-1>=90%; exact cursor; deterministic logits/hidden/KV state",
        },
        "threshold": {
            "selected_rows": selected_threshold,
            "eligible_rows": threshold_candidates,
            "policies": policies,
            "retained_median_sum_seconds": baseline_sum,
            "policy": "choose the smallest threshold whose selected shapes and aggregate wall are all positive",
        },
    }


def _run_one(
    owner: LagunaGGUFResidentSession,
    token_ids: Sequence[int],
    *,
    rows: int,
    mode: str,
    repetition: int,
) -> tuple[dict[str, Any], np.ndarray]:
    owner.reset_state()
    owner.set_selected_down_mode(MODE_ROUTES[mode])
    started = time.perf_counter()
    result = owner.prefill(token_ids[:rows], use_bulk=True)
    owner.runtime.device_synchronize()
    elapsed = time.perf_counter() - started
    logits = _device_array(owner.runtime, result.logits, np.float32)
    final_hidden = _device_array(owner.runtime, result.final_hidden, np.uint16)
    post_layer_hidden = _device_array(
        owner.runtime, result.post_layer_hidden, np.uint16
    )
    return (
        {
            "rows": rows,
            "mode": mode,
            "selected_down_mode": owner.selected_down_mode,
            "repetition": repetition,
            "prefill_seconds": elapsed,
            "prefill_tok_s": rows / elapsed,
            "next_token_id": int(result.next_token_id),
            "next_token_logit_hex": float(result.next_token_logit).hex(),
            "logits_sha256": _sha256_array(logits),
            "final_hidden_sha256": _sha256_array(final_hidden),
            "post_layer_hidden_sha256": _sha256_array(post_layer_hidden),
            "kv_sha256": _kv_digest(owner),
            "final_position": int(owner.position),
        },
        logits,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = tuple(int(value) for value in args.rows)
    if rows != ROWS:
        raise ValueError(f"expert-major full-model screen requires exact rows {ROWS}")
    if args.backend != "hip_gfx1151":
        raise ValueError("expert-major full-model screen is qualified only for hip_gfx1151")
    if args.context_length < max(ROWS):
        raise ValueError("context length must cover the largest row shape")
    if args.repetitions < 3 or args.warmups < 1:
        raise ValueError("expert-major full-model screen requires >=3 repetitions and >=1 warmup")
    if not args.model.is_file() or not args.model_sha256:
        raise ValueError("expert-major full-model screen requires the pinned model and SHA-256")
    leaf = _load_leaf_screen(args.leaf_screen)
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("expert-major full-model screen requires a clean tracked worktree")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_expert_major_compensated_wmma_screen",
        timing_protocol="same_resident_weights_counterbalanced_m32_512_full_logits",
        warmups=args.warmups * len(MODES),
        repetitions=args.repetitions,
    )
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    token_stream, token_source = _profile_token_stream(prompts, max(ROWS))
    runtime = get_hip_runtime()
    tracked_before = memory_stats()
    gpu_free_before, gpu_total = runtime.mem_get_info()
    owner: LagunaGGUFResidentSession | None = None
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    load_started = time.perf_counter()
    try:
        owner = LagunaGGUFResidentSession(
            args.model,
            context_length=args.context_length,
            backend=args.backend,
            runtime=runtime,
            compiler_version=_compiler_version(args.compiler_version_file),
            require_cached_build=args.require_cached_build,
            progress=_progress,
            repacked_cache=args.repacked_cache,
            model_sha256=args.model_sha256,
            prefill_chunk_size=max(ROWS),
        )
        load_seconds = time.perf_counter() - load_started
        for _ in range(args.warmups):
            for mode in MODES:
                owner.reset_state()
                owner.set_selected_down_mode(MODE_ROUTES[mode])
                owner.prefill(token_stream[:128], use_bulk=True)
                runtime.device_synchronize()
        for repetition in range(args.repetitions):
            for shape_index, row_count in enumerate(ROWS):
                pair: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
                for mode in _mode_order(shape_index, repetition):
                    record, logits = _run_one(
                        owner,
                        token_stream,
                        rows=row_count,
                        mode=mode,
                        repetition=repetition,
                    )
                    records.append(record)
                    pair[mode] = (record, logits)
                    print(
                        f"rep={repetition} rows={row_count} mode={mode} "
                        f"prefill={record['prefill_tok_s']:.3f} tok/s "
                        f"next={record['next_token_id']}",
                        file=sys.stderr,
                        flush=True,
                    )
                quality = _quality(pair["retained"][1], pair["expert_major_comp"][1])
                comparisons.append(
                    {
                        "rows": row_count,
                        "repetition": repetition,
                        **quality,
                        "cursor_exact": (
                            pair["retained"][0]["final_position"]
                            == pair["expert_major_comp"][0]["final_position"]
                        ),
                        "complete_state_exact": all(
                            pair["retained"][0][field]
                            == pair["expert_major_comp"][0][field]
                            for field in EXACT_STATE_FIELDS
                        ),
                    }
                )
        resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total:
        raise RuntimeError("HIP total memory changed during expert-major screen")
    recovered = bool(
        tracked_after["current_allocated_bytes"]
        == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"]
        == tracked_before["active_allocations"]
    )
    summary = _summarize(records, comparisons, rows=rows)
    if not recovered:
        summary["pass"] = False
        summary["failed_checks"].append("tracked_lifecycle_not_recovered")
    manifest_path = args.repacked_cache / "manifest.json"
    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_expert_major_wmma_full_model_screen",
        "status": "quality_lane_admitted" if summary["pass"] else "measured_rejected",
        "pass": bool(summary["pass"]),
        "performance_claim": False,
        "performance_claim_scope": "same-weight full-model shape and quality screen; no category/default claim",
        "provenance": provenance,
        "repo": repo,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": "Q4_K_M mixed GGUF v3",
            "repacked_cache": str(args.repacked_cache.resolve()),
            "repacked_cache_manifest_sha256": (
                _sha256_bytes(manifest_path.read_bytes())
                if manifest_path.is_file()
                else None
            ),
        },
        "platform": {
            "backend": args.backend,
            "target_arch": args.backend.removeprefix("hip_"),
            "device_name": provenance["device_name"],
            "machine": platform.machine(),
            "hip_total_bytes": gpu_total,
        },
        "protocol": {
            "rows": list(ROWS),
            "modes": list(MODES),
            "mode_routes": MODE_ROUTES,
            "one_physical_chunk": True,
            "prefill_chunk_size": max(ROWS),
            "context_length": args.context_length,
            "repetitions": args.repetitions,
            "warmups_per_mode": args.warmups,
            "timed_order": "counterbalanced by shape and repetition",
            "timing_scope": "reset through synchronized final projection; hashing and model load excluded",
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(args.prompts.read_bytes()),
            "token_stream_sha256": _sha256_json(token_stream),
            "token_source": token_source,
            "leaf_screen": str(args.leaf_screen.resolve()),
            "leaf_screen_sha256": _sha256_bytes(args.leaf_screen.read_bytes()),
            "leaf_variant": leaf["candidate"]["registry_variant"],
        },
        "load": {
            "seconds_excluded": load_seconds,
            "resident_nbytes": resident_nbytes,
        },
        "records": records,
        "comparisons": comparisons,
        "summary": summary,
        "correctness": {
            **summary["quality"],
            "tracked_returned_to_baseline": recovered,
        },
        "memory": {
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "limitations": [
            "One deterministic canonical-prompt-derived token stream screens shape policy.",
            "Category/heldout teacher-forced and free-running admission is separate.",
            "No backend default changes until this screen selects a threshold and the category gate passes."
        ],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
