#!/usr/bin/env python3
"""Benchmark exact Laguna target-only AR on the canonical category suite."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.chat.poolside_v1 import render_poolside_v1_chat
from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import host_array_ptr, memory_stats
from hipengine.kernels.backends import backend_package_capability
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.tokenization.gguf import LagunaGGUFTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
DEFAULT_PROMPTS = ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
DEFAULT_TEMPLATE = ROOT / "tests/fixtures/laguna_poolside_v1_template.json"
DEFAULT_ORACLE = ROOT / "tests/fixtures/laguna_poolside_v1_oracle.json"
DEFAULT_ORACLE_LOGPROBS = ROOT / "tests/fixtures/laguna_poolside_v1_first_token_logprobs.npy"
DEFAULT_BULK_GATE = (
    ROOT
    / "benchmarks/results/2026-07-22-gfx1151-laguna-bulk-prefill-verifier-correctness.json"
)
DEFAULT_CACHE = Path(
    "/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.hipengine-repacked-v1"
)
DEFAULT_MODEL_SHA256 = "7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f"
EXPECTED_CATEGORIES = frozenset(("code", "general_en", "general_ja", "mixed_ja_en"))
EXPECTED_PROMPT_COUNT = 10
RETAINED_HORIZONS = (16, 32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--oracle-logprobs", type=Path, default=DEFAULT_ORACLE_LOGPROBS)
    parser.add_argument("--bulk-correctness-artifact", type=Path, default=DEFAULT_BULK_GATE)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--iq3-c1-down-schedule",
        choices=("serial_weighted", "wave4_reduce"),
        default=None,
    )
    parser.add_argument(
        "--disable-iq2-grid64",
        action="store_true",
        help="roll back the exact expanded-magnitude IQ2 c=1 default",
    )
    parser.add_argument(
        "--disable-q5-fixed-meta-output",
        action="store_true",
        help="roll back exact fixed-metadata Q5 c=1 attention output",
    )
    parser.add_argument(
        "--disable-q5-fixed-meta-query-gate",
        action="store_true",
        help="roll back exact fixed-metadata Q5 c=1 query/gate",
    )
    parser.add_argument("--global-split-min-live", type=int)
    parser.add_argument("--swa-split-min-live", type=int)
    parser.add_argument("--swa-split-tile16-min-live", type=int)
    parser.add_argument(
        "--disable-swa-split-tile16",
        action="store_true",
        help="roll back the architecture-qualified SWA exact tile16 crossover",
    )
    parser.add_argument(
        "--disable-split-attention",
        action="store_true",
        help="roll back architecture-qualified Laguna split-attention defaults",
    )
    parser.add_argument(
        "--disable-split-gate-fusion",
        action="store_true",
        help="use the exact unfused split-reducer plus attention-gate chain",
    )
    parser.add_argument(
        "--disable-swa-split-wave-local",
        action="store_true",
        help="use the exact shared-statistics SWA split reducer",
    )
    parser.add_argument(
        "--disable-head-kv-fusion",
        action="store_true",
        help="use separate exact head RMSNorm/RoPE and BF16 KV append launches",
    )
    parser.add_argument(
        "--output-horizons",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=(16, 32),
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--warmup-output-tokens", type=int, default=2)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--direct-gguf", action="store_true")
    parser.add_argument("--safety-reserve-gib", type=float, default=8.0)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--quant-label", default="Q4_K_M mixed GGUF v3")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _compiler_version(path: Path | None) -> str | None:
    return None if path is None else path.read_text(encoding="utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _load_prompts(path: Path, tokenizer: LagunaGGUFTokenizer) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    categories: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        source = json.loads(raw_line)
        prompt_id = str(source.get("id", "")).strip()
        category = str(source.get("category", "")).strip()
        messages = source.get("messages")
        if not prompt_id or prompt_id in ids:
            raise ValueError(f"prompt line {line_number} has a blank or duplicate id")
        if not category or not isinstance(messages, list) or not messages:
            raise ValueError(f"prompt {prompt_id!r} lacks category/messages")
        rendered = render_poolside_v1_chat(
            messages,
            enable_thinking=False,
            add_generation_prompt=True,
        )
        token_ids = tuple(int(token) for token in tokenizer.encode(rendered))
        if not token_ids:
            raise ValueError(f"prompt {prompt_id!r} tokenized to no IDs")
        ids.add(prompt_id)
        categories.add(category)
        rows.append(
            {
                "id": prompt_id,
                "category": category,
                "messages_sha256": _sha256_json(messages),
                "rendered_sha256": _sha256_bytes(rendered.encode("utf-8")),
                "token_ids": token_ids,
                "token_ids_sha256": _sha256_json(token_ids),
                "prompt_tokens": len(token_ids),
            }
        )
    if len(rows) != EXPECTED_PROMPT_COUNT:
        raise ValueError(
            f"Laguna target AR requires all {EXPECTED_PROMPT_COUNT} canonical prompts, "
            f"got {len(rows)}"
        )
    if categories != EXPECTED_CATEGORIES:
        raise ValueError(
            "Laguna target AR requires canonical categories "
            f"{sorted(EXPECTED_CATEGORIES)}, got {sorted(categories)}"
        )
    return rows


def _progress(completed: int, total: int, spec) -> None:
    if completed == 1 or completed == total or completed % 100 == 0:
        print(
            f"load {completed}/{total}: {spec.source.name} ({spec.layout})",
            file=sys.stderr,
            flush=True,
        )


def _session(owner: LagunaGGUFResidentSession, args: argparse.Namespace):
    assert owner.weights is not None
    return LagunaGGUFResidentSession(
        resident_weights=owner.weights,
        context_length=args.context_length,
        backend=args.backend,
        runtime=owner.runtime,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached_build=args.require_cached_build,
        prefill_chunk_size=args.chunk_size,
        iq3_c1_down_schedule=args.iq3_c1_down_schedule,
        use_iq2_grid64=False if args.disable_iq2_grid64 else None,
        use_q5_fixed_meta_output=(
            False if args.disable_q5_fixed_meta_output else None
        ),
        use_q5_fixed_meta_query_gate=(
            False if args.disable_q5_fixed_meta_query_gate else None
        ),
        global_split_min_live=args.global_split_min_live,
        swa_split_min_live=args.swa_split_min_live,
        swa_split_tile16_min_live=args.swa_split_tile16_min_live,
        use_swa_split_tile16=False if args.disable_swa_split_tile16 else None,
        use_split_attention=False if args.disable_split_attention else None,
        use_split_gate_fusion=False if args.disable_split_gate_fusion else None,
        use_swa_split_wave_local=(
            False if args.disable_swa_split_wave_local else None
        ),
        use_head_kv_fusion=False if args.disable_head_kv_fusion else None,
    )


def _run_target(
    owner: LagunaGGUFResidentSession,
    prompt: dict[str, Any],
    *,
    mode: str,
    horizons: tuple[int, ...],
    repetition: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if mode not in {"serial", "bulk"}:
        raise ValueError(f"unknown Laguna benchmark mode {mode!r}")
    session = _session(owner, args)
    try:
        prefill_started = time.perf_counter()
        result = session.prefill(prompt["token_ids"], use_bulk=mode == "bulk")
        prefill_seconds = time.perf_counter() - prefill_started
        generated = [int(result.next_token_id)]
        decode_steps: list[float] = []
        while len(generated) < max(horizons):
            started = time.perf_counter()
            result = session.forward_token(result.next_token_id)
            decode_steps.append(time.perf_counter() - started)
            generated.append(int(result.next_token_id))
        checkpoints = {}
        for horizon in horizons:
            decode_seconds = float(sum(decode_steps[: max(0, horizon - 1)]))
            total_seconds = prefill_seconds + decode_seconds
            checkpoints[str(horizon)] = {
                "output_tokens": int(horizon),
                "generated_token_ids": generated[:horizon],
                "generated_ids_sha256": _sha256_json(generated[:horizon]),
                "decode_forward_calls": max(0, int(horizon) - 1),
                "decode_seconds": decode_seconds,
                "decode_tok_s": (
                    (int(horizon) - 1) / decode_seconds if horizon > 1 else None
                ),
                "total_seconds": total_seconds,
                "e2e_output_tok_s": int(horizon) / total_seconds,
            }
        return {
            "prompt_id": prompt["id"],
            "category": prompt["category"],
            "prompt_tokens": prompt["prompt_tokens"],
            "prompt_token_ids_sha256": prompt["token_ids_sha256"],
            "mode": mode,
            "repetition": int(repetition),
            "prefill_seconds": prefill_seconds,
            "ttft_seconds": prefill_seconds,
            "prefill_tok_s": prompt["prompt_tokens"] / prefill_seconds,
            "checkpoints": checkpoints,
        }
    finally:
        session.close()


def _warmup(owner: LagunaGGUFResidentSession, prompt: dict[str, Any], args: argparse.Namespace) -> None:
    for mode in ("serial", "bulk"):
        session = _session(owner, args)
        try:
            result = session.prefill(prompt["token_ids"], use_bulk=mode == "bulk")
            for _ in range(max(0, args.warmup_output_tokens - 1)):
                result = session.forward_token(result.next_token_id)
        finally:
            session.close()


def _normalized_log_probs(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(logits))
    return logits - (maximum + math.log(float(np.exp(logits - maximum).sum())))


def _oracle_gate(owner: LagunaGGUFResidentSession, args: argparse.Namespace) -> dict[str, Any]:
    template = json.loads(args.template.read_text(encoding="utf-8"))
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    prompt_case = next(
        case for case in template["cases"] if case["name"] == oracle["prompt"]["case"]
    )
    prompt_ids = tuple(int(value) for value in prompt_case["token_ids"])
    session = _session(owner, args)
    try:
        result = session.prefill(prompt_ids, use_bulk=True)
        logits = np.empty(session.config.vocab_size, dtype=np.float32)
        owner.runtime.memcpy(
            host_array_ptr(logits),
            result.logits.ptr,
            logits.nbytes,
            HipMemcpyKind.DEVICE_TO_HOST,
        )
    finally:
        session.close()
    reference = _normalized_log_probs(np.load(args.oracle_logprobs, allow_pickle=False))
    candidate = _normalized_log_probs(logits)
    probabilities = np.exp(reference)
    kl = float(np.sum(probabilities * (reference - candidate)))
    candidate_top1 = int(np.argmax(logits))
    reference_top1 = int(oracle["first_token"]["id"])
    finite = bool(np.isfinite(logits).all())
    return {
        "pass": bool(finite and kl <= 0.05 and candidate_top1 == reference_top1),
        "kl_divergence": kl,
        "kl_threshold": 0.05,
        "candidate_top1": candidate_top1,
        "reference_top1": reference_top1,
        "top1_agreement": float(candidate_top1 == reference_top1),
        "top1_threshold": 0.9,
        "finite_logits": finite,
        "oracle_artifact": str(args.oracle.resolve()),
        "oracle_distribution": str(args.oracle_logprobs.resolve()),
    }


def _paired_correctness(rows: list[dict[str, Any]], horizons: tuple[int, ...]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["prompt_id"], row["repetition"])][row["mode"]] = row
    comparisons = []
    for (prompt_id, repetition), modes in sorted(grouped.items()):
        if set(modes) != {"serial", "bulk"}:
            raise ValueError(f"missing serial/bulk pair for {prompt_id} rep {repetition}")
        checks = {}
        for horizon in horizons:
            serial_ids = modes["serial"]["checkpoints"][str(horizon)]["generated_token_ids"]
            bulk_ids = modes["bulk"]["checkpoints"][str(horizon)]["generated_token_ids"]
            checks[str(horizon)] = serial_ids == bulk_ids
        comparisons.append(
            {
                "prompt_id": prompt_id,
                "repetition": repetition,
                "horizons_exact": checks,
                "pass": all(checks.values()),
            }
        )
    deterministic = True
    for mode in ("serial", "bulk"):
        for prompt_id in {row["prompt_id"] for row in rows}:
            selected = [row for row in rows if row["mode"] == mode and row["prompt_id"] == prompt_id]
            for horizon in horizons:
                hashes = {
                    row["checkpoints"][str(horizon)]["generated_ids_sha256"]
                    for row in selected
                }
                deterministic = deterministic and len(hashes) == 1
    return {
        "pass": bool(all(item["pass"] for item in comparisons) and deterministic),
        "serial_bulk_pairs": comparisons,
        "same_mode_repeat_deterministic": bool(deterministic),
    }


def _aggregate(rows: list[dict[str, Any]], horizons: tuple[int, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in ("serial", "bulk"):
        selected = [row for row in rows if row["mode"] == mode]
        mode_result: dict[str, Any] = {
            "runs": len(selected),
            "prompt_tokens": sum(row["prompt_tokens"] for row in selected),
            "prefill_seconds": sum(row["prefill_seconds"] for row in selected),
            "prefill_tok_s": sum(row["prompt_tokens"] for row in selected)
            / sum(row["prefill_seconds"] for row in selected),
            "ttft_median_seconds": statistics.median(
                row["ttft_seconds"] for row in selected
            ),
            "horizons": {},
        }
        for horizon in horizons:
            checkpoints = [row["checkpoints"][str(horizon)] for row in selected]
            decode_calls = sum(item["decode_forward_calls"] for item in checkpoints)
            decode_seconds = sum(item["decode_seconds"] for item in checkpoints)
            output_tokens = sum(item["output_tokens"] for item in checkpoints)
            total_seconds = sum(item["total_seconds"] for item in checkpoints)
            mode_result["horizons"][str(horizon)] = {
                "output_tokens": output_tokens,
                "decode_forward_calls": decode_calls,
                "decode_seconds": decode_seconds,
                "decode_tok_s": decode_calls / decode_seconds,
                "total_seconds": total_seconds,
                "e2e_output_tok_s": output_tokens / total_seconds,
            }
        result[mode] = mode_result
    comparisons = {}
    for horizon in horizons:
        serial = result["serial"]["horizons"][str(horizon)]
        bulk = result["bulk"]["horizons"][str(horizon)]
        comparisons[str(horizon)] = {
            "prefill_speedup": result["bulk"]["prefill_tok_s"]
            / result["serial"]["prefill_tok_s"],
            "decode_speedup": bulk["decode_tok_s"] / serial["decode_tok_s"],
            "e2e_speedup": bulk["e2e_output_tok_s"] / serial["e2e_output_tok_s"],
        }
    result["bulk_vs_serial"] = comparisons
    return result


def _category_aggregate(
    rows: list[dict[str, Any]], horizons: tuple[int, ...]
) -> dict[str, Any]:
    categories: dict[str, Any] = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        categories[category] = _aggregate(selected, horizons)
    return categories


def _promotion_gate(
    aggregate: dict[str, Any],
    categories: dict[str, Any],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    """Require a suite win and fail closed on every category."""

    failed: list[str] = []
    for horizon in horizons:
        key = str(horizon)
        comparison = aggregate["bulk_vs_serial"][key]
        for metric, threshold, inclusive in (
            ("prefill_speedup", 1.0, False),
            ("decode_speedup", 0.98, True),
            ("e2e_speedup", 1.0, False),
        ):
            value = float(comparison[metric])
            accepted = value >= threshold if inclusive else value > threshold
            if not accepted:
                failed.append(f"suite:h{horizon}:{metric}")
        for category, category_result in sorted(categories.items()):
            category_comparison = category_result["bulk_vs_serial"][key]
            for metric, threshold, inclusive in (
                ("prefill_speedup", 1.0, False),
                ("decode_speedup", 0.98, True),
                ("e2e_speedup", 0.98, True),
            ):
                value = float(category_comparison[metric])
                accepted = value >= threshold if inclusive else value > threshold
                if not accepted:
                    failed.append(f"{category}:h{horizon}:{metric}")
    return {
        "pass": not failed,
        "failed_checks": failed,
        "policy": {
            "suite": {
                "prefill_speedup": ">1.0",
                "decode_speedup": ">=0.98",
                "e2e_speedup": ">1.0",
            },
            "each_category": {
                "prefill_speedup": ">1.0",
                "decode_speedup": ">=0.98",
                "e2e_speedup": ">=0.98",
            },
        },
    }


def _laguna_f16_prefill_configuration(backend: str) -> dict[str, Any]:
    requested = os.environ.get("HIPENGINE_LAGUNA_F16_PREFILL", "auto").strip().lower() or "auto"
    if requested not in {"auto", "gemv", "tiled"}:
        raise ValueError(
            "HIPENGINE_LAGUNA_F16_PREFILL must be one of: auto, gemv, tiled"
        )
    backend_strategy = backend_package_capability(
        backend, "LAGUNA_F16_PREFILL_STRATEGY", None
    )
    backend_min_rows = int(
        backend_package_capability(backend, "LAGUNA_F16_PREFILL_MIN_ROWS", 0) or 0
    )
    if requested == "tiled":
        effective_strategy = "tiled"
        effective_min_rows = 2
    elif requested == "gemv":
        effective_strategy = "gemv"
        effective_min_rows = None
    elif backend_strategy == "tiled" and backend_min_rows > 1:
        effective_strategy = "tiled"
        effective_min_rows = backend_min_rows
    else:
        effective_strategy = "gemv"
        effective_min_rows = None
    return {
        "requested": requested,
        "backend_strategy": backend_strategy,
        "backend_min_rows": backend_min_rows or None,
        "effective_strategy": effective_strategy,
        "effective_min_rows": effective_min_rows,
        "rows_one_always_gemv": True,
    }


def _repo_state() -> dict[str, Any]:
    revision = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    tracked = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"), cwd=ROOT, text=True
    ).strip()
    return {
        "revision": revision,
        "tracked_clean": not bool(tracked),
        "tracked_status": tracked.splitlines(),
        "untracked_shared_tree_artifacts_excluded": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    horizons = tuple(sorted(set(int(value) for value in args.output_horizons)))
    if len(horizons) < 2 or min(horizons) <= 1:
        raise ValueError("at least two distinct output horizons greater than one are required")
    if args.repetitions < 2:
        raise ValueError("at least two benchmark repetitions are required")
    if args.warmup_output_tokens <= 0:
        raise ValueError("warmup output tokens must be positive")
    if args.safety_reserve_gib <= 0.0:
        raise ValueError("--safety-reserve-gib must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(f"Laguna model not found: {args.model}")
    if not args.model_sha256:
        raise ValueError("--model-sha256 is required for a retained Laguna AR run")
    repo = _repo_state()
    if not repo["tracked_clean"]:
        raise RuntimeError("retained Laguna AR benchmark requires a clean tracked worktree")
    f16_prefill = _laguna_f16_prefill_configuration(args.backend)
    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant=args.quant_label,
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_target_ar_category",
        timing_protocol="balanced_serial_vs_bulk_prefill_plus_eager_c1_decode",
        warmups=2,
        repetitions=args.repetitions,
    )
    bulk_gate = json.loads(args.bulk_correctness_artifact.read_text(encoding="utf-8"))
    if not bulk_gate.get("pass") or bulk_gate.get("status") != "accepted":
        raise ValueError("Laguna bulk correctness artifact is not accepted")
    if bulk_gate.get("model", {}).get("sha256") != args.model_sha256:
        raise ValueError("Laguna bulk correctness artifact model SHA-256 mismatch")

    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _load_prompts(args.prompts, tokenizer)
    if max(prompt["prompt_tokens"] for prompt in prompts) + max(horizons) - 1 > args.context_length:
        raise ValueError("prompt/output shape exceeds admitted context")

    runtime = get_hip_runtime()
    gpu_free_before, gpu_total_bytes = runtime.mem_get_info()
    tracked_before = memory_stats()
    owner = None
    rows: list[dict[str, Any]] = []
    load_started = time.perf_counter()
    try:
        owner = LagunaGGUFResidentSession(
            args.model,
            context_length=args.context_length,
            backend=args.backend,
            runtime=runtime,
            compiler_version=_compiler_version(args.compiler_version_file),
            require_cached_build=args.require_cached_build,
            safety_reserve_nbytes=int(args.safety_reserve_gib * 2**30),
            progress=_progress,
            repacked_cache=None if args.direct_gguf else args.repacked_cache,
            model_sha256=args.model_sha256,
            prefill_chunk_size=args.chunk_size,
            iq3_c1_down_schedule=args.iq3_c1_down_schedule,
            use_iq2_grid64=False if args.disable_iq2_grid64 else None,
            use_q5_fixed_meta_output=(
                False if args.disable_q5_fixed_meta_output else None
            ),
            use_q5_fixed_meta_query_gate=(
                False if args.disable_q5_fixed_meta_query_gate else None
            ),
            global_split_min_live=args.global_split_min_live,
            swa_split_min_live=args.swa_split_min_live,
            swa_split_tile16_min_live=args.swa_split_tile16_min_live,
            use_swa_split_tile16=False if args.disable_swa_split_tile16 else None,
            use_split_attention=False if args.disable_split_attention else None,
            use_split_gate_fusion=False if args.disable_split_gate_fusion else None,
            use_swa_split_wave_local=(
                False if args.disable_swa_split_wave_local else None
            ),
            use_head_kv_fusion=False if args.disable_head_kv_fusion else None,
        )
        load_seconds = time.perf_counter() - load_started
        oracle_gate = _oracle_gate(owner, args)
        _warmup(owner, prompts[0], args)
        for repetition in range(args.repetitions):
            for prompt_index, prompt in enumerate(prompts):
                modes = (
                    ("serial", "bulk")
                    if (repetition + prompt_index) % 2 == 0
                    else ("bulk", "serial")
                )
                for mode in modes:
                    row = _run_target(
                        owner,
                        prompt,
                        mode=mode,
                        horizons=horizons,
                        repetition=repetition,
                        args=args,
                    )
                    rows.append(row)
                    checkpoint = row["checkpoints"][str(max(horizons))]
                    print(
                        f"rep={repetition} prompt={prompt['id']} mode={mode} "
                        f"prefill={row['prefill_tok_s']:.2f} tok/s "
                        f"decode={checkpoint['decode_tok_s']:.2f} tok/s",
                        file=sys.stderr,
                        flush=True,
                    )
        resident_nbytes = owner.resident_nbytes
    finally:
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    paired = _paired_correctness(rows, horizons)
    aggregate = _aggregate(rows, horizons)
    categories = _category_aggregate(rows, horizons)
    promotion = _promotion_gate(aggregate, categories, horizons)
    passed = bool(oracle_gate["pass"] and paired["pass"] and recovered)
    protocol_eligible = bool(
        horizons == RETAINED_HORIZONS
        and args.repetitions >= 2
        and len(prompts) == EXPECTED_PROMPT_COUNT
        and {prompt["category"] for prompt in prompts} == EXPECTED_CATEGORIES
    )
    claim = bool(passed and promotion["pass"] and protocol_eligible)
    gpu_free_after, gpu_total_after = runtime.mem_get_info()
    if gpu_total_after != gpu_total_bytes:
        raise RuntimeError("HIP total memory changed during Laguna target benchmark")
    created_at = datetime.now(timezone.utc).isoformat()
    prompt_payload = args.prompts.read_bytes()
    bulk_payload = args.bulk_correctness_artifact.read_bytes()
    active_cache = None if args.direct_gguf else args.repacked_cache
    manifest_path = None if active_cache is None else active_cache / "manifest.json"
    manifest_sha256 = (
        _sha256_bytes(manifest_path.read_bytes())
        if manifest_path is not None and manifest_path.is_file()
        else None
    )
    return {
        "schema": 1,
        "created_at": created_at,
        "kind": "hipengine_laguna_target_ar_category_benchmark",
        "status": "retained" if claim else ("accepted_correctness" if passed else "rejected"),
        "pass": passed,
        "performance_claim": claim,
        "performance_claim_scope": (
            "target-only c=1 greedy AR; canonical 18-prompt train+heldout four-category suite; "
            f"output horizons {list(horizons)}; model load excluded"
        ),
        "provenance": provenance,
        "repo": repo,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": args.quant_label,
            "repacked_cache": None if active_cache is None else str(active_cache.resolve()),
            "repacked_cache_manifest_sha256": manifest_sha256,
        },
        "platform": {
            "backend": args.backend,
            "target_arch": args.backend.removeprefix("hip_"),
            "device_name": provenance["device_name"],
            "machine": platform.machine(),
            "hip_total_bytes": gpu_total_bytes,
        },
        "protocol": {
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_suite_sha256": _sha256_bytes(prompt_payload),
            "prompt_count": len(prompts),
            "categories": sorted(EXPECTED_CATEGORIES),
            "prompt_tokens_min": min(prompt["prompt_tokens"] for prompt in prompts),
            "prompt_tokens_max": max(prompt["prompt_tokens"] for prompt in prompts),
            "context_length": args.context_length,
            "prefill_chunk_size": args.chunk_size,
            "iq3_c1_down_schedule": owner.iq3_c1_down_schedule,
            "global_split_min_live": owner.global_split_min_live,
            "swa_split_min_live": owner.swa_split_min_live,
            "swa_split_tile16_min_live": owner.swa_split_tile16_min_live,
            "use_swa_split_tile16": owner.use_swa_split_tile16,
            "use_split_attention": owner.use_split_attention,
            "use_split_gate_fusion": owner.use_split_gate_fusion,
            "use_swa_split_wave_local": owner.use_swa_split_wave_local,
            "use_head_kv_fusion": owner.use_head_kv_fusion,
            "use_iq2_grid64": owner.use_iq2_grid64,
            "use_q5_fixed_meta_output": owner.use_q5_fixed_meta_output,
            "use_q5_fixed_meta_query_gate": owner.use_q5_fixed_meta_query_gate,
            "output_horizons": list(horizons),
            "repetitions": args.repetitions,
            "warmups": {
                "serial": 1,
                "bulk": 1,
                "output_tokens": args.warmup_output_tokens,
            },
            "order": "alternating serial/bulk by repetition plus prompt index",
            "sampling": "greedy argmax",
            "ttft_scope": "prefill start through synchronized first-token argmax",
            "decode_scope": "horizon-1 synchronized forward_token calls after first token",
            "e2e_scope": "TTFT plus decode scope; model load excluded",
            "fixed_horizon_after_stop": True,
            "f16_prefill": f16_prefill,
            "protocol_eligible": protocol_eligible,
        },
        "load": {
            "seconds_excluded_from_claim": load_seconds,
            "cache_manifest_sha256": manifest_sha256,
        },
        "correctness": {
            "pass": passed,
            "poolside_oracle": oracle_gate,
            "serial_bulk_suite": paired,
            "bulk_correctness_artifact": {
                "path": str(args.bulk_correctness_artifact.resolve()),
                "sha256": _sha256_bytes(bulk_payload),
                "source_revision": bulk_gate.get("source_revision"),
                "pass": bool(bulk_gate.get("pass")),
                "status": bulk_gate.get("status"),
            },
            "tracked_returned_to_baseline": recovered,
        },
        "promotion_gate": promotion,
        "aggregate": aggregate,
        "categories": categories,
        "prompt_runs": rows,
        "memory": {
            "resident_nbytes": resident_nbytes,
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
            "tracked_peak_allocated_bytes": tracked_after["peak_allocated_bytes"],
            "gpu_free_before": gpu_free_before,
            "gpu_free_after": gpu_free_after,
            "hip_total_bytes": gpu_total_bytes,
        },
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "notes": [
            "Bulk and serial differ only in prompt prefill; both use the same eager c=1 decode path.",
            "Every prompt/mode/repetition emits exact generated IDs at both horizons.",
            "The Poolside distribution oracle and accepted bulk logits/hidden/KV artifact are hard gates.",
            "Profiler launch-family evidence is separate; rocprofv3 does not wrap this timing run.",
        ],
    }


def main() -> int:
    args = _parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
