#!/usr/bin/env python3
"""Gate a non-byte-exact GGUF GDN prefill schedule on same-context logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.gguf_gdn_trajectory_gate import (
    CANDIDATE_MODES,
    SUPPORTED_MODES,
    _gdn_mode,
    _run_logits_trajectory,
    _run_timed_decode,
)
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import load_prompt_rows, prompt_sha256


KIND = "hipengine_gguf_gdn_prefill_teacher_forced_semantic_gate"
SCHEMA_VERSION = 1
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_PROMPTS = (
    REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
    REPO_ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl",
)
REQUIRED_CATEGORIES = frozenset({"code", "general_en", "general_ja", "mixed_ja_en"})


class GateError(RuntimeError):
    """Raised when the semantic gate cannot be evaluated safely."""


def _compare_teacher_forced(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    kl_threshold: float,
    top1_threshold: float,
) -> dict[str, Any]:
    """Compare logits produced from identical prompt and decode-token contexts."""

    if not baseline or len(baseline) != len(candidate):
        raise GateError("baseline and candidate trajectories must have equal nonzero length")
    transitions: list[dict[str, Any]] = []
    for index, (baseline_step, candidate_step) in enumerate(
        zip(baseline, candidate, strict=True)
    ):
        baseline_logits = np.asarray(baseline_step["logits"], dtype=np.float32)
        candidate_logits = np.asarray(candidate_step["logits"], dtype=np.float32)
        if baseline_logits.shape != candidate_logits.shape:
            raise GateError(f"logit shape mismatch at transition {index}")
        if not np.all(np.isfinite(baseline_logits)) or not np.all(
            np.isfinite(candidate_logits)
        ):
            raise GateError(f"non-finite logits at transition {index}")
        result = evaluate_logits(
            baseline_logits,
            candidate_logits,
            kl_threshold=float(kl_threshold),
            top1_threshold=0.0,
        )
        baseline_top1 = int(np.argmax(baseline_logits))
        candidate_top1 = int(np.argmax(candidate_logits))
        transitions.append(
            {
                "transition": int(index),
                "baseline_token_id": baseline_top1,
                "candidate_token_id": candidate_top1,
                "top1_match": baseline_top1 == candidate_top1,
                "kl_mean": float(result.kl_mean),
                "kl_max": float(result.kl_max),
                "kl_passed": bool(result.kl_max <= float(kl_threshold)),
            }
        )
    total = len(transitions)
    top1_matches = sum(bool(row["top1_match"]) for row in transitions)
    top1_agreement = float(top1_matches / total)
    kl_passed = all(bool(row["kl_passed"]) for row in transitions)
    return {
        "passed": bool(kl_passed and top1_agreement >= float(top1_threshold)),
        "kl_passed": kl_passed,
        "kl_threshold": float(kl_threshold),
        "kl_mean_max": float(max(row["kl_mean"] for row in transitions)),
        "kl_max": float(max(row["kl_max"] for row in transitions)),
        "top1_threshold": float(top1_threshold),
        "top1_matches": int(top1_matches),
        "top1_agreement": top1_agreement,
        "transitions_total": total,
        "first_top1_divergence": next(
            (
                int(row["transition"])
                for row in transitions
                if not bool(row["top1_match"])
            ),
            None,
        ),
        "transitions": transitions,
    }


def _run_teacher_forced_candidate(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    forced_input_ids: Sequence[int],
    mode: str,
    bulk_attention_mode: str,
) -> list[dict[str, Any]]:
    session.reset()
    with _gdn_mode(mode):
        result = session.prefill(
            [int(token) for token in prompt_ids],
            use_bulk=True,
            bulk_attention_mode=bulk_attention_mode,
            return_logits=True,
            capture_hidden_seed_fp32=False,
        )
    trajectory = [
        {
            "token_id": int(result.token_id),
            "logits": np.ascontiguousarray(result.logits, dtype=np.float32),
        }
    ]
    for token_id in forced_input_ids:
        result = session.step(int(token_id), return_logits=True)
        trajectory.append(
            {
                "token_id": int(result.token_id),
                "logits": np.ascontiguousarray(result.logits, dtype=np.float32),
            }
        )
    return trajectory


def _timed_prompt_pair(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    baseline_mode: str,
    candidate_mode: str,
    decode_steps: int,
    repetitions: int,
    bulk_attention_mode: str,
    graph_replay_decode: bool,
) -> dict[str, Any]:
    samples: dict[str, list[float]] = {baseline_mode: [], candidate_mode: []}
    trajectories: dict[str, list[list[int]]] = {
        baseline_mode: [],
        candidate_mode: [],
    }
    modes = (baseline_mode, candidate_mode)
    measurements: list[dict[str, Any]] = []
    for repetition in range(int(repetitions)):
        order = modes if repetition % 2 == 0 else tuple(reversed(modes))
        row: dict[str, Any] = {
            "repetition": repetition,
            "order": list(order),
            "modes": {},
        }
        for mode in order:
            result = _run_timed_decode(
                session,
                prompt_ids=prompt_ids,
                mode=mode,
                decode_steps=decode_steps,
                bulk_attention_mode=bulk_attention_mode,
                graph_replay_decode=graph_replay_decode,
            )
            samples[mode].append(float(result["wall_ms"]))
            trajectories[mode].append([int(token) for token in result["token_ids"]])
            row["modes"][mode] = result
        measurements.append(row)
    baseline_median = float(statistics.median(samples[baseline_mode]))
    candidate_median = float(statistics.median(samples[candidate_mode]))
    reference = trajectories[baseline_mode][0]
    trajectories_exact = all(
        tokens == reference for values in trajectories.values() for tokens in values
    )
    return {
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "candidate_wall_delta_percent": float(
            ((candidate_median - baseline_median) / baseline_median) * 100.0
        ),
        "candidate_speedup": float(baseline_median / candidate_median),
        "trajectories_exact_diagnostic": trajectories_exact,
        "measurements": measurements,
    }


def _aggregate_gate(
    prompts: Sequence[Mapping[str, Any]],
    *,
    kl_threshold: float,
    top1_threshold: float,
) -> dict[str, Any]:
    if not prompts:
        raise GateError("semantic gate requires at least one prompt")
    total = sum(int(row["correctness"]["transitions_total"]) for row in prompts)
    top1_matches = sum(int(row["correctness"]["top1_matches"]) for row in prompts)
    top1_agreement = float(top1_matches / total)
    kl_passed = all(bool(row["correctness"]["kl_passed"]) for row in prompts)
    kl_max = float(max(float(row["correctness"]["kl_max"]) for row in prompts))
    baseline_ms = float(
        sum(float(row["decode_performance"]["baseline_median_ms"]) for row in prompts)
    )
    candidate_ms = float(
        sum(float(row["decode_performance"]["candidate_median_ms"]) for row in prompts)
    )
    decode_non_regressive = candidate_ms <= baseline_ms
    return {
        "passed": bool(
            kl_passed
            and kl_max <= float(kl_threshold)
            and top1_agreement >= float(top1_threshold)
            and decode_non_regressive
        ),
        "prompts_total": len(prompts),
        "transitions_total": total,
        "kl_passed": kl_passed,
        "kl_threshold": float(kl_threshold),
        "kl_max": kl_max,
        "top1_threshold": float(top1_threshold),
        "top1_matches": top1_matches,
        "top1_agreement": top1_agreement,
        "decode_non_regressive": decode_non_regressive,
        "baseline_decode_median_wall_sum_ms": baseline_ms,
        "candidate_decode_median_wall_sum_ms": candidate_ms,
        "candidate_decode_speedup": float(baseline_ms / candidate_ms),
        "candidate_decode_wall_delta_percent": float(
            ((candidate_ms - baseline_ms) / baseline_ms) * 100.0
        ),
        "free_running_trajectories_exact_diagnostic": all(
            bool(row["decode_performance"].get("trajectories_exact_diagnostic", True))
            for row in prompts
        ),
    }


def _load_suites(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise GateError(f"prompt suite does not exist: {path}")
        for row in load_prompt_rows(path):
            prompt_id = str(row["id"])
            if prompt_id in seen:
                raise GateError(f"duplicate prompt id across suites: {prompt_id}")
            seen.add(prompt_id)
            rows.append({**row, "suite": str(path.resolve())})
    categories = {str(row["category"]) for row in rows}
    missing = REQUIRED_CATEGORIES - categories
    if missing:
        raise GateError(f"prompt suites are missing required categories: {sorted(missing)}")
    return rows


def _configure_gate_environment(*, decode_repack: bool) -> None:
    """Reserve diagnostic scratch before the resident session is allocated.

    The production direct-GDN route omits materialized Q/K/V buffers. Semantic
    gates switch modes inside one session, so they must disable that compact
    allocation before constructing the session or a materialized candidate
    would receive null device pointers.
    """

    os.environ["HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE"] = "1"
    os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "1" if decode_repack else "0"


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    baseline_mode = str(args.baseline_mode)
    candidate_mode = str(args.candidate_mode)
    if baseline_mode not in SUPPORTED_MODES or candidate_mode not in CANDIDATE_MODES:
        raise GateError("unsupported baseline or candidate mode")
    if baseline_mode == candidate_mode:
        raise GateError("baseline and candidate modes must differ")
    correctness_steps = int(args.correctness_decode_steps)
    performance_steps = int(args.performance_decode_steps)
    repetitions = int(args.performance_repetitions)
    if correctness_steps <= 0 or performance_steps <= 0 or repetitions <= 0:
        raise GateError("step counts and repetitions must be positive")
    if repetitions % 2:
        raise GateError("performance repetitions must be even")
    if not args.model.is_file():
        raise GateError(f"model does not exist: {args.model}")
    prompt_rows = _load_suites(args.prompts)
    if args.limit is not None:
        prompt_rows = prompt_rows[: max(0, int(args.limit))]
    if not prompt_rows:
        raise GateError("selected prompt suites are empty")

    _configure_gate_environment(decode_repack=bool(args.decode_repack))

    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    prompt_tokens = {
        str(row["id"]): build_chat_prompt(tokenizer, str(row["prompt"]))
        for row in prompt_rows
    }
    max_prompt_tokens = max(len(tokens) for tokens in prompt_tokens.values())
    max_sequence_length = (
        max_prompt_tokens + max(correctness_steps, performance_steps) + 2
    )
    records: list[dict[str, Any]] = []
    with Qwen35GGUFResidentSession(
        args.model,
        backend=str(args.backend),
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=bool(args.use_gemv_decode),
        prefill_config=PrefillConfig(
            attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens)
        ),
    ) as session:
        if session.runner is None:
            raise GateError("GGUF resident session closed during setup")
        resolved_backend = str(session.runner.backend)
        target_arch = str(session.runner.target_arch)
        for index, row in enumerate(prompt_rows):
            prompt_id = str(row["id"])
            tokens = prompt_tokens[prompt_id]
            baseline = _run_logits_trajectory(
                session,
                prompt_ids=tokens,
                mode=baseline_mode,
                decode_steps=correctness_steps,
                bulk_attention_mode=str(args.bulk_attention_mode),
            )
            candidate = _run_teacher_forced_candidate(
                session,
                prompt_ids=tokens,
                forced_input_ids=[int(step["token_id"]) for step in baseline[:-1]],
                mode=candidate_mode,
                bulk_attention_mode=str(args.bulk_attention_mode),
            )
            correctness = _compare_teacher_forced(
                baseline,
                candidate,
                kl_threshold=float(args.kl_threshold),
                top1_threshold=float(args.top1_threshold),
            )
            performance = _timed_prompt_pair(
                session,
                prompt_ids=tokens,
                baseline_mode=baseline_mode,
                candidate_mode=candidate_mode,
                decode_steps=performance_steps,
                repetitions=repetitions,
                bulk_attention_mode=str(args.bulk_attention_mode),
                graph_replay_decode=bool(args.graph_replay_decode),
            )
            records.append(
                {
                    "id": prompt_id,
                    "category": str(row["category"]),
                    "suite": str(row["suite"]),
                    "prompt_chars": len(str(row["prompt"])),
                    "prompt_sha256": prompt_sha256(str(row["prompt"])),
                    "prompt_tokens": len(tokens),
                    "prompt_token_sha256_i64": hashlib.sha256(
                        np.asarray(tokens, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "correctness": correctness,
                    "decode_performance": performance,
                }
            )
            print(
                f"{index + 1}/{len(prompt_rows)} {prompt_id}: "
                f"KL={correctness['kl_max']:.6g} "
                f"top1={correctness['top1_agreement']:.4f} "
                f"decode_delta={performance['candidate_wall_delta_percent']:+.3f}%",
                flush=True,
            )

    summary = _aggregate_gate(
        records,
        kl_threshold=float(args.kl_threshold),
        top1_threshold=float(args.top1_threshold),
    )
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "HIPENGINE_GGUF_GDN_PREFILL_MODE": (
                f"explicit {baseline_mode} versus {candidate_mode}"
            ),
            "HIPENGINE_GGUF_DECODE_REPACK": os.environ.get(
                "HIPENGINE_GGUF_DECODE_REPACK"
            ),
            "HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE": os.environ.get(
                "HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE"
            ),
        },
        build_profile="gguf_gdn_teacher_forced_semantic_gate",
        timing_protocol="multi_suite_teacher_forced_logits_balanced_decode_v1",
        warmups=1,
        repetitions=repetitions,
        profiler={"enabled": False, "kind": None, "command": None},
    )
    clean = not bool(provenance.get("dirty"))
    gate_passed = bool(clean and summary["passed"])
    if not clean:
        status = "invalid_dirty_measurement"
    elif summary["passed"]:
        status = "accepted_semantic_and_decode_nonregression"
    elif not summary["kl_passed"] or summary["top1_agreement"] < float(
        args.top1_threshold
    ):
        status = "rejected_semantic_correctness"
    else:
        status = "rejected_decode_regression"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate_passed": gate_passed,
        "measurement_valid": clean,
        "performance_claim": gate_passed,
        "correctness_claim": gate_passed,
        "protocol": {
            "model": str(args.model.resolve()),
            "prompt_suites": [str(path.resolve()) for path in args.prompts],
            "required_categories": sorted(REQUIRED_CATEGORIES),
            "baseline_mode": baseline_mode,
            "candidate_mode": candidate_mode,
            "correctness_decode_steps": correctness_steps,
            "performance_decode_steps": performance_steps,
            "performance_repetitions_per_mode_prompt": repetitions,
            "same_context_rule": (
                "candidate consumes the exact baseline-generated token prefix at "
                "every compared transition"
            ),
            "kl_threshold": float(args.kl_threshold),
            "aggregate_top1_threshold": float(args.top1_threshold),
            "decode_nonregression_rule": (
                "candidate sum of per-prompt median walls must not exceed baseline"
            ),
        },
        "summary": summary,
        "prompts": records,
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--prompts", action="append", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--baseline-mode", choices=SUPPORTED_MODES, default="chain_lds32_direct"
    )
    parser.add_argument(
        "--candidate-mode",
        choices=CANDIDATE_MODES,
        default="chain_wave32_tree",
    )
    parser.add_argument("--correctness-decode-steps", type=int, default=24)
    parser.add_argument("--performance-decode-steps", type=int, default=128)
    parser.add_argument("--performance-repetitions", type=int, default=2)
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.99)
    parser.add_argument(
        "--bulk-attention-mode", choices=("bulk", "native"), default="bulk"
    )
    parser.add_argument(
        "--graph-replay-decode", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--decode-repack", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-wmma-prefill", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-gemv-decode", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=512)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    if args.prompts is None:
        args.prompts = list(DEFAULT_PROMPTS)
    command = [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *raw_argv]
    try:
        artifact = run(args, command=command)
    except (GateError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.json)
    print(json.dumps({"status": artifact["status"], **artifact["summary"]}, indent=2))
    return 0 if artifact["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
