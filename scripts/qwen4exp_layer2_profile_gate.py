#!/usr/bin/env python3
"""Qualify a Qwen4Exp execution-profile candidate against its strict parent.

The strict and candidate sessions use the same GGUF/BF16-KV configuration and
prompt/teacher-token schedule. Strict runs under the registered strict profile.
T0 exact candidates also use strict plus their post-binder override; T1/T2
candidates use the registered production stack plus their override. The gate records 450
full-vocabulary rows by default (18 prompts x prefill-last plus 24 c1 rows),
three candidate repeats, request-local state fingerprints, and 32-token free
outputs. A cross-route free-output difference requires a separate task review;
it is never silently accepted by this harness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.benchmark.execution_profiles import EvaluationThresholds
from scripts.execution_profile_gdn_calibration import (
    PromptCalibrationCapture,
    build_candidate_quality,
)
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import prompt_sha256
from scripts.qwen4exp_canonical_ar_bench import (
    _git_metadata,
    _host_metadata,
    _write_json,
    sha256_path,
    token_ids_sha256,
)

KIND = "qwen4exp_execution_profile_candidate_gate"


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    name: str
    classification: str
    mechanism: str
    environment: Mapping[str, str]
    base_profile: str
    scenario_id: str
    candidate_key: tuple[str, str, str, str]
    fallback_key: tuple[str, str, str, str]


CANDIDATES = {
    "layer2_grouped_q5k": CandidateSpec(
        name="layer2_grouped_q5k_wmma",
        classification="T2",
        mechanism="layer-2 Q5_K/Q5_K compact f16-WMMA grouped prefill",
        environment={"HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL": "1"},
        base_profile="production",
        scenario_id="qwen4exp-ud-q4-k-xl-layer2-grouped-q5k",
        candidate_key=(
            "hip_gfx1151", "moe_linear", "gguf_q5_k",
            "selected_wmma_prefill_compact_bf16_bf16_out",
        ),
        fallback_key=(
            "hip_gfx1151", "linear", "gguf_q5_k",
            "selected_gemv_bf16_bf16_out",
        ),
    ),
    "device_argmax": CandidateSpec(
        name="device_argmax",
        classification="T0",
        mechanism="select greedy token on device and copy one int64",
        environment={"HIPENGINE_QWEN4_EXP_DEVICE_ARGMAX": "1"},
        base_profile="strict",
        scenario_id="qwen4exp-ud-q4-k-xl-device-argmax",
        candidate_key=("hip_gfx1151", "argmax", "f32", "top1_i64"),
        fallback_key=("hip_gfx1151", "argmax", "f32", "top1_i64"),
    ),
    "q8_mmq_attn_gate": CandidateSpec(
        name="q8_mmq_attn_gate",
        classification="T2",
        mechanism="extend guarded Q8 MMQ F32 prefill to GDN attn_gate K2560/N6144",
        environment={"HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE": "1"},
        base_profile="production",
        scenario_id="qwen4exp-ud-q4-k-xl-q8-mmq-attn-gate",
        candidate_key=(
            "hip_gfx1151", "linear", "gguf_q8_0",
            "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out",
        ),
        fallback_key=(
            "hip_gfx1151", "linear", "gguf_q8_0",
            "coltile8_rowbatch4_f32_f32_out",
        ),
    ),
}


class GateError(RuntimeError):
    """Raised when the profile gate cannot be evaluated safely."""


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint16)
    return (values.astype(np.uint32) << 16).view(np.float32)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _state_summary(runner: Any) -> dict[str, Any]:
    snapshot = runner.snapshot()
    digest = hashlib.sha256()
    layout: dict[str, int] = {}
    finite = True
    for name, raw in sorted(snapshot.decode_state.buffers.items()):
        value = np.ascontiguousarray(raw, dtype=np.uint8)
        layout[str(name)] = int(value.nbytes)
        digest.update(str(name).encode())
        digest.update(b"\0")
        digest.update(value)
        if name == "residual":
            finite = finite and bool(np.isfinite(_bf16_to_f32(value.view(np.uint16))).all())
        else:
            finite = finite and bool(np.isfinite(value.view(np.float32)).all())
    ple = {
        str(layer): {
            "tokens": [int(token) for token in state.tokens],
            "next_position": int(state.next_position),
        }
        for layer, state in sorted(snapshot.ple_hash_states.items())
    }
    attention_positions = [
        [int(state.position_host[0]), int(state.context_host[0])]
        for state in runner.attention_states
    ]
    index_counts = [
        [int(state.count), int(state.pooled_count)] for state in runner.index_states
    ]
    metadata = {
        "position": int(snapshot.position),
        "ple": ple,
        "attention_positions": attention_positions,
        "index_counts": index_counts,
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    return {
        "state_sha256": digest.hexdigest(),
        "layout_sha256": _hash_json(layout),
        "buffer_bytes": layout,
        "finite": bool(finite),
        **metadata,
    }


def _state_repeat_gate(
    strict_by_prompt: Sequence[Mapping[str, Any]],
    candidate_by_prompt: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if len(strict_by_prompt) != len(candidate_by_prompt):
        raise ValueError("strict and candidate state prompt counts differ")
    prompts: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for strict, candidate_values in zip(
        strict_by_prompt, candidate_by_prompt, strict=True
    ):
        runs = tuple(candidate_values)
        if len(runs) < 3:
            raise ValueError("state repeat gate requires at least three candidate runs")
        first = runs[0]
        finite = bool(strict["finite"]) and all(bool(run["finite"]) for run in runs)
        layout_exact = all(
            run["layout_sha256"] == strict["layout_sha256"] for run in runs
        )
        metadata_exact = all(
            run["position"] == strict["position"]
            and run["attention_positions"] == strict["attention_positions"]
            and run["index_counts"] == strict["index_counts"]
            for run in runs
        )
        repeat_exact = all(
            run["state_sha256"] == first["state_sha256"] for run in runs[1:]
        )
        prompt_id = str(strict["prompt_id"])
        result = {
            "prompt_id": prompt_id,
            "finite": finite,
            "layout_exact": layout_exact,
            "metadata_exact": metadata_exact,
            "repeat_exact": repeat_exact,
            "strict_state_sha256": strict["state_sha256"],
            "candidate_state_sha256": [run["state_sha256"] for run in runs],
            "strict_candidate_state_exact": (
                strict["state_sha256"] == first["state_sha256"]
            ),
        }
        prompts.append(result)
        if not (finite and layout_exact and metadata_exact and repeat_exact):
            mismatches.append(result)
    return {"passed": not mismatches, "mismatches": mismatches, "prompts": prompts}


def _task_gate(
    strict: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    categories: Mapping[str, str],
) -> dict[str, Any]:
    if set(strict) != set(candidate) or set(strict) != set(categories):
        raise ValueError("task result prompt sets differ")
    divergences: list[dict[str, Any]] = []
    repeat_mismatches: list[str] = []
    rows: list[dict[str, Any]] = []
    for prompt_id in strict:
        runs = tuple(candidate[prompt_id])
        if len(runs) < 2:
            raise ValueError("task gate requires at least two candidate runs")
        first = runs[0]
        repeat_exact = all(
            run["ids"] == first["ids"] and run["text"] == first["text"]
            for run in runs[1:]
        )
        strict_exact = (
            strict[prompt_id]["ids"] == first["ids"]
            and strict[prompt_id]["text"] == first["text"]
        )
        if not repeat_exact:
            repeat_mismatches.append(prompt_id)
        if not strict_exact:
            divergences.append(
                {
                    "id": prompt_id,
                    "category": categories[prompt_id],
                    "strict_text": strict[prompt_id]["text"],
                    "candidate_text": first["text"],
                    "strict_ids_sha256": strict[prompt_id]["ids_sha256"],
                    "candidate_ids_sha256": first["ids_sha256"],
                }
            )
        rows.append(
            {
                "id": prompt_id,
                "category": categories[prompt_id],
                "candidate_repeat_exact": repeat_exact,
                "strict_exact": strict_exact,
                "strict_ids_sha256": strict[prompt_id]["ids_sha256"],
                "candidate_ids_sha256": first["ids_sha256"],
            }
        )
    repeat_pass = not repeat_mismatches
    status = (
        "failed_nondeterminism"
        if not repeat_pass
        else "requires_review"
        if divergences
        else "passed_exact"
    )
    return {
        "status": status,
        "candidate_repeat_exact": repeat_pass,
        "repeat_mismatches": repeat_mismatches,
        "strict_exact_count": len(rows) - len(divergences),
        "total": len(rows),
        "divergences": divergences,
        "rows": rows,
    }


def _ids_sha256(ids: Sequence[int]) -> str:
    return token_ids_sha256(int(value) for value in ids)


def _free_trajectory(runner: Any, tokenizer: Any, prompt_ids: Sequence[int], steps: int) -> dict[str, Any]:
    runner.reset()
    result = runner.prefill([int(token) for token in prompt_ids])
    ids: list[int] = []
    for index in range(int(steps)):
        token = int(result.token_id)
        ids.append(token)
        if index + 1 < steps:
            result = runner.step(token)
    return {
        "ids": ids,
        "ids_sha256": _ids_sha256(ids),
        "text": str(tokenizer.decode(ids, skip_special=False)),
    }


def _strict_trajectory(runner: Any, prompt_ids: Sequence[int], decode_steps: int) -> tuple[dict[str, Any], ...]:
    runner.reset()
    result = runner.prefill([int(token) for token in prompt_ids])
    trajectory = [
        {
            "token_id": int(result.token_id),
            "logits": np.ascontiguousarray(result.logits, dtype=np.float32),
        }
    ]
    current = int(result.token_id)
    for _ in range(int(decode_steps)):
        result = runner.step(current)
        current = int(result.token_id)
        trajectory.append(
            {
                "token_id": current,
                "logits": np.ascontiguousarray(result.logits, dtype=np.float32),
            }
        )
    return tuple(trajectory)


def _candidate_trajectory(
    runner: Any,
    prompt_ids: Sequence[int],
    forced_input_ids: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    runner.reset()
    result = runner.prefill([int(token) for token in prompt_ids])
    trajectory = [
        {
            "token_id": int(result.token_id),
            "logits": np.ascontiguousarray(result.logits, dtype=np.float32),
        }
    ]
    for token in forced_input_ids:
        result = runner.step(int(token))
        trajectory.append(
            {
                "token_id": int(result.token_id),
                "logits": np.ascontiguousarray(result.logits, dtype=np.float32),
            }
        )
    return tuple(trajectory)


def _make_generator(args: argparse.Namespace, profile: str):
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
    )
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model

    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    plugin = resolve_model(index.architecture or "")
    requested = ExecutionProfile(profile)
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=requested,
    )

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=args.model_root,
            weight_index=index,
            model_plugin=plugin,
            backend=QWEN4_EXP_BACKEND,
            max_sequence_length=int(args.max_sequence_length),
            prefill_chunk_size=int(args.prefill_chunk_size),
        )

    return resolved.construct_generator(factory), resolved, index


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model_root.is_dir():
        raise GateError(f"model root does not exist: {args.model_root}")
    if args.decode_steps <= 0 or args.repeat_runs < 3 or args.free_runs < 2:
        raise GateError("decode steps must be positive; candidate repeats >=3; free runs >=2")
    prompt_rows = _load_suites(args.prompts)
    if args.limit is not None:
        prompt_rows = prompt_rows[: max(0, int(args.limit))]
    if not prompt_rows:
        raise GateError("selected prompt suite is empty")
    complete_suite = args.limit is None
    candidate_spec = CANDIDATES[str(args.candidate)]

    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    candidate_kernel = resolve(
        backend=candidate_spec.candidate_key[0],
        layer=candidate_spec.candidate_key[1],
        quant=candidate_spec.candidate_key[2],
        variant=candidate_spec.candidate_key[3],
    )
    strict_fallback = resolve(
        backend=candidate_spec.fallback_key[0],
        layer=candidate_spec.fallback_key[1],
        quant=candidate_spec.fallback_key[2],
        variant=candidate_spec.fallback_key[3],
    )

    prompt_tokens: dict[str, list[int]] = {}
    strict_trajectories: dict[str, tuple[Mapping[str, Any], ...]] = {}
    strict_states: list[dict[str, Any]] = []
    strict_tasks: dict[str, dict[str, Any]] = {}
    prompt_manifest: list[dict[str, Any]] = []

    reset_memory_stats()
    strict_generator, strict_profile, index = _make_generator(args, "strict")
    try:
        for number, row in enumerate(prompt_rows, 1):
            prompt_id = str(row["id"])
            tokens = build_chat_prompt(strict_generator.tokenizer, str(row["prompt"]))
            prompt_tokens[prompt_id] = [int(token) for token in tokens]
            strict = _strict_trajectory(
                strict_generator.runner,
                prompt_tokens[prompt_id],
                int(args.decode_steps),
            )
            state = _state_summary(strict_generator.runner)
            state["prompt_id"] = prompt_id
            strict_trajectories[prompt_id] = strict
            strict_states.append(state)
            strict_tasks[prompt_id] = _free_trajectory(
                strict_generator.runner,
                strict_generator.tokenizer,
                prompt_tokens[prompt_id],
                int(args.free_tokens),
            )
            prompt_manifest.append(
                {
                    "id": prompt_id,
                    "category": str(row["category"]),
                    "suite": str(row["suite"]),
                    "prompt_sha256": prompt_sha256(str(row["prompt"])),
                    "prompt_tokens": len(tokens),
                    "prompt_token_ids_sha256": _ids_sha256(tokens),
                }
            )
            print(f"strict {number}/{len(prompt_rows)} {prompt_id}", flush=True)
    finally:
        strict_generator.close()
    strict_after_close = memory_stats()

    candidate_runs: dict[str, list[tuple[Mapping[str, Any], ...]]] = {}
    candidate_states: list[list[dict[str, Any]]] = []
    candidate_tasks: dict[str, list[dict[str, Any]]] = {}
    reset_memory_stats()
    candidate_generator, candidate_profile, _ = _make_generator(
        args, candidate_spec.base_profile
    )
    bound_environment = {
        key: os.environ.get(key) for key in candidate_spec.environment
    }
    os.environ.update(candidate_spec.environment)
    try:
        for number, row in enumerate(prompt_rows, 1):
            prompt_id = str(row["id"])
            strict = strict_trajectories[prompt_id]
            forced = [int(step["token_id"]) for step in strict[:-1]]
            runs: list[tuple[Mapping[str, Any], ...]] = []
            states: list[dict[str, Any]] = []
            for _ in range(int(args.repeat_runs)):
                run = _candidate_trajectory(
                    candidate_generator.runner,
                    prompt_tokens[prompt_id],
                    forced,
                )
                state = _state_summary(candidate_generator.runner)
                state["prompt_id"] = prompt_id
                runs.append(run)
                states.append(state)
            candidate_runs[prompt_id] = runs
            candidate_states.append(states)
            candidate_tasks[prompt_id] = [
                _free_trajectory(
                    candidate_generator.runner,
                    candidate_generator.tokenizer,
                    prompt_tokens[prompt_id],
                    int(args.free_tokens),
                )
                for _ in range(int(args.free_runs))
            ]
            print(
                f"candidate {number}/{len(prompt_rows)} {prompt_id}: "
                f"{args.repeat_runs} numerical + {args.free_runs} free",
                flush=True,
            )
    finally:
        for key, value in bound_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        candidate_generator.close()
    candidate_after_close = memory_stats()

    captures = tuple(
        PromptCalibrationCapture(
            prompt_id=str(row["id"]),
            category=str(row["category"]),
            strict=strict_trajectories[str(row["id"])],
            candidate_runs={
                candidate_spec.name: tuple(candidate_runs[str(row["id"])])
            },
        )
        for row in prompt_rows
    )
    thresholds = EvaluationThresholds()
    quality = build_candidate_quality(
        captures,
        candidate_mode=candidate_spec.name,
        scenario_id=candidate_spec.scenario_id,
        thresholds=thresholds,
    )
    state_gate = _state_repeat_gate(strict_states, candidate_states)
    categories = {str(row["id"]): str(row["category"]) for row in prompt_rows}
    task_gate = _task_gate(strict_tasks, candidate_tasks, categories=categories)
    source = _git_metadata(ROOT)
    tracked_clean = bool(source["tracked_clean"])
    teardown = bool(
        strict_after_close["current_allocated_bytes"] == 0
        and candidate_after_close["current_allocated_bytes"] == 0
    )
    numerical_pass = bool(
        quality["quality"]["hard_gates_passed"]
        and quality["repeat_determinism"]["passed"]
    )
    task_pass = task_gate["status"] == "passed_exact"
    measurement_valid = bool(
        complete_suite and tracked_clean and teardown and state_gate["passed"]
    )
    passed = bool(measurement_valid and numerical_pass and task_pass)
    status = (
        "passed"
        if passed
        else "requires_task_review"
        if measurement_valid and numerical_pass and task_gate["status"] == "requires_review"
        else "failed"
    )
    return {
        "schema": 1,
        "kind": KIND,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "measurement_valid": measurement_valid,
        "source": source,
        "host": _host_metadata(),
        "command": list(command),
        "model": {
            "root": str(args.model_root.resolve()),
            "architecture": index.architecture,
        },
        "candidate": {
            "name": candidate_spec.name,
            "candidate_id": str(args.candidate),
            "classification": candidate_spec.classification,
            "mechanism": candidate_spec.mechanism,
            "environment": dict(candidate_spec.environment),
            "base_profile": candidate_spec.base_profile,
            "bound_environment": bound_environment,
            "candidate_kernel": getattr(candidate_kernel, "__name__", str(candidate_kernel)),
            "strict_fallback": getattr(strict_fallback, "__name__", str(strict_fallback)),
            "candidate_kernel_registered": True,
            "strict_fallback_registered": True,
        },
        "profiles": {
            "strict_manifest": _json_value(strict_profile.manifest),
            "strict_manifest_sha256": strict_profile.manifest_sha256,
            "candidate_base_profile": candidate_spec.base_profile,
            "candidate_manifest": _json_value(candidate_profile.manifest),
            "candidate_manifest_sha256": candidate_profile.manifest_sha256,
            "candidate_named_profile_intact": False,
        },
        "protocol": {
            "prompt_suites": [str(path.resolve()) for path in args.prompts],
            "prompt_suite_sha256": {
                str(path.resolve()): sha256_path(path.resolve()) for path in args.prompts
            },
            "complete_prompt_and_heldout_suite": complete_suite,
            "prompt_count": len(prompt_rows),
            "decode_steps": int(args.decode_steps),
            "teacher_forced_rows": sum(len(capture.strict) for capture in captures),
            "candidate_repeat_runs": int(args.repeat_runs),
            "free_tokens": int(args.free_tokens),
            "candidate_free_runs": int(args.free_runs),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "kv": "BF16",
            "thresholds": thresholds.to_dict(),
        },
        "prompts": prompt_manifest,
        "quality": quality,
        "state_repeat_gate": state_gate,
        "task_gate": task_gate,
        "lifecycle": {
            "strict_after_close": strict_after_close,
            "candidate_after_close": candidate_after_close,
            "passed": teardown,
        },
        "decision": {
            "passed": passed,
            "numerical_passed": numerical_pass,
            "state_passed": state_gate["passed"],
            "task_passed": task_pass,
            "lifecycle_passed": teardown,
            "next": (
                "complete prompt and heldout suite required"
                if not complete_suite
                else "task review required for deterministic cross-route divergences"
                if status == "requires_task_review"
                else "eligible for manifest/c2/canonical-depth promotion work"
                if status == "passed"
                else "reject candidate and keep strict fallback"
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=tuple(CANDIDATES),
        default="layer2_grouped_q5k",
    )
    parser.add_argument("--prompts", action="append", type=Path, default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--free-tokens", type=int, default=32)
    parser.add_argument("--free-runs", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=2051)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--hip-arch", default="gfx1151")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prompts is None:
        args.prompts = list(DEFAULT_PROMPTS)
    os.environ.setdefault("HIPENGINE_HIP_ARCH", args.hip_arch)
    if args.compiler_version_file is not None:
        os.environ.setdefault(
            "HIPENGINE_COMPILER_VERSION_FILE",
            str(args.compiler_version_file.resolve()),
        )
    if args.require_cached_build:
        os.environ.setdefault("HIPENGINE_REQUIRE_CACHED_BUILD", "1")
    payload = run(args, command=[str(Path(sys.argv[0]).name), *sys.argv[1:]])
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload["status"] == "passed" else 2 if payload["status"] == "requires_task_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
