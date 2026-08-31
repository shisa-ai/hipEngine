#!/usr/bin/env python3
"""Gate the default-off Qwen4Exp H256 wave8 sparse-attention candidate at 4K."""

from __future__ import annotations

import argparse
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
from scripts.execution_profile_gdn_calibration import PromptCalibrationCapture, build_candidate_quality
from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, _git_metadata, _host_metadata, sha256_path, token_ids_sha256
from scripts.qwen4exp_layer2_profile_gate import _make_generator, _state_repeat_gate, _state_summary

ENV = "HIPENGINE_QWEN4_EXP_QSA_WAVE8_CONTIGUOUS_H256"
DEFAULT_CASES = ["code-p4096", "general_en-p4096", "general_ja-p4096", "mixed_ja_en-p4096"]


def _ids_sha256(values: Sequence[int]) -> str:
    return token_ids_sha256(int(value) for value in values)


def _task_summary(strict: Mapping[str, Sequence[int]], candidate: Mapping[str, Sequence[Sequence[int]]]) -> dict[str, Any]:
    divergences: list[str] = []
    repeat_mismatches: list[str] = []
    for prompt_id, expected in strict.items():
        runs = [list(run) for run in candidate[prompt_id]]
        if not runs or any(run != runs[0] for run in runs[1:]):
            repeat_mismatches.append(prompt_id)
        if not runs or runs[0] != list(expected):
            divergences.append(prompt_id)
    return {
        "prompts": len(strict),
        "strict_exact": len(strict) - len(divergences),
        "candidate_repeat_exact": not repeat_mismatches,
        "passed": not divergences and not repeat_mismatches,
        "divergences": divergences,
        "repeat_mismatches": repeat_mismatches,
    }


def _restore_env(value: str | None) -> None:
    if value is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = value


def _numerical_trajectory(runner: Any, snapshot: Any, root_row: Mapping[str, Any], forced: Sequence[int]) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    runner.restore(snapshot)
    rows: list[dict[str, Any]] = [{"token_id": int(root_row["token_id"]), "logits": np.ascontiguousarray(root_row["logits"], dtype=np.float32)}]
    for token in forced:
        result = runner.step(int(token))
        rows.append({"token_id": int(result.token_id), "logits": np.ascontiguousarray(result.logits, dtype=np.float32)})
    return tuple(rows), _state_summary(runner)


def _strict_trajectory(runner: Any, snapshot: Any, root_row: Mapping[str, Any], steps: int) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    runner.restore(snapshot)
    rows: list[dict[str, Any]] = [{"token_id": int(root_row["token_id"]), "logits": np.ascontiguousarray(root_row["logits"], dtype=np.float32)}]
    current = int(root_row["token_id"])
    for _ in range(int(steps)):
        result = runner.step(current)
        current = int(result.token_id)
        rows.append({"token_id": current, "logits": np.ascontiguousarray(result.logits, dtype=np.float32)})
    return tuple(rows), _state_summary(runner)


def _free_trajectory(runner: Any, snapshot: Any, root_token: int, steps: int) -> tuple[list[int], dict[str, Any]]:
    runner.restore(snapshot)
    ids = [int(root_token)]
    current = int(root_token)
    for _ in range(1, int(steps)):
        result = runner.step(current, capture_logits=False, capture_target_hidden=False)
        current = int(result.token_id)
        ids.append(current)
    return ids, _state_summary(runner)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--free-tokens", type=int, default=32)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model_root.is_dir() or not args.fixture.is_file():
        raise ValueError("model root and fixture must exist")
    if args.decode_steps <= 0 or args.repeat_runs < 3 or args.free_tokens <= 1:
        raise ValueError("decode steps >0, repeats >=3, and free tokens >1 are required")
    fixture = json.loads(args.fixture.read_text())
    by_id = {str(row["id"]): row for row in fixture["cases"]}
    cases = [by_id[str(case_id)] for case_id in args.case_id]
    if len(cases) != 4 or {str(row["category"]) for row in cases} != {"code", "general_en", "general_ja", "mixed_ja_en"}:
        raise ValueError("the binding gate requires all four canonical p4096 categories")

    if args.compiler_version_file is not None:
        os.environ.setdefault("HIPENGINE_COMPILER_VERSION_FILE", str(args.compiler_version_file))
    if args.require_cached_build:
        os.environ.setdefault("HIPENGINE_REQUIRE_CACHED_BUILD", "1")
    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")

    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    candidate_fn = resolve(backend="hip_gfx1151", layer="qsa_sparse_attention", quant="bf16_kv", variant="production_wave8_contiguous_h256_spans")
    fallback_fn = resolve(backend="hip_gfx1151", layer="qsa_sparse_attention", quant="bf16_kv", variant="strict_spans")
    reset_memory_stats()
    factory_args = argparse.Namespace(model_root=args.model_root, max_sequence_length=4352, prefill_chunk_size=int(args.prefill_chunk_size))
    generator, resolved, _index = _make_generator(factory_args, "production")
    bound = os.environ.get(ENV)
    captures: list[PromptCalibrationCapture] = []
    strict_states: list[dict[str, Any]] = []
    candidate_states: list[list[dict[str, Any]]] = []
    strict_tasks: dict[str, list[int]] = {}
    candidate_tasks: dict[str, list[list[int]]] = {}
    rows_manifest: list[dict[str, Any]] = []
    try:
        runner = generator.runner
        for number, row in enumerate(cases, 1):
            prompt_id = str(row["id"])
            prompt_ids = tuple(int(value) for value in row["prompt_token_ids"])
            os.environ[ENV] = "0"
            root = runner.prefill(prompt_ids)
            root_row = {"token_id": int(root.token_id), "logits": np.ascontiguousarray(root.logits, dtype=np.float32)}
            snapshot = runner.snapshot()
            strict, strict_state = _strict_trajectory(runner, snapshot, root_row, int(args.decode_steps))
            strict_state["prompt_id"] = prompt_id
            forced = [int(step["token_id"]) for step in strict[:-1]]
            strict_ids, _strict_task_state = _free_trajectory(runner, snapshot, int(root.token_id), int(args.free_tokens))
            os.environ[ENV] = "1"
            runs: list[tuple[dict[str, Any], ...]] = []
            states: list[dict[str, Any]] = []
            task_runs: list[list[int]] = []
            for _ in range(int(args.repeat_runs)):
                candidate_run, state = _numerical_trajectory(runner, snapshot, root_row, forced)
                state["prompt_id"] = prompt_id
                runs.append(candidate_run)
                states.append(state)
                task_ids, _task_state = _free_trajectory(runner, snapshot, int(root.token_id), int(args.free_tokens))
                task_runs.append(task_ids)
            captures.append(PromptCalibrationCapture(prompt_id=prompt_id, category=str(row["category"]), strict=strict, candidate_runs={"qsa_wave8_contiguous_h256": tuple(runs)}))
            strict_states.append(strict_state)
            candidate_states.append(states)
            strict_tasks[prompt_id] = strict_ids
            candidate_tasks[prompt_id] = task_runs
            rows_manifest.append({"id": prompt_id, "category": str(row["category"]), "prompt_tokens": len(prompt_ids), "prompt_token_ids_sha256": _ids_sha256(prompt_ids)})
            print(f"candidate {number}/{len(cases)} {prompt_id}: {args.repeat_runs} numerical + task", flush=True)
    finally:
        _restore_env(bound)
        generator.close()
    after_close = memory_stats()

    quality = build_candidate_quality(tuple(captures), candidate_mode="qsa_wave8_contiguous_h256", scenario_id="qwen4exp-qsa-contiguous-h256-p4096", thresholds=EvaluationThresholds())
    state_gate = _state_repeat_gate(strict_states, candidate_states)
    task_gate = _task_summary(strict_tasks, candidate_tasks)
    source = _git_metadata(ROOT)
    measurement_valid = bool(source and source["tracked_clean"] and after_close["current_allocated_bytes"] == 0)
    passed = bool(measurement_valid and quality["quality"]["hard_gates_passed"] and quality["repeat_determinism"]["passed"] and state_gate["passed"] and task_gate["passed"])
    return {
        "schema": 1, "kind": "qwen4exp_qsa_h256_candidate_gate", "status": "passed" if passed else "failed", "measurement_valid": measurement_valid,
        "command": list(command), "source": source, "host": _host_metadata(), "model_root": str(args.model_root),
        "fixture": str(args.fixture), "fixture_sha256": sha256_path(args.fixture), "cases": rows_manifest,
        "candidate": {"classification": "T1", "environment": {ENV: "1"}, "registry_key": ["hip_gfx1151", "qsa_sparse_attention", "bf16_kv", "production_wave8_contiguous_h256_spans"], "fallback_key": ["hip_gfx1151", "qsa_sparse_attention", "bf16_kv", "strict_spans"], "candidate_registered": callable(candidate_fn), "fallback_registered": callable(fallback_fn)},
        "profile": {"manifest_sha256": resolved.manifest_sha256, "strict_manifest_sha256": resolved.strict_manifest_sha256},
        "protocol": {"decode_steps": int(args.decode_steps), "repeat_runs": int(args.repeat_runs), "free_tokens": int(args.free_tokens), "prefill_chunk_size": int(args.prefill_chunk_size), "kv": "BF16"},
        "quality": quality, "state_repeat_gate": state_gate, "task_gate": task_gate,
        "lifecycle": {"after_close": after_close, "passed": after_close["current_allocated_bytes"] == 0},
        "decision": {"passed": passed, "next": "physical c2, trace, and canonical p4096" if passed else "reject candidate"},
    }


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args, command=[str(Path(sys.argv[0]).name), *sys.argv[1:]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
