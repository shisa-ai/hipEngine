#!/usr/bin/env python3
"""Gate one GGUF c1 numerical route against the strict same-quant fallback.

The adapter keeps strict and candidate arithmetic in one resident session,
teacher-forces every candidate onto the strict token trajectory, evaluates all
full-vocabulary rows with the calibrated execution-profile envelope, and
requires three bit-stable candidate repeats.  It also fingerprints the live
Conv/GDN/KV state after every trajectory to prove same-shape repeatability.
Performance is measured separately so full-logit D2H traffic cannot contaminate
wall claims.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.execution_profiles import EvaluationThresholds
from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.execution_profile_gdn_calibration import (
    PromptCalibrationCapture,
    build_candidate_quality,
    validate_strict_baseline,
)
from scripts.gguf_decode_graph_g5 import _capture_checkpoint, _checkpoint_summary
from scripts.gguf_gdn_semantic_gate import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    _configure_gate_environment,
    _load_suites,
    _run_teacher_forced_candidate,
)
from scripts.gguf_gdn_trajectory_gate import _run_logits_trajectory
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import prompt_sha256

KIND = "hipengine_execution_profile_gguf_c1_route_gate"
SCHEMA_VERSION = 1
_ROUTE_ENV_KEYS = (
    "HIPENGINE_GGUF_ROUTER_F32W_COOP",
    "HIPENGINE_GGUF_ROUTER_F32W_PERSISTENT_COUNTER",
)


class GateError(RuntimeError):
    """Raised when the c1 candidate packet cannot be evaluated honestly."""


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    classification: str
    mechanism: str
    strict_fallback: str
    environment: Mapping[str, str]


CANDIDATES = {
    "router_f32w_coop_persistent": Candidate(
        name="router_f32w_coop_persistent",
        classification="T2",
        mechanism=(
            "contract separate F32 expert logits, shared logit, and top-k into "
            "one cooperative launch with a self-resetting device counter"
        ),
        strict_fallback="separate router projections plus qwen35_router_select",
        environment={
            "HIPENGINE_GGUF_ROUTER_F32W_COOP": "1",
            "HIPENGINE_GGUF_ROUTER_F32W_PERSISTENT_COUNTER": "1",
        },
    ),
}
STRICT_ENVIRONMENT = {
    "HIPENGINE_GGUF_ROUTER_F32W_COOP": "0",
    "HIPENGINE_GGUF_ROUTER_F32W_PERSISTENT_COUNTER": "0",
}


@contextmanager
def route_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Apply a complete c1 route environment and restore the caller exactly."""

    unknown = sorted(set(values) - set(_ROUTE_ENV_KEYS))
    missing = sorted(set(_ROUTE_ENV_KEYS) - set(values))
    if unknown or missing:
        raise ValueError(f"route environment mismatch; missing={missing}, unknown={unknown}")
    previous = {key: os.environ.get(key) for key in _ROUTE_ENV_KEYS}
    try:
        for key in _ROUTE_ENV_KEYS:
            os.environ[key] = str(values[key])
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _state_summary(
    session: Any,
    trajectory: Sequence[Mapping[str, object]],
    forced_input_ids: Sequence[int],
) -> dict[str, Any]:
    if not trajectory:
        raise ValueError("state summary needs a non-empty trajectory")
    input_token_id = (
        int(forced_input_ids[-1])
        if forced_input_ids
        else int(trajectory[0]["token_id"])
    )
    checkpoint = _capture_checkpoint(
        session,
        position=int(session.position),
        input_token_id=input_token_id,
        predicted_token_id=int(trajectory[-1]["token_id"]),
    )
    return _checkpoint_summary(checkpoint)


def build_state_repeat_gate(
    strict_by_prompt: Sequence[Mapping[str, Any]],
    candidate_by_prompt: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Require finite, layout-stable, byte-repeatable candidate state."""

    if len(strict_by_prompt) != len(candidate_by_prompt):
        raise ValueError("strict and candidate state prompt counts differ")
    mismatches: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    for strict, candidate_runs in zip(strict_by_prompt, candidate_by_prompt, strict=True):
        runs = tuple(candidate_runs)
        if len(runs) < 3:
            raise ValueError("state repeat gate requires at least three candidate runs")
        first = runs[0]
        finite = bool(strict.get("finite")) and all(bool(run.get("finite")) for run in runs)
        layout_stable = all(
            run.get("position") == strict.get("position")
            and run.get("linear_state_pairs") == strict.get("linear_state_pairs")
            and run.get("full_attention_kv_pairs") == strict.get("full_attention_kv_pairs")
            for run in runs
        )
        repeatable = all(run.get("state_sha256") == first.get("state_sha256") for run in runs[1:])
        prompt_id = str(strict.get("prompt_id"))
        if not finite or not layout_stable or not repeatable:
            mismatches.append(
                {
                    "prompt_id": prompt_id,
                    "finite": finite,
                    "layout_stable": layout_stable,
                    "repeatable": repeatable,
                }
            )
        prompts.append(
            {
                "prompt_id": prompt_id,
                "strict_state_sha256": strict.get("state_sha256"),
                "candidate_state_sha256": [run.get("state_sha256") for run in runs],
                "strict_and_candidate_bytes_equal": (
                    strict.get("state_sha256") == first.get("state_sha256")
                ),
                "finite": finite,
                "layout_stable": layout_stable,
                "repeatable": repeatable,
            }
        )
    return {"passed": not mismatches, "mismatches": mismatches, "prompts": prompts}


def _capture(
    args: argparse.Namespace,
    *,
    prompt_rows: Sequence[Mapping[str, Any]],
    candidate: Candidate,
) -> tuple[
    tuple[PromptCalibrationCapture, ...],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[list[dict[str, Any]]],
    str,
    str,
]:
    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFResidentSession,
        _gguf_gdn_prefill_backend_exact_mode,
    )
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    compiler_version = (
        None
        if args.compiler_version_file is None
        else args.compiler_version_file.read_text(encoding="utf-8")
    )
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    prompt_tokens = {
        str(row["id"]): build_chat_prompt(tokenizer, str(row["prompt"]))
        for row in prompt_rows
    }
    max_sequence_length = max(len(tokens) for tokens in prompt_tokens.values()) + int(args.decode_steps) + 2
    captures: list[PromptCalibrationCapture] = []
    prompt_manifest: list[dict[str, Any]] = []
    strict_states: list[dict[str, Any]] = []
    candidate_states: list[list[dict[str, Any]]] = []
    with Qwen35GGUFResidentSession(
        args.model,
        backend=str(args.backend),
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=bool(args.use_gemv_decode),
        prefill_config=PrefillConfig(attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens)),
    ) as session:
        if session.runner is None:
            raise GateError("GGUF resident session closed during setup")
        resolved_backend = str(session.runner.backend)
        target_arch = str(session.runner.target_arch)
        validate_strict_baseline(
            requested_mode=str(args.baseline_gdn_mode),
            backend_exact_mode=_gguf_gdn_prefill_backend_exact_mode(resolved_backend),
        )
        for index, row in enumerate(prompt_rows):
            prompt_id = str(row["id"])
            tokens = prompt_tokens[prompt_id]
            with route_environment(STRICT_ENVIRONMENT):
                strict = tuple(
                    _run_logits_trajectory(
                        session,
                        prompt_ids=tokens,
                        mode=str(args.baseline_gdn_mode),
                        decode_steps=int(args.decode_steps),
                        bulk_attention_mode=str(args.bulk_attention_mode),
                    )
                )
                forced = [int(step["token_id"]) for step in strict[:-1]]
                strict_state = _state_summary(session, strict, forced)
            strict_state["prompt_id"] = prompt_id
            runs: list[tuple[Mapping[str, object], ...]] = []
            state_runs: list[dict[str, Any]] = []
            for _ in range(int(args.repeat_runs)):
                with route_environment(candidate.environment):
                    run = tuple(
                        _run_teacher_forced_candidate(
                            session,
                            prompt_ids=tokens,
                            forced_input_ids=forced,
                            mode=str(args.baseline_gdn_mode),
                            bulk_attention_mode=str(args.bulk_attention_mode),
                        )
                    )
                    state = _state_summary(session, run, forced)
                state["prompt_id"] = prompt_id
                runs.append(run)
                state_runs.append(state)
            captures.append(
                PromptCalibrationCapture(
                    prompt_id=prompt_id,
                    category=str(row["category"]),
                    strict=strict,
                    candidate_runs={candidate.name: tuple(runs)},
                )
            )
            strict_states.append(strict_state)
            candidate_states.append(state_runs)
            prompt_manifest.append(
                {
                    "id": prompt_id,
                    "category": str(row["category"]),
                    "suite": str(row["suite"]),
                    "prompt_sha256": prompt_sha256(str(row["prompt"])),
                    "prompt_tokens": len(tokens),
                    "prompt_token_ids_sha256": hashlib.sha256(
                        np.asarray(tokens, dtype="<i8").tobytes()
                    ).hexdigest(),
                }
            )
            print(
                f"{index + 1}/{len(prompt_rows)} {prompt_id}: strict + "
                f"{args.repeat_runs} candidate trajectories/states",
                flush=True,
            )
    return (
        tuple(captures),
        prompt_manifest,
        strict_states,
        candidate_states,
        resolved_backend,
        target_arch,
    )


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model.is_file():
        raise GateError(f"model does not exist: {args.model}")
    if int(args.decode_steps) <= 0 or int(args.repeat_runs) < 3:
        raise GateError("decode steps must be positive and repeat runs must be at least three")
    candidate = CANDIDATES[str(args.candidate)]
    prompt_rows = _load_suites(args.prompts)
    if args.limit is not None:
        prompt_rows = prompt_rows[: max(0, int(args.limit))]
    if not prompt_rows:
        raise GateError("selected prompt suites are empty")
    complete_suite = args.limit is None
    _configure_gate_environment(decode_repack=bool(args.decode_repack))
    (
        captures,
        prompt_manifest,
        strict_states,
        candidate_states,
        resolved_backend,
        target_arch,
    ) = _capture(args, prompt_rows=prompt_rows, candidate=candidate)
    thresholds = EvaluationThresholds()
    evaluated = build_candidate_quality(
        captures,
        candidate_mode=candidate.name,
        scenario_id=f"gguf-{args.model.stem}-c1-{candidate.name}",
        thresholds=thresholds,
    )
    state_gate = build_state_repeat_gate(strict_states, candidate_states)
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
            "HIPENGINE_GGUF_DECODE_REPACK": os.environ.get("HIPENGINE_GGUF_DECODE_REPACK"),
            "strict_route_environment": dict(STRICT_ENVIRONMENT),
            "candidate_route_environment": dict(candidate.environment),
        },
        build_profile="execution_profile_gguf_c1_route_gate",
        timing_protocol="none_full_logits_and_state_only_v1",
        warmups=0,
        repetitions=int(args.repeat_runs),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    measurement_valid = bool(
        not provenance.get("dirty")
        and complete_suite
        and evaluated["repeat_determinism"]["passed"]
        and state_gate["passed"]
    )
    gate_passed = bool(measurement_valid and evaluated["quality"]["hard_gates_passed"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "passed" if gate_passed else "failed_or_screen_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "measurement_valid": measurement_valid,
        "performance_claim": False,
        "candidate": {
            "name": candidate.name,
            "classification": candidate.classification,
            "mechanism": candidate.mechanism,
            "strict_fallback": candidate.strict_fallback,
            "strict_environment": dict(STRICT_ENVIRONMENT),
            "candidate_environment": dict(candidate.environment),
        },
        "protocol": {
            "model": str(args.model.resolve()),
            "prompt_suites": [str(path.resolve()) for path in args.prompts],
            "complete_prompt_and_heldout_suite": complete_suite,
            "prompt_count": len(prompt_rows),
            "baseline_gdn_mode": str(args.baseline_gdn_mode),
            "decode_steps": int(args.decode_steps),
            "teacher_forced_rows": sum(len(capture.strict) for capture in captures),
            "candidate_repeat_runs": int(args.repeat_runs),
            "thresholds": thresholds.to_dict(),
        },
        "prompts": prompt_manifest,
        "quality": evaluated,
        "state_repeat_gate": state_gate,
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--prompts", action="append", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--baseline-gdn-mode", default="chain_lds32_direct_nonvolatile")
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--bulk-attention-mode", choices=("bulk", "native"), default="bulk")
    parser.add_argument("--decode-repack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-wmma-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gemv-decode", action=argparse.BooleanOptionalAction, default=True)
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
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.json)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "measurement_valid": artifact["measurement_valid"],
                "hard_gates_passed": artifact["quality"]["quality"]["hard_gates_passed"],
                "repeat_deterministic": artifact["quality"]["repeat_determinism"]["passed"],
                "state_repeat_passed": artifact["state_repeat_gate"]["passed"],
                "summary": artifact["quality"]["quality"]["summary"],
            },
            indent=2,
        )
    )
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
