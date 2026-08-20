#!/usr/bin/env python3
"""Gate the gfx1151 c1 short-batch thread-geometry override (threads=1024)
against the retained exact 256-thread leaf using the calibrated
execution-profile envelope.

The 1024-thread leaf runs the same fixed256 body at a wider block width; it is
a T2 association/layout drift (split value reduction + different warp reduction
tree) and is therefore NOT byte-exact. This gate teacher-forces the candidate
onto the strict (256-thread) token trajectory across the full multi-prompt
category suite, evaluates mean/p95/p99/max KL and top-1 with the calibrated
envelope, and requires three bit-stable candidate repeats plus layout-stable
state. The env var HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS switches the leaf; the
runner caches the resolved leaf, so the cache is reset on every leg boundary.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
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
from scripts.execution_profile_gguf_c1_route_gate import (
    _state_summary,
    build_state_repeat_gate,
)
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

KIND = "hipengine_execution_profile_gguf_c1_threads_gate"
SCHEMA_VERSION = 1
_ROUTE_ENV_KEYS = ("HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS",)
STRICT_ENVIRONMENT = {_ROUTE_ENV_KEYS[0]: "256"}
CANDIDATE_THREADS = 1024
CANDIDATE_ENVIRONMENT = {_ROUTE_ENV_KEYS[0]: str(CANDIDATE_THREADS)}
_CACHE_KEY = "_gguf_full_attn_decode_short_batch_fn_cache"


class GateError(RuntimeError):
    """Raised when the threads candidate packet cannot be evaluated honestly."""


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    classification: str
    mechanism: str
    strict_fallback: str
    environment: Mapping[str, str]


CANDIDATE = Candidate(
    name=f"short_c1_attn_{CANDIDATE_THREADS}_threads",
    classification="T2",
    mechanism=(
        "run the fixed256 context-batch body at threads=1024 (split value "
        "reduction, wider warp reduction tree); not byte-exact vs 256-thread"
    ),
    strict_fallback="retained exact 256-thread fixed256 leaf (c1_exact_spans)",
    environment=CANDIDATE_ENVIRONMENT,
)


@contextmanager
def route_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Apply the threads route environment and restore exactly."""

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


def _reset_leaf_cache(session: Any) -> None:
    runner = session.runner
    if runner is not None:
        runner.__dict__.pop(_CACHE_KEY, None)


def _capture(
    args: argparse.Namespace,
    *,
    prompt_rows: Sequence[Mapping[str, Any]],
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
    max_sequence_length = (
        max(len(tokens) for tokens in prompt_tokens.values()) + int(args.decode_steps) + 2
    )
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
                _reset_leaf_cache(session)
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
                with route_environment(CANDIDATE.environment):
                    _reset_leaf_cache(session)
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
                    candidate_runs={CANDIDATE.name: tuple(runs)},
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
    ) = _capture(args, prompt_rows=prompt_rows)
    thresholds = EvaluationThresholds()
    evaluated = build_candidate_quality(
        captures,
        candidate_mode=CANDIDATE.name,
        scenario_id=f"gguf-{args.model.stem}-c1-{CANDIDATE.name}",
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
            "candidate_route_environment": dict(CANDIDATE.environment),
        },
        build_profile="execution_profile_gguf_c1_threads_gate",
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
    quality_passed = bool(
        evaluated["quality"]["hard_gates_passed"]
        and evaluated["quality"]["eligible_for_automatic_admission"]
        and not evaluated["quality"].get("requires_outlier_review", False)
    )
    passed = bool(measurement_valid and quality_passed)
    decision = "accepted" if passed else "blocked"
    status = {
        "passed": passed,
        "decision": decision,
        "reason": None if passed else "one or more binding gates failed",
    }
    report = {
        "id": f"2026-08-20-gfx1151-qwen36-35b-c1-{CANDIDATE.name}",
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "measurement_host": provenance.get("measurement_host"),
        "hardware": provenance.get("hardware"),
        "model": str(args.model),
        "backend": str(args.backend),
        "candidate": asdict(CANDIDATE),
        "strict_environment": dict(STRICT_ENVIRONMENT),
        "candidate_environment": dict(CANDIDATE.environment),
        "prompts": prompt_manifest,
        "evaluated": evaluated,
        "state_gate": state_gate,
        "provenance": provenance,
        "status": status,
    }
    print(json.dumps({"status": status, "evaluated": evaluated}, indent=2, default=str))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--prompts", type=Path, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--baseline-gdn-mode", default="chain_lds32_direct_nonvolatile")
    parser.add_argument("--bulk-attention-mode", choices=("bulk", "native"), default="bulk")
    parser.add_argument("--decode-repack", action="store_true")
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=0)
    parser.add_argument("--use-wmma-prefill", action="store_true")
    parser.add_argument("--use-gemv-decode", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prompts is None:
        args.prompts = list(DEFAULT_PROMPTS)
    report = run(args, command=sys.argv)
    if args.output is not None:
        args.output.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
    return 0 if report["status"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
