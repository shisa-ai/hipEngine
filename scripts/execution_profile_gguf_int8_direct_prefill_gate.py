#!/usr/bin/env python3
"""Gate the oracle-free direct INT8 prefill attention against the strict route.

The pure-INT8 route's strict prefill arithmetic is the oracle-bridge path
(``HIPENGINE_GGUF_INT8_PREFILL_DIRECT`` unset): full-attention prefill
reads the write-through BF16 oracle pair via AOTriton while the retained
INT8 store is written. The candidate (default ``twopass`` kernel) reads
the retained INT8 store directly through the two-pass tiled kernel, so
this gate isolates exactly the INT8-read prefill arithmetic drift.

The adapter keeps strict and candidate in one resident session with the
pure-INT8 KV policy, teacher-forces every candidate onto the strict token
trajectory, evaluates all full-vocabulary rows with the calibrated
execution-profile envelope, and requires three bit-stable candidate
repeats plus state-fingerprint stability. The retained INT8 KV bytes are
expected to remain identical between arms (write-through quantization is
unchanged); the linear/GDN states legitimately differ because full-attention
layer outputs feed them. Performance is measured separately so full-logit
D2H traffic cannot contaminate wall claims.
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

KIND = "hipengine_execution_profile_gguf_int8_direct_prefill_gate"
SCHEMA_VERSION = 1

PURE_INT8_ENV = {
    "HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG": "1",
    "HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS": "none",
}
DIRECT_ENV = "HIPENGINE_GGUF_INT8_PREFILL_DIRECT"
KERNEL_ENV = "HIPENGINE_GGUF_INT8_PREFILL_KERNEL"
ROUTE_ENV_KEYS = (*PURE_INT8_ENV, DIRECT_ENV, KERNEL_ENV)


class GateError(RuntimeError):
    """Raised when the candidate packet cannot be evaluated honestly."""


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    classification: str
    mechanism: str
    strict_fallback: str
    environment: Mapping[str, str]


def _candidate(kernel: str) -> Candidate:
    base = dict(PURE_INT8_ENV)
    base[DIRECT_ENV] = "1"
    base[KERNEL_ENV] = kernel
    kernel_desc = (
        "the GQA-grouped flash kernel (LDS tile staging, online softmax in registers)"
        if kernel == "flash"
        else "the sequential online-softmax kernel"
    )
    return Candidate(
        name=f"int8_direct_prefill_{kernel}",
        classification="T2",
        mechanism=(
            "prefill full-attention reads the retained INT8 K/V store directly "
            f"with per-token/head scales through {kernel_desc}, removing the "
            "BF16 oracle pair and the AOTriton bridge"
        ),
        strict_fallback="pure-INT8 oracle-bridge prefill (BF16 oracle + AOTriton)",
        environment=base,
    )


@contextmanager
def route_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Apply a complete route environment and restore the caller exactly."""

    unknown = sorted(set(values) - set(ROUTE_ENV_KEYS))
    if unknown:
        raise ValueError(f"route environment has unknown keys: {unknown}")
    previous = {key: os.environ.get(key) for key in ROUTE_ENV_KEYS}
    try:
        for key in ROUTE_ENV_KEYS:
            if key in values:
                os.environ[key] = str(values[key])
            else:
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _rearm_bulk_workspace(session: Any) -> None:
    """Release the bulk prefill workspace so the next arm re-plans its env."""

    release = getattr(session, "_release_bulk_prefill_workspace", None)
    if release is None:
        raise GateError("resident session lacks the bulk prefill workspace release")
    release()


def _state_summary(session: Any, trajectory: Sequence[Mapping[str, Any]], forced_input_ids: Sequence[int]) -> dict[str, Any]:
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
    from hipengine.kvcache import resolve_kv_policy
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
    policy = resolve_kv_policy(
        "int8_per_token_head",
        scale_dtype="fp32",
        scale_granularity="per_token_head",
    )
    with Qwen35GGUFResidentSession(
        args.model,
        backend=str(args.backend),
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        prefill_config=PrefillConfig(
            attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens)
        ),
        kv_policy=policy.create_policy(),
        kv_scale_dtype="fp32",
        kv_scale_granularity="per_token_head",
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
            with route_environment(PURE_INT8_ENV):
                _rearm_bulk_workspace(session)
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
                    _rearm_bulk_workspace(session)
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
    candidate = _candidate(str(args.kernel))
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
        scenario_id=f"gguf-{args.model.stem}-int8-direct-prefill-{args.kernel}",
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
        kv_dtype="int8_per_token_head_fp32_scale",
        command=command,
        environment={
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "HIPENGINE_GGUF_DECODE_REPACK": os.environ.get("HIPENGINE_GGUF_DECODE_REPACK"),
            "strict_route_environment": dict(PURE_INT8_ENV),
            "candidate_route_environment": dict(candidate.environment),
        },
        build_profile="execution_profile_gguf_int8_direct_prefill_gate",
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
            "strict_environment": dict(PURE_INT8_ENV),
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
            "kv_policy": "int8_per_token_head fp32 per-token-head scales",
        },
        "prompts": prompt_manifest,
        "quality": evaluated,
        "state_repeat_gate": state_gate,
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--kernel", choices=("flash", "sequential"), default="flash")
    parser.add_argument("--prompts", action="append", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--baseline-gdn-mode", default="chain_lds32_direct_nonvolatile")
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--decode-repack", action="store_true")
    parser.add_argument(
        "--bulk-attention-mode",
        default="bulk",
        choices=("bulk", "native"),
    )
    parser.add_argument(
        "--attn-aotriton-min-tokens",
        type=int,
        default=512,
    )
    parser.add_argument("--use-wmma-prefill", action="store_true")
    parser.add_argument("--use-gemv-decode", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prompt_paths = list(args.prompts) if args.prompts else [Path(DEFAULT_PROMPTS)]
    args.prompts = prompt_paths
    command = ["python3", str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else list(argv))]
    artifact = run(args, command=command)
    payload = json.dumps(artifact, indent=2, ensure_ascii=False)
    print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
