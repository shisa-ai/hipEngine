#!/usr/bin/env python3
"""Gate the fp16 recurrent-state route against the fp32 production route.

The fp16-state route (``HIPENGINE_GGUF_FP16_RECURRENT_STATE``) stores the GDN
recurrent state as fp16 (fp32 accumulate, RNE half round-trip) instead of fp32,
halving recurrent-state traffic and buffer size (see docs/REFACTOR.md).  The
strict denominator for this gate is the fp32-state version of the *same*
production route (``chain_compact_peer_wave32``) -- the fp16 flag changes only
the state storage dtype, not the route or the surrounding arithmetic.

Because the flag changes scratch buffer sizing at session construction, strict
(fp32 state) and candidate (fp16 state) each run in their own resident session
with the flag fixed for the whole session (as in real deployment), rather than
toggled mid-session.  Every candidate run is teacher-forced onto the strict
token trajectory, all full-vocabulary rows are evaluated with the calibrated
execution-profile envelope, and three bit-stable candidate repeats plus
finite/layout-stable/byte-repeatable state fingerprints are required.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
)
from scripts.execution_profile_gguf_c1_route_gate import (
    build_state_repeat_gate,
    _state_summary,
)
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import prompt_sha256

KIND = "hipengine_execution_profile_gguf_fp16_state_gate"
SCHEMA_VERSION = 1
FP16_STATE_ENV = "HIPENGINE_GGUF_FP16_RECURRENT_STATE"
GDN_MODE_ENV = "HIPENGINE_GGUF_GDN_PREFILL_MODE"
PRODUCTION_ROUTE = "chain_compact_peer_wave32"
CANDIDATE_NAME = "fp16_state"
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf")


class GateError(RuntimeError):
    """Raised when the fp16-state gate cannot be evaluated safely."""


@contextmanager
def fp16_state_environment(enabled: bool) -> Iterator[None]:
    """Apply/remove the fp16 recurrent-state flag, restoring the caller exactly."""

    previous = os.environ.get(FP16_STATE_ENV)
    if enabled:
        os.environ[FP16_STATE_ENV] = "1"
    else:
        os.environ.pop(FP16_STATE_ENV, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(FP16_STATE_ENV, None)
        else:
            os.environ[FP16_STATE_ENV] = previous


@contextmanager
def production_route_environment() -> Iterator[None]:
    """Pin the production route (compact-peer wave32) for both phases."""

    previous = os.environ.get(GDN_MODE_ENV)
    os.environ[GDN_MODE_ENV] = PRODUCTION_ROUTE
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(GDN_MODE_ENV, None)
        else:
            os.environ[GDN_MODE_ENV] = previous


def _run_logits_trajectory(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    decode_steps: int,
    bulk_attention_mode: str,
) -> list[dict[str, Any]]:
    session.reset()
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
    current = int(result.token_id)
    for _ in range(int(decode_steps)):
        result = session.step(current, return_logits=True)
        current = int(result.token_id)
        trajectory.append(
            {
                "token_id": current,
                "logits": np.ascontiguousarray(result.logits, dtype=np.float32),
            }
        )
    return trajectory


def _run_teacher_forced_candidate(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    forced_input_ids: Sequence[int],
    bulk_attention_mode: str,
) -> list[dict[str, Any]]:
    session.reset()
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


def _make_session(args: argparse.Namespace, *, fp16: bool):
    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    compiler_version = (
        None
        if args.compiler_version_file is None
        else args.compiler_version_file.read_text(encoding="utf-8")
    )
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    with fp16_state_environment(fp16):
        session = Qwen35GGUFResidentSession(
            args.model,
            backend=str(args.backend),
            compiler_version=compiler_version,
            require_cached_build=bool(args.require_cached_build),
            max_sequence_length=args.max_sequence_length,
            use_wmma_prefill=bool(args.use_wmma_prefill),
            use_gemv_decode=bool(args.use_gemv_decode),
            prefill_config=PrefillConfig(
                attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens)
            ),
        )
    return session, tokenizer


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
    strict_trajectories: dict[str, tuple[Mapping[str, object], ...]] = {}
    strict_states: dict[str, dict[str, Any]] = {}
    candidate_runs_by_prompt: dict[str, list[tuple[Mapping[str, object], ...]]] = {}
    candidate_states: dict[str, list[dict[str, Any]]] = {}
    prompt_tokens: dict[str, list[int]] = {}
    prompt_manifest: list[dict[str, Any]] = []
    resolved_backend = ""
    target_arch = ""

    tokenizer = None
    strict_session, tokenizer = _make_session(args, fp16=False)
    try:
        if strict_session.runner is None:
            raise GateError("strict GGUF resident session closed during setup")
        resolved_backend = str(strict_session.runner.backend)
        target_arch = str(strict_session.runner.target_arch)
        for row in prompt_rows:
            prompt_id = str(row["id"])
            tokens = build_chat_prompt(tokenizer, str(row["prompt"]))
            prompt_tokens[prompt_id] = [int(token) for token in tokens]
            with production_route_environment(), fp16_state_environment(False):
                strict = tuple(
                    _run_logits_trajectory(
                        strict_session,
                        prompt_ids=prompt_tokens[prompt_id],
                        decode_steps=int(args.decode_steps),
                        bulk_attention_mode=str(args.bulk_attention_mode),
                    )
                )
                forced = [int(step["token_id"]) for step in strict[:-1]]
                strict_state = _state_summary(strict_session, strict, forced)
            strict_state["prompt_id"] = prompt_id
            strict_trajectories[prompt_id] = strict
            strict_states[prompt_id] = strict_state
            prompt_manifest.append(
                {
                    "id": prompt_id,
                    "category": str(row["category"]),
                    "suite": str(row["suite"]),
                    "prompt_sha256": prompt_sha256(str(row["prompt"])),
                    "prompt_tokens": len(prompt_tokens[prompt_id]),
                    "prompt_token_ids_sha256": hashlib.sha256(
                        np.asarray(prompt_tokens[prompt_id], dtype="<i8").tobytes()
                    ).hexdigest(),
                }
            )
    finally:
        strict_session.close()

    candidate_session, _ = _make_session(args, fp16=True)
    try:
        if candidate_session.runner is None:
            raise GateError("candidate GGUF resident session closed during setup")
        for index, row in enumerate(prompt_rows):
            prompt_id = str(row["id"])
            tokens = prompt_tokens[prompt_id]
            forced = [int(step["token_id"]) for step in strict_trajectories[prompt_id][:-1]]
            runs: list[tuple[Mapping[str, object], ...]] = []
            state_runs: list[dict[str, Any]] = []
            for _ in range(int(args.repeat_runs)):
                with production_route_environment(), fp16_state_environment(True):
                    run = tuple(
                        _run_teacher_forced_candidate(
                            candidate_session,
                            prompt_ids=tokens,
                            forced_input_ids=forced,
                            bulk_attention_mode=str(args.bulk_attention_mode),
                        )
                    )
                    state = _state_summary(candidate_session, run, forced)
                state["prompt_id"] = prompt_id
                runs.append(run)
                state_runs.append(state)
            candidate_runs_by_prompt[prompt_id] = runs
            candidate_states[prompt_id] = state_runs
            print(
                f"{index + 1}/{len(prompt_rows)} {prompt_id}: candidate "
                f"{args.repeat_runs} trajectories/states",
                flush=True,
            )
    finally:
        candidate_session.close()

    captures = [
        PromptCalibrationCapture(
            prompt_id=str(row["id"]),
            category=str(row["category"]),
            strict=strict_trajectories[str(row["id"])],
            candidate_runs={CANDIDATE_NAME: tuple(candidate_runs_by_prompt[str(row["id"])])},
        )
        for row in prompt_rows
    ]
    strict_by_prompt = [strict_states[str(row["id"])] for row in prompt_rows]
    candidate_by_prompt = [candidate_states[str(row["id"])] for row in prompt_rows]
    return (
        tuple(captures),
        prompt_manifest,
        strict_by_prompt,
        candidate_by_prompt,
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
        candidate_mode=CANDIDATE_NAME,
        scenario_id=f"gguf-{args.model.stem}-{CANDIDATE_NAME}",
        thresholds=thresholds,
    )
    state_gate = build_state_repeat_gate(strict_states, candidate_states)
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=args.model,
        quant="gguf_q4_k_s",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "HIPENGINE_GGUF_DECODE_REPACK": os.environ.get("HIPENGINE_GGUF_DECODE_REPACK"),
            "strict_route_environment": {FP16_STATE_ENV: "0"},
            "candidate_route_environment": {FP16_STATE_ENV: "1"},
        },
        build_profile="execution_profile_gguf_fp16_state_gate",
        timing_protocol="none_full_logits_and_state_only_v1",
        warmups=0,
        repetitions=int(args.repeat_runs),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    # The shared worktree carries unrelated untracked benchmark artifacts from
    # other agents; they do not alter the tested code.  The code under test is
    # exactly the committed state, so measurement validity requires tracked
    # cleanliness only and records the untracked count as an explicit caveat.
    tracked_clean = not bool(
        provenance.get("staged_dirty") or provenance.get("unstaged_dirty")
    )
    worktree_note = {
        "tracked_clean": bool(tracked_clean),
        "untracked_dirty": bool(provenance.get("untracked_dirty")),
        "untracked_count": int(provenance.get("untracked_count", 0)),
        "note": (
            "untracked benchmark artifacts are unrelated pre-existing outputs; "
            "tracked code is at the committed state"
        ),
    }
    measurement_valid = bool(
        tracked_clean
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
        "worktree_note": worktree_note,
        "performance_claim": False,
        "candidate": {
            "name": CANDIDATE_NAME,
            "classification": "T2",
            "mechanism": (
                "fp16 recurrent-state storage (fp32 accumulate, RNE half "
                "round-trip) on the chain_compact_peer_wave32 production route"
            ),
            "strict_fallback": "fp32 recurrent-state storage, same production route",
            "strict_environment": {FP16_STATE_ENV: "0"},
            "candidate_environment": {FP16_STATE_ENV: "1"},
        },
        "protocol": {
            "model": str(args.model.resolve()),
            "prompt_suites": [str(path.resolve()) for path in args.prompts],
            "complete_prompt_and_heldout_suite": complete_suite,
            "prompt_count": len(prompt_rows),
            "production_route": PRODUCTION_ROUTE,
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
    parser.add_argument("--prompts", action="append", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--bulk-attention-mode", choices=("bulk", "native"), default="bulk")
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--decode-repack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-wmma-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gemv-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=512)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.prompts is None:
        args.prompts = list(DEFAULT_PROMPTS)
    if args.decode_repack:
        os.environ.setdefault("HIPENGINE_GGUF_DECODE_REPACK", "1")
    command = [str(Path(sys.argv[0]).name), *sys.argv[1:]]
    payload = run(args, command=command)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
