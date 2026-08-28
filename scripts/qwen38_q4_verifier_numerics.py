#!/usr/bin/env python3
"""Evaluate the scoped Qwen3.8 C2/K3 production Q4 verifier rowtiles.

The strict side uses FP32 recurrent state and the strict small-M/shared-B Q4
WMMA target owners. The candidate uses FP16 recurrent state and the profile-
selected Q4 rowtiles only inside an R8 packed target verifier. Both sides replay
identical strict-teacher token blocks and emit full-vocabulary logits.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.benchmark.execution_profiles import EvaluationThresholds
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    host_array_ptr,
    memory_stats,
)
from hipengine.runtime.gguf_linear import (
    TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV,
)
from scripts.execution_profile_gguf_batch_route_gate import (
    BatchRouteCapture,
    build_batch_route_quality,
)
from scripts.execution_profile_gguf_fp16_state_batch_gate import _reset_sessions
from scripts.execution_profile_gguf_fp16_state_gate import (
    FP16_STATE_ENV,
    fp16_state_environment,
    production_route_environment,
)
from scripts.gguf_gdn_semantic_gate import _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import prompt_sha256

KIND = "qwen38_c2k3_production_q4_verifier_numerics"
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_PROMPTS = ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
VERIFY_CAPTURE_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"
ROWS_PER_JOB = 4
PHYSICAL_ROWS = 8


class GateError(RuntimeError):
    """Raised when verifier numerical evidence is not comparable."""


@contextmanager
def _environment(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _make_sessions(
    args: argparse.Namespace,
    *,
    fp16_state: bool,
    q4_rowtile: bool,
    max_sequence_length: int,
) -> tuple[ExitStack, tuple[Any, Any]]:
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    compiler_version = (
        None
        if args.compiler_version_file is None
        else args.compiler_version_file.read_text(encoding="utf-8")
    )
    stack = ExitStack()
    with (
        fp16_state_environment(fp16_state),
        _environment(
            TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV,
            "1" if q4_rowtile else "0",
        ),
    ):
        owner = stack.enter_context(
            Qwen35GGUFResidentSession(
                args.model,
                backend=str(args.backend),
                compiler_version=compiler_version,
                require_cached_build=bool(args.require_cached_build),
                max_sequence_length=max_sequence_length,
                use_wmma_prefill=True,
                use_gemv_decode=True,
                prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
            )
        )
        peer = stack.enter_context(
            Qwen35GGUFResidentSession(
                args.model,
                backend=str(args.backend),
                runtime=owner.runtime,
                shared_runner=owner.runner,
                compiler_version=compiler_version,
                require_cached_build=bool(args.require_cached_build),
                max_sequence_length=max_sequence_length,
                use_wmma_prefill=True,
                use_gemv_decode=True,
                prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
            )
        )
    sessions = (owner, peer)
    if any(
        session.runner is None
        or bool(session.runner.fp16_recurrent_state) is not bool(fp16_state)
        for session in sessions
    ):
        stack.close()
        raise GateError("resident session did not freeze the requested state dtype")
    if any(
        bool(session.target_verifier_production_q4_rowtile)
        is not bool(q4_rowtile)
        for session in sessions
    ):
        stack.close()
        raise GateError("resident session did not freeze the requested Q4 route")
    return stack, sessions


def _prefill(
    sessions: Sequence[Any],
    token_rows: Sequence[Sequence[int]],
) -> Sequence[Any]:
    _reset_sessions(sessions)
    return sessions[0].prefill_batch_native(
        token_rows,
        sessions=sessions,
        return_logits=True,
        require_logits=True,
    )


def _strict_teacher_tokens(
    sessions: Sequence[Any],
    token_rows: Sequence[Sequence[int]],
    decode_steps: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    results = _prefill(sessions, token_rows)
    sequences = [[int(result.token_id)] for result in results]
    for _ in range(int(decode_steps)):
        results = sessions[0].step_batch_native(
            [sequence[-1] for sequence in sequences],
            sessions=sessions,
            return_logits=True,
            require_logits=True,
            scatter_state=True,
        )
        for index, result in enumerate(results):
            sequences[index].append(int(result.token_id))
    return tuple(tuple(sequence) for sequence in sequences)  # type: ignore[return-value]


def _read_packed_logits(owner: Any, rows: int) -> np.ndarray:
    logits_buffer = owner._verify_logits_buf
    if logits_buffer is None or owner.runner is None:
        raise GateError("packed verifier did not materialize full logits")
    result = np.empty((int(rows), int(owner.runner.vocab_size)), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(result),
        DeviceBuffer(logits_buffer.ptr, result.nbytes),
        result.nbytes,
        runtime=owner.runtime,
    )
    return result


def _capture_verifier_pair(
    sessions: Sequence[Any],
    token_rows: Sequence[Sequence[int]],
    teacher_tokens: Sequence[Sequence[int]],
    decode_steps: int,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    _prefill(sessions, token_rows)
    trajectories: list[list[dict[str, object]]] = [[], []]
    for step in range(0, int(decode_steps), ROWS_PER_JOB):
        jobs = [
            {
                "session": sessions[index],
                "input_token_ids": tuple(
                    int(token)
                    for token in teacher_tokens[index][
                        step : step + ROWS_PER_JOB
                    ]
                ),
                "bulk_attention_mode": "bulk",
                "use_wmma_prefill": True,
            }
            for index in range(2)
        ]
        positions_before = [int(session.position) for session in sessions]
        results = sessions[0].verify_target_blocks_batch(jobs)
        logits = _read_packed_logits(sessions[0], PHYSICAL_ROWS)
        for request_index in range(2):
            result = results[request_index]
            expected_inputs = tuple(jobs[request_index]["input_token_ids"])
            if (
                int(result.start_position) != positions_before[request_index]
                or tuple(result.input_token_ids) != expected_inputs
                or int(sessions[request_index].position)
                != positions_before[request_index] + ROWS_PER_JOB
            ):
                raise GateError("packed verifier request/position control drifted")
            row_base = request_index * ROWS_PER_JOB
            for row in range(ROWS_PER_JOB):
                trajectories[request_index].append(
                    {
                        "token_id": int(result.token_ids[row]),
                        "logits": np.ascontiguousarray(
                            logits[row_base + row], dtype=np.float32
                        ),
                    }
                )
    return tuple(tuple(rows) for rows in trajectories)  # type: ignore[return-value]


def _trajectory_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(np.asarray(int(row["token_id"]), dtype="<i8").tobytes())
        digest.update(
            np.ascontiguousarray(row["logits"], dtype="<f4").tobytes()
        )
    return digest.hexdigest()


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, object]:
    if not args.model.is_file():
        raise GateError(f"model does not exist: {args.model}")
    if int(args.decode_steps) <= 0 or int(args.decode_steps) % ROWS_PER_JOB:
        raise GateError("decode steps must be a positive multiple of four")
    if int(args.repeat_runs) < 3:
        raise GateError("candidate gate requires at least three repeats")

    prompt_rows = _load_suites((args.prompts,))
    if args.limit is not None:
        prompt_rows = prompt_rows[: max(0, int(args.limit))]
    if not prompt_rows or len(prompt_rows) % 2:
        raise GateError("selected prompt count must be positive and even")

    from hipengine.loading.gguf import scan_gguf
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    prompt_tokens = {
        str(row["id"]): tuple(
            int(token)
            for token in build_chat_prompt(tokenizer, str(row["prompt"]))
        )
        for row in prompt_rows
    }
    max_sequence_length = (
        max(len(tokens) for tokens in prompt_tokens.values())
        + int(args.decode_steps)
        + ROWS_PER_JOB
        + 4
    )
    memory_before = memory_stats()

    strict: dict[str, tuple[dict[str, object], ...]] = {}
    teachers: dict[str, tuple[int, ...]] = {}
    stack, sessions = _make_sessions(
        args,
        fp16_state=False,
        q4_rowtile=False,
        max_sequence_length=max_sequence_length,
    )
    try:
        for pair_start in range(0, len(prompt_rows), 2):
            pair_rows = prompt_rows[pair_start : pair_start + 2]
            token_rows = tuple(
                prompt_tokens[str(row["id"])] for row in pair_rows
            )
            teacher_pair = _strict_teacher_tokens(
                sessions, token_rows, int(args.decode_steps)
            )
            capture = _capture_verifier_pair(
                sessions, token_rows, teacher_pair, int(args.decode_steps)
            )
            for index, row in enumerate(pair_rows):
                request_id = str(row["id"])
                strict[request_id] = capture[index]
                teachers[request_id] = teacher_pair[index]
            print(
                f"strict {pair_start // 2 + 1}/{len(prompt_rows) // 2}",
                flush=True,
            )
    finally:
        stack.close()

    candidate: dict[str, list[tuple[dict[str, object], ...]]] = {
        str(row["id"]): [] for row in prompt_rows
    }
    stack, sessions = _make_sessions(
        args,
        fp16_state=True,
        q4_rowtile=True,
        max_sequence_length=max_sequence_length,
    )
    try:
        for repeat in range(int(args.repeat_runs)):
            for pair_start in range(0, len(prompt_rows), 2):
                pair_rows = prompt_rows[pair_start : pair_start + 2]
                token_rows = tuple(
                    prompt_tokens[str(row["id"])] for row in pair_rows
                )
                teacher_pair = tuple(
                    teachers[str(row["id"])] for row in pair_rows
                )
                capture = _capture_verifier_pair(
                    sessions,
                    token_rows,
                    teacher_pair,
                    int(args.decode_steps),
                )
                for index, row in enumerate(pair_rows):
                    candidate[str(row["id"])].append(capture[index])
            print(
                f"candidate {repeat + 1}/{int(args.repeat_runs)}",
                flush=True,
            )
    finally:
        stack.close()

    captures = tuple(
        BatchRouteCapture(
            scenario_id="c2_k3_r8",
            request_id=str(row["id"]),
            category=str(row["category"]),
            strict=strict[str(row["id"])],
            candidate_runs=tuple(candidate[str(row["id"])]),
            shapes=("c2_k3_r8",) * int(args.decode_steps),
            transitions=("prefill_to_c2_k3_r8",)
            + ("steady",) * (int(args.decode_steps) - 1),
            teacher_steps=tuple(range(int(args.decode_steps))),
        )
        for row in prompt_rows
    )
    quality = build_batch_route_quality(
        captures,
        thresholds=EvaluationThresholds(),
    )
    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=str(args.backend),
        resolved_backend=str(args.backend),
        target_arch="gfx1151",
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "strict": {
                FP16_STATE_ENV: "0",
                TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV: "0",
            },
            "candidate": {
                FP16_STATE_ENV: "1",
                TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV: "1",
            },
        },
        build_profile="qwen38_q4_verifier_numerics",
        timing_protocol="none_full_logits_teacher_forced_c2_k3_r8_v1",
        warmups=0,
        repetitions=int(args.repeat_runs),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    tracked_clean = not bool(
        provenance.get("staged_dirty") or provenance.get("unstaged_dirty")
    )
    hard_passed = bool(
        quality["quality"]["hard_gates_passed"]
        and quality["repeat_determinism"]["passed"]
    )
    measurement_valid = bool(
        tracked_clean
        and int(args.decode_steps) == 24
        and int(args.repeat_runs) >= 3
        and len(prompt_rows) >= 8
    )
    memory_after = memory_stats()
    return {
        "schema": 1,
        "kind": KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "passed"
            if measurement_valid and hard_passed
            else "failed_or_screen_only"
        ),
        "measurement_valid": measurement_valid,
        "hard_gates_passed": hard_passed,
        "performance_claim": False,
        "candidate": {
            "arithmetic_class": "T1+T2",
            "strict": "FP32 state + strict small-M/shared-B Q4 WMMA",
            "production": "FP16 state + scoped R8 Q4 singleton/pair rowtiles",
            "strict_fallbacks": [
                "linear/gguf_q4_k_t16_v1/t16_wmma_prefill_smallm_bf16_bf16_out",
                "linear_pair_silu/gguf_q4_k_t16_v1/dense_dual_wmma_prefill_bf16_bf16_out",
            ],
            "excluded": ["rows!=8", "narrow K5120/N1024", "peer backends"],
        },
        "protocol": {
            "model": str(args.model.resolve()),
            "prompts": str(args.prompts.resolve()),
            "prompt_count": len(prompt_rows),
            "decode_steps": int(args.decode_steps),
            "teacher_rows": len(prompt_rows) * int(args.decode_steps),
            "repeat_runs": int(args.repeat_runs),
            "physical_shape": "C2/K3 R8",
            "teacher_source": "strict fixed-schedule trajectory",
        },
        "quality": quality,
        "control": {
            "request_positions_exact": True,
            "input_tokens_exact": True,
            "same_width_pairing": True,
        },
        "prompts": [
            {
                "id": str(row["id"]),
                "category": str(row["category"]),
                "heldout": bool(row.get("heldout", False)),
                "prompt_sha256": prompt_sha256(str(row["prompt"])),
            }
            for row in prompt_rows
        ],
        "trajectory_sha256": {
            str(row["id"]): {
                "strict": _trajectory_sha256(strict[str(row["id"])]),
                "candidate": [
                    _trajectory_sha256(run)
                    for run in candidate[str(row["id"])]
                ],
            }
            for row in prompt_rows
        },
        "memory": {
            "before": memory_before,
            "after": memory_after,
            "teardown_exact": (
                int(memory_before["current_allocated_bytes"])
                == int(memory_after["current_allocated_bytes"])
                and int(memory_after["active_allocations"]) == 0
            ),
        },
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    command = [sys.executable, str(Path(__file__).relative_to(ROOT)), *raw_argv]
    try:
        with (
            production_route_environment(),
            _environment(VERIFY_CAPTURE_ENV, "1"),
        ):
            payload = run(args, command=command)
    except (GateError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "measurement_valid": payload["measurement_valid"],
                "hard_gates_passed": payload["hard_gates_passed"],
                "summary": payload["quality"]["quality"]["summary"],
            },
            indent=2,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
