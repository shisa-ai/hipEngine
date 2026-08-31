#!/usr/bin/env python3
"""Evaluate scoped Qwen3.8 C2/C3 K3 production Q4 verifier rowtiles.

The strict side uses FP32 recurrent state and the strict small-M/shared-B Q4
WMMA target owners. The candidate uses FP16 recurrent state and profile-selected
Q4 rowtiles inside an R8 or R12 packed target verifier. Both sides replay
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
    target_verifier_wide_q6_shared4_session,
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

KIND = "qwen38_production_q4_verifier_numerics"
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_PROMPTS = ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
VERIFY_CAPTURE_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"


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
) -> tuple[ExitStack, tuple[Any, ...]]:
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
        sessions_list = [owner]
        for _ in range(1, int(args.concurrency)):
            sessions_list.append(
                stack.enter_context(
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
            )
    sessions = tuple(sessions_list)
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
) -> tuple[tuple[int, ...], ...]:
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


def _capture_verifier_group(
    sessions: Sequence[Any],
    token_rows: Sequence[Sequence[int]],
    teacher_tokens: Sequence[Sequence[int]],
    decode_steps: int,
    *,
    rows_per_job: int,
    wide_q6_shared4: bool = False,
) -> tuple[tuple[dict[str, object], ...], ...]:
    _prefill(sessions, token_rows)
    trajectories: list[list[dict[str, object]]] = [[] for _ in sessions]
    for step in range(0, int(decode_steps), int(rows_per_job)):
        jobs = [
            {
                "session": sessions[index],
                "input_token_ids": tuple(
                    int(token)
                    for token in teacher_tokens[index][
                        step : step + int(rows_per_job)
                    ]
                ),
                "bulk_attention_mode": "bulk",
                "use_wmma_prefill": True,
            }
            for index in range(len(sessions))
        ]
        positions_before = [int(session.position) for session in sessions]
        with target_verifier_wide_q6_shared4_session(wide_q6_shared4):
            results = sessions[0].verify_target_blocks_batch(jobs)
        physical_rows = len(sessions) * int(rows_per_job)
        logits = _read_packed_logits(sessions[0], physical_rows)
        for request_index in range(len(sessions)):
            result = results[request_index]
            expected_inputs = tuple(jobs[request_index]["input_token_ids"])
            if (
                int(result.start_position) != positions_before[request_index]
                or tuple(result.input_token_ids) != expected_inputs
                or int(sessions[request_index].position)
                != positions_before[request_index] + int(rows_per_job)
            ):
                raise GateError("packed verifier request/position control drifted")
            row_base = request_index * int(rows_per_job)
            for row in range(int(rows_per_job)):
                trajectories[request_index].append(
                    {
                        "token_id": int(result.token_ids[row]),
                        "logits": np.ascontiguousarray(
                            logits[row_base + row], dtype=np.float32
                        ),
                    }
                )
    return tuple(tuple(rows) for rows in trajectories)


def _prompt_groups(
    prompt_rows: Sequence[Mapping[str, object]],
    concurrency: int,
) -> tuple[tuple[tuple[Mapping[str, object], ...], int], ...]:
    """Return fixed-width groups, padding only the final physical batch."""

    width = int(concurrency)
    if width <= 0:
        raise ValueError("concurrency must be positive")
    groups: list[tuple[tuple[Mapping[str, object], ...], int]] = []
    for start in range(0, len(prompt_rows), width):
        selected = list(prompt_rows[start : start + width])
        valid = len(selected)
        if not selected:
            continue
        selected.extend(selected[-1] for _ in range(width - valid))
        groups.append((tuple(selected), valid))
    return tuple(groups)


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
    rows_per_job = int(args.candidate_budget) + 1
    if int(args.decode_steps) <= 0 or int(args.decode_steps) % rows_per_job:
        raise GateError(
            "decode steps must be a positive multiple of candidate budget + one"
        )
    if int(args.repeat_runs) < 3:
        raise GateError("candidate gate requires at least three repeats")

    prompt_rows = _load_suites((args.prompts,))
    if args.limit is not None:
        prompt_rows = prompt_rows[: max(0, int(args.limit))]
    if not prompt_rows:
        raise GateError("selected prompt count must be positive")
    prompt_groups = _prompt_groups(prompt_rows, int(args.concurrency))

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
        + rows_per_job
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
        for group_index, (group_rows, valid_rows) in enumerate(prompt_groups):
            token_rows = tuple(
                prompt_tokens[str(row["id"])] for row in group_rows
            )
            teacher_group = _strict_teacher_tokens(
                sessions, token_rows, int(args.decode_steps)
            )
            capture = _capture_verifier_group(
                sessions,
                token_rows,
                teacher_group,
                int(args.decode_steps),
                rows_per_job=rows_per_job,
                wide_q6_shared4=False,
            )
            for index, row in enumerate(group_rows[:valid_rows]):
                request_id = str(row["id"])
                strict[request_id] = capture[index]
                teachers[request_id] = teacher_group[index]
            print(
                f"strict {group_index + 1}/{len(prompt_groups)}",
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
            for group_rows, valid_rows in prompt_groups:
                token_rows = tuple(
                    prompt_tokens[str(row["id"])] for row in group_rows
                )
                teacher_group = tuple(
                    teachers[str(row["id"])] for row in group_rows
                )
                capture = _capture_verifier_group(
                    sessions,
                    token_rows,
                    teacher_group,
                    int(args.decode_steps),
                    rows_per_job=rows_per_job,
                    wide_q6_shared4=True,
                )
                for index, row in enumerate(group_rows[:valid_rows]):
                    candidate[str(row["id"])].append(capture[index])
            print(
                f"candidate {repeat + 1}/{int(args.repeat_runs)}",
                flush=True,
            )
    finally:
        stack.close()

    shape_label = (
        f"c{int(args.concurrency)}_k{int(args.candidate_budget)}_"
        f"r{int(args.concurrency) * rows_per_job}"
    )
    captures = tuple(
        BatchRouteCapture(
            scenario_id=shape_label,
            request_id=str(row["id"]),
            category=str(row["category"]),
            strict=strict[str(row["id"])],
            candidate_runs=tuple(candidate[str(row["id"])]),
            shapes=(shape_label,) * int(args.decode_steps),
            transitions=(f"prefill_to_{shape_label}",)
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
        timing_protocol=f"none_full_logits_teacher_forced_{shape_label}_v1",
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
            "production": (
                "FP16 state + shape-scoped Q4/Q5 rowtiles and the W1 "
                "B-stationary Q6 shared4 candidate at "
                f"physical R{int(args.concurrency) * rows_per_job}"
            ),
            "strict_fallbacks": [
                "linear/gguf_q4_k_t16_v1/t16_wmma_prefill_smallm_bf16_bf16_out",
                "linear_pair_silu/gguf_q4_k_t16_v1/dense_dual_wmma_prefill_bf16_bf16_out",
            ],
            "excluded": [
                "rows outside the selected physical cell",
                "narrow K5120/N1024",
                "peer backends",
            ],
        },
        "protocol": {
            "model": str(args.model.resolve()),
            "prompts": str(args.prompts.resolve()),
            "prompt_count": len(prompt_rows),
            "decode_steps": int(args.decode_steps),
            "teacher_rows": len(prompt_rows) * int(args.decode_steps),
            "repeat_runs": int(args.repeat_runs),
            "concurrency": int(args.concurrency),
            "candidate_budget": int(args.candidate_budget),
            "rows_per_job": rows_per_job,
            "physical_shape": shape_label.upper(),
            "teacher_source": "strict fixed-schedule trajectory",
        },
        "quality": quality,
        "control": {
            "request_positions_exact": True,
            "input_tokens_exact": True,
            "same_width_grouping": True,
            "final_group_padding": -len(prompt_rows) % int(args.concurrency),
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
    parser.add_argument(
        "--concurrency", type=int, choices=(2, 3, 5, 6, 8), default=2
    )
    parser.add_argument("--candidate-budget", type=int, choices=(1, 2, 3), default=3)
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
