#!/usr/bin/env python3
"""C8-P4 L4 numerics: grouped q8_1 DP4A planar-Q6 substitution.

Paired resident sessions replay identical strict-teacher token blocks over
the ten-prompt category suite (plus any heldouts passed via --prompts) and
emit full-vocabulary logits. The strict side uses the retained exact BF16
decode configuration. The candidate differs ONLY by
``HIPENGINE_C8_Q6_DP4A_GROUPED=1``: the planar-Q6 grouped decode routes
rows 8-64 through the integer-dp4a sibling (x quantized to q8_1 per call).
Q4 rowtile, fp16 recurrent state, and Q5 MMQ axes stay strict on both sides
so the measurement attributes purely to the Q6 dp4a axis.

Quality is evaluated by the campaign's normative batch-route evaluator
(``build_batch_route_quality`` with ``EvaluationThresholds``) plus repeat
determinism across ``--repeat-runs`` candidate replays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.benchmark.execution_profiles import EvaluationThresholds
from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.execution_profile_gguf_batch_route_gate import (
    BatchRouteCapture,
    build_batch_route_quality,
)
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.qwen38_q4_verifier_numerics import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    GateError,
    VERIFY_CAPTURE_ENV,
    _capture_verifier_group,  # noqa: F401 - retained for protocol parity
    _environment,
    _load_suites,
    _prompt_groups,
    _reset_sessions,
    _strict_teacher_tokens,
    _trajectory_sha256,
)
from scripts.execution_profile_gguf_fp16_state_gate import (
    production_route_environment,
)

DP4A_GROUPED_ENV = "HIPENGINE_C8_Q6_DP4A_GROUPED"
KIND = "qwen38_c8_p4_q6_dp4a_numerics"


@contextmanager
def _dp4a_environment(enabled: bool):
    with _environment(DP4A_GROUPED_ENV, "1" if enabled else "0"):
        yield


def _capture_decode_regime_group(
    sessions: Sequence[Any],
    token_rows: Sequence[Sequence[int]],
    teacher_tokens: Sequence[Sequence[int]],
    decode_steps: int,
    *,
    rows_per_job: int,
) -> tuple[tuple[dict[str, object], ...], ...]:
    """Verify-block capture in the production decode regime (WMMA off).

    The retained C8/K3 profile exercises the grouped-GEMV decode family for
    rows 8-64 (census-proven); the WMMA prefill owns populated prefill rows.
    Forcing per-job ``use_wmma_prefill`` False reproduces the decode regime
    on both sides so the measurement attributes purely to the Q6 dp4a axis.
    """

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
                "use_wmma_prefill": False,
            }
            for index in range(len(sessions))
        ]
        positions_before = [int(session.position) for session in sessions]
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


def _read_packed_logits(owner: Any, rows: int) -> np.ndarray:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

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


def _prefill(sessions: Sequence[Any], token_rows: Sequence[Sequence[int]]) -> Sequence[Any]:
    _reset_sessions(sessions)
    return sessions[0].prefill_batch_native(
        token_rows,
        sessions=sessions,
        return_logits=True,
        require_logits=True,
    )


def _make_sessions(
    args: argparse.Namespace,
    *,
    dp4a: bool,
    max_sequence_length: int,
):
    from contextlib import ExitStack

    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    compiler_version = (
        None
        if args.compiler_version_file is None
        else args.compiler_version_file.read_text(encoding="utf-8")
    )
    stack = ExitStack()
    try:
        stack.enter_context(_dp4a_environment(dp4a))
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
                        prefill_config=PrefillConfig(
                            attn_aotriton_min_tokens=512
                        ),
                    )
                )
            )
    except BaseException:
        stack.close()
        raise
    return stack, tuple(sessions_list)


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
    physical_rows = int(args.concurrency) * rows_per_job
    if physical_rows < 8 or physical_rows % 8:
        raise GateError("grouped dp4a route requires physical rows multiple of 8")

    suite_paths = (
        tuple(args.prompts)
        if args.prompts
        else (
            DEFAULT_PROMPTS,
            ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl",
        )
    )
    prompt_rows = _load_suites(suite_paths)
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

    strict: dict[str, tuple[dict[str, object], ...]] = {}
    teachers: dict[str, tuple[int, ...]] = {}
    stack, sessions = _make_sessions(
        args, dp4a=False, max_sequence_length=max_sequence_length
    )
    try:
        for group_index, (group_rows, valid_rows) in enumerate(prompt_groups):
            token_rows = tuple(
                prompt_tokens[str(row["id"])] for row in group_rows
            )
            teacher_group = _strict_teacher_tokens(
                sessions, token_rows, int(args.decode_steps)
            )
            capture = _capture_decode_regime_group(
                sessions,
                token_rows,
                teacher_group,
                int(args.decode_steps),
                rows_per_job=rows_per_job,
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
        args, dp4a=True, max_sequence_length=max_sequence_length
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
                capture = _capture_decode_regime_group(
                    sessions,
                    token_rows,
                    teacher_group,
                    int(args.decode_steps),
                    rows_per_job=rows_per_job,
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
        f"r{physical_rows}"
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
        target_arch=str(args.backend).removeprefix("hip_"),
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "strict": {DP4A_GROUPED_ENV: "0"},
            "candidate": {DP4A_GROUPED_ENV: "1"},
        },
        build_profile="qwen38_c8_p4_q6_dp4a_numerics",
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
    status = "passed" if hard_passed and tracked_clean else "failed"
    payload = {
        "schema": 1,
        "kind": KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "hard_gates_passed": hard_passed,
        "tracked_clean": tracked_clean,
        "axis": "q6_grouped_dp4a_only",
        "shape_label": shape_label,
        "decode_steps": int(args.decode_steps),
        "repeat_runs": int(args.repeat_runs),
        "prompts": len(prompt_rows),
        "quality": quality,
        "trajectory_sha256": {
            "strict": {
                str(row["id"]): _trajectory_sha256(strict[str(row["id"])])
                for row in prompt_rows
            },
            "candidate_first": {
                str(row["id"]): _trajectory_sha256(
                    candidate[str(row["id"])][0]
                )
                for row in prompt_rows
            },
        },
        "provenance": provenance,
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--concurrency", type=int, choices=(2, 3, 8), default=8)
    parser.add_argument("--candidate-budget", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--prompts",
        type=Path,
        action="append",
        default=None,
        help="prompt suite; repeatable, defaults to the full category suite",
    )
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
                "hard_gates_passed": payload["hard_gates_passed"],
                "summary": payload["quality"]["quality"]["summary"],
            },
            indent=2,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
