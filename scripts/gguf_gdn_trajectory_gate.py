#!/usr/bin/env python3
"""Gate a GGUF GDN prefill candidate on natural trajectories and decode wall."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import load_prompt_rows, prompt_sha256


KIND = "hipengine_gguf_gdn_prefill_trajectory_decode_gate"
SCHEMA_VERSION = 1
BASELINE_MODE = "fused"
CANDIDATE_MODES = (
    "chain",
    "chain_tile64",
    "chain_tile32",
    "chain_wave32",
    "chain_wave32_tree",
    "chain_lds64",
    "chain_lds32",
    "chain_lds32_direct",
)
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"


class GateError(RuntimeError):
    """Raised when the trajectory/decode gate cannot be evaluated safely."""


@contextlib.contextmanager
def _gdn_mode(mode: str) -> Iterator[None]:
    if mode not in (BASELINE_MODE, *CANDIDATE_MODES):
        raise GateError(f"unsupported GDN mode: {mode!r}")
    name = "HIPENGINE_GGUF_GDN_PREFILL_MODE"
    previous = os.environ.get(name)
    os.environ[name] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _compare_trajectories(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    kl_threshold: float = 0.05,
    top1_threshold: float = 0.90,
) -> dict[str, Any]:
    if not baseline or len(baseline) != len(candidate):
        raise GateError("baseline and candidate trajectories must have equal nonzero length")
    rows: list[dict[str, Any]] = []
    for index, (baseline_step, candidate_step) in enumerate(
        zip(baseline, candidate, strict=True)
    ):
        baseline_token = int(baseline_step["token_id"])
        candidate_token = int(candidate_step["token_id"])
        baseline_logits = np.asarray(baseline_step["logits"], dtype=np.float32)
        candidate_logits = np.asarray(candidate_step["logits"], dtype=np.float32)
        if not np.all(np.isfinite(baseline_logits)) or not np.all(
            np.isfinite(candidate_logits)
        ):
            raise GateError(f"non-finite logits at trajectory transition {index}")
        try:
            result = evaluate_logits(
                baseline_logits,
                candidate_logits,
                kl_threshold=float(kl_threshold),
                top1_threshold=float(top1_threshold),
            )
        except ValueError as exc:
            raise GateError(f"logit comparison failed at transition {index}: {exc}") from exc
        token_exact = baseline_token == candidate_token
        rows.append(
            {
                "transition": int(index),
                "baseline_token_id": baseline_token,
                "candidate_token_id": candidate_token,
                "token_exact": token_exact,
                "kl_mean": float(result.kl_mean),
                "kl_max": float(result.kl_max),
                "top1_agreement": float(result.top1_agreement),
                "logit_gate_passed": bool(result.passed),
                "passed": bool(token_exact and result.passed),
            }
        )
    token_divergences = [row["transition"] for row in rows if not row["token_exact"]]
    passed_rows = sum(bool(row["passed"]) for row in rows)
    return {
        "passed": passed_rows == len(rows),
        "tokens_exact": not token_divergences,
        "first_token_divergence": (
            None if not token_divergences else int(token_divergences[0])
        ),
        "transitions_total": len(rows),
        "transitions_passed": int(passed_rows),
        "kl_mean_max": float(max(row["kl_mean"] for row in rows)),
        "kl_max": float(max(row["kl_max"] for row in rows)),
        "top1_agreement_min": float(min(row["top1_agreement"] for row in rows)),
        "kl_threshold": float(kl_threshold),
        "top1_threshold": float(top1_threshold),
        "transitions": rows,
    }


def _mode_statistics(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise GateError("decode wall samples must be finite and positive")
    return {
        "samples_ms": values,
        "count": len(values),
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.fmean(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "stdev_ms": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _summarize_decode_measurements(
    measurements: Sequence[Mapping[str, Any]],
    *,
    decode_steps: int,
    candidate_mode: str,
) -> dict[str, Any]:
    if not measurements:
        raise GateError("decode summary requires measured repetitions")
    modes = (BASELINE_MODE, candidate_mode)
    statistics_by_mode = {
        mode: _mode_statistics(
            [float(row["modes"][mode]["wall_ms"]) for row in measurements]
        )
        for mode in modes
    }
    paired_delta_ms = [
        float(row["modes"][candidate_mode]["wall_ms"])
        - float(row["modes"][BASELINE_MODE]["wall_ms"])
        for row in measurements
    ]
    reference_tokens = [
        int(token) for token in measurements[0]["modes"][BASELINE_MODE]["token_ids"]
    ]
    trajectories_exact = all(
        [int(token) for token in row["modes"][mode]["token_ids"]]
        == reference_tokens
        for row in measurements
        for mode in modes
    )
    baseline_median = float(statistics_by_mode[BASELINE_MODE]["median_ms"])
    candidate_median = float(statistics_by_mode[candidate_mode]["median_ms"])
    return {
        "measurements": [dict(row) for row in measurements],
        "statistics": statistics_by_mode,
        "paired_candidate_minus_baseline_ms": paired_delta_ms,
        "paired_candidate_minus_baseline_median_ms": float(
            statistics.median(paired_delta_ms)
        ),
        "decode_steps": int(decode_steps),
        "trajectories_exact": trajectories_exact,
        "reference_token_ids": reference_tokens,
        "candidate_speedup_vs_baseline": float(baseline_median / candidate_median),
        "candidate_wall_delta_percent": float(
            ((candidate_median - baseline_median) / baseline_median) * 100.0
        ),
        "candidate_wins": bool(candidate_median < baseline_median),
    }


def _run_logits_trajectory(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    mode: str,
    decode_steps: int,
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


def _run_timed_decode(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    mode: str,
    decode_steps: int,
    bulk_attention_mode: str,
    graph_replay_decode: bool,
) -> dict[str, Any]:
    session.reset()
    with _gdn_mode(mode):
        first = session.prefill(
            [int(token) for token in prompt_ids],
            use_bulk=True,
            bulk_attention_mode=bulk_attention_mode,
            return_logits=False,
            capture_hidden_seed_fp32=False,
        )
    current = int(first.token_id)
    generated = [current]
    minimum_fn = getattr(session, "decode_graph_min_replay_steps", None)
    graph_minimum = minimum_fn() if callable(minimum_fn) else None
    use_graph = bool(
        graph_replay_decode
        and graph_minimum is not None
        and int(decode_steps) >= int(graph_minimum)
        and callable(getattr(session, "capture_decode_graph", None))
    )
    runtime = session.runtime
    if runtime is None:
        raise GateError("GGUF session has no HIP runtime")
    runtime.device_synchronize()
    started_ns = time.perf_counter_ns()
    if use_graph:
        graph = session.capture_decode_graph(
            position=int(session.position),
            steps_per_replay=1,
            max_replay_steps=int(decode_steps),
            attention_max_context_len=int(session.position) + int(decode_steps),
        )
        try:
            for _ in range(int(decode_steps)):
                graph.replay(1)
                result = graph.read_sample(return_logits=False)
                current = int(result.token_id)
                generated.append(current)
        finally:
            graph.close()
    else:
        for _ in range(int(decode_steps)):
            result = session.step(current, return_logits=False)
            current = int(result.token_id)
            generated.append(current)
    runtime.device_synchronize()
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
        raise GateError(f"invalid decode wall for {mode}: {elapsed_ms!r}")
    return {
        "wall_ms": float(elapsed_ms),
        "decode_tok_s": float(1000.0 * int(decode_steps) / elapsed_ms),
        "token_ids": generated,
        "graph_replay_decode": use_graph,
        "graph_replay_min_steps": (
            None if graph_minimum is None else int(graph_minimum)
        ),
    }


def _aggregate_gate(
    prompts: Sequence[Mapping[str, Any]],
    *,
    candidate_mode: str,
    decode_steps: int,
) -> dict[str, Any]:
    if not prompts:
        raise GateError("gate requires at least one prompt")
    correctness_passed = all(bool(row["correctness"]["passed"]) for row in prompts)
    trajectories_exact = all(
        bool(row["decode_performance"]["trajectories_exact"]) for row in prompts
    )
    baseline_total_ms = sum(
        float(row["decode_performance"]["statistics"][BASELINE_MODE]["median_ms"])
        for row in prompts
    )
    candidate_total_ms = sum(
        float(row["decode_performance"]["statistics"][candidate_mode]["median_ms"])
        for row in prompts
    )
    total_decode_steps = int(decode_steps) * len(prompts)
    decode_non_regressive = candidate_total_ms <= baseline_total_ms
    transition_rows = [
        transition
        for prompt in prompts
        for transition in prompt["correctness"]["transitions"]
    ]
    paired_deltas = [
        float(delta)
        for prompt in prompts
        for delta in prompt["decode_performance"][
            "paired_candidate_minus_baseline_ms"
        ]
    ]
    return {
        "passed": bool(correctness_passed and trajectories_exact and decode_non_regressive),
        "correctness_passed": correctness_passed,
        "trajectory_tokens_exact": trajectories_exact,
        "prompts_total": len(prompts),
        "correctness_transitions_total": len(transition_rows),
        "correctness_transitions_passed": sum(
            bool(row["passed"]) for row in transition_rows
        ),
        "kl_mean_max": float(max(row["kl_mean"] for row in transition_rows)),
        "kl_max": float(max(row["kl_max"] for row in transition_rows)),
        "top1_agreement_min": float(
            min(row["top1_agreement"] for row in transition_rows)
        ),
        "decode_non_regressive": decode_non_regressive,
        "decode_steps_per_prompt": int(decode_steps),
        "decode_steps_total_per_mode": total_decode_steps,
        "baseline_decode_median_wall_sum_ms": float(baseline_total_ms),
        "candidate_decode_median_wall_sum_ms": float(candidate_total_ms),
        "baseline_decode_tok_s_weighted": float(
            1000.0 * total_decode_steps / baseline_total_ms
        ),
        "candidate_decode_tok_s_weighted": float(
            1000.0 * total_decode_steps / candidate_total_ms
        ),
        "candidate_decode_speedup": float(baseline_total_ms / candidate_total_ms),
        "candidate_decode_wall_delta_percent": float(
            ((candidate_total_ms - baseline_total_ms) / baseline_total_ms) * 100.0
        ),
        "paired_candidate_minus_baseline_median_ms": float(
            statistics.median(paired_deltas)
        ),
    }


def _classify_gate(
    summary: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    candidate_mode: str,
) -> dict[str, Any]:
    clean = not bool(provenance.get("dirty"))
    performance_comparison_valid = bool(clean and summary["trajectory_tokens_exact"])
    gate_passed = bool(clean and summary["passed"])
    if gate_passed:
        status = "accepted_project_trajectory_and_decode_nonregression"
        conclusion = (
            f"{candidate_mode} passes every natural-prompt trajectory logit/token "
            "gate and does not regress the balanced aggregate decode wall."
        )
    elif not clean:
        status = "invalid_measurement"
        conclusion = "The trajectory/decode measurement has dirty provenance."
    elif not summary["correctness_passed"] or not summary["trajectory_tokens_exact"]:
        status = "rejected_correctness"
        conclusion = f"{candidate_mode} changes a required natural trajectory."
    else:
        status = "rejected_decode_regression"
        conclusion = f"{candidate_mode} regresses the paired aggregate decode wall."
    return {
        "status": status,
        "conclusion": conclusion,
        "gate_passed": gate_passed,
        "measurement_valid": clean,
        "performance_comparison_valid": performance_comparison_valid,
    }


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    candidate_mode = str(args.candidate_mode)
    if candidate_mode not in CANDIDATE_MODES:
        raise GateError(f"unsupported candidate mode: {candidate_mode!r}")
    correctness_steps = int(args.correctness_decode_steps)
    performance_steps = int(args.performance_decode_steps)
    repetitions = int(args.performance_repetitions)
    if correctness_steps <= 0 or performance_steps <= 0:
        raise GateError("decode step counts must be positive")
    if repetitions <= 0 or repetitions % 2:
        raise GateError("performance repetitions must be a positive even integer")
    if not args.model.is_file():
        raise GateError(f"model does not exist: {args.model}")
    if not args.prompts.is_file():
        raise GateError(f"prompt suite does not exist: {args.prompts}")

    prompts = load_prompt_rows(args.prompts)
    if args.limit is not None:
        prompts = prompts[: max(0, int(args.limit))]
    if not prompts:
        raise GateError("selected prompt suite is empty")

    if bool(args.decode_repack):
        os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    else:
        os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "0"

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
        for row in prompts
    }
    max_prompt_tokens = max(len(tokens) for tokens in prompt_tokens.values())
    max_sequence_length = max_prompt_tokens + max(correctness_steps, performance_steps) + 2
    prefill_config = PrefillConfig(
        attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens)
    )

    prompt_records: list[dict[str, Any]] = []
    with Qwen35GGUFResidentSession(
        args.model,
        backend=str(args.backend),
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=bool(args.use_gemv_decode),
        prefill_config=prefill_config,
    ) as session:
        if session.runner is None:
            raise GateError("GGUF resident session closed during setup")
        resolved_backend = str(session.runner.backend)
        target_arch = str(session.runner.target_arch)
        for prompt_index, row in enumerate(prompts):
            prompt_id = str(row["id"])
            tokens = prompt_tokens[prompt_id]
            baseline_trajectory = _run_logits_trajectory(
                session,
                prompt_ids=tokens,
                mode=BASELINE_MODE,
                decode_steps=correctness_steps,
                bulk_attention_mode=str(args.bulk_attention_mode),
            )
            candidate_trajectory = _run_logits_trajectory(
                session,
                prompt_ids=tokens,
                mode=candidate_mode,
                decode_steps=correctness_steps,
                bulk_attention_mode=str(args.bulk_attention_mode),
            )
            correctness = _compare_trajectories(
                baseline_trajectory,
                candidate_trajectory,
                kl_threshold=float(args.kl_threshold),
                top1_threshold=float(args.top1_threshold),
            )
            measured: list[dict[str, Any]] = []
            modes = (BASELINE_MODE, candidate_mode)
            for repetition in range(repetitions):
                order = modes if repetition % 2 == 0 else tuple(reversed(modes))
                measured_row: dict[str, Any] = {
                    "repetition": repetition,
                    "order": list(order),
                    "modes": {},
                }
                for mode in order:
                    measured_row["modes"][mode] = _run_timed_decode(
                        session,
                        prompt_ids=tokens,
                        mode=mode,
                        decode_steps=performance_steps,
                        bulk_attention_mode=str(args.bulk_attention_mode),
                        graph_replay_decode=bool(args.graph_replay_decode),
                    )
                measured.append(measured_row)
            decode_performance = _summarize_decode_measurements(
                measured,
                decode_steps=performance_steps,
                candidate_mode=candidate_mode,
            )
            prompt_records.append(
                {
                    "id": prompt_id,
                    "category": str(row["category"]),
                    "prompt_chars": len(str(row["prompt"])),
                    "prompt_sha256": prompt_sha256(str(row["prompt"])),
                    "prompt_tokens": len(tokens),
                    "prompt_token_sha256_i64": hashlib.sha256(
                        np.asarray(tokens, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "correctness": correctness,
                    "decode_performance": decode_performance,
                }
            )
            print(
                f"{prompt_index + 1}/{len(prompts)} {prompt_id}: "
                f"correct={correctness['passed']} "
                f"decode_exact={decode_performance['trajectories_exact']}",
                flush=True,
            )

    summary = _aggregate_gate(
        prompt_records,
        candidate_mode=candidate_mode,
        decode_steps=performance_steps,
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
            "HIPENGINE_BACKEND": os.environ.get("HIPENGINE_BACKEND"),
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIPENGINE_GGUF_GDN_PREFILL_MODE": (
                f"explicit {BASELINE_MODE} versus {candidate_mode}"
            ),
            "HIPENGINE_GGUF_DECODE_REPACK": os.environ.get(
                "HIPENGINE_GGUF_DECODE_REPACK"
            ),
        },
        build_profile="gguf_gdn_trajectory_decode_gate",
        timing_protocol="multi_prompt_logits_and_balanced_decode_v1",
        warmups=1,
        repetitions=repetitions,
        profiler={"enabled": False, "kind": None, "command": None},
    )
    classification = _classify_gate(
        summary,
        provenance=provenance,
        candidate_mode=candidate_mode,
    )
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": classification["status"],
        "performance_claim": classification["performance_comparison_valid"],
        "correctness_claim": True,
        "gate_passed": classification["gate_passed"],
        "measurement_valid": classification["measurement_valid"],
        "performance_comparison_valid": classification[
            "performance_comparison_valid"
        ],
        "conclusion": classification["conclusion"],
        "workload": {
            "model": str(args.model.resolve()),
            "quant": "gguf_q4_k_m",
            "kv_dtype": "bf16",
            "prompt_suite": str(args.prompts.resolve()),
            "prompt_count": len(prompt_records),
            "prompt_ids": [record["id"] for record in prompt_records],
            "baseline_mode": BASELINE_MODE,
            "candidate_mode": candidate_mode,
            "correctness_decode_steps": correctness_steps,
            "performance_decode_steps": performance_steps,
            "bulk_attention_mode": str(args.bulk_attention_mode),
            "graph_replay_decode": bool(args.graph_replay_decode),
        },
        "protocol": {
            "correctness": (
                "own-token greedy fused/candidate trajectories; logits compared at "
                "the prefill sample and every decoded transition"
            ),
            "performance": (
                "same-session balanced mode order; synchronized production decode "
                "wall with graph capture/instantiate/close included when admitted"
            ),
            "performance_repetitions_per_mode_prompt": repetitions,
            "kl_threshold": float(args.kl_threshold),
            "top1_threshold": float(args.top1_threshold),
            "decode_nonregression_rule": (
                "candidate sum of per-prompt median walls must not exceed baseline; "
                "no percentage allowance"
            ),
        },
        "summary": summary,
        "prompts": prompt_records,
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--candidate-mode",
        choices=CANDIDATE_MODES,
        default="chain_wave32_tree",
    )
    parser.add_argument("--correctness-decode-steps", type=int, default=24)
    parser.add_argument("--performance-decode-steps", type=int, default=128)
    parser.add_argument("--performance-repetitions", type=int, default=2)
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--bulk-attention-mode", choices=("bulk", "native"), default="bulk")
    parser.add_argument(
        "--graph-replay-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--decode-repack",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-wmma-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-gemv-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=512)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    command = [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *raw_argv]
    try:
        artifact = run(args, command=command)
    except (GateError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.json)
    print(json.dumps({"status": artifact["status"], **artifact["summary"]}, indent=2))
    return 0 if artifact["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
