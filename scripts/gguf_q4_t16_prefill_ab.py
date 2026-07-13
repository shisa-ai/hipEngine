#!/usr/bin/env python3
"""Gate the Q4T16 shared-X prefill kernel on full-model wall and decode state."""

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

from hipengine.benchmark.correctness import evaluate_logits  # noqa: E402
from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402


KIND = "hipengine_gguf_q4_t16_prefill_full_model_ab"
SCHEMA_VERSION = 1
BASELINE_MODE = "baseline"
CANDIDATE_MODE = "shared_x"
MODES = (BASELINE_MODE, CANDIDATE_MODE)
SELECTOR_ENV = "HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE"
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_KERNEL_CORRECTNESS = (
    REPO_ROOT
    / "benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf3a-q4t16-shared-x-replay.json"
)


class GateError(RuntimeError):
    """Raised when the full-model A/B cannot produce valid evidence."""


def _parse_contexts(raw: str) -> tuple[int, ...]:
    try:
        contexts = tuple(int(part.strip()) for part in str(raw).split(","))
    except ValueError as exc:
        raise GateError("contexts must be comma-separated positive integers") from exc
    if not contexts or any(context <= 0 for context in contexts):
        raise GateError("contexts must be comma-separated positive integers")
    if len(set(contexts)) != len(contexts):
        raise GateError("contexts must not contain duplicates")
    return contexts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_kernel_correctness_gate(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise GateError(f"kernel correctness artifact does not exist: {resolved}")
    try:
        artifact = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"could not read kernel correctness artifact: {resolved}") from exc
    correctness = artifact.get("correctness")
    if not isinstance(correctness, Mapping):
        raise GateError("kernel correctness artifact has no correctness record")
    bf16_exact = correctness.get("bf16_raw_bytes_exact") is True
    fp16_exact = correctness.get("fp16_raw_bytes_exact") is True
    try:
        tests_passed = int(correctness.get("tests_passed", 0))
    except (TypeError, ValueError) as exc:
        raise GateError("kernel correctness test count is malformed") from exc
    if not bf16_exact or not fp16_exact or tests_passed <= 0:
        raise GateError("kernel correctness artifact is not BF16/FP16 byte-exact")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "kind": artifact.get("kind"),
        "tests_passed": tests_passed,
        "bf16_raw_bytes_exact": bf16_exact,
        "fp16_raw_bytes_exact": fp16_exact,
        "passed": True,
    }


@contextlib.contextmanager
def _q4_t16_mode(mode: str) -> Iterator[None]:
    if mode not in MODES:
        raise GateError(f"unsupported Q4T16 prefill mode: {mode!r}")
    previous = os.environ.get(SELECTOR_ENV)
    os.environ[SELECTOR_ENV] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SELECTOR_ENV, None)
        else:
            os.environ[SELECTOR_ENV] = previous


def _mode_statistics(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise GateError("timing samples must be finite and positive")
    return {
        "samples_ms": values,
        "count": len(values),
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.fmean(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "stdev_ms": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _phase_summary(
    measurements: Sequence[Mapping[str, Any]],
    *,
    field: str,
    work_items: int,
) -> dict[str, Any]:
    statistics_by_mode = {
        mode: _mode_statistics([float(row["modes"][mode][field]) for row in measurements])
        for mode in MODES
    }
    paired_delta_ms = [
        float(row["modes"][CANDIDATE_MODE][field]) - float(row["modes"][BASELINE_MODE][field])
        for row in measurements
    ]
    baseline_median = float(statistics_by_mode[BASELINE_MODE]["median_ms"])
    candidate_median = float(statistics_by_mode[CANDIDATE_MODE]["median_ms"])
    return {
        "statistics": statistics_by_mode,
        "median_throughput": {
            mode: float(1000.0 * int(work_items) / statistics_by_mode[mode]["median_ms"])
            for mode in MODES
        },
        "paired_candidate_minus_baseline_ms": paired_delta_ms,
        "paired_candidate_minus_baseline_median_ms": float(statistics.median(paired_delta_ms)),
        "candidate_speedup_vs_baseline": float(baseline_median / candidate_median),
        "candidate_wall_delta_percent": float(
            ((candidate_median - baseline_median) / baseline_median) * 100.0
        ),
        "candidate_wins": bool(candidate_median < baseline_median),
    }


def _summarize_context(
    measurements: Sequence[Mapping[str, Any]],
    *,
    context_tokens: int,
) -> dict[str, Any]:
    if not measurements:
        raise GateError("context summary requires measured repetitions")
    reference_tokens = [
        int(token) for token in measurements[0]["modes"][BASELINE_MODE]["token_ids"]
    ]
    if len(reference_tokens) < 2:
        raise GateError("trajectory must contain the prefill sample and decode steps")
    trajectories_exact = all(
        [int(token) for token in row["modes"][mode]["token_ids"]] == reference_tokens
        for row in measurements
        for mode in MODES
    )
    decode_steps = len(reference_tokens) - 1
    return {
        "measurements": [dict(row) for row in measurements],
        "prefill": _phase_summary(
            measurements,
            field="prefill_wall_ms",
            work_items=int(context_tokens),
        ),
        "decode": _phase_summary(
            measurements,
            field="decode_wall_ms",
            work_items=decode_steps,
        ),
        "decode_steps": int(decode_steps),
        "trajectories_exact": trajectories_exact,
        "reference_token_ids": reference_tokens,
    }


def _compare_prefill_results(baseline: Any, candidate: Any) -> dict[str, Any]:
    baseline_logits = np.ascontiguousarray(baseline.logits, dtype=np.float32)
    candidate_logits = np.ascontiguousarray(candidate.logits, dtype=np.float32)
    if baseline_logits.shape != candidate_logits.shape:
        raise GateError(
            f"prefill logit shape mismatch: {baseline_logits.shape} != {candidate_logits.shape}"
        )
    if not np.all(np.isfinite(baseline_logits)) or not np.all(np.isfinite(candidate_logits)):
        raise GateError("prefill logits contain NaN or Inf")
    result = evaluate_logits(baseline_logits, candidate_logits)
    token_exact = int(baseline.token_id) == int(candidate.token_id)
    logits_byte_exact = baseline_logits.tobytes() == candidate_logits.tobytes()
    return {
        "passed": bool(token_exact and logits_byte_exact and result.passed),
        "token_exact": token_exact,
        "baseline_token_id": int(baseline.token_id),
        "candidate_token_id": int(candidate.token_id),
        "logits_byte_exact": logits_byte_exact,
        "logit_count": int(baseline_logits.size),
        "max_abs_diff": float(np.max(np.abs(baseline_logits - candidate_logits), initial=0.0)),
        "kl_mean": float(result.kl_mean),
        "kl_max": float(result.kl_max),
        "top1_agreement": float(result.top1_agreement),
    }


def _run_prefill_correctness(
    session: Any,
    *,
    prompt_ids: Sequence[int],
    bulk_attention_mode: str,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mode in MODES:
        with _q4_t16_mode(mode):
            results[mode] = session.prefill(
                [int(token) for token in prompt_ids],
                use_bulk=True,
                bulk_attention_mode=bulk_attention_mode,
                return_logits=True,
                capture_hidden_seed_fp32=False,
            )
    return _compare_prefill_results(
        results[BASELINE_MODE],
        results[CANDIDATE_MODE],
    )


def _run_mode_leg(
    session: Any,
    *,
    mode: str,
    prompt_ids: Sequence[int],
    decode_steps: int,
    bulk_attention_mode: str,
    graph_replay_decode: bool,
) -> dict[str, Any]:
    runtime = session.runtime
    if runtime is None:
        raise GateError("GGUF session has no HIP runtime")
    runtime.device_synchronize()
    prefill_started_ns = time.perf_counter_ns()
    with _q4_t16_mode(mode):
        first = session.prefill(
            [int(token) for token in prompt_ids],
            use_bulk=True,
            bulk_attention_mode=bulk_attention_mode,
            return_logits=False,
            capture_hidden_seed_fp32=False,
        )
    runtime.device_synchronize()
    prefill_wall_ms = (time.perf_counter_ns() - prefill_started_ns) / 1_000_000.0
    graph_capture_ms = 0.0
    generated = [int(first.token_id)]
    if graph_replay_decode:
        minimum_fn = getattr(session, "decode_graph_min_replay_steps", None)
        graph_minimum = minimum_fn() if callable(minimum_fn) else None
        if graph_minimum is None or int(decode_steps) < int(graph_minimum):
            raise GateError("requested graph replay is not admitted for this session/decode window")
        capture_started_ns = time.perf_counter_ns()
        graph = session.capture_decode_graph(
            position=int(session.position),
            steps_per_replay=1,
            max_replay_steps=int(decode_steps),
            record_steps=int(decode_steps),
            attention_max_context_len=int(session.position) + int(decode_steps),
        )
        runtime.device_synchronize()
        graph_capture_ms = (time.perf_counter_ns() - capture_started_ns) / 1_000_000.0
        try:
            runtime.device_synchronize()
            decode_started_ns = time.perf_counter_ns()
            graph.replay(int(decode_steps))
            runtime.device_synchronize()
            decode_wall_ms = (time.perf_counter_ns() - decode_started_ns) / 1_000_000.0
            generated.extend(graph.read_generated_token_ids(int(decode_steps)))
        finally:
            graph.close()
    else:
        current = int(first.token_id)
        runtime.device_synchronize()
        decode_started_ns = time.perf_counter_ns()
        for _ in range(int(decode_steps)):
            result = session.step(current, return_logits=False)
            current = int(result.token_id)
            generated.append(current)
        runtime.device_synchronize()
        decode_wall_ms = (time.perf_counter_ns() - decode_started_ns) / 1_000_000.0
    if prefill_wall_ms <= 0.0 or decode_wall_ms <= 0.0:
        raise GateError("prefill/decode walls must be positive")
    return {
        "prefill_wall_ms": float(prefill_wall_ms),
        "prefill_tok_s": float(1000.0 * len(prompt_ids) / prefill_wall_ms),
        "prefill_token_id": int(first.token_id),
        "graph_capture_ms_excluded": float(graph_capture_ms),
        "decode_wall_ms": float(decode_wall_ms),
        "decode_tok_s": float(1000.0 * int(decode_steps) / decode_wall_ms),
        "token_ids": generated,
        "graph_replay_decode": bool(graph_replay_decode),
    }


def _classify_gate(
    contexts: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not contexts:
        raise GateError("classification requires at least one context")
    clean = not bool(provenance.get("dirty"))
    correctness_passed = all(
        bool(row["correctness"]["passed"]) and bool(row["correctness"]["logits_byte_exact"])
        for row in contexts
    )
    trajectories_exact = all(bool(row["trajectories_exact"]) for row in contexts)
    candidate_wins_all = all(bool(row["prefill"]["candidate_wins"]) for row in contexts)
    baseline_decode_ms = sum(
        float(row["decode"]["statistics"][BASELINE_MODE]["median_ms"]) for row in contexts
    )
    candidate_decode_ms = sum(
        float(row["decode"]["statistics"][CANDIDATE_MODE]["median_ms"]) for row in contexts
    )
    decode_non_regressive = candidate_decode_ms <= baseline_decode_ms
    measurement_valid = bool(clean and correctness_passed and trajectories_exact)
    if not clean:
        status = "invalid_measurement"
        selected_default = "unchanged"
        conclusion = "The full-model A/B has dirty provenance."
    elif not correctness_passed or not trajectories_exact:
        status = "reject_correctness"
        selected_default = BASELINE_MODE
        conclusion = "shared_x changes full-model logits or a decoded token trajectory."
    elif not candidate_wins_all:
        status = "reject_prefill_performance"
        selected_default = BASELINE_MODE
        conclusion = "shared_x does not reduce median prefill wall at every focus context."
    elif not decode_non_regressive:
        status = "reject_decode_regression"
        selected_default = BASELINE_MODE
        conclusion = "shared_x regresses the aggregate balanced decode wall."
    else:
        status = "promote_shared_x"
        selected_default = CANDIDATE_MODE
        conclusion = (
            "shared_x is byte-exact, wins prefill at every focus context, and does "
            "not regress aggregate decode wall."
        )
    return {
        "status": status,
        "selected_default": selected_default,
        "conclusion": conclusion,
        "measurement_valid": measurement_valid,
        "clean_provenance": clean,
        "correctness_passed": correctness_passed,
        "trajectories_exact": trajectories_exact,
        "candidate_wins_all_contexts": candidate_wins_all,
        "decode_non_regressive": decode_non_regressive,
        "baseline_decode_median_wall_sum_ms": float(baseline_decode_ms),
        "candidate_decode_median_wall_sum_ms": float(candidate_decode_ms),
        "candidate_decode_speedup": float(baseline_decode_ms / candidate_decode_ms),
        "candidate_decode_wall_delta_percent": float(
            ((candidate_decode_ms - baseline_decode_ms) / baseline_decode_ms) * 100.0
        ),
    }


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    contexts = _parse_contexts(args.contexts)
    warmups = int(args.warmups)
    repetitions = int(args.repetitions)
    decode_steps = int(args.decode_steps)
    if warmups < 0:
        raise GateError("warmups must be non-negative")
    if repetitions <= 0 or repetitions % 2:
        raise GateError("repetitions must be a positive even integer")
    if decode_steps <= 0:
        raise GateError("decode steps must be positive")
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise GateError(f"model does not exist: {model}")
    kernel_correctness = _load_kernel_correctness_gate(args.kernel_correctness_artifact)
    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
    os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "1" if args.decode_repack else "0"

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    context_records: list[dict[str, Any]] = []
    with Qwen35GGUFResidentSession(
        model,
        backend=str(args.backend),
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max(contexts) + decode_steps + 2,
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=True,
    ) as session:
        if session.runner is None:
            raise GateError("GGUF resident session closed during setup")
        resolved_backend = str(session.runner.backend)
        target_arch = str(session.runner.target_arch)
        for context in contexts:
            prompt_ids = [int(args.prompt_token_id)] * int(context)
            correctness = _run_prefill_correctness(
                session,
                prompt_ids=prompt_ids,
                bulk_attention_mode=str(args.bulk_attention_mode),
            )
            warmup_records: list[dict[str, Any]] = []
            for warmup in range(warmups):
                order = MODES if warmup % 2 == 0 else tuple(reversed(MODES))
                row: dict[str, Any] = {
                    "warmup": int(warmup),
                    "order": list(order),
                    "modes": {},
                }
                for mode in order:
                    row["modes"][mode] = _run_mode_leg(
                        session,
                        mode=mode,
                        prompt_ids=prompt_ids,
                        decode_steps=decode_steps,
                        bulk_attention_mode=str(args.bulk_attention_mode),
                        graph_replay_decode=bool(args.graph_replay_decode),
                    )
                warmup_records.append(row)
            measured: list[dict[str, Any]] = []
            for repetition in range(repetitions):
                order = MODES if repetition % 2 == 0 else tuple(reversed(MODES))
                row = {
                    "repetition": int(repetition),
                    "order": list(order),
                    "modes": {},
                }
                for mode in order:
                    row["modes"][mode] = _run_mode_leg(
                        session,
                        mode=mode,
                        prompt_ids=prompt_ids,
                        decode_steps=decode_steps,
                        bulk_attention_mode=str(args.bulk_attention_mode),
                        graph_replay_decode=bool(args.graph_replay_decode),
                    )
                measured.append(row)
            summary = _summarize_context(measured, context_tokens=int(context))
            context_records.append(
                {
                    "context_tokens": int(context),
                    "prompt_token_id": int(args.prompt_token_id),
                    "prompt_sha256_i64": hashlib.sha256(
                        np.asarray(prompt_ids, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "correctness": correctness,
                    "warmups": warmup_records,
                    **summary,
                }
            )
            print(
                f"context={context} exact={correctness['passed']} "
                f"prefill_speedup={summary['prefill']['candidate_speedup_vs_baseline']:.6f} "
                f"decode_speedup={summary['decode']['candidate_speedup_vs_baseline']:.6f}",
                flush=True,
            )

    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_BACKEND": os.environ.get("HIPENGINE_BACKEND"),
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIPENGINE_GGUF_DECODE_REPACK": os.environ.get("HIPENGINE_GGUF_DECODE_REPACK"),
            SELECTOR_ENV: f"balanced {BASELINE_MODE} versus {CANDIDATE_MODE}",
        },
        build_profile="gguf_q4_t16_prefill_full_model_ab",
        timing_protocol="same_session_balanced_prefill_decode_v1",
        warmups=warmups,
        repetitions=repetitions,
        profiler={"enabled": False, "kind": None, "command": None},
    )
    decision = _classify_gate(context_records, provenance=provenance)
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": decision["status"],
        "performance_claim": bool(decision["measurement_valid"]),
        "correctness_claim": True,
        "workload": {
            "model": str(model),
            "quant": "gguf_q4_k_m",
            "kv_dtype": "bf16",
            "contexts": list(contexts),
            "decode_steps": decode_steps,
            "prompt_token_id": int(args.prompt_token_id),
            "modes": list(MODES),
            "bulk_attention_mode": str(args.bulk_attention_mode),
            "graph_replay_decode": bool(args.graph_replay_decode),
            "decode_repack": bool(args.decode_repack),
        },
        "protocol": {
            "correctness": (
                "same-session baseline/candidate prefill logits must be byte-exact; "
                "every measured prefill sample plus decode token trajectory must match"
            ),
            "prefill_wall": "synchronized production session.prefill; token readback included",
            "decode_wall": (
                "synchronized graph replay; graph capture and generated-token readback excluded"
                if args.graph_replay_decode
                else "synchronized eager session.step loop"
            ),
            "ordering": "balanced baseline-candidate / candidate-baseline by repetition",
            "session_scope": "one resident model/session shared by every mode and context",
            "warmups_per_mode_context": warmups,
            "measured_repetitions_per_mode_context": repetitions,
            "promotion_rule": (
                "clean provenance, BF16/FP16 kernel bytes exact, full-model logits and "
                "trajectories exact, lower median prefill wall at every context, and no "
                "aggregate median decode-wall regression; no percentage threshold"
            ),
        },
        "kernel_correctness_gate": kernel_correctness,
        "contexts": context_records,
        "decision": decision,
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--contexts", default="512,1024,4096")
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument(
        "--bulk-attention-mode",
        choices=("bulk", "native"),
        default="bulk",
    )
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
        "--kernel-correctness-artifact",
        type=Path,
        default=DEFAULT_KERNEL_CORRECTNESS,
    )
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
    print(json.dumps(artifact["decision"], indent=2, sort_keys=True))
    return 0 if artifact["decision"]["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
