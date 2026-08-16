#!/usr/bin/env python3
"""Measure a quality-gated GGUF c1 route against its strict fallback.

The benchmark loads one resident model, warms both routes, and alternates their
order across repetitions.  It emits no quality verdict of its own: a complete,
passing same-host route-gate artifact is mandatory input.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.execution_profile_gguf_c1_route_gate import (
    CANDIDATES,
    KIND as QUALITY_KIND,
    STRICT_ENVIRONMENT,
    route_environment,
)
from scripts.gguf_gdn_semantic_gate import DEFAULT_MODEL, _configure_gate_environment

KIND = "hipengine_execution_profile_gguf_c1_route_performance"
SCHEMA_VERSION = 1


class PerformanceGateError(RuntimeError):
    """Raised when timing inputs or measurements are not retainable."""


def validate_quality_artifact(
    artifact: Mapping[str, Any],
    *,
    candidate_name: str,
    model: Path,
) -> None:
    if artifact.get("kind") != QUALITY_KIND or artifact.get("status") != "passed":
        raise PerformanceGateError("quality artifact is not a passing GGUF c1 route gate")
    if artifact.get("measurement_valid") is not True:
        raise PerformanceGateError("quality artifact measurement is not valid")
    candidate = artifact.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("name") != candidate_name:
        raise PerformanceGateError("quality artifact candidate differs from timing candidate")
    protocol = artifact.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("complete_prompt_and_heldout_suite") is not True:
        raise PerformanceGateError("quality artifact does not cover the complete prompt suite")
    quality_model = Path(str(protocol.get("model", ""))).resolve()
    if quality_model != model.resolve():
        raise PerformanceGateError("quality artifact model differs from timing model")


def summarize_samples(
    samples: Mapping[str, Sequence[float]],
    token_ids: Mapping[str, Sequence[Sequence[int]]],
) -> dict[str, Any]:
    required = {"strict", "candidate"}
    if set(samples) != required or set(token_ids) != required:
        raise ValueError("timing samples require strict and candidate routes")
    summary: dict[str, Any] = {}
    for route in ("strict", "candidate"):
        values = [float(value) for value in samples[route]]
        trajectories = [tuple(int(token) for token in row) for row in token_ids[route]]
        if not values or len(values) != len(trajectories):
            raise ValueError(f"{route} timing samples and token trajectories must align")
        if any(value <= 0.0 for value in values):
            raise ValueError("timing samples must be positive")
        summary[route] = {
            "samples_tok_s": values,
            "median_tok_s": statistics.median(values),
            "min_tok_s": min(values),
            "max_tok_s": max(values),
            "stdev_tok_s": statistics.stdev(values) if len(values) > 1 else 0.0,
            "generated_ids_repeatable": all(row == trajectories[0] for row in trajectories[1:]),
            "generated_token_ids": [list(row) for row in trajectories],
        }
    strict = float(summary["strict"]["median_tok_s"])
    candidate = float(summary["candidate"]["median_tok_s"])
    summary["candidate_vs_strict_pct"] = (candidate / strict - 1.0) * 100.0
    summary["candidate_faster"] = candidate > strict
    summary["candidate_generated_ids_repeatable"] = bool(
        summary["candidate"]["generated_ids_repeatable"]
    )
    summary["strict_candidate_generated_ids_equal"] = (
        summary["strict"]["generated_token_ids"][0]
        == summary["candidate"]["generated_token_ids"][0]
    )
    return summary


def _run_route(session: Any, *, environment: Mapping[str, str], args: argparse.Namespace) -> tuple[float, list[int]]:
    with route_environment(environment):
        session.reset()
        result = session.prefill(
            [int(args.prompt_token_id)] * int(args.prompt_length),
            use_bulk=True,
            bulk_attention_mode="bulk",
            return_logits=False,
        )
        for _ in range(int(args.warmup_decode_steps)):
            result = session.step(int(result.token_id), return_logits=False)
        session.runtime.device_synchronize()
        started = time.perf_counter()
        token_ids: list[int] = []
        for _ in range(int(args.decode_steps)):
            result = session.step(int(result.token_id), return_logits=False)
            token_ids.append(int(result.token_id))
        session.runtime.device_synchronize()
        elapsed = time.perf_counter() - started
    return float(args.decode_steps) / elapsed, token_ids


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model.is_file():
        raise PerformanceGateError(f"model does not exist: {args.model}")
    if not args.quality_artifact.is_file():
        raise PerformanceGateError(f"quality artifact does not exist: {args.quality_artifact}")
    if int(args.prompt_length) <= 0 or int(args.decode_steps) <= 0:
        raise PerformanceGateError("prompt length and decode steps must be positive")
    if int(args.repetitions) < 3 or int(args.warmup_decode_steps) < 0:
        raise PerformanceGateError("repetitions must be at least three and warmup steps non-negative")
    quality = json.loads(args.quality_artifact.read_text(encoding="utf-8"))
    if not isinstance(quality, Mapping):
        raise PerformanceGateError("quality artifact root must be an object")
    candidate = CANDIDATES[str(args.candidate)]
    validate_quality_artifact(quality, candidate_name=candidate.name, model=args.model)
    _configure_gate_environment(decode_repack=bool(args.decode_repack))

    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    compiler_version = (
        None
        if args.compiler_version_file is None
        else args.compiler_version_file.read_text(encoding="utf-8")
    )
    samples: dict[str, list[float]] = {"strict": [], "candidate": []}
    token_ids: dict[str, list[list[int]]] = {"strict": [], "candidate": []}
    measurements: list[dict[str, Any]] = []
    max_sequence_length = (
        int(args.prompt_length) + int(args.warmup_decode_steps) + int(args.decode_steps) + 2
    )
    with Qwen35GGUFResidentSession(
        args.model,
        backend=str(args.backend),
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=True,
        use_gemv_decode=True,
        prefill_config=PrefillConfig(attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens)),
    ) as session:
        if session.runner is None:
            raise PerformanceGateError("GGUF resident session closed during setup")
        resolved_backend = str(session.runner.backend)
        target_arch = str(session.runner.target_arch)
        _run_route(session, environment=STRICT_ENVIRONMENT, args=args)
        _run_route(session, environment=candidate.environment, args=args)
        routes = ("strict", "candidate")
        environments = {"strict": STRICT_ENVIRONMENT, "candidate": candidate.environment}
        for repetition in range(int(args.repetitions)):
            order = routes if repetition % 2 == 0 else tuple(reversed(routes))
            row: dict[str, Any] = {"repetition": repetition, "order": list(order), "routes": {}}
            for route in order:
                tok_s, ids = _run_route(session, environment=environments[route], args=args)
                samples[route].append(tok_s)
                token_ids[route].append(ids)
                row["routes"][route] = {"tok_s": tok_s, "final_token_id": ids[-1]}
                print(route, repetition, f"{tok_s:.6f}", ids[-1], flush=True)
            measurements.append(row)

    summary = summarize_samples(samples, token_ids)
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
        build_profile="execution_profile_gguf_c1_route_performance",
        timing_protocol="same_resident_alternating_route_wall_v1",
        warmups=1,
        repetitions=int(args.repetitions),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    quality_provenance = quality.get("provenance", {})
    same_quality_host = (
        isinstance(quality_provenance, Mapping)
        and quality_provenance.get("host_name") == provenance.get("host_name")
    )
    measurement_valid = bool(
        not provenance.get("dirty")
        and same_quality_host
        and summary["candidate_generated_ids_repeatable"]
    )
    retained = bool(measurement_valid and summary["candidate_faster"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "retained" if retained else "rejected_or_invalid",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "measurement_valid": measurement_valid,
        "candidate": {
            "name": candidate.name,
            "classification": candidate.classification,
            "strict_environment": dict(STRICT_ENVIRONMENT),
            "candidate_environment": dict(candidate.environment),
        },
        "quality_artifact": {
            "path": str(args.quality_artifact.resolve()),
            "same_host": same_quality_host,
            "quality_summary": quality["quality"]["quality"]["summary"],
            "state_repeat_passed": quality["state_repeat_gate"]["passed"],
        },
        "workload": {
            "prompt_token_id": int(args.prompt_token_id),
            "prompt_length": int(args.prompt_length),
            "decode_steps": int(args.decode_steps),
            "warmup_decode_steps": int(args.warmup_decode_steps),
            "repetitions": int(args.repetitions),
            "graph_replay": False,
        },
        "measurements": measurements,
        "summary": summary,
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--quality-artifact", type=Path, required=True)
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--warmup-decode-steps", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--decode-repack", action=argparse.BooleanOptionalAction, default=True)
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
    except (json.JSONDecodeError, OSError, PerformanceGateError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.json)
    print(json.dumps({"status": artifact["status"], "summary": artifact["summary"]}, indent=2))
    return 0 if artifact["status"] == "retained" else 1


if __name__ == "__main__":
    raise SystemExit(main())
