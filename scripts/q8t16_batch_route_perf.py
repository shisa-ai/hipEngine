#!/usr/bin/env python3
"""Counterbalanced current-route A/B for the gfx1151 Q8T16 batch rowtile.

Both routes share one resident model and the current package. The candidate
changes only the explicit all-rowtile diagnostic for physical c2/c4/c8 packed
graph decode; the current exact c8 pair-col8 owner remains active in both routes.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats
from scripts.execution_profile_gdn_calibration import CalibrationError
from scripts.execution_profile_gguf_batch_route_gate import (
    DEFAULT_GDN_MODE,
    DEFAULT_HISTORICAL_SOURCE,
    DEFAULT_MODEL,
    POLICY_CAPABILITY,
    POLICY_ENV,
    _rowtile_policy,
)
from scripts.gguf_packed_ar_bench import (
    CONFIGURATIONS,
    _prompt_rows,
    _run_sample,
)

KIND = "qwen36_gfx1151_q8t16_batch_route_counterbalanced_perf"
DEFAULT_QUALITY_ARTIFACT = Path(
    "benchmarks/results/2026-08-16-gfx1151-q8t16-batch-route-requalification.json"
)


def summarize_by_configuration(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Summarize strict/candidate medians and matched ratios per packed width."""

    result: dict[str, dict[str, Any]] = {}
    names = sorted({str(row["configuration"]) for row in runs})
    for name in names:
        config_runs = [row for row in runs if str(row["configuration"]) == name]
        samples = {
            label: [
                float(row["decode_tok_s"])
                for row in config_runs
                if str(row["label"]) == label
            ]
            for label in ("strict", "candidate")
        }
        if not samples["strict"] or len(samples["strict"]) != len(samples["candidate"]):
            raise ValueError(f"configuration {name} has unbalanced route samples")
        medians = {
            label: statistics.median(values) for label, values in samples.items()
        }
        pair_ids = sorted({int(row["pair"]) for row in config_runs})
        ratios: list[float] = []
        exact: list[bool] = []
        for pair_id in pair_ids:
            pair_rows = [row for row in config_runs if int(row["pair"]) == pair_id]
            by_label = {str(row["label"]): row for row in pair_rows}
            if set(by_label) != {"strict", "candidate"}:
                raise ValueError(f"configuration {name} pair {pair_id} is incomplete")
            ratios.append(
                float(by_label["candidate"]["decode_tok_s"])
                / float(by_label["strict"]["decode_tok_s"])
            )
            exact.append(
                list(by_label["candidate"]["trajectory_sha256"])
                == list(by_label["strict"]["trajectory_sha256"])
                if "trajectory_sha256" in by_label["candidate"]
                else True
            )
        result[name] = {
            "samples": samples,
            "medians": medians,
            "candidate_over_strict": medians["candidate"] / medians["strict"],
            "paired_ratios": ratios,
            "paired_median": statistics.median(ratios),
            "candidate_wins": sum(value > 1.0 for value in ratios),
            "trajectory_exact_by_pair": exact,
            "all_trajectories_exact": all(exact),
        }
    return result


def _run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model.is_file():
        raise CalibrationError(f"model does not exist: {args.model}")
    if not args.compiler_version_file.is_file():
        raise CalibrationError("a readable compiler-version file is required")
    if not args.require_cached_build:
        raise CalibrationError("performance capture requires --require-cached-build")
    if int(args.pairs) < 1 or int(args.decode_steps) <= 0:
        raise CalibrationError("pairs and decode steps must be positive")
    configurations = tuple(
        part.strip() for part in str(args.configurations).split(",") if part.strip()
    )
    if not configurations or any(
        name not in {"c2", "c4", "native_c8"} for name in configurations
    ):
        raise CalibrationError(
            "configurations must be a non-empty subset of c2,c4,native_c8"
        )

    os.environ["HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"] = "1"
    os.environ["HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE"] = "1"
    os.environ["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    os.environ["HIPENGINE_GGUF_GDN_PREFILL_MODE"] = str(args.gdn_mode)
    package = __import__(f"hipengine.kernels.{args.backend}", fromlist=[POLICY_CAPABILITY])
    if getattr(package, POLICY_CAPABILITY) is not False:
        raise CalibrationError(f"current package {POLICY_CAPABILITY} is not False")

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    compiler_version = args.compiler_version_file.read_text(encoding="utf-8")
    runtime = get_hip_runtime()
    initial_current = int(memory_stats()["current_allocated_bytes"])
    max_rows = max(CONFIGURATIONS[name].logical_rows for name in configurations)
    prompts = _prompt_rows(
        rows=max_rows,
        prompt_length=int(args.prompt_tokens),
        token_id=int(args.prompt_token_id),
    )
    runs: list[dict[str, Any]] = []
    resolved_backend = str(args.backend)
    target_arch = resolved_backend.removeprefix("hip_")
    stack = ExitStack()
    try:
        owner = stack.enter_context(
            Qwen35GGUFResidentSession(
                args.model,
                backend=str(args.backend),
                compiler_version=compiler_version,
                require_cached_build=True,
                max_sequence_length=(
                    int(args.prompt_tokens) + int(args.decode_steps) + 2
                ),
                use_wmma_prefill=True,
                use_gemv_decode=True,
            )
        )
        if owner.runner is None:
            raise CalibrationError("GGUF resident session closed during setup")
        resolved_backend = str(owner.runner.backend)
        target_arch = str(owner.runner.target_arch)
        sessions = [owner]
        while len(sessions) < max_rows:
            sessions.append(
                stack.enter_context(
                    Qwen35GGUFResidentSession(
                        args.model,
                        backend=str(args.backend),
                        runtime=owner.runtime,
                        shared_runner=owner.runner,
                        compiler_version=compiler_version,
                        require_cached_build=True,
                        max_sequence_length=(
                            int(args.prompt_tokens) + int(args.decode_steps) + 2
                        ),
                        use_wmma_prefill=True,
                        use_gemv_decode=True,
                    )
                )
            )

        def run_once(
            configuration: str,
            label: str,
            pair: int,
            *,
            measured: bool,
        ) -> dict[str, Any]:
            config = CONFIGURATIONS[configuration]
            with _rowtile_policy(label == "candidate"):
                sample = _run_sample(
                    config=config,
                    sessions=sessions,
                    prompts=prompts,
                    decode_steps=int(args.decode_steps),
                    measured=measured,
                    run_index=pair,
                )
            row = {
                "configuration": configuration,
                "label": label,
                "pair": pair,
                "measured": measured,
                "decode_tok_s": sample["throughput"]["decode_tok_s_aggregate"],
                "decode_seconds": sample["timings"]["decode_seconds"],
                "graph_capture_seconds": sample["timings"]["graph_capture_seconds"],
                "trajectory_sha256": [
                    value["sha256"] for value in sample["trajectory_fingerprints"]
                ],
                "route": sample["route"],
                "graph_manifests": sample["graph_manifests"],
                "flush_results": sample["flush_results"],
                "memory": sample["memory"],
            }
            print(configuration, label, pair, row["decode_tok_s"], flush=True)
            return row

        for configuration in configurations:
            run_once(configuration, "strict", 0, measured=False)
            run_once(configuration, "candidate", 0, measured=False)
            for pair in range(1, int(args.pairs) + 1):
                order = (
                    ("strict", "candidate")
                    if pair % 2
                    else ("candidate", "strict")
                )
                for label in order:
                    runs.append(
                        run_once(configuration, label, pair, measured=True)
                    )
    finally:
        stack.close()
    runtime.device_synchronize()
    final_current = int(memory_stats()["current_allocated_bytes"])

    summary = summarize_by_configuration(runs)
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
            "HIPENGINE_GGUF_GDN_PREFILL_MODE": os.environ.get(
                "HIPENGINE_GGUF_GDN_PREFILL_MODE"
            ),
            POLICY_ENV: os.environ.get(POLICY_ENV),
        },
        build_profile="q8t16_batch_route_counterbalanced_perf",
        timing_protocol="persistent_counterbalanced_packed_graph_wall_v1",
        warmups=2 * len(configurations),
        repetitions=int(args.pairs),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    complete_protocol = bool(
        int(args.pairs) >= 7
        and int(args.prompt_tokens) == 512
        and int(args.decode_steps) == 128
    )
    measurement_valid = bool(
        complete_protocol
        and not provenance.get("dirty")
        and initial_current == final_current
        and all(value["all_trajectories_exact"] for value in summary.values())
    )
    return {
        "schema_version": 1,
        "kind": KIND,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if measurement_valid else "invalid_or_screen_only",
        "measurement_valid": measurement_valid,
        "performance_claim": measurement_valid,
        "hardware": {"device": "AMD Radeon 8060S Graphics", "arch": "gfx1151"},
        "model": str(args.model.resolve()),
        "route": {
            "policy_capability": POLICY_CAPABILITY,
            "package_value_verified": False,
            "candidate_environment": {POLICY_ENV: "1"},
            "current_c8_pair_col8_preserved_in_both_routes": True,
        },
        "protocol": {
            "persistent_session": True,
            "configurations": list(configurations),
            "warmup_runs_per_route_and_configuration": 1,
            "counterbalanced_pairs": int(args.pairs),
            "prompt_tokens": int(args.prompt_tokens),
            "decode_steps": int(args.decode_steps),
            "prompt_token_id": int(args.prompt_token_id),
            "graph_replay": True,
            "strict_gdn_mode": str(args.gdn_mode),
            "timing": "packed graph replay wall excluding graph capture",
        },
        "runs": runs,
        "summary": summary,
        "quality_gate_artifact": str(args.quality_artifact),
        "historical_source": str(args.historical_source),
        "memory": {
            "tracked_current_before_bytes": initial_current,
            "tracked_current_after_bytes": final_current,
            "teardown_exact": initial_current == final_current,
        },
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--configurations", default="c2,c4,native_c8")
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--gdn-mode", default=DEFAULT_GDN_MODE)
    parser.add_argument("--historical-source", type=Path, default=DEFAULT_HISTORICAL_SOURCE)
    parser.add_argument("--quality-artifact", type=Path, default=DEFAULT_QUALITY_ARTIFACT)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    command = [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *raw_argv]
    try:
        artifact = _run(args, command=command)
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(json.dumps(artifact["summary"], indent=2))
    return 0 if artifact["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
