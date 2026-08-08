#!/usr/bin/env python3
"""Pair HIP-graph and PM4 on real packed GGUF c2/c4/c8 decode graphs.

The harness keeps one loaded model and one eight-session pool, alternates
transport order by round, resets and prefills before every graph generation,
and measures replay, capture-inclusive, and request-inclusive wall separately.
It also runs the complete natural category/heldout prompt suite through packed
power-of-two groups and requires exact HIP/PM4 token trajectories. Logical
c3/c5/c6/c7 are not mislabeled: production lowers those occupancies through
packed eager buckets, so submission transport does not participate.
"""

from __future__ import annotations

import argparse
import copy
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

from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402
from hipengine.loading import load_gguf_index  # noqa: E402
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession  # noqa: E402
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer  # noqa: E402
from scripts.gguf_packed_ar_bench import (  # noqa: E402
    CONFIGURATIONS,
    PackedARConfiguration,
    _memory_snapshot,
    _prompt_rows,
    _read_compiler_version,
    _run_sample,
    _temporary_environment,
)
from scripts.pm4_promotion_gate import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_SUITES,
    _load_prompt_cases,
)

_TRANSPORTS = ("hipgraph", "pm4")
_DEFAULT_CONFIGURATIONS = ("c2", "c4", "native_c8")
_EXACT_ENV = {
    "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1",
    "HIPENGINE_GGUF_GDN_PREFILL_MODE": "exact",
}
_DEFAULT_MEMORY_RECOVERY_TOLERANCE = 64 * 1024 * 1024


def _rotation(values: Sequence[str], index: int) -> tuple[str, ...]:
    rows = tuple(str(value) for value in values)
    if not rows:
        raise ValueError("transport rotation must not be empty")
    offset = int(index) % len(rows)
    return (*rows[offset:], *rows[:offset])


def _sample_fingerprints(sample: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row["sha256"]) for row in sample["trajectory_fingerprints"])


def _validate_sample_transport(
    sample: Mapping[str, Any],
    *,
    expected: str,
    steps: int,
) -> list[str]:
    blockers: list[str] = []
    if sample.get("passed") is not True:
        blockers.append("packed sample did not pass its graph/trajectory/flush gate")
    manifests = sample.get("graph_manifests")
    if not isinstance(manifests, list) or not manifests:
        return [*blockers, "packed sample has no graph manifests"]
    for index, manifest in enumerate(manifests):
        graph = manifest.get("graph", {}) if isinstance(manifest, Mapping) else {}
        proof = graph.get("transport", {}) if isinstance(graph, Mapping) else {}
        if proof.get("transport") != expected:
            blockers.append(f"graph {index} used {proof.get('transport')!r}, expected {expected!r}")
            continue
        if int(proof.get("launches", -1)) != int(steps):
            blockers.append(f"graph {index} launch count did not match decode steps")
        if int(proof.get("native_fallbacks", -1)) != 0:
            blockers.append(f"graph {index} recorded a native fallback")
        if expected != "pm4":
            continue
        if proof.get("stateful_registers") is not True:
            blockers.append(f"graph {index} did not use the canonical stateful encoder")
        if proof.get("local_cache_dependencies") is not True:
            blockers.append(
                f"graph {index} did not use the canonical local-cache dependency encoder"
            )
        context = proof.get("context", {})
        executable = proof.get("executable", {})
        if not isinstance(context, Mapping) or int(context.get("unretired_submissions", -1)) != 0:
            blockers.append(f"graph {index} retained an unretired PM4 submission")
        if not isinstance(context, Mapping) or int(context.get("callback_status", -1)) != 0:
            blockers.append(f"graph {index} recorded a PM4 callback error")
        if not isinstance(executable, Mapping) or executable.get("retired") is not True:
            blockers.append(f"graph {index} PM4 executable did not retire")
    return blockers


def _paired_transport_summary(
    pairs: Sequence[Mapping[str, Any]],
    *,
    logical_rows: int,
    decode_steps: int,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("paired transport summary requires at least one round")
    rows = int(logical_rows)
    steps = int(decode_steps)
    if rows <= 0 or steps <= 0:
        raise ValueError("logical_rows and decode_steps must be positive")

    result: dict[str, Any] = {}
    medians: dict[str, dict[str, float]] = {}
    generated = rows * steps
    for mode in _TRANSPORTS:
        samples = [pair[mode] for pair in pairs]
        replay = [float(sample["timings"]["decode_seconds"]) for sample in samples]
        capture = [float(sample["timings"]["graph_capture_seconds"]) for sample in samples]
        prefill = [float(sample["timings"]["prefill_seconds"]) for sample in samples]
        capture_inclusive = [a + b for a, b in zip(capture, replay, strict=True)]
        request_inclusive = [
            a + b + c for a, b, c in zip(prefill, capture, replay, strict=True)
        ]
        medians[mode] = {
            "replay": float(statistics.median(replay)),
            "capture": float(statistics.median(capture)),
            "prefill": float(statistics.median(prefill)),
            "capture_inclusive": float(statistics.median(capture_inclusive)),
            "request_inclusive": float(statistics.median(request_inclusive)),
        }
        value = medians[mode]
        result[mode] = {
            "replay_seconds_samples": replay,
            "capture_seconds_samples": capture,
            "replay_ms_per_step": value["replay"] * 1000.0 / steps,
            "aggregate_replay_tok_s": generated / value["replay"],
            "capture_ms": value["capture"] * 1000.0,
            "capture_inclusive_ms_per_step": value["capture_inclusive"] * 1000.0 / steps,
            "capture_inclusive_output_tok_s": generated / value["capture_inclusive"],
            "prefill_ms": value["prefill"] * 1000.0,
            "request_inclusive_ms": value["request_inclusive"] * 1000.0,
            "request_inclusive_output_tok_s": generated / value["request_inclusive"],
        }

    hip = medians["hipgraph"]
    pm4 = medians["pm4"]
    result.update(
        {
            "paired_rounds": len(pairs),
            "paired_replay_wins": sum(
                float(pair["pm4"]["timings"]["decode_seconds"])
                < float(pair["hipgraph"]["timings"]["decode_seconds"])
                for pair in pairs
            ),
            "replay_wall_delta_percent": (pm4["replay"] / hip["replay"] - 1.0) * 100.0,
            "capture_wall_delta_percent": (pm4["capture"] / hip["capture"] - 1.0) * 100.0,
            "capture_inclusive_wall_delta_percent": (
                pm4["capture_inclusive"] / hip["capture_inclusive"] - 1.0
            )
            * 100.0,
            "request_inclusive_wall_delta_percent": (
                pm4["request_inclusive"] / hip["request_inclusive"] - 1.0
            )
            * 100.0,
            "trajectories_exact": all(
                _sample_fingerprints(pair["hipgraph"])
                == _sample_fingerprints(pair["pm4"])
                for pair in pairs
            ),
            "transport_repeatable": all(
                all(
                    _sample_fingerprints(pair[mode])
                    == _sample_fingerprints(pairs[0][mode])
                    for pair in pairs[1:]
                )
                for mode in _TRANSPORTS
            ),
        }
    )
    return result


def _compact_transport_proof(proof: Mapping[str, Any]) -> dict[str, Any]:
    context = proof.get("context", {})
    executable = proof.get("executable", {})
    return {
        "transport": proof.get("transport"),
        "graph_fingerprint": proof.get("graph_fingerprint"),
        "node_count": proof.get("node_count"),
        "launches": proof.get("launches"),
        "native_fallbacks": proof.get("native_fallbacks"),
        "stateful_registers": proof.get("stateful_registers"),
        "local_cache_dependencies": proof.get("local_cache_dependencies"),
        "context": {
            key: context.get(key)
            for key in (
                "queue_id",
                "queue_size",
                "generation",
                "submissions",
                "children",
                "unretired_submissions",
                "callback_status",
                "usable",
            )
        }
        if isinstance(context, Mapping)
        else {},
        "executable": {
            key: executable.get(key)
            for key in (
                "nodes",
                "modules",
                "pm4_dwords",
                "pm4_submissions",
                "retired",
                "usable",
            )
        }
        if isinstance(executable, Mapping)
        else {},
    }


def _compact_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "configuration": sample["configuration"],
        "run_index": sample["run_index"],
        "measured": sample["measured"],
        "passed": sample["passed"],
        "route": copy.deepcopy(sample["route"]),
        "timings": copy.deepcopy(sample["timings"]),
        "throughput": copy.deepcopy(sample["throughput"]),
        "trajectory_fingerprints": copy.deepcopy(sample["trajectory_fingerprints"]),
        "flush_results": copy.deepcopy(sample["flush_results"]),
        "transport_proofs": [],
    }
    for manifest in sample["graph_manifests"]:
        graph = manifest["graph"]
        value["transport_proofs"].append(
            {
                "bucket_key_sha256": graph["bucket_key"]["key_sha256"],
                "physical_rows": graph["bucket_key"]["physical_rows"],
                **_compact_transport_proof(graph["transport"]),
            }
        )
    return value


def _context_teardown_ok(proofs: Mapping[str, Mapping[str, Any]]) -> bool:
    for row in proofs.values():
        before = row.get("before", {})
        after = row.get("after", {})
        native_before = before.get("native", before)
        if (
            int(before.get("children", -1)) != 0
            or not isinstance(native_before, Mapping)
            or int(native_before.get("unretired_submissions", -1)) != 0
            or int(native_before.get("callback_status", -1)) != 0
            or after.get("closed") is not True
        ):
            return False
    return True


def _configuration_names(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not names:
        raise ValueError("configurations must not be empty")
    unknown = sorted(set(names) - set(_DEFAULT_CONFIGURATIONS))
    if unknown:
        raise ValueError(f"unsupported PM4 packed configurations: {unknown!r}")
    if len(set(names)) != len(names):
        raise ValueError("configurations must be unique")
    return names


def _natural_groups(case_count: int) -> tuple[tuple[int, int], ...]:
    groups: list[tuple[int, int]] = []
    start = 0
    remaining = int(case_count)
    while remaining:
        width = next(value for value in (8, 4, 2, 1) if value <= remaining)
        groups.append((start, width))
        start += width
        remaining -= width
    return tuple(groups)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.prompt_length) < 4 or int(args.decode_steps) <= 0:
        raise ValueError("prompt length must be >=4 and decode steps must be positive")
    if int(args.prompt_length) + int(args.decode_steps) >= 1024:
        raise ValueError("packed graph benchmark requires context <1024")
    if int(args.warmups) < 0 or int(args.repetitions) <= 0:
        raise ValueError("warmups must be non-negative and repetitions positive")
    names = _configuration_names(args.configurations)
    model = Path(args.model).expanduser().resolve(strict=True)
    compiler_version = _read_compiler_version(args.compiler_version_file)
    if args.require_cached and compiler_version is None:
        raise ValueError("--require-cached requires --compiler-version-file")

    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(model))
    natural_cases = _load_prompt_cases(args.suite_files, tokenizer)
    if not natural_cases:
        raise ValueError("natural prompt suites contain no cases")
    if any(len(case.token_ids) + int(args.natural_steps) >= 1024 for case in natural_cases):
        raise ValueError("natural prompt suite exceeds packed graph context limit")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import reset_memory_stats

    runtime = get_hip_runtime()
    reset_memory_stats()
    free_before_load, total_bytes = runtime.mem_get_info()
    memory: dict[str, Any] = {"before_load": _memory_snapshot("before_load", runtime)}
    max_rows = max(8, *(CONFIGURATIONS[name].logical_rows for name in names))
    max_sequence_length = max(
        int(args.prompt_length) + int(args.decode_steps) + 2,
        max(len(case.token_ids) for case in natural_cases) + int(args.natural_steps) + 2,
    )
    prompts = _prompt_rows(
        rows=max_rows,
        prompt_length=int(args.prompt_length),
        token_id=int(args.prompt_token_id),
    )
    warmup_samples: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {mode: [] for mode in _TRANSPORTS} for name in names
    }
    paired_runs: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    context_teardown: dict[str, dict[str, dict[str, Any]]] = {}
    natural_rows: list[dict[str, Any]] = []
    pci_bdf = runtime.device_pci_bus_id()
    target_arch = str(args.backend).removeprefix("hip_")

    stack = ExitStack()
    try:
        with _temporary_environment(_EXACT_ENV):
            owner = stack.enter_context(
                Qwen35GGUFResidentSession(
                    model,
                    backend=str(args.backend),
                    compiler_version=compiler_version,
                    require_cached_build=bool(args.require_cached),
                    max_sequence_length=max_sequence_length,
                    use_wmma_prefill=True,
                    use_gemv_decode=True,
                )
            )
            if owner.runner is None:
                raise RuntimeError("packed PM4 benchmark owner is closed")
            target_arch = str(owner.runner.target_arch)
            sessions = [owner]
            while len(sessions) < max_rows:
                sessions.append(
                    stack.enter_context(
                        Qwen35GGUFResidentSession(
                            model,
                            backend=str(args.backend),
                            runtime=owner.runtime,
                            shared_runner=owner.runner,
                            compiler_version=compiler_version,
                            require_cached_build=bool(args.require_cached),
                            max_sequence_length=max_sequence_length,
                            use_wmma_prefill=True,
                            use_gemv_decode=True,
                        )
                    )
                )
            memory["after_load"] = _memory_snapshot("after_load", runtime)

            for config_index, name in enumerate(names):
                config = CONFIGURATIONS[name]
                for warmup_index in range(int(args.warmups)):
                    for mode in _rotation(_TRANSPORTS, warmup_index + config_index):
                        sample = _run_sample(
                            config=config,
                            sessions=sessions,
                            prompts=prompts,
                            decode_steps=int(args.decode_steps),
                            measured=False,
                            run_index=warmup_index,
                            submission_transport=mode,
                        )
                        blockers = _validate_sample_transport(
                            sample, expected=mode, steps=int(args.decode_steps)
                        )
                        if blockers:
                            raise RuntimeError(f"{name}/{mode} warmup failed: {blockers}")
                        warmup_samples[name][mode].append(_compact_sample(sample))

            runtime.device_synchronize()
            free_before_measured, total_before_measured = runtime.mem_get_info()
            memory["before_measured"] = _memory_snapshot("before_measured", runtime)

            for config_index, name in enumerate(names):
                config = CONFIGURATIONS[name]
                for round_index in range(int(args.repetitions)):
                    pair: dict[str, Any] = {
                        "round": round_index,
                        "order": list(_rotation(_TRANSPORTS, round_index + config_index)),
                    }
                    for mode in pair["order"]:
                        sample = _run_sample(
                            config=config,
                            sessions=sessions,
                            prompts=prompts,
                            decode_steps=int(args.decode_steps),
                            measured=True,
                            run_index=round_index,
                            submission_transport=mode,
                        )
                        blockers = _validate_sample_transport(
                            sample, expected=mode, steps=int(args.decode_steps)
                        )
                        if blockers:
                            raise RuntimeError(f"{name}/{mode} measured run failed: {blockers}")
                        pair[mode] = sample
                        print(
                            f"{name} round={round_index} {mode}: "
                            f"{sample['timings']['decode_seconds'] * 1000.0 / int(args.decode_steps):.6f} "
                            "ms/step",
                            file=sys.stderr,
                            flush=True,
                        )
                    paired_runs[name].append(pair)

            for group_index, (start, width) in enumerate(_natural_groups(len(natural_cases))):
                cases = natural_cases[start : start + width]
                config = PackedARConfiguration(
                    name=f"natural_c{width}_{group_index}",
                    logical_rows=width,
                    native_group_width=width,
                    native_group_count=1,
                    execution_class="direct_native_group",
                )
                pair: dict[str, Any] = {
                    "group_index": group_index,
                    "physical_rows": width,
                    "case_ids": [case.name for case in cases],
                    "categories": [case.category for case in cases],
                }
                for mode in _rotation(_TRANSPORTS, group_index):
                    sample = _run_sample(
                        config=config,
                        sessions=sessions,
                        prompts=[case.token_ids for case in cases],
                        decode_steps=int(args.natural_steps),
                        measured=False,
                        run_index=group_index,
                        submission_transport=mode,
                    )
                    blockers = _validate_sample_transport(
                        sample, expected=mode, steps=int(args.natural_steps)
                    )
                    if blockers:
                        raise RuntimeError(f"natural group {group_index}/{mode} failed: {blockers}")
                    pair[mode] = sample
                pair["trajectories_exact"] = (
                    _sample_fingerprints(pair["hipgraph"])
                    == _sample_fingerprints(pair["pm4"])
                )
                natural_rows.append(pair)

            runtime.device_synchronize()
            memory["before_context_close"] = _memory_snapshot(
                "before_context_close", runtime
            )
            for index, session in enumerate(sessions):
                proofs = session.close_decode_graph_submission_contexts()
                if proofs:
                    context_teardown[f"session_{index}"] = proofs
            runtime.device_synchronize()
            free_after_contexts, total_after_contexts = runtime.mem_get_info()
            memory["after_context_close"] = _memory_snapshot("after_context_close", runtime)
    finally:
        stack.close()
        runtime.device_synchronize()
        free_after_close, total_after_close = runtime.mem_get_info()
        memory["after_close"] = _memory_snapshot("after_close", runtime)

    compact_pairs = {
        name: [
            {
                "round": pair["round"],
                "order": pair["order"],
                "hipgraph": _compact_sample(pair["hipgraph"]),
                "pm4": _compact_sample(pair["pm4"]),
            }
            for pair in rows
        ]
        for name, rows in paired_runs.items()
    }
    summaries = {
        name: _paired_transport_summary(
            rows,
            logical_rows=CONFIGURATIONS[name].logical_rows,
            decode_steps=int(args.decode_steps),
        )
        for name, rows in paired_runs.items()
    }
    transport_exact = all(
        summary["trajectories_exact"] and summary["transport_repeatable"]
        for summary in summaries.values()
    )
    natural_exact = bool(natural_rows) and all(row["trajectories_exact"] for row in natural_rows)
    category_counts: dict[str, int] = {}
    for case in natural_cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1
    context_passed = bool(context_teardown) and all(
        _context_teardown_ok(proofs) for proofs in context_teardown.values()
    )
    measured_memory_delta = int(free_before_measured) - int(free_after_contexts)
    memory_recovered = bool(
        total_before_measured == total_after_contexts
        and measured_memory_delta <= int(args.memory_recovery_tolerance_bytes)
    )
    performance_non_regressive = all(
        summary["replay_wall_delta_percent"] <= 0.0
        and summary["paired_replay_wins"] > summary["paired_rounds"] // 2
        for summary in summaries.values()
    )
    passed = bool(
        transport_exact
        and natural_exact
        and context_passed
        and memory_recovered
        and all(
            _validate_sample_transport(pair[mode], expected=mode, steps=int(args.decode_steps))
            == []
            for rows in paired_runs.values()
            for pair in rows
            for mode in _TRANSPORTS
        )
    )

    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=str(args.backend),
        target_arch=target_arch,
        model_path=model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=[str(part) for part in sys.argv],
        environment={
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
            "HIPENGINE_HIP_ARCH": target_arch,
        },
        build_profile="gfx1100_pm4_packed_c2_c4_c8",
        timing_protocol=(
            "one model/eight-session pool; reset+prefill+fresh graph per transport; "
            "alternating same-session order; exact natural-suite trajectories; "
            "replay/capture/request wall reported separately"
        ),
        warmups=int(args.warmups),
        repetitions=int(args.repetitions),
        profiler={"used": False, "reason": "host-wall paired transport comparison"},
        hipcc_version=compiler_version,
    )
    return {
        "schema_version": 1,
        "kind": "gfx1100_pm4_packed_graph_promotion_benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if passed and performance_non_regressive else "measured_blocked",
        "passed": passed,
        "performance_non_regressive": performance_non_regressive,
        "performance_claim": bool(passed and performance_non_regressive),
        "hardware": {
            "backend": str(args.backend),
            "gfx_arch": target_arch,
            "pci_bdf": pci_bdf,
        },
        "workload": {
            "configurations": list(names),
            "physical_widths": [CONFIGURATIONS[name].native_group_width for name in names],
            "logical_c3_c5_c6_c7_transport_path": "packed_eager_unaffected",
            "prompt_token_id": int(args.prompt_token_id),
            "prompt_length": int(args.prompt_length),
            "decode_steps": int(args.decode_steps),
            "warmups": int(args.warmups),
            "repetitions": int(args.repetitions),
            "natural_steps": int(args.natural_steps),
            "natural_cases": len(natural_cases),
            "suite_files": [str(Path(path).resolve()) for path in args.suite_files],
        },
        "summaries": summaries,
        "correctness": {
            "paired_performance_trajectories_exact": transport_exact,
            "natural_suite_trajectories_exact": natural_exact,
            "natural_case_count": len(natural_cases),
            "category_counts": category_counts,
            "natural_groups": [
                {
                    "group_index": row["group_index"],
                    "physical_rows": row["physical_rows"],
                    "case_ids": row["case_ids"],
                    "categories": row["categories"],
                    "trajectories_exact": row["trajectories_exact"],
                    "hipgraph": _compact_sample(row["hipgraph"]),
                    "pm4": _compact_sample(row["pm4"]),
                }
                for row in natural_rows
            ],
        },
        "warmup_samples": warmup_samples,
        "paired_runs": compact_pairs,
        "context_teardown": context_teardown,
        "context_lifecycle_passed": context_passed,
        "memory": {
            "snapshots": memory,
            "free_before_load": int(free_before_load),
            "free_after_close": int(free_after_close),
            "total_bytes": int(total_bytes),
            "total_after_close": int(total_after_close),
            "measured_free_delta_bytes": measured_memory_delta,
            "recovery_tolerance_bytes": int(args.memory_recovery_tolerance_bytes),
            "recovered": memory_recovered,
        },
        "provenance": provenance,
        "notes": [
            "c1 whole-step PM4 is covered by the separate promotion matrix/graph benchmark.",
            "Logical c3/c5/c6/c7 use packed eager buckets, so PM4 is neither claimed nor measured for them.",
            "No submit-plus-queue-recreate lifecycle arm is performed.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("hip_gfx1100",), default="hip_gfx1100")
    parser.add_argument("--configurations", default=",".join(_DEFAULT_CONFIGURATIONS))
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--suite-files", type=Path, nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--natural-steps", type=int, default=3)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument(
        "--memory-recovery-tolerance-bytes",
        type=int,
        default=_DEFAULT_MEMORY_RECOVERY_TOLERANCE,
    )
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
