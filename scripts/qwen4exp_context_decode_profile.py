#!/usr/bin/env python3
"""Profile exact Qwen4Exp decode transitions at declared live-context counts.

The harness pre-fills one canonical token array to ``live_count - 1``, captures
request state, and restores that snapshot before every measured transition.
Each marker window therefore executes the same compact-output decode step at the
same live count. State/output repeatability and lifecycle closure are binding;
profiled wall values remain diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE,
    _git_metadata,
    _host_metadata,
    sha256_path,
    token_ids_sha256,
)
from scripts.qwen4exp_layer2_profile_gate import _make_generator, _state_summary
from scripts.qwen4exp_profile_gap import RoleMarkers, Roctx, RuntimeCensus, _graph_snapshot


def _select_case(fixture: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    matches = [row for row in fixture.get("cases", ()) if str(row.get("id")) == case_id]
    if len(matches) != 1:
        raise ValueError(f"fixture must contain exactly one case {case_id!r}")
    return matches[0]


def _prefix_for_live_count(
    case: Mapping[str, Any], live_count: int
) -> tuple[int, ...]:
    count = int(live_count)
    if count < 2:
        raise ValueError("live count must be at least 2")
    values = tuple(int(value) for value in case["prompt_token_ids"])
    prefix_count = count - 1
    if prefix_count > len(values):
        raise ValueError(
            f"live count {count} exceeds fixture capacity {len(values) + 1}"
        )
    return values[:prefix_count]


def _path_is_tracked(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(ROOT)
    except ValueError:
        return False
    return subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).returncode == 0


def _repeat_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("repeat summary needs at least one row")
    tokens = [int(row["token_id"]) for row in rows]
    states = [str(row["state_sha256"]) for row in rows]
    walls = [float(row["wall_seconds"]) for row in rows]
    token_exact = len(set(tokens)) == 1
    state_exact = len(set(states)) == 1
    return {
        "runs": len(rows),
        "token_exact": token_exact,
        "state_exact": state_exact,
        "passed": bool(token_exact and state_exact),
        "wall_seconds": {
            "mean": statistics.mean(walls),
            "median": statistics.median(walls),
            "min": min(walls),
            "max": max(walls),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", default="code-p4096")
    parser.add_argument(
        "--live-count",
        type=int,
        nargs="+",
        default=[2051, 2052, 4097],
        help="Exact attention live counts to profile",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--role-markers", action="store_true")
    parser.add_argument("--hip-arch", default="gfx1151")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model_root.is_dir():
        raise ValueError(f"model root does not exist: {args.model_root}")
    if not args.fixture.is_file():
        raise ValueError(f"fixture does not exist: {args.fixture}")
    if int(args.repetitions) < 3:
        raise ValueError("context decode profiling requires at least three repeats")
    if args.role_markers and not args.profile:
        raise ValueError("--role-markers requires --profile")
    live_counts = tuple(dict.fromkeys(int(value) for value in args.live_count))
    if not live_counts:
        raise ValueError("at least one live count is required")

    fixture = json.loads(args.fixture.read_text())
    case = _select_case(fixture, str(args.case_id))
    prefixes = {
        live_count: _prefix_for_live_count(case, live_count)
        for live_count in live_counts
    }

    os.environ.setdefault("HIPENGINE_HIP_ARCH", str(args.hip_arch))
    if args.compiler_version_file is not None:
        os.environ.setdefault(
            "HIPENGINE_COMPILER_VERSION_FILE", str(args.compiler_version_file)
        )
    if args.require_cached_build:
        os.environ.setdefault("HIPENGINE_REQUIRE_CACHED_BUILD", "1")

    from hipengine.core.memory import (
        copy_host_to_device,
        host_array_ptr,
        memory_stats,
        reset_memory_stats,
    )
    from hipengine.generation.qwen4_exp_profiles import (
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    import hipengine.runtime.qwen4_exp_runner as runner_module

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    reset_memory_stats()
    max_sequence_length = max(live_counts) + 1
    factory_args = argparse.Namespace(
        model_root=args.model_root,
        max_sequence_length=max_sequence_length,
        prefill_chunk_size=int(args.prefill_chunk_size),
    )
    generator, resolved, _index = _make_generator(factory_args, "production")
    roctx = Roctx() if args.profile else None
    contexts: list[dict[str, Any]] = []
    memory_after_warmup: dict[str, Any] | None = None
    memory_after_measurement: dict[str, Any] | None = None
    try:
        runner = generator.runner
        if not runner.index_states:
            raise RuntimeError("Qwen4Exp context profiler requires QSA index state")
        dense_limit = int(runner.index_states[0].dense_equivalent_limit)
        for live_count in live_counts:
            prefix = prefixes[live_count]
            root = runner.prefill(
                prefix,
                capture_logits=False,
                capture_target_hidden=False,
            )
            root_token = int(root.token_id)
            snapshot = runner.snapshot()
            runner.restore(snapshot)
            warm_token_host = np.asarray([root_token], dtype=np.int64)
            copy_host_to_device(
                runner.token_id_buffer,
                host_array_ptr(warm_token_host),
                warm_token_host.nbytes,
                runtime=runner.runtime,
            )
            warm_result = runner.step(
                root_token,
                capture_logits=False,
                capture_target_hidden=False,
                token_id_resident=True,
            )
            warm_state = _state_summary(runner)
            if memory_after_warmup is None:
                runner.runtime.device_synchronize()
                memory_after_warmup = memory_stats()
            runner.restore(snapshot)
            graph_before = _graph_snapshot(runner.moe_graph_cache, runner.runtime)
            rows: list[dict[str, Any]] = []
            for repetition in range(int(args.repetitions)):
                runner.restore(snapshot)
                token_host = np.asarray([root_token], dtype=np.int64)
                copy_host_to_device(
                    runner.token_id_buffer,
                    host_array_ptr(token_host),
                    token_host.nbytes,
                    runtime=runner.runtime,
                )
                census = RuntimeCensus(runner.runtime)
                census.install()
                roles = (
                    RoleMarkers(runner_module, roctx)
                    if args.role_markers and roctx is not None
                    else None
                )
                if roles is not None:
                    roles.install()
                marker = f"qwen4exp_decode_live{live_count}_rep{repetition}"
                try:
                    if roctx is not None:
                        roctx.push(marker)
                    if roles is not None:
                        roctx.push("qwen4exp_role:decode_boundary")
                    started = time.perf_counter()
                    try:
                        result = runner.step(
                            root_token,
                            capture_logits=False,
                            capture_target_hidden=False,
                            token_id_resident=True,
                        )
                    finally:
                        if roles is not None:
                            roctx.pop()
                    wall_seconds = time.perf_counter() - started
                    if roctx is not None:
                        roctx.pop()
                finally:
                    if roles is not None:
                        roles.close()
                    census.close()
                state = _state_summary(runner)
                rows.append(
                    {
                        "repetition": repetition,
                        "marker": marker,
                        "token_id": int(result.token_id),
                        "state_sha256": str(state["state_sha256"]),
                        "layout_sha256": str(state["layout_sha256"]),
                        "state_finite": bool(state["finite"]),
                        "wall_seconds": wall_seconds,
                        "runtime_census": census.snapshot(),
                    }
                )
            summary = _repeat_summary(rows)
            summary["finite"] = all(bool(row["state_finite"]) for row in rows)
            summary["passed"] = bool(summary["passed"] and summary["finite"])
            contexts.append(
                {
                    "live_count": live_count,
                    "prefix_tokens": len(prefix),
                    "prefix_token_ids_sha256": token_ids_sha256(prefix),
                    "root_token_id": root_token,
                    "warmup": {
                        "token_id": int(warm_result.token_id),
                        "state_sha256": str(warm_state["state_sha256"]),
                        "finite": bool(warm_state["finite"]),
                    },
                    "qsa_path": "dense_equivalent" if live_count <= dense_limit else "indexed_sparse",
                    "dense_equivalent_limit": dense_limit,
                    "graph_before": graph_before,
                    "graph_after": _graph_snapshot(
                        runner.moe_graph_cache, runner.runtime
                    ),
                    "repeats": rows,
                    "summary": summary,
                }
            )
            print(
                f"live_count={live_count} path={contexts[-1]['qsa_path']} "
                f"token_exact={summary['token_exact']} "
                f"state_exact={summary['state_exact']} "
                f"median_ms={summary['wall_seconds']['median'] * 1e3:.3f}",
                flush=True,
            )
        runner.runtime.device_synchronize()
        memory_after_measurement = memory_stats()
    finally:
        generator.close()
    after_close = memory_stats()
    source = _git_metadata(ROOT)
    script_tracked = _path_is_tracked(Path(__file__))
    if source is not None:
        source["script_tracked"] = script_tracked
    passed = bool(
        source
        and source["tracked_clean"]
        and script_tracked
        and all(row["summary"]["passed"] for row in contexts)
        and memory_after_warmup is not None
        and memory_after_measurement is not None
        and int(memory_after_measurement["active_allocations"])
        == int(memory_after_warmup["active_allocations"])
        and int(memory_after_measurement["current_allocated_bytes"])
        == int(memory_after_warmup["current_allocated_bytes"])
        and int(after_close["active_allocations"]) == 0
        and int(after_close["current_allocated_bytes"]) == 0
    )
    return {
        "schema": 1,
        "kind": "qwen4exp_context_decode_profile",
        "status": "passed" if passed else "failed",
        "measurement_class": "diagnostic_profiled_not_performance",
        "command": list(command),
        "source": source,
        "host": _host_metadata(),
        "model_root": str(args.model_root),
        "fixture": str(args.fixture),
        "fixture_sha256": sha256_path(args.fixture),
        "case": {
            "id": str(case["id"]),
            "category": str(case["category"]),
            "prompt_tokens": int(case["prompt_tokens"]),
            "prompt_token_ids_sha256": str(case["prompt_token_ids_sha256"]),
        },
        "profile": {
            "execution_profile": "production",
            "manifest_sha256": resolved.manifest_sha256,
            "strict_manifest_sha256": resolved.strict_manifest_sha256,
            "fell_back_to_strict": resolved.fell_back_to_strict,
        },
        "protocol": {
            "live_counts": list(live_counts),
            "repetitions": int(args.repetitions),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "compact_output": True,
            "snapshot_restore_between_repeats": True,
            "warmup_transitions_per_context": 1,
            "resident_input_token": True,
            "profile": bool(args.profile),
            "role_markers": bool(args.role_markers),
        },
        "contexts": contexts,
        "lifecycle": {
            "after_warmup": memory_after_warmup,
            "after_measurement": memory_after_measurement,
            "steady_allocation_growth": (
                int(memory_after_measurement["active_allocations"])
                - int(memory_after_warmup["active_allocations"])
                if memory_after_warmup is not None
                and memory_after_measurement is not None
                else None
            ),
            "steady_allocated_byte_growth": (
                int(memory_after_measurement["current_allocated_bytes"])
                - int(memory_after_warmup["current_allocated_bytes"])
                if memory_after_warmup is not None
                and memory_after_measurement is not None
                else None
            ),
            "after_close": after_close,
            "passed": bool(
                memory_after_warmup is not None
                and memory_after_measurement is not None
                and int(memory_after_measurement["active_allocations"])
                == int(memory_after_warmup["active_allocations"])
                and int(memory_after_measurement["current_allocated_bytes"])
                == int(memory_after_warmup["current_allocated_bytes"])
                and int(after_close["active_allocations"]) == 0
                and int(after_close["current_allocated_bytes"]) == 0
            ),
        },
        "decision": {
            "passed": passed,
            "next": "analyze marker-scoped kernel/API/role owners",
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args, command=[str(Path(sys.argv[0]).name), *sys.argv[1:]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
