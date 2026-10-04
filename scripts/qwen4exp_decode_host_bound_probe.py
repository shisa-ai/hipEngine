"""Separate the host enqueue cost of a decode step from the device time.

The canonical AR bench (`scripts/qwen4exp_canonical_ar_bench.py`) measures decode
as 128 `runner.step()` calls in one tight Python loop with a single
`device_synchronize()` at the end. That number is `max(host_enqueue, device)` per
step, and it cannot say which one it measured. This probe measures both
separately, for one canonical case, on the production profile.

`device_drain_ms` is the time from "host finished enqueuing" to "device
synchronize returned". A value near zero means the GPU had already completed
every queued step and the loop was host-bound for its entire length.

This is a diagnostic, not a benchmark: it runs two 128-step loops on one
growing sequence, so its absolute rates are not comparable with the canonical
bench's. Only the host-vs-device split and the host cost's growth with context
are meaningful.

Usage:
    uv run python scripts/qwen4exp_decode_host_bound_probe.py \
        --model-root <gguf dir> --case-id code-p512 --case-id code-p4096
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = REPO_ROOT / "benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json"
DEFAULT_CASES = ("code-p512", "code-p4096")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", action="append", default=None, help=f"Repeatable; default {DEFAULT_CASES}")
    parser.add_argument("--transitions", type=int, default=None, help="Defaults to the fixture value")
    parser.add_argument("--warm-steps", type=int, default=8)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--execution-profile", default="production")
    parser.add_argument("--hip-arch", default="gfx1151")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ.setdefault("HIPENGINE_HIP_ARCH", args.hip_arch)
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model

    fixture = json.loads(args.fixture.read_text())
    by_id = {str(case["id"]): case for case in fixture["cases"]}
    case_ids = tuple(args.case_id) if args.case_id else DEFAULT_CASES
    transitions = int(args.transitions if args.transitions is not None else fixture["decode_transitions"])
    model_root = args.model_root.resolve()

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile(str(args.execution_profile)),
    )

    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        if case_id not in by_id:
            raise SystemExit(f"unknown case id {case_id!r}")
        case = by_id[case_id]
        token_ids = [int(t) for t in case["prompt_token_ids"]]
        # Two 128-step loops plus warm steps run on one growing sequence.
        max_sequence_length = len(token_ids) + 2 * transitions + args.warm_steps + 32
        generator = Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        runner = generator.runner
        runtime = runner.runtime

        result = runner.prefill(token_ids)
        current = int(result.token_id)
        for _ in range(int(args.warm_steps)):
            current = int(runner.step(current).token_id)
        runtime.device_synchronize()

        # Host-only: enqueue every step, synchronize once at the end.
        enqueue_started = time.perf_counter_ns()
        for _ in range(transitions):
            current = int(runner.step(current).token_id)
        enqueue_ended = time.perf_counter_ns()
        runtime.device_synchronize()
        drained_ended = time.perf_counter_ns()

        enqueue_ms = (enqueue_ended - enqueue_started) / 1e6
        drain_ms = (drained_ended - enqueue_ended) / 1e6

        # Device-exposed: synchronize after every step.
        sync_started = time.perf_counter_ns()
        for _ in range(transitions):
            current = int(runner.step(current).token_id)
            runtime.device_synchronize()
        sync_ended = time.perf_counter_ns()
        sync_ms = (sync_ended - sync_started) / 1e6

        row = {
            "case_id": case_id,
            "prompt_tokens": len(token_ids),
            "transitions": transitions,
            "max_sequence_length": max_sequence_length,
            "host_enqueue_ms_total": round(enqueue_ms, 2),
            "host_enqueue_ms_per_step": round(enqueue_ms / transitions, 3),
            "device_drain_ms": round(drain_ms, 2),
            "loop_wall_ms_per_step": round((enqueue_ms + drain_ms) / transitions, 3),
            "sync_each_ms_per_step": round(sync_ms / transitions, 3),
            "host_bound": drain_ms < 1.0,
            "implied_tok_s_loop": round(1000.0 * transitions / (enqueue_ms + drain_ms), 3),
            "implied_tok_s_sync_each": round(1000.0 * transitions / sync_ms, 3),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        generator.close()

    if len(rows) > 1:
        first, last = rows[0], rows[-1]
        print(
            json.dumps(
                {
                    "host_cost_growth": {
                        "from": first["case_id"],
                        "to": last["case_id"],
                        "host_enqueue_ms_per_step": [
                            first["host_enqueue_ms_per_step"],
                            last["host_enqueue_ms_per_step"],
                        ],
                        "ratio": round(
                            last["host_enqueue_ms_per_step"] / first["host_enqueue_ms_per_step"], 3
                        ),
                    }
                }
            ),
            flush=True,
        )

    if args.output is not None:
        args.output.write_text(json.dumps({"schema": 1, "kind": "qwen4exp_decode_host_bound_probe", "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
