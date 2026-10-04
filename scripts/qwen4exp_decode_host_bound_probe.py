"""Separate the host enqueue cost of a decode step from the device time.

The canonical AR bench (`scripts/qwen4exp_canonical_ar_bench.py`) measures decode
as 128 `runner.step()` calls in one tight Python loop with a single
`device_synchronize()` at the end. That number is `max(host_enqueue, device)` per
step, and it cannot say which one it measured. This probe measures both
separately, for one canonical case, on the production profile.

**`--skip-token-readback` is required for the split to mean anything.**
`runner.step()` ends with a *blocking* device-to-host copy of the argmax result
(`_finish_lm_head` -> `copy_device_to_host`), so with the default settings the
"host enqueue" window already contains the device wait and `device_drain_ms` is
zero by construction. That zero is a tautology, not evidence of a host-bound
loop: the canonical bench's `--capture-mode compact`, which has no explicit
per-step synchronize at all, reports the same zero. With the readback skipped
the drain becomes a measurement, and on this host it is small but non-zero
(0.343 ms/step at p512, 0.575 at p4096 over 128 steps) while the two arms'
walls agree to under 1% - host enqueue and device execution are within ~1% of
each other and the loop wall is their maximum.

`device_drain_ms` is the time from "host finished enqueuing" to "device
synchronize returned".

This is a diagnostic, not a benchmark: it runs two 128-step loops on one
growing sequence and never warms the PLE page cache, so its absolute rates are
not comparable with the canonical bench's - they run about 15% (p512) and 47%
(p4096) high. Only within-arm deltas and the shape of the host cost's growth
with context are meaningful.

Usage:
    uv run python scripts/qwen4exp_decode_host_bound_probe.py \
        --model-root <gguf dir> --case-id code-p512 --case-id code-p4096 \
        --skip-token-readback
"""

from __future__ import annotations

import argparse
import ctypes
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
    parser.add_argument(
        "--host-profile",
        action="store_true",
        help=(
            "Run the decode loop under cProfile and print the functions that dominate the "
            "host side. Absolute times are inflated by the profiler; the attribution is not."
        ),
    )
    parser.add_argument("--profile-steps", type=int, default=64)
    parser.add_argument(
        "--skip-token-readback",
        action="store_true",
        help=(
            "Skip the per-step blocking device-to-host copy of the argmax token, so the "
            "trailing device drain becomes a measurement instead of a tautology. The "
            "generated token sequence is unchanged (the runner takes its next token from "
            "the device buffer) but the returned token is stale, so only timing is valid."
        ),
    )
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
        max_sequence_length = (
            len(token_ids) + 2 * transitions + args.warm_steps
            + (int(args.profile_steps) if args.host_profile else 0) + 32
        )
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

        if args.skip_token_readback:
            import hipengine.runtime.qwen4_exp_runner as runner_module

            real_copy = runner_module.copy_device_to_host
            readback_state: dict[str, int | bool] = {"skip": False, "token": 0}

            def patched_copy(host_ptr: int, buffer: Any, nbytes: int | None = None, *, runtime=None) -> None:
                count = buffer.nbytes if nbytes is None else int(nbytes)
                if readback_state["skip"] and count == 8:
                    ctypes.c_int64.from_address(int(host_ptr)).value = int(readback_state["token"])
                    return
                real_copy(host_ptr, buffer, nbytes, runtime=runtime)
                if count == 8:
                    readback_state["token"] = int(ctypes.c_int64.from_address(int(host_ptr)).value)

            runner_module.copy_device_to_host = patched_copy

            def set_skip(value: bool) -> None:
                readback_state["skip"] = value

        else:

            def set_skip(value: bool) -> None:  # noqa: ARG001 - signature parity
                return

        result = runner.prefill(token_ids)
        current = int(result.token_id)
        for _ in range(int(args.warm_steps)):
            current = int(runner.step(current).token_id)
        runtime.device_synchronize()

        if args.host_profile:
            # Attribute the host side of a decode step. cProfile inflates absolute
            # times, so read the ordering and the shares, not the seconds.
            import cProfile
            import pstats

            steps = int(args.profile_steps)

            def _profile_loop() -> None:
                nonlocal current
                for _ in range(steps):
                    current = int(runner.step(current).token_id)

            profiler = cProfile.Profile()
            profiler.enable()
            _profile_loop()
            profiler.disable()
            runtime.device_synchronize()
            stats = pstats.Stats(profiler)
            total = stats.total_tt
            print(json.dumps({"case_id": case_id, "host_profile_steps": steps,
                              "host_profile_total_s": round(total, 3)}), flush=True)
            entries = sorted(stats.stats.items(), key=lambda kv: kv[1][2], reverse=True)[:25]
            callers_of = {k: v[4] for k, v in stats.stats.items()}
            print(f"  {'tottime_s':>10} {'share':>7} {'ncalls':>8}  function", flush=True)
            for key, (_cc, _nc, tt, _ct, _callers) in entries:
                filename, lineno, func = key
                print(f"  {tt:10.3f} {100 * tt / total:6.1f}% {_nc:8d}  {func}  ({filename}:{lineno})", flush=True)
                if tt / total < 0.02:
                    continue
                for caller, cdata in sorted(callers_of.get(key, {}).items(), key=lambda kv: -kv[1][3])[:3]:
                    print(f"        <- {caller[2]} ({caller[0]}:{caller[1]}) x{cdata[1]} tt={cdata[3]:.3f}s", flush=True)

        # Host-only: enqueue every step, synchronize once at the end.
        # With the readback in place this window is per-step *wall*, not host cost;
        # `--skip-token-readback` is what turns it into a host measurement.
        set_skip(True)
        enqueue_started = time.perf_counter_ns()
        for _ in range(transitions):
            current = int(runner.step(current).token_id)
        enqueue_ended = time.perf_counter_ns()
        runtime.device_synchronize()
        drained_ended = time.perf_counter_ns()
        set_skip(False)

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
            "token_readback_skipped": bool(args.skip_token_readback),
            "host_bound": (drain_ms < 1.0) if args.skip_token_readback else None,
            "host_bound_note": (
                "drain measured with the per-step token readback skipped"
                if args.skip_token_readback
                else "NOT MEASURED: the per-step blocking token readback already synchronizes, "
                "so the host window is per-step wall and the drain is zero by construction; "
                "re-run with --skip-token-readback"
            ),
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
