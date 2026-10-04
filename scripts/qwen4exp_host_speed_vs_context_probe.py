"""Does the host CPU get slower as the GPU's context grows?

The canonical bench's decode wall is 47.66 ms/step at p512 and 57.69 at p4096
while the profiled device kernel sum is 46.19 and 47.13 (+2.0%). Ablating
2-2.5 ms/step of host Python work moves that wall by -0.45% to +2.53%, so the
+10.04 ms/step of wall growth is not removable host work.

The host profile shows why that is consistent: every host function's call count
grows 2-4% from p512 to p4096 but its *tottime* grows ~25%. The same Python
calls cost more at depth. On a unified-memory APU the obvious mechanism is that
the GPU's larger working set starves the CPU's own cache and memory bandwidth,
which would make host work more expensive exactly in proportion to context.

This probe measures that directly: it runs a fixed, purely host-side workload
inside the decode loop, one block per step, and reports what that block costs.
The workload does not depend on context at all, so if its cost rises with
context the CPU is slower, not the code.

  host_burn_us_per_step = the measured cost of the same fixed workload
  wall_ms_per_step      = the decode loop's wall, for reference

If host_burn_us_per_step is flat, the +10.04 ms/step is device-side or
structural. If it rises with context, host ablations cannot pay for it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/home/lhl/hipEngine")
DEFAULT_FIXTURE = REPO_ROOT / "benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json"

STEP_KWARGS = {
    "capture_logits": False,
    "capture_target_hidden": False,
    "token_id_resident": True,
}
PREFILL_KWARGS = {"capture_logits": False, "capture_target_hidden": False}

# A fixed, context-independent host workload: environment lookups through the
# same four-frame path the profile blames, plus a small pure-Python arithmetic
# loop. Neither depends on sequence length.
ENV_KEYS = (
    "HIPENGINE_COMPILER_VERSION_TEXT",
    "HIPENGINE_COMPILER_VERSION_FILE",
    "HIPENGINE_HIP_ARCH",
    "HIPENGINE_HIP_OFFLOAD_ARCH",
    "HIPENGINE_BUILD_CACHE_ROOT",
    "HIPENGINE_REQUIRE_CACHED_BUILD",
    "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS",
    "HIPENGINE_QWEN4_EXP_Q8_DPU_REDUCTION",
)


def _burn() -> int:
    """One fixed unit of host work. No dependence on context or on the GPU."""

    total = 0
    for _ in range(32):
        for key in ENV_KEYS:
            value = os.environ.get(key, "")
            total += len(value)
    return total


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--transitions", type=int, default=64)
    parser.add_argument("--blocks-per-step", type=int, default=1)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
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
    case_ids = tuple(args.case_id) if args.case_id else ("code-p512", "code-p4096")
    transitions = int(args.transitions)
    blocks = int(args.blocks_per_step)
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

    # Unloaded baseline: the same workload with no GPU work outstanding at all.
    baseline_ns = 0
    for _ in range(200):
        started = time.perf_counter_ns()
        _burn()
        baseline_ns += time.perf_counter_ns() - started
    baseline_us = baseline_ns / 200 / 1000.0
    print(json.dumps({"phase": "baseline", "idle_gpu_host_burn_us": round(baseline_us, 3)}), flush=True)

    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        case = by_id[case_id]
        token_ids = [int(t) for t in case["prompt_token_ids"]]
        generator = Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=len(token_ids) + transitions + 8,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        runner = generator.runner
        runtime = runner.runtime

        # Warm pass so graph capture and the page cache settle.
        result = runner.prefill(token_ids, **PREFILL_KWARGS)
        current = int(result.token_id)
        for _ in range(8):
            current = int(runner.step(current, **STEP_KWARGS).token_id)
        runtime.device_synchronize()

        burn_ns = 0
        started = time.perf_counter_ns()
        for _ in range(transitions):
            for _ in range(blocks):
                mark = time.perf_counter_ns()
                _burn()
                burn_ns += time.perf_counter_ns() - mark
            current = int(runner.step(current, **STEP_KWARGS).token_id)
        enqueued = time.perf_counter_ns()
        runtime.device_synchronize()
        drained = time.perf_counter_ns()

        wall_ms = (drained - started) / 1e6
        burn_us = burn_ns / transitions / 1000.0
        row = {
            "case_id": case_id,
            "prompt_tokens": len(token_ids),
            "transitions": transitions,
            "blocks_per_step": blocks,
            "host_burn_us_per_step": round(burn_us, 3),
            "host_burn_ratio_vs_idle": round(burn_us / baseline_us, 4),
            "wall_ms_per_step": round(wall_ms / transitions, 3),
            "drain_ms_per_step": round((drained - enqueued) / 1e6 / transitions, 3),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        generator.close()

    if len(rows) > 1:
        first, last = rows[0], rows[-1]
        print(
            json.dumps(
                {
                    "growth": {
                        "from": first["case_id"],
                        "to": last["case_id"],
                        "host_burn_us_per_step": [
                            first["host_burn_us_per_step"],
                            last["host_burn_us_per_step"],
                        ],
                        "host_burn_ratio": round(
                            last["host_burn_us_per_step"] / first["host_burn_us_per_step"], 4
                        ),
                        "wall_ms_per_step": [
                            first["wall_ms_per_step"],
                            last["wall_ms_per_step"],
                        ],
                        "wall_ratio": round(last["wall_ms_per_step"] / first["wall_ms_per_step"], 4),
                    }
                }
            ),
            flush=True,
        )

    if args.output is not None:
        args.output.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "kind": "qwen4exp_host_speed_vs_context",
                    "idle_gpu_host_burn_us": round(baseline_us, 3),
                    "rows": rows,
                },
                indent=2,
            )
        )
    print("HOSTSPEED_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
