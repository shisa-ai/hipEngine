"""Measure decode submission-loop wall time and the trailing synchronization wait.

Neither interval is a CPU/device attribution: submission may block in runtime
calls, and a small trailing wait does not establish a host-bound workload.
--skip-token-readback suppresses only the compact argmax-token copy while the
next step consumes the resident device token. Returned host tokens are stale in
that interval and must not be used as output evidence.

The two loops run consecutively on a growing sequence, so their context lengths
differ. This diagnostic is not a matched throughput benchmark.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_FIXTURE = REPO_ROOT / "benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json"
DEFAULT_CASES = ("code-p512", "code-p4096")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--transitions", type=int, default=None)
    parser.add_argument("--warm-steps", type=int, default=8)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--execution-profile", default="production")
    parser.add_argument("--hip-arch", default="gfx1151")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--host-profile", action="store_true", help="Print cProfile call timings, which can include blocking device waits.")
    parser.add_argument("--profile-steps", type=int, default=64)
    parser.add_argument("--skip-token-readback", action="store_true", help="Suppress only the compact token readback during the submission loop; host token results are stale.")
    parser.add_argument("--output", type=Path, default=None)
    return parser


@contextmanager
def _token_readback_scope(runner, enabled, *, module=None):
    if not enabled:
        yield
        return
    if module is None:
        from hipengine.runtime import qwen4_exp_runner as module
    original = module.copy_device_to_host
    token_ptr = int(runner.token_id_buffer.ptr)

    def copy(host_ptr, buffer, nbytes=None, *, runtime=None):
        count = buffer.nbytes if nbytes is None else int(nbytes)
        if int(buffer.ptr) == token_ptr and count == 8:
            # The loop consumes the device token, never this placeholder.
            ctypes.c_int64.from_address(int(host_ptr)).value = 0
            return
        return original(host_ptr, buffer, nbytes, runtime=runtime)

    module.copy_device_to_host = copy
    try:
        yield
    finally:
        module.copy_device_to_host = original


def _step(runner, token):
    return int(runner.step(token, capture_logits=False, capture_target_hidden=False,
                           token_id_resident=True).token_id)


def _timing_fields(submit_ms, wait_ms, sync_ms, transitions, skipped):
    return {
        "submission_loop_ms_total": round(submit_ms, 2),
        "submission_loop_ms_per_step": round(submit_ms / transitions, 3),
        "trailing_sync_wait_ms": round(wait_ms, 2),
        "loop_wall_ms_per_step": round((submit_ms + wait_ms) / transitions, 3),
        "sync_each_ms_per_step": round(sync_ms / transitions, 3),
        "token_readback_skipped": bool(skipped),
        "timing_note": "Submission includes possible runtime blocking; trailing wait is not total device time. Consecutive loops use different context lengths.",
    }


def _measure_case(generator, token_ids, args, transitions):
    runner = generator.runner
    runtime = runner.runtime
    try:
        current = int(runner.prefill(token_ids, capture_logits=False,
                                     capture_target_hidden=False).token_id)
        for _ in range(args.warm_steps):
            current = _step(runner, current)
        runtime.device_synchronize()
        if args.host_profile:
            import cProfile
            import pstats
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                for _ in range(args.profile_steps):
                    current = _step(runner, current)
            finally:
                profiler.disable()
            runtime.device_synchronize()
            pstats.Stats(profiler).sort_stats("tottime").print_stats(25)
        with _token_readback_scope(runner, args.skip_token_readback):
            started = time.perf_counter_ns()
            for _ in range(transitions):
                current = _step(runner, current)
            submitted = time.perf_counter_ns()
            runtime.device_synchronize()
            drained = time.perf_counter_ns()
        started_sync = time.perf_counter_ns()
        for _ in range(transitions):
            current = _step(runner, current)
            runtime.device_synchronize()
        ended_sync = time.perf_counter_ns()
        return _timing_fields((submitted-started)/1e6, (drained-submitted)/1e6,
                              (ended_sync-started_sync)/1e6, transitions,
                              args.skip_token_readback)
    finally:
        generator.close()


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    fixture = json.loads(args.fixture.read_text())
    transitions = int(args.transitions if args.transitions is not None else fixture["decode_transitions"])
    if transitions <= 0 or args.warm_steps < 0 or args.profile_steps <= 0:
        parser.error("transitions/profile-steps must be positive and warm-steps non-negative")
    by_id = {str(case["id"]): case for case in fixture["cases"]}
    case_ids = tuple(args.case_id) if args.case_id else DEFAULT_CASES
    for case_id in case_ids:
        if case_id not in by_id:
            parser.error(f"unknown case id {case_id!r}")
    os.environ.setdefault("HIPENGINE_HIP_ARCH", args.hip_arch)
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND, QWEN4_EXP_MODEL, QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    model_root = args.model_root.resolve()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    profile = resolve_runtime_profile(model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile(str(args.execution_profile)))
    rows = []
    for case_id in case_ids:
        tokens = [int(t) for t in by_id[case_id]["prompt_token_ids"]]
        capacity = len(tokens) + 2*transitions + args.warm_steps + (args.profile_steps if args.host_profile else 0) + 32
        generator = profile.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
            model_path=model_root, weight_index=index, model_plugin=plugin,
            backend="hip_gfx1151", max_sequence_length=capacity,
            prefill_chunk_size=args.prefill_chunk_size))
        row = {"case_id": case_id, "prompt_tokens": len(tokens),
               "transitions": transitions, "max_sequence_length": capacity,
               **_measure_case(generator, tokens, args, transitions)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps({"schema": 2, "kind": "qwen4exp_decode_host_bound_probe", "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
