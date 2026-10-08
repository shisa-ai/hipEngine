#!/usr/bin/env python3
"""Natural-input owner screen for the checked/full-row dense Q8 DPP sibling.

The production runner is left on the current exact DPP parent.  A registry hook
runs the parent into the model-owned output and runs the diagnostic sibling into
one private output buffer on the identical natural activation and resident raw
weight.  HIP-event timings cover only the complete launcher call; captures
compare the full natural output, input bytes and raw-weight bytes.  This is a
qualification screen, not a runtime selector or model-rate claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc, memory_stats
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    Q8_0_DPP_CHECKED_FULL_ROWS_DECLARED_SHAPES as DECLARED_SHAPES,
    q8_0_coltile_dpp_checked_full_rows_qualified,
)
from hipengine.kernels.registry import KernelKey, register, resolve
from scripts import qwen4exp_row4_state_gate as gate
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity

PARENT = "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
CANDIDATE = "coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out"
BACKEND = "hip_gfx1151"
QUANT = "gguf_q8_0"
CHUNK_SIZE = 4096
# DECLARED_SHAPES and the admission contract are imported from the engine
# (q8_0_coltile_dpp_checked_full_rows_qualified) so the screen cannot drift
# from the production qualification gate.


def admitted_shape(rows: int, in_features: int, out_features: int) -> bool:
    """The production qualification contract; the engine helper is the source."""

    return q8_0_coltile_dpp_checked_full_rows_qualified(rows, in_features, out_features)


def gate_q8_admissible(rows: int, in_features: int, out_features: int) -> bool:
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        q8_0_coltile_checked_int32_admissible,
    )
    return q8_0_coltile_checked_int32_admissible(rows, in_features, out_features)


def sha256_device(ptr: int, nbytes: int, runtime) -> str:
    host = np.empty(int(nbytes), dtype=np.uint8)
    copy_device_to_host(host_array_ptr(host), DeviceBuffer(int(ptr), int(nbytes)), runtime=runtime)
    return hashlib.sha256(host.tobytes()).hexdigest()


def sha256_output(ptr: int, rows: int, out_features: int, runtime) -> tuple[str, np.ndarray]:
    host = np.empty((int(rows), int(out_features)), dtype=np.float32)
    copy_device_to_host(host_array_ptr(host), DeviceBuffer(int(ptr), host.nbytes), runtime=runtime)
    if not np.isfinite(host).all():
        raise AssertionError("natural dense output is non-finite")
    return hashlib.sha256(host.view(np.uint8).tobytes()).hexdigest(), host


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--repetitions", type=int, default=4,
                        help="warmup plus this many measured owner pairs")
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("repetitions must be at least 2")

    check_host()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ["HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION"] = "1"

    gate.register_gfx1151_kernels(replace=True)
    gate.register_qwen4_exp_gfx1151_profiles()
    profile = gate.resolve_runtime_profile(
        model=gate.QWEN4_EXP_MODEL,
        backend=BACKEND,
        quant=gate.QWEN4_EXP_QUANTS[1],
        profile=gate.ExecutionProfile.PRODUCTION,
    )
    fixture, fixture_sha = gate.load_fixture(gate.DEFAULT_FIXTURE)
    case_ids = args.case_id or [
        "code-p512", "code-p1024", "code-p4096",
        "general_en-p4096", "general_ja-p4096", "mixed_ja_en-p4096",
    ]
    cases = [next(c for c in fixture["cases"] if c["id"] == name) for name in case_ids]
    index = gate.load_gguf_index(gate.discover_gguf_files(args.model_root)[0])
    generator = profile.construct_generator(lambda: gate.Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root,
        weight_index=index,
        model_plugin=gate.resolve_model(index.architecture or ""),
        backend=BACKEND,
        max_sequence_length=4352,
        prefill_chunk_size=CHUNK_SIZE,
    ))
    runtime = generator.runner.runtime
    key = KernelKey(BACKEND, "linear", QUANT, PARENT)
    parent = resolve(backend=BACKEND, layer="linear", quant=QUANT, variant=PARENT)
    candidate = resolve(backend=BACKEND, layer="linear", quant=QUANT, variant=CANDIDATE)
    start, stop = runtime.event_create(), runtime.event_create()
    report = {
        "schema": 1,
        "kind": "qwen4exp_dense_dpp_checked_full_rows_natural_owner_screen",
        "status": "running",
        "performance_claim": False,
        "source": gate._git_metadata(ROOT),
        "host": gate._host_metadata(),
        "command": sys.argv,
        "model_identity": model_identity(args.model_root),
        "fixture_sha256": fixture_sha,
        "manifest_sha256": profile.manifest_sha256,
        "manifest_scope": "production profile with diagnostic registry hook; no default change",
        "parent_variant": PARENT,
        "candidate_variant": CANDIDATE,
        "backend": BACKEND,
        "quant": QUANT,
        "chunk_size": CHUNK_SIZE,
        "repetitions": args.repetitions,
        "cases": [],
        "skipped_calls": 0,
    }
    captures: list[dict] = []
    current_case = ""

    def hooked(*call_args, **kwargs):
        nonlocal captures
        if len(call_args) < 6:
            raise AssertionError("raw linear ABI hook received too few arguments")
        x_ptr, raw_ptr, out_ptr = map(int, call_args[:3])
        rows, in_features, out_features = map(int, call_args[3:6])
        if not admitted_shape(rows, in_features, out_features):
            report["skipped_calls"] += 1
            return parent(*call_args, **kwargs)
        if int(kwargs.get("stream", 0)) != 0:
            raise ValueError("natural owner hook requires default stream")

        raw_nbytes = out_features * (in_features // 32) * 34
        x_nbytes = rows * in_features * 4
        out_nbytes = rows * out_features * 4
        x_before = sha256_device(x_ptr, x_nbytes, runtime)
        w_before = sha256_device(raw_ptr, raw_nbytes, runtime)
        private = malloc(out_nbytes, runtime=runtime)
        private_args = list(call_args)
        private_args[2] = private.ptr
        parent_samples: list[float] = []
        candidate_samples: list[float] = []
        parent_hashes: list[str] = []
        candidate_hashes: list[str] = []
        try:
            # Warm both arms before recording samples.
            parent(*call_args, **kwargs)
            candidate(*private_args, **kwargs)
            runtime.device_synchronize()
            for rep in range(args.repetitions):
                order = (0, 1) if (rep + len(captures)) % 2 == 0 else (1, 0)
                for arm in order:
                    runtime.event_record(start)
                    if arm == 0:
                        parent(*call_args, **kwargs)
                    else:
                        candidate(*private_args, **kwargs)
                    runtime.event_record(stop)
                    runtime.event_synchronize(stop)
                    elapsed = runtime.event_elapsed_time_ms(start, stop)
                    digest, _ = sha256_output(
                        int(out_ptr) if arm == 0 else int(private.ptr),
                        rows, out_features, runtime,
                    )
                    if arm == 0:
                        parent_samples.append(elapsed)
                        parent_hashes.append(digest)
                    else:
                        candidate_samples.append(elapsed)
                        candidate_hashes.append(digest)
            # Re-publish the exact parent output for the model continuation.
            parent(*call_args, **kwargs)
            runtime.device_synchronize()
            parent_digest, parent_output = sha256_output(int(out_ptr), rows, out_features, runtime)
            candidate_digest, candidate_output = sha256_output(int(private.ptr), rows, out_features, runtime)
            np.testing.assert_array_equal(parent_output.view(np.uint32), candidate_output.view(np.uint32))
            x_after = sha256_device(int(x_ptr), x_nbytes, runtime)
            w_after = sha256_device(raw_ptr, raw_nbytes, runtime)
            if x_after != x_before or w_after != w_before:
                raise AssertionError("natural dense input or raw weight was mutated")
            captures.append({
                "case_id": current_case,
                "rows": rows,
                "in_features": in_features,
                "out_features": out_features,
                "weight_ptr": raw_ptr,
                "weight_nbytes": raw_nbytes,
                "weight_sha256": w_before,
                "parent_samples_ms": parent_samples,
                "candidate_samples_ms": candidate_samples,
                "parent_ms": statistics.median(parent_samples),
                "candidate_ms": statistics.median(candidate_samples),
                "speedup": statistics.median(parent_samples) / statistics.median(candidate_samples),
                "parent_hashes": parent_hashes,
                "candidate_hashes": candidate_hashes,
                "output_sha256": parent_digest,
                "output_exact": parent_digest == candidate_digest,
                "input_immutable": x_after == x_before,
                "weight_immutable": w_after == w_before,
            })
        finally:
            free(private, runtime=runtime)

    try:
        register(key, hooked, replace=True)
        for case in cases:
            current_case = case["id"]
            before = len(captures)
            token = generator.runner.prefill(case["prompt_token_ids"])
            own = captures[before:]
            if not own:
                raise AssertionError((current_case, "no admitted natural dense Q8 calls"))
            if any(not row["output_exact"] for row in own):
                raise AssertionError((current_case, "natural candidate output mismatch"))
            row = {
                "id": current_case,
                "first_token": token.token_id,
                "calls": len(own),
                "candidate_calls": len(own),
                "all_output_exact": all(x["output_exact"] for x in own),
                "all_input_immutable": all(x["input_immutable"] for x in own),
                "all_weight_immutable": all(x["weight_immutable"] for x in own),
                "parent_ms": sum(x["parent_ms"] for x in own),
                "candidate_ms": sum(x["candidate_ms"] for x in own),
                "speedup": sum(x["parent_ms"] for x in own) / sum(x["candidate_ms"] for x in own),
                "shapes": sorted({tuple((x["rows"], x["in_features"], x["out_features"])) for x in own}),
                "captures": own,
            }
            report["cases"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "captures"}), flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        register(key, parent, replace=True)
        runtime.event_destroy(start)
        runtime.event_destroy(stop)
        generator.close()
        report["memory_after_close"] = memory_stats()
        report["capture_count"] = len(captures)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
