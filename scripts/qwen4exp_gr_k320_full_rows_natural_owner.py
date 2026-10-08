#!/usr/bin/env python3
"""Natural-input owner screen for the K320/full-row GR-up DPP sibling.

The production runner stays on the promoted GR-up DPP owner.  A registry hook
runs the parent into the model-owned gate/mixed outputs and runs the K320
full-row sibling into private output buffers on the identical natural
activation, normalized input and resident raw weight.  HIP-event timings on the
launch stream cover only the complete fused GR-up operation; captures compare
both outputs (gate and mixed) bit-for-bit and hash the activation, normalized
input and raw weight for immutability.  This is a qualification screen, not a
runtime selector or model-rate claim.

The sibling is admitted only for the screen's declared window: rows > 256 in
complete groups of 4, four branches, low rank 320, hidden 2560 - exactly the
predicate the runner's default-off selector applies behind
``HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS``.
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
from hipengine.kernels.registry import KernelKey, register, resolve
from scripts import qwen4exp_row4_state_gate as gate
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity

PARENT = "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact"
CANDIDATE = "coltile2_branch4_rowbatch4_wave_scale_dpp_k320_full_rows_f32_exact"
BACKEND = "hip_gfx1151"
LAYER = "linear+gr_gated_mean"
QUANT = "gguf_q8_0"
CHUNK_SIZE = 4096
LOW_RANK = 320
BRANCHES = 4
HIDDEN = 2560


def admitted_shape(rows: int, in_features: int, branches: int, hidden: int) -> bool:
    """The screen's declared window; matches the runner's default-off selector."""

    rows, in_features, branches, hidden = map(int, (rows, in_features, branches, hidden))
    return (
        rows > 256
        and rows % 4 == 0
        and branches == BRANCHES
        and in_features == LOW_RANK
        and hidden == HIDDEN
    )


def sha256_device(ptr: int, nbytes: int, runtime) -> str:
    host = np.empty(int(nbytes), dtype=np.uint8)
    copy_device_to_host(host_array_ptr(host), DeviceBuffer(int(ptr), int(nbytes)), runtime=runtime)
    return hashlib.sha256(host.tobytes()).hexdigest()


def sha256_f32(ptr: int, rows: int, cols: int, runtime) -> tuple[str, np.ndarray]:
    host = np.empty((int(rows), int(cols)), dtype=np.float32)
    copy_device_to_host(host_array_ptr(host), DeviceBuffer(int(ptr), host.nbytes), runtime=runtime)
    if not np.isfinite(host).all():
        raise AssertionError("natural GR-up output is non-finite")
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
    os.environ["HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE"] = "1"
    os.environ["HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION"] = "1"
    os.environ["HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS"] = "0"

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
    key = KernelKey(BACKEND, LAYER, QUANT, PARENT)
    parent = resolve(backend=BACKEND, layer=LAYER, quant=QUANT, variant=PARENT)
    candidate = resolve(backend=BACKEND, layer=LAYER, quant=QUANT, variant=CANDIDATE)
    start, stop = runtime.event_create(), runtime.event_create()
    report = {
        "schema": 1,
        "kind": "qwen4exp_gr_k320_full_rows_natural_owner_screen",
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
        "layer": LAYER,
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
        if len(call_args) < 9:
            raise AssertionError("fused GR-up hook received too few arguments")
        x_ptr, raw_ptr, normalized_ptr = map(int, call_args[:3])
        gate_ptr, mixed_ptr = map(int, call_args[3:5])
        rows, in_features, branches, hidden = map(int, call_args[5:9])
        stream = int(kwargs.get("stream", 0))
        if not admitted_shape(rows, in_features, branches, hidden):
            report["skipped_calls"] += 1
            return parent(*call_args, **kwargs)

        raw_nbytes = (branches * hidden) * (in_features // 32) * 34
        x_nbytes = rows * in_features * 4
        normalized_nbytes = rows * branches * hidden * 4
        gate_nbytes = rows * branches * hidden * 4
        mixed_nbytes = rows * hidden * 4
        x_before = sha256_device(x_ptr, x_nbytes, runtime)
        w_before = sha256_device(raw_ptr, raw_nbytes, runtime)
        n_before = sha256_device(normalized_ptr, normalized_nbytes, runtime)
        gate_private = malloc(gate_nbytes, runtime=runtime)
        mixed_private = malloc(mixed_nbytes, runtime=runtime)
        private_args = list(call_args)
        private_args[3] = gate_private.ptr
        private_args[4] = mixed_private.ptr
        parent_samples: list[float] = []
        candidate_samples: list[float] = []
        parent_gate_hashes: list[str] = []
        candidate_gate_hashes: list[str] = []
        parent_mixed_hashes: list[str] = []
        candidate_mixed_hashes: list[str] = []
        try:
            # Warm both arms before recording samples.
            parent(*call_args, **kwargs)
            candidate(*private_args, **kwargs)
            runtime.device_synchronize()
            for rep in range(args.repetitions):
                order = (0, 1) if (rep + len(captures)) % 2 == 0 else (1, 0)
                for arm in order:
                    runtime.event_record(start, stream=stream)
                    if arm == 0:
                        parent(*call_args, **kwargs)
                    else:
                        candidate(*private_args, **kwargs)
                    runtime.event_record(stop, stream=stream)
                    runtime.event_synchronize(stop)
                    elapsed = runtime.event_elapsed_time_ms(start, stop)
                    if arm == 0:
                        gate_digest, _ = sha256_f32(gate_ptr, rows, branches * hidden, runtime)
                        mixed_digest, _ = sha256_f32(mixed_ptr, rows, hidden, runtime)
                        parent_samples.append(elapsed)
                        parent_gate_hashes.append(gate_digest)
                        parent_mixed_hashes.append(mixed_digest)
                    else:
                        gate_digest, _ = sha256_f32(gate_private.ptr, rows, branches * hidden, runtime)
                        mixed_digest, _ = sha256_f32(mixed_private.ptr, rows, hidden, runtime)
                        candidate_samples.append(elapsed)
                        candidate_gate_hashes.append(gate_digest)
                        candidate_mixed_hashes.append(mixed_digest)
            # Re-publish the exact parent outputs for the model continuation.
            parent(*call_args, **kwargs)
            runtime.device_synchronize()
            parent_gate_digest, parent_gate_out = sha256_f32(gate_ptr, rows, branches * hidden, runtime)
            parent_mixed_digest, parent_mixed_out = sha256_f32(mixed_ptr, rows, hidden, runtime)
            candidate_gate_digest, candidate_gate_out = sha256_f32(
                gate_private.ptr, rows, branches * hidden, runtime)
            candidate_mixed_digest, candidate_mixed_out = sha256_f32(
                mixed_private.ptr, rows, hidden, runtime)
            np.testing.assert_array_equal(parent_gate_out.view(np.uint32), candidate_gate_out.view(np.uint32))
            np.testing.assert_array_equal(parent_mixed_out.view(np.uint32), candidate_mixed_out.view(np.uint32))
            x_after = sha256_device(x_ptr, x_nbytes, runtime)
            w_after = sha256_device(raw_ptr, raw_nbytes, runtime)
            n_after = sha256_device(normalized_ptr, normalized_nbytes, runtime)
            if x_after != x_before or w_after != w_before or n_after != n_before:
                raise AssertionError("natural GR-up input, normalized input or raw weight was mutated")
            captures.append({
                "case_id": current_case,
                "rows": rows,
                "in_features": in_features,
                "branches": branches,
                "hidden": hidden,
                "weight_ptr": raw_ptr,
                "weight_nbytes": raw_nbytes,
                "weight_sha256": w_before,
                "normalized_sha256": n_before,
                "parent_samples_ms": parent_samples,
                "candidate_samples_ms": candidate_samples,
                "parent_ms": statistics.median(parent_samples),
                "candidate_ms": statistics.median(candidate_samples),
                "speedup": statistics.median(parent_samples) / statistics.median(candidate_samples),
                "parent_gate_hashes": parent_gate_hashes,
                "candidate_gate_hashes": candidate_gate_hashes,
                "parent_mixed_hashes": parent_mixed_hashes,
                "candidate_mixed_hashes": candidate_mixed_hashes,
                "gate_sha256": parent_gate_digest,
                "mixed_sha256": parent_mixed_digest,
                "outputs_exact": (
                    parent_gate_digest == candidate_gate_digest
                    and parent_mixed_digest == candidate_mixed_digest
                ),
                "input_immutable": x_after == x_before,
                "normalized_immutable": n_after == n_before,
                "weight_immutable": w_after == w_before,
            })
        finally:
            free(gate_private, runtime=runtime)
            free(mixed_private, runtime=runtime)

    try:
        register(key, hooked, replace=True)
        for case in cases:
            current_case = case["id"]
            before = len(captures)
            token = generator.runner.prefill(case["prompt_token_ids"])
            own = captures[before:]
            if not own:
                raise AssertionError((current_case, "no admitted natural GR-up calls"))
            if any(not row["outputs_exact"] for row in own):
                raise AssertionError((current_case, "natural candidate GR-up output mismatch"))
            row = {
                "id": current_case,
                "first_token": token.token_id,
                "calls": len(own),
                "candidate_calls": len(own),
                "all_outputs_exact": all(x["outputs_exact"] for x in own),
                "all_input_immutable": all(x["input_immutable"] for x in own),
                "all_normalized_immutable": all(x["normalized_immutable"] for x in own),
                "all_weight_immutable": all(x["weight_immutable"] for x in own),
                "parent_ms": sum(x["parent_ms"] for x in own),
                "candidate_ms": sum(x["candidate_ms"] for x in own),
                "speedup": sum(x["parent_ms"] for x in own) / sum(x["candidate_ms"] for x in own),
                "shapes": sorted({tuple((x["rows"], x["in_features"], x["branches"], x["hidden"])) for x in own}),
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
