#!/usr/bin/env python3
"""Counterbalanced real-weight screen: dense Q8_0 checked/full-row DPP sibling.

Times the diagnostic ``coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows``
sibling against the current dense DPP owner
``coltile8_rowbatch4_wave_scale_pipelined_dpp`` on identical real GGUF weights
and identical activations. This is an operation-complete host-wall comparison:
each sample wraps the full launcher call including the stream synchronize, and
compilation happens before any timed region.

The comparator is the DPP wrapper the engine currently selects, not the stale
shuffle parent named by ``scripts/qwen4exp_dense_projection_ab.py``.

Activations are synthetic (seeded per layer/rows); the weight bytes are the
real model tensors. A win here is operation evidence only and does not qualify
natural-input owner/state behavior or any default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload

PARENT = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
CANDIDATE = (
    "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out"
)

# Dense Q8_0 projection shapes of the Qwen3.8-Flash-Next layer, by GGUF suffix.
DENSE_TENSORS = [
    "attn_qkv.weight",  # K=2560, N=10240
    "attn_gate.weight",  # K=2560, N=6144
    "ssm_out.weight",  # K=6144, N=2560
    "ffn_gate_shexp.weight",  # K=2560, N=640
    "ffn_up_shexp.weight",  # K=2560, N=640
    "ffn_down_shexp.weight",  # K=640, N=2560
    "hc_attn_down.weight",  # K=10240, N=320
    "hc_ffn_down.weight",  # K=10240, N=320
]


def _load_raw(readers, name: str) -> np.ndarray | None:
    for reader in readers:
        if any(t.name == name for t in reader.info.tensors):
            info = reader.tensor_info(name)
            if info.ggml_type_name != "Q8_0":
                raise SystemExit(f"{name}: expected Q8_0, found {info.ggml_type_name}")
            return np.ascontiguousarray(reader.tensor_data(name)).view(np.uint8).reshape(-1)
    return None


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _build_identity(library, kernel_py: Path) -> dict:
    """Loaded shared-library and build-cache identity for this run."""
    identity: dict = {"loaded_library": None, "loaded_library_sha256": None}
    name = getattr(library, "_name", None)
    if name:
        identity["loaded_library"] = str(name)
        identity["loaded_library_sha256"] = _file_sha256(Path(name))
        manifest = Path(name).with_name("manifest.txt")
        if manifest.exists():
            text = manifest.read_text()
            identity["build_manifest"] = text
            for line in text.splitlines():
                if line.startswith("cache_key="):
                    identity["cache_key"] = line.split("=", 1)[1]
                elif line.startswith("compiler_version<<"):
                    pass
        device_object = next(
            iter(Path(name).parent.glob("*.hipv4-amdgcn-amd-amdhsa--gfx1151")), None
        )
        if device_object is not None:
            identity["device_object"] = str(device_object)
            identity["device_object_sha256"] = _file_sha256(device_object)
    identity["wrapper_source_sha256"] = _file_sha256(kernel_py)
    return identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,22")
    parser.add_argument("--rows", default="512,1024,4096")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--pressure-mib", type=int, default=0)
    parser.add_argument("--tensors", default=",".join(DENSE_TENSORS))
    args = parser.parse_args()

    layers = [int(part) for part in args.layers.split(",") if part]
    row_cases = [int(part) for part in args.rows.split(",") if part]
    tensors = [name for name in args.tensors.split(",") if name]
    if args.pairs < 2 or args.pairs % 2:
        parser.error("use an even pair count >= 2 to balance first/second arm order")
    if any(rows % 4 for rows in row_cases):
        parser.error("all row cases must be divisible by 4 for the checked sibling")

    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    runtime = get_hip_runtime()
    library = q8.build_gguf_k_gemv(load=True, require_cached=True)
    parent_fn = getattr(q8, PARENT)
    candidate_fn = getattr(q8, CANDIDATE)

    readers = [GGUFReader(path) for path in discover_gguf_files(args.model_root)]
    source_files = {
        "kernel_hip": str(ROOT / "hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip"),
        "wrapper_py": str(ROOT / "hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.py"),
        "harness_py": str(ROOT / "scripts/qwen4exp_dense_dpp_checked_full_rows_screen.py"),
    }
    report = {
        "schema": 1,
        "kind": "qwen4exp_dense_dpp_checked_full_rows_screen",
        "performance_claim": False,
        "status": "diagnostic",
        "source": _git_metadata(ROOT),
        "source_files": {
            key: {"path": path, "sha256": _file_sha256(Path(path))}
            for key, path in source_files.items()
        },
        "build_identity": _build_identity(
            library, ROOT / "hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.py"
        ),
        "host": _host_metadata(),
        "command": sys.argv,
        "model": "Qwen3.8-Flash-Next UD-Q4_K_XL",
        "quant": "Q8_0",
        "parent_variant": PARENT,
        "candidate_variant": CANDIDATE,
        "activation_note": (
            "Synthetic per-(layer,rows) seeded standard normal scaled by 0.1. "
            "Weight bytes are the real model tensors. Operation evidence only."
        ),
        "order_balanced": True,
        "device_fill_precondition_mib": args.pressure_mib,
        "default_changed": False,
        "cases": [],
        "skipped": [],
    }

    for layer in layers:
        for tensor in tensors:
            name = f"blk.{layer}.{tensor}"
            raw = _load_raw(readers, name)
            if raw is None:
                report["skipped"].append(name)
                print(f"skip {name}: not present", flush=True)
                continue
            info = next(
                reader.tensor_info(name)
                for reader in readers
                if any(t.name == name for t in reader.info.tensors)
            )
            n, k = int(info.shape[0]), int(info.shape[1])
            if k % 32 or n % 8:
                report["skipped"].append(name)
                print(f"skip {name}: K={k} N={n} unsupported", flush=True)
                continue
            weight_sha256 = hashlib.sha256(raw).hexdigest()
            for rows in row_cases:
                allocations = []
                try:
                    x = (
                        np.random.default_rng(4000 + layer * 100 + rows)
                        .normal(0, 0.1, (rows, k))
                        .astype(np.float32)
                    )
                    dx, dw = [_upload(v, runtime, allocations) for v in (x, raw)]
                    outputs = [
                        _alloc((rows, n), np.float32, runtime, allocations)
                        for _ in range(2)
                    ]
                    pressure = (
                        _alloc((args.pressure_mib * 1024 * 1024,), np.uint8, runtime, allocations)
                        if args.pressure_mib
                        else None
                    )

                    def run(arm: int):
                        fn = parent_fn if arm == 0 else candidate_fn
                        fn(
                            dx.ptr,
                            dw.ptr,
                            outputs[arm].ptr,
                            rows,
                            k,
                            n,
                            runtime=runtime,
                            library=library,
                        )
                        runtime.device_synchronize()

                    run(0)
                    run(1)
                    # Exactness is verified once per case: every pair is
                    # bit-identical by construction (no arithmetic change), and
                    # the focused GPU gate already proves it across shapes. A
                    # per-pair download of a 4096x10240 tensor would dominate the
                    # screen's wall time without adding evidence.
                    a, b = [
                        _download(o, (rows, n), np.float32, runtime) for o in outputs
                    ]
                    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
                    del a, b
                    samples: list[list[float]] = [[], []]
                    for pair in range(args.pairs):
                        order = (0, 1) if pair % 2 == 0 else (1, 0)
                        for arm in order:
                            if pressure is not None:
                                runtime.memset(pressure.ptr, pair % 256, pressure.nbytes)
                                runtime.device_synchronize()
                            start = time.perf_counter()
                            run(arm)
                            samples[arm].append((time.perf_counter() - start) * 1000.0)
                    a, b = [
                        _download(o, (rows, n), np.float32, runtime) for o in outputs
                    ]
                    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
                    del a, b
                    medians = [statistics.median(s) for s in samples]
                    row = {
                        "tensor": name,
                        "shape": [rows, k, n],
                        "weight_sha256": weight_sha256,
                        "parent_ms": medians[0],
                        "candidate_ms": medians[1],
                        "speedup": medians[0] / medians[1],
                        "mean_speedup": statistics.mean(samples[0]) / statistics.mean(samples[1]),
                        "speedup_by_order": {
                            order: statistics.median(samples[0][offset::2])
                            / statistics.median(samples[1][offset::2])
                            for order, offset in (("parent_first", 0), ("candidate_first", 1))
                        },
                        "samples_ms": samples,
                        "exact": True,
                    }
                    report["cases"].append(row)
                    print(json.dumps(row), flush=True)
                finally:
                    for ptr in reversed(allocations):
                        free(ptr, runtime=runtime)

    if report["cases"]:
        speedups = [case["speedup"] for case in report["cases"]]
        report["geometric_mean_speedup"] = float(
            np.exp(np.mean(np.log(speedups)))
        )
        report["min_speedup"] = float(min(speedups))
        report["max_speedup"] = float(max(speedups))
        report["cases_improved"] = sum(1 for s in speedups if s > 1.0)
        report["cases_total"] = len(speedups)
        by_rows: dict[int, list[float]] = {}
        for case in report["cases"]:
            by_rows.setdefault(case["shape"][0], []).append(case["speedup"])
        report["speedup_by_rows"] = {
            str(rows): {
                "n": len(vals),
                "min": min(vals),
                "max": max(vals),
                "geomean": float(np.exp(np.mean(np.log(vals)))),
            }
            for rows, vals in sorted(by_rows.items())
        }
        # Raw per-sample arrays are hashed and written beside the output so a
        # compact Git artifact can cite them without committing the samples.
        samples_payload = {
            f"{case['tensor']}|rows={case['shape'][0]}": case["samples_ms"]
            for case in report["cases"]
        }
        samples_bytes = json.dumps(samples_payload, indent=1).encode()
        report["raw_samples_sha256"] = hashlib.sha256(samples_bytes).hexdigest()
        samples_path = args.output.with_name(args.output.stem + ".samples.json")
        samples_path.write_bytes(samples_bytes)
        report["raw_samples_path"] = str(samples_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
