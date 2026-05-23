#!/usr/bin/env python3
"""Microbenchmark the native DFlash BF16 dense kernels.

This is a small W7900/RDNA3 sanity tool for the R3.3 drafter-dense roofline.
It intentionally benchmarks only the existing naive kernels; it is not a
correctness test and does not load model weights.  Inputs are random BF16 bit
patterns because the measured quantity is kernel wall time / effective bandwidth.
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.backends import hip_target_arch_environment
from hipengine.kernels.hip_gfx1100.speculative import build_dflash_drafter
from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_dense_bf16_to_bf16, dflash_dense_bf16_to_f32

DEFAULT_SHAPES: tuple[tuple[str, int, int, int], ...] = (
    ("bf16", 16, 2048, 2048),
    ("f32", 16, 2048, 2048),
    ("bf16", 16, 2048, 6144),
    ("bf16", 16, 6144, 2048),
    ("bf16", 16, 2048, 512),
    ("f32", 16, 2048, 512),
)


def _git_context() -> dict[str, Any]:
    def run(cmd: list[str]) -> str | None:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True, timeout=10)
        return proc.stdout.strip() if proc.returncode == 0 else None

    status = run(["git", "status", "--porcelain"])
    return {
        "hipengine_commit": run(["git", "rev-parse", "HEAD"]),
        "hipengine_branch": run(["git", "branch", "--show-current"]),
        "hipengine_dirty": bool(status),
        "hipengine_status_porcelain": status,
    }


def _dev(runtime, buffers, array: np.ndarray):
    contiguous = np.ascontiguousarray(array)
    buf = malloc(contiguous.nbytes, runtime=runtime)
    buffers.append(buf)
    copy_host_to_device(buf, host_array_ptr(contiguous), runtime=runtime)
    return buf


def _empty(runtime, buffers, nbytes: int):
    buf = malloc(int(nbytes), runtime=runtime)
    buffers.append(buf)
    return buf


def _bench_one(*, library, runtime, kind: str, rows: int, in_features: int, out_features: int, loops: int, warmup: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + rows + in_features + out_features + (0 if kind == "bf16" else 1))
    x = rng.integers(0, 65535, size=(rows, in_features), dtype=np.uint16)
    weight = rng.integers(0, 65535, size=(out_features, in_features), dtype=np.uint16)
    out_itemsize = 2 if kind == "bf16" else 4
    buffers = []
    try:
        x_dev = _dev(runtime, buffers, x)
        weight_dev = _dev(runtime, buffers, weight)
        out_dev = _empty(runtime, buffers, rows * out_features * out_itemsize)
        fn = dflash_dense_bf16_to_bf16 if kind == "bf16" else dflash_dense_bf16_to_f32
        for _ in range(warmup):
            fn(x_dev.ptr, weight_dev.ptr, out_dev.ptr, rows, in_features, out_features, threads=128, library=library, runtime=runtime)
        runtime.device_synchronize()
        times_ms: list[float] = []
        for _ in range(loops):
            t0 = time.perf_counter()
            fn(x_dev.ptr, weight_dev.ptr, out_dev.ptr, rows, in_features, out_features, threads=128, library=library, runtime=runtime)
            runtime.device_synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = float(sum(times_ms) / len(times_ms))
        weight_bytes = int(out_features) * int(in_features) * 2
        input_bytes = int(rows) * int(in_features) * 2
        output_bytes = int(rows) * int(out_features) * out_itemsize
        flops = 2 * int(rows) * int(in_features) * int(out_features)
        return {
            "kind": kind,
            "rows": int(rows),
            "in_features": int(in_features),
            "out_features": int(out_features),
            "loops": int(loops),
            "warmup": int(warmup),
            "mean_ms": mean_ms,
            "min_ms": float(min(times_ms)),
            "max_ms": float(max(times_ms)),
            "times_ms": [float(x) for x in times_ms],
            "bytes": {
                "input": input_bytes,
                "weight": weight_bytes,
                "output": output_bytes,
                "total_nominal": input_bytes + weight_bytes + output_bytes,
            },
            "flops": flops,
            "effective_weight_gb_s_mean": (weight_bytes / 1e9) / (mean_ms / 1000.0),
            "effective_nominal_gb_s_mean": ((input_bytes + weight_bytes + output_bytes) / 1e9) / (mean_ms / 1000.0),
            "effective_tflops_mean": (flops / 1e12) / (mean_ms / 1000.0),
        }
    finally:
        for buf in reversed(buffers):
            free(buf, runtime=runtime)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hardware-gpu", default="AMD Radeon Pro W7900")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.loops <= 0 or args.warmup < 0:
        raise ValueError("--loops must be positive and --warmup must be non-negative")
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8") if args.compiler_version_file else None
    runtime = get_hip_runtime()
    with hip_target_arch_environment("gfx1100"):
        library = build_dflash_drafter(load=True, compiler_version=compiler_version)
    rows = [
        _bench_one(
            library=library,
            runtime=runtime,
            kind=kind,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            loops=args.loops,
            warmup=args.warmup,
            seed=args.seed,
        )
        for kind, rows, in_features, out_features in DEFAULT_SHAPES
    ]
    artifact = {
        "run_tag": "dflash-dense-microbench",
        "summary": "Native DFlash naive BF16 dense kernel microbench for R3.3 roofline sanity",
        "status": "diagnostic",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {"backend": "hip_gfx1100", "arch": "gfx1100", "gpu": args.hardware_gpu},
        "software": {**_git_context(), "python": platform.python_version(), "platform": platform.platform(), "hipcc_version": compiler_version},
        "workload": {"rows": 16, "shapes": [list(shape) for shape in DEFAULT_SHAPES], "loops": args.loops, "warmup": args.warmup},
        "measurements": {"rows": rows},
        "commands": {"benchmark": " ".join(shlex.quote(part) for part in ["python3", "scripts/dflash_dense_microbench.py", *(argv if argv is not None else sys.argv[1:])])},
        "notes": [
            "Random BF16 bit-pattern inputs; timing only, not a correctness test.",
            "Effective GB/s uses weight bytes as the primary roofline denominator because weights dominate these small-row GEMV shapes.",
        ],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({f"{row['kind']}:{row['rows']}x{row['in_features']}x{row['out_features']}": row["mean_ms"] for row in rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
