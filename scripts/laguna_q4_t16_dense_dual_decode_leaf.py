#!/usr/bin/env python3
"""Screen a decode-only T16 Q4 gate/up+SiLU owner on actual Laguna weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from typing import Callable

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_pack8_dual_silu_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
DEFAULT_CACHE = Path(
    "/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.hipengine-repacked-v1"
)
DEFAULT_OUTPUT = Path("/tmp/laguna-q4-t16-dense-dual-decode-leaf.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--burst", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _tracked_status() -> list[str]:
    output = subprocess.check_output(
        ("git", "status", "--short", "--untracked-files=no"),
        cwd=ROOT,
        text=True,
    )
    return [line for line in output.splitlines() if line]


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def _upload(runtime, values: np.ndarray):
    array = np.asarray(values)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _read_bf16(runtime, buffer, shape: tuple[int, ...]) -> np.ndarray:
    result = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(result), buffer, runtime=runtime)
    return result


def _event_ms(runtime, fn: Callable[[], None], *, burst: int) -> float:
    start = runtime.event_create()
    end = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(burst):
            fn()
        runtime.event_record(end)
        runtime.event_synchronize(end)
        return float(runtime.event_elapsed_time_ms(start, end)) / burst
    finally:
        runtime.event_destroy(end)
        runtime.event_destroy(start)


def _load_pack8(
    cache_root: Path,
    manifest: dict,
    *,
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entry = manifest["entries"][key]
    if entry["layout"] != "q4_k_pack8":
        raise ValueError(f"{key} is not q4_k_pack8")
    return tuple(
        np.load(cache_root / entry["allocations"][name]["file"], mmap_mode="r")
        for name in ("qweight", "scales", "mins")
    )


def _screen(
    *,
    runtime,
    q4_library,
    t16_library,
    x: np.ndarray,
    gate_pack8: tuple[np.ndarray, np.ndarray, np.ndarray],
    up_pack8: tuple[np.ndarray, np.ndarray, np.ndarray],
    gate_t16: np.ndarray,
    up_t16: np.ndarray,
    in_features: int,
    out_features: int,
    warmups: int,
    samples: int,
    burst: int,
) -> dict:
    buffers = []
    try:
        x_dev = _upload(runtime, x)
        gate_pack8_dev = tuple(_upload(runtime, value) for value in gate_pack8)
        up_pack8_dev = tuple(_upload(runtime, value) for value in up_pack8)
        gate_t16_dev = _upload(runtime, gate_t16)
        up_t16_dev = _upload(runtime, up_t16)
        control_out_dev = malloc(out_features * 2, runtime=runtime)
        candidate_out_dev = malloc(out_features * 2, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                *gate_pack8_dev,
                *up_pack8_dev,
                gate_t16_dev,
                up_t16_dev,
                control_out_dev,
                candidate_out_dev,
            )
        )

        def control() -> None:
            gguf_q4_k_pack8_dual_silu_bf16_bf16_out(
                x_dev.ptr,
                gate_pack8_dev[0].ptr,
                gate_pack8_dev[1].ptr,
                gate_pack8_dev[2].ptr,
                up_pack8_dev[0].ptr,
                up_pack8_dev[1].ptr,
                up_pack8_dev[2].ptr,
                control_out_dev.ptr,
                1,
                in_features,
                out_features,
                threads=32,
                library=q4_library,
                runtime=runtime,
            )

        def candidate() -> None:
            gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out(
                x_dev.ptr,
                gate_t16_dev.ptr,
                up_t16_dev.ptr,
                candidate_out_dev.ptr,
                1,
                in_features,
                out_features,
                library=t16_library,
                runtime=runtime,
            )

        launchers = {"control_pack8": control, "candidate_t16": candidate}
        for _ in range(warmups):
            control()
            candidate()
        runtime.device_synchronize()
        timings = {name: [] for name in launchers}
        candidate_wins = 0
        for sample in range(samples):
            order = (
                ("control_pack8", "candidate_t16")
                if sample % 2 == 0
                else ("candidate_t16", "control_pack8")
            )
            pair = {}
            for name in order:
                pair[name] = _event_ms(
                    runtime, launchers[name], burst=burst
                )
                timings[name].append(pair[name])
            candidate_wins += int(
                pair["candidate_t16"] < pair["control_pack8"]
            )

        control()
        candidate()
        runtime.device_synchronize()
        control_out = _read_bf16(runtime, control_out_dev, (1, out_features))
        candidate_out = _read_bf16(
            runtime, candidate_out_dev, (1, out_features)
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    medians = {
        name: statistics.median(values) for name, values in timings.items()
    }
    ratio = medians["candidate_t16"] / medians["control_pack8"]
    mismatch = candidate_out != control_out
    return {
        "samples_ms": timings,
        "median_ms": medians,
        "candidate_over_control": ratio,
        "candidate_delta_percent": (ratio - 1.0) * 100.0,
        "candidate_wins": candidate_wins,
        "pair_count": samples,
        "bf16_mismatches": int(np.count_nonzero(mismatch)),
        "bf16_max_ulp": int(
            np.max(
                np.abs(
                    candidate_out.astype(np.int32)
                    - control_out.astype(np.int32)
                )
            )
        ),
        "candidate_sha256": hashlib.sha256(
            candidate_out.astype("<u2").tobytes()
        ).hexdigest(),
        "control_sha256": hashlib.sha256(
            control_out.astype("<u2").tobytes()
        ).hexdigest(),
        "resident_bytes": {
            "pack8_gate_up": int(
                sum(value.nbytes for value in gate_pack8)
                + sum(value.nbytes for value in up_pack8)
            ),
            "t16_gate_up": int(gate_t16.nbytes + up_t16.nbytes),
        },
    }


def main() -> int:
    args = _parse_args()
    if _tracked_status() and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree must be clean; pass --allow-dirty for a screen"
        )
    if args.warmups < 0 or min(args.samples, args.burst) <= 0:
        raise ValueError("warmups must be non-negative; samples/burst positive")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )

    manifest = json.loads(
        (args.repacked_cache / "manifest.json").read_text(encoding="utf-8")
    )
    reader = GGUFReader(args.model)
    rng = np.random.default_rng(args.seed)
    x = _bf16_bits(
        rng.normal(0.0, 0.55, size=(1, 3072)).astype(np.float32)
    )
    runtime = get_hip_runtime()
    q4_library = build_gguf_q4_k_gemv(
        load=True,
        require_cached=args.require_cached_build,
    )
    t16_library = build_gguf_t16_selected_gemv(
        load=True,
        require_cached=args.require_cached_build,
    )
    cases = {
        "shared_m1_k3072_n1024": (
            "layers.1.ffn_gate_shexp",
            "layers.1.ffn_up_shexp",
            "blk.1.ffn_gate_shexp.weight",
            "blk.1.ffn_up_shexp.weight",
            1024,
        ),
        "dense_m1_k3072_n12288": (
            "layers.0.ffn_gate",
            "layers.0.ffn_up",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            12288,
        ),
    }
    results = {}
    for name, (
        gate_key,
        up_key,
        gate_tensor,
        up_tensor,
        out_features,
    ) in cases.items():
        gate_raw = np.asarray(reader.tensor_data(gate_tensor))[None, ...]
        up_raw = np.asarray(reader.tensor_data(up_tensor))[None, ...]
        gate_t16 = repack_gguf_q4_k_tile16(gate_raw).tiles
        up_t16 = repack_gguf_q4_k_tile16(up_raw).tiles
        results[name] = _screen(
            runtime=runtime,
            q4_library=q4_library,
            t16_library=t16_library,
            x=x,
            gate_pack8=_load_pack8(
                args.repacked_cache, manifest, key=gate_key
            ),
            up_pack8=_load_pack8(
                args.repacked_cache, manifest, key=up_key
            ),
            gate_t16=gate_t16,
            up_t16=up_t16,
            in_features=3072,
            out_features=out_features,
            warmups=args.warmups,
            samples=args.samples,
            burst=args.burst,
        )

    exact = all(row["bf16_mismatches"] == 0 for row in results.values())
    all_positive = all(
        row["candidate_over_control"] < 1.0 for row in results.values()
    )
    artifact = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_q4_t16_dense_dual_decode_leaf",
        "status": "candidate" if exact and all_positive else "rejected",
        "performance_claim": False,
        "repo_revision": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip(),
        "hardware": {
            "gpu": "AMD Radeon 8060S Graphics",
            "architecture": "gfx1151",
        },
        "protocol": {
            "warmups": args.warmups,
            "samples": args.samples,
            "burst": args.burst,
            "timing": "counterbalanced HIP-event elapsed time",
            "weights": "actual layer-0 dense and layer-1 shared Q4_K",
            "activation": "deterministic synthetic BF16 post-norm-shaped row",
            "control": "resident pack8 fused dual gate/up plus SiLU",
            "candidate": "decode-only T16 local32 fused dual gate/up plus SiLU",
        },
        "results": results,
        "correctness_pass": exact,
        "all_shapes_positive": all_positive,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
