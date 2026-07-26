#!/usr/bin/env python3
"""Compare an exact byte-neutral Q4_K layout with retained T16 on gfx1151.

The benchmark temporarily holds one actual layer's candidate and T16 gate/up
pairs side by side for counterbalanced timing. Candidate residency is the
byte-neutral pair alone; the comparison does not propose keeping both layouts
resident.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Callable

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
    reset_memory_stats,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_x8_selected_gemv import (
    build_gguf_x8_selected_gemv,
    gguf_q4_k_qmicro_selected_dual_exact_gemv_bf16_bf16_out,
    gguf_q4_k_x8_selected_dual_exact_gemv_bf16_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16_qmicro
from hipengine.quant.gguf_x8 import repack_gguf_q4_k_x8

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
DEFAULT_CACHE = Path(
    "/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.hipengine-repacked-v1"
)
DEFAULT_OUTPUT = Path("/tmp/laguna-q4-k-x8-exact-decode.json")
MODEL_SHA256 = "7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f"
HIDDEN = 3_072
OUT_FEATURES = 1_024
EXPERTS = 256
TOP_K = 10
Q4_K_BLOCK_BYTES = 144
Q4_T16_BLOCK_BYTES = 2_368


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        )
    if tuple(sorted(set(parsed))) != parsed:
        raise argparse.ArgumentTypeError("values must be sorted and unique")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument(
        "--producer-rows",
        type=_parse_csv_ints,
        default=(1, 2, 4, 8),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--burst", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--candidate",
        choices=("x8", "qmicro"),
        default="x8",
    )
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _git_revision() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        text=True,
    ).strip()


def _tracked_status() -> list[str]:
    out = subprocess.check_output(
        ("git", "status", "--short", "--untracked-files=no"),
        cwd=ROOT,
        text=True,
    )
    return [line for line in out.splitlines() if line]


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    return (
        np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    ).view(np.float32)


def _selected_experts(
    producer_rows: int,
    *,
    layer: int,
    top_k: int = TOP_K,
    experts: int = EXPERTS,
) -> np.ndarray:
    if producer_rows <= 0 or top_k <= 0 or experts < top_k:
        raise ValueError("invalid selected-expert shape")
    source = np.arange(producer_rows, dtype=np.int64)[:, None]
    slot = np.arange(top_k, dtype=np.int64)[None, :]
    selected = (
        source * (top_k * 17 + 1) + slot * 37 + layer * 13 + 11
    ) % experts
    if any(np.unique(row).size != top_k for row in selected):
        raise AssertionError("each producer row must select distinct experts")
    return np.ascontiguousarray(selected.reshape(-1))


def _cache_tiles(
    cache_root: Path,
    *,
    layer: int,
    slot: str,
) -> tuple[np.ndarray, dict]:
    manifest = json.loads(
        (cache_root / "manifest.json").read_text(encoding="utf-8")
    )
    entry = manifest["entries"][f"layers.{layer}.{slot}"]
    if entry["layout"] != "gguf_q4_k_t16_v1":
        raise ValueError(f"{slot} does not use Q4_K T16")
    allocation = entry["allocations"]["tiles"]
    tiles = np.load(cache_root / allocation["file"], mmap_mode="r")
    expected = (
        EXPERTS,
        OUT_FEATURES // 16,
        HIDDEN // 256,
        Q4_T16_BLOCK_BYTES,
    )
    if tuple(tiles.shape) != expected or tiles.dtype != np.uint8:
        raise ValueError(f"unexpected {slot} T16 payload: {tiles.shape}")
    return tiles, entry


def _upload(runtime, values: np.ndarray):
    array = np.asarray(values)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _event_ms(
    runtime,
    fn: Callable[[], None],
    *,
    burst: int,
) -> float:
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


def _copy_output(runtime, buffer, shape: tuple[int, int]) -> np.ndarray:
    host = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(host), buffer, runtime=runtime)
    return host


def main() -> None:
    args = _parse_args()
    tracked_status = _tracked_status()
    if tracked_status and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree must be clean; pass --allow-dirty only for a smoke"
        )
    if args.layer <= 0:
        raise SystemExit("--layer must name a sparse layer above zero")
    if args.warmups < 0 or args.samples <= 0 or args.burst <= 0:
        raise SystemExit(
            "warmups must be non-negative; samples/burst must be positive"
        )
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )

    reader = GGUFReader(args.model)
    raw_gate = np.asarray(
        reader.tensor_data(f"blk.{args.layer}.ffn_gate_exps.weight")
    )
    raw_up = np.asarray(
        reader.tensor_data(f"blk.{args.layer}.ffn_up_exps.weight")
    )
    expected_raw_shape = (
        EXPERTS,
        OUT_FEATURES,
        HIDDEN * Q4_K_BLOCK_BYTES // 256,
    )
    if (
        raw_gate.shape != expected_raw_shape
        or raw_up.shape != expected_raw_shape
        or raw_gate.dtype != np.uint8
        or raw_up.dtype != np.uint8
    ):
        raise ValueError(
            f"expected actual Q4_K expert shape {expected_raw_shape}, got "
            f"{raw_gate.shape}/{raw_up.shape}"
        )

    if args.candidate == "qmicro":
        candidate_gate = repack_gguf_q4_k_tile16_qmicro(raw_gate).tiles
        candidate_up = repack_gguf_q4_k_tile16_qmicro(raw_up).tiles
        candidate_wrapper = (
            gguf_q4_k_qmicro_selected_dual_exact_gemv_bf16_bf16_out
        )
        candidate_layout = "gguf_q4_k_tile16_qmicro"
    else:
        candidate_gate = repack_gguf_q4_k_x8(raw_gate).tiles
        candidate_up = repack_gguf_q4_k_x8(raw_up).tiles
        candidate_wrapper = (
            gguf_q4_k_x8_selected_dual_exact_gemv_bf16_bf16_out
        )
        candidate_layout = "gguf_q4_k_x8_v1"
    t16_gate, gate_entry = _cache_tiles(
        args.repacked_cache,
        layer=args.layer,
        slot="ffn_gate_exps",
    )
    t16_up, up_entry = _cache_tiles(
        args.repacked_cache,
        layer=args.layer,
        slot="ffn_up_exps",
    )

    runtime = get_hip_runtime()
    x8_library = build_gguf_x8_selected_gemv(
        load=True,
        require_cached=args.require_cached_build,
    )
    t16_library = build_gguf_t16_selected_gemv(
        load=True,
        require_cached=args.require_cached_build,
    )
    reset_memory_stats()
    resident_buffers = []
    results: dict[str, dict] = {}
    try:
        candidate_gate_dev = _upload(runtime, candidate_gate)
        candidate_up_dev = _upload(runtime, candidate_up)
        t16_gate_dev = _upload(runtime, t16_gate)
        t16_up_dev = _upload(runtime, t16_up)
        resident_buffers.extend(
            (
                candidate_gate_dev,
                candidate_up_dev,
                t16_gate_dev,
                t16_up_dev,
            )
        )

        for producer_rows in args.producer_rows:
            routes = producer_rows * TOP_K
            rng = np.random.default_rng(
                args.seed + args.layer * 1_000 + producer_rows
            )
            x_bits = _bf16_bits(
                rng.normal(
                    0.0,
                    0.55,
                    size=(producer_rows, HIDDEN),
                ).astype(np.float32)
            )
            selected = _selected_experts(
                producer_rows,
                layer=args.layer,
            )
            shape = (routes, OUT_FEATURES)
            output_bytes = routes * OUT_FEATURES * np.dtype(np.uint16).itemsize
            shape_buffers = []
            try:
                x_dev = _upload(runtime, x_bits)
                selected_dev = _upload(runtime, selected)
                out_a_dev = malloc(output_bytes, runtime=runtime)
                out_b_dev = malloc(output_bytes, runtime=runtime)
                shape_buffers.extend(
                    (x_dev, selected_dev, out_a_dev, out_b_dev)
                )

                def launch_t16() -> None:
                    gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out(
                        x_dev.ptr,
                        selected_dev.ptr,
                        t16_gate_dev.ptr,
                        t16_up_dev.ptr,
                        out_a_dev.ptr,
                        out_b_dev.ptr,
                        producer_rows,
                        routes,
                        EXPERTS,
                        HIDDEN,
                        OUT_FEATURES,
                        library=t16_library,
                        runtime=runtime,
                    )

                def launch_candidate() -> None:
                    candidate_wrapper(
                        x_dev.ptr,
                        selected_dev.ptr,
                        candidate_gate_dev.ptr,
                        candidate_up_dev.ptr,
                        out_a_dev.ptr,
                        out_b_dev.ptr,
                        producer_rows,
                        routes,
                        EXPERTS,
                        HIDDEN,
                        OUT_FEATURES,
                        library=x8_library,
                        runtime=runtime,
                    )

                launchers = {
                    "t16": launch_t16,
                    "candidate_exact": launch_candidate,
                }
                for _ in range(args.warmups):
                    launch_t16()
                    launch_candidate()
                runtime.device_synchronize()

                samples = {"t16": [], "candidate_exact": []}
                for sample in range(args.samples):
                    order = (
                        ("t16", "candidate_exact")
                        if sample % 2 == 0
                        else ("candidate_exact", "t16")
                    )
                    for mode in order:
                        samples[mode].append(
                            _event_ms(
                                runtime,
                                launchers[mode],
                                burst=args.burst,
                            )
                        )

                launch_t16()
                runtime.device_synchronize()
                t16_a = _copy_output(runtime, out_a_dev, shape)
                t16_b = _copy_output(runtime, out_b_dev, shape)
                launch_candidate()
                runtime.device_synchronize()
                candidate_a = _copy_output(runtime, out_a_dev, shape)
                candidate_b = _copy_output(runtime, out_b_dev, shape)
                mismatch_a = int(np.count_nonzero(t16_a != candidate_a))
                mismatch_b = int(np.count_nonzero(t16_b != candidate_b))
                medians = {
                    mode: statistics.median(values)
                    for mode, values in samples.items()
                }
                ratio = medians["candidate_exact"] / medians["t16"]
                values_f32 = np.concatenate(
                    (
                        _bf16_to_f32(candidate_a),
                        _bf16_to_f32(candidate_b),
                    ),
                    axis=1,
                )
                results[str(producer_rows)] = {
                    "producer_rows": producer_rows,
                    "selected_routes": routes,
                    "selected_sha256": hashlib.sha256(
                        selected.astype("<i8").tobytes()
                    ).hexdigest(),
                    "samples_ms": samples,
                    "median_ms": medians,
                    "candidate_over_t16": ratio,
                    "candidate_regression_percent": (ratio - 1.0) * 100.0,
                    "bf16_mismatch_gate": mismatch_a,
                    "bf16_mismatch_up": mismatch_b,
                    "finite": bool(np.isfinite(values_f32).all()),
                    "checksum_f64": float(
                        values_f32.astype(np.float64).sum()
                    ),
                }
                print(
                    f"producer_rows={producer_rows} routes={routes} "
                    f"t16={medians['t16']:.6f}ms "
                    f"{args.candidate}={medians['candidate_exact']:.6f}ms "
                    f"ratio={ratio:.6f} "
                    f"mismatch={mismatch_a + mismatch_b}",
                    flush=True,
                )
            finally:
                for buffer in reversed(shape_buffers):
                    free(buffer, runtime=runtime)
    finally:
        for buffer in reversed(resident_buffers):
            free(buffer, runtime=runtime)

    stats = memory_stats()
    exact = all(
        item["finite"]
        and item["bf16_mismatch_gate"] == 0
        and item["bf16_mismatch_up"] == 0
        for item in results.values()
    )
    within_gate = all(
        item["candidate_over_t16"] <= 1.02 for item in results.values()
    )
    artifact = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_q4_k_exact_decode_leaf",
        "status": "retained" if exact and within_gate else "rejected",
        "performance_claim": bool(exact and within_gate),
        "scope": (
            "actual layer-1 K3072/N1024 Q4_K gate/up selected exact leaf"
        ),
        "hardware": {
            "gpu": "AMD Radeon 8060S Graphics",
            "architecture": "gfx1151",
            "gpu_max_hw_queues": os.environ.get("GPU_MAX_HW_QUEUES"),
        },
        "model": {
            "path": str(args.model.resolve()),
            "sha256": MODEL_SHA256,
            "layer": args.layer,
            "gate_tensor": f"blk.{args.layer}.ffn_gate_exps.weight",
            "up_tensor": f"blk.{args.layer}.ffn_up_exps.weight",
        },
        "workload": {
            "producer_rows": list(args.producer_rows),
            "top_k": TOP_K,
            "in_features": HIDDEN,
            "out_features": OUT_FEATURES,
            "experts": EXPERTS,
            "selected_policy": (
                "deterministic distinct affine expert IDs; independent of "
                "prompt, token ID, output, or category"
            ),
        },
        "layouts": {
            "candidate_pair_bytes": int(
                candidate_gate.nbytes + candidate_up.nbytes
            ),
            "t16_pair_bytes": int(t16_gate.nbytes + t16_up.nbytes),
            "candidate_single_resident_layout": candidate_layout,
            "temporary_side_by_side_one_layer_only": True,
            "t16_entries": {"gate": gate_entry, "up": up_entry},
        },
        "protocol": {
            "warmups": args.warmups,
            "samples": args.samples,
            "burst": args.burst,
            "timing": "counterbalanced HIP-event elapsed time",
            "correctness": (
                "BF16 bit equality to retained exact T16 for gate and up; "
                "focused CPU-source primitive gate is in "
                "tests/test_gguf_x8_selected_gemv.py"
            ),
            "promotion_gate": (
                "all shapes exact and candidate/T16 <= 1.02"
            ),
            "command": shlex.join(sys.argv),
        },
        "repo": {
            "revision": _git_revision(),
            "tracked_status": tracked_status,
        },
        "memory": {
            "tracked_peak_bytes": stats["peak_allocated_bytes"],
            "tracked_after_bytes": stats["current_allocated_bytes"],
        },
        "results": results,
        "gate": {
            "exact": exact,
            "within_two_percent": within_gate,
            "passed": exact and within_gate,
        },
        "notes": [
            "This qualifies the exact candidate leaf, not full-model decode.",
            "No raw-Q4 or T16 sidecar is required by the candidate kernel.",
            "The comparison temporarily co-resides one candidate layer and T16.",
            "Runtime dispatch and materialization remain unchanged.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "gate": artifact["gate"],
                "tracked_after_bytes": stats["current_allocated_bytes"],
            },
            sort_keys=True,
        )
    )
    if not exact:
        raise SystemExit("candidate exact output differs from retained T16")
    if not within_gate:
        raise SystemExit("candidate exact leaf exceeds the 2% decode gate")


if __name__ == "__main__":
    main()
