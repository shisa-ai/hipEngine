#!/usr/bin/env python3
"""R8 original-product full FFN leaf: current Q4 versus Kairic PFS IU4.

The candidate uses the immutable published FFN sidecar, block-Hadamard U4
packing, combined gate/up IU4 projection, fused SwiGLU+Hadamard packing, and IU4
down projection. The control is the current operation-complete Q4 pack8 gate/up
+ SiLU followed by the current gfx1151 Q4 pack8 down projection.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import DeviceBuffer, free, malloc, memory_stats
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
    build_gguf_q4_k_prefill,
    gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_bf16_out,
    gguf_q4_k_pack8_wmma_prefill_gfx1151_bf16_bf16_out,
    plan_gguf_q4_k_prefill_build,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    build_gguf_k_t16_selected_prefill,
    gguf_q5_k_t16_wmma_prefill_bf16_bf16_out,
    plan_gguf_k_t16_selected_prefill_build,
)
from hipengine.kernels.hip_gfx1151.quant.iu4_s4_ffn_product import (
    build_iu4_s4_ffn_product,
    iu4_pfs_linear_bf16_out,
    iu4_pfs_pack_gate_bf16,
    iu4_pfs_pack_swiglu_down_bf16,
    iu4_pfs_packed_nbytes,
    plan_iu4_s4_ffn_product_build,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8
from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16
from hipengine.quant.iu4_ffn_pfs import (
    KAIRIC_QWEN38_FFN_REVISION,
    KAIRIC_QWEN38_FFN_SHA256,
    open_kairic_qwen38_ffn,
    pfs_s4_to_n16_k32_tiles,
)
from hipengine.quant.iu4_s4 import bf16_bits_to_f32, f32_to_bf16_bits
from scripts.qwen38_iu4_prefill_gate_up_leaf import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    _event_ms,
    _read_bf16,
    _sha256_array,
    _timing_summary,
    _tracked_status,
    _upload,
)
from scripts.qwen38_iu4_s4_gate_up_leaf import _softmax_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = (64, 128, 256, 512)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--pfs", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--rows", default=",".join(str(value) for value in DEFAULT_ROWS))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0xF8F4)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _screen_rows(
    *,
    rows: int,
    hidden: int,
    intermediate: int,
    runtime,
    control_library,
    control_q5_library,
    candidate_library,
    control_gate: tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer],
    control_up: tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer],
    control_down: tuple[DeviceBuffer, ...],
    control_down_kind: str,
    candidate_gate: tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer],
    candidate_down: tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer],
    warmups: int,
    samples: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + rows)
    x = f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, hidden)).astype(np.float32)
    )
    buffers: list[DeviceBuffer] = []
    try:
        x_dev = _upload(runtime, x)
        control_swiglu_dev = malloc(rows * intermediate * 2, runtime=runtime)
        control_out_dev = malloc(rows * hidden * 2, runtime=runtime)
        gate_packed_dev = malloc(iu4_pfs_packed_nbytes(rows, hidden), runtime=runtime)
        gate_scale_dev = malloc(rows * 4, runtime=runtime)
        gate_zero_dev = malloc(rows * 4, runtime=runtime)
        candidate_gate_up_dev = malloc(rows * 2 * intermediate * 2, runtime=runtime)
        down_packed_dev = malloc(iu4_pfs_packed_nbytes(rows, intermediate), runtime=runtime)
        down_scale_dev = malloc(rows * 4, runtime=runtime)
        down_zero_dev = malloc(rows * 4, runtime=runtime)
        candidate_out_dev = malloc(rows * hidden * 2, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                control_swiglu_dev,
                control_out_dev,
                gate_packed_dev,
                gate_scale_dev,
                gate_zero_dev,
                candidate_gate_up_dev,
                down_packed_dev,
                down_scale_dev,
                down_zero_dev,
                candidate_out_dev,
            )
        )

        def control_gate_up() -> None:
            gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_bf16_out(
                x_dev.ptr,
                control_gate[0].ptr,
                control_gate[1].ptr,
                control_gate[2].ptr,
                control_up[0].ptr,
                control_up[1].ptr,
                control_up[2].ptr,
                control_swiglu_dev.ptr,
                rows,
                hidden,
                intermediate,
                library=control_library,
                runtime=runtime,
            )

        def control_down_project() -> None:
            if control_down_kind == "Q4_K":
                gguf_q4_k_pack8_wmma_prefill_gfx1151_bf16_bf16_out(
                    control_swiglu_dev.ptr,
                    control_down[0].ptr,
                    control_down[1].ptr,
                    control_down[2].ptr,
                    control_out_dev.ptr,
                    rows,
                    intermediate,
                    hidden,
                    library=control_library,
                    runtime=runtime,
                )
            else:
                gguf_q5_k_t16_wmma_prefill_bf16_bf16_out(
                    control_swiglu_dev.ptr,
                    control_down[0].ptr,
                    control_out_dev.ptr,
                    rows,
                    intermediate,
                    hidden,
                    library=control_q5_library,
                    runtime=runtime,
                )

        def control() -> None:
            control_gate_up()
            control_down_project()

        def candidate_gate_pack() -> None:
            iu4_pfs_pack_gate_bf16(
                x_dev.ptr,
                gate_packed_dev.ptr,
                gate_scale_dev.ptr,
                gate_zero_dev.ptr,
                rows,
                hidden,
                library=candidate_library,
                runtime=runtime,
            )

        def candidate_gate_project() -> None:
            iu4_pfs_linear_bf16_out(
                gate_packed_dev.ptr,
                gate_scale_dev.ptr,
                gate_zero_dev.ptr,
                candidate_gate[0].ptr,
                candidate_gate[1].ptr,
                candidate_gate[2].ptr,
                candidate_gate_up_dev.ptr,
                rows,
                hidden,
                2 * intermediate,
                library=candidate_library,
                runtime=runtime,
            )

        def candidate_down_pack() -> None:
            iu4_pfs_pack_swiglu_down_bf16(
                candidate_gate_up_dev.ptr,
                down_packed_dev.ptr,
                down_scale_dev.ptr,
                down_zero_dev.ptr,
                rows,
                intermediate,
                library=candidate_library,
                runtime=runtime,
            )

        def candidate_down_project() -> None:
            iu4_pfs_linear_bf16_out(
                down_packed_dev.ptr,
                down_scale_dev.ptr,
                down_zero_dev.ptr,
                candidate_down[0].ptr,
                candidate_down[1].ptr,
                candidate_down[2].ptr,
                candidate_out_dev.ptr,
                rows,
                intermediate,
                hidden,
                library=candidate_library,
                runtime=runtime,
            )

        def candidate() -> None:
            candidate_gate_pack()
            candidate_gate_project()
            candidate_down_pack()
            candidate_down_project()

        for _ in range(warmups):
            control()
            candidate()
        runtime.device_synchronize()

        timings: dict[str, list[float]] = {"control": [], "candidate": []}
        candidate_wins = 0
        for sample in range(samples):
            order: list[tuple[str, Callable[[], None]]] = [
                ("control", control),
                ("candidate", candidate),
            ]
            if sample & 1:
                order.reverse()
            pair: dict[str, float] = {}
            for name, launch in order:
                pair[name] = _event_ms(runtime, launch)
                timings[name].append(pair[name])
            candidate_wins += int(pair["candidate"] < pair["control"])

        candidate_gate_pack()
        candidate_gate_project()
        candidate_down_pack()
        runtime.device_synchronize()
        stage_launches = {
            "control_gate_up_silu": control_gate_up,
            "control_down": control_down_project,
            "candidate_gate_pack": candidate_gate_pack,
            "candidate_gate_project": candidate_gate_project,
            "candidate_swiglu_down_pack": candidate_down_pack,
            "candidate_down_project": candidate_down_project,
        }
        stage_timings = {
            name: _timing_summary([_event_ms(runtime, launch) for _ in range(samples)])
            for name, launch in stage_launches.items()
        }

        control()
        candidate()
        runtime.device_synchronize()
        control_bits = _read_bf16(runtime, control_out_dev, (rows, hidden))
        candidate_bits = _read_bf16(runtime, candidate_out_dev, (rows, hidden))
        candidate_gate_up_bits = _read_bf16(
            runtime, candidate_gate_up_dev, (rows, 2 * intermediate)
        )
        candidate()
        runtime.device_synchronize()
        candidate_repeat_bits = _read_bf16(
            runtime, candidate_out_dev, (rows, hidden)
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    control_summary = _timing_summary(timings["control"])
    candidate_summary = _timing_summary(timings["candidate"])
    control_ms = float(control_summary["median_ms"])
    candidate_ms = float(candidate_summary["median_ms"])
    return {
        "rows": rows,
        "timing": {
            "control_inclusive": control_summary,
            "candidate_inclusive": candidate_summary,
            "stages": stage_timings,
            "candidate_wins": candidate_wins,
            "pair_count": samples,
        },
        "performance": {
            "inclusive_speedup": control_ms / candidate_ms,
            "inclusive_delta_percent": (candidate_ms / control_ms - 1.0) * 100.0,
            "candidate_tokens_per_second_equivalent": 1000.0 * rows / candidate_ms,
            "control_tokens_per_second_equivalent": 1000.0 * rows / control_ms,
        },
        "correctness_vs_current_control": _softmax_metrics(
            bf16_bits_to_f32(control_bits),
            bf16_bits_to_f32(candidate_bits),
        ),
        "candidate_gate_up_finite": bool(
            np.isfinite(bf16_bits_to_f32(candidate_gate_up_bits)).all()
        ),
        "candidate_deterministic_bits": bool(
            np.array_equal(candidate_bits, candidate_repeat_bits)
        ),
        "control_output_sha256": _sha256_array(control_bits),
        "candidate_output_sha256": _sha256_array(candidate_bits),
    }


def main() -> int:
    args = _parse_args()
    rows_values = tuple(int(value) for value in args.rows.split(",") if value)
    if rows_values != DEFAULT_ROWS:
        raise ValueError(f"rows must be exactly {DEFAULT_ROWS} for the R8 leaf")
    if args.layer < 0 or args.warmups < 0 or args.samples <= 0:
        raise ValueError("layer/warmups non-negative; samples positive")
    compiler_version = None
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8").strip()

    reader = GGUFReader(args.model)
    names = {
        "gate": f"blk.{args.layer}.ffn_gate.weight",
        "up": f"blk.{args.layer}.ffn_up.weight",
        "down": f"blk.{args.layer}.ffn_down.weight",
    }
    infos = {role: reader.tensor_info(name) for role, name in names.items()}
    if any(infos[role].ggml_type_name != "Q4_K" or len(infos[role].shape) != 2 for role in ("gate", "up")):
        raise ValueError("R8 control requires rank-2 Q4_K gate/up")
    down_kind = infos["down"].ggml_type_name
    if down_kind not in ("Q4_K", "Q5_K") or len(infos["down"].shape) != 2:
        raise ValueError("R8 control down must be rank-2 Q4_K or Q5_K")
    if infos["gate"].shape != infos["up"].shape:
        raise ValueError("gate/up tensor shapes differ")
    intermediate, hidden = (int(value) for value in infos["gate"].shape)
    if tuple(int(value) for value in infos["down"].shape) != (hidden, intermediate):
        raise ValueError("down tensor must transpose gate/up geometry")

    raw = {role: np.asarray(reader.tensor_data(name)) for role, name in names.items()}
    control_host = {
        "gate": repack_gguf_q4_k_pack8(raw["gate"]),
        "up": repack_gguf_q4_k_pack8(raw["up"]),
        "down": (
            repack_gguf_q4_k_pack8(raw["down"])
            if down_kind == "Q4_K"
            else repack_gguf_q5_k_tile16(raw["down"][None, ...])
        ),
    }
    runtime = get_hip_runtime()
    before_bytes = memory_stats()["current_allocated_bytes"]
    require_cached = bool(args.require_cached_build)
    control_library = build_gguf_q4_k_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached,
    )
    control_q5_library = build_gguf_k_t16_selected_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached,
    )
    candidate_library = build_iu4_s4_ffn_product(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached,
    )
    control_plan = plan_gguf_q4_k_prefill_build(compiler_version=compiler_version)
    control_q5_plan = plan_gguf_k_t16_selected_prefill_build(compiler_version=compiler_version)
    candidate_plan = plan_iu4_s4_ffn_product_build(compiler_version=compiler_version)
    persistent: list[DeviceBuffer] = []
    started = time.perf_counter()
    with open_kairic_qwen38_ffn(args.pfs, verify_sha256=True) as sidecar:
        product = sidecar.layer(args.layer)
        try:
            control_device = {
                "gate": tuple(
                    _upload(runtime, array)
                    for array in (
                        control_host["gate"].qweight,
                        control_host["gate"].scales,
                        control_host["gate"].mins,
                    )
                ),
                "up": tuple(
                    _upload(runtime, array)
                    for array in (
                        control_host["up"].qweight,
                        control_host["up"].scales,
                        control_host["up"].mins,
                    )
                ),
                "down": (
                    tuple(
                        _upload(runtime, array)
                        for array in (
                            control_host["down"].qweight,
                            control_host["down"].scales,
                            control_host["down"].mins,
                        )
                    )
                    if down_kind == "Q4_K"
                    else (_upload(runtime, control_host["down"].tiles),)
                ),
            }
            candidate_gate = tuple(
                _upload(runtime, array)
                for array in (
                    pfs_s4_to_n16_k32_tiles(product.gate_weight),
                    product.gate_scales,
                    product.gate_sums,
                )
            )
            candidate_down = tuple(
                _upload(runtime, array)
                for array in (
                    pfs_s4_to_n16_k32_tiles(product.down_weight),
                    product.down_scales,
                    product.down_sums,
                )
            )
            for value in control_device.values():
                persistent.extend(value)
            persistent.extend((*candidate_gate, *candidate_down))
            results = [
                _screen_rows(
                    rows=rows,
                    hidden=hidden,
                    intermediate=intermediate,
                    runtime=runtime,
                    control_library=control_library,
                    control_q5_library=control_q5_library,
                    candidate_library=candidate_library,
                    control_gate=control_device["gate"],
                    control_up=control_device["up"],
                    control_down=control_device["down"],
                    control_down_kind=down_kind,
                    candidate_gate=candidate_gate,
                    candidate_down=candidate_down,
                    warmups=args.warmups,
                    samples=args.samples,
                    seed=args.seed,
                )
                for rows in rows_values
            ]
        finally:
            for buffer in reversed(persistent):
                free(buffer, runtime=runtime)
    runtime.device_synchronize()
    after_bytes = memory_stats()["current_allocated_bytes"]

    all_faster = all(float(row["performance"]["inclusive_speedup"]) > 1.0 for row in results)
    all_finite = all(
        bool(row["correctness_vs_current_control"]["finite"])
        and bool(row["candidate_gate_up_finite"])
        for row in results
    )
    all_deterministic = all(bool(row["candidate_deterministic_bits"]) for row in results)
    teardown = before_bytes == after_bytes
    artifact = {
        "schema_version": 1,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "kind": "qwen38_gfx1151_kairic_pfs_iu4_full_ffn_leaf_ab",
        "status": (
            "speed_screen_passed_quality_pending"
            if all_faster and all_finite and all_deterministic and teardown
            else "diagnostic_rejected"
        ),
        "performance_claim": False,
        "scope": "R8 original-product layer-local full FFN leaf; no runtime route",
        "hardware": {
            "hostname": platform.node(),
            "cpu": "AMD Ryzen AI MAX+ 395",
            "gpu": "AMD Radeon 8060S Graphics",
            "arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        },
        "software": {
            "commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "tracked_dirty_paths": _tracked_status(),
            "require_cached_build": require_cached,
            "compiler_version_file": str(args.compiler_version_file) if args.compiler_version_file else None,
            "control_build": {
                "cache_key": control_plan.cache_key,
                "output": str(control_plan.output_path),
                "command": list(control_plan.command),
            },
            "control_q5_build": {
                "cache_key": control_q5_plan.cache_key,
                "output": str(control_q5_plan.output_path),
                "command": list(control_q5_plan.command),
            },
            "candidate_build": {
                "cache_key": candidate_plan.cache_key,
                "output": str(candidate_plan.output_path),
                "command": list(candidate_plan.command),
            },
        },
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "size_bytes": args.model.stat().st_size,
            "quant": "Q4_K_S",
            "layer": args.layer,
            "shape": {"hidden": hidden, "intermediate": intermediate},
            "control_tensors": names,
            "control_down_quant": down_kind,
        },
        "product": {
            "quant": "iu4_s4_kairic_ffn_v1",
            "pfs_path": str(args.pfs.resolve()),
            "pfs_sha256": KAIRIC_QWEN38_FFN_SHA256,
            "pfs_revision": KAIRIC_QWEN38_FFN_REVISION,
            "activation_transform": "block_hadamard1024; gate seed 0xA511E9B3; down seed 0x63D83595",
            "gate_layout": "combined gate/up S4 [2I,H] with per-output scale/sum",
            "down_layout": "S4 [H,I] with per-output scale/sum",
        },
        "protocol": {
            "rows": list(rows_values),
            "warmups": args.warmups,
            "samples": args.samples,
            "timing": "counterbalanced HIP events; full gate/up/SwiGLU/down operation",
            "seed": args.seed,
            "elapsed_seconds_including_hash_and_setup": time.perf_counter() - started,
        },
        "results": results,
        "gates": {
            "all_shapes_inclusive_faster": all_faster,
            "all_finite": all_finite,
            "all_deterministic": all_deterministic,
            "teardown_exact": teardown,
            "full_model_quality_run": False,
            "runtime_integration_authorized": False,
        },
        "memory": {
            "tracked_current_before_bytes": before_bytes,
            "tracked_current_after_bytes": after_bytes,
            "teardown_exact": teardown,
        },
        "command": " ".join(
            [
                f"HIPENGINE_HIP_ARCH={os.environ.get('HIPENGINE_HIP_ARCH', '')}",
                "PYTHONPATH=.",
                Path(os.sys.executable).name,
                *os.sys.argv,
            ]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "rows": [
                    {
                        "m": row["rows"],
                        "control_ms": row["timing"]["control_inclusive"]["median_ms"],
                        "candidate_ms": row["timing"]["candidate_inclusive"]["median_ms"],
                        "speedup": row["performance"]["inclusive_speedup"],
                        "wins": row["timing"]["candidate_wins"],
                        "nrmse": row["correctness_vs_current_control"]["normalized_rmse"],
                    }
                    for row in results
                ],
            },
            indent=2,
        )
    )
    return 0 if all_finite and all_deterministic and teardown else 1


if __name__ == "__main__":
    raise SystemExit(main())
