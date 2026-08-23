#!/usr/bin/env python3
"""Operation-complete R7 IU4 gate/up prefill A/B on actual Qwen3.8 weights.

The control is the current pack8 Q4_K->FP16 WMMA gate/up+SiLU owner. The
candidate dynamically packs BF16 activations to U4 and runs the rebuilt
paired-K S4 DOT8/WMMA family, including I32 correction, scales, BF16 gate/up
publication, and SiLU. This remains a T3 speed screen: its S4 sidecar is
re-quantized from dequantized Q4_K_S rather than original BF16/F16 weights.
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
    plan_gguf_q4_k_prefill_build,
)
from hipengine.kernels.hip_gfx1151.quant.iu4_s4_sidecar import (
    build_iu4_s4_sidecar,
    iu4_s4_dual_silu_bf16_out,
    iu4_u4_quantize_bf16,
    iu4_u4_wmma_nbytes,
    plan_iu4_s4_sidecar_build,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8
from hipengine.quant.iu4_s4 import (
    bf16_bits_to_f32,
    f32_to_bf16_bits,
    pack_s4_wmma_tiles,
)
from scripts.qwen38_iu4_prefill_gate_up_leaf import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    _event_ms,
    _prefill_control_metrics,
    _read_bf16,
    _sha256_array,
    _timing_summary,
    _tracked_status,
    _upload,
)
from scripts.qwen38_iu4_s4_gate_up_leaf import (
    IU4_ARITHMETIC_ROOF_TOPS,
    _build_s4_from_q4_k,
    _softmax_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = (64, 128, 256, 512)
QWEN38_LAYERS = 64
Q4KS_PP512_TOK_S = 396.09
Q4KS_PP512_SOURCE = (
    "worklog/entries/20260817T052043.375715Z-pi-qwen38-gfx1151-"
    "q4ks-q5-prefill-shared-c116b1.md"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", default=",".join(str(value) for value in DEFAULT_ROWS))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0x1A47)
    parser.add_argument("--dequant-chunk-rows", type=int, default=128)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--target-arch", default=os.environ.get("HIPENGINE_HIP_ARCH", "gfx1151"))
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _candidate_metrics(
    *,
    rows: int,
    hidden: int,
    out_features: int,
    inclusive_ms: float,
    core_ms: float,
    pair_bytes: int,
) -> dict[str, float | int | str]:
    padded_rows = ((rows + 255) // 256) * 256
    weight_sweeps = (rows + 255) // 256
    useful_ops = 4 * rows * hidden * out_features
    executed_ops = 4 * padded_rows * hidden * out_features
    core_executed_tops = executed_ops / (core_ms / 1000.0) / 1e12
    return {
        "route": "iu4_wmma_bulk_m256",
        "accumulators_per_wave": 16,
        "padded_rows": padded_rows,
        "row_utilization": rows / padded_rows,
        "weight_sweeps": weight_sweeps,
        "pair_payload_bytes": pair_bytes,
        "executed_payload_bytes": pair_bytes * weight_sweeps,
        "useful_ops": useful_ops,
        "executed_ops": executed_ops,
        "inclusive_useful_tops": useful_ops / (inclusive_ms / 1000.0) / 1e12,
        "core_executed_tops": core_executed_tops,
        "core_effective_weight_gbps": pair_bytes * weight_sweeps / core_ms / 1e6,
        "iu4_arithmetic_roof_tops": IU4_ARITHMETIC_ROOF_TOPS,
        "fraction_of_iu4_arithmetic_roof": core_executed_tops / IU4_ARITHMETIC_ROOF_TOPS,
        "percent_of_iu4_arithmetic_roof": (
            100.0 * core_executed_tops / IU4_ARITHMETIC_ROOF_TOPS
        ),
    }


def _family_amdahl_projection(
    *,
    control_layer_ms: float,
    candidate_layer_ms: float,
    layers: int,
    prompt_tokens: int,
    complete_prefill_tok_s: float,
) -> dict[str, float | int | str]:
    complete_ms = 1000.0 * prompt_tokens / complete_prefill_tok_s
    control_family_ms = layers * control_layer_ms
    candidate_family_ms = layers * candidate_layer_ms
    projected_ms = complete_ms - control_family_ms + candidate_family_ms
    return {
        "classification": "inferred_from_leaf_times_and_prior_complete-prefill_baseline",
        "layers": layers,
        "prompt_tokens": prompt_tokens,
        "complete_prefill_tok_s": complete_prefill_tok_s,
        "complete_prefill_ms": complete_ms,
        "control_family_ms": control_family_ms,
        "candidate_family_ms": candidate_family_ms,
        "control_family_wall_share": control_family_ms / complete_ms,
        "candidate_family_wall_share": candidate_family_ms / complete_ms,
        "projected_complete_prefill_ms": projected_ms,
        "projected_complete_prefill_speedup": complete_ms / projected_ms,
        "projected_complete_prefill_tok_s": 1000.0 * prompt_tokens / projected_ms,
    }


def _screen_rows(
    *,
    rows: int,
    hidden: int,
    out_features: int,
    control_pair_bytes: int,
    candidate_pair_bytes: int,
    runtime,
    control_library,
    candidate_library,
    control_gate: tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer],
    control_up: tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer],
    candidate_gate: tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer],
    candidate_up: tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer],
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
        u4_dev = malloc(iu4_u4_wmma_nbytes(rows, hidden), runtime=runtime)
        u4_scale_dev = malloc(rows * 4, runtime=runtime)
        u4_zero_dev = malloc(rows * 4, runtime=runtime)
        control_out_dev = malloc(rows * out_features * 2, runtime=runtime)
        candidate_out_dev = malloc(rows * out_features * 2, runtime=runtime)
        buffers.extend(
            (x_dev, u4_dev, u4_scale_dev, u4_zero_dev, control_out_dev, candidate_out_dev)
        )

        def control() -> None:
            gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_bf16_out(
                x_dev.ptr,
                control_gate[0].ptr,
                control_gate[1].ptr,
                control_gate[2].ptr,
                control_up[0].ptr,
                control_up[1].ptr,
                control_up[2].ptr,
                control_out_dev.ptr,
                rows,
                hidden,
                out_features,
                library=control_library,
                runtime=runtime,
            )

        def candidate_quant() -> None:
            iu4_u4_quantize_bf16(
                x_dev.ptr,
                u4_dev.ptr,
                u4_scale_dev.ptr,
                u4_zero_dev.ptr,
                rows,
                hidden,
                library=candidate_library,
                runtime=runtime,
            )

        def candidate_core() -> None:
            iu4_s4_dual_silu_bf16_out(
                u4_dev.ptr,
                u4_scale_dev.ptr,
                u4_zero_dev.ptr,
                candidate_gate[0].ptr,
                candidate_gate[1].ptr,
                candidate_gate[2].ptr,
                candidate_up[0].ptr,
                candidate_up[1].ptr,
                candidate_up[2].ptr,
                candidate_out_dev.ptr,
                rows,
                hidden,
                out_features,
                library=candidate_library,
                runtime=runtime,
            )

        def candidate() -> None:
            candidate_quant()
            candidate_core()

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

        candidate_quant()
        runtime.device_synchronize()
        quant_samples = [_event_ms(runtime, candidate_quant) for _ in range(samples)]
        core_samples = [_event_ms(runtime, candidate_core) for _ in range(samples)]

        control()
        candidate()
        runtime.device_synchronize()
        control_bits = _read_bf16(runtime, control_out_dev, (rows, out_features))
        candidate_bits = _read_bf16(runtime, candidate_out_dev, (rows, out_features))
        candidate()
        runtime.device_synchronize()
        candidate_repeat_bits = _read_bf16(
            runtime, candidate_out_dev, (rows, out_features)
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    control_summary = _timing_summary(timings["control"])
    candidate_summary = _timing_summary(timings["candidate"])
    quant_summary = _timing_summary(quant_samples)
    core_summary = _timing_summary(core_samples)
    control_ms = float(control_summary["median_ms"])
    candidate_ms = float(candidate_summary["median_ms"])
    core_ms = float(core_summary["median_ms"])
    return {
        "rows": rows,
        "timing": {
            "control_inclusive": control_summary,
            "candidate_inclusive": candidate_summary,
            "candidate_activation_pack": quant_summary,
            "candidate_core_correction_bf16_silu": core_summary,
            "candidate_wins": candidate_wins,
            "pair_count": samples,
        },
        "performance": {
            "inclusive_speedup": control_ms / candidate_ms,
            "inclusive_delta_percent": (candidate_ms / control_ms - 1.0) * 100.0,
            "control": _prefill_control_metrics(
                rows=rows,
                hidden=hidden,
                out_features=out_features,
                median_ms=control_ms,
                pair_bytes=control_pair_bytes,
            ),
            "candidate": _candidate_metrics(
                rows=rows,
                hidden=hidden,
                out_features=out_features,
                inclusive_ms=candidate_ms,
                core_ms=core_ms,
                pair_bytes=candidate_pair_bytes,
            ),
        },
        "correctness_vs_current_control": _softmax_metrics(
            bf16_bits_to_f32(control_bits),
            bf16_bits_to_f32(candidate_bits),
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
        raise ValueError(f"rows must be exactly {DEFAULT_ROWS} for the R7 screen")
    if args.layer < 0 or args.warmups < 0 or min(args.samples, args.dequant_chunk_rows) <= 0:
        raise ValueError("layer/warmups non-negative; samples/chunk positive")
    compiler_version = None
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8").strip()

    reader = GGUFReader(args.model)
    names = {
        "gate": f"blk.{args.layer}.ffn_gate.weight",
        "up": f"blk.{args.layer}.ffn_up.weight",
    }
    infos = {role: reader.tensor_info(name) for role, name in names.items()}
    if any(info.ggml_type_name != "Q4_K" or len(info.shape) != 2 for info in infos.values()):
        raise ValueError("gate/up must be rank-2 Q4_K tensors")
    if infos["gate"].shape != infos["up"].shape:
        raise ValueError("gate/up tensor shapes differ")
    out_features, hidden = (int(value) for value in infos["gate"].shape)
    raw_gate = np.asarray(reader.tensor_data(names["gate"]))
    raw_up = np.asarray(reader.tensor_data(names["up"]))

    build_started = time.perf_counter()
    control_gate_host = repack_gguf_q4_k_pack8(raw_gate)
    control_up_host = repack_gguf_q4_k_pack8(raw_up)
    candidate_gate_host, gate_quant = _build_s4_from_q4_k(
        raw_gate,
        out_features=out_features,
        in_features=hidden,
        chunk_rows=args.dequant_chunk_rows,
    )
    candidate_up_host, up_quant = _build_s4_from_q4_k(
        raw_up,
        out_features=out_features,
        in_features=hidden,
        chunk_rows=args.dequant_chunk_rows,
    )
    host_build_seconds = time.perf_counter() - build_started
    control_pair_bytes = sum(
        array.nbytes
        for packed in (control_gate_host, control_up_host)
        for array in (packed.qweight, packed.scales, packed.mins)
    )
    candidate_pair_bytes = candidate_gate_host.nbytes + candidate_up_host.nbytes
    if min(control_pair_bytes, candidate_pair_bytes) <= 64 * 1024 * 1024:
        raise ValueError("both control and candidate pairs must exceed the cold-pool gate")

    runtime = get_hip_runtime()
    before_bytes = memory_stats()["current_allocated_bytes"]
    require_cached = bool(args.require_cached_build)
    control_library = build_gguf_q4_k_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached,
    )
    candidate_library = build_iu4_s4_sidecar(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached,
        target_arch=args.target_arch,
    )
    control_plan = plan_gguf_q4_k_prefill_build(compiler_version=compiler_version)
    candidate_plan = plan_iu4_s4_sidecar_build(
        compiler_version=compiler_version, target_arch=args.target_arch
    )
    persistent: list[DeviceBuffer] = []
    try:
        control_gate = tuple(
            _upload(runtime, array)
            for array in (
                control_gate_host.qweight,
                control_gate_host.scales,
                control_gate_host.mins,
            )
        )
        control_up = tuple(
            _upload(runtime, array)
            for array in (
                control_up_host.qweight,
                control_up_host.scales,
                control_up_host.mins,
            )
        )
        candidate_gate = (
            _upload(runtime, pack_s4_wmma_tiles(candidate_gate_host)),
            _upload(runtime, candidate_gate_host.scales),
            _upload(runtime, candidate_gate_host.sums),
        )
        candidate_up = (
            _upload(runtime, pack_s4_wmma_tiles(candidate_up_host)),
            _upload(runtime, candidate_up_host.scales),
            _upload(runtime, candidate_up_host.sums),
        )
        persistent.extend(
            (*control_gate, *control_up, *candidate_gate, *candidate_up)
        )
        results = [
            _screen_rows(
                rows=rows,
                hidden=hidden,
                out_features=out_features,
                control_pair_bytes=control_pair_bytes,
                candidate_pair_bytes=candidate_pair_bytes,
                runtime=runtime,
                control_library=control_library,
                candidate_library=candidate_library,
                control_gate=control_gate,
                control_up=control_up,
                candidate_gate=candidate_gate,
                candidate_up=candidate_up,
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
    all_pairs_win = all(
        int(row["timing"]["candidate_wins"]) == args.samples for row in results
    )
    all_finite = all(bool(row["correctness_vs_current_control"]["finite"]) for row in results)
    all_deterministic = all(bool(row["candidate_deterministic_bits"]) for row in results)
    teardown = before_bytes == after_bytes
    row512 = next(row for row in results if int(row["rows"]) == 512)
    amdahl = _family_amdahl_projection(
        control_layer_ms=float(row512["timing"]["control_inclusive"]["median_ms"]),
        candidate_layer_ms=float(row512["timing"]["candidate_inclusive"]["median_ms"]),
        layers=QWEN38_LAYERS,
        prompt_tokens=512,
        complete_prefill_tok_s=Q4KS_PP512_TOK_S,
    )

    artifact = {
        "schema_version": 1,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "kind": "qwen38_gfx1151_iu4_s4_prefill_gate_up_ab",
        "status": (
            "speed_screen_passed_unqualified_t3"
            if all_faster and all_pairs_win and all_finite and all_deterministic and teardown
            else "diagnostic_rejected"
        ),
        "performance_claim": False,
        "scope": "R7 one-layer operation-complete speed screen; no runtime route or quality promotion",
        "hardware": {
            "hostname": platform.node(),
            "cpu": "AMD Ryzen AI MAX+ 395",
            "gpu": "AMD Radeon 8060S Graphics",
            "arch": os.environ.get("HIPENGINE_HIP_ARCH"),
        },
        "software": {
            "commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "tracked_dirty_paths": _tracked_status(),
            "require_cached_build": require_cached,
            "compiler_version_file": (
                str(args.compiler_version_file) if args.compiler_version_file else None
            ),
            "control_build": {
                "cache_key": control_plan.cache_key,
                "output": str(control_plan.output_path),
                "loaded_output": str(getattr(control_library, "_name", "")),
                "command": list(control_plan.command),
            },
            "candidate_build": {
                "cache_key": candidate_plan.cache_key,
                "output": str(candidate_plan.output_path),
                "loaded_output": str(getattr(candidate_library, "_name", "")),
                "command": list(candidate_plan.command),
            },
        },
        "model": {
            "path": str(args.model.resolve()),
            "sha256": str(args.model_sha256),
            "size_bytes": args.model.stat().st_size,
            "quant": "Q4_K_S",
            "layer": args.layer,
            "gate_tensor": names["gate"],
            "up_tensor": names["up"],
            "shape": [out_features, hidden],
        },
        "representations": {
            "control": {
                "variant": "pack8_dual_wmma_prefill_bf16_bf16_out",
                "layout": "gguf_q4_k_pack8",
                "pair_bytes": control_pair_bytes,
            },
            "candidate": {
                "variant": "iu4_wmma_bulk_m256",
                "layout": "iu4_s4_sidecar_v1_n16_k32pair",
                "arithmetic_class": "T3",
                "source": "re-quantized dequantized Q4_K_S (not original BF16/F16)",
                "pair_bytes": candidate_pair_bytes,
                "gate_quantization": gate_quant,
                "up_quantization": up_quant,
            },
            "host_build_seconds": host_build_seconds,
        },
        "protocol": {
            "rows": list(rows_values),
            "warmups": args.warmups,
            "samples": args.samples,
            "timing": "counterbalanced HIP events; operation-complete on both sides",
            "cold_pool": "both actual gate/up execution views exceed 64 MiB",
            "seed": args.seed,
        },
        "results": results,
        "family_amdahl_projection": {
            **amdahl,
            "baseline_source": Q4KS_PP512_SOURCE,
            "note": "Projection only; R8 must measure the complete model and add down+quality.",
        },
        "gates": {
            "all_shapes_inclusive_faster": all_faster,
            "all_pairs_candidate_wins": all_pairs_win,
            "all_finite": all_finite,
            "all_deterministic": all_deterministic,
            "teardown_exact": teardown,
            "model_logit_quality_gate_run": False,
            "model_logit_quality_qualified": False,
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
                        "candidate_tops": row["performance"]["candidate"]["core_executed_tops"],
                    }
                    for row in results
                ],
                "amdahl": amdahl,
            },
            indent=2,
        )
    )
    return 0 if artifact["status"] == "speed_screen_passed_unqualified_t3" else 1


if __name__ == "__main__":
    raise SystemExit(main())
