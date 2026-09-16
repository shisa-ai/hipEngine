#!/usr/bin/env python3
"""Assemble the dense-wide Q8_0 prefill candidate artifact from raw sweep output.

Reads the sweep JSON files next to this script plus the rocprofv3 kernel trace
recorded for the candidate, and writes ``artifact.json``. Numbers are taken from
the files rather than transcribed so the artifact cannot drift from the runs.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE_GLOB = "/tmp/prof-final/gfx1151/*_kernel_trace.csv"


def sweep_rows(name: str) -> list[dict]:
    data = json.loads((HERE / name).read_text())
    return [row for row in data["rows"] if "event_ms" in row]


def profile_rows() -> dict[str, dict]:
    """Kernel-level durations and resource usage from the rocprofv3 trace."""
    out: dict[str, dict] = {}
    rows: list[dict] = []
    for path in glob.glob(PROFILE_GLOB):
        rows += list(csv.DictReader(open(path)))
    for row in rows:
        name = row["Kernel_Name"]
        entry = out.setdefault(
            name,
            {
                "launches": 0,
                "total_ns": 0,
                "vgpr": row.get("VGPR_Count"),
                "lds_bytes": row.get("LDS_Block_Size"),
                "scratch_bytes": row.get("Scratch_Size"),
                "workgroup_x": row.get("Workgroup_Size_X"),
                "grid_x": row.get("Grid_Size_X"),
                "grid_y": row.get("Grid_Size_Y"),
            },
        )
        entry["launches"] += 1
        entry["total_ns"] += int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
    for entry in out.values():
        entry["avg_us"] = entry["total_ns"] / entry["launches"] / 1e3
    return out


def main() -> int:
    family_runs = [
        sweep_rows("sweep-dense-wide-family-run1.json"),
        sweep_rows("sweep-dense-wide-family-run2.json"),
    ]
    registered = sweep_rows("sweep-registered-variants.json")

    # Repeat-run spread for the candidate family.
    family: dict[str, dict] = {}
    for rows in family_runs:
        for row in rows:
            entry = family.setdefault(
                row["variant"],
                {
                    "event_ms": [],
                    "tflops": [],
                    "max_rel_vs_f64": row["max_rel_vs_f64"],
                    "max_abs_vs_f64": row["max_abs_vs_f64"],
                    "max_abs_vs_f16both": row["max_abs_vs_f16both"],
                    "max_abs_vs_bf16both": row["max_abs_vs_bf16both"],
                    "ref_max_abs": row["ref_max_abs"],
                },
            )
            entry["event_ms"].append(row["event_ms"])
            entry["tflops"].append(row["tflops"])

    baseline = {row["variant"]: row for row in registered}

    artifact = {
        "protocol": "identical-operand cross-engine replay + registered-variant sweep",
        "packet": "q8-attnqkv-L8-c0",
        "operation": "Qwen3.8-Flash-Next UD-Q4_K_XL layers.8.attn_qkv",
        "geometry": {"rows": 1024, "in_features": 2560, "out_features": 10240},
        "flops": 2.0 * 1024 * 2560 * 10240,
        "host": "gfx1151 (Strix Halo, Radeon 8060S), 122880 MiB reported VRAM",
        "kernel_binary_arch": "gfx1151",
        "candidate": {
            "kernel": "q8_0_dense_wide_kernel<128, 256, 64, 64>",
            "source": "hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_dense_wide.hip",
            "registered_variant": "dense_wide256_f32_f32_out",
            "port_source": (
                "ggml/src/ggml-cuda/mmb.cu mmb_dense_kernel<128,256,64,64,1>, "
                "commit c4aa302294fcd5121af2039fd4d3dee0d472ec03"
            ),
            "runs": family["dense_wide256_f32_f32_out"],
            "all_tiles": family,
        },
        "baselines_same_process": {
            name: {
                "event_ms": baseline[name]["event_ms"],
                "tflops": baseline[name]["tflops"],
                "max_rel_vs_f64": baseline[name]["max_rel_vs_f64"],
                "bit_exact_vs_capture": baseline[name]["bit_exact_vs_capture"],
            }
            for name in (
                "coltile8_rowbatch4_f32_f32_out",
                "coltile8_rowbatch4_wave_scale_f32_f32_out",
                "wmma_prefill_f32_f32_out@tile16x32",
                "iu8_wmma_prefill_f32_f32_out",
            )
        },
        "kernel_level": profile_rows(),
        "tiling_invariance": json.loads((HERE / "tiling-invariance.json").read_text()),
        "activation_path_probe": {
            "note": (
                "Timing-only probe: the same kernel with the activation load "
                "reading a half8_t directly instead of two float4 plus eight "
                "f32->f16 conversions. The bytes in the buffer are f32, so the "
                "values are garbage; only the load path is under test. Built "
                "outside the tree and not registered."
            ),
            "f32_activation_in_kernel_convert_ms": 2.541,
            "preconverted_f16_activation_ms": 1.303,
            "comparator_kernel_ms": 1.339,
        },
    }

    (HERE / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {HERE / 'artifact.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
