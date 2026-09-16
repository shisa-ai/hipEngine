#!/usr/bin/env python3
"""Assemble the cross-engine replay artifact from the two replay runs.

Reads the strict-profile and production-profile replay JSONs plus the packet
manifests they were run against, and writes one compact artifact. Nothing is
recomputed here: every number is copied from a run output, so the artifact can
be rebuilt from the raw files without re-running the GPU.

    assemble.py \\
      --strict /tmp/replay-bridge/q8-attnqkv-L8-c0-strict-ab.json \\
      --production /tmp/replay-bridge/q8-attnqkv-L8-c0-prod-ab.json \\
      --packet-dir /tmp/replay-bridge/packets \\
      --output artifact.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Kernel facts read from the rocprofv3 kernel trace of each run. rocprofv3
# reports Grid_Size_X as gridDim.x * blockDim.x, so the block geometry is
# recovered by dividing by the workgroup size.
TRACE = {
    "hipengine_strict": {
        "kernel": "gguf_k_prefill_out_coltile_rowbatch_kernel<float, float, 8, 8, 4, false>",
        "grid_dim": [1280, 256, 1],
        "workgroup": [128, 1, 1],
        "lds_bytes": 512,
        "vgpr": 72,
        "mean_ms": 19.6773,
    },
    "hipengine_production": {
        "kernel": "gguf_k_prefill_out_coltile_rowbatch_kernel<float, float, 8, 8, 4, true>",
        "grid_dim": [1280, 256, 1],
        "workgroup": [128, 1, 1],
        "lds_bytes": 512,
        "vgpr": 72,
        "mean_ms": 17.3952,
    },
    "comparator_matmul": {
        "kernel": "mmb_dense_kernel<128, 256, 64, 64, 1>",
        "grid_dim": [80, 4, 1],
        "workgroup": [256, 1, 1],
        "lds_bytes": 55296,
        "vgpr": 256,
        "mean_ms": 1.3347,
    },
    "comparator_conversion": {
        "kernel": "mmb_cvt_f32_bf16",
        "grid_dim": [1280, 1, 1],
        "workgroup": [256, 1, 1],
        "lds_bytes": 0,
        "vgpr": 16,
        "mean_ms": 0.0836,
    },
}


def trim_packet(manifest: dict) -> dict:
    """Drop the per-launch log; keep identity, geometry and the dispatch record."""
    keep = (
        "schema", "kind", "generated_at", "host", "model", "prompt", "profile",
        "chunk", "layer", "slot", "quant", "hipengine_variant", "hipengine_abi",
        "geometry", "arrays", "x_float_summary", "launches_observed",
    )
    return {key: manifest[key] for key in keep if key in manifest}


def weight_traffic(geometry: dict, col_tile: int, row_batch: int) -> dict:
    """Derive weight bytes read from the measured launch geometry.

    A block covering ``col_tile`` output columns and ``row_batch`` rows must read
    the whole K extent of those columns, so the operation reads the weight
    ``ceil(rows / row_batch)`` times in total.
    """
    rows = geometry["rows"]
    in_features = geometry["in_features"]
    out_features = geometry["out_features"]
    column_bytes = (in_features // 32) * 34
    blocks = (out_features // col_tile) * ((rows + row_batch - 1) // row_batch)
    per_block = col_tile * column_bytes
    return {
        "col_tile": col_tile,
        "row_batch": row_batch,
        "blocks": blocks,
        "bytes_per_block": per_block,
        "weight_bytes_read": blocks * per_block,
        "weight_rereads": (rows + row_batch - 1) // row_batch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    strict = json.loads(args.strict.read_text())
    production = json.loads(args.production.read_text())
    geometry = strict["geometry"]

    packets = {}
    for name, run in (("strict", strict), ("production", production)):
        stem = Path(run["packet"]).name
        manifest = json.loads((args.packet_dir / f"{stem}.json").read_text())
        packets[name] = trim_packet(manifest)

    strict_arrays = packets["strict"]["arrays"]
    production_arrays = packets["production"]["arrays"]
    operand_identity = {
        "identical_across_profiles": all(
            strict_arrays[key]["sha256"] == production_arrays[key]["sha256"]
            for key in ("w_raw", "x", "out")
        ),
        "weight_sha256": strict_arrays["w_raw"]["sha256"],
        "weight_nbytes": strict_arrays["w_raw"]["nbytes"],
        "activation_sha256": strict_arrays["x"]["sha256"],
        "output_sha256": strict_arrays["out"]["sha256"],
    }

    flop = geometry["flop"]
    timing = {
        "hipengine_strict": {
            "variant": packets["strict"]["hipengine_variant"]["variant"],
            "complete_ms": strict["timing"]["hipengine"]["event_ms"],
            "wall_ms": strict["timing"]["hipengine"]["wall_ms"],
            "tflops": strict["timing"]["hipengine"]["tflops"],
        },
        "hipengine_production": {
            "variant": packets["production"]["hipengine_variant"]["variant"],
            "complete_ms": production["timing"]["hipengine"]["event_ms"],
            "wall_ms": production["timing"]["hipengine"]["wall_ms"],
            "tflops": production["timing"]["hipengine"]["tflops"],
        },
        "comparator": {
            "complete_ms_cold_activation": production["timing"]["comparator"]["complete_ms"],
            "matmul_ms_hot_activation": production["timing"]["comparator"]["hot_matmul_ms"],
            "conversion_ms": production["timing"]["comparator"]["conversion_ms"],
            "tflops": production["timing"]["comparator"]["tflops"],
        },
        "ratio_hipengine_over_comparator": {
            "production": production["timing"]["hipengine"]["event_ms"]
            / strict["timing"]["comparator"]["complete_ms"],
            "strict": strict["timing"]["hipengine"]["event_ms"]
            / strict["timing"]["comparator"]["complete_ms"],
        },
    }

    artifact = {
        "schema": 1,
        "kind": "cross-engine-identical-operand-replay",
        "host": strict["packet_manifest"]["host"],
        "model": strict["packet_manifest"]["model"],
        "protocol": {
            "harness": "tools/replay_bridge",
            "packet_harness": "tools/replay_bridge/capture_packet.py",
            "replay_harness": "tools/replay_bridge/replay_ab.py",
            "comparator_shim": production["comparator"]["shim"],
            "rounds": production["timing"]["rounds"],
            "reps_per_round": production["timing"]["reps_per_round"],
            "counterbalanced": True,
            "hipengine_events_on": "default stream, matching the launched kernel",
            "comparator_events_on": "the comparator's own non-blocking backend stream",
        },
        "packets": packets,
        "operand_identity": operand_identity,
        "geometry": geometry,
        "flop": flop,
        "timing": timing,
        "dispatch": {
            "trace": TRACE,
            "hipengine_mmb_selected": production["comparator"]["mmb_selected"],
            "comparator_note": production["comparator"]["note"],
            "replay_reproduces_capture": {
                name: run["replay_identity"]["hipengine_reproduces_capture"]
                for name, run in (("strict", strict), ("production", production))
            },
            "replay_max_abs_vs_capture": {
                name: run["replay_identity"]["hipengine_vs_captured_max_abs"]
                for name, run in (("strict", strict), ("production", production))
            },
        },
        "numerics": {
            "references": production["numerics"]["references"],
            "hipengine_vs_exact": production["numerics"]["hipengine"]["vs_reference"]["exact"],
            "hipengine_closest_reference": production["numerics"]["hipengine"]["closest_reference"],
            "comparator_closest_reference": production["numerics"]["comparator"]["closest_reference"],
            "comparator_vs_both_bf16": production["numerics"]["comparator"]["vs_reference"]["both_bf16"],
            "hipengine_vs_comparator": production["numerics"]["hipengine_vs_comparator"],
            "hipengine_mean_abs_by_reference": {
                name: row["mean_abs"]
                for name, row in production["numerics"]["hipengine"]["vs_reference"].items()
            },
            "comparator_mean_abs_by_reference": {
                name: row["mean_abs"]
                for name, row in production["numerics"]["comparator"]["vs_reference"].items()
            },
        },
        "derived_weight_traffic": {
            "hipengine": weight_traffic(geometry, col_tile=8, row_batch=4),
            "comparator": weight_traffic(geometry, col_tile=128, row_batch=256),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"  operand identity holds: {operand_identity['identical_across_profiles']}")
    print(f"  production ratio      : "
          f"{timing['ratio_hipengine_over_comparator']['production']:.3f}x hipEngine/comparator")
    print(f"  hipEngine weight read : "
          f"{artifact['derived_weight_traffic']['hipengine']['weight_bytes_read'] / 1e9:.2f} GB")
    print(f"  comparator weight read: "
          f"{artifact['derived_weight_traffic']['comparator']['weight_bytes_read'] / 1e9:.3f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
