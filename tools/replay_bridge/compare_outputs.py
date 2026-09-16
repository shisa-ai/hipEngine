#!/usr/bin/env python3
"""Do two fast Q8_0 prefill kernels actually agree, or only in their error statistic?

The registered-variant sweep reports one number per kernel per reference. Ten
f16-WMMA variants with different tilings all reported the *same* maximum
absolute error against every reference, which is either evidence that operand
rounding dominates the summation order, or evidence that the sweep is not
resolving the differences. This script decides between the two by comparing the
two kernels' outputs elementwise.

Usage:

    python tools/replay_bridge/compare_outputs.py \
        --packet /tmp/replay-bridge/packets/q8-attnqkv-L8-c0 \
        --left dense_wide256_f32_f32_out \
        --right wmma_prefill_f32_f32_out --right-tile 16 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hipengine_adapter import load_packet  # noqa: E402
from sweep_variants import Sweep, references  # noqa: E402

# The tile-swept family takes its geometry as keyword arguments.
TILED_ABI = "wmma_prefill"


def capture(sweep: Sweep, name: str, tile: tuple[int, int] | None) -> np.ndarray:
    # The sweep registers the gfx1100 modules directly and resolves under that
    # key; the packet's own backend is gfx1151 and reaches them via the alias
    # pass, which this diagnostic deliberately does not depend on.
    from hipengine.kernels.hip_gfx1100.quant import (  # noqa: F401 - registers
        gguf_k_gemv,
        gguf_q8_0_dense_wide,
        gguf_q8_0_mmq_prefill,
        gguf_q8_0_pack8_gemv,
        gguf_q8_0_prefill,
        gguf_q8_0_raw_to_t16,
        gguf_q8_0_t16_gemv,
        gguf_q8_0_t16_prefill,
    )
    from hipengine.kernels.registry import resolve

    fn = resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant=sweep.packet.key["quant"],
        variant=name,
    )
    if tile is not None:
        fn(
            sweep._buffers[1],
            sweep._buffers[0],
            sweep._buffers[2],
            sweep.packet.rows,
            sweep.packet.in_features,
            sweep.packet.out_features,
            tile_m=tile[0],
            tile_n=tile[1],
            stream=0,
            runtime=sweep.runtime,
        )
    else:
        fn(
            sweep._buffers[1],
            sweep._buffers[0],
            sweep._buffers[2],
            sweep.packet.rows,
            sweep.packet.in_features,
            sweep.packet.out_features,
            stream=0,
            runtime=sweep.runtime,
        )
    sweep.runtime.device_synchronize()
    out = np.empty((sweep.packet.rows, sweep.packet.out_features), dtype=np.float32)
    sweep.runtime.memcpy(
        out.ctypes.data,
        sweep._buffers[2],
        int(out.nbytes),
        sweep._kind.DEVICE_TO_HOST,
    )
    return out


def compare(
    packet_path: Path,
    left: str,
    left_tile: tuple[int, int] | None,
    right: str,
    right_tile: tuple[int, int] | None,
) -> dict:
    """Run both kernels on the packet and return the elementwise comparison."""

    packet = load_packet(packet_path)
    sweep = Sweep(packet, warmup=2, reps=1)
    try:
        a = capture(sweep, left, left_tile)
        b = capture(sweep, right, right_tile)
        weight = sweep.weight
    finally:
        sweep.close()

    exact, both, f16both = references(packet, weight)
    scale = float(np.max(np.abs(exact))) or 1.0
    diff = np.abs(a - b)
    return {
        "left": left,
        "left_tile": list(left_tile) if left_tile else None,
        "right": right,
        "right_tile": list(right_tile) if right_tile else None,
        "bit_identical": bool(np.array_equal(a, b)),
        "differing_elements": int((diff > 0).sum()),
        "elements": int(diff.size),
        "max_abs_delta": float(diff.max()),
        "mean_abs_delta": float(diff.mean()),
        "max_abs_delta_relative": float(diff.max()) / scale,
        "refs": {
            name: {
                "left": float(np.max(np.abs(a - ref))),
                "right": float(np.max(np.abs(b - ref))),
                "identical": bool(
                    np.max(np.abs(a - ref)) == np.max(np.abs(b - ref))
                ),
            }
            for name, ref in (
                ("f64_exact", exact),
                ("bf16_both", both),
                ("f16_both", f16both),
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-tile", type=int, nargs=2, default=None)
    parser.add_argument("--right-tile", type=int, nargs=2, default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the comparison as JSON instead of a report",
    )
    args = parser.parse_args()

    result = compare(
        args.packet,
        args.left,
        tuple(args.left_tile) if args.left_tile else None,
        args.right,
        tuple(args.right_tile) if args.right_tile else None,
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    left, right = result["left"], result["right"]
    lt = f" @ {result['left_tile']}" if result["left_tile"] else ""
    rt = f" @ {result['right_tile']}" if result["right_tile"] else ""
    print(f"left  : {left}{lt}")
    print(f"right : {right}{rt}")
    print()
    print(f"bit-identical       : {result['bit_identical']}")
    print(
        f"max |left - right|  : {result['max_abs_delta']:.6e}  "
        f"({result['max_abs_delta_relative']:.3e} relative)"
    )
    print(f"mean |left - right| : {result['mean_abs_delta']:.6e}")
    print(
        f"differing elements  : {result['differing_elements']} / "
        f"{result['elements']} "
        f"({100.0 * result['differing_elements'] / result['elements']:.4f}%)"
    )
    print()
    for name, ref in result["refs"].items():
        verdict = "identical" if ref["identical"] else "DIFFER"
        print(
            f"max|d| vs {name:<10}: left {ref['left']:.6e}   "
            f"right {ref['right']:.6e}   {verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
