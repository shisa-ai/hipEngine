#!/usr/bin/env python3
"""Identical-operand cross-engine replay: hipEngine against the pinned comparator.

Both engines receive the same packet bytes. hipEngine runs the dispatch key the
packet recorded; the comparator runs its own dense MMB entry point through the
shim built by ``build_shim.sh``. Neither adapter may fall back: the hipEngine
side requires an exact-key registration, and the comparator side requires
``ggml_cuda_mmb_supported_mm`` to select MMB for the operand shapes.

Reported for each engine:

* the complete-operation time, from GPU events on that engine's own stream, with
  the host-wall equivalent beside it so a mis-targeted event shows up instead of
  hiding;
* the comparator's activation conversion cost separately, from the difference
  between a hot conversion cache and a rotating one;
* the full-output difference against four high-precision references that isolate
  where the rounding comes from: an exact float64 product on the packet bytes,
  the same with the weight rounded to bf16, the same with the activation rounded
  to bf16, and both rounded. Which reference an engine tracks is the finding;
  bit inequality on its own is not.

Repetitions are counterbalanced: the engine that runs first alternates by round,
so a drifting clock cannot favour one side.

Example:

    tools/replay_bridge/replay_ab.py \\
      --packet /tmp/replay-bridge/packets/q8-attnqkv-L8-c0 \\
      --shim /tmp/replay-bridge/libmmb_replay.so \\
      --output /tmp/replay-bridge/q8-attnqkv-L8-c0-ab.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hipengine_adapter import HipEngineAdapter, load_packet  # noqa: E402

GGML_TYPE_Q8_0 = 8


class ShimResult(ctypes.Structure):
    _fields_ = [
        ("upload_ms", ctypes.c_double),
        ("convert_ms", ctypes.c_double),
        ("matmul_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("wall_matmul_ms", ctypes.c_double),
        ("wall_total_ms", ctypes.c_double),
        ("spread_ms", ctypes.c_double),
        ("supported", ctypes.c_int),
        ("samples", ctypes.c_int),
        ("wtype", ctypes.c_int),
        ("note", ctypes.c_char * 256),
    ]


class ComparatorAdapter:
    """Drives the pinned comparator's dense MMB entry point on packet operands."""

    def __init__(self, shim_path: Path, *, strict: bool = True):
        self.path = Path(shim_path)
        if not self.path.exists():
            raise FileNotFoundError(f"comparator shim not built: {self.path}")
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.he_replay_mmb_available.restype = ctypes.c_int
        self.lib.he_replay_mmb_min_rotate.restype = ctypes.c_int
        self.lib.he_replay_mmb_run.restype = ctypes.c_int
        self.lib.he_replay_mmb_run.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ShimResult),
        ]
        self.strict = strict
        self.available = bool(self.lib.he_replay_mmb_available())
        self.min_rotate = int(self.lib.he_replay_mmb_min_rotate())
        self.last_note = ""

    def run(self, packet, *, reps: int, rotate: int) -> tuple[ShimResult, np.ndarray]:
        w = np.ascontiguousarray(packet.w_raw)
        x = np.ascontiguousarray(packet.x_float())
        out = np.zeros((packet.rows, packet.out_features), dtype=np.float32)
        result = ShimResult()
        code = self.lib.he_replay_mmb_run(
            w.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p),
            packet.in_features, packet.out_features, packet.rows,
            GGML_TYPE_Q8_0, int(reps), int(rotate),
            1 if self.strict else 0,
            ctypes.byref(result),
        )
        self.last_note = result.note.decode()
        if code == 4:
            raise RuntimeError(f"comparator MMB not selected for these operands: {self.last_note}")
        if code != 0:
            raise RuntimeError(f"comparator shim failed with code {code}: {self.last_note}")
        return result, out


def build_references(packet) -> dict[str, np.ndarray]:
    """High-precision references that separate the two rounding sources."""
    x = packet.x_float().astype(np.float64)
    w = packet.weight_matrix().astype(np.float64)
    x_bf16 = bf16_round(x)
    w_bf16 = bf16_round(w)
    return {
        "exact": x @ w.T,
        "weight_bf16": x @ w_bf16.T,
        "activation_bf16": x_bf16 @ w.T,
        "both_bf16": x_bf16 @ w_bf16.T,
    }


def bf16_round(array: np.ndarray) -> np.ndarray:
    """Round a float array to bf16 and back, matching the hardware's truncation."""
    as32 = array.astype(np.float32)
    bits = as32.view(np.uint32)
    # round-to-nearest-even on the low 16 bits
    lsb = (bits >> 16) & 1
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32).astype(np.float64)


def summarize(engine: np.ndarray, references: dict[str, np.ndarray]) -> dict:
    engine64 = engine.astype(np.float64)
    report = {}
    for name, reference in references.items():
        delta = np.abs(engine64 - reference)
        scale = float(np.abs(reference).mean())
        report[name] = {
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "rms_abs": float(np.sqrt((delta ** 2).mean())),
            "rel_to_mean_abs": float(delta.mean() / scale) if scale else 0.0,
        }
    best = min(report, key=lambda name: report[name]["max_abs"])
    return {"vs_reference": report, "closest_reference": best}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packet", type=Path, required=True,
                        help="packet stem produced by capture_packet.py")
    parser.add_argument("--shim", type=Path,
                        default=Path("/tmp/replay-bridge/libmmb_replay.so"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--skip-references", action="store_true",
                        help="skip the float64 references (they cost ~1 min)")
    args = parser.parse_args()

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)

    packet = load_packet(args.packet)
    flop = 2.0 * packet.rows * packet.in_features * packet.out_features
    print(f"packet     : {packet.stem}")
    print(f"prompt     : {packet.manifest['prompt']['case_id']} "
          f"({packet.manifest['prompt']['category']}) "
          f"chunk {packet.manifest['chunk']['index']} of "
          f"{packet.manifest['prompt']['tokens']} tokens")
    print(f"layer/slot : {packet.manifest['layer']} / {packet.manifest['slot']}")
    print(f"geometry   : rows={packet.rows} K={packet.in_features} M={packet.out_features} "
          f"({flop / 1e9:.1f} GFLOP)")
    print(f"dtypes     : activation={packet.activation_dtype} output={packet.output_dtype}")
    print(f"he variant : {packet.key['variant']}")
    print()

    comparator = ComparatorAdapter(args.shim)
    print(f"comparator : available={comparator.available} min_rotate={comparator.min_rotate}")

    he_rounds: list[dict] = []
    cmp_rounds: list[dict] = []
    he_output = None
    cmp_output = None

    with HipEngineAdapter(packet) as he:
        # Warm both engines fully before any counterbalanced round.
        he.replay(reps=args.reps, warmup=args.warmup, read_output=False)
        comparator.run(packet, reps=args.reps, rotate=comparator.min_rotate)

        for round_index in range(int(args.rounds)):
            he_first = round_index % 2 == 0

            def run_he():
                return he.replay(reps=args.reps, warmup=0, read_output=round_index == 0)

            def run_cmp():
                hot, _ = comparator.run(packet, reps=args.reps, rotate=1)
                rotating, output = comparator.run(
                    packet, reps=args.reps, rotate=comparator.min_rotate
                )
                return hot, rotating, output

            if he_first:
                he_result = run_he()
                hot, rotating, output = run_cmp()
            else:
                hot, rotating, output = run_cmp()
                he_result = run_he()

            he_rounds.append({
                "order": "he-first" if he_first else "comparator-first",
                "event_ms": he_result.event_ms,
                "wall_ms": he_result.wall_ms,
                "spread_ms": he_result.spread_ms,
            })
            cmp_rounds.append({
                "order": "he-first" if he_first else "comparator-first",
                "hot_matmul_ms": hot.matmul_ms,
                "hot_wall_ms": hot.wall_matmul_ms,
                "rotating_total_ms": rotating.total_ms,
                "rotating_convert_ms": rotating.convert_ms,
                "rotating_wall_ms": rotating.wall_total_ms,
                "spread_ms": rotating.spread_ms,
                "supported": int(rotating.supported),
            })
            if he_result.output.size:
                he_output = he_result.output
            if cmp_output is None:
                cmp_output = output

    he_event = statistics.median(r["event_ms"] for r in he_rounds)
    he_wall = statistics.median(r["wall_ms"] for r in he_rounds)
    cmp_hot = statistics.median(r["hot_matmul_ms"] for r in cmp_rounds)
    cmp_convert = statistics.median(r["rotating_convert_ms"] for r in cmp_rounds)
    cmp_total = statistics.median(r["rotating_total_ms"] for r in cmp_rounds)

    def rate(ms: float) -> float:
        return flop / (ms * 1e-3) / 1e12

    print()
    print(f"{'Operation':<34} {'complete ms':>12} {'TFLOP/s':>9} {'conversion ms':>14}")
    print("-" * 74)
    print(f"{'hipEngine ' + packet.key['variant']:<34} {he_event:>12.4f} {rate(he_event):>9.2f} "
          f"{'fused (n/a)':>14}")
    print(f"{'comparator mmb_dense (cold act)':<34} {cmp_total:>12.4f} {rate(cmp_total):>9.2f} "
          f"{cmp_convert:>14.4f}")
    print(f"{'comparator mmb_dense (hot act)':<34} {cmp_hot:>12.4f} {rate(cmp_hot):>9.2f} "
          f"{'0.0000':>14}")
    print(f"{'hipEngine / comparator (x)':<34} {he_event / cmp_total:>12.3f}")
    print()
    print(f"host wall    : hipEngine {he_wall:.4f} ms  comparator {cmp_total:.4f} ms "
          f"(hot {cmp_hot:.4f})")
    print(f"per-round HE : " + ", ".join(f"{r['event_ms']:.4f}" for r in he_rounds))
    print(f"per-round CMP: " + ", ".join(f"{r['rotating_total_ms']:.4f}" for r in cmp_rounds))

    artifact: dict = {
        "schema": 1,
        "kind": "identical-operand-cross-engine-replay",
        "packet": str(packet.stem),
        "packet_manifest": packet.manifest,
        "geometry": {"rows": packet.rows, "in_features": packet.in_features,
                     "out_features": packet.out_features, "flop": flop},
        "comparator": {
            "shim": str(args.shim),
            "available": comparator.available,
            "min_rotate": comparator.min_rotate,
            "mmb_selected": all(r["supported"] == 1 for r in cmp_rounds),
            "note": comparator.last_note,
        },
        "timing": {
            "rounds": int(args.rounds),
            "reps_per_round": int(args.reps),
            "hipengine": {"event_ms": he_event, "wall_ms": he_wall,
                          "tflops": rate(he_event), "per_round": he_rounds},
            "comparator": {"complete_ms": cmp_total, "hot_matmul_ms": cmp_hot,
                           "conversion_ms": cmp_convert, "tflops": rate(cmp_total),
                           "per_round": cmp_rounds},
            "ratio_hipengine_over_comparator": he_event / cmp_total,
        },
    }

    if he_output is not None and cmp_output is not None:
        exact_agreement = float(np.abs(he_output.astype(np.float64)
                                       - packet.out_float().astype(np.float64)).max())
        artifact["replay_identity"] = {
            "hipengine_vs_captured_max_abs": exact_agreement,
            "hipengine_reproduces_capture": bool(exact_agreement == 0.0),
        }
        if not args.skip_references:
            print()
            print("building float64 references on the packet bytes ...")
            references = build_references(packet)
            he_numerics = summarize(he_output, references)
            cmp_numerics = summarize(cmp_output, references)
            artifact["numerics"] = {
                "references": {
                    name: {"rms": float(np.sqrt((ref ** 2).mean())),
                           "max_abs": float(np.abs(ref).max())}
                    for name, ref in references.items()
                },
                "hipengine": he_numerics,
                "comparator": cmp_numerics,
                "hipengine_vs_comparator": {
                    "max_abs": float(np.abs(he_output.astype(np.float64)
                                            - cmp_output.astype(np.float64)).max()),
                    "mean_abs": float(np.abs(he_output.astype(np.float64)
                                             - cmp_output.astype(np.float64)).mean()),
                },
            }
            print()
            print(f"{'Engine':<14} {'closest reference':<20} {'max|d|':>10} {'mean|d|':>10} "
                  f"{'rel':>10}")
            print("-" * 70)
            for label, report in (("hipEngine", he_numerics), ("comparator", cmp_numerics)):
                best = report["closest_reference"]
                row = report["vs_reference"][best]
                print(f"{label:<14} {best:<20} {row['max_abs']:>10.5f} {row['mean_abs']:>10.5f} "
                      f"{row['rel_to_mean_abs']:>10.3e}")
            print()
            for name in references:
                he_row = he_numerics["vs_reference"][name]
                cmp_row = cmp_numerics["vs_reference"][name]
                print(f"  vs {name:<18} hipEngine mean|d|={he_row['mean_abs']:.5f}  "
                      f"comparator mean|d|={cmp_row['mean_abs']:.5f}")
            print()
            cross = artifact["numerics"]["hipengine_vs_comparator"]
            print(f"hipEngine vs comparator: max|d|={cross['max_abs']:.5f} "
                  f"mean|d|={cross['mean_abs']:.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    print()
    print(f"artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
