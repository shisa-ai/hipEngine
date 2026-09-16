#!/usr/bin/env python3
"""Sweep every registered hipEngine Q8_0 prefill variant on one identical-operand packet.

The replay bridge establishes what the production dispatch key costs on a real
projection. This sweep answers the next question: which of the *already
registered* variants is fastest on those same bytes. A variant that beats the
production key here is a dispatch candidate; one that reaches the pinned
comparator's rate needs no new kernel at all.

Every variant receives the packet's exact ``w_raw``, ``x`` and geometry. Nothing
is regenerated and no variant may fall back: the key is resolved with
``is_registered`` (exact key, no backend/quant/variant fallback) and the callable
is invoked directly.

Each row reports the complete-operation time from GPU events, the achieved
TFLOP/s, and where the result sits against two high-precision references so a
faster variant that silently loses precision is visible rather than promoted:

* ``f64``      - exact float64 product of the dequantized weight and the packet
                 activation. This is the arithmetic the strict profile tracks.
* ``bf16both`` - the same product with both operands rounded to bf16, which is
                 the comparator's arithmetic.

Example:

    tools/replay_bridge/sweep_variants.py \\
      --packet /tmp/replay-bridge/packets/q8-attnqkv-L8-c0 \\
      --output /tmp/replay-bridge/q8-attnqkv-L8-c0-sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hipengine_adapter import load_packet, to_float32  # noqa: E402

# Variants whose 7th positional slot is an ignored ``threads`` argument rather
# than a WMMA tile shape. Kept explicit so an ABI mismatch fails loudly instead
# of silently launching with a tile value reinterpreted as a thread count.
THREADS_ABI = "iu8_wmma_prefill_f32_f32_out"

# Variants that accept explicit ``tile_m``/``tile_n`` kwargs. The registered
# callables are ``*args, **kwargs`` launch shims, so the parameter cannot be
# introspected; the family is declared by prefix instead. Getting this wrong is
# silent, not loud -- a missed family just measures its default tile and the
# sweep looks like it succeeded -- so the check is asserted by a test.
TILE_ABI_PREFIX = "wmma_prefill_"


def accepts_tiles(variant: str) -> bool:
    """True when the registered variant takes ``tile_m``/``tile_n`` kwargs."""
    return variant.startswith(TILE_ABI_PREFIX) and variant != THREADS_ABI

# Variants that do not take the standard (x, w, out, rows, in, out) ABI at all.
SKIP_ABI = {
    "dual_gemv_f32_f32_out": "dual_gemv takes a fused operand pair, not a single projection",
    "dual_gemv_bf16_bf16_out": "dual_gemv takes a fused operand pair, not a single projection",
}

# WMMA prefill tiles allowed by the gguf_q8_0_prefill wrapper.
WMMA_TILES = ((16, 16), (16, 32), (32, 16), (32, 32), (64, 16), (64, 32))


def references(packet, weight):
    """Return the high-precision references described in the module docstring.

    ``f16both`` simulates what a kernel that rounds both operands to f16 and
    accumulates in f32 should produce. A candidate whose output tracks it is
    implementing the arithmetic it claims; a candidate that misses it has a
    structural bug, not a rounding difference.
    """
    x64 = packet.x_float().astype(np.float64)
    w64 = weight.astype(np.float64)
    # Row-major (rows, out_features), matching the packet's output layout.
    exact = x64 @ w64.T

    wb = round_to_bf16(weight).astype(np.float64)
    xb = round_to_bf16(packet.x_float()).astype(np.float64)
    both = xb @ wb.T

    wh = weight.astype(np.float16).astype(np.float64)
    xh = packet.x_float().astype(np.float16).astype(np.float64)
    f16both = xh @ wh.T
    return exact, both, f16both


def round_to_bf16(array: np.ndarray) -> np.ndarray:
    """Round float32 to bf16 and back, using round-to-nearest-even on the low 16 bits."""
    as_uint = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    low = as_uint & np.uint32(0xFFFF)
    high = as_uint >> np.uint32(16)
    round_bit = np.uint32(0x8000)
    lsb = (high & np.uint32(1)) << np.uint32(16)
    # Round half to even, matching hardware bf16 conversion.
    rounded = as_uint + round_bit - np.uint32(1) + lsb
    out = (rounded & np.uint32(0xFFFF0000)).astype(np.uint32)
    return out.view(np.float32)


class Sweep:
    def __init__(self, packet, *, warmup: int, reps: int):
        from hipengine.core.hip import MemcpyKind, get_hip_runtime

        self.packet = packet
        self.warmup = warmup
        self.reps = reps
        self.runtime = get_hip_runtime()
        self._kind = MemcpyKind
        self.weight = packet.weight_matrix()
        self.exact, self.both, self.f16both = references(packet, self.weight)
        self.flops = 2.0 * packet.rows * packet.in_features * packet.out_features

        self._buffers = [
            self._upload(packet.w_raw),
            self._upload(packet.x),
            self.runtime.malloc(int(packet.out.nbytes)),
        ]

    def _upload(self, array: np.ndarray) -> int:
        contiguous = np.ascontiguousarray(array)
        ptr = self.runtime.malloc(int(contiguous.nbytes))
        self.runtime.memcpy(
            ptr, contiguous.ctypes.data, int(contiguous.nbytes), self._kind.HOST_TO_DEVICE
        )
        return ptr

    def close(self) -> None:
        for ptr in self._buffers:
            try:
                self.runtime.free(ptr)
            except Exception:  # noqa: BLE001 - teardown must not mask a result
                pass
        self._buffers = []

    def call(self, fn, *, variant: str, tile: tuple[int, int] | None) -> None:
        x_ptr, w_ptr, out_ptr = self._buffers[1], self._buffers[0], self._buffers[2]
        packet = self.packet
        common = (x_ptr, w_ptr, out_ptr, packet.rows, packet.in_features, packet.out_features)
        if tile is not None:
            fn(*common, tile_m=tile[0], tile_n=tile[1], stream=0, runtime=self.runtime)
        elif variant == THREADS_ABI:
            fn(*common, threads=128, stream=0, runtime=self.runtime)
        else:
            fn(*common, stream=0, runtime=self.runtime)

    def run(self, fn, *, variant: str, tile: tuple[int, int] | None) -> dict:
        packet = self.packet
        for _ in range(self.warmup):
            self.call(fn, variant=variant, tile=tile)
        self.runtime.device_synchronize()

        start = self.runtime.event_create()
        stop = self.runtime.event_create()
        try:
            self.runtime.event_record(start, 0)
            for _ in range(self.reps):
                self.call(fn, variant=variant, tile=tile)
            self.runtime.event_record(stop, 0)
            self.runtime.event_synchronize(stop)
            event_ms = self.runtime.event_elapsed_time_ms(start, stop) / self.reps
        finally:
            self.runtime.event_destroy(start)
            self.runtime.event_destroy(stop)

        out = np.empty((packet.rows, packet.out_features), dtype=np.float32)
        self.runtime.memcpy(
            out.ctypes.data, self._buffers[2], int(out.nbytes), self._kind.DEVICE_TO_HOST
        )
        captured = packet.out_float()
        # Relative error makes the absolute numbers interpretable: this projection's
        # output magnitude sets what 1e-3 absolute actually means.
        ref_scale = float(np.max(np.abs(self.exact))) or 1.0
        return {
            "variant": variant,
            "tile": list(tile) if tile else None,
            "event_ms": event_ms,
            "tflops": self.flops / (event_ms * 1e-3) / 1e12,
            "max_abs_vs_f64": float(np.max(np.abs(out - self.exact))),
            "max_rel_vs_f64": float(np.max(np.abs(out - self.exact))) / ref_scale,
            "mean_abs_vs_f64": float(np.mean(np.abs(out - self.exact))),
            "max_abs_vs_bf16both": float(np.max(np.abs(out - self.both))),
            "mean_abs_vs_bf16both": float(np.mean(np.abs(out - self.both))),
            "max_abs_vs_f16both": float(np.max(np.abs(out - self.f16both))),
            "mean_abs_vs_f16both": float(np.mean(np.abs(out - self.f16both))),
            "ref_max_abs": ref_scale,
            "out_max_abs": float(np.max(np.abs(out))),
            "bit_exact_vs_capture": bool(np.array_equal(out, captured)),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument(
        "--filter",
        default="f32_f32_out",
        help="substring the variant name must contain (default: f32_f32_out)",
    )
    parser.add_argument(
        "--sweep-tiles",
        action="store_true",
        help="also sweep every allowed WMMA tile for the wmma_prefill family",
    )
    args = parser.parse_args()

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
    from hipengine.kernels.registry import KernelKey, is_registered, registered_keys, resolve

    packet = load_packet(args.packet)
    key = packet.key
    print(f"packet   : {args.packet}")
    print(f"quant    : {key['quant']}  geometry: {packet.manifest['geometry']}")
    print(f"recorded : {key['variant']}")

    sweep = Sweep(packet, warmup=args.warmup, reps=args.reps)
    rows: list[dict] = []
    try:
        keys = [
            k
            for k in registered_keys()
            if k.quant == key["quant"] and k.layer == "linear" and args.filter in k.variant
        ]
        # The recorded key first, so its number is the baseline in the same process.
        keys.sort(key=lambda k: (k.variant != key["variant"], k.variant))

        if args.sweep_tiles:
            expandable = [k.variant for k in keys if accepts_tiles(k.variant)]
            print(
                f"tile sweep: {len(expandable)} variant(s) accept tile_m/tile_n: "
                f"{', '.join(expandable) if expandable else 'none'}"
            )
            if not expandable:
                print(
                    "WARNING: --sweep-tiles expanded nothing; "
                    "TILE_ABI_PREFIX no longer matches the registered family"
                )

        for kernel_key in keys:
            exact_key = KernelKey(kernel_key.backend, kernel_key.layer, kernel_key.quant, kernel_key.variant)
            if not is_registered(exact_key):
                continue
            fn = resolve(
                backend=kernel_key.backend,
                layer=kernel_key.layer,
                quant=kernel_key.quant,
                variant=kernel_key.variant,
            )
            variant = kernel_key.variant
            tiles: list[tuple[int, int] | None] = [None]
            # Expand tiles only where the wrapper actually accepts them. A
            # substring match on the variant name is wrong: iu8_wmma_prefill
            # contains "wmma_prefill" but takes an ignored ``threads`` argument
            # in that slot, so expanding it fails with a TypeError instead of
            # producing a row. Introspecting the registered callable is wrong
            # too: it is a ``*args, **kwargs`` shim with no visible parameters.
            if args.sweep_tiles and accepts_tiles(variant):
                tiles = list(WMMA_TILES)
            if variant in SKIP_ABI:
                rows.append({"variant": variant, "skipped": SKIP_ABI[variant]})
                print(f"  {variant:<52} SKIPPED  {SKIP_ABI[variant]}")
                continue
            for tile in tiles:
                label = variant if tile is None else f"{variant}@tile{tile[0]}x{tile[1]}"
                try:
                    result = sweep.run(fn, variant=variant, tile=tile)
                except Exception as exc:  # noqa: BLE001 - one bad variant must not stop the sweep
                    rows.append(
                        {
                            "variant": label,
                            "failed": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=3),
                        }
                    )
                    print(f"  {label:<52} FAILED  {type(exc).__name__}: {exc}")
                    continue
                result["variant"] = label
                rows.append(result)
                print(
                    f"  {label:<52} {result['event_ms']:8.3f} ms  "
                    f"{result['tflops']:6.2f} TFLOP/s  "
                    f"rel {result['max_rel_vs_f64']:.2e}  "
                    f"max|d| f64 {result['max_abs_vs_f64']:.3e}  "
                    f"bf16both {result['max_abs_vs_bf16both']:.3e}"
                    f"  f16both {result['max_abs_vs_f16both']:.3e}"
                    f"{'  BIT-EXACT' if result['bit_exact_vs_capture'] else ''}"
                )
    finally:
        sweep.close()

    ranked = sorted((r for r in rows if "event_ms" in r), key=lambda r: r["event_ms"])
    print("\nfastest:")
    for r in ranked[:8]:
        print(f"  {r['event_ms']:8.3f} ms  {r['tflops']:6.2f} TFLOP/s  {r['variant']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "packet": str(args.packet),
                    "quant": key["quant"],
                    "geometry": packet.manifest["geometry"],
                    "recorded_variant": key["variant"],
                    "warmup": args.warmup,
                    "reps": args.reps,
                    "flops": sweep.flops,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
