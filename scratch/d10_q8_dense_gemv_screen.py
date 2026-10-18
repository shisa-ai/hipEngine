"""D9/D10 screen: registered q8_0 rows==1 decode candidates at Gemma dense shapes.

The Gemma punchlist D9 row asks for a screen of every registered variant at the
dense decode shapes (2112 / 4096 / 8192) so the pack8 GEMV can be tuned against
them. This script times each candidate on GPU at rows == 1 with raw GGUF Q8_0
weight bytes (t16 candidates get the byte-neutral Q8T16 repack), touch-checks
every launch so a silently dead wrapper cannot win the table, and compares every
candidate's output against both an f32 dequant reference and the legacy
incumbent (parent parity).

Production decode shapes (rows == 1, bf16 activations, hidden 2816):
  gate/up single  K=2816 -> out=2112
  q projection    K=2816 -> out=4096
  fused qkv       K=2816 -> out=8192
  o projection    K=4096 -> out=2816
  down projection K=2112 -> out=2816
plus the fused gate+up pair shape (K=2816 -> 2112 + 2112) for the pair-only
candidates (pack8_dual / t16 dual / rowvec8), timed separately.

Usage:
  env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1 PYTHONPATH=. \
      .venv/bin/python scratch/d10_q8_dense_gemv_screen.py
"""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.runtime import MemcpyKind  # noqa: E402

Q8_0_BLOCK = 32
BLOCK_BYTES = 2 + Q8_0_BLOCK  # fp16 d + qs[32]
WARMUP = 20
ITERS = 200
_SEED = 20260930


def _alloc(nbytes: int) -> int:
    runtime = get_hip_runtime()
    return runtime.malloc(nbytes)


def _upload(device_ptr: int, buf: bytes) -> None:
    runtime = get_hip_runtime()
    host = (ctypes.c_ubyte * len(buf)).from_buffer_copy(buf)
    runtime.memcpy(device_ptr, ctypes.addressof(host), len(buf), MemcpyKind.HOST_TO_DEVICE)


def _download(device_ptr: int, nbytes: int) -> bytes:
    runtime = get_hip_runtime()
    host = (ctypes.c_ubyte * nbytes)()
    runtime.memcpy(ctypes.addressof(host), device_ptr, nbytes, MemcpyKind.DEVICE_TO_HOST)
    return bytes(host)


def _free_all(ptrs) -> None:
    runtime = get_hip_runtime()
    for p in sorted({int(p) for p in ptrs if p}):
        runtime.free(p)


def make_x(k: int) -> int:
    """bf16 activation row [1, K], deterministic, finite."""
    rng = np.random.default_rng(_SEED)
    f32 = rng.standard_normal(k, dtype=np.float32)
    x = (f32.view(np.uint32) >> 16).astype(np.uint16)
    buf = x.tobytes()
    ptr = _alloc(len(buf))
    _upload(ptr, buf)
    return ptr


def make_w(out_features: int, in_features: int) -> int:
    """Raw GGUF Q8_0 bytes [out, K/32 * 34]: d = 1.0 (0x3C00), qs random per row."""
    blocks = in_features // Q8_0_BLOCK
    rng = np.random.default_rng(_SEED + 1)
    qs = rng.integers(-127, 127, size=(out_features, blocks, Q8_0_BLOCK), dtype=np.int8)
    row = np.empty((out_features, blocks, BLOCK_BYTES), dtype=np.uint8)
    row[..., 0] = 0x00
    row[..., 1] = 0x3C
    row[..., 2:BLOCK_BYTES] = qs.view(np.uint8)
    buf = row.tobytes()
    ptr = _alloc(len(buf))
    _upload(ptr, buf)
    return ptr


def make_out(nbytes: int) -> int:
    """Output row pre-filled with 0xAA so touch-check sees fresh writes."""
    ptr = _alloc(nbytes)
    get_hip_runtime().memset(ptr, 0xAA, nbytes)
    return ptr


def make_t16_tiles(w_raw_device: int, out_features: int, in_features: int) -> int:
    """Byte-neutral Q8T16 repack of the raw q8_0 bytes (CPU) -> device tiles."""
    from hipengine.quant.gguf_t16 import repack_gguf_q8_0_tile16

    raw_bytes = out_features * (in_features // Q8_0_BLOCK) * BLOCK_BYTES
    raw = np.frombuffer(_download(w_raw_device, raw_bytes), dtype=np.uint8).reshape(
        out_features, in_features // Q8_0_BLOCK * BLOCK_BYTES
    )
    packed = repack_gguf_q8_0_tile16(raw)
    buf = np.ascontiguousarray(packed.tiles).tobytes()
    ptr = _alloc(len(buf))
    _upload(ptr, buf)
    return ptr


def dequant_ref(x_device: int, w_device: int, out_features: int, in_features: int) -> np.ndarray:
    """f32 reference row: dequant q8_0 weights with the same bf16 input bits."""
    x_raw = np.frombuffer(_download(x_device, in_features * 2), dtype=np.uint16)
    x_f32 = (x_raw.astype(np.uint32) << 16).view(np.float32)
    w_bytes = out_features * (in_features // Q8_0_BLOCK) * BLOCK_BYTES
    w = np.frombuffer(_download(w_device, w_bytes), dtype=np.uint8).reshape(
        out_features, in_features // Q8_0_BLOCK, BLOCK_BYTES
    )
    d = w[..., 0:2].copy().view(np.float16).astype(np.float32).reshape(out_features, -1)
    qs = w[..., 2:34].copy().view(np.int8).reshape(out_features, -1).astype(np.float32)
    w_deq = np.repeat(d, Q8_0_BLOCK, axis=1) * qs  # [out, blocks*32] == [out, K]
    return (w_deq @ x_f32).astype(np.float32)


def bf16_as_f32(out_device: int, out_features: int) -> np.ndarray:
    bits = np.frombuffer(_download(out_device, out_features * 2), dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def round_bf16_rne(values: np.ndarray) -> np.ndarray:
    """f32 -> bf16 round-to-nearest-even (the store the kernels perform)."""
    bits = values.astype(np.float32).copy().view(np.uint32)
    lsb = (bits >> 16) & np.uint32(1)
    bits = bits + np.uint32(0x7FFF) + lsb
    return (bits >> 16).astype(np.uint16).astype(np.uint32).__lshift__(16).view(np.float32)


def timed(fn, args, warmup=WARMUP, iters=ITERS) -> float:
    runtime = get_hip_runtime()
    for _ in range(warmup):
        fn(*args)
    runtime.device_synchronize()
    start = runtime.event_create()
    stop = runtime.event_create()
    runtime.event_record(start)
    for _ in range(iters):
        fn(*args)
    runtime.event_record(stop)
    runtime.event_synchronize(stop)
    ms = runtime.event_elapsed_time_ms(start, stop)
    runtime.event_destroy(start)
    runtime.event_destroy(stop)
    return ms * 1000.0 / iters


def touch_ok(out_ptr: int, nbytes: int) -> bool:
    return any(b != 0xAA for b in _download(out_ptr, nbytes))


def call_single(fn, x: int, w: int, out: int, rows: int, k: int, o: int) -> None:
    fn(x, w, out, rows, k, o)


# ---------------------------------------------------------------- candidates
def candidates_single():
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        gguf_q8_0_pack8_gemv_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_pack8_gemv import (
        gguf_q8_0_pack8_gemv_decode_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv import (
        gguf_q8_0_t16_gemv_decode_bf16_bf16_out,
        gguf_q8_0_t16_gemv_decode_rowtile4_bf16_bf16_out,
    )

    # (label, fn, weight_kind) where kind is "raw" or "t16tiles"
    return [
        ("pack8_gemv (legacy, prod)", gguf_q8_0_pack8_gemv_bf16_bf16_out, "raw"),
        ("pack8_gemv_decode (P9.B3)", gguf_q8_0_pack8_gemv_decode_bf16_bf16_out, "raw"),
        ("t16_gemv_decode", gguf_q8_0_t16_gemv_decode_bf16_bf16_out, "t16tiles"),
        (
            "t16_gemv_decode_rowtile4",
            gguf_q8_0_t16_gemv_decode_rowtile4_bf16_bf16_out,
            "t16tiles",
        ),
    ]


def candidates_pair():
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_pack8_gemv import (
        gguf_q8_0_pack8_dual_gate_up_gemv_decode_bf16_bf16_out,
        gguf_q8_0_rowvec8_dual_split_gemv_decode_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv import (
        gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out,
        gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out,
        gguf_q8_0_t16_dual_gemv_decode_rowtile4_col8_bf16_bf16_out,
    )

    fused = [
        ("pack8_dual_gate_up (P9.B3)", gguf_q8_0_pack8_dual_gate_up_gemv_decode_bf16_bf16_out, "raw"),
        ("t16_dual_gemv_decode", gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out, "t16tiles"),
        ("t16_dual_rowtile4", gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out, "t16tiles"),
        ("t16_dual_rowtile4_col8", gguf_q8_0_t16_dual_gemv_decode_rowtile4_col8_bf16_bf16_out, "t16tiles"),
    ]
    split = [
        ("rowvec8_dual_split t=32", gguf_q8_0_rowvec8_dual_split_gemv_decode_bf16_bf16_out, 32),
        ("rowvec8_dual_split t=64", gguf_q8_0_rowvec8_dual_split_gemv_decode_bf16_bf16_out, 64),
        ("rowvec8_dual_split t=128", gguf_q8_0_rowvec8_dual_split_gemv_decode_bf16_bf16_out, 128),
    ]
    return fused, split


# single-output production shapes: (label, K, out). Covers the post-fusion
# resident singles: fused qkv and fused gate_up buffers (materialized concat),
# attn_output at both column widths, k_eq_v layers' separate q/k, and down.
SINGLE_SHAPES = [
    ("gate/up single", 2816, 2112),
    ("fused gate/up", 2816, 4224),
    ("q projection", 2816, 4096),
    ("fused qkv", 2816, 8192),
    ("k_eq_v k (1kv)", 2816, 1024),
    ("k_eq_v k (2kv)", 2816, 2048),
    ("o projection", 4096, 2816),
    ("o projection wide", 8192, 2816),
    ("down projection", 2112, 2816),
]
PAIR_SHAPE = ("gate+up pair", 2816, 2112, 2112)


def main() -> int:
    runtime = get_hip_runtime()
    # (shape, candidate, out, us, abs_err_ref, abs_err_legacy, status)
    rows: list[tuple] = []

    # ---- single-output matrix
    single_cands = candidates_single()
    for label, k, out in SINGLE_SHAPES:
        x = make_x(k)
        w = make_w(out, k)
        w_t16 = make_t16_tiles(w, out, k)
        o_ptr = make_out(out * 2)
        ref = round_bf16_rne(dequant_ref(x, w, out, k))
        legacy_out: np.ndarray | None = None
        for name, fn, kind in single_cands:
            w_arg = w if kind == "raw" else w_t16
            err_ref: float | None = None
            err_leg: float | None = None
            try:
                call_single(fn, x, w_arg, o_ptr, 1, k, out)
                runtime.device_synchronize()
            except Exception as exc:  # noqa: BLE001 - record, don't die
                rows.append((label, name, out, None, None, None, f"launch-error: {type(exc).__name__}"))
                continue
            if not touch_ok(o_ptr, out * 2):
                rows.append((label, name, out, None, None, None, "touch-fail (no writes)"))
                continue
            got = bf16_as_f32(o_ptr, out)
            err_ref = float(np.max(np.abs(got - ref)))
            if name.startswith("pack8_gemv (legacy"):
                legacy_out = got.copy()
            elif legacy_out is not None:
                err_leg = float(np.max(np.abs(got - legacy_out)))
            try:
                us = timed(call_single, (fn, x, w_arg, o_ptr, 1, k, out))
                rows.append((label, name, out, us, err_ref, err_leg, "ok"))
            except Exception as exc:  # noqa: BLE001
                rows.append((label, name, out, None, err_ref, err_leg, f"time-error: {type(exc).__name__}"))

    # ---- pair matrix (fused out [1, a+b] and split out pair)
    label, k, a, b = PAIR_SHAPE
    x = make_x(k)
    w_a = make_w(a, k)
    w_b = make_w(b, k)
    ta = make_t16_tiles(w_a, a, k)
    tb = make_t16_tiles(w_b, b, k)
    fused_out = make_out((a + b) * 2)
    out_a = make_out(a * 2)
    out_b = make_out(b * 2)
    ref_a = round_bf16_rne(dequant_ref(x, w_a, a, k))
    ref_b = round_bf16_rne(dequant_ref(x, w_b, b, k))
    # parent parity baseline: legacy singles into split outputs
    single_legacy = single_cands[0][1]
    call_single(single_legacy, x, w_a, out_a, 1, k, a)
    call_single(single_legacy, x, w_b, out_b, 1, k, b)
    runtime.device_synchronize()
    leg_a = bf16_as_f32(out_a, a)
    leg_b = bf16_as_f32(out_b, b)

    fused, split = candidates_pair()
    for name, fn, kind in fused:
        w_arg_a, w_arg_b = (w_a, w_b) if kind == "raw" else (ta, tb)
        try:
            if kind == "raw":
                fn(x, w_arg_a, w_arg_b, fused_out, 1, k, a, b)
            else:
                # t16 duals are split-output: out_a, out_b positional
                fn(x, w_arg_a, w_arg_b, out_a, out_b, 1, k, a, b)
            runtime.device_synchronize()
            if kind == "raw":
                if not touch_ok(fused_out, (a + b) * 2):
                    rows.append((label, name, a + b, None, None, None, "touch-fail (no writes)"))
                    continue
                got = bf16_as_f32(fused_out, a + b)
                got_a, got_b = got[:a], got[a:]
                args = (x, w_arg_a, w_arg_b, fused_out, 1, k, a, b)
            else:
                if not touch_ok(out_a, a * 2) or not touch_ok(out_b, b * 2):
                    rows.append((label, name, a + b, None, None, None, "touch-fail (no writes)"))
                    continue
                got_a, got_b = bf16_as_f32(out_a, a), bf16_as_f32(out_b, b)
                args = (x, w_arg_a, w_arg_b, out_a, out_b, 1, k, a, b)
            err_ref = max(
                float(np.max(np.abs(got_a - ref_a))),
                float(np.max(np.abs(got_b - ref_b))),
            )
            err_leg = max(
                float(np.max(np.abs(got_a - leg_a))),
                float(np.max(np.abs(got_b - leg_b))),
            )
            us = timed(fn, args)
            rows.append((label, name, a + b, us, err_ref, err_leg, "ok"))
        except Exception as exc:  # noqa: BLE001
            rows.append((label, name, a + b, None, None, None, f"launch-error: {type(exc).__name__}"))
    for name, fn, threads in split:
        try:
            fn(x, w_a, w_b, out_a, out_b, 1, k, a, b, threads=threads)
            runtime.device_synchronize()
            if not touch_ok(out_a, a * 2) and not touch_ok(out_b, b * 2):
                rows.append((label, name, a + b, None, None, None, "touch-fail (no writes)"))
                continue
            got_a, got_b = bf16_as_f32(out_a, a), bf16_as_f32(out_b, b)
            err_ref = max(
                float(np.max(np.abs(got_a - ref_a))),
                float(np.max(np.abs(got_b - ref_b))),
            )
            err_leg = max(
                float(np.max(np.abs(got_a - leg_a))),
                float(np.max(np.abs(got_b - leg_b))),
            )
            us = timed(
                lambda *args, _t=threads: fn(*args, threads=_t),
                (x, w_a, w_b, out_a, out_b, 1, k, a, b),
            )
            rows.append((label, name, a + b, us, err_ref, err_leg, "ok"))
        except Exception as exc:  # noqa: BLE001
            rows.append((label, name, a + b, None, None, None, f"launch-error: {type(exc).__name__}"))

    # ---- report
    print("shape,out,candidate,us_per_launch,abs_err_vs_f32ref,abs_err_vs_legacy,status")
    for shape, name, out, us, err_ref, err_leg, status in rows:
        us_s = f"{us:.1f}" if us is not None else ""
        er_s = f"{err_ref:.4g}" if err_ref is not None else ""
        el_s = f"{err_leg:.4g}" if err_leg is not None else ""
        print(f"{shape},{out},{name},{us_s},{er_s},{el_s},{status}")
    print("\n== per-shape best (ok only) ==")
    by_shape: dict[str, list] = {}
    for shape, name, out, us, err_ref, err_leg, status in rows:
        if status == "ok" and us is not None:
            by_shape.setdefault(shape, []).append((us, name))
    for shape, vals in by_shape.items():
        vals.sort()
        best_us, best_name = vals[0]
        worst = vals[-1]
        print(
            f"{shape}: best={best_us:.1f}us ({best_name})  "
            f"worst={worst[0]:.1f}us ({worst[1]})  spread={worst[0] / best_us:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())