#!/usr/bin/env python3
"""Gemma 4 layer-29 expert-decode screen: production selected GEMVs vs candidates.

Punchlist D11 (with P3 for the prefill half): layer 29 carries the artifact's
only Q5_K gate_up stack and only Q8_0 down expert stack, and its decode costs
``moe.l29_experts`` 0.244 ms per token -- 0.1792 ms for the Q5_K gate_up and
0.0625 ms for the Q8_0 down, two launches on grid (180224, 8, 1) with 128
threads, measured in the 2026-09-30 census kernel trace against a 0.03 ms
family floor. Both projections run the shared raw-layout selected leaf
(``gguf_k_selected_prefill_out_kernel``) rather than a tuned owner; the 29
normal layers already run tuned owners (``qk_t16_selected_direct_gemv`` at
~62 us for gate_up, ``q5_1_selected_gemv_logical256_t64`` for down).

This screen measures, at the exact production geometry, same-run same-bytes:

  gate_up (Q5_K, in 2816, out 1408, E 128, rows 8)
    production   gguf_q5_k_selected_gemv_bf16_bf16_out   (incumbent)
    pack8        gguf_q5_k_selected_pack8_gemv_bf16_bf16_out
    pack8_dp4a   gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out
    t16_direct   gguf_q5_k_t16_selected_gemv_bf16_bf16_out  (repacked tiles)

  down (Q8_0, in 704, out 2816, E 128, rows 8)
    production   gguf_q8_0_selected_gemv_bf16_bf16_out   (incumbent)
    pack8        gguf_q8_0_selected_pack8_gemv_bf16_bf16_out

The grouped selected variants (``selected_grouped*`` for both quants) were
screened first and are excluded here: they fault the device under this
caller's per-lane ABI (memory access fault, lane_to_row=None path) -- they
serve the CSR expert_start grouped path, not the per-lane selected caller.
The q5_k pack8_dp4a arm is kept and reported as rejected: it launches but
produces garbage under this ABI (rel_to_peak 5.4e4).

There is no registered Q8_0 t16 selected leaf (the t16 family covers
q4_k/q5_k/q6_k), so no t16 arm runs for down; that gap is the row's finding if
the raw candidates do not win.

Every arm reads the same raw byte stack on device (the t16 arm reads the
bit-lossless repack of the gate_up stack). Agreement runs every arm against the
incumbent before any timing, D3 screen protocol. Diagnostic only; no runtime
default changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.hip import get_hip_runtime

QK_K = 256
QK_Q8_0 = 32
Q5_K_BLOCK_BYTES = 176
Q8_0_BLOCK_BYTES = 34

# In-situ census reference, 2026-09-30 kernel trace of the fixture at
# prompt 8 / 8 decode steps: moe.l29_experts, grid (180224, 8, 1), wg 128.
CENSUS_SITU_US = {"gate_up": 179.2, "down": 62.5}


def _git_head() -> str:
    import subprocess

    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return "unknown"


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f = np.ascontiguousarray(arr, dtype=np.float32)
    bits = f.view(np.uint32)
    lsb = (bits >> 16) & 1
    bits = bits + np.uint32(0x7FFF + lsb)
    return (bits >> 16).astype(np.uint16)


def _bf16_u16_to_f32(bits: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32)
    return (b << 16).view(np.float32)


def _softmax_kl(ref: np.ndarray, cand: np.ndarray) -> float:
    def logsm(x: np.ndarray) -> np.ndarray:
        x = x.ravel().astype(np.float64)
        x = x - x.max()
        return x - np.log(np.exp(x).sum())

    p = logsm(ref)
    q = logsm(cand)
    return float((np.exp(p) * (p - q)).sum())


def _stack_experts(base: np.ndarray, experts: int) -> np.ndarray:
    """One raw stack of ``experts`` rows-varied copies, D3 screen protocol."""

    return np.ascontiguousarray(
        np.stack([np.roll(base, e + 1, 0) for e in range(experts)])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-up-in", type=int, default=2816)
    ap.add_argument("--gate-up-out", type=int, default=1408)
    ap.add_argument("--down-in", type=int, default=704)
    ap.add_argument("--down-out", type=int, default=2816)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--rows", type=int, default=8, help="compact decode lanes")
    ap.add_argument(
        "--arms",
        default=None,
        help="comma list: restrict to named arms (debug bisect)",
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        build_gguf_k_gemv,
        gguf_q5_k_selected_gemv_bf16_bf16_out,
        gguf_q5_k_selected_pack8_gemv_bf16_bf16_out,
        gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out,
        gguf_q8_0_selected_gemv_bf16_bf16_out,
        gguf_q8_0_selected_pack8_gemv_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        gguf_q5_k_t16_selected_gemv_bf16_bf16_out,
    )
    from hipengine.quant.gguf_t16 import repack_gguf_q5_k_tile16
    from tests._gguf_synthetic_weights import make_q5_k_weight, make_q8_0_weight

    G_H, G_O = args.gate_up_in, args.gate_up_out
    D_H, D_O = args.down_in, args.down_out
    E, R = args.experts, args.rows
    if G_H % QK_K:
        raise SystemExit(f"gate_up in_features {G_H} must be a multiple of {QK_K}")
    if D_H % QK_Q8_0:
        raise SystemExit(f"down in_features {D_H} must be a multiple of {QK_Q8_0}")
    build_gguf_k_gemv(load=True)
    rt = get_hip_runtime()

    rng = np.random.default_rng(20260931)

    def bf16_x(h: int) -> np.ndarray:
        return _f32_to_bf16_u16(
            (rng.standard_normal((R, h)) * 1e-3).astype(np.float32)
        )

    # Distinct ascending experts, like production top-8 lanes.
    selected = np.ascontiguousarray(np.arange(R, dtype=np.int64) % E)

    raw_gate = _stack_experts(make_q5_k_weight(G_O, G_H), E)
    raw_down = _stack_experts(make_q8_0_weight(D_O, D_H), E)
    gate_tiles = repack_gguf_q5_k_tile16(raw_gate)
    tiles = np.ascontiguousarray(gate_tiles.tiles)

    bufs: list = []

    def dev(arr: np.ndarray) -> int:
        b = malloc(arr.nbytes)
        copy_host_to_device(b, host_array_ptr(arr), arr.nbytes)
        bufs.append(b)
        return b

    def bench(fn) -> float:
        for _ in range(args.warmup):
            fn()
        rt.device_synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        rt.device_synchronize()
        return (time.perf_counter() - t0) / args.iters * 1e3  # ms/call

    results: dict = {}

    try:
        xb_g, xb_d = dev(bf16_x(G_H)), dev(bf16_x(D_H))
        sb = dev(selected)
        gb, db, tb = dev(raw_gate), dev(raw_down), dev(tiles)

        for proj, cfg in (
            (
                "gate_up",
                {
                    "quant": "gguf_q5_k",
                    "x_ptr": xb_g,
                    "w_ptr": gb,
                    "in_features": G_H,
                    "out_features": G_O,
                    "bytes_per_row": G_H // QK_K * Q5_K_BLOCK_BYTES,
                    "raw_nbytes": raw_gate.nbytes,
                    "extra_bytes": {"t16_tiles": tiles.nbytes},
                },
            ),
            (
                "down",
                {
                    "quant": "gguf_q8_0",
                    "x_ptr": xb_d,
                    "w_ptr": db,
                    "in_features": D_H,
                    "out_features": D_O,
                    "bytes_per_row": D_H // QK_Q8_0 * Q8_0_BLOCK_BYTES,
                    "raw_nbytes": raw_down.nbytes,
                    "extra_bytes": {},
                },
            ),
        ):
            H, O = cfg["in_features"], cfg["out_features"]
            out = np.zeros((R, O), np.uint16)
            ob = dev(out)

            def selected_arm(fn, out_buf=ob, w=cfg["w_ptr"], x=cfg["x_ptr"]):
                return lambda: fn(
                    x.ptr, sb.ptr, w.ptr, out_buf.ptr, R, R, E, H, O
                )

            arms: list[tuple[str, str, object]] = [
                (
                    "production",
                    "selected",
                    selected_arm(
                        gguf_q5_k_selected_gemv_bf16_bf16_out
                        if proj == "gate_up"
                        else gguf_q8_0_selected_gemv_bf16_bf16_out
                    ),
                ),
            ]
            if proj == "gate_up":
                arms += [
                    ("pack8", "selected", selected_arm(gguf_q5_k_selected_pack8_gemv_bf16_bf16_out)),
                    ("pack8_dp4a", "selected", selected_arm(gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out)),
                    (
                        "t16_direct",
                        "selected",
                        lambda: gguf_q5_k_t16_selected_gemv_bf16_bf16_out(
                            xb_g.ptr, sb.ptr, tb.ptr, ob.ptr, R, R, E, H, O
                        ),
                    ),
                ]
            else:
                arms += [
                    ("pack8", "selected", selected_arm(gguf_q8_0_selected_pack8_gemv_bf16_bf16_out)),
                ]

            # Launch health: a wrapper that refuses its geometry is recorded,
            # not fatal -- the screen reports which candidates are runnable.
            if args.arms:
                wanted = {a.strip() for a in args.arms.split(",")}
                arms = [a for a in arms if a[0] in wanted]
            live: list[tuple[str, object]] = []
            errors: dict[str, str] = {}
            for name, abi, fn in arms:
                try:
                    fn()
                    rt.device_synchronize()
                    live.append((name, fn))
                except Exception as exc:  # noqa: BLE001 - screen records, not raises
                    errors[name] = f"{type(exc).__name__}: {exc}"

            # Agreement first: a broken fixture must not yield a bandwidth
            # number. Production must produce; every other live arm must
            # agree with it (D3 gate: rel_to_peak <= 2e-2, finite KL).
            runs: dict[str, np.ndarray] = {}
            for name, fn in live:
                fn()
                rt.device_synchronize()
                copy_device_to_host(host_array_ptr(out), ob, out.nbytes)
                runs[name] = out.copy()
            ref = _bf16_u16_to_f32(runs["production"])
            ref_scale = float(np.max(np.abs(ref)))
            if not np.isfinite(ref_scale) or ref_scale == 0.0:
                raise SystemExit(
                    f"{proj}: production arm produced no usable output "
                    f"(max_abs={ref_scale})"
                )
            agreement: dict[str, dict] = {}
            for name in runs:
                if name == "production":
                    continue
                cand = _bf16_u16_to_f32(runs[name])
                kl = _softmax_kl(ref, cand)
                max_abs = float(np.max(np.abs(ref - cand)))
                rel = max_abs / ref_scale
                bit_equal = float(np.mean(runs["production"] == runs[name]))
                agreement[name] = {
                    "kl_vs_production": kl,
                    "max_abs": max_abs,
                    "rel_to_peak": rel,
                    "bit_equal_fraction": bit_equal,
                    "accepted": bool(np.isfinite(kl) and rel <= 2e-2),
                }

            timing = {name: bench(fn) for name, fn in live}
            selected_bytes = R * O * cfg["bytes_per_row"]

            def gbs(ms: float) -> float:
                return selected_bytes / (ms * 1e-3) / 1e9

            situ = CENSUS_SITU_US[proj]
            prod_ms = timing.get("production")
            results[proj] = {
                "shape": {
                    "quant": cfg["quant"],
                    "in_features": H,
                    "out_features": O,
                    "experts": E,
                    "rows": R,
                    "bytes_per_row": cfg["bytes_per_row"],
                    "selected_bytes": selected_bytes,
                    "raw_stack_nbytes": cfg["raw_nbytes"],
                    **{
                        f"extra_{k}_nbytes": v
                        for k, v in cfg["extra_bytes"].items()
                    },
                },
                "timing": {
                    name: {
                        "ms_per_call": ms,
                        "gbps": gbs(ms),
                        "accepted": agreement.get(name, {}).get(
                            "accepted", name == "production"
                        ),
                    }
                    for name, ms in timing.items()
                },
                "speedup_production_over": {
                    name: prod_ms / ms for name, ms in timing.items()
                },
                "agreement": agreement,
                "launch_errors": errors,
                "reference": {
                    "census_situ_us": situ,
                    "production_ms_over_situ": (
                        prod_ms * 1e3 / situ if prod_ms else None
                    ),
                    "note": (
                        "in-situ census: moe.l29_experts launch at grid "
                        "(180224, 8, 1), wg 128, from the 2026-09-30 kernel "
                        "trace of the fixture (prompt 8, 8 decode steps)"
                    ),
                },
            }

            print(f"== {proj}: in={H} out={O} E={E} rows={R}")
            for name, ms in timing.items():
                acc = results[proj]["timing"][name]["accepted"]
                print(
                    f"  {name:<24}: {ms:.4f} ms/call  {gbs(ms):6.1f} GB/s  "
                    f"({prod_ms / ms:.3f}x vs production)  "
                    f"accepted={acc}"
                )
            for name, agr in agreement.items():
                print(
                    f"  {name:<24}: kl={agr['kl_vs_production']:.3e} "
                    f"rel={agr['rel_to_peak']:.3e} "
                    f"bit_equal={agr['bit_equal_fraction']:.4f}"
                )
            for name, err in errors.items():
                print(f"  {name:<24}: LAUNCH ERROR {err}")

        artifact = {
            "kind": "gemma4-d11-l29-expert-decode-screen",
            "provenance": {
                "git": _git_head(),
                "gpu_env": {
                    k: os.environ[k]
                    for k in ("ROCR_VISIBLE_DEVICES",)
                    if k in os.environ
                },
                "iters": args.iters,
                "warmup": args.warmup,
                "command_args": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()
                },
                "protocol": (
                    "same-run same-bytes A/B; all launchable arms warmed "
                    "before timing (agreement phase runs every arm against "
                    "production first; D3 screen protocol)"
                ),
            },
            "projections": results,
            "fixture": {
                "tensor_shapes": {
                    "blk.29.ffn_gate_up_exps.weight": [E, G_O, G_H],
                    "blk.29.ffn_down_exps.weight": [E, D_O, D_H],
                },
                "note": (
                    "read from the benchmark fixture; byte_shape drives the "
                    "screen geometry, synthetic valid blocks stand in for the "
                    "tensor bytes"
                ),
            },
        }
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(artifact, indent=1, sort_keys=True) + "\n"
            )
            print(f"wrote {args.out}")
    finally:
        for b in bufs:
            free(b)


if __name__ == "__main__":
    main()