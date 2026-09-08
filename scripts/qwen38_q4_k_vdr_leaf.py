#!/usr/bin/env python3
"""Leaf A/B/C: dense single-row Q4_K decode — production pack8 float GEMV
versus the q8_1-DP4A control and the nasone32-style VDR load-reuse kernel.

2026-09-08 engine comparison follow-up (benchmarks/results/
2026-09-08-rx7900xtx-engine-comparison.md, Optimization Review item 1:
"nasone32 K-quant load reuse ... worth a shaped kernel comparison").
Three arms at the production decode shape (rows=1) on actual Qwen3.8
Q4_K_M weights:

  * production: gguf_q4_k_pack8_gemv_decode_bf16_bf16_out — the retained
    exact float-FMA decode owner (raw GGUF bytes, BF16 activations, no
    activation-quantize launch).
  * ctl: gguf_q4_k_quantize_bf16_q8_1 + the unamortized q8_1-dp4a control
    (per-4-pack header/scale/min re-decode, subblock-strided).
  * vdr: same producer + the adjacent-chunk amortized kernel (nasone32
    efa4e864 vecdotq.cuh VDR idea: header/scale/min hoisted once per
    32-element subblock, activation packs reused across 8 columns).

ctl and vdr are bit-identical by construction (tests/
test_gguf_q4_k_q8_1_dp4a_vdr_gemv.py); production is a different
arithmetic class (float activations) and is compared informationally
(mismatch/KL), not gated on exactness.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def _bf16_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def _kl(reference: np.ndarray, actual: np.ndarray) -> float:
    ref = _bf16_f32(reference).astype(np.float64)
    act = _bf16_f32(actual).astype(np.float64)
    ref = ref - ref.max(axis=1, keepdims=True)
    act = act - act.max(axis=1, keepdims=True)
    ref_p = np.exp(ref)
    ref_p /= ref_p.sum(axis=1, keepdims=True)
    act_p = np.exp(act)
    act_p /= act_p.sum(axis=1, keepdims=True)
    return float(np.sum(ref_p * (np.log(ref_p) - np.log(act_p))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--burst", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.loading.gguf import GGUFReader
    from hipengine.benchmark.provenance import detect_device_name
    from hipengine.kernels.backends import detect_hip_target_arches
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_quantize_bf16_q8_1,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_pack8_gemv import (
        build_gguf_q4_k_pack8_gemv,
        gguf_q4_k_pack8_gemv_decode_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        build_gguf_t16_selected_gemv,
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
        gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out,
    )
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_dp4a_vdr_gemv import (
        build_gguf_q4_k_q8_1_dp4a_vdr_gemv,
        gguf_q4_k_q8_1_dp4a_ctl_bf16_bf16_out,
        gguf_q4_k_q8_1_dp4a_vdr_bf16_bf16_out,
    )

    runtime = get_hip_runtime()
    reader = GGUFReader(args.model)
    roles = (
        ("ffn_down", "blk.8.ffn_down.weight"),
        ("recurrent_qkv", "blk.8.attn_qkv.weight"),
        ("attention_v", "blk.11.attn_v.weight"),
        ("attention_q", "blk.11.attn_q.weight"),
    )
    cases = []
    for role, name in roles:
        info = reader.tensor_info(name)
        if info.ggml_type_name != "Q4_K":
            raise ValueError(f"{name} is {info.ggml_type_name}, expected Q4_K")
        cases.append((role, name, int(info.shape[0]), int(info.shape[1])))
    if args.limit:
        cases = cases[: args.limit]

    q4_library = build_gguf_q4_k_gemv(load=True)
    pack8_library = build_gguf_q4_k_pack8_gemv(load=True)
    vdr_library = build_gguf_q4_k_q8_1_dp4a_vdr_gemv(load=True)
    t16_library = build_gguf_t16_selected_gemv(load=True)

    def upload(host: np.ndarray):
        host = np.ascontiguousarray(host)
        buffer = malloc(host.nbytes, runtime=runtime)
        copy_host_to_device(buffer, host_array_ptr(host), runtime=runtime)
        return buffer

    def download_bf16(buffer, shape):
        host = np.empty(shape, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(host), buffer, runtime=runtime)
        return host

    def event_ms(function) -> float:
        start = runtime.event_create()
        stop = runtime.event_create()
        try:
            runtime.event_record(start)
            for _ in range(args.burst):
                function()
            runtime.event_record(stop)
            runtime.event_synchronize(stop)
            return float(runtime.event_elapsed_time_ms(start, stop)) / args.burst
        finally:
            runtime.event_destroy(stop)
            runtime.event_destroy(start)

    results = []
    wins = {"production": 0, "ctl": 0, "vdr": 0}
    for index, (role, name, out_features, in_features) in enumerate(cases):
        raw = np.ascontiguousarray(reader.tensor_data(name))
        raw_device = upload(raw)
        t16_tiles = np.ascontiguousarray(
            repack_gguf_q4_k_tile16(raw[None, ...]).tiles
        )
        t16_device = upload(t16_tiles)
        buffers = [raw_device, t16_device]
        try:
            for rows in args.rows:
                rng = np.random.default_rng(
                    2_026_090_900 + index * 1000 + rows
                )
                x = _bf16_bits(
                    rng.normal(0.0, 0.2, size=(rows, in_features)).astype(
                        np.float32
                    )
                )
                x_device = upload(x)
                xq_device = malloc(
                    rows * (in_features // 32) * 36, runtime=runtime
                )
                prod_device = malloc(rows * out_features * 2, runtime=runtime)
                ctl_device = malloc(rows * out_features * 2, runtime=runtime)
                vdr_device = malloc(rows * out_features * 2, runtime=runtime)
                buffers.extend(
                    (x_device, xq_device, prod_device, ctl_device, vdr_device)
                )

                def quantize() -> None:
                    gguf_q4_k_quantize_bf16_q8_1(
                        x_device.ptr,
                        xq_device.ptr,
                        rows,
                        in_features,
                        library=q4_library,
                        runtime=runtime,
                    )

                def production() -> None:
                    if rows == 1:
                        owner = gguf_q4_k_t16_dense_single_local32_bf16_bf16_out
                    else:
                        owner = gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out
                    owner(
                        x_device.ptr,
                        t16_device.ptr,
                        prod_device.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=t16_library,
                        runtime=runtime,
                    )

                def pack8_raw() -> None:
                    gguf_q4_k_pack8_gemv_decode_bf16_bf16_out(
                        x_device.ptr,
                        raw_device.ptr,
                        prod_device.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=pack8_library,
                        runtime=runtime,
                    )

                def ctl() -> None:
                    quantize()
                    gguf_q4_k_q8_1_dp4a_ctl_bf16_bf16_out(
                        xq_device.ptr,
                        raw_device.ptr,
                        ctl_device.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=vdr_library,
                        runtime=runtime,
                    )

                def vdr() -> None:
                    quantize()
                    gguf_q4_k_q8_1_dp4a_vdr_bf16_bf16_out(
                        xq_device.ptr,
                        raw_device.ptr,
                        vdr_device.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=vdr_library,
                        runtime=runtime,
                    )

                production()
                pack8_raw()
                ctl()
                vdr()
                prod_out = download_bf16(prod_device, (rows, out_features))
                ctl_out = download_bf16(ctl_device, (rows, out_features))
                vdr_out = download_bf16(vdr_device, (rows, out_features))
                ctl_vdr_mismatch = int(np.count_nonzero(ctl_out != vdr_out))
                prod_ctl_mismatch = int(np.count_nonzero(prod_out != ctl_out))
                finite = bool(
                    np.isfinite(_bf16_f32(prod_out)).all()
                    and np.isfinite(_bf16_f32(ctl_out)).all()
                    and np.isfinite(_bf16_f32(vdr_out)).all()
                )
                if ctl_vdr_mismatch:
                    raise AssertionError(
                        f"{name} rows={rows}: ctl/vdr differ in "
                        f"{ctl_vdr_mismatch} outputs (RED contract)"
                    )

                samples = {"production": [], "pack8_raw": [], "ctl": [], "vdr": []}
                functions = {
                    "production": production,
                    "pack8_raw": pack8_raw,
                    "ctl": ctl,
                    "vdr": vdr,
                }
                orderings = [
                    ["production", "pack8_raw", "ctl", "vdr"],
                    ["vdr", "production", "pack8_raw", "ctl"],
                    ["ctl", "vdr", "production", "pack8_raw"],
                    ["pack8_raw", "ctl", "vdr", "production"],
                ]
                for sample in range(args.samples):
                    for arm in orderings[sample % 4]:
                        samples[arm].append(event_ms(functions[arm]))
                for _ in range(args.warmups):
                    production()
                    pack8_raw()
                    ctl()
                    vdr()

                medians = {
                    arm: statistics.median(vals) for arm, vals in samples.items()
                }
                best = min(medians, key=medians.get)
                wins[best] += 1
                results.append(
                    {
                        "role": role,
                        "tensor": name,
                        "rows": rows,
                        "in_features": in_features,
                        "out_features": out_features,
                        "production_ms": medians["production"],
                        "pack8_raw_ms": medians["pack8_raw"],
                        "ctl_ms": medians["ctl"],
                        "vdr_ms": medians["vdr"],
                        "vdr_vs_ctl": medians["vdr"] / medians["ctl"],
                        "vdr_vs_production": medians["vdr"] / medians["production"],
                        "ctl_vs_production": medians["ctl"] / medians["production"],
                        "pack8_vs_production": medians["pack8_raw"] / medians["production"],
                        "ctl_vdr_mismatch": ctl_vdr_mismatch,
                        "prod_ctl_mismatch": prod_ctl_mismatch,
                        "kl_prod_vs_ctl": _kl(prod_out, ctl_out),
                        "finite": finite,
                    }
                )
                print(
                    f"[{index + 1}/{len(cases)}] {role} {name} rows={rows}: "
                    f"prod {medians['production']:.4f} ms "
                    f"(pack8 {medians['pack8_raw']:.4f}) "
                    f"ctl {medians['ctl']:.4f} ms "
                    f"vdr {medians['vdr']:.4f} ms "
                    f"vdr/ctl {medians['vdr'] / medians['ctl']:.4f} "
                    f"vdr/prod {medians['vdr'] / medians['production']:.4f}",
                    flush=True,
                )
        finally:
            for buffer in buffers:
                free(buffer, runtime=runtime)

    payload = {
        "schema": 1,
        "kind": "qwen38_q4km_dense_q4_k_vdr_load_reuse_leaf",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": detect_device_name(),
        "target_arches": detect_hip_target_arches(),
        "model": args.model,
        "arms": {
            "production": (
                "rows=1: gguf_q4_k_t16_dense_single_local32_bf16_bf16_out; "
                "rows 2-8: gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out "
                "(retained exact T16 decode owners, DECODE_REPACK=1 route)"
            ),
            "pack8_raw": (
                "gguf_q4_k_pack8_gemv_decode_bf16_bf16_out (raw-GGUF float-FMA "
                "decode fallback owner)"
            ),
            "ctl": (
                "gguf_q4_k_quantize_bf16_q8_1 + "
                "gguf_q4_k_q8_1_dp4a_ctl_bf16_bf16_out (unamortized q8_1 "
                "dp4a control, per-pack metadata re-decode)"
            ),
            "vdr": (
                "gguf_q4_k_quantize_bf16_q8_1 + "
                "gguf_q4_k_q8_1_dp4a_vdr_bf16_bf16_out (nasone32 efa4e864 "
                "adjacent-chunk load amortization, subblock-hoisted "
                "metadata, activations reused across 8 columns)"
            ),
        },
        "donor_source": (
            "nasone32/llama.cpp-RDNA3-7900xtx-opt "
            "efa4e86410c07723deaa458bdadd8c08f1029928 "
            "ggml/src/ggml-cuda/vecdotq.cuh VDR_Q4_K_Q8_1_MMVQ 2->4"
        ),
        "rows": args.rows,
        "samples": args.samples,
        "burst": args.burst,
        "warmups": args.warmups,
        "wins": wins,
        "results": results,
    }
    with open(args.output, "w") as handle:
        import json

        json.dump(payload, handle, indent=1)
        handle.write("\n")
    print(f"wins {wins}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
