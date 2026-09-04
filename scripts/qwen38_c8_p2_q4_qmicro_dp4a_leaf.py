#!/usr/bin/env python3
"""Leaf A/B: grouped q8_1 DP4A qmicro-Q4 versus the retained rowtile16-w2 owner.

C8-P2 Q4 pool screen (iteration 43). The new grouped dp4a kernel (rows
8-64, qmicro T16 tiles consumed directly) is timed against the retained
grouped Q4 owner at the production row counts on actual Qwen3.8 Q4_K_M
weights. Both arms share the SAME plain q8_1 activation producer
(gguf_q4_k_quantize_bf16_q8_1), so the arm difference is purely the GEMV
kernel. Arms differ in arithmetic (integer dp4a decode vs BF16 decode);
outputs are compared informationally (mismatch, KL), not gated on
exactness. Kernel-level oracle/floor/determinism contracts live in
tests/test_gguf_q4_k_qmicro_dp4a_grouped_gemv.py.
"""
from __future__ import annotations

import argparse
import statistics
from datetime import datetime, timezone

import numpy as np


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
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--burst", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 24, 32])
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
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_qmicro_dp4a_grouped import (
        build_gguf_q4_k_qmicro_dp4a_grouped,
        gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        build_gguf_t16_selected_gemv,
        gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out,
        gguf_q4_k_t16_dense_rowtile16_w2_grouped_rows8_bf16_bf16_out,
    )
    from hipengine.quant.gguf_q4_k import (
        GGUF_Q4_K_TILE16_BLOCK_BYTES,
        repack_gguf_q4_k_tile16,
        repack_gguf_q4_k_tile16_qmicro,
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
    t16_library = build_gguf_t16_selected_gemv(load=True)
    dp4a_library = build_gguf_q4_k_qmicro_dp4a_grouped(load=True)

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
    control_wins = 0
    candidate_wins = 0
    for index, (role, name, out_features, in_features) in enumerate(cases):
        raw = np.ascontiguousarray(reader.tensor_data(name))
        t16_tiles = np.ascontiguousarray(
            repack_gguf_q4_k_tile16(raw[None, ...]).tiles
        )
        qmicro_tiles = np.ascontiguousarray(
            repack_gguf_q4_k_tile16_qmicro(raw[None, ...]).tiles
        )
        assert qmicro_tiles.shape[-1] == 2304
        assert t16_tiles.shape[-1] == GGUF_Q4_K_TILE16_BLOCK_BYTES
        t16_device = upload(t16_tiles)
        qmicro_device = upload(qmicro_tiles)
        buffers = [t16_device, qmicro_device]
        try:
            for rows in args.rows:
                rng = np.random.default_rng(
                    2_026_090_600 + index * 1000 + rows
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
                control_device = malloc(rows * out_features * 2, runtime=runtime)
                cand_device = malloc(rows * out_features * 2, runtime=runtime)
                buffers.extend((x_device, xq_device, control_device, cand_device))

                def quantize() -> None:
                    gguf_q4_k_quantize_bf16_q8_1(
                        x_device.ptr,
                        xq_device.ptr,
                        rows,
                        in_features,
                        library=q4_library,
                        runtime=runtime,
                    )

                control_fn = (
                    gguf_q4_k_t16_dense_rowtile16_w2_grouped_rows8_bf16_bf16_out
                    if rows >= 16
                    else gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out
                )

                def control() -> None:
                    quantize()
                    control_fn(
                        x_device.ptr,
                        t16_device.ptr,
                        control_device.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=t16_library,
                        runtime=runtime,
                    )

                def candidate() -> None:
                    quantize()
                    gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_bf16_out(
                        xq_device.ptr,
                        qmicro_device.ptr,
                        cand_device.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=dp4a_library,
                        runtime=runtime,
                    )

                control()
                candidate()
                control_out = download_bf16(control_device, (rows, out_features))
                cand_out = download_bf16(cand_device, (rows, out_features))
                mismatches = int(np.count_nonzero(control_out != cand_out))
                finite = bool(
                    np.isfinite(_bf16_f32(control_out)).all()
                    and np.isfinite(_bf16_f32(cand_out)).all()
                )

                control_samples: list[float] = []
                cand_samples: list[float] = []
                for sample in range(args.samples):
                    if (index + sample) % 2 == 0:
                        control_samples.append(event_ms(control))
                        cand_samples.append(event_ms(candidate))
                    else:
                        cand_samples.append(event_ms(candidate))
                        control_samples.append(event_ms(control))
                for _ in range(args.warmups):
                    control()
                    candidate()

                control_med = statistics.median(control_samples)
                cand_med = statistics.median(cand_samples)
                if cand_med < control_med:
                    candidate_wins += 1
                elif control_med < cand_med:
                    control_wins += 1
                results.append(
                    {
                        "role": role,
                        "tensor": name,
                        "rows": rows,
                        "in_features": in_features,
                        "out_features": out_features,
                        "control_ms": control_med,
                        "candidate_ms": cand_med,
                        "candidate_ratio": cand_med / control_med,
                        "mismatches_vs_control": mismatches,
                        "kl_vs_control": _kl(control_out, cand_out),
                        "finite": finite,
                    }
                )
                print(
                    f"[{index + 1}/{len(cases)}] {role} {name} rows={rows}: "
                    f"control {control_med:.4f} ms candidate {cand_med:.4f} ms "
                    f"ratio {cand_med / control_med:.4f} mismatches {mismatches}",
                    flush=True,
                )
        finally:
            for buffer in buffers:
                free(buffer, runtime=runtime)

    payload = {
        "schema": 1,
        "kind": "w7900_qwen38_q4km_k3_c8_p2_q4_qmicro_dp4a_grouped_leaf",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": detect_device_name(),
        "target_arches": detect_hip_target_arches(),
        "model": args.model,
        "arms": {
            "control": (
                "gguf_q4_k_quantize_bf16_q8_1 + gguf_q4_k_t16_dense_rowtile16_w2"
                "[_grouped_rows8]_bf16_bf16_out (retained owner, T16 tiles)"
            ),
            "candidate": (
                "gguf_q4_k_quantize_bf16_q8_1 + "
                "gguf_q4_k_qmicro_q8_1_dp4a_grouped_bf16_bf16_out (grouped "
                "integer decode, qmicro T16 tiles consumed directly)"
            ),
        },
        "rows": args.rows,
        "samples": args.samples,
        "burst": args.burst,
        "warmups": args.warmups,
        "control_wins": control_wins,
        "candidate_wins": candidate_wins,
        "results": results,
    }
    with open(args.output, "w") as handle:
        import json

        json.dump(payload, handle, indent=1)
        handle.write("\n")
    print(f"candidate wins {candidate_wins} / control wins {control_wins}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
