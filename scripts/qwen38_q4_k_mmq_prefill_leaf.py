#!/usr/bin/env python3
"""Leaf prefill A/B: dense bulk Q4_K projection — retained T16 WMMA owners
versus the raw-Q4_K x DS4-Q8_1 integer MMQ screen kernels.

2026-09-08 engine-comparison follow-up (worklog pp8192-gap-attribution:
the PP8192 deficit is projection-prefill-dominated, ~63% of timed prefill is
the T16 WMMA projection family; the candidate arithmetic class is nasone32's
raw-Q4_K x Q8_1 integer MMQ with efa4e8641-family load reuse). The
2026-08-12 T16-payload integer rejections bound the screen, so the MMQ arms
consume raw GGUF Q4_K bytes (tests/test_gpu_gguf_q4_k_q8_1_mmq_prefill.py proves
bit-exact ctl/vdr pairs per class plus the DS4 CPU oracle).

Arms per case (real Qwen3.8-27B Q4_K_M weights, bulk prefill rows):

  * t16_prod: the retained float production owner for the shape — the
    gate/up pair runs the fused dual+SiLU 256-row parent
    (gguf_q4_k_t16_dense_dual_wmma_prefill_silu), single matrices run the
    dense 256-row parent (gguf_q4_k_t16_wmma_prefill).
  * pack: gguf_q8_1_mmq_ds4_pack_bf16, timed separately (production would
    pay it once per activation tensor; the pair case shares one pack across
    both matrices).
  * mmq32_ctl / mmq32_vdr: local128 staged-dp4a 32x32-tile consumer.
  * wmma32_ctl / wmma32_vdr: direct-global iu8-WMMA 32x16-tile consumer.

ctl and vdr are bit-identical by construction; t16_prod is a different
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


def _concat(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--burst", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--rows", type=int, nargs="+", default=[512, 1024, 4096])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    from hipengine.core.hip import get_hip_runtime
    from hipengine.benchmark.provenance import detect_device_name
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.loading.gguf import GGUFReader
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
        build_gguf_k_t16_selected_prefill,
        gguf_q4_k_t16_wmma_prefill_bf16_bf16_out,
        gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
        build_gguf_q4_k_q8_1_selected_prefill,
        gguf_q8_1_mmq_ds4_pack_bf16,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_mmq_prefill import (
        build_gguf_q4_k_q8_1_mmq_prefill,
        gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_bf16_out,
        gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_bf16_out,
        gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_bf16_out,
        gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_bf16_out,
    )
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

    runtime = get_hip_runtime()
    reader = GGUFReader(args.model)

    # (role, [tensor names]) — the gate/up pair keeps both matrices so the
    # pack amortizes across them, matching production's shared activation.
    case_defs = [
        ("ffn_gate_up_pair", ["blk.8.ffn_gate.weight", "blk.8.ffn_up.weight"]),
        ("ffn_down", ["blk.8.ffn_down.weight"]),
        ("attention_q", ["blk.11.attn_q.weight"]),
    ]
    if args.limit:
        case_defs = case_defs[: args.limit]

    cases = []
    for role, names in case_defs:
        infos = []
        for name in names:
            info = reader.tensor_info(name)
            if info.ggml_type_name != "Q4_K":
                raise ValueError(
                    f"{name} is {info.ggml_type_name}, expected Q4_K"
                )
            infos.append((name, int(info.shape[0]), int(info.shape[1])))
        cases.append((role, infos))

    t16_library = build_gguf_k_t16_selected_prefill(load=True)
    pack_library = build_gguf_q4_k_q8_1_selected_prefill(load=True)
    mmq_library = build_gguf_q4_k_q8_1_mmq_prefill(load=True)

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
    wins = {
        "t16_prod": 0,
        "mmq32_ctl": 0,
        "mmq32_vdr": 0,
        "wmma32_ctl": 0,
        "wmma32_vdr": 0,
    }
    for index, (role, infos) in enumerate(cases):
        raws = [np.ascontiguousarray(reader.tensor_data(name)) for name, _, _ in infos]
        out_features_total = sum(out for _, out, _ in infos)
        in_features = infos[0][2]
        if any(info[2] != in_features for info in infos):
            raise ValueError(f"{role}: mixed in_features")
        raw_devices = [upload(raw) for raw in raws]
        t16_devices = [
            upload(np.ascontiguousarray(repack_gguf_q4_k_tile16(raw[None, ...]).tiles))
            for raw in raws
        ]
        buffers = [*raw_devices, *t16_devices]
        try:
            for rows in args.rows:
                rng = np.random.default_rng(2_026_091_000 + index * 1000 + rows)
                x = _bf16_bits(
                    rng.normal(0.0, 0.2, size=(rows, in_features)).astype(
                        np.float32
                    )
                )
                x_device = upload(x)
                xq_device = malloc(
                    rows * (in_features // 128) * 144, runtime=runtime
                )
                t16_out_bufs = [
                    malloc(rows * out_f * 2, runtime=runtime)
                    for _, out_f, _ in infos
                ]
                # The fused dual+SiLU owner outputs the SwiGLU product
                # [rows, out_a]; informational production-dispatch timing
                # for the pair case only.
                pair = len(infos) == 2
                out_a = int(infos[0][1])
                dual_silu_buf = (
                    malloc(rows * out_a * 2, runtime=runtime) if pair else None
                )
                mmq_out_bufs = [
                    malloc(rows * out_f * 2, runtime=runtime)
                    for _, out_f, _ in infos
                ]
                buffers.extend(
                    (x_device, xq_device, *t16_out_bufs, *mmq_out_bufs)
                )
                if dual_silu_buf is not None:
                    buffers.append(dual_silu_buf)

                def pack() -> None:
                    gguf_q8_1_mmq_ds4_pack_bf16(
                        x_device.ptr,
                        xq_device.ptr,
                        rows,
                        in_features,
                        library=pack_library,
                        runtime=runtime,
                    )

                def t16_prod() -> None:
                    for tiles, (_, out_f, _), obuf in zip(
                        t16_devices, infos, t16_out_bufs
                    ):
                        gguf_q4_k_t16_wmma_prefill_bf16_bf16_out(
                            x_device.ptr,
                            tiles.ptr,
                            obuf.ptr,
                            rows,
                            in_features,
                            out_f,
                            library=t16_library,
                            runtime=runtime,
                        )

                def t16_dual_silu() -> None:
                    gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out(
                        x_device.ptr,
                        t16_devices[0].ptr,
                        t16_devices[1].ptr,
                        dual_silu_buf.ptr,
                        rows,
                        in_features,
                        out_a,
                        library=t16_library,
                        runtime=runtime,
                    )

                # Build the four MMQ arms; each writes its own per-matrix
                # contiguous outputs.
                arm_specs = {
                    "mmq32_ctl": gguf_q4_k_q8_1_mmq32_ctl_dense_bf16_bf16_out,
                    "mmq32_vdr": gguf_q4_k_q8_1_mmq32_vdr_dense_bf16_bf16_out,
                    "wmma32_ctl": gguf_q4_k_q8_1_wmma32_ctl_dense_bf16_bf16_out,
                    "wmma32_vdr": gguf_q4_k_q8_1_wmma32_vdr_dense_bf16_bf16_out,
                }

                def make_fn(sym):
                    def fn() -> None:
                        for dev, (_, out_f, _), obuf in zip(
                            raw_devices, infos, mmq_out_bufs
                        ):
                            sym(
                                xq_device.ptr,
                                dev.ptr,
                                obuf.ptr,
                                rows,
                                in_features,
                                out_f,
                                library=mmq_library,
                                runtime=runtime,
                            )

                    return fn

                functions = {"t16_prod": t16_prod, "pack": pack}
                for arm_name, sym in arm_specs.items():
                    functions[arm_name] = make_fn(sym)

                # Correctness pass.
                pack()
                t16_prod()
                for arm_name in arm_specs:
                    functions[arm_name]()
                runtime.device_synchronize()
                t16_outs = [
                    download_bf16(obuf, (rows, int(info[1])))
                    for obuf, info in zip(t16_out_bufs, infos)
                ]
                mmq_outs = {}
                for arm_name in arm_specs:
                    mmq_outs[arm_name] = [
                        download_bf16(obuf, (rows, int(info[1])))
                        for obuf, info in zip(mmq_out_bufs, infos)
                    ]
                for cls_arms in (
                    ("mmq32_ctl", "mmq32_vdr"),
                    ("wmma32_ctl", "wmma32_vdr"),
                ):
                    for part_a, part_b in zip(
                        mmq_outs[cls_arms[0]], mmq_outs[cls_arms[1]]
                    ):
                        mismatch = int(np.count_nonzero(part_a != part_b))
                        if mismatch:
                            raise AssertionError(
                                f"{role} rows={rows}: {cls_arms[0]}/{cls_arms[1]} "
                                f"differ in {mismatch} outputs (RED contract)"
                            )
                finite = all(
                    bool(np.isfinite(_bf16_f32(v)).all())
                    for parts in mmq_outs.values()
                    for v in parts
                ) and all(
                    bool(np.isfinite(_bf16_f32(v)).all()) for v in t16_outs
                )

                samples = {arm: [] for arm in functions}
                if pair:
                    functions["t16_dual_silu"] = t16_dual_silu
                    samples["t16_dual_silu"] = []
                orderings = [
                    ["t16_prod", "pack", "mmq32_ctl", "mmq32_vdr", "wmma32_ctl", "wmma32_vdr"],
                    ["wmma32_vdr", "t16_prod", "pack", "mmq32_ctl", "mmq32_vdr", "wmma32_ctl"],
                    ["wmma32_ctl", "wmma32_vdr", "t16_prod", "pack", "mmq32_ctl", "mmq32_vdr"],
                    ["mmq32_vdr", "wmma32_ctl", "wmma32_vdr", "t16_prod", "pack", "mmq32_ctl"],
                    ["mmq32_ctl", "mmq32_vdr", "wmma32_ctl", "wmma32_vdr", "t16_prod", "pack"],
                    ["pack", "mmq32_ctl", "mmq32_vdr", "wmma32_ctl", "wmma32_vdr", "t16_prod"],
                ]
                for sample in range(args.samples):
                    order = list(orderings[sample % len(orderings)])
                    if pair:
                        order.insert(sample % (len(order) + 1), "t16_dual_silu")
                    for arm in order:
                        samples[arm].append(event_ms(functions[arm]))
                for _ in range(args.warmups):
                    t16_prod()
                    pack()
                    for arm_name in arm_specs:
                        functions[arm_name]()

                medians = {
                    arm: statistics.median(vals) for arm, vals in samples.items()
                }
                n_consume = len(infos)
                totals = {
                    arm: medians["pack"] + medians[arm] * n_consume
                    for arm in arm_specs
                }
                best = min(totals, key=totals.get)
                if totals[best] < medians["t16_prod"]:
                    wins[best] += 1
                else:
                    wins["t16_prod"] += 1
                results.append(
                    {
                        "role": role,
                        "tensors": [name for name, _, _ in infos],
                        "rows": rows,
                        "in_features": in_features,
                        "out_features_total": out_features_total,
                        "n_consume": n_consume,
                        "t16_prod_ms": medians["t16_prod"],
                        "pack_ms": medians["pack"],
                        "mmq32_ctl_ms": medians["mmq32_ctl"],
                        "mmq32_vdr_ms": medians["mmq32_vdr"],
                        "wmma32_ctl_ms": medians["wmma32_ctl"],
                        "wmma32_vdr_ms": medians["wmma32_vdr"],
                        "totals_with_pack": totals,
                        "t16_vs_best_mmq": medians["t16_prod"] / min(totals.values()),
                        "mmq32_vdr_vs_ctl": medians["mmq32_vdr"] / medians["mmq32_ctl"],
                        "wmma32_vdr_vs_ctl": medians["wmma32_vdr"] / medians["wmma32_ctl"],
                        "t16_vs_mmq32_vdr_mismatch": int(
                            sum(
                                np.count_nonzero(a != b)
                                for a, b in zip(
                                    t16_outs, mmq_outs["mmq32_vdr"]
                                )
                            )
                        ),
                        "t16_vs_wmma32_vdr_mismatch": int(
                            sum(
                                np.count_nonzero(a != b)
                                for a, b in zip(
                                    t16_outs, mmq_outs["wmma32_vdr"]
                                )
                            )
                        ),
                        "kl_t16_vs_mmq32_vdr": sum(
                            _kl(a, b)
                            for a, b in zip(t16_outs, mmq_outs["mmq32_vdr"])
                        ),
                        "kl_t16_vs_wmma32_vdr": sum(
                            _kl(a, b)
                            for a, b in zip(t16_outs, mmq_outs["wmma32_vdr"])
                        ),
                        "t16_dual_silu_ms": (
                            medians["t16_dual_silu"] if pair else None
                        ),
                        "finite": finite,
                    }
                )
                print(
                    f"[{index + 1}/{len(cases)}] {role} rows={rows}: "
                    f"t16 {medians['t16_prod']:.3f} ms"
                    + (
                        f" (dual_silu {medians['t16_dual_silu']:.3f})"
                        if pair
                        else ""
                    )
                    + f" | pack {medians['pack']:.3f} | "
                    f"mmq32 ctl/vdr {medians['mmq32_ctl']:.3f}/{medians['mmq32_vdr']:.3f} | "
                    f"wmma32 ctl/vdr {medians['wmma32_ctl']:.3f}/{medians['wmma32_vdr']:.3f} | "
                    f"best mmq total {min(totals.values()):.3f} "
                    f"({medians['t16_prod'] / min(totals.values()):.3f}x t16)",
                    flush=True,
                )
        finally:
            for buffer in buffers:
                free(buffer, runtime=runtime)

    payload = {
        "schema": 1,
        "kind": "qwen38_q4km_dense_q4_k_int_mmq_prefill_leaf",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": detect_device_name(),
        "model": args.model,
        "arms": {
            "t16_prod": (
                "gguf_q4_k_t16_wmma_prefill (dense 256-row parent) per matrix — "
                "the retained float projection owner; the pair case runs it "
                "twice (gate + up)"
            ),
            "t16_dual_silu": (
                "gguf_q4_k_t16_dense_dual_wmma_prefill_silu (fused dual+SiLU "
                "256-row parent, SwiGLU product output) — the production "
                "gate/up dispatch; informational timing for the pair case"
            ),
            "pack": (
                "gguf_q8_1_mmq_ds4_pack_bf16 (BF16 activations to llama.cpp "
                "DS4 block_q8_1_mmq, one residual pass)"
            ),
            "mmq32_ctl": (
                "raw-Q4_K staged-dp4a 32x32-tile MMQ, unamortized metadata"
            ),
            "mmq32_vdr": (
                "mmq32 with efa4e8641-family load reuse: block header and "
                "subblock scale/min products hoisted once per 256-K block"
            ),
            "wmma32_ctl": (
                "raw-Q4_K direct-global iu8-WMMA 32x16-tile MMQ, unamortized"
            ),
            "wmma32_vdr": (
                "wmma32 with the block header read once per 256-K block"
            ),
        },
        "donor_source": (
            "nasone32/llama.cpp-RDNA3-7900xtx-opt "
            "7dc2f0cb28326816f67f6b979008383344e2038b bulk-prefill arithmetic "
            "class (raw-Q4_K x Q8_1 integer MMQ, efa4e8641 load-reuse family)"
        ),
        "prior_boundary": (
            "2026-08-12 T16-payload integer MMQ rejections "
            "(Q4 2.1x slower; Q5 +3.0-4.4%) bar the T16-payload dataflow; this "
            "screen consumes raw Q4_K bytes instead"
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
