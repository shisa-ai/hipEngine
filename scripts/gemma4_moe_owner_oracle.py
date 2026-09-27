"""Kernel-level oracle for the grouped MoE gate_up owners.

Wraps the production grouped-dual owner to capture its real arguments on a live
forward, then replays those exact arguments through the known-good fp32 owner and
through the Q4_K ds4 MMQ owner, comparing outputs. Varies one argument at a time
so a single run localizes which wiring is wrong instead of costing a 10-minute
end-to-end gate cycle per hypothesis.

Usage: PYTHONPATH=. .venv/bin/python /tmp/gemma4_moe_oracle.py [artifact]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import hipengine
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts
from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
    qwen35_moe_mmq32_tile_map,
    qwen35_moe_wmma_tile_map,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    build_gguf_q4_k_q8_1_selected_prefill,
    gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
    gguf_q4_k_x8_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
    gguf_q8_1_mmq_ds4_pack_bf16,
    gguf_q8_1_mmq_ds4_pack_bf16_d4x3,
)

ARTIFACT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)

captured: list[tuple[tuple, dict]] = []
_orig = gemma4_experts.gemma4_project_experts_grouped_dual


def _spy(*args, **kwargs):
    captured.append((args, kwargs))
    return _orig(*args, **kwargs)


def read(host: np.ndarray, buffer, nbytes: int, runtime) -> None:
    copy_device_to_host(host_array_ptr(host), buffer, nbytes, runtime=runtime)


def main() -> int:
    runtime = get_hip_runtime()
    llm = hipengine.LLM(model=ARTIFACT)
    generator = llm._get_text_generator()
    generator.context_length = 8192
    runner = generator._ensure_runner()

    gemma4_experts.gemma4_project_experts_grouped_dual = _spy
    try:
        runner.reset()
        runner.forward([9707] * 64)
        captured.clear()
        runner.reset()
        runner.forward([9707] * 2048)
        runtime.device_synchronize()
    finally:
        gemma4_experts.gemma4_project_experts_grouped_dual = _orig

    print(f"captured {len(captured)} grouped-dual calls")
    if not captured:
        print("FAIL: no grouped-dual call reached the MoE gate_up")
        return 1

    # Prefer a call with the most rows: the tile plan only exists above the
    # WMMA lane threshold, and a tiny call would not exercise it.
    (args, kwargs) = max(captured, key=lambda c: int(c[0][4]))
    (
        weight,
        x_ptr,
        expert_start_ptr,
        out_ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        fused_width,
    ) = args[:9]
    stream = kwargs.get("stream", 0)
    print(
        f"rows={compact_rows} experts={num_experts} in={in_features} "
        f"half={out_features} fused={fused_width}"
    )

    width = compact_rows * fused_width
    out_ref = malloc(width * 2, runtime=runtime)
    out_mmq = malloc(width * 2, runtime=runtime)
    workspace = malloc(compact_rows * (in_features // 128) * 144, runtime=runtime)
    identity = malloc((compact_rows + 32 * num_experts) * 8, runtime=runtime)
    plan_expert_start = malloc((num_experts + 1) * 8, runtime=runtime)
    plan_tile_expert = malloc(4096 * 8, runtime=runtime)
    plan_total = malloc(8, runtime=runtime)

    # Reference: the production fp32 grouped-dual owner, same arguments.
    _orig(
        weight,
        x_ptr,
        expert_start_ptr,
        out_ref.ptr,
        compact_rows,
        num_experts,
        in_features,
        out_features,
        fused_width,
        stream=stream,
        runtime=runtime,
    )

    library = build_gguf_q4_k_q8_1_selected_prefill(load=True)

    base_ptr = weight.allocation("raw").buffer.ptr
    half_bytes = weight.expert_stride_bytes // 2

    def pack(pack_fn) -> None:
        pack_fn(
            x_ptr,
            workspace.ptr,
            compact_rows,
            in_features,
            library=library,
            runtime=runtime,
        )

    def build_plan(builder, tile_capacity: int) -> int:
        builder(
            expert_start_ptr,
            plan_expert_start.ptr,
            plan_tile_expert.ptr,
            plan_total.ptr,
            num_experts,
            tile_capacity=tile_capacity,
            stream=stream,
            runtime=runtime,
        )
        host = np.empty(1, dtype=np.int64)
        read(host, plan_total, 8, runtime)
        return int(host[0])

    def fill_identity(n: int) -> None:
        arr = np.arange(n, dtype=np.int64)
        copy_host_to_device(identity, host_array_ptr(arr), arr.nbytes, runtime=runtime)

    def run_mmq(
        fn,
        total_rows: int,
        es_compact: int,
        es_plan: int,
        tile_ptr: int,
        weight_ptrs: tuple[int, int] | None = None,
    ) -> np.ndarray:
        a_ptr, b_ptr = weight_ptrs or (base_ptr, base_ptr + half_bytes)
        fn(
            workspace.ptr,
            identity.ptr,
            es_compact,
            es_plan,
            tile_ptr,
            a_ptr,
            b_ptr,
            out_mmq.ptr,
            compact_rows,
            in_features,
            out_features,
            out_features,
            num_experts,
            total_rows,
            stream=stream,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        buf = np.empty(width, dtype=np.uint16)
        read(buf, out_mmq, width * 2, runtime)
        return buf.view(np.float16).astype(np.float32)

    ref_buf = np.empty(width, dtype=np.uint16)
    read(ref_buf, out_ref, width * 2, runtime)
    ref = ref_buf.view(np.float16).astype(np.float32)
    print(f"reference finite={np.isfinite(ref).all()} absmax={np.abs(ref).max():.4f}")

    # The plan only exists above the WMMA lane threshold; below it the compact
    # starts are the only array available.
    tile_capacity = 4096
    H = out_features
    total = build_plan(qwen35_moe_mmq32_tile_map, tile_capacity)
    print(f"mmq32-plan total_rows={total}")
    first_tile = np.empty(1, dtype=np.int64)
    read(first_tile, plan_tile_expert, 8, runtime)
    print(f"tile_expert[0]={int(first_tile[0])}  (row 0 belongs to this expert)")
    fill_identity(compact_rows + 32 * num_experts)
    pack(gguf_q8_1_mmq_ds4_pack_bf16)

    # The owner computes expert_bytes = local_out_features * weight_row_bytes, so
    # qweight_a and qweight_b are independently addressed with an expert stride
    # of one *half's* rows. Gemma's fused tensor has an expert stride of the full
    # fused width, so build the two expert-major half tensors the owner expects
    # and check whether the output matches the reference.
    from hipengine.core.hip import HipMemcpyKind

    blocks_per_row = in_features // 256
    weight_row_bytes = blocks_per_row * 144
    half_bytes = out_features * weight_row_bytes
    expert_bytes = 2 * half_bytes
    half_a = malloc(half_bytes * num_experts, runtime=runtime)
    half_b = malloc(half_bytes * num_experts, runtime=runtime)
    for e in range(num_experts):
        base = base_ptr + e * expert_bytes
        runtime.memcpy(
            half_a.ptr + e * half_bytes, base, half_bytes, HipMemcpyKind.DEVICE_TO_DEVICE
        )
        runtime.memcpy(
            half_b.ptr + e * half_bytes,
            base + half_bytes,
            half_bytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
        )
    runtime.device_synchronize()
    print(f"split tensors: half={half_bytes} expert={expert_bytes} bytes")

    for label, a_ptr, b_ptr in (
        ("fused ", base_ptr, base_ptr + half_bytes),
        ("split ", half_a.ptr, half_b.ptr),
    ):
        got = run_mmq(
            gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
            total,
            expert_start_ptr,
            plan_expert_start.ptr,
            plan_tile_expert.ptr,
            weight_ptrs=(a_ptr, b_ptr),
        )
        tag = f"weights={label}"
        if not np.isfinite(got).all():
            print(f"  {tag} NON-FINITE")
            continue
        g = got.reshape(compact_rows, 2 * H)
        r = ref.reshape(compact_rows, 2 * H)
        gate = np.abs(g[:, :H] - r[:, :H]).max()
        up = np.abs(g[:, H:] - r[:, H:]).max()
        rowerr = np.abs(g - r).max(axis=1)
        good = float((rowerr < 1e-2).mean())
        print(
            f"  {tag} direct={np.abs(g - r).max():.5f} gate={gate:.5f} "
            f"up={up:.5f} rows_ok={good:.1%}"
        )
        # The grid is (out_features_total/32, row_tiles): output columns are
        # 32-wide tiles. If the weight rows are consumed in tile order rather
        # than row order, the output columns are a tile-major permutation of the
        # reference. Test that on one row per half before assuming anything.
        tiles = H // 32
        for half, sl in (("gate", slice(0, H)), ("up", slice(H, 2 * H))):
            a = g[0, sl]
            b = r[0, sl]
            variants = {
                "row-major": b,
                "tile-major": b.reshape(tiles, 32).T.reshape(-1),
                "tile-rev": b.reshape(tiles, 32)[::-1].reshape(-1),
            }
            best = min(
                ((np.abs(a - v).max(), k) for k, v in variants.items()), key=lambda t: t[0]
            )
            print(f"    row0 {half}: best={best[1]} err={best[0]:.4f}")
            # Which 32-wide column tiles are wrong? A residual that is confined
            # to particular tiles localizes the remaining addressing difference.
            per_tile = np.abs(a - b).reshape(tiles, 32).max(axis=1)
            bad = [i for i, e in enumerate(per_tile) if e > 1e-2]
            print(
                f"      tiles_bad={len(bad)}/{tiles} first={bad[:6]} "
                f"worst={per_tile.max():.4f} median={np.median(per_tile):.4f}"
            )

    for label, builder in (
        ("mmq32-plan", qwen35_moe_mmq32_tile_map),
        ("wmma-plan", qwen35_moe_wmma_tile_map),
    ):
        try:
            total = build_plan(builder, tile_capacity)
        except Exception as exc:  # noqa: BLE001
            print(f"{label}: build failed: {exc}")
            continue
        print(f"\n--- {label}: total_rows={total}")
        for id_label, id_n in (("arange(rows)", compact_rows), ("arange(padded)", compact_rows + 32 * num_experts)):
            fill_identity(id_n)
            for es_label, es_plan in (
                ("plan", plan_expert_start.ptr),
                ("compact", expert_start_ptr),
            ):
                got = run_mmq(
                    gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out,
                    total,
                    expert_start_ptr,
                    es_plan,
                    plan_tile_expert.ptr,
                )
                if not np.isfinite(got).all():
                    print(f"  {id_label:16s} es_mmq32={es_label:8s} NON-FINITE")
                    continue
                # Characterize the error rather than only its magnitude: a half
                # swap, a uniform scale, or a row-index mismatch each look
                # different here, and they point at different causes.
                g = got.reshape(compact_rows, 2 * H)
                r = ref.reshape(compact_rows, 2 * H)
                same = np.abs(g - r).max()
                swap = np.abs(np.concatenate([g[:, H:], g[:, :H]], axis=1) - r).max()
                k = float((g.ravel() @ r.ravel()) / max(g.ravel() @ g.ravel(), 1e-9))
                rowerr = np.abs(g - r).max(axis=1)
                good = float((rowerr < 1e-2).mean())
                print(
                    f"  {id_label:16s} es_mmq32={es_label:8s} direct={same:.4f} "
                    f"swapped={swap:.4f} scale_k={k:.3f} rows_ok={good:.1%}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
