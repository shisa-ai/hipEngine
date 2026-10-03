"""Discriminator: does t16 fused-stride mmq32 match raw fused-stride mmq32?

The production regression: gemma gate_up converts to t16 tiles, then
gemma4_project_experts_mmq_dual wires the fused stack (qweight_b = base +
half_bytes, expert_stride_rows = fused_width) into the t16 mmq32 symbol --
a combination test_q4_k_q8_1... models only for the raw layout
("fused_stride requires the raw single-pass layout").

Cases per shape: raw split, raw fused, t16 split, t16 fused (repack of the
concatenated buffer). Any t16-fused != raw-fused reproduces the defect
without the full model.
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np

spec = importlib.util.spec_from_file_location(
    "pt", "tests/test_gpu_gguf_q4_k_q8_1_selected_prefill.py"
)
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)


def run(fixture, *, layout: str, fused: bool) -> np.ndarray:
    """The test runner with the fused+t16 restriction lifted."""
    runtime = get_hip_runtime()
    library = pt.build_gguf_q4_k_q8_1_selected_prefill(load=True)
    host_out = np.zeros(
        (fixture.compact_rows, fixture.out_features_a + fixture.out_features_b),
        dtype=np.uint16,
    )
    compact_to_source = np.arange(fixture.compact_rows, dtype=np.int64)
    q8_ds4 = pt.pack_q8_1_mmq_ds4_from_bf16(fixture.x_host)
    expert_start_mmq32, tile_expert, mmq_total_rows = pt._mmq32_metadata(fixture)

    fused_up_offset = 0
    if layout == "raw":
        qweight_a = fixture.qweight_a
        qweight_b = fixture.qweight_b
        launcher = pt.gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out
    elif layout == "t16":
        qweight_a = pt.repack_gguf_q4_k_tile16(fixture.qweight_a).tiles
        qweight_b = pt.repack_gguf_q4_k_tile16(fixture.qweight_b).tiles
        launcher = (
            pt.gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out
        )
    else:
        raise ValueError(layout)
    fused_stride = 0
    if fused:
        if layout == "raw":
            row_bytes = int(qweight_a.shape[-1])
            fused_up_offset = fixture.out_features_a * row_bytes
        else:
            bpr = int(qweight_a.shape[2])
            tb = int(qweight_a.shape[3])
            fused_up_offset = (
                fixture.out_features_a // pt.GGUF_Q4_K_TILE16_COLS
            ) * bpr * tb
        fused_stride = fixture.out_features_a + fixture.out_features_b
        qweight_a = np.ascontiguousarray(np.concatenate([qweight_a, qweight_b], axis=1))
        qweight_b = None

    bufs = []
    try:
        source_dev = malloc(fixture.x_host.nbytes, runtime=runtime)
        q8_dev = malloc(q8_ds4.nbytes, runtime=runtime)
        c2s_dev = malloc(compact_to_source.nbytes, runtime=runtime)
        sc_dev = malloc(fixture.expert_start_compact.nbytes, runtime=runtime)
        sm_dev = malloc(expert_start_mmq32.nbytes, runtime=runtime)
        te_dev = malloc(tile_expert.nbytes, runtime=runtime)
        wa_dev = malloc(qweight_a.nbytes, runtime=runtime)
        wb_dev = None if qweight_b is None else malloc(qweight_b.nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend(
            b
            for b in (source_dev, q8_dev, c2s_dev, sc_dev, sm_dev, te_dev, wa_dev, wb_dev, out_dev)
            if b is not None
        )
        for dev, arr in (
            (source_dev, fixture.x_host),
            (q8_dev, q8_ds4),
            (c2s_dev, compact_to_source),
            (sc_dev, fixture.expert_start_compact),
            (sm_dev, expert_start_mmq32),
            (te_dev, tile_expert),
            (wa_dev, qweight_a),
            (wb_dev, qweight_b),
        ):
            if dev is None:
                continue
            copy_host_to_device(dev, host_array_ptr(np.ascontiguousarray(arr)), runtime=runtime)
        launcher(
            q8_dev.ptr,
            c2s_dev.ptr,
            sc_dev.ptr,
            sm_dev.ptr,
            te_dev.ptr,
            wa_dev.ptr,
            (wa_dev.ptr + fused_up_offset if wb_dev is None else wb_dev.ptr),
            out_dev.ptr,
            fixture.compact_rows,
            fixture.in_features,
            fixture.out_features_a,
            fixture.out_features_b,
            fixture.num_experts,
            mmq_total_rows,
            **({"expert_stride_rows": fused_stride} if fused else {}),
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)
    return pt._decode_output(host_out, "bf16")


def main() -> int:
    shapes = [
        dict(counts=[4, 0, 5], in_features=256, out_features_a=32, out_features_b=32),
        dict(
            counts=[64] * 8,
            in_features=4608,
            out_features_a=4608,
            out_features_b=4608,
        ),
    ]
    for kw in shapes:
        fx = pt._build_compact_fixture(dtype="bf16", seed=23, **kw)
        ref = run(fx, layout="raw", fused=False)
        raw_f = run(fx, layout="raw", fused=True)
        t16 = run(fx, layout="t16", fused=False)
        t16_f = run(fx, layout="t16", fused=True)
        def cmp(name, x):
            same = np.array_equal(x, ref)
            md = float(np.abs(x.astype(np.float32) - ref.astype(np.float32)).max())
            print(f"  {name:12s} ==ref {same}  maxabs {md:.4f}")
            return same
        print(f"shape {kw}")
        cmp("raw-fused", raw_f)
        cmp("t16-split", t16)
        cmp("t16-fused", t16_f)
    return 0


if __name__ == "__main__":
    sys.exit(main())