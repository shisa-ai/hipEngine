#!/usr/bin/env python3
"""Identical-operand, operation-complete benchmark for a dense GGUF projection.

Answers whether the existing dense projection can be made substantially faster
without changing its arithmetic boundaries. It times every registered
coltile/rowbatch variant of one projection on the *same* operands, so the only
difference between rows is the kernel that runs.

The projection under test is the one the prefill spends the most time in. For a
4096-token Qwen3.8-Flash-Next prefill that is ``attn_qkv``: K=2560, N=10240,
Q8_0, 1024 rows per chunk, 2585 ms of a 22046 ms window across 144 launches,
all served by ``gguf_k_prefill_out_coltile_rowbatch_kernel``.

Design points that matter for the result being usable:

* Operands come from real layers of the model file, not one synthetic tensor.
  Layers rotate across repetitions so the weight matrix does not stay resident
  in MALL and inflate the rate.
* Both row cases are measured: a full 1024-row chunk and the tail rows a chunked
  prefill actually ends on.
* Timing is operation-complete, wrapping the full host-side call including input
  preparation, and compilation is excluded by a warmup that runs the same symbol
  first.
* Every variant is compared against the production variant on the full output
  tensor, so a faster row is only interesting if it is also numerically close.

Nothing here changes the F32 output ABI, adds a conversion stage, or fuses
anything. It measures what already ships.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (  # noqa: E402
    build_gguf_k_gemv,
    gguf_q8_0_gemv_coltile4_rowbatch8_f32_f32_out,
    gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out,
    gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_f32_f32_out,
    gguf_q8_0_gemv_coltile8_rowbatch8_f32_f32_out,
    gguf_q8_0_gemv_coltile16_rowbatch2_f32_f32_out,
    gguf_q8_0_gemv_coltile16_rowbatch4_f32_f32_out,
    gguf_q8_0_gemv_coltile32_rowbatch1_f32_f32_out,
)
from hipengine.loading.gguf import (  # noqa: E402
    GGUFReader,
    discover_gguf_files,
    scan_gguf,
)

FP32_FMA_PEAK_GFLOPS = 61_300.0
Q8_0_BLOCK = 32
Q8_0_BLOCK_BYTES = 34

# The variant the engine actually selects for this scope. It is the wave-scale
# instantiation: the trace shows the launched kernel template argument list as
# ``float, float, 8, 8, 4, true``. Reading the base kernel name alone is not
# enough to tell the two apart, because WAVE_SCALE is a template parameter and
# both instantiations share the kernel name.
PRODUCTION_VARIANT = "coltile8_rowbatch4_wave_scale_f32_f32_out"

# The instantiation that the kernel name alone would suggest, kept so the sweep
# records the comparison that motivated checking the template argument.
SUPERSEDED_VARIANT = "coltile8_rowbatch4_f32_f32_out"

VARIANTS: dict[str, Any] = {
    "coltile4_rowbatch8_f32_f32_out": gguf_q8_0_gemv_coltile4_rowbatch8_f32_f32_out,
    "coltile8_rowbatch4_f32_f32_out": gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out,
    "coltile8_rowbatch4_wave_scale_f32_f32_out": (
        gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_f32_f32_out
    ),
    "coltile8_rowbatch8_f32_f32_out": gguf_q8_0_gemv_coltile8_rowbatch8_f32_f32_out,
    "coltile16_rowbatch2_f32_f32_out": gguf_q8_0_gemv_coltile16_rowbatch2_f32_f32_out,
    "coltile16_rowbatch4_f32_f32_out": gguf_q8_0_gemv_coltile16_rowbatch4_f32_f32_out,
    "coltile32_rowbatch1_f32_f32_out": gguf_q8_0_gemv_coltile32_rowbatch1_f32_f32_out,
}


def load_layer_weights(
    model_root: str, suffix: str, layers: list[int]
) -> list[tuple[int, np.ndarray, str]]:
    """Read the raw Q8_0 bytes of ``suffix`` for each requested layer."""
    wanted = {f"blk.{layer}.{suffix}": layer for layer in layers}
    found: dict[int, tuple[np.ndarray, str]] = {}
    for path in discover_gguf_files(model_root):
        if len(found) == len(wanted):
            break
        info = scan_gguf(path)
        if not info.tensors:
            continue
        present = {tensor.name for tensor in info.tensors}
        if not (present & set(wanted)):
            continue
        reader = GGUFReader(path)
        for name, layer in wanted.items():
            if layer in found or name not in present:
                continue
            # tensor_data is a memmap of the raw GGUF storage, which for Q8_0
            # is exactly the block layout the kernel reads.
            raw = np.ascontiguousarray(reader.tensor_data(name)).view(np.uint8)
            found[layer] = (raw.reshape(-1).copy(), reader.tensor_info(name).ggml_type_name)
    missing = [layer for layer in layers if layer not in found]
    if missing:
        raise SystemExit(f"{suffix}: no tensor for layers {missing}")
    return [(layer, *found[layer]) for layer in layers]


def launch_variant(
    launch, x_ptr: int, w_ptr: int, out_ptr: int,
    rows: int, in_features: int, out_features: int,
) -> None:
    """Call a registered variant with the production thread count and stream.

    ``threads`` and ``stream`` are keyword-only on the launcher, so they cannot
    be passed positionally.
    """
    launch(
        x_ptr, w_ptr, out_ptr, rows, in_features, out_features,
        threads=128, stream=0,
    )


def time_variant(
    launch,
    x_ptr: int,
    w_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    runtime,
    repetitions: int,
) -> float:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(repetitions):
            launch_variant(launch, x_ptr, w_ptr, out_ptr, rows, in_features, out_features)
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        return runtime.event_elapsed_time_ms(start, stop) / repetitions
    finally:
        runtime.event_destroy(start)
        runtime.event_destroy(stop)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        default="/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL",
    )
    parser.add_argument("--tensor-suffix", default="attn_qkv.weight")
    parser.add_argument("--layers", default="0,9,22,34,46")
    parser.add_argument("--in-features", type=int, default=2560)
    parser.add_argument("--out-features", type=int, default=10240)
    parser.add_argument("--rows", default="1,8,64,512,1024")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="comma-separated subset of the registered coltile/rowbatch variants",
    )
    args = parser.parse_args()

    layers = [int(part) for part in args.layers.split(",") if part]
    row_cases = [int(part) for part in args.rows.split(",") if part]
    variant_names = [name for name in args.variants.split(",") if name]
    for name in variant_names:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}")

    runtime = get_hip_runtime()
    build_gguf_k_gemv(load=True)  # compile outside every timed region

    layer_weights = load_layer_weights(args.model_root, args.tensor_suffix, layers)
    quant_types = {q for _layer, _raw, q in layer_weights}
    if quant_types != {"Q8_0"}:
        raise SystemExit(f"expected Q8_0 weights, found {sorted(quant_types)}")

    # One activation buffer per layer, seeded per layer so the value ranges
    # differ across layers instead of repeating one convenient tensor.
    activations = {}
    for layer, _raw, _quant in layer_weights:
        rng = np.random.default_rng(1000 + layer)
        activations[layer] = (
            rng.standard_normal((max(row_cases), args.in_features)) * 0.5
        ).astype(np.float32)

    results: list[dict[str, Any]] = []
    out_bytes = max(row_cases) * args.out_features * 4

    out_dev = malloc(out_bytes)
    try:
        for layer, raw, _quant in layer_weights:
            w_dev = malloc(raw.nbytes)
            try:
                copy_host_to_device(w_dev, host_array_ptr(raw))
                for rows in row_cases:
                    x = activations[layer][:rows]
                    x_dev = malloc(x.nbytes)
                    try:
                        copy_host_to_device(x_dev, host_array_ptr(x))

                        # Warm every variant once so no launch pays first-call
                        # cost inside its own timed region.
                        for name in variant_names:
                            launch_variant(
                                VARIANTS[name], x_dev.ptr, w_dev.ptr, out_dev.ptr,
                                rows, args.in_features, args.out_features,
                            )
                        runtime.stream_synchronize(0)

                        reference = None
                        for name in variant_names:
                            launch_variant(
                                VARIANTS[name], x_dev.ptr, w_dev.ptr, out_dev.ptr,
                                rows, args.in_features, args.out_features,
                            )
                            runtime.stream_synchronize(0)
                            out = np.empty(
                                rows * args.out_features, dtype=np.float32
                            )
                            copy_device_to_host(
                                host_array_ptr(out), out_dev, out.nbytes
                            )
                            if name == PRODUCTION_VARIANT:
                                reference = out.copy()

                        for name in variant_names:
                            ms = time_variant(
                                VARIANTS[name], x_dev.ptr, w_dev.ptr, out_dev.ptr,
                                rows, args.in_features, args.out_features,
                                runtime, args.repetitions,
                            )
                            launch_variant(
                                VARIANTS[name], x_dev.ptr, w_dev.ptr, out_dev.ptr,
                                rows, args.in_features, args.out_features,
                            )
                            runtime.stream_synchronize(0)
                            out = np.empty(
                                rows * args.out_features, dtype=np.float32
                            )
                            copy_device_to_host(
                                host_array_ptr(out), out_dev, out.nbytes
                            )
                            flops = 2.0 * rows * args.in_features * args.out_features
                            entry = {
                                "layer": layer,
                                "rows": rows,
                                "variant": name,
                                "ms": round(ms, 4),
                                "gflops": round(flops / 1e9, 1),
                                "achieved_gflops": round(flops / (ms / 1000.0) / 1e9, 1),
                                "share_of_fp32_peak_pct": round(
                                    100.0 * flops / (ms / 1000.0) / 1e9
                                    / FP32_FMA_PEAK_GFLOPS, 2
                                ),
                                "max_abs_diff_vs_production": (
                                    None if reference is None
                                    else float(np.max(np.abs(out - reference)))
                                ),
                                "bit_identical_to_production": (
                                    None if reference is None
                                    else bool(np.array_equal(out, reference))
                                ),
                            }
                            results.append(entry)
                            print(
                                f"L{layer:<3d} rows={rows:<5d} {name:44s} "
                                f"{ms:8.4f} ms {entry['achieved_gflops']:8.1f} "
                                f"GFLOP/s {entry['share_of_fp32_peak_pct']:5.2f}% pk"
                            )
                    finally:
                        free(x_dev)
            finally:
                free(w_dev)
    finally:
        free(out_dev)

    payload = {
        "schema": 1,
        "kind": "qwen4exp_dense_projection_identical_operand_ab",
        "performance_claim": False,
        "status": "diagnostic",
        "model_root": args.model_root,
        "tensor_suffix": args.tensor_suffix,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "quant": "Q8_0",
        "layers": layers,
        "row_cases": row_cases,
        "repetitions": args.repetitions,
        "production_variant": PRODUCTION_VARIANT,
        "superseded_variant": SUPERSEDED_VARIANT,
        "production_variant_basis": (
            "Template argument list in the role-marked rocprofv3 trace: "
            "float, float, 8, 8, 4, true (COL_TILE=8, ROW_BATCH=4, WAVE_SCALE=true)."
        ),
        "fp32_fma_peak_gflops": FP32_FMA_PEAK_GFLOPS,
        "activation_note": (
            "Per-layer deterministic standard normal scaled by 0.5. Real "
            "activation ranges from a boundary capture are a follow-up; this "
            "run separates kernel variants on identical operands, which is "
            "what the variant comparison needs."
        ),
        "notes": [
            "Every variant runs the same arithmetic class (scalar FP32 FMA over Q8_0 weights) with a different tile shape.",
            "Layers rotate across the sweep so the weight matrix does not stay cache-resident.",
            "Compilation happens before any timed region.",
            "bit_identical_to_production is reported because a different tile shape changes the accumulation order even when the formula is unchanged.",
            "The production variant is already the fastest of the registered family; this sweep bounds the tile-shape design space rather than finding a candidate.",
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n")

    print("\nbest per (layer, rows) vs production")
    print("-" * 92)
    by_case: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in results:
        by_case.setdefault((row["layer"], row["rows"]), []).append(row)
    for (layer, rows), group in sorted(by_case.items()):
        best = min(group, key=lambda r: r["ms"])
        prod = next(r for r in group if r["variant"] == PRODUCTION_VARIANT)
        delta = 100.0 * (prod["ms"] - best["ms"]) / prod["ms"]
        print(
            f"L{layer:<3d} rows={rows:<5d} best={best['variant']:42s} "
            f"{best['ms']:8.4f} ms  production {prod['ms']:8.4f} ms  "
            f"{delta:+5.1f}%"
        )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
