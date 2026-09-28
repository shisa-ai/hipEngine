#!/usr/bin/env python3
"""Microbench the GGUF Q5_1 selected-expert down-projection MMQ prefill kernel.

One kernel, one geometry, timed in isolation. The harness exists because the
down projection is the largest DRAM consumer in a gemma4 prefill step and its
byte accounting could not be reconciled from source reading alone: at a
512-token prompt it fetched 62.33 GB per step, of which the weight stream
explained only part, and the fraction that remained was attributed to activation
re-reads by elimination rather than by measurement.

The fixture is the kernel's real input shape, not a reduced one: gemma4 26B-A4B
stores ``ffn_down_exps`` as Q5_1 ``(128, 2816, 704)``, so ``--experts 128
--out-features 2816 --in-features 704 --rows-per-expert 32`` reproduces
production exactly, including the uniform expert counts that a balanced router
produces at that prompt length. Weight rows are indexed ``(expert * out + row)``
and the activations are the three-plane DS4 layout the route packs.

Timing is the kernel only: the activation pack and the metadata uploads happen
once, outside the measured loop. Run it under a counter to price the kernel's
bytes:

    rocprofv3 --pmc FETCH_SIZE --output-format csv -d <dir> -- \
      .venv/bin/python scripts/gguf_q5_1_down_mmq_microbench.py ...

``Counter_Value`` is kilobytes. Collect one PMC per request on this part: two or
more in one request fail with "Request exceeds the capabilities of the hardware
to collect".

This is a kernel-design harness. It does not load a model and does not validate
model quality; correctness for this kernel lives in
``tests/test_gpu_gguf_q5_1_mmq_selected_prefill.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Run against THIS checkout even when the shared environment's editable install
# points at a different worktree. Observed in this repository: the system
# interpreter resolved ``hipengine`` to another checkout, so a kernel edit here
# was silently not the code under test.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _make_q5_1_weights(
    experts: int, out_features: int, in_features: int, *, seed: int
) -> np.ndarray:
    """Random Q5_1 blocks in the kernel's ``(expert * out + row)`` order."""

    rng = np.random.default_rng(seed)
    blocks = experts * out_features * (in_features // 32)
    inter = np.empty(
        blocks,
        dtype=[("d", "<f2"), ("m", "<f2"), ("qh", "<u4"), ("qs", "u1", (16,))],
    )
    inter["d"] = (rng.standard_normal(blocks) * 0.03).astype(np.float16)
    inter["m"] = (rng.standard_normal(blocks) * 0.01).astype(np.float16)
    inter["qh"] = rng.integers(0, 1 << 32, size=blocks, dtype=np.uint32)
    inter["qs"] = rng.integers(0, 256, size=(blocks, 16), dtype=np.uint8)
    assert inter.dtype.itemsize == 24, inter.dtype.itemsize
    return np.frombuffer(inter.tobytes(), dtype=np.uint8)


def _make_metadata(
    experts: int,
    rows_per_expert: int,
    distribution: str,
    *,
    seed: int,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Expert row counts, uniform or drawn the way a balanced router lands.

    A router that spreads top-k assignments evenly across experts still gives
    each expert a Poisson-distributed share, and that matters here: the kernel's
    row loop advances in ``ROWS_PER_PASS`` strides, so an expert whose count
    crosses a multiple of that pays for another whole pass over its weight rows.
    At a 512-token prompt the mean is 32 and ``ROWS_PER_PASS`` is 32, which puts
    about half the experts over the line -- a uniform fixture hides exactly the
    effect this harness exists to price.
    """

    if distribution == "uniform":
        counts = np.full(experts, rows_per_expert, dtype=np.int64)
    elif distribution == "poisson":
        counts = np.random.default_rng(seed).poisson(rows_per_expert, experts)
        counts = np.maximum(counts, 0).astype(np.int64)
    else:
        raise SystemExit(f"unknown --distribution {distribution!r}")
    expert_start = np.zeros(experts + 1, dtype=np.int64)
    expert_start[1:] = np.cumsum(counts)
    return expert_start, int(expert_start[-1]), counts


# The fp32-scale DS4 block is 32 header bytes plus the 128 int8 quants. The
# down projection's input is a post-SiLU GeGLU output, whose magnitude can
# exceed what an fp16 scale or block sum holds, so this is the layout its route
# packs and the layout the kernel under test reads.
_DS4_F32_BLOCK_BYTES = 32 + 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--out-features", type=int, default=2816)
    parser.add_argument("--in-features", type=int, default=704)
    parser.add_argument("--rows-per-expert", type=int, default=32)
    parser.add_argument(
        "--distribution",
        choices=("uniform", "poisson"),
        default="uniform",
        help="expert row-count distribution; poisson is what a balanced router lands",
    )
    parser.add_argument("--planes", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.in_features % 32:
        raise SystemExit("--in-features must be a multiple of 32")
    if args.out_features % 128:
        raise SystemExit("--out-features must be a multiple of 128")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
        build_gguf_q4_k_q8_1_selected_prefill,
        gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3 as pack_activations,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q5_1_mmq_selected_prefill import (
        build_gguf_q5_1_mmq_selected_prefill,
        gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out as down_mmq,
    )

    expert_start, compact_rows, counts = _make_metadata(
        args.experts, args.rows_per_expert, args.distribution, seed=args.seed
    )
    raw_weights = _make_q5_1_weights(
        args.experts,
        args.out_features,
        args.in_features,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed + 1)
    rows_f32 = (rng.standard_normal((compact_rows, args.in_features)) * 0.4).astype(
        np.float32
    )
    rows_bf16 = np.ascontiguousarray(
        _bf16_bits(rows_f32).reshape(compact_rows, args.in_features)
    )
    out = np.empty((compact_rows, args.out_features), dtype=np.uint16)

    weight_bytes = args.experts * args.out_features * (args.in_features // 32) * 24
    ds4_bytes = (
        args.planes
        * compact_rows
        * ((args.in_features + 127) // 128)
        * _DS4_F32_BLOCK_BYTES
    )

    runtime = get_hip_runtime()
    library = build_gguf_q5_1_mmq_selected_prefill(load=True)
    pack_library = build_gguf_q4_k_q8_1_selected_prefill(load=True)

    allocations = []
    try:
        w_dev = malloc(raw_weights.nbytes, runtime=runtime)
        rows_dev = malloc(rows_bf16.nbytes, runtime=runtime)
        ds4_dev = malloc(ds4_bytes, runtime=runtime)
        start_dev = malloc(expert_start.nbytes, runtime=runtime)
        out_dev = malloc(out.nbytes, runtime=runtime)
        allocations += [w_dev, rows_dev, ds4_dev, start_dev, out_dev]
        copy_host_to_device(w_dev, host_array_ptr(raw_weights), runtime=runtime)
        copy_host_to_device(rows_dev, host_array_ptr(rows_bf16), runtime=runtime)
        copy_host_to_device(start_dev, host_array_ptr(expert_start), runtime=runtime)
        pack_activations(
            rows_dev.ptr,
            ds4_dev.ptr,
            compact_rows,
            args.in_features,
            residual_passes=args.planes,
            library=pack_library,
            runtime=runtime,
        )

        def launch() -> None:
            down_mmq(
                ds4_dev.ptr,
                start_dev.ptr,
                w_dev.ptr,
                out_dev.ptr,
                compact_rows,
                args.experts,
                args.in_features,
                args.out_features,
                args.planes,
                f32_scales=True,
                library=library,
                runtime=runtime,
            )

        for _ in range(args.warmup):
            launch()
        runtime.device_synchronize()

        timings = []
        for _ in range(args.iters):
            start = runtime.event_create()
            stop = runtime.event_create()
            runtime.event_record(start)
            launch()
            runtime.event_record(stop)
            runtime.device_synchronize()
            timings.append(runtime.event_elapsed_time_ms(start, stop))
            runtime.event_destroy(start)
            runtime.event_destroy(stop)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    timings.sort()
    median = timings[len(timings) // 2]
    mean = sum(timings) / len(timings)
    macs = compact_rows * args.in_features * args.out_features
    passes = int(np.ceil(counts / 32.0).sum())
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experts": args.experts,
        "out_features": args.out_features,
        "in_features": args.in_features,
        "rows_per_expert": args.rows_per_expert,
        "distribution": args.distribution,
        "planes": args.planes,
        "compact_rows": compact_rows,
        "counts_min": int(counts.min()),
        "counts_max": int(counts.max()),
        "row_passes_at_32": passes,
        "row_passes_ratio": passes / max(args.experts, 1),
        "weight_bytes": weight_bytes,
        "weight_bytes_x_passes": weight_bytes * passes,
        "ds4_activation_bytes": ds4_bytes,
        "ms_per_call_min": timings[0],
        "ms_per_call_median": median,
        "ms_per_call_mean": mean,
        "ms_per_call_max": timings[-1],
        "gmac_per_s": macs / median / 1e6,
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
