#!/usr/bin/env python3
"""Event-time IQ2_XS grouped-prefill schedules on synthetic Laguna shapes."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill import (
    build_gguf_iq_selected_prefill,
    gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out,
    gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out,
)
from tests.test_gguf_iq2_xs_selected_prefill import _weights
from tests.test_gguf_iq_gemv import _f32_to_bf16_u16, _make_x
from tests.test_gguf_iq_selected_prefill import _compact_meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--in-features", type=int, default=3072)
    parser.add_argument("--out-features", type=int, default=128)
    parser.add_argument("--rows-per-expert", default="1,2,3,4,8,16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def _copy(array: np.ndarray):
    array = np.ascontiguousarray(array)
    buffer = malloc(array.nbytes)
    copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
    return buffer


def _time_launch(runtime, launch, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        launch()
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(iters):
            launch()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        return runtime.event_elapsed_time_ms(start, stop) / iters
    finally:
        runtime.event_destroy(start)
        runtime.event_destroy(stop)


def main() -> None:
    args = _parse_args()
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    rows_values = tuple(int(value) for value in args.rows_per_expert.split(","))
    if (
        args.experts <= 0
        or args.in_features <= 0
        or args.in_features % 256
        or args.out_features <= 0
        or any(value <= 0 for value in rows_values)
        or args.warmup < 0
        or args.iters <= 0
    ):
        raise SystemExit("invalid positive shape/timing arguments")

    compiler_version = (
        args.compiler_version_file.read_text()
        if args.compiler_version_file is not None
        else None
    )
    library = build_gguf_iq_selected_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    runtime = get_hip_runtime()
    gate, up = _weights(args.experts, args.out_features, args.in_features)
    gate_buf = _copy(gate)
    up_buf = _copy(up)
    rows = []
    try:
        for rows_per_expert in rows_values:
            meta = _compact_meta([rows_per_expert] * args.experts)
            x = _f32_to_bf16_u16(_make_x(meta.compact_rows, args.in_features))
            out = np.zeros(
                (meta.compact_rows, 2 * args.out_features), dtype=np.uint16
            )
            x_buf = _copy(x)
            start_buf = _copy(meta.expert_start_compact)
            out_buf = malloc(out.nbytes)
            try:
                common = dict(
                    compact_rows=meta.compact_rows,
                    in_features=args.in_features,
                    out_features=args.out_features,
                    num_experts=args.experts,
                    library=library,
                    runtime=runtime,
                )

                def launch_base() -> None:
                    gguf_iq2_xs_selected_dual_grouped_prefill_compact_bf16_bf16_out(
                        x_buf.ptr,
                        start_buf.ptr,
                        gate_buf.ptr,
                        up_buf.ptr,
                        out_buf.ptr,
                        **common,
                    )

                def launch_rowbatch4() -> None:
                    gguf_iq2_xs_selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out(
                        x_buf.ptr,
                        start_buf.ptr,
                        gate_buf.ptr,
                        up_buf.ptr,
                        out_buf.ptr,
                        **common,
                    )

                base_ms = _time_launch(
                    runtime, launch_base, warmup=args.warmup, iters=args.iters
                )
                rowbatch4_ms = _time_launch(
                    runtime, launch_rowbatch4, warmup=args.warmup, iters=args.iters
                )
                rows.append(
                    {
                        "rows_per_expert": rows_per_expert,
                        "compact_rows": meta.compact_rows,
                        "base_ms": base_ms,
                        "rowbatch4_ms": rowbatch4_ms,
                        "rowbatch4_vs_base_percent": 100.0
                        * (rowbatch4_ms / base_ms - 1.0),
                    }
                )
            finally:
                free(out_buf)
                free(start_buf)
                free(x_buf)
    finally:
        free(up_buf)
        free(gate_buf)

    payload = {
        "schema": "hipengine.iq2_xs_laguna_microbench.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shape": {
            "experts": args.experts,
            "in_features": args.in_features,
            "out_features": args.out_features,
        },
        "warmup": args.warmup,
        "iterations": args.iters,
        "rows": rows,
    }
    print(json.dumps(payload, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
