#!/usr/bin/env python3
"""Measure current Q4-T16 rowtile owners against small-M on production weights.

This harness deliberately obtains each tiles pointer from the normal Qwen GGUF
materializer.  Do not replace that path with a reconstructed tile payload: a
prior guessed payload wedged the GPU.  It inventories the requested FFN/QKV
roles too, and times only roles whose shipping resident layout is the exact
``gguf_q4_k_t16_v1`` ABI shared by rowtile, small-M, and strict shared-B.

Each cell first requires bit-exact BF16 output across incumbent, small-M, and
strict shared-B.  Timings use counterbalanced arm order, HIP events, and
operation-complete event synchronization.  Ratios are incumbent/small-M, so a
value above one means small-M is faster.

Example:
  timeout 300 .venv/bin/python scripts/gguf_t16_sidecar_row_bench.py \
    --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
    --output /tmp/he-t16/production-sidecar-rows2-8.json
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_ROWS = (2, 3, 4, 5, 6, 7, 8)
# This inventory spans the disputed FFN projection, recurrent QKV, and the
# standard-Q4 full/recurrent attention roles that are actually candidate-safe.
ROLE_SPECS = (
    (0, "ffn_gate"),
    (0, "ffn_down"),
    (0, "attn_gate"),
    (0, "attn_qkv"),
    (3, "attn_q"),
    (3, "attn_k"),
    (3, "attn_v"),
    (3, "attn_output"),
)
ARMS = ("incumbent", "smallm", "strict_shared_b")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _tracked_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--untracked-files=no"))


def _parse_rows(value: str) -> tuple[int, ...]:
    rows = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not rows or any(row < 2 or row > 8 for row in rows):
        raise argparse.ArgumentTypeError("rows must be a non-empty subset of 2..8")
    if len(set(rows)) != len(rows):
        raise argparse.ArgumentTypeError("rows must not contain duplicates")
    return rows


def _counterbalanced_order(
    names: Sequence[str], *, cell_index: int, sample_index: int
) -> tuple[str, ...]:
    """Rotate every arm through every order position across adjacent samples."""

    values = tuple(names)
    if not values:
        return ()
    offset = (int(cell_index) + int(sample_index)) % len(values)
    return values[offset:] + values[:offset]


def _sample_summary(samples: Sequence[float]) -> dict[str, float | int | list[float]]:
    ordered = sorted(float(value) for value in samples)
    if not ordered:
        raise ValueError("cannot summarize zero samples")
    return {
        "samples": len(ordered),
        "median_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "values_ms": ordered,
    }


def _sampled_model_fingerprint(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    sample_bytes = min(1 << 20, size)
    offsets = sorted({0, max(0, size // 2 - sample_bytes // 2), max(0, size - sample_bytes)})
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            digest.update(handle.read(sample_bytes))
    return {
        "algorithm": "sha256-sampled-v1",
        "value": digest.hexdigest(),
        "size_bytes": size,
        "sampled_bytes": sample_bytes * len(offsets),
        "sample_offsets": offsets,
    }


def _bf16_values(rows: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 0.35, size=(rows, width)).astype(np.float32)
    bits = values.view(np.uint32)
    lsb = (bits >> 16) & np.uint32(1)
    return np.ascontiguousarray(((bits + 0x7FFF + lsb) >> 16).astype(np.uint16))


def _read_bf16(runtime: Any, buffer: Any, shape: tuple[int, int]) -> np.ndarray:
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    host = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(host), buffer, host.nbytes, runtime=runtime)
    return host


def _event_sample_ms(
    runtime: Any,
    launch: Callable[[], None],
    *,
    burst: int,
) -> tuple[float, float]:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        wall_started = time.perf_counter_ns()
        for _ in range(int(burst)):
            launch()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        wall_ms = (time.perf_counter_ns() - wall_started) / 1.0e6 / int(burst)
        event_ms = float(runtime.event_elapsed_time_ms(start, stop)) / int(burst)
        return event_ms, wall_ms
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)


def _resolve_functions() -> tuple[dict[str, Callable[..., None]], dict[str, Any]]:
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
        build_gguf_k_t16_selected_prefill,
        gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant import gguf_k_t16_selected_prefill
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        build_gguf_t16_selected_gemv,
        gguf_q4_k_t16_dense_rowtile_bf16_bf16_out,
        gguf_q4_k_t16_dense_rowtile_col4_bf16_bf16_out,
    )

    smallm = getattr(
        gguf_k_t16_selected_prefill,
        "gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out",
        None,
    )
    if not callable(smallm):
        raise RuntimeError("gfx1100 small-M Q4-T16 wrapper is unavailable")
    return (
        {
            "dense_rowtile_bf16_bf16_out": gguf_q4_k_t16_dense_rowtile_bf16_bf16_out,
            "dense_rowtile_col4_bf16_bf16_out": (
                gguf_q4_k_t16_dense_rowtile_col4_bf16_bf16_out
            ),
            "smallm": smallm,
            "strict_shared_b": gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out,
        },
        {
            "rowtile": build_gguf_t16_selected_gemv(load=True),
            "prefill": build_gguf_k_t16_selected_prefill(load=True),
        },
    )


def _incumbent_variant(rows: int, in_features: int, out_features: int) -> str:
    from hipengine.runtime.gguf_linear import _q4_t16_sidecar_decode_variants

    variants = _q4_t16_sidecar_decode_variants(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    )
    if not variants:
        raise RuntimeError("production sidecar selector returned no incumbent")
    return str(variants[0])


def run(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.loading.qwen35_gguf_materialize import (
        LAYOUT_GGUF_Q4_K_T16,
        materialize_qwen35_gguf_weights,
    )

    model = Path(args.model).resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    if _tracked_dirty():
        raise RuntimeError("benchmark requires tracked-clean source")

    runtime = get_hip_runtime()
    functions, libraries = _resolve_functions()
    selected_slots = {f"layers.{layer_id}.{role}" for layer_id, role in ROLE_SPECS}
    weights = materialize_qwen35_gguf_weights(
        model,
        selected_slots=selected_slots,
        runtime=runtime,
        backend="hip_gfx1100",
    )
    cells: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    buffers: list[Any] = []
    try:
        measurable: list[tuple[int, str, Any]] = []
        for layer_id, role in ROLE_SPECS:
            weight = weights.layer(layer_id).weight(role)
            out_features, in_features = map(int, weight.spec.source.shape)
            eligible = weight.spec.layout == LAYOUT_GGUF_Q4_K_T16
            inventory.append(
                {
                    "layer": layer_id,
                    "role": role,
                    "source_name": weight.spec.source.name,
                    "source_quant": weight.spec.source.ggml_type_name,
                    "resident_layout": weight.spec.layout,
                    "in_features": in_features,
                    "out_features": out_features,
                    "smallm_abi_eligible": eligible,
                    "reason": (
                        None
                        if eligible
                        else "shipping resident layout does not match gguf_q4_k_t16_v1"
                    ),
                }
            )
            if eligible:
                measurable.append((layer_id, role, weight))

        cell_index = 0
        for layer_id, role, weight in measurable:
            out_features, in_features = map(int, weight.spec.source.shape)
            tiles = weight.allocation("tiles")
            for rows in args.rows:
                incumbent_variant = _incumbent_variant(rows, in_features, out_features)
                incumbent = functions[incumbent_variant]
                host_x = _bf16_values(
                    rows,
                    in_features,
                    int(args.seed) + cell_index * 131 + rows,
                )
                x_buffer = malloc(host_x.nbytes, runtime=runtime)
                buffers.append(x_buffer)
                copy_host_to_device(
                    x_buffer,
                    host_array_ptr(host_x),
                    host_x.nbytes,
                    runtime=runtime,
                )
                output_buffers = {
                    name: malloc(rows * out_features * 2, runtime=runtime)
                    for name in ARMS
                }
                buffers.extend(output_buffers.values())

                launches: dict[str, Callable[[], None]] = {
                    "incumbent": lambda fn=incumbent, out=output_buffers["incumbent"]: fn(
                        x_buffer.ptr,
                        tiles.tensor.ptr,
                        out.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=libraries["rowtile"],
                        runtime=runtime,
                    ),
                    "smallm": lambda out=output_buffers["smallm"]: functions["smallm"](
                        x_buffer.ptr,
                        tiles.tensor.ptr,
                        out.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=libraries["prefill"],
                        runtime=runtime,
                    ),
                    "strict_shared_b": lambda out=output_buffers["strict_shared_b"]: functions[
                        "strict_shared_b"
                    ](
                        x_buffer.ptr,
                        tiles.tensor.ptr,
                        out.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=libraries["prefill"],
                        runtime=runtime,
                    ),
                }

                outputs: dict[str, np.ndarray] = {}
                for name in ARMS:
                    launches[name]()
                    runtime.device_synchronize()
                    outputs[name] = _read_bf16(
                        runtime,
                        output_buffers[name],
                        (rows, out_features),
                    )
                exact = {
                    name: bool(np.array_equal(outputs[name], outputs["strict_shared_b"]))
                    for name in ARMS
                }
                finite = {
                    name: bool(
                        np.isfinite(
                            (outputs[name].astype(np.uint32) << 16).view(np.float32)
                        ).all()
                    )
                    for name in ARMS
                }

                for _ in range(int(args.warmup)):
                    for name in ARMS:
                        launches[name]()
                runtime.device_synchronize()
                event_samples: dict[str, list[float]] = {name: [] for name in ARMS}
                wall_samples: dict[str, list[float]] = {name: [] for name in ARMS}
                orders: list[list[str]] = []
                for sample_index in range(int(args.samples)):
                    order = _counterbalanced_order(
                        ARMS,
                        cell_index=cell_index,
                        sample_index=sample_index,
                    )
                    orders.append(list(order))
                    for name in order:
                        event_ms, wall_ms = _event_sample_ms(
                            runtime,
                            launches[name],
                            burst=int(args.burst),
                        )
                        event_samples[name].append(event_ms)
                        wall_samples[name].append(wall_ms)

                timings = {
                    name: {
                        "event": _sample_summary(event_samples[name]),
                        "operation_complete_wall": _sample_summary(wall_samples[name]),
                    }
                    for name in ARMS
                }
                incumbent_ms = float(timings["incumbent"]["event"]["median_ms"])
                smallm_ms = float(timings["smallm"]["event"]["median_ms"])
                cells.append(
                    {
                        "layer": layer_id,
                        "role": role,
                        "source_name": weight.spec.source.name,
                        "rows": rows,
                        "in_features": in_features,
                        "out_features": out_features,
                        "incumbent_variant": incumbent_variant,
                        "candidate_variant": "t16_wmma_prefill_smallm_bf16_bf16_out",
                        "strict_variant": "t16_wmma_prefill_shared_b_bf16_bf16_out",
                        "correctness": {
                            "exact_vs_strict": exact,
                            "finite": finite,
                            "passed": all(exact.values()) and all(finite.values()),
                        },
                        "timings": timings,
                        "incumbent_over_smallm": incumbent_ms / smallm_ms,
                        "orders": orders,
                    }
                )
                print(
                    f"{role:>11} rows={rows} shape=({in_features},{out_features}) "
                    f"{incumbent_variant.removesuffix('_bf16_bf16_out')}={incumbent_ms:.4f}ms "
                    f"smallm={smallm_ms:.4f}ms ratio={incumbent_ms / smallm_ms:.3f}x "
                    f"exact={all(exact.values())}",
                    flush=True,
                )
                cell_index += 1
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        weights.free(runtime=runtime)

    passed = all(cell["correctness"]["passed"] for cell in cells)
    return {
        "schema": 1,
        "kind": "gguf_t16_production_sidecar_rows2_8",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if passed else "failed",
        "passed": passed,
        "source": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": _tracked_dirty(),
        },
        "model": {
            "path": str(model),
            "fingerprint": _sampled_model_fingerprint(model),
        },
        "hardware": {
            "physical_host": platform.node(),
            "machine": platform.machine(),
            "backend": "hip_gfx1100",
        },
        "protocol": {
            "rows": list(args.rows),
            "samples": int(args.samples),
            "burst": int(args.burst),
            "warmup": int(args.warmup),
            "seed": int(args.seed),
            "timing": "counterbalanced HIP-event burst plus operation-complete event-synchronize wall",
            "payload": "normal Qwen GGUF materializer allocation; no reconstructed tiles",
            "ratio": "incumbent event median / small-M event median; >1 means small-M faster",
        },
        "inventory": inventory,
        "cells": cells,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--rows",
        type=_parse_rows,
        default=DEFAULT_ROWS,
        help="comma-separated subset of rows 2..8",
    )
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--burst", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.samples < 3 or args.burst < 1 or args.warmup < 1:
        raise SystemExit("samples>=3, burst>=1, and warmup>=1 are required")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
