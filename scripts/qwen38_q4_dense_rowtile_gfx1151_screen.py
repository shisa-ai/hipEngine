#!/usr/bin/env python3
"""Screen the withheld dense Q4T16 col4 and down+residual rowtiles on gfx1151.

Both variants are registered on hip_gfx1100 and withheld from gfx1151 by
``_GFX1151_ALIAS_EXCLUSIONS`` pending an independent gfx1151 shape crossover.
This harness measures that crossover on actual Qwen3.8-27B Q4_K weights with
counterbalanced HIP-event bursts, and verifies bit-exactness against the
retained gfx1151 owner before any timing is trusted.

Scope, from the dispatch code rather than from the variant names:

* ``dense_rowtile_col4_bf16_bf16_out`` is only reachable through
  ``_q4_t16_sidecar_decode_variants``, which gates it to ``rows <= 4`` and
  ``(in_features, out_features) == (5_120, 1_024)`` -- the narrow K/V family.
  The retained gfx1151 alternative at that shape and row band is
  ``dense_rowtile_bf16_bf16_out``.
* ``dense_rowtile_bf16_residual_bf16_out`` is the composite sibling of
  ``dense_rowtile_bf16_bf16_out``. gfx1151 does not register the composite, so
  the retained route is the rowtile projection followed by the primitive
  ``gguf_bf16_add``.

The harness is a leaf screen: it measures per-call cost, not complete-model
throughput. A favourable crossover is a precondition for a full-model gate,
not a substitute for one.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import build_gguf_ops, gguf_bf16_add
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out,
    gguf_q4_k_t16_dense_rowtile_bf16_bf16_out,
    gguf_q4_k_t16_dense_rowtile_bf16_residual_bf16_out,
    gguf_q4_k_t16_dense_rowtile_col4_bf16_bf16_out,
    gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
    gguf_q4_k_t16_dense_single_local32_bf16_residual_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_OUTPUT = Path("/tmp/hip1151-t6/q4-dense-rowtile-screen.json")

# ``(in_features, out_features)`` for the two screened families.
COL4_SHAPE = (5_120, 1_024)
RESIDUAL_SHAPE = (17_408, 5_120)

# Actual Qwen3.8-27B Q4_K tensors. attn_v has 8 Q4_K instances
# (11/15/23/27/35/39/47/51) and attn_k has 17; ffn_down has 32.
COL4_TENSORS = (
    "blk.11.attn_v.weight",
    "blk.23.attn_v.weight",
    "blk.47.attn_v.weight",
    "blk.51.attn_v.weight",
    "blk.15.attn_k.weight",
    "blk.39.attn_k.weight",
)
RESIDUAL_TENSORS = (
    "blk.8.ffn_down.weight",
    "blk.23.ffn_down.weight",
    "blk.47.ffn_down.weight",
    "blk.54.ffn_down.weight",
)

# Q4_K tensors per screened family in the complete model. Used only to project
# a per-call saving onto a per-token figure.
COL4_TENSOR_COUNT = 25
RESIDUAL_TENSOR_COUNT = 32


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--burst", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0x1151_6)
    parser.add_argument(
        "--family",
        choices=("col4", "residual", "both"),
        default="both",
    )
    parser.add_argument("--rows", type=str, default=None)
    parser.add_argument("--tensors", type=str, default=None)
    return parser.parse_args()


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def _upload(runtime, values: np.ndarray) -> DeviceBuffer:
    array = np.ascontiguousarray(values)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _read(runtime, buffer: DeviceBuffer, shape: tuple[int, ...], dtype) -> np.ndarray:
    output = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(output), buffer, runtime=runtime)
    return output


def _tile16(reader: GGUFReader, name: str, runtime) -> tuple[DeviceBuffer, int, int]:
    info = reader.tensor_info(name)
    if info.ggml_type_name != "Q4_K" or len(info.shape) != 2:
        raise ValueError(f"{name} is not rank-2 Q4_K: {info}")
    out_features, in_features = int(info.shape[0]), int(info.shape[1])
    raw = np.ascontiguousarray(reader.tensor_data(name))
    tiles = np.ascontiguousarray(repack_gguf_q4_k_tile16(raw[None, ...]).tiles)
    del raw
    return _upload(runtime, tiles), in_features, out_features


class Op:
    """One logical projection, possibly spanning several kernel launches."""

    def __init__(self, label: str, launches: int, run, out: DeviceBuffer,
                 shape: tuple[int, int]):
        self.label = label
        self.launches = launches
        self.run = run
        self.out = out
        self.shape = shape

    def read(self, runtime) -> np.ndarray:
        return _read(runtime, self.out, self.shape, np.uint16)


def _build_col4_ops(
    library,
    ops_library,
    runtime,
    x: DeviceBuffer,
    tiles: DeviceBuffer,
    rows: int,
    in_features: int,
    out_features: int,
) -> dict[str, Op]:
    out_shape = (rows, out_features)
    bytes_out = rows * out_features * 2

    def make(label, fn, launches=1):
        out = malloc(bytes_out, runtime=runtime)

        def run():
            fn(
                x.ptr,
                tiles.ptr,
                out.ptr,
                rows,
                in_features,
                out_features,
                stream=0,
                library=library,
                runtime=runtime,
            )

        return Op(label, launches, run, out, out_shape)

    candidates = {
        "col4": make(
            "col4", gguf_q4_k_t16_dense_rowtile_col4_bf16_bf16_out
        ),
        "rowtile": make(
            "rowtile", gguf_q4_k_t16_dense_rowtile_bf16_bf16_out
        ),
    }
    if rows <= 8:
        candidates["rowtile16_w2"] = make(
            "rowtile16_w2", gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out
        )
    # Per-row c1 owner, launched once per row (the physical fallback route).
    per_row_out = malloc(bytes_out, runtime=runtime)

    def run_per_row():
        for row in range(rows):
            gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
                x.ptr + row * in_features * 2,
                tiles.ptr,
                per_row_out.ptr + row * out_features * 2,
                1,
                in_features,
                out_features,
                stream=0,
                library=library,
                runtime=runtime,
            )

    candidates["per_row_c1"] = Op(
        "per_row_c1", rows, run_per_row, per_row_out, out_shape
    )
    return candidates


def _build_residual_ops(
    library,
    ops_library,
    runtime,
    x: DeviceBuffer,
    tiles: DeviceBuffer,
    residual: DeviceBuffer,
    rows: int,
    in_features: int,
    out_features: int,
) -> dict[str, Op]:
    out_shape = (rows, out_features)
    bytes_out = rows * out_features * 2
    elements = rows * out_features

    def make_fused(label, fn):
        out = malloc(bytes_out, runtime=runtime)

        def run():
            fn(
                x.ptr,
                tiles.ptr,
                residual.ptr,
                out.ptr,
                rows,
                in_features,
                out_features,
                stream=0,
                library=library,
                runtime=runtime,
            )

        return Op(label, 1, run, out, out_shape)

    def make_chain(label, proj_fn):
        """Rowtile projection followed by the primitive rounded BF16 add."""

        out = malloc(bytes_out, runtime=runtime)

        def run():
            proj_fn(
                x.ptr,
                tiles.ptr,
                out.ptr,
                rows,
                in_features,
                out_features,
                stream=0,
                library=library,
                runtime=runtime,
            )
            gguf_bf16_add(
                out.ptr,
                residual.ptr,
                out.ptr,
                elements,
                stream=0,
                library=ops_library,
                runtime=runtime,
            )

        return Op(label, 2, run, out, out_shape)

    ops = {
        "fused_rowtile_residual": make_fused(
            "fused_rowtile_residual",
            gguf_q4_k_t16_dense_rowtile_bf16_residual_bf16_out,
        ),
        "rowtile_plus_add": make_chain(
            "rowtile_plus_add", gguf_q4_k_t16_dense_rowtile_bf16_bf16_out
        ),
    }
    if rows <= 8:
        ops["rowtile16_w2_plus_add"] = make_chain(
            "rowtile16_w2_plus_add",
            gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out,
        )
    # Registered gfx1151 c1 residual owner, launched once per row.
    per_row_out = malloc(bytes_out, runtime=runtime)

    def run_per_row():
        for row in range(rows):
            gguf_q4_k_t16_dense_single_local32_bf16_residual_bf16_out(
                x.ptr + row * in_features * 2,
                tiles.ptr,
                residual.ptr + row * out_features * 2,
                per_row_out.ptr + row * out_features * 2,
                1,
                in_features,
                out_features,
                stream=0,
                library=library,
                runtime=runtime,
            )

    ops["per_row_c1_residual"] = Op(
        "per_row_c1_residual", rows, run_per_row, per_row_out, out_shape
    )
    return ops


def _time_op(runtime, op: Op, burst: int, start_event: int, stop_event: int) -> float:
    runtime.event_record(start_event)
    for _ in range(burst):
        op.run()
    runtime.event_record(stop_event)
    runtime.event_synchronize(stop_event)
    return runtime.event_elapsed_time_ms(start_event, stop_event) * 1000.0 / burst


def _screen_case(
    args,
    runtime,
    library,
    ops_library,
    reader: GGUFReader,
    *,
    family: str,
    name: str,
    rows: int,
    start_event: int,
    stop_event: int,
) -> dict[str, object]:
    tiles, in_features, out_features = _tile16(reader, name, runtime)
    expected_shape = COL4_SHAPE if family == "col4" else RESIDUAL_SHAPE
    if (in_features, out_features) != expected_shape:
        raise ValueError(
            f"{name} has shape {(in_features, out_features)}, expected {expected_shape}"
        )
    rng = np.random.default_rng(args.seed + rows * 131 + hash(name) % 9973)
    x = _upload(
        runtime, _bf16_bits(rng.normal(0.0, 0.2, size=(rows, in_features)))
    )
    residual = None
    try:
        if family == "col4":
            ops = _build_col4_ops(
                library, ops_library, runtime, x, tiles, rows, in_features, out_features
            )
            reference = "rowtile"
        else:
            residual = _upload(
                runtime,
                _bf16_bits(rng.normal(0.0, 0.2, size=(rows, out_features))),
            )
            ops = _build_residual_ops(
                library,
                ops_library,
                runtime,
                x,
                tiles,
                residual,
                rows,
                in_features,
                out_features,
            )
            reference = "rowtile_plus_add"

        # Probe support first. Some arms are defined only on part of the row
        # band (the 2-4-row rowtile, the c1-only single owner), so an
        # unsupported arm is dropped with its reason instead of failing the
        # whole screen. The wrappers gate shape before any launch, so a
        # rejected arm enqueues no device work.
        supported: dict[str, Op] = {}
        unsupported: dict[str, str] = {}
        for label, op in ops.items():
            try:
                op.run()
                runtime.device_synchronize()
            except ValueError as error:
                unsupported[label] = str(error)
                continue
            supported[label] = op
        if reference not in supported:
            raise RuntimeError(
                f"reference arm {reference} is unsupported at rows={rows}: "
                f"{unsupported.get(reference)}"
            )

        # Exactness before timing: every arm must reproduce the retained owner
        # bit-for-bit, otherwise the timing is meaningless.
        baseline = supported[reference].read(runtime)
        exactness: dict[str, object] = {}
        for label, op in supported.items():
            values = op.read(runtime)
            mismatches = int(np.count_nonzero(values != baseline))
            exactness[label] = {
                "sha256": _sha256(values),
                "mismatches": mismatches,
                "exact": mismatches == 0,
            }

        candidate_label = "col4" if family == "col4" else "fused_rowtile_residual"
        candidate = supported[candidate_label]

        for op in supported.values():
            for _ in range(args.warmups):
                op.run()
        runtime.device_synchronize()

        samples: dict[str, list[float]] = {label: [] for label in supported}
        for sample in range(args.samples):
            order = list(supported)
            if sample % 2:
                order.reverse()
            for label in order:
                samples[label].append(
                    _time_op(runtime, supported[label], args.burst, start_event, stop_event)
                )

        medians = {label: statistics.median(values) for label, values in samples.items()}
        comparisons: dict[str, object] = {}
        for label, median in medians.items():
            if label == candidate.label:
                continue
            speedup = median / medians[candidate.label]
            comparisons[label] = {
                "control_median_us": median,
                "candidate_median_us": medians[candidate.label],
                "speedup": speedup,
                "candidate_delta_pct": (speedup - 1.0) * 100.0,
                "control_launches": supported[label].launches,
                "candidate_launches": candidate.launches,
                "wins": sum(
                    1
                    for c, k in zip(samples[label], samples[candidate.label])
                    if k < c
                ),
                "pairs": len(samples[label]),
                "exact": exactness[label]["exact"],
                "mismatches": exactness[label]["mismatches"],
            }
        return {
            "family": family,
            "tensor": name,
            "rows": rows,
            "in_features": in_features,
            "out_features": out_features,
            "candidate": candidate.label,
            "candidate_median_us": medians[candidate.label],
            "candidate_launches": candidate.launches,
            "medians_us": medians,
            "samples_us": samples,
            "exactness": exactness,
            "unsupported": unsupported,
            "comparisons": comparisons,
        }
    finally:
        for op in locals().get("ops", {}).values():
            free(op.out, runtime=runtime)
        if residual is not None:
            free(residual, runtime=runtime)
        free(x, runtime=runtime)
        free(tiles, runtime=runtime)


def _rows_for(family: str, requested: str | None) -> tuple[int, ...]:
    """Row band per family. col4 is dispatch-gated to rows<=4; residual to <=4."""

    if requested:
        return tuple(int(part) for part in requested.split(","))
    return (2, 3, 4)


def main() -> int:
    args = _parse_args()
    runtime = get_hip_runtime()
    command = [sys.executable, *sys.argv]

    build_artifact = build_gguf_t16_selected_gemv(load=False)
    ops_artifact = build_gguf_ops(load=False)
    library = build_gguf_t16_selected_gemv()
    ops_library = build_gguf_ops()
    reader = GGUFReader(args.model)

    families: list[tuple[str, tuple[str, ...]]] = []
    if args.family in ("col4", "both"):
        families.append(("col4", COL4_TENSORS))
    if args.family in ("residual", "both"):
        families.append(("residual", RESIDUAL_TENSORS))
    if args.tensors:
        wanted = set(args.tensors.split(","))
        families = [
            (family, tuple(n for n in names if n in wanted))
            for family, names in families
        ]

    start_event = runtime.event_create()
    stop_event = runtime.event_create()
    cases: list[dict[str, object]] = []
    try:
        for family, names in families:
            for name in names:
                for rows in _rows_for(family, args.rows):
                    case = _screen_case(
                        args,
                        runtime,
                        library,
                        ops_library,
                        reader,
                        family=family,
                        name=name,
                        rows=rows,
                        start_event=start_event,
                        stop_event=stop_event,
                    )
                    cases.append(case)
                    primary = case["comparisons"].get(
                        "rowtile" if family == "col4" else "rowtile_plus_add"
                    )
                    if primary is None:
                        print(
                            f"{family:8s} {name:22s} rows={rows} "
                            f"cand={case['candidate_median_us']:8.2f}us "
                            f"(reference arm unsupported)",
                            flush=True,
                        )
                        continue
                    print(
                        f"{family:8s} {name:22s} rows={rows} "
                        f"cand={case['candidate_median_us']:8.2f}us "
                        f"vs {primary['control_median_us']:8.2f}us "
                        f"speedup={primary['speedup']:.4f} "
                        f"wins={primary['wins']}/{primary['pairs']} "
                        f"exact={primary['exact']}",
                        flush=True,
                    )
    finally:
        runtime.event_destroy(start_event)
        runtime.event_destroy(stop_event)

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend="hip_gfx1151",
        resolved_backend="hip_gfx1151",
        target_arch="gfx1151",
        model_path=args.model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
        },
        timing_protocol=(
            f"counterbalanced HIP-event bursts: {args.warmups} warmups, "
            f"{args.samples} samples, {args.burst} launches/sample"
        ),
        warmups=args.warmups,
        repetitions=args.samples,
    )
    git_head = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    payload = {
        "kind": "qwen38_gfx1151_q4_dense_rowtile_screen",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "git_status": subprocess.check_output(
            ("git", "status", "--short", "--untracked-files=no"),
            cwd=ROOT,
            text=True,
        ).splitlines(),
        "host": os.uname().nodename,
        "build": {
            "t16_selected_gemv": getattr(build_artifact, "cache_key", None),
            "gguf_ops": getattr(ops_artifact, "cache_key", None),
        },
        "model": str(args.model),
        "settings": {
            "warmups": args.warmups,
            "samples": args.samples,
            "burst": args.burst,
            "seed": args.seed,
        },
        "scope": {
            "col4_shape": list(COL4_SHAPE),
            "col4_tensor_count": COL4_TENSOR_COUNT,
            "residual_shape": list(RESIDUAL_SHAPE),
            "residual_tensor_count": RESIDUAL_TENSOR_COUNT,
        },
        "provenance": provenance,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
