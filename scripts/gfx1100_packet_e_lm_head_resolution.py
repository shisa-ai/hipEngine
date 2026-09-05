#!/usr/bin/env python3
"""Root LM-head rowtile resolution probe for the W7900 packet E gate.

Materializes only ``root.lm_head`` from a real GGUF, resolves the four-axis
linear dispatch exactly as ``_verify_lm_head_rowtile()`` does, and reports the
registered rowtile primitive's ``_hipengine_max_rows`` plus the effective
chunk partition that ``_verify_lm_head_rowtile_chunked()`` would produce for
every physical width under each candidate ``GGUF_Q6_LM_HEAD_MAX_CHUNK``.

Diagnostic only: no kernel, route, or capability is changed. The audit's
packet E RED step asks whether the W7900-resolved rowtile caps below 8; this
answers it from the resolved primitive rather than from the package constant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.kernels.registry import KernelKey, is_registered, resolve  # noqa: E402
from hipengine.loading.qwen35_gguf_materialize import (  # noqa: E402
    materialize_qwen35_gguf_weights,
)
from hipengine.runtime.gguf_linear import (  # noqa: E402
    GGUF_ACTIVATION_BF16,
    GGUF_OUTPUT_F32,
    resolve_gguf_linear_dispatch,
)
from hipengine.runtime.qwen35_gguf_runner import _small_b_rowtile_chunks  # noqa: E402

ROWTILE_VARIANT = "t16_gemv_rowtile_bf16_f32_out"


def _key_fields(key: KernelKey) -> list[str]:
    return [key.backend, key.layer, key.quant, key.variant]


def _chunk_partition(rows: int, max_chunk: int) -> tuple[int, ...]:
    """Mirror ``_verify_lm_head_rowtile_chunked()``'s partition decision."""

    if rows <= max_chunk:
        return (rows,)
    return tuple(int(chunk) for chunk in _small_b_rowtile_chunks(rows, max_chunk=max_chunk))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument(
        "--candidate-chunks",
        default="4,6,8",
        help="comma-separated GGUF_Q6_LM_HEAD_MAX_CHUNK values to report",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    runtime = get_hip_runtime()
    weights = materialize_qwen35_gguf_weights(
        args.model,
        selected_slots=["root.lm_head"],
        backend=args.backend,
        runtime=runtime,
    )
    try:
        weight = weights.root("lm_head")
        spec = weight.spec
        report: dict[str, Any] = {
            "model": str(args.model),
            "backend": args.backend,
            "lm_head": {
                "slot_path": spec.slot_path,
                "quant_key": spec.quant_key,
                "layout": spec.layout,
                "allocation_names": list(spec.allocation_names),
                "shape": [int(dim) for dim in spec.source.shape],
            },
            "resolution_by_rows": {},
        }
        max_rows_by_rows: dict[int, int] = {}
        for rows in range(1, 9):
            dispatch = resolve_gguf_linear_dispatch(
                weight,
                activation_dtype=GGUF_ACTIVATION_BF16,
                output_dtype=GGUF_OUTPUT_F32,
                backend=args.backend,
                rows=rows,
            )
            rowtile_key = KernelKey(
                dispatch.key.backend,
                dispatch.key.layer,
                dispatch.key.quant,
                ROWTILE_VARIANT,
            )
            registered = is_registered(rowtile_key)
            max_rows = 0
            fn_name = None
            if registered:
                fn = resolve(
                    backend=rowtile_key.backend,
                    layer=rowtile_key.layer,
                    quant=rowtile_key.quant,
                    variant=rowtile_key.variant,
                )
                max_rows = int(getattr(fn, "_hipengine_max_rows", 0))
                fn_name = getattr(fn, "__name__", None)
            max_rows_by_rows[rows] = max_rows
            report["resolution_by_rows"][str(rows)] = {
                "decode_dispatch_key": _key_fields(dispatch.key),
                "decode_abi": dispatch.abi,
                "rowtile_key": _key_fields(rowtile_key),
                "rowtile_registered": registered,
                "rowtile_function": fn_name,
                "rowtile_max_rows": max_rows,
            }

        # ``_verify_lm_head_rowtile_max_rows()`` probes at rows=2 and fails
        # closed outside [2, 8].
        probe_max_rows = max_rows_by_rows[2]
        effective_primitive_max_rows = probe_max_rows if 2 <= probe_max_rows <= 8 else 0
        report["verify_lm_head_rowtile_max_rows"] = effective_primitive_max_rows

        partitions: dict[str, Any] = {}
        for raw in str(args.candidate_chunks).split(","):
            raw = raw.strip()
            if not raw:
                continue
            requested = int(raw)
            effective = min(requested, effective_primitive_max_rows)
            rows_map = {}
            for rows in range(2, 9):
                if effective < 2:
                    rows_map[str(rows)] = None
                    continue
                partition = _chunk_partition(rows, effective)
                rows_map[str(rows)] = {
                    "partition": list(partition),
                    "launches": len(partition),
                }
            partitions[raw] = {
                "requested_max_chunk": requested,
                "effective_max_chunk": effective,
                "rows": rows_map,
            }
        report["chunk_partitions"] = partitions

        print(json.dumps(report, indent=2, sort_keys=True))
        if args.json is not None:
            args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    finally:
        weights.free(runtime=runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
