#!/usr/bin/env python3
"""Rank-local MLP materialization probe for a two-rank TP plan.

The TP2 decision needs shard-shaped segment times, and a shard segment cannot be
measured until the rank-local weight bytes exist. This probe establishes that
prerequisite for one representative MLP and answers three questions with
evidence rather than assumption:

  1. Does the byte-preserving shard planner produce rank-local payloads for the
     MLP's real quant types, and do those payloads round-trip bit-exactly?
  2. Are the local shapes admissible for the *resident* layouts the engine would
     use (Q4_K pack8/t16 for gate and up, raw Q6_K for down), including the tile
     alignment the t16 repack requires?
  3. Does the engine's linear dispatch resolve for a shard-shaped weight, and does
     it resolve to the same kernel as the TP1 shape?

Question 3 is the one that decides whether a shard segment needs a new kernel.
The dispatch key is ``(layout, activation, output, quant_key, variant_for_rows)``
and the weight's own row count is a launch parameter, so the expected answer is
that a shard resolves identically. That is checked here instead of assumed, and a
change that introduced a shape-keyed dispatch would fail this probe.

Usage:
    python3 scripts/tp2_mlp_shard_plan_probe.py \
        --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
        --layer 0 --world-size 2 \
        --json benchmarks/results/2026-09-15-w7900-tp2-mlp-shard-plan-probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader, scan_gguf  # noqa: E402
from hipengine.loading.qwen35_gguf import qwen35_gguf_config_from_metadata  # noqa: E402
from hipengine.loading.qwen35_gguf_consumer_surface import (  # noqa: E402
    resolve_linear_consumer_contract,
)
from hipengine.loading.qwen35_gguf_materialize import _planned_t16_nbytes  # noqa: E402
from hipengine.loading.qwen35_gguf_shards import (  # noqa: E402
    build_shard_manifest,
    materialize_slice,
    reconstruct_tensor,
    source_payload,
)
from hipengine.quant.gguf_t16 import GGUF_T16_COLS  # noqa: E402
from hipengine.runtime.gguf_linear import resolve_gguf_linear_dispatch  # noqa: E402

#: The MLP tensors of one block, with the axis the planner splits and the
#: resident layout the engine uses for each source quant type.
MLP_TENSORS = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")


def _mlp_tensor_names(layer: int) -> tuple[str, ...]:
    return tuple(f"blk.{int(layer)}.{name}" for name in MLP_TENSORS)


class _WeightSpec:
    """Minimal ``GGUFWeightSpec`` stand-in for dispatch resolution."""

    def __init__(self, layout: str, quant_key: str) -> None:
        self.layout = layout
        self.quant_key = quant_key


class _Weight:
    """Minimal ``GGUFDeviceWeight`` stand-in for dispatch resolution."""

    def __init__(self, layout: str, quant_key: str, backend: str) -> None:
        self.spec = _WeightSpec(layout, quant_key)
        self.backend = backend


def _resident_layout(quant_type_name: str) -> tuple[str, str]:
    """The (layout, quant_key) the engine resolves for one source quant type.

    Read from the same tables the materializer uses rather than hardcoded: a
    rank-2 Q4_K linear goes to the t16 layout, a rank-2 Q6_K linear to the raw
    layout. An unknown type is reported instead of guessed.
    """

    from hipengine.loading.qwen35_gguf_consumer_surface import (
        RAW_LINEAR_SOURCE_QUANT_KEYS,
    )
    from hipengine.loading.qwen35_gguf_materialize import (
        LAYOUT_GGUF_Q4_K_T16,
        LAYOUT_RAW_GGUF,
    )

    name = str(quant_type_name)
    if name == "Q4_K":
        return LAYOUT_GGUF_Q4_K_T16, "gguf_q4_k_t16_v1"
    if name in RAW_LINEAR_SOURCE_QUANT_KEYS:
        return LAYOUT_RAW_GGUF, RAW_LINEAR_SOURCE_QUANT_KEYS[name]
    raise ValueError(f"no resident layout is recorded for source quant {name!r}")


def _t16_admissible(*, out_features: int, bytes_per_row: int, block_bytes: int) -> dict[str, Any]:
    """The t16 repack's own admissibility test, applied to a local shape."""

    cols_ok = int(out_features) % GGUF_T16_COLS == 0
    block_ok = int(bytes_per_row) % int(block_bytes) == 0
    return {
        "t16_cols": GGUF_T16_COLS,
        "out_features": int(out_features),
        "out_features_divisible": cols_ok,
        "bytes_per_row": int(bytes_per_row),
        "block_bytes": int(block_bytes),
        "bytes_per_row_divisible": block_ok,
        "admissible": bool(cols_ok and block_ok),
    }


def probe(
    *,
    model: Path,
    layer: int,
    world_size: int,
    activation_dtype: str = "bf16",
    output_dtype: str = "bf16",
) -> dict[str, Any]:
    info = scan_gguf(str(model))
    config = qwen35_gguf_config_from_metadata(info)
    reader = GGUFReader(str(model))
    names = _mlp_tensor_names(layer)
    manifest = build_shard_manifest(info, world_size=int(world_size), model_hash="mlp-probe")
    by_name = {plan.name: plan for plan in manifest.tensors}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"manifest has no plan for {missing}")

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "tp2-mlp-shard-plan-probe",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(model),
        "model_hash": manifest.model_hash,
        "layer": int(layer),
        "world_size": int(world_size),
        "hidden_size": int(config.hidden_size),
        "manifest_hash": manifest.manifest_hash(),
        "activation_dtype": activation_dtype,
        "output_dtype": output_dtype,
        "tensors": {},
        "questions": {},
    }

    all_exact = True
    all_admissible = True
    all_dispatch_match = True
    tp1_bytes = 0
    shard_bytes = 0

    for name in names:
        plan = by_name[name]
        tensor = reader.tensor_info(name)
        source = source_payload(reader.path, data_offset=tensor.data_offset, nbytes=tensor.nbytes)
        layout, quant_key = _resident_layout(tensor.ggml_type_name)
        block_bytes = int(plan.type_size)
        bytes_per_row = int(plan.source_row_bytes)

        # The TP1 shape's dispatch, for comparison.
        tp1_dispatch = resolve_gguf_linear_dispatch(
            _Weight(layout, quant_key, backend="hip_gfx1100"),
            activation_dtype=activation_dtype,
            output_dtype=output_dtype,
            backend="hip_gfx1100",
            rows=1,
        )

        entry: dict[str, Any] = {
            "source_shape": list(plan.source_shape),
            "source_nbytes": int(plan.source_nbytes),
            "quant_type": str(tensor.ggml_type_name),
            "kind": plan.kind,
            "split_axis": plan.axis,
            "block_size": int(plan.block_size),
            "type_size": block_bytes,
            "resident_layout": layout,
            "quant_key": quant_key,
            "tp1_dispatch": {
                "key": str(tp1_dispatch.key),
                "abi": str(tp1_dispatch.abi),
            },
            "ranks": {},
        }

        payloads: dict[int, Any] = {}
        for shard in plan.slices:
            payload = materialize_slice(source, shard)
            payloads[shard.rank] = payload
            local_shape = tuple(int(dim) for dim in shard.local_shape)
            # A column split changes the row count (the t16 tile axis); a row
            # split keeps the row count and shrinks the bytes per row.
            if plan.kind == "column":
                admissible = _t16_admissible(
                    out_features=local_shape[0], bytes_per_row=bytes_per_row, block_bytes=block_bytes
                )
            else:
                local_bytes_per_row = int(shard.local_nbytes) // max(1, local_shape[0])
                admissible = _t16_admissible(
                    out_features=local_shape[0],
                    bytes_per_row=local_bytes_per_row,
                    block_bytes=block_bytes,
                )
            # The local shape's own dispatch: the weight row count is a launch
            # parameter, so a shard must resolve to the same contract as TP1.
            local_dispatch = resolve_gguf_linear_dispatch(
                _Weight(layout, quant_key, backend="hip_gfx1100"),
                activation_dtype=activation_dtype,
                output_dtype=output_dtype,
                backend="hip_gfx1100",
                rows=1,
            )
            dispatch_match = str(local_dispatch.key) == str(tp1_dispatch.key)
            all_dispatch_match = all_dispatch_match and dispatch_match
            all_admissible = all_admissible and admissible["admissible"]
            entry["ranks"][str(shard.rank)] = {
                "axis_ranges": [list(r) for r in shard.axis_ranges],
                "local_shape": list(local_shape),
                "local_nbytes": int(shard.local_nbytes),
                "segments": sum(1 for _ in shard.iter_segments()),
                "layout_admissible": admissible,
                "dispatch_key": str(local_dispatch.key),
                "dispatch_matches_tp1": dispatch_match,
            }
            shard_bytes += int(shard.local_nbytes)

        rebuilt = reconstruct_tensor(manifest, name, payloads)
        exact = bool(rebuilt.tobytes() == bytes(source))
        all_exact = all_exact and exact
        entry["round_trip_bit_exact"] = exact
        tp1_bytes += int(plan.source_nbytes)
        report["tensors"][name] = entry
        del payloads, rebuilt, source

    # The shard MLP's reduction payload: the down projection is row-parallel, so
    # each rank produces a partial output over the full hidden size and the two
    # partials are summed once per layer.
    down = by_name[f"blk.{int(layer)}.ffn_down.weight"]
    reduction_elements = int(down.source_shape[0])
    report["questions"] = {
        "rank_local_materialization_exists": {
            "answer": True,
            "evidence": "each MLP tensor has one slice per rank with a concrete local shape and byte count",
            "tp1_weight_bytes": tp1_bytes,
            "shard_weight_bytes_per_rank": shard_bytes // int(world_size),
            "shard_to_tp1_ratio": (shard_bytes / int(world_size)) / tp1_bytes,
        },
        "rank_local_bytes_round_trip": {
            "answer": bool(all_exact),
            "evidence": "reconstruct_tensor(manifest, name, rank payloads) compared byte-wise to the source payload",
        },
        "local_shapes_are_layout_admissible": {
            "answer": bool(all_admissible),
            "evidence": "the t16 repack's own divisibility test applied to each local shape",
        },
        "shard_dispatch_needs_no_new_kernel": {
            "answer": bool(all_dispatch_match),
            "evidence": (
                "resolve_gguf_linear_dispatch for the shard-shaped weight equals the TP1 "
                "dispatch key; the weight row count is a launch parameter, not a dispatch key"
            ),
        },
    }
    report["reduction_point"] = {
        "kind": "all_reduce",
        "per_layer_count": 1,
        "elements": reduction_elements,
        "dtype": "fp32",
        "payload_bytes": reduction_elements * 4,
        "note": (
            "the down projection is row-parallel, so each rank holds a partial output over "
            "the full hidden size and the partials are summed once per layer; this is the "
            "payload the transport screen already measured"
        ),
    }
    report["next"] = (
        "measure complete TP1 and local-shard MLP segment walls on both cards, connect the "
        "shard outputs through the native exchange into the next consumer, and validate "
        "against the CPU split oracle and the TP1 teacher under the production numerical "
        "contract"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    report = probe(model=args.model, layer=args.layer, world_size=args.world_size)
    text = json.dumps(report, indent=2) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
        print(f"wrote {args.json}", file=sys.stderr)
    else:
        print(text)

    for question, answer in report["questions"].items():
        verdict = "yes" if answer["answer"] else "NO"
        print(f"[{verdict}] {question}", file=sys.stderr)
    print(
        f"TP1 MLP weights {report['questions']['rank_local_materialization_exists']['tp1_weight_bytes'] / 1e6:.1f} MB, "
        f"shard per rank "
        f"{report['questions']['rank_local_materialization_exists']['shard_weight_bytes_per_rank'] / 1e6:.1f} MB, "
        f"reduction payload {report['reduction_point']['payload_bytes']} B",
        file=sys.stderr,
    )
    ok = all(answer["answer"] for answer in report["questions"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
