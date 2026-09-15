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
  3. Does the repack commute with the split, so a rank can repack its own local
     slice instead of needing the full tensor - for the layout the incumbent
     model actually uses, and for the split axis that tensor actually uses?

This probe is a **host-payload** prerequisite check. It does not qualify device
execution: passing here means the rank-local bytes are the right bytes in the
right layout, not that a kernel consumes them correctly. Device execution is
qualified separately by an execution test that runs the real segment and compares
it against an independent CPU split oracle.

The resident layout per tensor is resolved through the engine's own planner
(``plan_qwen35_gguf_materialization`` with the same capability and environment
flags a real load resolves), never hardcoded: whether the Q6_K down projection is
raw or t16 depends on ``decode_repack`` and the dense capability set, so guessing
it would benchmark a different TP1 path than the incumbent.

Usage:
    python3 scripts/tp2_mlp_shard_plan_probe.py \
        --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
        --layer 0 --world-size 2 \
        --json benchmarks/results/2026-09-15-w7900-tp2-mlp-shard-plan-probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader, scan_gguf  # noqa: E402
from hipengine.distributed.kv import full_attention_layer_count  # noqa: E402
from hipengine.loading.qwen35_gguf import (  # noqa: E402
    build_qwen35_gguf_tensor_map,
    qwen35_gguf_config_from_metadata,
)
from hipengine.loading.qwen35_gguf_admission import (  # noqa: E402
    build_qwen35_gguf_role_manifest,
)
from hipengine.kernels.backends import backend_package_capability  # noqa: E402
from hipengine.loading.qwen35_gguf_materialize import (  # noqa: E402
    gguf_decode_repack_enabled,
    plan_qwen35_gguf_materialization,
)
from hipengine.loading.qwen35_gguf_policy import resolve_gguf_dense_flags  # noqa: E402
from hipengine.loading.qwen35_gguf_consumer_surface import (  # noqa: E402
    resolve_linear_consumer_contract,
)
from hipengine.loading.qwen35_gguf_materialize import (  # noqa: E402
    LAYOUT_GGUF_Q4_K_T16,
    _planned_t16_nbytes,
)
from hipengine.loading.qwen35_gguf_shards import (  # noqa: E402
    build_shard_manifest,
    materialize_slice,
    reconstruct_tensor,
    source_payload,
)
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16  # noqa: E402
from hipengine.quant.gguf_t16 import (  # noqa: E402
    repack_gguf_q6_k_tile16,
    repack_gguf_q6_k_tile16_qmicro_planar,
)
from hipengine.quant.gguf_t16 import GGUF_T16_COLS  # noqa: E402

#: The MLP tensors of one block, with the axis the planner splits and the
#: resident layout the engine uses for each source quant type.
MLP_TENSORS = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")


def _mlp_tensor_names(layer: int) -> tuple[str, ...]:
    return tuple(f"blk.{int(layer)}.{name}" for name in MLP_TENSORS)


def resolve_incumbent_plan(
    info: Any,
    *,
    backend: str = "hip_gfx1100",
    environ: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve the resident-layout plan the incumbent model actually loads.

    Runs the engine's own chain - ``build_qwen35_gguf_tensor_map`` then
    ``plan_qwen35_gguf_materialization`` with the flags
    ``resolve_gguf_dense_flags`` derives from backend capabilities and the
    environment - so a layout is never guessed. The returned plan carries one
    ``Qwen35GGUFWeightSpec`` per slot with its real ``layout``, ``quant_key`` and
    ``allocation_names``.
    """

    env = os.environ if environ is None else environ
    model_map = build_qwen35_gguf_tensor_map(info)
    dense_flags = resolve_gguf_dense_flags(
        backend,
        getattr(info, "file_type_name", None),
        capability_reader=backend_package_capability,
        environ=env,
    )
    plan = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=gguf_decode_repack_enabled(None),
        **dense_flags,
    )
    context = {
        "decode_repack": bool(gguf_decode_repack_enabled(None)),
        "dense_flags": {key: value for key, value in sorted(dense_flags.items())},
        "backend": str(backend),
    }
    return plan, context


def _spec_for_slot(plan: Any, *, layer: int, slot: str) -> Any:
    """The planned spec for ``layers.<layer>.<slot>``, or a named refusal."""

    if not 0 <= int(layer) < len(plan.layer_specs):
        raise SystemExit(f"the plan has no layer {layer}")
    layer_specs = plan.layer_specs[int(layer)]
    if slot not in layer_specs:
        raise SystemExit(f"the plan has no slot {slot!r} in layer {layer}")
    return layer_specs[slot]


def _spec_identity(spec: Any) -> dict[str, Any]:
    return {
        "layout": str(spec.layout),
        "quant_key": str(spec.quant_key),
        "allocation_names": [str(name) for name in spec.allocation_names],
    }


#: The t16 repack for each resident layout, keyed by the layout string rather
#: than by source quant type: whether a tensor is repacked at all is a layout
#: decision, and the same source type can resolve to raw or t16.
T16_REPACKS_BY_LAYOUT = {
    "gguf_q4_k_t16_v1": repack_gguf_q4_k_tile16,
    "gguf_q6_k_t16_v1": repack_gguf_q6_k_tile16,
    "gguf_q6_k_t16_qmicro_planar_v1": repack_gguf_q6_k_tile16_qmicro_planar,
}


def _t16_repack_for(layout: str, quant_type: str) -> Any:
    """The repack callable for a resident layout, or None for a raw layout."""

    repack = T16_REPACKS_BY_LAYOUT.get(str(layout))
    if repack is None and str(layout).startswith("raw"):
        return None
    if repack is None:
        raise ValueError(
            f"no repack is recorded for resident layout {layout!r} "
            f"(source quant {quant_type!r}); this layout cannot be materialized "
            "rank-locally by this probe"
        )
    return repack


def _t16_repack_tiles(
    payload: Any, *, rows: int, bytes_per_row: int, quant_type: str, repack: Any
) -> Any:
    """Repack one rank-local (or full) slice into its t16 tile array.

    The repack takes a rank-3 expert byte shape, so the rank-2 payload is viewed
    as one expert.
    """

    if repack is None:
        raise ValueError(f"no t16 repack is recorded for source quant {quant_type!r}")
    expert = np.asarray(payload, dtype=np.uint8).reshape(1, int(rows), int(bytes_per_row))
    return np.asarray(repack(expert).tiles)


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
    materialization, plan_context = resolve_incumbent_plan(info)
    model_map = build_qwen35_gguf_tensor_map(info)
    # Content-bound model identity: role names, layer scopes, shapes and storage
    # types. A placeholder here would let a different file inherit this
    # artifact's evidence.
    fingerprint = build_qwen35_gguf_role_manifest(model_map).fingerprint
    manifest = build_shard_manifest(info, world_size=int(world_size), model_hash=fingerprint)
    by_name = {plan.name: plan for plan in manifest.tensors}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"manifest has no plan for {missing}")

    full_attention = full_attention_layer_count(config)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "tp2-mlp-shard-plan-probe",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(model),
        "model_hash": fingerprint,
        "layer": int(layer),
        "world_size": int(world_size),
        "hidden_size": int(config.hidden_size),
        "manifest_hash": manifest.manifest_hash(),
        "activation_dtype": activation_dtype,
        "output_dtype": output_dtype,
        "plan_context": plan_context,
        "inventory": {
            "ar_blocks": int(config.block_count),
            "full_attention_blocks": int(full_attention),
            "linear_attention_gdn_blocks": int(config.block_count) - int(full_attention),
            "excluded_blocks": [int(b) for b in (getattr(config, "ignored_block_ids", ()) or ())],
            "note": (
                "the excluded block is the MTP/NextN block and is not part of the AR cost "
                "model; the AR model is 16 full-attention + 48 GDN blocks"
            ),
        },
        "scope": (
            "host-payload prerequisite only: this artifact does not qualify device "
            "execution, and passing it does not show that any kernel consumes these "
            "bytes correctly"
        ),
        "tensors": {},
        "questions": {},
    }

    all_exact = True
    all_admissible = True
    all_repack_commutes = True
    repack_checked: list[str] = []
    tp1_bytes = 0
    shard_bytes = 0

    for name in names:
        plan = by_name[name]
        # Slot paths in the materialization plan drop the storage suffix:
        # ``blk.0.ffn_gate.weight`` plans as slot ``ffn_gate``.
        slot = name.split(".", 2)[2]
        slot = slot[: -len(".weight")] if slot.endswith(".weight") else slot
        spec = _spec_for_slot(materialization, layer=int(layer), slot=slot)
        layout = str(spec.layout)
        quant_key = str(spec.quant_key)
        tensor = reader.tensor_info(name)
        source = source_payload(reader.path, data_offset=tensor.data_offset, nbytes=tensor.nbytes)
        block_bytes = int(plan.type_size)
        bytes_per_row = int(plan.source_row_bytes)
        quant_type = str(tensor.ggml_type_name)
        repack = _t16_repack_for(layout, quant_type)

        entry: dict[str, Any] = {
            "source_shape": list(plan.source_shape),
            "source_nbytes": int(plan.source_nbytes),
            "quant_type": quant_type,
            "kind": plan.kind,
            "split_axis": plan.axis,
            "block_size": int(plan.block_size),
            "type_size": block_bytes,
            "incumbent_spec": _spec_identity(spec),
            "repack": None if repack is None else repack.__name__,
            "ranks": {},
        }

        payloads: dict[int, Any] = {}
        for shard in plan.slices:
            payload = materialize_slice(source, shard)
            payloads[shard.rank] = payload
            local_shape = tuple(int(dim) for dim in shard.local_shape)
            # A column split changes the out-feature count; a row split keeps it
            # and shrinks the bytes per row. The t16 repack's own divisibility
            # test is applied to whichever of the two actually moved.
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
            all_admissible = all_admissible and admissible["admissible"]
            entry["ranks"][str(shard.rank)] = {
                "axis_ranges": [list(r) for r in shard.axis_ranges],
                "local_shape": list(local_shape),
                "local_nbytes": int(shard.local_nbytes),
                "segments": sum(1 for _ in shard.iter_segments()),
                "layout_admissible": admissible,
            }
            shard_bytes += int(shard.local_nbytes)

        # Does the repack commute with the split? A rank that had to see the whole
        # tensor to repack its own slice would defeat a shard-local materializer,
        # and a non-aligned split would make the local repack differ from the
        # global one while still producing a correctly shaped, plausible weight.
        # A raw layout has no repack, which is recorded as not-applicable rather
        # than as a pass.
        if repack is None:
            entry["repack_commutes_with_split"] = None
        else:
            full_tiles = _t16_repack_tiles(
                source,
                rows=int(plan.source_shape[0]),
                bytes_per_row=bytes_per_row,
                quant_type=quant_type,
                repack=repack,
            )
            commutes = True
            for shard in plan.slices:
                axis_start, axis_stop = shard.axis_ranges[0]
                local = materialize_slice(source, shard)
                if plan.kind == "column":
                    # The split moves out-features, which is the tile axis.
                    local_tiles = _t16_repack_tiles(
                        local,
                        rows=int(axis_stop - axis_start),
                        bytes_per_row=bytes_per_row,
                        quant_type=quant_type,
                        repack=repack,
                    )
                    tile_start = int(axis_start) // GGUF_T16_COLS
                    tile_stop = int(axis_stop) // GGUF_T16_COLS
                    same = bool(np.array_equal(local_tiles[0], full_tiles[0, tile_start:tile_stop]))
                    where = f"out_tiles[{tile_start}:{tile_stop}]"
                else:
                    # The split moves the K axis, which is the block axis.
                    local_bytes_per_row = int(shard.local_nbytes) // max(1, int(local_shape[0]))
                    local_tiles = _t16_repack_tiles(
                        local,
                        rows=int(local_shape[0]),
                        bytes_per_row=local_bytes_per_row,
                        quant_type=quant_type,
                        repack=repack,
                    )
                    block_start = int(axis_start) // int(plan.block_size)
                    block_stop = int(axis_stop) // int(plan.block_size)
                    same = bool(
                        np.array_equal(local_tiles[0], full_tiles[0, :, block_start:block_stop])
                    )
                    where = f"blocks[{block_start}:{block_stop}]"
                entry["ranks"][str(shard.rank)]["repack_commutes_with_split"] = same
                entry["ranks"][str(shard.rank)]["repack_slice_compared"] = where
                commutes = commutes and same
            entry["repack_commutes_with_split"] = commutes
            all_repack_commutes = all_repack_commutes and commutes
            repack_checked.append(name)
            del full_tiles

        rebuilt = reconstruct_tensor(manifest, name, payloads)
        exact = bool(rebuilt.tobytes() == bytes(source))
        all_exact = all_exact and exact
        entry["round_trip_bit_exact"] = exact
        tp1_bytes += int(plan.source_nbytes)
        report["tensors"][name] = entry
        del payloads, rebuilt, source

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
        "repack_commutes_with_the_split": {
            "answer": bool(all_repack_commutes),
            "evidence": (
                "repack of each rank-local slice equals the corresponding range of the "
                "full-tensor repack, on the axis the split actually moves"
            ),
            "tensors_checked": repack_checked,
            "not_applicable": [name for name in names if name not in repack_checked],
        },
        "layouts_resolved_from_the_incumbent_planner": {
            "answer": True,
            "evidence": (
                "each tensor's layout, quant_key and allocations come from "
                "plan_qwen35_gguf_materialization under the capability and environment "
                "flags a real load resolves, not from a hardcoded table"
            ),
            "tensors": {
                name: report["tensors"][name]["incumbent_spec"] for name in names
            },
        },
    }
    down_plan = by_name[f"blk.{int(layer)}.ffn_down.weight"]
    # The down projection is row-parallel, so each rank produces a partial output
    # over the full hidden size and the partials are summed once per layer.
    reduction_elements = int(down_plan.source_shape[0])
    down_spec = report["tensors"][f"blk.{int(layer)}.ffn_down.weight"]["incumbent_spec"]
    report["reduction_point"] = {
        "kind": "all_reduce",
        "per_layer_count": 1,
        "elements": reduction_elements,
        "options": [
            {
                "name": "fp32_partials",
                "partial_dtype": "f32",
                "reduce_dtype": "f32",
                "payload_bytes": reduction_elements * 4,
                "note": "the down projection must emit f32 for this option",
                "incumbent_down_output_dtype": output_dtype,
                "requires_output_dtype_change": output_dtype != "f32",
            },
            {
                "name": "bf16_partials_then_fp32_sum",
                "partial_dtype": "bf16",
                "reduce_dtype": "f32",
                "payload_bytes": reduction_elements * 2,
                "note": (
                    "converting bf16 partials to f32 before summing does not recover the "
                    "precision already lost at the bf16 rounding of each partial; the "
                    "conversion also costs host or device time that the payload halves"
                ),
                "incumbent_down_output_dtype": output_dtype,
                "requires_output_dtype_change": False,
            },
        ],
        "selected": None,
        "selection_reason": (
            "unresolved: the partial dtype must match the down projection's actual output "
            "dtype, and the incumbent down spec resolves to "
            f"{down_spec['quant_key']!r} with output {output_dtype!r}. Choosing the payload "
            "before resolving that would compare a bf16 exchange against an f32 arithmetic "
            "chain. Resolve in the execution test."
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
        f"reduction payload unresolved: {report['reduction_point']['elements']} elements, "
        f"{len(report['reduction_point']['options'])} dtype options",
        file=sys.stderr,
    )
    ok = all(answer["answer"] for answer in report["questions"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
