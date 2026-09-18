"""Runtime materialization of rank-local MLP shard payloads.

This is the shard-payload half of the TP2 design
(``docs/QWEN38-27B-GFX1100-TP2.md`` "Weight and state ownership") moved out of
``scripts/tp2_mlp_shard_plan_probe.py`` / ``scripts/tp2_mlp_slice_e2e.py`` into
the runtime, where the model-owning TP2 loop needs it. The layouts are never
guessed: every rank's payload is the incumbent materialization plan's own slice
of the GGUF tensor, repacked into the resident t16 layout that the planner
resolved for the same slot the TP1 engine loads. Quant blocks are copied
verbatim - there is no dequantize/requantize step.

One MLP projection is one rank's payload per role:

* ``ffn_gate`` / ``ffn_up`` are column-parallel (the output-feature axis is
  split; the nonlinearity stays local to each rank's paired slices);
* ``ffn_down`` is row-parallel (the input-feature axis is split on quant-block
  boundaries; the rank's down GEMV writes a full-hidden f32 partial that the
  staged exchange sums).

Nothing here touches a device: payloads are host numpy arrays. Upload is a
separate explicit step (:func:`upload_mlp_shard_weights`) so a caller can
preflight memory before a single byte reaches a card.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from hipengine.loading.qwen35_gguf import (
    build_qwen35_gguf_tensor_map,
    qwen35_gguf_config_from_metadata,
)
from hipengine.loading.qwen35_gguf_admission import build_qwen35_gguf_role_manifest
from hipengine.loading.qwen35_gguf_shards import (
    build_shard_manifest,
    materialize_slice,
    source_payload,
)

#: The three dense-MLP projections one block shards, in planner order. Slot
#: roles, e.g. ``ffn_gate`` from ``blk.3.ffn_gate.weight``.
MLP_TENSORS = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
MLP_ROLES = tuple(name.split(".")[0] for name in MLP_TENSORS)


class MLPShardError(ValueError):
    """A requested MLP shard plan is not materializable as asked."""


@dataclass(frozen=True)
class MlpShardPayload:
    """One rank's resident payload for one MLP role of one layer."""

    role: str
    layout: str
    quant_key: str
    payload: np.ndarray
    local_shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return int(self.payload.nbytes)


@dataclass(frozen=True)
class MlpShardLayer:
    """Every rank's payloads for one layer's MLP."""

    layer_id: int
    ranks: Mapping[int, Mapping[str, MlpShardPayload]]

    def rank_payloads(self, rank: int) -> Mapping[str, MlpShardPayload]:
        if int(rank) not in self.ranks:
            raise MLPShardError(f"layer {self.layer_id} has no shard for rank {rank}")
        return self.ranks[int(rank)]


def _t16_repack_for_layout(layout: str, quant_type: str) -> Any:
    """The repack callable for a resident layout, or ``None`` for raw layouts.

    Keyed by the layout string rather than by source quant type: whether a
    tensor is repacked at all is a layout decision, and the same source type
    can resolve to raw or t16. Mirrors the probe's table; the probe keeps its
    own copy because it must also probe layouts the runtime refuses.
    """

    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
    from hipengine.quant.gguf_t16 import (
        repack_gguf_q6_k_tile16,
        repack_gguf_q6_k_tile16_qmicro_planar,
    )

    table = {
        "gguf_q4_k_t16_v1": repack_gguf_q4_k_tile16,
        "gguf_q6_k_t16_v1": repack_gguf_q6_k_tile16,
        "gguf_q6_k_t16_qmicro_planar_v1": repack_gguf_q6_k_tile16_qmicro_planar,
    }
    repack = table.get(str(layout))
    if repack is None and str(layout).startswith("raw"):
        return None
    if repack is None:
        raise MLPShardError(
            f"no t16 repack is recorded for resident layout {layout!r} "
            f"(source quant {quant_type!r}); this layout cannot be materialized "
            "rank-locally"
        )
    return repack


def _spec_for_slot(materialization: Any, *, layer: int, slot: str) -> Any:
    """The planned spec for ``layers.<layer>.<slot>``, or a named refusal."""

    if not 0 <= int(layer) < len(materialization.layer_specs):
        raise MLPShardError(f"the plan has no layer {layer}")
    layer_specs = materialization.layer_specs[int(layer)]
    if slot not in layer_specs:
        raise MLPShardError(f"the plan has no slot {slot!r} in layer {layer}")
    return layer_specs[slot]


def resolve_mlp_shard_context(
    model_path: str | Path,
    *,
    world_size: int,
    backend: str = "hip_gfx1100",
    uneven_split: Any = None,
) -> tuple[Any, dict[str, Any], dict[int, Any], Any]:
    """Resolve everything the shard payloads are derived from, once.

    Returns ``(materialization_plan, plan_context, tensor_plans_by_layer,
    config)``. The materialization plan is the engine's own chain
    (``build_qwen35_gguf_tensor_map`` then ``plan_qwen35_gguf_materialization``
    under ``resolve_gguf_dense_flags``), so a layout is never guessed; the
    tensor plans come from the byte-preserving shard manifest bound to the
    model fingerprint. Raises :class:`MLPShardError` before any allocation when
    the degree does not split the MLP axes.
    """

    from hipengine.loading.gguf import GGUFReader, scan_gguf
    from hipengine.loading.qwen35_gguf_materialize import (
        gguf_decode_repack_enabled,
        plan_qwen35_gguf_materialization,
    )
    from hipengine.loading.qwen35_gguf_policy import resolve_gguf_dense_flags
    from hipengine.kernels.backends import backend_package_capability

    world_size = int(world_size)
    if world_size < 1:
        raise MLPShardError("world_size must be positive")

    reader = GGUFReader(str(model_path))
    info = scan_gguf(str(model_path))
    config = qwen35_gguf_config_from_metadata(info)
    ffn = int(config.feed_forward_length)
    if uneven_split is None and ffn % world_size:
        raise MLPShardError(
            f"feed_forward_length {ffn} does not split across {world_size} ranks"
        )

    env_gguf_decode_repack = gguf_decode_repack_enabled(None)
    model_map = build_qwen35_gguf_tensor_map(info)
    dense_flags = resolve_gguf_dense_flags(
        backend,
        getattr(info, "file_type_name", None),
        capability_reader=backend_package_capability,
        environ=os.environ,
    )
    materialization = plan_qwen35_gguf_materialization(
        model_map,
        decode_repack=env_gguf_decode_repack,
        **dense_flags,
    )
    plan_context = {
        "decode_repack": bool(env_gguf_decode_repack),
        "dense_flags": {key: value for key, value in sorted(dense_flags.items())},
        "backend": str(backend),
    }

    fingerprint = build_qwen35_gguf_role_manifest(model_map).fingerprint
    manifest = build_shard_manifest(
        info,
        world_size=world_size,
        model_hash=fingerprint,
        uneven_split=uneven_split,
    )
    by_name = {plan.name: plan for plan in manifest.tensors}

    tensor_plans: dict[int, Any] = {}
    for layer_id in range(int(config.block_count)):
        names = tuple(f"blk.{layer_id}.{name}" for name in MLP_TENSORS)
        missing = [name for name in names if name not in by_name]
        if missing:
            raise MLPShardError(f"shard manifest has no plan for {missing}")
        plans = {name.split(".", 2)[2].removesuffix(".weight"): by_name[name] for name in names}
        for role, plan in plans.items():
            if plan.replicated or plan.kind not in {"column", "row"}:
                raise MLPShardError(
                    f"blk.{layer_id}.{role}: the shard plan is {plan.kind!r}; "
                    "the MLP projections must split column/row for TP"
                )
            for rank in range(world_size):
                shard = plan.slice_for(rank)
                if shard.local_nbytes <= 0:
                    raise MLPShardError(
                        f"blk.{layer_id}.{role}: rank {rank} owns no bytes"
                    )
        tensor_plans[layer_id] = plans
    return materialization, plan_context, tensor_plans, config


def _rank_payload(
    reader: Any,
    materialization: Any,
    tensor_plans: Mapping[str, Any],
    *,
    layer: int,
    rank: int,
) -> dict[str, MlpShardPayload]:
    """One rank's payloads for one layer, from the planner's own slices."""

    payloads: dict[str, MlpShardPayload] = {}
    for role, plan in tensor_plans.items():
        slot = str(plan.name).split(".", 2)[2]
        slot = slot[: -len(".weight")] if slot.endswith(".weight") else slot
        spec = _spec_for_slot(materialization, layer=int(layer), slot=slot)
        layout = str(spec.layout)
        quant_key = str(spec.quant_key)
        info = reader.tensor_info(plan.name)
        source = source_payload(
            reader.path, data_offset=info.data_offset, nbytes=info.nbytes
        )
        repack = _t16_repack_for_layout(layout, str(info.ggml_type_name))
        if repack is None:
            raise MLPShardError(
                f"{plan.name}: resident layout {layout!r} is raw and cannot be "
                "sliced rank-locally by this path"
            )
        shard = plan.slice_for(int(rank))
        local = materialize_slice(source, shard)
        if plan.kind == "column":
            rows = int(shard.axis_stop - shard.axis_start)
            bytes_per_row = int(plan.source_row_bytes)
        else:
            rows = int(shard.local_shape[0])
            bytes_per_row = int(shard.local_nbytes) // max(1, int(rows))
        expert = np.ascontiguousarray(local, dtype=np.uint8).reshape(
            1, rows, bytes_per_row
        )
        tiles = np.ascontiguousarray(np.asarray(repack(expert).tiles))
        payloads[role] = MlpShardPayload(
            role=role,
            layout=layout,
            quant_key=quant_key,
            payload=tiles.reshape(-1),
            local_shape=tuple(int(dim) for dim in shard.local_shape),
        )
    return payloads


def materialize_mlp_shards(
    model_path: str | Path,
    *,
    world_size: int,
    layer_ids: Iterable[int] | None = None,
    backend: str = "hip_gfx1100",
    uneven_split: Any = None,
) -> dict[int, MlpShardLayer]:
    """Materialize every requested layer's rank-local MLP payloads.

    ``layer_ids=None`` means every block. The returned payloads are host bytes
    in the resident layouts the incumbent planner resolved - the same split the
    TP2-A slice validated against the device-resident TP1 weights.
    """

    materialization, _context, tensor_plans, config = resolve_mlp_shard_context(
        model_path, world_size=int(world_size), backend=backend, uneven_split=uneven_split
    )
    from hipengine.loading.gguf import GGUFReader

    reader = GGUFReader(str(model_path))
    requested = (
        range(int(config.block_count))
        if layer_ids is None
        else sorted({int(layer) for layer in layer_ids})
    )
    shards: dict[int, MlpShardLayer] = {}
    for layer_id in requested:
        if not 0 <= layer_id < int(config.block_count):
            raise MLPShardError(
                f"layer {layer_id} outside the model's {config.block_count} blocks"
            )
        ranks = {
            rank: _rank_payload(
                reader,
                materialization,
                tensor_plans[layer_id],
                layer=layer_id,
                rank=rank,
            )
            for rank in range(int(world_size))
        }
        shards[layer_id] = MlpShardLayer(layer_id=layer_id, ranks=ranks)
    return shards


def shard_bytes(shards: Mapping[int, MlpShardLayer], *, rank: int) -> int:
    """One rank's total resident shard bytes across all materialized layers."""

    total = 0
    for layer in shards.values():
        for payload in layer.rank_payloads(rank).values():
            total += payload.nbytes
    return total


def upload_mlp_shard_weights(
    runtime: Any,
    shards: Mapping[int, MlpShardLayer],
    *,
    devices: Iterable[int],
) -> dict[int, dict[int, dict[str, Any]]]:
    """Upload every rank's payloads as persistent resident shard weights.

    Returns ``{layer_id: {device: {role: ShardWeight}}}`` ready for
    :class:`hipengine.distributed.shard_group.MlpShardGroup`. Every upload is
    device-scoped to its rank's device.
    """

    from hipengine.distributed.shard_exec import upload_shard_weight

    device_list = [int(device) for device in devices]
    uploaded: dict[int, dict[int, dict[str, Any]]] = {}
    for layer_id, layer in shards.items():
        per_device: dict[int, dict[str, Any]] = {}
        for device in device_list:
            payloads = layer.rank_payloads(device)
            weights = {
                role: upload_shard_weight(
                    runtime,
                    device=device,
                    # The allocation name is the launcher's ABI operand name
                    # (hipengine.loading.qwen35_gguf_consumer_surface.
                    # LINEAR_WEIGHT_OPERANDS): t16 resident layouts resolve
                    # their pointer as 'tiles', raw/dense as 'raw'.
                    name="tiles" if "t16" in payload.layout else "raw",
                    layout=payload.layout,
                    quant_key=payload.quant_key,
                    payload=payload.payload,
                )
                for role, payload in payloads.items()
            }
            per_device[device] = weights
        uploaded[layer_id] = per_device
    return uploaded
