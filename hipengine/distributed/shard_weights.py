"""Runtime materialization of rank-local TP2 shard payloads.

This is the shard-payload half of the TP2 design
(``docs/QWEN38-27B-GFX1100-TP2.md`` "Weight and state ownership") moved out of
``scripts/tp2_mlp_shard_plan_probe.py`` / ``scripts/tp2_mlp_slice_e2e.py`` into
the runtime, where the model-owning TP2 loop needs it. The layouts are never
guessed: every rank's payload is the incumbent materialization plan's own slice
of the GGUF tensor, repacked into the resident t16 layout that the planner
resolved for the same slot the TP1 engine loads. Quant blocks are copied
verbatim - there is no dequantize/requantize step.

One *shard family* is one set of slots that shards together
(:class:`ShardFamily`):

* the dense MLP - ``ffn_gate`` / ``ffn_up`` are column-parallel (the
  output-feature axis is split; the nonlinearity stays local to each rank's
  paired slices) and ``ffn_down`` is row-parallel (the input-feature axis is
  split on quant-block boundaries; the rank's down GEMV writes a full-hidden
  f32 partial that the staged exchange sums);
* attention under head sharding - ``attn_q`` / ``attn_k`` / ``attn_v`` are
  split on the head-group axis and the linear-attention (GDN) tensors on the
  value-head axis, while ``attn_output`` and ``ssm_out`` are row-parallel, so
  each rank produces a hidden-size partial for the exchange to sum.

Nothing here touches a device: payloads are host numpy arrays. Upload is a
separate explicit step (:func:`upload_shard_weights`) so a caller can preflight
memory before a single byte reaches a card.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from hipengine.loading.qwen35_gguf import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
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

#: Attention slots a rank owns under head sharding, per layer type. These are
#: materialization slot names, not GGUF tensor names: ``blk.0.ssm_dt.bias`` is
#: the slot ``ssm_dt_bias``, and the layer map resolves the source name, so no
#: string surgery on the tensor name is needed.
FULL_ATTENTION_SLOTS = ("attn_q", "attn_k", "attn_v", "attn_output")
LINEAR_ATTENTION_SLOTS = (
    "attn_qkv",
    "attn_gate",
    "ssm_alpha",
    "ssm_beta",
    "ssm_conv1d",
    "ssm_a",
    "ssm_dt_bias",
    "ssm_out",
)


class MLPShardError(ValueError):
    """A requested shard plan is not materializable as asked."""


@dataclass(frozen=True)
class ShardFamily:
    """One shardable tensor family and the split contract it must satisfy.

    ``slots_by_layer_type`` maps a layer type to the slots that shard together
    in it; the ``None`` key is the default for any layer type not named. The
    roles a materialized layer carries are exactly these slot names, which is
    also what the runner's ``layer.weight(slot)`` resolves.
    """

    name: str
    slots_by_layer_type: Mapping[str | None, tuple[str, ...]]
    allowed_kinds: frozenset[str]
    allow_raw: bool = False

    def slots_for(self, layer_type: str) -> tuple[str, ...]:
        slots = self.slots_by_layer_type.get(str(layer_type))
        if slots is None:
            slots = self.slots_by_layer_type.get(None)
        if slots is None:
            raise MLPShardError(
                f"shard family {self.name!r} has no tensor set for layer type "
                f"{layer_type!r}"
            )
        return slots


#: The dense MLP: every layer, column/row split.
MLP_FAMILY = ShardFamily(
    name="mlp",
    slots_by_layer_type={None: MLP_ROLES},
    allowed_kinds=frozenset({"column", "row"}),
)

#: Attention under head sharding: head-group splits plus row-parallel outputs.
#: Raw/dense slots (``ssm_alpha``/``ssm_beta``/``ssm_conv1d``/``ssm_a``/
#: ``ssm_dt_bias`` are ``dense_f32``) are allowed because their payload is the
#: rank's slice bytes verbatim - there is nothing to repack.
ATTENTION_FAMILY = ShardFamily(
    name="attention",
    slots_by_layer_type={
        FULL_ATTENTION: FULL_ATTENTION_SLOTS,
        LINEAR_ATTENTION: LINEAR_ATTENTION_SLOTS,
    },
    allowed_kinds=frozenset({"group", "row"}),
    allow_raw=True,
)


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

    ``None`` means the rank's payload is its slice's bytes verbatim. A layout
    in the t16 family with no recorded repack is a named refusal rather than a
    silently wrong payload.
    """

    from hipengine.loading.qwen35_gguf_materialize import (
        LAYOUT_DENSE_BF16,
        LAYOUT_DENSE_F32,
        LAYOUT_RAW_GGUF,
    )
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
    from hipengine.quant.gguf_t16 import (
        repack_gguf_q5_k_tile16,
        repack_gguf_q6_k_tile16,
        repack_gguf_q6_k_tile16_qmicro_planar,
    )

    table = {
        "gguf_q4_k_t16_v1": repack_gguf_q4_k_tile16,
        "gguf_q5_k_t16_v1": repack_gguf_q5_k_tile16,
        "gguf_q6_k_t16_v1": repack_gguf_q6_k_tile16,
        "gguf_q6_k_t16_qmicro_planar_v1": repack_gguf_q6_k_tile16_qmicro_planar,
    }
    repack = table.get(str(layout))
    if repack is None and (
        str(layout).startswith("raw")
        or str(layout) in {LAYOUT_RAW_GGUF, LAYOUT_DENSE_F32, LAYOUT_DENSE_BF16}
    ):
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


def resolve_shard_context(
    model_path: str | Path,
    *,
    world_size: int,
    family: ShardFamily = MLP_FAMILY,
    backend: str = "hip_gfx1100",
    uneven_split: Any = None,
) -> tuple[Any, dict[str, Any], dict[int, Any], Any]:
    """Resolve everything one shard family's payloads are derived from, once.

    Returns ``(materialization_plan, plan_context, tensor_plans_by_layer,
    config)``. The materialization plan is the engine's own chain
    (``build_qwen35_gguf_tensor_map`` then ``plan_qwen35_gguf_materialization``
    under ``resolve_gguf_dense_flags``), so a layout is never guessed; the
    tensor plans come from the byte-preserving shard manifest bound to the
    model fingerprint. Each layer's roles are the family's slot names, resolved
    through the layer map rather than by editing the GGUF tensor name.

    Raises :class:`MLPShardError` before any allocation when the degree does not
    split the family's axes, or when a requested slot does not shard.
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
    _preflight_family_axes(
        config, family, world_size=world_size, uneven_split=uneven_split
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
        "family": family.name,
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
        layer_map = model_map.layer(layer_id)
        layer_type = str(layer_map.layer_type)
        plans: dict[str, Any] = {}
        for slot in family.slots_for(layer_type):
            source = layer_map.tensors.get(slot)
            if source is None:
                raise MLPShardError(
                    f"blk.{layer_id} ({layer_type}) has no slot {slot!r}; the "
                    f"{family.name} family cannot shard this layer"
                )
            plan = by_name.get(source.name)
            if plan is None:
                raise MLPShardError(
                    f"shard manifest has no plan for {source.name}"
                )
            if plan.replicated or plan.kind not in family.allowed_kinds:
                raise MLPShardError(
                    f"{source.name}: the shard plan is {plan.kind!r} "
                    f"(replicated={plan.replicated}); the {family.name} family "
                    f"requires one of {sorted(family.allowed_kinds)}"
                )
            for rank in range(world_size):
                shard = plan.slice_for(rank)
                if shard.local_nbytes <= 0:
                    raise MLPShardError(
                        f"{source.name}: rank {rank} owns no bytes"
                    )
            plans[slot] = plan
        tensor_plans[layer_id] = plans
    return materialization, plan_context, tensor_plans, config


def _preflight_family_axes(
    config: Any,
    family: ShardFamily,
    *,
    world_size: int,
    uneven_split: Any,
) -> None:
    """Refuse a degree that cannot split the family's axes, before allocation."""

    if uneven_split is not None:
        return
    if family is MLP_FAMILY:
        ffn = int(config.feed_forward_length)
        if ffn % world_size:
            raise MLPShardError(
                f"feed_forward_length {ffn} does not split across {world_size} ranks"
            )
        return
    axes = {
        "attention.head_count": int(config.head_count),
        "attention.head_count_kv": int(config.head_count_kv),
        "ssm.group_count": int(config.ssm_group_count),
        "ssm.inner_size": int(config.ssm_inner_size),
        "ssm.time_step_rank": int(config.ssm_time_step_rank),
    }
    for axis, value in axes.items():
        if value % world_size:
            raise MLPShardError(
                f"{axis} {value} does not split across {world_size} ranks"
            )


def resolve_mlp_shard_context(
    model_path: str | Path,
    *,
    world_size: int,
    backend: str = "hip_gfx1100",
    uneven_split: Any = None,
) -> tuple[Any, dict[str, Any], dict[int, Any], Any]:
    """The dense-MLP family's :func:`resolve_shard_context`."""

    return resolve_shard_context(
        model_path,
        world_size=world_size,
        family=MLP_FAMILY,
        backend=backend,
        uneven_split=uneven_split,
    )


def resolve_attention_shard_context(
    model_path: str | Path,
    *,
    world_size: int,
    backend: str = "hip_gfx1100",
    uneven_split: Any = None,
) -> tuple[Any, dict[str, Any], dict[int, Any], Any]:
    """The attention (head-sharding) family's :func:`resolve_shard_context`."""

    return resolve_shard_context(
        model_path,
        world_size=world_size,
        family=ATTENTION_FAMILY,
        backend=backend,
        uneven_split=uneven_split,
    )


def _rank_payload(
    reader: Any,
    materialization: Any,
    tensor_plans: Mapping[str, Any],
    *,
    layer: int,
    rank: int,
    allow_raw: bool = False,
) -> dict[str, MlpShardPayload]:
    """One rank's payloads for one layer, from the planner's own slices.

    ``tensor_plans`` maps a family's slot names to their shard plans, so the
    role a caller receives is the runner's own slot name rather than a name
    reconstructed from the GGUF tensor. ``allow_raw`` admits resident layouts
    with no t16 repack (``dense_f32`` and friends), whose rank payload is the
    slice's bytes verbatim; without it such a layout stays a named refusal,
    which is what the dense-MLP family wants.
    """

    payloads: dict[str, MlpShardPayload] = {}
    for role, plan in tensor_plans.items():
        spec = _spec_for_slot(materialization, layer=int(layer), slot=str(role))
        layout = str(spec.layout)
        quant_key = str(spec.quant_key)
        info = reader.tensor_info(plan.name)
        source = source_payload(
            reader.path, data_offset=info.data_offset, nbytes=info.nbytes
        )
        repack = _t16_repack_for_layout(layout, str(info.ggml_type_name))
        shard = plan.slice_for(int(rank))
        local = materialize_slice(source, shard)
        local_shape = tuple(int(dim) for dim in shard.local_shape)
        if repack is None:
            if not allow_raw:
                raise MLPShardError(
                    f"{plan.name}: resident layout {layout!r} is raw and cannot be "
                    "sliced rank-locally by this path"
                )
            payloads[role] = MlpShardPayload(
                role=role,
                layout=layout,
                quant_key=quant_key,
                payload=np.ascontiguousarray(local, dtype=np.uint8).reshape(-1),
                local_shape=local_shape,
            )
            continue
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
            local_shape=local_shape,
        )
    return payloads


def materialize_shards(
    model_path: str | Path,
    *,
    world_size: int,
    family: ShardFamily = MLP_FAMILY,
    layer_ids: Iterable[int] | None = None,
    backend: str = "hip_gfx1100",
    uneven_split: Any = None,
) -> dict[int, MlpShardLayer]:
    """Materialize every requested layer's rank-local payloads for one family.

    ``layer_ids=None`` means every block. The returned payloads are host bytes
    in the resident layouts the incumbent planner resolved - the same split the
    TP2-A slice validated against the device-resident TP1 weights. The
    container type keeps its historical ``Mlp`` name because it carries any
    family's roles; see ``docs/REFACTOR.md``.
    """

    materialization, _context, tensor_plans, config = resolve_shard_context(
        model_path,
        world_size=int(world_size),
        family=family,
        backend=backend,
        uneven_split=uneven_split,
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
                allow_raw=family.allow_raw,
            )
            for rank in range(int(world_size))
        }
        shards[layer_id] = MlpShardLayer(layer_id=layer_id, ranks=ranks)
    return shards


def materialize_mlp_shards(
    model_path: str | Path,
    *,
    world_size: int,
    layer_ids: Iterable[int] | None = None,
    backend: str = "hip_gfx1100",
    uneven_split: Any = None,
) -> dict[int, MlpShardLayer]:
    """The dense-MLP family's :func:`materialize_shards`."""

    return materialize_shards(
        model_path,
        world_size=world_size,
        family=MLP_FAMILY,
        layer_ids=layer_ids,
        backend=backend,
        uneven_split=uneven_split,
    )


def materialize_attention_shards(
    model_path: str | Path,
    *,
    world_size: int,
    layer_ids: Iterable[int] | None = None,
    backend: str = "hip_gfx1100",
    uneven_split: Any = None,
) -> dict[int, MlpShardLayer]:
    """The attention (head-sharding) family's :func:`materialize_shards`.

    Every layer's roles are its own layer type's slots: ``attn_q`` / ``attn_k``
    / ``attn_v`` / ``attn_output`` on a full-attention block, and the GDN set
    on a linear-attention block.
    """

    return materialize_shards(
        model_path,
        world_size=world_size,
        family=ATTENTION_FAMILY,
        layer_ids=layer_ids,
        backend=backend,
        uneven_split=uneven_split,
    )


def shard_bytes(shards: Mapping[int, MlpShardLayer], *, rank: int) -> int:
    """One rank's total resident shard bytes across all materialized layers."""

    total = 0
    for layer in shards.values():
        for payload in layer.rank_payloads(rank).values():
            total += payload.nbytes
    return total


def upload_shard_weights(
    runtime: Any,
    shards: Mapping[int, MlpShardLayer],
    *,
    devices: Iterable[int],
) -> dict[int, dict[int, dict[str, Any]]]:
    """Upload every rank's payloads as persistent resident shard weights.

    Returns ``{layer_id: {device: {role: ShardWeight}}}``. Every upload is
    device-scoped to its rank's device. The allocation name is the launcher's
    ABI operand name (``hipengine.loading.qwen35_gguf_consumer_surface``
    ``LINEAR_WEIGHT_OPERANDS``): t16 resident layouts resolve their pointer as
    ``tiles``, raw/dense as ``raw``.
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


def upload_mlp_shard_weights(
    runtime: Any,
    shards: Mapping[int, MlpShardLayer],
    *,
    devices: Iterable[int],
) -> dict[int, dict[int, dict[str, Any]]]:
    """The dense-MLP family's :func:`upload_shard_weights`."""

    return upload_shard_weights(runtime, shards, devices=devices)
