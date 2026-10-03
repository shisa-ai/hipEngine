"""Raw GGUF device residency for the Gemma 4 text tower.

The reference materializer in :mod:`hipengine.loading.gemma4_gguf_materialize`
dequantizes every tensor to a NumPy array so the CPU reference can run. That is
the right shape for validation and the wrong shape for serving: the artifact is
25.23 B parameters, so an f32 copy is 100.9 GB and even a bf16 copy is 50.5 GB
against a 48 GB device.

This module keeps the blocks as they are stored. Each tensor becomes one
``raw`` device allocation holding its GGUF bytes unmodified, described by a
:class:`Gemma4GGUFWeightSpec` that names its quant key, layout, and allocation
names. The quantized linear dispatch in :mod:`hipengine.runtime.gguf_linear`
reads those blocks directly.

Two tensor families also carry a derived allocation:

* A rank-3 Q4_K expert stack keeps its ``raw`` blocks and adds the Q4T16
  ``t16_gate`` / ``t16_up`` tiles the expert gate/up prefill route reads. The
  two halves are repacked separately because the T16 leaf takes two independent
  tile pointers with no expert stride. The tiles are 1.02778x the raw bytes, so
  the tensor holds 2.02778x its file size.
* A rank-2 dense Q8_0 projection leaf is stored as Q8T16 ``tiles`` and *only*
  as tiles: the raw ``gguf_q8_0`` key's only WMMA prefill is f32 -> f32 output
  and these are bf16 hidden projections, while the ``gguf_q8_0_t16_v1`` key
  carries the admitted two-wave and four-wave schedules. The repack is a pure
  permutation, so tiles-only costs exactly what raw cost.

A third allocation family is planned only when the grouped int8 MMQ prefill
leaf is the selected route (``split_gate_up``): the fused ``gate | up`` stack
becomes two expert-strided ``gate`` / ``up`` tensors beside the ``raw`` copy,
because that leaf derives an expert's stride from its own output width and
cannot read a fused stack.

Everything else (Q5_1 / Q5_K / Q8_0 stacked expert tensors, projections in
other quants, norms) keeps raw residency alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from hipengine.core.dtype import DType
from hipengine.loading.gguf import GGUFReader, GGUFTensorInfo, MissingGGUFTensorError
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    load_host_array_to_device_as_dtype,
)
from hipengine.quant.gguf import GGMLQuantizationType, quant_layout
from hipengine.quant.gguf_repack import Q4_K_T16_SHAPE, Q5_K_T16_SHAPE, Q8_0_T16_SHAPE
from hipengine.quant.gguf_t16 import (
    GGUF_Q5_K_BLOCK_BYTES,
    GGUF_Q5_K_T16_BLOCK_BYTES,
    GGUF_T16_COLS,
)

__all__ = [
    "LAYOUT_DENSE_F32",
    "LAYOUT_GGUF_Q4_K_T16",
    "LAYOUT_GGUF_Q8_0_T16",
    "LAYOUT_RAW_GGUF",
    "Gemma4GGUFDeviceWeight",
    "Gemma4GGUFWeightSpec",
    "materialize_fused_gguf_device_weight",
    "can_fuse_gguf_device_weights",
    "materialize_gemma4_gguf_device_weight",
    "plan_gemma4_gguf_resident_specs",
    "resident_bytes",
]

LAYOUT_RAW_GGUF = "raw_gguf"
LAYOUT_DENSE_F32 = "dense_f32"
# The T16 tiles resident layout for Q4_K expert stacks. The string is the
# registry quant key it dispatches under, by design: the four-axis key carries
# the layout, so no dispatch site branches on it.
LAYOUT_GGUF_Q4_K_T16 = "gguf_q4_k_t16_v1"
# A dense Q8_0 leaf stored as Q8T16 tiles. The value matches the qwen35
# materializer's constant of the same name so the shared consumer-surface route
# table resolves the same rows for both loaders.
LAYOUT_GGUF_Q8_0_T16 = "gguf_q8_0_t16_v1"

# The quant types this artifact is built from, plus the f32 norms. This is the
# set the *loader* can carry, not an admission list for a model: a Gemma 4 GGUF
# whose tensors are all in this set loads, and one with a type outside it fails
# with the type named. Widening it is a matter of adding the dispatch the
# quantized linear layer already needs for that type.
_RAW_BLOCK_TYPES = frozenset(
    {
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q5_1,
        GGMLQuantizationType.Q6_K,
        GGMLQuantizationType.Q8_0,
    }
)


@dataclass(frozen=True)
class Gemma4GGUFWeightSpec:
    """One planned resident GGUF weight record."""

    slot_path: str
    source: GGUFTensorInfo
    quant_key: str
    layout: str
    allocation_names: tuple[str, ...] = ("raw",)
    # Registry quant key for the tiles side copy's decode owner, set only
    # where a raw-plus-tiles dual plans one (Q5_K gate_up stacks). The
    # decode rewrite resolves this key when -- and only when -- the weight
    # carries the tiles allocation; ``None`` means raw keeps the owner it
    # has. The Q8_0 dense dual predates the field and derives its t16 key
    # in the linear dispatch itself.
    tiles_quant_key: str | None = None
    # The split ``gate``/``up`` views of a fused rank-3 expert stack, built only
    # when the grouped int8 MMQ leaf is the selected route. ``raw`` stays
    # resident either way.
    split_gate_up: bool = False


@dataclass(frozen=True)
class Gemma4GGUFDeviceWeight:
    """Owned device allocations for one logical GGUF weight."""

    spec: Gemma4GGUFWeightSpec
    allocations: Mapping[str, DeviceTensorAllocation]
    backend: str

    def allocation(self, name: str | None = None) -> DeviceTensorAllocation:
        """The resident allocation ``name``, or the spec's primary one.

        The primary is ``spec.allocation_names[0]``: ``raw`` for the layouts
        that hold GGUF bytes as stored, ``tiles`` for the T16 conversion. Call
        sites that read weights through dispatch want the primary; callers
        that address a known layout by name keep passing it.
        """

        if name is None:
            name = self.spec.allocation_names[0]
        return self.allocations[name]

    def has_allocation(self, name: str) -> bool:
        return name in self.allocations

    @property
    def nbytes(self) -> int:
        return sum(int(allocation.buffer.nbytes) for allocation in self.allocations.values())

    @property
    def expert_stride_bytes(self) -> int:
        """Byte stride between consecutive experts in a stacked expert tensor.

        A rank-3 ``(num_experts, out, in)`` expert tensor is one contiguous
        allocation, and the caller selects an expert by offsetting into it. Each
        expert's rows are contiguous, so the stride is the allocation's byte
        count divided by the expert count -- the right expression for any
        resident layout: for raw GGUF bytes a row is a whole number of
        256-element blocks rather than ``in_features * itemsize``, and for T16
        tiles the allocation is exactly the tiled stack, so one expression
        serves both and half-offsets stay on the layout's own boundaries.
        """

        shape = self.spec.source.shape
        if len(shape) != 3:
            raise ValueError(
                f"{self.spec.slot_path} is rank {len(shape)}, not a stacked expert tensor"
            )
        experts = int(shape[0])
        nbytes = int(self.allocations[self.spec.allocation_names[0]].buffer.nbytes)
        if experts <= 0 or nbytes % experts:
            raise ValueError(
                f"{self.spec.slot_path}: {nbytes} bytes does not divide evenly across "
                f"{experts} experts"
            )
        return nbytes // experts

    def free(self, *, runtime=None) -> None:
        for allocation in reversed(tuple(self.allocations.values())):
            allocation.free(runtime=runtime)


def _plan_one(
    slot_path: str,
    source: GGUFTensorInfo,
    *,
    split_gate_up: bool = False,
) -> Gemma4GGUFWeightSpec:
    qtype = GGMLQuantizationType(int(source.ggml_type))
    if qtype is GGMLQuantizationType.F32:
        return Gemma4GGUFWeightSpec(
            slot_path=slot_path,
            source=source,
            quant_key="f32",
            layout=LAYOUT_DENSE_F32,
        )
    if qtype not in _RAW_BLOCK_TYPES:
        raise ValueError(
            f"{slot_path}: {source.name} has ggml type {qtype.name}, which this "
            f"loader has no resident layout for; supported block types are "
            f"{sorted(t.name for t in _RAW_BLOCK_TYPES)}"
        )
    if len(source.shape) not in (2, 3):
        raise ValueError(
            f"{slot_path}: {source.name} is rank {len(source.shape)}; raw GGUF "
            "residency needs rank-2 or rank-3 block storage"
        )
    # One allocation holds the whole tensor, including a stacked expert tensor.
    # A rank-3 expert tensor is *not* split per expert: the per-expert gather is
    # the kernel's job, and the dispatch selects the expert by index. Splitting
    # here would multiply the allocation count by 128 for no benefit.
    quant_key = f"gguf_{source.ggml_type_name.lower()}"
    allocation_names = ("raw",)
    # Keep the local tiles-only fused stack on the default path. Duplicating
    # every Q4_K expert tensor as raw plus split T16 slabs would lose the
    # request capacity preserved by this branch. Explicit WMMA pins retain the
    # split layout their registered leaf consumes.
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import _prefill_mode

    if (
        qtype is GGMLQuantizationType.Q4_K
        and slot_path.endswith(".ffn_gate_up_exps")
        and len(source.shape) == 3
        and _prefill_mode() not in {"wmma", "wmma_plain"}
    ):
        try:
            Q4_K_T16_SHAPE.validate(source.byte_shape)
        except ValueError:
            pass
        else:
            return Gemma4GGUFWeightSpec(
                slot_path=slot_path, source=source,
                quant_key=LAYOUT_GGUF_Q4_K_T16,
                layout=LAYOUT_GGUF_Q4_K_T16,
                allocation_names=("tiles",),
            )
    if quant_key == "gguf_q4_k" and len(source.shape) == 3 and source.shape[1] % 32 == 0:
        # A stacked Q4_K expert tensor also carries its Q4T16 tiles, because the
        # expert gate/up prefill route reads that layout. The raw blocks stay, so
        # the tensor holds both layouts at once: a tile is 2368 bytes where the 16
        # Q4_K blocks it is built from are 2304, which makes the tiles 1.02778
        # times the raw bytes and the tensor 2.02778 times raw.
        # Rank-3 is what makes a tensor an expert stack; a dense Q4_K matrix has
        # no route that reads tiles, so it does not pay for them.
        #
        # The gate/up tensor is stored gate-rows-first *per expert*, and the T16
        # leaf takes two independent tile pointers with no expert stride, so the
        # two halves are repacked separately rather than as one fused slab.
        #
        # The down projection has no route that reads tiles, and cannot have one
        # in this layout: a Q4T16 tile is one 16-column tile per 256-element K
        # superblock, so it needs ``in_features % 256 == 0``, while the expert
        # down projection reduces a 704-wide intermediate -- 704 is 5 x 128 + 64.
        # No K-family quant can store a 704-wide row at all, so this branch never
        # fires for a down tensor; a half-split of its 2816 output rows would be
        # an arbitrary cut rather than a gate/up one. The branch is also
        # unreachable for Gemma 4 26B-A4B UD-Q4_K_XL for a second reason: its
        # expert down projections are Q5_1 (29 layers) and Q8_0 (1 layer), so
        # they take the ``("raw",)`` default above.
        allocation_names = ("raw", "t16_gate", "t16_up")
    elif (
        quant_key == "gguf_q8_0"
        and len(source.shape) == 2
        and slot_path.startswith("layers.")
        and slot_path.rsplit(".", 1)[-1]
        in {
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        }
    ):
        try:
            Q8_0_T16_SHAPE.validate(source.byte_shape)
        except ValueError:
            pass
        else:
            # A dense Q8_0 projection keeps its raw blocks and ships a
            # byte-neutral Q8T16 ``tiles`` copy beside them.
            #
            # Raw stays primary because the fused ``gate | up`` and ``q | k |
            # v`` residency concatenates the members' stored rows
            # (:func:`materialize_fused_gguf_device_weight`), which a
            # tiles-only spec could not supply. The tiles are the capability
            # signal the rows==1 decode rewrite keys on
            # (:func:`hipengine.runtime.gguf_linear._q8_t16_tiles_decode_dispatch`),
            # which measured 1.25x-2.09x the legacy pack8 decoder with 0/844800
            # association-bit divergence against a 9e-5..1.2e-4 control rate.
            #
            # The repack is a pure permutation -- 16 columns * 34 bytes per
            # tile row equals 16 raw Q8_0 blocks of 34 bytes -- so the side
            # copy costs exactly what the raw copy cost and the resident total
            # grows by one copy of the Q8_0 projections, which the fixture's
            # capacity headroom absorbs.
            #
            # The slot list is what makes this a dense projection: the rank-3
            # expert stacks are Q4_K and take the branch above, and the
            # embedding and lm head are outside ``layers.`` and stay raw
            # because they are GEMV-shaped. The shape contract is
            # ``Q8_0_T16_SHAPE``'s, so a tensor the tile-16 repack would reject
            # keeps raw residency alone -- the same fail-open gate the Q4_K
            # stacks use above.
            return Gemma4GGUFWeightSpec(
                slot_path=slot_path,
                source=source,
                quant_key=quant_key,
                layout=LAYOUT_RAW_GGUF,
                allocation_names=("raw", "tiles"),
            )
    elif (
        quant_key == "gguf_q5_k"
        and slot_path.endswith(".ffn_gate_up_exps")
        and len(source.shape) == 3
    ):
        try:
            Q5_K_T16_SHAPE.validate(source.byte_shape)
        except ValueError:
            pass
        else:
            # A Q5_K gate_up expert stack keeps raw residency -- the iu8 and
            # grouped prefill owners read those blocks -- and adds a Q5T16
            # ``tiles`` side copy whose decode rewrite is bit-exact with the raw
            # incumbent and 4.35x faster at the production geometry (0.1830 ->
            # 0.0421 ms at rows 8). ``tiles_quant_key`` names the registry key
            # the decode rewrite resolves under; raw stays primary so prefill
            # resolution is unchanged.
            #
            # Unlike Q8T16, Q5T16 is not byte-neutral: 16 raw rows of 176-byte
            # Q5_K blocks (2816 bytes) repack to one 2880-byte tile block, so
            # the side copy is ~2.3% larger than the raw rows it holds.
            return Gemma4GGUFWeightSpec(
                slot_path=slot_path,
                source=source,
                quant_key=quant_key,
                layout=LAYOUT_RAW_GGUF,
                allocation_names=("raw", "tiles"),
                tiles_quant_key="gguf_q5_k_t16_v1",
            )
    #
    # ``split_gate_up`` is the one exception, and it is not a per-expert split:
    # the fused gate|up stack becomes two expert-strided tensors instead of one,
    # because the grouped int8 MMQ leaf derives an expert's stride from its own
    # output width and so cannot read a fused stack. That is 2 allocations, not
    # 128, and the expert gather stays the kernel's job in both.
    if split_gate_up:
        # The split is additive: ``raw`` stays resident so every other prefill
        # route keeps reading the allocation it always read, and gate/up are the
        # two expert-strided halves that leaf strides. Declaring them here is
        # what prices them in ``resident_bytes`` and what
        # :func:`materialize_gemma4_gguf_device_weight` builds them from.
        allocation_names = allocation_names + ("gate", "up")
    return Gemma4GGUFWeightSpec(
        slot_path=slot_path,
        source=source,
        quant_key=quant_key,
        layout=LAYOUT_RAW_GGUF,
        allocation_names=allocation_names,
        split_gate_up=split_gate_up,
    )


def plan_gguf_weight_spec(slot_path: str, source: GGUFTensorInfo) -> Gemma4GGUFWeightSpec:
    """Plan one resident spec for a GGUF tensor, raw blocks or dense F32.

    Public because the assistant head's device loader plans its tensors through
    the same function the backbone's planner uses, and the two must not disagree
    about how a GGUF block type becomes a resident layout. The block types this
    accepts are the ones this loader carries, so a head whose linear weights are
    a type outside that set fails at plan time with the slot named rather than
    at first launch.
    """

    return _plan_one(slot_path, source)


def plan_gemma4_gguf_resident_specs(
    reader: GGUFReader,
    *,
    model_map=None,
) -> tuple[Gemma4GGUFWeightSpec, ...]:
    """Plan a resident spec for every tensor in the text tower.

    Walks the same slot mapping the reference materializer walks, so the two
    cannot disagree about which tensors a layer owns. Planning is separate from
    materializing so a caller can compute the residency total before allocating
    anything.

    Every slot the reference materializer would read is required here too: a
    missing tensor is an error naming the slot, not a weight that quietly does
    not load.
    """

    from hipengine.loading.gemma4_gguf import build_gemma4_gguf_tensor_map
    from hipengine.loading.gemma4_gguf_materialize import gemma4_layer_slot_names

    info = reader.info
    resolved = model_map or build_gemma4_gguf_tensor_map(info)
    config = resolved.config

    specs: list[Gemma4GGUFWeightSpec] = []
    for layer_id in range(config.block_count):
        layer_map = resolved.layer(layer_id)
        for slot in gemma4_layer_slot_names(config, layer_id):
            if not layer_map.has(slot):
                raise MissingGGUFTensorError(
                    f"Gemma 4 layer {layer_id} is missing GGUF tensor slot {slot!r}"
                )
            tensor = layer_map.tensor(slot)
            specs.append(_plan_one(f"layers.{layer_id}.{slot}", tensor))

    specs.append(_plan_one("token_embedding", resolved.root("token_embedding")))
    specs.append(_plan_one("output_norm", resolved.root("output_norm")))
    if not config.tied_embeddings:
        specs.append(_plan_one("lm_head", resolved.root("lm_head")))
    return tuple(specs)


def resident_bytes(specs: tuple[Gemma4GGUFWeightSpec, ...]) -> int:
    """Return the device bytes ``specs`` would occupy, without allocating.

    The total is the stored representation, not a dequantized one. For a spec
    whose only allocation is ``raw`` that is exactly the artifact's own byte
    count, since the tensor is stored as it came out of the file. A spec that
    declares a derived layout is charged for it as well, so the figure stays an
    allocation total rather than a file-size total.
    """

    total = 0
    for spec in specs:
        # The raw blocks are charged only when the spec keeps them. A tiles-only
        # spec replaced raw rather than adding to it, so charging source.nbytes
        # as well would report double the memory it actually uses.
        if "raw" in spec.allocation_names:
            total += int(spec.source.nbytes)
        for name in spec.allocation_names:
            if name != "raw":
                total += derived_allocation_bytes(spec, name)
    return total


def derived_allocation_bytes(spec: Gemma4GGUFWeightSpec, name: str) -> int:
    """Return the device bytes one derived allocation of ``spec`` needs."""

    if name == "tiles":
        if spec.layout == LAYOUT_GGUF_Q4_K_T16:
            from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16_tile_bytes

            return repack_gguf_q4_k_tile16_tile_bytes(spec.source.byte_shape)
        # A dense Q8_0 leaf's Q8T16 slab: [out // 16, blocks_per_row, 544].
        # This is exactly the source byte count -- 16 columns * 34 bytes per
        # tile row equals 16 raw Q8_0 blocks of 34 bytes -- which is why the
        # repack is memory-neutral rather than costing a second tensor.
        if spec.layout == LAYOUT_GGUF_Q8_0_T16:
            if len(spec.source.shape) != 2:
                raise ValueError(
                    f"{spec.slot_path}: a tiles-only Q8T16 tensor must be rank 2, "
                    f"not rank {len(spec.source.shape)}"
                )
            return int(spec.source.nbytes)
        # The side-copy form: the spec keeps its raw blocks and adds tiles
        # beside them, so this returns the tiles alone and ``resident_bytes``
        # charges the raw copy separately.
        if spec.layout != LAYOUT_RAW_GGUF or "tiles" not in spec.allocation_names:
            raise ValueError(
                f"{spec.slot_path}: a tiles allocation needs a raw-resident or "
                f"tiles-only Q8T16 tensor, not {spec.layout}"
            )
        if spec.tiles_quant_key is not None:
            # Q5T16 tiles are not byte-neutral: 16 raw rows of 176-byte Q5_K
            # blocks (2816 bytes) repack to one 2880-byte tile block, so the
            # side copy is ~2.3% larger than the raw rows it holds.
            experts, out_features, bytes_per_row = spec.source.byte_shape
            blocks = bytes_per_row // GGUF_Q5_K_BLOCK_BYTES
            return (
                experts
                * (out_features // GGUF_T16_COLS)
                * blocks
                * GGUF_Q5_K_T16_BLOCK_BYTES
            )
        # Q8T16 is the same 34-byte Q8_0 blocks per 16 rows, transposed:
        # exactly the raw byte count again, so the side copy doubles the leaf.
        return int(spec.source.nbytes)
    if name not in ("t16_gate", "t16_up"):
        raise ValueError(f"{spec.slot_path}: unknown derived allocation {name!r}")
    if spec.quant_key != "gguf_q4_k" or len(spec.source.shape) != 3:
        raise ValueError(
            f"{spec.slot_path}: a t16 allocation needs a rank-3 Q4_K expert tensor, "
            f"not {spec.quant_key} rank {len(spec.source.shape)}"
        )
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16_tile_bytes

    return repack_gguf_q4_k_tile16_tile_bytes(derived_half_shape(spec))


def derived_half_shape(spec: Gemma4GGUFWeightSpec) -> tuple[int, int, int]:
    """Return the byte shape of one half of a fused rank-3 Q4_K expert tensor.

    The third element is a row's *block bytes*, not its element count. GGUF
    reports a tensor's shape in elements -- ``ffn_gate_up_exps`` is
    ``(experts, 2 * intermediate, hidden)`` -- while the repack and its size
    formula take byte shapes and validate the row against the quant's block
    size. Reading ``shape[2]`` as a byte count left every real Q4_K expert
    stack unpriced: ``hidden`` is a multiple of 256 but not of the 144-byte Q4_K
    block, so ``resident_bytes`` raised rather than returning a total. The row
    length is derived from the tensor's own byte count instead, which is exact
    for any block type the loader carries.
    """

    experts, out_features, _ = spec.source.shape
    if int(out_features) % 2:
        raise ValueError(
            f"{spec.slot_path}: {out_features} output rows do not split into a "
            "gate half and an up half"
        )
    rows = int(experts) * int(out_features)
    nbytes = int(spec.source.nbytes)
    if rows <= 0 or nbytes <= 0 or nbytes % rows:
        raise ValueError(
            f"{spec.slot_path}: {nbytes} stored bytes do not divide into {rows} "
            "rows, so one row's block bytes cannot be derived"
        )
    return (int(experts), int(out_features) // 2, nbytes // rows)


def _mmq_split_requested() -> bool:
    """True when the resident split gate_up layout is worth its extra memory.

    The grouped int8 MMQ leaf cannot read a fused ``gate | up`` stack, so that
    route needs gate and up resident separately. Materializing the split costs a
    second copy of the largest expert tensor, so it is built only when the route
    that needs it is selected; every other route keeps reading the fused ``raw``
    allocation it has always read.

    This resolves the route through the same function the forward pass dispatches
    on, rather than re-reading the env var. Matching on the literal string
    ``"mmq"`` silently disagreed with the dispatch once ``auto`` began selecting
    the MMQ route: the forward pass ran the int8 leaf while the loader kept the
    fused layout, so the leaf read a fused ``gate | up`` stack as if it were
    split. Deriving both from one resolver is what keeps them from drifting.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        _prefill_mode,
        _prefill_route_flags,
    )

    return bool(_prefill_route_flags(_prefill_mode())[2])


def _is_fused_expert_gate_up(spec_source, *, experts: int, fused_width: int) -> bool:
    """True for the rank-3 ``(experts, 2 * intermediate, hidden)`` gate_up stack.

    Matched on shape rather than on the slot name so the raw device materializer
    and the reference materializer cannot drift apart about which tensor is the
    fused one: the reference path validates this exact shape too.
    """

    shape = tuple(int(dim) for dim in spec_source.shape)
    return shape[:2] == (experts, fused_width)


def materialize_gemma4_gguf_device_weight(
    reader: GGUFReader,
    spec: Gemma4GGUFWeightSpec,
    *,
    device=None,
    runtime=None,
    backend: str = "hip_gfx1100",
    allocator=None,
) -> Gemma4GGUFDeviceWeight:
    """Upload one planned weight's blocks to device in its resident layout.

    Raw and dense layouts upload the stored bytes unmodified; the T16 layout
    uploads the bit-lossless repack of those bytes, which is the layout's
    definition rather than a change to the tensor.
    """

    raw = reader.tensor_data(spec.source.name)
    allocations: dict[str, DeviceTensorAllocation] = {}
    uploads: dict[str, tuple] = {}
    if spec.layout == LAYOUT_RAW_GGUF:
        # ``storage_dtype`` is the byte view the kernels index; every block type
        # this loader carries stores as uint8 blocks.
        storage = quant_layout(int(spec.source.ggml_type)).storage_dtype
        if storage != "uint8_blocks":
            raise ValueError(
                f"{spec.slot_path}: {spec.source.name} has storage dtype {storage!r}; "
                "raw GGUF residency expects uint8 block storage"
            )
        dtype, source_dtype = DType.INT8, "I8"
        if "tiles" in spec.allocation_names:
            # Type-keyed like the planner's rules: Q8_0 dense projections
            # ship the byte-neutral Q8T16 copy, Q5_K gate_up stacks the
            # ~102.3% Q5T16 copy their decode rewrite reads.
            from hipengine.quant.gguf_t16 import (
                repack_gguf_q5_k_tile16,
                repack_gguf_q8_0_tile16,
            )

            repacks = {
                GGMLQuantizationType.Q8_0: repack_gguf_q8_0_tile16,
                GGMLQuantizationType.Q5_K: repack_gguf_q5_k_tile16,
            }
            repack = repacks[int(spec.source.ggml_type)]
            uploads["tiles"] = (
                repack(
                    np.frombuffer(raw, dtype=np.uint8).reshape(spec.source.byte_shape)
                ).tiles,
            )
    elif spec.layout == LAYOUT_GGUF_Q4_K_T16:
        from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

        dtype, source_dtype = DType.INT8, "I8"
        uploads["tiles"] = (
            repack_gguf_q4_k_tile16(
                np.frombuffer(raw, dtype=np.uint8).reshape(spec.source.byte_shape)
            ).tiles,
        )
    elif spec.layout == LAYOUT_DENSE_F32:
        dtype, source_dtype = DType.FP32, "F32"
    elif spec.layout == LAYOUT_GGUF_Q8_0_T16:
        # The tiles are uint8 blocks exactly like the raw form, so the storage
        # view is the same; only the arrangement differs.
        dtype, source_dtype = DType.INT8, "I8"
    else:
        raise ValueError(f"unsupported resident layout {spec.layout!r}")

    # The raw blocks are uploaded only when the spec asks for them. A
    # tiles-only spec -- the dense Q8_0 leaves -- still reads ``raw`` for the
    # repack below, but must not *retain* it: nothing reads those blocks once the
    # T16 route is selected, and keeping them would double the leaves' resident
    # memory for no benefit.
    if "raw" in spec.allocation_names:
        allocations["raw"] = load_host_array_to_device_as_dtype(
            spec.source.name,
            raw,
            dtype,
            source_dtype=source_dtype,
            device=device,
            runtime=runtime,
            allocator=allocator,
        )
    if spec.split_gate_up:
        # The split is additive: ``raw`` stays resident so every other prefill
        # route keeps reading the allocation it always read, and gate/up are
        # extra expert-strided views the grouped int8 MMQ leaf can index. The
        # gather is one-time, at load, so the forward pass pays nothing for it.
        experts = int(spec.source.shape[0])
        expert_bytes = int(spec.source.nbytes) // experts
        if expert_bytes % 2:
            raise ValueError(
                f"{spec.slot_path}: fused expert gate_up expert stride "
                f"{expert_bytes} bytes is not even"
            )
        half = expert_bytes // 2
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(experts, expert_bytes)
        for name, half_blocks in (("gate", blocks[:, :half]), ("up", blocks[:, half:])):
            allocations[name] = load_host_array_to_device_as_dtype(
                f"{spec.source.name}.{name}",
                np.ascontiguousarray(half_blocks),
                dtype,
                source_dtype=source_dtype,
                device=device,
                runtime=runtime,
                allocator=allocator,
            )
    for name in spec.allocation_names:
        # ``gate`` and ``up`` are the split halves uploaded above, and ``tiles``
        # is the side copy staged in ``uploads``; the remaining names are the
        # Q4T16 halves this loop repacks from the host bytes.
        if name in ("raw", "gate", "up", "tiles"):
            continue

        if spec.layout == LAYOUT_GGUF_Q8_0_T16:
            # A dense Q8_0 leaf: one flat tile slab, no expert or half
            # dimension, so neither ``derived_half_shape`` nor the Q4_K repack
            # below applies.
            from hipengine.quant.gguf_t16 import repack_gguf_q8_0_tile16

            tiles = repack_gguf_q8_0_tile16(np.asarray(raw)).tiles
        else:
            from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16

            experts, half_rows, _ = derived_half_shape(spec)
            first = name == "t16_gate"
            start = 0 if first else half_rows
            stacked = np.asarray(raw).reshape(experts, 2 * half_rows, -1)
            # K-major Q region, staged through LDS: the kernel copies each
            # sub-block's 512 bytes into shared memory with one coalesced
            # 128-bit load per lane and reads one weight fragment per 128-bit
            # LDS load, instead of sixteen scalar global byte loads.  Same tile
            # size, same decoded values, and only this route reads these tiles.
            tiles = repack_gguf_q4_k_tile16(
                stacked[:, start : start + half_rows, :], column_major=True
            ).tiles
        allocations[name] = load_host_array_to_device_as_dtype(
            f"{spec.source.name}.{name}",
            tiles,
            DType.INT8,
            source_dtype="I8",
            device=device,
            runtime=runtime,
            allocator=allocator,
        )
    for name, (array,) in uploads.items():
        allocations[name] = load_host_array_to_device_as_dtype(
            f"{spec.source.name}:{name}",
            array,
            DType.INT8,
            source_dtype="I8",
            device=device,
            runtime=runtime,
            allocator=allocator,
        )

    return Gemma4GGUFDeviceWeight(
        spec=spec,
        allocations=allocations,
        backend=backend,
    )
def can_fuse_gguf_device_weights(specs: Sequence[Gemma4GGUFWeightSpec]) -> bool:
    """Whether ``materialize_fused_gguf_device_weight`` would accept these specs.

    Fusion is a capability of the artifact's *storage*: the rows must be
    uint8 block storage and every weight must have the same column width, or
    concatenating along axis 0 would not describe a valid weight. A caller that
    needs to fall back to separate projections therefore has to ask first, and
    cannot do it by catching the materializer's ``ValueError`` -- that would
    also swallow a genuine loading failure and silently split weights that
    should have fused.

    This is what keeps the fused path capability-gated rather than gated on
    anything about the model's identity: an artifact the kernels can fuse
    fuses, whatever it is called, and one that cannot keeps the separate
    weights it always had.

    The predicate mirrors every precondition the materializer enforces -- one
    shared layout, quant key, quant type, and bytes-per-row, over uint8 block
    storage -- so that a ``True`` here is a guarantee the materializer will not
    raise. A spec that carries a derived allocation (the Q8T16 ``tiles`` a
    dense Q8_0 leaf resides as) is not raw-resident and so is not fusible; that
    is the layout check below rather than a property of the quant type.
    """
    if len(specs) < 2:
        return False
    head = specs[0]
    if head.layout != LAYOUT_RAW_GGUF:
        return False
    if any(
        spec.layout != head.layout
        or spec.quant_key != head.quant_key
        or int(spec.source.ggml_type) != int(head.source.ggml_type)
        for spec in specs
    ):
        return False
    widths = {int(spec.source.byte_shape[1]) for spec in specs}
    if len(widths) != 1:
        return False
    return all(
        quant_layout(int(spec.source.ggml_type)).storage_dtype == "uint8_blocks"
        for spec in specs
    )


def materialize_fused_gguf_device_weight(
    reader: GGUFReader,
    specs: Sequence[Gemma4GGUFWeightSpec],
    *,
    device=None,
    runtime=None,
    backend: str = "hip_gfx1100",
    allocator=None,
) -> Gemma4GGUFDeviceWeight:
    """Upload several same-shaped weights as one contiguous resident block.

    The dense MLP's gate and up projections both read the same input and are
    numerically separable, so one resident weight carries both and one launch
    computes both.

    Block-quantized GGUF storage is row-contiguous -- each row is a whole
    number of block bytes -- so concatenating the raw blocks along axis 0
    yields exactly the fused ``(sum(rows), bytes_per_row)`` layout. The bytes
    and their order are unchanged: this arranges existing bytes in memory and
    does no reblocking, no dequantization, and no arithmetic.

    The fused spec's ``source`` is synthesized rather than read from the file,
    because the artifact stores the members as separate tensors. Nothing
    resolves ``source.name`` back through the reader for a fused weight --
    :func:`materialize_fused_gguf_device_weight` reads each member's name
    before it is replaced -- and the quantized linear dispatch keys only on
    ``spec.layout`` and ``spec.quant_key``.
    """

    if not specs:
        raise ValueError("fused materialization requires at least one weight")

    head = specs[0]
    if head.layout != LAYOUT_RAW_GGUF:
        raise ValueError(
            f"{head.slot_path}: fused resident weights support only the raw GGUF "
            f"layout, got {head.layout!r}"
        )
    for other in specs[1:]:
        if other.layout != head.layout or other.quant_key != head.quant_key:
            raise ValueError(
                f"{other.slot_path}: fused weights must share one layout and quant "
                f"key with {head.slot_path}, got {other.layout}/{other.quant_key} "
                f"against {head.layout}/{head.quant_key}"
            )
        if (
            other.source.ggml_type != head.source.ggml_type
            or other.source.shape[1] != head.source.shape[1]
            or other.source.byte_shape[1] != head.source.byte_shape[1]
        ):
            raise ValueError(
                f"{other.slot_path}: fused weights must share a quant type, "
                f"in_features, and bytes-per-row with {head.slot_path}, got "
                f"{other.source.ggml_type_name}/shape {other.source.shape}/"
                f"byte_shape {other.source.byte_shape} against "
                f"{head.source.ggml_type_name}/shape {head.source.shape}/"
                f"byte_shape {head.source.byte_shape}"
            )

    storage = quant_layout(int(head.source.ggml_type)).storage_dtype
    if storage != "uint8_blocks":
        raise ValueError(
            f"{head.slot_path}: fused residency expects uint8 block storage, "
            f"got {storage!r}"
        )

    arrays = [reader.tensor_data(spec.source.name) for spec in specs]
    raw = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
    expected = (
        sum(int(spec.source.byte_shape[0]) for spec in specs),
        head.source.byte_shape[1],
    )
    if tuple(raw.shape) != expected:
        raise ValueError(
            f"{head.slot_path}: fused blocks landed at shape {tuple(raw.shape)}, "
            f"expected {expected}"
        )

    source = replace(
        head.source,
        name="+".join(spec.source.name for spec in specs),
        shape=(sum(spec.source.shape[0] for spec in specs), *head.source.shape[1:]),
        ggml_shape=(
            head.source.ggml_shape[0],
            sum(spec.source.ggml_shape[1] for spec in specs),
        ),
        n_elements=sum(int(spec.source.n_elements) for spec in specs),
        nbytes=sum(int(spec.source.nbytes) for spec in specs),
        byte_shape=expected,
    )
    spec = replace(
        head,
        slot_path="+".join(spec.slot_path for spec in specs),
        source=source,
    )

    allocations = {
        "raw": load_host_array_to_device_as_dtype(
            spec.source.name,
            raw,
            DType.INT8,
            source_dtype="I8",
            device=device,
            runtime=runtime,
            allocator=allocator,
        )
    }
    if "tiles" in spec.allocation_names:
        # The members planned a raw-plus-tiles side allocation, so the fused
        # buffer does too: the Q8T16 repack of the concatenated rows is the
        # concat of their tiles (the block column layout is per-row), and the
        # rows==1 rewrite reads it as one fused t16 weight.
        from hipengine.quant.gguf_t16 import repack_gguf_q8_0_tile16

        allocations["tiles"] = load_host_array_to_device_as_dtype(
            f"{spec.source.name}:tiles",
            repack_gguf_q8_0_tile16(raw).tiles,
            DType.INT8,
            source_dtype="I8",
            device=device,
            runtime=runtime,
            allocator=allocator,
        )

    return Gemma4GGUFDeviceWeight(
        spec=spec,
        allocations=allocations,
        backend=backend,
    )
