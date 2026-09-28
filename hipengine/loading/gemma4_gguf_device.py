"""Raw GGUF device residency for the Gemma 4 text tower.

The reference materializer in :mod:`hipengine.loading.gemma4_gguf_materialize`
dequantizes every tensor to a NumPy array so the CPU reference can run. That is
the right shape for validation and the wrong shape for serving: the artifact is
25.23 B parameters, so an f32 copy is 100.9 GB and even a bf16 copy is 50.5 GB
against a 48 GB device.

This module keeps the blocks as they are stored. Each tensor becomes one
``raw`` device allocation holding its GGUF bytes unmodified, described by a
:class:`Gemma4GGUFWeightSpec` that names its quant key and layout. The
quantized linear dispatch in :mod:`hipengine.runtime.gguf_linear` reads those
blocks directly.

No repack step. ``launch_gguf_linear`` serves the ``raw`` layout for every quant
type this artifact uses (Q4_K, Q5_1, Q5_K, Q8_0), at both decode and prefill
shapes, so resident-layout conversion is a later performance question rather
than a prerequisite for running.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from hipengine.core.dtype import DType
from hipengine.loading.gguf import GGUFReader, GGUFTensorInfo, MissingGGUFTensorError
from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    load_host_array_to_device_as_dtype,
)
from hipengine.quant.gguf import GGMLQuantizationType, quant_layout

__all__ = [
    "LAYOUT_DENSE_F32",
    "LAYOUT_RAW_GGUF",
    "Gemma4GGUFDeviceWeight",
    "Gemma4GGUFWeightSpec",
    "materialize_gemma4_gguf_device_weight",
    "plan_gemma4_gguf_resident_specs",
    "resident_bytes",
]

LAYOUT_RAW_GGUF = "raw_gguf"
LAYOUT_DENSE_F32 = "dense_f32"
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


@dataclass(frozen=True)
class Gemma4GGUFDeviceWeight:
    """Owned device allocations for one logical GGUF weight."""

    spec: Gemma4GGUFWeightSpec
    allocations: Mapping[str, DeviceTensorAllocation]
    backend: str

    def allocation(self, name: str = "raw") -> DeviceTensorAllocation:
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
        expert's rows are contiguous, so the stride is the tensor's byte count
        divided by the expert count -- which is the right expression for a
        quantized tensor, where a row is a whole number of 256-element blocks
        rather than ``in_features * itemsize``.
        """

        shape = self.spec.source.shape
        if len(shape) != 3:
            raise ValueError(
                f"{self.spec.slot_path} is rank {len(shape)}, not a stacked expert tensor"
            )
        experts = int(shape[0])
        nbytes = int(self.spec.source.nbytes)
        if experts <= 0 or nbytes % experts:
            raise ValueError(
                f"{self.spec.slot_path}: {nbytes} bytes does not divide evenly across "
                f"{experts} experts"
            )
        return nbytes // experts

    def free(self, *, runtime=None) -> None:
        for allocation in reversed(tuple(self.allocations.values())):
            allocation.free(runtime=runtime)





def _plan_one(slot_path: str, source: GGUFTensorInfo) -> Gemma4GGUFWeightSpec:
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
    if quant_key == "gguf_q4_k" and len(source.shape) == 3 and source.shape[1] % 32 == 0:
        # A stacked Q4_K expert tensor also carries its Q4T16 tiles, because the
        # expert gate/up prefill route reads that layout and it is 2.78 percent
        # larger than the raw blocks rather than a second copy of the tensor.
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
    ):
        # A dense Q8_0 leaf is stored as Q8T16 tiles and *only* as tiles.
        #
        # The reason is a capability gap, not a preference: the raw gguf_q8_0
        # key's one WMMA prefill is iu8_wmma_prefill_f32_f32_out, i.e. f32 ->
        # f32 output, and these are hidden projections with bf16 output. The T16
        # key carries the whole wmma_prefill_* family including the admitted
        # two-wave and four-wave schedules
        # (worklog/entries/20260929T153000). At rows=512 the four-wave schedule
        # measures 12.2-15.5 TFLOP/s against this route's ~8.5, a 1.4-1.8x
        # lever on the largest single prefill term (20260929T163000).
        #
        # Tiles replace raw rather than accompanying it, unlike the Q4_K expert
        # precedent above. Q4_K keeps both because its expert routes read either
        # layout; nothing here reads the raw blocks once the T16 route is
        # selected, and the repack is a pure permutation -- 16 columns * 34
        # bytes per tile row equals 16 raw Q8_0 blocks of 34 bytes -- so
        # tiles-only costs exactly 1.0x what raw cost (e4583bb8c).
        #
        # The layout is what selects the launch ABI, not the quant key:
        # resolve_gguf_linear_dispatch passes spec.layout as the first argument
        # of resolve_linear_consumer_contract, so a tiles-only spec that kept
        # LAYOUT_RAW_GGUF matched the raw rows, got abi "raw", and crashed in
        # _launch_wmma_raw asking for an allocation it correctly does not have
        # (20260929T183000).
        #
        # Rank-2 plus the `layers.` prefix is what makes this a dense leaf: the
        # rank-3 expert stacks are Q4_K and take the branch above, and the
        # embedding and lm head are outside the prefix and stay raw because they
        # are GEMV-shaped.
        return Gemma4GGUFWeightSpec(
            slot_path=slot_path,
            source=source,
            quant_key="gguf_q8_0_t16_v1",
            layout=LAYOUT_GGUF_Q8_0_T16,
            allocation_names=("tiles",),
        )
    return Gemma4GGUFWeightSpec(
        slot_path=slot_path,
        source=source,
        quant_key=quant_key,
        layout=LAYOUT_RAW_GGUF,
        allocation_names=allocation_names,
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
        # A dense Q8_0 leaf's Q8T16 slab: [out // 16, blocks_per_row, 544].
        # This is exactly the source byte count -- 16 columns * 34 bytes per
        # tile row equals 16 raw Q8_0 blocks of 34 bytes -- which is why the
        # repack is memory-neutral rather than costing a second tensor.
        if spec.layout != LAYOUT_GGUF_Q8_0_T16 or len(spec.source.shape) != 2:
            raise ValueError(
                f"{spec.slot_path}: a tiles allocation needs a rank-2 Q8T16 tensor, "
                f"not {spec.layout} rank {len(spec.source.shape)}"
            )
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



def materialize_gemma4_gguf_device_weight(
    reader: GGUFReader,
    spec: Gemma4GGUFWeightSpec,
    *,
    device=None,
    runtime=None,
    backend: str = "hip_gfx1100",
    allocator=None,
) -> Gemma4GGUFDeviceWeight:
    """Upload one planned weight's blocks to device, unmodified."""

    raw = reader.tensor_data(spec.source.name)
    allocations: dict[str, DeviceTensorAllocation] = {}
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
    for name in spec.allocation_names:
        if name == "raw":
            continue
        import numpy as np

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
            tiles = repack_gguf_q4_k_tile16(stacked[:, start : start + half_rows, :]).tiles
        allocations[name] = load_host_array_to_device_as_dtype(
            f"{spec.source.name}.{name}",
            tiles,
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

