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

One tensor family converts at load: the fused Q4_K ``gate_up`` expert stack
resides as bit-lossless T16 tiles (``LAYOUT_GGUF_Q4_K_T16``), whose decode
GEMV measured 4.9-5.1x the raw-layout selected GEMV at bitwise-equal output
and whose mmq32 / WMMA prefill owners resolve under the same quant key. The
repack runs here, once per tensor, replacing the raw copy rather than
adding one -- a second resident copy of the gate_up stacks would not fit the
capacity the artifact already occupies.

Q8_0 dense projections additionally carry a *side* allocation: the raw
copy stays primary (the prefill owners keep reading it, byte for byte) and a
byte-neutral ``tiles`` copy is uploaded beside it so the rows==1 decode
launch can rewrite to the Q8T16 GEMV sibling (see
:func:`hipengine.runtime.gguf_linear._q8_t16_tiles_decode_dispatch`). Q8T16
tiles are the same 34-byte Q8_0 blocks per 16 rows, transposed -- the same
byte count as the raw copy -- and only rank-2 tensors at projection slots
plan it, so the resident total grows by exactly one copy of the Q8_0
projections (measured to fit the fixture's capacity headroom). Everything
else (Q5_1 / Q5_K / Q8_0 stacked expert tensors, projections in other
quants, norms) keeps raw residency alone.
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
from hipengine.quant.gguf_q4_k import (
    GGUF_Q4_K_BLOCK_BYTES,
    GGUF_Q4_K_TILE16_BLOCK_BYTES,
    GGUF_Q4_K_TILE16_COLS,
    repack_gguf_q4_k_tile16,
)
from hipengine.quant.gguf_repack import Q4_K_T16_SHAPE, Q8_0_T16_SHAPE

__all__ = [
    "LAYOUT_DENSE_F32",
    "LAYOUT_GGUF_Q4_K_T16",
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
    # here would multiply the allocation count by 128 for no benefit, and the
    # fused ``gate | up`` stack stays whole for the same reason -- the int8 MMQ
    # leaf reads it through an explicit expert stride rather than a second copy.
    #
    # The fused Q4_K gate_up stack is the one tensor whose owners are all
    # registered for the T16 tiles quant key (decode's selected GEMV, the mmq32
    # prefill leaf, the WMMA alias), so it resides tiled, replacing its raw
    # copy. The slot names the tensor's dispatch role -- down stacks run a
    # different owner chain with no t16 service -- and the shape contract is
    # ``Q4_K_T16_SHAPE``'s, so a stack it rejects keeps raw residency.
    if (
        qtype is GGMLQuantizationType.Q4_K
        and slot_path.endswith(".ffn_gate_up_exps")
        and len(source.shape) == 3
    ):
        try:
            Q4_K_T16_SHAPE.validate(source.byte_shape)
        except ValueError:
            pass
        else:
            return Gemma4GGUFWeightSpec(
                slot_path=slot_path,
                source=source,
                quant_key=LAYOUT_GGUF_Q4_K_T16,
                layout=LAYOUT_GGUF_Q4_K_T16,
                allocation_names=("tiles",),
            )
    # Q8_0 dense projections plan a raw-plus-tiles side allocation. The slot
    # names the tensor's dispatch role (root slots like ``token_embedding``
    # and stacked expert tensors have no rows==1 t16 owner), and the shape
    # contract is ``Q8_0_T16_SHAPE``'s, so a tensor the tile-16 repack would
    # reject keeps raw residency alone -- the same fail-open shape gate the
    # Q4_K stacks use above, just keeping both allocations instead of the
    # one.
    if (
        qtype is GGMLQuantizationType.Q8_0
        and len(source.shape) == 2
        and slot_path.startswith("layers.")
        and slot_path.rsplit(".", 1)[-1]
        in {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"}
    ):
        try:
            Q8_0_T16_SHAPE.validate(source.byte_shape)
        except ValueError:
            pass
        else:
            return Gemma4GGUFWeightSpec(
                slot_path=slot_path,
                source=source,
                quant_key=f"gguf_{source.ggml_type_name.lower()}",
                layout=LAYOUT_RAW_GGUF,
                allocation_names=("raw", "tiles"),
            )
    return Gemma4GGUFWeightSpec(
        slot_path=slot_path,
        source=source,
        quant_key=f"gguf_{source.ggml_type_name.lower()}",
        layout=LAYOUT_RAW_GGUF,
    )


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


def _planned_nbytes(spec: Gemma4GGUFWeightSpec) -> int:
    """Device bytes the spec's resident layout will occupy.

    Raw and dense layouts occupy exactly their stored bytes; the T16 tiles
    carry per-16-row scale blocks and occupy the repack's planned size, which
    is what capacity checks must sum.
    """

    if spec.layout == LAYOUT_GGUF_Q4_K_T16:
        experts, out_features, bytes_per_row = spec.source.byte_shape
        blocks = bytes_per_row // GGUF_Q4_K_BLOCK_BYTES
        return (
            experts
            * (out_features // GGUF_Q4_K_TILE16_COLS)
            * blocks
            * GGUF_Q4_K_TILE16_BLOCK_BYTES
        )
    if spec.layout == LAYOUT_RAW_GGUF:
        if "tiles" in spec.allocation_names:
            # The Q8T16 side allocation is the same 34-byte Q8_0 blocks per
            # 16 rows, transposed: exactly the raw byte count again.
            # Capacity checks must sum both copies, so the doubling is
            # explicit here rather than implied by the repack's internals.
            return 2 * int(spec.source.nbytes)
        return int(spec.source.nbytes)
    return int(spec.source.nbytes)


def resident_bytes(specs: tuple[Gemma4GGUFWeightSpec, ...]) -> int:
    """Return the device bytes ``specs`` would occupy, without allocating."""

    return sum(_planned_nbytes(spec) for spec in specs)


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
    allocation_name = "raw"
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
            from hipengine.quant.gguf_t16 import repack_gguf_q8_0_tile16

            uploads["tiles"] = (
                repack_gguf_q8_0_tile16(
                    np.frombuffer(raw, dtype=np.uint8).reshape(spec.source.byte_shape)
                ).tiles,
            )
    elif spec.layout == LAYOUT_DENSE_F32:
        dtype, source_dtype = DType.FP32, "F32"
    elif spec.layout == LAYOUT_GGUF_Q4_K_T16:
        raw = repack_gguf_q4_k_tile16(
            np.frombuffer(raw, dtype=np.uint8).reshape(spec.source.byte_shape)
        ).tiles
        allocation_name = "tiles"
        dtype, source_dtype = DType.INT8, "I8"
    else:
        raise ValueError(f"unsupported resident layout {spec.layout!r}")

    allocations = {
        allocation_name: load_host_array_to_device_as_dtype(
            spec.source.name,
            raw,
            dtype,
            source_dtype=source_dtype,
            device=device,
            runtime=runtime,
            allocator=allocator,
        )
    }
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
    """
    if len(specs) < 2:
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
