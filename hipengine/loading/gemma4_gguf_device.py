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

__all__ = [
    "LAYOUT_DENSE_F32",
    "LAYOUT_RAW_GGUF",
    "Gemma4GGUFDeviceWeight",
    "Gemma4GGUFWeightSpec",
    "materialize_fused_gguf_device_weight",
    "materialize_gemma4_gguf_device_weight",
    "plan_gemma4_gguf_resident_specs",
    "resident_bytes",
]

LAYOUT_RAW_GGUF = "raw_gguf"
LAYOUT_DENSE_F32 = "dense_f32"

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


def resident_bytes(specs: tuple[Gemma4GGUFWeightSpec, ...]) -> int:
    """Return the device bytes ``specs`` would occupy, without allocating."""

    return sum(int(spec.source.nbytes) for spec in specs)


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
    else:
        raise ValueError(f"unsupported resident layout {spec.layout!r}")

    return Gemma4GGUFDeviceWeight(
        spec=spec,
        allocations={
            "raw": load_host_array_to_device_as_dtype(
                spec.source.name,
                raw,
                dtype,
                source_dtype=source_dtype,
                device=device,
                runtime=runtime,
                allocator=allocator,
            )
        },
        backend=backend,
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

    return Gemma4GGUFDeviceWeight(
        spec=spec,
        allocations={
            "raw": load_host_array_to_device_as_dtype(
                spec.source.name,
                raw,
                DType.INT8,
                source_dtype="I8",
                device=device,
                runtime=runtime,
                allocator=allocator,
            )
        },
        backend=backend,
    )
