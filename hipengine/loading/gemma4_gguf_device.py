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

import os
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
    "LAYOUT_Q4_K_PACK8",
    "LAYOUT_RAW_GGUF",
    "Gemma4GGUFDeviceWeight",
    "Gemma4GGUFWeightSpec",
    "materialize_gemma4_gguf_device_weight",
    "pack8_arrays",
    "pack8_layout_enabled",
    "plan_gemma4_gguf_resident_specs",
    "resident_bytes",
]

LAYOUT_RAW_GGUF = "raw_gguf"
LAYOUT_DENSE_F32 = "dense_f32"
# A Q4_K tensor that also carries the pack8 GEMV layout. The packed arrays sit
# alongside the raw blocks rather than replacing them, because three of the four
# expert routes read the raw allocation and a missing one would be a KeyError
# rather than a fallback.
LAYOUT_Q4_K_PACK8 = "q4_k_pack8"

# The packed components a pack8 weight carries, in the order the kernel takes
# them. ``qweight_high`` is not among them: the kernel requires it only for q5_k
# and q6_k, and for q4_k it passes null.
_PACK8_ALLOCATION_NAMES = ("qweight", "scales", "mins")


def pack8_layout_enabled() -> bool:
    """Whether the planner gives a Q4_K expert tensor the pack8 layout.

    **Default off, with a measured cause.** On 2026-09-27 this box measured decode
    at 21.03 tok/s before the layout and 7.64 tok/s after it, on the same
    `--prompt 512 --output 128 --samples 3` run, with `pack8_selected` confirmed
    as the route that ran (928 calls). The layout is correct -- the same-weight
    parity gate in `tests/test_unit_gemma4_gguf_device.py` passes -- but it is
    slower and it costs 19.2 GB.

    The reason is a representation trade that loses on this projection. The
    packed form replaces the raw Q4_K block metadata with precomputed fp32 scale
    and min terms, which is 1.33x the raw bytes, and the projection is
    bandwidth-bound rather than metadata-bound, so decoding less metadata means
    reading more bytes. The kernel is also shaped for prefill
    (`gguf_expert_pack8_selected_prefill_kernel`), and the ladder reaches it
    exactly where rows are fewest: `grouped_prefill` takes every block with at
    least one row per expert, so a packed weight is only selected below that
    threshold, which is the decode case.

    Clearing this needs a measurement showing the packed layout ahead at some
    other row count, most plausibly a large-batch decode where `rows` exceeds the
    expert count. Set the variable, bench, and record the row against the raw
    route. Evidence:
    `benchmarks/results/2026-09-27-gemma4-gfx1151-pack8-expert-route-measured.json`.
    """

    return os.environ.get("HIPENGINE_GEMMA4_EXPERT_PACK8_LAYOUT", "0") not in {
        "",
        "0",
        "false",
        "False",
    }

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


def _pack8_shapes(shape: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the packed ``(qweight, scales)`` shapes for an expert tensor.

    Derived from the source shape alone so a caller can size the layout without
    building it, which is what :func:`resident_bytes` needs. ``scales`` and
    ``mins`` share a shape: one value per 32-value group, per expert, per output
    column.
    """

    if len(shape) != 3:
        raise ValueError(f"pack8 needs a rank-3 expert tensor, got rank {len(shape)}")
    experts, out_features, in_features = (int(dim) for dim in shape)
    from hipengine.quant.gguf_q4_k import GGUF_Q4_K_PACK, GGUF_Q4_K_SUBBLOCKS, QK_K

    if out_features % GGUF_Q4_K_PACK:
        raise ValueError(
            f"pack8 needs out_features divisible by {GGUF_Q4_K_PACK}; "
            f"{out_features} is not, so groups would straddle experts"
        )
    if in_features % QK_K:
        raise ValueError(f"Q4_K needs in_features divisible by {QK_K}; {in_features} is not")
    groups = (in_features // QK_K) * GGUF_Q4_K_SUBBLOCKS
    return (
        (experts, out_features // GGUF_Q4_K_PACK, in_features),
        (experts, groups, out_features),
    )


def _pack8_nbytes(shape: tuple[int, ...]) -> int:
    """Device bytes the pack8 arrays for ``shape`` occupy, without building them."""

    qweight_shape, scale_shape = _pack8_shapes(shape)
    qweight_bytes = 4 * _product(qweight_shape)
    scale_bytes = 4 * _product(scale_shape)
    return qweight_bytes + 2 * scale_bytes


def _product(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


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
    # A rank-3 Q4_K expert tensor can also carry the pack8 GEMV layout, which
    # precomputes the per-32-value scale and min terms so the kernel does not
    # re-decode raw block metadata on every weight read. Only Q4_K: the pack8
    # expert GEMV family registers q4_k, q5_k and q6_k, and this artifact's
    # expert tensors are Q4_K.
    #
    # Off by default because it measured slower -- see ``pack8_layout_enabled``
    # for the number, the reason, and what would clear it.
    if (
        len(source.shape) == 3
        and qtype is GGMLQuantizationType.Q4_K
        and pack8_layout_enabled()
    ):
        return Gemma4GGUFWeightSpec(
            slot_path=slot_path,
            source=source,
            quant_key="gguf_q4_k",
            layout=LAYOUT_Q4_K_PACK8,
            # The raw copy stays alongside the packed arrays. Three of the four
            # expert routes read it, so dropping it would turn them into a
            # missing-key error rather than a working fallback.
            allocation_names=("raw",) + _PACK8_ALLOCATION_NAMES,
        )
    # One allocation holds the whole tensor, including a stacked expert tensor.
    # A rank-3 expert tensor is *not* split per expert: the per-expert gather is
    # the kernel's job, and the dispatch selects the expert by index. Splitting
    # here would multiply the allocation count by 128 for no benefit.
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
    """Return the device bytes ``specs`` would occupy, without allocating.

    A pack8 weight is larger than its stored bytes, because the packed arrays sit
    alongside the raw blocks rather than replacing them. Planning the total from
    the artifact size alone would under-report that allocation by the expansion.
    """

    total = 0
    for spec in specs:
        total += int(spec.source.nbytes)
        if spec.layout == LAYOUT_Q4_K_PACK8:
            total += _pack8_nbytes(tuple(int(dim) for dim in spec.source.shape))
    return total


def pack8_arrays(raw, shape: tuple[int, ...]):
    """Repack one stacked expert tensor's raw bytes into the pack8 arrays.

    ``repack_gguf_q4_k_pack8`` handles one expert and rejects a rank-3 input, so
    this loops per expert and stacks. The stacked shapes are exactly what the
    kernel indexes -- ``qweight [experts, out_features/8, in_features]`` and one
    scale and min per 32-value group.
    """

    import numpy as np

    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8

    raw = np.asarray(raw, dtype=np.uint8)
    experts = int(shape[0])
    if raw.shape[0] != experts:
        raise ValueError(
            f"raw storage has {raw.shape[0]} experts but the shape declares {experts}"
        )
    packed = [repack_gguf_q4_k_pack8(raw[expert]) for expert in range(experts)]
    return (
        np.ascontiguousarray(np.stack([item.qweight for item in packed])),
        np.ascontiguousarray(np.stack([item.scales for item in packed])),
        np.ascontiguousarray(np.stack([item.mins for item in packed])),
    )


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
    elif spec.layout == LAYOUT_Q4_K_PACK8:
        return _materialize_pack8(
            reader,
            spec,
            raw,
            device=device,
            runtime=runtime,
            backend=backend,
            allocator=allocator,
        )
    else:
        raise ValueError(f"unsupported resident layout {spec.layout!r}")

    allocation = load_host_array_to_device_as_dtype(
        spec.source.name,
        raw,
        dtype,
        source_dtype=source_dtype,
        device=device,
        runtime=runtime,
        allocator=allocator,
    )
    allocations["raw"] = allocation
    return Gemma4GGUFDeviceWeight(
        spec=spec,
        allocations=allocations,
        backend=backend,
    )


def _materialize_pack8(
    reader: GGUFReader,
    spec: Gemma4GGUFWeightSpec,
    raw,
    *,
    device=None,
    runtime=None,
    backend: str = "hip_gfx1100",
    allocator=None,
) -> Gemma4GGUFDeviceWeight:
    """Upload a Q4_K expert tensor's raw blocks and its pack8 arrays.

    The raw blocks go up first and stay: the grouped-prefill, grouped-row4 and
    per-expert-offset routes all read the ``raw`` allocation, and the pack8 route
    is one more option in the ladder rather than a replacement for the ladder.

    If a packed upload fails partway, the allocations already made are released
    before the error propagates, so a failed load does not leak device memory.
    """

    shape = tuple(int(dim) for dim in spec.source.shape)
    expected = _pack8_shapes(shape)
    qweight, scales, mins = pack8_arrays(raw, shape)
    actual = (tuple(qweight.shape), tuple(scales.shape))
    if actual != expected:
        raise ValueError(
            f"{spec.slot_path}: pack8 produced shapes {actual} but the plan sized {expected}"
        )

    allocations: dict[str, DeviceTensorAllocation] = {}
    try:
        allocations["raw"] = load_host_array_to_device_as_dtype(
            spec.source.name,
            raw,
            DType.INT8,
            source_dtype="I8",
            device=device,
            runtime=runtime,
            allocator=allocator,
        )
        for name, array, dtype in (
            ("qweight", qweight, DType.INT32),
            ("scales", scales, DType.FP32),
            ("mins", mins, DType.FP32),
        ):
            allocations[name] = load_host_array_to_device_as_dtype(
                f"{spec.source.name}.{name}",
                array,
                dtype,
                device=device,
                runtime=runtime,
                allocator=allocator,
            )
    except Exception:
        for allocation in reversed(tuple(allocations.values())):
            allocation.free(runtime=runtime)
        raise
    return Gemma4GGUFDeviceWeight(
        spec=spec,
        allocations=allocations,
        backend=backend,
    )
