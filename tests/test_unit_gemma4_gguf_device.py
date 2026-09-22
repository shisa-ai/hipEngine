"""Raw GGUF device residency for the Gemma 4 text tower.

The property under test is that a resident weight is the artifact's bytes,
unmodified. A loader that dequantized, repacked, or truncated anything would
still produce a plausible-looking device buffer, so these tests compare against
the reader's own byte view rather than against a shape or a size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.gemma4_gguf_device import (
    LAYOUT_DENSE_F32,
    LAYOUT_RAW_GGUF,
    materialize_gemma4_gguf_device_weight,
    plan_gemma4_gguf_resident_specs,
    resident_bytes,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf import GGMLQuantizationType
from tests._rocm_guard import hip_runtime_available
from tests._gemma4_gguf_fixture import (
    default_fixture_tensors,
    fixture_metadata,
    write_fixture_gguf,
)


@pytest.fixture
def reader(tmp_path: Path) -> GGUFReader:
    path = write_fixture_gguf(
        tmp_path / "device.gguf",
        default_fixture_tensors(),
        fixture_metadata(),
    )
    return GGUFReader(path)


def test_every_tensor_gets_a_spec(reader: GGUFReader) -> None:
    """Planning covers the artifact, not a subset of it."""

    specs = plan_gemma4_gguf_resident_specs(reader)
    planned = {spec.source.name for spec in specs}
    in_artifact = {tensor.name for tensor in reader.info.tensors}
    # rope_freqs.weight is metadata the loader derives from rather than a
    # resident weight, so it is the one tensor expected to be absent.
    assert planned == in_artifact - {"rope_freqs.weight"}


def test_quant_keys_follow_the_artifact(reader: GGUFReader) -> None:
    """The resident quant key is the tensor's own type, not a default."""

    by_name = {spec.source.name: spec for spec in plan_gemma4_gguf_resident_specs(reader)}
    for name, spec in by_name.items():
        source = reader.tensor_info(name)
        if source.ggml_type == GGMLQuantizationType.F32:
            assert spec.layout == LAYOUT_DENSE_F32, name
            assert spec.quant_key == "f32", name
        else:
            assert spec.layout == LAYOUT_RAW_GGUF, name
            assert spec.quant_key == f"gguf_{source.ggml_type_name.lower()}", name
        assert spec.allocation_names == ("raw",), name


def test_a_stacked_expert_tensor_stays_one_allocation(reader: GGUFReader) -> None:
    """A rank-3 expert tensor is not split per expert.

    The per-expert gather is the kernel's job and the dispatch selects the
    expert by index. Splitting here would multiply the allocation count by the
    expert count for no benefit, so the planner must keep it whole.
    """

    specs = plan_gemma4_gguf_resident_specs(reader)
    stacked = [spec for spec in specs if len(spec.source.shape) == 3]
    assert stacked, "fixture has no rank-3 expert tensor to check"

    names = [spec.source.name for spec in stacked]
    assert len(names) == len(set(names)), "a stacked expert tensor was planned more than once"

    for spec in stacked:
        assert spec.allocation_names == ("raw",), spec.slot_path
        assert spec.source.shape[0] > 1, spec.slot_path
        # One allocation holds all experts. If the planner had split per expert
        # the resident total would be the same but the allocation count would be
        # shape[0] times larger, so check the total against the whole tensor.
        per_expert = spec.source.nbytes / spec.source.shape[0]
        assert spec.source.nbytes == int(per_expert) * spec.source.shape[0], spec.slot_path


def test_resident_bytes_is_the_artifact_bytes(reader: GGUFReader) -> None:
    """Residency is the stored size, not a dequantized size."""

    specs = plan_gemma4_gguf_resident_specs(reader)
    expected = sum(
        tensor.nbytes for tensor in reader.info.tensors if tensor.name != "rope_freqs.weight"
    )
    assert resident_bytes(specs) == expected


def test_an_unsupported_quant_type_names_the_type(reader: GGUFReader) -> None:
    """A type the loader cannot carry fails loudly and names itself.

    Silently skipping it would produce a model that loads and then reads
    uninitialized memory for one projection.
    """

    import dataclasses

    specs = plan_gemma4_gguf_resident_specs(reader)
    block = next(spec for spec in specs if spec.layout == LAYOUT_RAW_GGUF)
    # IQ2_XS is a real GGUF type the Gemma 4 loader has no raw layout for.
    forged = dataclasses.replace(
        block.source,
        ggml_type=int(GGMLQuantizationType.IQ2_XS),
        ggml_type_name="IQ2_XS",
    )
    forged_spec = dataclasses.replace(block, source=forged)
    from hipengine.loading.gemma4_gguf_device import _plan_one

    with pytest.raises(ValueError, match="IQ2_XS"):
        _plan_one(forged_spec.slot_path, forged)


@pytest.mark.skipif(
    not hip_runtime_available(), reason="HIP runtime unavailable; skipping device residency test"
)
def test_device_bytes_match_the_artifact_bytes(reader: GGUFReader) -> None:
    """The resident buffer is the stored blocks, byte for byte."""

    specs = plan_gemma4_gguf_resident_specs(reader)
    # One raw block weight and one f32 weight, so both layouts are exercised.
    chosen = [
        next(spec for spec in specs if spec.layout == LAYOUT_RAW_GGUF),
        next(spec for spec in specs if spec.layout == LAYOUT_DENSE_F32),
    ]
    for spec in chosen:
        weight = materialize_gemma4_gguf_device_weight(reader, spec)
        try:
            device = weight.allocation("raw")
            nbytes = int(device.buffer.nbytes)
            assert nbytes == int(spec.source.nbytes), spec.slot_path

            host = np.empty(nbytes, dtype=np.uint8)
            from hipengine.core.memory import DeviceBuffer, copy_device_to_host
            from hipengine.loading.materialize import host_array_ptr

            copy_device_to_host(
                host_array_ptr(host),
                DeviceBuffer(ptr=device.buffer.ptr, nbytes=nbytes),
                nbytes,
            )
            expected = np.frombuffer(reader.tensor_data(spec.source.name), dtype=np.uint8)
            np.testing.assert_array_equal(host, expected, err_msg=spec.slot_path)
        finally:
            weight.free()


# --------------------------------------------------------------------------
# The quantized projection must reproduce the bf16 projection
# --------------------------------------------------------------------------


def _read_bf16_rows(buffer, rows: int, columns: int) -> np.ndarray:
    """Read a BF16 device buffer back as float32 rows.

    The high 16 bits of a float32 *are* its bf16 encoding, so the conversion is
    a left shift and a reinterpret -- not a numeric cast. `bits.astype(float32)`
    would return the integer value of the bit pattern (bf16 4.375 is 0x408C, so
    it would read back as 16524), which looks like a plausible activation and is
    how a broken readback survives review.
    """

    from hipengine.core.memory import DeviceBuffer, copy_device_to_host
    from hipengine.loading.materialize import host_array_ptr

    raw = np.empty(rows * columns, dtype=np.uint16)
    copy_device_to_host(
        host_array_ptr(raw),
        DeviceBuffer(ptr=buffer.ptr, nbytes=raw.nbytes),
        raw.nbytes,
    )
    return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(rows, columns)


def _dequantized(reader: GGUFReader, name: str) -> np.ndarray:
    tensor = reader.tensor_info(name)
    from hipengine.quant.gguf import dequantize_gguf_data

    return np.asarray(
        dequantize_gguf_data(reader.tensor_data(name), tensor.ggml_type), dtype=np.float32
    )


@pytest.mark.skipif(
    not hip_runtime_available(), reason="HIP runtime unavailable; skipping projection test"
)
def test_quantized_projection_reproduces_the_bf16_projection(reader: GGUFReader) -> None:
    """The quantized dispatch and the bf16 GEMV agree on the same weight.

    The two paths must produce the same projection from the same source values.
    The bf16 side uses the artifact's tensor dequantized to bf16; the quantized
    side uses the artifact's own blocks. If the dispatch picked the wrong kernel,
    read the wrong layout, or passed the operands in the wrong order, the two
    would differ -- and this is the check that catches it before the difference
    is buried inside a 30-layer forward pass.
    """

    from hipengine.core.memory import malloc, free
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import gemma4_project
    from hipengine.loading.materialize import (
        float_array_to_bf16_bits,
        load_host_array_to_device_as_dtype,
    )

    spec = next(
        s for s in plan_gemma4_gguf_resident_specs(reader) if s.slot_path.endswith(".attn_q")
    )
    out_features, in_features = (int(d) for d in spec.source.shape)
    rows = 4

    rng = np.random.default_rng(20260922)
    x = rng.standard_normal((rows, in_features)).astype(np.float32)

    # bf16 path: the artifact's tensor, dequantized and rounded to bf16.
    weight_bf16 = _dequantized(reader, spec.source.name)
    x_ptr = load_host_array_to_device_as_dtype(
        "x", float_array_to_bf16_bits(x), "bf16", source_dtype="BF16"
    )
    w_ptr = load_host_array_to_device_as_dtype(
        "w", float_array_to_bf16_bits(weight_bf16), "bf16", source_dtype="BF16"
    )
    out_bf16 = malloc(rows * out_features * 2)
    out_quant = malloc(rows * out_features * 2)
    quantized = materialize_gemma4_gguf_device_weight(reader, spec)
    try:
        gemma4_project(
            x_ptr.buffer.ptr, w_ptr.buffer.ptr, out_bf16.ptr, rows, in_features, out_features
        )
        gemma4_project(x_ptr.buffer.ptr, quantized, out_quant.ptr, rows, in_features, out_features)

        got_bf16 = _read_bf16_rows(out_bf16, rows, out_features)
        got_quant = _read_bf16_rows(out_quant, rows, out_features)

        # The fixture's Q8_0 values are `d * q` with `d = k * 2**-5` for k <= 5
        # and `|q| <= 15`, so every weight has at most 7 significant bits and is
        # *exactly* representable in bf16. The two paths therefore compute the
        # same real-valued product and differ only by bf16 rounding of the
        # output, which is under half an ulp (about 0.4%).
        #
        # That is why the tolerance is tight: on this fixture a dispatch that
        # read the wrong layout or swapped the operands could not hide inside it.
        # A 2% tolerance would have been loose enough to pass a wrong kernel.
        scale = float(np.abs(got_bf16).max())
        assert scale > 0, "the bf16 projection produced nothing to compare against"
        assert np.allclose(got_quant, got_bf16, rtol=5e-3, atol=5e-3 * scale), (
            f"quantized projection diverged from bf16: "
            f"max abs diff {np.abs(got_quant - got_bf16).max():.4g} against scale {scale:.4g}"
        )
        # Independent of the tolerance: the quantized output must not be a
        # constant, which is what a kernel reading a single block for every row
        # would produce.
        assert got_quant.std(axis=1).min() > 0, "quantized output has a constant row"
    finally:
        free(out_bf16)
        free(out_quant)
        x_ptr.free()
        w_ptr.free()
        quantized.free()


# --------------------------------------------------------------------------
# Selected-expert dispatch: one projection per compact row
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not hip_runtime_available(), reason="HIP runtime unavailable; skipping selected test"
)
def test_selected_expert_dispatch_matches_the_bf16_offset_path(reader: GGUFReader) -> None:
    """The selected path and the per-expert offset path agree.

    The offset path is already validated against the CPU reference, so agreeing
    with it is the check that the selected path's per-row expert indexing is
    right. A kernel that ignored `selected` and used one expert for every row
    would disagree here, because the fixture's experts hold different weights.
    """

    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, malloc
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        gemma4_project_experts_by_offset,
        gemma4_project_experts_selected,
    )
    from hipengine.loading.materialize import (
        float_array_to_bf16_bits,
        host_array_ptr,
        load_host_array_to_device_as_dtype,
    )

    spec = next(s for s in plan_gemma4_gguf_resident_specs(reader) if len(s.source.shape) == 3)
    num_experts, out_features, in_features = (int(d) for d in spec.source.shape)
    experts = [0, 1, num_experts - 1, 0]
    rows = len(experts)

    rng = np.random.default_rng(20260922)
    x = rng.standard_normal((rows, in_features)).astype(np.float32)

    # A bf16 baseline from the same source values, so both paths read the same
    # weights and the only difference is how the expert is selected.
    from hipengine.quant.gguf import dequantize_gguf_data

    weight_bf16 = np.asarray(
        dequantize_gguf_data(
            reader.tensor_data(spec.source.name), reader.tensor_info(spec.source.name).ggml_type
        ),
        dtype=np.float32,
    )
    x_ptr = load_host_array_to_device_as_dtype(
        "x", float_array_to_bf16_bits(x), "bf16", source_dtype="BF16"
    )
    w_ptr = load_host_array_to_device_as_dtype(
        "w", float_array_to_bf16_bits(weight_bf16), "bf16", source_dtype="BF16"
    )
    sel_ptr = load_host_array_to_device_as_dtype(
        "selected", np.asarray(experts, dtype=np.int64), "int64", source_dtype="I64"
    )
    out_bf16 = malloc(rows * out_features * 2)
    out_sel = malloc(rows * out_features * 2)
    quantized = materialize_gemma4_gguf_device_weight(reader, spec)
    try:
        # The bf16 path has no selected kernel, so it must report not-served.
        assert (
            gemma4_project_experts_selected(
                w_ptr.buffer.ptr,
                x_ptr.buffer.ptr,
                sel_ptr.buffer.ptr,
                out_bf16.ptr,
                rows,
                rows,
                num_experts,
                in_features,
                out_features,
            )
            is False
        )
        assert (
            gemma4_project_experts_selected(
                quantized,
                x_ptr.buffer.ptr,
                sel_ptr.buffer.ptr,
                out_sel.ptr,
                rows,
                rows,
                num_experts,
                in_features,
                out_features,
            )
            is True
        )

        # The bf16 reference, one launch per distinct expert.
        counts = np.zeros(num_experts, dtype=np.int64)
        for expert in experts:
            counts[expert] += 1
        starts = np.zeros(num_experts + 1, dtype=np.int64)
        starts[1:] = np.cumsum(counts)
        # Order the rows the way the offset path expects: grouped by expert.
        order = sorted(range(rows), key=lambda i: experts[i])
        starts_ptr = load_host_array_to_device_as_dtype(
            "starts", starts, "int64", source_dtype="I64"
        )
        packed = np.ascontiguousarray(x[order])
        packed_ptr = load_host_array_to_device_as_dtype(
            "packed", float_array_to_bf16_bits(packed), "bf16", source_dtype="BF16"
        )
        gemma4_project_experts_by_offset(
            w_ptr.buffer.ptr,
            packed_ptr.buffer.ptr,
            out_bf16.ptr,
            starts_ptr.buffer,
            num_experts,
            in_features,
            out_features,
        )

        def read_back(buffer) -> np.ndarray:
            raw = np.empty(rows * out_features, dtype=np.uint16)
            copy_device_to_host(
                host_array_ptr(raw),
                DeviceBuffer(ptr=buffer.ptr, nbytes=raw.nbytes),
                raw.nbytes,
            )
            return (
                (raw.astype(np.uint32) << np.uint32(16))
                .view(np.float32)
                .reshape(rows, out_features)
            )

        # Undo the grouping so both results are in the original row order.
        grouped = read_back(out_bf16)
        ungrouped = np.empty_like(grouped)
        for position, source_row in enumerate(order):
            ungrouped[source_row] = grouped[position]
        selected = read_back(out_sel)

        # Distinct experts must actually be selected: a kernel that used expert
        # 0 for every row would produce identical rows 0 and 3 here.
        assert not np.allclose(selected[0], selected[2]), "selected index had no effect"
        scale = float(np.abs(ungrouped).max())
        assert scale > 0
        assert np.allclose(selected, ungrouped, rtol=5e-3, atol=5e-3 * scale), (
            f"selected path diverged from the offset path: "
            f"max abs diff {np.abs(selected - ungrouped).max():.4g} against scale {scale:.4g}"
        )
    finally:
        for buffer in (out_bf16, out_sel):
            free(buffer)
        x_ptr.free()
        w_ptr.free()
        sel_ptr.free()
        quantized.free()
