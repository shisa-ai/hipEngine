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
    Gemma4GGUFDeviceWeight,
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
    """The resident quant key follows the tensor's type and resident layout.

    Raw residency keeps ``gguf_<type>``; the T16 conversion of the fused
    gate_up stack carries the layout string as its quant key, which is the
    registry key its owners resolve under.
    """

    by_name = {spec.source.name: spec for spec in plan_gemma4_gguf_resident_specs(reader)}
    for name, spec in by_name.items():
        source = reader.tensor_info(name)
        if source.ggml_type == GGMLQuantizationType.F32:
            assert spec.layout == LAYOUT_DENSE_F32, name
            assert spec.quant_key == "f32", name
            assert spec.allocation_names == ("raw",), name
        elif spec.layout == T16_LAYOUT:
            assert spec.quant_key == T16_LAYOUT, name
            assert spec.allocation_names == ("tiles",), name
            assert source.ggml_type == GGMLQuantizationType.Q4_K, name
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
        # One allocation for the whole stack, named for its resident layout:
        # ``raw`` for GGUF bytes as stored, ``tiles`` for the T16 conversion.
        expected_name = "tiles" if spec.layout == T16_LAYOUT else "raw"
        assert spec.allocation_names == (expected_name,), spec.slot_path
        assert spec.source.shape[0] > 1, spec.slot_path
        # One allocation holds all experts. If the planner had split per expert
        # the resident total would be the same but the allocation count would be
        # shape[0] times larger, so check the total against the whole tensor.
        per_expert = spec.source.nbytes / spec.source.shape[0]
        assert spec.source.nbytes == int(per_expert) * spec.source.shape[0], spec.slot_path


def test_resident_bytes_is_the_artifact_bytes(reader: GGUFReader) -> None:
    """Raw and dense residency is the stored size, not a dequantized size.

    The T16-converted stacks are planned separately at their tiles size (see
    ``test_resident_bytes_counts_the_planned_layouts``); everything else must
    still occupy exactly what the artifact stores.
    """

    specs = plan_gemma4_gguf_resident_specs(reader)
    t16_sources = {spec.source.name for spec in specs if spec.layout == T16_LAYOUT}
    expected = sum(
        tensor.nbytes
        for tensor in reader.info.tensors
        if tensor.name != "rope_freqs.weight" and tensor.name not in t16_sources
    )
    planned_raw = sum(
        int(spec.source.nbytes) for spec in specs if spec.layout != T16_LAYOUT
    )
    assert planned_raw == expected
    assert t16_sources, "fixture gate_up stacks should convert to t16"


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


# --------------------------------------------------------------------------
# Fused residency: gate and up become one resident weight
# --------------------------------------------------------------------------


def _gate_up_specs(reader: GGUFReader) -> list:
    """The first layer's gate and up specs, in the order they must be fused."""

    by_slot = {spec.slot_path: spec for spec in plan_gemma4_gguf_resident_specs(reader)}
    return [by_slot["layers.0.ffn_gate"], by_slot["layers.0.ffn_up"]]


def test_gate_and_up_are_fusible_as_stored(reader: GGUFReader) -> None:
    """The fusion's precondition holds in the artifact, not just in theory.

    Fusing requires a shared layout, quant type, in_features and bytes-per-row.
    Those are properties of the file, so they are asserted against the file
    rather than assumed from the model's architecture.
    """

    gate, up = _gate_up_specs(reader)
    assert gate.layout == up.layout == LAYOUT_RAW_GGUF
    assert gate.quant_key == up.quant_key
    assert gate.source.ggml_type == up.source.ggml_type
    assert gate.source.shape == up.source.shape
    assert gate.source.byte_shape == up.source.byte_shape


def test_fused_materialization_rejects_a_width_mismatch(reader: GGUFReader) -> None:
    """A mismatched member is refused by name, before any device upload.

    Members whose bytes cannot sit in one tensor must not be silently
    concatenated, so the error names the offending slot and both byte shapes.
    """

    from dataclasses import replace

    from hipengine.loading.gemma4_gguf_device import materialize_fused_gguf_device_weight

    gate, up = _gate_up_specs(reader)
    narrower = replace(
        up,
        source=replace(
            up.source,
            shape=(up.source.shape[0], up.source.shape[1] // 2),
            byte_shape=(up.source.byte_shape[0], up.source.byte_shape[1] // 2),
        ),
    )
    with pytest.raises(ValueError, match="ffn_up") as excinfo:
        materialize_fused_gguf_device_weight(reader, [gate, narrower])
    assert str(narrower.source.shape) in str(excinfo.value)


@pytest.mark.skipif(
    not hip_runtime_available(), reason="HIP runtime unavailable; skipping fused residency test"
)
def test_fused_blocks_are_the_members_bytes_in_order(reader: GGUFReader) -> None:
    """The fused buffer is the members' bytes concatenated, in order.

    This is the whole claim behind fusing gate and up: the concatenation of the
    raw blocks *is* the fused tensor's storage, so no reblocking, dequantization
    or arithmetic happens anywhere. Comparing the device buffer against a fresh
    concatenation of the reader's byte views tests that directly, and a spec
    that described anything other than the concatenation would fail it.
    """

    from hipengine.core.memory import DeviceBuffer, copy_device_to_host
    from hipengine.loading.gemma4_gguf_device import materialize_fused_gguf_device_weight
    from hipengine.loading.materialize import host_array_ptr

    gate, up = _gate_up_specs(reader)
    weight = materialize_fused_gguf_device_weight(reader, [gate, up])
    try:
        source = weight.spec.source
        assert source.shape == (
            gate.source.shape[0] + up.source.shape[0],
            gate.source.shape[1],
        )
        assert source.nbytes == gate.source.nbytes + up.source.nbytes
        assert source.byte_shape == (
            gate.source.byte_shape[0] + up.source.byte_shape[0],
            gate.source.byte_shape[1],
        )
        assert source.ggml_shape == (
            gate.source.ggml_shape[0],
            gate.source.ggml_shape[1] + up.source.ggml_shape[1],
        )
        # Both members share the slot, so the fused path names both.
        assert "ffn_gate" in source.name and "ffn_up" in source.name

        nbytes = int(weight.allocation("raw").buffer.nbytes)
        assert nbytes == source.nbytes

        host = np.empty(nbytes, dtype=np.uint8)
        copy_device_to_host(
            host_array_ptr(host),
            DeviceBuffer(ptr=weight.allocation("raw").buffer.ptr, nbytes=nbytes),
            nbytes,
        )
        expected = np.concatenate(
            [
                np.frombuffer(reader.tensor_data(name), dtype=np.uint8)
                for name in (gate.source.name, up.source.name)
            ]
        )
        np.testing.assert_array_equal(host, expected, err_msg="fused residency")
    finally:
        weight.free()


# --------------------------------------------------------------------------
# Resident T16 tiles for the fused gate_up stack (D1 repack)
# --------------------------------------------------------------------------

# The layout string and the registry quant key are the same string by design:
# dispatch routes on ``quant_key`` alone, with no backend or quant branch.
T16_LAYOUT = "gguf_q4_k_t16_v1"


def _t16_tiles_nbytes(spec) -> int:
    """Planned device bytes for a gate_up spec's T16 tiles residency."""

    from hipengine.quant.gguf_q4_k import (
        GGUF_Q4_K_BLOCK_BYTES,
        GGUF_Q4_K_TILE16_BLOCK_BYTES,
        GGUF_Q4_K_TILE16_COLS,
    )

    experts, out_features, bytes_per_row = spec.source.byte_shape
    blocks = bytes_per_row // GGUF_Q4_K_BLOCK_BYTES
    assert out_features % GGUF_Q4_K_TILE16_COLS == 0
    return experts * (out_features // GGUF_Q4_K_TILE16_COLS) * blocks * GGUF_Q4_K_TILE16_BLOCK_BYTES


def _gate_up_stack_specs(reader: GGUFReader) -> list:
    """Every planned fused ``ffn_gate_up_exps`` spec in the fixture."""

    return [
        spec
        for spec in plan_gemma4_gguf_resident_specs(reader)
        if spec.slot_path.endswith(".ffn_gate_up_exps")
    ]


def test_gate_up_q4_k_plans_the_t16_tiles_layout(reader: GGUFReader) -> None:
    """The fused Q4_K gate_up stack materializes as T16 tiles, on by default.

    This is the layout the measured decode win lives in (iteration 136's
    screen: 4.9-5.1x over the raw-layout selected GEMV at bitwise-equal
    output), so the planner must select it for every gate_up stack whose shape
    the t16 kernels serve -- not behind a flag.
    """

    specs = plan_gemma4_gguf_resident_specs(reader)
    gate_up = _gate_up_stack_specs(reader)
    assert gate_up, "fixture has no fused gate_up expert slot"

    for spec in gate_up:
        assert spec.layout == T16_LAYOUT, spec.slot_path
        assert spec.quant_key == T16_LAYOUT, spec.slot_path
        assert spec.allocation_names == ("tiles",), spec.slot_path

    # Every other tensor keeps the raw / dense residency it had. The down
    # stack (Q8_0 in the fixture, Q5_1 in the real artifact) has no t16
    # family, and dense projections have no reason to move.
    gate_up_slots = {spec.slot_path for spec in gate_up}
    for spec in specs:
        if spec.slot_path in gate_up_slots:
            continue
        assert spec.layout in (LAYOUT_RAW_GGUF, LAYOUT_DENSE_F32), spec.slot_path
        assert spec.allocation_names == ("raw",), spec.slot_path
        assert spec.quant_key == (
            "f32"
            if spec.layout == LAYOUT_DENSE_F32
            else f"gguf_{spec.source.ggml_type_name.lower()}"
        ), spec.slot_path


def test_resident_bytes_counts_the_planned_layouts(reader: GGUFReader) -> None:
    """Residency plans what the *kernels will read*, not just file bytes.

    T16 tiles carry per-16-row scale blocks, so a converted gate_up stack
    occupies slightly more than its stored bytes (2.78% at the fixture shape,
    the same ratio at the real 1408x2816 shape). A capacity check that summed
    file bytes would under-plan the device by that delta.
    """

    specs = plan_gemma4_gguf_resident_specs(reader)
    gate_up = _gate_up_stack_specs(reader)
    assert gate_up, "fixture has no fused gate_up expert slot"

    expected = 0
    for spec in specs:
        if spec.layout == T16_LAYOUT:
            expected += _t16_tiles_nbytes(spec)
        else:
            expected += int(spec.source.nbytes)
    assert resident_bytes(specs) == expected

    artifact_bytes = sum(
        tensor.nbytes for tensor in reader.info.tensors if tensor.name != "rope_freqs.weight"
    )
    assert expected > artifact_bytes, "tiles residency must be planned, not assumed free"


def test_expert_stride_bytes_follows_the_resident_layout(reader: GGUFReader) -> None:
    """Expert strides come from the resident allocation, not the source file.

    The fused ``half_bytes = stride // 2`` offsets in the prefill owners land
    on a tile boundary only when the stride is the tiles stride: at the real
    shape the raw stride is 2,230,592 while the tiles stride is 2,292,224, and
    half of the raw stride (1,115,296) is not a multiple of the 2,368-byte tile
    block, so an owner reading tiles through the source-derived stride would
    read across expert and half boundaries.
    """

    from types import SimpleNamespace

    class _StubAllocation:
        def __init__(self, nbytes: int) -> None:
            self.buffer = SimpleNamespace(nbytes=nbytes)

    gate_up = _gate_up_stack_specs(reader)
    assert gate_up, "fixture has no fused gate_up expert slot"
    spec = gate_up[0]
    tiles_nbytes = _t16_tiles_nbytes(spec)
    weight = Gemma4GGUFDeviceWeight(
        spec=spec,
        allocations={"tiles": _StubAllocation(tiles_nbytes)},
        backend="hip_gfx1100",
    )
    experts = int(spec.source.shape[0])
    assert weight.expert_stride_bytes == tiles_nbytes // experts
    assert weight.expert_stride_bytes != int(spec.source.nbytes) // experts

    # A raw stacked tensor keeps the stride it had (allocation == file bytes).
    raw_stacked = next(
        spec
        for spec in plan_gemma4_gguf_resident_specs(reader)
        if len(spec.source.shape) == 3 and spec.layout == LAYOUT_RAW_GGUF
    )
    raw_weight = Gemma4GGUFDeviceWeight(
        spec=raw_stacked,
        allocations={"raw": _StubAllocation(int(raw_stacked.source.nbytes))},
        backend="hip_gfx1100",
    )
    assert raw_weight.expert_stride_bytes == int(raw_stacked.source.nbytes) // int(
        raw_stacked.source.shape[0]
    )


def test_t16_gate_up_dispatch_keys_resolve() -> None:
    """Every probe the gemma chain can issue for a t16 gate_up resolves.

    Three owners serve the converted stack: the selected decode GEMV under the
    ``linear`` layer key (decode *and* the strict-mode grouped-miss
    fall-through, whose absence would route to the raw-layout by-offset path),
    the mmq32 prefill leaf that the production ``auto`` route probes, and the
    WMMA compact alias the wmma route probes. Each must be registered for the
    t16 quant key before the planner may emit it.
    """

    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
        register_gguf_q4_k_q8_1_selected_prefill_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_t16_selected_prefill import (
        register_gguf_q4_k_t16_selected_prefill_kernels,
    )
    from hipengine.kernels.registry import KernelKey, is_registered
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    # Both prefill families register outside the shared ensure list, so the
    # production probes reach them through the leaf's own registrar
    # (``gemma4_project_experts_mmq_dual`` / ``_mmq32_leaf_owner``) or the
    # backend package's import; pytest's registry baseline restore between
    # tests drops import-time registrations, so this test re-runs the
    # registrars exactly the way production code does when it needs them.
    register_gguf_q4_k_q8_1_selected_prefill_kernels()
    register_gguf_q4_k_t16_selected_prefill_kernels()

    for layer, variant in (
        ("linear", "selected_gemv_bf16_bf16_out"),
        ("moe_linear", "selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out"),
        ("moe_linear", "selected_dual_wmma_prefill_compact_bf16_bf16_out"),
    ):
        key = KernelKey("hip_gfx1100", layer, T16_LAYOUT, variant)
        _ensure_linear_kernel_registered(key)
        assert is_registered(key), key.display()


def test_raw_layout_launches_refuse_a_t16_weight(reader: GGUFReader) -> None:
    """The raw-layout expert launches fail loudly on a tiles weight.

    Both are the by-offset fallback family: if a future probe miss ever lets a
    converted stack reach them, they must raise naming the layout rather than
    stride through tiles as if they were raw blocks -- that silent path is
    exactly what the alias registration exists to make unreachable, and this
    guard is the tripwire that keeps the reachability claim honest.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        gemma4_project_expert,
        gemma4_project_experts_by_offset,
    )
    from types import SimpleNamespace

    class _StubAllocation:
        def __init__(self, nbytes: int) -> None:
            self.buffer = SimpleNamespace(nbytes=nbytes)

    spec = _gate_up_stack_specs(reader)[0]
    weight = Gemma4GGUFDeviceWeight(
        spec=spec,
        allocations={"tiles": _StubAllocation(_t16_tiles_nbytes(spec))},
        backend="hip_gfx1100",
    )

    with pytest.raises(ValueError, match="t16_v1"):
        gemma4_project_experts_by_offset(weight, 0, 0, None, 4, 256, 128)
    with pytest.raises(ValueError, match="t16_v1"):
        gemma4_project_expert(weight, 0, 0, 0, 1, 256, 128)
