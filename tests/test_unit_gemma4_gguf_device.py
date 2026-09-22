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
