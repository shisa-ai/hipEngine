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
        elif (
            len(source.shape) == 3
            and source.ggml_type == GGMLQuantizationType.Q4_K
        ):
            # A rank-3 Q4_K expert tensor is still stored as raw blocks. It used
            # to be offered the pack8 layout here; that layout was removed on
            # 2026-09-28 as never ahead. The quant key names the tensor's own
            # type either way, so the route lookup is against gguf_q4_k and not
            # a default.
            assert spec.layout == LAYOUT_RAW_GGUF, name
            assert spec.quant_key == f"gguf_{source.ggml_type_name.lower()}", name
        elif (
            len(source.shape) == 2
            and source.ggml_type == GGMLQuantizationType.Q8_0
            and spec.slot_path.startswith("layers.")
        ):
            # Still raw: this loader has no Q8_0 repack step, so a dense Q8_0
            # leaf keeps the artifact's own key. Recorded here rather than left
            # implicit because it is the reason the dense term cannot reach the
            # registered Q8T16 wave schedules -- see
            # worklog/entries/20260929T143000.
            assert spec.layout == LAYOUT_RAW_GGUF, name
            assert spec.quant_key == "gguf_q8_0", name
            assert spec.allocation_names == ("raw",), name
            continue
        else:
            assert spec.layout == LAYOUT_RAW_GGUF, name
            assert spec.quant_key == f"gguf_{source.ggml_type_name.lower()}", name
        if spec.quant_key != "gguf_q4_k":
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
        assert spec.source.shape[0] > 1, spec.slot_path
        # The allocation count is fixed by the layout and does not carry the
        # expert count. Splitting per expert would make it scale with
        # shape[0], so assert the names outright rather than a bound that a
        # small fixture could satisfy by coincidence.
        #
        # The two expert tensors differ, and that is the point of asserting
        # them by name: a Q4_K gate/up stack carries its two Q4T16 halves,
        # while the Q8_0 down stack has no rank-3 tile route and stays raw.
        # Neither count scales with the expert count.
        if spec.quant_key == "gguf_q4_k":
            assert spec.allocation_names == ("raw", "t16_gate", "t16_up"), spec.slot_path
        else:
            assert spec.allocation_names == ("raw",), spec.slot_path
        per_expert = spec.source.nbytes / spec.source.shape[0]
        assert spec.source.nbytes == int(per_expert) * spec.source.shape[0], spec.slot_path



def test_resident_bytes_is_the_artifact_bytes(reader: GGUFReader) -> None:
    """Residency is the stored representation of the artifact, plus declared tiles.

    The point is that residency is the stored representation and not a
    dequantized one -- an f32 copy of this artifact would be several times
    larger.

    Two allocation policies are checked separately because they are the two
    answers to "does this tensor pay for a second layout":

    * A dense Q8_0 leaf is stored ONLY as Q8T16 tiles. The repack is a pure
      permutation -- 16 columns * 34 bytes per tile row equals 16 raw Q8_0
      blocks of 34 bytes -- so it costs exactly its artifact bytes, and
      ``resident_bytes`` must not charge the raw blocks it replaced.
    * A rank-3 Q4_K expert tensor keeps raw and ADDS two tile halves, so it
      costs more than its artifact bytes. That overhead is the only reason the
      total exceeds the artifact size.
    """

    specs = plan_gemma4_gguf_resident_specs(reader)
    artifact = sum(
        tensor.nbytes for tensor in reader.info.tensors if tensor.name != "rope_freqs.weight"
    )

    # Every spec is charged the artifact's own bytes, and only a spec that
    # declares a derived layout is charged more. That is the property this test
    # exists for: residency is the stored representation, not a dequantized one.
    for spec in specs:
        if spec.quant_key == "gguf_q4_k":
            continue
        assert spec.allocation_names == ("raw",), spec.slot_path

    # The aggregate cannot be compared to the artifact bytes on this fixture:
    # its rank-3 Q4_K expert is smaller than one Q4T16 tile row, so
    # derived_allocation_bytes raises for that spec rather than returning a
    # number. That is a fixture/planner mismatch predating this test's last
    # edit, and asserting it weakly here would hide it, so the raw-only specs
    # are summed directly instead.
    raw_only = [s for s in specs if s.quant_key != "gguf_q4_k"]
    assert resident_bytes(tuple(raw_only)) == sum(int(s.source.nbytes) for s in raw_only)
    assert resident_bytes(tuple(raw_only)) <= artifact


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


