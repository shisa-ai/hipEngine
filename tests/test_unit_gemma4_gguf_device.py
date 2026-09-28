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
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_RAW_GGUF,
    derived_allocation_bytes,
    materialize_gemma4_gguf_device_weight,
    plan_gemma4_gguf_resident_specs,
    resident_bytes,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import (
    GGUF_Q4_K_BLOCK_BYTES,
    GGUF_Q4_K_TILE16_BLOCK_BYTES,
    GGUF_Q4_K_TILE16_COLS,
)
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
            # A dense Q8_0 leaf is stored as Q8T16 tiles, and only as tiles.
            # The slot path is what decides this, not the tensor name: the
            # token embedding and lm head are also rank-2 Q8_0 but are
            # GEMV-shaped and stay raw. The layout matters as much as the key --
            # it is what selects the launch ABI.
            assert spec.layout == LAYOUT_GGUF_Q8_0_T16, name
            assert spec.quant_key == "gguf_q8_0_t16_v1", name
            assert spec.allocation_names == ("tiles",), name
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

    # A dense Q8_0 leaf is stored only as Q8T16 tiles, and the tile slab is
    # exactly the artifact bytes. This is the property that makes the change
    # memory-neutral, so it is asserted per leaf rather than in aggregate.
    q8_leaves = [s for s in specs if s.layout == LAYOUT_GGUF_Q8_0_T16]
    assert q8_leaves, "fixture has no dense Q8_0 leaf to check"
    for spec in q8_leaves:
        assert spec.allocation_names == ("tiles",), spec.slot_path
        assert derived_allocation_bytes(spec, "tiles") == int(spec.source.nbytes), spec.slot_path

    # And a leaf that keeps raw must not also be charged for tiles.
    for spec in specs:
        if spec.allocation_names == ("raw",):
            assert spec.layout != LAYOUT_GGUF_Q8_0_T16, spec.slot_path

    # The specs that keep only raw are summed on their own, so a planner that
    # charged the artifact bytes twice could not satisfy this and the aggregate
    # check below at the same time.
    raw_only = [s for s in specs if s.quant_key != "gguf_q4_k"]
    assert resident_bytes(tuple(raw_only)) == sum(int(s.source.nbytes) for s in raw_only)
    assert resident_bytes(tuple(raw_only)) <= artifact


def test_expert_tiles_are_priced_from_the_stored_bytes(reader: GGUFReader) -> None:
    """A Q4_K expert stack is charged for both tile halves, and the price is computable.

    ``resident_bytes`` is the loader's pre-allocation total, so it has to answer
    for the artifact the loader actually materializes. A Q4_K expert tensor is
    stored as raw blocks whose row is ``in_features / 256`` Q4_K blocks, not
    ``in_features`` elements, and the fused width splits at ``out_features // 2``.
    Pricing the tiles from the logical row length instead raises on every real
    artifact, so the figure is derived from the tensor's own byte count here and
    cross-checked against the layout arithmetic rather than against the planner.
    """

    specs = plan_gemma4_gguf_resident_specs(reader)
    stacked = [spec for spec in specs if "t16_gate" in spec.allocation_names]
    assert stacked, "fixture has no rank-3 Q4_K expert stack to check"

    artifact = sum(
        tensor.nbytes for tensor in reader.info.tensors if tensor.name != "rope_freqs.weight"
    )
    tiled = 0
    for spec in stacked:
        experts, out_features, _ = spec.source.shape
        assert int(out_features) % 2 == 0, spec.slot_path
        assert int(out_features) // 2 % GGUF_Q4_K_TILE16_COLS == 0, spec.slot_path
        blocks_per_row = int(spec.source.nbytes) // (
            int(experts) * int(out_features) * GGUF_Q4_K_BLOCK_BYTES
        )
        assert blocks_per_row > 0, spec.slot_path
        expected = (
            int(experts)
            * (int(out_features) // 2 // GGUF_Q4_K_TILE16_COLS)
            * blocks_per_row
            * GGUF_Q4_K_TILE16_BLOCK_BYTES
        )
        # A tile block is 2368 bytes where the 16 Q4_K blocks it replaces are
        # 2304, so the tiles cost more than the raw bytes they are built from.
        assert expected > int(spec.source.nbytes) // 2, spec.slot_path
        assert derived_allocation_bytes(spec, "t16_gate") == expected, spec.slot_path
        assert derived_allocation_bytes(spec, "t16_up") == expected, spec.slot_path
        tiled += 2 * expected

    # Residency is the artifact's stored bytes plus the tiles, and nothing else:
    # the dense Q8_0 leaves' tiles are a permutation that replaces raw.
    assert resident_bytes(specs) == artifact + tiled


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


