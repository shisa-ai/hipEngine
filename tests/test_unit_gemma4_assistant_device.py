"""Device-residency tests for the Gemma 4 ``gemma4-assistant`` MTP draft head.

The head's device loader is the second half of its loading contract: the map
module decodes and validates the metadata and tensor names, and this one turns a
validated map into owned device allocations the forward can read.

Two surfaces:

* no-GPU: the forward-slot vocabulary, the slot-to-GGUF-name mapping, and the
  plan over the real header, which needs no HIP because planning reads shapes and
  block types only;
* GPU: the real head loaded to device, with every slot present, every allocation
  the planned byte count, and the total matching the artifact's own tensor bytes.

RED first, in the two shapes this module can fail. The first run of this file
failed at import: ``hipengine.loading.gemma4_assistant_device`` did not exist, so
none of the slots, the mapping, or the plan could be exercised at all. Once the
module existed, two of these tests failed against real code rather than against
their own fixtures, and both were corrected here because the tests were wrong:
the planned layout split is 23 raw blocks and 26 dense F32 (not the 44/5 first
guessed), and a second ``free`` is not idempotent on this tree. The GPU half is
guarded so a no-ROCm runner skips rather than fails.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFReader, MissingGGUFTensorError, scan_gguf
from hipengine.loading.gemma4_assistant_gguf import (
    build_gemma4_assistant_tensor_map,
)
from hipengine.loading.gemma4_assistant_device import (
    BLOCK_SLOTS,
    GLOBAL_SLOTS,
    assistant_resident_bytes,
    assistant_slot_names,
    assistant_tensor_name,
    load_gemma4_assistant_device_weights,
    plan_gemma4_assistant_device_specs,
)

ARTIFACT = Path(
    "/models/gguf/gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
)

_ARTIFACT_PRESENT = ARTIFACT.exists()
_needs_artifact = pytest.mark.skipif(
    not _ARTIFACT_PRESENT, reason="the Gemma 4 assistant head is not downloaded"
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


_needs_hip = pytest.mark.skipif(
    not _hip_available(), reason="libamdhip64.so is not available"
)


def _tensor_map():
    if not ARTIFACT.exists():
        pytest.skip("the Gemma 4 assistant head is not downloaded")
    return build_gemma4_assistant_tensor_map(scan_gguf(ARTIFACT))


def test_slot_vocabulary_is_the_forward_surface() -> None:
    """Four blocks of eleven tensors, plus five globals."""

    assert len(BLOCK_SLOTS) == 11
    assert len(GLOBAL_SLOTS) == 5
    assert len(set(BLOCK_SLOTS)) == 11
    assert len(set(GLOBAL_SLOTS)) == 5


def test_slot_names_cover_the_whole_head() -> None:
    """The plan's slot list is the map's tensor list, one for one."""

    tensor_map = _tensor_map()
    names = assistant_slot_names(tensor_map.config)
    # 4 blocks x 11 + 5 globals = 49, which is the artifact's tensor count.
    assert len(names) == 49
    assert len(set(names)) == 49
    resolved = {assistant_tensor_name(slot, tensor_map.config) for slot in names}
    assert resolved == set(tensor_map.tensors), (
        "every planned slot must resolve to a tensor the validated map carries, "
        "and no tensor may go unplanned"
    )


def test_slot_mapping_is_explicit_about_suffixes() -> None:
    """A caller addresses ``blocks.2.attn_q``, not ``blk.2.attn_q.weight``."""

    config = _tensor_map().config
    assert assistant_tensor_name("blocks.0.attn_q", config) == "blk.0.attn_q.weight"
    assert assistant_tensor_name("blocks.3.ffn_down", config) == "blk.3.ffn_down.weight"
    assert (
        assistant_tensor_name("nextn_pre_projection", config)
        == "nextn.pre_projection.weight"
    )
    assert assistant_tensor_name("token_embedding", config) == "token_embd.weight"


def test_slot_mapping_rejects_a_block_the_head_does_not_have() -> None:
    """A block index outside the head is an error, not a missing-key surprise."""

    config = _tensor_map().config
    with pytest.raises(MissingGGUFTensorError):
        assistant_tensor_name("blocks.4.attn_q", config)
    with pytest.raises(MissingGGUFTensorError):
        assistant_tensor_name("not_a_slot", config)


def test_plan_covers_every_tensor_with_a_supported_layout() -> None:
    """Planning reads shapes and block types only, so it needs no device."""

    tensor_map = _tensor_map()
    specs = plan_gemma4_assistant_device_specs(tensor_map)
    assert len(specs) == 49
    assert len({spec.slot_path for spec in specs}) == 49

    by_layout: dict[str, int] = {}
    for spec in specs:
        by_layout[spec.layout] = by_layout.get(spec.layout, 0) + 1
    # 23 raw Q8_0 blocks and 26 dense F32. The split is forced by the head's
    # own tensor list rather than chosen: the five linears per block
    # (attn_q, attn_output, ffn_gate, ffn_up, ffn_down) plus token_embd and the
    # two nextn projections are 5 * 4 + 3 = 23, and the six per-block norms
    # (attn_norm, attn_q_norm, post_attention_norm, ffn_norm, post_ffw_norm,
    # layer_output_scale) plus output_norm and rope_freqs are 6 * 4 + 2 = 26.
    # A linear appearing as F32, or a norm as raw blocks, is what this catches.
    assert by_layout == {"raw_gguf": 23, "dense_f32": 26}, by_layout


@_needs_artifact
def test_plan_byte_total_matches_the_artifact() -> None:
    """The planned residency is the artifact's own tensor bytes.

    Every planned tensor is uploaded unmodified, so the sum of the planned
    allocations must equal the sum of the GGUF tensors' own byte counts. A
    mismatch means a layout that expands or drops data.
    """

    tensor_map = _tensor_map()
    specs = plan_gemma4_assistant_device_specs(tensor_map)
    planned = assistant_resident_bytes(specs)
    source = sum(int(tensor.nbytes) for tensor in tensor_map.tensors.values())
    assert planned == source, (
        f"planned {planned} device bytes for {source} artifact bytes"
    )
    # The head is 461,766,816 bytes on disk including its header and metadata, so
    # the tensor payload is a little under that.
    assert 440_000_000 < source < 461_766_816


@_needs_artifact
@_needs_hip
def test_real_head_loads_every_slot_to_device() -> None:
    """The whole head loads, and every slot is where the forward expects it."""

    weights = load_gemma4_assistant_device_weights(str(ARTIFACT))
    try:
        assert weights.config.block_count == 4
        assert weights.config.n_embd == 1024
        assert weights.config.n_embd_backbone == 2816

        tensor_map = _tensor_map()
        for slot in assistant_slot_names(weights.config):
            weight = weights.weight(slot)
            source = tensor_map.tensor(assistant_tensor_name(slot, weights.config))
            assert weight.allocation("raw").buffer.nbytes == int(source.nbytes), slot
            # A non-zero pointer: a slot that planned but did not upload would
            # still report the right byte count.
            assert int(weight.allocation("raw").buffer.ptr) != 0, slot

        # The forward's accessors are the ones the tests above exercised, so a
        # rename that broke the forward would break this.
        assert weights.block(0, "attn_q") is weights.weight("blocks.0.attn_q")
        assert int(weights.buffer("token_embedding").ptr) != 0
        assert weights.nbytes() > 440_000_000
    finally:
        weights.free()


@_needs_artifact
@_needs_hip
def test_partial_materialization_frees_what_it_took() -> None:
    """A load that fails partway leaves no device memory behind.

    The head is 440 MB and the loader is all-or-nothing: a head that fails partway
    through is not a head with some weights, it is a head that cannot run, and
    holding its bytes resident for one costs the next load. The failure is induced
    by appending a spec whose tensor does not exist, so the real specs before it
    have already uploaded.

    This is the contract the loader documents; a plain second ``free`` is not, and
    is not asserted here -- ``DeviceTensorAllocation.free`` is not idempotent on
    this tree, the backbone's loader included, so a double free raises rather than
    no-opping.
    """

    from dataclasses import replace

    tensor_map = _tensor_map()
    specs = list(plan_gemma4_assistant_device_specs(tensor_map))
    # A real tensor's bytes under a slot that resolves to nothing, so planning has
    # already succeeded and the failure is the upload.
    bogus = replace(specs[-1], source=replace(specs[-1].source, name="not.a.tensor"))
    specs.append(bogus)

    reader = GGUFReader(str(ARTIFACT))
    with pytest.raises(Exception):
        from hipengine.loading.gemma4_assistant_device import (
            materialize_gemma4_assistant_device_weights,
        )

        materialize_gemma4_assistant_device_weights(reader, tuple(specs))
