"""Device residency for the Gemma 4 assistant (MTP) head.

The head is a separate GGUF artifact with its own architecture string, so its
tensors need their own residency plan even though the layouts and the upload
path are the backbone's. This module is the second half of
:mod:`hipengine.loading.gemma4_assistant_gguf`, which decodes and validates the
head's metadata and tensor names; this one turns a validated tensor map into
owned device allocations the forward can read.

Planning and materializing are separate for the same reason they are on the
backbone: a caller can compute the residency total before allocating anything.
Both go through :func:`hipengine.loading.gemma4_gguf_device.plan_gguf_weight_spec`
so the head and the backbone cannot disagree about how a GGUF block type becomes
a resident layout.

The head's linear weights are Q8_0 and its norms F32. `rope_freqs` is F32 and is
the one tensor here that is not a weight the forward multiplies by: it is the
`freq_factors` table the full-attention blocks read. It is carried as a dense F32
allocation like the norms, because that is what it is on device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from hipengine.core.memory import DeviceBuffer
from hipengine.loading.gguf import GGUFReader, MissingGGUFTensorError
from hipengine.loading.gemma4_gguf_device import (
    Gemma4GGUFDeviceWeight,
    Gemma4GGUFWeightSpec,
    materialize_gemma4_gguf_device_weight,
    plan_gguf_weight_spec,
    resident_bytes,
)
from hipengine.loading.gemma4_assistant_gguf import (
    Gemma4AssistantConfig,
    Gemma4AssistantTensorMap,
    build_gemma4_assistant_tensor_map,
)

# The slot names the forward looks tensors up by. They are stable identifiers
# rather than GGUF names: the GGUF name for block 2's query weight is
# ``blk.2.attn_q.weight`` and it is addressed here as ``blocks.2.attn_q``, so a
# caller does not have to build the name or know whether it carries a suffix.
GLOBAL_SLOTS: tuple[str, ...] = (
    "token_embedding",
    "output_norm",
    "rope_freqs",
    "nextn_pre_projection",
    "nextn_post_projection",
)

# ``slot -> GGUF tensor name`` for the five globals. ``token_embd`` is also the
# head's output projection: the reference loader marks the output tensor
# ``TENSOR_DUPLICATED`` against ``token_embd``, so the vocabulary projection is
# tied and there is no second tensor to load.
_GLOBAL_TENSOR_NAMES: Mapping[str, str] = {
    "token_embedding": "token_embd.weight",
    "output_norm": "output_norm.weight",
    "rope_freqs": "rope_freqs.weight",
    "nextn_pre_projection": "nextn.pre_projection.weight",
    "nextn_post_projection": "nextn.post_projection.weight",
}

BLOCK_SLOTS: tuple[str, ...] = (
    "attn_norm",
    "layer_output_scale",
    "attn_q",
    "attn_q_norm",
    "attn_output",
    "post_attention_norm",
    "ffn_norm",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "post_ffw_norm",
)


def assistant_slot_names(config: Gemma4AssistantConfig) -> tuple[str, ...]:
    """Return every slot the head's forward reads, blocks first."""

    names: list[str] = []
    for block_id in range(int(config.block_count)):
        names.extend(f"blocks.{block_id}.{slot}" for slot in BLOCK_SLOTS)
    names.extend(GLOBAL_SLOTS)
    return tuple(names)


def assistant_tensor_name(slot: str, config: Gemma4AssistantConfig) -> str:
    """Map one forward slot to the GGUF tensor name that carries it."""

    if slot in _GLOBAL_TENSOR_NAMES:
        return _GLOBAL_TENSOR_NAMES[slot]
    parts = slot.split(".")
    if len(parts) == 3 and parts[0] == "blocks":
        block_id = int(parts[1])
        if not 0 <= block_id < int(config.block_count):
            raise MissingGGUFTensorError(
                f"assistant head slot {slot!r} names block {block_id}, but the head "
                f"has {config.block_count}"
            )
        return f"blk.{block_id}.{parts[2]}.weight"
    raise MissingGGUFTensorError(f"assistant head has no tensor slot {slot!r}")


def plan_gemma4_assistant_device_specs(
    tensor_map: Gemma4AssistantTensorMap,
) -> tuple[Gemma4GGUFWeightSpec, ...]:
    """Plan a resident spec for every tensor in the assistant head.

    Every slot the forward reads is planned, and a slot the validated map does
    not carry is an error naming it rather than a weight that quietly does not
    load. The map has already checked the shapes, so a failure here is a slot
    this module and the map disagree about rather than a malformed artifact.
    """

    config = tensor_map.config
    specs: list[Gemma4GGUFWeightSpec] = []
    for slot in assistant_slot_names(config):
        name = assistant_tensor_name(slot, config)
        try:
            tensor = tensor_map.tensor(name)
        except MissingGGUFTensorError as exc:
            raise MissingGGUFTensorError(
                f"assistant head slot {slot!r} resolves to {name!r}, which the "
                f"validated tensor map does not carry"
            ) from exc
        specs.append(plan_gguf_weight_spec(slot, tensor))
    return tuple(specs)


@dataclass(frozen=True)
class Gemma4AssistantDeviceWeights:
    """Owned device allocations for one assistant head, keyed by forward slot."""

    config: Gemma4AssistantConfig
    weights: Mapping[str, Gemma4GGUFDeviceWeight]

    def weight(self, slot: str) -> Gemma4GGUFDeviceWeight:
        try:
            return self.weights[slot]
        except KeyError as exc:
            raise MissingGGUFTensorError(
                f"assistant head device weights have no slot {slot!r}"
            ) from exc

    def block(self, block_id: int, slot: str) -> Gemma4GGUFDeviceWeight:
        return self.weight(f"blocks.{int(block_id)}.{slot}")

    def buffer(self, slot: str) -> DeviceBuffer:
        """Return the raw device buffer for one slot.

        The forward reads ``.ptr``; this exists so callers do not reach through
        two attributes and a default allocation name at every call site.
        """

        return self.weight(slot).allocation("raw").buffer

    def nbytes(self) -> int:
        return sum(int(weight.nbytes) for weight in self.weights.values())

    def free(self, *, runtime=None) -> None:
        for weight in reversed(tuple(self.weights.values())):
            weight.free(runtime=runtime)


def materialize_gemma4_assistant_device_weights(
    reader: GGUFReader,
    specs: tuple[Gemma4GGUFWeightSpec, ...],
    *,
    device=None,
    runtime=None,
    backend: str = "hip_gfx1100",
    allocator=None,
) -> Gemma4AssistantDeviceWeights:
    """Upload every planned assistant weight, or raise having freed what it took.

    All-or-nothing: a head that fails partway through is not a head with some
    weights, it is a head that cannot run, and leaving 440 MB resident for one is
    worse than failing. The same reasoning the backbone's loader uses.
    """

    config = gemma4_assistant_config_from_reader(reader)
    weights: dict[str, Gemma4GGUFDeviceWeight] = {}
    try:
        for spec in specs:
            weights[spec.slot_path] = materialize_gemma4_gguf_device_weight(
                reader,
                spec,
                device=device,
                runtime=runtime,
                backend=backend,
                allocator=allocator,
            )
    except BaseException:
        for weight in reversed(tuple(weights.values())):
            weight.free(runtime=runtime)
        raise
    return Gemma4AssistantDeviceWeights(config=config, weights=weights)


def gemma4_assistant_config_from_reader(reader: GGUFReader) -> Gemma4AssistantConfig:
    """Decode the head's config from a reader without validating the tensors."""

    from hipengine.loading.gemma4_assistant_gguf import (
        gemma4_assistant_config_from_metadata,
    )

    return gemma4_assistant_config_from_metadata(reader.info.metadata)


def load_gemma4_assistant_device_weights(
    path: str,
    *,
    device=None,
    runtime=None,
    backend: str = "hip_gfx1100",
    allocator=None,
) -> Gemma4AssistantDeviceWeights:
    """Open one assistant GGUF and load all of it to device.

    Validates the tensor map first, so a head whose metadata or shapes disagree
    with the contract fails before any allocation rather than after 440 MB of
    uploads.
    """

    reader = GGUFReader(path)
    tensor_map = build_gemma4_assistant_tensor_map(reader.info)
    specs = plan_gemma4_assistant_device_specs(tensor_map)
    return materialize_gemma4_assistant_device_weights(
        reader,
        specs,
        device=device,
        runtime=runtime,
        backend=backend,
        allocator=allocator,
    )


def assistant_resident_bytes(specs: tuple[Gemma4GGUFWeightSpec, ...]) -> int:
    """Return the device bytes ``specs`` would occupy, without allocating."""

    return resident_bytes(specs)


__all__ = [
    "BLOCK_SLOTS",
    "GLOBAL_SLOTS",
    "Gemma4AssistantDeviceWeights",
    "assistant_resident_bytes",
    "assistant_slot_names",
    "assistant_tensor_name",
    "gemma4_assistant_config_from_reader",
    "load_gemma4_assistant_device_weights",
    "materialize_gemma4_assistant_device_weights",
    "plan_gemma4_assistant_device_specs",
]
