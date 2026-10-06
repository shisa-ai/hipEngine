"""Gemma 4 GPU runner over a resident GGUF artifact.

Loads the artifact into device-resident weights, then runs the decoder forward
pass and returns logits. This is the layer between the GGUF loader and whatever
wants to generate tokens from it.

Nothing here chooses a kernel by quant name or backend name. Each stage resolves
its own dispatch through the four-axis registry:

===================  ==================================================
stage                route
===================  ==================================================
embedding gather     ``embedding/<quant>/lookup_bf16_out`` (raw layout)
embedding scale      ``scale/<quant>/gemma4_plain``
layer forward        the Gemma 4 layer family
expert projections   ``linear/<quant>/selected_gemv_bf16_bf16_out``
final norm           ``rmsnorm/<quant>/gemma4_plain``
lm head              ``launch_gguf_linear``, ``output_dtype="f32"``
===================  ==================================================

so an unfamiliar artifact whose tensors the kernels can execute runs, and one
whose tensors they cannot fails inside the dispatch that could not serve it.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    enqueue_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.gemma4 import (
    SLIDING_ATTENTION,
    Gemma4AttentionGeometry,
    Gemma4RopeConfig,
    Gemma4TextConfig,
    gemma4_text_config_from_hf,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    Gemma4AttentionScratch,
    gemma4_attention_serves_keys,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rope import gemma4_rope_cos_sin_tables
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
    Gemma4LayerGeometry,
    Gemma4LayerKV,
    Gemma4LayerPointers,
    Gemma4LayerScratch,
    gemma4_layer_forward_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
    gemma4_logit_argmax_f32,
    gemma4_logit_argmax_scratch_bytes,
    gemma4_logit_softcap_f32,
    gemma4_rmsnorm_f32w_bf16,
    gemma4_scale_bf16,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.gemma4_gguf import Gemma4GGUFConfig, gemma4_gguf_config_from_metadata
from hipengine.loading.gemma4_gguf_device import (
    Gemma4GGUFDeviceWeight,
    Gemma4GGUFWeightSpec,
    can_fuse_gguf_device_weights,
    materialize_fused_gguf_device_weight,
    materialize_gemma4_gguf_device_weight,
    plan_gemma4_gguf_resident_specs,
)
from hipengine.loading.materialize import DeviceTensorAllocation, load_host_array_to_device_as_dtype
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import (
    _WMMA_PREFILL_ENV,
    launch_gguf_linear,
    wmma_prefill_session,
)
from hipengine.runtime.gemma4_int8_kv import (
    Gemma4Int8KVCache,
    int8_kv_consumer_shared_bytes,
    int8_kv_resident_bytes,
)

_BF16_BYTES = 2


@dataclass(frozen=True)
class Gemma4SharedKV:
    """A read view of one layer's KV cache, for a consumer that shares it.

    The assistant head attends against this model's KV rather than holding its
    own, so it needs the two buffer addresses and the live position count, but
    must not be able to append. This type is that capability: it carries no
    write offset and no reference to the runner.
    """

    layer_index: int
    key_cache: int
    value_cache: int
    capacity: int
    live: int
    num_kv_heads: int
    head_dim: int

    @property
    def kv_width(self) -> int:
        """Elements per position: ``num_kv_heads * head_dim``."""

        return self.num_kv_heads * self.head_dim

    @property
    def live_bytes(self) -> int:
        """BF16 bytes of the live region, which is the readable span."""

        return self.live * self.kv_width * _BF16_BYTES
_F32_BYTES = 4
_I64_BYTES = 8

# Widest block a single forward call hands to the kernels when the caller does
# not ask for a specific one. The per-layer scratch is sized from this, and the
# scratch is what makes the difference between a context that loads and one
# that does not: on the real 26B UD-Q4_K_XL artifact, sizing it for the whole
# 8192-token context costs 52.68 GB against 3.29 GB for a 512-token block, on
# top of 17.00 GB of resident blocks and 1.85 GB of KV. A prompt wider than
# this is forwarded as consecutive blocks, which is exact.
# Tokens per prefill pass. This is the ubatch equivalent: a prompt wider than
# this is forwarded as consecutive blocks.
#
# Raising this to 1024 to match llama.cpp's ubatch was measured in an earlier
# pass and did not pay: the weight-reuse argument (64 tokens per expert against
# 32) was expected to help, but the int8 MMQ path was compute-bound on the WMMA
# units rather than memory-bound on those weights, so halving the reads bought
# almost nothing. That pass recorded prompt 1024 at 1373/1367/1367 (512) against
# 1351/1397/1397 (1024) and called it a wash, so 512 stayed.
#
# Re-tested on 2026-09-29 (GEMMA4-26B-A4B-PUNCHLIST P13) after the Q5_K depth-16
# kernel, the lm_head final-block fix, the Q5_K plan-override removal, the packed
# Q4_K gate_up, the int8 scratch-contract fix and the two-stream MoE branch had
# all landed. The conclusion reversed:
#
#     prompt 1024   block 512:  prefill 456.9 ms (2241 tok/s)  decode 40.99
#                   block 1024: prefill 417.3 ms (2454 tok/s)  decode 40.78
#     prompt 2048   block 512:  prefill 1293.3 ms (1584)       decode 39.62
#                   block 1024: prefill 1218.7 ms (1681)       decode 39.55
#
# Prefill improves 9.5% and 6.1%; decode moves 0.5% and 0.2%, i.e. it is flat,
# not the 2% regression the earlier pass measured. End-to-end at prompt 1024
# plus 64 new tokens is 1987 ms against 2018 ms. The cost is the per-layer
# scratch, which scales with the block: 6.58 GB against 3.29 GB, so a resident
# context goes from 22.14 GB to 25.43 GB -- still inside the 48 GB the real
# 26B UD-Q4_K_XL artifact already needs room for, and self-limiting because
# max_block is min(capacity, this). 1024 stays while the measurement holds.
DEFAULT_PREFILL_BLOCK = 1024


# Headroom kept free above the computed resident set: allocator rounding,
# per-forward temporaries the layers do not name as scratch, and the kernel
# image. Half a gigabyte against a 3.29 GB difference between the two blocks.
_FIT_RESERVE_BYTES = 512 * 1024 * 1024


def _resident_bytes(
    *,
    block: int,
    capacity: int,
    hidden: int,
    config: Any,
    dense_widths: Sequence[int],
    kv_storage: str = "bf16",
    kv_scale_dtype: DType = DType.FP16,
) -> int:
    """Bytes the runner holds for a prefill block of ``block`` rows.

    Everything except the weights, which are already resident by the time the
    runner is constructed: the runner's own buffers, the per-layer KV cache, and
    every scratch buffer the layers can take. Asking a scratch its size
    allocates no device memory, because every buffer in one is created on first
    use. The scratch sum is deliberately an upper bound over its whole table
    rather than over the names one path touches.

    The KV term follows the selected storage: BF16 K/V planes, or the INT8
    payload, scale planes and FP32 consumer scratch of the INT8 owner.
    """
    total = (
        block * _I64_BYTES
        + block * hidden * _BF16_BYTES
        + hidden * _BF16_BYTES
        + int(config.vocab_size or 0) * _F32_BYTES
    )
    for index, attention in enumerate(config.attention):
        scratch = Gemma4LayerScratch(
            tokens=block,
            hidden_size=hidden,
            dense_intermediate=dense_widths[index],
            geometry=_layer_geometry(attention),
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            expert_intermediate=config.moe_intermediate_size,
        )
        total += scratch.resident_bytes()
    if kv_storage == "int8_per_token_head":
        attentions = tuple(
            (a.num_heads, a.num_kv_heads, a.head_dim) for a in config.attention
        )
        total += int8_kv_resident_bytes(
            capacity=capacity,
            max_block=block,
            attentions=attentions,
            scale_dtype=kv_scale_dtype,
        )
    else:
        for attention in config.attention:
            total += (
                capacity * attention.num_kv_heads * attention.head_dim * _BF16_BYTES * 2
            )
    return total


def _fit_prefill_block(
    block: int,
    *,
    capacity: int,
    hidden: int,
    config: Any,
    dense_widths: Sequence[int],
    kv_storage: str = "bf16",
    kv_scale_dtype: DType = DType.FP16,
) -> int:
    """Shrink ``block`` until the resident set fits free device memory.

    Scratch scales with the block and is taken lazily on the first forward, so a
    block that does not fit does not fail at load -- it fails as an out of memory
    error partway through decode, after the model has already loaded and long
    after anything could diagnose it. Sizing against ``hipMemGetInfo`` keeps the
    larger block on cards that can hold it and steps down where they cannot:
    automatic selection between two working block sizes, not a flag and not a
    blanket revert. A runtime that will not answer the query returns the block
    unchanged, because a missing measurement is not a reason to refuse to run.
    """
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    mem_get_info = getattr(runtime, "mem_get_info", None)
    if not callable(mem_get_info):
        return block
    try:
        free_bytes, _total_bytes = mem_get_info()
    except Exception:
        return block
    budget = max(0, int(free_bytes) - _FIT_RESERVE_BYTES)
    while block > 1:
        needed = _resident_bytes(
            block=block,
            capacity=capacity,
            hidden=hidden,
            config=config,
            dense_widths=dense_widths,
            kv_storage=kv_storage,
            kv_scale_dtype=kv_scale_dtype,
        )
        if needed <= budget:
            return block
        block //= 2
    return 1

# Which layer field each artifact slot feeds. The slot names are the loader's;
# the fields are the layer's. Kept as one table so the two cannot drift apart
# silently.
#
# ``ffn_gate`` and ``ffn_up`` are deliberately absent: both read the same
# normalized hidden state and are numerically separable, so they are fused into
# one resident weight and one launch (see ``_GATE_UP_FUSION``). The artifact
# still stores them as two tensors, so both slots remain required.
_GATE_UP_FUSION = ("ffn_gate", "ffn_up", "mlp_gate_up_proj")

_SLOT_TO_FIELD = {
    "attn_norm": "input_layernorm",
    "attn_q": "q_proj",
    "attn_k": "k_proj",
    "attn_v": "v_proj",
    "attn_output": "o_proj",
    "attn_q_norm": "q_norm",
    "attn_k_norm": "k_norm",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_norm": "pre_feedforward_layernorm",
    "ffn_down": "mlp_down_proj",
    "post_ffw_norm": "post_feedforward_layernorm",
    "ffn_gate_inp": "router_proj",
    "ffn_gate_inp_scale": "router_scale",
    "ffn_down_exps_scale": "router_per_expert_scale",
    "pre_ffw_norm_2": "pre_feedforward_layernorm_2",
    "ffn_gate_up_exps": "experts_gate_up_proj",
    "ffn_down_exps": "experts_down_proj",
    "post_ffw_norm_1": "post_feedforward_layernorm_1",
    "post_ffw_norm_2": "post_feedforward_layernorm_2",
    "layer_output_scale": "layer_scalar",
}

# Fields that hold a plain f32 device pointer rather than a projection.
_SCALAR_FIELDS = frozenset(
    {
        "input_layernorm",
        "q_norm",
        "k_norm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "router_scale",
        "router_proj",
        "router_per_expert_scale",
        "pre_feedforward_layernorm_2",
        "post_feedforward_layernorm_1",
        "post_feedforward_layernorm_2",
        "layer_scalar",
    }
)

# Projection fields, in the order the layer reads them. Only used to report
# which slots were missing when a layer cannot be assembled.
_PROJECTION_FIELDS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "qkv_proj",
    "o_proj",
    "mlp_gate_up_proj",
    "mlp_down_proj",
    "experts_gate_up_proj",
    "experts_down_proj",
)


@dataclass
class Gemma4DeviceWeights:
    """Every device-resident tensor the forward pass needs, plus ownership."""

    config: Gemma4TextConfig
    embed_tokens: Gemma4GGUFDeviceWeight
    final_norm: DeviceTensorAllocation
    layers: tuple[Gemma4LayerPointers, ...]
    backend: str
    lm_head: Gemma4GGUFDeviceWeight | None = None
    # Per-layer dense-MLP widths, in layer order. The artifact's
    # ``feed_forward_length`` may differ by layer, and the kernels read the
    # width from the scratch they are handed; carrying the real widths here is
    # what keeps each layer's scratch at its own size instead of the maximum.
    # Empty on a synthetic/legacy weights object, which falls back to the
    # config's single value.
    dense_intermediate: tuple[int, ...] = ()
    _owned: list[Any] = field(default_factory=list, repr=False)
    _freed: bool = field(default=False, repr=False)

    def free(self) -> None:
        """Release every allocation this object owns. Idempotent."""

        if self._freed:
            return
        self._freed = True
        for item in self._owned:
            item.free()
        self._owned.clear()


def _tensor_values(reader: GGUFReader, spec: Gemma4GGUFWeightSpec) -> np.ndarray:
    """Read one f32 tensor as a contiguous float32 array.

    GGML stores f32 little-endian, so this is a byte reinterpretation rather
    than a dequantization; the dtype check is what makes that safe to assume.
    """

    info = reader.tensor_info(spec.source.name)
    if info.ggml_type != GGMLQuantizationType.F32:
        raise ValueError(
            f"{spec.slot_path} is {info.ggml_type.name}, not F32; the scalar slots are "
            "F32 in every artifact this loader plans"
        )
    values = np.frombuffer(reader.tensor_data(spec.source.name), dtype="<f4").astype(np.float32)
    expected = int(np.prod(spec.source.shape))
    if values.size != expected:
        raise ValueError(
            f"{spec.slot_path} holds {values.size} values but declares {expected}"
        )
    return np.ascontiguousarray(values)


def gemma4_text_config_from_gguf(
    gguf: Gemma4GGUFConfig,
    *,
    tensor_names: Sequence[str] = (),
) -> Gemma4TextConfig:
    """Build the text config from the artifact's own metadata.

    A user who has a ``.gguf`` file has a ``.gguf`` file; requiring a separate HF
    ``config.json`` beside it would be a second thing to get wrong. Every field
    the forward pass reads is in the GGUF metadata, including the per-layer
    attention geometry and both RoPE contracts.

    ``tensor_names`` is optional and only used for the two facts the metadata
    cannot state: whether the lm head is tied, and which layers reuse the raw K
    projection as V. Both are derivable from the tensor list, so they are read
    from it rather than assumed.
    """

    names = set(tensor_names)
    block_count = gguf.block_count
    if len(gguf.layer_types) != block_count:
        raise ValueError(
            f"metadata declares {block_count} blocks but {len(gguf.layer_types)} layer types"
        )

    def per_layer(values: Sequence[int], index: int) -> int:
        if not values:
            raise ValueError("metadata carries no per-layer values for a required field")
        return int(values[index] if len(values) > 1 else values[0])

    attention = []
    for index in range(block_count):
        layer_type = gguf.layer_types[index]
        sliding = layer_type == SLIDING_ATTENTION
        head_dim = int(gguf.key_length_swa if sliding else gguf.key_length)
        rope = gguf.swa_rope if sliding else gguf.full_rope
        attention.append(
            Gemma4AttentionGeometry(
                layer_type=layer_type,
                num_heads=per_layer(gguf.head_counts, index),
                num_kv_heads=per_layer(gguf.head_count_kv, index),
                head_dim=head_dim,
                rope=Gemma4RopeConfig(
                    rope_theta=float(rope.freq_base),
                    head_dim=head_dim,
                    rope_angles=int(rope.rotated_pairs),
                    rope_type=rope.rope_type,
                ),
                sliding_window=int(gguf.sliding_window) if sliding else None,
                # A layer with no attn_v tensor takes the raw K projection as V.
                k_eq_v=names and f"blk.{index}.attn_v.weight" not in names,
            )
        )

    # GGUF omits output.weight when the lm head is tied to the embedding.
    tied = not names or "output.weight" not in names
    return Gemma4TextConfig(
        hidden_size=int(gguf.hidden_size),
        intermediate_size=max(int(v) for v in gguf.feed_forward_lengths),
        moe_intermediate_size=int(gguf.expert_feed_forward_length),
        num_experts=int(gguf.expert_count),
        top_k_experts=int(gguf.expert_used_count),
        rms_norm_eps=float(gguf.rms_norm_eps),
        attention=tuple(attention),
        final_logit_softcapping=gguf.final_logit_softcapping,
        tie_word_embeddings=bool(tied),
        vocab_size=int(gguf.vocab_size),
    )


def gemma4_text_config_from_reader(
    reader: GGUFReader,
    *,
    hf_config: Mapping[str, Any] | None = None,
) -> Gemma4TextConfig:
    """Derive the text config from artifact metadata, loading no tensor.

    Split out of :func:`load_gemma4_device_weights` so the attention geometry is
    available before any weight is resident. The geometry is what decides whether
    a requested context length is servable, and a caller that discovers that only
    after loading the artifact has already paid the load.
    """

    if hf_config is not None:
        return gemma4_text_config_from_hf(hf_config)
    info = reader.info
    tensor_names = tuple(t.name for t in info.tensors)
    gguf = gemma4_gguf_config_from_metadata(info)
    return gemma4_text_config_from_gguf(gguf, tensor_names=tensor_names)


def gemma4_layer_key_count(
    attention: Gemma4AttentionGeometry, capacity: int, *, rows: int = 1
) -> int:
    """Return the cached key span presented by a layer's widest block."""
    capacity = int(capacity)
    if attention.sliding_window is None:
        return capacity
    return min(capacity, int(attention.sliding_window) + int(rows) - 1)


def gemma4_require_context_capacity(
    config: Gemma4TextConfig,
    capacity: int,
    *,
    rows: int | None = None,
    kv_storage: str = "bf16",
    variants: Sequence[str] | None = None,
) -> None:
    """Raise if no selected attention path can serve ``capacity`` positions.

    The question is asked of the *selected* consumer, and of the family rather
    than of one kernel in it.

    BF16 256/512-head attention moves its logits to request-owned global scratch
    when a resident row would exceed the LDS budget, so that geometry is
    servable at any key count. Other BF16 geometries are charged the layer's
    actual key band. The INT8 consumer has its own context-sized LDS contract
    and is checked separately.

    ``gemma4_attention_serves_keys`` holds the strict kernel to one logit per
    live key in LDS. Each layer is charged the key count it actually presents,
    from :func:`gemma4_layer_key_count`: the whole context for a windowless
    layer, and ``window + rows - 1`` for a windowed one.

    A layer the strict kernel cannot serve is not automatically unservable. The
    WMMA flash prefills stage a fixed K/V tile and walk the keys in online-softmax
    batches, so their shared memory does not grow with the context, and a
    ``variants`` request that selects one of them for this geometry serves the
    layer. The question is asked through
    :func:`gemma4_attention_serves_keys`, which owns it, so the answer cannot
    drift from what the launchers do.

    ``rows`` is the widest block that will be submitted, and it widens a windowed
    layer's band. It defaults to the block ``Gemma4Runner`` derives when it is not
    told one, so a caller checking the geometry before a load reaches the same
    answer the runner will.

    This is a capability refusal and it is deliberately loud and named. It is also
    deliberately callable without weights, so a caller can raise it before paying
    for a load rather than after.
    """

    capacity = int(capacity)
    if rows is None:
        rows = min(capacity, DEFAULT_PREFILL_BLOCK)
    resolved = str(kv_storage or "auto")
    if resolved == "auto":
        resolved = "bf16"
    if resolved == "int8_per_token_head":
        for attention in config.attention:
            int8_kv_consumer_shared_bytes(
                capacity=int(capacity), head_dim=attention.head_dim
            )
        return
    for attention in config.attention:
        keys = gemma4_layer_key_count(attention, capacity, rows=int(rows))
        reason = gemma4_attention_serves_keys(
            head_dim=attention.head_dim,
            num_heads=attention.num_heads,
            num_kv_heads=attention.num_kv_heads,
            keys=keys,
            requested_variant=variants,
        )
        if reason is not None:
            raise NotImplementedError(
                f"no attention path serves {keys} keys for head_dim="
                f"{attention.head_dim} with {attention.num_heads}q/"
                f"{attention.num_kv_heads}kv at capacity {capacity}: {reason}"
            )


def load_gemma4_device_weights(
    reader: GGUFReader,
    *,
    hf_config: Mapping[str, Any] | None = None,
    backend: str = "hip_gfx1100",
) -> Gemma4DeviceWeights:
    """Load a Gemma 4 GGUF artifact into device-resident weights.

    The config comes from the artifact's GGUF metadata. ``hf_config`` overrides
    it for a checkpoint whose HF config.json is authoritative, such as a
    conversion the GGUF metadata cannot describe.
    """

    config = gemma4_text_config_from_reader(reader, hf_config=hf_config)
    specs = plan_gemma4_gguf_resident_specs(reader)
    by_slot = {spec.slot_path: spec for spec in specs}

    def require(slot: str) -> Gemma4GGUFWeightSpec:
        try:
            return by_slot[slot]
        except KeyError:
            raise KeyError(
                f"artifact has no tensor for {slot!r}; "
                f"planned slots are {sorted(by_slot)[:6]}..."
            ) from None

    owned: list[Any] = []
    try:
        embed = materialize_gemma4_gguf_device_weight(
            reader, require("token_embedding"), backend=backend
        )
        owned.append(embed)
        final_norm = load_host_array_to_device_as_dtype(
            "output_norm", _tensor_values(reader, require("output_norm")), "fp32"
        )
        owned.append(final_norm)

        # A tied checkpoint has no separate lm-head tensor, so the embedding
        # matrix is reused. That is a property of the artifact, not a fallback:
        # the reference ties the two when `tie_word_embeddings` is set.
        lm_head = None
        if not config.tie_word_embeddings:
            lm_head = materialize_gemma4_gguf_device_weight(
                reader, require("lm_head"), backend=backend
            )
            owned.append(lm_head)

        layers = []
        for index in range(config.num_hidden_layers):
            layers.append(_build_layer(reader, index, by_slot, config, backend, owned))

        # Dense MLP width is a per-layer property of the artifact. The metadata
        # carries a ``feed_forward_length`` array that may differ across layers,
        # and the kernels size the dense projections from the width they are
        # handed. Read each layer's width from its planned ``ffn_gate`` spec (the
        # logical shape's leading dimension is the projection's out_features) so
        # a narrower layer is never dispatched at the widest layer's size.
        dense_intermediate = tuple(
            int(by_slot[f"layers.{index}.ffn_gate"].source.shape[0])
            for index in range(config.num_hidden_layers)
        )
    except BaseException:
        for item in owned:
            item.free()
        raise

    return Gemma4DeviceWeights(
        config=config,
        embed_tokens=embed,
        final_norm=final_norm,
        layers=tuple(layers),
        backend=backend,
        lm_head=lm_head,
        dense_intermediate=dense_intermediate,
        _owned=owned,
    )


def _build_layer(
    reader: GGUFReader,
    index: int,
    by_slot: Mapping[str, Gemma4GGUFWeightSpec],
    config: Gemma4TextConfig,
    backend: str,
    owned: list[Any],
) -> Gemma4LayerPointers:
    """Assemble one layer's pointers from the artifact's slots."""

    prefix = f"layers.{index}."
    fields: dict[str, Any] = {}
    missing: list[str] = []

    gate_slot, up_slot, fused_field = _GATE_UP_FUSION
    gate_spec = by_slot.get(prefix + gate_slot)
    up_spec = by_slot.get(prefix + up_slot)
    if gate_spec is None or up_spec is None:
        # Both halves are required. A layer missing either is a real gap and is
        # reported by name rather than defaulted to a split projection.
        for slot, spec in ((gate_slot, gate_spec), (up_slot, up_spec)):
            if spec is None:
                missing.append(slot)
    else:
        fused = materialize_fused_gguf_device_weight(
            reader, (gate_spec, up_spec), backend=backend
        )
        owned.append(fused)
        fields[fused_field] = fused

    # Attention projections: one resident q|k|v weight and one launch instead
    # of three, whenever the artifact's storage can express it (see
    # ``can_fuse_gguf_device_weights``).
    #
    # Deliberately *not* fused on k_eq_v layers. Those run two projections
    # today (q and k -- there is no attn_v), and fusing them would be one
    # projection plus the post-projection split: still two launches, so the
    # split's 5.7-19.7 us and the transient fused buffer would be bought for
    # no reduction at all. Measured on the fixture geometry in the P6 split
    # investigation; re-check that number if the split's cost moves.
    fused_slots: frozenset[str] = frozenset()
    qkv_candidates = [by_slot.get(prefix + slot) for slot in ("attn_q", "attn_k", "attn_v")]
    if (
        not config.geometry(index).k_eq_v
        and all(spec is not None for spec in qkv_candidates)
    ):
        qkv_specs = tuple(spec for spec in qkv_candidates if spec is not None)
        if can_fuse_gguf_device_weights(qkv_specs):
            qkv = materialize_fused_gguf_device_weight(reader, qkv_specs, backend=backend)
            owned.append(qkv)
            fields["qkv_proj"] = qkv
            fused_slots = frozenset({"attn_q", "attn_k", "attn_v"})

    for slot, field_name in _SLOT_TO_FIELD.items():
        if slot in fused_slots:
            # Already carried by the fused weight above; materialising them
            # again would hold two copies of the same rows resident.
            continue
        spec = by_slot.get(prefix + slot)
        if spec is None:
            # v_proj genuinely does not exist on attention_k_eq_v layers; the
            # layer treats 0 as "reuse the raw K projection". Anything else
            # missing is a real gap and is reported rather than defaulted.
            if field_name == "v_proj" and config.geometry(index).k_eq_v:
                fields[field_name] = 0
                continue
            missing.append(slot)
            continue
        if field_name in _SCALAR_FIELDS:
            allocation = load_host_array_to_device_as_dtype(
                f"layers.{index}.{slot}", _tensor_values(reader, spec), "fp32"
            )
            owned.append(allocation)
            fields[field_name] = allocation.buffer.ptr
        else:
            weight = materialize_gemma4_gguf_device_weight(reader, spec, backend=backend)
            owned.append(weight)
            fields[field_name] = weight

    if missing:
        raise KeyError(
            f"layer {index} is missing {sorted(missing)}; the artifact does not "
            f"carry every tensor this layer's forward pass reads"
        )
    for name in _PROJECTION_FIELDS:
        if name not in fields:
            fields[name] = 0
    if "v_proj" not in fields:
        fields["v_proj"] = 0
    return Gemma4LayerPointers(**fields)


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 bit patterns, the way the kernels read them."""

    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    # Round to nearest even on the 16 discarded bits rather than truncating.
    rounding = ((bits >> 16) & 1) + np.uint32(0x7FFF)
    return ((bits + rounding) >> 16).astype(np.uint16)


def _gemma4_block_wmma_session(enabled: bool):
    """Scope the WMMA dense prefill opt-in to one Gemma 4 prefill block.

    An explicitly set ``HIPENGINE_GGUF_WMMA_PREFILL`` wins outright, so the env
    var stays a working rollback lever rather than something this wrapper
    silently overrides. When it is unset, a full block opts in and a partial
    block opts out.
    """

    if os.environ.get(_WMMA_PREFILL_ENV, "").strip():
        return contextlib.nullcontext()
    return wmma_prefill_session(enabled)
# The dense Q8_0 projections, admitted to the guarded d4x3 MMQ chain per shape.
#
# **Calibrated against the exact owner, and only reachable where that is the
# incumbent.** Every value here compares the chain with the strict
# `prefill_bf16_bf16_out` route, which is what a shape runs when WMMA prefill is
# off. With WMMA prefill on -- the default -- `_wmma_prefill_dispatch` claims the
# variant name first and the chain never sees the shape. That is the right
# outcome rather than a gap: measured against the WMMA owner at 512 rows the
# chain is **1.44x slower** (258.6 -> 373.2 ms for a 512-token prefill, campaign
# iteration 154), so a reorder that let it outrank the WMMA rewrite would
# regress the default prefill path by 44%. `tests/
# test_unit_gguf_q8_mmq_prefill_ordering.py` is the guard that makes such a
# reorder fail loudly instead of silently.
#
# risk_threshold governs how much of the matrix the sparse correction repairs,
# and the guard queues *more* as the threshold rises: measured on this model's
# real Q8_0 weights at 512 rows, 1e-8 queues 0.0% (and leaves 271-985 elements
# wrong), 1e-6 queues 0.2-0.3%, 1e-5 queues 1.5-2.5% and is bit-identical to the
# exact owner on every shape tried, 1e-4 queues 10-16%, and 1e-2 queues 100%.
# So 1e-5 is the setting where the chain is both exact and cheap; the same value
# the Qwen policies use. Repairing 10-16% of the elements at in_features MACs
# each costs about what the GEMM itself costs, which is why 1e-4 measured slower
# than the exact owner rather than faster.
#
# min_rows is per shape because the chain loses below 512 rows (0.45x-0.55x)
# against the exact owner, so the 64-row tail keeps it. (2112, 2816) is absent
# deliberately: 2112 is not a multiple of 128, so the d4 packing cannot serve it
# and that shape keeps its exact owner too. The runner's block size is
# `DEFAULT_PREFILL_BLOCK` = 512, so in the shipped configuration the only row
# count a prefill block can present is 512 (or fewer, on a short tail) -- which
# is exactly the crossover this table encodes. A raised block size would need a
# fresh measurement before these values mean anything.
GEMMA4_Q8_MMQ_MIN_ROWS: dict[tuple[int, int], int] = {
    (2816, 2112): 512,
    (2816, 2048): 512,
    (4096, 2816): 512,
    (2816, 4096): 512,
    (8192, 2816): 512,
    (2816, 8192): 512,
}
GEMMA4_Q8_MMQ_MAX_ROWS = 4096
GEMMA4_Q8_MMQ_RISK_THRESHOLD = 1.0e-5
GEMMA4_Q8_MMQ_MAX_OUT_FEATURES = 8192


@dataclass
class Gemma4Runner:
    """Runs the Gemma 4 decoder forward pass over a KV cache.

    ``capacity`` is the number of positions the cache holds. Prefill and decode
    share one path: both append their tokens at the current offset and attend
    over everything written so far, so a decode step is a prefill of one token.
    """

    weights: Gemma4DeviceWeights
    capacity: int
    max_block: int = 0
    #: Bound on the logits projection buffers, in rows. A speculative verify
    #: reads one row per verified position; the default of one row is what keeps
    #: a wide prefill from sizing its logits scratch for the whole block.
    max_logits_rows: int = 1
    #: The prefill-attention variants the execution profile requests, or ``None``
    #: for the layer's own automatic route selection. Verification passes the
    #: explicit strict variant rather than ``None`` so a verify block does not
    #: fall onto a production prefill route.
    prefill_attention_variants: tuple[str, ...] | None = None
    #: KV storage request. ``bf16`` is the BF16 comparison path; ``auto`` is
    #: resolved to ``bf16``. ``int8_per_token_head`` selects the BF16-source INT8
    #: writer and the direct INT8 consumers over the owned INT8 cache.
    kv_storage: str = "bf16"
    kv_scale_dtype: str = "fp16"
    kv_scale_granularity: str = "per_token_head"
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    _buffers: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _scratches: list[Gemma4LayerScratch] = field(default_factory=list, repr=False)
    _kv: list[Gemma4LayerKV] = field(default_factory=list, repr=False)
    _int8_kv: Any = field(default=None, repr=False)
    _kv_storage_resolved: str = field(default="bf16", repr=False)
    _kv_scale_dtype_resolved: DType = field(default=DType.FP16, repr=False)
    _caches: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _staging: dict[str, tuple[DeviceBuffer, int]] = field(default_factory=dict, repr=False)
    # One host-only multirow range per immutable RoPE config. Decode bypasses
    # this cache so single-token steps do not evict reusable prefill tables.
    _rope_prefill_tables: dict[
        Gemma4RopeConfig, tuple[int, int, np.ndarray, np.ndarray]
    ] = field(default_factory=dict, init=False, repr=False)
    _position: int = field(default=0, repr=False)
    # Rows of the most recent block, which is what ``hidden_state`` indexes.
    _last_rows: int = field(default=0, repr=False)
    # Logits rows the most recent launch projected, which is what ``_collect_block``
    # copies home (one, or the verify block's trailing rows).
    _last_logits_rows: int = field(default=1, repr=False)
    # Rows of post-output_norm state left in ``_normalized`` by the most recent
    # forward that asked for ``return_hidden``; 0 when the caller did not.
    _normalized_hidden_rows: int = field(default=0, repr=False)
    _closed: bool = field(default=False, repr=False)
    _q8_mmq_policy: Any = field(default=None, repr=False)
    _q8_mmq_library: Any = field(default=None, repr=False)
    _q8_mmq_workspace: Any = field(default=None, repr=False)
    _q8_mmq_risk_count: Any = field(default=None, repr=False)
    _q8_mmq_risk_indices: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        config = self.weights.config
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        # Remember whether the block was chosen here rather than handed in, so
        # the fit check below may shrink only the block we picked. An explicit
        # block is the caller's request and fails loudly if it does not fit.
        auto_block = self.max_block <= 0
        if auto_block:
            self.max_block = min(self.capacity, DEFAULT_PREFILL_BLOCK)
        if self.max_block > self.capacity:
            raise ValueError("max_block must not exceed capacity")
        # A capability refusal, raised here rather than at request time. Callers
        # that can reach the geometry without loading weights should call
        # gemma4_require_context_capacity first so the refusal costs nothing.
        self._resolve_kv_storage()
        self._validate_attention_shared_memory(config)

        hidden = config.hidden_size
        # Each layer's dense MLP can have its own width. Use the per-layer
        # widths the loader recorded from the artifact; a synthetic weights
        # object that carries none falls back to the config's single value.
        dense_widths = getattr(self.weights, "dense_intermediate", ()) or (
            config.intermediate_size,
        ) * len(config.attention)
        if len(dense_widths) != len(config.attention):
            raise ValueError(
                f"weights carry {len(dense_widths)} dense widths but the config "
                f"has {len(config.attention)} layers"
            )
        if auto_block:
            self.max_block = _fit_prefill_block(
                self.max_block,
                capacity=self.capacity,
                hidden=hidden,
                config=config,
                dense_widths=dense_widths,
                kv_storage=self._kv_storage_resolved,
                kv_scale_dtype=self._kv_scale_dtype_resolved,
            )
        # A verify block projects one logits row per verified position, so the
        # projection buffers are sized from this bound. Validated after the fit
        # shrink because a block smaller than the bound cannot carry the rows.
        if self.max_logits_rows <= 0:
            raise ValueError("max_logits_rows must be positive")
        if self.max_logits_rows > self.max_block:
            raise ValueError("max_logits_rows must not exceed max_block")
        # Every buffer taken below is owned by this runner, so a failure
        # partway through construction must release the ones already taken
        # rather than leaving them to the garbage collector, which does not own
        # device memory. The scratch objects are in the same list: they hold no
        # device buffers until a forward pass asks for one, but cleanup still
        # runs over them for symmetry with close().
        try:
            self._token_ids = self._alloc(self.max_block * _I64_BYTES)
            self._hidden = self._alloc(self.max_block * hidden * _BF16_BYTES)
            # Block-sized rather than single-row: the default path still
            # normalizes and projects only the last row, but ``return_hidden``
            # (M2) keeps every row's post-output_norm state here so a multi-row
            # lm_head can consume them without recomputing. 2.9 MB at a
            # 512-token block of 2816-wide states, against 25 GB in use.
            self._normalized = self._alloc(self.max_block * hidden * _BF16_BYTES)
            # Sized for the largest verify block's trailing rows, not one row:
            # the default of one row still only fills the base slice, so the
            # extra capacity is unread unless a caller asks for ``logits_rows``.
            self._logits = self._alloc(
                self.max_logits_rows * int(config.vocab_size or 0) * _F32_BYTES
            )
            # Greedy argmax route (D10): the 16-byte (index, value) result and
            # its partial scratch, pre-allocated beside _logits so no malloc
            # can happen inside a decode step or a captured graph.
            self._argmax_out = self._alloc(16)
            self._argmax_scratch = self._alloc(
                gemma4_logit_argmax_scratch_bytes(int(config.vocab_size or 0))
            )

            # One attention workspace serves every layer. The workspace is
            # transient -- a layer's launch chain finishes with it before the
            # next layer's starts -- and the block submits its layers serially
            # on one stream (the layer forward's default stream 0), so a later
            # layer's kernels are ordered behind the earlier layer's reads and
            # one allocation can serve them all. A per-layer owner instead
            # retains each layer's growth chain separately, which is what makes
            # a deep prefill's workspace scale with the layer count. Sharing
            # stays safe for anything queued elsewhere: the scratch keys its
            # buffers by stream, so another stream gets its own allocation
            # rather than one a kernel here may still be reading. Cleanup is
            # unchanged: every layer scratch still closes the owner it was
            # handed, and ``Gemma4AttentionScratch.close`` is idempotent, so the
            # workspace is synchronized and freed exactly once.
            shared_attention = Gemma4AttentionScratch()
            for index, attention in enumerate(config.attention):
                self._scratches.append(
                    Gemma4LayerScratch(
                        tokens=self.max_block,
                        hidden_size=hidden,
                        dense_intermediate=dense_widths[index],
                        geometry=_layer_geometry(attention),
                        num_experts=config.num_experts,
                        top_k=config.top_k_experts,
                        expert_intermediate=config.moe_intermediate_size,
                        attention=shared_attention,
                    )
                )
                if self._kv_storage_resolved == "int8_per_token_head":
                    continue
                key = self._alloc(
                    self.capacity * attention.num_kv_heads * attention.head_dim * _BF16_BYTES
                )
                value = self._alloc(
                    self.capacity * attention.num_kv_heads * attention.head_dim * _BF16_BYTES
                )
                self._caches.extend((key, value))
                self._kv.append(
                    Gemma4LayerKV(
                        key_cache=key.ptr,
                        value_cache=value.ptr,
                        capacity=self.capacity,
                        write_offset=0,
                    )
                )
            if self._kv_storage_resolved == "int8_per_token_head":
                # The INT8 owner allocates the payload, scale planes, page table
                # and FP32 consumer scratch; the runner keeps no BF16 K/V cache
                # and no full BF16 shadow.
                attentions = tuple(
                    (a.num_heads, a.num_kv_heads, a.head_dim) for a in config.attention
                )
                self._int8_kv = Gemma4Int8KVCache(
                    capacity=self.capacity,
                    max_block=self.max_block,
                    attentions=attentions,
                    backend=str(getattr(self.weights, "backend", "hip_gfx1100")),
                    scale_dtype=self._kv_scale_dtype_resolved,
                )
        except BaseException:
            # Mark closed first so a later close() cannot free the same buffers
            # a second time, then release everything taken so far.
            self._closed = True
            self._release_owned()
            raise

    def _validate_attention_shared_memory(self, config: Any) -> None:
        """Refuse a context the *selected* attention consumer cannot fit.

        Delegates to :func:`gemma4_require_context_capacity` so the pre-load
        check the generator runs and the construction check here cannot drift.
        """

        gemma4_require_context_capacity(
            config,
            self.capacity,
            rows=self.max_block,
            kv_storage=self._kv_storage_resolved,
            variants=self.prefill_attention_variants,
        )

    def _resolve_kv_storage(self) -> None:
        """Resolve the storage request to a concrete, supported mode.

        ``auto`` is the BF16 comparison path. ``int8_per_token_head`` is the
        only INT8 layout the writer and consumers implement (per-token/head
        scales, fp16 or fp32). Anything else is a named capability miss rather
        than a silent downgrade to BF16.
        """

        requested = str(self.kv_storage or "auto")
        if requested == "auto":
            requested = "bf16"
        if requested not in {"bf16", "int8_per_token_head"}:
            raise ValueError(
                f"unsupported KV storage {self.kv_storage!r}; the Gemma 4 runner "
                "implements 'bf16' and 'int8_per_token_head'"
            )
        self._kv_storage_resolved = requested
        if requested != "int8_per_token_head":
            self._kv_scale_dtype_resolved = DType.FP16
            return
        granularity = str(self.kv_scale_granularity or "per_token_head")
        if granularity != "per_token_head":
            raise ValueError(
                "the Gemma 4 INT8 KV cache implements per_token_head scale "
                f"granularity only; got {granularity!r}"
            )
        scale_dtype = DType.parse(self.kv_scale_dtype or "fp16")
        if scale_dtype not in {DType.FP16, DType.FP32}:
            raise ValueError(
                "the Gemma 4 INT8 KV cache implements fp16 or fp32 scales; "
                f"got {self.kv_scale_dtype!r}"
            )
        self._kv_scale_dtype_resolved = scale_dtype

    @property
    def uses_int8_kv(self) -> bool:
        """Whether this runner holds the INT8 per-token/head KV cache."""

        return self._int8_kv is not None

    @property
    def kv_cache(self) -> Gemma4Int8KVCache | None:
        """The owned INT8 KV cache, or ``None`` on the BF16 path."""

        return self._int8_kv

    @property
    def kv_storage_resolved(self) -> str:
        """The concrete storage mode this runner was built with."""

        return self._kv_storage_resolved

    @property
    def kv_scale_dtype_resolved(self) -> DType:
        """The concrete INT8 scale dtype this runner was built with."""

        return self._kv_scale_dtype_resolved

    def _alloc(self, nbytes: int) -> DeviceBuffer:
        if nbytes <= 0:
            raise ValueError(f"refusing to allocate {nbytes} bytes")
        buffer = malloc(nbytes)
        self._buffers.append(buffer)
        return buffer

    def _staging_buffer(self, name: str, nbytes: int) -> DeviceBuffer:
        """Return a reusable device buffer of at least ``nbytes``.

        Grown on demand and otherwise kept, never freed after a kernel launch.
        Freeing a buffer that an in-flight kernel still reads is a use-after-free
        the driver may only notice later, as a page-not-present fault somewhere
        else entirely. It also puts a device synchronization point on every layer
        of every token, which is most of what a decode step would otherwise cost.
        Growth is rare and geometric: a buffer is replaced only when a call
        needs more than any previous call did, and then it at least doubles, so a
        rising request replaces the buffer O(log) times over a session rather
        than several times per token. Replaced allocations are retained, never
        freed, so the memory a name holds stays within a constant factor of its
        largest request instead of growing with the number of requests.
        """

        held = self._staging.get(name)
        if held is not None:
            if held[1] >= nbytes:
                return held[0]
            # Geometric growth: at least double the current capacity, so a
            # request that keeps rising replaces the buffer O(log) times over a
            # session. Keeping the replaced allocation makes the retained total
            # O(largest request) rather than the sum of every request; freeing
            # it would be a use-after-free for an in-flight kernel.
            capacity = max(nbytes, held[1] * 2)
        else:
            capacity = nbytes
        buffer = malloc(capacity)
        self._buffers.append(buffer)
        self._staging[name] = (buffer, capacity)
        return buffer

    def reset(self) -> None:
        """Rewind to an empty sequence without freeing the cache."""

        self._position = 0
        self._last_rows = 0
        # The exposed states belonged to the sequence just rewound; a consumer
        # must not read them for a prompt that no longer exists.
        self._normalized_hidden_rows = 0
        if self._int8_kv is not None:
            self._int8_kv.reset()
        for index in range(len(self._kv)):
            self._kv[index] = Gemma4LayerKV(
                key_cache=self._kv[index].key_cache,
                value_cache=self._kv[index].value_cache,
                capacity=self.capacity,
                write_offset=0,
            )

    def _release_owned(self) -> None:
        """Free every allocation this runner owns, once.

        Shared by :meth:`close` and the constructor's failure path. Staging
        buffers are members of ``_buffers``, so iterating that list releases
        them too, including the replaced allocations growth deliberately kept.
        """

        for scratch in self._scratches:
            scratch.free()
        if self._int8_kv is not None:
            self._int8_kv.close()
            self._int8_kv = None
        for buffer in self._buffers:
            free(buffer)
        self._buffers.clear()
        self._staging.clear()
        self._rope_prefill_tables.clear()
        self._scratches.clear()
        self._kv.clear()
        self._caches.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._release_owned()

    @property
    def position(self) -> int:
        return self._position

    @property
    def layer_count(self) -> int:
        """Number of blocks this runner holds KV for.

        Read from the config rather than from the BF16 cache list so an INT8
        runner still reports the model's layer count; a consumer that then asks
        for a BF16 :meth:`shared_kv` view gets a named capability miss instead
        of an empty iteration that silently drafts nothing.
        """

        return len(self.weights.config.attention)

    def shared_kv(self, layer_index: int) -> Gemma4SharedKV:
        """Return a read view of one layer's BF16 KV cache.

        The Gemma 4 assistant (MTP) head allocates no KV of its own: each of its
        four blocks attends against one of this model's last two layers, sharing
        the buffers written here rather than receiving a copy. This is the read
        side of that binding -- a consumer gets the two buffer addresses, how
        many positions are live, and the geometry it needs to index them, and
        has no way to append.

        ``live`` is the position count after the most recent forward, which is
        what a shared reader must attend over. A consumer that reads while a
        forward is in flight would see the previous count, so the draft step is
        expected to run between forwards rather than concurrently with one.

        The INT8 per-token/head cache has no BF16 read view -- the direct INT8
        consumer owns its own layout -- so this refuses that storage by name
        rather than returning pointers into a cache that does not exist.
        """

        if self._int8_kv is not None:
            raise NotImplementedError(
                "shared_kv requires BF16 KV storage; the INT8 per-token/head "
                "cache exposes no BF16 read view for the assistant head"
            )
        if not 0 <= layer_index < len(self._kv):
            raise IndexError(
                f"layer {layer_index} is out of range for a model with "
                f"{len(self._kv)} blocks"
            )
        attention = self.weights.config.attention[layer_index]
        entry = self._kv[layer_index]
        return Gemma4SharedKV(
            layer_index=int(layer_index),
            key_cache=int(entry.key_cache),
            value_cache=int(entry.value_cache),
            capacity=int(self.capacity),
            live=int(self._position),
            num_kv_heads=int(attention.num_kv_heads),
            head_dim=int(attention.head_dim),
        )

    def hidden_state(self, row: int = -1) -> DeviceBuffer:
        """Return the raw hidden state after the last block, for one row.

        This is the tensor the assistant head's pre-projection consumes as
        ``h_backbone``. ``row`` defaults to the last row of the most recent
        forward, which is the position the next draft step predicts from; a
        negative index counts back from that row and a positive one from the
        start of the block.

        The row count is the most recent forward's, not the accumulated position
        count: a prefill of 512 followed by a decode of 1 leaves one valid row,
        and indexing the 513 accumulated positions would read stale hidden state
        from before the decode.

        The buffer is BF16 and ``hidden_size`` wide, and is the runner's own
        scratch: it is valid until the next forward pass overwrites it.
        """

        hidden = int(self.weights.config.hidden_size)
        rows = int(self._last_rows)
        if rows <= 0:
            raise ValueError("no forward pass has run, so there is no hidden state")
        index = int(row)
        if index < 0:
            index += rows
        if not 0 <= index < rows:
            raise IndexError(
                f"row {row} is out of range for the last forward's {rows} rows"
            )
        return DeviceBuffer(
            ptr=int(self._hidden.ptr) + index * hidden * _BF16_BYTES,
            nbytes=hidden * _BF16_BYTES,
        )

    def rewind(self, position: int) -> None:
        """Drop back to an earlier position without clearing the cache.

        A speculative verify pass appends a whole draft and then accepts only a
        prefix of it, so the cache has to give back the rejected tail. The K/V
        written past ``position`` is left in place rather than cleared: the next
        forward overwrites a position before reading it, so clearing would cost
        the same write as the token that replaces it.

        ``_last_rows`` deliberately survives, because the hidden rows of the
        forward that was just rewound are still in the scratch buffer and are
        still the rows the accepted prefix was computed from. That is what lets
        a caller draft from the last accepted position after rewinding. The next
        forward replaces both, so a draft must not be taken across one.
        """

        if self._closed:
            raise RuntimeError("runner is closed")
        target = int(position)
        if not 0 <= target <= self._position:
            raise ValueError(
                f"cannot rewind to {target} from position {self._position}"
            )
        self._position = target
        for index in range(len(self._kv)):
            entry = self._kv[index]
            self._kv[index] = Gemma4LayerKV(
                key_cache=entry.key_cache,
                value_cache=entry.value_cache,
                capacity=self.capacity,
                write_offset=target,
            )

    # --- cross-process KV state -----------------------------------------

    # The save/restore file schema this runner writes and accepts. Class
    # attributes, deliberately not dataclass fields: they are format constants,
    # not per-runner state.
    KV_STATE_SCHEMA = 1
    _KV_STATE_CHUNK_BYTES = 16 * 1024 * 1024

    def save_kv_state(
        self,
        path: str | Path,
        *,
        identity: Mapping[str, Any] | None = None,
        logits: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Serialize the live KV prefix to ``path`` for a later restore.

        The BF16 key and value planes of every layer are copied out up to the
        current position -- exactly the bytes a continuation's attention would
        read -- under a JSON header carrying the position, the per-layer plane
        sizes and the caller's ``identity`` mapping. ``logits``, the last
        row's FP32 logits, is stored alongside when given, so a restored
        caller can sample its first token without re-running the prefill.

        The file is a checkpoint of device bytes, not an engine input:
        :meth:`restore_kv_state` validates the capacity, the plane sizes and
        the identity, and refuses a mismatch rather than guessing.
        """
        if self._closed:
            raise RuntimeError("runner is closed")
        if self._int8_kv is not None:
            raise NotImplementedError(
                "save_kv_state requires the BF16 KV route; this runner holds "
                "an int8_per_token_head cache"
            )
        from hipengine.core.hip import HipMemcpyKind, get_hip_runtime

        runtime = get_hip_runtime()
        runtime.device_synchronize()
        position = int(self._position)
        if not 0 <= position <= self.capacity:
            raise RuntimeError(
                f"position {position} is outside capacity {self.capacity}"
            )
        if logits is not None:
            logits = np.ascontiguousarray(logits, dtype=np.float32)

        header = {
            "schema": self.KV_STATE_SCHEMA,
            "kind": "gemma4_kv_state",
            "capacity": int(self.capacity),
            "position": position,
            "plane_bytes": [int(cache.nbytes) for cache in self._caches],
            "identity": dict(identity) if identity is not None else None,
            "logits": None if logits is None else int(logits.size),
        }

        chunk = np.empty(self._KV_STATE_CHUNK_BYTES, dtype=np.uint8)
        chunk_ptr = host_array_ptr(chunk)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.write(json.dumps(header).encode("utf-8") + b"\n")
            for cache in self._caches:
                per_position = int(cache.nbytes) // int(self.capacity)
                remaining = position * per_position
                done = 0
                while remaining > 0:
                    take = min(self._KV_STATE_CHUNK_BYTES, remaining)
                    runtime.memcpy(
                        chunk_ptr, cache.ptr + done, take, HipMemcpyKind.DEVICE_TO_HOST
                    )
                    handle.write(memoryview(chunk)[:take])
                    done += take
                    remaining -= take
            if logits is not None:
                handle.write(logits.tobytes())
        return header

    def restore_kv_state(
        self,
        path: str | Path,
        *,
        identity: Mapping[str, Any] | None = None,
    ) -> np.ndarray | None:
        """Restore a KV prefix saved by :meth:`save_kv_state`.

        Returns the saved last-row logits when the file carries them. The
        planes are copied into this runner's own caches and the position is
        set to the saved one, so the next :meth:`forward` decodes from exactly
        the cached prefix without the prefill that produced it.

        The saved capacity, the per-layer plane sizes and -- when ``identity``
        is given -- the saved identity must match this runner. A file whose
        identity is null while the caller passes one is also a mismatch: the
        caller knows something the file does not. Every mismatch raises
        ``ValueError`` naming it; nothing is partially restored, because the
        validation runs before the first byte lands in a cache.
        """
        if self._closed:
            raise RuntimeError("runner is closed")
        if self._int8_kv is not None:
            raise NotImplementedError(
                "restore_kv_state requires the BF16 KV route; this runner "
                "holds an int8_per_token_head cache"
            )
        from hipengine.core.hip import HipMemcpyKind, get_hip_runtime

        runtime = get_hip_runtime()
        path = Path(path)
        with path.open("rb") as handle:
            header = json.loads(handle.readline().decode("utf-8"))
            if header.get("schema") != self.KV_STATE_SCHEMA:
                raise ValueError(
                    f"kv state schema {header.get('schema')!r} is not "
                    f"{self.KV_STATE_SCHEMA}"
                )
            if header.get("kind") != "gemma4_kv_state":
                raise ValueError(f"kv state kind {header.get('kind')!r} is unrecognized")
            if int(header["capacity"]) != int(self.capacity):
                raise ValueError(
                    f"kv state capacity {header['capacity']} does not match this "
                    f"runner's capacity {self.capacity}"
                )
            planes = [int(cache.nbytes) for cache in self._caches]
            if [int(size) for size in header["plane_bytes"]] != planes:
                raise ValueError(
                    f"kv state plane sizes {header['plane_bytes']} do not match "
                    f"this runner's {planes}"
                )
            saved_identity = header.get("identity")
            if identity is not None and saved_identity != dict(identity):
                raise ValueError(
                    f"kv state identity mismatch: file has {saved_identity!r}, "
                    f"caller asked for {dict(identity)!r}"
                )
            position = int(header["position"])
            if not 0 <= position <= self.capacity:
                raise ValueError(
                    f"kv state position {position} is outside this runner's "
                    f"capacity {self.capacity}"
                )

            # Validation is done: rewind the exposed state, then land the bytes.
            self.reset()
            chunk = np.empty(self._KV_STATE_CHUNK_BYTES, dtype=np.uint8)
            for cache in self._caches:
                per_position = int(cache.nbytes) // int(self.capacity)
                remaining = position * per_position
                done = 0
                while remaining > 0:
                    take = min(self._KV_STATE_CHUNK_BYTES, remaining)
                    data = handle.read(take)
                    if len(data) != take:
                        raise ValueError(
                            f"kv state file {path} ends inside its planes"
                        )
                    runtime.memcpy(
                        cache.ptr + done,
                        ctypes.cast(data, ctypes.c_void_p).value or 0,
                        take,
                        HipMemcpyKind.HOST_TO_DEVICE,
                    )
                    done += take
                    remaining -= take
            logits = None
            if header.get("logits") is not None:
                raw = handle.read(int(header["logits"]) * 4)
                if len(raw) != int(header["logits"]) * 4:
                    raise ValueError(f"kv state file {path} ends inside its logits")
                logits = np.frombuffer(raw, dtype=np.float32).copy()

        for index in range(len(self._kv)):
            entry = self._kv[index]
            self._kv[index] = Gemma4LayerKV(
                key_cache=entry.key_cache,
                value_cache=entry.value_cache,
                capacity=self.capacity,
                write_offset=position,
            )
        self._position = position
        return logits

    @property
    def normalized_hidden(self) -> DeviceBuffer:
        """Post-``output_norm`` states for the last block that asked for them.

        ``rows`` worth of rows are live; see :attr:`normalized_hidden_rows`.
        The buffer is device-resident so a multi-row lm_head can launch against
        it without a round trip. Reading 0 rows means the last forward did not
        ask for ``return_hidden``.
        """

        return self._normalized

    @property
    def normalized_hidden_rows(self) -> int:
        """How many rows of :attr:`normalized_hidden` are valid (0 if none)."""

        return self._normalized_hidden_rows

    def _validated_tokens(self, token_ids: Sequence[int]) -> list[int]:
        """Admission checks shared by :meth:`forward` and :meth:`forward_argmax`."""

        if self._closed:
            raise RuntimeError("runner is closed")
        tokens = [int(t) for t in token_ids]
        rows = len(tokens)
        if rows == 0:
            raise ValueError("token_ids must not be empty")
        if self._position + rows > self.capacity:
            raise ValueError(
                f"{rows} tokens from position {self._position} exceeds capacity {self.capacity}"
            )
        vocab = int(self.weights.config.vocab_size or 0)
        if vocab <= 0:
            raise ValueError("config carries no vocab_size, so logits cannot be sized")
        for token in tokens:
            if not 0 <= token < vocab:
                raise ValueError(f"token id {token} is outside the vocabulary of {vocab}")
        return tokens

    def forward(
        self,
        token_ids: Sequence[int],
        *,
        apply_softcap: bool = True,
        return_hidden: bool = False,
        logits_rows: int = 1,
        capture_layers: list[np.ndarray] | None = None,
        verification: bool = False,
    ) -> np.ndarray:
        """Run ``token_ids`` through the model and return the last row's logits.

        The tokens append to the sequence at the current position, so calling
        this repeatedly with single tokens is incremental decode and calling it
        once with a prompt is prefill. Both are the same computation.

        A prompt wider than ``max_block`` is forwarded as consecutive blocks.
        ``max_block`` bounds the per-layer scratch, which is a property of the
        widest block a device is asked to hold rather than of the context: at
        the real 26B artifact's default context the difference is 52.68 GB of
        scratch against 3.29 GB. Chunking is exact here -- the mask is built
        from absolute positions, and a one-token-at-a-time path is already
        bit-identical to a dense prefill.

        ``apply_softcap=False`` returns the raw projection, before
        ``final_logit_softcapping``. The cap is part of the model's output
        distribution and so is applied by default; the raw values are exposed
        because a saturated cap destroys the ordering information a parity
        comparison needs.

        ``return_hidden=True`` additionally keeps the final block's
        post-``output_norm`` states for **every** row of that block instead of
        only the last, exposed through :attr:`normalized_hidden` and
        :attr:`normalized_hidden_rows`. This is ``h`` -- the tensor llama.cpp
        names ``t_h_nextn`` -- which a multi-row lm_head consumes without
        recomputing. The last row's logits are unchanged by it: the norm writes
        each row independently, so the default path and this one agree
        byte-for-byte on the row the projection reads.

        ``logits_rows=k`` returns the last ``k`` rows' logits as a ``(k, vocab)``
        array instead of one row, which is what a speculative verify pass reads:
        it forwards a draft in one call and needs the target's own distribution
        at every drafted position, not just the last. ``k`` is bounded by
        ``max_logits_rows``, the constructor's bound on the projection buffers.
        Each returned row is the same computation a single-token forward would
        produce for that position: the mask is built from absolute positions, so
        a row never depends on how many rows accompany it.

        ``verification=True`` keeps a batched speculative block on the strict
        attention arithmetic used by target decode rather than the production
        prompt-prefill variants. The strict variant is named explicitly so a
        verify block does not fall onto the layer's automatic tiled route.

        ``capture_layers`` appends the residual stream after each block, as a
        ``(rows, hidden)`` BF16 array, to the list it is given. It is a
        diagnostic: it costs a device-to-host copy per layer, so a generation
        path does not pass it.
        """

        tokens = self._validated_tokens(token_ids)
        rows = len(tokens)

        wanted = int(logits_rows)
        if not 1 <= wanted <= self.max_logits_rows:
            raise ValueError(
                f"logits_rows {wanted} is outside 1..{self.max_logits_rows}, the "
                f"runner's projection buffer bound"
            )

        if wanted > rows:
            raise ValueError(f"logits_rows {wanted} exceeds this forward's {rows} rows")
        logits = None
        last_start = min(((rows - 1) // self.max_block) * self.max_block, rows - wanted)
        block_starts = [*range(0, last_start, self.max_block), last_start]
        for start in block_starts:
            # Only this loop's last block's logits are returned; every earlier
            # block's copy is overwritten on the next iteration, so its final
            # norm, projection and device-to-host copy are unread.
            last = start == last_start
            end = rows if last else min(start + self.max_block, last_start)
            block = tokens[start:end]
            # A full block opts into the dense WMMA prefill; a partial block or a
            # verify block stays on the strict route. An explicitly set
            # HIPENGINE_GGUF_WMMA_PREFILL wins outright inside the session.
            with _gemma4_block_wmma_session(
                not verification and len(block) == self.max_block
            ):
                logits = self._forward_block(
                    block,
                    apply_softcap=apply_softcap,
                    needs_logits=last,
                    return_hidden=return_hidden,
                    logits_rows=wanted if last else 1,
                    capture_layers=capture_layers,
                    verification=verification,
                )
        assert logits is not None
        return logits

    def forward_argmax(
        self,
        token_ids: Sequence[int],
        *,
        apply_softcap: bool = True,
    ) -> int:
        """Run ``token_ids`` and return the greedy token without the vocab copy.

        Identical computation, admission, and chunking to :meth:`forward`, but
        the collect phase argmaxes on the device behind the softcap and
        transfers the winning index — 16 bytes instead of the full-vocab logits
        (D10's greedy route; the host-side cost it removes is the ~0.118 ms
        1 MB sync D2H plus the 0.016 ms ``np.argmax`` measured on this box).

        The token equals ``int(np.argmax(forward(...)))`` exactly: the battery
        in ``tests/test_gpu_gemma4_argmax_kernel.py`` pins first-maximum index
        equality, bitwise value equality, NaN semantics, and the chained
        after-softcap comparator against what the host path reads back. The
        full-logits path stays as :meth:`forward` for diagnostics, gates, and
        any non-greedy sampler.
        """

        tokens = self._validated_tokens(token_ids)
        rows = len(tokens)
        token = -1
        last_start = ((rows - 1) // self.max_block) * self.max_block
        for start in range(0, rows, self.max_block):
            block = tokens[start : start + self.max_block]
            with _gemma4_block_wmma_session(len(block) == self.max_block):
                token = self._forward_block(
                    block,
                    apply_softcap=apply_softcap,
                    needs_logits=start == last_start,
                    collect_argmax=True,
                )
        assert token is not None
        return int(token)

    @contextmanager
    def _q8_mmq_prefill_session(self) -> Iterator[None]:
        """Admit this block's dense Q8_0 projections to the guarded MMQ chain.

        The owners and the guard live in the shared linear dispatch, which reads
        the policy off the session rather than re-resolving one from the
        registry. So a model carries its own crossover map and its own
        threshold without colliding with another model that shares its file
        type, and a shape the map does not name keeps its exact owner.
        """

        from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import (
            Q8MMQPrefillPolicy,
            build_gguf_q8_0_mmq_prefill,
            q8_mmq_d4x3_nbytes,
        )
        from hipengine.runtime.gguf_linear import q8_mmq_prefill_session

        if self._q8_mmq_policy is None:
            self._q8_mmq_policy = Q8MMQPrefillPolicy(
                min_rows=GEMMA4_Q8_MMQ_MIN_ROWS,
                max_rows=GEMMA4_Q8_MMQ_MAX_ROWS,
                risk_threshold=GEMMA4_Q8_MMQ_RISK_THRESHOLD,
                max_out_features=GEMMA4_Q8_MMQ_MAX_OUT_FEATURES,
            )
        if self._q8_mmq_library is None:
            self._q8_mmq_library = build_gguf_q8_0_mmq_prefill(load=True)
        if self._q8_mmq_workspace is None:
            # Sized from the policy's own shapes, not from the block: the policy
            # is what bounds which shapes reach the chain, so its widest entry
            # is the widest activation the workspace can ever be asked to hold.
            rows = min(
                int(self.max_block or 0) or GEMMA4_Q8_MMQ_MAX_ROWS,
                GEMMA4_Q8_MMQ_MAX_ROWS,
            )
            hidden = int(self.weights.config.hidden_size)
            widest_in = max([key[0] for key in GEMMA4_Q8_MMQ_MIN_ROWS] + [hidden])
            widest_out = max([key[1] for key in GEMMA4_Q8_MMQ_MIN_ROWS] + [hidden])
            self._q8_mmq_workspace = self._alloc(q8_mmq_d4x3_nbytes(rows, widest_in))
            self._q8_mmq_risk_count = self._alloc(4)
            self._q8_mmq_risk_indices = self._alloc(rows * widest_out * 4)
        with q8_mmq_prefill_session(
            workspace_ptr=self._q8_mmq_workspace.ptr,
            workspace_nbytes=self._q8_mmq_workspace.nbytes,
            risk_count_ptr=self._q8_mmq_risk_count.ptr,
            risk_count_nbytes=self._q8_mmq_risk_count.nbytes,
            risk_indices_ptr=self._q8_mmq_risk_indices.ptr,
            risk_indices_nbytes=self._q8_mmq_risk_indices.nbytes,
            policy=self._q8_mmq_policy,
            library=self._q8_mmq_library,
        ):
            yield

    def _forward_block(
        self,
        tokens: Sequence[int],
        *,
        apply_softcap: bool = True,
        needs_logits: bool = True,
        return_hidden: bool = False,
        collect_argmax: bool = False,
        logits_rows: int = 1,
        capture_layers: list[np.ndarray] | None = None,
        verification: bool = False,
    ) -> np.ndarray | int:
        """Run one block of at most ``max_block`` tokens; see :meth:`forward`.

        The dense WMMA prefill opt-in is scoped by the caller, one block at a
        time, so this stays a plain session nest.
        """

        with self._q8_mmq_prefill_session():
            return self._forward_block_inner(
                tokens,
                apply_softcap=apply_softcap,
                needs_logits=needs_logits,
                return_hidden=return_hidden,
                collect_argmax=collect_argmax,
                logits_rows=logits_rows,
                capture_layers=capture_layers,
                verification=verification,
            )

    def _forward_block_inner(
        self,
        tokens: Sequence[int],
        *,
        apply_softcap: bool = True,
        needs_logits: bool = True,
        return_hidden: bool = False,
        collect_argmax: bool = False,
        logits_rows: int = 1,
        capture_layers: list[np.ndarray] | None = None,
        verification: bool = False,
    ) -> np.ndarray | int:
        """The block body, run under whatever session :meth:`_forward_block` set.

        ``needs_logits`` is False for a block whose output ``forward`` discards.
        ``verification``, ``logits_rows`` and ``capture_layers`` are per-call: a
        speculative verify pass keeps its batched rows on the strict attention
        kernel and projects the trailing ``logits_rows`` logit rows.

        The body splits into three phases so the decode-graph path can drive
        them separately: stage (position-dependent content, enqueued), launch
        (every device kernel, on one stream), collect (the logits copy and the
        host bookkeeping). This default path runs them in their historical
        stream order on the default stream with the actual position.
        ``collect_argmax=True`` swaps the collect phase for the device-argmax
        variant :meth:`_collect_block_argmax`; the launch phase is identical.
        """

        config = self.weights.config
        rows = len(tokens)
        if rows > self.max_block:
            raise ValueError(f"{rows} tokens exceeds max_block {self.max_block}")
        projected = int(logits_rows)
        if not 1 <= projected <= rows:
            raise ValueError(
                f"logits_rows {projected} is outside 1..{rows}, this block's rows"
            )

        tables, masks = self._stage_block_content(tokens, stream=0)
        self._launch_block(
            tokens,
            tables=tables,
            masks=masks,
            kv_write_offset=self._position,
            key_begin_at=None,
            stream=0,
            stream_moe=-1,
            needs_logits=needs_logits,
            return_hidden=return_hidden,
            logits_rows=projected,
            capture_layers=capture_layers,
            verification=verification,
        )
        if collect_argmax:
            return self._collect_block_argmax(
                tokens,
                apply_softcap=apply_softcap,
                needs_logits=needs_logits,
            )
        return self._collect_block(
            tokens,
            apply_softcap=apply_softcap,
            needs_logits=needs_logits,
        )

    def _collect_block_argmax(
        self,
        tokens: Sequence[int],
        *,
        apply_softcap: bool,
        needs_logits: bool,
        stream: int | None = None,
    ) -> int:
        """Softcap, argmax on the device, and bring home just the token.

        Same admission and stream contract as :meth:`_collect_block`; only the
        transfer changes. The argmax kernel is enqueued behind the softcap on
        the same stream, then a sync copy moves 16 bytes instead of the vocab.
        The drain the copy waits for is the token dependency itself — the
        sampler needs the result before the next step can launch — so what
        shrinks is the transfer and the host ``np.argmax``, not the wait.
        Non-final blocks mirror the logits path: no collect, position advance
        only, and the sentinel return is never read (the caller keeps only the
        last block's result).
        """

        rows = len(tokens)
        if not needs_logits:
            self._position += rows
            self._last_rows = rows
            return -1

        vocab = int(self.weights.config.vocab_size or 0)
        cap = self.weights.config.final_logit_softcapping
        launch_stream = int(stream) if stream is not None else 0
        if apply_softcap and cap:
            gemma4_logit_softcap_f32(
                self._logits.ptr,
                vocab,
                float(np.float32(cap)),
                stream=launch_stream,
            )
        gemma4_logit_argmax_f32(
            self._logits.ptr,
            vocab,
            self._argmax_out.ptr,
            scratch_ptr=self._argmax_scratch.ptr,
            scratch_blocks=self._argmax_scratch.nbytes // 12,
            stream=launch_stream,
        )

        if launch_stream != 0:
            from hipengine.core.hip import get_hip_runtime

            get_hip_runtime().stream_synchronize(launch_stream)

        raw = np.empty(4, dtype=np.int64)  # 16 bytes: i64 index + f32 value
        copy_device_to_host(host_array_ptr(raw), self._argmax_out, 16)
        self._position += rows
        self._last_rows = rows
        return int(raw[0])

    def _stage_block_content(
        self,
        tokens: Sequence[int],
        *,
        stream: int,
        keys_extent: int | None = None,
        key_begin_at: Callable[[Gemma4AttentionGeometry], int] | None = None,
    ) -> tuple[
        dict[Gemma4RopeConfig, tuple[DeviceBuffer, DeviceBuffer]],
        dict[int | None, DeviceBuffer],
    ]:
        """Compute this block's position-dependent content and enqueue uploads.

        Token ids, the rope tables for this block's positions and the keep-mask
        change every block, but each lands in a reusable staging buffer whose
        device pointer does not, so they are enqueued on ``stream`` ahead of
        every launch that reads them.

        ``keys_extent`` overrides the mask's column count: the decode-graph
        capture freezes it to its context bucket so the captured attention
        nodes never change shape, while the mask's *content* still describes
        this block's actual positions.

        ``key_begin_at`` supplies the captured launch's frozen first key.
        Otherwise masks use this block's actual sliding read range. Both the
        column extent and origin must match the launch that consumes the mask.
        """

        config = self.weights.config
        rows = len(tokens)
        start = self._position
        # Enqueued, not sync: the embedding kernel launched after this returns
        # sees the ids in stream order, while the host keeps submitting instead
        # of draining the previous step's work (0.19 ms/step of blocking
        # hipMemcpy, D8 row).
        ids = np.ascontiguousarray(tokens, dtype=np.int64)
        enqueue_host_to_device(
            self._token_ids, host_array_ptr(ids), ids.nbytes, stream=stream
        )

        positions = np.arange(start, start + rows, dtype=np.int64)
        # Tables depend only on the rope contract and this block's positions,
        # the mask only on the sliding window and this block's range, so each
        # distinct geometry is staged once per block instead of once per layer:
        # layers of the same kind uploaded byte-identical copies 30 times per
        # decode step on the production model. Both config dataclasses are
        # frozen, so the rope config is a value key, and no buffer is rewritten
        # while the block runs — the next block's copies are stream-ordered
        # behind this block's kernels, exactly as the per-layer staging was.
        tables: dict[Gemma4RopeConfig, tuple[DeviceBuffer, DeviceBuffer]] = {}
        masks: dict[int | None, DeviceBuffer] = {}
        extent = start + rows if keys_extent is None else int(keys_extent)
        if extent < start + rows:
            raise ValueError(
                f"keys_extent {extent} must cover this block's {start + rows} key positions"
            )
        for index in range(len(self.weights.layers)):
            attention = config.geometry(index)
            # The rope tables are F32 (rows, head_dim) with the rotated half
            # doubled, which is the layout gemma4_partial_rotary_bf16 indexes.
            staged = tables.get(attention.rope)
            if staged is None:
                cached = self._rope_prefill_tables.get(attention.rope) if rows > 1 else None
                if cached is not None and cached[:2] == (start, rows):
                    cos, sin = cached[2:]
                else:
                    cos, sin = gemma4_rope_cos_sin_tables(attention.rope, positions)
                    if rows > 1:
                        self._rope_prefill_tables[attention.rope] = (start, rows, cos, sin)
                # Upload even on a host hit: decode or graph staging may have
                # overwritten these reusable device buffers since the prefill.
                staged = (
                    self._stage_upload(
                        f"cos{attention.rope}",
                        np.ascontiguousarray(cos, dtype=np.float32),
                        stream=stream,
                    ),
                    self._stage_upload(
                        f"sin{attention.rope}",
                        np.ascontiguousarray(sin, dtype=np.float32),
                        stream=stream,
                    ),
                )
                tables[attention.rope] = staged
            if attention.sliding_window not in masks:
                masks[attention.sliding_window] = self._stage_upload(
                    f"mask{attention.sliding_window}",
                    _keep_mask(
                        attention, start, rows, extent,
                        key_begin=None if key_begin_at is None else key_begin_at(attention),
                    ),
                    stream=stream,
                )
        return tables, masks

    def _launch_block(
        self,
        tokens: Sequence[int],
        *,
        tables: dict[Gemma4RopeConfig, tuple[DeviceBuffer, DeviceBuffer]],
        masks: dict[int | None, DeviceBuffer],
        kv_write_offset: int,
        key_begin_at: Callable[[Gemma4AttentionGeometry], int] | None,
        stream: int = 0,
        stream_moe: int = -1,
        needs_logits: bool = True,
        return_hidden: bool = False,
        logits_rows: int = 1,
        capture_layers: list[np.ndarray] | None = None,
        verification: bool = False,
    ) -> None:
        """Enqueue every device launch for this block on ``stream``.

        No host copy and no position bookkeeping happens here: the caller
        staged the content first and collects the logits after. Two parameters
        are position-dependent and are passed in rather than read from the
        runner so the decode-graph capture can freeze them to a context bucket:
        ``kv_write_offset`` (where this block's keys and values are appended,
        which also bounds the keys each layer attends) and ``key_begin_at``
        (how far into the cache each layer's attention starts; ``None`` takes
        the actual sliding-window range for ``kv_write_offset``).

        ``verification`` keeps this block on the strict attention variant. It
        names ``gemma4_plain`` explicitly rather than leaving the variants at
        ``None``, because ``None`` selects the layer's own automatic tiled/AOTriton
        route, which is not what a target-verify block must run.
        """

        config = self.weights.config
        rows = len(tokens)
        vocab = int(config.vocab_size or 0)
        hidden = config.hidden_size

        launch_gguf_embedding(
            self.weights.embed_tokens,
            self._token_ids.ptr,
            self._hidden.ptr,
            rows,
            hidden,
            vocab,
            stream=stream,
        )
        # sqrt(hidden_size), applied to the residual stream rather than folded
        # into the first norm: the norm is scale-invariant and would hide it,
        # but every layer adds back into this stream.
        gemma4_scale_bf16(
            self._hidden.ptr,
            self._hidden.ptr,
            rows,
            hidden,
            float(config.embed_scale),
            stream=stream,
        )

        # INT8 storage stages this block's append positions and live counts once
        # for every layer, on the same stream, ahead of the writer and consumer
        # launches that read them.
        int8_block = None
        if self._int8_kv is not None:
            int8_block = self._int8_kv.begin_block(
                write_offset=kv_write_offset, rows=rows, stream=stream
            )

        for index, layer in enumerate(self.weights.layers):
            attention = config.geometry(index)
            cos_buf, sin_buf = tables[attention.rope]
            mask_buf = masks[attention.sliding_window]
            key_begin = (
                _sliding_read_range(attention, kv_write_offset, rows)
                if key_begin_at is None
                else int(key_begin_at(attention))
            )
            gemma4_layer_forward_bf16(
                self._hidden.ptr,
                cos_buf.ptr,
                sin_buf.ptr,
                mask_buf.ptr,
                layer,
                scratch=self._scratches[index],
                kv=None
                if int8_block is not None
                else Gemma4LayerKV(
                    key_cache=self._kv[index].key_cache,
                    value_cache=self._kv[index].value_cache,
                    capacity=self.capacity,
                    write_offset=kv_write_offset,
                ),
                int8_kv=None
                if int8_block is None
                else self._int8_kv.layer_kv(index, int8_block),
                rows=rows,
                eps=config.rms_norm_eps,
                key_begin=key_begin,
                # ``_keep_mask`` builds exactly ``key <= query`` plus the window
                # bound, and the window bound is vacuous while the attended range
                # is no wider than the window. The layer re-checks the range; this
                # flag is the mask's *semantics*, which only the builder knows.
                attention_mask_is_causal=True,
                # A global layer's window is the whole attended range, so saying
                # so lets the kernel trim the trailing masked run as it already
                # trims a sliding layer's leading one. A sliding layer keeps its
                # own window.
                window=(
                    0 if key_begin_at is not None else (
                        kv_write_offset + rows - key_begin
                        if attention.sliding_window is None
                        else int(attention.sliding_window)
                    )
                ),
                prefill_attention_variants=(
                    ("gemma4_plain",) if verification else self.prefill_attention_variants
                ),
                stream=stream,
                stream_moe=stream_moe,
            )
            if capture_layers is not None:
                residual = np.empty((rows, hidden), dtype=np.uint16)
                copy_device_to_host(
                    host_array_ptr(residual), self._hidden, residual.nbytes
                )
                capture_layers.append(residual)

        # The caller wants one next-token distribution, so only the final block
        # computes one. Skipping it here drops a norm, a vocab-wide projection
        # and a device-to-host copy per discarded block: 1 at 1024 tokens and 7
        # at 4096 under the default 512-token block.
        if not needs_logits:
            return

        # ``return_hidden`` normalizes the whole block so a multi-row lm_head
        # can read the states already computed; otherwise the projection reads
        # the trailing ``logits_rows`` rows, one for the default next-token path
        # and the verify block's rows for a speculative pass. Each row is
        # independent under rmsnorm, so the row the projection consumes is
        # identical either way -- asserted byte-for-byte by the live M2 test.
        projected = int(logits_rows)
        if not 1 <= projected <= rows:
            raise ValueError(
                f"logits_rows {projected} is outside 1..{rows}, this block's rows"
            )
        if return_hidden:
            norm_rows, norm_input = rows, self._hidden.ptr
            self._normalized_hidden_rows = rows
            head_rows = projected
            head_input = self._normalized.ptr + (rows - projected) * hidden * _BF16_BYTES
        else:
            self._normalized_hidden_rows = 0
            norm_rows = projected
            norm_input = self._hidden.ptr + (rows - projected) * hidden * _BF16_BYTES
            head_rows = projected
            head_input = self._normalized.ptr
        gemma4_rmsnorm_f32w_bf16(
            norm_input,
            self.weights.final_norm.buffer.ptr,
            self._normalized.ptr,
            norm_rows,
            hidden,
            config.rms_norm_eps,
            stream=stream,
        )
        head = self.weights.lm_head or self.weights.embed_tokens
        # ``_collect_block`` copies this many rows home; the decode-graph path
        # reads the same field so a capture that leaves it at one still copies
        # exactly the row it projected.
        self._last_logits_rows = head_rows
        launch_gguf_linear(
            head,
            head_input,
            self._logits.ptr,
            head_rows,
            hidden,
            vocab,
            output_dtype="f32",
            stream=stream,
        )

    def _collect_block(
        self,
        tokens: Sequence[int],
        *,
        apply_softcap: bool,
        needs_logits: bool,
        stream: int | None = None,
    ) -> np.ndarray:
        """Copy this block's logits to the host and advance the position.

        ``stream`` names a non-default stream the launches ran on: the copy
        below is synchronous on the default stream, so work replayed on a
        capture stream has to drain there first.
        """

        rows = len(tokens)
        if not needs_logits:
            self._position += rows
            self._last_rows = rows
            return np.empty(0, dtype=np.float32)

        vocab = int(self.weights.config.vocab_size or 0)
        # The most recent launch records how many rows it projected: one for the
        # default next-token path, the verify block's trailing rows otherwise.
        # The decode-graph path reads the same field, so a capture that projected
        # one row copies exactly one.
        projected = max(1, int(self._last_logits_rows))
        # The cap is part of the model's output, applied here rather than in
        # the sampler so every consumer sees the distribution the model
        # defines (both CPU references do the same). It now runs on the
        # device: the host np.tanh over the full vocab measured 0.528 ms per
        # decode step — about a third of the measured host wall gap (X7 row)
        # — so the kernel is enqueued on the block's stream against the
        # logits buffer before the copy below. The device tanhf rounds
        # independently of numpy's libm; tests/test_gpu_gemma4_softcap_kernel
        # pins the argmax / tie / saturation contract on adversarial
        # batteries, and the workload parity tests pin the tokens.
        cap = self.weights.config.final_logit_softcapping
        if apply_softcap and cap:
            gemma4_logit_softcap_f32(
                self._logits.ptr,
                projected * vocab,
                float(np.float32(cap)),
                stream=int(stream) if stream is not None else 0,
            )

        if stream is not None and int(stream) != 0:
            from hipengine.core.hip import get_hip_runtime

            get_hip_runtime().stream_synchronize(int(stream))

        shape = (projected, vocab) if projected > 1 else (vocab,)
        logits = np.empty(shape, dtype=np.float32)
        copy_device_to_host(host_array_ptr(logits), self._logits, logits.nbytes)
        self._position += rows
        self._last_rows = rows
        return logits

    def _stage_upload(self, name: str, values: np.ndarray, *, stream: int = 0) -> DeviceBuffer:
        """Enqueue an upload of ``values`` into the reusable staging buffer."""

        buffer = self._staging_buffer(name, values.nbytes)
        # Enqueued on the caller's stream (the default stream for the normal
        # path, the capture stream for the decode graph): readers launched
        # after this return see the data in stream order, and the next step's
        # upload of the same buffer lands behind this block's readers. The sync
        # hipMemcpy here was draining the whole queued forward mid-pass -- six
        # times per decode step, 0.55 ms of host stall for KB-scale buffers
        # (D8 row; screen in scratch/d8_async_h2d_screen.py).
        enqueue_host_to_device(buffer, host_array_ptr(values), values.nbytes, stream=stream)
        return buffer

    def next_token(self, logits: np.ndarray, *, temperature: float = 0.0) -> int:
        """Pick the next token from ``logits``.

        ``temperature <= 0`` is greedy. The logits are expected to have already
        had ``final_logit_softcapping`` applied by :meth:`forward`, so this does
        not reapply it.
        """

        values = np.asarray(logits, dtype=np.float32).reshape(-1)
        if temperature <= 0:
            return int(np.argmax(values))
        scaled = values / np.float32(temperature)
        scaled -= scaled.max()
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(probabilities.size, p=probabilities))


def _layer_geometry(attention: Gemma4AttentionGeometry) -> Gemma4LayerGeometry:
    """Narrow the config's attention geometry to the fields the layer reads."""

    return Gemma4LayerGeometry(
        num_heads=attention.num_heads,
        num_kv_heads=attention.num_kv_heads,
        head_dim=attention.head_dim,
        scale=attention.scale,
        k_eq_v=attention.k_eq_v,
        sliding_window=attention.sliding_window,
    )


def _sliding_read_range(
    attention: Gemma4AttentionGeometry,
    start: int,
    rows: int,
) -> int:
    """First cached key this block must walk, or 0 when nothing is skipped.

    A sliding layer's keep-mask zeroes every key outside its window, but the
    mask is full width, so the attention kernel walks the whole live context
    and the walk is what costs: measured on the RX 7900 XTX, the decode kernel's
    time tracks ``keys``, not the number of live keys. Gemma 4 has 25 sliding
    layers of 30, so at context 4096 that is 3073 keys walked per layer whose
    weight is exactly zero. Prefill pays that per row, so its attention cost
    grows with the square of the prompt length while a sliding layer can never
    need more than ``window`` keys for any one row.

    Skipping them is bit-exact. A masked key contributes ``exp(-inf) = 0`` to
    the denominator and the same to the weighted sum, and dropping terms whose
    value is zero leaves the surviving terms in their original order, so the
    reduction is unchanged rather than merely close. The one thing that has to
    hold is that the skipped keys are *exactly* the masked ones; the unit tests
    pin that against the mask itself for a range of window, context and block
    lengths.

    The bound is the first row's. A block's rows sit at ``start .. start + rows
    - 1``, and a sliding layer keeps key ``k`` for query ``q`` only while
    ``q - k < window``, so ``start`` is the earliest query in the block and
    ``start - window + 1`` is the earliest key any row of it can read. Later
    rows have later windows and cannot need anything earlier, so this is exact
    for the block rather than a heuristic. For a one-row block it reduces to
    ``live - window``, which is what the decode path has always used.

    The mask has to move with this. The kernel indexes it as ``keep_mask + token
    * keys`` with ``keys`` shortened by ``key_begin``, so ``_keep_mask`` builds
    exactly that many columns starting at ``key_begin`` and the layer wrapper
    passes the mask base unshifted.
    """

    window = attention.sliding_window
    if window is None:
        return 0
    window = int(window)
    if window <= 0:
        raise ValueError(f"sliding_window must be positive, got {window}")
    return max(0, start - window + 1)


def _keep_mask(
    attention: Gemma4AttentionGeometry,
    start: int,
    rows: int,
    keys_extent: int | None = None,
    *,
    key_begin: int | None = None,
) -> np.ndarray:
    """Build the ``(rows, keys - key_begin)`` uint8 keep-mask for one block.

    A position may attend to a key at or before it, and on a sliding layer only
    within ``sliding_window`` of it. The mask is shortened to the cached range
    this block attends: column 0 is key ``key_begin`` and the last column is key
    ``keys - 1``, where ``keys`` is ``start + rows`` by default or ``keys_extent``
    when the decode-graph capture freezes the shape to its context bucket.
    Columns past the live range are kept False, so one bucket-shaped mask reads
    as the exact mask for every position inside that bucket.
    """

    keys = start + rows if keys_extent is None else int(keys_extent)
    if keys < start + rows:
        raise ValueError(f"keys_extent {keys} must cover {start + rows} key positions")
    # Shortened to the block's own read range: column 0 is key ``key_begin``,
    # which is the same first key ``_sliding_read_range`` hands the layer and the
    # kernel indexes as ``keep_mask + token * keys`` with ``keys`` shortened by
    # the same amount. A full-width mask would address the wrong row from the
    # second row on.
    first_live = _sliding_read_range(attention, start, rows)
    key_begin = first_live if key_begin is None else int(key_begin)
    if not 0 <= key_begin <= first_live:
        raise ValueError(f"key_begin {key_begin} must be in 0..{first_live}")
    queries = np.arange(start, start + rows, dtype=np.int64)[:, None]
    key_positions = np.arange(key_begin, keys, dtype=np.int64)[None, :]
    keep = key_positions <= queries
    if attention.sliding_window is not None:
        keep &= (queries - key_positions) < int(attention.sliding_window)
    return np.ascontiguousarray(keep.astype(np.uint8))
