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

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
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
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import gemma4_attention_shared_bytes
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rope import gemma4_rope_cos_sin_tables
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
    Gemma4LayerGeometry,
    Gemma4LayerKV,
    Gemma4LayerPointers,
    Gemma4LayerScratch,
    gemma4_layer_forward_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
    gemma4_rmsnorm_f32w_bf16,
    gemma4_scale_bf16,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.gemma4_gguf import Gemma4GGUFConfig, gemma4_gguf_config_from_metadata
from hipengine.loading.gemma4_gguf_device import (
    Gemma4GGUFDeviceWeight,
    Gemma4GGUFWeightSpec,
    materialize_gemma4_gguf_device_weight,
    plan_gemma4_gguf_resident_specs,
)
from hipengine.loading.materialize import DeviceTensorAllocation, load_host_array_to_device_as_dtype
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import launch_gguf_linear

_BF16_BYTES = 2
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
# Raising this to 1024 to match llama.cpp's ubatch was measured and does not pay.
# The argument for it was weight reuse -- a 1024-token block routes about 64
# tokens per expert against 32, halving the number of passes over the expert
# weights. But the int8 MMQ path is compute-bound on the WMMA units rather than
# memory-bound on those weights, so halving the reads buys almost nothing:
#
#     prompt 1024   block 512: 1373/1367/1367     block 1024: 1351/1397/1397
#     prompt 2048   block 512: 1118/1119          block 1024: 1141/1131  (+1.5%)
#     decode        block 512: 38.86-39.09        block 1024: 37.96-38.15  (-2%)
#
# That is a wash, and it costs a larger per-layer scratch, so 512 stays. The
# reuse argument would apply to a memory-bound prefill owner, not this one.
DEFAULT_PREFILL_BLOCK = 512

# Which layer field each artifact slot feeds. The slot names are the loader's;
# the fields are the layer's. Kept as one table so the two cannot drift apart
# silently.
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
    "ffn_gate": "mlp_gate_proj",
    "ffn_up": "mlp_up_proj",
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
    "o_proj",
    "mlp_gate_proj",
    "mlp_up_proj",
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

    info = reader.info
    tensor_names = tuple(t.name for t in info.tensors)
    if hf_config is not None:
        config = gemma4_text_config_from_hf(hf_config)
    else:
        gguf = gemma4_gguf_config_from_metadata(info)
        config = gemma4_text_config_from_gguf(gguf, tensor_names=tensor_names)
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

    for slot, field_name in _SLOT_TO_FIELD.items():
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


# The dense Q8_0 projections, admitted to the guarded d4x3 MMQ chain per shape.
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
# min_rows is per shape because the chain loses below 512 rows (0.45x-0.55x),
# so the 64-row tail keeps the exact owner. (2112, 2816) is absent
# deliberately: 2112 is not a multiple of 128, so the d4 packing cannot serve it
# and that shape keeps its exact owner too.
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
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    _buffers: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _scratches: list[Gemma4LayerScratch] = field(default_factory=list, repr=False)
    _kv: list[Gemma4LayerKV] = field(default_factory=list, repr=False)
    _caches: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _staging: dict[str, tuple[DeviceBuffer, int]] = field(default_factory=dict, repr=False)
    _position: int = field(default=0, repr=False)
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
        if self.max_block <= 0:
            self.max_block = min(self.capacity, DEFAULT_PREFILL_BLOCK)
        if self.max_block > self.capacity:
            raise ValueError("max_block must not exceed capacity")
        for attention in config.attention:
            gemma4_attention_shared_bytes(head_dim=attention.head_dim, keys=self.capacity)

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
        # Every buffer taken below is owned by this runner, so a failure
        # partway through construction must release the ones already taken
        # rather than leaving them to the garbage collector, which does not own
        # device memory. The scratch objects are in the same list: they hold no
        # device buffers until a forward pass asks for one, but cleanup still
        # runs over them for symmetry with close().
        try:
            self._token_ids = self._alloc(self.max_block * _I64_BYTES)
            self._hidden = self._alloc(self.max_block * hidden * _BF16_BYTES)
            self._normalized = self._alloc(hidden * _BF16_BYTES)
            self._logits = self._alloc(int(config.vocab_size or 0) * _F32_BYTES)

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
                    )
                )
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
        except BaseException:
            # Mark closed first so a later close() cannot free the same buffers
            # a second time, then release everything taken so far.
            self._closed = True
            self._release_owned()
            raise

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
        for buffer in self._buffers:
            free(buffer)
        self._buffers.clear()
        self._staging.clear()
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

    def forward(self, token_ids: Sequence[int], *, apply_softcap: bool = True) -> np.ndarray:
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
        """

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

        logits = None
        for start in range(0, rows, self.max_block):
            logits = self._forward_block(
                tokens[start : start + self.max_block], apply_softcap=apply_softcap
            )
        assert logits is not None
        return logits

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
        self, tokens: Sequence[int], *, apply_softcap: bool = True
    ) -> np.ndarray:
        """Run one block of at most ``max_block`` tokens; see :meth:`forward`."""

        with self._q8_mmq_prefill_session():
            return self._forward_block_inner(tokens, apply_softcap=apply_softcap)

    def _forward_block_inner(
        self, tokens: Sequence[int], *, apply_softcap: bool = True
    ) -> np.ndarray:
        """The block body, run under whatever session :meth:`_forward_block` set."""

        config = self.weights.config
        rows = len(tokens)
        if rows > self.max_block:
            raise ValueError(f"{rows} tokens exceeds max_block {self.max_block}")
        vocab = int(config.vocab_size or 0)

        hidden = config.hidden_size
        start = self._position

        # --- embedding ------------------------------------------------------
        ids = np.ascontiguousarray(tokens, dtype=np.int64)
        copy_host_to_device(
            self._token_ids, host_array_ptr(ids), ids.nbytes
        )
        launch_gguf_embedding(
            self.weights.embed_tokens,
            self._token_ids.ptr,
            self._hidden.ptr,
            rows,
            hidden,
            vocab,
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
        )

        # --- layers ---------------------------------------------------------
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
        for index, layer in enumerate(self.weights.layers):
            attention = config.geometry(index)
            # The rope tables are F32 (rows, head_dim) with the rotated half
            # doubled, which is the layout gemma4_partial_rotary_bf16 indexes.
            staged = tables.get(attention.rope)
            if staged is None:
                cos, sin = gemma4_rope_cos_sin_tables(attention.rope, positions)
                staged = (
                    self._stage_upload(
                        f"cos{attention.rope}",
                        np.ascontiguousarray(cos, dtype=np.float32),
                    ),
                    self._stage_upload(
                        f"sin{attention.rope}",
                        np.ascontiguousarray(sin, dtype=np.float32),
                    ),
                )
                tables[attention.rope] = staged
            cos_buf, sin_buf = staged
            mask_buf = masks.get(attention.sliding_window)
            if mask_buf is None:
                mask_buf = self._stage_upload(
                    f"mask{attention.sliding_window}",
                    _keep_mask(attention, start, rows),
                )
                masks[attention.sliding_window] = mask_buf
            gemma4_layer_forward_bf16(
                self._hidden.ptr,
                cos_buf.ptr,
                sin_buf.ptr,
                mask_buf.ptr,
                layer,
                scratch=self._scratches[index],
                kv=Gemma4LayerKV(
                    key_cache=self._kv[index].key_cache,
                    value_cache=self._kv[index].value_cache,
                    capacity=self.capacity,
                    write_offset=start,
                ),
                rows=rows,
                eps=config.rms_norm_eps,
                key_begin=_sliding_read_range(attention, start, rows),
                # ``_keep_mask`` builds exactly ``key <= query`` plus the window
                # bound, and the window bound is vacuous while the attended range
                # is no wider than the window. The layer re-checks the range; this
                # flag is the mask's *semantics*, which only the builder knows.
                attention_mask_is_causal=True,
            )

        # --- final norm and lm head -----------------------------------------
        # Only the last row is needed: the caller wants the next-token
        # distribution, and the earlier rows' logits are never read.
        last = (rows - 1) * hidden * _BF16_BYTES
        gemma4_rmsnorm_f32w_bf16(
            self._hidden.ptr + last,
            self.weights.final_norm.buffer.ptr,
            self._normalized.ptr,
            1,
            hidden,
            config.rms_norm_eps,
        )
        head = self.weights.lm_head or self.weights.embed_tokens
        launch_gguf_linear(
            head,
            self._normalized.ptr,
            self._logits.ptr,
            1,
            hidden,
            vocab,
            output_dtype="f32",
        )

        logits = np.empty(vocab, dtype=np.float32)
        copy_device_to_host(host_array_ptr(logits), self._logits, logits.nbytes)
        self._position += rows
        # Applied here rather than in the sampler so that every consumer of the
        # model's output sees the distribution the model defines. Both the dense
        # and the streaming CPU references do the same.
        cap = config.final_logit_softcapping
        if apply_softcap and cap:
            cap = np.float32(cap)
            logits = (np.tanh(logits / cap) * cap).astype(np.float32)
        return logits

    def _stage_upload(self, name: str, values: np.ndarray) -> DeviceBuffer:
        """Copy ``values`` into the reusable staging buffer for ``name``."""

        buffer = self._staging_buffer(name, values.nbytes)
        copy_host_to_device(buffer, host_array_ptr(values), values.nbytes)
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
    """First cached key a decode step must walk, or 0 when nothing is skipped.

    A sliding layer's keep-mask zeroes every key outside its window, but the
    mask is full width, so the attention kernel walks the whole live context
    and the walk is what costs: measured on the RX 7900 XTX, the decode kernel's
    time tracks ``keys``, not the number of live keys. Gemma 4 has 25 sliding
    layers of 30, so at context 4096 that is 3073 keys walked per layer whose
    weight is exactly zero.

    Skipping them is bit-exact. A masked key contributes ``exp(-inf) = 0`` to
    the denominator and the same to the weighted sum, and dropping terms whose
    value is zero leaves the surviving terms in their original order, so the
    reduction is unchanged rather than merely close. The one thing that has to
    hold is that the skipped keys are *exactly* the masked ones; the unit tests
    pin that against the mask itself for a range of window and context lengths.

    Only a one-row block may skip. A prefill block's rows sit at different
    positions and so have different windows, and its mask rows are strided by
    the full key count, so the single pointer offset this enables would read the
    wrong mask row rather than a shorter one.
    """

    window = attention.sliding_window
    if window is None or rows != 1:
        return 0
    window = int(window)
    if window <= 0:
        raise ValueError(f"sliding_window must be positive, got {window}")
    live = start + rows
    return max(0, live - window)


def _keep_mask(
    attention: Gemma4AttentionGeometry,
    start: int,
    rows: int,
) -> np.ndarray:
    """Build the ``(rows, start + rows)`` uint8 keep-mask for one block.

    A position may attend to a key at or before it, and on a sliding layer only
    within ``sliding_window`` of it. The mask covers exactly the cached range
    this block writes, which is ``start + rows`` columns.
    """

    keys = start + rows
    queries = np.arange(start, start + rows, dtype=np.int64)[:, None]
    key_positions = np.arange(keys, dtype=np.int64)[None, :]
    keep = key_positions <= queries
    if attention.sliding_window is not None:
        keep &= (queries - key_positions) < int(attention.sliding_window)
    return np.ascontiguousarray(keep.astype(np.uint8))
