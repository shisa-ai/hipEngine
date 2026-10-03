"""The Gemma 4 assistant (MTP) head forward pass.

The head is a four-block draft model that predicts the *next* token from the
target model's last hidden state. It is not a decoder in its own right: it holds
no KV cache, and each of its blocks attends against one of the backbone's last
two layers. The contract is specified in
``docs/reference/GEMMA4-ASSISTANT-MTP.md``; this module implements the forward
from it, and the two facts that are easiest to get wrong are called out at their
call sites below (``wo`` before the post-attention norm, and the input embedding
being the backbone's rather than the head's).

Where this differs from :func:`gemma4_layer_forward_bf16`
-------------------------------------------------------

The head's block is *not* the backbone's block with different weights, so it does
not call the backbone's fused layer:

* the norm order is **post-norm** on both branches -- ``attn_post_norm``
  normalizes the attention output *before* the residual add, and ``post_ffw_norm``
  normalizes the FFN output before its residual add, where the backbone adds then
  normalizes;
* there is no KV write, and K/V come from the backbone;
* the attention scale is 1.0 rather than the backbone's geometry scale (which is
  also 1.0 for Gemma 4, but the head has no geometry object to read it from);
* ``attn_q_norm`` is a per-head norm with no K counterpart.

So the forward composes the primitives directly. Every one of them already
existed; no new kernel was needed.

Batch scope
-----------

This runs one token per call. A draft/verify loop that batches proposals must run
under the ``batch_invariant`` execution profile: the backbone selects its expert
routes by batch width, so a draft pass and a verify pass of the same tokens can
otherwise use different arithmetic. See
``worklog/entries/20260928T010526.507968Z-lhl-gemma4-mmq-divergence-profile-scope-b58b30.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference.gemma4 import Gemma4RopeConfig
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    Gemma4AttentionScratch,
    gemma4_attention_prefill_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import gemma4_gelu_tanh_mul_bf16
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
    gemma4_branch_add_bf16,
    gemma4_head_rmsnorm_f32w_bf16,
    gemma4_rmsnorm_f32w_bf16,
    gemma4_scale_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rope import gemma4_rope_cos_sin_tables
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rotary import gemma4_partial_rotary_bf16
from hipengine.loading.gemma4_assistant_device import Gemma4AssistantDeviceWeights
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.gguf_linear import launch_gguf_linear

if TYPE_CHECKING:
    from hipengine.runtime.gemma4 import Gemma4Runner

from hipengine.loading.gemma4_assistant_gguf import Gemma4AssistantConfig
from hipengine.loading.gemma4_gguf import Gemma4GGUFConfig

_BF16_BYTES = 2
_F32_BYTES = 4


def _bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    """Decode BF16 storage as float32.

    BF16 is the *top* 16 bits of a float32: 8 exponent bits and 7 mantissa bits.
    It is not float16, which has 5 exponent bits and 10 mantissa bits, so
    ``bits.view(np.float16)`` reinterprets rather than converts and produces
    nonsense for any value outside float16's much smaller range. Widening to
    uint32 and shifting left is the conversion.
    """

    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def _float32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Encode float32 as BF16 storage, rounding to nearest even."""

    as_int = np.asarray(values, dtype=np.float32).view(np.uint32)
    # Round-half-to-even on the bit that is about to become the LSB.
    rounded = as_int + np.uint32(0x7FFF) + ((as_int >> 16) & np.uint32(1))
    return (rounded >> 16).astype(np.uint16)

# The two backbone layers the head reads, counted back from the end. A
# sliding-window head block reads ``n_layer - 2`` and a full-attention block
# reads ``n_layer - 1``; ``docs/reference/GEMMA4-ASSISTANT-MTP.md`` records the
# cache-construction code that fixes this, and
# ``tests/test_unit_gemma4_assistant_kv_binding.py`` asserts the geometry of the
# two layers against both artifacts.
_SWA_LAYER_FROM_END = 2
_FULL_LAYER_FROM_END = 1


@dataclass(frozen=True)
class Gemma4AssistantGeometry:
    """Resolved attention geometry for one head block.

    ``rope`` is the *bound backbone layer's* rope config rather than a second
    schedule derived from the head's metadata: the two were verified equal, so
    the head reads the backbone's and there is one definition.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope: Gemma4RopeConfig
    sliding_window: int | None
    kv_layer: int

    @property
    def q_width(self) -> int:
        return self.num_heads * self.head_dim


def gemma4_assistant_geometry(
    config: Gemma4AssistantConfig,
    backbone: Gemma4GGUFConfig,
) -> tuple[Gemma4AssistantGeometry, ...]:
    """Resolve the head's per-block geometry against a backbone config.

    Fails closed if the head's block does not match the backbone layer it binds
    to. The equality is a property of how the two artifacts were built rather
    than of either file, so it is checked here at construction instead of being
    assumed and producing silently wrong attention.
    """

    block_count = int(config.block_count)
    if block_count <= 0:
        raise ValueError("the assistant head has no blocks")
    layers = int(backbone.block_count)
    if layers < 2:
        raise ValueError(f"a {layers}-layer backbone has no last two layers to share")
    if block_count > layers:
        raise ValueError(
            f"assistant head has {block_count} blocks, more than the backbone's "
            f"{layers} layers; its blocks cannot each bind a distinct last layer"
        )

    geometry: list[Gemma4AssistantGeometry] = []
    for block_id in range(block_count):
        is_swa = bool(config.is_swa[block_id])
        layer_id = layers - (_SWA_LAYER_FROM_END if is_swa else _FULL_LAYER_FROM_END)
        head_dim = int(config.key_length_swa if is_swa else config.key_length)
        kv_heads = int(config.n_head_kv[block_id])

        if bool(backbone.is_sliding(layer_id)) != is_swa:
            raise ValueError(
                f"assistant block {block_id} binds backbone layer {layer_id}, which is "
                f"{'sliding' if backbone.is_sliding(layer_id) else 'full'} attention "
                f"while the head block is {'sliding' if is_swa else 'full'}"
            )
        if int(backbone.head_count(layer_id)) != int(config.n_head):
            raise ValueError(
                f"assistant block {block_id} has {config.n_head} Q heads but backbone "
                f"layer {layer_id} has {backbone.head_count(layer_id)}"
            )
        if int(backbone.head_count_kv_for(layer_id)) != kv_heads:
            raise ValueError(
                f"assistant block {block_id} declares {kv_heads} KV heads but backbone "
                f"layer {layer_id} carries {backbone.head_count_kv_for(layer_id)}"
            )
        if int(backbone.head_dim(layer_id)) != head_dim:
            raise ValueError(
                f"assistant block {block_id} has head width {head_dim} but backbone "
                f"layer {layer_id} has {backbone.head_dim(layer_id)}"
            )

        # The bound layer's config, narrowed to the CPU-reference type the rope
        # table builder takes. ``rotated_pairs`` becomes ``rope_angles`` and
        # ``freq_base`` becomes ``rope_theta``; the two types carry the same
        # schedule, so this is a field rename rather than a second derivation.
        rope = backbone.rope_for_layer(layer_id)
        geometry.append(
            Gemma4AssistantGeometry(
                num_heads=int(config.n_head),
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                rope=Gemma4RopeConfig(
                    rope_theta=float(rope.freq_base),
                    head_dim=head_dim,
                    rope_angles=int(rope.rotated_pairs),
                    rope_type=rope.rope_type,
                ),
                sliding_window=int(config.sliding_window) if is_swa else None,
                kv_layer=layer_id,
            )
        )
    return tuple(geometry)


def gemma4_assistant_keep_mask(
    position: int,
    keys: int,
    *,
    sliding_window: int | None,
) -> np.ndarray:
    """Return the ``(1, keys)`` uint8 keep-mask for one query row.

    Causal, and windowed when the block is a sliding one: key ``k`` is kept when
    ``k <= position`` and, with a window, when ``k > position - window``. The
    head is single-token, so there is no row-to-row variation to encode and this
    is the same predicate ``_keep_mask`` uses for the backbone.
    """

    if position < 0:
        raise ValueError(f"position must be non-negative, got {position}")
    if keys <= 0:
        raise ValueError(f"keys must be positive, got {keys}")
    index = np.arange(keys, dtype=np.int64)
    keep = index <= position
    if sliding_window is not None:
        keep &= index > position - int(sliding_window)
    return keep.astype(np.uint8).reshape(1, keys)




@dataclass
class Gemma4AssistantScratch:
    """Per-block device scratch, sized from the block's geometry."""

    hidden_size: int
    q_width: int
    intermediate: int
    attention: Gemma4AttentionScratch = field(default_factory=Gemma4AttentionScratch)
    _buffers: dict[str, DeviceBuffer] = field(default_factory=dict, repr=False)

    def buffer(self, name: str, nbytes: int) -> DeviceBuffer:
        existing = self._buffers.get(name)
        if existing is not None and existing.nbytes >= nbytes:
            return existing
        if existing is not None:
            free(existing)
        buffer = malloc(nbytes)
        self._buffers[name] = buffer
        return buffer

    def release(self) -> None:
        for buffer in self._buffers.values():
            free(buffer)
        self._buffers.clear()
        self.attention.close()


@dataclass
class Gemma4AssistantHead:
    """Runs the assistant head's forward over a backbone's shared KV.

    ``weights`` is the head's own 49-tensor device residency, ``backbone`` is the
    target model's config (used to resolve the two bound layers), and
    ``backbone_embedding`` is the target's ``embed_tokens`` weight -- the head's
    input embedding is the *backbone's* table, not the head's own, which is the
    output projection.

    ``backbone_output_norm`` is the target's final norm weight, applied to the
    hidden state the caller seeds with. The head's recurrent input is the target's
    **post-output-norm** state, not its raw last-block output: llama.cpp's
    backbone assigns ``res->t_h_nextn`` *after* ``build_norm(cur,
    model.output_norm, ...)`` and describes it as "the LM-head input feature"
    handed to the drafter "as the recurrent h input". Passing the raw residual
    stream instead is a whole-vector substitution, not a rounding difference.

    The recurrent state is owned by the head. :meth:`prime` seeds it from the
    target's raw hidden state at the last position the target wrote; each
    :meth:`forward` consumes it and replaces it with the head's own projected
    state, which is the input the next draft step wants. Only the seed goes
    through the target's final norm -- later steps must not re-norm a value the
    head produced.

    The head allocates no KV: :meth:`forward` takes the backbone's read views.
    """

    weights: Gemma4AssistantDeviceWeights
    backbone: object
    backbone_embedding: object
    backbone_output_norm: object
    capacity: int
    eps: float
    stream: int = 0
    _geometry: tuple[Gemma4AssistantGeometry, ...] = field(default=(), repr=False)
    _scratch: list[Gemma4AssistantScratch] = field(default_factory=list, repr=False)
    _owned: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _staged: dict[str, DeviceBuffer] = field(default_factory=dict, repr=False)
    _closed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        config = self.weights.config
        self._geometry = gemma4_assistant_geometry(config, self.backbone)
        hidden = int(config.n_embd)
        for block in self._geometry:
            self._scratch.append(
                Gemma4AssistantScratch(
                    hidden_size=hidden,
                    q_width=block.q_width,
                    intermediate=int(config.n_ff),
                )
            )
        # x (backbone width), the concatenated pre-projection input, the running
        # activation, and the two outputs. BF16 except the logits.
        backbone_width = int(config.n_embd_backbone)
        self._xh = self._alloc((2 * backbone_width) * _BF16_BYTES)
        self._x = self._alloc(backbone_width * _BF16_BYTES)
        # One buffer is both the recurrent input and the projected output. The
        # input half is copied to host for the concat before any layer runs, and
        # the output is written after the last one, so the overlap is safe and
        # saves the device-to-device copy that core/memory.py cannot do.
        self._recurrent = self._alloc(backbone_width * _BF16_BYTES)
        self._cur = self._alloc(hidden * _BF16_BYTES)
        self._logits = self._alloc(262144 * _F32_BYTES)

    @property
    def geometry(self) -> tuple[Gemma4AssistantGeometry, ...]:
        return self._geometry

    @property
    def block_count(self) -> int:
        return len(self._geometry)

    def _alloc(self, nbytes: int) -> DeviceBuffer:
        buffer = malloc(nbytes)
        self._owned.append(buffer)
        return buffer

    def _stage(self, name: str, values: np.ndarray) -> DeviceBuffer:
        """Upload host values once and keep them for the head's lifetime.

        For the keep-mask and rope tables, which depend on the position. A
        position change rebuilds them; within one draft step they are constant
        across the four blocks of the same attention kind.
        """

        key = f"{name}"
        buffer = self._staged.get(key)
        if buffer is None or buffer.nbytes < values.nbytes:
            if buffer is not None:
                free(buffer)
                self._owned.remove(buffer)
            buffer = self._alloc(values.nbytes)
            self._staged[key] = buffer
        copy_host_array_to_device(buffer, values)
        return buffer

    def _linear(self, slot: str, src: int, dst: int, rows: int, in_features: int, out: int) -> None:
        launch_gguf_linear(
            self.weights.weight(slot),
            src,
            dst,
            rows,
            in_features,
            out,
            stream=self.stream,
        )

    def prime(self, backbone_hidden: DeviceBuffer) -> None:
        """Seed the recurrent state from the target's raw hidden state.

        ``backbone_hidden`` is the target's last-block hidden state at the last
        position it wrote -- the row that produced the token about to be drafted.
        This method applies the target's final norm, which is what makes it the
        state the head was trained to consume. The backbone's own eps is used
        rather than the head's, because the norm is the target's.
        """

        if self._closed:
            raise RuntimeError("the assistant head is closed")
        gemma4_rmsnorm_f32w_bf16(
            backbone_hidden.ptr,
            int(self.backbone_output_norm.ptr),
            self._recurrent.ptr,
            1,
            int(self.backbone.hidden_size),
            float(self.backbone.rms_norm_eps),
            stream=self.stream,
            runtime=get_hip_runtime(),
        )

    def forward(
        self,
        token_id: int,
        *,
        position: int,
        shared_kv: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one draft step and return ``(logits, h_next)``.

        ``logits`` is F32 of the head's vocabulary width; ``h_next`` is the
        head's projected state in the *backbone's* width. The head's recurrent
        state is advanced in place, so successive calls chain: the state returned
        by one step is the state consumed by the next.

        ``position`` is the draft query's absolute position, which is one past
        the last row the target wrote and stays constant across a draft round:
        llama.cpp's ``common/speculative.cpp`` takes the ``is_mem_shared`` branch
        for Gemma 4 assistants and adds every draft token at ``dp.n_past``, citing
        the Hugging Face doc's "the position_ids value are constant".
        """

        if self._closed:
            raise RuntimeError("the assistant head is closed")
        config = self.weights.config
        hidden = int(config.n_embd)
        # The backbone's width and vocabulary, taken from the backbone's config
        # rather than the head's ``embedding_length_out``. They agree on the real
        # artifacts, but the head describes the *target* it predicts, so reading
        # the target's own numbers is what makes a mismatch impossible to hide.
        backbone_width = int(self.backbone.hidden_size)
        backbone_vocab = int(self.backbone.vocab_size)
        eps = float(self.eps)
        runtime = get_hip_runtime()
        kwargs = {"stream": self.stream, "runtime": runtime}

        # --- input: the *backbone's* embedding, scaled ---------------------
        # The reference reads `model_other->tok_embd`, the target's table, and
        # multiplies by sqrt(n_embd_backbone). Gathering the head's own
        # `token_embd` here would produce a 1024-wide vector where 2816 is
        # required; that table is the output projection.
        tokens = np.asarray([int(token_id)], dtype=np.int64)
        token_buf = self._stage("token", tokens)
        launch_gguf_embedding(
            self.backbone_embedding,
            token_buf.ptr,
            self._x.ptr,
            1,
            backbone_width,
            backbone_vocab,
            stream=self.stream,
        )
        gemma4_scale_bf16(
            self._x.ptr,
            self._x.ptr,
            1,
            backbone_width,
            float(backbone_width) ** 0.5,
            **kwargs,
        )

        # --- concat(x, recurrent state) ------------------------------------
        # Host-side: 2 * 2816 BF16 is 11 KB, and core/memory.py has no
        # device-to-device copy. The gather above wrote x to the first half of
        # xh's staging buffer, so only the hidden half is copied per step.
        host_x = np.empty(backbone_width, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(host_x), self._x, host_x.nbytes, runtime=runtime)
        host_h = np.empty(backbone_width, dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(host_h), self._recurrent, host_h.nbytes, runtime=runtime
        )
        copy_host_array_to_device(
            self._xh, np.concatenate((host_x, host_h)).view(np.uint8)
        )

        # --- pre-projection -------------------------------------------------
        self._linear(
            "nextn_pre_projection", self._xh.ptr, self._cur.ptr, 1, 2 * backbone_width, hidden
        )

        for block_id, block in enumerate(self._geometry):
            scratch = self._scratch[block_id]
            prefix = f"blocks.{block_id}"
            head_dim = block.head_dim
            q_width = block.q_width
            intermediate = int(config.n_ff)

            normed = scratch.buffer("normed", hidden * _BF16_BYTES)
            q = scratch.buffer("q", q_width * _BF16_BYTES)
            q_rot = scratch.buffer("q_rot", q_width * _BF16_BYTES)
            context = scratch.buffer("context", q_width * _BF16_BYTES)
            attn = scratch.buffer("attn", hidden * _BF16_BYTES)
            residual = scratch.buffer("residual", hidden * _BF16_BYTES)
            ffn_in = scratch.buffer("ffn_in", hidden * _BF16_BYTES)
            gate_up = scratch.buffer("gate_up", 2 * intermediate * _BF16_BYTES)
            activated = scratch.buffer("activated", intermediate * _BF16_BYTES)
            ffn = scratch.buffer("ffn", hidden * _BF16_BYTES)

            # pre-norm, then Q
            gemma4_rmsnorm_f32w_bf16(
                self._cur.ptr,
                self.weights.buffer(f"{prefix}.attn_norm").ptr,
                normed.ptr,
                1,
                hidden,
                eps,
                **kwargs,
            )
            self._linear(f"{prefix}.attn_q", normed.ptr, q.ptr, 1, hidden, q_width)

            # per-head Q norm over head_dim, with no K counterpart
            gemma4_head_rmsnorm_f32w_bf16(
                q.ptr,
                self.weights.buffer(f"{prefix}.attn_q_norm").ptr,
                q.ptr,
                block.num_heads,
                head_dim,
                eps,
                **kwargs,
            )

            # Q-only rotation: the head has no K, and the bound layer's K in the
            # shared cache is already rotated. ``key_ptr = 0`` is the documented
            # way to ask for queries only.
            cos_values, sin_values = gemma4_rope_cos_sin_tables(
                block.rope, np.asarray([int(position)], dtype=np.int64)
            )
            cos = self._stage(
                f"cos{block_id}", np.ascontiguousarray(cos_values, dtype=np.float32)
            )
            sin = self._stage(
                f"sin{block_id}", np.ascontiguousarray(sin_values, dtype=np.float32)
            )
            gemma4_partial_rotary_bf16(
                q.ptr,
                0,
                cos.ptr,
                sin.ptr,
                q_rot.ptr,
                0,
                1,
                block.num_heads,
                0,
                head_dim,
                **kwargs,
            )

            # attention against the backbone's cache, scale 1.0
            view = shared_kv[int(block.kv_layer)]
            mask = gemma4_assistant_keep_mask(
                position, int(view.live), sliding_window=block.sliding_window
            )
            mask_buf = self._stage(f"mask{block_id}", np.ascontiguousarray(mask))
            gemma4_attention_prefill_bf16(
                q_rot.ptr,
                view.key_cache,
                view.value_cache,
                mask_buf.ptr,
                context.ptr,
                tokens=1,
                keys=int(view.live),
                num_heads=block.num_heads,
                num_kv_heads=block.num_kv_heads,
                head_dim=head_dim,
                scale=1.0,
                # The single query's absolute position, which is what the kernel
                # uses to skip a row's own masked prefix. The head passes
                # key_begin = 0 (it reads the whole live cache), so this is the
                # position itself; leaving it at the default 0 would make the
                # kernel treat the query as position 0 and attend to almost
                # nothing.
                row_offset=int(position),
                # A global block's window is the whole live cache, and saying
                # so rather than leaving it at 0 is what lets the kernel trim
                # the trailing masked run: `window > 0` is the promise that the
                # mask is sliding-causal, which is zero above the query's own
                # position too. The single query here sits at `position`, so
                # `keys` is exactly the sliding-causal mask with an unbounded
                # window. This is a no-op when the query is the last live token,
                # which is the ordinary decode step.
                window=int(view.live)
                if block.sliding_window is None
                else int(block.sliding_window),
                scratch=scratch.attention,
                **kwargs,
            )

            # wo, *then* the post-attention norm
            self._linear(f"{prefix}.attn_output", context.ptr, attn.ptr, 1, q_width, hidden)
            gemma4_rmsnorm_f32w_bf16(
                attn.ptr,
                self.weights.buffer(f"{prefix}.post_attention_norm").ptr,
                attn.ptr,
                1,
                hidden,
                eps,
                **kwargs,
            )
            gemma4_branch_add_bf16(
                attn.ptr, self._cur.ptr, residual.ptr, hidden, **kwargs
            )

            # FFN: pre-norm, gate/up, GeGLU, down, post-norm
            gemma4_rmsnorm_f32w_bf16(
                residual.ptr,
                self.weights.buffer(f"{prefix}.ffn_norm").ptr,
                ffn_in.ptr,
                1,
                hidden,
                eps,
                **kwargs,
            )
            self._linear(
                f"{prefix}.ffn_gate", ffn_in.ptr, gate_up.ptr, 1, hidden, intermediate
            )
            self._linear(
                f"{prefix}.ffn_up",
                ffn_in.ptr,
                gate_up.ptr + intermediate * _BF16_BYTES,
                1,
                hidden,
                intermediate,
            )
            gemma4_gelu_tanh_mul_bf16(
                gate_up.ptr, activated.ptr, 1, intermediate, **kwargs
            )
            self._linear(
                f"{prefix}.ffn_down", activated.ptr, ffn.ptr, 1, intermediate, hidden
            )
            gemma4_rmsnorm_f32w_bf16(
                ffn.ptr,
                self.weights.buffer(f"{prefix}.post_ffw_norm").ptr,
                ffn.ptr,
                1,
                hidden,
                eps,
                **kwargs,
            )

            # cur = (ffn + attn_out) * layer_output_scale
            gemma4_branch_add_bf16(ffn.ptr, residual.ptr, self._cur.ptr, hidden, **kwargs)
            scale = self._layer_scale(prefix)
            gemma4_scale_bf16(
                self._cur.ptr, self._cur.ptr, 1, hidden, scale, **kwargs
            )

        # --- output norm, logits, h_next ------------------------------------
        gemma4_rmsnorm_f32w_bf16(
            self._cur.ptr,
            self.weights.buffer("output_norm").ptr,
            self._cur.ptr,
            1,
            hidden,
            eps,
            **kwargs,
        )
        vocab = 262144
        launch_gguf_linear(
            self.weights.weight("token_embedding"),
            self._cur.ptr,
            self._logits.ptr,
            1,
            hidden,
            vocab,
            output_dtype="f32",
            stream=self.stream,
        )
        logits = np.empty(vocab, dtype=np.float32)
        copy_device_to_host(host_array_ptr(logits), self._logits, logits.nbytes, runtime=runtime)

        self._linear(
            "nextn_post_projection",
            self._cur.ptr,
            self._recurrent.ptr,
            1,
            hidden,
            backbone_width,
        )
        host_next = np.empty(backbone_width, dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(host_next), self._recurrent, host_next.nbytes, runtime=runtime
        )
        return logits, _bf16_to_float32(host_next)

    def _layer_scale(self, prefix: str) -> float:
        """Read a block's ``layer_output_scale``, which is a 1-element F32."""

        buffer = self.weights.buffer(f"{prefix}.layer_output_scale")
        value = np.empty(1, dtype=np.float32)
        copy_device_to_host(host_array_ptr(value), buffer, 4)
        return float(value[0])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for scratch in self._scratch:
            scratch.release()
        self._staged.clear()
        for buffer in reversed(self._owned):
            free(buffer)
        self._owned.clear()


@dataclass
class Gemma4MtpDrafter:
    """Chains a :class:`Gemma4AssistantHead` into a multi-token draft.

    The head drafts one token per call, so a draft of *n* tokens is *n* calls and
    the head's recurrent state carries the chain. What this class adds is the
    part that is easy to get wrong from outside:

    * the seed. :meth:`draft` primes the head from the backbone's own last hidden
      row, so it must be called straight after the forward that produced
      ``token`` and before any later one. ``Gemma4Runner.hidden_state``
      documents the same constraint from the other side.
    * the views. ``Gemma4Runner.shared_kv`` captures the live position count when
      it is called, so the views are taken once per draft, before the verify pass
      extends the cache. Taking them per step, or after the verify, would either
      re-read a moved count or expose the query's own K/V to the head.
    * the position. Every step of one draft uses the same ``n_past``; the head's
      blocks are all shared-KV, which is llama.cpp's ``is_mem_shared`` case.

    This proposes tokens; it does not accept them. Acceptance is the caller's
    verify pass against the target's own logits.
    """

    head: Gemma4AssistantHead
    runner: Gemma4Runner
    max_drafts: int = 4

    def __post_init__(self) -> None:
        if self.max_drafts <= 0:
            raise ValueError("max_drafts must be positive")

    def draft(self, token: int, *, hidden_row: int = -1) -> list[int]:
        """Propose up to ``max_drafts`` tokens to follow ``token``.

        ``token`` is the token the backbone just sampled, not the last token it
        processed: the head's first step is fed the sampled token together with
        the hidden row that produced it.
        """

        self.head.prime(self.runner.hidden_state(row=hidden_row))
        shared = {
            index: self.runner.shared_kv(index) for index in range(self.runner.layer_count)
        }
        position = self.runner.position
        drafts: list[int] = []
        for _ in range(self.max_drafts):
            logits, _h_next = self.head.forward(token, position=position, shared_kv=shared)
            token = int(np.argmax(logits))
            drafts.append(token)
        return drafts


__all__ = [
    "Gemma4AssistantGeometry",
    "Gemma4AssistantHead",
    "Gemma4AssistantScratch",
    "Gemma4MtpDrafter",
    "gemma4_assistant_geometry",
    "gemma4_assistant_keep_mask",
]
