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

:meth:`Gemma4AssistantHead.forward` runs one token per call and keeps the
recurrent state on the head. :meth:`Gemma4AssistantHead.forward_batch` runs C
*independent* requests through one call: every dense projection, the norms, the
Q rotation, the GeGLU and the output projection are single launches at physical
row width C, and only attention is per request, because each request reads its
own shared-KV view, its own position, and its own keep-mask. The batched path
holds no state: the caller passes each request's recurrent row in and receives
the next rows back, so a provider cycle can keep, discard, or reorder rows
without leaving a cursor behind on the head.

The two paths are not required to be bit-identical. A projection at row width C
can select a different linear schedule than the same weight at row width 1, so a
batched proposal need not reproduce C single-row proposals to the last bit. That
is admissible because a draft is only a proposal: the target verifies every
token under its own arithmetic. A draft/verify loop that batches proposals must
still run the *target's* passes under the ``batch_invariant`` execution profile:
the backbone selects its expert routes by batch width, so a draft pass and a
verify pass of the same tokens can otherwise use different arithmetic. See
``worklog/entries/20260928T010526.507968Z-lhl-gemma4-mmq-divergence-profile-scope-b58b30.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Mapping, Sequence
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
    gemma4_attention_decode_flash_long_bf16,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import (
    gemma4_gelu_tanh_mul_bf16,
    gemma4_gelu_tanh_mul_split_bf16,
)
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


def _head_logits_width(weights: Gemma4AssistantDeviceWeights, *, hidden: int) -> int:
    """Return the head's output-projection width, read off its own weight table.

    The head's ``token_embd`` *is* its output projection -- the reference loader
    marks the output tensor ``TENSOR_DUPLICATED`` against it -- so a logits row is
    that table applied to the head's hidden state. One width serves both
    :meth:`Gemma4AssistantHead.forward` and :meth:`Gemma4AssistantHead.forward_batch`,
    so the two paths cannot disagree about the vocabulary a logits row spans.

    ``GGUFTensorInfo.shape`` is the **row-major** logical shape, i.e. ``ne``
    reversed, so a linear's output count is ``shape[0]`` and its input width is
    ``shape[1]`` -- the same reading the backbone loader uses for ``ffn_gate``'s
    ``out_features``. The input width is checked against the head's hidden size
    here, because a transposed table would otherwise present the hidden width as
    a vocabulary and silently truncate every logits row to it.
    """

    shape = tuple(
        int(value) for value in weights.weight("token_embedding").spec.source.shape
    )
    if len(shape) != 2 or shape[0] <= 0 or shape[1] != int(hidden):
        raise ValueError(
            f"the assistant head's token_embedding has shape {shape}; a 2-D "
            f"(n_vocab, {int(hidden)}) row-major table is required for the logits "
            "row, with the head's hidden width as its input dimension"
        )
    return shape[0]


def _integer_sequence(values: object, name: str) -> tuple[int, ...]:
    """Read a sequence of integers, refusing a scalar or a non-integral element."""

    if isinstance(values, (bool, np.bool_)) or isinstance(values, (int, np.integer)):
        raise TypeError(f"{name} must be a sequence of integers, not a single value")
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers") from exc
    for index, item in enumerate(items):
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise TypeError(f"{name}[{index}] is {item!r}; an integer is required")
    return tuple(int(item) for item in items)


def _host_bf16_matrix(values: object, *, width: int) -> np.ndarray | None:
    """Return ``values`` as ``(rows, width)`` BF16 bits, or ``None`` if it is not a matrix.

    Accepted host forms are BF16 bits as ``uint16``, their raw-byte ``uint8`` view,
    and ``float32`` values, which encode to BF16 with the round-to-nearest-even the
    device uses. A value that is not a 2-D numeric array at all -- a list of device
    buffers, for instance -- returns ``None`` so the caller can resolve it another
    way; a numeric matrix of the wrong width raises, because that is a mistake
    rather than a different representation.
    """

    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.dtype == object:
        return None
    if array.dtype == np.uint8:
        if array.shape[1] != 2 * width:
            raise ValueError(
                f"a uint8 byte view of BF16 rows must be {2 * width} wide, got "
                f"{array.shape[1]}"
            )
        return np.ascontiguousarray(array).view(np.uint16)
    if array.dtype == np.uint16:
        if array.shape[1] != width:
            raise ValueError(
                f"BF16 rows must be {width} wide, got {array.shape[1]}"
            )
        return array
    if array.dtype == np.float32:
        bits = _float32_to_bf16(array)
        if bits.shape[1] != width:
            raise ValueError(f"float32 rows must be {width} wide, got {bits.shape[1]}")
        return bits
    raise ValueError(
        f"a host BF16 row matrix has dtype {array.dtype}; pass uint16 bits, their "
        "uint8 byte view, or float32 values"
    )


@dataclass(frozen=True)
class _BatchRow:
    """One resolved request row in a batched head forward."""

    token_id: int
    position: int
    #: One shared-KV read view per block, in ``geometry`` order.
    views: tuple[object, ...]
    #: The query's slot in each block's cache frame, ``position - view.base``.
    slot_positions: tuple[int, ...]
    #: The block's ``(1, keys)`` keep-mask, evaluated on absolute positions.
    masks: tuple[np.ndarray, ...]

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
    base: int = 0,
) -> np.ndarray:
    """Return the ``(1, keys)`` uint8 keep-mask for one query row.

    Causal, and windowed when the block is a sliding one: key ``k`` is kept when
    ``k <= position`` and, with a window, when ``k > position - window``. The
    head is single-token, so there is no row-to-row variation to encode and this
    is the same predicate ``_keep_mask`` uses for the backbone.

    ``base`` is the cache's base. Slot ``s`` holds absolute position
    ``base + s``, so the predicate is evaluated on absolute positions while the
    returned mask stays indexed by slot; a global block's base is 0 and this is
    the identity.
    """

    if position < 0:
        raise ValueError(f"position must be non-negative, got {position}")
    if keys <= 0:
        raise ValueError(f"keys must be positive, got {keys}")
    if base < 0:
        raise ValueError(f"base must be non-negative, got {base}")
    index = np.arange(keys, dtype=np.int64) + int(base)
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

    :meth:`forward_batch` is the C-request form and holds no state of its own:
    the caller passes every request's recurrent row and receives the next rows
    back, so no provider cycle can leave a cursor on the head. :meth:`prime_batch`
    is its seed, and applies the target's final norm to each raw row. The two
    forms share the loaded weights and are otherwise independent; :meth:`forward`
    and :class:`Gemma4MtpDrafter` are unchanged by the batched path.

    The head allocates no KV: :meth:`forward` takes the backbone's read views, and
    :meth:`forward_batch` takes one per request.
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
    #: Packed C-row scratch, one per block, allocated on the first batched call.
    _batch_scratch: list[Gemma4AssistantScratch] = field(default_factory=list, repr=False)
    _owned: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _staged: dict[str, DeviceBuffer] = field(default_factory=dict, repr=False)
    #: Packed C-row buffers, kept apart from ``_staged`` so the single-row path
    #: and the batched path never resize each other's staging.
    _batch: dict[str, DeviceBuffer] = field(default_factory=dict, repr=False)
    _closed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        config = self.weights.config
        self._geometry = gemma4_assistant_geometry(config, self.backbone)
        hidden = int(config.n_embd)
        self._logits_width = _head_logits_width(self.weights, hidden=hidden)
        for block in self._geometry:
            self._scratch.append(
                Gemma4AssistantScratch(
                    hidden_size=hidden,
                    q_width=block.q_width,
                    intermediate=int(config.n_ff),
                )
            )
            self._batch_scratch.append(
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
        self._logits = self._alloc(self._logits_width * _F32_BYTES)

    @property
    def geometry(self) -> tuple[Gemma4AssistantGeometry, ...]:
        return self._geometry

    @property
    def block_count(self) -> int:
        return len(self._geometry)

    @property
    def logits_width(self) -> int:
        """Width of a logits row: the head's tied output projection."""

        return int(self._logits_width)

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

        buffer = self._grow(self._staged, name, values.nbytes)
        copy_host_array_to_device(buffer, values)
        return buffer

    def _grow(self, pool: dict[str, DeviceBuffer], name: str, nbytes: int) -> DeviceBuffer:
        """Return ``pool[name]`` grown to at least ``nbytes``, allocating on first use."""

        buffer = pool.get(name)
        if buffer is not None and buffer.nbytes >= nbytes:
            return buffer
        if buffer is not None:
            free(buffer)
            self._owned.remove(buffer)
        buffer = self._alloc(nbytes)
        pool[name] = buffer
        return buffer

    def _batch_buffer(self, name: str, nbytes: int) -> DeviceBuffer:
        """Packed C-row scratch for the batched path, grown on demand."""

        return self._grow(self._batch, name, nbytes)

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
            # The cache may be rebased: slot ``s`` holds absolute position
            # ``base + s``. The head reads a shared view by absolute position,
            # so both the mask predicate and the kernel's ``row_offset`` are
            # taken in the cache's slot frame, ``position - base``; that is the
            # identity for a global layer, whose base is 0.
            slot_position = int(position) - int(view.base)
            mask = gemma4_assistant_keep_mask(
                position,
                int(view.live),
                sliding_window=block.sliding_window,
                base=int(view.base),
            )
            mask_buf = self._stage(f"mask{block_id}", np.ascontiguousarray(mask))
            # A draft is only a proposal -- the target verifies every token,
            # under every profile -- so the head takes the fastest decode route
            # rather than a profile's exact one: key-sliced flash from 256 keys
            # with the long-row slice count, the native kernel below that. The
            # native launch puts a long global row on the owned-logits class
            # kernel, 12.8 ms a draft token at 32K keys on a W7900.
            gemma4_attention_decode_flash_long_bf16(
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
                # The single query's position in the cache's slot frame, which
                # is what the kernel uses to skip a row's own masked prefix. The
                # head passes key_begin = 0 (it reads the whole live cache), so
                # this is ``position - base``; leaving it at the default 0 would
                # make the kernel treat the query as slot 0 and attend to almost
                # nothing.
                row_offset=slot_position,
                # A global block's window is the whole live cache, and saying
                # so rather than leaving it at 0 is what lets the kernel trim
                # the trailing masked run: `window > 0` is the promise that the
                # mask is sliding-causal, which is zero above the query's own
                # position too. The single query here sits at ``slot_position``,
                # so `keys` is exactly the sliding-causal mask with an unbounded
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
        vocab = int(self._logits_width)
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

    def prime_batch(self, hidden_states: object) -> np.ndarray:
        """Seed C requests' recurrent rows from the target's raw hidden rows.

        Two input forms, one result:

        * a sequence of device buffers, one per request, each holding that
          request's raw last-block hidden row -- the target's own hidden state at
          the last position it wrote for that request. A buffer larger than one
          row is read from its start; pass ``DeviceBuffer(buffer.ptr + offset,
          row_bytes)`` to seed from a row that is not at the start.
        * a host ``(C, backbone_width)`` BF16 matrix of those rows, as ``uint16``
          bits, their ``uint8`` byte view, or ``float32`` values.

        This is :meth:`prime` for a batch: the target's final norm is applied to
        every row in one launch, because the norm is the target's and the rows are
        independent. The result is a fresh host ``(C, backbone_width)`` array of
        BF16 bits, which is exactly what :meth:`forward_batch` takes as
        ``recurrent_states``.
        """

        if self._closed:
            raise RuntimeError("the assistant head is closed")
        width = int(self.backbone.hidden_size)
        row_nbytes = width * _BF16_BYTES
        runtime = get_hip_runtime()

        matrix = _host_bf16_matrix(hidden_states, width=width)
        if matrix is not None:
            count = int(matrix.shape[0])
            if count <= 0:
                raise ValueError("prime_batch needs at least one hidden row")
            raw = np.ascontiguousarray(matrix)
        else:
            try:
                sources = tuple(hidden_states)  # type: ignore[arg-type]
            except TypeError as exc:
                raise TypeError(
                    "hidden_states must be a sequence of per-request device buffers "
                    "or a host (rows, backbone_width) BF16 matrix"
                ) from exc
            count = len(sources)
            if count <= 0:
                raise ValueError("prime_batch needs at least one hidden row")
            for index, source in enumerate(sources):
                if not hasattr(source, "ptr") or not hasattr(source, "nbytes"):
                    raise TypeError(
                        f"hidden_states[{index}] is not a device buffer, and the "
                        "argument is not a host BF16 matrix either"
                    )
                if int(source.nbytes) < row_nbytes:
                    raise ValueError(
                        f"hidden_states[{index}] holds {int(source.nbytes)} bytes, less "
                        f"than one {width}-wide BF16 row ({row_nbytes} bytes)"
                    )
            # No device-to-device copy exists in core/memory.py, so the rows are
            # gathered on the host once and uploaded as one packed block. Each row
            # is ``row_nbytes`` bytes, so this is C small reads rather than a
            # strided copy.
            raw = np.empty((count, width), dtype=np.uint16)
            for index, source in enumerate(sources):
                copy_device_to_host(
                    host_array_ptr(raw[index]), source, row_nbytes, runtime=runtime
                )

        staged = self._batch_buffer("prime_in", count * row_nbytes)
        copy_host_array_to_device(staged, raw)
        primed = self._batch_buffer("prime_out", count * row_nbytes)
        gemma4_rmsnorm_f32w_bf16(
            staged.ptr,
            int(self.backbone_output_norm.ptr),
            primed.ptr,
            count,
            width,
            float(self.backbone.rms_norm_eps),
            stream=self.stream,
            runtime=runtime,
        )
        result = np.empty((count, width), dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(result), primed, result.nbytes, runtime=runtime
        )
        return result

    def _batch_view(
        self,
        shared_kv: object,
        index: int,
        block: Gemma4AssistantGeometry,
    ) -> object:
        """Resolve one request's read view of the backbone layer a block binds."""

        try:
            view = shared_kv[int(block.kv_layer)]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"request {index}'s shared_kv carries no view for backbone layer "
                f"{block.kv_layer}, which one of the head's blocks attends"
            ) from exc
        for attribute in ("key_cache", "value_cache", "live", "base"):
            if not hasattr(view, attribute):
                raise TypeError(
                    f"request {index}'s view of backbone layer {block.kv_layer} has no "
                    f"{attribute!r}; a shared-KV read view needs key_cache, "
                    "value_cache, live, and base"
                )
        return view

    def _resolve_batch_rows(
        self,
        token_ids: object,
        positions: object,
        shared_kv: object,
    ) -> tuple[_BatchRow, ...]:
        """Validate one batch and resolve it against the head's geometry.

        Everything a batched forward can refuse is refused here, before the first
        device write: a malformed request list must not leave the packed scratch
        holding some requests' rows and not others.
        """

        if self._closed:
            raise RuntimeError("the assistant head is closed")
        tokens = _integer_sequence(token_ids, "token_ids")
        places = _integer_sequence(positions, "positions")
        if not tokens:
            raise ValueError("forward_batch needs at least one request row")
        if len(tokens) != len(places):
            raise ValueError(
                f"forward_batch got {len(tokens)} token ids and {len(places)} positions"
            )
        try:
            requests = tuple(shared_kv)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "shared_kv must be a sequence of per-request {layer: view} mappings"
            ) from exc
        if len(requests) != len(tokens):
            raise ValueError(
                f"forward_batch got {len(tokens)} token ids and {len(requests)} "
                "shared_kv entries"
            )

        vocabulary = int(self.backbone.vocab_size)
        rows: list[_BatchRow] = []
        for index, (token, position) in enumerate(zip(tokens, places)):
            if not 0 <= token < vocabulary:
                raise ValueError(
                    f"request {index} token id {token} is outside the backbone "
                    f"vocabulary [0, {vocabulary})"
                )
            if position < 0:
                raise ValueError(
                    f"request {index} position must be non-negative, got {position}"
                )
            views: list[object] = []
            slots: list[int] = []
            masks: list[np.ndarray] = []
            for block in self._geometry:
                view = self._batch_view(requests[index], index, block)
                live = int(view.live)  # type: ignore[attr-defined]
                base = int(view.base)  # type: ignore[attr-defined]
                if live <= 0:
                    raise ValueError(
                        f"request {index} has {live} live positions in its view of "
                        f"backbone layer {block.kv_layer}; attention needs at least "
                        "one key"
                    )
                if base < 0:
                    raise ValueError(
                        f"request {index}'s view of backbone layer {block.kv_layer} "
                        f"has base {base}; a cache base cannot be negative"
                    )
                slot = position - base
                if not 0 <= slot <= live:
                    raise ValueError(
                        f"request {index} position {position} is slot {slot} in its "
                        f"view of backbone layer {block.kv_layer}, which holds live "
                        f"slots [0, {live}) from base {base}"
                    )
                views.append(view)
                slots.append(slot)
                masks.append(
                    gemma4_assistant_keep_mask(
                        position,
                        live,
                        sliding_window=block.sliding_window,
                        base=base,
                    )
                )
            rows.append(
                _BatchRow(
                    token_id=token,
                    position=position,
                    views=tuple(views),
                    slot_positions=tuple(slots),
                    masks=tuple(masks),
                )
            )
        return tuple(rows)

    def _host_state_rows(
        self, recurrent_states: object, *, count: int, width: int
    ) -> np.ndarray:
        """Return one batch's recurrent rows as host BF16 bits, ``(count, width)``.

        The caller owns these rows, so only the forms a provider can hand over
        explicitly are accepted: a ``(count, width)`` array of BF16 bits as
        ``uint16``, their raw-byte ``uint8`` view, or ``float32`` values, which
        encode to BF16 with the round-to-nearest-even the device uses. A
        ``float32`` round trip through :func:`_bf16_to_float32` is exact, so
        re-encoding the single-row path's ``h_next`` reproduces the same bits.
        """

        matrix = _host_bf16_matrix(recurrent_states, width=width)
        if matrix is None:
            if hasattr(recurrent_states, "ptr") and hasattr(recurrent_states, "nbytes"):
                raise TypeError(
                    "recurrent_states must be a host array of per-request BF16 rows; "
                    "pass prime_batch(...) or a previous forward_batch(...)'s "
                    "next_states, or copy a resident buffer to host first"
                )
            raise TypeError(
                "recurrent_states must be a host (rows, backbone_width) array of "
                "BF16 rows"
            )
        if matrix.shape != (count, width):
            raise ValueError(
                f"recurrent_states has shape {matrix.shape}; {count} rows of width "
                f"{width} are required"
            )
        return np.ascontiguousarray(matrix)

    def forward_batch(
        self,
        token_ids: Sequence[int],
        *,
        positions: Sequence[int],
        shared_kv: Sequence[Mapping[int, object]],
        recurrent_states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run C independent draft steps and return ``(logits, next_states)``.

        ``token_ids[i]``, ``positions[i]``, ``shared_kv[i]`` and
        ``recurrent_states[i]`` all describe request ``i`` and nothing else:
        ``shared_kv[i]`` is that request's ``{backbone_layer: view}`` mapping, so
        two requests may read different caches, different live counts, and
        different positions in one call. Rows need not be contiguous or ordered
        by request id -- the caller decides which requests are active this cycle.

        Returns fresh host arrays: ``logits`` is ``(C, logits_width)`` float32 and
        ``next_states`` is ``(C, backbone_width)`` BF16 bits (``uint16``), ready to
        be the next cycle's ``recurrent_states``. The head keeps no cursor, so the
        caller owns both arrays and may subset, reorder, or drop rows.

        The dense work is packed: one embedding gather, one pre-projection, one
        launch per block projection, one Q rotation over all C rows, and one
        output projection each run at physical row width C. Attention is the
        exception -- a request's K/V, live count and mask are its own, so each
        block issues one ``tokens=1`` attention launch per request. That split is
        the point of the method, and it is why the packed projections are the
        only launches whose width can be amortized.

        The gather runs the *backbone's* embedding table at ``rows=C``, which the
        raw GGUF lookup ABI supports. A backbone whose ``token_embd`` is a dense
        BF16 table has a single-row lookup ABI only, and that case fails by name
        in the shared embedding consumer surface rather than running a wrong row.
        """

        rows = self._resolve_batch_rows(token_ids, positions, shared_kv)
        count = len(rows)
        config = self.weights.config
        hidden = int(config.n_embd)
        intermediate = int(config.n_ff)
        # The backbone's width and vocabulary, from the backbone's own config:
        # the head describes the target it predicts, so a mismatch must be
        # impossible to hide. Same two numbers ``forward`` reads.
        backbone_width = int(self.backbone.hidden_size)
        backbone_vocab = int(self.backbone.vocab_size)
        vocab = int(self._logits_width)
        eps = float(self.eps)
        runtime = get_hip_runtime()
        kwargs = {"stream": self.stream, "runtime": runtime}
        host_state = self._host_state_rows(
            recurrent_states, count=count, width=backbone_width
        )

        # --- packed input: gather, scale, concat with each row's state -------
        xh = self._batch_buffer("xh", count * 2 * backbone_width * _BF16_BYTES)
        x = self._batch_buffer("x", count * backbone_width * _BF16_BYTES)
        tokens = self._batch_buffer("tokens", count * 8)
        copy_host_array_to_device(
            tokens, np.asarray([row.token_id for row in rows], dtype=np.int64)
        )
        launch_gguf_embedding(
            self.backbone_embedding,
            tokens.ptr,
            x.ptr,
            count,
            backbone_width,
            backbone_vocab,
            stream=self.stream,
        )
        gemma4_scale_bf16(
            x.ptr,
            x.ptr,
            count,
            backbone_width,
            float(backbone_width) ** 0.5,
            **kwargs,
        )
        # Host-side concat, one upload: 2 * 2816 BF16 per row is 11 KB, and
        # core/memory.py has no device-to-device copy to interleave it in place.
        host_x = np.empty((count, backbone_width), dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(host_x), x, host_x.nbytes, runtime=runtime
        )
        copy_host_array_to_device(
            xh, np.concatenate((host_x, host_state), axis=1)
        )

        cur = self._batch_buffer("cur", count * hidden * _BF16_BYTES)
        self._linear(
            "nextn_pre_projection",
            xh.ptr,
            cur.ptr,
            count,
            2 * backbone_width,
            hidden,
        )

        for block_id, block in enumerate(self._geometry):
            scratch = self._batch_scratch[block_id]
            prefix = f"blocks.{block_id}"
            head_dim = block.head_dim
            q_width = block.q_width
            q_row = q_width * _BF16_BYTES
            hidden_row = hidden * _BF16_BYTES

            normed = scratch.buffer("normed", count * hidden_row)
            q = scratch.buffer("q", count * q_row)
            q_rot = scratch.buffer("q_rot", count * q_row)
            context = scratch.buffer("context", count * q_row)
            attn = scratch.buffer("attn", count * hidden_row)
            residual = scratch.buffer("residual", count * hidden_row)
            ffn_in = scratch.buffer("ffn_in", count * hidden_row)
            # Gate and up are separate packed buffers rather than one fused
            # ``(rows, 2 * intermediate)`` block: two packed projections can only
            # write row-contiguous destinations, so the fused layout would force
            # the up projection back to one launch per request. The split GeGLU
            # is element-for-element the fused kernel's arithmetic.
            gate = scratch.buffer("gate", count * intermediate * _BF16_BYTES)
            up = scratch.buffer("up", count * intermediate * _BF16_BYTES)
            activated = scratch.buffer("activated", count * intermediate * _BF16_BYTES)
            ffn = scratch.buffer("ffn", count * hidden_row)

            gemma4_rmsnorm_f32w_bf16(
                cur.ptr,
                self.weights.buffer(f"{prefix}.attn_norm").ptr,
                normed.ptr,
                count,
                hidden,
                eps,
                **kwargs,
            )
            self._linear(f"{prefix}.attn_q", normed.ptr, q.ptr, count, hidden, q_width)
            # One per-head norm over ``count * num_heads`` rows: the head axis and
            # the request axis are both flat in this kernel.
            gemma4_head_rmsnorm_f32w_bf16(
                q.ptr,
                self.weights.buffer(f"{prefix}.attn_q_norm").ptr,
                q.ptr,
                count * block.num_heads,
                head_dim,
                eps,
                **kwargs,
            )

            # Per-request rope tables: row i of the table is request i's position.
            cos_values, sin_values = gemma4_rope_cos_sin_tables(
                block.rope, np.asarray([row.position for row in rows], dtype=np.int64)
            )
            cos_host = np.ascontiguousarray(cos_values, dtype=np.float32)
            sin_host = np.ascontiguousarray(sin_values, dtype=np.float32)
            cos = scratch.buffer("cos", cos_host.nbytes)
            copy_host_array_to_device(cos, cos_host)
            sin = scratch.buffer("sin", sin_host.nbytes)
            copy_host_array_to_device(sin, sin_host)
            gemma4_partial_rotary_bf16(
                q.ptr,
                0,
                cos.ptr,
                sin.ptr,
                q_rot.ptr,
                0,
                count,
                block.num_heads,
                0,
                head_dim,
                **kwargs,
            )

            # Attention is per request. Each request's keys, live count, cache
            # base and keep-mask are its own, and there is no batched decode
            # kernel that takes a different cache per row, so the block issues C
            # ``tokens=1`` launches. ``row_offset`` and the mask predicate are
            # both in the cache's slot frame, ``position - base``, exactly as in
            # the single-row path.
            packed_masks = np.concatenate(
                [row.masks[block_id].reshape(-1) for row in rows]
            ).astype(np.uint8)
            mask = scratch.buffer("mask", packed_masks.nbytes)
            copy_host_array_to_device(mask, packed_masks)
            mask_offset = 0
            for index, row in enumerate(rows):
                view = row.views[block_id]
                live = int(view.live)  # type: ignore[attr-defined]
                gemma4_attention_decode_flash_long_bf16(
                    q_rot.ptr + index * q_row,
                    view.key_cache,  # type: ignore[attr-defined]
                    view.value_cache,  # type: ignore[attr-defined]
                    mask.ptr + mask_offset,
                    context.ptr + index * q_row,
                    tokens=1,
                    keys=live,
                    num_heads=block.num_heads,
                    num_kv_heads=block.num_kv_heads,
                    head_dim=head_dim,
                    scale=1.0,
                    row_offset=row.slot_positions[block_id],
                    window=live
                    if block.sliding_window is None
                    else int(block.sliding_window),
                    scratch=scratch.attention,
                    **kwargs,
                )
                mask_offset += row.masks[block_id].size

            # wo, *then* the post-attention norm
            self._linear(
                f"{prefix}.attn_output", context.ptr, attn.ptr, count, q_width, hidden
            )
            gemma4_rmsnorm_f32w_bf16(
                attn.ptr,
                self.weights.buffer(f"{prefix}.post_attention_norm").ptr,
                attn.ptr,
                count,
                hidden,
                eps,
                **kwargs,
            )
            gemma4_branch_add_bf16(
                attn.ptr, cur.ptr, residual.ptr, count * hidden, **kwargs
            )

            # FFN: pre-norm, gate/up, GeGLU, down, post-norm
            gemma4_rmsnorm_f32w_bf16(
                residual.ptr,
                self.weights.buffer(f"{prefix}.ffn_norm").ptr,
                ffn_in.ptr,
                count,
                hidden,
                eps,
                **kwargs,
            )
            self._linear(
                f"{prefix}.ffn_gate", ffn_in.ptr, gate.ptr, count, hidden, intermediate
            )
            self._linear(
                f"{prefix}.ffn_up", ffn_in.ptr, up.ptr, count, hidden, intermediate
            )
            gemma4_gelu_tanh_mul_split_bf16(
                gate.ptr, up.ptr, activated.ptr, count * intermediate, **kwargs
            )
            self._linear(
                f"{prefix}.ffn_down", activated.ptr, ffn.ptr, count, intermediate, hidden
            )
            gemma4_rmsnorm_f32w_bf16(
                ffn.ptr,
                self.weights.buffer(f"{prefix}.post_ffw_norm").ptr,
                ffn.ptr,
                count,
                hidden,
                eps,
                **kwargs,
            )

            # cur = (ffn + attn_out) * layer_output_scale
            gemma4_branch_add_bf16(
                ffn.ptr, residual.ptr, cur.ptr, count * hidden, **kwargs
            )
            gemma4_scale_bf16(
                cur.ptr, cur.ptr, count, hidden, self._layer_scale(prefix), **kwargs
            )

        # --- output norm, logits, next states -------------------------------
        gemma4_rmsnorm_f32w_bf16(
            cur.ptr,
            self.weights.buffer("output_norm").ptr,
            cur.ptr,
            count,
            hidden,
            eps,
            **kwargs,
        )
        logits_buffer = self._batch_buffer("logits", count * vocab * _F32_BYTES)
        launch_gguf_linear(
            self.weights.weight("token_embedding"),
            cur.ptr,
            logits_buffer.ptr,
            count,
            hidden,
            vocab,
            output_dtype="f32",
            stream=self.stream,
        )
        logits = np.empty((count, vocab), dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(logits), logits_buffer, logits.nbytes, runtime=runtime
        )

        next_buffer = self._batch_buffer("next", count * backbone_width * _BF16_BYTES)
        self._linear(
            "nextn_post_projection",
            cur.ptr,
            next_buffer.ptr,
            count,
            hidden,
            backbone_width,
        )
        next_states = np.empty((count, backbone_width), dtype=np.uint16)
        copy_device_to_host(
            host_array_ptr(next_states), next_buffer, next_states.nbytes, runtime=runtime
        )
        return logits, next_states

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
        for scratch in (*self._scratch, *self._batch_scratch):
            scratch.release()
        self._staged.clear()
        self._batch.clear()
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

    def draft(
        self,
        token: int,
        *,
        hidden_row: int = -1,
        select: Callable[[np.ndarray], int] | None = None,
    ) -> list[int]:
        """Propose up to ``max_drafts`` tokens to follow ``token``.

        ``token`` is the token the backbone just sampled, not the last token it
        processed: the head's first step is fed the sampled token together with
        the hidden row that produced it.

        ``select`` chooses each drafted token from the head's logits row and is
        called once per step, in order; the next step is conditioned on what it
        returned. ``None`` takes the row's argmax. A sampling caller passes its
        own draw and keeps the distribution it drew from, which the exact
        speculative accept needs.
        """

        self.head.prime(self.runner.hidden_state(row=hidden_row))
        shared = {
            index: self.runner.shared_kv(index) for index in range(self.runner.layer_count)
        }
        position = self.runner.position
        drafts: list[int] = []
        for _ in range(self.max_drafts):
            logits, _h_next = self.head.forward(token, position=position, shared_kv=shared)
            token = int(np.argmax(logits)) if select is None else int(select(logits))
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
