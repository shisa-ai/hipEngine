"""The assistant head's batched forward, checked on a simulated device.

``Gemma4AssistantHead.forward_batch`` packs C independent requests into one
launch per dense projection and keeps attention per request. Two claims matter
and neither needs a GPU to check:

* the projections run at **physical row width C** -- one launch of ``rows=C``
  per weight, not C launches of ``rows=1`` -- while attention runs once per
  request, because each request has its own KV, live count, position and mask;
* rows do not contaminate each other. Request ``i``'s token, position, state and
  view reach row ``i`` and nothing else, a ragged active set (C1/C2/C3/C4) leaves
  no cursor behind, and the single-row ``forward`` path is untouched.

The harness replaces ``malloc``/``free``/the copy helpers and every launch with a
fake device: a byte-addressed store plus a launch log. Launches that write an
output write a row-identifying pattern, so a test can read back exactly which
request's data landed in which row. That is stronger than inspecting arguments
alone, and it is why the pointer arithmetic (``buffer.ptr + row * width``) is
exercised rather than assumed.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.gemma4_assistant as assistant_module
from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rope import gemma4_rope_cos_sin_tables
from hipengine.loading.gemma4_assistant_device import assistant_slot_names
from hipengine.loading.gemma4_assistant_gguf import Gemma4AssistantConfig
from hipengine.loading.gemma4_gguf import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    Gemma4GGUFConfig,
    Gemma4GGUFRoPEConfig,
)
from hipengine.runtime.gemma4_assistant import (
    Gemma4AssistantHead,
    _bf16_to_float32,
    _float32_to_bf16,
    gemma4_assistant_keep_mask,
)

# Fake allocations are spaced far apart so that ``base.ptr + row * width`` can be
# resolved back to (allocation, offset) unambiguously.
_STRIDE = 1 << 32

_BACKBONE_WIDTH = 6
_BACKBONE_VOCAB = 32
_HEAD_HIDDEN = 4
_HEAD_INTERMEDIATE = 6
_HEAD_HEADS = 2
_HEAD_KV_HEADS = 1
_HEAD_KEY_LENGTH = 4
_HEAD_KEY_LENGTH_SWA = 2
_HEAD_SLIDING_WINDOW = 8
_HEAD_VOCAB = 32
_LAYER_SCALE = 0.5


class FakeDevice:
    """A byte-addressed device stand-in plus a launch log."""

    def __init__(self) -> None:
        self._next_ptr = _STRIDE
        self.blocks: dict[int, bytearray] = {}
        self.launches: list[tuple[str, dict]] = []

    # --- allocation and copies ------------------------------------------

    def malloc(self, nbytes: int, *, runtime=None) -> DeviceBuffer:
        ptr = self._next_ptr
        self._next_ptr += _STRIDE
        self.blocks[ptr] = bytearray(int(nbytes))
        return DeviceBuffer(ptr, int(nbytes))

    def free(self, buffer: DeviceBuffer, *, runtime=None) -> None:
        self.blocks.pop(int(buffer.ptr), None)

    def _locate(self, ptr: int, nbytes: int) -> tuple[bytearray, int]:
        base = max(base for base in self.blocks if base <= ptr)
        offset = int(ptr) - base
        block = self.blocks[base]
        assert offset + nbytes <= len(block), (
            f"access of {nbytes} bytes at +{offset} runs past an "
            f"{len(block)}-byte allocation"
        )
        return block, offset

    def write(self, ptr: int, payload: bytes) -> None:
        block, offset = self._locate(ptr, len(payload))
        block[offset : offset + len(payload)] = payload

    def read(self, ptr: int, nbytes: int) -> bytes:
        block, offset = self._locate(ptr, nbytes)
        return bytes(block[offset : offset + nbytes])

    def copy_host_array_to_device(self, buffer, array, nbytes=None, *, runtime=None) -> None:
        count = int(array.nbytes) if nbytes is None else int(nbytes)
        payload = np.ascontiguousarray(array).view(np.uint8).reshape(-1)[:count]
        self.write(int(buffer.ptr), payload.tobytes())

    def copy_device_to_host(self, host_ptr: int, buffer, nbytes=None, *, runtime=None) -> None:
        count = int(buffer.nbytes) if nbytes is None else int(nbytes)
        ctypes.memmove(host_ptr, self.read(int(buffer.ptr), count), count)

    # --- launch log ------------------------------------------------------

    def record(self, name: str, **kwargs) -> None:
        self.launches.append((name, kwargs))

    def named(self, name: str) -> list[dict]:
        return [kwargs for launch, kwargs in self.launches if launch == name]

    def only(self, name: str) -> dict:
        calls = self.named(name)
        assert len(calls) == 1, f"{name} ran {len(calls)} launches, expected one"
        return calls[0]


# --- synthetic configs -------------------------------------------------


def _backbone_config() -> Gemma4GGUFConfig:
    """A four-layer backbone whose last two layers are the two the head binds."""

    return Gemma4GGUFConfig(
        architecture="gemma4",
        block_count=4,
        hidden_size=_BACKBONE_WIDTH,
        vocab_size=_BACKBONE_VOCAB,
        context_length=64,
        rms_norm_eps=1e-5,
        head_counts=(_HEAD_HEADS,) * 4,
        head_count_kv=(_HEAD_KV_HEADS,) * 4,
        key_length=_HEAD_KEY_LENGTH,
        key_length_swa=_HEAD_KEY_LENGTH_SWA,
        value_length=_HEAD_KEY_LENGTH,
        value_length_swa=_HEAD_KEY_LENGTH_SWA,
        sliding_window=_HEAD_SLIDING_WINDOW,
        # Head blocks bind ``block_count - 2`` (sliding) and ``block_count - 1``
        # (full), so those two entries are what the geometry check reads.
        layer_types=(FULL_ATTENTION, FULL_ATTENTION, SLIDING_ATTENTION, FULL_ATTENTION),
        feed_forward_lengths=(8,) * 4,
        expert_count=0,
        expert_used_count=0,
        expert_feed_forward_length=0,
        final_logit_softcapping=None,
        full_rope=Gemma4GGUFRoPEConfig(
            rope_type="proportional",
            head_dim=_HEAD_KEY_LENGTH,
            rotated_pairs=_HEAD_KEY_LENGTH // 2,
            freq_base=1.0e6,
        ),
        swa_rope=Gemma4GGUFRoPEConfig(
            rope_type="default",
            head_dim=_HEAD_KEY_LENGTH_SWA,
            rotated_pairs=_HEAD_KEY_LENGTH_SWA // 2,
            freq_base=1.0e4,
        ),
        shared_kv_layers=4,
        tied_embeddings=True,
        bos_token_id=2,
        eos_token_ids=(1,),
    )


def _assistant_config() -> Gemma4AssistantConfig:
    return Gemma4AssistantConfig(
        block_count=4,
        n_embd=_HEAD_HIDDEN,
        n_embd_backbone=_BACKBONE_WIDTH,
        n_ff=_HEAD_INTERMEDIATE,
        n_head=_HEAD_HEADS,
        n_head_kv=(_HEAD_KV_HEADS,) * 4,
        key_length=_HEAD_KEY_LENGTH,
        value_length=_HEAD_KEY_LENGTH,
        key_length_swa=_HEAD_KEY_LENGTH_SWA,
        value_length_swa=_HEAD_KEY_LENGTH_SWA,
        sliding_window=_HEAD_SLIDING_WINDOW,
        is_swa=(True, False, True, False),
        n_rot=_HEAD_KEY_LENGTH,
        n_rot_swa=_HEAD_KEY_LENGTH_SWA,
        nextn_predict_layers=4,
    )


# --- fake weights ------------------------------------------------------


class FakeWeight:
    """The three things the head reads off a weight: slot, shape, and buffer."""

    def __init__(self, slot: str, shape: tuple[int, ...], buffer: DeviceBuffer) -> None:
        self.slot_name = slot
        self.spec = SimpleNamespace(source=SimpleNamespace(shape=tuple(shape)))
        self._buffer = buffer

    def allocation(self, name: str | None = None):
        return SimpleNamespace(buffer=self._buffer)


class FakeWeights:
    def __init__(
        self,
        config: Gemma4AssistantConfig,
        device: FakeDevice,
        *,
        token_embedding_shape: tuple[int, int] | None = None,
    ) -> None:
        self.config = config
        self._device = device
        self._weights: dict[str, FakeWeight] = {}
        for slot in assistant_slot_names(config):
            if slot.endswith("layer_output_scale"):
                continue
            # ``GGUFTensorInfo.shape`` is row-major, so a linear's shape is
            # ``(out_features, in_features)`` -- the same order the backbone
            # loader reads ``ffn_gate``'s output count from.
            shape = (
                token_embedding_shape or (_HEAD_VOCAB, config.n_embd)
                if slot == "token_embedding"
                else (config.n_embd, config.n_embd)
            )
            self._weights[slot] = FakeWeight(slot, shape, device.malloc(shape[0] * shape[1] * 2))
        # ``layer_output_scale`` is a 1-element F32 the forward reads back.
        for block_id in range(config.block_count):
            buffer = device.malloc(4)
            device.write(buffer.ptr, np.asarray([_LAYER_SCALE], dtype=np.float32).tobytes())
            self._weights[f"blocks.{block_id}.layer_output_scale"] = FakeWeight(
                f"blocks.{block_id}.layer_output_scale", (1, 1), buffer
            )

    def weight(self, slot: str) -> FakeWeight:
        return self._weights[slot]

    def buffer(self, slot: str) -> DeviceBuffer:
        return self._weights[slot].allocation("raw").buffer


@dataclass
class FakeView:
    """A shared-KV read view, the four fields the head consumes."""

    key_cache: int
    value_cache: int
    live: int
    base: int = 0


def _install_launches(device: FakeDevice, monkeypatch: pytest.MonkeyPatch) -> None:
    def linear(
        weight,
        x_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        *,
        output_dtype="bf16",
        stream=0,
        **kwargs,
    ):
        device.record(
            "linear",
            slot=weight.slot_name,
            x=int(x_ptr),
            out=int(out_ptr),
            rows=int(rows),
            in_features=int(in_features),
            out_features=int(out_features),
            output_dtype=output_dtype,
        )
        if weight.slot_name == "nextn_post_projection":
            pattern = np.empty((rows, out_features), dtype=np.uint16)
            for index in range(rows):
                pattern[index] = np.uint16(index + 1)
            device.write(int(out_ptr), pattern.tobytes())
        elif weight.slot_name == "token_embedding":
            pattern = np.empty((rows, out_features), dtype=np.float32)
            for index in range(rows):
                pattern[index] = np.float32(index) + np.float32(0.5)
            device.write(int(out_ptr), pattern.tobytes())

    def embedding(
        weight,
        token_ids_ptr,
        out_ptr,
        rows,
        hidden_size,
        vocab_size,
        *,
        output_dtype="bf16",
        stream=0,
        **kwargs,
    ):
        ids = np.frombuffer(device.read(int(token_ids_ptr), rows * 8), dtype=np.int64)
        device.record(
            "embedding",
            token_ids=tuple(int(value) for value in ids),
            out=int(out_ptr),
            rows=int(rows),
            hidden_size=int(hidden_size),
            vocab_size=int(vocab_size),
        )
        # Raw BF16 bits ``token + 0x100``, so a row's identity is readable back.
        pattern = np.empty((rows, hidden_size), dtype=np.uint16)
        for index, token in enumerate(ids):
            pattern[index] = np.uint16(int(token) + 0x100)
        device.write(int(out_ptr), pattern.tobytes())

    def rmsnorm(src, weight, out, rows, hidden_size, eps=1e-6, *, stream=0, **kwargs):
        device.record(
            "rmsnorm",
            src=int(src),
            weight=int(weight),
            out=int(out),
            rows=int(rows),
            hidden=int(hidden_size),
            eps=float(eps),
        )
        # Deterministic +1 on the raw bits, so a caller can tell the norm's
        # output from its input without pretending to be an RMSNorm.
        payload = np.frombuffer(device.read(int(src), rows * hidden_size * 2), dtype=np.uint16)
        device.write(int(out), (payload + np.uint16(1)).tobytes())

    def head_rmsnorm(src, weight, out, rows, head_dim, eps=1e-6, *, stream=0, **kwargs):
        device.record(
            "head_rmsnorm",
            weight=int(weight),
            rows=int(rows),
            head_dim=int(head_dim),
            eps=float(eps),
        )

    def rotary(
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        *,
        stream=0,
        **kwargs,
    ):
        device.record(
            "rotary",
            q=int(q),
            cos=int(cos),
            sin=int(sin),
            q_out=int(q_out),
            tokens=int(tokens),
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
        )

    def attention(
        q,
        key,
        value,
        mask,
        out,
        *,
        tokens,
        keys,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        row_offset=0,
        window=0,
        scratch=None,
        stream=0,
        **kwargs,
    ):
        device.record(
            "attention",
            q=int(q),
            key=int(key),
            value=int(value),
            mask=int(mask),
            out=int(out),
            tokens=int(tokens),
            keys=int(keys),
            num_heads=int(num_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            scale=float(scale),
            row_offset=int(row_offset),
            window=int(window),
        )

    def branch_add(a, b, out, total, *, stream=0, **kwargs):
        device.record("branch_add", total=int(total))

    def scale(x, out, rows, hidden_size, factor, *, stream=0, **kwargs):
        device.record("scale", rows=int(rows), hidden=int(hidden_size), factor=float(factor))

    def gelu_split(gate, up, out, total, *, stream=0, **kwargs):
        device.record("gelu_split", gate=int(gate), up=int(up), out=int(out), total=int(total))

    def gelu_fused(gate_up, out, rows, intermediate, *, stream=0, **kwargs):
        device.record("gelu_fused", rows=int(rows), intermediate=int(intermediate))

    monkeypatch.setattr(assistant_module, "launch_gguf_linear", linear)
    monkeypatch.setattr(assistant_module, "launch_gguf_embedding", embedding)
    monkeypatch.setattr(assistant_module, "gemma4_rmsnorm_f32w_bf16", rmsnorm)
    monkeypatch.setattr(assistant_module, "gemma4_head_rmsnorm_f32w_bf16", head_rmsnorm)
    monkeypatch.setattr(assistant_module, "gemma4_partial_rotary_bf16", rotary)
    monkeypatch.setattr(assistant_module, "gemma4_attention_decode_flash_long_bf16", attention)
    monkeypatch.setattr(assistant_module, "gemma4_branch_add_bf16", branch_add)
    monkeypatch.setattr(assistant_module, "gemma4_scale_bf16", scale)
    monkeypatch.setattr(assistant_module, "gemma4_gelu_tanh_mul_split_bf16", gelu_split)
    monkeypatch.setattr(assistant_module, "gemma4_gelu_tanh_mul_bf16", gelu_fused)


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> FakeDevice:
    fake = FakeDevice()
    monkeypatch.setattr(assistant_module, "malloc", fake.malloc)
    monkeypatch.setattr(assistant_module, "free", fake.free)
    monkeypatch.setattr(
        assistant_module, "copy_host_array_to_device", fake.copy_host_array_to_device
    )
    monkeypatch.setattr(
        assistant_module, "copy_device_to_host", fake.copy_device_to_host
    )
    # No ROCm on a CPU runner, and nothing below needs it.
    monkeypatch.setattr(assistant_module, "get_hip_runtime", lambda: None)
    _install_launches(fake, monkeypatch)
    return fake


@pytest.fixture
def head(device: FakeDevice) -> Gemma4AssistantHead:
    return _make_head(device)


def _make_head(
    device: FakeDevice, *, token_embedding_shape: tuple[int, int] | None = None
) -> Gemma4AssistantHead:
    backbone = _backbone_config()
    weights = FakeWeights(
        _assistant_config(), device, token_embedding_shape=token_embedding_shape
    )
    return Gemma4AssistantHead(
        weights=weights,
        backbone=backbone,
        backbone_embedding=FakeWeight("token_embedding", (1, 1), device.malloc(8)),
        backbone_output_norm=device.malloc(_BACKBONE_WIDTH * 4),
        capacity=64,
        eps=1e-6,
    )


# --- request helpers ---------------------------------------------------


def _states(count: int, *, seed: int = 0) -> np.ndarray:
    """C distinct BF16 recurrent rows, ``(count, backbone_width)`` uint16."""

    rows = np.empty((count, _BACKBONE_WIDTH), dtype=np.uint16)
    for index in range(count):
        rows[index] = np.uint16(0x3F00 + seed * 0x10 + index)
    return rows


def _requests(head: Gemma4AssistantHead, device: FakeDevice, specs):
    """Build ``(token_ids, positions, shared_kv)`` from ``(token, position, live, base)``."""

    tokens: list[int] = []
    positions: list[int] = []
    shared: list[dict] = []
    for index, (token, position, live, base) in enumerate(specs):
        tokens.append(token)
        positions.append(position)
        shared.append(
            {
                block.kv_layer: FakeView(
                    key_cache=0x7000 + 0x100 * index + block.kv_layer,
                    value_cache=0x8000 + 0x100 * index + block.kv_layer,
                    live=live,
                    base=base,
                )
                for block in head.geometry
            }
        )
    return tokens, positions, shared


# --- packed projections ------------------------------------------------


def test_forward_batch_packs_every_projection_at_the_request_count(head, device) -> None:
    """One ``rows=C`` launch per weight, not C launches of ``rows=1``."""

    tokens, positions, shared = _requests(
        head,
        device,
        [(3, 5, 5, 0), (7, 9, 9, 0), (11, 4, 4, 0), (13, 12, 12, 0)],
    )
    logits, next_states = head.forward_batch(
        tokens, positions=positions, shared_kv=shared, recurrent_states=_states(4)
    )

    linear = device.named("linear")
    by_slot: dict[str, list[dict]] = {}
    for call in linear:
        by_slot.setdefault(call["slot"], []).append(call)

    expected = {"nextn_pre_projection", "nextn_post_projection", "token_embedding"}
    for block_id in range(head.block_count):
        expected |= {
            f"blocks.{block_id}.attn_q",
            f"blocks.{block_id}.attn_output",
            f"blocks.{block_id}.ffn_gate",
            f"blocks.{block_id}.ffn_up",
            f"blocks.{block_id}.ffn_down",
        }
    assert set(by_slot) == expected
    for slot, calls in by_slot.items():
        assert len(calls) == 1, f"{slot} ran {len(calls)} launches for four requests"
        assert calls[0]["rows"] == 4, f"{slot} ran at row width {calls[0]['rows']}"

    assert by_slot["nextn_pre_projection"][0]["in_features"] == 2 * _BACKBONE_WIDTH
    assert by_slot["nextn_pre_projection"][0]["out_features"] == _HEAD_HIDDEN
    assert by_slot["nextn_post_projection"][0]["out_features"] == _BACKBONE_WIDTH
    assert by_slot["token_embedding"][0]["output_dtype"] == "f32"
    for block_id, block in enumerate(head.geometry):
        assert by_slot[f"blocks.{block_id}.attn_q"][0]["out_features"] == block.q_width
        assert by_slot[f"blocks.{block_id}.ffn_gate"][0]["out_features"] == _HEAD_INTERMEDIATE
        assert by_slot[f"blocks.{block_id}.ffn_down"][0]["in_features"] == _HEAD_INTERMEDIATE

    # The gather and the rotation are packed too.
    assert device.only("embedding")["rows"] == 4
    assert device.only("embedding")["token_ids"] == (3, 7, 11, 13)
    rotaries = device.named("rotary")
    assert len(rotaries) == head.block_count
    assert {call["tokens"] for call in rotaries} == {4}

    # The per-head Q norm is one launch over requests * heads.
    head_norms = device.named("head_rmsnorm")
    assert len(head_norms) == head.block_count
    assert {call["rows"] for call in head_norms} == {4 * _HEAD_HEADS}

    # Every norm in the block loop runs at the packed width.
    assert {call["rows"] for call in device.named("rmsnorm")} == {4}

    assert logits.shape == (4, _HEAD_VOCAB)
    assert logits.dtype == np.float32
    assert next_states.shape == (4, _BACKBONE_WIDTH)
    assert next_states.dtype == np.uint16


def test_forward_batch_uses_one_split_geglu_per_block_over_disjoint_buffers(head, device) -> None:
    tokens, positions, shared = _requests(head, device, [(3, 5, 5, 0), (7, 9, 9, 0)])
    head.forward_batch(
        tokens, positions=positions, shared_kv=shared, recurrent_states=_states(2)
    )

    gelu = device.named("gelu_split")
    assert len(gelu) == head.block_count
    for call in gelu:
        assert call["total"] == 2 * _HEAD_INTERMEDIATE
        assert call["gate"] != call["up"]
        assert call["out"] not in {call["gate"], call["up"]}
    assert not device.named("gelu_fused")


# --- attention is per request -----------------------------------------


def test_attention_runs_once_per_request_with_that_requests_kv_position_and_mask(
    head, device
) -> None:
    specs = [(3, 5, 5, 0), (7, 9, 9, 0), (11, 4, 4, 0)]
    tokens, positions, shared = _requests(head, device, specs)
    head.forward_batch(
        tokens, positions=positions, shared_kv=shared, recurrent_states=_states(3)
    )

    attention = device.named("attention")
    assert len(attention) == len(specs) * head.block_count
    for block_id, block in enumerate(head.geometry):
        launches = attention[block_id * len(specs) : (block_id + 1) * len(specs)]
        for index, (launch, spec) in enumerate(zip(launches, specs)):
            _token, position, live, base = spec
            view = shared[index][block.kv_layer]
            assert launch["tokens"] == 1
            assert launch["keys"] == live
            assert launch["row_offset"] == position - base
            assert launch["key"] == view.key_cache
            assert launch["value"] == view.value_cache
            assert launch["num_heads"] == block.num_heads
            assert launch["num_kv_heads"] == block.num_kv_heads
            assert launch["head_dim"] == block.head_dim
            assert launch["scale"] == 1.0
            assert launch["window"] == (
                live if block.sliding_window is None else block.sliding_window
            )
            # The mask this request's launch reads is its own predicate.
            expected = gemma4_assistant_keep_mask(
                position, live, sliding_window=block.sliding_window, base=base
            )
            assert device.read(launch["mask"], live) == expected.reshape(-1).tobytes()

        # Each request's query and context rows are its own slice of the packed
        # buffer, in request order.
        q_rows = [launch["q"] for launch in launches]
        out_rows = [launch["out"] for launch in launches]
        q_row_bytes = block.q_width * 2
        assert q_rows == [q_rows[0] + index * q_row_bytes for index in range(len(specs))]
        assert out_rows == [out_rows[0] + index * q_row_bytes for index in range(len(specs))]


def test_each_block_reads_its_own_layers_slot_frame_and_window(head, device) -> None:
    """A rebased sliding layer and a global layer do not share a slot or a mask.

    ``Gemma4SharedKV.base`` is nonzero only for a bounded sliding layer whose
    retained band was rebased, so the two blocks a request binds can sit in
    different slot frames. Reading the block's own view -- rather than request
    0's, or block 0's -- is what this pins.
    """

    swa_layer = head.geometry[0].kv_layer
    full_layer = head.geometry[1].kv_layer
    assert swa_layer != full_layer

    shared = [
        {
            swa_layer: FakeView(key_cache=0x1100, value_cache=0x1200, live=12, base=0),
            full_layer: FakeView(key_cache=0x2100, value_cache=0x2200, live=12, base=0),
        },
        {
            # Request 1's sliding band has been rebased; its global layer has not.
            swa_layer: FakeView(key_cache=0x3100, value_cache=0x3200, live=5, base=3),
            full_layer: FakeView(key_cache=0x4100, value_cache=0x4200, live=9, base=0),
        },
    ]
    head.forward_batch(
        [3, 7],
        positions=[12, 8],
        shared_kv=shared,
        recurrent_states=_states(2),
    )

    attention = device.named("attention")
    assert len(attention) == 2 * head.block_count
    for block_id, block in enumerate(head.geometry):
        launches = attention[block_id * 2 : (block_id + 1) * 2]
        for index, launch in enumerate(launches):
            view = shared[index][block.kv_layer]
            position = (12, 8)[index]
            assert launch["row_offset"] == position - view.base
            assert launch["keys"] == view.live
            expected = gemma4_assistant_keep_mask(
                position, view.live, sliding_window=block.sliding_window, base=view.base
            )
            assert device.read(launch["mask"], view.live) == expected.reshape(-1).tobytes()

    # The windowed block's mask really is narrower than the global block's on the
    # same request, so the two masks above are not accidentally equal.
    windowed = device.read(attention[0]["mask"], 12)
    global_row = device.read(attention[2]["mask"], 12)
    assert windowed != global_row
    assert windowed[:4] == bytes(4)


def test_rope_tables_carry_each_requests_own_position(head, device) -> None:
    positions_by_request = [5, 9, 4, 12]
    tokens = [3, 7, 11, 13]
    shared = []
    for index, position in enumerate(positions_by_request):
        shared.append(
            {
                block.kv_layer: FakeView(
                    key_cache=0x1000 * (index + 1),
                    value_cache=0x2000 * (index + 1),
                    live=position,
                    base=0,
                )
                for block in head.geometry
            }
        )
    head.forward_batch(
        tokens,
        positions=positions_by_request,
        shared_kv=shared,
        recurrent_states=_states(4),
    )

    rotaries = device.named("rotary")
    for block_id, block in enumerate(head.geometry):
        cos_values, sin_values = gemma4_rope_cos_sin_tables(
            block.rope, np.asarray(positions_by_request, dtype=np.int64)
        )
        launch = rotaries[block_id]
        row_nbytes = block.head_dim * 4
        assert device.read(launch["cos"], 4 * row_nbytes) == np.ascontiguousarray(
            cos_values, dtype=np.float32
        ).tobytes()
        assert device.read(launch["sin"], 4 * row_nbytes) == np.ascontiguousarray(
            sin_values, dtype=np.float32
        ).tobytes()


# --- state rows --------------------------------------------------------


def test_each_requests_token_and_state_reach_its_own_concat_row(head, device) -> None:
    specs = [(3, 5, 5, 0), (7, 9, 9, 0), (11, 4, 4, 0), (13, 12, 12, 0)]
    tokens, positions, shared = _requests(head, device, specs)
    states = _states(4, seed=3)
    head.forward_batch(
        tokens, positions=positions, shared_kv=shared, recurrent_states=states
    )

    pre_projection = [
        call for call in device.named("linear") if call["slot"] == "nextn_pre_projection"
    ][0]
    packed = np.frombuffer(
        device.read(pre_projection["x"], 4 * 2 * _BACKBONE_WIDTH * 2), dtype=np.uint16
    ).reshape(4, 2 * _BACKBONE_WIDTH)
    for index, token in enumerate(tokens):
        # The embedding stand-in writes ``token + 0x100`` as raw bits.
        assert packed[index, :_BACKBONE_WIDTH].tolist() == [token + 0x100] * _BACKBONE_WIDTH
        assert packed[index, _BACKBONE_WIDTH:].tolist() == states[index].tolist()


def test_next_states_come_back_in_request_order_and_are_caller_owned(head, device) -> None:
    tokens, positions, shared = _requests(head, device, [(3, 5, 5, 0), (7, 9, 9, 0), (11, 4, 4, 0)])
    logits, next_states = head.forward_batch(
        tokens, positions=positions, shared_kv=shared, recurrent_states=_states(3)
    )

    # The stand-in writes ``row_index + 1`` as raw bits, so a swapped row shows.
    assert [int(next_states[index, 0]) for index in range(3)] == [1, 2, 3]
    assert next_states.base is None and next_states.flags.writeable
    assert [float(logits[index, 0]) for index in range(3)] == [0.5, 1.5, 2.5]

    # Nothing on the head aliases the returned arrays: the caller may keep or
    # drop them, and a second call returns fresh objects.
    next_states[:] = 0
    logits[:] = 0.0
    again_logits, again_states = head.forward_batch(
        tokens, positions=positions, shared_kv=shared, recurrent_states=_states(3)
    )
    assert again_states is not next_states and again_logits is not logits
    assert [int(again_states[index, 0]) for index in range(3)] == [1, 2, 3]


def test_a_ragged_active_set_leaves_no_state_behind(head, device) -> None:
    """C4 then C1 then C2: each call is sized and filled by its own rows only."""

    four = [(3, 5, 5, 0), (7, 9, 9, 0), (11, 4, 4, 0), (13, 12, 12, 0)]
    tokens, positions, shared = _requests(head, device, four)
    head.forward_batch(
        tokens, positions=positions, shared_kv=shared, recurrent_states=_states(4)
    )

    # The single-row path's own state must not have been touched by the batch.
    recurrent_bytes = device.read(head._recurrent.ptr, head._recurrent.nbytes)
    assert recurrent_bytes == bytes(head._recurrent.nbytes)

    device.launches.clear()
    single = [(11, 4, 4, 0)]
    tokens, positions, shared = _requests(head, device, single)
    head.forward_batch(
        tokens, positions=positions, shared_kv=shared, recurrent_states=_states(1, seed=5)
    )

    for call in device.named("linear"):
        assert call["rows"] == 1, f"{call['slot']} kept the previous width"
    assert device.only("embedding")["rows"] == 1
    assert device.only("embedding")["token_ids"] == (11,)
    pre = [call for call in device.named("linear") if call["slot"] == "nextn_pre_projection"][0]
    # The upload is sized by this call's count, so no stale row is even present.
    packed = np.frombuffer(
        device.read(pre["x"], 2 * _BACKBONE_WIDTH * 2), dtype=np.uint16
    ).reshape(1, 2 * _BACKBONE_WIDTH)
    assert packed[0, :_BACKBONE_WIDTH].tolist() == [11 + 0x100] * _BACKBONE_WIDTH
    assert packed[0, _BACKBONE_WIDTH:].tolist() == _states(1, seed=5)[0].tolist()
    assert len(device.named("attention")) == head.block_count

    # A subset of a previous cycle's rows can be handed straight back in.
    device.launches.clear()
    carried = np.stack([_states(4, seed=7)[0], _states(4, seed=7)[2]])
    tokens, positions, shared = _requests(head, device, [four[0], four[2]])
    head.forward_batch(
        tokens, positions=positions, shared_kv=shared, recurrent_states=carried
    )
    pre = [call for call in device.named("linear") if call["slot"] == "nextn_pre_projection"][0]
    packed = np.frombuffer(
        device.read(pre["x"], 2 * 2 * _BACKBONE_WIDTH * 2), dtype=np.uint16
    ).reshape(2, 2 * _BACKBONE_WIDTH)
    assert packed[:, _BACKBONE_WIDTH:].tolist() == carried.tolist()
    assert device.only("embedding")["token_ids"] == (3, 11)
    assert device.read(head._recurrent.ptr, head._recurrent.nbytes) == recurrent_bytes


# --- validation happens before any device write ------------------------


def _valid_batch(head: Gemma4AssistantHead, device: FakeDevice):
    tokens, positions, shared = _requests(head, device, [(3, 5, 5, 0), (7, 9, 9, 0)])
    return {
        "token_ids": tokens,
        "positions": positions,
        "shared_kv": shared,
        "recurrent_states": _states(2),
    }


def _snapshot(device: FakeDevice, head: Gemma4AssistantHead) -> dict[str, bytes]:
    return {
        name: device.read(buffer.ptr, buffer.nbytes)
        for name, buffer in {**head._batch, **head._staged}.items()
    }


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda batch: batch.update(token_ids=[3]), ValueError),
        (lambda batch: batch.update(positions=[5]), ValueError),
        (lambda batch: batch.update(shared_kv=batch["shared_kv"][:1]), ValueError),
        (lambda batch: batch.update(token_ids=[3, _BACKBONE_VOCAB]), ValueError),
        (lambda batch: batch.update(positions=[5, -1]), ValueError),
        (lambda batch: batch.update(positions=[5, 40]), ValueError),
        (lambda batch: batch["shared_kv"][1].pop(3), ValueError),
        (
            lambda batch: batch.update(
                recurrent_states=np.zeros((2, _BACKBONE_WIDTH + 1), np.uint16)
            ),
            ValueError,
        ),
        (
            lambda batch: batch.update(
                recurrent_states=np.zeros((3, _BACKBONE_WIDTH), np.uint16)
            ),
            ValueError,
        ),
        (
            lambda batch: batch.update(
                recurrent_states=np.zeros((2, _BACKBONE_WIDTH), np.float16)
            ),
            ValueError,
        ),
        (lambda batch: batch.update(recurrent_states=DeviceBuffer(0x10, 64)), TypeError),
        (lambda batch: batch.update(token_ids=3), TypeError),
        (lambda batch: batch.update(token_ids=[3, True]), TypeError),
        (lambda batch: batch.update(token_ids=[]), ValueError),
    ],
)
def test_forward_batch_refuses_bad_requests_before_writing_anything(
    head, device, mutate, error
) -> None:
    # One good call first, so the packed buffers hold a previous cycle's rows.
    good = _valid_batch(head, device)
    head.forward_batch(**good)
    before = _snapshot(device, head)
    device.launches.clear()

    batch = _valid_batch(head, device)
    mutate(batch)
    with pytest.raises(error):
        head.forward_batch(**batch)

    assert device.launches == [], "a refused batch still launched device work"
    assert _snapshot(device, head) == before


def test_forward_batch_refuses_a_view_that_is_not_a_read_view(head, device) -> None:
    batch = _valid_batch(head, device)
    batch["shared_kv"][1][2] = SimpleNamespace(key_cache=1, value_cache=2, base=0)
    with pytest.raises(TypeError, match="live"):
        head.forward_batch(**batch)
    assert device.launches == []


def test_forward_batch_refuses_a_closed_head(head, device) -> None:
    head.close()
    batch = _valid_batch(head, device)
    with pytest.raises(RuntimeError, match="closed"):
        head.forward_batch(**batch)


# --- the seed ----------------------------------------------------------


def test_prime_batch_applies_the_backbone_final_norm_to_every_row(head, device) -> None:
    width = _BACKBONE_WIDTH
    raw = _states(3, seed=9)
    sources = []
    for index in range(3):
        buffer = device.malloc(width * 2)
        device.write(buffer.ptr, raw[index].tobytes())
        sources.append(buffer)

    primed = head.prime_batch(sources)

    assert primed.shape == (3, width)
    assert primed.dtype == np.uint16
    launch = device.only("rmsnorm")
    assert launch["rows"] == 3
    assert launch["hidden"] == width
    assert launch["eps"] == _backbone_config().rms_norm_eps
    assert launch["weight"] == int(head.backbone_output_norm.ptr)
    # The stand-in writes ``bits + 1``, so the result is the norm's output for
    # every row rather than the input echoed back.
    assert primed.tolist() == (raw + np.uint16(1)).tolist()


def test_prime_batch_accepts_a_host_bf16_matrix(head, device) -> None:
    """The parent hands over the backbone's rows as one host matrix."""

    raw = _states(3, seed=4)
    primed = head.prime_batch(raw)

    assert primed.shape == (3, _BACKBONE_WIDTH)
    assert primed.dtype == np.uint16
    launch = device.only("rmsnorm")
    assert launch["rows"] == 3
    assert launch["hidden"] == _BACKBONE_WIDTH
    assert launch["eps"] == _backbone_config().rms_norm_eps
    assert primed.tolist() == (raw + np.uint16(1)).tolist()

    # The byte view and the float32 form of the same rows give the same result.
    device.launches.clear()
    assert head.prime_batch(raw.view(np.uint8)).tolist() == primed.tolist()
    device.launches.clear()
    values = _bf16_to_float32(raw)
    assert head.prime_batch(values).tolist() == primed.tolist()
    assert head.prime_batch(values.astype(np.float32)).dtype == np.uint16


def test_prime_batch_refuses_a_short_or_non_device_row(head, device) -> None:
    with pytest.raises(ValueError, match="less than"):
        head.prime_batch([device.malloc(2)])
    with pytest.raises(TypeError):
        head.prime_batch([object()])
    with pytest.raises(ValueError, match="at least one"):
        head.prime_batch([])
    assert device.launches == []


def test_a_primed_batch_feeds_forward_batch(head, device) -> None:
    sources = [device.malloc(_BACKBONE_WIDTH * 2) for _ in range(2)]
    for index, buffer in enumerate(sources):
        device.write(buffer.ptr, _states(2, seed=1)[index].tobytes())
    primed = head.prime_batch(sources)

    tokens, positions, shared = _requests(head, device, [(3, 5, 5, 0), (7, 9, 9, 0)])
    device.launches.clear()
    head.forward_batch(
        tokens, positions=positions, shared_kv=shared, recurrent_states=primed
    )
    pre = [call for call in device.named("linear") if call["slot"] == "nextn_pre_projection"][0]
    packed = np.frombuffer(
        device.read(pre["x"], 2 * 2 * _BACKBONE_WIDTH * 2), dtype=np.uint16
    ).reshape(2, 2 * _BACKBONE_WIDTH)
    assert packed[:, _BACKBONE_WIDTH:].tolist() == primed.tolist()


# --- the single-row path is unchanged ---------------------------------


def test_forward_still_runs_one_row_at_a_time(head, device) -> None:
    _tokens, _positions, shared = _requests(head, device, [(3, 5, 5, 0)])
    logits, h_next = head.forward(3, position=5, shared_kv=shared[0])

    for call in device.named("linear"):
        assert call["rows"] == 1, f"{call['slot']} ran at row width {call['rows']}"
    assert device.only("embedding")["rows"] == 1
    assert {call["tokens"] for call in device.named("rotary")} == {1}
    assert {call["rows"] for call in device.named("head_rmsnorm")} == {_HEAD_HEADS}
    attention = device.named("attention")
    assert len(attention) == head.block_count
    assert {call["tokens"] for call in attention} == {1}
    assert device.named("gelu_fused") and not device.named("gelu_split")

    assert logits.shape == (head.logits_width,)
    assert h_next.shape == (_BACKBONE_WIDTH,)


def test_the_head_has_one_logits_width_and_both_paths_span_all_of_it(
    head, device
) -> None:
    """A truncated vocabulary is silent: sampling then only ever emits low ids.

    ``token_embedding`` is the tied output projection, and its row-major shape is
    ``(n_vocab, n_embd)``. Reading the input width as the vocabulary -- the
    transposed reading -- produced 1024-wide logits on the real artifact and a
    public smoke that could only emit ids below 1024.
    """

    assert head.logits_width == _HEAD_VOCAB

    _tokens, _positions, shared = _requests(head, device, [(3, 5, 5, 0)])
    single, _h_next = head.forward(3, position=5, shared_kv=shared[0])
    batched, _next = head.forward_batch(
        [3], positions=[5], shared_kv=shared, recurrent_states=_states(1)
    )
    assert single.shape == (head.logits_width,)
    assert batched.shape == (1, head.logits_width)
    vocabulary = [call for call in device.named("linear") if call["slot"] == "token_embedding"]
    assert len(vocabulary) == 2, "the single-row and batched paths each ran one"
    for call in vocabulary:
        assert call["in_features"] == _HEAD_HIDDEN
        assert call["out_features"] == _HEAD_VOCAB


def test_a_transposed_output_table_is_refused_at_construction(device) -> None:
    with pytest.raises(ValueError, match="row-major"):
        _make_head(device, token_embedding_shape=(_HEAD_HIDDEN, _HEAD_VOCAB))


def test_prime_seeds_the_heads_own_recurrent_buffer(head, device) -> None:
    hidden = device.malloc(_BACKBONE_WIDTH * 2)
    device.write(hidden.ptr, _states(1, seed=2)[0].tobytes())
    head.prime(hidden)
    launch = device.only("rmsnorm")
    assert launch["rows"] == 1
    assert launch["out"] == head._recurrent.ptr
    assert launch["weight"] == int(head.backbone_output_norm.ptr)


def test_bf16_bits_survive_a_float32_round_trip() -> None:
    """``next_states`` as BF16 bits is exact, and ``h_next``'s float32 is too."""

    bits = np.array([[0x3F80, 0x0000, 0x7F7F, 0x8001]], dtype=np.uint16)
    assert _float32_to_bf16(_bf16_to_float32(bits)).tolist() == bits.tolist()
