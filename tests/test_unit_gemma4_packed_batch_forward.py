"""Physically packed multi-request forward for the Gemma 4 runner.

``Gemma4Runner.forward_batch`` runs several requests through one physical block:
the norm, dense projection, router, expert and lm-head launches cover every
packed row at once, while the rotation, the KV append and the attention launch
repeat per request against that request's own cache, position, sliding base and
keep mask.

These tests are CPU-only. Every device launch is a recorder, the allocator hands
out numpy-backed buffers, and the HIP runtime stand-in moves real bytes, so the
assertions are about *what was launched against which memory* -- the one thing
that cannot be recovered from the output of a no-op kernel:

* the projections run once over all rows, never once per request;
* a request's attention reads its own cache, its own mask and its own positions,
  and the same request attends identically alone and beside a neighbor;
* a verify block stays row-exact per request, and the packed total never turns
  on the WMMA prefill session by itself;
* each request's position, hidden rows and logits are its own;
* every refusal happens before the first device call.

GPU validation of the same interface is still required: these tests prove the
launch structure and the ownership of every pointer, not the arithmetic the
kernels perform on them.
"""

from __future__ import annotations

import ctypes
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.cpu_reference.gemma4 import (
    Gemma4AttentionGeometry,
    Gemma4RopeConfig,
    Gemma4TextConfig,
)
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as layer_module
from hipengine.runtime import gemma4 as gemma4_module
from hipengine.runtime.gemma4 import Gemma4Runner, _sliding_read_range

_HIDDEN = 16
_VOCAB = 64
_WINDOW = 8
# A quantized-shaped projection: ``gemma4_project`` sends anything that is not a
# bf16 pointer through the runtime's quantized linear dispatch, which is the
# launch a real artifact takes, so the recorder sees every dense projection with
# its row count and its WMMA flag.
_PROJECTION = SimpleNamespace(backend="hip_gfx1100")


# --- fake device ----------------------------------------------------------


class _ArrayAllocator:
    """A malloc/free pair that backs every buffer with a numpy byte array."""

    def __init__(self) -> None:
        self.next_ptr = 0x100000
        self.buffers: dict[int, np.ndarray] = {}

    def malloc(self, nbytes: int) -> DeviceBuffer:
        buffer = DeviceBuffer(ptr=self.next_ptr, nbytes=int(nbytes))
        self.buffers[self.next_ptr] = np.zeros(int(nbytes), dtype=np.uint8)
        self.next_ptr += int(nbytes) + 0x1000
        return buffer

    def free(self, buffer: DeviceBuffer) -> None:
        self.buffers.pop(int(buffer.ptr), None)

    def array_at(self, ptr: int) -> tuple[np.ndarray, int]:
        for base, array in self.buffers.items():
            if base <= ptr < base + array.nbytes:
                return array, ptr - base
        raise KeyError(f"no allocation contains {ptr:#x}")

    def bytes_at(self, ptr: int, nbytes: int) -> np.ndarray:
        array, offset = self.array_at(int(ptr))
        return array[offset : offset + int(nbytes)]

    def write(self, ptr: int, values: np.ndarray) -> None:
        raw = np.ascontiguousarray(values)
        self.bytes_at(ptr, raw.nbytes)[:] = raw.view(np.uint8).reshape(-1)


def _host_view(ptr: int, nbytes: int) -> np.ndarray:
    ctype = ctypes.c_ubyte * int(nbytes)
    return np.frombuffer(ctype.from_address(int(ptr)), dtype=np.uint8)


class _RecordingRuntime:
    """A HIP runtime stand-in whose D2D copies move real bytes."""

    def __init__(self, allocator: _ArrayAllocator) -> None:
        self.allocator = allocator
        self.copies: list[tuple[int, int, int, int, int]] = []

    def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
        self.copies.append((int(dst), int(src), int(nbytes), int(kind), int(stream)))
        target, target_offset = self.allocator.array_at(int(dst))
        source, source_offset = self.allocator.array_at(int(src))
        target[target_offset : target_offset + int(nbytes)] = source[
            source_offset : source_offset + int(nbytes)
        ]

    def stream_create(self, **kwargs) -> int:  # pragma: no cover - unused here
        return 0

    def stream_synchronize(self, stream) -> None:  # pragma: no cover - unused here
        return None

    def device_synchronize(self) -> None:  # pragma: no cover - unused here
        return None


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return None

    def reset(self) -> None:
        self.calls.clear()


# --- geometry -------------------------------------------------------------


def _geometry(*, sliding: bool) -> Gemma4AttentionGeometry:
    return Gemma4AttentionGeometry(
        layer_type="sliding_attention" if sliding else "full_attention",
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        rope=Gemma4RopeConfig(rope_theta=10_000.0, head_dim=8, rope_angles=4),
        sliding_window=_WINDOW if sliding else None,
        k_eq_v=False,
    )


def _config() -> Gemma4TextConfig:
    """Two layers: one bounded sliding layer and one global layer."""

    return Gemma4TextConfig(
        hidden_size=_HIDDEN,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_experts=4,
        top_k_experts=2,
        rms_norm_eps=1e-6,
        attention=(_geometry(sliding=True), _geometry(sliding=False)),
        vocab_size=_VOCAB,
    )


def _layer_pointers() -> SimpleNamespace:
    return SimpleNamespace(
        input_layernorm=0x100,
        q_proj=_PROJECTION,
        k_proj=_PROJECTION,
        v_proj=_PROJECTION,
        o_proj=_PROJECTION,
        q_norm=0x200,
        k_norm=0x201,
        post_attention_layernorm=0x202,
        pre_feedforward_layernorm=0x203,
        mlp_gate_up_proj=_PROJECTION,
        mlp_down_proj=_PROJECTION,
        post_feedforward_layernorm_1=0x204,
        router_scale=0x205,
        router_proj=0x206,
        router_per_expert_scale=0x207,
        pre_feedforward_layernorm_2=0x208,
        experts_gate_up_proj=_PROJECTION,
        experts_down_proj=_PROJECTION,
        post_feedforward_layernorm_2=0x209,
        post_feedforward_layernorm=0x20A,
        layer_scalar=0x20B,
        qkv_proj=0,
    )


def _weights(config: Gemma4TextConfig) -> SimpleNamespace:
    return SimpleNamespace(
        config=config,
        layers=tuple(_layer_pointers() for _ in range(len(config.attention))),
        dense_intermediate=(),
        final_norm=SimpleNamespace(buffer=SimpleNamespace(ptr=0x300)),
        embed_tokens=SimpleNamespace(),
        lm_head=None,
    )


# --- stubs ----------------------------------------------------------------


class _Stubs:
    """Every device launch and allocation a CPU forward needs, recorded."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.allocator = _ArrayAllocator()
        self.runtime = _RecordingRuntime(self.allocator)
        self.config = _config()
        self.weights = _weights(self.config)
        self.projections = _Recorder()
        self.attention = _Recorder()
        self.rotary = _Recorder()
        self.router = _Recorder()
        self.experts = _Recorder()
        self.head_linear = _Recorder()
        self.wmma_session = _Recorder()
        self.staged: dict[int, tuple[dict, dict]] = {}
        self._monkeypatch = monkeypatch

        import hipengine.core.hip as hip
        import hipengine.runtime.gguf_linear as gguf_linear

        monkeypatch.setattr(hip, "get_hip_runtime", lambda: self.runtime)
        for module in (gemma4_module, layer_module):
            monkeypatch.setattr(module, "malloc", self.allocator.malloc)
        monkeypatch.setattr(gemma4_module, "free", self.allocator.free)
        monkeypatch.setattr(layer_module, "hip_free", self.allocator.free)

        monkeypatch.setattr(gemma4_module, "copy_device_to_host", self._device_to_host)
        monkeypatch.setattr(gemma4_module, "enqueue_host_to_device", self._host_to_device)
        monkeypatch.setattr(gemma4_module, "launch_gguf_embedding", _Recorder())
        monkeypatch.setattr(gemma4_module, "gemma4_scale_bf16", _Recorder())
        monkeypatch.setattr(gemma4_module, "gemma4_rmsnorm_f32w_bf16", _Recorder())
        monkeypatch.setattr(gemma4_module, "launch_gguf_linear", self.head_linear)
        monkeypatch.setattr(gemma4_module, "gemma4_logit_softcap_f32", _Recorder())
        monkeypatch.setattr(
            gemma4_module, "_gemma4_block_wmma_session", self._wmma_session
        )

        # The quantized linear dispatch is imported inside ``gemma4_project``, so
        # it is patched where that import looks for it.
        monkeypatch.setattr(gguf_linear, "launch_gguf_linear", self.projections)

        monkeypatch.setattr(layer_module, "gemma4_partial_rotary_bf16", self.rotary)
        monkeypatch.setattr(layer_module, "gemma4_router_topk_bf16", self.router)
        monkeypatch.setattr(layer_module, "gemma4_experts_forward_bf16", self.experts)
        for name in (
            "gemma4_rmsnorm_f32w_bf16",
            "gemma4_rmsnorm_weightless_bf16",
            "gemma4_head_rmsnorm_f32w_bf16",
            "gemma4_add_rmsnorm_scale_bf16",
            "gemma4_gelu_tanh_mul_bf16",
            "gemma4_dense_combine_rmsnorm_scale_bf16",
            "gemma4_qkv_split_bf16",
        ):
            monkeypatch.setattr(layer_module, name, _Recorder())
        # Both attention routes end in the same recorder: the strict/native
        # entry for a route no profile requested, and the selected variant's
        # launcher for a requested one (which a verify block always carries).
        monkeypatch.setattr(layer_module, "gemma4_attention_prefill_bf16", self.attention)
        self.select_prefill = _Recorder()
        monkeypatch.setattr(
            layer_module, "select_prefill_attention", self._select_prefill
        )
        monkeypatch.setattr(layer_module, "_moe_stream", lambda: 0)

        original_stage = Gemma4Runner._stage_block_content

        def record_stage(runner, tokens, **kwargs):
            result = original_stage(runner, tokens, **kwargs)
            self.staged[id(runner)] = result
            return result

        monkeypatch.setattr(Gemma4Runner, "_stage_block_content", record_stage)
        monkeypatch.setattr(
            Gemma4Runner, "_q8_mmq_prefill_session", lambda self: nullcontext()
        )

    # -- copies ---------------------------------------------------------

    def _device_to_host(self, host_ptr, buffer, nbytes=None) -> None:
        count = int(buffer.nbytes if nbytes is None else nbytes)
        _host_view(host_ptr, count)[:] = self.allocator.bytes_at(int(buffer.ptr), count)

    def _host_to_device(self, buffer, host_ptr, nbytes=None, *, stream=0) -> None:
        count = int(buffer.nbytes if nbytes is None else nbytes)
        self.allocator.bytes_at(int(buffer.ptr), count)[:] = _host_view(host_ptr, count)

    def _select_prefill(self, **kwargs):
        self.select_prefill.calls.append(((), dict(kwargs)))
        return SimpleNamespace(launcher=self.attention, variant="test")

    def _wmma_session(self, enabled):
        self.wmma_session.calls.append(((bool(enabled),), {}))
        return nullcontext()

    # -- runners --------------------------------------------------------

    def runner(
        self,
        *,
        capacity: int = 64,
        max_block: int = 16,
        max_logits_rows: int = 16,
        weights=None,
    ) -> Gemma4Runner:
        return Gemma4Runner(
            weights=self.weights if weights is None else weights,
            capacity=capacity,
            max_block=max_block,
            max_logits_rows=max_logits_rows,
        )

    def workspace(
        self, *, max_block: int = 8, max_logits_rows: int = 8, weights=None
    ) -> Gemma4Runner:
        return self.runner(
            capacity=max_block,
            max_block=max_block,
            max_logits_rows=max_logits_rows,
            weights=weights,
        )

    # -- buffers --------------------------------------------------------

    def fill_logits(self, workspace: Gemma4Runner, rows: int) -> np.ndarray:
        """Write a row-addressable pattern into the workspace's logits buffer."""

        pattern = (
            np.arange(rows, dtype=np.float32)[:, None] * 1000.0
            + np.arange(_VOCAB, dtype=np.float32)[None, :]
        )
        self.allocator.write(int(workspace._logits.ptr), pattern)
        return pattern

    def fill_hidden(self, workspace: Gemma4Runner, rows: int) -> np.ndarray:
        """Write a row-addressable bf16-bit pattern into the hidden buffer."""

        bits = (
            np.arange(rows, dtype=np.uint16)[:, None] * np.uint16(100)
            + np.arange(_HIDDEN, dtype=np.uint16)[None, :]
        )
        self.allocator.write(int(workspace._hidden.ptr), bits)
        return bits


@pytest.fixture()
def stubs(monkeypatch: pytest.MonkeyPatch) -> _Stubs:
    return _Stubs(monkeypatch)


# --- expectations ---------------------------------------------------------


def _expected_attention(
    attention: Gemma4AttentionGeometry, position: int, rows: int
) -> tuple[int, int, int]:
    """``(key_begin, keys, window)`` a one-request block at ``position`` carries."""

    key_begin = _sliding_read_range(attention, position, rows)
    keys = position + rows - key_begin
    window = (
        position + rows - key_begin
        if attention.sliding_window is None
        else int(attention.sliding_window)
    )
    return key_begin, keys, window


def _expected_mask(
    attention: Gemma4AttentionGeometry, position: int, rows: int
) -> np.ndarray:
    """The keep mask one request owns, derived here rather than read back.

    Column 0 is the first key the request's block reads, so a packed block that
    leaked another request's positions into this mask would change its shape
    before it changed a bit.
    """

    key_begin, keys, _window = _expected_attention(attention, position, rows)
    queries = np.arange(position, position + rows, dtype=np.int64)[:, None]
    key_positions = np.arange(key_begin, position + rows, dtype=np.int64)[None, :]
    keep = key_positions <= queries
    if attention.sliding_window is not None:
        keep &= (queries - key_positions) < int(attention.sliding_window)
    assert keep.shape == (rows, keys)
    return keep.astype(np.uint8)


def _layer_attention_calls(
    stubs: _Stubs, layer: int, per_layer: int
) -> list[tuple[tuple, dict]]:
    """The recorded attention launches belonging to one layer, in call order."""

    return stubs.attention.calls[layer * per_layer : (layer + 1) * per_layer]


# --- packed execution -----------------------------------------------------


def test_dense_and_moe_launch_once_over_the_packed_block(stubs: _Stubs) -> None:
    """Ragged requests share one projection launch per layer; attention does not."""

    config = stubs.config
    workspace = stubs.workspace(max_block=6, max_logits_rows=6)
    requests = [stubs.runner() for _ in range(3)]
    token_rows = [[1, 2], [3], [4, 5, 6]]
    total = 6
    pattern = stubs.fill_logits(workspace, total)

    logits = workspace.forward_batch(requests, token_rows, verification=False)

    # Every dense projection of every layer ran once, over all six rows. A loop
    # of one-request forwards would have produced row counts 2, 1 and 3 instead.
    assert {args[3] for args, _kwargs in stubs.projections.calls} == {total}
    assert len(stubs.projections.calls) == 6 * len(config.attention)

    # The router and the experts are block-wide too.
    assert [kwargs["tokens"] for _args, kwargs in stubs.router.calls] == [
        total
    ] * len(config.attention)
    assert [kwargs["rows"] for _args, kwargs in stubs.experts.calls] == [
        total
    ] * len(config.attention)

    # Rotation and attention are per request, with that request's own row count.
    assert [args[6] for args, _kwargs in stubs.rotary.calls] == [2, 1, 3] * len(
        config.attention
    )
    for layer, attention in enumerate(config.attention):
        calls = _layer_attention_calls(stubs, layer, len(requests))
        assert [kwargs["tokens"] for _args, kwargs in calls] == [2, 1, 3]
        kv_width = attention.num_kv_heads * attention.head_dim
        for (args, kwargs), rows, runner in zip(calls, (2, 1, 3), requests):
            key_begin, keys, _window = _expected_attention(attention, 0, rows)
            entry = runner._kv[layer]
            assert kwargs["keys"] == keys
            # The keys are read out of *this* request's cache, from its own
            # first live key -- not from the packed block's first row.
            assert args[1] == entry.key_cache + key_begin * kv_width * 2
            assert args[2] == entry.value_cache + key_begin * kv_width * 2
            # And the mask is this request's own, sized to its own positions.
            mask_ptr = stubs.staged[id(runner)][1][attention.sliding_window].ptr
            assert args[3] == mask_ptr
            mask = stubs.allocator.bytes_at(mask_ptr, rows * keys)
            np.testing.assert_array_equal(
                mask.reshape(rows, keys), _expected_mask(attention, 0, rows)
            )

    # Logits come back per request, sliced from its own rows of the packed block.
    assert [array.shape for array in logits] == [(2, _VOCAB), (1, _VOCAB), (3, _VOCAB)]
    offset = 0
    for array, rows in zip(logits, token_rows):
        np.testing.assert_array_equal(array, pattern[offset : offset + len(rows)])
        offset += len(rows)


def test_each_request_advances_its_own_position_and_keeps_its_own_hidden_rows(
    stubs: _Stubs,
) -> None:
    """Position, ``hidden_state`` and the KV write offset are per request."""

    workspace = stubs.workspace(max_block=6, max_logits_rows=6)
    requests = [stubs.runner() for _ in range(3)]
    # Different starting positions, so a shared position would be visible.
    requests[0]._position = 0
    requests[1]._position = 5
    requests[2]._position = 2
    token_rows = [[1, 2], [3], [4, 5, 6]]
    stubs.fill_logits(workspace, 6)
    hidden = stubs.fill_hidden(workspace, 6)

    workspace.forward_batch(requests, token_rows, verification=True)

    assert [runner.position for runner in requests] == [2, 6, 5]
    offset = 0
    for runner, rows in zip(requests, token_rows):
        count = len(rows)
        assert runner._last_rows == count
        # This request's verify rows are in its own buffer, so the assistant
        # head's ``hidden_state(row)`` reads that request's rows.
        kept = stubs.allocator.bytes_at(runner._hidden.ptr, count * _HIDDEN * 2)
        np.testing.assert_array_equal(
            kept.view(np.uint16).reshape(count, _HIDDEN),
            hidden[offset : offset + count],
        )
        # Its cache write offset tracks its own advanced position.
        for entry in runner._kv:
            assert entry.write_offset == runner.position
        offset += count
    assert [runner.normalized_hidden_rows for runner in requests] == [0, 0, 0]


def test_verification_keeps_every_request_row_exact(stubs: _Stubs) -> None:
    """A verify block repeats each request's own one-token decode step."""

    config = stubs.config
    workspace = stubs.workspace(max_block=4, max_logits_rows=4)
    requests = [stubs.runner() for _ in range(2)]
    requests[1]._position = 3
    token_rows = [[1], [2, 3, 4]]
    stubs.fill_logits(workspace, 4)

    workspace.forward_batch(requests, token_rows, verification=True)

    # The dense projections keep the row-exact schedule: no WMMA prefill.
    assert {kwargs["use_wmma_prefill"] for _args, kwargs in stubs.projections.calls} == {
        False
    }
    # The WMMA session is off even though the packed block fills the workspace.
    assert [args[0] for args, _kwargs in stubs.wmma_session.calls] == [False]
    # The router runs one row at a time, so a row's route cannot depend on its
    # neighbors -- four rows over two layers.
    assert [kwargs["tokens"] for _args, kwargs in stubs.router.calls] == [1] * 8

    # A packed block of one-row requests is the case that has no per-row decode
    # ranges to notice: four single-row verify blocks must still route one row
    # at a time, or the packed block would route four rows through the
    # multi-row chain instead of four one-token launches.
    stubs.router.reset()
    singles = [stubs.runner() for _ in range(2)]
    singles_workspace = stubs.workspace(max_block=2, max_logits_rows=2)
    stubs.fill_logits(singles_workspace, 2)
    singles_workspace.forward_batch(singles, [[1], [2]], verification=True)
    assert [kwargs["tokens"] for _args, kwargs in stubs.router.calls] == [1] * 4

    # Attention: the one-row request is one launch; the three-row request is
    # three launches, each over that row's own decode range.
    for layer, attention in enumerate(config.attention):
        calls = _layer_attention_calls(stubs, layer, 4)
        assert [kwargs["tokens"] for _args, kwargs in calls] == [1, 1, 1, 1]
        # Request 0 sits at position 0 (one key); request 1 at positions 3..5,
        # so its rows read 4, 5 and 6 keys.
        assert [kwargs["keys"] for _args, kwargs in calls] == [1, 4, 5, 6]
        # Each row of the multi-row request starts its mask at its own first
        # live key inside that request's staged mask.
        mask_ptr = stubs.staged[id(requests[1])][1][attention.sliding_window].ptr
        key_begin, _keys, _window = _expected_attention(attention, 3, 1)
        assert key_begin == 0
        block_keys = 3 + 3 - key_begin
        for row, (args, _kwargs) in enumerate(calls[1:]):
            assert args[3] == mask_ptr + row * block_keys


def test_production_packing_never_promotes_the_wmma_session_by_itself(
    stubs: _Stubs,
) -> None:
    """The WMMA session follows the requests' own decisions, not the packed total."""

    workspace = stubs.workspace(max_block=6, max_logits_rows=6)
    requests = [stubs.runner() for _ in range(3)]
    stubs.fill_logits(workspace, 6)

    workspace.forward_batch(requests, [[1, 2], [3], [4, 5, 6]], verification=False)

    # The packed block fills the workspace's max_block, but no request fills its
    # own block, so none of them would have enabled WMMA prefill alone.
    assert [args[0] for args, _kwargs in stubs.wmma_session.calls] == [False]

    # When every request does fill its own block, the packed block enables it,
    # which is exactly the schedule each of them would have run alone.
    stubs.wmma_session.reset()
    full = [
        stubs.runner(max_block=2, max_logits_rows=2) for _ in range(3)
    ]
    workspace = stubs.workspace(max_block=6, max_logits_rows=6)
    stubs.fill_logits(workspace, 6)
    workspace.forward_batch(full, [[1, 2], [3, 4], [5, 6]], verification=False)
    assert [args[0] for args, _kwargs in stubs.wmma_session.calls] == [True]


def test_a_request_attends_identically_alone_and_beside_a_neighbor(
    stubs: _Stubs,
) -> None:
    """Isolation: a neighbor cannot change what an observed request attends.

    The observed three-row request is run alone, then in slot 1 of a packed
    block behind a two-row neighbor. Every attention launch it produces is
    compared by cache offset, key count, window and mask bytes rather than by
    pointer identity, so a shared workspace or a packed mask would show up.
    """

    def observe(token_rows: list[list[int]], observed: int) -> list[tuple]:
        requests = [stubs.runner() for _ in token_rows]
        total = sum(len(rows) for rows in token_rows)
        workspace = stubs.workspace(max_block=8, max_logits_rows=8)
        stubs.fill_logits(workspace, total)
        stubs.attention.reset()
        workspace.forward_batch(requests, token_rows, verification=True)
        runner = requests[observed]
        before = sum(len(rows) for rows in token_rows[:observed])
        mine = len(token_rows[observed])
        records = []
        for layer, attention in enumerate(stubs.config.attention):
            calls = _layer_attention_calls(stubs, layer, total)
            entry = runner._kv[layer]
            for args, kwargs in calls[before : before + mine]:
                records.append(
                    (
                        layer,
                        kwargs["tokens"],
                        kwargs["keys"],
                        kwargs.get("window"),
                        args[1] - entry.key_cache,
                        args[2] - entry.value_cache,
                        bytes(stubs.allocator.bytes_at(args[3], kwargs["keys"])),
                    )
                )
            del attention
        return records

    alone = observe([[1, 2, 3]], observed=0)
    packed = observe([[7, 8], [1, 2, 3], [9]], observed=1)
    assert alone and alone == packed


def test_single_request_forward_is_unchanged(stubs: _Stubs) -> None:
    """``forward`` keeps its own path: one block, no per-request segmentation."""

    config = stubs.config
    runner = stubs.runner()
    stubs.fill_logits(runner, 3)

    logits = runner.forward([1, 2, 3], logits_rows=3, verification=True)

    assert logits.shape == (3, _VOCAB)
    assert runner.position == 3
    # One rotation and one attention launch per layer, over the whole block.
    assert [args[6] for args, _kwargs in stubs.rotary.calls] == [3] * len(
        config.attention
    )
    # A verify block attends one row per launch, exactly as before: three rows
    # over two layers, never one packed launch.
    assert [kwargs["tokens"] for _args, kwargs in stubs.attention.calls] == [1] * 6
    # No per-request hidden copy was enqueued: the only device copies are this
    # runner's own KV appends.
    assert all(dst != int(runner._hidden.ptr) for dst, *_rest in stubs.runtime.copies)


# --- admission ------------------------------------------------------------


def _refuses(stubs: _Stubs, call) -> str:
    """Run ``call``, require a refusal, and require that nothing was launched."""

    with pytest.raises((ValueError, NotImplementedError)) as raised:
        call()
    assert not stubs.projections.calls
    assert not stubs.attention.calls
    assert not stubs.rotary.calls
    assert not stubs.staged
    assert stubs.runtime.copies == []
    return str(raised.value)


def test_a_packed_block_wider_than_the_workspace_is_refused(stubs: _Stubs) -> None:
    workspace = stubs.workspace(max_block=3, max_logits_rows=3)
    requests = [stubs.runner(), stubs.runner()]
    message = _refuses(
        stubs,
        lambda: workspace.forward_batch(requests, [[1, 2], [3, 4]], verification=True),
    )
    assert "max_block" in message
    assert [runner.position for runner in requests] == [0, 0]


def test_a_packed_block_wider_than_the_logits_buffer_is_refused(stubs: _Stubs) -> None:
    workspace = stubs.runner(capacity=8, max_block=8, max_logits_rows=2)
    message = _refuses(
        stubs, lambda: workspace.forward_batch([stubs.runner()], [[1, 2, 3]])
    )
    assert "max_logits_rows" in message


def test_a_request_over_its_own_capacity_is_refused(stubs: _Stubs) -> None:
    workspace = stubs.workspace()
    tight = stubs.runner(capacity=2, max_block=2, max_logits_rows=2)
    tight._position = 1
    message = _refuses(stubs, lambda: workspace.forward_batch([tight], [[1, 2]]))
    assert "capacity" in message
    assert tight.position == 1


def test_an_empty_or_mismatched_request_list_is_refused(stubs: _Stubs) -> None:
    workspace = stubs.workspace()
    assert "at least one request" in _refuses(
        stubs, lambda: workspace.forward_batch([], [])
    )
    assert "one-to-one" in _refuses(
        stubs, lambda: workspace.forward_batch([stubs.runner()], [[1], [2]])
    )


def test_a_request_without_the_workspace_weights_is_refused(stubs: _Stubs) -> None:
    workspace = stubs.workspace()
    stranger = stubs.runner(weights=_weights(_config()))
    message = _refuses(stubs, lambda: workspace.forward_batch([stranger], [[1]]))
    assert "weights" in message


def test_the_workspace_may_not_be_one_of_its_requests(stubs: _Stubs) -> None:
    workspace = stubs.workspace()
    message = _refuses(stubs, lambda: workspace.forward_batch([workspace], [[1]]))
    assert "workspace" in message


def test_a_request_with_a_different_dense_schedule_is_refused(stubs: _Stubs) -> None:
    workspace = stubs.workspace()
    request = stubs.runner()
    request.verify_dense_exact = False
    message = _refuses(stubs, lambda: workspace.forward_batch([request], [[1]]))
    assert "verify_dense_exact" in message


def test_int8_kv_storage_is_refused_by_name(stubs: _Stubs) -> None:
    workspace = stubs.workspace()
    request = stubs.runner()
    # Stand in for the INT8 owner: this test is about the guard, and building a
    # real one would allocate device memory the CPU suite does not have.
    request._int8_kv = SimpleNamespace()
    with pytest.raises(NotImplementedError) as raised:
        workspace.forward_batch([request], [[1]])
    assert "INT8" in str(raised.value)
    assert not stubs.projections.calls
    assert not stubs.staged


# --- row-exact math under the shipping configuration ----------------------


def test_packed_verify_issues_the_same_launches_as_row_by_row_forwards(
    stubs: _Stubs,
) -> None:
    """The shipping configuration: row-exact dense, router and attention.

    The resident target is built with ``verify_dense_exact=True`` even under the
    production profile, and a production profile also carries prefill and
    exact-prefill variant requests. This runs one request twice -- once through
    the packed workspace beside nothing, once through its own ``forward`` -- and
    requires the attention requests and the dense schedule to be identical, so
    packing cannot promote a different arithmetic.
    """

    tokens = [1, 2, 3]

    def observe(run) -> tuple[list, list]:
        stubs.projections.reset()
        stubs.attention.reset()
        stubs.select_prefill.reset()
        run()
        return (
            [
                (kwargs["use_wmma_prefill"], args[3])
                for args, kwargs in stubs.projections.calls
            ],
            [
                (kwargs["requested_variant"], kwargs["tokens"], kwargs["keys"])
                for _args, kwargs in stubs.select_prefill.calls
            ],
        )

    def build():
        runner = stubs.runner()
        # A production profile's prefill requests, which a verify block must not
        # adopt: the strict scope replaces them.
        runner.prefill_attention_variants = ("production_prefill",)
        runner.exact_prefill_attention_variants = ("production_exact_prefill",)
        return runner

    alone = build()
    stubs.fill_logits(alone, len(tokens))
    expected = observe(
        lambda: alone.forward(tokens, logits_rows=len(tokens), verification=True)
    )

    request = build()
    workspace = stubs.workspace(max_block=len(tokens), max_logits_rows=len(tokens))
    stubs.fill_logits(workspace, len(tokens))
    packed = observe(
        lambda: workspace.forward_batch([request], [tokens], verification=True)
    )

    assert packed == expected
    # What the row-exact verify asked for is the runner's own one-token request
    # -- the same thing its decode step asks for -- and never the profile's
    # exact-prefill scope, which is a prefill scope.
    assert {requested for requested, _tokens, _keys in expected[1]} == {
        ("production_prefill",)
    }
    assert all(
        requested != ("production_exact_prefill",)
        for requested, _tokens, _keys in expected[1]
    )
    # And the dense schedule is the row-exact one for every packed projection.
    assert {wmma for wmma, _rows in expected[0]} == {False}


def test_the_packed_expert_stage_reads_route_inputs_that_a_request_does_not(
    stubs: _Stubs,
) -> None:
    """Where a packed verify can still differ from per-request verification.

    ``gemma4_experts_forward_bf16`` decides its route from ``scratch.tokens`` --
    the *declared* width of the scratch it is handed -- and several of its owners
    from the live lane count. A packed block hands it the workspace's scratch and
    every request's lanes at once, so those two inputs are not the request's own:

    * ``scratch.tokens`` is the workspace's ``max_block``, not the request
      runner's, and the route gate opens at ``lanes >= num_experts``;
    * the live lane count is ``total rows * top_k``, not ``rows_i * top_k``, and
      the WMMA owner gates open at ``16 * num_experts`` lanes.

    Both are existing properties of the expert dispatch, not of the packing. This
    pins what the two calls actually receive so the difference is visible, and
    the two numbers that bound it: on this fixture the declared widths straddle
    the gate, while on the 26B-A4B geometry (128 experts, top_k 8) the WMMA gate
    needs 256 rows -- far wider than any MTP verify block.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        _WMMA_PREFILL_MIN_LANES_PER_EXPERT,
        gemma4_moe_prefill_route_enabled,
    )

    num_experts = stubs.config.num_experts
    top_k = stubs.config.top_k_experts

    # Per-request verify: the request's own scratch and its own lanes.
    request = stubs.runner()
    request.forward([1], verification=True)
    own = stubs.experts.calls[0][1]
    assert own["scratch"] is request._scratches[0].experts
    assert own["rows"] == 1

    # Packed verify: the workspace's scratch and the block's lanes.
    stubs.experts.reset()
    workspace = stubs.workspace(max_block=2, max_logits_rows=2)
    stubs.fill_logits(workspace, 2)
    other = stubs.runner()
    workspace.forward_batch([request, other], [[1], [2]], verification=True)
    packed = stubs.experts.calls[0][1]
    assert packed["scratch"] is workspace._scratches[0].experts
    assert packed["rows"] == 2

    # The declared width each call reports to the route gate.
    declared_packed = packed["scratch"].tokens
    declared_alone = own["scratch"].tokens
    assert declared_packed != declared_alone
    assert declared_alone >= num_experts > declared_packed
    assert not gemma4_moe_prefill_route_enabled(
        lanes=declared_packed, num_experts=num_experts
    )
    assert gemma4_moe_prefill_route_enabled(
        lanes=declared_alone, num_experts=num_experts
    )

    # The live-lane WMMA gate needs this many lanes before it can open at all.
    assert _WMMA_PREFILL_MIN_LANES_PER_EXPERT * num_experts == 64
    assert packed["rows"] * top_k < 64
    assert own["rows"] * top_k < 64
    # On the 26B-A4B geometry the same gate needs 16 * 128 = 2048 lanes, i.e.
    # 256 rows at top_k 8 -- wider than any verify block this interface packs.
    assert _WMMA_PREFILL_MIN_LANES_PER_EXPERT * 128 // 8 == 256
