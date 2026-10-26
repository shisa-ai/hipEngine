"""CPU-mocked contract tests for the Qwen4Exp N-row full-model decode runner.

No GPU and no kernels: the runner is assembled over recording fakes so these
tests pin the parts a device test cannot isolate cheaply.

* **Physical-width dispatch.** One step must reach every stage once, with
  ``rows`` equal to the caller's row count and the layer order the model
  declares, including the PLE stage running before its layer's mixer.
* **Admission before mutation.** A rejected row set (unknown slot, duplicate
  slot, out-of-vocabulary token, exhausted capacity) must raise before any
  owner's ``prepare`` runs, so no host mirror or device slab moves.
* **Row-to-slot mapping.** Row ``r`` is whatever slot the caller names, in the
  caller's order; the owners receive that list unchanged.
* **Ownership.** Owner tokens are fresh per admission, a foreign or stale token
  cannot release a slot, and close is retry-safe.
* **Import checks.** A mismatched source (config, block size, index count) or an
  overlapping buffer is refused before the first copy.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen4_exp_batch_runner as runner_module
from hipengine.runtime.moe_graph import MoeGraphCache
from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner
from hipengine.runtime.qwen4_exp_batch_state import (
    STEP_CONTEXT_COLUMN,
    STEP_INDEX_POSITION_COLUMN,
    STEP_POSITION_COLUMN,
    Qwen4ExpBatchStepMetadata,
)
from tests import _qwen4_exp_batch_fixtures as fixture


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        # 1-based index of the copy/zero that raises, for fault injection.
        self.fail_memcpy_at: int | None = None
        self.fail_memset_at: int | None = None
        self.memcpy_count = 0
        self.memset_count = 0
        self.next_ptr = 0xF00000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        # Device memory, so a captured op can be proven to read what the
        # pointer holds at launch time rather than at capture time.
        self.memory: dict[int, bytearray] = {}
        # HIP stream/graph surface, enough for the real ``MoeGraphCache``.
        self.capturing = False
        self.recorded: list = []
        self.captured: dict[int, tuple] = {}
        self.execs: dict[int, int] = {}
        self.graphs = 0
        self.next_exec = 0
        self.streams = 0
        self.fail_capture = False

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(int(nbytes), 1) + 0x100
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.allocations.pop(ptr, None)
        self.freed.append(int(ptr))

    def memset(self, ptr, value, nbytes) -> None:
        self.memset_count += 1
        if self.fail_memset_at == self.memset_count:
            raise RuntimeError("injected device zero failure")
        self.calls.append(("memset", int(ptr), int(value), int(nbytes)))

    def memcpy(self, destination, source, nbytes, kind) -> None:
        self.memcpy_count += 1
        if self.fail_memcpy_at == self.memcpy_count:
            raise RuntimeError("injected device copy failure")
        self.calls.append(("memcpy", int(destination), int(source), int(nbytes)))
        if kind == HipMemcpyKind.HOST_TO_DEVICE:
            self.device_write(destination, ctypes.string_at(int(source), int(nbytes)))
        elif kind == HipMemcpyKind.DEVICE_TO_HOST:
            data = self.device_read(source, nbytes)
            ctypes.memmove(int(destination), data, int(nbytes))

    def device_synchronize(self) -> None:
        self.calls.append(("sync",))

    # -- device memory ----------------------------------------------------

    def device_write(self, ptr: int, data: bytes) -> None:
        self.memory[int(ptr)] = bytearray(data)

    def device_read(self, ptr: int, nbytes: int) -> bytes:
        stored = self.memory.get(int(ptr))
        if stored is None:
            return bytes(int(nbytes))
        return bytes(stored[: int(nbytes)]).ljust(int(nbytes), b"\0")

    def device_op(self, op) -> None:
        """Issue one device operation, recording it while a capture is open."""

        if self.capturing:
            self.recorded.append(op)
        op()

    # -- streams and graphs ----------------------------------------------

    def stream_create(self, *, nonblocking: bool = True) -> int:
        self.streams += 1
        self.calls.append(("stream_create", bool(nonblocking)))
        return 0x5000 + self.streams

    def stream_destroy(self, stream: int) -> None:
        self.calls.append(("stream_destroy", int(stream)))

    def stream_synchronize(self, stream: int) -> None:
        self.calls.append(("stream_sync", int(stream)))

    def stream_begin_capture(self, stream: int, mode: int) -> None:
        self.capturing = True
        self.recorded = []
        self.calls.append(("begin_capture", int(stream), int(mode)))

    def stream_end_capture(self, stream: int) -> int:
        self.capturing = False
        self.calls.append(("end_capture", int(stream)))
        if self.fail_capture:
            return 0
        self.graphs += 1
        self.captured[self.graphs] = tuple(self.recorded)
        return self.graphs

    def graph_instantiate(self, graph: int) -> int:
        self.next_exec += 1
        self.execs[self.next_exec] = int(graph)
        self.calls.append(("graph_instantiate", int(graph)))
        return self.next_exec

    def graph_launch(self, graph_exec: int, stream: int) -> None:
        self.calls.append(("graph_launch", int(graph_exec), int(stream)))
        for op in self.captured[self.execs[int(graph_exec)]]:
            op()

    def graph_exec_destroy(self, graph_exec: int) -> None:
        self.calls.append(("graph_exec_destroy", int(graph_exec)))

    def graph_destroy(self, graph: int) -> None:
        self.calls.append(("graph_destroy", int(graph)))


class _FakeAttention:
    def __init__(self, slots: int, blocks: int) -> None:
        self.state_slots = slots
        self.blocks_per_request = blocks
        self.block_size = fixture.BLOCK_SIZE
        self.kv_heads = fixture.KV_HEADS
        self.head_dim = fixture.HEAD_DIM
        self.max_positions = fixture.MAX_POSITIONS
        self.block_host = np.asarray(
            [
                [slot * blocks + block for block in range(blocks)]
                for slot in range(slots)
            ],
            dtype=np.int32,
        )
        row_bytes = self.kv_heads * self.head_dim * 2
        size = slots * blocks * self.block_size * row_bytes
        self.key_cache = DeviceBuffer(0x100000, size)
        self.value_cache = DeviceBuffer(0x200000, size)
        self.position_host = np.zeros(slots, dtype=np.int64)
        self.context_host = np.ones(slots, dtype=np.int64)
        self.prepared: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.published: list[tuple[tuple[int, ...], object]] = []
        self.active = None
        self.block_table = DeviceBuffer(0x300000, slots * blocks * 4)

    def validate_rows(self, slot_ids, positions):
        slots = tuple(int(slot) for slot in slot_ids)
        values = tuple(int(position) for position in positions)
        self.prepared.append((slots, values))
        return SimpleNamespace(
            slot_ids=slots,
            rows=len(slots),
            positions=np.asarray(values, dtype=np.int64),
            contexts=np.asarray([value + 1 for value in values], dtype=np.int64),
            block_table=np.asarray(
                [self.block_host[slot].tolist() for slot in slots], dtype=np.int32
            ),
            max_context_len=max(values) + 1,
        )

    def publish(self, plan, *, packed=None):
        self.published.append((plan.slot_ids, packed))
        rows = plan.rows
        self.position_host[:rows] = plan.positions
        self.context_host[:rows] = plan.contexts
        position = (
            DeviceBuffer(0x400000, rows * 8)
            if packed is None
            else packed.position
        )
        context = (
            DeviceBuffer(0x500000, rows * 8) if packed is None else packed.context
        )
        self.active = SimpleNamespace(
            slot_ids=plan.slot_ids,
            rows=rows,
            block_table=self.block_table,
            position=position,
            context=context,
            append_spans=object(),
            decode_spans=object(),
            max_context_len=plan.max_context_len,
        )
        return self.active


class _FakeIndex:
    def __init__(self, slots: int) -> None:
        self.state_slots = slots
        self.index_heads = fixture.INDEX_HEADS
        self.index_dim = fixture.INDEX_DIM
        self.capacity = fixture.MAX_POSITIONS
        self.output_stride = fixture.OUTPUT_STRIDE
        self.raw_keys = DeviceBuffer(
            0x600000, slots * self.capacity * self.index_dim * 4
        )
        self.block_size = fixture.BLOCK_SIZE
        self.cursor_host = np.zeros(slots, dtype=np.int64)
        self.pooled_count_host = np.zeros(slots, dtype=np.int64)
        self.reset_generation = np.zeros(slots, dtype=np.int64)
        self.pending_rollback = None
        self.count = 0
        self.prepared: list[tuple[int, ...]] = []
        self.published: list[tuple[tuple[int, ...], object]] = []
        self.resets: list[int] = []
        self.active = None

    def reset_slot(self, slot):
        self.resets.append(int(slot))
        self.cursor_host[slot] = 0
        self.pooled_count_host[slot] = 0
        self.reset_generation[slot] += 1
        self.active = None

    def validate_rows(self, slot_ids):
        slots = tuple(int(slot) for slot in slot_ids)
        self.prepared.append(slots)
        parsed = np.asarray(slots, dtype=np.int64)
        cursors = self.cursor_host[parsed]
        return SimpleNamespace(
            slot_ids=slots,
            rows=len(slots),
            slots=parsed,
            positions=cursors - 1,
            blocks=cursors // fixture.RATIO,
        )

    def publish(self, plan, *, packed=None):
        self.published.append((plan.slot_ids, packed))
        self.active = SimpleNamespace(
            slot_ids=plan.slot_ids,
            rows=plan.rows,
            positions_device=(
                DeviceBuffer(0x680000, plan.rows * 8)
                if packed is None
                else packed.index_position
            ),
        )
        return self.active


class _FakeGdn:
    def __init__(self, slots: int) -> None:
        self.state_slots = slots
        self.conv_bytes_per_slot = fixture.GDN_QKV_WIDTH * fixture.GDN_CONV_KERNEL * 4
        self.recurrent_bytes_per_slot = (
            fixture.GDN_V_HEADS * fixture.GDN_HEAD_DIM * fixture.GDN_HEAD_DIM * 4
        )
        self.conv_state = DeviceBuffer(0x700000, slots * self.conv_bytes_per_slot)
        self.recurrent_state = DeviceBuffer(
            0x800000, slots * self.recurrent_bytes_per_slot
        )
        self.prepared: list[tuple[int, ...]] = []

    def conv_slot_ptr(self, slot):
        return self.conv_state.ptr + slot * self.conv_bytes_per_slot

    def recurrent_slot_ptr(self, slot):
        return self.recurrent_state.ptr + slot * self.recurrent_bytes_per_slot

    def prepare(self, slot_ids):
        slots = tuple(int(slot) for slot in slot_ids)
        self.prepared.append(slots)
        return SimpleNamespace(slot_ids=slots, rows=len(slots))


class _FakePle:
    def __init__(self, slots: int) -> None:
        self.state_slots = slots
        self.history_bytes_per_slot = (
            (fixture.PLE_CONV_KERNEL - 1)
            * fixture.PLE_NGRAM
            * fixture.BRANCHES
            * fixture.HIDDEN
            * 4
        )
        self.history = DeviceBuffer(0x900000, slots * self.history_bytes_per_slot)
        self.prepared: list[tuple[int, ...]] = []

    def history_slot_ptr(self, slot):
        return self.history.ptr + slot * self.history_bytes_per_slot

    def prepare(self, slot_ids):
        slots = tuple(int(slot) for slot in slot_ids)
        self.prepared.append(slots)
        return SimpleNamespace(slot_ids=slots, rows=len(slots))


class _FakeWeight:
    def __init__(self, ptr: int, quant_key: str = "f32") -> None:
        self.spec = SimpleNamespace(quant_key=quant_key)
        self._allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=ptr))

    def allocation(self, name: str):
        assert name == "raw"
        return self._allocation


class _FakeResident:
    def __init__(self) -> None:
        self.plan = SimpleNamespace(config=fixture.config())
        self.ple_staging = object()

    def weight(self, name: str) -> _FakeWeight:
        return _FakeWeight(0x10000 + (abs(hash(name)) % 0x1000) * 16)


def _fake_binding(layer: int, kind: str) -> SimpleNamespace:
    return SimpleNamespace(
        layer_id=layer,
        layer_type=kind,
        gdn_state_index=sum(
            1 for value in fixture.LAYER_TYPES[:layer] if value == "gdn"
        ),
        qsa_state_index=sum(
            1 for value in fixture.LAYER_TYPES[:layer] if value == "qsa"
        ),
        attention_gr=object(),
        ffn_gr=object(),
        mixer=object(),
        moe=object(),
    )


def _fake_runner(*, slots: int = 3) -> Qwen4ExpGGUFBatchRunner:
    config = fixture.config()
    runner = object.__new__(Qwen4ExpGGUFBatchRunner)
    runner.resident = _FakeResident()
    runner.config = config
    runner.backend = "hip_gfx1151"
    runner.runtime = _FakeRuntime()
    runner.state_slots = slots
    runner.max_positions = fixture.MAX_POSITIONS
    runner.block_size = fixture.BLOCK_SIZE
    runner.closed = False
    runner.closing = False
    runner._owned = []
    runner._buffers = None
    runner._owners = [0] * slots
    runner._poisoned = set()
    runner._next_owner = 0
    runner._positions = np.zeros(slots, dtype=np.int64)
    runner._ple_hash_states = {}
    runner._ple_layers = frozenset(config.ple_layers)
    runner._graphs = None
    runner.gdn_bindings = {
        layer: _fake_binding(layer, kind)
        for layer, kind in enumerate(config.layer_types)
        if kind == "gdn"
    }
    runner.qsa_bindings = {
        layer: _fake_binding(layer, kind)
        for layer, kind in enumerate(config.layer_types)
        if kind == "qsa"
    }
    runner.attention_states = (_FakeAttention(slots, 2),)
    # Static layout: every slot owns its full window of the page table.
    runner.pooled = False
    runner._pool_request = None
    runner._block_host = runner.attention_states[0].block_host
    runner._free_blocks = []
    runner._slot_blocks = [runner.attention_states[0].blocks_per_request] * slots
    runner.index_states = (_FakeIndex(slots),)
    runner.step_metadata = Qwen4ExpBatchStepMetadata.allocate(
        layers=len(runner.attention_states),
        state_slots=slots,
        runtime=runner.runtime,
    )
    runner.gdn_states = (_FakeGdn(slots), _FakeGdn(slots))
    runner.ple_state = _FakePle(slots)
    runner.selection = SimpleNamespace(rows=slots, stride=fixture.OUTPUT_STRIDE)
    runner.gdn_scratch = SimpleNamespace(ptr=0xA00000, closed=False)
    runner.qsa_scratch = SimpleNamespace(ptr=0xB00000, closed=False)
    runner.ple_scratch = SimpleNamespace(ptr=0xC00000, closed=False)
    runner.head_scratch = SimpleNamespace(ptr=0xD00000, closed=False)
    runner.token_ids = DeviceBuffer(0xE00000, slots * 8)
    runner.embedding = DeviceBuffer(0xE10000, slots * fixture.HIDDEN * 2)
    runner.ple_embedding = DeviceBuffer(0xE20000, slots * fixture.HIDDEN * 4)
    runner.residual = DeviceBuffer(0xE30000, slots * fixture.RESIDUAL_WIDTH * 2)
    runner.logits = DeviceBuffer(0xE40000, slots * fixture.VOCAB * 4)
    runner.argmax_ids = DeviceBuffer(0xE50000, slots * 4)
    runner.argmax_values = DeviceBuffer(0xE60000, 64 * 4)
    runner.argmax_blocks_i32 = DeviceBuffer(0xE70000, 64 * 4)
    return runner


def _install_step_fakes(monkeypatch) -> list[tuple]:
    """Record every dispatch the step makes and its row width."""

    events: list[tuple] = []

    def record(name, rows, **extra):
        events.append((name, int(rows), extra))

    class _Kernel:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args, **kwargs):
            widths = [int(arg) for arg in args if isinstance(arg, int)]
            events.append((self.name, tuple(widths)))

    monkeypatch.setattr(
        runner_module,
        "resolve",
        lambda **kwargs: _Kernel(f"embedding:{kwargs['quant']}"),
    )
    monkeypatch.setattr(
        runner_module,
        "copy_host_array_to_device",
        lambda buffer, array, nbytes=None, runtime=None: events.append(
            (
                "h2d",
                int(buffer.ptr),
                int(nbytes if nbytes is not None else array.nbytes),
            )
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "copy_device_to_host",
        lambda host_ptr, buffer, nbytes=None, runtime=None: events.append(
            ("d2h", int(buffer.ptr))
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "qwen4_exp_repeat_bf16_branches",
        lambda *args, rows=1, runtime=None: record("repeat", rows),
    )
    monkeypatch.setattr(
        runner_module,
        "ple_hash_rows",
        lambda tokens, *, positions, sequence_ids, states, **kwargs: (
            np.zeros((len(tokens), fixture.PLE_HEAD_COUNT), dtype=np.int64),
            {
                int(seq): SimpleNamespace(tokens=(0, 1), next_position=pos + 1)
                for seq, pos in zip(sequence_ids, positions)
            },
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "stage_qwen4_exp_ple_rows",
        lambda staging, rows, *, hidden: np.zeros(
            (len(rows), hidden), dtype=np.float32
        ),
    )

    def ple_stage(
        residual_ptr, embedding_ptr, weights, *, state, scratch, rows, **kwargs
    ):
        record("ple", rows)
        return SimpleNamespace(ptr=residual_ptr + 0x10)

    def gdn_layer(residual_ptr, weights, *, state, scratch, rows, **kwargs):
        record("gdn", rows, state=id(state))
        return SimpleNamespace(ptr=residual_ptr + 0x10)

    def qsa_layer(
        residual_ptr,
        weights,
        *,
        attention_state,
        index_state,
        selection,
        scratch,
        rows,
        **kwargs,
    ):
        record("qsa", rows, index=id(index_state), selection=id(selection))
        return SimpleNamespace(ptr=residual_ptr + 0x10)

    def gr_read(residual_ptr, norm_ptr, down, up, inject, scratch, *, rows, **kwargs):
        record("gr_read", rows, scratch=id(scratch))
        return SimpleNamespace(
            mixed=DeviceBuffer(residual_ptr + 0x20, rows * fixture.HIDDEN * 4)
        )

    monkeypatch.setattr(runner_module, "run_qwen4_exp_batch_ple_stage", ple_stage)
    monkeypatch.setattr(runner_module, "run_qwen4_exp_batch_gdn_layer", gdn_layer)
    monkeypatch.setattr(runner_module, "run_qwen4_exp_batch_qsa_layer", qsa_layer)
    monkeypatch.setattr(runner_module, "run_qwen4_exp_gr_read", gr_read)
    monkeypatch.setattr(
        runner_module,
        "launch_gguf_linear",
        lambda weight, x_ptr, out_ptr, rows, in_features, out_features, **kwargs: record(
            "lm_head", rows, out_features=int(out_features)
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "argmax_f32_rows_i32",
        lambda logits_ptr, values_ptr, blocks_ptr, ids_ptr, out_values, rows, vocab, **kwargs: record(
            "argmax", rows, vocab=int(vocab)
        ),
    )
    return events


def _use_two_qsa_layers(runner: Qwen4ExpGGUFBatchRunner) -> None:
    """Give ``runner`` two QSA layers, each with its own owners and views."""

    two_qsa = fixture.two_qsa_config()
    runner.config = two_qsa
    runner._ple_layers = frozenset(two_qsa.ple_layers)
    runner.gdn_bindings = {
        layer: _fake_binding(layer, kind)
        for layer, kind in enumerate(two_qsa.layer_types)
        if kind == "gdn"
    }
    runner.qsa_bindings = {
        layer: _fake_binding(layer, kind)
        for layer, kind in enumerate(two_qsa.layer_types)
        if kind == "qsa"
    }
    runner.attention_states = (
        _FakeAttention(runner.state_slots, 2),
        _FakeAttention(runner.state_slots, 2),
    )
    runner.index_states = (_FakeIndex(runner.state_slots), _FakeIndex(runner.state_slots))
    runner.step_metadata = Qwen4ExpBatchStepMetadata.allocate(
        layers=2, state_slots=runner.state_slots, runtime=runner.runtime
    )


def _packed_uploads(
    runner: Qwen4ExpGGUFBatchRunner, metadata: Qwen4ExpBatchStepMetadata
) -> list[int]:
    """Byte counts of every device copy that targeted the packed buffer.

    Read from the runtime rather than the runner module's upload helper,
    because the packed owner lives in its own module and reaches the device
    through the same ``runtime.memcpy`` the real path uses.
    """

    return [
        call[3]
        for call in runner.runtime.calls
        if call[0] == "memcpy" and call[1] == metadata.buffer.ptr
    ]


def _packed_payload(runner: Qwen4ExpGGUFBatchRunner) -> np.ndarray:
    """The packed buffer's host staging as ``[layer, column, slot]`` int64."""

    metadata = runner.step_metadata
    assert metadata is not None
    return metadata.host


def test_step_dispatches_every_stage_at_physical_width(monkeypatch) -> None:
    events = _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=3)
    owner_two = runner.admit(2)
    owner_zero = runner.admit(0)
    result = runner.step(
        [7, 9], slots=[2, 0], owners=[owner_two, owner_zero], capture_logits=False
    )
    names = [event[0] for event in events]
    assert names.count("ple") == 1
    assert names.count("gdn") == 2
    assert names.count("qsa") == 1
    layer_events = [event for event in events if event[0] in {"ple", "gdn", "qsa"}]
    assert [event[0] for event in layer_events] == ["ple", "gdn", "qsa", "gdn"]
    for name, rows, *_ in layer_events:
        assert rows == 2, f"{name} dispatched {rows} rows for a 2-row step"
    for name in ("repeat", "lm_head", "argmax"):
        widths = [event[1] for event in events if event[0] == name]
        assert widths == [2], f"{name} dispatched widths {widths}"
    assert result.rows == 2
    assert result.slot_ids == (2, 0)
    assert result.positions.tolist() == [0, 0]
    assert result.logits is None
    assert runner.position(2) == 1 and runner.position(0) == 1
    # The owners were admitted with the caller's slot order, unchanged.
    attention = runner.attention_states[0]
    assert attention.prepared == [((2, 0), (0, 0))]
    assert runner.index_states[0].prepared == [(2, 0)]
    assert [state.prepared for state in runner.gdn_states] == [[(2, 0)], [(2, 0)]]
    assert runner.ple_state.prepared == [(2, 0)]
    # The QSA layer received the index owner and the shared selection scratch.
    qsa_event = next(event for event in events if event[0] == "qsa")
    assert qsa_event[2]["index"] == id(runner.index_states[0])
    assert qsa_event[2]["selection"] == id(runner.selection)


def test_step_keeps_the_persistent_state_of_unstepped_slots(monkeypatch) -> None:
    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=3)
    owner = runner.admit(1)
    runner.step([3], slots=[1], owners=[owner], capture_logits=False)
    assert runner.index_states[0].cursor_host.tolist() == [0, 0, 0]
    assert runner.position(1) == 1
    assert runner.free_slots == (0, 2)


def test_step_rejects_a_bad_row_set_before_any_prepare(monkeypatch) -> None:
    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=3)
    owner = runner.admit(2)
    attention = runner.attention_states[0]
    index_state = runner.index_states[0]
    gdn_state = runner.gdn_states[0]
    ple_state = runner.ple_state
    with pytest.raises(RuntimeError, match="not admitted"):
        runner.step([1], slots=[0], owners=[owner])
    with pytest.raises(ValueError, match="outside"):
        runner.step([1], slots=[3], owners=[owner])
    with pytest.raises(ValueError, match="equal length"):
        runner.step([1, 2], slots=[2], owners=[owner])
    with pytest.raises(ValueError, match="appears twice"):
        runner.step([1, 2], slots=[2, 2], owners=[owner, owner])
    with pytest.raises(ValueError, match="vocabulary"):
        runner.step([fixture.VOCAB], slots=[2], owners=[owner])
    with pytest.raises(ValueError, match="at least one token"):
        runner.step([], slots=[])
    assert attention.prepared == []
    assert index_state.prepared == []
    assert gdn_state.prepared == []
    assert ple_state.prepared == []
    assert runner.position(2) == 0
    # Capacity is checked against the per-slot position mirror before dispatch.
    runner._positions[2] = fixture.MAX_POSITIONS
    with pytest.raises(ValueError, match="capacity"):
        runner.step([1], slots=[2], owners=[owner])
    assert attention.prepared == []
    assert runner.position(2) == fixture.MAX_POSITIONS


def test_admission_ownership_and_release(monkeypatch) -> None:
    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=2)
    first = runner.admit(0)
    second = runner.admit(1)
    assert first != second
    assert runner.active_slots == (0, 1)
    with pytest.raises(RuntimeError, match="already owned"):
        runner.admit(0)
    with pytest.raises(RuntimeError, match="owned by request"):
        runner.release(0, second)
    with pytest.raises(ValueError, match="owner token must be an integer"):
        runner.release(0, "one")
    runner.release(0, first)
    assert runner.free_slots == (0,)
    assert runner.owner_of(0) == 0
    with pytest.raises(RuntimeError, match="not admitted"):
        runner.release(0, first)
    third = runner.admit(0)
    assert third not in (first, second)
    # reset_slot keeps the owner token but rewinds the position and history.
    runner._positions[0] = 4
    runner._ple_hash_states[0] = SimpleNamespace(tokens=(1, 2), next_position=4)
    runner.reset_slot(0, third)
    assert runner.owner_of(0) == third
    assert runner.position(0) == 0
    assert 0 not in runner.ple_hash_states


def test_admission_zeroes_the_whole_slot(monkeypatch) -> None:
    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=2)
    runner.index_states[0].cursor_host[1] = 5
    runner.index_states[0].pooled_count_host[1] = 2
    runner.runtime.calls.clear()
    runner.admit(1)
    memsets = [call for call in runner.runtime.calls if call[0] == "memset"]
    assert memsets, "admission must zero the slot's persistent state"
    total = sum(call[3] for call in memsets)
    attention = runner.attention_states[0]
    block_bytes = attention.block_size * attention.kv_heads * attention.head_dim * 2
    expected = 2 * attention.blocks_per_request * block_bytes
    expected += (
        attention.blocks_per_request * attention.block_size * fixture.INDEX_DIM * 4
    )
    expected += len(runner.gdn_states) * (
        runner.gdn_states[0].conv_bytes_per_slot
        + runner.gdn_states[0].recurrent_bytes_per_slot
    )
    expected += runner.ple_state.history_bytes_per_slot
    assert total == expected
    assert runner.index_states[0].cursor_host[1] == 0
    assert runner.index_states[0].pooled_count_host[1] == 0


def test_close_is_retry_safe(monkeypatch) -> None:
    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=2)
    closed: list[str] = []

    class _Owner:
        def __init__(self, name, *, fail=False):
            self.name = name
            self.fail = fail

        def close(self):
            if self.fail:
                self.fail = False
                raise RuntimeError(f"{self.name} free failed")
            closed.append(self.name)

    runner._owned = [
        _Owner("attention"),
        _Owner("index", fail=True),
        _Owner("gdn"),
    ]
    with pytest.raises(RuntimeError, match="free failed"):
        runner.close()
    assert closed == ["gdn"]
    assert [owner.name for owner in runner._owned] == ["attention", "index"]
    assert runner.closing
    with pytest.raises(RuntimeError, match="closed"):
        runner.step([1], slots=[0])
    runner.close()
    assert closed == ["gdn", "index", "attention"]
    assert runner.closed


def test_import_rejects_a_mismatched_source_before_copying(monkeypatch) -> None:
    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=2)
    owner = runner.admit(0)
    runner.runtime.calls.clear()

    def source(**overrides):
        config = fixture.config()
        for name, value in overrides.pop("config", {}).items():
            config = type(config)(**{**config.__dict__, name: value})
        attention = _FakeAttention(1, 2)
        index_state = _FakeIndex(1)
        index_state.count = overrides.pop("index_count", 4)
        index_state.capacity = 300
        if overrides.pop("pending_rollback", False):
            index_state.pending_rollback = object()
        decode = SimpleNamespace(
            gdn_conv=DeviceBuffer(
                0x5000000,
                len(runner.gdn_states) * runner.gdn_states[0].conv_bytes_per_slot,
            ),
            gdn_matrix=DeviceBuffer(
                0x6000000,
                len(runner.gdn_states) * runner.gdn_states[0].recurrent_bytes_per_slot,
            ),
            ple_conv=DeviceBuffer(0x7000000, runner.ple_state.history_bytes_per_slot),
        )
        return SimpleNamespace(
            closed=False,
            config=config,
            position=overrides.pop("position", 4),
            attention_states=(attention,),
            index_states=(index_state,),
            state=decode,
            ple_hash_states={0: SimpleNamespace(tokens=(1, 2), next_position=4)},
        )

    with pytest.raises(ValueError, match="differs from this runner"):
        runner.import_prefilled(0, owner, source=source(config={"hidden_size": 128}))
    mismatched = source()
    mismatched.attention_states[0].block_size = 128
    with pytest.raises(ValueError, match="block size"):
        runner.import_prefilled(0, owner, source=mismatched)
    with pytest.raises(ValueError, match="index count"):
        runner.import_prefilled(0, owner, source=source(index_count=3))
    with pytest.raises(ValueError, match="pending index rollback"):
        runner.import_prefilled(0, owner, source=source(pending_rollback=True))
    with pytest.raises(ValueError, match="exceeds max_positions"):
        runner.import_prefilled(
            0, owner, source=source(position=fixture.MAX_POSITIONS + 1)
        )
    assert runner.runtime.calls == [], "a rejected import must not write"
    assert runner.position(0) == 0
    # The nonalias guard rejects a source that shares the destination slab.
    aliased = source()
    aliased.attention_states[0].key_cache = runner.attention_states[0].key_cache
    with pytest.raises(ValueError, match="overlap"):
        runner.import_prefilled(0, owner, source=aliased)
    assert runner.position(0) == 0


def test_admission_resets_the_index_owner_generation(monkeypatch) -> None:
    """Slot reuse goes through the index owner's reset, not raw cursor writes.

    A generation bump is what invalidates a rollback token captured by the
    previous request; writing ``cursor_host`` directly would leave that token
    applicable to the new request's keys.
    """

    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=2)
    index_state = runner.index_states[0]
    owner = runner.admit(1)
    assert index_state.resets == [1]
    assert int(index_state.reset_generation[1]) == 1
    runner._positions[1] = 3
    runner.reset_slot(1, owner)
    assert index_state.resets == [1, 1]
    assert int(index_state.reset_generation[1]) == 2
    runner.release(1, owner)
    assert index_state.resets == [1, 1, 1]
    assert int(index_state.reset_generation[1]) == 3
    assert int(index_state.cursor_host[1]) == 0
    assert int(index_state.pooled_count_host[1]) == 0


def test_import_copies_each_block_prefix_from_its_own_physical_block(
    monkeypatch,
) -> None:
    """The import is a per-block copy, not a scatter through one block table.

    A c1 source addresses its raw index keys by *its* physical block ids, so a
    contiguous scatter would only be right when that table happens to be the
    identity. This source uses blocks ``[7, 9]`` against slot blocks ``[2, 3]``
    and a 300-position prefix, so the second block must copy 44 rows, not 256.
    """

    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=2)
    owner = runner.admit(0)
    attention = runner.attention_states[0]
    attention.block_host[0] = np.asarray([2, 3], dtype=np.int32)
    source_attention = _FakeAttention(1, 2)
    source_attention.block_host[0] = np.asarray([7, 9], dtype=np.int32)
    source_attention.key_cache = DeviceBuffer(
        0x1100000, source_attention.key_cache.nbytes
    )
    source_attention.value_cache = DeviceBuffer(
        0x1200000, source_attention.value_cache.nbytes
    )
    source_index = _FakeIndex(1)
    source_index.raw_keys = DeviceBuffer(0x1300000, source_index.raw_keys.nbytes)
    source_index.count = 300
    source_index.capacity = fixture.MAX_POSITIONS
    source = SimpleNamespace(
        closed=False,
        config=fixture.config(),
        position=300,
        attention_states=(source_attention,),
        index_states=(source_index,),
        state=SimpleNamespace(
            gdn_conv=DeviceBuffer(
                0x5000000,
                len(runner.gdn_states) * runner.gdn_states[0].conv_bytes_per_slot,
            ),
            gdn_matrix=DeviceBuffer(
                0x6000000,
                len(runner.gdn_states) * runner.gdn_states[0].recurrent_bytes_per_slot,
            ),
            ple_conv=DeviceBuffer(0x7000000, runner.ple_state.history_bytes_per_slot),
        ),
        ple_hash_states={0: SimpleNamespace(tokens=(1, 2), next_position=300)},
    )
    runner.runtime.calls.clear()
    assert runner.import_prefilled(0, owner, source=source) == 300

    row_bytes = attention.kv_heads * attention.head_dim * 2
    block_bytes = attention.block_size * row_bytes
    index_row_bytes = fixture.INDEX_DIM * 4
    index_block_bytes = attention.block_size * index_row_bytes
    copies = [call for call in runner.runtime.calls if call[0] == "memcpy"]
    # Per block: key, value, then this layer's index keys. The GDN and PLE
    # copies follow and are not part of this assertion.
    expected: list[tuple] = []
    for source_block, destination_block, rows in ((7, 2, 256), (9, 3, 44)):
        for source_cache, destination_cache in (
            (source_attention.key_cache.ptr, attention.key_cache.ptr),
            (source_attention.value_cache.ptr, attention.value_cache.ptr),
        ):
            expected.append(
                (
                    "memcpy",
                    destination_cache + destination_block * block_bytes,
                    source_cache + source_block * block_bytes,
                    rows * row_bytes,
                )
            )
        expected.append(
            (
                "memcpy",
                runner.index_states[0].raw_keys.ptr
                + destination_block * index_block_bytes,
                source_index.raw_keys.ptr + source_block * index_block_bytes,
                rows * index_row_bytes,
            )
        )
    assert copies[:6] == expected
    assert len(copies) == 6 + 2 * len(runner.gdn_states) + 1
    assert int(runner.index_states[0].cursor_host[0]) == 300
    assert int(runner.index_states[0].pooled_count_host[0]) == 0
    assert runner.position(0) == 300
    assert runner.ple_hash_states[0].next_position == 300


def test_step_rejects_non_integral_slots_tokens_and_owners(monkeypatch) -> None:
    """A float or bool slot/token/owner is refused, never truncated.

    ``int()`` and ``np.asarray(dtype=np.int64)`` both accept ``3.7`` as ``3`` and
    ``True`` as ``1``, which would silently advance the wrong slot or embed the
    wrong token.
    """

    events = _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=3)
    owner = runner.admit(2)
    attention = runner.attention_states[0]
    for call in (
        dict(token_ids=[1], slots=[2.0], owners=[owner]),
        dict(token_ids=[1], slots=[True], owners=[owner]),
        dict(token_ids=[1], slots=[np.float32(2)], owners=[owner]),
        dict(token_ids=[1], slots=[2], owners=[1.5]),
        dict(token_ids=[1], slots=[2], owners=[True]),
        dict(token_ids=[2.5], slots=[2], owners=[owner]),
        dict(token_ids=[True], slots=[2], owners=[owner]),
        dict(token_ids=np.asarray([[1, 2]]), slots=[2], owners=[owner]),
        dict(token_ids=3, slots=[2], owners=[owner]),
    ):
        with pytest.raises(ValueError, match="must be an integer|must be a sequence"):
            runner.step(**call)
    assert events == [], "a rejected call must dispatch nothing"
    assert attention.prepared == []
    assert runner.poisoned_slots == ()
    assert runner.position(2) == 0


def test_step_requires_the_owner_token_of_every_named_slot(monkeypatch) -> None:
    """A named slot without a matching owner token cannot be advanced.

    Without this a scheduler holding a handle to a released request would step
    whichever request was re-admitted into the same physical slot.
    """

    events = _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=3)
    stale = runner.admit(2)
    runner.release(2, stale)
    fresh = runner.admit(2)
    assert fresh != stale
    with pytest.raises(ValueError, match="requires the owner token"):
        runner.step([1], slots=[2])
    with pytest.raises(RuntimeError, match="owned by request"):
        runner.step([1], slots=[2], owners=[stale])
    with pytest.raises(ValueError, match="only meaningful with explicit slots"):
        runner.step([1], owners=[fresh])
    assert events == [], "a rejected call must dispatch nothing"
    assert runner.position(2) == 0
    assert runner.poisoned_slots == ()
    # The live token still works.
    runner.step([1], slots=[2], owners=[fresh], capture_logits=False)
    assert runner.position(2) == 1


def test_failed_step_poisons_only_its_rows(monkeypatch) -> None:
    """A failure after mutation started refuses a retry until an explicit reset."""

    events = _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=3)
    owner_zero = runner.admit(0)
    owner_two = runner.admit(2)

    def failing_qsa_layer(*args, **kwargs):
        raise RuntimeError("injected failure after the index append")

    monkeypatch.setattr(runner_module, "run_qwen4_exp_batch_qsa_layer", failing_qsa_layer)
    with pytest.raises(RuntimeError, match="injected failure"):
        runner.step([1], slots=[2], owners=[owner_two])
    assert runner.poisoned_slots == (2,)
    assert runner.is_poisoned(2)
    assert not runner.is_poisoned(0)
    # The failed step must not have advanced the position mirror.
    assert runner.position(2) == 0
    # A retry refuses instead of appending on top of the half-applied step.
    with pytest.raises(RuntimeError, match="poisoned"):
        runner.step([1], slots=[2], owners=[owner_two])
    with pytest.raises(RuntimeError, match="poisoned"):
        runner.step([1, 1], slots=[2, 0], owners=[owner_two, owner_zero])
    # The default row set (every admitted slot) refuses too.
    with pytest.raises(RuntimeError, match="poisoned"):
        runner.step([1, 1])
    # A slot that was not a row of the failed call is still usable.
    monkeypatch.setattr(
        runner_module,
        "run_qwen4_exp_batch_qsa_layer",
        lambda residual_ptr, weights, *, rows, **kwargs: SimpleNamespace(
            ptr=residual_ptr + 0x10
        ),
    )
    healthy = runner.step([1], slots=[0], owners=[owner_zero], capture_logits=False)
    assert healthy.slot_ids == (0,)
    assert runner.position(0) == 1
    assert runner.poisoned_slots == (2,)
    # reset_slot is the in-place recovery path.
    runner.reset_slot(2, owner_two)
    assert runner.poisoned_slots == ()
    runner.step([1], slots=[2], owners=[owner_two], capture_logits=False)
    assert runner.position(2) == 1
    # release + admit is the other one.
    monkeypatch.setattr(runner_module, "run_qwen4_exp_batch_qsa_layer", failing_qsa_layer)
    with pytest.raises(RuntimeError, match="injected failure"):
        runner.step([1], slots=[2], owners=[owner_two])
    assert runner.poisoned_slots == (2,)
    # A reset whose zeroing fails must leave the slot refusing work.
    runner.runtime.fail_memset_at = runner.runtime.memset_count + 1
    with pytest.raises(RuntimeError, match="injected device zero failure"):
        runner.reset_slot(2, owner_two)
    assert runner.poisoned_slots == (2,)
    with pytest.raises(RuntimeError, match="poisoned"):
        runner.step([1], slots=[2], owners=[owner_two])
    runner.runtime.fail_memset_at = None
    runner.reset_slot(2, owner_two)
    assert runner.poisoned_slots == ()
    monkeypatch.setattr(runner_module, "run_qwen4_exp_batch_qsa_layer", failing_qsa_layer)
    with pytest.raises(RuntimeError, match="injected failure"):
        runner.step([1], slots=[2], owners=[owner_two])
    assert runner.poisoned_slots == (2,)
    runner.release(2, owner_two)
    assert runner.poisoned_slots == ()
    again = runner.admit(2)
    assert runner.poisoned_slots == ()
    assert runner.position(2) == 0
    assert again not in (owner_two, owner_zero)
    assert len(events) > 0


def test_preflight_rejection_never_poisons_a_slot(monkeypatch) -> None:
    """Only a call that started mutating poisons; validation failures do not."""

    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=3)
    owner = runner.admit(2)
    runner._positions[2] = fixture.MAX_POSITIONS
    for call in (
        dict(token_ids=[1], slots=[2], owners=[owner]),
        dict(token_ids=[fixture.VOCAB], slots=[2], owners=[owner]),
        dict(token_ids=[1], slots=[9], owners=[owner]),
        dict(token_ids=[1], slots=[2], owners=[owner + 7]),
        dict(token_ids=[1.5], slots=[2], owners=[owner]),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            runner.step(**call)
        assert runner.poisoned_slots == (), f"{call} poisoned a healthy slot"
    assert runner.position(2) == fixture.MAX_POSITIONS


def test_failed_import_poisons_the_target_slot_only(monkeypatch) -> None:
    """A copy that dies part-way refuses a retry until the slot is reset."""

    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=2)
    owner_one = runner.admit(1)
    owner_zero = runner.admit(0)
    source_attention = _FakeAttention(1, 2)
    source_attention.key_cache = DeviceBuffer(
        0x1100000, source_attention.key_cache.nbytes
    )
    source_attention.value_cache = DeviceBuffer(
        0x1200000, source_attention.value_cache.nbytes
    )
    source_index = _FakeIndex(1)
    source_index.raw_keys = DeviceBuffer(0x1300000, source_index.raw_keys.nbytes)
    source_index.count = 300
    source_index.capacity = fixture.MAX_POSITIONS
    source = SimpleNamespace(
        closed=False,
        config=fixture.config(),
        position=300,
        attention_states=(source_attention,),
        index_states=(source_index,),
        state=SimpleNamespace(
            gdn_conv=DeviceBuffer(
                0x5000000,
                len(runner.gdn_states) * runner.gdn_states[0].conv_bytes_per_slot,
            ),
            gdn_matrix=DeviceBuffer(
                0x6000000,
                len(runner.gdn_states) * runner.gdn_states[0].recurrent_bytes_per_slot,
            ),
            ple_conv=DeviceBuffer(0x7000000, runner.ple_state.history_bytes_per_slot),
        ),
        ple_hash_states={0: SimpleNamespace(tokens=(1, 2), next_position=300)},
    )
    runner.runtime.fail_memcpy_at = 3
    with pytest.raises(RuntimeError, match="injected device copy failure"):
        runner.import_prefilled(1, owner_one, source=source)
    assert runner.poisoned_slots == (1,)
    assert not runner.is_poisoned(0)
    assert runner.position(1) == 0
    # The retry is refused: the slot holds a mix of copied and zeroed state.
    with pytest.raises(RuntimeError, match="poisoned"):
        runner.import_prefilled(1, owner_one, source=source)
    with pytest.raises(RuntimeError, match="poisoned"):
        runner.step([1], slots=[1], owners=[owner_one])
    # The untouched slot keeps working.
    runner.step([1], slots=[0], owners=[owner_zero], capture_logits=False)
    assert runner.position(0) == 1
    # A reset clears the poison and the import then succeeds end to end.
    runner.reset_slot(1, owner_one)
    assert runner.poisoned_slots == ()
    runner.runtime.fail_memcpy_at = None
    assert runner.import_prefilled(1, owner_one, source=source) == 300
    assert runner.position(1) == 300
    assert int(runner.index_states[0].cursor_host[1]) == 300


def test_each_qsa_layer_owns_its_own_paged_kv_cache(monkeypatch) -> None:
    """Two QSA layers must not share one K/V cache.

    The synthetic default geometry has a single QSA layer, so a shared cache
    looks correct there. The production model has 12 QSA layers: with one cache
    every layer would append over the previous layer's keys at the same physical
    block and position, and the import would keep only the last layer's state.
    """

    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=2)
    _use_two_qsa_layers(runner)

    seen: list[tuple[int, int]] = []

    def qsa_layer(
        residual_ptr,
        weights,
        *,
        attention_state,
        index_state,
        selection,
        scratch,
        rows,
        **kwargs,
    ):
        seen.append((id(attention_state), id(index_state)))
        return SimpleNamespace(ptr=residual_ptr + 0x10)

    monkeypatch.setattr(runner_module, "run_qwen4_exp_batch_qsa_layer", qsa_layer)
    owner = runner.admit(0)
    runner.step([3], slots=[0], owners=[owner], capture_logits=False)
    assert len(seen) == 2, "both QSA layers must dispatch"
    attention_ids = {entry[0] for entry in seen}
    index_ids = {entry[1] for entry in seen}
    assert len(attention_ids) == 2, "each QSA layer needs its own K/V owner"
    assert len(index_ids) == 2, "each QSA layer needs its own index owner"
    # Every layer's owner was prepared for this row set.
    for attention in runner.attention_states:
        assert attention.prepared == [((0,), (0,))]
    # Zeroing a slot touches every layer's cache, not only the first.
    runner.runtime.calls.clear()
    runner.admit(1)
    memsets = [call for call in runner.runtime.calls if call[0] == "memset"]
    attention = runner.attention_states[0]
    block_bytes = attention.block_size * attention.kv_heads * attention.head_dim * 2
    kv_bytes = 2 * len(runner.attention_states) * attention.blocks_per_request * block_bytes
    # Contiguous blocks are zeroed by one memset each, so count bytes, not calls.
    # The fakes share addresses, so each range is counted once per memset.
    caches = {
        (cache.ptr, cache.ptr + cache.nbytes)
        for state in runner.attention_states
        for cache in (state.key_cache, state.value_cache)
    }
    written = sum(
        call[3] for call in memsets
        for start, end in caches if start <= call[1] < end
    )
    assert written == kv_bytes


def test_step_uploads_packed_metadata_once_for_every_qsa_layer(monkeypatch) -> None:
    """One step publishes every layer's dynamic metadata with one upload.

    Before packing, each QSA layer uploaded its own position, context, and index
    query positions, so a step cost three copies per layer. The count below is
    the whole point of the unit: it must not grow with the layer count.
    """

    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=2)
    _use_two_qsa_layers(runner)
    metadata = runner.step_metadata
    assert metadata is not None
    owner_two = runner.admit(0)
    owner_zero = runner.admit(1)
    runner._positions[0] = 5
    runner._positions[1] = 9
    for index_state in runner.index_states:
        index_state.cursor_host[0] = 6
        index_state.cursor_host[1] = 10
    published_at_dispatch: list[bool] = []
    inner_qsa = runner_module.run_qwen4_exp_batch_qsa_layer

    def wrapped_qsa(*args, **kwargs):
        published_at_dispatch.append(
            all(state.active is not None for state in runner.attention_states)
            and all(state.active is not None for state in runner.index_states)
        )
        return inner_qsa(*args, **kwargs)

    monkeypatch.setattr(runner_module, "run_qwen4_exp_batch_qsa_layer", wrapped_qsa)
    runner.runtime.calls.clear()
    runner.step([7, 3], slots=[0, 1], owners=[owner_two, owner_zero], capture_logits=False)
    uploads = _packed_uploads(runner, metadata)
    assert uploads == [metadata.buffer.nbytes], (
        "a 2-layer step must publish its packed metadata exactly once"
    )
    # Every owner is published before the first QSA layer reads its metadata.
    assert published_at_dispatch == [True, True]
    # Row r carries row r's slot metadata, for every layer.
    packed = _packed_payload(runner)
    for layer in range(2):
        assert packed[layer, STEP_POSITION_COLUMN, :2].tolist() == [5, 9]
        assert packed[layer, STEP_CONTEXT_COLUMN, :2].tolist() == [6, 10]
        assert packed[layer, STEP_INDEX_POSITION_COLUMN, :2].tolist() == [5, 9]
    # Both owners published through the same buffer's per-layer views.
    for layer, attention in enumerate(runner.attention_states):
        slot_ids, packed_views = attention.published[-1]
        assert slot_ids == (0, 1)
        assert packed_views is metadata.views(layer)
        assert attention.active.position.ptr == metadata.views(layer).position.ptr
        assert attention.active.context.ptr == metadata.views(layer).context.ptr
    for layer, index_state in enumerate(runner.index_states):
        slot_ids, packed_views = index_state.published[-1]
        assert slot_ids == (0, 1)
        assert packed_views is metadata.views(layer)
        assert index_state.active.positions_device.ptr == (
            metadata.views(layer).index_position.ptr
        )


def test_step_packs_permuted_rows_in_the_callers_row_order(monkeypatch) -> None:
    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=3)
    metadata = runner.step_metadata
    assert metadata is not None
    owner_two = runner.admit(2)
    owner_zero = runner.admit(0)
    runner._positions[2] = 11
    runner._positions[0] = 2
    runner.index_states[0].cursor_host[2] = 12
    runner.index_states[0].cursor_host[0] = 3
    result = runner.step(
        [7, 9], slots=[2, 0], owners=[owner_two, owner_zero], capture_logits=False
    )
    assert result.slot_ids == (2, 0)
    packed = _packed_payload(runner)
    # Row 0 is slot 2 and row 1 is slot 0, in the caller's order.
    assert packed[0, STEP_POSITION_COLUMN, :2].tolist() == [11, 2]
    assert packed[0, STEP_CONTEXT_COLUMN, :2].tolist() == [12, 3]
    assert packed[0, STEP_INDEX_POSITION_COLUMN, :2].tolist() == [11, 2]
    # The permuted row set keeps the owners' row-to-slot mapping too.
    assert runner.attention_states[0].prepared[-1] == ((2, 0), (11, 2))
    assert runner.index_states[0].prepared[-1] == (2, 0)
    assert runner.position(2) == 12 and runner.position(0) == 3


def test_step_restages_the_packed_metadata_after_slot_reuse(monkeypatch) -> None:
    """A released and re-admitted slot must not leave a previous row published."""

    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=2)
    metadata = runner.step_metadata
    assert metadata is not None
    first = runner.admit(1)
    runner._positions[1] = 4
    runner.step([7], slots=[1], owners=[first], capture_logits=False)
    assert _packed_payload(runner)[0, STEP_POSITION_COLUMN, 0] == 4
    buffer_ptr = metadata.buffer.ptr
    runner.release(1, first)
    second = runner.admit(1)
    assert second != first
    runner.runtime.calls.clear()
    runner.step([7], slots=[1], owners=[second], capture_logits=False)
    # Same buffer, no reallocation, and the new request's position replaced the
    # retired request's row instead of leaving it in the packed copy.
    assert metadata.buffer.ptr == buffer_ptr
    assert _packed_payload(runner)[0, STEP_POSITION_COLUMN, 0] == 0
    assert _packed_uploads(runner, metadata) == [metadata.buffer.nbytes]
    assert runner.position(1) == 1


def test_step_validates_every_owner_before_the_packed_upload(monkeypatch) -> None:
    """A rejected owner row set must not upload or publish any metadata."""

    events = _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=2)
    metadata = runner.step_metadata
    assert metadata is not None
    owner = runner.admit(0)
    attention = runner.attention_states[0]

    def rejecting_validate_rows(slot_ids, positions):
        raise ValueError("injected row-set rejection")

    monkeypatch.setattr(attention, "validate_rows", rejecting_validate_rows)
    events.clear()
    runner.runtime.calls.clear()
    with pytest.raises(ValueError, match="injected row-set rejection"):
        runner.step([7], slots=[0], owners=[owner], capture_logits=False)
    assert _packed_uploads(runner, metadata) == []
    assert events == [], "a rejected row set must dispatch nothing"
    assert runner.poisoned_slots == ()
    assert attention.published == []
    assert runner.index_states[0].published == []
    assert runner.position(0) == 0


def test_step_rejects_a_packed_buffer_that_does_not_match_the_owners(monkeypatch) -> None:
    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=2)
    owner = runner.admit(0)
    runner.step_metadata = Qwen4ExpBatchStepMetadata.allocate(
        layers=2, state_slots=2, runtime=runner.runtime
    )
    with pytest.raises(RuntimeError, match="packed step metadata holds 2 layers"):
        runner.step([7], slots=[0], owners=[owner], capture_logits=False)
    runner.step_metadata = Qwen4ExpBatchStepMetadata.allocate(
        layers=1, state_slots=3, runtime=runner.runtime
    )
    with pytest.raises(RuntimeError, match="packed step metadata holds 3 slots"):
        runner.step([7], slots=[0], owners=[owner], capture_logits=False)
    runner.step_metadata = None
    with pytest.raises(RuntimeError, match="no packed step metadata buffer"):
        runner.step([7], slots=[0], owners=[owner], capture_logits=False)
    assert runner.poisoned_slots == ()
    assert runner.position(0) == 0


def test_step_survives_a_failed_packed_upload_by_poisoning_its_rows(monkeypatch) -> None:
    """A device copy that dies mid-step is still a poisoned row, not a retry."""

    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=2)
    metadata = runner.step_metadata
    assert metadata is not None
    owner_zero = runner.admit(0)
    owner_one = runner.admit(1)
    real_memcpy = runner.runtime.memcpy
    calls = {"n": 0}

    def failing_memcpy(destination, source, nbytes, kind) -> None:
        calls["n"] += 1
        if destination == metadata.buffer.ptr:
            raise RuntimeError("injected packed upload failure")
        real_memcpy(destination, source, nbytes, kind)

    runner.runtime.memcpy = failing_memcpy
    with pytest.raises(RuntimeError, match="injected packed upload failure"):
        runner.step([7], slots=[0], owners=[owner_zero], capture_logits=False)
    assert runner.poisoned_slots == (0,)
    # The untouched row keeps working, and its own packed upload succeeds.
    runner.runtime.memcpy = real_memcpy
    runner.step([7], slots=[1], owners=[owner_one], capture_logits=False)
    assert runner.position(1) == 1
    assert runner.position(0) == 0


# -- output-head graph capture --------------------------------------------


_HEAD_KEYS = ("head_gr_read", "lm_head")
_READBACK = ("d2h", "argmax")


def _install_head_graph_fakes(monkeypatch, runner: Qwen4ExpGGUFBatchRunner) -> None:
    """Head and readback fakes that read and write the fake device memory.

    The head writes ``mixed`` as a copy of the residual bytes and ``logits``
    derived from ``mixed``, so a replay that read a different pointer produces
    different tokens. Every device operation goes through ``runtime.device_op``,
    which records it while a capture is open, exactly as HIP captures a launch.
    """

    runtime = runner.runtime
    cfg = runner.config
    head_ptr = int(runner.head_scratch.ptr)
    mixed_ptr = head_ptr + 0x20

    def gr_read(
        residual_ptr, norm_ptr, down, up, inject, scratch, *, rows, stream=0, **kwargs
    ):
        assert int(scratch.ptr) == head_ptr
        runtime.calls.append(
            ("head_gr_read", int(residual_ptr), int(stream), int(rows))
        )

        def op() -> None:
            runtime.device_write(mixed_ptr, runtime.device_read(residual_ptr, 64))

        runtime.device_op(op)
        return SimpleNamespace(mixed=DeviceBuffer(mixed_ptr, int(rows) * 64))

    def lm_head(
        weight, x_ptr, out_ptr, rows, in_features, out_features, *, stream=0, **kwargs
    ):
        assert int(in_features) == cfg.hidden_size
        runtime.calls.append(("lm_head", int(x_ptr), int(out_ptr), int(stream)))

        def op() -> None:
            base = int.from_bytes(runtime.device_read(x_ptr, 2), "little") % int(
                out_features
            )
            values = np.zeros((int(rows), int(out_features)), dtype=np.float32)
            for row in range(int(rows)):
                values[row, (base + row) % int(out_features)] = 1.0
            runtime.device_write(out_ptr, values.tobytes())

        runtime.device_op(op)

    def argmax(
        logits_ptr, values_ptr, blocks_ptr, ids_ptr, out_values, rows, vocab, **kwargs
    ):
        runtime.calls.append(("argmax", int(logits_ptr), int(rows), int(vocab)))

        def op() -> None:
            values = np.frombuffer(
                runtime.device_read(logits_ptr, int(rows) * int(vocab) * 4),
                dtype=np.float32,
            ).reshape(int(rows), int(vocab))
            ids = np.ascontiguousarray(np.argmax(values, axis=1), dtype=np.int32)
            runtime.device_write(ids_ptr, ids.tobytes())

        runtime.device_op(op)

    def d2h(host_ptr, buffer, nbytes=None, runtime=None):
        runtime.calls.append(("d2h", int(buffer.ptr)))

    def h2d(buffer, array, nbytes=None, runtime=None):
        runtime.calls.append(("h2d", int(buffer.ptr)))

    monkeypatch.setattr(runner_module, "run_qwen4_exp_gr_read", gr_read)
    monkeypatch.setattr(runner_module, "launch_gguf_linear", lm_head)
    monkeypatch.setattr(runner_module, "argmax_f32_rows_i32", argmax)
    monkeypatch.setattr(runner_module, "copy_device_to_host", d2h)
    monkeypatch.setattr(runner_module, "copy_host_array_to_device", h2d)


def _graph_runner(
    monkeypatch, *, slots: int = 3, enabled: bool = True
) -> Qwen4ExpGGUFBatchRunner:
    """A fixture runner with the real graph cache and memory-modelling fakes."""

    _install_step_fakes(monkeypatch)
    runner = _fake_runner(slots=slots)
    _install_head_graph_fakes(monkeypatch, runner)
    runner._graphs = MoeGraphCache(runner.runtime, enabled=enabled)
    return runner


def _head_keys(runner: Qwen4ExpGGUFBatchRunner) -> list[tuple]:
    """Every captured key that names the output head, in capture order."""

    return [key for key in runner._graphs._execs if key[0] == "head"]


def _head_launches(calls: list[tuple]) -> list[tuple]:
    return [call for call in calls if call[0] in _HEAD_KEYS]


def _capture_brackets(calls: list[tuple]) -> list[tuple[int, int]]:
    """Index pairs of every ``begin_capture``/``end_capture`` in ``calls``."""

    brackets: list[tuple[int, int]] = []
    open_at: int | None = None
    for index, call in enumerate(calls):
        if call[0] == "begin_capture":
            assert open_at is None
            open_at = index
        elif call[0] == "end_capture":
            assert open_at is not None
            brackets.append((open_at, index))
            open_at = None
    assert open_at is None
    return brackets


def _head_residual_ptr(runner: Qwen4ExpGGUFBatchRunner) -> int:
    """The final residual pointer the last layer hands the head."""

    keys = _head_keys(runner)
    assert keys, "the head was never captured"
    return int(keys[0][2])


def test_step_captures_the_output_head_and_replays_it(monkeypatch) -> None:
    runner = _graph_runner(monkeypatch)
    owner_two = runner.admit(2)
    owner_zero = runner.admit(0)
    slots = [2, 0]
    owners = [owner_two, owner_zero]

    first = runner.step([7, 9], slots=slots, owners=owners, capture_logits=False)
    keys = _head_keys(runner)
    assert len(keys) == 1
    assert keys[0][0] == "head"
    assert keys[0][1] == 2
    assert keys[0][2] > 0
    # Capture runs the unit twice (eager reference, then the captured launch),
    # then replays it once for the bit-exact parity check.
    assert len(_head_launches(runner.runtime.calls)) == 4
    assert runner.graph_stats["capture"] >= 1

    launches_before = len(_head_launches(runner.runtime.calls))
    replays_before = runner.graph_stats["replay"]
    second = runner.step([7, 9], slots=slots, owners=owners, capture_logits=False)

    # A second step at the same width and residual pointer replays the graph.
    assert len(_head_launches(runner.runtime.calls)) == launches_before
    assert runner.graph_stats["replay"] == replays_before + 1
    assert _head_keys(runner) == keys
    assert np.array_equal(first.tokens, second.tokens)
    assert first.tokens.tolist() == [0, 1]


def test_head_graph_key_carries_the_row_width(monkeypatch) -> None:
    runner = _graph_runner(monkeypatch)
    owners = {slot: runner.admit(slot) for slot in (0, 1, 2)}

    runner.step([7, 9], slots=[2, 0], owners=[owners[2], owners[0]], capture_logits=False)
    wide = runner.step(
        [7, 9, 11], slots=[0, 1, 2], owners=[owners[0], owners[1], owners[2]],
        capture_logits=False,
    )
    assert wide.tokens.tolist() == [0, 1, 2]
    keys = _head_keys(runner)
    assert sorted(key[1] for key in keys) == [2, 3]
    launches_after_captures = len(_head_launches(runner.runtime.calls))

    # Returning to width 2 must replay its own graph, not the width-3 one.
    again = runner.step(
        [7, 9], slots=[2, 0], owners=[owners[2], owners[0]], capture_logits=False
    )
    assert again.tokens.tolist() == [0, 1]
    assert _head_keys(runner) == keys
    assert len(_head_launches(runner.runtime.calls)) == launches_after_captures


def test_head_graph_key_carries_the_residual_pointer(monkeypatch) -> None:
    runner = _graph_runner(monkeypatch)
    owners = {slot: runner.admit(slot) for slot in (0, 1)}
    shift = {"value": 0x10}
    last_layer = len(runner.config.layer_types) - 1
    base_gdn = runner_module.run_qwen4_exp_batch_gdn_layer

    def shifted_gdn(residual_ptr, weights, **kwargs):
        step = shift["value"] if weights.layer_id == last_layer else 0x10
        return SimpleNamespace(ptr=residual_ptr + step)

    monkeypatch.setattr(runner_module, "run_qwen4_exp_batch_gdn_layer", shifted_gdn)
    try:
        first = runner.step(
            [7, 9], slots=[1, 0], owners=[owners[1], owners[0]], capture_logits=False
        )
        assert first.tokens.tolist() == [0, 1]
        first_key = _head_keys(runner)[0]

        # Same width, different final residual pointer holding different bytes.
        shift["value"] = 0x1000
        stale_ptr = runner.residual.ptr + 0x1000 + 0x30
        runner.runtime.device_write(stale_ptr, b"\x05\x00" + bytes(62))
        second = runner.step(
            [7, 9], slots=[1, 0], owners=[owners[1], owners[0]], capture_logits=False
        )
        keys = _head_keys(runner)
        assert len(keys) == 2
        assert keys[1][1] == 2
        assert keys[1][2] == stale_ptr
        assert keys[1] != first_key
        # The head read the new pointer: the tokens follow its contents, which a
        # replay of the first graph could not produce.
        assert second.tokens.tolist() == [5, 6]
        launches_after_captures = len(_head_launches(runner.runtime.calls))

        # Back to the original pointer: its own graph replays.
        shift["value"] = 0x10
        third = runner.step(
            [7, 9], slots=[1, 0], owners=[owners[1], owners[0]], capture_logits=False
        )
        assert third.tokens.tolist() == [0, 1]
        assert _head_keys(runner) == keys
        assert len(_head_launches(runner.runtime.calls)) == launches_after_captures
    finally:
        monkeypatch.setattr(
            runner_module, "run_qwen4_exp_batch_gdn_layer", base_gdn
        )


def test_head_capture_reads_the_live_residual_at_the_baked_pointer(monkeypatch) -> None:
    runner = _graph_runner(monkeypatch)
    owner = runner.admit(0)
    runner.step([7], slots=[0], owners=[owner], capture_logits=False)
    residual_ptr = _head_residual_ptr(runner)
    launches = len(_head_launches(runner.runtime.calls))

    # Same pointer, new contents: the graph must read them, not a snapshot.
    runner.runtime.device_write(residual_ptr, b"\x07\x00" + bytes(62))
    result = runner.step([7], slots=[0], owners=[owner], capture_logits=False)

    assert result.tokens.tolist() == [7]
    assert len(_head_launches(runner.runtime.calls)) == launches


def test_head_readback_stays_outside_every_capture(monkeypatch) -> None:
    runner = _graph_runner(monkeypatch)
    owners = {slot: runner.admit(slot) for slot in (0, 1)}

    runner.step(
        [7, 9], slots=[1, 0], owners=[owners[1], owners[0]], capture_logits=False
    )
    logits_result = runner.step(
        [7, 9], slots=[1, 0], owners=[owners[1], owners[0]], capture_logits=True
    )

    brackets = _capture_brackets(runner.runtime.calls)
    assert brackets
    for start, end in brackets:
        inside = [call[0] for call in runner.runtime.calls[start : end + 1]]
        assert set(inside) == {"begin_capture", "head_gr_read", "lm_head", "end_capture"}
        assert not [name for name in inside if name in _READBACK]
    # The logits copy follows the head unit, and the argmax path likewise.
    d2h = [
        index
        for index, call in enumerate(runner.runtime.calls)
        if call[0] == "d2h" and call[1] == runner.logits.ptr
    ]
    assert d2h and d2h[0] > brackets[0][1]
    argmax = [
        index
        for index, call in enumerate(runner.runtime.calls)
        if call[0] == "argmax"
    ]
    assert argmax and min(argmax) > brackets[0][1]
    assert logits_result.logits is not None
    assert np.argmax(logits_result.logits, axis=1).tolist() == [0, 1]


def test_head_falls_back_to_eager_when_the_capture_is_refused(monkeypatch) -> None:
    runner = _graph_runner(monkeypatch)
    runner.runtime.fail_capture = True
    owners = {slot: runner.admit(slot) for slot in (0, 1)}

    first = runner.step(
        [7, 9], slots=[1, 0], owners=[owners[1], owners[0]], capture_logits=False
    )
    head_eager = [key for key in runner._graphs._eager_only if key[0] == "head"]
    assert len(head_eager) == 1
    assert head_eager[0][1] == 2
    assert runner.graph_stats["replay"] == 0
    launches_after_first = len(_head_launches(runner.runtime.calls))

    second = runner.step(
        [7, 9], slots=[1, 0], owners=[owners[1], owners[0]], capture_logits=False
    )
    assert len(_head_launches(runner.runtime.calls)) == launches_after_first + 2
    assert runner.graph_stats["replay"] == 0
    assert first.tokens.tolist() == second.tokens.tolist() == [0, 1]


def test_step_without_a_graph_cache_runs_the_head_eager(monkeypatch) -> None:
    runner = _graph_runner(monkeypatch)
    runner._graphs.close()
    runner._graphs = None
    owner = runner.admit(0)

    result = runner.step([7], slots=[0], owners=[owner], capture_logits=False)

    assert runner.graph_stats == {}
    assert result.tokens.tolist() == [0]
    assert len(_head_launches(runner.runtime.calls)) == 2
    assert not [call for call in runner.runtime.calls if call[0] == "begin_capture"]


def test_close_destroys_the_head_graph_before_its_buffers(monkeypatch) -> None:
    runner = _graph_runner(monkeypatch)
    owner = runner.admit(0)
    runner.step([7], slots=[0], owners=[owner], capture_logits=False)
    assert _head_keys(runner)

    class _Owner:
        def close(self) -> None:
            runner.runtime.calls.append(("close_owner",))

    runner._owned.append(_Owner())
    runner.close()

    calls = runner.runtime.calls
    destroyed = [
        index for index, call in enumerate(calls) if call[0] == "graph_exec_destroy"
    ]
    closed = [index for index, call in enumerate(calls) if call[0] == "close_owner"]
    assert destroyed and closed
    assert max(destroyed) < closed[0]


def test_batch_owner_declares_mixed_target_support() -> None:
    """Serving reads this off the owner to route a speculative row to the pass."""

    assert Qwen4ExpGGUFBatchRunner.supports_mixed_target is True
    # A class attribute, so the fixture's ``object.__new__`` instance has it too.
    runner = _fake_runner(slots=1)
    assert getattr(runner, "supports_mixed_target", False) is True
    # An owner that does not declare it keeps the c1 MTP path.
    assert getattr(SimpleNamespace(), "supports_mixed_target", False) is False


# -- host-source ownership ---------------------------------------------------


def test_step_routes_token_and_ple_uploads_through_the_owning_array_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_step_fakes(monkeypatch)
    calls: list[tuple[object, np.ndarray, int | None]] = []
    real_upload = runner_module.copy_host_array_to_device

    def recording_upload(buffer, array, nbytes=None, *, runtime=None):
        calls.append((buffer, array, nbytes))
        real_upload(buffer, array, nbytes, runtime=runtime)

    monkeypatch.setattr(
        runner_module, "copy_host_array_to_device", recording_upload
    )
    runner = _fake_runner(slots=2)
    owner = runner.admit(0)
    runner.step([3], slots=[0], owners=[owner], capture_logits=False)

    assert len(calls) == 2
    token_buffer, tokens, token_nbytes = calls[0]
    assert token_buffer is runner.token_ids
    assert tokens.tolist() == [3]
    assert tokens.flags.c_contiguous
    assert token_nbytes == tokens.nbytes == 8
    ple_buffer, ple_rows, ple_nbytes = calls[1]
    assert ple_buffer is runner.ple_embedding
    assert ple_rows.flags.c_contiguous
    assert ple_rows.shape[0] == 1
    assert ple_nbytes == ple_rows.nbytes


def test_step_contiguizes_a_strided_ple_stage_without_changing_its_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_step_fakes(monkeypatch)
    staged_ref: list[np.ndarray] = []

    def strided_stage(staging, rows, *, hidden):
        base = np.arange(len(rows) * hidden * 2, dtype=np.float32).reshape(
            len(rows), hidden * 2
        )
        staged = base[:, ::2]
        staged_ref.append(staged)
        return staged

    monkeypatch.setattr(
        runner_module, "stage_qwen4_exp_ple_rows", strided_stage
    )
    calls: list[tuple[object, np.ndarray, int | None]] = []
    real_upload = runner_module.copy_host_array_to_device

    def recording_upload(buffer, array, nbytes=None, *, runtime=None):
        calls.append((buffer, array, nbytes))
        real_upload(buffer, array, nbytes, runtime=runtime)

    monkeypatch.setattr(
        runner_module, "copy_host_array_to_device", recording_upload
    )
    runner = _fake_runner(slots=2)
    owner = runner.admit(0)
    runner.step([3], slots=[0], owners=[owner], capture_logits=False)

    assert len(staged_ref) == 1
    staged = staged_ref[0]
    assert not staged.flags.c_contiguous
    ple_calls = [call for call in calls if call[0] is runner.ple_embedding]
    assert len(ple_calls) == 1
    _, uploaded, nbytes = ple_calls[0]
    assert uploaded.flags.c_contiguous
    assert nbytes == staged.nbytes
    assert uploaded.nbytes == staged.nbytes
    assert np.array_equal(uploaded, staged)
