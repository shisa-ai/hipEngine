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

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen4_exp_batch_runner as runner_module
from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner
from tests import _qwen4_exp_batch_fixtures as fixture


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        # 1-based index of the copy/zero that raises, for fault injection.
        self.fail_memcpy_at: int | None = None
        self.fail_memset_at: int | None = None
        self.memcpy_count = 0
        self.memset_count = 0

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

    def device_synchronize(self) -> None:
        self.calls.append(("sync",))


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
        self.active = None
        self.block_table = DeviceBuffer(0x300000, slots * blocks * 4)

    def prepare(self, slot_ids, positions):
        slots = tuple(int(slot) for slot in slot_ids)
        values = tuple(int(position) for position in positions)
        self.prepared.append((slots, values))
        self.position_host[: len(slots)] = values
        self.context_host[: len(slots)] = [value + 1 for value in values]
        self.active = SimpleNamespace(
            slot_ids=slots,
            rows=len(slots),
            block_table=self.block_table,
            position=DeviceBuffer(0x400000, len(slots) * 8),
            context=DeviceBuffer(0x500000, len(slots) * 8),
            append_spans=object(),
            decode_spans=object(),
            max_context_len=max(values) + 1,
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
        self.resets: list[int] = []
        self.active = None

    def reset_slot(self, slot):
        self.resets.append(int(slot))
        self.cursor_host[slot] = 0
        self.pooled_count_host[slot] = 0
        self.reset_generation[slot] += 1
        self.active = None

    def prepare(self, slot_ids):
        slots = tuple(int(slot) for slot in slot_ids)
        self.prepared.append(slots)
        self.active = SimpleNamespace(slot_ids=slots, rows=len(slots))
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
    runner.index_states = (_FakeIndex(slots),)
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
        "copy_host_to_device",
        lambda buffer, host_ptr, nbytes=None, runtime=None: events.append(
            (
                "h2d",
                int(buffer.ptr),
                int(nbytes if nbytes is not None else buffer.nbytes),
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
        _FakeAttention(2, 2),
        _FakeAttention(2, 2),
    )
    runner.index_states = (_FakeIndex(2), _FakeIndex(2))

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
    assert sum(call[3] for call in memsets if call[3] == block_bytes) == (
        kv_bytes // block_bytes
    ) * block_bytes
