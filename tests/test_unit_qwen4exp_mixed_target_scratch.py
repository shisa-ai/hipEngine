"""Reuse, growth and release of the batch-owned mixed target scratch.

A mixed target pass is built and closed once per serving cycle. Its stateless
half -- the token/embedding/residual/logits slabs plus the per-layer GDN, QSA
and head stage scratch -- depends only on the physical row count and the model
geometry, so the batch runner keeps one set and every pass borrows it.

These tests pin the lifecycle a device run cannot isolate: a same-or-narrower
pass reuses the installed buffers without allocating, a wider pass releases
exactly the previous capacity, a failed growth leaks nothing and stays
retryable, a pass close leaves the shared set allocated for the next pass while
its own provisional state is released, the batch releases the set exactly once,
and a returned result is a host copy that a later pass cannot overwrite.

The fake runtime stores real bytes, so the pass runs its own ``verify``
orchestration (token upload, embedding lookup, residual repeat, GR read, LM head
and readback) against recording kernels instead of a GPU.
"""

from __future__ import annotations

import ctypes
from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen4_exp_mixed_target as mixed_target
from hipengine.core.memory import free, malloc
from hipengine.core.runtime import MemcpyKind
from hipengine.runtime.qwen4_exp_batch_runner import (
    Qwen4ExpGGUFBatchRunner,
    Qwen4ExpMixedTargetScratch,
)
from hipengine.runtime.qwen4_exp_mixed_layout import (
    Qwen4ExpMixedRequest,
    compile_mixed_layout,
)
from hipengine.runtime.qwen4_exp_mixed_target import Qwen4ExpMixedTargetPass
from tests import _qwen4_exp_batch_fixtures as fixture


# -- host-side device -------------------------------------------------------


class _FakeDevice:
    """A HIP runtime over host bytes: real allocations, real frees, real copies."""

    def __init__(self) -> None:
        self.next_ptr = 0x1000000
        self.live: dict[int, bytearray] = {}
        self.freed: list[int] = []
        self.malloc_count = 0
        # 1-based index of the allocation that raises, for fault injection.
        self.fail_malloc_at: int | None = None

    def malloc(self, nbytes: int) -> int:
        self.malloc_count += 1
        if self.fail_malloc_at == self.malloc_count:
            raise MemoryError("injected device allocation failure")
        ptr = self.next_ptr
        self.next_ptr += int(nbytes) + 0x1000
        self.live[ptr] = bytearray(int(nbytes))
        return ptr

    def free(self, ptr: int) -> None:
        if int(ptr) not in self.live:
            raise AssertionError(f"free of an unallocated device pointer {int(ptr):#x}")
        del self.live[int(ptr)]
        self.freed.append(int(ptr))

    def memcpy(self, destination, source, nbytes, kind) -> None:
        count = int(nbytes)
        if kind == MemcpyKind.HOST_TO_DEVICE:
            self.view(destination)[:count] = ctypes.string_at(int(source), count)
        elif kind == MemcpyKind.DEVICE_TO_HOST:
            ctypes.memmove(int(destination), bytes(self.view(source)[:count]), count)
        else:  # pragma: no cover - the pass never issues another kind
            raise AssertionError(f"unexpected copy kind {kind!r}")

    def view(self, ptr: int) -> memoryview:
        for base, data in self.live.items():
            if base <= int(ptr) < base + len(data):
                return memoryview(data)[int(ptr) - base :]
        raise AssertionError(f"device pointer {int(ptr):#x} is not inside a live allocation")

    def write(self, ptr: int, array) -> None:
        values = np.ascontiguousarray(array)
        self.view(ptr)[: values.nbytes] = values.tobytes()

    def read(self, ptr: int, dtype, shape) -> np.ndarray:
        count = int(np.prod(shape))
        itemsize = np.dtype(dtype).itemsize
        raw = bytes(self.view(ptr)[: count * itemsize])
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


# -- recording kernels ------------------------------------------------------


def _install_fake_kernels(monkeypatch, device: _FakeDevice) -> None:
    """Stand in for the four device stages a layer-free pass still runs."""

    def fake_resolve(**kwargs):
        def lookup(token_ids_ptr, weight_ptr, out_ptr, rows, hidden, vocab, *, runtime):
            tokens = device.read(token_ids_ptr, np.int64, (rows,))
            values = np.tile((tokens.astype(np.float32) + 0.5)[:, None], (1, hidden))
            # The embedding stage writes BF16 rows.
            device.write(out_ptr, (values.view(np.uint32) >> 16).astype(np.uint16))

        return lookup

    def fake_repeat(embedding_ptr, residual_ptr, branches, hidden, *, rows, runtime):
        values = device.read(embedding_ptr, np.uint16, (rows, hidden))
        device.write(residual_ptr, np.tile(values, (1, branches)))

    def fake_gr_read(residual_ptr, norm_ptr, down, up, inject, scratch, *,
                     rows, branches, hidden, low_rank, runtime):
        bits = device.read(residual_ptr, np.uint16, (rows, branches * hidden))
        mixed = (bits.astype(np.uint32) << 16).view(np.float32)[:, :hidden]
        device.write(scratch.mixed.ptr, mixed)
        return SimpleNamespace(mixed=scratch.mixed, inject_logits=scratch.inject_logits)

    def fake_lm_head(weight, mixed_ptr, out_ptr, rows, hidden, vocab, *,
                     activation_dtype, output_dtype, runtime):
        values = device.read(mixed_ptr, np.float32, (rows, hidden))
        logits = np.zeros((rows, vocab), dtype=np.float32)
        for row, token in enumerate(values[:, 0].astype(np.int64)):
            logits[row, int(token) % vocab] = 1.0
        device.write(out_ptr, logits)

    monkeypatch.setattr(mixed_target, "resolve", fake_resolve)
    monkeypatch.setattr(mixed_target, "qwen4_exp_repeat_bf16_branches", fake_repeat)
    monkeypatch.setattr(mixed_target, "run_qwen4_exp_gr_read", fake_gr_read)
    monkeypatch.setattr(mixed_target, "launch_gguf_linear", fake_lm_head)


# -- batch and layout doubles ----------------------------------------------


def _stateless_config():
    """The shared geometry with no layers, so a pass allocates no adapters."""

    return replace(
        fixture.config(),
        block_count=0,
        layer_types=(),
        attention_compress_ratios=(),
        ple_layers=(),
    )


def _gdn_config():
    """Two GDN layers, so a failed adapter allocation leaves one to clean up."""

    return replace(
        fixture.config(),
        block_count=2,
        layer_types=("gdn", "gdn"),
        attention_compress_ratios=(0, 0),
        ple_layers=(),
    )


def _fake_weight(name):
    return SimpleNamespace(
        spec=SimpleNamespace(quant_key="q4_k_m"),
        allocation=lambda view: SimpleNamespace(
            tensor=SimpleNamespace(ptr=0x900000 + (len(name) * 16))
        ),
    )


def _batch(device: _FakeDevice, *, config=None, slots: int = 3):
    batch = SimpleNamespace(
        config=config or _stateless_config(),
        runtime=device,
        backend="hip_gfx1100",
        closed=False,
        closing=False,
        _owned=[],
        _mixed_target_scratch=None,
        _graphs=None,
        _buffers=None,
        _positions=np.zeros(slots, dtype=np.int64),
        _owners=list(range(1, slots + 1)),
        _poisoned=set(),
        _ple_hash_states={},
        _ple_layers=frozenset(),
        gdn_bindings={},
        qsa_bindings={},
        gdn_states=(),
        attention_states=(),
        index_states=[],
        ple_state=None,
        max_positions=fixture.MAX_POSITIONS,
        _slot_blocks=[2] * slots,
        resident=SimpleNamespace(weight=_fake_weight),
    )

    def require_open():
        if batch.closed or batch.closing:
            raise RuntimeError("Qwen4Exp batch runner is closed")

    def require_slot(slot):
        if not 0 <= int(slot) < slots:
            raise ValueError(f"slot {slot} is outside [0, {slots})")
        return int(slot)

    def require_owner(slot, token):
        if batch._owners[slot] != token:
            raise ValueError("stale owner")

    batch._require_open = require_open
    batch._require_slot = require_slot
    batch._require_owner = require_owner
    batch._require_healthy = lambda slot: None
    batch.blocks_for_positions = lambda positions: (int(positions) + 255) // 256
    batch.mixed_target_scratch = MethodType(
        Qwen4ExpGGUFBatchRunner.mixed_target_scratch, batch
    )
    return batch


def _layout(tokens):
    return compile_mixed_layout(
        tuple(
            Qwen4ExpMixedRequest(100 + index, index, index + 1, 0, int(token))
            for index, token in enumerate(tokens)
        )
    )


def _pass(batch, *, tokens):
    return Qwen4ExpMixedTargetPass(batch, _layout(tokens))


def _stub_gdn_adapter(monkeypatch, device: _FakeDevice, *, fail_at: int | None = None):
    """A provisional GDN adapter that owns one real device buffer per pass."""

    created: list[object] = []

    class _Adapter:
        def __init__(self, buffer) -> None:
            self.buffer = buffer
            self.closed = False

        def close(self) -> None:
            if not self.closed:
                free(self.buffer, runtime=device)
                self.closed = True

        @classmethod
        def allocate(cls, *, layout, live_state, owner_of, runtime):
            if fail_at is not None and len(created) >= fail_at:
                raise MemoryError("injected provisional allocation failure")
            adapter = cls(malloc(layout.total_rows * 4, runtime=runtime))
            created.append(adapter)
            return adapter

    monkeypatch.setattr(mixed_target, "Qwen4ExpMixedGdnAdapter", _Adapter)
    return created


def _pass_buffers(target):
    return (
        target.token_ids,
        target.embedding,
        target.ple_embedding,
        target.residual,
        target.logits,
        target.gdn_scratch,
        target.qsa_scratch,
        target.head,
    )


# -- reuse ------------------------------------------------------------------


def test_a_narrower_pass_reuses_the_installed_capacity_without_allocating():
    device = _FakeDevice()
    batch = _batch(device)

    wide = batch.mixed_target_scratch(6)
    assert isinstance(wide, Qwen4ExpMixedTargetScratch)
    allocated = device.malloc_count

    narrow = batch.mixed_target_scratch(2)

    assert narrow is wide
    assert narrow.capacity_rows == 6
    assert device.malloc_count == allocated
    assert not wide.closed


def test_a_second_pass_at_the_same_width_borrows_the_same_buffers(monkeypatch):
    device = _FakeDevice()
    batch = _batch(device)
    _install_fake_kernels(monkeypatch, device)

    first = _pass(batch, tokens=(11, 12))
    buffers = _pass_buffers(first)
    allocated = device.malloc_count
    first.verify()
    first.close()

    second = _pass(batch, tokens=(13, 14))

    assert second.scratch is first.scratch
    assert _pass_buffers(second) == buffers
    assert device.malloc_count == allocated


def test_a_narrower_pass_runs_on_the_wider_capacity_without_allocating(monkeypatch):
    device = _FakeDevice()
    batch = _batch(device)
    _install_fake_kernels(monkeypatch, device)

    wide = _pass(batch, tokens=(11, 12, 13))
    wide.verify()
    wide.close()
    allocated = device.malloc_count

    narrow = _pass(batch, tokens=(14, 15))

    assert narrow.scratch is wide.scratch
    assert narrow.scratch.capacity_rows == 3
    result = narrow.verify()
    narrow.close()

    assert device.malloc_count == allocated
    assert result.logits.shape == (2, fixture.VOCAB)
    assert tuple(int(token) for token in result.token_ids) == (14, 15)
    assert np.allclose(result.hidden_seeds[:, 0], (14.5, 15.5))


def test_a_pass_borrows_the_scratch_and_owns_only_provisional_state(monkeypatch):
    device = _FakeDevice()
    batch = _batch(device, config=_gdn_config())
    batch.gdn_bindings = {0: SimpleNamespace(gdn_state_index=0),
                          1: SimpleNamespace(gdn_state_index=1)}
    batch.gdn_states = (object(), object())
    _stub_gdn_adapter(monkeypatch, device)

    target = _pass(batch, tokens=(11,))

    assert target.scratch.borrowers == 1
    # Every stateless slab is the batch's; only the depth adapters are per pass.
    assert sorted(id(owner) for owner in target._owned) == sorted(
        id(adapter) for adapter in target.gdn.values()
    )
    for buffer in _pass_buffers(target):
        assert buffer not in target._owned
    target.close()


# -- growth -----------------------------------------------------------------


def test_a_wider_pass_grows_once_and_frees_only_the_previous_capacity():
    device = _FakeDevice()
    batch = _batch(device)

    small = batch.mixed_target_scratch(2)
    small_ptrs = set(device.live)
    assert small_ptrs

    big = batch.mixed_target_scratch(5)

    assert big is not small
    assert small.closed
    assert big.capacity_rows == 5
    assert device.live, "the grown capacity allocated its own set"
    assert small_ptrs.isdisjoint(device.live)
    assert sorted(device.freed) == sorted(small_ptrs)
    assert batch._mixed_target_scratch is big
    assert batch._owned == [big]


def test_a_failed_growth_keeps_the_previous_capacity_and_leaks_nothing():
    device = _FakeDevice()
    batch = _batch(device)

    small = batch.mixed_target_scratch(2)
    installed = set(device.live)
    device.fail_malloc_at = device.malloc_count + 60

    with pytest.raises(MemoryError, match="injected device allocation failure"):
        batch.mixed_target_scratch(5)

    assert set(device.live) == installed, "the failed growth leaked no device bytes"
    assert installed.isdisjoint(device.freed), "the failed growth kept the previous capacity"
    assert not small.closed and not small.closing
    assert batch._mixed_target_scratch is small
    assert batch._owned == [small]

    device.fail_malloc_at = None
    grown = batch.mixed_target_scratch(5)

    assert grown.capacity_rows == 5
    assert small.closed
    assert installed <= set(device.freed)


def test_growth_is_refused_while_a_pass_leases_the_scratch():
    device = _FakeDevice()
    batch = _batch(device)

    target = _pass(batch, tokens=(11,))
    installed = set(device.live)

    with pytest.raises(RuntimeError, match="still leased"):
        batch.mixed_target_scratch(3)

    assert set(device.live) == installed
    assert not target.scratch.closed

    target.close()
    assert batch.mixed_target_scratch(3).capacity_rows == 3


@pytest.mark.parametrize("rows", [2, 1])
def test_a_second_live_pass_is_refused_before_it_touches_the_row_slabs(
        monkeypatch, rows):
    """The lease is exclusive whatever the row count: a narrower pass aliases.

    The set holds one pass's rows. A second pass -- even one that would fit the
    installed capacity -- would overwrite the first pass's token, residual,
    logit and stage bytes mid-verification, which per-pass allocation never
    allowed. It must be refused before it allocates provisional state or writes
    a byte, and the refusal must leave the first pass usable and the lease held.
    """

    device = _FakeDevice()
    batch = _batch(device, config=_gdn_config())
    batch.gdn_bindings = {0: SimpleNamespace(gdn_state_index=0),
                          1: SimpleNamespace(gdn_state_index=1)}
    batch.gdn_states = (object(), object())
    created = _stub_gdn_adapter(monkeypatch, device)

    owner = _pass(batch, tokens=(11, 12))
    scratch = owner.scratch
    assert scratch.capacity_rows == 2
    slabs = {buffer.ptr for buffer in (
        scratch.token_ids, scratch.embedding, scratch.ple_embedding,
        scratch.residual, scratch.logits, scratch.gdn.output, scratch.qsa.output,
        scratch.head.mixed,
    )}
    installed = set(device.live)
    allocated = device.malloc_count

    tokens = (12,) if rows == 1 else (12, 13)
    assert len(tokens) <= scratch.capacity_rows, "the refusal is the lease, not growth"
    with pytest.raises(RuntimeError, match="already leased by a live mixed target pass"):
        _pass(batch, tokens=tokens)

    assert device.malloc_count == allocated, "the refused pass allocated nothing"
    assert set(device.live) == installed
    assert device.freed == []
    assert scratch.borrowers == 1
    assert len(created) == 2, "the refused pass built no provisional adapter"
    assert batch._mixed_target_scratch is scratch

    # The owner is untouched: its adapters are open and its own close works.
    assert not owner.gdn[0].closed and not owner.gdn[1].closed
    owner.close()
    assert owner.gdn[0].closed and owner.gdn[1].closed
    assert scratch.borrowers == 0
    assert not scratch.closed

    # The lease is acquirable again, on the same row slabs. Only its own
    # provisional adapters are new bytes.
    successor = _pass(batch, tokens=tokens)

    assert successor.scratch is scratch
    assert successor.scratch.borrowers == 1
    assert slabs <= set(device.live), "the successor runs on the same row slabs"
    assert len(set(device.live) - installed) == len(successor.gdn) == 2
    successor.close()
    assert scratch.borrowers == 0


@pytest.mark.parametrize("closing", ["close", "failed_close"])
def test_the_lease_is_released_even_by_a_close_that_leaves_the_pass_retryable(
        monkeypatch, closing):
    """A stuck lease would refuse every later pass, so close always drops it."""

    device = _FakeDevice()
    batch = _batch(device, config=_gdn_config())
    batch.gdn_bindings = {0: SimpleNamespace(gdn_state_index=0)}
    batch.gdn_states = (object(),)
    created = _stub_gdn_adapter(monkeypatch, device)

    target = _pass(batch, tokens=(11,))
    scratch = target.scratch
    if closing == "failed_close":
        def fail():
            raise RuntimeError("adapter free failed")
        created[0].close = fail
        with pytest.raises(RuntimeError, match="adapter free failed"):
            target.close()
        assert not target.closed and target.closing
    else:
        target.close()

    assert scratch.borrowers == 0
    successor = _pass(batch, tokens=(12,))
    assert successor.scratch is scratch
    successor.close()


def test_a_failed_provisional_allocation_leaves_the_shared_scratch_usable(monkeypatch):
    device = _FakeDevice()
    batch = _batch(device, config=_gdn_config())
    batch.gdn_bindings = {0: SimpleNamespace(gdn_state_index=0),
                          1: SimpleNamespace(gdn_state_index=1)}
    batch.gdn_states = (object(), object())
    # The first adapter is allocated and then released by the failed pass; the
    # second raises, so the failure happens with provisional state already held.
    _stub_gdn_adapter(monkeypatch, device, fail_at=1)
    scratch = batch.mixed_target_scratch(1)
    installed = set(device.live)

    with pytest.raises(MemoryError, match="injected provisional allocation failure"):
        _pass(batch, tokens=(11,))

    assert set(device.live) == installed, "the failed adapter leaked no device bytes"
    assert not scratch.closed
    assert scratch.borrowers == 0

    _stub_gdn_adapter(monkeypatch, device)
    retried = _pass(batch, tokens=(11,))

    assert retried.scratch is scratch
    retried.close()


# -- release ----------------------------------------------------------------


def test_closing_a_pass_keeps_the_shared_scratch_allocated(monkeypatch):
    device = _FakeDevice()
    batch = _batch(device)
    _install_fake_kernels(monkeypatch, device)

    target = _pass(batch, tokens=(11, 12))
    scratch = target.scratch
    installed = set(device.live)
    target.verify()

    target.close()

    assert target.closed
    assert not scratch.closed and not scratch.closing
    assert set(device.live) == installed
    assert device.freed == []
    assert scratch.borrowers == 0

    again = _pass(batch, tokens=(13, 14))

    assert again.scratch is scratch
    again.close()


def test_each_pass_builds_its_own_provisional_state(monkeypatch):
    device = _FakeDevice()
    batch = _batch(device, config=_gdn_config())
    batch.gdn_bindings = {0: SimpleNamespace(gdn_state_index=0),
                          1: SimpleNamespace(gdn_state_index=1)}
    batch.gdn_states = (object(), object())
    created = _stub_gdn_adapter(monkeypatch, device)

    first = _pass(batch, tokens=(11,))
    assert first.scratch.borrowers == 1
    first_adapters = dict(first.gdn)

    first.close()

    assert first.gdn[0].closed and first.gdn[1].closed
    assert not first.scratch.closed
    assert first.scratch.borrowers == 0

    second = _pass(batch, tokens=(12,))

    assert second.scratch is first.scratch
    assert len(created) == 4
    for layer in (0, 1):
        assert second.gdn[layer] is not first_adapters[layer]
        assert not second.gdn[layer].closed

    second.close()
    assert second.gdn[0].closed and second.gdn[1].closed


def test_batch_close_releases_the_shared_scratch_exactly_once():
    device = _FakeDevice()
    batch = _batch(device)

    scratch = batch.mixed_target_scratch(3)
    installed = set(device.live)
    assert installed

    Qwen4ExpGGUFBatchRunner.close(batch)

    assert scratch.closed
    assert device.live == {}
    assert sorted(device.freed) == sorted(installed)
    assert batch._mixed_target_scratch is None

    Qwen4ExpGGUFBatchRunner.close(batch)

    assert sorted(device.freed) == sorted(installed)


def test_a_pass_closes_cleanly_after_the_runner_released_its_scratch(monkeypatch):
    """The runner's close frees the set regardless; the pass must not double-free.

    Serving closes the pass before the runner, but a runner torn down while a
    pass is still leased must not turn that pass's own close into a second free
    of the shared slabs, and the pass must still drop the lease so its close
    completes instead of failing on a closed set.
    """

    device = _FakeDevice()
    batch = _batch(device, config=_gdn_config())
    batch.gdn_bindings = {0: SimpleNamespace(gdn_state_index=0)}
    batch.gdn_states = (object(),)
    created = _stub_gdn_adapter(monkeypatch, device)

    target = _pass(batch, tokens=(11,))
    scratch = target.scratch
    adapter_ptr = created[0].buffer.ptr
    slabs = set(device.live) - {adapter_ptr}
    assert slabs

    Qwen4ExpGGUFBatchRunner.close(batch)

    assert scratch.closed
    assert slabs.isdisjoint(device.live), "the runner released the shared slabs"
    assert adapter_ptr in device.live, "the pass's provisional state is its own"
    assert scratch.borrowers == 1, "the lease is still held by the open pass"

    target.close()

    assert target.closed
    assert created[0].closed, "the pass still releases its own provisional state"
    assert scratch.borrowers == 0
    # Each shared slab was freed exactly once, by the runner.
    assert len(device.freed) == len(set(device.freed))


# -- result ownership -------------------------------------------------------


def _record_copies(monkeypatch):
    """Record every host/device copy the pass issues and its byte count."""

    calls: list[tuple[str, object, int | None]] = []
    real_h2d = mixed_target.copy_host_array_to_device
    real_d2h = mixed_target.copy_device_to_host

    def h2d(buffer, array, nbytes=None, *, runtime=None):
        calls.append(("h2d", buffer, nbytes))
        return real_h2d(buffer, array, nbytes, runtime=runtime)

    def d2h(host_ptr, buffer, nbytes=None, *, runtime=None):
        calls.append(("d2h", buffer, nbytes))
        return real_d2h(host_ptr, buffer, nbytes, runtime=runtime)

    monkeypatch.setattr(mixed_target, "copy_host_array_to_device", h2d)
    monkeypatch.setattr(mixed_target, "copy_device_to_host", d2h)
    return calls


def test_every_copy_names_the_pass_row_extent_not_the_capacity(monkeypatch):
    """A borrowed slab is wider than the pass, so no copy may default to it.

    ``copy_device_to_host`` defaults to the device buffer extent, whereas
    ``copy_host_array_to_device`` defaults to the host source extent. The
    scratch is allocated at the widest capacity admitted. Explicit row-sized
    lengths keep both directions bounded to the pass; a capacity-sized D2H
    copy would write past the host array.
    """

    device = _FakeDevice()
    batch = _batch(device)
    _install_fake_kernels(monkeypatch, device)
    calls = _record_copies(monkeypatch)

    wide = _pass(batch, tokens=(11, 12, 13))
    wide.close()
    target = _pass(batch, tokens=(11, 12))
    assert target.scratch.capacity_rows == 3 > target.layout.total_rows == 2
    target.verify()
    target.close()

    named = {kind: [(buffer, nbytes) for recorded, buffer, nbytes in calls
                    if recorded == kind] for kind in ("h2d", "d2h")}
    for kind, recorded in named.items():
        assert recorded, f"the pass issued {kind} copies"
        for buffer, nbytes in recorded:
            assert nbytes is not None, (
                f"{kind} on a {buffer.nbytes}-byte slab defaulted to the capacity size"
            )
    tokens = [nbytes for buffer, nbytes in named["h2d"] if buffer is target.token_ids]
    logits = [nbytes for buffer, nbytes in named["d2h"] if buffer is target.logits]
    hidden = [nbytes for buffer, nbytes in named["d2h"] if buffer is target.residual]
    assert tokens == [2 * 8]
    assert logits == [2 * fixture.VOCAB * 4]
    assert hidden == [2 * fixture.RESIDUAL_WIDTH * 2]


def test_the_ple_row_upload_names_the_pass_row_extent(monkeypatch):
    device = _FakeDevice()
    batch = _batch(device, config=replace(_stateless_config(), ple_layers=(0,)))
    batch.resident.ple_staging = object()
    calls = _record_copies(monkeypatch)

    wide = _pass(batch, tokens=(11, 12, 13))
    wide.close()
    target = _pass(batch, tokens=(11, 12))
    assert target.scratch.capacity_rows == 3

    staged = np.zeros((2, fixture.HIDDEN), dtype=np.float32)

    def fake_hash_rows(tokens, *, sequence_ids, states, **kwargs):
        updated = dict(states)
        for slot in sequence_ids:
            updated[slot] = SimpleNamespace(next_position=1)
        return ([[0] for _ in tokens], updated)

    monkeypatch.setattr(mixed_target, "ple_hash_rows", fake_hash_rows)
    monkeypatch.setattr(mixed_target, "stage_qwen4_exp_ple_rows", lambda *a, **k: staged)
    target._stage_hashes()
    target.close()

    uploads = [nbytes for kind, buffer, nbytes in calls if buffer is target.ple_embedding]
    assert uploads == [2 * fixture.HIDDEN * 4]


def test_a_returned_result_never_aliases_the_reused_scratch(monkeypatch):
    device = _FakeDevice()
    batch = _batch(device)
    _install_fake_kernels(monkeypatch, device)

    first = _pass(batch, tokens=(11, 12))
    result = first.verify()
    snapshot = (result.token_ids.copy(), result.logits.copy(), result.hidden_seeds.copy())
    # Rewrite the borrowed device slabs: a result that aliased them would move.
    device.write(first.logits.ptr, np.full((2, fixture.VOCAB), 7.0, dtype=np.float32))
    device.write(first.residual.ptr, np.zeros((2, fixture.RESIDUAL_WIDTH), dtype=np.uint16))
    first.close()

    second = _pass(batch, tokens=(13, 14))
    assert second.scratch is first.scratch
    other = second.verify()
    second.close()

    assert np.array_equal(result.token_ids, snapshot[0])
    assert np.array_equal(result.logits, snapshot[1])
    assert np.array_equal(result.hidden_seeds, snapshot[2])
    assert tuple(int(token) for token in result.token_ids) == (11, 12)
    assert tuple(int(token) for token in other.token_ids) == (13, 14)
    assert not np.shares_memory(result.logits, other.logits)
    assert not np.shares_memory(result.hidden_seeds, other.hidden_seeds)
    assert np.allclose(result.hidden_seeds[:, 0], (11.5, 12.5))
