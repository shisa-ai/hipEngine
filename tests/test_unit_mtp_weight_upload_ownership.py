"""Explicit ownership for resident MTP weight uploads, on CPU only.

``WeightUploadCache`` is the owner the module-global ``_WEIGHT_CACHE`` never
had: one object per owner, keyed by runtime identity and name, released through
the runtime that allocated each buffer. Nothing here touches HIP --
``hipengine.core.memory.malloc`` / ``copy_host_array_to_device`` / ``free`` are
replaced with recording fakes, so the real allocation, publish, and release
order is what gets exercised.
"""

from __future__ import annotations

import dataclasses
import itertools
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.speculative import weight_uploads as weight_uploads_mod
from hipengine.speculative.weight_uploads import WeightUploadCache

_PTR_BASE = 0x7F00_0000_0000
_PTR_STRIDE = 0x1000


class _FakeRuntime:
    """Distinct runtime identity; the cache must never confuse two of them."""


class _MemorySpy:
    """Fake HIP boundary: records allocations, copies, and releases."""

    def __init__(self) -> None:
        self._ptrs = itertools.count()
        self.allocations: list[tuple[DeviceBuffer, object]] = []
        self.copies: list[tuple[DeviceBuffer, int, object]] = []
        self.frees: list[tuple[DeviceBuffer, object]] = []
        self.copy_error: BaseException | None = None
        self.free_failures: set[int] = set()
        self.fail_next_free = False

    def malloc(self, nbytes: int, *, runtime: object | None = None) -> DeviceBuffer:
        buffer = DeviceBuffer(ptr=_PTR_BASE + next(self._ptrs) * _PTR_STRIDE, nbytes=int(nbytes))
        self.allocations.append((buffer, runtime))
        return buffer

    def copy_host_array_to_device(
        self,
        buffer: DeviceBuffer,
        array: object,
        nbytes: int | None = None,
        *,
        runtime: object | None = None,
    ) -> None:
        if self.copy_error is not None:
            raise self.copy_error
        self.copies.append((buffer, int(getattr(array, "nbytes")), runtime))

    def free(self, buffer: DeviceBuffer, *, runtime: object | None = None) -> None:
        self.frees.append((buffer, runtime))
        if self.fail_next_free:
            self.fail_next_free = False
            raise RuntimeError(f"free refused for {int(buffer.ptr):#x}")
        if int(buffer.ptr) in self.free_failures:
            raise RuntimeError(f"free refused for {int(buffer.ptr):#x}")

    def allocated_ptrs(self) -> list[int]:
        return [int(buffer.ptr) for buffer, _runtime in self.allocations]

    def freed_ptrs(self) -> list[int]:
        return [int(buffer.ptr) for buffer, _runtime in self.frees]


@pytest.fixture
def memory_spy(monkeypatch) -> _MemorySpy:
    """Replace the memory helpers wherever the cache module can see them."""

    spy = _MemorySpy()
    for name in ("malloc", "copy_host_array_to_device", "free"):
        monkeypatch.setattr(f"hipengine.core.memory.{name}", getattr(spy, name))
        # Also cover a module-level ``from hipengine.core.memory import ...``.
        if hasattr(weight_uploads_mod, name):
            monkeypatch.setattr(weight_uploads_mod, name, getattr(spy, name))
    return spy


def test_same_owner_runtime_and_name_uploads_once(memory_spy: _MemorySpy) -> None:
    runtime = _FakeRuntime()
    cache = WeightUploadCache()
    array = np.arange(16, dtype=np.float32)

    first = cache.upload("blk.40.attn_q.weight", array, runtime=runtime)
    second = cache.upload("blk.40.attn_q.weight", array.copy(), runtime=runtime)

    assert first == second
    assert memory_spy.allocated_ptrs() == [int(first.ptr)]
    assert [int(buffer.ptr) for buffer, _nbytes, _rt in memory_spy.copies] == [int(first.ptr)]
    assert int(first.nbytes) == array.nbytes


def test_cache_isolates_runtime_identity_and_cache_instances(memory_spy: _MemorySpy) -> None:
    runtime_a = _FakeRuntime()
    runtime_b = _FakeRuntime()
    cache = WeightUploadCache()
    other_cache = WeightUploadCache()
    array = np.zeros(8, dtype=np.float32)

    on_a = cache.upload("output.weight", array, runtime=runtime_a)
    on_b = cache.upload("output.weight", array, runtime=runtime_b)
    on_other = other_cache.upload("output.weight", array, runtime=runtime_a)

    assert len({int(on_a.ptr), int(on_b.ptr), int(on_other.ptr)}) == 3
    assert memory_spy.allocated_ptrs() == [int(on_a.ptr), int(on_b.ptr), int(on_other.ptr)]

    cache.close()
    other_cache.close()

    # Each buffer is released through the runtime that allocated it.
    assert sorted((int(buffer.ptr), id(runtime)) for buffer, runtime in memory_spy.frees) == sorted(
        [
            (int(on_a.ptr), id(runtime_a)),
            (int(on_b.ptr), id(runtime_b)),
            (int(on_other.ptr), id(runtime_a)),
        ]
    )


def test_failed_host_copy_frees_unpublished_buffer_and_keeps_error(
    memory_spy: _MemorySpy,
) -> None:
    runtime = _FakeRuntime()
    cache = WeightUploadCache()
    array = np.ones(4, dtype=np.float32)
    copy_error = RuntimeError("host copy refused")
    memory_spy.copy_error = copy_error

    with pytest.raises(RuntimeError) as caught:
        cache.upload("blk.40.attn_k.weight", array, runtime=runtime)

    assert caught.value is copy_error
    allocated = memory_spy.allocated_ptrs()
    assert len(allocated) == 1
    # The half-built buffer is released, and never becomes a cache hit.
    assert memory_spy.freed_ptrs() == allocated
    assert memory_spy.frees[0][1] is runtime

    memory_spy.copy_error = None
    retried = cache.upload("blk.40.attn_k.weight", array, runtime=runtime)
    assert len(memory_spy.allocated_ptrs()) == 2
    assert int(retried.ptr) == memory_spy.allocated_ptrs()[-1]


def test_changed_byte_size_under_one_name_is_refused(memory_spy: _MemorySpy) -> None:
    """A name is an immutable tensor; a different size must not be served stale."""

    runtime = _FakeRuntime()
    cache = WeightUploadCache()
    small = np.zeros(4, dtype=np.float32)
    original = cache.upload("blk.40.attn_q.weight", small, runtime=runtime)

    with pytest.raises(ValueError, match="changed its byte size"):
        cache.upload("blk.40.attn_q.weight", np.zeros(8, dtype=np.float32), runtime=runtime)

    assert len(memory_spy.allocations) == 1
    assert memory_spy.frees == []
    assert cache.upload("blk.40.attn_q.weight", small, runtime=runtime) == original


def test_failed_copy_and_failed_release_keep_the_buffer_for_close(
    memory_spy: _MemorySpy,
) -> None:
    """A double failure reports the copy error and still reaches the buffer."""

    runtime = _FakeRuntime()
    cache = WeightUploadCache()
    array = np.zeros(4, dtype=np.float32)
    copy_error = RuntimeError("host copy refused")
    memory_spy.copy_error = copy_error
    memory_spy.fail_next_free = True

    with pytest.raises(RuntimeError) as caught:
        cache.upload("blk.40.attn_q.weight", array, runtime=runtime)

    assert caught.value is copy_error
    notes = "\n".join(getattr(caught.value, "__notes__", ()))
    assert "Weight upload cleanup failed" in notes
    orphaned = memory_spy.allocated_ptrs()
    assert memory_spy.freed_ptrs() == orphaned
    assert memory_spy.frees[0][1] is runtime

    # The unpublished buffer is not a cache hit: the retry allocates fresh.
    memory_spy.copy_error = None
    published = cache.upload("blk.40.attn_q.weight", array, runtime=runtime)
    assert int(published.ptr) != orphaned[0]
    assert cache.upload("blk.40.attn_q.weight", array, runtime=runtime) == published
    assert memory_spy.allocated_ptrs() == [orphaned[0], int(published.ptr)]

    # close retries the orphaned buffer as well as releasing the published one.
    cache.close()
    assert memory_spy.freed_ptrs() == [orphaned[0], int(published.ptr), orphaned[0]]


def test_close_is_idempotent_and_frees_through_allocating_runtime(
    memory_spy: _MemorySpy,
) -> None:
    runtime = _FakeRuntime()
    cache = WeightUploadCache()
    buffers = [
        cache.upload(f"blk.40.weight{index}", np.zeros(4, dtype=np.float32), runtime=runtime)
        for index in range(3)
    ]

    cache.close()

    assert sorted(memory_spy.freed_ptrs()) == sorted(int(buffer.ptr) for buffer in buffers)
    assert all(released_by is runtime for _buffer, released_by in memory_spy.frees)

    cache.close()
    assert len(memory_spy.frees) == len(buffers)


def test_close_raises_the_first_release_failure_and_retains_all_failed_entries(
    memory_spy: _MemorySpy,
) -> None:
    runtime = _FakeRuntime()
    cache = WeightUploadCache()
    released = cache.upload("blk.40.attn_v.weight", np.zeros(4, dtype=np.float32), runtime=runtime)
    stuck = cache.upload("blk.40.nextn.eh_proj.weight", np.zeros(4, dtype=np.float32), runtime=runtime)
    memory_spy.free_failures.add(int(stuck.ptr))

    with pytest.raises(RuntimeError) as caught:
        cache.close()

    assert str(caught.value) == f"free refused for {int(stuck.ptr):#x}"
    # One release raising must not abandon the other entries.
    assert sorted(memory_spy.freed_ptrs()) == sorted([int(released.ptr), int(stuck.ptr)])

    memory_spy.free_failures.clear()
    cache.close()
    # Only the retained entry is retried; the released one is not freed twice.
    assert memory_spy.freed_ptrs() == [int(released.ptr), int(stuck.ptr), int(stuck.ptr)]

    cache.close()
    assert len(memory_spy.frees) == 3


def test_close_raises_the_first_of_several_release_failures(memory_spy: _MemorySpy) -> None:
    runtime = _FakeRuntime()
    cache = WeightUploadCache()
    first = cache.upload("blk.40.attn_q.weight", np.zeros(4, dtype=np.float32), runtime=runtime)
    second = cache.upload("blk.40.attn_k.weight", np.zeros(4, dtype=np.float32), runtime=runtime)
    memory_spy.free_failures.update({int(first.ptr), int(second.ptr)})

    with pytest.raises(RuntimeError) as caught:
        cache.close()

    assert str(caught.value) == f"free refused for {int(first.ptr):#x}"
    assert memory_spy.freed_ptrs() == [int(first.ptr), int(second.ptr)]

    # Both stay reachable, in insertion order, for a later retry.
    memory_spy.free_failures.clear()
    cache.close()
    assert memory_spy.freed_ptrs() == [
        int(first.ptr),
        int(second.ptr),
        int(first.ptr),
        int(second.ptr),
    ]
    cache.close()
    assert len(memory_spy.frees) == 4


def _bare_runner(
    cache: WeightUploadCache,
    runtime: object,
    weights: dict[str, tuple[np.ndarray, object, tuple[int, ...]]] | None = None,
):
    """A draft runner with the state ``__post_init__`` would establish.

    The real ``__post_init__`` JIT-builds a dozen HIP libraries, so the fields
    the ownership paths read are set directly and the real methods run.
    """

    from hipengine.speculative.mtp_resident_draft import Qwen35GGUFResidentMTPDraftRunner

    runner = object.__new__(Qwen35GGUFResidentMTPDraftRunner)
    runner.runtime = runtime
    runner.upload_cache = cache
    runner._owns_upload_cache = False  # a provided cache is borrowed
    runner._buffers = []
    runner._native_proposal_graphs = {}
    runner.weights = weights or {}
    runner._embed_table_f32 = None
    return runner


def test_mtp_serving_assets_each_own_a_distinct_upload_cache() -> None:
    """A per-asset cache is what stops two models sharing a name-keyed entry."""

    from hipengine.generation.qwen35_gguf import _GGUFMTPServingAssets

    fields = {field.name: field for field in dataclasses.fields(_GGUFMTPServingAssets)}
    assert fields["upload_cache"].default_factory is WeightUploadCache

    def assets() -> _GGUFMTPServingAssets:
        return _GGUFMTPServingAssets(
            weights={},
            token_embd_f32=np.zeros((2, 2), dtype=np.float32),
            rope_cos=np.zeros((2, 2), dtype=np.float32),
            rope_sin=np.zeros((2, 2), dtype=np.float32),
        )

    first, second = assets(), assets()
    assert isinstance(first.upload_cache, WeightUploadCache)
    assert first.upload_cache is not second.upload_cache


def test_new_mtp_draft_runner_borrows_the_asset_upload_cache(monkeypatch) -> None:
    """The factory hands the asset's cache to the runner instead of a global."""

    from hipengine.generation import qwen35_gguf as gguf
    from hipengine.speculative import mtp_resident_draft as resident

    recorded: dict[str, object] = {}

    class _RecordingRunner:
        def __init__(self, weights, token_embd_f32, **kwargs) -> None:
            recorded["weights"] = weights
            recorded.update(kwargs)

    monkeypatch.setattr(resident, "Qwen35GGUFResidentMTPDraftRunner", _RecordingRunner)
    cache = WeightUploadCache()
    assets = gguf._GGUFMTPServingAssets(
        weights={"output.weight": (np.zeros((8, 8), dtype=np.float32), 0, (8, 8))},
        token_embd_f32=np.zeros((8, 8), dtype=np.float32),
        rope_cos=np.zeros((4, 2), dtype=np.float32),
        rope_sin=np.zeros((4, 2), dtype=np.float32),
        upload_cache=cache,
    )

    gguf._new_mtp_draft_runner(assets, runtime=SimpleNamespace())

    assert recorded["upload_cache"] is cache


def test_draft_runners_sharing_a_cache_upload_each_weight_once(
    memory_spy: _MemorySpy,
) -> None:
    """Two pooled runners over one owner reuse one device buffer, not the global."""

    from hipengine.kernels.hip_gfx1100.speculative import mtp_nextn

    runtime = _FakeRuntime()
    cache = WeightUploadCache()
    array = np.zeros(4, dtype=np.float32)
    weights = {"blk.40.attn_q.weight": (array, 0, (4,))}
    first = _bare_runner(cache, runtime, weights)
    second = _bare_runner(cache, runtime, weights)

    uploaded = first._upload("blk.40.attn_q.weight")
    assert second._upload("blk.40.attn_q.weight") == uploaded
    assert len(memory_spy.allocations) == 1
    # The module-global cache is not the owner for resident runners.
    assert mtp_nextn._WEIGHT_CACHE == {}

    cache.close()
    assert memory_spy.freed_ptrs() == [int(uploaded.ptr)]


def test_ensure_embed_table_uploads_into_the_runner_cache(
    memory_spy: _MemorySpy,
) -> None:
    runtime = _FakeRuntime()
    cache = WeightUploadCache()
    runner = _bare_runner(cache, runtime)
    runner.vocab = 4
    runner.hidden_size = 8
    runner.token_embd_f32 = np.zeros((16, 8), dtype=np.float32)

    runner._ensure_embed_table()
    table = runner._embed_table_f32
    runner._ensure_embed_table()

    assert len(memory_spy.allocations) == 1
    assert int(table.nbytes) == 4 * 8 * 4
    cache.close()
    assert memory_spy.freed_ptrs() == [int(table.ptr)]


def test_draft_runner_close_releases_only_the_cache_it_owns(
    memory_spy: _MemorySpy,
) -> None:
    runtime = _FakeRuntime()
    array = np.zeros(4, dtype=np.float32)

    borrowed = WeightUploadCache()
    borrowed_buffer = borrowed.upload("blk.40.attn_q.weight", array, runtime=runtime)
    _bare_runner(borrowed, runtime).close()
    assert memory_spy.frees == []
    assert borrowed.upload("blk.40.attn_q.weight", array, runtime=runtime) == borrowed_buffer

    owned_cache = WeightUploadCache()
    owned_buffer = owned_cache.upload("blk.40.attn_q.weight", array, runtime=runtime)
    standalone = _bare_runner(owned_cache, runtime)
    standalone._owns_upload_cache = True  # what __post_init__ sets when it makes the cache
    standalone.close()
    assert memory_spy.freed_ptrs() == [int(owned_buffer.ptr)]


def test_generator_close_closes_draft_runners_before_asset_uploads() -> None:
    """A raising draft close must not skip the asset upload release."""

    from hipengine.generation.qwen35_gguf import (
        Qwen35GGUFBringupGenerator,
        _GGUFMTPServingAssets,
    )

    log: list[str] = []
    draft_error = RuntimeError("draft close refused")

    class _RecordingDraft:
        def close(self) -> None:
            log.append("draft")
            raise draft_error

    class _RecordingCache:
        def close(self) -> None:
            log.append("assets")

    assets = _GGUFMTPServingAssets(
        weights={},
        token_embd_f32=np.zeros((2, 2), dtype=np.float32),
        rope_cos=np.zeros((2, 2), dtype=np.float32),
        rope_sin=np.zeros((2, 2), dtype=np.float32),
        upload_cache=_RecordingCache(),
    )
    generator = object.__new__(Qwen35GGUFBringupGenerator)
    generator._closed = False
    generator._shared_session_pool = {}
    generator._shared_mtp_draft_pool = {7: [_RecordingDraft()]}
    generator._shared_mtp_nextn_weights = {}
    generator._shared_runner = None
    generator._shared_runner_lock = threading.Lock()
    generator._mtp_serving_assets = assets

    with pytest.raises(RuntimeError) as caught:
        generator.close()

    assert caught.value is draft_error
    assert log == ["draft", "assets"]
    assert generator._mtp_serving_assets is None
    generator.close()
    assert log == ["draft", "assets"]


_BUILDERS = (
    "build_mtp_nextn",
    "build_gguf_k_gemv",
    "build_gguf_q4_k_gemv",
    "build_gguf_q6_k_pack8_gemv",
    "build_gguf_q8_0_dp4a_gemv",
    "build_cast",
    "build_qwen35_router",
    "build_lm_head",
    "build_paro_silu",
    "build_paro_combine",
    "build_gather",
)


def _resident_mtp_weights() -> dict[str, tuple[np.ndarray, object, tuple[int, ...]]]:
    """Synthetic NextN weights: only the shapes and qtypes ``__post_init__`` reads."""

    from hipengine.quant.gguf import GGMLQuantizationType as QuantType

    def entry(shape: tuple[int, ...], qtype: QuantType):
        array = np.zeros(shape, dtype=np.float32)
        return array, qtype, shape

    return {
        "blk.40.nextn.eh_proj.weight": entry((4, 8), QuantType.Q8_0),
        "blk.40.nextn.hnorm.weight": entry((4,), QuantType.F32),
        "blk.40.nextn.enorm.weight": entry((4,), QuantType.F32),
        "blk.40.nextn.shared_head_norm.weight": entry((4,), QuantType.F32),
        "blk.40.attn_norm.weight": entry((4,), QuantType.F32),
        "blk.40.post_attention_norm.weight": entry((4,), QuantType.F32),
        "blk.40.attn_q.weight": entry((8, 4), QuantType.Q8_0),
        "blk.40.attn_k.weight": entry((2, 4), QuantType.Q8_0),
        "blk.40.attn_v.weight": entry((2, 4), QuantType.Q8_0),
        "blk.40.attn_output.weight": entry((4, 8), QuantType.Q8_0),
        "blk.40.attn_q_norm.weight": entry((8,), QuantType.F32),
        "blk.40.attn_k_norm.weight": entry((8,), QuantType.F32),
        "blk.40.ffn_gate_inp.weight": entry((2, 4), QuantType.F32),
        "blk.40.ffn_gate_inp_shexp.weight": entry((4,), QuantType.F32),
        "blk.40.ffn_gate_exps.weight": entry((2, 6), QuantType.Q4_K),
        "blk.40.ffn_up_exps.weight": entry((2, 6), QuantType.Q4_K),
        "blk.40.ffn_down_exps.weight": entry((2, 4), QuantType.Q5_K),
        "blk.40.ffn_gate_shexp.weight": entry((4, 4), QuantType.Q8_0),
        "blk.40.ffn_up_shexp.weight": entry((4, 4), QuantType.Q8_0),
        "blk.40.ffn_down_shexp.weight": entry((4, 4), QuantType.Q8_0),
        "output.weight": entry((16, 4), QuantType.Q6_K),
    }


def _fake_post_init_boundaries(monkeypatch) -> list[str]:
    """Fake the JIT builds and scratch allocation; keep metadata and uploads real."""

    from hipengine.speculative import mtp_resident_draft as resident

    built: list[str] = []

    def builder(name: str):
        def build(**kwargs):
            built.append(name)
            return SimpleNamespace(name=name, kwargs=kwargs)

        return build

    for name in _BUILDERS:
        monkeypatch.setattr(resident, name, builder(name))
    monkeypatch.setattr(
        resident.Qwen35GGUFResidentMTPDraftRunner, "_allocate_buffers", lambda self: None
    )
    return built


def test_post_init_standalone_runner_owns_the_cache_it_creates(
    memory_spy: _MemorySpy, monkeypatch
) -> None:
    from hipengine.speculative.mtp_resident_draft import Qwen35GGUFResidentMTPDraftRunner

    built = _fake_post_init_boundaries(monkeypatch)
    runtime = _FakeRuntime()
    runner = Qwen35GGUFResidentMTPDraftRunner(
        _resident_mtp_weights(),
        np.zeros((16, 4), dtype=np.float32),
        runtime=runtime,
        vocab_cap=16,
    )

    assert built
    assert runner._owns_upload_cache is True
    assert isinstance(runner.upload_cache, WeightUploadCache)
    assert len(memory_spy.allocations) == len(_resident_mtp_weights())
    assert all(owner is runtime for _buffer, owner in memory_spy.allocations)

    # The cache __post_init__ created is the one _upload reads.
    assert runner._upload("blk.40.attn_q.weight") == runner.wq
    assert len(memory_spy.allocations) == len(_resident_mtp_weights())

    runner.close()
    assert sorted(memory_spy.freed_ptrs()) == sorted(memory_spy.allocated_ptrs())
    assert all(owner is runtime for _buffer, owner in memory_spy.frees)


@pytest.mark.parametrize("retired", [None, "0", "false", "1", "true"])
def test_post_init_ignores_retired_shared_dual_selector(
    memory_spy: _MemorySpy, monkeypatch, retired: str | None,
) -> None:
    from hipengine.speculative import mtp_resident_draft as resident

    _fake_post_init_boundaries(monkeypatch)
    if retired is None:
        monkeypatch.delenv("HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL", retired)
    read_flag = resident._env_flag

    def checked_flag(name, default):
        assert name != "HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL"
        return read_flag(name, default)

    monkeypatch.setattr(resident, "_env_flag", checked_flag)
    runner = resident.Qwen35GGUFResidentMTPDraftRunner(
        _resident_mtp_weights(), np.zeros((16, 4), dtype=np.float32),
        runtime=_FakeRuntime(), vocab_cap=16,
    )
    try:
        assert not hasattr(runner, "_q8_shared_dual_enabled")
    finally:
        runner.close()
    assert sorted(memory_spy.freed_ptrs()) == sorted(memory_spy.allocated_ptrs())


def test_post_init_borrowed_cache_is_used_and_left_open(
    memory_spy: _MemorySpy, monkeypatch
) -> None:
    from hipengine.speculative.mtp_resident_draft import Qwen35GGUFResidentMTPDraftRunner

    _fake_post_init_boundaries(monkeypatch)
    runtime = _FakeRuntime()
    cache = WeightUploadCache()
    runner = Qwen35GGUFResidentMTPDraftRunner(
        _resident_mtp_weights(),
        np.zeros((16, 4), dtype=np.float32),
        runtime=runtime,
        vocab_cap=16,
        upload_cache=cache,
    )

    assert runner.upload_cache is cache
    assert runner._owns_upload_cache is False
    assert len(memory_spy.allocations) == len(_resident_mtp_weights())

    runner.close()
    assert memory_spy.frees == []

    cache.close()
    assert sorted(memory_spy.freed_ptrs()) == sorted(memory_spy.allocated_ptrs())


@pytest.mark.parametrize("borrowed", [False, True])
def test_post_init_allocation_failure_releases_only_owned_uploads(
    memory_spy: _MemorySpy, monkeypatch, borrowed: bool
) -> None:
    from hipengine.speculative.mtp_resident_draft import Qwen35GGUFResidentMTPDraftRunner

    _fake_post_init_boundaries(monkeypatch)
    failure = RuntimeError("scratch allocation failed")

    def fail_allocation(self):
        raise failure

    monkeypatch.setattr(Qwen35GGUFResidentMTPDraftRunner, "_allocate_buffers", fail_allocation)
    cache = WeightUploadCache() if borrowed else None
    with pytest.raises(RuntimeError) as caught:
        Qwen35GGUFResidentMTPDraftRunner(
            _resident_mtp_weights(),
            np.zeros((16, 4), dtype=np.float32),
            runtime=_FakeRuntime(),
            vocab_cap=16,
            upload_cache=cache,
        )
    assert caught.value is failure
    assert len(memory_spy.allocations) == len(_resident_mtp_weights())
    if borrowed:
        assert memory_spy.frees == []
        cache.close()
    assert sorted(memory_spy.freed_ptrs()) == sorted(memory_spy.allocated_ptrs())
