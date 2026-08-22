from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hipengine.generation.engine_loop import EngineLoopConfig
from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
from hipengine.generation.registry import GenerationRequest
from hipengine.dispatch import WorkItem, WorkKind
from hipengine.kvcache import DeviceChunkedKVPool


class _FakePrefixSnapshot:
    def __init__(self, *, source_slot_id: int, position: int, block_ids: tuple[int, ...]) -> None:
        self.source_slot_id = int(source_slot_id)
        self.position = int(position)
        self.block_ids = tuple(int(block_id) for block_id in block_ids)
        self.nbytes = 384
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakePrefixSession:
    def __init__(self, slot_id: int) -> None:
        self.slot_id = int(slot_id)
        self.scratch = SimpleNamespace(max_positions=1024)
        self.position = 0
        self.allocation = None
        self.pool = None
        self.prefill_calls: list[tuple[tuple[int, ...], int, int]] = []
        self.step_calls: list[tuple[int, int, int]] = []
        self.clone_calls: list[tuple[int, int]] = []
        self.snapshot_capture_calls: list[int] = []
        self.snapshot_clone_calls: list[tuple[int, int]] = []
        self.snapshots: list[_FakePrefixSnapshot] = []

    def create_device_kv_pool(self, **config):
        return DeviceChunkedKVPool(
            page_bytes=4096,
            initial_pages=int(config["initial_pages"]),
            low_water_pages=int(config["low_water_pages"]),
            high_water_pages=(
                None
                if config["high_water_pages"] is None
                else int(config["high_water_pages"])
            ),
            chunk_pages=int(config["chunk_pages"]),
            idle_grace_seconds=float(config["idle_grace_seconds"]),
            allocate_chunk=lambda start, pages: {
                "ptr": 0xA0000000 + int(start) * 4096,
                "pages": int(pages),
            },
            free_chunk=lambda backing: None,
            page_pointer=lambda backing, local_page: int(backing["ptr"]) + int(local_page) * 4096,
        )

    def bind_device_kv_allocation(self, pool, allocation) -> None:
        assert self.allocation is None
        self.pool = pool
        self.allocation = allocation

    def clone_prefix_state_from(self, source, *, position: int, stream: int = 0) -> int:
        assert stream == 0
        assert int(position) == int(source.position)
        assert self.allocation is not None
        assert source.allocation is not None
        assert self.allocation.reused_block_ids == source.allocation.block_ids[:1]
        self.position = int(source.position)
        self.clone_calls.append((int(source.slot_id), int(source.position)))
        return 384

    def capture_prefix_state_snapshot(self, *, position: int | None = None):
        boundary = int(self.position if position is None else position)
        assert boundary == self.position
        assert boundary > 0 and boundary % 256 == 0
        assert self.allocation is not None
        snapshot = _FakePrefixSnapshot(
            source_slot_id=self.slot_id,
            position=boundary,
            block_ids=tuple(self.allocation.block_ids[: boundary // 256]),
        )
        self.snapshot_capture_calls.append(boundary)
        self.snapshots.append(snapshot)
        return snapshot

    def clone_prefix_state_from_snapshot(self, snapshot, *, stream: int = 0) -> int:
        assert stream == 0
        assert not snapshot.closed
        assert self.allocation is not None
        assert self.allocation.reused_block_ids == snapshot.block_ids
        self.position = int(snapshot.position)
        self.snapshot_clone_calls.append((int(snapshot.source_slot_id), int(snapshot.position)))
        return int(snapshot.nbytes)

    @staticmethod
    def _result(*, return_logits: bool):
        logits = None
        if return_logits:
            logits = np.full((1, 1024), -100.0, dtype=np.float32)
            logits[0, 777] = 10.0
        return SimpleNamespace(token_id=777, logits=logits)

    def prefill(self, token_ids, *, return_logits: bool):
        assert self.position == 0
        prompt = tuple(int(token) for token in token_ids)
        start = int(self.position)
        self.position += len(prompt)
        self.prefill_calls.append((prompt, start, int(self.position)))
        return self._result(return_logits=return_logits)

    def prefill_batch_native(self, prompt_token_ids, *, sessions, **kwargs):
        assert sessions == [self]
        prompt = tuple(int(token) for token in prompt_token_ids[0])
        start = int(self.position)
        self.position += len(prompt)
        self.prefill_calls.append((prompt, start, int(self.position)))
        return [self._result(return_logits=bool(kwargs.get("return_logits", False)))]

    def step(self, token_id: int, *, return_logits: bool):
        start = int(self.position)
        self.position += 1
        self.step_calls.append((int(token_id), start, int(self.position)))
        return self._result(return_logits=return_logits)

    def invalidate_device_kv_graphs(self) -> int:
        return 0

    def unbind_device_kv_allocation(self):
        allocation = self.allocation
        assert allocation is not None
        self.allocation = None
        self.pool = None
        return allocation

    def reset(self) -> None:
        self.position = 0

    def close(self) -> None:
        pass


class _FakePrefixOwner:
    backend = "hip_gfx1151"
    target_arch = "gfx1151"
    _prepared_max_sequence_length = 1024
    tokenizer = SimpleNamespace(eos_token_id=None, decode=lambda tokens: "".join(str(token) for token in tokens))

    def __init__(self) -> None:
        self.sessions = [_FakePrefixSession(index) for index in range(3)]

    def _get_shared_runner(self):
        return SimpleNamespace(runtime=SimpleNamespace(mem_get_info=lambda: (100, 200)))

    def _acquire_shared_session(self, shared_runner, **kwargs):
        del shared_runner, kwargs
        session = self.sessions.pop(0)
        return session, ("continuous_ar_dynamic_kv", True, True, 1024), False

    def _release_shared_session(self, key, session) -> None:
        del key
        self.sessions.append(session)

    def _flush_ar_packed_decode_owners(self, slots) -> None:
        del slots


def _request(
    prompt: tuple[int, ...],
    *,
    max_tokens: int,
    temperature: float = 0.0,
    forced_token_id: int | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        prompts=(prompt,),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        ignore_eos=True,
        forced_tokens_pending=(
            () if forced_token_id is None else (int(forced_token_id),)
        ),
        forced_token_reason=(
            None if forced_token_id is None else "tool_choice_required"
        ),
    )


def test_resident_runner_reuses_exact_current_prefix_and_reclaims_source_first() -> None:
    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=3)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=3,
            kv_pool_initial_pages=6,
            kv_pool_low_water_pages=6,
            kv_pool_high_water_pages=6,
            kv_pool_chunk_pages=6,
            prefix_cache="radix",
        )
    )
    prefix = tuple(range(1, 257))
    source_request = _request(prefix, max_tokens=3)
    runner.register_batch((1,), source_request, prompt_rows=(prefix,))
    runner.reserve_admission(SimpleNamespace(request_id=1))
    source_row = runner._rows[1]
    assert source_row.lease is not None
    source_row.prefill_tokens_seen = len(prefix)
    source_row.lease.session.position = len(prefix)
    runner._refresh_prefix_cache(source_row)

    continued_prompt = (*prefix, 999)
    continued_request = _request(continued_prompt, max_tokens=2)
    runner.register_batch((2,), continued_request, prompt_rows=(continued_prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=2))
    continued_row = runner._rows[2]
    assert continued_row.lease is not None
    continued_session = continued_row.lease.session

    assert continued_row.prefix_reused_tokens == 256
    assert continued_row.prefix_source_request_id == 1
    assert continued_row.prefix_state_clone_bytes == 384
    assert continued_row.kv_allocation.reused_block_ids == source_row.kv_allocation.block_ids[:1]
    assert continued_session.clone_calls == [(source_row.lease.session.slot_id, 256)]
    assert runner.kv_pool.refcount(source_row.kv_allocation.block_ids[0]) == 2
    assert runner._prefix_request_telemetry(continued_row) == {
        "mode": "radix",
        "block_size_tokens": 256,
        "eligible": True,
        "lookup": True,
        "hit": True,
        "source": "active_current",
        "matched_tokens": 256,
        "reused_tokens": 256,
        "avoided_prefill_tokens": 256,
        "executed_prefill_tokens": 1,
        "reused_pages": 1,
        "reused_page_bytes": 4096,
        "state_clone_bytes": 384,
        "snapshot_hit": False,
        "admission_fallback": False,
        "fallback_reason": None,
        "cache_resident_entries": 1,
        "cache_resident_pages": 0,
        "cache_resident_bytes": 384,
    }

    runner.prefill_batch(
        WorkItem(
            kind=WorkKind.PREFILL,
            request_ids=(2,),
            row_to_request=(2,),
            token_rows=(prefix,),
        ),
        commit=True,
    )
    assert continued_session.prefill_calls == []
    runner.prefill_batch(
        WorkItem(
            kind=WorkKind.PREFILL,
            request_ids=(2,),
            row_to_request=(2,),
            token_rows=((999,),),
        ),
        commit=True,
    )
    assert continued_session.prefill_calls == []
    assert continued_session.step_calls == [(999, 256, 257)]
    assert continued_row.slot is not None
    assert continued_row.slot.generated_ids == [777]

    shared_block = source_row.kv_allocation.block_ids[0]
    runner.rollback_admission(SimpleNamespace(request_id=1))
    assert runner.kv_pool.refcount(shared_block) == 1
    assert continued_row.lease is not None
    assert continued_row.lease.session.position == 257

    snapshot = runner.observability_snapshot()["prefix_cache"]
    assert snapshot["mode"] == "radix"
    assert snapshot["usable_hits"] == 1
    assert snapshot["reused_tokens"] == 256
    assert snapshot["state_clone_bytes"] == 384

    runner.rollback_admission(SimpleNamespace(request_id=2))
    assert runner.kv_pool.refcount(shared_block) == 0
    assert runner.kv_pool.stats.refcounted_pages == 0
    assert runner.available_session_count == 3
    runner.close()


def test_processed_argmax_reuses_completed_prefix_with_suffix_only_prefill() -> None:
    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=3)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=3,
            kv_pool_initial_pages=9,
            kv_pool_low_water_pages=9,
            kv_pool_high_water_pages=9,
            kv_pool_chunk_pages=9,
            prefix_cache="radix",
        )
    )
    prefix = tuple(range(1, 257))
    source_request = _request(prefix, max_tokens=2, forced_token_id=811)
    runner.register_batch((10,), source_request, prompt_rows=(prefix,))
    runner.reserve_admission(SimpleNamespace(request_id=10))
    source = runner._rows[10]
    assert source.lease is not None
    source.prefill_tokens_seen = len(prefix)
    source.lease.session.position = len(prefix)
    assert runner._refresh_prefix_cache(source) is True
    source_snapshot = source.lease.session.snapshots[-1]
    shared_block = source.kv_allocation.block_ids[0]

    runner._release_row_resources(source, retain_prefix_snapshots=True)
    runner._rows.pop(10)
    assert runner.kv_pool.refcount(shared_block) == 1

    continued_prompt = (*prefix, 999)
    continued_request = _request(
        continued_prompt,
        max_tokens=2,
        forced_token_id=812,
    )
    runner.register_batch((11,), continued_request, prompt_rows=(continued_prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=11))
    continued = runner._rows[11]
    assert continued.lease is not None
    session = continued.lease.session
    assert continued.sampler_plan.mode.value == "processed_argmax"
    assert continued.prefix_reused_tokens == 256
    assert continued.prefix_snapshot_hit is True
    assert session.snapshot_clone_calls == [(source_snapshot.source_slot_id, 256)]

    runner.prefill_batch(
        WorkItem(
            kind=WorkKind.PREFILL,
            request_ids=(11,),
            row_to_request=(11,),
            token_rows=(prefix,),
        ),
        commit=True,
    )
    runner.prefill_batch(
        WorkItem(
            kind=WorkKind.PREFILL,
            request_ids=(11,),
            row_to_request=(11,),
            token_rows=((999,),),
        ),
        commit=True,
    )

    assert session.prefill_calls == []
    assert session.step_calls == [(999, 256, 257)]
    assert continued.slot is not None
    assert continued.slot.generated_ids == [812]
    assert continued.sampling_state is not None
    assert continued.sampling_state.generated_tokens == [812]
    assert continued.full_vocab_logits_d2h is True
    assert continued.logits_d2h_bytes == 4096
    telemetry = runner._prefix_request_telemetry(continued)
    assert telemetry["eligible"] is True
    assert telemetry["lookup"] is True
    assert telemetry["hit"] is True
    assert telemetry["source"] == "completed_snapshot"
    assert telemetry["reused_tokens"] == 256
    assert telemetry["executed_prefill_tokens"] == 1
    assert telemetry["fallback_reason"] is None

    runner.rollback_admission(SimpleNamespace(request_id=11))
    assert runner.kv_pool.refcount(shared_block) == 1
    assert runner._evict_prefix_snapshot(prefix) is True
    assert runner.kv_pool.refcount(shared_block) == 0
    runner.close()


def test_processed_argmax_radix_miss_captures_aligned_boundaries() -> None:
    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=3)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=3,
            kv_pool_initial_pages=9,
            kv_pool_low_water_pages=9,
            kv_pool_high_water_pages=9,
            kv_pool_chunk_pages=9,
            prefix_cache="radix",
        )
    )
    prompt = tuple(range(1, 514))
    request = _request(prompt, max_tokens=2, forced_token_id=811)
    runner.register_batch((12,), request, prompt_rows=(prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=12))
    row = runner._rows[12]
    assert row.prefix_eligible is True
    assert row.prefix_lookup is True
    assert row.prefix_fallback_reason == "miss"

    runner.prefill_batch(
        WorkItem(
            kind=WorkKind.PREFILL,
            request_ids=(12,),
            row_to_request=(12,),
            token_rows=(prompt,),
        ),
        commit=True,
    )

    assert row.lease is not None
    session = row.lease.session
    assert session.prefill_calls == [(prompt[:512], 0, 512)]
    assert session.step_calls == [(513, 512, 513)]
    assert session.snapshot_capture_calls == [512]
    assert session.snapshots[0].closed is False
    assert row.slot is not None
    assert row.slot.generated_ids == [811]
    assert runner._prefix_cache is not None
    match = runner._prefix_cache.match(prompt)
    assert match.hit is True
    assert match.matched_token_count == 512
    telemetry = runner._prefix_request_telemetry(row)
    assert telemetry["eligible"] is True
    assert telemetry["lookup"] is True
    assert telemetry["hit"] is False
    assert telemetry["matched_tokens"] == 0
    assert telemetry["executed_prefill_tokens"] == 513
    assert telemetry["fallback_reason"] == "miss"

    runner._release_row_resources(row, retain_prefix_snapshots=True)
    runner._rows.pop(12)
    assert runner._evict_prefix_snapshot(prompt[:512]) is True
    assert runner.kv_pool.stats.refcounted_pages == 0
    runner.close()


def test_resident_runner_reuses_completed_prefix_snapshot_and_evicts_cleanly() -> None:
    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=3)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=3,
            kv_pool_initial_pages=6,
            kv_pool_low_water_pages=6,
            kv_pool_high_water_pages=6,
            kv_pool_chunk_pages=6,
            prefix_cache="radix",
        )
    )
    prefix = tuple(range(1, 257))
    source_request = _request(prefix, max_tokens=3)
    runner.register_batch((1,), source_request, prompt_rows=(prefix,))
    runner.reserve_admission(SimpleNamespace(request_id=1))
    source_row = runner._rows[1]
    assert source_row.lease is not None
    source_session = source_row.lease.session
    source_row.prefill_tokens_seen = len(prefix)
    source_session.position = len(prefix)
    assert runner._refresh_prefix_cache(source_row) is True
    snapshot = source_session.snapshots[-1]
    shared_block = source_row.kv_allocation.block_ids[0]

    runner._release_row_resources(source_row, retain_prefix_snapshots=True)
    runner._rows.pop(1)
    assert snapshot.closed is False
    assert runner.kv_pool.refcount(shared_block) == 1
    assert runner.kv_pool.stats.refcounted_pages == 1

    continued_prompt = (*prefix, 999)
    continued_request = _request(continued_prompt, max_tokens=2)
    runner.register_batch((2,), continued_request, prompt_rows=(continued_prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=2))
    continued_row = runner._rows[2]
    assert continued_row.lease is not None
    assert continued_row.prefix_reused_tokens == 256
    assert continued_row.prefix_source_request_id is None
    assert continued_row.prefix_snapshot_hit is True
    assert continued_row.lease.session.snapshot_clone_calls == [(source_session.slot_id, 256)]
    assert runner.kv_pool.refcount(shared_block) == 2
    assert runner._prefix_request_telemetry(continued_row) == {
        "mode": "radix",
        "block_size_tokens": 256,
        "eligible": True,
        "lookup": True,
        "hit": True,
        "source": "completed_snapshot",
        "matched_tokens": 256,
        "reused_tokens": 256,
        "avoided_prefill_tokens": 256,
        "executed_prefill_tokens": 1,
        "reused_pages": 1,
        "reused_page_bytes": 4096,
        "state_clone_bytes": 384,
        "snapshot_hit": True,
        "admission_fallback": False,
        "fallback_reason": None,
        "cache_resident_entries": 1,
        "cache_resident_pages": 1,
        "cache_resident_bytes": 4480,
    }

    runner.rollback_admission(SimpleNamespace(request_id=2))
    assert runner.kv_pool.refcount(shared_block) == 1
    prefix_observability = runner.observability_snapshot()["prefix_cache"]
    assert prefix_observability["snapshot_entries"] == 1
    assert prefix_observability["snapshot_hits"] == 1
    assert prefix_observability["snapshot_bytes"] == 384

    assert runner._evict_prefix_snapshot(prefix) is True
    assert snapshot.closed is True
    assert runner.kv_pool.refcount(shared_block) == 0
    assert runner.kv_pool.stats.refcounted_pages == 0
    runner.close()


def test_completed_prefix_survives_unaligned_tail_and_lru_residency_is_bounded() -> None:
    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=1)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=1,
            kv_pool_initial_pages=4,
            kv_pool_low_water_pages=4,
            kv_pool_high_water_pages=4,
            kv_pool_chunk_pages=4,
            prefix_cache="radix",
        )
    )
    prompt = tuple(range(1, 514))
    source_request = _request(prompt, max_tokens=2)
    runner.register_batch((10,), source_request, prompt_rows=(prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=10))
    row = runner._rows[10]
    assert row.lease is not None
    session = row.lease.session
    row.prefill_tokens_seen = len(prompt)

    session.position = 256
    assert runner._refresh_prefix_cache(row) is True
    first_snapshot = session.snapshots[-1]
    session.position = 512
    assert runner._refresh_prefix_cache(row) is True
    second_snapshot = session.snapshots[-1]
    assert first_snapshot.closed is True
    assert second_snapshot.closed is False

    session.position = 513
    assert runner._refresh_prefix_cache(row) is False
    assert runner._prefix_cache is not None
    assert runner._prefix_cache.match(prompt).matched_token_count == 512

    runner._release_row_resources(row, retain_prefix_snapshots=True)
    runner._rows.pop(10)
    cache = runner.observability_snapshot()["prefix_cache"]
    assert cache["snapshot_limit"] == 1
    assert cache["snapshot_entries"] == 1
    assert cache["retained_snapshot_entries"] == 1
    assert cache["retained_kv_pages"] == 2
    assert cache["retained_kv_bytes"] == 8192
    assert cache["snapshot_bytes"] == 384
    assert cache["resident_bytes"] == 8576
    assert runner.kv_pool.stats.refcounted_pages == 2

    assert runner._evict_prefix_snapshot(prompt[:512]) is True
    assert second_snapshot.closed is True
    assert runner.kv_pool.stats.refcounted_pages == 0
    runner.close()


def test_prefix_reuse_falls_back_for_exact_prompt_and_sampled_boundary() -> None:
    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=3)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=3,
            kv_pool_initial_pages=9,
            kv_pool_low_water_pages=9,
            kv_pool_high_water_pages=9,
            kv_pool_chunk_pages=9,
            prefix_cache="radix",
        )
    )
    prefix = tuple(range(1, 513))
    source_request = _request(prefix, max_tokens=2)
    runner.register_batch((20,), source_request, prompt_rows=(prefix,))
    runner.reserve_admission(SimpleNamespace(request_id=20))
    source = runner._rows[20]
    assert source.lease is not None
    source.prefill_tokens_seen = len(prefix)
    source.lease.session.position = len(prefix)
    assert runner._refresh_prefix_cache(source) is True

    runner.register_batch((21,), source_request, prompt_rows=(prefix,))
    runner.reserve_admission(SimpleNamespace(request_id=21))
    exact = runner._rows[21]
    assert exact.kv_allocation.reused_block_ids == ()
    assert exact.prefix_lookup is True
    assert exact.prefix_matched_tokens == 512
    assert exact.prefix_fallback_reason == "full_prompt_boundary_requires_suffix"
    assert runner._prefix_request_telemetry(exact) == {
        "mode": "radix",
        "block_size_tokens": 256,
        "eligible": True,
        "lookup": True,
        "hit": False,
        "source": None,
        "matched_tokens": 512,
        "reused_tokens": 0,
        "avoided_prefill_tokens": 0,
        "executed_prefill_tokens": 512,
        "reused_pages": 0,
        "reused_page_bytes": 0,
        "state_clone_bytes": 0,
        "snapshot_hit": False,
        "admission_fallback": False,
        "fallback_reason": "full_prompt_boundary_requires_suffix",
        "cache_resident_entries": 1,
        "cache_resident_pages": 0,
        "cache_resident_bytes": 384,
    }
    runner.rollback_admission(SimpleNamespace(request_id=21))

    sampled_prompt = (*prefix, 999)
    sampled_request = _request(sampled_prompt, max_tokens=2, temperature=0.7)
    runner.register_batch((22,), sampled_request, prompt_rows=(sampled_prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=22))
    sampled = runner._rows[22]
    assert sampled.kv_allocation.reused_block_ids == ()
    assert sampled.prefix_lookup is False
    assert sampled.prefix_eligible is False
    assert sampled.prefix_fallback_reason == "sampling_unsupported"
    assert runner._prefix_request_telemetry(sampled) == {
        "mode": "radix",
        "block_size_tokens": 256,
        "eligible": False,
        "lookup": False,
        "hit": False,
        "source": None,
        "matched_tokens": 0,
        "reused_tokens": 0,
        "avoided_prefill_tokens": 0,
        "executed_prefill_tokens": 513,
        "reused_pages": 0,
        "reused_page_bytes": 0,
        "state_clone_bytes": 0,
        "snapshot_hit": False,
        "admission_fallback": False,
        "fallback_reason": "sampling_unsupported",
        "cache_resident_entries": 1,
        "cache_resident_pages": 0,
        "cache_resident_bytes": 384,
    }
    runner.rollback_admission(SimpleNamespace(request_id=22))

    runner.rollback_admission(SimpleNamespace(request_id=20))
    assert runner.kv_pool.stats.refcounted_pages == 0
    runner.close()


class _FakeGlobalPoolSession:
    """Fake resident session exposing the global-pool factory ABI."""

    kv_attention_source = None
    defer_kv_allocation = True

    def __init__(self, slot_id: int) -> None:
        self.slot_id = int(slot_id)
        # 768-token scratch: 3 pages per request, below the packed workspace's
        # 1024-token (4-page) per-slot union floor.
        self.scratch = SimpleNamespace(max_positions=768)
        self.created_pools = []
        self.bound_workspace_pools = []
        self.workspace_release_calls = 0
        self.closed = False
        self._reset_current_slot_only = False

    def resident_slot_view(self, index: int):
        return _FakeGlobalPoolSession(index)

    def create_global_device_kv_pool(self, *, page_capacity, generation):
        from hipengine.kvcache.device_global import GlobalDeviceKVPool

        pool = GlobalDeviceKVPool(
            page_bytes=4096,
            backend_fingerprint="test",
            generation=int(generation),
            backing=None,
            plane_page_pointers={
                "payload": tuple(0x10000 * (index + 1) for index in range(int(page_capacity)))
            },
            pointer_table_pointers={"payload": 0xF0000},
            metadata_descriptor_pointer=0xF1000,
            close_storage=lambda: None,
        )
        self.created_pools.append(pool)
        return pool

    def bind_workspace_kv_pool(self, pool) -> None:
        self.bound_workspace_pools.append(pool)

    def release_idle_packed_workspace(self) -> int:
        self.workspace_release_calls += 1
        return 0

    def close(self) -> None:
        self.closed = True


class _FakeGlobalPoolOwner:
    backend = "hip_gfx1100"
    target_arch = "gfx1100"
    _prepared_max_sequence_length = 1024
    _defer_resident_session_policy_resolution = True
    tokenizer = SimpleNamespace(eos_token_id=None, decode=lambda tokens: "")

    def __init__(self) -> None:
        self.sessions: list[_FakeGlobalPoolSession] = []

    def _get_shared_runner(self):
        return SimpleNamespace(runtime=SimpleNamespace(mem_get_info=lambda: (100, 200)))

    def _acquire_shared_session(self, shared_runner, **kwargs):
        del shared_runner
        session = _FakeGlobalPoolSession(int(kwargs.get("max_batch_size", 0)))
        self.sessions.append(session)
        return session, ("continuous_ar_dynamic_kv", True, True, 1024), False

    def _release_shared_session(self, key, session) -> None:
        del key, session

    def _flush_ar_packed_decode_owners(self, slots) -> None:
        del slots


def test_configure_engine_loop_leases_packed_workspace_pages() -> None:
    from hipengine.runtime.qwen35_gguf_runner import _GGUF_PACKED_WORKSPACE_LEASE_KEY

    owner = _FakeGlobalPoolOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=2)
    config = EngineLoopConfig(
        max_active_requests=2,
        kv_pool_initial_pages=8,
        kv_pool_low_water_pages=8,
        kv_pool_chunk_pages=8,
        prefix_cache="off",
    )
    runner._reserve_sessions()
    runner.configure_engine_loop(config)

    pool = runner.kv_pool
    batch_owner = owner.sessions[0]
    # capacity=2 requests * 3 pages/request = 6 request pages; the packed
    # workspace lease adds max(8, capacity) * max(3, 1024/256) = 32 pinned
    # pages on top (the union floor is 1024 tokens per slot even when the
    # request context is shorter).
    assert pool.current_pages == 38
    lease = pool.workspace_pages(_GGUF_PACKED_WORKSPACE_LEASE_KEY)
    assert lease is not None and len(lease) == 32
    assert pool.stats.free_pages == 6
    assert pool.stats.pinned_pages == 32
    assert batch_owner.bound_workspace_pools == [pool]

    # Reconfiguration releases the lease and the idle workspace before the
    # old pool closes, then re-leases on the fresh pool.
    runner.configure_engine_loop(config)
    new_pool = runner.kv_pool
    assert new_pool is not pool
    assert batch_owner.workspace_release_calls == 1
    assert pool.workspace_pages(_GGUF_PACKED_WORKSPACE_LEASE_KEY) is None
    assert len(new_pool.workspace_pages(_GGUF_PACKED_WORKSPACE_LEASE_KEY)) == 32
    assert batch_owner.bound_workspace_pools[-1] is new_pool

    runner.close()
    assert runner.kv_pool is None
