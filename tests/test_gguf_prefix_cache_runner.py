from __future__ import annotations

from types import SimpleNamespace

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

    def prefill_batch_native(self, prompt_token_ids, *, sessions, **kwargs):
        assert sessions == [self]
        prompt = tuple(int(token) for token in prompt_token_ids[0])
        start = int(self.position)
        self.position += len(prompt)
        self.prefill_calls.append((prompt, start, int(self.position)))
        return [SimpleNamespace(token_id=777)]

    def step(self, token_id: int, *, return_logits: bool):
        assert return_logits is False
        start = int(self.position)
        self.position += 1
        self.step_calls.append((int(token_id), start, int(self.position)))
        return SimpleNamespace(token_id=777)

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
) -> GenerationRequest:
    return GenerationRequest(
        prompts=(prompt,),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        ignore_eos=True,
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
