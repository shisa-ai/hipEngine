from __future__ import annotations

from types import SimpleNamespace

from hipengine.generation.engine_loop import EngineLoopConfig
from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
from hipengine.generation.registry import GenerationRequest
from hipengine.dispatch import WorkItem, WorkKind
from hipengine.kvcache import DeviceChunkedKVPool


class _FakePrefixSession:
    def __init__(self, slot_id: int) -> None:
        self.slot_id = int(slot_id)
        self.scratch = SimpleNamespace(max_positions=1024)
        self.position = 0
        self.allocation = None
        self.pool = None
        self.prefill_calls: list[tuple[tuple[int, ...], int, int]] = []
        self.clone_calls: list[tuple[int, int]] = []

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

    def prefill_batch_native(self, prompt_token_ids, *, sessions, **kwargs):
        assert sessions == [self]
        prompt = tuple(int(token) for token in prompt_token_ids[0])
        start = int(self.position)
        self.position += len(prompt)
        self.prefill_calls.append((prompt, start, int(self.position)))
        return [SimpleNamespace(token_id=777)]

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


def _request(prompt: tuple[int, ...], *, max_tokens: int) -> GenerationRequest:
    return GenerationRequest(
        prompts=(prompt,),
        max_tokens=max_tokens,
        temperature=0.0,
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
    assert continued_session.prefill_calls == [((999,), 256, 257)]
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
