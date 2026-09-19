from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation import qwen35_gguf as qwen35_gguf_generation
from hipengine.generation.engine_loop import EngineLoopConfig
from hipengine.generation.qwen35_gguf import (
    _GGUF_PREFIX_RETAINED_SNAPSHOTS_ENV,
    _PREFIX_RETAINED_SNAPSHOTS_WIDE,
    Qwen35GGUFResidentModelRunner,
)
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
        self.prefill_batch_kwargs: list[dict] = []
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
        assert self.allocation.reused_block_ids == source.allocation.block_ids[
            : int(position) // 256
        ]
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
        self.prefill_batch_kwargs.append(dict(kwargs))
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
    # The runner asks whether an MTP adapter exists before it will let a
    # reused-prefix row open a prompt sink; this owner has none.
    supports_speculative_mtp = False
    model_plugin = SimpleNamespace()
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
    # A one-token suffix stays on the serial step: a single-row prefill leaves
    # the bulk schedule (MoE has no rows==1 bulk path), and one token costs one
    # step either way, so there is no batching win to trade correctness for.
    assert continued_session.prefill_calls == []
    assert continued_session.prefill_batch_kwargs == []
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

    # One-token suffix: serial step, not a one-row batched prefill.
    assert session.prefill_calls == []
    assert session.prefill_batch_kwargs == []
    assert session.step_calls == [(999, 256, 257)]
    # The reused-suffix route - batched by default, serial on fallback - must
    # record the phase, or a served run cannot tell which route paid.
    assert runner._prefix_cache_observability()["phase_calls"]["suffix_prefill"] == 1
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


def test_reused_suffix_prefill_falls_back_to_serial_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HIPENGINE_GGUF_PREFIX_BATCHED_SUFFIX=0 restores the serial suffix loop."""

    monkeypatch.setenv("HIPENGINE_GGUF_PREFIX_BATCHED_SUFFIX", "0")
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

    runner.prefill_batch(
        WorkItem(
            kind=WorkKind.PREFILL,
            request_ids=(2,),
            row_to_request=(2,),
            token_rows=(continued_prompt,),
        ),
        commit=True,
    )
    assert continued_session.prefill_calls == []
    assert continued_session.step_calls == [(999, 256, 257)]
    assert continued_row.slot is not None
    assert continued_row.slot.generated_ids == [777]
    assert runner._fallback_reasons["prefix_batched_suffix_unavailable"] == 1
    assert runner._prefix_cache_observability()["phase_calls"]["suffix_prefill"] == 1

    runner.rollback_admission(SimpleNamespace(request_id=1))
    runner.rollback_admission(SimpleNamespace(request_id=2))
    assert runner.kv_pool.stats.refcounted_pages == 0
    runner.close()


def test_batched_reused_suffix_splits_at_an_interior_prompt_boundary() -> None:
    """A suffix crossing the deepest prompt boundary captures it mid-prefill.

    The snapshot is only capturable while the session sits exactly on the
    boundary, so the batched suffix must run as two segments with the capture
    in between: [reused, boundary) then [boundary, prompt_end).
    """

    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=3)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=3,
            kv_pool_initial_pages=8,
            kv_pool_low_water_pages=8,
            kv_pool_high_water_pages=8,
            kv_pool_chunk_pages=8,
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

    # 600-token prompt: reuse 256, suffix 344, deepest prompt boundary 512 sits
    # inside the suffix (256 < 512 < 600).
    continued_prompt = tuple(range(1, 601))
    continued_request = _request(continued_prompt, max_tokens=2)
    runner.register_batch((2,), continued_request, prompt_rows=(continued_prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=2))
    continued_row = runner._rows[2]
    assert continued_row.lease is not None
    continued_session = continued_row.lease.session
    assert continued_row.prefix_reused_tokens == 256

    runner.prefill_batch(
        WorkItem(
            kind=WorkKind.PREFILL,
            request_ids=(2,),
            row_to_request=(2,),
            token_rows=(continued_prompt,),
        ),
        commit=True,
    )
    assert continued_session.prefill_calls == [
        (continued_prompt[256:512], 256, 512),
        (continued_prompt[512:], 512, 600),
    ]
    assert continued_session.step_calls == []
    # The capture fired between the two segments, at the exact boundary.
    assert continued_session.snapshot_capture_calls == [512]
    assert continued_row.slot is not None
    assert continued_row.slot.generated_ids == [777]

    runner.rollback_admission(SimpleNamespace(request_id=1))
    runner.rollback_admission(SimpleNamespace(request_id=2))
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


def test_incremental_prefill_captures_only_the_prompt_aligned_boundary() -> None:
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
    prompt = tuple(range(1, 513))
    request = _request(prompt, max_tokens=2)
    runner.register_batch((13,), request, prompt_rows=(prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=13))
    row = runner._rows[13]
    assert row.native_greedy is True

    for chunk in (prompt[:256], prompt[256:]):
        runner.prefill_batch(
            WorkItem(
                kind=WorkKind.PREFILL,
                request_ids=(13,),
                row_to_request=(13,),
                token_rows=(chunk,),
            ),
            commit=True,
        )

    assert row.lease is not None
    session = row.lease.session
    assert session.prefill_calls == [(prompt[:256], 0, 256), (prompt[256:], 256, 512)]
    # One hybrid-state capture per request, at the deepest prompt-aligned
    # boundary.  Capturing every 256-token chunk clones the full Conv/GDN state
    # once per chunk and is the measured prefill regression on real multi-turn
    # traffic; only the deepest boundary is reachable by the next turn.
    assert session.snapshot_capture_calls == [512]
    assert tuple(runner._prefix_state_snapshots) == (prompt,)
    assert runner._prefix_cache is not None
    match = runner._prefix_cache.match(prompt)
    assert match.hit is True
    assert match.matched_token_count == 512

    # The captured boundary is what a following turn actually reuses.
    continued_prompt = (*prompt, 900, 901)
    continued_request = _request(continued_prompt, max_tokens=2)
    runner.register_batch((14,), continued_request, prompt_rows=(continued_prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=14))
    continued_row = runner._rows[14]
    assert continued_row.prefix_reused_tokens == 512
    assert continued_row.prefix_fallback_reason is None

    runner._release_row_resources(row, retain_prefix_snapshots=True)
    runner._rows.pop(13)
    assert runner._evict_prefix_snapshot(prompt) is True
    runner.rollback_admission(SimpleNamespace(request_id=14))
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


def test_retained_prefix_survives_a_later_requests_capture() -> None:
    """A retained boundary must outlive the next request's own captures.

    Multi-turn serving only reuses anything if the boundary captured for turn N
    is still resolvable when turn N+1 arrives. A transient (unretained) snapshot
    from the current request must be sacrificed first.
    """

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
    source = runner._rows[10]
    assert source.lease is not None
    source.prefill_tokens_seen = len(prompt)
    source.lease.session.position = 512
    assert runner._refresh_prefix_cache(source) is True
    retained_snapshot = source.lease.session.snapshots[-1]
    runner._release_row_resources(source, retain_prefix_snapshots=True)
    runner._rows.pop(10)
    assert runner.observability_snapshot()["prefix_cache"]["retained_snapshot_entries"] == 1

    later_prompt = tuple(range(1, 257))
    later_request = _request(later_prompt, max_tokens=2)
    runner.register_batch((11,), later_request, prompt_rows=(later_prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=11))
    later = runner._rows[11]
    assert later.lease is not None
    later.prefill_tokens_seen = len(later_prompt)
    later.lease.session.position = 256
    assert runner._refresh_prefix_cache(later) is True

    assert retained_snapshot.closed is False, (
        "the later request's capture evicted the retained snapshot"
    )
    assert runner._prefix_cache is not None
    assert runner._prefix_cache.match(prompt[:512]).matched_token_count == 512
    assert runner.observability_snapshot()["prefix_cache"]["retained_snapshot_entries"] == 1

    runner.rollback_admission(SimpleNamespace(request_id=11))
    runner._evict_prefix_snapshot(prompt[:512])
    assert runner.kv_pool.stats.refcounted_pages == 0
    runner.close()


def _retain_boundary(
    runner: Qwen35GGUFResidentModelRunner,
    request_id: int,
    prompt: tuple[int, ...],
    boundary: int,
) -> None:
    """Admit, capture one aligned boundary, and retain it like a finished turn."""

    request = _request(prompt, max_tokens=2, forced_token_id=811)
    runner.register_batch((request_id,), request, prompt_rows=(prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=request_id))
    row = runner._rows[request_id]
    assert row.lease is not None
    row.prefill_tokens_seen = len(prompt)
    row.lease.session.position = boundary
    assert runner._refresh_prefix_cache(row) is True
    runner._release_row_resources(row, retain_prefix_snapshots=True)
    runner._rows.pop(request_id)


def test_retained_prefixes_from_two_conversations_coexist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retention must cover the working set, not one boundary.

    Served lanes interleave, so a conversation's next turn arrives after other
    conversations have captured and retained their own boundaries. A retained
    budget of one made every conversation miss its second hand-off. The wider
    working set is opt-in (HIPENGINE_GGUF_PREFIX_RETAINED_SNAPSHOTS) because it
    is a measured regression until the shared-admission contiguity path lands.
    """

    monkeypatch.setenv(
        _GGUF_PREFIX_RETAINED_SNAPSHOTS_ENV, str(_PREFIX_RETAINED_SNAPSHOTS_WIDE)
    )
    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=1)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=1,
            kv_pool_initial_pages=8,
            kv_pool_low_water_pages=8,
            kv_pool_high_water_pages=8,
            kv_pool_chunk_pages=8,
            prefix_cache="radix",
        )
    )
    first = tuple(range(1, 301))
    second = tuple(range(1001, 1301))
    _retain_boundary(runner, 40, first, 256)
    _retain_boundary(runner, 41, second, 256)

    cache = runner.observability_snapshot()["prefix_cache"]
    assert cache["retained_snapshot_entries"] == 2
    assert runner._prefix_cache is not None
    assert runner._prefix_cache.match(first).matched_token_count == 256
    assert runner._prefix_cache.match(second).matched_token_count == 256

    for tokens in (first[:256], second[:256]):
        assert runner._evict_prefix_snapshot(tokens) is True
    assert runner.kv_pool.stats.refcounted_pages == 0
    runner.close()


def test_retained_budget_defaults_to_the_pre_fix_effective_value() -> None:
    """The wider retained working set is opt-in, not the shipped default.

    Enabling a 16-entry retained set today is a measured serving regression
    (docs/REFACTOR.md), so the default stays at the pre-fix effective budget
    max(1, capacity): at capacity 1 a second retained boundary evicts the first.
    """

    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=1)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=1,
            kv_pool_initial_pages=8,
            kv_pool_low_water_pages=8,
            kv_pool_high_water_pages=8,
            kv_pool_chunk_pages=8,
            prefix_cache="radix",
        )
    )
    first = tuple(range(1, 301))
    second = tuple(range(1001, 1301))
    _retain_boundary(runner, 40, first, 256)
    _retain_boundary(runner, 41, second, 256)

    cache = runner.observability_snapshot()["prefix_cache"]
    assert cache["retained_snapshot_entries"] == 1
    assert cache["retained_snapshot_evictions_by_reason"] == {"trim_retained": 1}
    assert runner._prefix_cache is not None
    assert runner._prefix_cache.match(first).matched_token_count == 0
    assert runner._prefix_cache.match(second).matched_token_count == 256

    assert runner._evict_prefix_snapshot(second[:256]) is True
    assert runner.kv_pool.stats.refcounted_pages == 0
    runner.close()


def test_reuse_chain_deepens_across_three_turns() -> None:
    """Turn N+1 must be able to match the deepest boundary turn N retained.

    A cumulative conversation resends its whole history, so the boundary a turn
    reaches during decode is a prefix of the next turn's prompt. If the chain
    works, the third turn matches deeper than the first hand-off.
    """

    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=1)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=1,
            kv_pool_initial_pages=8,
            kv_pool_low_water_pages=8,
            kv_pool_high_water_pages=8,
            kv_pool_chunk_pages=8,
            prefix_cache="radix",
        )
    )
    first_prompt = tuple(range(1, 301))
    reply_one = tuple(811 for _ in range(256))

    def admit(request_id: int, prompt: tuple[int, ...]):
        request = _request(prompt, max_tokens=2, forced_token_id=811)
        runner.register_batch((request_id,), request, prompt_rows=(prompt,))
        runner.reserve_admission(SimpleNamespace(request_id=request_id))
        return runner._rows[request_id]

    # Turn 1: no reuse, capture the prompt boundary, retain it.
    first = admit(30, first_prompt)
    assert first.prefix_fallback_reason == "miss"
    first.prefill_tokens_seen = len(first_prompt)
    first.lease.session.position = 256
    assert runner._refresh_prefix_cache(first) is True
    runner._release_row_resources(first, retain_prefix_snapshots=True)
    runner._rows.pop(30)

    # Turn 2: the client resends the transcript including the reply, so the
    # boundary turn 1 reached during decode is now inside the prompt.
    second_prompt = (*first_prompt, *reply_one[:212])
    second = admit(31, second_prompt)
    assert second.prefix_reused_tokens == 256
    runner.prefill_batch(
        WorkItem(
            kind=WorkKind.PREFILL,
            request_ids=(31,),
            row_to_request=(31,),
            token_rows=(second_prompt,),
        ),
        commit=True,
    )
    assert second.slot is not None
    second.slot.generated_ids = [811] * 256
    second.lease.session.position = 768
    assert runner._refresh_prefix_cache(second) is True
    runner._release_row_resources(second, retain_prefix_snapshots=True)
    runner._rows.pop(31)

    # Turn 3: the client adds the next user turn, so the prompt extends past the
    # boundary turn 2 retained and can reuse it.
    third_prompt = (*second_prompt, *reply_one, *range(5000, 5010))
    third = admit(32, third_prompt)
    assert third.prefix_reused_tokens == 768, (
        "turn 3 did not reach the boundary turn 2 retained"
    )

    runner.rollback_admission(SimpleNamespace(request_id=32))
    for tokens in tuple(runner._prefix_state_snapshots):
        runner._evict_prefix_snapshot(tokens)
    assert runner.kv_pool.stats.refcounted_pages == 0
    runner.close()


def test_prefix_fallback_reasons_are_counted_per_reason() -> None:
    """The served harness can only see counters, so each reason needs one."""

    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=1)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=1,
            kv_pool_initial_pages=8,
            kv_pool_low_water_pages=8,
            kv_pool_high_water_pages=8,
            kv_pool_chunk_pages=8,
            prefix_cache="radix",
        )
    )
    short = tuple(range(1, 101))
    request = _request(short, max_tokens=2, forced_token_id=811)
    runner.register_batch((50,), request, prompt_rows=(short,))
    runner.reserve_admission(SimpleNamespace(request_id=50))
    row = runner._rows[50]
    assert row.prefix_fallback_reason == "prompt_too_short"
    assert runner._prefix_cache_observability()["fallback_reasons"] == {
        "prompt_too_short": 1
    }

    runner.rollback_admission(SimpleNamespace(request_id=50))
    runner.close()


def test_prefix_phase_counters_attribute_admission_refresh_and_capture() -> None:
    """Serving diagnostics attribute prefix cost to the phase that paid it."""

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
    source_request = _request(prefix, max_tokens=2, forced_token_id=811)
    runner.register_batch((10,), source_request, prompt_rows=(prefix,))
    runner.reserve_admission(SimpleNamespace(request_id=10))
    source = runner._rows[10]
    assert source.prefix_fallback_reason == "miss"

    admission = runner._prefix_cache_observability()
    admission_calls = admission["phase_calls"]
    # A miss still pays the packed-owner flush, the trie probe, and the pool
    # reservation; it pays nothing for state restore because there is no source.
    assert admission_calls["admission_lookup"] == 1
    assert admission_calls["lookup_flush_packed"] == 1
    assert admission_calls["lookup_match"] == 1
    assert admission_calls["admission_pool"] == 1
    assert "admission_restore_state" not in admission_calls
    assert "capture_total" not in admission_calls

    assert source.lease is not None
    source.prefill_tokens_seen = len(prefix)
    source.lease.session.position = len(prefix)
    assert runner._refresh_prefix_cache(source) is True

    captured = runner._prefix_cache_observability()
    captured_calls = captured["phase_calls"]
    assert captured_calls["capture_total"] == 1
    assert captured_calls["capture_clone_state"] == 1
    assert captured_calls["refresh_trie"] == 1
    assert captured_calls["processed_tokens"] >= 1
    assert captured_calls["refresh_total"] == 1
    # Every recorded phase carries a non-negative wall time and a call count.
    assert set(captured["phase_ms"]) == set(captured_calls)
    for name, calls in captured_calls.items():
        assert calls >= 1, name
        assert float(captured["phase_ms"][name]) >= 0.0, name
    # Counters accumulate across requests rather than reporting one call.
    assert captured_calls["admission_lookup"] == 1
    assert captured["phase_ms"]["admission_lookup"] >= admission["phase_ms"]["admission_lookup"]

    runner._release_row_resources(source, retain_prefix_snapshots=True)
    runner._rows.pop(10)
    continued_prompt = (*prefix, 999, 998)
    continued_request = _request(continued_prompt, max_tokens=2, forced_token_id=812)
    runner.register_batch((11,), continued_request, prompt_rows=(continued_prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=11))
    continued = runner._rows[11]
    assert continued.prefix_snapshot_hit is True

    reused = runner._prefix_cache_observability()
    reused_calls = reused["phase_calls"]
    assert reused_calls["admission_lookup"] == 2
    assert reused_calls["lookup_resolve"] == 1
    assert reused_calls["admission_restore_state"] == 1
    # A hit refreshes the new owner's boundary from the cloned session position,
    # so it re-inserts the trie entry; the capture is skipped because the same
    # token tuple is already resident.
    assert reused_calls["admission_refresh"] == 1
    assert reused_calls["refresh_trie"] == 2
    assert reused_calls["capture_total"] == 1
    assert reused_calls["capture_clone_state"] == 1

    runner.rollback_admission(SimpleNamespace(request_id=11))
    runner._evict_prefix_snapshot(prefix)
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

    # A one-token suffix still reuses the prefix; only the suffix route
    # changes. It goes through the serial step rather than a one-row batched
    # prefill, which would leave the bulk schedule (MoE has no rows==1 bulk
    # path) for no speedup - one token costs one step either way.
    single_prompt = (*prefix, 998)
    single_request = _request(single_prompt, max_tokens=2)
    runner.register_batch((23,), single_request, prompt_rows=(single_prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=23))
    single = runner._rows[23]
    assert single.prefix_lookup is True
    assert single.prefix_matched_tokens == 512
    assert single.prefix_reused_tokens == 512
    assert single.prefix_fallback_reason is None
    runner.rollback_admission(SimpleNamespace(request_id=23))

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
        self.max_sequence_length = 768
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
    # Eight request pages plus two serving slots * four workspace pages.
    # Wider physical verifier layouts use separately budgeted private KV.
    assert pool.current_pages == 16
    lease = pool.workspace_pages(_GGUF_PACKED_WORKSPACE_LEASE_KEY)
    assert lease is not None and len(lease) == 8
    assert pool.stats.free_pages == 8
    assert pool.stats.pinned_pages == 8
    assert batch_owner.bound_workspace_pools == [pool]
    snapshot = runner.observability_snapshot()
    assert snapshot["model_runner"]["max_active_requests"] == 2
    assert snapshot["model_runner"]["max_context_tokens"] == 768
    assert "max_pages" in snapshot["kv_pool"]
    assert "budget_bytes" in snapshot["kv_pool"]

    # Reconfiguration releases the lease and the idle workspace before the
    # old pool closes, then re-leases on the fresh pool.
    runner.configure_engine_loop(config)
    new_pool = runner.kv_pool
    assert new_pool is not pool
    assert batch_owner.workspace_release_calls == 1
    assert pool.workspace_pages(_GGUF_PACKED_WORKSPACE_LEASE_KEY) is None
    assert len(new_pool.workspace_pages(_GGUF_PACKED_WORKSPACE_LEASE_KEY)) == 8
    assert batch_owner.bound_workspace_pools[-1] is new_pool

    runner.close()
    assert runner.kv_pool is None


def test_configure_engine_loop_leases_the_capacity_workspace_at_c1() -> None:
    """A C1 pool leases the serving capacity's workspace, not a fixed ceiling.

    The lease's slot term is the runner's serving capacity (one slot at C1)
    times the per-slot union floor of max(ceil(max_positions/256),
    _PACKED_VERIFY_MIN_MAX_SEQUENCE/256) pages. An above-capacity geometry
    (an MTP verify group packs four physical slots at C1) is deliberately
    not covered by the pool-creation-time lease: ``_GGUFPackedTargetState``
    uses private KV charged against the pool budget. The lease is an eager
    reservation, not an upper bound on total workspace demand.
    """

    from hipengine.runtime.qwen35_gguf_runner import (
        _GGUF_PACKED_WORKSPACE_LEASE_KEY,
        _PACKED_VERIFY_MIN_MAX_SEQUENCE,
        packed_verify_lease_slot_ceiling,
        packed_verify_workspace_lease_pages,
    )

    owner = _FakeGlobalPoolOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=1)
    config = EngineLoopConfig(
        max_active_requests=1,
        kv_pool_initial_pages=8,
        kv_pool_low_water_pages=8,
        kv_pool_chunk_pages=8,
        prefix_cache="off",
    )
    runner._reserve_sessions()
    runner.configure_engine_loop(config)

    pool = runner.kv_pool
    assert pool is not None
    expected_lease_pages = packed_verify_workspace_lease_pages(
        1,
        768,
    )
    assert expected_lease_pages == packed_verify_lease_slot_ceiling(1) * max(
        (768 + 255) // 256,
        _PACKED_VERIFY_MIN_MAX_SEQUENCE // 256,
    )
    assert expected_lease_pages == 4
    assert pool.current_pages == 8 + expected_lease_pages
    lease = pool.workspace_pages(_GGUF_PACKED_WORKSPACE_LEASE_KEY)
    assert lease is not None and len(lease) == expected_lease_pages
    assert pool.stats.pinned_pages == expected_lease_pages
    assert pool.stats.free_pages == 8

    runner.close()


def test_packed_verify_union_geometry_is_capacity_honest() -> None:
    """Serving capacity supplies a floor; wider physical layouts still fit."""

    from types import SimpleNamespace

    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFResidentSession,
        _PACKED_VERIFY_DEFAULT_SLOT_CAPACITY,
    )

    def geometry(max_batch_size, slot_count=1):
        namespace = SimpleNamespace(
            max_batch_size=max_batch_size,
            _packed_verify_state=None,
            _packed_verify_scratch=None,
            _bulk_prefill_scratch=None,
            _packed_verify_prefill_row_cap=lambda: 128,
        )
        return Qwen35GGUFResidentSession._packed_verify_union_geometry(
            namespace,
            slot_count=slot_count,
            rows=8,
            max_sequence_length=3072,
        )

    # C1 serves one slot: state slots and GDN segments follow the real cap.
    union_slots, _union_rows, _union_max_seq, union_segments = geometry(1)
    assert union_slots == 1
    assert union_segments == 1

    # The default C8 geometry is unchanged.
    union_slots, _union_rows, _union_max_seq, union_segments = geometry(8)
    assert union_slots == _PACKED_VERIFY_DEFAULT_SLOT_CAPACITY
    assert union_segments == _PACKED_VERIFY_DEFAULT_SLOT_CAPACITY

    # C4 right-sizes to four slots.
    union_slots, _union_rows, _union_max_seq, union_segments = geometry(4)
    assert union_slots == 4
    assert union_segments == 4

    # Physical verifier layouts may be wider than the serving capacity.
    union_slots, _, _, union_segments = geometry(1, slot_count=4)
    assert union_slots == union_segments == 4

    # Absent serving caps keep the historical 8-slot fallback.
    union_slots, _union_rows, _union_max_seq, union_segments = geometry(None)
    assert union_slots == _PACKED_VERIFY_DEFAULT_SLOT_CAPACITY


def test_gapped_placement_declines_a_hit_whose_suffix_exceeds_the_paged_budget(
    monkeypatch,
) -> None:
    """Placement decides the route, so a gapped hit is only worth a short suffix.

    A contiguous shared allocation keeps the fast slot-local prefill. A gapped
    one drops to the packed paged route, which is far slower per token, so past
    a few hundred suffix tokens the full prefill the hit would replace is the
    cheaper answer. Declining there is what turns a wide retained working set
    from a regression into a win.
    """

    from hipengine.kvcache.pool import DeviceKVContiguityError

    # Small budget so the shapes fit the fixture's resident capacity; the
    # default is 512 tokens, measured against the full prefill a hit replaces.
    monkeypatch.setenv("HIPENGINE_GGUF_PREFIX_GAPPED_SUFFIX_MAX", "64")
    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=3)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=3,
            kv_pool_initial_pages=32,
            kv_pool_low_water_pages=32,
            kv_pool_high_water_pages=32,
            kv_pool_chunk_pages=32,
            prefix_cache="radix",
        )
    )
    prefix = tuple(range(1, 513))
    source_request = _request(prefix, max_tokens=2)
    runner.register_batch((60,), source_request, prompt_rows=(prefix,))
    runner.reserve_admission(SimpleNamespace(request_id=60))
    source = runner._rows[60]
    source.prefill_tokens_seen = len(prefix)
    source.lease.session.position = len(prefix)
    assert runner._refresh_prefix_cache(source) is True

    pool = runner.kv_pool
    real_admit = pool.admit_with_shared_prefix

    def only_gapped(*args, **kwargs):
        if kwargs.get("require_contiguous"):
            raise DeviceKVContiguityError("no contiguous run for the test")
        return real_admit(*args, **kwargs)

    monkeypatch.setattr(pool, "admit_with_shared_prefix", only_gapped)

    # A long suffix is declined: the paged route would cost more than the miss.
    long_prompt = (*prefix, *range(9000, 9000 + 128))
    runner.register_batch((61,), _request(long_prompt, max_tokens=2), prompt_rows=(long_prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=61))
    long_row = runner._rows[61]
    assert long_row.prefix_matched_tokens == 512
    assert long_row.prefix_reused_tokens == 0
    assert long_row.prefix_fallback_reason == "gapped_suffix_exceeds_paged_budget"
    runner.rollback_admission(SimpleNamespace(request_id=61))

    # A short suffix still takes the gapped hit.
    short_prompt = (*prefix, *range(9000, 9000 + 32))
    runner.register_batch((62,), _request(short_prompt, max_tokens=2), prompt_rows=(short_prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=62))
    short_row = runner._rows[62]
    assert short_row.prefix_reused_tokens == 512
    assert short_row.prefix_fallback_reason is None
    assert runner._prefix_gapped_admissions >= 1
    runner.rollback_admission(SimpleNamespace(request_id=62))
    runner.close()


def test_gapped_placement_takes_a_long_suffix_when_the_gather_route_is_fast(
    monkeypatch,
) -> None:
    """With the gapped gather route available, any suffix length is worth a hit.

    The gapped gather route swaps a gapped slot's identity spans for its real
    block table and runs the same AOTriton attention the contiguous route
    uses, so a gapped hit no longer pays the paged-route penalty the suffix
    budget guarded against. The decline below only binds when that fast
    route is unavailable (kill-switch, no head-major KV, oversized context).
    """

    monkeypatch.setenv("HIPENGINE_GGUF_PREFIX_GAPPED_SUFFIX_MAX", "64")
    monkeypatch.setattr(
        qwen35_gguf_generation,
        "_gguf_prefix_gapped_fast_route_available",
        lambda lease, context_tokens: True,
    )
    from hipengine.kvcache.pool import DeviceKVContiguityError

    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=3)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=3,
            kv_pool_initial_pages=32,
            kv_pool_low_water_pages=32,
            kv_pool_high_water_pages=32,
            kv_pool_chunk_pages=32,
            prefix_cache="radix",
        )
    )
    prefix = tuple(range(1, 513))
    source_request = _request(prefix, max_tokens=2)
    runner.register_batch((70,), source_request, prompt_rows=(prefix,))
    runner.reserve_admission(SimpleNamespace(request_id=70))
    source = runner._rows[70]
    source.prefill_tokens_seen = len(prefix)
    source.lease.session.position = len(prefix)
    assert runner._refresh_prefix_cache(source) is True

    pool = runner.kv_pool
    real_admit = pool.admit_with_shared_prefix

    def only_gapped(*args, **kwargs):
        if kwargs.get("require_contiguous"):
            raise DeviceKVContiguityError("no contiguous run for the test")
        return real_admit(*args, **kwargs)

    monkeypatch.setattr(pool, "admit_with_shared_prefix", only_gapped)

    # The same long suffix the budget would decline on the slow route now
    # takes the gapped hit: the gather route makes it pay contiguous cost.
    long_prompt = (*prefix, *range(9000, 9000 + 128))
    runner.register_batch((71,), _request(long_prompt, max_tokens=2), prompt_rows=(long_prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=71))
    long_row = runner._rows[71]
    assert long_row.prefix_matched_tokens == 512
    assert long_row.prefix_reused_tokens == 512
    assert long_row.prefix_fallback_reason is None
    assert runner._prefix_gapped_admissions >= 1
    runner.rollback_admission(SimpleNamespace(request_id=71))
    runner.close()


def test_workspace_telemetry_separates_private_kv_and_request_pins():
    from hipengine.generation.engine_loop import EngineLoopConfig

    owner = _FakeGlobalPoolOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=1)
    runner._reserve_sessions()
    runner.configure_engine_loop(EngineLoopConfig(
        max_active_requests=1, kv_pool_initial_pages=8, kv_pool_low_water_pages=8,
    ))
    pool = runner.kv_pool
    allocation = pool.allocate(99, 1)
    pool.pin(allocation.block_ids)
    token = pool.reserve_private_workspace(4 * pool.page_bytes)
    snapshot = runner.kv_pool_memory_snapshot()
    assert snapshot["private_workspace_kv_bytes"] == 4 * pool.page_bytes
    assert snapshot["accounted_kv_bytes"] == 16 * pool.page_bytes
    model = runner.observability_snapshot()["model_runner"]
    assert model["packed_workspace_leased_pool_bytes"] == 4 * pool.page_bytes
    pool.release_private_workspace(token)
    pool.unpin(allocation.block_ids)
    pool.release(99)
    runner.close()
