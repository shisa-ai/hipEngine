"""Packed-verifier recorder lifetime tests.

``verify_target_blocks_batch`` used to start the opt-in HIP-event recorder
before embedding and resolve it only on success. An exception in embedding,
layer execution, sampling, or output scatter bypassed the recorder's own
resolve-time ``finally``, leaking every event handle created up to that point.
These tests drive the real method with CPU stand-ins for the GPU work and
inject failures at recorder start, verification work, and resolve. They assert
that the created handles are released, that the original exception survives a
failing cleanup, and that the default no-timing path still creates no events.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen35_gguf_runner as gguf_runner
from hipengine.core.dtype import DType
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

_TIMING_ENV = "HIPENGINE_GGUF_PACKED_VERIFY_GPU_STAGE_TIMINGS"


class _FakeEventRuntime:
    """CPU stand-in for the HIP calls the packed verifier reaches."""

    def __init__(self) -> None:
        self.next_event = 100
        self.created: list[int] = []
        self.recorded: list[tuple[int, int]] = []
        self.synchronized: list[int] = []
        self.destroyed: list[int] = []
        self.record_failure_at: int | None = None
        self.synchronize_failure_at: int | None = None
        self.destroy_failures: list[list[int]] = []

    def event_create(self) -> int:
        event = self.next_event
        self.next_event += 1
        self.created.append(event)
        return event

    def event_record(self, event: int, stream: int = 0) -> None:
        if (
            self.record_failure_at is not None
            and len(self.recorded) == self.record_failure_at
        ):
            raise RuntimeError("event record failed")
        self.recorded.append((int(event), int(stream)))

    def event_synchronize(self, event: int) -> None:
        if (
            self.synchronize_failure_at is not None
            and len(self.synchronized) == self.synchronize_failure_at
        ):
            raise RuntimeError("event synchronize failed")
        self.synchronized.append(int(event))

    def event_elapsed_time_ms(self, start: int, stop: int) -> float:
        return float(int(stop) - int(start))

    def event_destroy(self, event: int) -> None:
        for failure in self.destroy_failures:
            if failure[0] == event and failure[1] > 0:
                failure[1] -= 1
                raise RuntimeError(f"event destroy failed for {event}")
        self.destroyed.append(int(event))

    def memcpy_async(self, *args: object, **kwargs: object) -> None:
        return None

    def device_synchronize(self) -> None:
        return None

    def stream_synchronize(self, stream: int) -> None:
        return None


@dataclass
class _FakeSessionScratch:
    hidden_seed_fp32: object
    layer_conv_states: object | None = None
    layer_recurrent_states: object | None = None


@dataclass
class _FakePackedScratch:
    norm: object
    retained_decode_kernel: object | None = None


class _FakePackedScratchBase:
    def for_packed_verify_layout(self, layout, *, runtime, stream: int = 0):
        return _FakePackedScratch(norm=SimpleNamespace(ptr=0x300))


def _make_session(
    monkeypatch: pytest.MonkeyPatch,
    runtime: _FakeEventRuntime,
) -> tuple[Qwen35GGUFResidentSession, dict[str, object]]:
    """Build the smallest session shape that reaches the packed recorder."""

    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_RESIDUAL", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", "1")
    monkeypatch.delenv("HIPENGINE_GGUF_PACKED_VERIFY_CAPTURE_DIR", raising=False)

    weights = SimpleNamespace(
        config=SimpleNamespace(layer_types=(), rms_norm_eps=1e-6),
        root=lambda name: SimpleNamespace(
            allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=0x400))
        ),
    )
    runner = SimpleNamespace(weights=weights, hidden_size=8, vocab_size=128)
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = runner
    session.scratch = _FakeSessionScratch(hidden_seed_fp32=SimpleNamespace(ptr=0x101))
    session._prefill_token_buf = SimpleNamespace(ptr=0x102)
    session._prefill_hidden_a = SimpleNamespace(ptr=0x103)
    session._prefill_hidden_b = SimpleNamespace(ptr=0x104)
    session._verify_hidden_seed_buf = SimpleNamespace(ptr=0x105)
    session._bulk_prefill_scratch = SimpleNamespace(rows=16)
    session.kv_storage_dtype = DType.BF16
    session.use_expert_sidecar = False
    session.runtime = runtime
    session._position = 0
    session.use_gemv_decode = False
    session.target_verifier_production_q4_rowtile = False

    session._direct_resident_verify_linear_state = MethodType(
        lambda self, jobs: None, session
    )
    session._ensure_packed_verify_workspace = MethodType(
        lambda self, **kwargs: (
            SimpleNamespace(
                slot_count=1,
                max_sequence_length=1024,
                blocks_per_slot=4,
                page_ids=(),
                layer_conv_states=(None,),
                layer_recurrent_states=(None,),
            ),
            _FakePackedScratchBase(),
        ),
        session,
    )
    session._ensure_verify_block_buffers = MethodType(
        lambda self, rows, **kwargs: None, session
    )
    session._sync_packed_verify_initial_state = MethodType(
        lambda self, *args, **kwargs: None, session
    )
    session._device_token_embedding_weight = MethodType(
        lambda self, **kwargs: object(), session
    )
    session._sample_target_block_rows_from_hidden = MethodType(
        lambda self, *args, **kwargs: np.zeros(2, dtype=np.int64), session
    )
    session._scatter_packed_verify_outputs = MethodType(
        lambda self, *args, **kwargs: [], session
    )
    for name in (
        "_q5_raw_mmq_target_context",
        "_q6_dp4a_grouped_target_context",
        "_q6_integer_mmq_context",
        "_iq_dense_mmq_context",
    ):
        setattr(
            session,
            name,
            MethodType(lambda self, *args, **kwargs: nullcontext(), session),
        )

    monkeypatch.setattr(
        gguf_runner,
        "_rebind_packed_verify_layout_pages",
        lambda layout, state: layout,
    )
    monkeypatch.setattr(
        gguf_runner, "_stage_gguf_packed_verify_token_ids", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(gguf_runner, "launch_gguf_embedding", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gguf_runner, "gguf_rmsnorm_bf16_f32_weight", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        gguf_runner, "gguf_rmsnorm_bf16_f32_weight_out_f32", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(gguf_runner, "copy_device_to_host", lambda *args, **kwargs: None)
    monkeypatch.setattr(gguf_runner, "host_array_ptr", lambda array: 0)
    monkeypatch.setattr(
        gguf_runner, "wmma_prefill_session", lambda *args, **kwargs: nullcontext()
    )
    monkeypatch.setattr(
        gguf_runner, "gemv_decode_session", lambda *args, **kwargs: nullcontext()
    )
    monkeypatch.setattr(
        gguf_runner, "target_verifier_rowtile_session", lambda *args, **kwargs: nullcontext()
    )
    monkeypatch.setattr(
        gguf_runner,
        "target_verifier_production_q4_rowtile_session",
        lambda *args, **kwargs: nullcontext(),
    )

    job: dict[str, object] = {"session": session, "input_token_ids": (1, 2)}
    return session, job


def test_packed_verify_start_failure_releases_created_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeEventRuntime()
    runtime.record_failure_at = 0
    session, job = _make_session(monkeypatch, runtime)
    monkeypatch.setenv(_TIMING_ENV, "1")

    with pytest.raises(RuntimeError, match="event record failed"):
        session.verify_target_blocks_batch([job])

    assert runtime.created == [100]
    assert runtime.destroyed == [100]


def test_packed_verify_work_failure_releases_started_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeEventRuntime()
    session, job = _make_session(monkeypatch, runtime)
    monkeypatch.setenv(_TIMING_ENV, "1")

    def fail_embedding(*args: object, **kwargs: object) -> None:
        raise RuntimeError("embedding boom")

    monkeypatch.setattr(gguf_runner, "launch_gguf_embedding", fail_embedding)

    with pytest.raises(RuntimeError, match="embedding boom"):
        session.verify_target_blocks_batch([job])

    assert runtime.created == [100]
    assert runtime.destroyed == [100]


@pytest.mark.parametrize("stage", ["layer", "head", "scatter"])
def test_packed_verify_late_work_failure_releases_every_created_event(
    monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    runtime = _FakeEventRuntime()
    session, job = _make_session(monkeypatch, runtime)
    monkeypatch.setenv(_TIMING_ENV, "1")

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"{stage} boom")

    if stage == "layer":
        session.runner.weights.config.layer_types = (gguf_runner.LINEAR_ATTENTION,)
        session.runner._run_linear_attention_prefill_layer_rows = fail
    elif stage == "head":
        session._sample_target_block_rows_from_hidden = fail
    else:
        session._scatter_packed_verify_outputs = fail

    with pytest.raises(RuntimeError, match=f"{stage} boom"):
        session.verify_target_blocks_batch([job])

    assert len(runtime.created) > 1
    assert sorted(runtime.created) == sorted(runtime.destroyed)


def test_packed_verify_resolve_failure_retries_retained_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeEventRuntime()
    runtime.destroy_failures = [[100, 1]]
    session, job = _make_session(monkeypatch, runtime)
    monkeypatch.setenv(_TIMING_ENV, "1")

    with pytest.raises(RuntimeError, match="event destroy failed for 100"):
        session.verify_target_blocks_batch([job])

    assert runtime.created == [100, 101, 102, 103]
    assert sorted(runtime.destroyed) == [100, 101, 102, 103]


def test_packed_verify_resolve_synchronize_failure_still_releases_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeEventRuntime()
    runtime.synchronize_failure_at = 0
    session, job = _make_session(monkeypatch, runtime)
    monkeypatch.setenv(_TIMING_ENV, "1")

    with pytest.raises(RuntimeError, match="event synchronize failed"):
        session.verify_target_blocks_batch([job])

    assert sorted(runtime.destroyed) == [100, 101, 102, 103]


def test_packed_verify_resolve_failure_preserves_sync_error_during_destroy_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeEventRuntime()
    runtime.synchronize_failure_at = 0
    runtime.destroy_failures = [[100, 1]]
    session, job = _make_session(monkeypatch, runtime)
    monkeypatch.setenv(_TIMING_ENV, "1")

    with pytest.raises(RuntimeError, match="event synchronize failed"):
        session.verify_target_blocks_batch([job])

    assert sorted(runtime.destroyed) == [100, 101, 102, 103]


def test_packed_verify_success_resolves_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeEventRuntime()
    session, job = _make_session(monkeypatch, runtime)
    monkeypatch.setenv(_TIMING_ENV, "1")

    results = session.verify_target_blocks_batch([job])

    assert results == []
    assert sorted(runtime.destroyed) == [100, 101, 102, 103]
    timings = session.last_packed_verify_stage_timings_ms
    assert timings["packed_verify_gpu_embedding"] == 1.0
    assert timings["packed_verify_gpu_output_norm_hidden"] == 1.0
    assert timings["packed_verify_gpu_lm_head_sample"] == 1.0
    assert "packed_verify_total" in timings


def test_packed_verify_default_path_creates_no_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeEventRuntime()
    session, job = _make_session(monkeypatch, runtime)
    monkeypatch.delenv(_TIMING_ENV, raising=False)

    results = session.verify_target_blocks_batch([job])

    assert results == []
    assert runtime.created == []
    assert runtime.destroyed == []
    timings = session.last_packed_verify_stage_timings_ms
    assert "packed_verify_total" in timings
    assert not [name for name in timings if name.startswith("packed_verify_gpu_")]


class _FakeScopeRecorder:
    def __init__(
        self,
        *,
        start_error: BaseException | None = None,
        resolve_error: BaseException | None = None,
        close_errors: int = 0,
    ) -> None:
        self.start_error = start_error
        self.resolve_error = resolve_error
        self.close_errors = close_errors
        self.calls: list[str] = []

    def start(self) -> None:
        self.calls.append("start")
        if self.start_error is not None:
            raise self.start_error

    def resolve_into(self, timings: dict[str, float] | None) -> None:
        self.calls.append("resolve")
        if self.resolve_error is not None:
            raise self.resolve_error

    def close(self) -> None:
        self.calls.append("close")
        if self.close_errors > 0:
            self.close_errors -= 1
            raise RuntimeError("close failed")


def test_scope_work_failure_preserves_original_when_cleanup_fails() -> None:
    scope_type = gguf_runner._PackedVerifyGpuStageRecorderScope
    recorder = _FakeScopeRecorder(close_errors=2)

    with pytest.raises(RuntimeError, match="work boom"):
        with scope_type(recorder, {}):
            raise RuntimeError("work boom")

    assert recorder.calls == ["start", "close", "close"]


def test_scope_start_failure_preserves_original_when_cleanup_fails() -> None:
    scope_type = gguf_runner._PackedVerifyGpuStageRecorderScope
    recorder = _FakeScopeRecorder(start_error=RuntimeError("start boom"), close_errors=2)

    with pytest.raises(RuntimeError, match="start boom"):
        with scope_type(recorder, {}):
            raise AssertionError("body must not run")

    assert recorder.calls == ["start", "close", "close"]


def test_scope_success_surfaces_persistent_cleanup_failure() -> None:
    scope_type = gguf_runner._PackedVerifyGpuStageRecorderScope
    recorder = _FakeScopeRecorder(close_errors=2)

    with pytest.raises(RuntimeError, match="close failed"):
        with scope_type(recorder, {}):
            pass

    assert recorder.calls == ["start", "resolve", "close", "close"]


def test_scope_none_recorder_is_inert() -> None:
    scope_type = gguf_runner._PackedVerifyGpuStageRecorderScope
    timings: dict[str, float] = {}

    with scope_type(None, timings):
        timings["work"] = 1.0

    assert timings == {"work": 1.0}


def test_scope_none_recorder_preserves_body_exception() -> None:
    scope_type = gguf_runner._PackedVerifyGpuStageRecorderScope

    with pytest.raises(RuntimeError, match="work boom"):
        with scope_type(None, {}):
            raise RuntimeError("work boom")
