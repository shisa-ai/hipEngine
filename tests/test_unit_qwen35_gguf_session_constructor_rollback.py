"""Constructor rollback contracts for the GGUF resident session.

``Qwen35GGUFResidentSession.__post_init__`` acquires, in order, a full-stack
runner, a decode scratch owner, session buffers, a bulk-prefill workspace,
external-DMS state, and a prefill flight recorder. A failure at any boundary
must release everything already acquired, in reverse-safe order, exactly once,
without closing a borrowed runner and without replacing the original
exception.

These tests use a fake device, fake runner, fake scratch and fake DMS
components. They are CPU-only; no ROCm runtime or model file is required.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_runner as gguf_runner


class ConstructorFailure(Exception):
    """Synthetic failure raised at an injected acquisition boundary."""


class _FakeRuntime:
    def __init__(self, controller: "_Controller") -> None:
        self.controller = controller
        self.synchronizations = 0
        self.fail_host_get_pointer: BaseException | None = None
        self.fail_host_unregister: BaseException | None = None
        self.host_unregister_calls = 0

    def mem_get_info(self) -> tuple[int, int]:
        return (0, 0)

    def device_synchronize(self) -> None:
        self.synchronizations += 1
        if self.controller.fail_cleanup == "recorder-sync":
            raise ConstructorFailure("synthetic recorder synchronization failure")

    def host_register(self, *args, **kwargs) -> None:
        del args, kwargs

    def host_unregister(self, *args, **kwargs) -> None:
        del args, kwargs
        self.host_unregister_calls += 1
        if self.fail_host_unregister is not None:
            raise self.fail_host_unregister

    def host_get_device_pointer(self, pointer: int) -> int:
        if self.fail_host_get_pointer is not None:
            raise self.fail_host_get_pointer
        return int(pointer)


class _FakePool:
    def __init__(self, controller: "_Controller") -> None:
        self.controller = controller
        self.released: list[int] = []
        self.closed = 0

    def release(self, request_id: int, *, now_seconds: float) -> None:
        del now_seconds
        self.released.append(int(request_id))
        if self.controller.fail_cleanup in {
            "pool-release",
            "pool-release-and-close",
        }:
            raise ConstructorFailure("synthetic pool release failure")

    def close(self) -> None:
        self.closed += 1
        if self.controller.fail_cleanup in {
            "pool-close",
            "pool-release-and-close",
        }:
            raise ConstructorFailure("synthetic pool close failure")


class _FakeProjector:
    def __init__(self, controller: "_Controller") -> None:
        self.controller = controller
        self.closed = 0

    def close(self) -> None:
        self.closed += 1
        if self.controller.fail_cleanup == "projector-close":
            raise ConstructorFailure("synthetic projector close failure")


class _FakeRecorder:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        controller = _ACTIVE_CONTROLLER[0]
        if controller.fail_at == "recorder-init":
            raise controller.failure
        self.controller = controller
        self.closed = 0
        controller.recorders.append(self)

    def close(self) -> None:
        self.closed += 1
        if self.controller.fail_cleanup == "recorder-close":
            raise ConstructorFailure("synthetic recorder close failure")


class _FakeRunner:
    def __init__(self, controller: "_Controller") -> None:
        self.controller = controller
        self.backend = "hip_gfx1100"
        self.runtime = controller.runtime
        self.target_arch = "gfx1100"
        config = SimpleNamespace(
            context_length=256,
            layer_types=("full_attention",),
            is_moe=False,
            block_count=1,
            head_count=1,
            head_count_kv=1,
            key_length=128,
        )
        self.weights = SimpleNamespace(config=config, allocation_arena=None)
        self.hidden_size = 128
        self.vocab_size = 256
        self.fp16_recurrent_state = False
        self.verifier_prefill_gdn_capture = False
        self.router_f32w_coop = None
        self.router_f32w_persistent_counter = None
        self.q8_t16_rowtile_all = None
        self.verify_f32_residual = None
        self.verify_f32_post_norm = None
        self.q4_t16_dual_silu_r28 = None
        self.closed = 0

    def close(self) -> None:
        self.closed += 1
        if self.controller.fail_cleanup == "runner-close":
            raise ConstructorFailure("synthetic runner close failure")


class _FakeScratchOwner:
    def __init__(self, buffers: tuple[object, ...]) -> None:
        self.buffers = tuple(buffers)
        self.allocation_mode = "dedicated"
        self.slot_count = 1

    def for_slot(self, slot: int, *, span_role: str = "decode") -> SimpleNamespace:
        del slot, span_role
        return SimpleNamespace(max_positions=256, block_size=256)


class _FakeFullStackScratch:
    @classmethod
    def allocate(cls, runner, *, runtime, **kwargs):
        del runner, kwargs
        controller = _ACTIVE_CONTROLLER[0]
        if controller.fail_at == "scratch":
            raise controller.failure
        buffers: list[DeviceBuffer] = []
        try:
            for index in range(int(controller.scratch_buffer_count)):
                buffers.append(gguf_runner.malloc(32 + index, runtime=runtime))
        except BaseException:
            for buffer in reversed(buffers):
                gguf_runner.free(buffer, runtime=runtime)
            raise
        owner = _FakeScratchOwner(tuple(buffers))
        if controller.on_scratch_allocated is not None:
            controller.on_scratch_allocated(owner)
        return owner


class _Controller:
    def __init__(self) -> None:
        self.fail_at: str | None = None
        self.failure = ConstructorFailure("synthetic construction failure")
        self.fail_alloc_at: int | None = None
        self.fail_free_at: int | None = None
        self.fail_free_buffer: DeviceBuffer | None = None
        self.fail_cleanup: str | None = None
        self.scratch_buffer_count = 1
        self.on_scratch_allocated = None
        self.allocations: list[DeviceBuffer] = []
        self.freed: list[DeviceBuffer] = []
        self.alloc_attempts = 0
        self.free_attempts = 0
        self.next_ptr = 0x10000000
        self.runtime = _FakeRuntime(self)
        self.pool = _FakePool(self)
        self.projector = _FakeProjector(self)
        self.allocation = SimpleNamespace(request_id=7)
        self.recorders: list[_FakeRecorder] = []
        self.runners: list[_FakeRunner] = []


_ACTIVE_CONTROLLER: list[_Controller] = []


def _install_constructor_harness(monkeypatch: pytest.MonkeyPatch) -> _Controller:
    controller = _Controller()
    _ACTIVE_CONTROLLER[:] = [controller]

    def fake_malloc(nbytes: int, *, runtime):
        del runtime
        controller.alloc_attempts += 1
        if (
            controller.fail_alloc_at is not None
            and controller.alloc_attempts == controller.fail_alloc_at
        ):
            raise controller.failure
        buffer = DeviceBuffer(ptr=controller.next_ptr, nbytes=int(nbytes))
        controller.next_ptr += 256
        controller.allocations.append(buffer)
        return buffer

    def fake_free(buffer, *, runtime):
        del runtime
        controller.free_attempts += 1
        controller.freed.append(buffer)
        if (
            controller.fail_free_buffer is not None
            and buffer is controller.fail_free_buffer
        ):
            raise ConstructorFailure("synthetic targeted free failure")
        if (
            controller.fail_free_at is not None
            and controller.free_attempts == controller.fail_free_at
        ):
            raise ConstructorFailure("synthetic buffer free failure")

    def fake_runner_factory(*args, **kwargs):
        del args, kwargs
        if controller.fail_at == "runner":
            raise controller.failure
        runner = _FakeRunner(controller)
        controller.runners.append(runner)
        return runner

    def fake_allocate_bulk(self, runtime) -> None:
        if controller.fail_at == "bulk":
            raise controller.failure
        token = gguf_runner.malloc(8, runtime=runtime)
        hidden = None
        scratch_buffer = None
        try:
            hidden = gguf_runner.malloc(16, runtime=runtime)
            scratch_buffer = gguf_runner.malloc(32, runtime=runtime)
        except BaseException:
            # Mirrors the real workspace's existing partial-allocation
            # rollback for its own locals.
            if scratch_buffer is not None:
                gguf_runner.free(scratch_buffer, runtime=runtime)
            if hidden is not None:
                gguf_runner.free(hidden, runtime=runtime)
            gguf_runner.free(token, runtime=runtime)
            raise
        if controller.fail_at == "bulk-partial":
            self._prefill_token_buf = token
            self._prefill_hidden_a = hidden
            self._prefill_hidden_b = hidden
            self._buffers = (*self._buffers, token, hidden)
            gguf_runner.free(scratch_buffer, runtime=runtime)
            raise controller.failure
        self._prefill_token_buf = token
        self._prefill_hidden_a = hidden
        self._prefill_hidden_b = hidden
        self._bulk_prefill_scratch = SimpleNamespace(
            buffers=(scratch_buffer,),
            full_attn_split_growth_buffers=(),
        )
        self._buffers = (*self._buffers, token, hidden, scratch_buffer)

    def fake_initialize_dms(self, runtime) -> None:
        if controller.fail_at is None or not controller.fail_at.startswith("dms"):
            return
        self._dms_source = object()
        if controller.fail_at == "dms-source":
            raise controller.failure
        self._dms_decode_projector = controller.projector
        if controller.fail_at == "dms-projector":
            raise controller.failure
        self._dms_decode_decisions = gguf_runner.malloc(4, runtime=runtime)
        self._dms_decode_logits = gguf_runner.malloc(4, runtime=runtime)
        if controller.fail_at == "dms-buffers":
            raise controller.failure
        self._dms_dense_prefill_pool = controller.pool
        if controller.fail_at == "dms-pool":
            raise controller.failure
        self._device_kv_pool = controller.pool
        self._device_kv_allocation = controller.allocation
        if controller.fail_at == "dms-allocation":
            raise controller.failure

    def fake_reset(self, *, stream: int = 0) -> None:
        del stream
        if controller.fail_at == "reset":
            raise controller.failure

    def fake_resolve_decode_graph_min_replay_steps(self) -> int:
        if controller.fail_at == "late":
            raise controller.failure
        return 0

    def fake_unbind_device_kv_allocation(self):
        self._device_kv_allocation = None
        self._device_kv_pool = None
        return controller.allocation

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    monkeypatch.setattr(gguf_runner, "free", fake_free)
    monkeypatch.setattr(gguf_runner, "copy_host_array_to_device", lambda *a, **k: None)
    monkeypatch.setattr(gguf_runner, "get_hip_runtime", lambda: controller.runtime)
    monkeypatch.setattr(gguf_runner, "resolve_backend", lambda backend: "hip_gfx1100")
    monkeypatch.setattr(
        gguf_runner,
        "backend_package_capability",
        lambda backend, name, default=None: default,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_resolve_gguf_token_embedding_placement",
        lambda **kwargs: ("device", "test"),
    )
    monkeypatch.setattr(
        gguf_runner,
        "_resolve_gguf_private_c1_small_weight_arena",
        lambda **kwargs: (False, "test"),
    )
    monkeypatch.setattr(
        gguf_runner,
        "_resolve_gguf_private_c1_decode_scratch_arena",
        lambda **kwargs: (False, "test"),
    )
    monkeypatch.setattr(
        gguf_runner,
        "_resolve_target_verifier_production_q4_rowtile",
        lambda requested: False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_resolve_verifier_prefill_gdn_capture",
        lambda requested: False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "resolve_qwen35moe_fastpath_safety",
        lambda **kwargs: SimpleNamespace(
            effective_wmma_prefill=False,
            effective_gemv_decode=False,
        ),
    )
    monkeypatch.setattr(gguf_runner, "build_runtime_state", lambda **kwargs: object())
    monkeypatch.setattr(gguf_runner, "build_lm_head", lambda **kwargs: object())
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_verify_lm_head_q6_top1_dp4a_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "hip_target_arch_environment",
        lambda arch: nullcontext(),
    )
    monkeypatch.setattr(
        gguf_runner,
        "_qwen35_gguf_session_kv_chunk_layout",
        lambda session: object(),
    )
    monkeypatch.setattr(
        gguf_runner,
        "resolve_prefill_config_for_sequence",
        lambda *args, **kwargs: (SimpleNamespace(), {}),
    )
    monkeypatch.setattr(
        gguf_runner,
        "lm_head_argmax_stage1_blocks",
        lambda vocab_size, *, threads: 4,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_int8_kv_value_bf16_enabled",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_int8_kv_scale_granularity",
        lambda **kwargs: "per_token_head",
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_int8_bf16_prefix_full_attention_layers",
        lambda **kwargs: 0,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_int8_bf16_full_attention_layer_indices",
        lambda **kwargs: (),
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_int8_effective_scale_dtype",
        lambda **kwargs: DType.FP16,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_validate_gguf_int8_kv_context",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_admitted_no_mirror_int8_capability",
        lambda capability: False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_runnable_no_mirror_int8_capability",
        lambda capability: False,
    )
    monkeypatch.setattr(gguf_runner, "Qwen35GGUFFullStackRunner", fake_runner_factory)
    monkeypatch.setattr(gguf_runner, "_FullStackScratch", _FakeFullStackScratch)
    monkeypatch.setattr(gguf_runner, "PrefillFlightRecorder", _FakeRecorder)
    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_allocate_bulk_prefill_workspace",
        fake_allocate_bulk,
    )
    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_initialize_external_dms_serving",
        fake_initialize_dms,
    )
    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "reset",
        fake_reset,
    )
    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_resolve_decode_graph_min_replay_steps",
        fake_resolve_decode_graph_min_replay_steps,
    )
    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "unbind_device_kv_allocation",
        fake_unbind_device_kv_allocation,
    )
    return controller


def _construct(*, shared_runner=None, recorder_path=None, dms_path=None):
    return gguf_runner.Qwen35GGUFResidentSession(
        "unused",
        runtime=_ACTIVE_CONTROLLER[0].runtime,
        backend="hip_gfx1100",
        shared_runner=shared_runner,
        prefill_flight_recorder_path=recorder_path,
        dms_metadata_path=dms_path,
    )


def _assert_every_allocation_attempted_once(controller: _Controller) -> None:
    freed_ids = [id(buffer) for buffer in controller.freed]
    assert len(freed_ids) == len(set(freed_ids)), "a buffer was freed twice"
    assert {id(buffer) for buffer in controller.allocations} == set(freed_ids)


@pytest.mark.parametrize(
    "boundary",
    (
        "runner",
        "scratch",
        "bulk",
        "bulk-partial",
        "dms-source",
        "dms-projector",
        "dms-buffers",
        "dms-pool",
        "dms-allocation",
        "reset",
        "recorder-init",
        "late",
    ),
)
def test_constructor_failure_releases_each_boundary(
    monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    controller = _install_constructor_harness(monkeypatch)
    controller.fail_at = boundary

    with pytest.raises(ConstructorFailure) as caught:
        _construct(
            recorder_path=(
                "recorder.log"
                if boundary in {"late", "recorder-init"}
                else None
            ),
            dms_path="metadata.json" if boundary.startswith("dms") else None,
        )

    assert caught.value is controller.failure
    assert not getattr(caught.value, "__notes__", [])
    _assert_every_allocation_attempted_once(controller)

    if boundary != "runner":
        assert controller.runners
        assert controller.runners[0].closed == 1
    else:
        assert controller.runners == []

    if boundary in {"dms-projector", "dms-buffers", "dms-pool", "dms-allocation"}:
        assert controller.projector.closed == 1
    if boundary in {"dms-pool", "dms-allocation"}:
        assert controller.pool.closed == 1
    if boundary == "dms-allocation":
        assert controller.pool.released == [7]
    if boundary == "late":
        assert len(controller.recorders) == 1
        assert controller.recorders[0].closed == 1
    if boundary == "recorder-init":
        assert controller.recorders == []


def test_constructor_failure_releases_every_successful_acquisition_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Derive the successful-path allocation count from the real constructor
    # harness rather than pinning the current number of session fields.
    controller = _install_constructor_harness(monkeypatch)
    session = _construct()
    allocation_count = len(controller.allocations)
    assert allocation_count > 0
    session.close()
    _assert_every_allocation_attempted_once(controller)

    for index in range(1, allocation_count + 1):
        controller = _install_constructor_harness(monkeypatch)
        controller.fail_alloc_at = index
        with pytest.raises(ConstructorFailure) as caught:
            _construct()

        assert caught.value is controller.failure
        assert not getattr(caught.value, "__notes__", [])
        assert len(controller.allocations) == index - 1
        _assert_every_allocation_attempted_once(controller)
        assert controller.runners
        assert controller.runners[0].closed == 1


def test_borrowed_runner_is_not_closed_and_shared_weights_survive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_constructor_harness(monkeypatch)
    shared = _FakeRunner(controller)
    controller.fail_at = "reset"

    with pytest.raises(ConstructorFailure) as caught:
        _construct(shared_runner=shared)

    assert caught.value is controller.failure
    assert shared.closed == 0
    assert shared.weights is not None
    _assert_every_allocation_attempted_once(controller)


def test_repeated_failure_then_retry_and_successful_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_constructor_harness(monkeypatch)

    for _ in range(2):
        controller.fail_at = "bulk-partial"
        controller.allocations.clear()
        controller.freed.clear()
        controller.runners.clear()
        with pytest.raises(ConstructorFailure):
            _construct()
        _assert_every_allocation_attempted_once(controller)

    controller.fail_at = None
    controller.allocations.clear()
    controller.freed.clear()
    controller.runners.clear()
    session = _construct()
    assert controller.freed == []
    assert controller.runners[0].closed == 0
    assert session._buffers
    assert session._bulk_prefill_scratch is not None

    session.close()
    assert controller.runners[0].closed == 1
    _assert_every_allocation_attempted_once(controller)


def test_cleanup_errors_are_recorded_without_masking_the_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_constructor_harness(monkeypatch)
    controller.fail_at = "reset"
    controller.fail_free_at = 1
    controller.fail_cleanup = "runner-close"

    with pytest.raises(ConstructorFailure) as caught:
        _construct()

    assert caught.value is controller.failure
    notes = getattr(caught.value, "__notes__", [])
    assert any("buffer-free" in note for note in notes)
    assert any("synthetic buffer free failure" in note for note in notes)
    assert any("runner-close" in note for note in notes)
    assert any("synthetic runner close failure" in note for note in notes)
    _assert_every_allocation_attempted_once(controller)


def test_canonical_scratch_free_failure_attempts_later_buffers_and_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_constructor_harness(monkeypatch)
    controller.scratch_buffer_count = 3
    controller.fail_at = "reset"
    owners: list[_FakeScratchOwner] = []

    def select_middle(owner: _FakeScratchOwner) -> None:
        owners.append(owner)
        controller.fail_free_buffer = owner.buffers[1]

    controller.on_scratch_allocated = select_middle

    with pytest.raises(ConstructorFailure) as caught:
        _construct()

    assert caught.value is controller.failure
    assert owners
    scratch_buffers = owners[0].buffers
    # The middle release fails. The adapter must still attempt the buffer
    # before it (last in reverse order) and the runner close.
    assert all(buffer in controller.freed for buffer in scratch_buffers)
    assert controller.runners[0].closed == 1
    notes = _notes(controller.failure)
    assert any("state-buffer-free" in note for note in notes)
    assert any(
        f"0x{int(scratch_buffers[1].ptr):x}" in note for note in notes
    )
    _assert_every_allocation_attempted_once(controller)


def _install_dms_harness(monkeypatch: pytest.MonkeyPatch) -> _Controller:
    from hipengine.kvcache import dms as dms_module
    from hipengine.kvcache import dms_device as dms_device_module
    from hipengine.kvcache import dms_sidecar as dms_sidecar_module
    from hipengine.models import qwen35_dms as qwen35_dms_module

    real_initialize = (
        gguf_runner.Qwen35GGUFResidentSession._initialize_external_dms_serving
    )
    controller = _install_constructor_harness(monkeypatch)
    # The constructor harness replaces the method with a synthetic DMS
    # boundary. These tests exercise the real acquisition path instead.
    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_initialize_external_dms_serving",
        real_initialize,
    )
    config = SimpleNamespace(
        num_layers=2,
        num_kv_heads=1,
        window_size=4,
        physical_layer_ids=(0,),
    )
    monkeypatch.setattr(
        dms_module,
        "load_dms_retrofit_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        qwen35_dms_module,
        "resolve_qwen35_dms_decision_capability",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        dms_sidecar_module,
        "load_external_dms_sidecar",
        lambda source_config: SimpleNamespace(config=source_config),
    )
    monkeypatch.setattr(
        dms_device_module,
        "DMSExternalLinearDeviceProjector",
        lambda source, *, backend: controller.projector,
    )
    return controller


def _new_dms_session(controller: _Controller):
    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    session.__dict__.update(
        runner=_FakeRunner(controller),
        model_path="unused",
        dms_metadata_path="metadata.json",
        dms_decision_mode="sidecar",
        dms_prefill_mode="dense_pool",
        backend="hip_gfx1100",
        runtime=controller.runtime,
        _owns_runner=False,
        _dms_source=None,
        _dms_decode_projector=None,
        _dms_decode_decisions=None,
        _dms_decode_logits=None,
        _dms_dense_prefill_pool=None,
        _dms_pending_allocation=None,
        _dms_prefill_oracle_pair=None,
    )
    session.scratch = SimpleNamespace(max_positions=256)
    session.create_device_kv_pool = lambda **kwargs: controller.pool
    return session


def _notes(error: BaseException) -> list[str]:
    return list(getattr(error, "__notes__", ()))


def test_dms_pool_allocation_failure_is_released_by_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_dms_harness(monkeypatch)
    session = _new_dms_session(controller)

    def fail_allocate(*args, **kwargs):
        del args, kwargs
        raise controller.failure

    controller.pool.allocate = fail_allocate
    with pytest.raises(ConstructorFailure) as caught:
        session._initialize_external_dms_serving(controller.runtime)

    assert caught.value is controller.failure
    # The pool is still reachable by the rollback owner; the acquisition
    # method must not clear or close it locally.
    assert session._dms_dense_prefill_pool is controller.pool
    assert session._dms_pending_allocation is None
    assert controller.pool.released == []
    assert controller.pool.closed == 0

    session._rollback_construction(cause=caught.value)
    assert session._dms_dense_prefill_pool is None
    assert session._dms_pending_allocation is None
    assert controller.pool.released == []
    assert controller.pool.closed == 1
    _assert_every_allocation_attempted_once(controller)


def test_dms_bind_failure_releases_pending_allocation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_dms_harness(monkeypatch)
    session = _new_dms_session(controller)
    controller.pool.allocate = lambda request_id, pages: controller.allocation

    def fail_bind(*args, **kwargs):
        del args, kwargs
        raise controller.failure

    session.bind_device_kv_allocation = fail_bind
    with pytest.raises(ConstructorFailure) as caught:
        session._initialize_external_dms_serving(controller.runtime)

    assert caught.value is controller.failure
    assert session._dms_pending_allocation is controller.allocation
    assert session._dms_dense_prefill_pool is controller.pool
    assert controller.pool.released == []
    assert controller.pool.closed == 0

    session._rollback_construction(cause=caught.value)
    assert session._dms_pending_allocation is None
    assert controller.pool.released == [7]
    assert controller.pool.closed == 1
    _assert_every_allocation_attempted_once(controller)


def test_dms_bind_after_publication_releases_the_same_allocation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_dms_harness(monkeypatch)
    session = _new_dms_session(controller)
    controller.pool.allocate = lambda request_id, pages: controller.allocation

    def publish_then_fail(pool, allocation) -> None:
        session._device_kv_pool = pool
        session._device_kv_allocation = allocation
        raise controller.failure

    session.bind_device_kv_allocation = publish_then_fail
    with pytest.raises(ConstructorFailure) as caught:
        session._initialize_external_dms_serving(controller.runtime)

    assert caught.value is controller.failure
    assert session._dms_pending_allocation is controller.allocation
    assert session._device_kv_allocation is controller.allocation

    session._rollback_construction(cause=caught.value)
    assert session._dms_pending_allocation is None
    assert session._device_kv_allocation is None
    assert controller.pool.released == [7]
    assert controller.pool.closed == 1
    _assert_every_allocation_attempted_once(controller)


def test_dms_release_and_close_failures_are_recorded_on_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_dms_harness(monkeypatch)
    session = _new_dms_session(controller)
    controller.pool.allocate = lambda request_id, pages: controller.allocation
    controller.fail_cleanup = "pool-release-and-close"

    def fail_bind(*args, **kwargs):
        del args, kwargs
        raise controller.failure

    session.bind_device_kv_allocation = fail_bind
    with pytest.raises(ConstructorFailure) as caught:
        session._initialize_external_dms_serving(controller.runtime)

    session._rollback_construction(cause=caught.value)
    assert caught.value is controller.failure
    notes = _notes(caught.value)
    assert any("device-kv-release" in note for note in notes)
    assert any("synthetic pool release failure" in note for note in notes)
    assert any("dms-dense-prefill-pool" in note for note in notes)
    assert any("synthetic pool close failure" in note for note in notes)
    assert controller.pool.released == [7]
    assert controller.pool.closed == 1
    _assert_every_allocation_attempted_once(controller)


def _install_full_stack_runner_harness(monkeypatch: pytest.MonkeyPatch):
    controller = _Controller()

    monkeypatch.setattr(gguf_runner, "resolve_backend", lambda backend: "hip_gfx1100")
    monkeypatch.setattr(
        gguf_runner,
        "hip_target_arch_for_backend",
        lambda backend: "gfx1100",
    )
    monkeypatch.setattr(
        gguf_runner,
        "load_backend_kernel_package",
        lambda backend: None,
    )
    monkeypatch.setattr(gguf_runner, "resolve", lambda **kwargs: object())
    monkeypatch.setattr(gguf_runner, "get_hip_runtime", lambda: controller.runtime)
    monkeypatch.setattr(
        gguf_runner,
        "_resolve_verifier_prefill_gdn_capture",
        lambda requested: False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_shared_token_embedding_placement",
        lambda **kwargs: "device",
    )
    monkeypatch.setattr(
        gguf_runner,
        "_resolve_fp16_recurrent_state",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_dense_down_residual_decode_fused",
        lambda *args, **kwargs: False,
    )
    return controller


class _FakeWeights:
    def __init__(self, backend: str = "hip_gfx1100") -> None:
        self.backend = backend
        self.file_type_name = "MOSTLY_Q4_K_M"
        self.artifact_preset_key = None
        self.config = SimpleNamespace(
            feed_forward_length=256,
            hidden_size=128,
        )
        self.freed = 0

    def free(self, *, runtime) -> None:
        del runtime
        self.freed += 1


def test_full_stack_runner_failure_frees_materialized_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_full_stack_runner_harness(monkeypatch)
    weights = _FakeWeights()
    monkeypatch.setattr(
        gguf_runner,
        "materialize_qwen35_gguf_weights",
        lambda *args, **kwargs: weights,
    )

    def fail_after_materialization(self, **kwargs):
        del self, kwargs
        raise controller.failure

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFFullStackRunner,
        "_freeze_fp16_recurrent_state",
        fail_after_materialization,
    )

    with pytest.raises(ConstructorFailure) as caught:
        gguf_runner.Qwen35GGUFFullStackRunner(
            "unused",
            runtime=controller.runtime,
            backend="hip_gfx1100",
            token_embedding_placement="device",
        )

    assert caught.value is controller.failure
    assert weights.freed == 1
    assert not getattr(caught.value, "__notes__", [])


def test_full_stack_runner_late_failure_preserves_borrowed_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_full_stack_runner_harness(monkeypatch)
    borrowed = _FakeWeights()

    def fail_after_materialization(self, **kwargs):
        del self, kwargs
        raise controller.failure

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFFullStackRunner,
        "_freeze_fp16_recurrent_state",
        fail_after_materialization,
    )

    with pytest.raises(ConstructorFailure) as caught:
        gguf_runner.Qwen35GGUFFullStackRunner(
            "unused",
            runtime=controller.runtime,
            backend="hip_gfx1100",
            resident_weights=borrowed,
            token_embedding_placement="device",
        )

    assert caught.value is controller.failure
    assert borrowed.freed == 0


def test_full_stack_runner_backend_mismatch_does_not_free_borrowed_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_full_stack_runner_harness(monkeypatch)
    borrowed = _FakeWeights(backend="other_backend")

    with pytest.raises(ValueError, match="backend does not match"):
        gguf_runner.Qwen35GGUFFullStackRunner(
            "unused",
            runtime=controller.runtime,
            backend="hip_gfx1100",
            resident_weights=borrowed,
            token_embedding_placement="device",
        )

    assert borrowed.freed == 0


class _FailingWeights(_FakeWeights):
    def __init__(self, backend: str = "hip_gfx1100") -> None:
        super().__init__(backend)
        self.free_attempts = 0
        self.failure = ConstructorFailure("synthetic weights free failure")

    def free(self, *, runtime) -> None:
        del runtime
        self.free_attempts += 1
        raise self.failure


def _bare_full_stack_runner(controller: _Controller, weights: _FakeWeights):
    runner = object.__new__(gguf_runner.Qwen35GGUFFullStackRunner)
    runner.weights = weights
    runner.owns_resident_weights = True
    runner.runtime = controller.runtime
    runner.host_token_embedding_registered_ptr = 0
    runner.host_token_embedding_mapped_weight = None
    runner.host_token_embedding_mapped_storage = None
    runner.host_token_embedding_raw = object()
    runner.host_token_embedding_reader = object()
    return runner


def test_full_stack_runner_close_attempts_weights_after_mapping_failure() -> None:
    controller = _Controller()
    weights = _FakeWeights()
    runner = _bare_full_stack_runner(controller, weights)
    mapping_failure = ConstructorFailure("synthetic mapping cleanup failure")

    def fail_mapping() -> None:
        raise mapping_failure

    runner._unmap_host_token_embedding = fail_mapping
    with pytest.raises(ConstructorFailure) as caught:
        runner.close()

    assert caught.value is mapping_failure
    assert weights.freed == 1
    assert runner.weights is None
    assert runner.host_token_embedding_raw is None
    assert runner.host_token_embedding_reader is None


def test_full_stack_runner_close_attempts_mapping_after_weights_failure() -> None:
    controller = _Controller()
    weights = _FailingWeights()
    runner = _bare_full_stack_runner(controller, weights)
    mapping_calls: list[bool] = []
    runner._unmap_host_token_embedding = lambda: mapping_calls.append(True)

    with pytest.raises(ConstructorFailure) as caught:
        runner.close()

    assert caught.value is weights.failure
    assert mapping_calls == [True]
    assert weights.free_attempts == 1
    assert runner.weights is None


def test_full_stack_runner_close_records_secondary_failure() -> None:
    controller = _Controller()
    weights = _FailingWeights()
    runner = _bare_full_stack_runner(controller, weights)
    mapping_failure = ConstructorFailure("synthetic mapping cleanup failure")

    def fail_mapping() -> None:
        raise mapping_failure

    runner._unmap_host_token_embedding = fail_mapping
    with pytest.raises(ConstructorFailure) as caught:
        runner.close()

    assert caught.value is mapping_failure
    assert weights.free_attempts == 1
    notes = _notes(mapping_failure)
    assert any("additional runner close failure" in note for note in notes)
    assert any("synthetic weights free failure" in note for note in notes)


def test_full_stack_runner_rollback_records_close_failure_without_masking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_full_stack_runner_harness(monkeypatch)
    weights = _FailingWeights()
    monkeypatch.setattr(
        gguf_runner,
        "materialize_qwen35_gguf_weights",
        lambda *args, **kwargs: weights,
    )

    def fail_after_materialization(self, **kwargs):
        del self, kwargs
        raise controller.failure

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFFullStackRunner,
        "_freeze_fp16_recurrent_state",
        fail_after_materialization,
    )
    mapping_failure = ConstructorFailure("synthetic mapping cleanup failure")

    def fail_mapping(self) -> None:
        del self
        raise mapping_failure

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFFullStackRunner,
        "_unmap_host_token_embedding",
        fail_mapping,
    )

    with pytest.raises(ConstructorFailure) as caught:
        gguf_runner.Qwen35GGUFFullStackRunner(
            "unused",
            runtime=controller.runtime,
            backend="hip_gfx1100",
            token_embedding_placement="device",
        )

    assert caught.value is controller.failure
    assert weights.free_attempts == 1
    notes = _notes(controller.failure)
    assert any("construction cleanup failed" in note for note in notes)
    assert any("synthetic mapping cleanup failure" in note for note in notes)


def test_full_stack_runner_host_placement_failure_preserves_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_full_stack_runner_harness(monkeypatch)
    weights = _FakeWeights()
    monkeypatch.setattr(
        gguf_runner,
        "materialize_qwen35_gguf_weights",
        lambda *args, **kwargs: weights,
    )
    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFFullStackRunner,
        "_freeze_fp16_recurrent_state",
        lambda self, **kwargs: False,
    )
    weights.root = lambda name: (_ for _ in ()).throw(controller.failure)
    mapping_failure = ConstructorFailure("synthetic mapping cleanup failure")

    def fail_mapping(self) -> None:
        del self
        raise mapping_failure

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFFullStackRunner,
        "_unmap_host_token_embedding",
        fail_mapping,
    )

    with pytest.raises(ConstructorFailure) as caught:
        gguf_runner.Qwen35GGUFFullStackRunner(
            "unused",
            runtime=controller.runtime,
            backend="hip_gfx1100",
            token_embedding_placement="host",
        )

    assert caught.value is controller.failure
    assert weights.freed == 1
    notes = _notes(controller.failure)
    assert any("construction cleanup failed" in note for note in notes)
    assert any("synthetic mapping cleanup failure" in note for note in notes)


def test_host_embedding_pointer_failure_preserves_original_after_unregister_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _install_full_stack_runner_harness(monkeypatch)
    weights = _FakeWeights()
    monkeypatch.setattr(
        gguf_runner,
        "materialize_qwen35_gguf_weights",
        lambda *args, **kwargs: weights,
    )
    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFFullStackRunner,
        "_freeze_fp16_recurrent_state",
        lambda self, **kwargs: False,
    )
    raw = np.zeros(64, dtype=np.uint8)
    token_weight = SimpleNamespace(
        spec=SimpleNamespace(
            layout="raw_gguf",
            quant_key="gguf_q8_0",
            source=SimpleNamespace(name="token_embd.weight"),
        ),
        allocations={},
        backend="hip_gfx1100",
    )
    weights.root = lambda name: token_weight

    class _Reader:
        def __init__(self, path: str) -> None:
            del path

        def tensor_data(self, name: str) -> np.ndarray:
            del name
            return raw

    monkeypatch.setattr(gguf_runner, "GGUFReader", _Reader)
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_mapped_host_token_embedding_enabled",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_mapped_host_token_embedding_storage",
        lambda **kwargs: (kwargs["raw"], kwargs["raw"]),
    )
    mapping_failure = ConstructorFailure("synthetic host pointer failure")
    unregister_failure = ConstructorFailure("synthetic host unregister failure")
    controller.runtime.fail_host_get_pointer = mapping_failure
    controller.runtime.fail_host_unregister = unregister_failure

    with pytest.raises(ConstructorFailure) as caught:
        gguf_runner.Qwen35GGUFFullStackRunner(
            "unused",
            runtime=controller.runtime,
            backend="hip_gfx1100",
            token_embedding_placement="host",
        )

    assert caught.value is mapping_failure
    assert controller.runtime.host_unregister_calls == 1
    assert weights.freed == 1
    notes = _notes(mapping_failure)
    assert any("host-embedding unregister failed" in note for note in notes)
    assert any("synthetic host unregister failure" in note for note in notes)


# --- Real allocator ownership seams -----------------------------------------


def _scratch_config(case: str) -> SimpleNamespace:
    config = dict(
        context_length=512,
        layer_types=("linear_attention", "full_attention"),
        is_moe=False,
        block_count=2,
        head_count=2,
        head_count_kv=1,
        key_length=128,
        expert_used_count=1,
        expert_count=1,
        expert_shared_feed_forward_length=0,
        ssm_inner_size=64,
        ssm_time_step_rank=4,
        ssm_state_size=16,
        ssm_conv_kernel=4,
        rope_dimension_count=64,
        rope_freq_base=10000.0,
    )
    if case == "moe":
        config.update(
            is_moe=True,
            expert_used_count=2,
            expert_count=4,
            expert_shared_feed_forward_length=128,
        )
    return SimpleNamespace(**config)


class _ScratchRunner:
    def __init__(self, case: str) -> None:
        self.weights = SimpleNamespace(config=_scratch_config(case))
        self.hidden_size = 128
        self.ffn_size = 256
        self.q_width = 128
        self.kv_width = 128
        self.linear_qkv_width = 192
        self.fp16_recurrent_state = False


class _ScratchHarness:
    def __init__(self) -> None:
        self.runtime = object()
        self.failure = ConstructorFailure("synthetic scratch allocation failure")
        self.allocations: list[DeviceBuffer] = []
        self.freed: list[DeviceBuffer] = []
        self.alloc_attempts = 0
        self.free_attempts = 0
        self.copy_attempts = 0
        self.fail_alloc_at: int | None = None
        self.fail_alloc_exc: BaseException | None = None
        self.fail_free_at: int | None = None
        self.fail_copy_at: int | None = None
        self.next_ptr = 0x20000000


def _install_allocation_fakes(
    monkeypatch: pytest.MonkeyPatch, harness: _ScratchHarness
) -> None:
    def fake_malloc(nbytes: int, *, runtime):
        del runtime
        harness.alloc_attempts += 1
        if harness.fail_alloc_at == harness.alloc_attempts:
            raise (harness.fail_alloc_exc or harness.failure)
        buffer = DeviceBuffer(ptr=harness.next_ptr, nbytes=int(nbytes))
        harness.next_ptr += 256
        harness.allocations.append(buffer)
        return buffer

    def fake_free(buffer, *, runtime):
        del runtime
        harness.free_attempts += 1
        harness.freed.append(buffer)
        if harness.fail_free_at == harness.free_attempts:
            raise ConstructorFailure("synthetic scratch free failure")

    def fake_copy(*args, **kwargs):
        del args, kwargs
        harness.copy_attempts += 1
        if harness.fail_copy_at == harness.copy_attempts:
            raise harness.failure

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    monkeypatch.setattr(gguf_runner, "free", fake_free)
    monkeypatch.setattr(gguf_runner, "copy_host_array_to_device", fake_copy)


_SCRATCH_CASES = {
    "dense": {},
    "moe": {},
    "int8": {"kv_storage_dtype": DType.INT8_PER_TOKEN_HEAD},
    "deferred": {"allocate_kv_cache": False},
}


def _install_scratch_harness(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> tuple[_ScratchRunner, _ScratchHarness]:
    harness = _ScratchHarness()
    _install_allocation_fakes(monkeypatch, harness)
    return _ScratchRunner(case), harness


def _allocate_scratch(
    runner: _ScratchRunner,
    harness: _ScratchHarness,
    case: str,
    **overrides,
):
    kwargs = dict(_SCRATCH_CASES[case])
    kwargs.update(overrides)
    return gguf_runner._FullStackScratch.allocate(
        runner,
        runtime=harness.runtime,
        max_sequence_length=512,
        max_batch_size=1,
        **kwargs,
    )


def _assert_allocations_released_once(
    allocations: list[DeviceBuffer], freed: list[DeviceBuffer]
) -> None:
    freed_ids = [id(buffer) for buffer in freed]
    assert len(freed_ids) == len(set(freed_ids)), "a scratch buffer was freed twice"
    assert {id(buffer) for buffer in allocations} == set(freed_ids)


@pytest.mark.parametrize("case", tuple(_SCRATCH_CASES))
def test_full_stack_scratch_dedicated_success_keeps_owners(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    runner, harness = _install_scratch_harness(monkeypatch, case)
    owner = _allocate_scratch(runner, harness, case)

    assert owner.allocation_mode == "dedicated"
    assert len(owner.buffers) == len(harness.allocations)
    assert harness.freed == []
    assert harness.copy_attempts >= 5


@pytest.mark.parametrize("case", tuple(_SCRATCH_CASES))
def test_full_stack_scratch_every_allocation_failure_releases_prior_owners(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    _runner, baseline = _install_scratch_harness(monkeypatch, case)
    _allocate_scratch(_runner, baseline, case)
    allocation_count = len(baseline.allocations)
    assert allocation_count > 1

    for index in range(1, allocation_count + 1):
        runner, harness = _install_scratch_harness(monkeypatch, case)
        harness.fail_alloc_at = index
        with pytest.raises(ConstructorFailure) as caught:
            _allocate_scratch(runner, harness, case)

        assert caught.value is harness.failure
        assert not _notes(caught.value)
        assert len(harness.allocations) == index - 1
        _assert_allocations_released_once(harness.allocations, harness.freed)


@pytest.mark.parametrize("case", tuple(_SCRATCH_CASES))
def test_full_stack_scratch_every_copy_failure_releases_acquired_owners(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    _runner, baseline = _install_scratch_harness(monkeypatch, case)
    _allocate_scratch(_runner, baseline, case)
    copy_count = baseline.copy_attempts
    assert copy_count >= 5

    for index in range(1, copy_count + 1):
        runner, harness = _install_scratch_harness(monkeypatch, case)
        harness.fail_copy_at = index
        with pytest.raises(ConstructorFailure) as caught:
            _allocate_scratch(runner, harness, case)

        assert caught.value is harness.failure
        assert not _notes(caught.value)
        assert harness.allocations
        _assert_allocations_released_once(harness.allocations, harness.freed)


@pytest.mark.parametrize("case", tuple(_SCRATCH_CASES))
def test_full_stack_scratch_mid_setup_failure_releases_acquired_owners(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    runner, harness = _install_scratch_harness(monkeypatch, case)

    def fail_spans(*args, **kwargs):
        del args, kwargs
        raise harness.failure

    monkeypatch.setattr(gguf_runner.KVLiveSpans, "paged_uniform", fail_spans)
    with pytest.raises(ConstructorFailure) as caught:
        _allocate_scratch(runner, harness, case)

    assert caught.value is harness.failure
    assert harness.allocations
    _assert_allocations_released_once(harness.allocations, harness.freed)


@pytest.mark.parametrize("case", tuple(_SCRATCH_CASES))
def test_full_stack_scratch_constructor_failure_releases_every_owner(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    _runner, baseline = _install_scratch_harness(monkeypatch, case)
    _allocate_scratch(_runner, baseline, case)
    allocation_count = len(baseline.allocations)

    runner, harness = _install_scratch_harness(monkeypatch, case)

    def fail_init(self, *args, **kwargs):
        del self, args, kwargs
        raise harness.failure

    monkeypatch.setattr(gguf_runner._FullStackScratch, "__init__", fail_init)
    with pytest.raises(ConstructorFailure) as caught:
        _allocate_scratch(runner, harness, case)

    assert caught.value is harness.failure
    assert len(harness.allocations) == allocation_count
    _assert_allocations_released_once(harness.allocations, harness.freed)


def test_full_stack_scratch_cleanup_failure_preserves_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner, baseline = _install_scratch_harness(monkeypatch, "dense")
    _allocate_scratch(_runner, baseline, "dense")
    allocation_count = len(baseline.allocations)

    runner, harness = _install_scratch_harness(monkeypatch, "dense")
    harness.fail_alloc_at = allocation_count
    harness.fail_free_at = 1

    with pytest.raises(ConstructorFailure) as caught:
        _allocate_scratch(runner, harness, "dense")

    assert caught.value is harness.failure
    assert harness.free_attempts == len(harness.allocations)
    assert {id(buffer) for buffer in harness.allocations} == {
        id(buffer) for buffer in harness.freed
    }
    notes = _notes(caught.value)
    assert any("allocation rollback failed" in note for note in notes)
    assert any("synthetic scratch free failure" in note for note in notes)


def test_full_stack_scratch_single_arena_owns_only_the_arena(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, harness = _install_scratch_harness(monkeypatch, "dense")
    owner = _allocate_scratch(runner, harness, "dense", use_single_arena=True)

    assert owner.allocation_mode == "single_arena"
    assert len(harness.allocations) == 1
    assert owner.buffers == (harness.allocations[0],)
    assert harness.freed == []


def test_full_stack_scratch_single_arena_failure_releases_only_the_arena(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, harness = _install_scratch_harness(monkeypatch, "dense")
    harness.fail_copy_at = 1

    with pytest.raises(ConstructorFailure) as caught:
        _allocate_scratch(runner, harness, "dense", use_single_arena=True)

    assert caught.value is harness.failure
    assert len(harness.allocations) == 1
    assert len(harness.freed) == 1
    assert harness.freed[0] is harness.allocations[0]


def test_full_stack_scratch_failed_arena_falls_back_to_dedicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, harness = _install_scratch_harness(monkeypatch, "dense")
    harness.fail_alloc_at = 1
    harness.fail_alloc_exc = MemoryError("synthetic arena allocation failure")
    owner = _allocate_scratch(runner, harness, "dense", use_single_arena=True)

    assert owner.allocation_mode == "dedicated"
    assert len(owner.buffers) == len(harness.allocations)
    assert harness.alloc_attempts == len(harness.allocations) + 1


def test_full_stack_scratch_failed_arena_fallback_failure_releases_dedicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, harness = _install_scratch_harness(monkeypatch, "dense")
    harness.fail_alloc_at = 1
    harness.fail_alloc_exc = MemoryError("synthetic arena allocation failure")
    harness.fail_copy_at = 1

    with pytest.raises(ConstructorFailure) as caught:
        _allocate_scratch(runner, harness, "dense", use_single_arena=True)

    assert caught.value is harness.failure
    assert harness.allocations
    _assert_allocations_released_once(harness.allocations, harness.freed)


def test_prefill_hidden_second_allocation_failure_releases_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _ScratchHarness()
    _install_allocation_fakes(monkeypatch, harness)
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_prefill_hidden_buffer_count",
        lambda runner, *, rows: 2,
    )
    harness.fail_alloc_at = 2

    with pytest.raises(ConstructorFailure) as caught:
        gguf_runner._allocate_prefill_hidden_buffers(
            object(),
            rows=4,
            nbytes=64,
            runtime=harness.runtime,
        )

    assert caught.value is harness.failure
    assert len(harness.allocations) == 1
    _assert_allocations_released_once(harness.allocations, harness.freed)


def test_prefill_hidden_cleanup_failure_preserves_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _ScratchHarness()
    _install_allocation_fakes(monkeypatch, harness)
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_prefill_hidden_buffer_count",
        lambda runner, *, rows: 2,
    )
    harness.fail_alloc_at = 2
    harness.fail_free_at = 1

    with pytest.raises(ConstructorFailure) as caught:
        gguf_runner._allocate_prefill_hidden_buffers(
            object(),
            rows=4,
            nbytes=64,
            runtime=harness.runtime,
        )

    assert caught.value is harness.failure
    assert harness.free_attempts == 1
    notes = _notes(caught.value)
    assert any("allocation rollback failed" in note for note in notes)
    assert any("synthetic scratch free failure" in note for note in notes)


def test_prefill_hidden_single_plane_aliases_without_a_second_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _ScratchHarness()
    _install_allocation_fakes(monkeypatch, harness)

    hidden_a, hidden_b = gguf_runner._allocate_prefill_hidden_buffers(
        object(),
        rows=4,
        nbytes=64,
        runtime=harness.runtime,
        single_plane=True,
    )

    assert hidden_a is hidden_b
    assert harness.allocations == [hidden_a]
    assert harness.freed == []


def test_prefill_hidden_count_failure_releases_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _ScratchHarness()
    _install_allocation_fakes(monkeypatch, harness)

    def fail_count(runner, *, rows):
        del runner, rows
        raise harness.failure

    monkeypatch.setattr(
        gguf_runner,
        "_gguf_prefill_hidden_buffer_count",
        fail_count,
    )

    with pytest.raises(ConstructorFailure) as caught:
        gguf_runner._allocate_prefill_hidden_buffers(
            object(),
            rows=4,
            nbytes=64,
            runtime=harness.runtime,
        )

    assert caught.value is harness.failure
    assert len(harness.allocations) == 1
    _assert_allocations_released_once(harness.allocations, harness.freed)


@dataclass
class _FakePrefillScratch:
    buffers: tuple[object, ...]
    head_major_key_cache: object | None = None
    head_major_value_cache: object | None = None
    head_major_kv_capacity: int = 0


class _FakePrefillScratchAllocator:
    @classmethod
    def allocate(cls, runner, *, runtime, **kwargs):
        del runner, kwargs
        return _FakePrefillScratch(
            buffers=(gguf_runner.malloc(64, runtime=runtime),)
        )


def _install_bulk_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    head_major: bool = True,
    hidden_planes: int = 2,
) -> tuple[object, _ScratchHarness]:
    harness = _ScratchHarness()
    _install_allocation_fakes(monkeypatch, harness)
    session = object.__new__(gguf_runner.Qwen35GGUFResidentSession)
    session.runner = _ScratchRunner("dense")
    session.scratch = SimpleNamespace(max_positions=256)
    session._bulk_prefill_scratch = None
    session._int8_prefill_lifetime_plan = SimpleNamespace(
        mode="chunk_outer",
        required_hidden_capacity=128,
    )
    session.prefill_chunk_tuning = {}
    session.dms_prefill_mode = "dense_pool"
    session.use_expert_sidecar = False
    session.dms_metadata_path = None
    session.defer_kv_allocation = False
    session._runtime_state_library = object()
    session.backend = "hip_gfx1100"
    session._buffers = ()
    session._prefill_scratch_rows = lambda capacity: 128
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_bulk_prefill_full_prompt_capacity",
        lambda **kwargs: 128,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_bulk_prefill_initial_hidden_capacity",
        lambda **kwargs: 128,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_bulk_prefill_hidden_plane_count",
        lambda **kwargs: hidden_planes,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_prefill_hidden_buffer_count",
        lambda runner, *, rows: 2,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_GGUFFullAttentionPrefillScratch",
        _FakePrefillScratchAllocator,
    )

    if head_major:
        def allocate_head_major(**kwargs):
            del kwargs
            return (
                gguf_runner.malloc(32, runtime=harness.runtime),
                gguf_runner.malloc(32, runtime=harness.runtime),
            )
    else:
        def allocate_head_major(**kwargs):
            del kwargs
            return None

    monkeypatch.setattr(
        gguf_runner,
        "_try_allocate_gguf_aotriton_head_major_kv_scratch",
        allocate_head_major,
    )
    return session, harness


def test_bulk_prefill_success_publishes_every_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, harness = _install_bulk_harness(monkeypatch)
    session._allocate_bulk_prefill_workspace(harness.runtime)

    assert session._prefill_token_buf is harness.allocations[0]
    assert session._prefill_hidden_a is harness.allocations[1]
    assert session._prefill_hidden_b is harness.allocations[2]
    assert session._bulk_prefill_scratch.buffers
    assert session._bulk_prefill_scratch.head_major_key_cache is not None
    assert session._bulk_prefill_scratch.head_major_value_cache is not None
    assert harness.freed == []
    assert set(id(buffer) for buffer in session._buffers) == set(
        id(buffer) for buffer in harness.allocations
    )


def test_bulk_prefill_second_hidden_failure_releases_token_and_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, harness = _install_bulk_harness(monkeypatch)
    harness.fail_alloc_at = 3

    with pytest.raises(ConstructorFailure) as caught:
        session._allocate_bulk_prefill_workspace(harness.runtime)

    assert caught.value is harness.failure
    assert session._bulk_prefill_scratch is None
    assert len(harness.allocations) == 2
    _assert_allocations_released_once(harness.allocations, harness.freed)


def test_bulk_prefill_token_allocation_failure_preserves_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, harness = _install_bulk_harness(monkeypatch)
    harness.fail_alloc_at = 1

    with pytest.raises(ConstructorFailure) as caught:
        session._allocate_bulk_prefill_workspace(harness.runtime)

    assert caught.value is harness.failure
    assert harness.allocations == []
    assert harness.freed == []
    assert session._bulk_prefill_scratch is None


def test_bulk_prefill_first_hidden_failure_releases_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, harness = _install_bulk_harness(monkeypatch)
    harness.fail_alloc_at = 2

    with pytest.raises(ConstructorFailure) as caught:
        session._allocate_bulk_prefill_workspace(harness.runtime)

    assert caught.value is harness.failure
    assert len(harness.allocations) == 1
    _assert_allocations_released_once(harness.allocations, harness.freed)


def test_bulk_prefill_scratch_allocation_failure_releases_token_and_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, harness = _install_bulk_harness(monkeypatch)

    class _FailingScratchAllocator:
        @classmethod
        def allocate(cls, runner, *, runtime, **kwargs):
            del cls, runner, runtime, kwargs
            raise harness.failure

    monkeypatch.setattr(
        gguf_runner,
        "_GGUFFullAttentionPrefillScratch",
        _FailingScratchAllocator,
    )

    with pytest.raises(ConstructorFailure) as caught:
        session._allocate_bulk_prefill_workspace(harness.runtime)

    assert caught.value is harness.failure
    assert len(harness.allocations) == 3
    _assert_allocations_released_once(harness.allocations, harness.freed)


def test_bulk_prefill_hidden_alias_failure_frees_hidden_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, harness = _install_bulk_harness(monkeypatch, hidden_planes=1)

    def fail_replace(*args, **kwargs):
        del args, kwargs
        raise harness.failure

    monkeypatch.setattr(gguf_runner, "replace", fail_replace)

    with pytest.raises(ConstructorFailure) as caught:
        session._allocate_bulk_prefill_workspace(harness.runtime)

    assert caught.value is harness.failure
    hidden = harness.allocations[1]
    assert sum(buffer is hidden for buffer in harness.freed) == 1
    _assert_allocations_released_once(harness.allocations, harness.freed)


def test_bulk_prefill_head_major_failure_releases_scratch_and_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, harness = _install_bulk_harness(monkeypatch)

    def fail_replace(*args, **kwargs):
        del args, kwargs
        raise harness.failure

    monkeypatch.setattr(gguf_runner, "replace", fail_replace)

    with pytest.raises(ConstructorFailure) as caught:
        session._allocate_bulk_prefill_workspace(harness.runtime)

    assert caught.value is harness.failure
    assert session._bulk_prefill_scratch is None
    assert session._prefill_token_buf is None
    assert session._prefill_hidden_a is None
    assert session._prefill_hidden_b is None
    _assert_allocations_released_once(harness.allocations, harness.freed)


def test_bulk_prefill_head_major_allocation_failure_releases_scratch_and_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, harness = _install_bulk_harness(monkeypatch, head_major=False)

    def fail_head_major(**kwargs):
        del kwargs
        raise harness.failure

    monkeypatch.setattr(
        gguf_runner,
        "_try_allocate_gguf_aotriton_head_major_kv_scratch",
        fail_head_major,
    )

    with pytest.raises(ConstructorFailure) as caught:
        session._allocate_bulk_prefill_workspace(harness.runtime)

    assert caught.value is harness.failure
    assert session._bulk_prefill_scratch is None
    _assert_allocations_released_once(harness.allocations, harness.freed)


def test_bulk_prefill_cleanup_failure_preserves_original_and_attempts_each_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, harness = _install_bulk_harness(monkeypatch)

    def fail_replace(*args, **kwargs):
        del args, kwargs
        raise harness.failure

    monkeypatch.setattr(gguf_runner, "replace", fail_replace)
    harness.fail_free_at = 1

    with pytest.raises(ConstructorFailure) as caught:
        session._allocate_bulk_prefill_workspace(harness.runtime)

    assert caught.value is harness.failure
    assert harness.free_attempts == len(harness.allocations)
    assert {id(buffer) for buffer in harness.allocations} == {
        id(buffer) for buffer in harness.freed
    }
    notes = _notes(caught.value)
    assert any("allocation rollback failed" in note for note in notes)
    assert any("synthetic scratch free failure" in note for note in notes)
