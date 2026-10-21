"""Layer-outer DMS prefill must release the bulk workspace it no longer reads.

The two-pass layer-outer prefill used to leave the full-capacity bulk prefill
workspace (and the pass-1 sidecar score storage) alive after the compact store
was complete, so it coexisted with the verifier workspace during
``finish_prompt_streaming``. The dense-pool finalize already retains the
published trunk row and then releases the bulk workspace; the layer-outer
route now does the same, and the collector can drop its projection inputs once
the decisions and the cold snapshot are captured.

CPU-testable ownership contract (fake device):
- on success the order is: pass 1, decision finalize, pass 2, cold capture,
  collector input release, retain published row, release bulk workspace;
- the published trunk row is moved to the persistent hidden buffer before the
  workspace that backed it is freed;
- a missing persistent row buffer is allocated before workspace release;
- the collector input release is idempotent, keeps ``_decisions`` for pass-2
  packing, and ``close`` stays safe afterwards;
- pass 2 can never project new decisions from a released collector;
- a failed pass 2 closes the partially built compact store and leaves the
  workspace for the next prefill.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import dms_prefill_cache
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import _ExternalDMSDevicePrefillCollector

from tests.test_unit_gguf_bulk_prefill_workspace_release import _fake_session


class _FakeCollector:
    """Minimal stand-in for the device collector; records its own lifetime."""

    requires_teacher_logits = False

    def __init__(self, events, *, decisions):
        self.events = events
        self._decisions = DeviceBuffer(0x70000000, int(np.prod(decisions.shape)))
        self._logits = DeviceBuffer(0x71000000, int(np.prod(decisions.shape)) * 4)
        self._projector = None
        self._closed = False
        self._decisions_host = decisions
        self._decision_stride = int(decisions.shape[0] * decisions.shape[2])
        self.release_calls = 0

    def finalize(self, *, stream: int = 0):
        self.events.append("finalize")
        return self._decisions_host

    def decision_ptr(self, compact_layer_index: int) -> int:
        return self._decisions.ptr + int(compact_layer_index) * self._decision_stride

    def release_decision_inputs(self) -> None:
        self.events.append("release_inputs")
        self.release_calls += 1
        self._logits = None

    def close(self) -> None:
        self.events.append("close")
        self._closed = True


class _FakeBackend:
    def __init__(self, events, *, fail_finalize: bool = False):
        self.events = events
        self.fail_finalize = fail_finalize
        self.closed = False

    def estimate(self, request, *args, **kwargs):  # noqa: ARG002 - signature match
        return SimpleNamespace(request_id=0, tokens=0)

    def reserve(self, claims) -> None:  # noqa: ARG002 - signature match
        return None

    def finalize_device_streaming_pack(self, request_id, *, eviction, tokens):
        del request_id, eviction, tokens
        if self.fail_finalize:
            raise RuntimeError("compact pack failed")

    def close(self) -> None:
        self.events.append("backend_close")
        self.closed = True


def _dms_source(*, layers: int = 1, tokens: int = 4, kv_heads: int = 1):
    return SimpleNamespace(
        config=SimpleNamespace(
            num_layers=layers,
            num_kv_heads=kv_heads,
            window_size=tokens,
            head_dim=256,
            physical_layer_ids=tuple(range(layers)),
            # These fixtures cover the threshold policy's two-pass lifetime.
            # Exact-budget one-pass ordering is tested separately.
            prefill_selection_mode="threshold",
            target_compression_ratio=2,
        ),
        compact_layer_index=lambda physical_layer: int(physical_layer),
    )


def _layer_outer_dms_session(
    monkeypatch, *, capacity: int = 1024, fail_finalize: bool = False
):
    """A standalone layer-outer DMS owner with a live bulk workspace."""

    session, freed = _fake_session(monkeypatch, capacity=capacity)
    events: list[str] = []
    tokens = 4
    # ``finalize`` returns token-major decisions: [tokens, layers, kv_heads].
    decisions = np.zeros((tokens, 1, 1), dtype=bool)
    decisions[:, 0, 0] = (0, 1, 0, 1)
    collector = _FakeCollector(events, decisions=decisions)
    backend = _FakeBackend(events, fail_finalize=fail_finalize)
    session.__dict__.update(
        dms_prefill_mode="layer_outer",
        dms_metadata_path="fake-metadata.json",
        dms_max_new_tokens=2,
        dms_decision_mode="sidecar",
        dms_backend_factory=lambda **kwargs: backend,
        dms_metadata_source=None,
        _dms_source=_dms_source(tokens=tokens),
        _dms_prefill_oracle_pair=None,
        _dms_layer_outer_pack=None,
        _dms_cache_layers={},
        _dms_backend=None,
        _dms_cold_prefill=None,
        _resident_batch_owner=None,
        _resident_slot_views=(),
    )
    runtime = SimpleNamespace(
        memcpy_async=lambda *args, **kwargs: events.append("retain_copy"),
        device_synchronize=lambda: None,
        stream_synchronize=lambda stream: None,
    )
    session.runtime = runtime
    session._hidden_a = DeviceBuffer(0x40000000, session.runner.hidden_size * 2)
    session._last_target_hidden_ptr = 0
    session._allocate_bulk_prefill_workspace(runtime)
    workspace_ptr = int(session._prefill_hidden_a.ptr)

    def _bulk_prefill_and_sample(token_ids, **kwargs):
        events.append("pass2" if "dms_pack_sink" in kwargs else "pass1")
        assert "dms_pack_sink" not in kwargs or kwargs["dms_pack_sink"] is session
        # The second pass publishes the final trunk row from the bulk plane.
        session._last_target_hidden_ptr = (
            workspace_ptr + (len(token_ids) - 1) * session.runner.hidden_size * 2
        )
        return SimpleNamespace(logits=np.zeros(4, dtype=np.float32))

    session._run_bulk_prefill_and_sample = _bulk_prefill_and_sample
    return session, collector, backend, events, freed


def _record_ownership_calls(monkeypatch, session, events):
    real_retain = session._retain_published_target_hidden
    real_release = session._release_bulk_prefill_workspace

    def retain(**kwargs):
        events.append("retain")
        return real_retain(**kwargs)

    def release(**kwargs):
        events.append("release_workspace")
        return real_release(**kwargs)

    session._retain_published_target_hidden = retain
    session._release_bulk_prefill_workspace = release


def test_success_path_releases_bulk_workspace_after_retaining_last_row(monkeypatch) -> None:
    session, collector, backend, events, freed = _layer_outer_dms_session(monkeypatch)
    _record_ownership_calls(monkeypatch, session, events)
    hidden = session._hidden_a
    workspace = (
        *session._bulk_prefill_scratch.buffers,
        session._prefill_token_buf,
        session._prefill_hidden_a,
    )

    def capture(session_arg, collector_arg, token_ids):  # noqa: ARG001
        events.append("cold_capture")
        # The snapshot copies the sidecar scores, so the collector inputs must
        # still be alive when it runs.
        assert collector_arg._logits is not None
        assert collector_arg._decisions is not None
        return "cold-sentinel"

    monkeypatch.setattr(dms_prefill_cache, "capture_cold_prefill", capture)
    session._dms_cache_capture_tokens = (1, 2)

    result = session._prefill_layer_outer_dms(
        collector,
        [1, 2, 3, 4],
        bulk_attention_mode="bulk",
        return_logits=True,
        record_gpu_stage_timings=False,
    )

    assert result is not None
    assert events == [
        "pass1",
        "finalize",
        "pass2",
        "cold_capture",
        "release_inputs",
        "retain",
        "retain_copy",
        "release_workspace",
        "close",
    ]
    # Last row preserved: the published pointer now names the persistent buffer.
    assert session._last_target_hidden_ptr == int(hidden.ptr)
    # Bulk workspace released exactly once and pruned from the ledger.
    assert session._bulk_prefill_scratch is None
    assert session._prefill_hidden_a is None
    assert session._prefill_hidden_b is None
    assert session._prefill_token_buf is None
    for buffer in workspace:
        assert freed.count(buffer) == 1, "workspace buffer freed exactly once"
        assert buffer not in session._buffers
    # Session ownership moved to the compact store; cold capture kept.
    assert session._dms_backend is backend
    assert session.runner._dms_decode_owner is session
    assert session._dms_cold_prefill == "cold-sentinel"
    assert session._dms_layer_outer_pack is None
    assert collector.release_calls == 1
    assert collector._closed is True


def test_missing_persistent_hidden_gets_a_row_before_compact_restore(monkeypatch) -> None:
    session, collector, backend, events, freed = _layer_outer_dms_session(monkeypatch)
    _record_ownership_calls(monkeypatch, session, events)
    session._hidden_a = None
    workspace = session._prefill_hidden_a
    finalize = backend.finalize_device_streaming_pack

    def restore(*args, **kwargs):
        assert session._hidden_a is not None
        assert session._last_target_hidden_ptr == session._hidden_a.ptr
        assert session._bulk_prefill_scratch is None
        assert session._prefill_hidden_a is None
        return finalize(*args, **kwargs)

    backend.finalize_device_streaming_pack = restore
    session._prefill_layer_outer_dms(
        collector,
        [1, 2, 3, 4],
        bulk_attention_mode="bulk",
        return_logits=True,
        record_gpu_stage_timings=False,
    )

    assert "retain_copy" in events
    assert "release_workspace" in events
    assert freed.count(workspace) == 1
    assert session._hidden_a.nbytes == session.runner.hidden_size * 2
    assert session._hidden_a in session._buffers
    assert session._hidden_a not in freed


@pytest.mark.parametrize(
    "failure_method",
    ["release_decision_inputs", "_retain_published_target_hidden", "_release_bulk_prefill_workspace"],
)
def test_final_cleanup_failure_closes_unpublished_compact_store(monkeypatch, failure_method) -> None:
    session, collector, backend, events, _freed = _layer_outer_dms_session(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("final cleanup failed")

    owner = collector if failure_method == "release_decision_inputs" else session
    monkeypatch.setattr(owner, failure_method, fail)
    with pytest.raises(RuntimeError, match="final cleanup failed"):
        session._prefill_layer_outer_dms(
            collector, [1, 2, 3, 4], bulk_attention_mode="bulk",
            return_logits=True, record_gpu_stage_timings=False,
        )
    assert backend.closed
    assert session._dms_backend is None
    assert collector._closed
    assert "backend_close" in events


def test_compact_layers_restore_only_after_bulk_and_oracle_release(monkeypatch) -> None:
    session, collector, backend, events, _freed = _layer_outer_dms_session(monkeypatch)
    session._release_dms_prefill_oracle = lambda: events.append("release_oracle")
    original_finalize = backend.finalize_device_streaming_pack

    def finalize(*args, **kwargs):
        assert session._bulk_prefill_scratch is None
        assert "release_oracle" in events
        return original_finalize(*args, **kwargs)

    backend.finalize_device_streaming_pack = finalize
    session._prefill_layer_outer_dms(
        collector, [1, 2, 3, 4], bulk_attention_mode="bulk",
        return_logits=True, record_gpu_stage_timings=False,
    )
    assert session._dms_backend is backend


def test_packed_layer_is_synchronized_and_host_staged(monkeypatch) -> None:
    session, collector, backend, events, _freed = _layer_outer_dms_session(monkeypatch)
    session._dms_layer_outer_pack = (backend, collector, 4)
    session._dms_layer_outer_oracle_pair = lambda: (DeviceBuffer(100, 2048), DeviceBuffer(200, 2048))
    session.runtime.stream_synchronize = lambda stream: events.append(("sync", stream))
    backend.device_streaming_pack_layer = lambda *args, **kwargs: events.append("pack")
    backend.stage_device_prefill_layer = lambda layer: events.append(("stage", layer))
    session.pack_dms_prefill_layer(0, stream=17)
    assert events == [("sync", 17), "pack", ("sync", 17), ("stage", 0)]


@pytest.mark.parametrize("phase", ["pass2", "restore_finalize"])
def test_failed_pack_closes_store_and_preserves_workspace_lifetime(monkeypatch, phase) -> None:
    session, collector, backend, events, freed = _layer_outer_dms_session(
        monkeypatch, fail_finalize=phase == "restore_finalize"
    )
    if phase == "pass2":
        original = session._run_bulk_prefill_and_sample

        def run(token_ids, **kwargs):
            if "dms_pack_sink" in kwargs:
                raise RuntimeError("compact pack failed")
            return original(token_ids, **kwargs)

        session._run_bulk_prefill_and_sample = run
    _record_ownership_calls(monkeypatch, session, events)
    workspace = session._prefill_hidden_a

    with pytest.raises(RuntimeError, match="compact pack failed"):
        session._prefill_layer_outer_dms(
            collector, [1, 2, 3, 4], bulk_attention_mode="bulk",
            return_logits=True, record_gpu_stage_timings=False,
        )

    released = phase == "restore_finalize"
    assert ("release_workspace" in events) == released
    assert ("release_inputs" in events) == released
    assert backend.closed is True
    assert collector._closed is True
    assert session._dms_backend is None
    assert (session._bulk_prefill_scratch is None) == released
    assert session._prefill_hidden_a is (None if released else workspace)
    assert freed.count(workspace) == int(released)


def _bare_collector(monkeypatch, *, decision_mode: str = "sidecar"):
    """A collector with only the decision-score inputs a pass-1 sweep allocates."""

    allocated: list[DeviceBuffer] = []
    freed: list[DeviceBuffer] = []

    def fake_malloc(nbytes: int, *, runtime):
        buffer = DeviceBuffer(0x50000000 + 4096 * len(allocated), int(nbytes))
        allocated.append(buffer)
        return buffer

    monkeypatch.setattr(gguf_runner, "malloc", fake_malloc)
    monkeypatch.setattr(gguf_runner, "free", lambda buffer, *, runtime=None: freed.append(buffer))

    class _Projector:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    projector = _Projector() if decision_mode == "sidecar" else None
    collector = object.__new__(_ExternalDMSDevicePrefillCollector)
    collector.__dict__.update(
        source=SimpleNamespace(
            compact_layer_index=lambda physical_layer: int(physical_layer),
            config=SimpleNamespace(num_layers=2, num_kv_heads=1),
        ),
        physical_layer_ids=(0, 1),
        hidden_size=5120,
        num_q_heads=24,
        num_kv_heads=1,
        head_dim=256,
        input_stage="attn_norm",
        token_count=4,
        decision_mode=decision_mode,
        _diagnostic_injection=None,
        _runtime=SimpleNamespace(),
        _projector=projector,
        _decisions=fake_malloc(2 * 4 * 1, runtime=None),
        _logits=fake_malloc(2 * 4 * 1 * 4, runtime=None) if decision_mode == "sidecar" else None,
        _next=np.zeros(2, dtype=np.int32),
        _closed=False,
    )
    return collector, freed, projector


def test_release_decision_inputs_is_idempotent_and_keeps_decisions(monkeypatch) -> None:
    collector, freed, projector = _bare_collector(monkeypatch)
    logits = collector._logits
    decisions = collector._decisions
    first_decision_ptr = collector.decision_ptr(1)

    collector.release_decision_inputs()
    assert collector._logits is None
    assert projector.closed == 1
    assert freed == [logits]
    # Decisions and layer metadata survive for pass-2 packing.
    assert collector._decisions is decisions
    assert collector.decision_ptr(1) == first_decision_ptr
    assert collector._closed is False

    # Idempotent: a second call frees nothing and does not re-close.
    collector.release_decision_inputs()
    assert projector.closed == 1
    assert freed == [logits]

    # Pass 2 must never project new decisions from a released collector.
    with pytest.raises(RuntimeError, match="collector is incomplete"):
        collector.capture_device_chunk(
            physical_layer_id=0,
            compact_layer_index=0,
            start=0,
            rows=4,
            hidden_ptr=0x60000000,
            stream=0,
        )

    # close stays safe after the release: decisions freed once, nothing else.
    collector.close()
    assert freed == [logits, decisions]
    collector.close()
    assert freed == [logits, decisions]
    assert projector.closed == 1


def test_release_decision_inputs_handles_no_evict_mode(monkeypatch) -> None:
    collector, freed, projector = _bare_collector(monkeypatch, decision_mode="no_evict")
    assert collector._logits is None
    assert projector is None
    collector.release_decision_inputs()
    assert freed == []
    assert collector._decisions is not None
    collector.close()
    assert len(freed) == 1
