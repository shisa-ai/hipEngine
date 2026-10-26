"""CPU coverage for the PARO verifier graph cache lifecycle.

The resident session is constructed with ``__new__`` so these tests exercise the
shared graph-stage cache, the verifier binding, capture cleanup and the adapter's
route reporting without a HIP device. Every graph handle is a plain integer and
every runtime call is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.generation.qwen35_paro_mtp2 import Qwen35ParoMTP2Adapter
from hipengine.runtime import qwen35_paro_runner as runner_module
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoResidentSession,
    Qwen35ParoVerifierGraphEntry,
)
from hipengine.speculative.graph_stage import (
    GRAPH_STAGE_BINDING_MISS,
    SpecGraphStageCache,
    describe_failure,
)

_DEVICE = Device("hip", 0)


class _Runtime:
    """Record every graph/stream call, optionally failing one of them."""

    def __init__(
        self,
        *,
        fail: str | None = None,
        fail_times: int | None = None,
        strict: bool = False,
        exception: type[BaseException] = RuntimeError,
    ) -> None:
        self.calls: list[tuple[str, int]] = []
        self.destroyed: set[tuple[str, int]] = set()
        self.synchronized = 0
        self._fail = fail
        self._fail_times = fail_times
        self._failures_used = 0
        self._strict = bool(strict)
        self._exception = exception
        self._next_handle = 0x1000

    def _handle(self) -> int:
        self._next_handle += 0x100
        return self._next_handle

    def _record(self, name: str, handle: int) -> None:
        handle = int(handle)
        if self._strict and (name, handle) in self.destroyed:
            raise RuntimeError(f"{name} refused duplicate destroy for {handle:#x}")
        self.calls.append((name, handle))
        if name == self._fail and (
            self._fail_times is None or self._failures_used < self._fail_times
        ):
            self._failures_used += 1
            raise self._exception(f"{name} failed for handle {handle:#x}")
        if name.endswith("_destroy"):
            self.destroyed.add((name, handle))

    def attempts(self, name: str) -> list[int]:
        return [handle for call, handle in self.calls if call == name]

    def stream_create(self) -> int:
        handle = self._handle()
        self.calls.append(("stream_create", handle))
        return handle

    def stream_begin_capture(self, stream: int, mode: int = 2) -> None:
        self._record("stream_begin_capture", stream)

    def stream_end_capture(self, stream: int) -> int:
        self._record("stream_end_capture", stream)
        return self._handle()

    def graph_instantiate(self, graph: int) -> int:
        self._record("graph_instantiate", graph)
        return self._handle()

    def graph_launch(self, graph_exec: int, stream: int) -> None:
        self._record("graph_launch", graph_exec)

    def graph_exec_destroy(self, handle: int) -> None:
        self._record("graph_exec_destroy", handle)

    def graph_destroy(self, handle: int) -> None:
        self._record("graph_destroy", handle)

    def stream_destroy(self, handle: int) -> None:
        self._record("stream_destroy", handle)

    def stream_synchronize(self, stream: int) -> None:
        self.synchronized += 1

    def device_synchronize(self) -> None:
        self.synchronized += 1


class _Workspace:
    """Minimal named workspace exposing the allocation table the binding walks."""

    def __init__(self, pointers: dict[str, int]) -> None:
        self._pointers = {name: int(pointer) for name, pointer in pointers.items()}
        self.freed = 0

    def free(self) -> None:
        self.freed += 1

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._pointers)

    def allocation(self, name: str) -> object:
        return SimpleNamespace(buffer=DeviceBuffer(self._pointers[name], 128))


@dataclass
class _Scratch:
    """Dataclass scratch stand-in so pointer collection runs for real."""

    attn_input: Tensor
    qkv_z: Tensor
    tree_conv_state: Tensor


@dataclass
class _MoeScratch:
    normed: Tensor
    shared_out: Tensor


def _tensor(pointer: int, shape: tuple[int, ...] = (2, 8)) -> Tensor:
    return Tensor.from_handle(int(pointer), shape, DType.FP16, _DEVICE)


def _session(*, runtime: _Runtime | None = None) -> Qwen35ParoResidentSession:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.runtime = runtime or _Runtime()
    session.closed = False
    session.config = SimpleNamespace(hidden_size=8, num_hidden_layers=1)
    session.vocab_size = 32
    session.max_sequence_length = 4096
    session.block_size = 256
    session.decode_chunk_size = 256
    session.max_splits = 16
    session.max_batch_size = 2
    session.layer_limit = 1
    session.kv_storage_dtype = DType.BF16
    session.kv_storage_layout = "uniform"
    session.states = [SimpleNamespace(layer_weights=object())]
    session.runner = object()
    session.linear_states = {
        0: (
            _tensor(0x3000, (8, 4)),
            _tensor(0x3100, (2, 4, 4)),
            DeviceBuffer(0x3000, 128),
            DeviceBuffer(0x3100, 128),
            None,
            None,
        )
    }
    session.full_caches = {}
    session.full_cache_scales = {}
    session.linear_scratch = {
        0: _Scratch(
            attn_input=_tensor(0x4000),
            qkv_z=_tensor(0x4100),
            tree_conv_state=_tensor(0x4200),
        )
    }
    session.moe_scratch = {
        0: _MoeScratch(normed=_tensor(0x5000), shared_out=_tensor(0x5100))
    }
    session.full_scratch = {}
    session.prefill_linear_scratch = None
    session.prefill_full_scratch = None
    session.prefill_moe_scratch = None
    session.prefill_workspace = _Workspace(
        {"attn.decode_batch.partial_out": 0x6000, "attn.decode_batch.partial_m": 0x6100}
    )
    for name, pointer in (
        ("verify_token_ids_i64", 0x7000),
        ("verify_positions_i32", 0x7100),
        ("verify_top1_i32", 0x7200),
        ("verify_commit_rows", 0x7300),
        ("verify_accept_payload_i32", 0x7400),
        ("verify_capture_hidden_concat", 0x7500),
        ("prefill_block_table_buf", 0x7600),
        ("prefill_position_buf", 0x7700),
        ("prefill_context_count_buf", 0x7800),
        ("position_buf", 0x7900),
        ("context_buf", 0x7A00),
    ):
        setattr(session, name, DeviceBuffer(pointer, 128))
    session.verify_trunk_hidden = _tensor(0x7B00)
    session.verify_trunk_next_hidden = _tensor(0x7C00)
    session.cos = _tensor(0x7D00, (4096, 4))
    session.sin = _tensor(0x7E00, (4096, 4))
    session._verify_scratch_cache_generation = 0
    session._verify_linear_scratch_cache = {}
    session._verify_mlp_scratch_cache = {}
    session._verify_graphs = SpecGraphStageCache(max_entries=4)
    session._verify_graph_partials = []
    session._verify_graph_partial_buffers = []
    session._verify_graph_last_status = "unobserved"
    session._verify_graph_last_error = None
    return session


def _stub_launches(session: Qwen35ParoResidentSession) -> list[dict[str, object]]:
    launches: list[dict[str, object]] = []
    session._launch_verify_chain_forward_accept = lambda *args, **kwargs: launches.append(
        dict(kwargs)
    )
    session._read_verify_top1 = lambda rows: ((11, 12), (0.5, 0.25))
    session._read_verify_accept_payload = (
        lambda count, stream=0, already_synchronized=False: (1, 12, 11, 12)
    )
    session._native_spec_provider_target_commit_requested = (
        lambda *args, **kwargs: False
    )
    session._try_submit_native_spec_provider_target_graph = (
        lambda *args, **kwargs: (False, None)
    )
    return launches


def _direct_launches(launches: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the launches issued on the caller stream (not the capture stream)."""

    return [launch for launch in launches if int(launch.get("stream", 0) or 0) == 0]


def _batch(*, positions: tuple[int, ...] = (36, 37), mode: str = "verify_chain"):
    return SimpleNamespace(
        mode=str(mode),
        rows=len(positions),
        positions=tuple(int(position) for position in positions),
        request_ids=(7,),
    )


def _run(
    session: Qwen35ParoResidentSession,
    batch=None,
    *,
    graph_mode: str = "auto",
    capture_ids: tuple[int, ...] = (),
    capture_hidden_concat: Tensor | None = None,
    capture_row_start: int = 0,
):
    payload = batch if batch is not None else _batch()
    return session._run_verify_graph_or_direct(
        payload,
        base_slot=0,
        capture_ids=capture_ids,
        capture_hidden_concat=(
            _tensor(0x7500, (int(payload.rows), 0))
            if capture_hidden_concat is None
            else capture_hidden_concat
        ),
        capture_row_start=int(capture_row_start),
        rows=int(payload.rows),
        graph_mode=str(graph_mode),
        chain_attn_mode="decode_batched",
        linear_attn_mode="tree_tloop",
        stream=0,
    )


def test_paro_verify_graph_reuses_a_matching_capture() -> None:
    runtime = _Runtime()
    session = _session(runtime=runtime)
    launches = _stub_launches(session)

    first = _run(session)
    second = _run(session)
    third = _run(session)

    assert first["status"] == "captured"
    assert first["replayed"] is False
    assert second["status"] == "replayed"
    assert second["replayed"] is True
    assert third["replay_count"] == 2
    # One direct pass for the capture cycle, then no caller-stream launches.
    assert len(_direct_launches(launches)) == 1
    assert runtime.attempts("graph_instantiate") != []
    assert len(runtime.attempts("graph_launch")) == 2
    assert len(session._verify_graphs) == 1
    assert session.verify_graph_cleanup()["reusable"] == 1


def test_paro_verify_capture_does_not_claim_replay_or_validation() -> None:
    session = _session()
    _stub_launches(session)

    result = _run(session)

    assert result["status"] == "captured"
    assert result["replayed"] is False
    assert result["validation_passed"] is None
    assert result["native_spec_target_commit"] is False
    assert result["replay_count"] == 0


def test_paro_verify_validate_executes_the_graph_and_records_validation() -> None:
    session = _session()
    launches = _stub_launches(session)

    result = _run(session, graph_mode="validate")

    assert result["status"] == "captured_validated"
    assert result["replayed"] is True
    assert result["validation_passed"] is True
    # The direct pass ran once; the graph execution is a graph launch.
    assert len(_direct_launches(launches)) == 1
    assert len(session.runtime.attempts("graph_launch")) == 1


def test_paro_verify_validate_mismatch_falls_back_without_caching() -> None:
    runtime = _Runtime()
    session = _session(runtime=runtime)
    launches = _stub_launches(session)
    top1 = iter([((11, 12), (0.5, 0.25)), ((11, 13), (0.5, 0.25))])
    session._read_verify_top1 = lambda rows: next(top1)

    result = _run(session, graph_mode="validate")

    assert result["status"] == "capture_failed_fallback"
    assert result["replayed"] is False
    assert result["validation_passed"] is None
    assert "diverged" in str(result["fallback_reason"])
    assert len(session._verify_graphs) == 0
    # Direct pass for the capture cycle, then the restore pass.
    assert len(_direct_launches(launches)) == 2
    # Every handle the failed attempt acquired was released.
    assert runtime.attempts("graph_exec_destroy")
    assert runtime.attempts("graph_destroy")
    assert runtime.attempts("stream_destroy")


def test_paro_verify_graph_recaptures_after_scratch_drift() -> None:
    runtime = _Runtime()
    session = _session(runtime=runtime)
    launches = _stub_launches(session)

    assert _run(session)["status"] == "captured"
    entry = next(iter(session._verify_graphs._slots.values())).value
    stale = (entry.stream, entry.graph, entry.graph_exec)

    # Re-reserving one verifier scratch row changes a pointer the captured
    # kernels dereference. Nothing else about the cycle changed.
    session.linear_scratch = {
        0: _Scratch(
            attn_input=_tensor(0x9000),
            qkv_z=_tensor(0x9100),
            tree_conv_state=_tensor(0x9200),
        )
    }

    result = _run(session)

    assert result["status"] == "captured"
    assert result["replay_count"] == 0
    assert len(_direct_launches(launches)) == 2
    assert ("stream_destroy", stale[0]) in runtime.calls
    assert ("graph_destroy", stale[1]) in runtime.calls
    assert ("graph_exec_destroy", stale[2]) in runtime.calls
    assert (entry.stream, entry.graph, entry.graph_exec) == (0, 0, 0)


def test_paro_verify_graph_recaptures_after_prefill_workspace_drift() -> None:
    runtime = _Runtime()
    session = _session(runtime=runtime)
    _stub_launches(session)

    assert _run(session)["status"] == "captured"

    session.prefill_workspace = _Workspace(
        {"attn.decode_batch.partial_out": 0x9999, "attn.decode_batch.partial_m": 0x6100}
    )

    result = _run(session)

    assert result["status"] == "captured"
    assert len(runtime.attempts("graph_destroy")) == 1


def test_paro_verify_graph_split_drift_captures_a_separate_entry() -> None:
    runtime = _Runtime()
    session = _session(runtime=runtime)
    launches = _stub_launches(session)

    short = _run(session, _batch(positions=(36, 37)))
    # Crossing a split boundary changes the baked split-K reduction, so the
    # capture is keyed separately instead of replaying the narrower one.
    wide = _run(session, _batch(positions=(256, 257)))

    assert short["status"] == "captured"
    assert wide["status"] == "captured"
    assert short["bucket_key"]["attention_splits"] == [1]
    assert wide["bucket_key"]["attention_splits"] == [2]
    assert len(_direct_launches(launches)) == 2
    assert len(session._verify_graphs) == 2

    replay = _run(session, _batch(positions=(258, 259)))

    assert replay["status"] == "replayed"
    assert replay["bucket_key"]["attention_splits"] == [2]


def test_paro_verify_c1_loop_split_bucket_covers_every_row() -> None:
    session = _session()
    _stub_launches(session)
    batch = _batch(positions=(255, 256))

    bucket = session._verify_graph_split_bucket(
        batch, rows=2, chain_attn_mode="c1_loop"
    )

    # Row 0 still fits one chunk; row 1 needs the second split.
    assert bucket == (1, 2)
    # Prefill-style batched attention has no split axis at all.
    assert session._verify_graph_split_bucket(
        batch, rows=2, chain_attn_mode="batched"
    ) == ()


def test_paro_verify_graph_capture_failure_falls_back_and_stays_retryable() -> None:
    runtime = _Runtime(fail="graph_instantiate", fail_times=1)
    session = _session(runtime=runtime)
    launches = _stub_launches(session)

    failed = _run(session)

    assert failed["status"] == "capture_failed_fallback"
    assert failed["replayed"] is False
    assert "graph_instantiate failed" in str(failed["fallback_reason"])
    # Direct pass for the cycle plus the restore pass after the failure.
    assert len(_direct_launches(launches)) == 2
    assert len(session._verify_graphs) == 0
    assert runtime.attempts("graph_destroy")
    assert runtime.attempts("stream_destroy")
    assert session.verify_graph_cleanup()["last_status"] == "capture_fallback"

    retried = _run(session)

    assert retried["status"] == "captured"
    assert session._verify_graphs.last_failure(next(iter(session._verify_graphs.keys()))) is None


def test_paro_verify_capture_partial_release_is_retained_and_retried() -> None:
    runtime = _Runtime(fail="graph_destroy", fail_times=1, strict=True)
    session = _session(runtime=runtime)

    unresolved = session._release_capture_partial(
        graph_stream=0x2000,
        graph=0x3000,
        graph_exec=0x4000,
        capturing=False,
    )

    assert len(unresolved) == 1
    assert "graph_destroy failed" in describe_failure(unresolved[0])
    assert len(session._verify_graph_partials) == 1
    cleanup = session.verify_graph_cleanup()
    assert cleanup["pending_partial_releases"] == 1
    assert cleanup["pending_partial_handles"] == [[0x3000]]
    # The refused handle is owned, not dropped; the rest were released.
    assert runtime.attempts("graph_exec_destroy") == [0x4000]
    assert runtime.attempts("stream_destroy") == [0x2000]

    assert session._retry_verify_graph_partials() == []

    assert session._verify_graph_partials == []
    assert runtime.attempts("graph_destroy") == [0x3000, 0x3000]
    assert session.verify_graph_cleanup()["pending_partial_handles"] == []


def test_paro_verify_capture_refuses_to_allocate_behind_a_retained_partial() -> None:
    runtime = _Runtime(fail="graph_destroy", strict=True)
    session = _session(runtime=runtime)
    _stub_launches(session)
    session._release_capture_partial(
        graph_stream=0x2000,
        graph=0x3000,
        graph_exec=0x4000,
        capturing=False,
    )

    blocked = _run(session)

    assert blocked["status"] == "capture_failed_fallback"
    assert "graph_destroy failed" in str(blocked["fallback_reason"])
    # No new graph was allocated while the release backlog was outstanding.
    assert runtime.attempts("stream_create") == []
    assert len(session._verify_graph_partials) == 1

    runtime._fail = None
    retried = _run(session)

    assert retried["status"] == "captured"
    assert len(runtime.attempts("stream_create")) == 1
    assert session._verify_graph_partials == []


def test_paro_verify_graph_release_skips_destroyed_handles() -> None:
    entry = Qwen35ParoVerifierGraphEntry(
        rows=2,
        capture_width=0,
        base_slot=0,
        graph=0x3000,
        graph_exec=0x2000,
        stream=0x1000,
        validation_passed=None,
    )
    runtime = _Runtime(fail="graph_exec_destroy", fail_times=1, strict=True)

    with pytest.raises(BaseExceptionGroup) as first:
        entry.release(runtime)

    assert len(first.value.exceptions) == 1
    assert entry.graph_exec != 0
    assert entry.graph == 0 and entry.stream == 0
    assert entry.pending_handles() == (0x2000,)

    entry.release(runtime)

    assert entry.graph_exec == 0
    assert entry.pending_handles() == ()
    assert runtime.attempts("graph_exec_destroy") == [0x2000, 0x2000]
    assert runtime.attempts("graph_destroy") == [0x3000]
    assert runtime.attempts("stream_destroy") == [0x1000]


def test_paro_invalidate_releases_every_entry_and_reports_a_refusal() -> None:
    runtime = _Runtime()
    session = _session(runtime=runtime)
    _stub_launches(session)
    assert _run(session, _batch(positions=(36, 37)))["status"] == "captured"
    assert _run(session, _batch(positions=(256, 257)))["status"] == "captured"
    assert len(session._verify_graphs) == 2

    entry = next(iter(session._verify_graphs._slots.values())).value
    runtime._fail = "graph_destroy"
    runtime._fail_times = None

    session._invalidate_verify_graph_cache()

    assert len(session._verify_graphs) == 0
    # Each refusal is retained and reported instead of dropped.
    cleanup = session.verify_graph_cleanup()
    assert cleanup["pending_close_count"] == 2
    assert cleanup["close_failure_count"] >= 2
    assert "graph_destroy failed" in cleanup["close_failures"][0][1]
    assert entry.graph != 0

    runtime._fail = None
    session._invalidate_verify_graph_cache()

    assert session.verify_graph_cleanup()["pending_close_count"] == 0
    assert entry.graph == 0


def test_paro_verify_graph_binding_covers_capture_metadata_and_scratch() -> None:
    session = _session()
    _stub_launches(session)

    binding = session._verify_graph_binding(
        _batch(),
        rows=2,
        capture_hidden_concat=_tensor(0x7500, (2, 0)),
        capture_row_start=0,
        capture_ids=(),
        base_slot=0,
        chain_attn_mode="decode_batched",
        linear_attn_mode="tree_tloop",
        native_spec_target_commit=False,
        split_bucket=(1,),
    )
    addresses = dict(binding.addresses)

    for name in (
        "verify_capture_hidden_concat",
        "verify_top1_i32",
        "verify_accept_payload_i32",
        "verify_commit_rows",
        "prefill_block_table_buf",
        "prefill_position_buf",
        "prefill_context_count_buf",
        "position_buf",
        "context_buf",
        "cos",
        "sin",
        "verify_trunk_hidden",
        "linear_state[0][0]",
        "linear_scratch[0].attn_input",
        "linear_scratch[0].tree_conv_state",
        "moe_scratch[0].normed",
        "prefill_workspace.attn.decode_batch.partial_out",
    ):
        assert name in addresses, name
    assert binding.bucket[7] == (1,)
    assert binding.bucket[8:] == (0, ())
    assert id(session.states) in binding.configuration
    assert (id(session.states[0].layer_weights),) in binding.configuration


def test_paro_verify_graph_acquire_reports_a_binding_miss() -> None:
    session = _session()
    _stub_launches(session)
    batch = _batch()
    binding = session._verify_graph_binding(
        batch,
        rows=2,
        capture_hidden_concat=_tensor(0x7500, (2, 0)),
        capture_row_start=0,
        capture_ids=(),
        base_slot=0,
        chain_attn_mode="decode_batched",
        linear_attn_mode="tree_tloop",
        native_spec_target_commit=False,
        split_bucket=(1,),
    )
    entry = Qwen35ParoVerifierGraphEntry(
        rows=2,
        capture_width=0,
        base_slot=0,
        graph=0x3000,
        graph_exec=0x2000,
        stream=0x1000,
        validation_passed=None,
    )
    session._verify_graphs.capture("key", binding, lambda: entry, session._close_verify_graph)

    session.cos = _tensor(0xABCD, (4096, 4))
    drifted = session._verify_graph_binding(
        batch,
        rows=2,
        capture_hidden_concat=_tensor(0x7500, (2, 0)),
        capture_row_start=0,
        capture_ids=(),
        base_slot=0,
        chain_attn_mode="decode_batched",
        linear_attn_mode="tree_tloop",
        native_spec_target_commit=False,
        split_bucket=(1,),
    )

    reused, reason = session._verify_graphs.acquire("key", drifted)

    assert reused is None
    assert reason == GRAPH_STAGE_BINDING_MISS
    assert len(session._verify_graphs) == 0


def test_paro_verify_graph_keepalive_follows_the_cache() -> None:
    session = _session()
    _stub_launches(session)

    assert session._verify_graph_keepalive_active() is False
    assert _run(session)["status"] == "captured"
    assert session._verify_graph_keepalive_active() is True
    session._invalidate_verify_graph_cache()
    assert session._verify_graph_keepalive_active() is False


def test_paro_verify_graph_cleanup_reports_a_clean_teardown() -> None:
    session = _session()
    _stub_launches(session)
    assert _run(session)["status"] == "captured"

    session._invalidate_verify_graph_cache()

    cleanup = session.verify_graph_cleanup()
    assert cleanup["reusable"] == 0
    assert cleanup["pending_partial_releases"] == 0
    assert cleanup["pending_close"] == []
    assert cleanup["close_failure_count"] == 0
    assert cleanup["close_failures"] == []
    assert cleanup["last_status"] == "invalidated"


@pytest.mark.parametrize("route, mode", [("eager", "off"), ("graph", "auto")])
def test_paro_adapter_honors_the_planned_graph_route(route, mode) -> None:
    plan = SimpleNamespace(execution_route=route)

    assert Qwen35ParoMTP2Adapter._target_graph_mode(plan) == mode


@pytest.mark.parametrize(
    ("graph", "route"),
    [
        # A replay executes the graph and reads its outputs.
        ({"status": "replayed", "replayed": True}, "graph"),
        # A validate capture executes the graph and keeps its outputs.
        ({"status": "captured_validated", "replayed": True}, "graph"),
        # A recapture cycle executes the freshly captured graph.
        ({"status": "captured", "replayed": True}, "graph"),
        # A capture that never launched its graph leaves the direct outputs live.
        ({"status": "captured", "replayed": False}, "eager"),
        ({"status": "capture_failed_fallback", "replayed": False}, "eager"),
        ({"status": "disabled", "replayed": False}, "eager"),
        # A provider record without the executed signal still reports its status.
        ({"status": "replayed"}, "graph"),
        ({"status": "captured"}, "eager"),
    ],
)
def test_paro_adapter_reports_the_route_the_target_executed(graph, route) -> None:
    verify = SimpleNamespace(graph=graph)

    assert Qwen35ParoMTP2Adapter._executed_graph_route(verify) == route


def test_paro_adapter_reports_eager_without_a_graph_record() -> None:
    assert Qwen35ParoMTP2Adapter._executed_graph_route(SimpleNamespace(graph=None)) == "eager"


def test_paro_adapter_capability_declares_the_graph_route() -> None:
    row = SimpleNamespace(
        model_slot=0,
        native_greedy=True,
        first_token_emitted=True,
    )
    session = SimpleNamespace(max_sequence_length=4096)
    owner = SimpleNamespace(
        generator=SimpleNamespace(backend="hip_gfx1100", execution_profile="production"),
        _row=lambda request_id: row,
        _session=session,
    )
    adapter = Qwen35ParoMTP2Adapter(owner)
    adapter._intents[7] = 1
    adapter._states[7] = SimpleNamespace()
    from hipengine.speculative import SpeculativeRequestSemantics

    semantics = (
        SpeculativeRequestSemantics(
            request_id=7,
            sampling_mode="greedy",
            mode="verify_chain",
            context_tokens=128,
            remaining_decode=8,
        ),
    )

    capability = adapter.capability(semantics)

    assert capability is not None
    assert capability.graph_supported is True
    assert capability.eager_supported is True


def test_paro_verify_graph_module_constants_are_documented_defaults() -> None:
    assert runner_module._VERIFY_GRAPH_CACHE_MAX_ENTRIES >= 1
    assert "verify_capture_hidden_concat" in runner_module._VERIFY_GRAPH_SESSION_BUFFERS
    assert "prefill_block_table_buf" in runner_module._VERIFY_GRAPH_SESSION_BUFFERS


def test_paro_verify_graph_binding_covers_the_capture_destination() -> None:
    """The capture writes into the caller's tensor at a baked row offset."""

    session = _session()
    _stub_launches(session)
    destination = _tensor(0x8800, (4, 8))

    binding = session._verify_graph_binding(
        _batch(),
        rows=2,
        capture_hidden_concat=destination,
        capture_row_start=1,
        capture_ids=(3, 5),
        base_slot=0,
        chain_attn_mode="decode_batched",
        linear_attn_mode="tree_tloop",
        native_spec_target_commit=False,
        split_bucket=(1,),
    )

    assert dict(binding.addresses)["verify_capture_destination"] == 0x8800
    # The row offset and the layer->column axis are baked launch arguments.
    assert binding.bucket[-2:] == (1, (3, 5))


def test_paro_verify_graph_recaptures_after_capture_destination_drift() -> None:
    runtime = _Runtime()
    session = _session(runtime=runtime)
    _stub_launches(session)

    first = _run(
        session,
        capture_ids=(3,),
        capture_hidden_concat=_tensor(0x8800, (2, 8)),
    )
    # Same width and axes, different destination allocation: the recorded taps
    # would write into the previous tensor.
    second = _run(
        session,
        capture_ids=(3,),
        capture_hidden_concat=_tensor(0x8900, (2, 8)),
    )

    assert first["status"] == "captured"
    assert second["status"] == "captured"
    assert second["replay_count"] == 0
    # The stale capture was released, not replayed.
    assert len(runtime.attempts("graph_destroy")) == 1
    assert len(session._verify_graphs) == 1


def test_paro_verify_graph_recaptures_after_capture_axis_drift() -> None:
    session = _session()
    _stub_launches(session)
    destination = _tensor(0x8800, (4, 8))

    first = _run(
        session, capture_ids=(3,), capture_hidden_concat=destination, capture_row_start=0
    )
    # A different layer->column axis writes the taps somewhere else.
    second = _run(
        session, capture_ids=(5,), capture_hidden_concat=destination, capture_row_start=0
    )
    # A different row offset shifts every recorded tap.
    third = _run(
        session, capture_ids=(3,), capture_hidden_concat=destination, capture_row_start=2
    )

    assert first["status"] == "captured"
    assert second["status"] == "captured"
    assert second["replay_count"] == 0
    assert third["status"] == "captured"
    assert third["replay_count"] == 0
    assert second["bucket_key"]["capture_row_start"] == 0
    assert third["bucket_key"]["capture_row_start"] == 2
    assert third["bucket_key"]["capture_ids"] == [3]


def test_paro_verify_capture_binds_commit_tables_before_the_capture_stream() -> None:
    runtime = _Runtime()
    session = _session(runtime=runtime)
    _stub_launches(session)
    session._native_spec_provider_target_commit_requested = lambda *a, **k: True

    def tables():
        # A real table build mallocs and copies host->device, which HIP forbids
        # inside a capture. Record where it happened.
        runtime.calls.append(("commit_tables", 0))
        return (DeviceBuffer(0xA000, 128), DeviceBuffer(0xA100, 128))

    session._native_spec_provider_target_commit_tables = tables

    result = _run(session)

    assert result["status"] == "captured"
    names = [name for name, _handle in runtime.calls]
    assert "commit_tables" in names and "stream_begin_capture" in names
    assert names.index("commit_tables") < names.index("stream_begin_capture")


def test_paro_verify_capture_commit_table_failure_creates_no_stream() -> None:
    runtime = _Runtime()
    session = _session(runtime=runtime)
    _stub_launches(session)
    session._native_spec_provider_target_commit_requested = lambda *a, **k: True

    def tables():
        raise RuntimeError("state table build failed")

    session._native_spec_provider_target_commit_tables = tables

    result = _run(session)

    assert result["status"] == "capture_failed_fallback"
    assert "state table build failed" in str(result["fallback_reason"])
    # The prerequisite is fetched before anything is allocated or captured.
    assert runtime.attempts("stream_create") == []
    assert runtime.attempts("stream_begin_capture") == []
    assert session._verify_graph_partials == []


def test_paro_target_commit_tables_are_owned_by_the_graph_entry(monkeypatch) -> None:
    session = _session()
    session.buffers = []
    session.linear_layer_ids = [0]
    session._slot_linear_state = lambda layer_id, slot: (
        _tensor(0x3000, (8, 4)),
        _tensor(0x3100, (2, 4, 4)),
    )
    session.linear_scratch = {
        0: SimpleNamespace(
            tree_conv_state=_tensor(0x4200), tree_recurrent_state=_tensor(0x4300)
        )
    }
    allocated: list[int] = []
    freed: list[int] = []

    def fake_malloc(nbytes, runtime=None):
        buffer = DeviceBuffer(0xB000 + 0x100 * len(allocated), int(nbytes))
        allocated.append(buffer.ptr)
        return buffer

    monkeypatch.setattr(runner_module, "malloc", fake_malloc)
    monkeypatch.setattr(runner_module, "copy_host_array_to_device", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_module,
        "free",
        lambda buffer, runtime=None: freed.append(int(buffer.ptr)),
    )

    tables = session._native_spec_provider_target_commit_tables()

    assert tables is not None
    # The graph entry frees these when it retires, so the session lifetime list
    # must not own them as well.
    assert session.buffers == []
    entry = Qwen35ParoVerifierGraphEntry(
        rows=2,
        capture_width=0,
        base_slot=0,
        graph=0x3000,
        graph_exec=0x2000,
        stream=0x1000,
        validation_passed=None,
        native_spec_commit_source_table=tables[0],
        native_spec_commit_destination_table=tables[1],
    )

    entry.release(_Runtime())

    assert freed == [int(tables[0].ptr), int(tables[1].ptr)]
    # A resumed release must not free the same buffer twice.
    entry.release(_Runtime())
    assert len(freed) == 2


def test_paro_invalidate_surfaces_a_cleanup_interrupt_and_retains_the_handle() -> None:
    runtime = _Runtime(exception=KeyboardInterrupt)
    session = _session(runtime=runtime)
    _stub_launches(session)
    assert _run(session)["status"] == "captured"
    entry = next(iter(session._verify_graphs._slots.values())).value
    runtime._fail = "graph_destroy"

    with pytest.raises(KeyboardInterrupt):
        session._invalidate_verify_graph_cache()

    # The interrupt is not swallowed, and the handle stays owned for a retry.
    assert entry.graph != 0
    cleanup = session.verify_graph_cleanup()
    assert cleanup["pending_close_count"] == 1
    assert cleanup["last_status"] == "cleanup_failed"

    runtime._fail = None

    assert session.retry_verify_graph_cleanup() == []
    assert entry.graph == 0
    assert session.verify_graph_cleanup()["pending_close_count"] == 0


def test_paro_retry_verify_graph_cleanup_drains_after_close() -> None:
    runtime = _Runtime(fail="graph_destroy")
    session = _session(runtime=runtime)
    _stub_launches(session)
    assert _run(session)["status"] == "captured"
    entry = next(iter(session._verify_graphs._slots.values())).value

    # ``close`` sets ``closed`` first and then invalidates; the refusal below is
    # what a teardown leaves behind.
    session.closed = True
    session._invalidate_verify_graph_cache()

    assert session.verify_graph_cleanup()["pending_close_count"] == 1
    assert entry.graph != 0

    runtime._fail = None

    assert session.retry_verify_graph_cleanup() == []

    assert entry.graph == 0
    assert session.verify_graph_cleanup()["pending_close_count"] == 0
    attempts = len(runtime.attempts("graph_destroy"))
    # Repeated retries are idempotent and cannot destroy a released handle.
    assert session.retry_verify_graph_cleanup() == []
    assert len(runtime.attempts("graph_destroy")) == attempts


def test_paro_verify_graph_binding_covers_the_full_cache_scale_pointers() -> None:
    session = _session()
    _stub_launches(session)
    session.full_cache_scales = {
        0: (
            _tensor(0xC000, (8,)),
            _tensor(0xC100, (8,)),
            DeviceBuffer(0xC000, 128),
            DeviceBuffer(0xC100, 128),
        )
    }
    kwargs = dict(
        rows=2,
        capture_hidden_concat=_tensor(0x7500, (2, 0)),
        capture_row_start=0,
        capture_ids=(),
        base_slot=0,
        chain_attn_mode="decode_batched",
        linear_attn_mode="tree_tloop",
        native_spec_target_commit=False,
        split_bucket=(1,),
    )

    addresses = dict(session._verify_graph_binding(_batch(), **kwargs).addresses)

    assert addresses["full_cache_scale[0][0]"] == 0xC000
    assert addresses["full_cache_scale[0][1]"] == 0xC100

    session.full_cache_scales = {
        0: (
            _tensor(0xD000, (8,)),
            _tensor(0xD100, (8,)),
            DeviceBuffer(0xD000, 128),
            DeviceBuffer(0xD100, 128),
        )
    }
    drifted = dict(session._verify_graph_binding(_batch(), **kwargs).addresses)

    assert drifted["full_cache_scale[0][0]"] == 0xD000


def test_paro_verify_graph_binding_follows_weight_identity() -> None:
    session = _session()
    _stub_launches(session)
    kwargs = dict(
        rows=2,
        capture_hidden_concat=_tensor(0x7500, (2, 0)),
        capture_row_start=0,
        capture_ids=(),
        base_slot=0,
        chain_attn_mode="decode_batched",
        linear_attn_mode="tree_tloop",
        native_spec_target_commit=False,
        split_bucket=(1,),
    )

    before = session._verify_graph_binding(_batch(), **kwargs)
    states_id = id(session.states)

    session.states = [SimpleNamespace(layer_weights=object())]

    after = session._verify_graph_binding(_batch(), **kwargs)

    # Layer weights are materialized once and immutable afterwards, so their
    # object identity is the sound guard against a wholesale replacement.
    assert states_id in before.configuration
    assert before.configuration != after.configuration


def test_paro_verify_capture_only_dereferences_fingerprinted_buffers() -> None:
    """Every session buffer the capture launch bakes is in the fingerprint list."""

    import ast
    import pathlib

    source = pathlib.Path(runner_module.__file__).read_text()
    tree = ast.parse(source)
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen35ParoResidentSession"
    )
    methods = {
        node.name: node for node in owner.body if isinstance(node, ast.FunctionDef)
    }

    def callees(node):
        return {
            call.func.attr
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
        }

    reachable: set[str] = set()
    stack = ["_launch_verify_chain_forward_accept"]
    while stack:
        name = stack.pop()
        if name in reachable or name not in methods:
            continue
        reachable.add(name)
        stack.extend(callee for callee in callees(methods[name]) if callee in methods)

    referenced: set[str] = set()
    for name in reachable:
        for node in ast.walk(methods[name]):
            if not isinstance(node, ast.Attribute) or node.attr != "ptr":
                continue
            inner = node.value
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "self"
            ):
                referenced.add(inner.attr)

    assert len(reachable) > 10, sorted(reachable)
    missing = sorted(referenced - set(runner_module._VERIFY_GRAPH_SESSION_BUFFERS))
    assert missing == [], missing


def test_paro_verify_capture_failure_releases_the_commit_tables(monkeypatch) -> None:
    """A capture that dies after the prerequisite owns and frees the tables."""

    runtime = _Runtime(fail="graph_instantiate", fail_times=1)
    session = _session(runtime=runtime)
    _stub_launches(session)
    session._native_spec_provider_target_commit_requested = lambda *a, **k: True
    source = DeviceBuffer(0xA000, 128)
    destination = DeviceBuffer(0xA100, 128)
    session._native_spec_provider_target_commit_tables = lambda: (source, destination)
    freed: list[int] = []
    monkeypatch.setattr(
        runner_module,
        "free",
        lambda buffer, runtime=None: freed.append(int(buffer.ptr)),
    )

    result = _run(session)

    assert result["status"] == "capture_failed_fallback"
    # The entry that would have owned them never existed, so the capture's own
    # partial release frees each table exactly once.
    assert freed == [0xA000, 0xA100]
    assert session._verify_graph_partials == []
    assert len(session._verify_graphs) == 0


def _detached_allocation_patches(monkeypatch, *, copy_error, free_error=None):
    """Patch malloc/copy/free and return the allocation and free records."""

    allocated: list[int] = []
    freed: list[int] = []

    def fake_malloc(nbytes, runtime=None):
        buffer = DeviceBuffer(0xE000 + 0x100 * len(allocated), int(nbytes))
        allocated.append(buffer.ptr)
        return buffer

    def fake_copy(*args, **kwargs):
        if copy_error is not None:
            raise copy_error

    def fake_free(buffer, runtime=None):
        if free_error is not None:
            raise free_error
        freed.append(int(buffer.ptr))

    monkeypatch.setattr(runner_module, "malloc", fake_malloc)
    monkeypatch.setattr(runner_module, "copy_host_array_to_device", fake_copy)
    monkeypatch.setattr(runner_module, "free", fake_free)
    return allocated, freed


def test_paro_dev_detached_releases_the_buffer_when_the_copy_fails(monkeypatch) -> None:
    session = _session()
    allocated, freed = _detached_allocation_patches(
        monkeypatch, copy_error=RuntimeError("copy failed")
    )

    with pytest.raises(RuntimeError, match="copy failed"):
        session._dev_detached(np.zeros(2, dtype=np.uint64))

    # The allocation was made before the copy, so it must be released.
    assert allocated != []
    assert freed == allocated
    assert session._verify_graph_partial_buffers == []


def test_paro_dev_detached_retains_a_buffer_the_runtime_refuses_to_release(
    monkeypatch,
) -> None:
    session = _session()
    allocated, _freed = _detached_allocation_patches(
        monkeypatch,
        copy_error=RuntimeError("copy failed"),
        free_error=RuntimeError("free refused"),
    )

    with pytest.raises(RuntimeError) as raised:
        session._dev_detached(np.zeros(2, dtype=np.uint64))

    assert "free refused" in describe_failure(raised.value)
    # The refused buffer stays owned and reachable for a retry.
    assert len(session._verify_graph_partial_buffers) == 1
    assert session.verify_graph_cleanup()["pending_partial_handles"] == [allocated]

    monkeypatch.setattr(
        runner_module, "free", lambda buffer, runtime=None: _freed.append(int(buffer.ptr))
    )

    assert session.retry_verify_graph_cleanup() == []

    assert session._verify_graph_partial_buffers == []
    assert session.verify_graph_cleanup()["pending_partial_handles"] == []


def test_paro_target_commit_tables_release_the_first_when_the_second_fails(
    monkeypatch,
) -> None:
    session = _session()
    session.linear_layer_ids = [0]
    session._slot_linear_state = lambda layer_id, slot: (
        _tensor(0x3000, (8, 4)),
        _tensor(0x3100, (2, 4, 4)),
    )
    session.linear_scratch = {
        0: SimpleNamespace(
            tree_conv_state=_tensor(0x4200), tree_recurrent_state=_tensor(0x4300)
        )
    }
    allocated: list[int] = []
    freed: list[int] = []

    def fake_malloc(nbytes, runtime=None):
        if allocated:
            raise RuntimeError("second allocation failed")
        buffer = DeviceBuffer(0xB000, int(nbytes))
        allocated.append(buffer.ptr)
        return buffer

    monkeypatch.setattr(runner_module, "malloc", fake_malloc)
    monkeypatch.setattr(runner_module, "copy_host_array_to_device", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_module, "free", lambda buffer, runtime=None: freed.append(int(buffer.ptr))
    )

    with pytest.raises(RuntimeError, match="second allocation failed"):
        session._native_spec_provider_target_commit_tables()

    # The first table was already allocated, so the attempt owns and frees it.
    assert allocated == [0xB000]
    assert freed == [0xB000]
    assert session._verify_graph_partial_buffers == []


def test_paro_target_commit_tables_release_the_first_when_the_second_copy_fails(
    monkeypatch,
) -> None:
    session = _session()
    session.linear_layer_ids = [0]
    session._slot_linear_state = lambda layer_id, slot: (
        _tensor(0x3000, (8, 4)),
        _tensor(0x3100, (2, 4, 4)),
    )
    session.linear_scratch = {
        0: SimpleNamespace(
            tree_conv_state=_tensor(0x4200), tree_recurrent_state=_tensor(0x4300)
        )
    }
    allocated: list[int] = []
    freed: list[int] = []
    copies = 0

    def fake_malloc(nbytes, runtime=None):
        buffer = DeviceBuffer(0xB000 + 0x100 * len(allocated), int(nbytes))
        allocated.append(buffer.ptr)
        return buffer

    def fake_copy(*args, **kwargs):
        nonlocal copies
        copies += 1
        if copies == 2:
            raise RuntimeError("second copy failed")

    monkeypatch.setattr(runner_module, "malloc", fake_malloc)
    monkeypatch.setattr(runner_module, "copy_host_array_to_device", fake_copy)
    monkeypatch.setattr(
        runner_module, "free", lambda buffer, runtime=None: freed.append(int(buffer.ptr))
    )

    with pytest.raises(RuntimeError, match="second copy failed"):
        session._native_spec_provider_target_commit_tables()

    # Both the second table's failed copy and the first table are released.
    assert allocated == [0xB000, 0xB100]
    assert sorted(freed) == [0xB000, 0xB100]
    assert session._verify_graph_partial_buffers == []


def _close_ready_session(runtime: _Runtime, *, released: list[object]):
    """Build a session whose ``close`` reaches every teardown step."""

    session = _session(runtime=runtime)
    _stub_launches(session)
    session.states = [
        SimpleNamespace(
            layer_weights=object(), free=lambda: released.append("state")
        )
    ]
    session.allocations = [
        SimpleNamespace(free=lambda runtime=None: released.append("allocation"))
    ]
    session.buffers = [DeviceBuffer(0xEE00, 128)]
    session.prefill_workspace = _Workspace({"prefill.x": 0xAA00})
    session.prefill_hidden_buffer = None
    session._prefill_scratch_state = None
    return session


def test_paro_close_completes_session_cleanup_when_a_graph_release_interrupts(
    monkeypatch,
) -> None:
    runtime = _Runtime(exception=KeyboardInterrupt)
    released: list[object] = []
    session = _close_ready_session(runtime, released=released)
    monkeypatch.setattr(
        runner_module,
        "free",
        lambda buffer, runtime=None: released.append(("buffer", int(buffer.ptr))),
    )
    assert _run(session)["status"] == "captured"
    entry = next(iter(session._verify_graphs._slots.values())).value
    runtime._fail = "graph_destroy"

    with pytest.raises(KeyboardInterrupt):
        session.close()

    # The interrupt surfaced, and it did not strand the rest of the teardown.
    assert session.closed is True
    assert released == ["state", "allocation", ("buffer", 0xEE00)]
    assert session.prefill_workspace.freed == 1
    # The refused graph handle is still owned and retryable.
    assert entry.graph != 0
    assert session.verify_graph_cleanup()["pending_close_count"] == 1

    runtime._fail = None

    assert session.retry_verify_graph_cleanup() == []
    assert entry.graph == 0

    # A repeated close stays a no-op: nothing is freed a second time.
    session.close()

    assert released == ["state", "allocation", ("buffer", 0xEE00)]
    assert session.prefill_workspace.freed == 1


def test_paro_close_surfaces_a_refused_graph_release(monkeypatch) -> None:
    runtime = _Runtime(fail="graph_destroy")
    released: list[object] = []
    session = _close_ready_session(runtime, released=released)
    monkeypatch.setattr(
        runner_module,
        "free",
        lambda buffer, runtime=None: released.append(("buffer", int(buffer.ptr))),
    )
    assert _run(session)["status"] == "captured"

    with pytest.raises(RuntimeError, match="graph_destroy failed"):
        session.close()

    # A refused graph release is reported at close, not only recorded as status.
    assert session.verify_graph_cleanup()["last_status"] == "cleanup_failed"
    assert session.closed is True
    assert released == ["state", "allocation", ("buffer", 0xEE00)]
    assert session.prefill_workspace.freed == 1


@dataclass
class _ContainerScratch:
    """Scratch stand-in whose field holds a container instead of one tensor."""

    rows: list
    single: Tensor


def test_paro_graph_pointer_walk_follows_a_container_field_on_a_cached_scratch() -> None:
    """A cached plan describes field names only; nested addresses stay live."""

    first = _ContainerScratch(
        rows=[_tensor(0x8000), _tensor(0x8100)], single=_tensor(0x8200)
    )
    walked: dict[str, int] = {}
    runner_module._collect_graph_pointers("scratch[0]", first, walked)
    assert walked == {
        "scratch[0].rows[0]": 0x8000,
        "scratch[0].rows[1]": 0x8100,
        "scratch[0].single": 0x8200,
    }

    # A second instance of the same type reuses the plan and still reads live,
    # including a shorter container.
    second = _ContainerScratch(rows=[_tensor(0x9000)], single=_tensor(0x9100))
    again: dict[str, int] = {}
    runner_module._collect_graph_pointers("scratch[0]", second, again)
    assert again == {"scratch[0].rows[0]": 0x9000, "scratch[0].single": 0x9100}


def test_paro_graph_pointer_walk_records_int_pointers_and_skips_bools() -> None:
    class _IntPointer(int):
        pass

    class _SubclassPointer:
        ptr = _IntPointer(0x6000)

    class _BoolPointer:
        ptr = True

    walked: dict[str, int] = {}
    runner_module._collect_graph_pointers("subclass", _SubclassPointer(), walked)
    runner_module._collect_graph_pointers("flag", _BoolPointer(), walked)
    assert walked == {"subclass": 0x6000}


def test_paro_graph_pointer_walk_falls_back_when_a_scratch_grows_an_attribute() -> None:
    """An attribute the cached plan does not describe must not be dropped."""

    scratch = _Scratch(
        attn_input=_tensor(0x4000),
        qkv_z=_tensor(0x4100),
        tree_conv_state=_tensor(0x4200),
    )
    warm: dict[str, int] = {}
    runner_module._collect_graph_pointers("linear_scratch[0]", scratch, warm)
    assert len(warm) == 3

    scratch.replacement = _tensor(0x4300)
    walked: dict[str, int] = {}
    runner_module._collect_graph_pointers("linear_scratch[0]", scratch, walked)
    assert walked["linear_scratch[0].replacement"] == 0x4300
    assert set(warm) <= set(walked)


def test_paro_verify_graph_recaptures_after_a_scratch_field_is_replaced() -> None:
    """Same scratch instance, same structure, one re-reserved field."""

    runtime = _Runtime()
    session = _session(runtime=runtime)
    launches = _stub_launches(session)

    assert _run(session)["status"] == "captured"
    entry = next(iter(session._verify_graphs._slots.values())).value
    stale = (entry.stream, entry.graph, entry.graph_exec)

    session.linear_scratch[0].qkv_z = _tensor(0x9500)

    result = _run(session)

    assert result["status"] == "captured"
    assert result["replay_count"] == 0
    assert len(_direct_launches(launches)) == 2
    assert ("graph_destroy", stale[1]) in runtime.calls


@dataclass
class _SwapScratch:
    """Mutable scratch stand-in whose field set can change in place."""

    first: Tensor
    second: Tensor
    third: Tensor


@dataclass
class _DefaultSwapScratch:
    """Mutable scratch stand-in whose deleted field still has a class default."""

    first: Tensor
    second: Tensor
    third: object = None


def test_paro_graph_pointer_walk_falls_back_when_a_scratch_swaps_a_field() -> None:
    """Same member count, different names: the cached plan must not be used."""

    scratch = _SwapScratch(
        first=_tensor(0x4000), second=_tensor(0x4100), third=_tensor(0x4200)
    )
    warm: dict[str, int] = {}
    runner_module._collect_graph_pointers("swap[0]", scratch, warm)
    assert set(warm) == {"swap[0].first", "swap[0].second", "swap[0].third"}

    # One field removed, another added: the count a plan could compare stays the
    # same while the names do not.
    del scratch.third
    scratch.fourth = _tensor(0x4300)
    assert len(scratch.__dict__) == 3

    walked: dict[str, int] = {}
    runner_module._collect_graph_pointers("swap[0]", scratch, walked)
    assert walked == {
        "swap[0].first": 0x4000,
        "swap[0].second": 0x4100,
        "swap[0].fourth": 0x4300,
    }


def test_paro_graph_pointer_walk_ignores_a_class_default_behind_a_swap() -> None:
    """A swapped-out field must not resolve to its class default instead."""

    scratch = _DefaultSwapScratch(
        first=_tensor(0x5000), second=_tensor(0x5100), third=_tensor(0x5200)
    )
    warm: dict[str, int] = {}
    runner_module._collect_graph_pointers("swap[1]", scratch, warm)
    assert set(warm) == {"swap[1].first", "swap[1].second", "swap[1].third"}

    del scratch.third
    scratch.fourth = _tensor(0x5300)
    assert len(scratch.__dict__) == 3
    # The trap a member-count check steps into: the deleted name still resolves.
    assert scratch.third is None

    walked: dict[str, int] = {}
    runner_module._collect_graph_pointers("swap[1]", scratch, walked)
    assert walked == {
        "swap[1].first": 0x5000,
        "swap[1].second": 0x5100,
        "swap[1].fourth": 0x5300,
    }
