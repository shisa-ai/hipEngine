from __future__ import annotations

import ctypes
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from hipengine.speculative.native_cycle import (
    NATIVE_SPEC_CYCLE_ABI_VERSION,
    NativeSpecCycleControl,
    NativeSpecCycleControlC,
    NativeSpecCycleDType,
    NativeSpecCycleError,
    NativeSpecCycleKVLiveSpanPointers,
    NativeSpecCycleMetadataPointers,
    NativeSpecCycleMode,
    NativeSpecCycleOutputPointers,
    NativeSpecCyclePointers,
    NativeSpecCycleResultC,
    NativeSpecCycleShape,
    NativeSpecCycleStage,
    NativeSpecCycleStatePointers,
    NativeSpecCycleStatus,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.speculative.native_cycle_graph import (
    NativeSpecProposalGraphLauncher,
    NativeSpecProviderTargetGraphLauncher,
    NativeSpecTargetGraphLauncher,
    build_native_spec_cycle_graph_launcher,
    create_native_spec_proposal_graph_launcher,
    create_native_spec_provider_target_graph_launcher,
    create_native_spec_target_graph_launcher,
    plan_native_spec_cycle_graph_launcher_build,
    register_native_spec_provider_target_graph,
)


def _b2_control(*, cycle_id: int = 7) -> NativeSpecCycleControl:
    return NativeSpecCycleControl(
        cycle_id=cycle_id,
        transaction_id=23,
        stages=NativeSpecCycleStage.VERIFY,
        mode=NativeSpecCycleMode.CHAIN,
        shape=NativeSpecCycleShape(
            request_count=1,
            request_capacity=1,
            row_count=3,
            active_row_count=3,
            row_capacity=3,
            candidate_count=2,
            active_candidate_count=2,
            candidate_capacity=2,
            candidate_budget=2,
            span_count=3,
            span_capacity=3,
            max_live_count=31,
            context_bucket=128,
            hidden_size=2048,
            hidden_row_capacity=3,
            output_stride=3,
            metadata_dtype=NativeSpecCycleDType.INT64,
            hidden_dtype=NativeSpecCycleDType.FP32,
            kv_dtype=NativeSpecCycleDType.BF16,
        ),
        pointers=NativeSpecCyclePointers(
            metadata=NativeSpecCycleMetadataPointers(
                token_ids=0x1000,
                positions=0x1100,
                parent_rows=0x1200,
                draft_depths=0x1300,
                row_to_request=0x1400,
                active_mask=0x1500,
            ),
            kv_live_spans=NativeSpecCycleKVLiveSpanPointers(
                base_offsets=0x2000,
                live_counts=0x2100,
                row_positions=0x2200,
            ),
            state=NativeSpecCycleStatePointers(hidden_seed_rows=0x3000),
            outputs=NativeSpecCycleOutputPointers(target_top1=0x4000),
        ),
        stream=0x5000,
    )


def _b1_control(*, cycle_id: int = 9) -> NativeSpecCycleControl:
    control = _b2_control(cycle_id=cycle_id)
    return replace(
        control,
        shape=replace(
            control.shape,
            row_count=2,
            active_row_count=2,
            candidate_count=1,
            active_candidate_count=1,
            candidate_budget=1,
        ),
    )


def _b3_control(*, cycle_id: int = 10) -> NativeSpecCycleControl:
    control = _b2_control(cycle_id=cycle_id)
    return replace(
        control,
        shape=replace(
            control.shape,
            row_count=4,
            active_row_count=4,
            row_capacity=4,
            candidate_count=3,
            active_candidate_count=3,
            candidate_capacity=3,
            candidate_budget=3,
            span_count=4,
            span_capacity=4,
            hidden_row_capacity=4,
            output_stride=4,
        ),
    )


def _proposal_control(*, cycle_id: int = 10) -> NativeSpecCycleControl:
    control = _b2_control(cycle_id=cycle_id)
    return replace(
        control,
        stages=NativeSpecCycleStage.PROPOSE,
        shape=replace(control.shape, kv_dtype=NativeSpecCycleDType.FP32),
        pointers=replace(
            control.pointers,
            state=NativeSpecCycleStatePointers(
                hidden_seed_in=0x3000,
                candidate_token_ids=0x3100,
                draft_key_cache=0x3200,
                draft_value_cache=0x3300,
            ),
        ),
    )


def _provider_b4_control(*, cycle_id: int = 13) -> NativeSpecCycleControl:
    control = _b2_control(cycle_id=cycle_id)
    return replace(
        control,
        stages=NativeSpecCycleStage.VERIFY | NativeSpecCycleStage.ACCEPT,
        shape=replace(
            control.shape,
            row_count=5,
            active_row_count=5,
            row_capacity=5,
            candidate_count=4,
            active_candidate_count=4,
            candidate_capacity=4,
            candidate_budget=4,
            span_count=5,
            span_capacity=5,
            hidden_row_capacity=5,
            output_stride=5,
            metadata_dtype=NativeSpecCycleDType.INT32,
            hidden_dtype=NativeSpecCycleDType.BF16,
        ),
        pointers=replace(
            control.pointers,
            metadata=replace(control.pointers.metadata, candidate_counts=0x1600),
            outputs=replace(
                control.pointers.outputs,
                accepted_counts=0x4100,
                commit_rows=0x4200,
                commit_tokens=0x4300,
                commit_positions=0x4400,
                next_tokens=0x4500,
                full_accept=0x4600,
                committed_output_ids=0x4700,
                committed_output_lengths=0x4800,
            ),
        ),
        stream=0,
    )


def _n2_control(*, cycle_id: int = 11) -> NativeSpecCycleControl:
    control = _b2_control(cycle_id=cycle_id)
    stages = (
        NativeSpecCycleStage.VERIFY
        | NativeSpecCycleStage.ACCEPT
        | NativeSpecCycleStage.COMMIT
        | NativeSpecCycleStage.UPDATE_CURSORS
    )
    return replace(
        control,
        stages=stages,
        shape=replace(control.shape, metadata_dtype=NativeSpecCycleDType.INT32),
        pointers=replace(
            control.pointers,
            metadata=replace(
                control.pointers.metadata,
                candidate_counts=0x1600,
                remaining_decode=0x1700,
            ),
            state=replace(
                control.pointers.state,
                linear_state_rows=0x3100,
                linear_state_dst=0x3200,
                hidden_seed_dst=0x3300,
            ),
            outputs=replace(
                control.pointers.outputs,
                accepted_counts=0x4100,
                commit_rows=0x4200,
                commit_tokens=0x4300,
                commit_positions=0x4400,
                next_tokens=0x4500,
                full_accept=0x4600,
                committed_output_ids=0x4700,
                committed_output_lengths=0x4800,
                output_ids=0x4900,
                output_lengths=0x4A00,
                last_positions=0x4B00,
                context_lengths=0x4C00,
            ),
        ),
    )


def _provider_commit_control(*, cycle_id: int = 13) -> NativeSpecCycleControl:
    control = _provider_b4_control(cycle_id=cycle_id)
    commit_stages = (
        NativeSpecCycleStage.VERIFY
        | NativeSpecCycleStage.ACCEPT
        | NativeSpecCycleStage.COMMIT
        | NativeSpecCycleStage.UPDATE_CURSORS
    )
    return replace(
        control,
        stages=commit_stages,
        shape=replace(control.shape, hidden_dtype=NativeSpecCycleDType.FP16),
        pointers=replace(
            control.pointers,
            state=replace(
                control.pointers.state,
                linear_state_rows=0x5100,
                linear_state_dst=0x5200,
            ),
            outputs=replace(
                control.pointers.outputs,
                output_ids=0x5300,
                output_lengths=0x5400,
                last_positions=0x5500,
                context_lengths=0x5600,
            ),
        ),
    )


class _FakeNativeLibrary:
    def __init__(self, *, status: NativeSpecCycleStatus = NativeSpecCycleStatus.COMPLETE) -> None:
        self.status = status
        self.calls: list[tuple[int, int, int]] = []
        self.hipengine_native_spec_target_graph_launch_v1 = self._launch
        self.hipengine_native_spec_proposal_graph_launch_v1 = self._launch

    def _launch(self, control_ptr, result_ptr, graph_exec, graph_launch_fn, stream_sync_fn):
        control = ctypes.cast(control_ptr, ctypes.POINTER(NativeSpecCycleControlC)).contents
        result = ctypes.cast(result_ptr, ctypes.POINTER(NativeSpecCycleResultC)).contents
        graph_exec_value = int(getattr(graph_exec, "value", graph_exec) or 0)
        graph_launch_value = int(getattr(graph_launch_fn, "value", graph_launch_fn) or 0)
        stream_sync_value = int(getattr(stream_sync_fn, "value", stream_sync_fn) or 0)
        self.calls.append((graph_exec_value, graph_launch_value, stream_sync_value))
        result.abi_version = NATIVE_SPEC_CYCLE_ABI_VERSION
        result.struct_size = ctypes.sizeof(NativeSpecCycleResultC)
        result.status = int(self.status)
        result.error_code = int(
            NativeSpecCycleError.NONE
            if self.status is NativeSpecCycleStatus.COMPLETE
            else NativeSpecCycleError.KERNEL_LAUNCH
        )
        result.completed_stage_mask = (
            int(control.stage_mask)
            if self.status is NativeSpecCycleStatus.COMPLETE
            else 0
        )
        result.failed_stage = (
            0
            if self.status is NativeSpecCycleStatus.COMPLETE
            else int(NativeSpecCycleStage.VERIFY)
        )
        result.request_count = int(control.request_count)
        result.visible_output_count = 0
        result.cycle_id = int(control.cycle_id)
        result.transaction_id = int(control.transaction_id)
        result.backend_error_code = 0 if self.status is NativeSpecCycleStatus.COMPLETE else 700
        result.reserved = 0
        return 0


def test_native_target_graph_build_plan_is_host_only_and_versioned(tmp_path: Path) -> None:
    artifact = plan_native_spec_cycle_graph_launcher_build(
        cache_root=tmp_path,
        compiler_version="hipcc-test",
        target_arch="gfx1100",
    )

    assert artifact.family == "native_spec_cycle_graph"
    assert artifact.output_path.name == "native_spec_cycle_graph.so"
    assert artifact.sources[0].name == "native_cycle_graph.cpp"
    abi_header = artifact.sources[0].with_name("native_cycle_abi.h")
    abi_digest = hashlib.sha256(abi_header.read_bytes()).hexdigest()
    assert f"-DHIPENGINE_NATIVE_SPEC_CYCLE_ABI_HEADER_SHA256_{abi_digest}=1" in artifact.flags
    source = artifact.sources[0].read_text()
    assert "native_cycle_abi.h" in source
    assert "hipengine_native_spec_target_graph_launch_v1" in source
    assert "hipengine_native_spec_proposal_graph_launch_v1" in source
    assert "__global__" not in source


def test_gguf_native_target_graph_has_dedicated_gfx11_backend_registrations() -> None:
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        assert (
            resolve(
                backend=backend,
                layer="speculative_cycle",
                quant="w4_gguf",
                variant="native_v1_b2_target_graph",
                missing="none",
            )
            is create_native_spec_target_graph_launcher
        )

    assert (
        resolve(
            backend="hip_gfx1151",
            layer="speculative_cycle",
            quant="w4_gguf",
            variant="native_v1_b2_proposal_graph",
            missing="none",
        )
        is None
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="speculative_cycle",
            quant="w4_gguf",
            variant="native_v1_b2_proposal_graph",
        )
        is create_native_spec_proposal_graph_launcher
    )


def test_native_target_graph_launcher_calls_one_pre_resolved_submission_boundary() -> None:
    library = _FakeNativeLibrary()
    launcher = NativeSpecTargetGraphLauncher(
        graph_exec=0x6000,
        graph_launch_fn=0x7000,
        stream_synchronize_fn=0x8000,
        library=library,
    )

    result = launcher.launch(_b2_control())

    assert result.status is NativeSpecCycleStatus.COMPLETE
    assert result.completed_stages is NativeSpecCycleStage.VERIFY
    assert result.cycle_id == 7
    assert launcher.launch_count == 1
    assert library.calls == [(0x6000, 0x7000, 0x8000)]


def test_native_proposal_graph_launcher_calls_one_pre_resolved_submission_boundary() -> None:
    library = _FakeNativeLibrary()
    launcher = NativeSpecProposalGraphLauncher(
        graph_exec=0x6000,
        graph_launch_fn=0x7000,
        stream_synchronize_fn=0x8000,
        bound_control=_proposal_control(),
        library=library,
    )

    result = launcher.launch(_proposal_control(cycle_id=12))

    assert result.status is NativeSpecCycleStatus.COMPLETE
    assert result.completed_stages is NativeSpecCycleStage.PROPOSE
    assert result.cycle_id == 12
    assert launcher.launch_count == 1
    assert library.calls == [(0x6000, 0x7000, 0x8000)]


def test_provider_target_graph_launcher_accepts_paro_b4_verify_accept_bucket() -> None:
    library = _FakeNativeLibrary()
    control = _provider_b4_control()
    launcher = NativeSpecProviderTargetGraphLauncher(
        graph_exec=0x6000,
        graph_launch_fn=0x7000,
        stream_synchronize_fn=0x8000,
        bound_control=control,
        library=library,
    )

    result = launcher.launch(replace(control, cycle_id=14))

    assert result.status is NativeSpecCycleStatus.COMPLETE
    assert result.completed_stages == (
        NativeSpecCycleStage.VERIFY | NativeSpecCycleStage.ACCEPT
    )
    assert result.cycle_id == 14
    assert launcher.launch_count == 1
    assert library.calls == [(0x6000, 0x7000, 0x8000)]

    fp16_launcher = NativeSpecProviderTargetGraphLauncher(
        graph_exec=0x6001,
        graph_launch_fn=0x7000,
        stream_synchronize_fn=0x8000,
        library=library,
    )
    fp16 = fp16_launcher.launch(
        replace(control, shape=replace(control.shape, hidden_dtype=NativeSpecCycleDType.FP16))
    )
    assert fp16.status is NativeSpecCycleStatus.COMPLETE


def test_provider_target_graph_launcher_accepts_paro_commit_cursor_bucket_only() -> None:
    library = _FakeNativeLibrary()
    commit_control = _provider_commit_control()
    commit_stages = commit_control.stages
    launcher = NativeSpecProviderTargetGraphLauncher(
        graph_exec=0x6000,
        graph_launch_fn=0x7000,
        stream_synchronize_fn=0x8000,
        bound_control=commit_control,
        library=library,
    )

    result = launcher.launch(replace(commit_control, cycle_id=14))

    assert result.completed_stages == commit_stages
    with pytest.raises(ValueError, match="FP16"):
        launcher.launch(
            replace(
                commit_control,
                shape=replace(commit_control.shape, hidden_dtype=NativeSpecCycleDType.BF16),
            )
        )


def test_bound_provider_launcher_reuses_marshaled_control_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = _FakeNativeLibrary()
    control = _provider_b4_control(cycle_id=20)
    marshal_calls = 0
    original = NativeSpecCycleControl.to_ctypes

    def counted_to_ctypes(observed: NativeSpecCycleControl):
        nonlocal marshal_calls
        marshal_calls += 1
        return original(observed)

    monkeypatch.setattr(NativeSpecCycleControl, "to_ctypes", counted_to_ctypes)
    launcher = NativeSpecProviderTargetGraphLauncher(
        graph_exec=0x6000,
        graph_launch_fn=0x7000,
        stream_synchronize_fn=0x8000,
        bound_control=control,
        library=library,
    )

    first = launcher.launch_bound(cycle_id=21, transaction_id=31)
    second = launcher.launch_bound(cycle_id=22, transaction_id=32)

    assert (first.cycle_id, first.transaction_id) == (21, 31)
    assert (second.cycle_id, second.transaction_id) == (22, 32)
    assert marshal_calls == 1
    assert launcher.launch_count == 2
    assert library.calls == [
        (0x6000, 0x7000, 0x8000),
        (0x6000, 0x7000, 0x8000),
    ]


def test_provider_target_graph_has_registered_w4_paro_plugin_boundary() -> None:
    register_native_spec_provider_target_graph()
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            "speculative_cycle",
            "w4_paro",
            "native_v1_target_graph",
        )
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="speculative_cycle",
            quant="w4_paro",
            variant="native_v1_target_graph",
        )
        is create_native_spec_provider_target_graph_launcher
    )


def test_gguf_target_graph_keeps_strict_b1_b2_b3_contract() -> None:
    launcher = NativeSpecTargetGraphLauncher(
        graph_exec=0x6000,
        graph_launch_fn=0x7000,
        stream_synchronize_fn=0x8000,
        library=_FakeNativeLibrary(),
    )

    with pytest.raises(ValueError, match="B1/B2/B3"):
        launcher.launch(
            replace(_provider_b4_control(), stages=NativeSpecCycleStage.VERIFY)
        )


def test_native_target_graph_launcher_accepts_n2_device_accept_commit_control() -> None:
    library = _FakeNativeLibrary()
    launcher = NativeSpecTargetGraphLauncher(
        graph_exec=0x6000,
        graph_launch_fn=0x7000,
        stream_synchronize_fn=0x8000,
        library=library,
    )

    result = launcher.launch(_n2_control())

    assert result.status is NativeSpecCycleStatus.COMPLETE
    assert result.completed_stages == (
        NativeSpecCycleStage.VERIFY
        | NativeSpecCycleStage.ACCEPT
        | NativeSpecCycleStage.COMMIT
        | NativeSpecCycleStage.UPDATE_CURSORS
    )
    assert launcher.launch_count == 1


def test_native_target_graph_launcher_accepts_b1_b2_and_b3_shape_buckets() -> None:
    library = _FakeNativeLibrary()
    launcher = NativeSpecTargetGraphLauncher(
        graph_exec=0x6000,
        graph_launch_fn=0x7000,
        stream_synchronize_fn=0x8000,
        library=library,
    )

    b1 = launcher.launch(_b1_control())
    b2 = launcher.launch(_b2_control())
    b3 = launcher.launch(_b3_control())

    assert b1.status is NativeSpecCycleStatus.COMPLETE
    assert b2.status is NativeSpecCycleStatus.COMPLETE
    assert b3.status is NativeSpecCycleStatus.COMPLETE
    assert b1.request_count == b2.request_count == b3.request_count == 1
    assert launcher.launch_count == 3


def test_native_target_graph_keeps_n2_device_commit_bounded_to_b1_b2() -> None:
    launcher = NativeSpecTargetGraphLauncher(
        graph_exec=0x6000,
        graph_launch_fn=0x7000,
        stream_synchronize_fn=0x8000,
        library=_FakeNativeLibrary(),
    )
    b3_n2 = replace(
        _n2_control(),
        shape=replace(_b3_control().shape, metadata_dtype=NativeSpecCycleDType.INT32),
    )

    with pytest.raises(ValueError, match="N2 B1/B2"):
        launcher.launch(b3_n2)


def test_native_target_graph_launcher_rejects_control_that_drifted_from_bound_graph() -> None:
    control = _b2_control()
    launcher = NativeSpecTargetGraphLauncher(
        graph_exec=0x6000,
        graph_launch_fn=0x7000,
        stream_synchronize_fn=0x8000,
        bound_control=control,
        library=_FakeNativeLibrary(),
    )
    drifted = replace(
        control,
        pointers=replace(
            control.pointers,
            outputs=replace(control.pointers.outputs, target_top1=0x4100),
        ),
    )

    with pytest.raises(RuntimeError, match="state-bound"):
        launcher.launch(drifted)

    # Cycle/transaction identity is result bookkeeping, not a graph node binding.
    result = launcher.launch(replace(control, cycle_id=8, transaction_id=24))
    assert result.cycle_id == 8


def test_native_target_graph_launcher_surfaces_terminal_backend_failure() -> None:
    library = _FakeNativeLibrary(status=NativeSpecCycleStatus.FAILED)
    launcher = NativeSpecTargetGraphLauncher(
        graph_exec=0x6000,
        graph_launch_fn=0x7000,
        stream_synchronize_fn=0x8000,
        library=library,
    )

    result = launcher.launch(_b2_control())

    assert result.status is NativeSpecCycleStatus.FAILED
    assert result.error is NativeSpecCycleError.KERNEL_LAUNCH
    assert result.failed_stage is NativeSpecCycleStage.VERIFY
    assert result.backend_error_code == 700


def test_native_target_graph_launcher_rejects_non_b2_or_deadline_control() -> None:
    launcher = NativeSpecTargetGraphLauncher(
        graph_exec=0x6000,
        graph_launch_fn=0x7000,
        stream_synchronize_fn=0x8000,
        library=_FakeNativeLibrary(),
    )
    control = _b2_control()

    with pytest.raises(ValueError, match="B2"):
        launcher.launch(replace(control, shape=replace(control.shape, candidate_budget=1)))
    with pytest.raises(ValueError, match="deadline"):
        launcher.launch(replace(control, deadline_ns=1))


def test_native_target_graph_launcher_validates_bound_handles() -> None:
    with pytest.raises(ValueError, match="graph_exec"):
        NativeSpecTargetGraphLauncher(
            graph_exec=0,
            graph_launch_fn=0x7000,
            stream_synchronize_fn=0x8000,
            library=_FakeNativeLibrary(),
        )
    with pytest.raises(ValueError, match="graph_launch_fn"):
        NativeSpecTargetGraphLauncher(
            graph_exec=0x6000,
            graph_launch_fn=0,
            stream_synchronize_fn=0x8000,
            library=_FakeNativeLibrary(),
        )


def test_native_target_graph_cpp_launcher_calls_fake_hip_functions(
    hip_test_target_arch: str,
) -> None:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")

    library = build_native_spec_cycle_graph_launcher(target_arch=hip_test_target_arch)
    calls: list[tuple[str, int, int]] = []
    graph_callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    sync_callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)

    @graph_callback
    def graph_launch(graph_exec, stream):
        calls.append(("launch", int(graph_exec or 0), int(stream or 0)))
        return 0

    @sync_callback
    def stream_synchronize(stream):
        calls.append(("sync", int(stream or 0), 0))
        return 0

    launcher = NativeSpecTargetGraphLauncher(
        graph_exec=0x6000,
        graph_launch_fn=ctypes.cast(graph_launch, ctypes.c_void_p).value or 0,
        stream_synchronize_fn=ctypes.cast(stream_synchronize, ctypes.c_void_p).value or 0,
        library=library,
    )
    b1 = launcher.launch(_b1_control())
    b2 = launcher.launch(_b2_control())
    b3 = launcher.launch(_b3_control())
    proposal_launcher = NativeSpecProposalGraphLauncher(
        graph_exec=0x6001,
        graph_launch_fn=ctypes.cast(graph_launch, ctypes.c_void_p).value or 0,
        stream_synchronize_fn=ctypes.cast(stream_synchronize, ctypes.c_void_p).value or 0,
        library=library,
    )
    proposal = proposal_launcher.launch(_proposal_control())
    provider_launcher = NativeSpecProviderTargetGraphLauncher(
        graph_exec=0x6002,
        graph_launch_fn=ctypes.cast(graph_launch, ctypes.c_void_p).value or 0,
        stream_synchronize_fn=ctypes.cast(stream_synchronize, ctypes.c_void_p).value or 0,
        library=library,
    )
    provider = provider_launcher.launch(_provider_b4_control())
    provider_commit = provider_launcher.launch(_provider_commit_control())

    assert b1.status is NativeSpecCycleStatus.COMPLETE
    assert b2.status is NativeSpecCycleStatus.COMPLETE
    assert b3.status is NativeSpecCycleStatus.COMPLETE
    assert proposal.status is NativeSpecCycleStatus.COMPLETE
    assert proposal.completed_stages is NativeSpecCycleStage.PROPOSE
    assert provider.status is NativeSpecCycleStatus.COMPLETE
    assert provider.completed_stages == (
        NativeSpecCycleStage.VERIFY | NativeSpecCycleStage.ACCEPT
    )
    assert provider_commit.status is NativeSpecCycleStatus.COMPLETE
    assert provider_commit.completed_stages == _provider_commit_control().stages
    assert calls == [
        ("launch", 0x6000, 0x5000),
        ("sync", 0x5000, 0),
        ("launch", 0x6000, 0x5000),
        ("sync", 0x5000, 0),
        ("launch", 0x6000, 0x5000),
        ("sync", 0x5000, 0),
        ("launch", 0x6001, 0x5000),
        ("sync", 0x5000, 0),
        ("launch", 0x6002, 0),
        ("sync", 0, 0),
        ("launch", 0x6002, 0),
        ("sync", 0, 0),
    ]
