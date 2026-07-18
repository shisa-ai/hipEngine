from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path
import re

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.speculative import DraftBatch, TargetVerifyBatch, TargetVerifyBuffers
from hipengine.speculative.native_cycle import (
    NATIVE_SPEC_CYCLE_ABI_VERSION,
    FakeNativeSpecCycleLauncher,
    NativeSpecCycleControl,
    NativeSpecCycleControlC,
    NativeSpecCycleDType,
    NativeSpecCycleError,
    NativeSpecCycleKVLiveSpanPointers,
    NativeSpecCycleMetadataPointers,
    NativeSpecCycleMode,
    NativeSpecCycleOutputPointers,
    NativeSpecCyclePointers,
    NativeSpecCycleResult,
    NativeSpecCycleResultC,
    NativeSpecCycleShape,
    NativeSpecCycleStage,
    NativeSpecCycleStatePointers,
    NativeSpecCycleStatus,
)


def _tensor(ptr: int, shape: tuple[int, ...], dtype: DType | str, *, device: Device | None = None) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, device or Device("hip", 0))


def _target_and_buffers(*, metadata_dtype: DType = DType.INT32) -> tuple[TargetVerifyBatch, TargetVerifyBuffers]:
    draft = DraftBatch(
        request_ids=(17,),
        candidate_tokens=(101, 102),
        parent_positions=(8, 9),
        draft_depths=(1, 2),
        row_to_request=(17, 17),
        tree_parents=(-1, 0),
    )
    target = TargetVerifyBatch.from_draft(
        draft,
        root_tokens=(100,),
        root_positions=(8,),
    )
    buffers = TargetVerifyBuffers.for_batch(
        target,
        token_ids=_tensor(0x1000, (3,), metadata_dtype),
        positions=_tensor(0x1100, (3,), metadata_dtype),
        parent_rows=_tensor(0x1200, (3,), metadata_dtype),
        draft_depths=_tensor(0x1300, (3,), metadata_dtype),
        row_to_request=_tensor(0x1400, (3,), metadata_dtype),
        active_mask=_tensor(0x1500, (3,), DType.BOOL),
        target_top1=_tensor(0x1600, (3,), metadata_dtype),
        accepted_counts=_tensor(0x1700, (1,), metadata_dtype),
        commit_rows=_tensor(0x1800, (1,), metadata_dtype),
        commit_tokens=_tensor(0x1900, (1,), metadata_dtype),
        commit_positions=_tensor(0x1A00, (1,), metadata_dtype),
        next_tokens=_tensor(0x1B00, (1,), metadata_dtype),
        full_accept=_tensor(0x1C00, (1,), DType.BOOL),
        committed_output_ids=_tensor(0x1D00, (1, 4), metadata_dtype),
        committed_output_lengths=_tensor(0x1E00, (1,), metadata_dtype),
        transaction_id=23,
    )
    return target, buffers


def _spans(*, device: Device | None = None) -> KVLiveSpans:
    device = device or Device("hip", 0)
    return KVLiveSpans.paged_uniform(
        block_table=_tensor(0x2000, (1, 4), DType.INT32, device=device),
        live_counts=_tensor(0x2100, (3,), DType.INT32, device=device),
        max_live_count=11,
        storage_dtype=DType.BF16,
        row_positions=_tensor(0x2200, (3,), DType.INT32, device=device),
        span_role="verify_chain",
    )


def _verify_control() -> NativeSpecCycleControl:
    _target, buffers = _target_and_buffers()
    return NativeSpecCycleControl.for_target_verify(
        cycle_id=7,
        buffers=buffers,
        kv_live_spans=_spans(),
        hidden_seed_rows=_tensor(0x3000, (3, 2048), DType.FP32),
        context_bucket=128,
        stream=0x4000,
        output_stride=4,
    )


def test_native_spec_cycle_v1_maps_existing_verify_and_kvlivespans_abis() -> None:
    control = _verify_control()

    assert control.abi_version == NATIVE_SPEC_CYCLE_ABI_VERSION == 1
    assert control.stages == NativeSpecCycleStage.VERIFY
    assert control.mode is NativeSpecCycleMode.CHAIN
    assert control.transaction_id == 23
    assert control.shape == NativeSpecCycleShape(
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
        max_live_count=11,
        context_bucket=128,
        hidden_size=2048,
        hidden_row_capacity=3,
        output_stride=4,
        metadata_dtype=NativeSpecCycleDType.INT32,
        hidden_dtype=NativeSpecCycleDType.FP32,
        kv_dtype=NativeSpecCycleDType.BF16,
    )
    assert control.pointers.metadata.token_ids == 0x1000
    assert control.pointers.metadata.active_mask == 0x1500
    assert control.pointers.kv_live_spans.base_offsets == 0x2000
    assert control.pointers.kv_live_spans.live_counts == 0x2100
    assert control.pointers.kv_live_spans.row_positions == 0x2200
    assert control.pointers.state.hidden_seed_rows == 0x3000
    assert control.pointers.outputs.target_top1 == 0x1600

    raw = control.to_ctypes()
    assert raw.abi_version == 1
    assert raw.struct_size == ctypes.sizeof(NativeSpecCycleControlC)
    assert raw.stage_mask == int(NativeSpecCycleStage.VERIFY)
    assert raw.mode == int(NativeSpecCycleMode.CHAIN)
    assert raw.cycle_id == 7
    assert raw.transaction_id == 23
    assert raw.stream == 0x4000
    assert raw.row_capacity == 3
    assert raw.metadata_dtype == int(NativeSpecCycleDType.INT32)
    assert raw.hidden_dtype == int(NativeSpecCycleDType.FP32)
    assert raw.kv_dtype == int(NativeSpecCycleDType.BF16)
    assert raw.metadata_token_ids == 0x1000
    assert raw.kv_base_offsets == 0x2000
    assert raw.state_hidden_seed_rows == 0x3000
    assert raw.output_target_top1 == 0x1600
    assert NativeSpecCycleControl.from_ctypes(raw) == control

    raw.abi_version = 2
    with pytest.raises(ValueError, match="ABI version"):
        NativeSpecCycleControl.from_ctypes(raw)
    raw.abi_version = 1
    raw.struct_size -= 8
    with pytest.raises(ValueError, match="struct_size"):
        NativeSpecCycleControl.from_ctypes(raw)


def test_native_spec_cycle_c_layout_is_fixed_width_and_versioned() -> None:
    assert NativeSpecCycleControlC.abi_version.offset == 0
    assert NativeSpecCycleControlC.struct_size.offset == 4
    assert NativeSpecCycleControlC.stage_mask.offset == 8
    assert NativeSpecCycleControlC.mode.offset == 12
    assert NativeSpecCycleControlC.cycle_id.offset == 16
    assert ctypes.sizeof(NativeSpecCycleControlC) == 496

    assert NativeSpecCycleResultC.abi_version.offset == 0
    assert NativeSpecCycleResultC.struct_size.offset == 4
    assert NativeSpecCycleResultC.status.offset == 8
    assert ctypes.sizeof(NativeSpecCycleResultC) == 64


def test_native_spec_cycle_header_field_order_matches_ctypes_mirror() -> None:
    header = (
        Path(__file__).parents[1]
        / "hipengine"
        / "speculative"
        / "native_cycle_abi.h"
    ).read_text()

    def declarations(struct_name: str) -> list[tuple[str, str]]:
        match = re.search(
            rf"typedef struct {struct_name} \{{(?P<body>.*?)\}} {struct_name};",
            header,
            flags=re.DOTALL,
        )
        assert match is not None
        return [
            (name, c_type)
            for c_type, name in re.findall(
                r"^\s*(uint32_t|uint64_t|int64_t)\s+([a-z0-9_]+);",
                match.group("body"),
                flags=re.MULTILINE,
            )
        ]

    ctypes_names = {
        ctypes.c_uint32: "uint32_t",
        ctypes.c_uint64: "uint64_t",
        ctypes.c_int64: "int64_t",
    }
    assert declarations("HipengineNativeSpecCycleControlV1") == [
        (name, ctypes_names[c_type]) for name, c_type in NativeSpecCycleControlC._fields_
    ]
    assert declarations("HipengineNativeSpecCycleResultV1") == [
        (name, ctypes_names[c_type]) for name, c_type in NativeSpecCycleResultC._fields_
    ]
    assert "HIPENGINE_NATIVE_SPEC_CYCLE_ABI_VERSION 1u" in header
    assert "sizeof(HipengineNativeSpecCycleControlV1) == 496" in header
    assert "sizeof(HipengineNativeSpecCycleResultV1) == 64" in header


def test_native_spec_cycle_shape_rejects_unbounded_or_inconsistent_counts() -> None:
    shape = _verify_control().shape

    with pytest.raises(ValueError, match="row_count must equal request_count plus candidate_count"):
        replace(shape, row_count=4)
    with pytest.raises(ValueError, match="active_row_count"):
        replace(shape, active_row_count=2)
    with pytest.raises(ValueError, match="active_candidate_count"):
        replace(shape, active_candidate_count=3)
    with pytest.raises(ValueError, match="row_capacity"):
        replace(shape, row_capacity=2)
    with pytest.raises(ValueError, match="candidate_capacity"):
        replace(shape, candidate_capacity=1)
    with pytest.raises(ValueError, match="request_capacity"):
        replace(shape, request_capacity=0)
    with pytest.raises(ValueError, match="span_capacity"):
        replace(shape, span_capacity=2)
    with pytest.raises(ValueError, match="context_bucket"):
        replace(shape, context_bucket=10)
    with pytest.raises(ValueError, match="metadata_dtype"):
        replace(shape, metadata_dtype=NativeSpecCycleDType.FP16)
    with pytest.raises(ValueError, match="uint32"):
        replace(shape, row_capacity=1 << 32)


def test_native_spec_cycle_validates_stage_dependencies_and_every_required_group() -> None:
    control = _verify_control()

    with pytest.raises(ValueError, match="ACCEPT requires VERIFY"):
        replace(control, stages=NativeSpecCycleStage.ACCEPT)
    with pytest.raises(ValueError, match="COMMIT requires ACCEPT"):
        replace(control, stages=NativeSpecCycleStage.VERIFY | NativeSpecCycleStage.COMMIT)
    with pytest.raises(ValueError, match="UPDATE_CURSORS requires COMMIT"):
        replace(
            control,
            stages=NativeSpecCycleStage.VERIFY | NativeSpecCycleStage.UPDATE_CURSORS,
        )

    bad_metadata = replace(control.pointers.metadata, positions=0)
    with pytest.raises(ValueError, match="metadata.positions"):
        replace(control, pointers=replace(control.pointers, metadata=bad_metadata))

    bad_spans = replace(control.pointers.kv_live_spans, live_counts=0)
    with pytest.raises(ValueError, match="kv_live_spans.live_counts"):
        replace(control, pointers=replace(control.pointers, kv_live_spans=bad_spans))

    bad_state = replace(control.pointers.state, hidden_seed_rows=0)
    with pytest.raises(ValueError, match="state.hidden_seed_rows"):
        replace(control, pointers=replace(control.pointers, state=bad_state))

    bad_outputs = replace(control.pointers.outputs, target_top1=0)
    with pytest.raises(ValueError, match="outputs.target_top1"):
        replace(control, pointers=replace(control.pointers, outputs=bad_outputs))


def test_native_spec_cycle_rejects_partial_pointer_pairs_and_invalid_raw_addresses() -> None:
    with pytest.raises(ValueError, match="draft_key_cache.*draft_value_cache"):
        NativeSpecCycleStatePointers(draft_key_cache=0x1000)
    with pytest.raises(ValueError, match="k_scale.*v_scale"):
        NativeSpecCycleKVLiveSpanPointers(k_scale=0x1000)
    with pytest.raises(ValueError, match="must fit uint64"):
        NativeSpecCycleMetadataPointers(token_ids=1 << 64)
    with pytest.raises(TypeError, match="must be an integer"):
        NativeSpecCycleOutputPointers(target_top1=True)


def test_target_verify_adapter_rejects_cross_device_and_mixed_metadata_dtypes() -> None:
    _target, buffers = _target_and_buffers()
    hidden = _tensor(0x3000, (3, 2048), DType.FP32)

    with pytest.raises(ValueError, match="one device"):
        NativeSpecCycleControl.for_target_verify(
            cycle_id=1,
            buffers=buffers,
            kv_live_spans=_spans(device=Device("hip", 1)),
            hidden_seed_rows=hidden,
            context_bucket=128,
        )

    mixed = replace(buffers, positions=_tensor(0x1100, (3,), DType.INT64))
    with pytest.raises(ValueError, match="one integer dtype"):
        NativeSpecCycleControl.for_target_verify(
            cycle_id=1,
            buffers=mixed,
            kv_live_spans=_spans(),
            hidden_seed_rows=hidden,
            context_bucket=128,
        )

    with pytest.raises(ValueError, match="hidden_seed_rows must have shape"):
        NativeSpecCycleControl.for_target_verify(
            cycle_id=1,
            buffers=buffers,
            kv_live_spans=_spans(),
            hidden_seed_rows=_tensor(0x3000, (2, 2048), DType.FP32),
            context_bucket=128,
        )

    strided = replace(
        buffers,
        token_ids=Tensor.from_handle(
            0x1000,
            (3,),
            DType.INT32,
            Device("hip", 0),
            strides=(2,),
        ),
    )
    with pytest.raises(ValueError, match="token_ids must be a contiguous"):
        NativeSpecCycleControl.for_target_verify(
            cycle_id=1,
            buffers=strided,
            kv_live_spans=_spans(),
            hidden_seed_rows=hidden,
            context_bucket=128,
        )


def test_native_spec_cycle_result_contract_and_ctypes_roundtrip() -> None:
    control = _verify_control()
    result = NativeSpecCycleResult.complete(control, visible_output_count=0)

    assert result.status is NativeSpecCycleStatus.COMPLETE
    assert result.error is NativeSpecCycleError.NONE
    result.validate_for(control)

    raw = result.to_ctypes()
    assert raw.struct_size == ctypes.sizeof(NativeSpecCycleResultC)
    assert NativeSpecCycleResult.from_ctypes(raw) == result

    with pytest.raises(ValueError, match="completed stages"):
        replace(result, completed_stages=NativeSpecCycleStage(0)).validate_for(control)
    with pytest.raises(ValueError, match="failed result requires"):
        replace(result, status=NativeSpecCycleStatus.FAILED).validate_for(control)
    with pytest.raises(ValueError, match="successful result cannot carry"):
        replace(result, error=NativeSpecCycleError.INTERNAL).validate_for(control)
    with pytest.raises(ValueError, match="visible_output_count"):
        replace(result, visible_output_count=5).validate_for(control)


def test_fake_native_spec_cycle_launcher_validates_lifecycle_and_cpu_oracle() -> None:
    target, _buffers = _target_and_buffers()
    control = _verify_control()
    expected = target.accept_from_top1((101, 102, 999), transaction_id=23)

    def execute(value: NativeSpecCycleControl) -> NativeSpecCycleResult:
        assert value is control
        assert expected.accepted_counts == (2,)
        assert expected.next_tokens == (999,)
        return NativeSpecCycleResult.complete(value, visible_output_count=3)

    launcher = FakeNativeSpecCycleLauncher(execute)
    result = launcher.launch(control)

    assert result.visible_output_count == 3
    assert launcher.launch_count == 1
    assert launcher.history == ((control, result),)

    reentrant: FakeNativeSpecCycleLauncher

    def recurse(value: NativeSpecCycleControl) -> NativeSpecCycleResult:
        with pytest.raises(RuntimeError, match="already in flight"):
            reentrant.launch(value)
        return NativeSpecCycleResult.complete(value)

    reentrant = FakeNativeSpecCycleLauncher(recurse)
    reentrant.launch(control)


def test_full_cycle_stage_contract_requires_accept_commit_and_cursor_outputs() -> None:
    base = _verify_control()
    stages = (
        NativeSpecCycleStage.PROPOSE
        | NativeSpecCycleStage.VERIFY
        | NativeSpecCycleStage.ACCEPT
        | NativeSpecCycleStage.COMMIT
        | NativeSpecCycleStage.UPDATE_CURSORS
    )
    pointers = NativeSpecCyclePointers(
        metadata=replace(base.pointers.metadata, candidate_counts=0x5000, remaining_decode=0x5100),
        kv_live_spans=replace(
            base.pointers.kv_live_spans,
            key_cache=0x5200,
            value_cache=0x5300,
        ),
        state=replace(
            base.pointers.state,
            hidden_seed_in=0x5400,
            candidate_token_ids=0x5500,
            linear_state_rows=0x5600,
            linear_state_dst=0x5700,
            key_rows=0x5800,
            value_rows=0x5900,
            hidden_seed_dst=0x5A00,
        ),
        outputs=replace(
            base.pointers.outputs,
            output_ids=0x5B00,
            output_lengths=0x5C00,
            last_positions=0x5D00,
            context_lengths=0x5E00,
        ),
    )
    full = replace(base, stages=stages, pointers=pointers)
    full.validate()

    with pytest.raises(ValueError, match="outputs.output_ids"):
        replace(full, pointers=replace(pointers, outputs=replace(pointers.outputs, output_ids=0)))
