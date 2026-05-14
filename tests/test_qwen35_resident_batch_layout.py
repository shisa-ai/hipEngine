from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoResidentBatchLayout,
    Qwen35ParoResidentSession,
    qwen35_paro_native_prefill_plan,
)


def test_qwen35_resident_batch_layout_is_batch_shaped_with_slot0_aliases() -> None:
    layout = Qwen35ParoResidentBatchLayout(
        max_batch_size=4,
        hidden_size=4096,
        max_sequence_length=1024,
        block_size=256,
        blocks=4,
        num_key_value_heads=2,
        head_dim=256,
    )

    assert layout.hidden_shape == (4, 4096)
    assert layout.slot_scalar_shape == (4,)
    assert layout.slot0_hidden_shape == (1, 4096)
    assert layout.full_kv_shape == (4, 4, 256, 2, 256)
    assert layout.slot0_full_kv_shape == (4, 256, 2, 256)


def test_qwen35_resident_native_prefill_plan_reports_linear_prefix_blocker() -> None:
    layer_types = ("linear_attention", "linear_attention", "full_attention", "linear_attention")
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.layer_limit = 4
    session.config = SimpleNamespace(layer_types=layer_types)

    plan = session.native_prefill_plan()
    pure_plan = qwen35_paro_native_prefill_plan(layer_types, layer_limit=4)

    assert pure_plan == plan

    assert plan.path == "linear_attention_prefix_only"
    assert plan.layer_limit == 4
    assert plan.linear_prefix_layers == 2
    assert not plan.full_layer_limit_native
    assert plan.first_unsupported_layer == 2
    assert plan.first_unsupported_type == "full_attention"
    assert any("first unsupported layer 2" in blocker for blocker in plan.blockers)
    assert plan.to_json_dict()["linear_prefix_layers"] == 2


def test_qwen35_resident_native_prefill_plan_rejects_invalid_layer_limit() -> None:
    with pytest.raises(ValueError, match="exceeds available"):
        qwen35_paro_native_prefill_plan(("linear_attention",), layer_limit=2)


def test_qwen35_resident_native_prefill_plan_accepts_all_linear_layer_limit() -> None:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.layer_limit = 2
    session.config = SimpleNamespace(layer_types=("linear_attention", "linear_attention", "full_attention"))

    plan = session.native_prefill_plan()

    assert plan.path == "linear_attention_native_full_layer_limit"
    assert plan.linear_prefix_layers == 2
    assert plan.full_layer_limit_native
    assert plan.first_unsupported_layer is None
    assert plan.first_unsupported_type is None
    assert plan.blockers == ()


def test_qwen35_resident_prefill_linear_tokens_native_requires_rejected_correctness_opt_in() -> None:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.closed = False
    session.max_sequence_length = 8
    session.vocab_size = 100
    session.layer_limit = 2
    session.config = SimpleNamespace(layer_types=("linear_attention", "linear_attention"))

    with pytest.raises(NotImplementedError, match="rejected_correctness"):
        session.prefill_linear_tokens_native([1, 2], sample=True)


def test_qwen35_resident_batch_execution_metadata_labels_serial_fallback() -> None:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.layer_limit = 3
    session.config = SimpleNamespace(layer_types=("linear_attention", "linear_attention", "full_attention"))

    metadata = session.batch_execution_metadata(scheduler_owned=True)

    assert metadata.path == "scheduler_serial_slot_bridge"
    assert metadata.scheduler_owned
    assert metadata.row_execution == "serial_c1_layer_path"
    assert metadata.native_prefill_plan.linear_prefix_layers == 2
    assert not metadata.native_prefill_plan.full_layer_limit_native
    assert not metadata.native_compact_prefill
    assert not metadata.native_caware_decode
    assert not metadata.throughput_claim_eligible
    assert any("unsupported layer 2" in blocker and "full_attention" in blocker for blocker in metadata.blockers)
    payload = metadata.to_json_dict()
    assert payload["native_prefill_plan"]["linear_prefix_layers"] == 2
    assert payload["blockers"] == list(metadata.blockers)


def test_qwen35_resident_session_slot_views_offset_batch_state() -> None:
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.max_batch_size = 3
    session.max_sequence_length = 16
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.config = SimpleNamespace(
        hidden_size=8,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        num_key_value_heads=1,
        head_dim=4,
    )
    session.batch_hidden = Tensor.from_handle(0x1000, (3, 8), DType.FP16, device)
    session.batch_next_hidden = Tensor.from_handle(0x2000, (3, 8), DType.FP16, device)
    session.position_buf = DeviceBuffer(0x3000, 3 * DType.INT64.itemsize)
    session.context_buf = DeviceBuffer(0x4000, 3 * DType.INT64.itemsize)
    session.block_table = Tensor.from_handle(0x5000, (4,), DType.INT32, device)

    conv = Tensor.from_handle(0x6000, (8, 4), DType.FP32, device)
    recurrent = Tensor.from_handle(0x7000, (2, 4, 4), DType.FP32, device)
    session.linear_states = {
        1: (
            conv,
            recurrent,
            DeviceBuffer(0x6000, 3 * conv.numel * conv.dtype.itemsize),
            DeviceBuffer(0x7000, 3 * recurrent.numel * recurrent.dtype.itemsize),
            None,
            None,
        )
    }
    key = Tensor.from_handle(0x8000, (4, 256, 1, 4), DType.BF16, device)
    value = Tensor.from_handle(0x9000, (4, 256, 1, 4), DType.BF16, device)
    session.full_caches = {
        2: (
            key,
            value,
            DeviceBuffer(0x8000, 3 * key.numel * key.dtype.itemsize),
            DeviceBuffer(0x9000, 3 * value.numel * value.dtype.itemsize),
        )
    }

    assert session._slot_hidden_view(session.batch_hidden, 2).ptr == 0x1000 + 2 * 8 * DType.FP16.itemsize
    assert session._slot_scalar_tensor(session.position_buf, 2, DType.INT64).ptr == 0x3000 + 2 * DType.INT64.itemsize

    conv2, recurrent2 = session._slot_linear_state(1, 2)
    assert conv2.ptr == 0x6000 + 2 * conv.numel * DType.FP32.itemsize
    assert recurrent2.ptr == 0x7000 + 2 * recurrent.numel * DType.FP32.itemsize
    assert conv2.shape == conv.shape
    assert recurrent2.shape == recurrent.shape

    key2, value2 = session._slot_full_cache(2, 2)
    assert key2.ptr == 0x8000 + 2 * key.numel * DType.BF16.itemsize
    assert value2.ptr == 0x9000 + 2 * value.numel * DType.BF16.itemsize
    assert key2.shape == key.shape
    assert value2.shape == value.shape

    position, append_spans, decode_spans = session._slot_spans(2)
    assert position.ptr == 0x3000 + 2 * DType.INT64.itemsize
    assert append_spans.live_counts.ptr == position.ptr
    assert decode_spans.live_counts.ptr == 0x4000 + 2 * DType.INT64.itemsize
    assert append_spans.max_live_count == 15
    assert decode_spans.max_live_count == 16

    with pytest.raises(ValueError, match="slot"):
        session._slot_hidden_view(session.batch_hidden, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_batch_size": 0},
        {"hidden_size": 0},
        {"max_sequence_length": 0},
        {"block_size": 0},
        {"blocks": 0},
        {"num_key_value_heads": 0},
        {"head_dim": 0},
    ],
)
def test_qwen35_resident_batch_layout_validates_positive_dimensions(kwargs) -> None:
    base = dict(
        max_batch_size=1,
        hidden_size=4096,
        max_sequence_length=1024,
        block_size=256,
        blocks=4,
        num_key_value_heads=2,
        head_dim=256,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        Qwen35ParoResidentBatchLayout(**base)
