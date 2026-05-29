from __future__ import annotations

from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.generation import CompactPromptSlab
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.runtime import PrefillConfig
from hipengine.runtime.prefill import resolve_prefill_config_for_sequence
from hipengine.runtime.qwen35_paro import (
    Qwen35ParoGroupedMoeScratch,
    qwen35_grouped_moe_expert_lane_groups,
    qwen35_grouped_moe_expert_starts,
    qwen35_grouped_moe_lane_rows,
    qwen35_grouped_moe_lane_to_sorted_row,
    qwen35_grouped_moe_sorted_lanes_from_selected_experts,
    qwen35_grouped_moe_sorted_routing_weights,
    qwen35_grouped_moe_sorted_token_rows,
    qwen35_grouped_moe_weighted_token_sums,
)
from hipengine.runtime import qwen35_paro_runner as runner_module
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoResidentBatchLayout,
    Qwen35ParoResidentSession,
    Qwen35ParoResidentSpeculativeExecution,
    qwen35_paro_native_prefill_plan,
)
from hipengine.speculative import DraftBatch, TargetCommitPlan, TargetStateCommitBuffers


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str | DType) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


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
    assert layout.full_kv_scale_shape == (4, 4, 256, 2)
    assert layout.flat_full_kv_scale_shape == (16, 256, 2)
    assert layout.slot0_full_kv_scale_shape == (4, 256, 2)


def _resident_allocation_session(*, storage_dtype: str = "bf16", scale_dtype: DType = DType.FP16):
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.max_batch_size = 2
    session.blocks = 3
    session.block_size = 256
    session.config = SimpleNamespace(num_key_value_heads=2, head_dim=4)
    session.batch_layout = Qwen35ParoResidentBatchLayout(
        max_batch_size=session.max_batch_size,
        hidden_size=8,
        max_sequence_length=512,
        block_size=session.block_size,
        blocks=session.blocks,
        num_key_value_heads=session.config.num_key_value_heads,
        head_dim=session.config.head_dim,
    )
    session.kv_policy = FixedPagedKVPolicy(block_size=session.block_size, storage_dtype=storage_dtype)
    session.kv_storage_dtype = session.kv_policy.storage_dtype
    session.kv_scale_dtype = scale_dtype
    session.kv_scale_granularity = "per_token_head"
    session.full_caches = {}
    session.full_cache_scales = {}
    session.full_cache_scale_metadata = {}
    session.buffers = []
    session.allocations = []
    captured: list[tuple[DeviceBuffer, tuple[int, ...], np.dtype]] = []
    next_ptr = 0x100000

    def fake_dev(self, array: np.ndarray) -> DeviceBuffer:
        nonlocal next_ptr
        contiguous = np.ascontiguousarray(array)
        buf = DeviceBuffer(next_ptr, contiguous.nbytes)
        next_ptr += max(contiguous.nbytes, 1) + 0x100
        self.buffers.append(buf)
        captured.append((buf, tuple(contiguous.shape), contiguous.dtype))
        return buf

    session._dev = MethodType(fake_dev, session)
    return session, captured


def test_qwen35_resident_prefill_hidden_buffer_is_lazy_single_buffer() -> None:
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.config = SimpleNamespace(hidden_size=8)
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.prefill_capacity_rows = 4

    class FakeRuntime:
        def __init__(self) -> None:
            self.next_ptr = 0x7000
            self.mallocs: list[tuple[int, int]] = []
            self.frees: list[int] = []

        def malloc(self, nbytes: int) -> int:
            ptr = self.next_ptr
            self.next_ptr += max(int(nbytes), 1) + 0x100
            self.mallocs.append((ptr, int(nbytes)))
            return ptr

        def free(self, ptr: int) -> None:
            self.frees.append(int(ptr))

    runtime = FakeRuntime()
    session.runtime = runtime
    session.prefill_hidden_buffer = None
    session.prefill_hidden_capacity_rows = 0
    session._set_empty_prefill_hidden_views()

    hidden = session._ensure_prefill_hidden_capacity(3)

    assert hidden.shape == (3, 8)
    assert session.prefill_hidden.ptr == hidden.ptr
    assert session.prefill_next_hidden.ptr == hidden.ptr
    assert runtime.mallocs == [(0x7000, 3 * session.hidden_nbytes)]

    same = session._ensure_prefill_hidden_capacity(2)
    assert same.ptr == hidden.ptr
    assert runtime.mallocs == [(0x7000, 3 * session.hidden_nbytes)]

    larger = session._ensure_prefill_hidden_capacity(4)
    assert larger.ptr != hidden.ptr
    assert runtime.frees == [hidden.ptr]
    assert runtime.mallocs[-1] == (0x7000 + 3 * session.hidden_nbytes + 0x100, 4 * session.hidden_nbytes)

    session._release_prefill_hidden_buffer()

    assert runtime.frees == [hidden.ptr, larger.ptr]
    assert session.prefill_hidden.ptr == 0
    assert session.prefill_next_hidden.ptr == 0


def test_qwen35_resident_release_decode_scratch_for_prefill_frees_state_workspaces() -> None:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.linear_scratch = {0: object()}
    session.full_scratch = {1: object()}
    session.moe_scratch = {0: object(), 1: object()}

    class FakeWorkspace:
        def __init__(self) -> None:
            self.free_calls = 0

        def free(self) -> None:
            self.free_calls += 1

    state0 = SimpleNamespace(workspace=FakeWorkspace(), _rotate_fuse_ready={0x1000})
    state1 = SimpleNamespace(workspace=FakeWorkspace(), _rotate_fuse_ready={0x2000})
    session.states = [state0, state1]

    session._release_decode_scratch_for_prefill()

    assert state0.workspace.free_calls == 1
    assert state1.workspace.free_calls == 1
    assert state0._rotate_fuse_ready == set()
    assert state1._rotate_fuse_ready == set()
    assert session.linear_scratch == {}
    assert session.full_scratch == {}
    assert session.moe_scratch == {}


def test_qwen35_resident_full_kv_allocation_defaults_to_bf16_payload_only() -> None:
    session, captured = _resident_allocation_session(storage_dtype="bf16")

    session._allocate_full_attention_cache(2)

    key_cache, value_cache, key_buf, value_buf = session.full_caches[2]
    assert key_cache.dtype is DType.BF16
    assert value_cache.dtype is DType.BF16
    assert key_cache.shape == session.batch_layout.slot0_full_kv_shape
    assert value_cache.shape == session.batch_layout.slot0_full_kv_shape
    assert key_buf.nbytes == np.prod(session.batch_layout.full_kv_shape) * DType.BF16.itemsize
    assert value_buf.nbytes == key_buf.nbytes
    assert session.full_cache_scales == {}
    assert [item[2] for item in captured] == [np.dtype(np.uint16), np.dtype(np.uint16)]

    summary = session.owned_buffer_summary()
    layer = summary["full_attention_layers"][0]
    assert summary["kv_storage_dtype"] == "bf16"
    assert summary["full_attention_kv_scale_bytes"] == 0
    assert layer["storage_dtype"] == "bf16"
    assert layer["payload_dtype"] == "bf16"
    assert layer["scale_metadata"] is None


def test_qwen35_resident_full_kv_allocation_uses_int8_payload_and_scales() -> None:
    session, captured = _resident_allocation_session(storage_dtype="int8_per_token_head", scale_dtype=DType.FP16)

    session._allocate_full_attention_cache(3)

    key_cache, value_cache, key_buf, value_buf = session.full_caches[3]
    assert key_cache.dtype is DType.INT8
    assert value_cache.dtype is DType.INT8
    assert key_cache.shape == session.batch_layout.slot0_full_kv_shape
    payload_slot_bytes = np.prod(session.batch_layout.slot0_full_kv_shape) * DType.INT8.itemsize
    assert key_buf.nbytes == np.prod(session.batch_layout.full_kv_shape) * DType.INT8.itemsize
    assert value_buf.nbytes == key_buf.nbytes
    key_slot1, value_slot1 = session._slot_full_cache(3, 1)
    assert key_slot1.ptr == key_buf.ptr + payload_slot_bytes
    assert value_slot1.ptr == value_buf.ptr + payload_slot_bytes
    assert key_slot1.dtype is DType.INT8
    k_scale, v_scale, k_scale_buf, v_scale_buf = session.full_cache_scales[3]
    assert k_scale.shape == session.batch_layout.slot0_full_kv_scale_shape
    assert v_scale.shape == session.batch_layout.slot0_full_kv_scale_shape
    assert k_scale.dtype is DType.FP16
    assert k_scale_buf.nbytes == np.prod(session.batch_layout.flat_full_kv_scale_shape) * DType.FP16.itemsize
    assert v_scale_buf.nbytes == k_scale_buf.nbytes
    assert [item[2] for item in captured] == [
        np.dtype(np.int8),
        np.dtype(np.int8),
        np.dtype(np.float16),
        np.dtype(np.float16),
    ]

    slot1_metadata = session._slot_full_scale_metadata(3, 1)
    assert slot1_metadata is not None
    slot_scale_bytes = np.prod(session.batch_layout.slot0_full_kv_scale_shape) * DType.FP16.itemsize
    assert slot1_metadata.k_scale.ptr == k_scale_buf.ptr + slot_scale_bytes
    assert slot1_metadata.v_scale.ptr == v_scale_buf.ptr + slot_scale_bytes
    assert slot1_metadata.k_scale.shape == session.batch_layout.slot0_full_kv_scale_shape
    all_slots_metadata = session._full_cache_scale_metadata_all_slots(3)
    assert all_slots_metadata is not None
    assert all_slots_metadata.k_scale.shape == session.batch_layout.flat_full_kv_scale_shape

    summary = session.owned_buffer_summary()
    layer = summary["full_attention_layers"][0]
    assert summary["kv_storage_dtype"] == "int8_per_token_head"
    assert summary["kv_scale_dtype"] == "fp16"
    assert summary["kv_scale_granularity"] == "per_token_head"
    assert summary["full_attention_kv_payload_bytes"] == key_buf.nbytes + value_buf.nbytes
    assert summary["full_attention_kv_payload_bytes_per_element"] == 1.0
    assert summary["full_attention_kv_scale_bytes"] == k_scale_buf.nbytes + v_scale_buf.nbytes
    assert layer["storage_dtype"] == "int8_per_token_head"
    assert layer["payload_dtype"] == "int8"
    assert layer["payload_bytes_per_element"] == 1.0
    assert layer["scale_metadata"]["scale_dtype"] == "fp16"
    assert layer["scale_metadata"]["granularity"] == "per_token_head"

    audit = session.kv_memory_audit()
    assert audit["required"] is True
    assert audit["passed"] is True
    assert audit["retained_kv_payload_bytes_per_element"] == 1.0
    assert audit["retained_kv_payload_bytes"] == key_buf.nbytes + value_buf.nbytes
    assert audit["retained_kv_scale_bytes"] == k_scale_buf.nbytes + v_scale_buf.nbytes
    assert audit["persistent_bf16_kv_layers"] == []
    assert audit["bf16_shadow_candidates"] == []
    assert audit["persistent_bf16_shadow_exists"] is False


def test_qwen35_resident_kv_memory_audit_flags_persistent_bf16_kv_cache() -> None:
    session, _captured = _resident_allocation_session(storage_dtype="int8_per_token_head")
    session._allocate_full_attention_cache(3)
    key_cache, value_cache, _key_buf, _value_buf = session.full_caches[3]
    bf16_key = Tensor.from_handle(key_cache.ptr, key_cache.shape, DType.BF16, key_cache.device)
    bf16_value = Tensor.from_handle(value_cache.ptr, value_cache.shape, DType.BF16, value_cache.device)
    key_buf = DeviceBuffer(bf16_key.ptr, bf16_key.numel * DType.BF16.itemsize)
    value_buf = DeviceBuffer(bf16_value.ptr, bf16_value.numel * DType.BF16.itemsize)
    session.full_caches[3] = (bf16_key, bf16_value, key_buf, value_buf)

    audit = session.kv_memory_audit()

    assert audit["passed"] is False
    assert audit["persistent_bf16_kv_layers"] == [3]
    assert audit["payload_dtype_mismatch_layers"] == [3]
    assert audit["payload_element_size_mismatch_layers"] == [3]
    assert audit["persistent_bf16_shadow_exists"] is True


def test_qwen35_resident_kv_memory_audit_flags_persistent_bf16_shadow() -> None:
    session, _captured = _resident_allocation_session(storage_dtype="int8_per_token_head")
    session._allocate_full_attention_cache(3)
    shadow_tensor = _tensor(0x900000, session.batch_layout.slot0_full_kv_shape, DType.BF16)
    shadow_buffer = DeviceBuffer(shadow_tensor.ptr, shadow_tensor.numel * shadow_tensor.dtype.itemsize)
    shadow_allocation = SimpleNamespace(tensor=shadow_tensor, buffer=shadow_buffer)
    session.prefill_workspace = SimpleNamespace(
        names=("prefill.int8_oracle_key.3",),
        allocation=lambda name: shadow_allocation,
    )
    session.states = []

    audit = session.kv_memory_audit()

    assert audit["passed"] is False
    assert audit["persistent_bf16_shadow_exists"] is True
    assert audit["bf16_shadow_candidates"] == [
        {
            "workspace": "prefill_workspace",
            "name": "prefill.int8_oracle_key.3",
            "dtype": "bf16",
            "shape": list(session.batch_layout.slot0_full_kv_shape),
            "bytes": shadow_buffer.nbytes,
            "reasons": ["int8_prefill_oracle", "full_cache_shape"],
        }
    ]


def test_qwen35_resident_slot_full_spans_follow_int8_policy_metadata() -> None:
    session, _captured = _resident_allocation_session(storage_dtype="int8_per_token_head")
    session.max_sequence_length = 512
    session.block_table = _tensor(0x300000, (session.blocks,), DType.INT32)
    session.position_buf = DeviceBuffer(0x310000, session.max_batch_size * DType.INT64.itemsize)
    session.context_buf = DeviceBuffer(0x320000, session.max_batch_size * DType.INT64.itemsize)
    session._allocate_full_attention_cache(4)

    position, append_spans, decode_spans = session._slot_full_spans(4, 1)

    assert position.ptr == session.position_buf.ptr + DType.INT64.itemsize
    assert append_spans.storage_dtype is DType.INT8_PER_TOKEN_HEAD
    assert decode_spans.storage_dtype is DType.INT8_PER_TOKEN_HEAD
    assert append_spans.scale_metadata is not None
    assert decode_spans.scale_metadata is append_spans.scale_metadata
    assert append_spans.scale_metadata.k_scale.shape == session.batch_layout.slot0_full_kv_scale_shape
    slot_scale_elems = int(np.prod(session.batch_layout.slot0_full_kv_scale_shape))
    assert append_spans.scale_metadata.k_scale.ptr == session.full_cache_scales[4][2].ptr + slot_scale_elems * DType.FP16.itemsize


def test_qwen35_resident_slot_full_spans_use_live_counts_for_decode_threshold() -> None:
    session, _captured = _resident_allocation_session()
    session.max_sequence_length = 1024
    session.block_table = _tensor(0x300000, (session.blocks,), DType.INT32)
    session.position_buf = DeviceBuffer(0x310000, session.max_batch_size * DType.INT64.itemsize)
    session.context_buf = DeviceBuffer(0x320000, session.max_batch_size * DType.INT64.itemsize)
    session.position_arr = np.asarray([512, 7], dtype=np.int64)
    session.context_arr = session.position_arr + np.int64(1)

    _position, append_spans, decode_spans = session._slot_full_spans(0, 0)
    _position1, append_spans1, decode_spans1 = session._slot_full_spans(0, 1)

    assert append_spans.max_live_count == 512
    assert decode_spans.max_live_count == 513
    assert append_spans1.max_live_count == 7
    assert decode_spans1.max_live_count == 8


def test_qwen35_resident_native_prefill_layers_use_int8_retained_cache_and_bf16_oracle() -> None:
    device = Device("hip", 0)
    session, _captured = _resident_allocation_session(storage_dtype="int8_per_token_head")
    session.config = SimpleNamespace(
        hidden_size=8,
        layer_types=("full_attention",),
        num_key_value_heads=2,
        head_dim=4,
        linear_conv_kernel_dim=1,
    )
    session.max_sequence_length = 8
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.prefill_config = PrefillConfig(attn_aotriton_min_tokens=1)
    session.prefill_hidden = _tensor(0x1000, (4, 8), DType.FP16)
    session.prefill_next_hidden = _tensor(0x2000, (4, 8), DType.FP16)
    session.prefill_positions = _tensor(0x3000, (4,), DType.INT64)
    session.prefill_context_count_buf = DeviceBuffer(0x4000, 4 * DType.INT64.itemsize)
    session.prefill_block_table_buf = DeviceBuffer(0x5000, 4 * session.blocks * DType.INT32.itemsize)
    session.cos = _tensor(0x6000, (8, 4), DType.FP32)
    session.sin = _tensor(0x7000, (8, 4), DType.FP32)
    session.libraries = {}
    session._allocate_full_attention_cache(0)
    session._full_attention_prefill_layer_chunk_size = MethodType(lambda self, tokens: 2, session)
    session._prefill_single_cu_seqlens_pair = MethodType(
        lambda self, query_tokens, key_tokens: (
            _tensor(0x5100 + int(query_tokens), (2,), DType.INT32),
            _tensor(0x5200 + int(key_tokens), (2,), DType.INT32),
        ),
        session,
    )
    session._ensure_full_prefill_scratch = MethodType(lambda self, *, tokens, **_kwargs: object(), session)
    session._ensure_moe_prefill_scratch = MethodType(lambda self, layer_id=None, *, tokens: object(), session)

    class FakeRuntime:
        def __init__(self) -> None:
            self.memcpy_async_calls = []

        def memcpy_async(self, *args):
            self.memcpy_async_calls.append(args)

    class FakeWorkspace:
        def __init__(self) -> None:
            self.calls = []
            self.next_ptr = 0x8000

        def reserve_tensor(self, name, shape, dtype):
            self.calls.append((name, tuple(shape), DType.parse(dtype)))
            tensor = Tensor.from_handle(self.next_ptr, tuple(shape), dtype, device)
            self.next_ptr += tensor.numel * tensor.dtype.itemsize + 0x100
            return tensor

    class FakeFullPrefillState:
        def __init__(self) -> None:
            self.run_calls = []

        def run_full_attention_moe_prefill_layer_fp16(self, hidden, **kwargs):
            self.run_calls.append((hidden, kwargs))
            return Tensor.from_handle(0xA000 + len(self.run_calls) * 0x100, hidden.shape, DType.FP16, device)

    runtime = FakeRuntime()
    workspace = FakeWorkspace()
    state = FakeFullPrefillState()
    session.runtime = runtime
    session.prefill_workspace = workspace
    session.states = [state]

    out = session._run_native_prefill_layers(tokens=4)

    assert out.shape == (4, 8)
    assert len(state.run_calls) == 2
    assert [call[1]["tokens"] for call in state.run_calls] == [2, 2]
    assert all(call[1]["aotriton_attention"] is True for call in state.run_calls)
    assert [call[1]["aotriton_kv_rows"] for call in state.run_calls] == [2, 4]
    assert all(call[1]["cu_seqlens_q"] is not None for call in state.run_calls)
    assert all(call[1]["cu_seqlens_k"] is not None for call in state.run_calls)
    assert [item[0] for item in workspace.calls] == ["prefill.int8_oracle_key", "prefill.int8_oracle_value"]
    for _hidden, kwargs in state.run_calls:
        assert kwargs["key_cache"].dtype is DType.BF16
        assert kwargs["value_cache"].dtype is DType.BF16
        assert kwargs["append_spans"].storage_dtype is DType.BF16
        assert kwargs["prefill_spans"].storage_dtype is DType.BF16
        assert kwargs["retained_key_cache"].dtype is DType.INT8
        assert kwargs["retained_value_cache"].dtype is DType.INT8
        assert kwargs["retained_append_spans"].storage_dtype is DType.INT8_PER_TOKEN_HEAD
        assert kwargs["retained_append_spans"].scale_metadata is not None
        assert kwargs["retained_append_spans"].scale_metadata.k_scale.dtype is DType.FP16
    assert len(runtime.memcpy_async_calls) == 2
    assert [call[0] for call in runtime.memcpy_async_calls] == [0x1000, 0x1000 + 2 * session.hidden_nbytes]


def test_qwen35_resident_native_prefill_plan_accepts_full_attention_layers() -> None:
    layer_types = ("linear_attention", "linear_attention", "full_attention", "linear_attention")
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.layer_limit = 4
    session.config = SimpleNamespace(layer_types=layer_types)

    plan = session.native_prefill_plan()
    pure_plan = qwen35_paro_native_prefill_plan(layer_types, layer_limit=4)

    assert pure_plan == plan

    assert plan.path == "single_request_native_full"
    assert plan.layer_limit == 4
    assert plan.linear_prefix_layers == 2
    assert plan.full_layer_limit_native
    assert plan.first_unsupported_layer is None
    assert plan.first_unsupported_type is None
    assert plan.blockers == ()
    assert plan.to_json_dict()["linear_prefix_layers"] == 2


def test_qwen35_resident_native_prefill_plan_rejects_invalid_layer_limit() -> None:
    with pytest.raises(ValueError, match="exceeds available"):
        qwen35_paro_native_prefill_plan(("linear_attention",), layer_limit=2)


def test_qwen35_resident_native_prefill_plan_reports_unknown_layer_blocker() -> None:
    plan = qwen35_paro_native_prefill_plan(("linear_attention", "weird"), layer_limit=2)

    assert plan.path == "unsupported_layer_type"
    assert not plan.full_layer_limit_native
    assert plan.first_unsupported_layer == 1
    assert plan.first_unsupported_type == "weird"
    assert any("first unsupported layer 1" in blocker for blocker in plan.blockers)


def test_qwen35_resident_native_prefill_plan_accepts_all_linear_layer_limit() -> None:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.layer_limit = 2
    session.config = SimpleNamespace(layer_types=("linear_attention", "linear_attention", "full_attention"))

    plan = session.native_prefill_plan()

    assert plan.path == "single_request_native_full"
    assert plan.linear_prefix_layers == 2
    assert plan.full_layer_limit_native
    assert plan.first_unsupported_layer is None
    assert plan.first_unsupported_type is None
    assert plan.blockers == ()


def _prefill_validation_session() -> Qwen35ParoResidentSession:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.closed = False
    session.max_sequence_length = 8
    session.vocab_size = 100
    session.layer_limit = 2
    session.config = SimpleNamespace(
        layer_types=("linear_attention", "full_attention"),
        linear_conv_kernel_dim=4,
    )
    session.prefill_config = PrefillConfig()
    session._check_position = MethodType(lambda self, position: None, session)
    return session


def test_qwen35_resident_prefill_linear_tokens_native_validates_prompt_tokens() -> None:
    session = _prefill_validation_session()

    with pytest.raises(ValueError, match="token_ids must be non-empty"):
        session.prefill_linear_tokens_native([], sample=True)
    with pytest.raises(ValueError, match="outside"):
        session.prefill_linear_tokens_native([100], sample=True)


def test_prefill_config_validates_chunk_sizes_and_defaults_to_full_native() -> None:
    config = PrefillConfig(linear_chunk_size="4", require_full_native=False)

    assert config.linear_chunk_size == 4
    assert config.attn_aotriton_min_tokens == 512
    assert config.require_full_native is False
    assert config.auto_tune_chunk_sizes is True
    assert config.moe_grouped_device_gather is True
    assert PrefillConfig(attn_aotriton_min_tokens="1024").attn_aotriton_min_tokens == 1024
    assert PrefillConfig(moe_chunk_size="1024").moe_chunk_size == 1024

    with pytest.raises(ValueError, match="chunk_tune_memory_budget_gib"):
        PrefillConfig(chunk_tune_memory_budget_gib=-1)
    with pytest.raises(ValueError, match="full_attn_query_chunk_size"):
        PrefillConfig(full_attn_query_chunk_size=-1)
    with pytest.raises(ValueError, match="moe_chunk_size"):
        PrefillConfig(moe_chunk_size=-1)
    with pytest.raises(ValueError, match="attn_aotriton_min_tokens"):
        PrefillConfig(attn_aotriton_min_tokens=-1)


def test_prefill_config_autotunes_gt1k_chunks_from_budget() -> None:
    short, short_tuning = resolve_prefill_config_for_sequence(
        PrefillConfig(),
        max_sequence_length=1024,
        total_memory_bytes=48 * 1024**3,
    )
    assert short.linear_chunk_size == 0
    assert short_tuning["reason"] == "below_min_tokens"

    mid, mid_tuning = resolve_prefill_config_for_sequence(
        PrefillConfig(),
        max_sequence_length=1025,
        total_memory_bytes=48 * 1024**3,
    )
    assert (mid.linear_chunk_size, mid.moe_chunk_size, mid.full_attn_query_chunk_size) == (1024, 1024, 4096)
    assert mid.full_attn_post_chunk_size == 1024
    assert mid.full_attn_rope_chunk_size == 1024
    assert mid_tuning["reason"] == "manual_long_equiv_gt1k"

    very_long, very_long_tuning = resolve_prefill_config_for_sequence(
        PrefillConfig(),
        max_sequence_length=131072 + 129,
        total_memory_bytes=48 * 1024**3,
    )
    assert very_long.full_attn_query_chunk_size == 4096
    assert very_long_tuning["reason"] == "manual_long_equiv_gt1k"

    budget_limited, budget_tuning = resolve_prefill_config_for_sequence(
        PrefillConfig(chunk_tune_memory_budget_gib=24.0),
        max_sequence_length=131072 + 129,
    )
    assert budget_limited.full_attn_query_chunk_size == 4096
    assert budget_tuning["reason"] == "manual_long_equiv_gt1k"

    manual, manual_tuning = resolve_prefill_config_for_sequence(
        PrefillConfig(linear_chunk_size=2048),
        max_sequence_length=131072 + 129,
        total_memory_bytes=48 * 1024**3,
    )
    assert manual.linear_chunk_size == 2048
    assert manual.full_attn_query_chunk_size == 0
    assert manual_tuning["reason"] == "manual_chunk_sizes"


def test_qwen35_resident_decode_split_config_caps_128k_context(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_PAGED_ATTN_MAX_SPLITS", raising=False)
    monkeypatch.delenv("NANOVLLM_AMD_PAGED_ATTN_MAX_SPLITS", raising=False)

    assert runner_module._paged_attn_decode_split_config(32768 + 129, block_size=256, chunk_size=256) == (256, 129)
    assert runner_module._paged_attn_decode_split_config(131072 + 129, block_size=256, chunk_size=256) == (256, 513)

    monkeypatch.setenv("HIPENGINE_PAGED_ATTN_MAX_SPLITS", "512")
    assert runner_module._paged_attn_decode_split_config(131072 + 129, block_size=256, chunk_size=256) == (512, 257)
    monkeypatch.setenv("HIPENGINE_PAGED_ATTN_MAX_SPLITS", "128")
    assert runner_module._paged_attn_decode_split_config(32768 + 129, block_size=256, chunk_size=256) == (512, 65)


def test_qwen35_resident_prefill_chunk_helpers_select_safe_ranges() -> None:
    assert Qwen35ParoResidentSession._chunk_ranges(5, 2) == ((0, 2), (2, 4), (4, 5))
    assert Qwen35ParoResidentSession._chunk_ranges(7, 2, min_chunk_size=4) == ((0, 2), (2, 7))
    assert Qwen35ParoResidentSession._chunk_ranges(4, 0) == ((0, 4),)

    session = _prefill_validation_session()
    session.prefill_config = PrefillConfig(
        linear_chunk_size=1024,
        moe_chunk_size=512,
        full_attn_query_chunk_size=4096,
        full_attn_post_chunk_size=1024,
    )

    assert session._linear_prefill_layer_chunk_size(4096) == 512
    assert session._full_attention_prefill_layer_chunk_size(8192) == 4096


def test_qwen35_resident_prefill_native_contract_uses_full_native_by_default() -> None:
    session = _prefill_validation_session()
    calls: list[tuple[tuple[int, ...], bool]] = []

    def fake_full(self, token_ids, *, sample=True):
        calls.append((tuple(token_ids), bool(sample)))
        return "full-native-result"

    session._prefill_tokens_native_full = MethodType(fake_full, session)

    with pytest.raises(ValueError, match="linear_conv_kernel_dim"):
        session.prefill_native([1, 2, 3], sample=False)
    assert session.prefill_native([1, 2, 3, 4], sample=False) == "full-native-result"
    assert calls == [((1, 2, 3, 4), False)]


def test_qwen35_resident_prefill_native_allows_explicit_oracle_bringup_path() -> None:
    session = _prefill_validation_session()
    calls: list[tuple[tuple[int, ...], bool, bool]] = []

    def fake_legacy(self, token_ids, *, sample=True, allow_rejected_correctness=False):
        calls.append((tuple(token_ids), bool(sample), bool(allow_rejected_correctness)))
        return "legacy-result"

    session._prefill_linear_tokens_native_legacy = MethodType(fake_legacy, session)

    result = session.prefill_native([1, 2, 3, 4], sample=False, require_full_native=False)

    assert result == "legacy-result"
    assert calls == [((1, 2, 3, 4), False, False)]


def test_qwen35_resident_prefill_native_uses_config_default_for_full_native() -> None:
    session = _prefill_validation_session()
    session.prefill_config = PrefillConfig(require_full_native=False)
    session._prefill_linear_tokens_native_legacy = MethodType(
        lambda self, token_ids, *, sample=True, allow_rejected_correctness=False: tuple(token_ids),
        session,
    )

    assert session.prefill_native([1, 2, 3, 4], sample=False) == (1, 2, 3, 4)


def test_qwen35_resident_prefill_native_packed_wires_metadata_layers_and_commit(monkeypatch) -> None:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.closed = False
    session.max_batch_size = 2
    session.max_sequence_length = 8
    session.blocks = 1
    session.block_size = 256
    session.config = SimpleNamespace(hidden_size=4)
    session.vocab_size = 100
    session.libraries = {"runtime_state": object()}
    session.embedding = SimpleNamespace(tensor=_tensor(0x1000, (100, 4), DType.FP16))
    session.prefill_hidden = _tensor(0x1800, (8, 4), DType.FP16)
    calls: list[str] = []

    class FakeRuntime:
        def stream_synchronize(self, stream):
            calls.append(f"sync:{stream}")

    session.runtime = FakeRuntime()
    session.native_prefill_plan = lambda: SimpleNamespace(
        full_layer_limit_native=True,
        blockers=(),
        linear_prefix_layers=1,
        layer_limit=2,
    )
    metadata = SimpleNamespace(token_ids=_tensor(0x2000, (3,), DType.INT64), temp_buffers=())
    session._materialize_packed_prefill_metadata = lambda slab: calls.append("metadata") or metadata
    hidden = _tensor(0x3000, (3, 4), DType.FP16)
    session._run_native_prefill_packed_layers = lambda slab, metadata, stream=0: calls.append("layers") or hidden
    session._commit_packed_prefill_final_rows = (
        lambda hidden_arg, slab, sample=True, stream=0: calls.append(f"commit:{sample}") or ("result",)
    )
    session._restore_decode_scratch_after_prefill = lambda: calls.append("restore")
    monkeypatch.setattr(runner_module, "embedding_lookup_batch_fp16_i64", lambda *args, **kwargs: calls.append("embed"))
    slab = CompactPromptSlab.from_token_rows(
        request_ids=(10, 11),
        token_rows=((1, 2), (3,)),
        start_positions=(0, 0),
        block_count=1,
        slot_ids=(0, 1),
    )

    assert session.prefill_native_packed(slab, sample=False) == ("result",)

    assert calls == ["metadata", "embed", "layers", "sync:0", "commit:False", "restore"]
    assert session.last_prefill_execution["path"] == "native_prefill_compact_cN"
    assert session.last_prefill_execution["slot_ids"] == [0, 1]
    assert session.last_prefill_execution["linear_attention_prefill_path"] == "packed_segments"
    assert session.last_prefill_execution["full_attention_prefill_path"] == "packed_varlen"
    assert session.last_prefill_execution["blockers"] == []


def test_qwen35_resident_trace_prefill_linear_input_copies_bits(monkeypatch) -> None:
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.config = SimpleNamespace(hidden_size=3)
    synced: list[int] = []
    session.runtime = SimpleNamespace(stream_synchronize=lambda stream: synced.append(int(stream)))
    session._prefill_linear_input_trace = []
    arrays: dict[int, np.ndarray] = {}

    def fake_host_array_ptr(array):
        arrays[0xABC] = array
        return 0xABC

    def fake_copy_device_to_host(host_ptr, buffer, *, runtime=None):
        assert int(buffer.ptr) == 0x1000
        assert int(buffer.nbytes) == 2 * 3 * DType.FP16.itemsize
        arrays[int(host_ptr)][:] = np.array([[0x3C00, 0x4000, 0x4200], [0x4400, 0x4500, 0x4600]], dtype=np.uint16)

    monkeypatch.setattr(runner_module, "host_array_ptr", fake_host_array_ptr)
    monkeypatch.setattr(runner_module, "copy_device_to_host", fake_copy_device_to_host)

    session._trace_prefill_linear_input(
        layer_id=5,
        hidden=Tensor.from_handle(0x1000, (2, 3), DType.FP16, device),
        rows=2,
        stream=7,
    )

    assert synced == [7]
    assert session._prefill_linear_input_trace[0]["layer_index"] == 5
    np.testing.assert_array_equal(
        session._prefill_linear_input_trace[0]["bits"],
        np.array([[0x3C00, 0x4000, 0x4200], [0x4400, 0x4500, 0x4600]], dtype=np.uint16),
    )


def test_qwen35_resident_run_native_prefill_packed_layers_can_force_per_segment_linear(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR", "1")
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.config = SimpleNamespace(hidden_size=4, layer_types=("linear_attention",))
    session.prefill_hidden = Tensor.from_handle(0x1000, (3, 4), DType.FP16, device)
    session.hidden_nbytes = 4 * DType.FP16.itemsize
    session.max_batch_size = 3
    session.libraries = {}
    conv = Tensor.from_handle(0x3000, (2, 2), DType.FP32, device)
    recurrent = Tensor.from_handle(0x4000, (1, 2, 1), DType.FP32, device)
    conv_nbytes = 2 * 2 * DType.FP32.itemsize
    recurrent_nbytes = 1 * 2 * 1 * DType.FP32.itemsize
    session.linear_states = {
        0: (
            conv,
            recurrent,
            SimpleNamespace(ptr=0x3000, nbytes=3 * conv_nbytes),
            SimpleNamespace(ptr=0x4000, nbytes=3 * recurrent_nbytes),
            np.zeros((3, 2, 2), dtype=np.float32),
            np.zeros((3, 1, 2, 1), dtype=np.float32),
        )
    }
    session._ensure_linear_prefill_scratch = lambda *, tokens: SimpleNamespace(name="linear", tokens=tokens)
    session._ensure_moe_prefill_scratch = lambda layer_id, *, tokens: SimpleNamespace(name="moe", layer_id=layer_id, tokens=tokens)
    copies: list[tuple[int, int, int, int]] = []

    class FakeRuntime:
        def memcpy_async(self, dst, src, nbytes, kind, stream):
            copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.calls = []

        def run_linear_attention_moe_c1_layer_fp16(self, hidden, **kwargs):
            self.calls.append((hidden, kwargs))
            return Tensor.from_handle(0x8000 + 0x100 * len(self.calls), hidden.shape, DType.FP16, device)

    state = FakeState()
    session.runtime = FakeRuntime()
    session.states = [state]
    slab = SimpleNamespace(rows=3, request_count=2, cu_seqlens_q=(0, 2, 3), physical_slot_ids=(0, 2))
    metadata = SimpleNamespace()

    out = session._run_native_prefill_packed_layers(slab, metadata, stream=9)

    assert out.ptr == 0x1000
    assert session._last_packed_prefill_linear_path == "per_segment"
    assert session._last_packed_prefill_blockers == ["linear-attention packed prefill forced to per-segment diagnostic path"]
    assert [call[0].ptr for call in state.calls] == [0x1000, 0x1000 + 2 * session.hidden_nbytes]
    assert [call[1]["tokens"] for call in state.calls] == [2, 1]
    assert [call[1]["conv_state"].ptr for call in state.calls] == [0x3000, 0x3000 + 2 * conv_nbytes]
    assert [call[1]["recurrent_state"].ptr for call in state.calls] == [0x4000, 0x4000 + 2 * recurrent_nbytes]
    assert copies == [
        (0x1000, 0x8100, 2 * session.hidden_nbytes, 9),
        (0x1000 + 2 * session.hidden_nbytes, 0x8200, session.hidden_nbytes, 9),
    ]


def test_qwen35_resident_run_native_prefill_packed_layers_uses_aotriton_varlen_when_resolved(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN", raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR", raising=False)
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.config = SimpleNamespace(hidden_size=4, layer_types=("full_attention",))
    session.prefill_hidden = Tensor.from_handle(0x1000, (4, 4), DType.FP16, device)
    session.hidden_nbytes = 4 * DType.FP16.itemsize
    session.block_size = 256
    session.max_sequence_length = 8
    session.libraries = {}
    session.cos = Tensor.from_handle(0x2000, (8, 2), DType.FP16, device)
    session.sin = Tensor.from_handle(0x3000, (8, 2), DType.FP16, device)
    key = Tensor.from_handle(0x5000, (4, 256, 1, 2), DType.BF16, device)
    value = Tensor.from_handle(0x6000, (4, 256, 1, 2), DType.BF16, device)
    session._full_cache_all_slots = lambda layer_id: (key, value)
    session._prefill_use_aotriton_attention_resolved = lambda rows: True
    session._ensure_full_prefill_scratch = lambda *, tokens, aotriton_attention=False: SimpleNamespace(
        name="attention",
        tokens=tokens,
        aotriton_attention=aotriton_attention,
    )
    session._ensure_grouped_moe_prefill_scratch = lambda layer_id, *, tokens: SimpleNamespace(
        name="grouped_moe",
        layer_id=layer_id,
        tokens=tokens,
    )
    copies: list[tuple[int, int, int, int]] = []

    class FakeRuntime:
        def memcpy_async(self, dst, src, nbytes, kind, stream):
            copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.calls = []

        def run_full_attention_moe_prefill_varlen_layer_fp16(self, hidden, **kwargs):
            self.calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9000, hidden.shape, DType.FP16, device)

    state = FakeState()
    session.runtime = FakeRuntime()
    session.states = [state]
    slab = SimpleNamespace(rows=4, request_count=2, cu_seqlens_q=(0, 2, 4), physical_slot_ids=(0, 1))
    metadata = SimpleNamespace(
        append_spans="append",
        prefill_spans="prefill",
        cu_seqlens_q=Tensor.from_handle(0xA000, (3,), DType.INT32, device),
        cu_seqlens_k=Tensor.from_handle(0xB000, (3,), DType.INT32, device),
        positions=Tensor.from_handle(0xC000, (4,), DType.INT64, device),
    )

    out = session._run_native_prefill_packed_layers(slab, metadata, stream=3)

    assert out.ptr == 0x1000
    assert session._last_packed_prefill_full_attention_path == "packed_varlen_aotriton"
    assert session._last_packed_prefill_blockers == []
    assert state.calls[0][0].ptr == 0x1000
    assert state.calls[0][1]["aotriton_attention"] is True
    assert state.calls[0][1]["aotriton_max_seqlen_q"] == 2
    assert state.calls[0][1]["aotriton_max_seqlen_k"] == 2
    assert state.calls[0][1]["attention_scratch"].aotriton_attention is True
    assert state.calls[0][1]["tokens"] == 4
    assert copies == [(0x1000, 0x9000, 4 * session.hidden_nbytes, 3)]


def test_qwen35_resident_run_native_prefill_packed_layers_can_force_per_segment_full_attention(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN", "1")
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.config = SimpleNamespace(hidden_size=4, layer_types=("full_attention",))
    session.prefill_hidden = Tensor.from_handle(0x1000, (4, 4), DType.FP16, device)
    session.hidden_nbytes = 4 * DType.FP16.itemsize
    session.block_size = 256
    session.max_sequence_length = 8
    session.libraries = {}
    session.cos = Tensor.from_handle(0x2000, (8, 2), DType.FP16, device)
    session.sin = Tensor.from_handle(0x3000, (8, 2), DType.FP16, device)
    session.prefill_positions = Tensor.from_handle(0x4000, (4,), DType.INT64, device)
    session.prefill_block_table_buf = SimpleNamespace(ptr=0xA000, nbytes=4 * DType.INT32.itemsize)
    session.prefill_context_count_buf = SimpleNamespace(ptr=0xB000, nbytes=4 * DType.INT64.itemsize)
    copies: list[tuple[int, int, int, int]] = []
    local_table_copies: list[tuple[int, int]] = []

    def fake_copy_host_to_device(buffer, host_ptr, nbytes=None, *, runtime=None):
        local_table_copies.append((int(buffer.ptr), int(buffer.nbytes if nbytes is None else nbytes)))

    monkeypatch.setattr(runner_module, "copy_host_to_device", fake_copy_host_to_device)
    session._slot_full_cache = lambda layer_id, slot: (
        Tensor.from_handle(0x5000 + int(slot) * 0x100, (4, 256, 1, 2), DType.BF16, device),
        Tensor.from_handle(0x6000 + int(slot) * 0x100, (4, 256, 1, 2), DType.BF16, device),
    )
    session._prefill_rows_tensor = lambda tensor, rows, start=0: Tensor.from_handle(0x7000 + int(start) * 8, (rows,), DType.INT64, device)
    session._prefill_use_aotriton_attention_resolved = lambda tokens: False
    session._ensure_full_prefill_scratch = lambda *, tokens, aotriton_attention=False: SimpleNamespace(
        name="attention",
        tokens=tokens,
        aotriton_attention=aotriton_attention,
    )
    session._ensure_moe_prefill_scratch = lambda layer_id, *, tokens: SimpleNamespace(name="moe", layer_id=layer_id, tokens=tokens)

    class FakeRuntime:
        def memcpy_async(self, dst, src, nbytes, kind, stream):
            copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.calls = []

        def run_full_attention_moe_prefill_layer_fp16(self, hidden, **kwargs):
            self.calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9000 + 0x100 * len(self.calls), hidden.shape, DType.FP16, device)

    state = FakeState()
    session.runtime = FakeRuntime()
    session.states = [state]
    slab = SimpleNamespace(
        rows=4,
        request_count=2,
        cu_seqlens_q=(0, 2, 4),
        physical_slot_ids=(0, 2),
        block_count=1,
        block_tables=((0,), (0,), (0,), (0,)),
    )
    metadata = SimpleNamespace()

    out = session._run_native_prefill_packed_layers(slab, metadata, stream=5)

    assert out.ptr == 0x1000
    assert session._last_packed_prefill_linear_path == "packed_segments"
    assert session._last_packed_prefill_full_attention_path == "per_segment"
    assert session._last_packed_prefill_blockers == ["full-attention packed prefill forced to per-segment diagnostic path"]
    assert [call[0].ptr for call in state.calls] == [0x1000, 0x1000 + 2 * session.hidden_nbytes]
    assert [call[1]["tokens"] for call in state.calls] == [2, 2]
    assert [call[1]["key_cache"].ptr for call in state.calls] == [0x5000, 0x5000 + 2 * 0x100]
    assert [call[1]["positions"].ptr for call in state.calls] == [0x7000, 0x7000 + 2 * 8]
    assert [call[1]["append_spans"].base_offsets.shape for call in state.calls] == [(2, 1), (2, 1)]
    assert [call[1]["append_spans"].base_offsets.ptr for call in state.calls] == [0xA000, 0xA000 + 2 * DType.INT32.itemsize]
    assert [call[1]["prefill_spans"].live_counts.ptr for call in state.calls] == [0xB000, 0xB000 + 2 * DType.INT64.itemsize]
    assert local_table_copies == [(0xA000, 2 * DType.INT32.itemsize), (0xA000 + 2 * DType.INT32.itemsize, 2 * DType.INT32.itemsize)]
    assert copies == [
        (0x1000, 0x9100, 2 * session.hidden_nbytes, 5),
        (0x1000 + 2 * session.hidden_nbytes, 0x9200, 2 * session.hidden_nbytes, 5),
    ]


def test_qwen35_resident_commit_packed_prefill_final_rows_updates_slots(monkeypatch) -> None:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.max_batch_size = 3
    session.max_sequence_length = 8
    session.device = Device("hip", 0)
    session.config = SimpleNamespace(hidden_size=4)
    session.hidden_nbytes = 4 * DType.FP16.itemsize
    session.batch_hidden = _tensor(0x1000, (3, 4), DType.FP16)
    session.position_arr = np.zeros((3,), dtype=np.int64)
    session.context_arr = np.ones((3,), dtype=np.int64)
    session.position_buf = SimpleNamespace(ptr=0x2000, nbytes=session.position_arr.nbytes)
    session.context_buf = SimpleNamespace(ptr=0x3000, nbytes=session.context_arr.nbytes)
    copies: list[tuple[int, int, int, int]] = []

    class FakeRuntime:
        def memcpy_async(self, dst, src, nbytes, kind, stream):
            copies.append((int(dst), int(src), int(nbytes), int(stream)))

    session.runtime = FakeRuntime()
    monkeypatch.setattr(runner_module, "copy_host_to_device", lambda *args, **kwargs: copies.append((0, 0, 0, -1)))
    sampled: list[int] = []
    session._sample_from_hidden = lambda hidden: sampled.append(hidden.ptr) or SimpleNamespace(token_id=hidden.ptr)
    slab = CompactPromptSlab.from_token_rows(
        request_ids=(10, 11),
        token_rows=((1, 2), (3, 4, 5)),
        start_positions=(0, 4),
        block_count=1,
        slot_ids=(2, 0),
    )
    hidden = _tensor(0x8000, (5, 4), DType.FP16)

    result = session._commit_packed_prefill_final_rows(hidden, slab, sample=True)

    assert [item.token_id for item in result] == [0x1000 + 2 * session.hidden_nbytes, 0x1000]
    assert sampled == [0x1000 + 2 * session.hidden_nbytes, 0x1000]
    assert (0x1000 + 2 * session.hidden_nbytes, 0x8000 + 1 * session.hidden_nbytes, session.hidden_nbytes, 0) in copies
    assert (0x1000, 0x8000 + 4 * session.hidden_nbytes, session.hidden_nbytes, 0) in copies
    assert session.position_arr.tolist() == [6, 0, 1]
    assert session.context_arr.tolist() == [7, 1, 2]


def test_qwen35_resident_target_verify_batch_materializes_metadata_only() -> None:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.closed = False
    session.max_batch_size = 5
    session.max_sequence_length = 16
    session.vocab_size = 100
    session.device = Device("hip", 0)
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )

    target = session.target_verify_batch(draft, root_tokens=(9, 19), root_positions=(5, 3))

    assert target.request_ids == (1, 2)
    assert target.rows == 5
    assert target.candidate_count == 3
    assert target.tokens == (9, 19, 10, 11, 20)
    assert target.positions == (5, 3, 6, 7, 4)
    assert target.parent_rows == (-1, -1, 0, 2, 1)
    assert target.tree_shape == (0, 1, 0)
    assert target.mode == "verify_tree"

    buffers = session.verify_speculative_batch(
        target,
        token_ids=_tensor(0x3000, (5,), "int32"),
        positions=_tensor(0x3100, (5,), "int32"),
        parent_rows=_tensor(0x3200, (5,), "int32"),
        draft_depths=_tensor(0x3300, (5,), "int32"),
        row_to_request=_tensor(0x3400, (5,), "int32"),
        active_mask=_tensor(0x3500, (5,), "bool"),
        target_top1=_tensor(0x3600, (5,), "int32"),
        accepted_counts=_tensor(0x3700, (2,), "int32"),
        commit_rows=_tensor(0x3800, (2,), "int32"),
        commit_tokens=_tensor(0x3900, (2,), "int32"),
        commit_positions=_tensor(0x3A00, (2,), "int32"),
        next_tokens=_tensor(0x3B00, (2,), "int32"),
        transaction_id=7,
    )
    assert buffers.transaction_id == 7
    assert buffers.candidate_counts == (2, 1)
    assert buffers.draft_depth == 2
    assert buffers.tree_shape == (0, 1, 0)
    assert buffers.next_tokens is not None
    assert buffers.next_tokens.shape == (2,)
    assert buffers.rows == 5
    assert buffers.candidate_rows == 3
    assert buffers.request_count == 2
    assert str(buffers.device) == "hip:0"

    with pytest.raises(ValueError, match="transaction_id"):
        session.verify_speculative_batch(
            target,
            token_ids=_tensor(0x3B00, (5,), "int32"),
            positions=_tensor(0x3C00, (5,), "int32"),
            parent_rows=_tensor(0x3D00, (5,), "int32"),
            draft_depths=_tensor(0x3E00, (5,), "int32"),
            row_to_request=_tensor(0x3F00, (5,), "int32"),
            active_mask=_tensor(0x4000, (5,), "bool"),
            target_top1=_tensor(0x4100, (5,), "int32"),
            accepted_counts=_tensor(0x4200, (2,), "int32"),
            commit_rows=_tensor(0x4300, (2,), "int32"),
            commit_tokens=_tensor(0x4400, (2,), "int32"),
            commit_positions=_tensor(0x4500, (2,), "int32"),
            transaction_id=-1,
        )

    other_device = Device("hip", 1)
    with pytest.raises(ValueError, match="resident device"):
        session.verify_speculative_batch(
            target,
            token_ids=Tensor.from_handle(0x4200, (5,), "int32", other_device),
            positions=Tensor.from_handle(0x4300, (5,), "int32", other_device),
            parent_rows=Tensor.from_handle(0x4400, (5,), "int32", other_device),
            draft_depths=Tensor.from_handle(0x4500, (5,), "int32", other_device),
            row_to_request=Tensor.from_handle(0x4600, (5,), "int32", other_device),
            active_mask=Tensor.from_handle(0x4700, (5,), "bool", other_device),
            target_top1=Tensor.from_handle(0x4800, (5,), "int32", other_device),
            accepted_counts=Tensor.from_handle(0x4900, (2,), "int32", other_device),
            commit_rows=Tensor.from_handle(0x4A00, (2,), "int32", other_device),
            commit_tokens=Tensor.from_handle(0x4B00, (2,), "int32", other_device),
            commit_positions=Tensor.from_handle(0x4C00, (2,), "int32", other_device),
        )

    plan = TargetCommitPlan(
        transaction_id=0,
        request_ids=(1, 2),
        accepted_counts=(2, 1),
        commit_rows=(3, 4),
        commit_tokens=(11, 20),
        commit_positions=(7, 4),
        candidate_counts=(2, 1),
        mode="verify_tree",
    )
    state_buffers = TargetStateCommitBuffers.for_plan(
        plan,
        accepted_counts=_tensor(0x3B00, (2,), "int32"),
        commit_rows=_tensor(0x3C00, (2,), "int32"),
        commit_positions=_tensor(0x3D00, (2,), "int32"),
        linear_state_src=_tensor(0x3E00, (5, 40, 128), "bf16"),
        linear_state_dst=_tensor(0x3F00, (2, 40, 128), "bf16"),
        kv_rows_src=_tensor(0x4000, (5, 8, 128), "bf16"),
        kv_rows_dst=_tensor(0x4100, (3, 8, 128), "bf16"),
    )
    assert state_buffers.transaction_id == plan.transaction_id
    assert session.commit_verified_state(plan, state_buffers) is state_buffers
    wrong_transaction_buffers = replace(state_buffers, transaction_id=plan.transaction_id + 1)
    with pytest.raises(ValueError, match="transaction_id"):
        session.commit_verified_state(plan, wrong_transaction_buffers)
    short_linear_src = TargetStateCommitBuffers.for_plan(
        plan,
        accepted_counts=_tensor(0x4200, (2,), "int32"),
        commit_rows=_tensor(0x4300, (2,), "int32"),
        commit_positions=_tensor(0x4400, (2,), "int32"),
        linear_state_src=_tensor(0x4500, (4, 40, 128), "bf16"),
        linear_state_dst=_tensor(0x4600, (2, 40, 128), "bf16"),
    )
    with pytest.raises(ValueError, match="selected commit rows"):
        session.commit_verified_state(plan, short_linear_src)
    short_kv_dst = TargetStateCommitBuffers.for_plan(
        plan,
        accepted_counts=_tensor(0x4700, (2,), "int32"),
        commit_rows=_tensor(0x4800, (2,), "int32"),
        commit_positions=_tensor(0x4900, (2,), "int32"),
        kv_rows_src=_tensor(0x4A00, (5, 8, 128), "bf16"),
        kv_rows_dst=_tensor(0x4B00, (2, 8, 128), "bf16"),
    )
    with pytest.raises(ValueError, match="accepted token rows"):
        session.commit_verified_state(plan, short_kv_dst)
    with pytest.raises(ValueError, match="request_ids"):
        session.commit_verified_state(
            TargetCommitPlan(
                transaction_id=0,
                request_ids=(1,),
                accepted_counts=(1,),
                commit_rows=(3,),
                commit_tokens=(11,),
                commit_positions=(7,),
                candidate_counts=(1,),
                mode="verify_tree",
            ),
            state_buffers,
        )

    with pytest.raises(ValueError, match="row tensors"):
        session.verify_speculative_batch(
            target,
            token_ids=_tensor(0x3000, (4,), "int32"),
            positions=_tensor(0x3100, (5,), "int32"),
            parent_rows=_tensor(0x3200, (5,), "int32"),
            draft_depths=_tensor(0x3300, (5,), "int32"),
            row_to_request=_tensor(0x3400, (5,), "int32"),
            active_mask=_tensor(0x3500, (5,), "bool"),
            target_top1=_tensor(0x3600, (5,), "int32"),
            accepted_counts=_tensor(0x3700, (2,), "int32"),
            commit_rows=_tensor(0x3800, (2,), "int32"),
            commit_tokens=_tensor(0x3900, (2,), "int32"),
            commit_positions=_tensor(0x3A00, (2,), "int32"),
        )

    session.max_batch_size = 4
    with pytest.raises(ValueError, match="max_batch_size"):
        session.target_verify_batch(draft, root_tokens=(9, 19), root_positions=(5, 3))
    session.max_batch_size = 5
    with pytest.raises(ValueError, match="outside"):
        session.target_verify_batch(draft, root_tokens=(9, 100), root_positions=(5, 3))


def test_qwen35_resident_speculative_execution_metadata_stays_blocked() -> None:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)

    metadata = session.speculative_execution_metadata()

    assert isinstance(metadata, Qwen35ParoResidentSpeculativeExecution)
    assert metadata.target_verify_batch_metadata
    assert metadata.verify_speculative_batch_metadata
    assert metadata.commit_verified_state_metadata
    assert not metadata.native_target_verify_executes_kernels
    assert not metadata.commit_verified_state_executes_copies
    assert not metadata.native_target_verify_ready
    assert not metadata.throughput_claim_eligible
    assert any("metadata-only" in blocker for blocker in metadata.blockers)
    payload = metadata.to_json_dict()
    assert payload["native_target_verify_batch"]
    assert payload["speculative_verify_batch"]
    assert payload["commit_verified_state"]
    assert not payload["native_target_verify_ready"]


def test_qwen35_resident_batch_execution_metadata_labels_serial_fallback() -> None:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.layer_limit = 3
    session.config = SimpleNamespace(layer_types=("linear_attention", "linear_attention", "full_attention"))

    metadata = session.batch_execution_metadata(scheduler_owned=True)

    assert metadata.path == "scheduler_serial_slot_bridge"
    assert metadata.scheduler_owned
    assert metadata.row_execution == "serial_c1_layer_path"
    assert metadata.native_prefill_plan.linear_prefix_layers == 2
    assert metadata.native_prefill_plan.full_layer_limit_native
    assert metadata.native_compact_prefill
    assert not metadata.native_caware_decode
    assert not metadata.throughput_claim_eligible
    assert any("decode" in blocker for blocker in metadata.blockers)
    payload = metadata.to_json_dict()
    assert payload["native_prefill_plan"]["linear_prefix_layers"] == 2
    assert payload["blockers"] == list(metadata.blockers)


def test_qwen35_resident_batch_execution_metadata_keeps_native_diagnostics_ineligible() -> None:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.layer_limit = 3
    session.config = SimpleNamespace(layer_types=("linear_attention", "linear_attention", "full_attention"))

    metadata = session.batch_execution_metadata(scheduler_owned=True, native_decode=True, active_rows=4)

    assert metadata.path == "scheduler_native_compact_batch"
    assert metadata.native_caware_decode
    assert not metadata.throughput_claim_eligible
    assert metadata.projection_dispatch is not None
    assert metadata.projection_dispatch["rows"] == 4
    assert metadata.projection_dispatch["path"] == "row_gemv_until_caware_benchmark"
    assert metadata.projection_dispatch["selected_candidate"] == "row_gemv"
    assert any("generated-token equality" in blocker for blocker in metadata.blockers)
    assert any("projection dispatch: no c-aware projection candidate applies" in blocker for blocker in metadata.blockers)
    assert metadata.to_json_dict()["projection_dispatch"] == metadata.projection_dispatch


def test_qwen35_resident_step_batch_native_requires_experimental_env(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE", raising=False)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.closed = False

    with pytest.raises(NotImplementedError, match="HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE"):
        session.step_batch_native([1], positions=[0], slots=[0])


def test_qwen35_resident_step_batch_native_rejects_int8_kv_when_experimental(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE", "1")
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.closed = False
    session.kv_storage_dtype = DType.INT8_PER_TOKEN_HEAD

    with pytest.raises(NotImplementedError, match="BF16 KV"):
        session.step_batch_native([1], positions=[0], slots=[0])


def test_qwen35_resident_step_batch_native_accepts_sparse_slots(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE", "1")
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.closed = False
    session.kv_storage_dtype = DType.BF16
    session.max_batch_size = 3
    session.max_sequence_length = 16
    calls: list[tuple[str, object]] = []

    class FakeRuntime:
        def device_synchronize(self):
            calls.append(("sync", None))

    def fake_set_tokens(tokens, *, stream=0):
        calls.append(("tokens", (tuple(tokens), stream)))

    def fake_set_positions(positions, *, stream=0):
        calls.append(("positions", (tuple(positions), stream)))

    def fake_run_layers(*, rows, positions, slots, stream=0):
        calls.append(("run", (rows, tuple(positions), tuple(slots), stream)))
        return Tensor.from_handle(0x7000, (rows, 8), DType.FP16, Device("hip", 0))

    session.runtime = FakeRuntime()
    session._set_batch_token_embeddings = fake_set_tokens
    session._set_batch_positions = fake_set_positions
    session._run_layers_batch_decode = fake_run_layers

    results = session.step_batch_native([10, 20], positions=[5, 6], slots=[0, 2], sample=False)

    assert results == (None, None)
    assert calls == [
        ("tokens", ((10, 20), 0)),
        ("positions", ((5, 6), 0)),
        ("run", (2, (5, 6), (0, 2), 0)),
        ("sync", None),
    ]


@pytest.mark.parametrize(
    ("slots", "match"),
    [
        ((1, 0), "physical-slot order"),
        ((0, 0), "unique"),
        ((0, 3), "within max_batch_size"),
    ],
)
def test_qwen35_resident_step_batch_native_rejects_invalid_sparse_slots(monkeypatch, slots, match) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE", "1")
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.closed = False
    session.kv_storage_dtype = DType.BF16
    session.max_batch_size = 3

    with pytest.raises((ValueError, NotImplementedError), match=match):
        session.step_batch_native([1, 2], positions=[0, 0], slots=slots)


def test_qwen35_resident_batch_full_spans_maps_sparse_slots(monkeypatch) -> None:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.blocks = 3
    session.prefill_block_table_buf = DeviceBuffer(0x1000, 2 * 3 * DType.INT32.itemsize)
    session.position_buf = DeviceBuffer(0x2000, 2 * DType.INT64.itemsize)
    session.context_buf = DeviceBuffer(0x3000, 2 * DType.INT64.itemsize)
    session.device = Device("hip", 0)
    session.kv_storage_dtype = DType.BF16
    session.runtime = object()
    captured: list[np.ndarray] = []

    def fake_host_array_ptr(array):
        captured.append(np.asarray(array).copy())
        return 0xDEADBEEF

    def fake_copy_host_to_device(*args, **kwargs):
        return None

    monkeypatch.setattr(runner_module, "host_array_ptr", fake_host_array_ptr)
    monkeypatch.setattr(runner_module, "copy_host_to_device", fake_copy_host_to_device)

    position_tensor, append_spans, decode_spans = session._batch_full_spans(
        0,
        rows=2,
        positions=(5, 6),
        slots=(0, 2),
    )

    assert len(captured) == 1
    assert np.array_equal(captured[0], np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32))
    assert position_tensor.ptr == 0x2000
    assert append_spans.base_offsets.ptr == 0x1000
    assert decode_spans.base_offsets.ptr == 0x1000
    assert append_spans.max_live_count == 6
    assert decode_spans.max_live_count == 7
    assert session._last_batch_full_spans_metadata == {
        "layer_index": 0,
        "rows": 2,
        "slots": [0, 2],
        "positions": [5, 6],
        "append_live_counts": [5, 6],
        "decode_live_counts": [6, 7],
        "append_max_live_count": 6,
        "decode_max_live_count": 7,
        "block_size": 256,
        "block_table_len_per_row": 3,
        "block_table_rows": [[0, 1, 2], [3, 4, 5]],
        "storage_dtype": "bf16",
    }


def test_qwen35_resident_run_layers_batch_decode_reports_native_batch_for_short_context() -> None:
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.layer_limit = 1
    session.config = SimpleNamespace(hidden_size=8, layer_types=("full_attention",))
    session.batch_hidden = Tensor.from_handle(0x1000, (2, 8), DType.FP16, device)
    session.batch_next_hidden = Tensor.from_handle(0x2000, (2, 8), DType.FP16, device)
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.decode_chunk_size = 512
    session.max_sequence_length = 1024
    session.cos = Tensor.from_handle(0xA000, (1,), DType.BF16, device)
    session.sin = Tensor.from_handle(0xB000, (1,), DType.BF16, device)
    session.libraries = {}
    session._batch_decode_segment_metadata = lambda *, rows, slots: (
        Tensor.from_handle(0x3000, (rows + 1,), DType.INT32, device),
        Tensor.from_handle(0x4000, (rows,), DType.INT64, device),
        (),
    )
    session._full_cache_all_slots = lambda layer_id: (
        Tensor.from_handle(0x5000, (1,), DType.BF16, device),
        Tensor.from_handle(0x6000, (1,), DType.BF16, device),
    )
    session._batch_full_spans = lambda layer_id, *, rows, positions, slots: (
        Tensor.from_handle(0x7000, (rows,), DType.INT64, device),
        SimpleNamespace(rows=rows, slots=slots, span="append"),
        SimpleNamespace(rows=rows, slots=slots, span="decode"),
    )
    session._ensure_full_decode_batch_scratch = lambda layer_id, rows: SimpleNamespace(name="attention", rows=rows)
    session._ensure_moe_decode_batch_scratch = lambda layer_id, rows: SimpleNamespace(name="moe", rows=rows)
    copies: list[tuple[int, int, int, int]] = []

    class FakeRuntime:
        def memcpy_async(self, dst, src, nbytes, kind, stream):
            copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.calls = []

        def run_full_attention_moe_decode_batch_layer_fp16(self, hidden, **kwargs):
            self.calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9000, (kwargs["tokens"], 8), DType.FP16, device)

    state = FakeState()
    session.runtime = FakeRuntime()
    session.states = [state]

    out = session._run_layers_batch_decode(rows=2, positions=(4, 7), slots=(0, 2), stream=5)

    assert out.ptr == 0x2000
    assert len(state.calls) == 1
    hidden, kwargs = state.calls[0]
    assert hidden.ptr == 0x1000
    assert kwargs["tokens"] == 2
    assert kwargs["key_cache"].ptr == 0x5000
    assert kwargs["value_cache"].ptr == 0x6000
    assert copies == [(0x2000, 0x9000, 2 * session.hidden_nbytes, 5)]
    assert session.last_batch_decode_execution == {
        "rows": 2,
        "slots": [0, 2],
        "max_full_attention_context": 8,
        "native_full_attention_layers": 1,
        "full_attention_decode_path": "native_batch",
        "native_caware_decode": True,
        "linear_attention_segment_metadata": {"cu_seqlens": [0, 1, 2], "state_indices": [0, 2]},
        "moe_decode_path": "grouped_compact",
        "moe_decode_rows": 2,
        "moe_grouped_compact_layers": 1,
        "moe_selected_c1_fallback_layers": 0,
        "layer_executions": [
            {
                "layer_index": 0,
                "layer_type": "full_attention",
                "rows": 2,
                "slots": [0, 2],
                "max_context": 8,
                "full_attention_decode_path": "native_batch",
                "native_caware_decode": True,
                "moe_decode_path": "grouped_compact",
                "attn_context_trace_source": "attention_scratch.query_raw",
            }
        ],
        "blockers": [],
    }
    metadata = session.batch_execution_metadata(scheduler_owned=True, native_decode=True)
    assert metadata.native_caware_decode
    assert metadata.row_execution == "native_compact_caware_layers"
    assert metadata.decode_execution == session.last_batch_decode_execution
    assert metadata.to_json_dict()["decode_execution"]["full_attention_decode_path"] == "native_batch"


def test_qwen35_resident_run_layers_batch_decode_can_force_per_row_post_attention_probe(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_POST_ATTN", "1")
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.layer_limit = 1
    session.config = SimpleNamespace(hidden_size=8, layer_types=("full_attention",))
    session.batch_hidden = Tensor.from_handle(0x1000, (2, 8), DType.FP16, device)
    session.batch_next_hidden = Tensor.from_handle(0x2000, (2, 8), DType.FP16, device)
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.decode_chunk_size = 512
    session.max_sequence_length = 1024
    session.cos = Tensor.from_handle(0xA000, (1,), DType.BF16, device)
    session.sin = Tensor.from_handle(0xB000, (1,), DType.BF16, device)
    session.libraries = {}
    session._batch_decode_segment_metadata = lambda *, rows, slots: (
        Tensor.from_handle(0x3000, (rows + 1,), DType.INT32, device),
        Tensor.from_handle(0x4000, (rows,), DType.INT64, device),
        (),
    )
    session._full_cache_all_slots = lambda layer_id: (
        Tensor.from_handle(0x5000, (1,), DType.BF16, device),
        Tensor.from_handle(0x6000, (1,), DType.BF16, device),
    )
    session._batch_full_spans = lambda layer_id, *, rows, positions, slots: (
        Tensor.from_handle(0x7000, (rows,), DType.INT64, device),
        SimpleNamespace(rows=rows, slots=slots, span="append"),
        SimpleNamespace(rows=rows, slots=slots, span="decode"),
    )
    session._ensure_full_decode_batch_scratch = lambda layer_id, rows: SimpleNamespace(name="attention", rows=rows)
    session._ensure_moe_decode_batch_scratch = lambda layer_id, rows: SimpleNamespace(name="moe", rows=rows)
    copies: list[tuple[int, int, int, int]] = []

    class FakeRuntime:
        def memcpy_async(self, dst, src, nbytes, kind, stream):
            copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.calls = []

        def run_full_attention_moe_decode_batch_layer_fp16(self, hidden, **kwargs):
            self.calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9000, (kwargs["tokens"], 8), DType.FP16, device)

    state = FakeState()
    session.runtime = FakeRuntime()
    session.states = [state]

    out = session._run_layers_batch_decode(rows=2, positions=(4, 7), slots=(0, 2), stream=5)

    assert out.ptr == 0x2000
    assert len(state.calls) == 1
    assert state.calls[0][1]["force_per_row_post_attention"] is True
    assert copies == [(0x2000, 0x9000, 2 * session.hidden_nbytes, 5)]
    assert session.last_batch_decode_execution["full_attention_decode_path"] == "native_batch"
    assert session.last_batch_decode_execution["native_caware_decode"] is False
    assert session.last_batch_decode_execution["blockers"] == [
        "post-attention add/rmsnorm forced to per-row diagnostic path"
    ]
    layer_execution = session.last_batch_decode_execution["layer_executions"][0]
    assert layer_execution["native_caware_decode"] is False
    assert layer_execution["post_attention_decode_path"] == "per_row_add_rmsnorm_fallback"
    metadata = session.batch_execution_metadata(scheduler_owned=True, native_decode=True)
    assert not metadata.native_caware_decode
    assert metadata.row_execution == "native_batch_with_diagnostic_fallback"
    assert "post-attention add/rmsnorm forced to per-row diagnostic path" in metadata.blockers


def test_qwen35_resident_run_layers_batch_decode_can_force_per_row_full_attention_input_probe(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_INPUT", "1")
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.layer_limit = 1
    session.config = SimpleNamespace(hidden_size=8, layer_types=("full_attention",))
    session.batch_hidden = Tensor.from_handle(0x1000, (2, 8), DType.FP16, device)
    session.batch_next_hidden = Tensor.from_handle(0x2000, (2, 8), DType.FP16, device)
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.decode_chunk_size = 512
    session.max_sequence_length = 1024
    session.cos = Tensor.from_handle(0xA000, (1,), DType.BF16, device)
    session.sin = Tensor.from_handle(0xB000, (1,), DType.BF16, device)
    session.libraries = {}
    session._batch_decode_segment_metadata = lambda *, rows, slots: (
        Tensor.from_handle(0x3000, (rows + 1,), DType.INT32, device),
        Tensor.from_handle(0x4000, (rows,), DType.INT64, device),
        (),
    )
    session._full_cache_all_slots = lambda layer_id: (
        Tensor.from_handle(0x5000, (1,), DType.BF16, device),
        Tensor.from_handle(0x6000, (1,), DType.BF16, device),
    )
    session._batch_full_spans = lambda layer_id, *, rows, positions, slots: (
        Tensor.from_handle(0x7000, (rows,), DType.INT64, device),
        SimpleNamespace(rows=rows, slots=slots, span="append"),
        SimpleNamespace(rows=rows, slots=slots, span="decode"),
    )
    session._ensure_full_decode_batch_scratch = lambda layer_id, rows: SimpleNamespace(name="attention", rows=rows)
    session._ensure_moe_decode_batch_scratch = lambda layer_id, rows: SimpleNamespace(name="moe", rows=rows)
    copies: list[tuple[int, int, int, int]] = []

    class FakeRuntime:
        def memcpy_async(self, dst, src, nbytes, kind, stream):
            copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.calls = []

        def run_full_attention_moe_decode_batch_layer_fp16(self, hidden, **kwargs):
            self.calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9000, (kwargs["tokens"], 8), DType.FP16, device)

    state = FakeState()
    session.runtime = FakeRuntime()
    session.states = [state]

    out = session._run_layers_batch_decode(rows=2, positions=(4, 7), slots=(0, 2), stream=5)

    assert out.ptr == 0x2000
    assert len(state.calls) == 1
    assert state.calls[0][1]["force_per_row_input_rmsnorm"] is True
    assert copies == [(0x2000, 0x9000, 2 * session.hidden_nbytes, 5)]
    assert session.last_batch_decode_execution["full_attention_decode_path"] == "native_batch"
    assert session.last_batch_decode_execution["native_caware_decode"] is False
    assert session.last_batch_decode_execution["blockers"] == [
        "full-attention input RMSNorm forced to per-row diagnostic path"
    ]
    layer_execution = session.last_batch_decode_execution["layer_executions"][0]
    assert layer_execution["native_caware_decode"] is False
    assert layer_execution["full_attention_input_decode_path"] == "per_row_rmsnorm_fallback"
    metadata = session.batch_execution_metadata(scheduler_owned=True, native_decode=True)
    assert not metadata.native_caware_decode
    assert metadata.row_execution == "native_batch_with_diagnostic_fallback"
    assert "full-attention input RMSNorm forced to per-row diagnostic path" in metadata.blockers


def test_qwen35_resident_run_layers_batch_decode_can_force_selected_c1_moe_probe(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE", "1")
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.layer_limit = 1
    session.config = SimpleNamespace(hidden_size=8, layer_types=("full_attention",), num_experts=4)
    session.batch_hidden = Tensor.from_handle(0x1000, (2, 8), DType.FP16, device)
    session.batch_next_hidden = Tensor.from_handle(0x2000, (2, 8), DType.FP16, device)
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.decode_chunk_size = 512
    session.max_sequence_length = 1024
    session.cos = Tensor.from_handle(0xA000, (1,), DType.BF16, device)
    session.sin = Tensor.from_handle(0xB000, (1,), DType.BF16, device)
    session.libraries = {}
    session._batch_decode_segment_metadata = lambda *, rows, slots: (
        Tensor.from_handle(0x3000, (rows + 1,), DType.INT32, device),
        Tensor.from_handle(0x4000, (rows,), DType.INT64, device),
        (),
    )
    session._full_cache_all_slots = lambda layer_id: (
        Tensor.from_handle(0x5000, (1,), DType.BF16, device),
        Tensor.from_handle(0x6000, (1,), DType.BF16, device),
    )
    session._batch_full_spans = lambda layer_id, *, rows, positions, slots: (
        Tensor.from_handle(0x7000, (rows,), DType.INT64, device),
        SimpleNamespace(rows=rows, slots=slots, span="append"),
        SimpleNamespace(rows=rows, slots=slots, span="decode"),
    )
    session._ensure_full_decode_batch_scratch = lambda layer_id, rows: SimpleNamespace(name="attention", rows=rows)
    force_flags: list[bool] = []

    def fake_moe_scratch(layer_id, rows, *, force_selected_c1_moe=False):
        force_flags.append(bool(force_selected_c1_moe))
        return SimpleNamespace(name="moe", rows=rows, force_selected_c1_moe=force_selected_c1_moe)

    session._ensure_moe_decode_batch_scratch = fake_moe_scratch
    copies: list[tuple[int, int, int, int]] = []

    class FakeRuntime:
        def memcpy_async(self, dst, src, nbytes, kind, stream):
            copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.calls = []

        def run_full_attention_moe_decode_batch_layer_fp16(self, hidden, **kwargs):
            self.calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9000, (kwargs["tokens"], 8), DType.FP16, device)

    state = FakeState()
    session.runtime = FakeRuntime()
    session.states = [state]

    out = session._run_layers_batch_decode(rows=2, positions=(4, 7), slots=(0, 2), stream=5)

    assert out.ptr == 0x2000
    assert force_flags == [True]
    assert state.calls[0][1]["force_selected_c1_moe"] is True
    assert copies == [(0x2000, 0x9000, 2 * session.hidden_nbytes, 5)]
    assert session.last_batch_decode_execution == {
        "rows": 2,
        "slots": [0, 2],
        "max_full_attention_context": 8,
        "native_full_attention_layers": 1,
        "full_attention_decode_path": "native_batch",
        "native_caware_decode": False,
        "linear_attention_segment_metadata": {"cu_seqlens": [0, 1, 2], "state_indices": [0, 2]},
        "moe_decode_path": "selected_c1_forced",
        "moe_decode_rows": 2,
        "moe_grouped_compact_layers": 0,
        "moe_selected_c1_fallback_layers": 1,
        "layer_executions": [
            {
                "layer_index": 0,
                "layer_type": "full_attention",
                "rows": 2,
                "slots": [0, 2],
                "max_context": 8,
                "full_attention_decode_path": "native_batch",
                "native_caware_decode": False,
                "moe_decode_path": "selected_c1_forced",
                "attn_context_trace_source": "attention_scratch.query_raw",
            }
        ],
        "blockers": ["MoE decode forced to selected-c1 diagnostic path"],
    }


def test_qwen35_resident_run_layers_batch_decode_uses_per_row_splitk_fallback_for_long_context(monkeypatch) -> None:
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.layer_limit = 1
    session.config = SimpleNamespace(hidden_size=8, layer_types=("full_attention",))
    session.batch_hidden = Tensor.from_handle(0x1000, (2, 8), DType.FP16, device)
    session.batch_next_hidden = Tensor.from_handle(0x2000, (2, 8), DType.FP16, device)
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.decode_chunk_size = 512
    session.max_sequence_length = 2048
    session.cos = Tensor.from_handle(0xA000, (1,), DType.BF16, device)
    session.sin = Tensor.from_handle(0xB000, (1,), DType.BF16, device)
    session.libraries = {}
    session.full_scratch = {0: SimpleNamespace(name="full")}
    session.moe_scratch = {0: SimpleNamespace(name="moe")}
    session._batch_decode_segment_metadata = lambda *, rows, slots: (
        Tensor.from_handle(0x3000, (rows + 1,), DType.INT32, device),
        Tensor.from_handle(0x4000, (rows,), DType.INT64, device),
        (),
    )
    session._slot_full_cache = lambda layer_id, slot: (
        Tensor.from_handle(0x5000 + slot * 0x100, (1,), DType.BF16, device),
        Tensor.from_handle(0x6000 + slot * 0x100, (1,), DType.BF16, device),
    )
    session._slot_full_spans = lambda layer_id, slot: (
        Tensor.from_handle(0x7000 + slot * 8, (1,), DType.INT64, device),
        SimpleNamespace(slot=slot, span="append"),
        SimpleNamespace(slot=slot, span="decode"),
    )
    copies: list[tuple[int, int, int, int]] = []

    class FakeRuntime:
        def memcpy_async(self, dst, src, nbytes, kind, stream):
            copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.calls = []

        def run_full_attention_moe_c1_layer_fp16(self, hidden, **kwargs):
            self.calls.append((hidden, kwargs))
            slot = kwargs["append_spans"].slot
            return Tensor.from_handle(0x9000 + slot * 0x100, (1, 8), DType.FP16, device)

    state = FakeState()
    session.runtime = FakeRuntime()
    session.states = [state]

    out = session._run_layers_batch_decode(rows=2, positions=(1023, 1024), slots=(0, 2), stream=7)

    assert out.ptr == 0x2000
    assert [call[0].ptr for call in state.calls] == [0x1000, 0x1000 + session.hidden_nbytes]
    assert [call[1]["append_spans"].slot for call in state.calls] == [0, 2]
    assert [call[1]["num_splits"] for call in state.calls] == [2, 3]
    assert copies == [
        (0x2000, 0x9000, session.hidden_nbytes, 7),
        (0x2000 + session.hidden_nbytes, 0x9000 + 2 * 0x100, session.hidden_nbytes, 7),
    ]
    assert session.last_batch_decode_execution == {
        "rows": 2,
        "slots": [0, 2],
        "max_full_attention_context": 1025,
        "native_full_attention_layers": 0,
        "full_attention_decode_path": "per_row_splitk_fallback",
        "native_caware_decode": False,
        "linear_attention_segment_metadata": {"cu_seqlens": [0, 1, 2], "state_indices": [0, 2]},
        "moe_decode_path": "mixed_grouped_compact_with_per_row_full_attention_fallback",
        "moe_decode_rows": 2,
        "moe_grouped_compact_layers": 0,
        "moe_selected_c1_fallback_layers": 1,
        "layer_executions": [
            {
                "layer_index": 0,
                "layer_type": "full_attention",
                "rows": 2,
                "slots": [0, 2],
                "max_context": 1025,
                "full_attention_decode_path": "per_row_splitk_fallback",
                "native_caware_decode": False,
                "moe_decode_path": "selected_c1_per_row_fallback",
                "num_splits_per_row": [2, 3],
            }
        ],
        "blockers": ["full-attention decode used a per-row fallback"],
    }
    metadata = session.batch_execution_metadata(scheduler_owned=True, native_decode=True)
    assert not metadata.native_caware_decode
    assert metadata.row_execution == "native_linear_batch_with_per_row_full_attention_fallback"
    assert metadata.decode_execution == session.last_batch_decode_execution
    assert any("per-row fallback" in blocker for blocker in metadata.blockers)
    assert metadata.to_json_dict()["decode_execution"]["full_attention_decode_path"] == "per_row_splitk_fallback"
    assert metadata.to_json_dict()["decode_execution"]["blockers"] == ["full-attention decode used a per-row fallback"]


def test_qwen35_resident_run_layers_batch_decode_reports_selected_c1_with_per_row_full_fallback(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE", "1")
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_FULL_ATTN_NATIVE", "0")
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.layer_limit = 1
    session.config = SimpleNamespace(hidden_size=8, layer_types=("full_attention",), num_experts=4)
    session.batch_hidden = Tensor.from_handle(0x1000, (2, 8), DType.FP16, device)
    session.batch_next_hidden = Tensor.from_handle(0x2000, (2, 8), DType.FP16, device)
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.decode_chunk_size = 512
    session.max_sequence_length = 1024
    session.cos = Tensor.from_handle(0xA000, (1,), DType.BF16, device)
    session.sin = Tensor.from_handle(0xB000, (1,), DType.BF16, device)
    session.libraries = {}
    session.full_scratch = {0: SimpleNamespace(name="full")}
    session.moe_scratch = {0: SimpleNamespace(name="moe")}
    session._batch_decode_segment_metadata = lambda *, rows, slots: (
        Tensor.from_handle(0x3000, (rows + 1,), DType.INT32, device),
        Tensor.from_handle(0x4000, (rows,), DType.INT64, device),
        (),
    )
    session._slot_full_cache = lambda layer_id, slot: (
        Tensor.from_handle(0x5000 + slot * 0x100, (1,), DType.BF16, device),
        Tensor.from_handle(0x6000 + slot * 0x100, (1,), DType.BF16, device),
    )
    session._slot_full_spans = lambda layer_id, slot: (
        Tensor.from_handle(0x7000 + slot * 8, (1,), DType.INT64, device),
        SimpleNamespace(slot=slot, span="append"),
        SimpleNamespace(slot=slot, span="decode"),
    )
    copies: list[tuple[int, int, int, int]] = []

    class FakeRuntime:
        def memcpy_async(self, dst, src, nbytes, kind, stream):
            copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.calls = []

        def run_full_attention_moe_c1_layer_fp16(self, hidden, **kwargs):
            self.calls.append((hidden, kwargs))
            slot = kwargs["append_spans"].slot
            return Tensor.from_handle(0x9000 + slot * 0x100, (1, 8), DType.FP16, device)

    state = FakeState()
    session.runtime = FakeRuntime()
    session.states = [state]

    out = session._run_layers_batch_decode(rows=2, positions=(4, 7), slots=(0, 2), stream=7)

    assert out.ptr == 0x2000
    assert [call[0].ptr for call in state.calls] == [0x1000, 0x1000 + session.hidden_nbytes]
    assert copies == [
        (0x2000, 0x9000, session.hidden_nbytes, 7),
        (0x2000 + session.hidden_nbytes, 0x9000 + 2 * 0x100, session.hidden_nbytes, 7),
    ]
    assert session.last_batch_decode_execution["full_attention_decode_path"] == "per_row_context_fallback"
    assert session.last_batch_decode_execution["moe_decode_path"] == "selected_c1_forced_with_per_row_full_attention_fallback"
    assert session.last_batch_decode_execution["moe_grouped_compact_layers"] == 0
    assert session.last_batch_decode_execution["moe_selected_c1_fallback_layers"] == 1
    assert session.last_batch_decode_execution["blockers"] == [
        "MoE decode forced to selected-c1 diagnostic path",
        "full-attention decode used a per-row fallback",
    ]


def test_qwen35_resident_step_batch_native_accepts_long_context_for_splitk_fallback(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE", "1")
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.closed = False
    session.kv_storage_dtype = DType.BF16
    session.max_batch_size = 2
    session.max_sequence_length = 2048
    calls: list[tuple[str, object]] = []

    class FakeRuntime:
        def device_synchronize(self):
            calls.append(("sync", None))

    session.runtime = FakeRuntime()
    session._set_batch_token_embeddings = lambda tokens, *, stream=0: calls.append(("tokens", (tuple(tokens), stream)))
    session._set_batch_positions = lambda positions, *, stream=0: calls.append(("positions", (tuple(positions), stream)))

    def fake_run_layers(*, rows, positions, slots, stream=0):
        calls.append(("run", (rows, tuple(positions), tuple(slots), stream)))
        return Tensor.from_handle(0x7100, (rows, 8), DType.FP16, Device("hip", 0))

    session._run_layers_batch_decode = fake_run_layers

    results = session.step_batch_native([1, 2], positions=[1023, 1023], slots=[0, 1], sample=False)

    assert results == (None, None)
    assert calls == [
        ("tokens", ((1, 2), 0)),
        ("positions", ((1023, 1023), 0)),
        ("run", (2, (1023, 1023), (0, 1), 0)),
        ("sync", None),
    ]


def test_qwen35_resident_sample_batch_defaults_to_serial_lm_head(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_QWEN35_BATCH_SAMPLE_MODE", raising=False)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.max_batch_size = 2
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.config = SimpleNamespace(hidden_size=8)
    hidden = Tensor.from_handle(0x5000, (2, 8), DType.FP16, Device("hip", 0))
    sampled_ptrs: list[int] = []

    def fake_sample(row_hidden):
        sampled_ptrs.append(row_hidden.ptr)
        return SimpleNamespace(token_id=row_hidden.ptr)

    session._sample_from_hidden = fake_sample

    results = session._sample_batch_from_hidden(hidden, rows=2)

    assert [result.token_id for result in results] == [0x5000, 0x5000 + session.hidden_nbytes]
    assert sampled_ptrs == [0x5000, 0x5000 + session.hidden_nbytes]


def test_qwen35_resident_sample_batch_batched_lm_head_falls_back_without_c2_evidence(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_SAMPLE_MODE", "batched_lm_head")
    monkeypatch.delenv("HIPENGINE_QWEN35_BATCH_SAMPLE_C2_EQ_OK", raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN35_BATCH_SAMPLE_EQ_ARTIFACT", raising=False)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.max_batch_size = 2
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.config = SimpleNamespace(hidden_size=8)
    hidden = Tensor.from_handle(0x5100, (2, 8), DType.FP16, Device("hip", 0))
    sampled_ptrs: list[int] = []

    def fake_sample(row_hidden):
        sampled_ptrs.append(row_hidden.ptr)
        return SimpleNamespace(token_id=row_hidden.ptr)

    session._sample_from_hidden = fake_sample

    results = session._sample_batch_from_hidden(hidden, rows=2)

    assert [result.token_id for result in results] == [0x5100, 0x5100 + session.hidden_nbytes]
    assert sampled_ptrs == [0x5100, 0x5100 + session.hidden_nbytes]
    assert session.last_batch_sampler_execution["requested_mode"] == "batched_lm_head"
    assert session.last_batch_sampler_execution["mode"] == "serial_lm_head"
    assert session.last_batch_sampler_execution["native_row_aware_lm_head"] is False
    assert "batched LM-head requires green c>N generated-token equality evidence" in session.last_batch_sampler_execution["blockers"]


def test_qwen35_resident_sample_batch_requires_retained_equality_artifact(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_SAMPLE_MODE", "batched_lm_head")
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_SAMPLE_C2_EQ_OK", "1")
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_SAMPLE_EQ_ARTIFACT", "/tmp/qwen35-c2-eq.json")
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.max_batch_size = 2
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.config = SimpleNamespace(hidden_size=8)
    hidden = Tensor.from_handle(0x5200, (2, 8), DType.FP16, Device("hip", 0))
    sampled_ptrs: list[int] = []

    def fake_sample(row_hidden):
        sampled_ptrs.append(row_hidden.ptr)
        return SimpleNamespace(token_id=row_hidden.ptr)

    session._sample_from_hidden = fake_sample

    results = session._sample_batch_from_hidden(hidden, rows=2)

    assert [result.token_id for result in results] == [0x5200, 0x5200 + session.hidden_nbytes]
    assert sampled_ptrs == [0x5200, 0x5200 + session.hidden_nbytes]
    assert session.last_batch_sampler_execution["mode"] == "serial_lm_head"
    assert "batched LM-head equality artifact path must be under benchmarks/results" in session.last_batch_sampler_execution["blockers"]


def test_qwen35_resident_sample_batch_rejects_unknown_mode(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_SAMPLE_MODE", "surprise")
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.max_batch_size = 1
    hidden = Tensor.from_handle(0x6000, (1, 8), DType.FP16, Device("hip", 0))

    with pytest.raises(ValueError, match="HIPENGINE_QWEN35_BATCH_SAMPLE_MODE"):
        session._sample_batch_from_hidden(hidden, rows=1)


class _FakePrefillRuntime:
    def __init__(self) -> None:
        self.memcpy_async_calls = []

    def memcpy_async(self, *args):
        self.memcpy_async_calls.append(args)


class _FakePrefillState:
    def __init__(self, device: Device) -> None:
        self.device = device
        self.linear_reservations = []
        self.moe_reservations = []
        self.grouped_reservations = []
        self.run_calls = []

    def reserve_linear_attention_scratch(self, *, tokens: int, activation_dtype):
        scratch = SimpleNamespace(
            attn_input=Tensor.from_handle(0x10000 + tokens * 0x100, (tokens, 8), DType.parse(activation_dtype), self.device),
        )
        self.linear_reservations.append(scratch)
        return scratch

    def reserve_moe_c1_scratch(self, *, tokens: int, activation_dtype):
        scratch = SimpleNamespace(
            normed=Tensor.from_handle(0x20000 + tokens * 0x100, (tokens, 8), DType.parse(activation_dtype), self.device),
        )
        self.moe_reservations.append(scratch)
        return scratch

    def reserve_moe_grouped_prefill_scratch(self, *, tokens: int, activation_dtype):
        tensor = Tensor.from_handle(0x24000 + tokens * 0x100, (tokens, 8), DType.parse(activation_dtype), self.device)
        scratch = Qwen35ParoGroupedMoeScratch(
            normed=tensor,
            residual=tensor,
            router_logits=tensor,
            routing_weights=tensor,
            selected_experts=tensor,
            counts=tensor,
            padded_counts=tensor,
            expert_start=tensor,
            total_padded=tensor,
            scatter_offsets=tensor,
            sorted_lanes=tensor,
            sorted_experts=tensor,
            sorted_weights=tensor,
            lane_to_row=tensor,
            wmma_expert_start=tensor,
            tile_expert=tensor,
            wmma_total=tensor,
            packed_hidden=tensor,
            packed_gate_up_input=tensor,
            gate_up=tensor,
            down_input=tensor,
            down_out=tensor,
            selected_out=tensor,
            shared_gate_input=tensor,
            shared_up_input=tensor,
            shared_gate_out=tensor,
            shared_up_out=tensor,
            shared_up=tensor,
            shared_intermediate=tensor,
            shared_down_input=tensor,
            shared_out=tensor,
            moe_out=tensor,
        )
        self.grouped_reservations.append(scratch)
        return scratch

    def run_linear_attention_moe_c1_layer_fp16(self, hidden, **kwargs):
        self.run_calls.append((hidden, kwargs))
        tokens = kwargs["tokens"]
        return Tensor.from_handle(0x30000 + tokens * 0x100, (tokens, 8), DType.FP16, self.device)


def test_qwen35_grouped_moe_lane_helpers_map_sorted_lanes_to_token_rows() -> None:
    sorted_lanes = (1, 3, 5, 0, 2, 4)

    assert qwen35_grouped_moe_lane_rows(tokens=3, top_k=2) == (0, 0, 1, 1, 2, 2)
    assert qwen35_grouped_moe_sorted_token_rows(sorted_lanes, tokens=3, top_k=2) == (0, 1, 2, 0, 1, 2)
    lane_to_sorted_row = qwen35_grouped_moe_lane_to_sorted_row(sorted_lanes, tokens=3, top_k=2)

    assert lane_to_sorted_row == (3, 0, 4, 1, 5, 2)
    for lane, sorted_row in enumerate(lane_to_sorted_row):
        token_row = lane // 2
        assert sorted_lanes[sorted_row] == lane
        assert qwen35_grouped_moe_sorted_token_rows(sorted_lanes, tokens=3, top_k=2)[sorted_row] == token_row

    for bad_sorted_lanes in ((1, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 6), (0, 1, 2, 3, 4, True)):
        with pytest.raises(ValueError, match="sorted_lanes entries must be unique lane ints in range"):
            qwen35_grouped_moe_lane_to_sorted_row(bad_sorted_lanes, tokens=3, top_k=2)


def test_qwen35_grouped_moe_selected_experts_build_expert_groups() -> None:
    selected_experts = ((2, 0), (1, 2), (0, 1))

    assert qwen35_grouped_moe_expert_lane_groups(selected_experts, num_experts=3) == ((1, 4), (2, 5), (0, 3))
    assert qwen35_grouped_moe_expert_starts(selected_experts, num_experts=3) == (0, 2, 4, 6)
    sorted_lanes = qwen35_grouped_moe_sorted_lanes_from_selected_experts(selected_experts, num_experts=3)

    assert sorted_lanes == (1, 4, 2, 5, 0, 3)
    assert qwen35_grouped_moe_sorted_token_rows(sorted_lanes, tokens=3, top_k=2) == (0, 2, 1, 2, 0, 1)
    assert qwen35_grouped_moe_lane_to_sorted_row(sorted_lanes, tokens=3, top_k=2) == (4, 0, 2, 5, 1, 3)

    with pytest.raises(ValueError, match="selected_experts rows must have a consistent top_k"):
        qwen35_grouped_moe_expert_lane_groups(((0, 1), (2,)), num_experts=3)
    with pytest.raises(ValueError, match="selected_experts entries must be expert ints in range"):
        qwen35_grouped_moe_expert_lane_groups(((0, True),), num_experts=3)
    with pytest.raises(ValueError, match="selected_experts entries must be expert ints in range"):
        qwen35_grouped_moe_expert_lane_groups(((0, 3),), num_experts=3)


def test_qwen35_grouped_moe_weighted_sums_match_token_major_selected_branch() -> None:
    selected_experts = ((2, 0), (1, 2), (0, 1))
    routing_weights = ((0.25, 0.75), (0.5, 0.125), (0.875, 0.375))
    token_major_values = ((10.0, 1.0), (20.0, 2.0), (30.0, 3.0), (40.0, 4.0), (50.0, 5.0), (60.0, 6.0))
    sorted_lanes = qwen35_grouped_moe_sorted_lanes_from_selected_experts(selected_experts, num_experts=3)
    sorted_values = tuple(token_major_values[lane] for lane in sorted_lanes)
    sorted_weights = qwen35_grouped_moe_sorted_routing_weights(routing_weights, sorted_lanes, tokens=3, top_k=2)

    grouped = qwen35_grouped_moe_weighted_token_sums(sorted_values, sorted_weights, sorted_lanes, tokens=3, top_k=2)
    selected_c1 = tuple(
        tuple(
            sum(token_major_values[token * 2 + expert_rank][col] * routing_weights[token][expert_rank] for expert_rank in range(2))
            for col in range(2)
        )
        for token in range(3)
    )

    assert sorted_weights == (0.75, 0.875, 0.5, 0.375, 0.25, 0.125)
    for grouped_row, selected_row in zip(grouped, selected_c1, strict=True):
        assert grouped_row == pytest.approx(selected_row)

    with pytest.raises(ValueError, match=r"routing_weights shape must match tokens \* top_k"):
        qwen35_grouped_moe_sorted_routing_weights(((1.0,),), sorted_lanes, tokens=3, top_k=2)
    with pytest.raises(ValueError, match="sorted_values rows must have a consistent non-empty feature size"):
        qwen35_grouped_moe_weighted_token_sums(((1.0,), (2.0, 3.0)), (1.0, 1.0), (0, 1), tokens=1, top_k=2)
    with pytest.raises(ValueError, match="sorted_weights entries must be numeric"):
        qwen35_grouped_moe_weighted_token_sums(((1.0,), (2.0,)), (1.0, True), (0, 1), tokens=1, top_k=2)


def test_qwen35_resident_decode_batch_uses_grouped_moe_scratch_for_rows_gt1() -> None:
    device = Device("hip", 0)
    state = _FakePrefillState(device)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.config = SimpleNamespace(num_experts=128)
    session.states = [state]
    session.moe_scratch = {0: SimpleNamespace(residual=Tensor.from_handle(0x5000, (1, 8), DType.FP16, device))}

    scratch = session._ensure_moe_decode_batch_scratch(0, rows=2)

    assert isinstance(scratch, Qwen35ParoGroupedMoeScratch)
    assert scratch is state.grouped_reservations[0]
    assert session.moe_scratch[0] is scratch


def test_qwen35_resident_linear_prefill_restores_decode_scratch_token1() -> None:
    device = Device("hip", 0)
    runtime = _FakePrefillRuntime()
    state = _FakePrefillState(device)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.runtime = runtime
    session.config = SimpleNamespace(hidden_size=8, layer_types=("linear_attention",))
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.prefill_hidden = Tensor.from_handle(0x1000, (4, 8), DType.FP16, device)
    session.prefill_next_hidden = Tensor.from_handle(0x2000, (4, 8), DType.FP16, device)
    session.states = [state]
    session.libraries = {}
    conv = Tensor.from_handle(0x3000, (8, 4), DType.FP32, device)
    recurrent = Tensor.from_handle(0x4000, (2, 4, 4), DType.FP32, device)
    session.linear_states = {0: (conv, recurrent, DeviceBuffer(0x3000, 1), DeviceBuffer(0x4000, 1), None, None)}
    decode_linear = SimpleNamespace(attn_input=Tensor.from_handle(0x5000, (1, 8), DType.FP16, device))
    decode_moe = SimpleNamespace(normed=Tensor.from_handle(0x6000, (1, 8), DType.FP16, device))
    session.linear_scratch = {0: decode_linear}
    session.moe_scratch = {0: decode_moe}

    out = session._run_linear_prefill_layers(tokens=4)

    assert out.shape == (4, 8)
    assert session.linear_scratch[0] is decode_linear
    assert session.moe_scratch[0] is decode_moe
    assert session.prefill_linear_scratch is state.linear_reservations[0]
    assert session.prefill_moe_scratch is state.grouped_reservations[0]
    call_kwargs = state.run_calls[0][1]
    assert call_kwargs["linear_scratch"] is session.prefill_linear_scratch
    assert call_kwargs["moe_scratch"] is session.prefill_moe_scratch
    assert call_kwargs["tokens"] == 4
    assert runtime.memcpy_async_calls

    session._restore_decode_scratch_after_prefill()

    assert session.prefill_linear_scratch is None
    assert session.prefill_moe_scratch is None
    assert session.linear_scratch[0] is state.linear_reservations[1]
    assert session.moe_scratch[0] is state.moe_reservations[0]
    assert session.linear_scratch[0].attn_input.shape == (1, 8)
    assert session.moe_scratch[0].normed.shape == (1, 8)


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


def test_qwen35_resident_linear_batch_decode_uses_state_indices_for_c2_slots() -> None:
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.max_batch_size = 3
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.config = SimpleNamespace(hidden_size=8, layer_types=("linear_attention",))
    session.batch_hidden = Tensor.from_handle(0x1000, (3, 8), DType.FP16, device)
    session.batch_next_hidden = Tensor.from_handle(0x2000, (3, 8), DType.FP16, device)
    conv = Tensor.from_handle(0x3000, (8, 4), DType.FP32, device)
    recurrent = Tensor.from_handle(0x4000, (2, 4, 4), DType.FP32, device)
    session.linear_states = {
        0: (
            conv,
            recurrent,
            DeviceBuffer(0x3000, 3 * conv.numel * conv.dtype.itemsize),
            DeviceBuffer(0x4000, 3 * recurrent.numel * recurrent.dtype.itemsize),
            None,
            None,
        )
    }
    metadata_slots: list[tuple[int, ...]] = []
    cu_seqlens = Tensor.from_handle(0xA000, (3,), DType.INT32, device)
    state_indices = Tensor.from_handle(0xB000, (2,), DType.INT64, device)

    def fake_metadata(*, rows: int, slots: tuple[int, ...]):
        metadata_slots.append(tuple(slots))
        return cu_seqlens, state_indices, ()

    class FakeRuntime:
        def __init__(self) -> None:
            self.copies: list[tuple[int, int, int, int]] = []

        def memcpy_async(self, dst, src, nbytes, kind, stream):
            self.copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.calls = []

        def run_linear_attention_moe_decode_batch_layer_fp16(self, hidden, **kwargs):
            self.calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9000, (kwargs["tokens"], 8), DType.FP16, device)

    runtime = FakeRuntime()
    state = FakeState()
    session.runtime = runtime
    session.states = [state]
    session.libraries = {}
    session._batch_decode_segment_metadata = fake_metadata
    session._ensure_linear_decode_batch_scratch = lambda layer_id, rows: SimpleNamespace(attn_input=Tensor.from_handle(0x5000, (rows, 8), DType.FP16, device))
    session._ensure_moe_decode_batch_scratch = lambda layer_id, rows: SimpleNamespace(residual=Tensor.from_handle(0x6000, (rows, 8), DType.FP16, device))

    out = session._run_layers_batch_decode(rows=2, positions=(4, 7), slots=(0, 2), stream=5)

    slot0_conv, slot0_recurrent = session._slot_linear_state(0, 0)
    slot2_conv, slot2_recurrent = session._slot_linear_state(0, 2)
    assert metadata_slots == [(0, 2)]
    assert (slot0_conv.ptr, slot2_conv.ptr) == (0x3000, 0x3000 + 2 * conv.numel * DType.FP32.itemsize)
    assert (slot0_recurrent.ptr, slot2_recurrent.ptr) == (
        0x4000,
        0x4000 + 2 * recurrent.numel * DType.FP32.itemsize,
    )
    assert out.ptr == 0x2000
    assert len(state.calls) == 1
    hidden, kwargs = state.calls[0]
    assert hidden.ptr == 0x1000
    assert kwargs["conv_state"] is conv
    assert kwargs["recurrent_state"] is recurrent
    assert kwargs["cu_seqlens"] is cu_seqlens
    assert kwargs["state_indices"] is state_indices
    assert kwargs["segments"] == 2
    assert kwargs["tokens"] == 2
    assert runtime.copies == [(0x2000, 0x9000, 2 * session.hidden_nbytes, 5)]


def test_qwen35_resident_linear_batch_decode_selected_c1_projection_state_is_non_native(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_PROJECTIONS", "1")
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_STATE", "1")
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.max_batch_size = 3
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.config = SimpleNamespace(hidden_size=8, layer_types=("linear_attention",), num_experts=4)
    session.batch_hidden = Tensor.from_handle(0x1000, (3, 8), DType.FP16, device)
    session.batch_next_hidden = Tensor.from_handle(0x2000, (3, 8), DType.FP16, device)
    conv = Tensor.from_handle(0x3000, (8, 4), DType.FP32, device)
    recurrent = Tensor.from_handle(0x4000, (2, 4, 4), DType.FP32, device)
    session.linear_states = {
        0: (
            conv,
            recurrent,
            DeviceBuffer(0x3000, 3 * conv.numel * conv.dtype.itemsize),
            DeviceBuffer(0x4000, 3 * recurrent.numel * recurrent.dtype.itemsize),
            None,
            None,
        )
    }
    cu_seqlens = Tensor.from_handle(0xA000, (3,), DType.INT32, device)
    state_indices = Tensor.from_handle(0xB000, (2,), DType.INT64, device)
    session._batch_decode_segment_metadata = lambda *, rows, slots: (cu_seqlens, state_indices, ())
    session._ensure_linear_decode_batch_scratch = lambda layer_id, rows: SimpleNamespace(
        attn_input=Tensor.from_handle(0x5000, (rows, 8), DType.FP16, device)
    )

    moe_force_flags: list[bool] = []

    def fake_moe_scratch(layer_id, rows, *, force_selected_c1_moe=False):
        moe_force_flags.append(bool(force_selected_c1_moe))
        return SimpleNamespace(residual=Tensor.from_handle(0x6000, (rows, 8), DType.FP16, device))

    session._ensure_moe_decode_batch_scratch = fake_moe_scratch

    class FakeRuntime:
        def __init__(self) -> None:
            self.copies: list[tuple[int, int, int, int]] = []

        def memcpy_async(self, dst, src, nbytes, kind, stream):
            self.copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.calls = []

        def run_linear_attention_moe_decode_batch_layer_fp16(self, hidden, **kwargs):
            self.calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9000, (kwargs["tokens"], 8), DType.FP16, device)

    runtime = FakeRuntime()
    state = FakeState()
    session.runtime = runtime
    session.states = [state]
    session.libraries = {}

    out = session._run_layers_batch_decode(rows=2, positions=(4, 7), slots=(0, 2), stream=5)

    slot0_conv, slot0_recurrent = session._slot_linear_state(0, 0)
    slot2_conv, slot2_recurrent = session._slot_linear_state(0, 2)
    assert out.ptr == 0x2000
    assert len(state.calls) == 1
    hidden, kwargs = state.calls[0]
    assert hidden.ptr == 0x1000
    assert kwargs["force_selected_c1_moe"] is False
    assert kwargs["force_selected_c1_linear_projections"] is True
    assert kwargs["force_selected_c1_linear_state"] is True
    assert kwargs["force_selected_c1_linear_out"] is None
    assert kwargs["force_batch_gemv_linear_out"] is False
    assert kwargs["selected_c1_linear_state_pairs"] == (
        (slot0_conv, slot0_recurrent),
        (slot2_conv, slot2_recurrent),
    )
    assert kwargs["conv_state"] is conv
    assert kwargs["recurrent_state"] is recurrent
    assert kwargs["cu_seqlens"] is cu_seqlens
    assert kwargs["state_indices"] is state_indices
    assert kwargs["segments"] == 2
    assert kwargs["tokens"] == 2
    assert moe_force_flags == [False]
    assert runtime.copies == [(0x2000, 0x9000, 2 * session.hidden_nbytes, 5)]

    execution = session.last_batch_decode_execution
    assert execution["native_caware_decode"] is False
    assert execution["moe_decode_path"] == "grouped_compact"
    assert execution["moe_grouped_compact_layers"] == 1
    assert execution["moe_selected_c1_fallback_layers"] == 0
    assert execution["blockers"] == [
        "linear-attention projections forced to selected-c1 diagnostic path",
        "linear-attention state forced to selected-c1 diagnostic path",
        "linear-attention output projection forced to selected-c1 diagnostic path",
    ]
    layer = execution["layer_executions"][0]
    assert layer["native_caware_decode"] is False
    assert layer["linear_attention_projection_path"] == "selected_c1_forced"
    assert layer["linear_attention_state_path"] == "selected_c1_forced"
    assert layer["linear_attention_output_path"] == "selected_c1_forced"
    assert layer["moe_decode_path"] == "grouped_compact"


@pytest.mark.parametrize(
    ("env_value", "expected_selected_c1", "expected_batch_gemv", "expected_path", "expected_blocker"),
    [
        (
            "selected_c1",
            True,
            False,
            "selected_c1_forced",
            "linear-attention output projection forced to selected-c1 diagnostic path",
        ),
        (
            "batch_gemv",
            False,
            True,
            "batch_gemv",
            "linear-attention output projection forced to batch GEMV diagnostic path",
        ),
    ],
)
def test_qwen35_resident_linear_batch_decode_output_diagnostics_are_non_native(
    monkeypatch,
    env_value: str,
    expected_selected_c1: bool,
    expected_batch_gemv: bool,
    expected_path: str,
    expected_blocker: str,
) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_OUT", env_value)
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.max_batch_size = 3
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.config = SimpleNamespace(hidden_size=8, layer_types=("linear_attention",), num_experts=4)
    session.batch_hidden = Tensor.from_handle(0x1000, (3, 8), DType.FP16, device)
    session.batch_next_hidden = Tensor.from_handle(0x2000, (3, 8), DType.FP16, device)
    conv = Tensor.from_handle(0x3000, (8, 4), DType.FP32, device)
    recurrent = Tensor.from_handle(0x4000, (2, 4, 4), DType.FP32, device)
    session.linear_states = {
        0: (
            conv,
            recurrent,
            DeviceBuffer(0x3000, 3 * conv.numel * conv.dtype.itemsize),
            DeviceBuffer(0x4000, 3 * recurrent.numel * recurrent.dtype.itemsize),
            None,
            None,
        )
    }
    cu_seqlens = Tensor.from_handle(0xA000, (3,), DType.INT32, device)
    state_indices = Tensor.from_handle(0xB000, (2,), DType.INT64, device)
    session._batch_decode_segment_metadata = lambda *, rows, slots: (cu_seqlens, state_indices, ())
    session._ensure_linear_decode_batch_scratch = lambda layer_id, rows: SimpleNamespace(
        attn_input=Tensor.from_handle(0x5000, (rows, 8), DType.FP16, device)
    )

    moe_force_flags: list[bool] = []

    def fake_moe_scratch(layer_id, rows, *, force_selected_c1_moe=False):
        moe_force_flags.append(bool(force_selected_c1_moe))
        return SimpleNamespace(residual=Tensor.from_handle(0x6000, (rows, 8), DType.FP16, device))

    session._ensure_moe_decode_batch_scratch = fake_moe_scratch

    class FakeRuntime:
        def __init__(self) -> None:
            self.copies: list[tuple[int, int, int, int]] = []

        def memcpy_async(self, dst, src, nbytes, kind, stream):
            self.copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.calls = []

        def run_linear_attention_moe_decode_batch_layer_fp16(self, hidden, **kwargs):
            self.calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9000, (kwargs["tokens"], 8), DType.FP16, device)

    runtime = FakeRuntime()
    state = FakeState()
    session.runtime = runtime
    session.states = [state]
    session.libraries = {}

    out = session._run_layers_batch_decode(rows=2, positions=(4, 7), slots=(0, 2), stream=5)

    assert out.ptr == 0x2000
    assert len(state.calls) == 1
    hidden, kwargs = state.calls[0]
    assert hidden.ptr == 0x1000
    assert kwargs["force_selected_c1_linear_out"] is expected_selected_c1
    assert kwargs["force_batch_gemv_linear_out"] is expected_batch_gemv
    assert kwargs["force_selected_c1_moe"] is False
    assert kwargs["force_selected_c1_linear_projections"] is False
    assert kwargs["force_selected_c1_linear_state"] is False
    assert kwargs["selected_c1_linear_state_pairs"] is None
    assert kwargs["conv_state"] is conv
    assert kwargs["recurrent_state"] is recurrent
    assert kwargs["cu_seqlens"] is cu_seqlens
    assert kwargs["state_indices"] is state_indices
    assert kwargs["segments"] == 2
    assert kwargs["tokens"] == 2
    assert moe_force_flags == [False]
    assert runtime.copies == [(0x2000, 0x9000, 2 * session.hidden_nbytes, 5)]

    execution = session.last_batch_decode_execution
    assert execution["native_caware_decode"] is False
    assert execution["moe_decode_path"] == "grouped_compact"
    assert execution["moe_grouped_compact_layers"] == 1
    assert execution["moe_selected_c1_fallback_layers"] == 0
    assert execution["blockers"] == [expected_blocker]
    layer = execution["layer_executions"][0]
    assert layer["native_caware_decode"] is False
    assert layer["linear_attention_projection_path"] == "native_batch"
    assert layer["linear_attention_state_path"] == "native_segments"
    assert layer["linear_attention_output_path"] == expected_path
    assert layer["moe_decode_path"] == "grouped_compact"


def test_qwen35_resident_linear_batch_decode_can_force_per_row_probe(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR", "1")
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.max_batch_size = 3
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.config = SimpleNamespace(hidden_size=8, layer_types=("linear_attention",), num_experts=4)
    session.batch_hidden = Tensor.from_handle(0x1000, (3, 8), DType.FP16, device)
    session.batch_next_hidden = Tensor.from_handle(0x2000, (3, 8), DType.FP16, device)
    conv = Tensor.from_handle(0x3000, (8, 4), DType.FP32, device)
    recurrent = Tensor.from_handle(0x4000, (2, 4, 4), DType.FP32, device)
    session.linear_states = {
        0: (
            conv,
            recurrent,
            DeviceBuffer(0x3000, 3 * conv.numel * conv.dtype.itemsize),
            DeviceBuffer(0x4000, 3 * recurrent.numel * recurrent.dtype.itemsize),
            None,
            None,
        )
    }
    session._batch_decode_segment_metadata = lambda *, rows, slots: (
        Tensor.from_handle(0xA000, (rows + 1,), DType.INT32, device),
        Tensor.from_handle(0xB000, (rows,), DType.INT64, device),
        (),
    )
    linear_rows: list[int] = []
    moe_rows: list[tuple[int, bool]] = []
    session._ensure_linear_decode_batch_scratch = lambda layer_id, rows: linear_rows.append(rows) or SimpleNamespace(attn_input=Tensor.from_handle(0x5000, (rows, 8), DType.FP16, device))

    def fake_moe_scratch(layer_id, rows, *, force_selected_c1_moe=False):
        moe_rows.append((rows, bool(force_selected_c1_moe)))
        return SimpleNamespace(residual=Tensor.from_handle(0x6000, (rows, 8), DType.FP16, device))

    session._ensure_moe_decode_batch_scratch = fake_moe_scratch

    class FakeRuntime:
        def __init__(self) -> None:
            self.copies: list[tuple[int, int, int, int]] = []

        def memcpy_async(self, dst, src, nbytes, kind, stream):
            self.copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.calls = []

        def run_linear_attention_moe_c1_layer_fp16(self, hidden, **kwargs):
            self.calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9000 + len(self.calls) * 0x100, (1, 8), DType.FP16, device)

    runtime = FakeRuntime()
    state = FakeState()
    session.runtime = runtime
    session.states = [state]
    session.libraries = {}

    out = session._run_layers_batch_decode(rows=2, positions=(4, 7), slots=(0, 2), stream=5)

    assert out.ptr == 0x2000
    assert linear_rows == [1, 1]
    assert moe_rows == [(1, True), (1, True)]
    assert [call[0].ptr for call in state.calls] == [0x1000, 0x1000 + session.hidden_nbytes]
    assert [call[1]["tokens"] for call in state.calls] == [1, 1]
    assert [call[1]["conv_state"].ptr for call in state.calls] == [0x3000, 0x3000 + 2 * conv.numel * DType.FP32.itemsize]
    assert runtime.copies == [
        (0x2000, 0x9100, session.hidden_nbytes, 5),
        (0x2000 + session.hidden_nbytes, 0x9200, session.hidden_nbytes, 5),
    ]
    assert session.last_batch_decode_execution == {
        "rows": 2,
        "slots": [0, 2],
        "max_full_attention_context": 0,
        "native_full_attention_layers": 0,
        "full_attention_decode_path": "none",
        "native_caware_decode": False,
        "linear_attention_segment_metadata": {"cu_seqlens": [0, 1, 2], "state_indices": [0, 2]},
        "moe_decode_path": "mixed_grouped_compact_with_per_row_linear_attention_fallback",
        "moe_decode_rows": 2,
        "moe_grouped_compact_layers": 0,
        "moe_selected_c1_fallback_layers": 1,
        "layer_executions": [
            {
                "layer_index": 0,
                "layer_type": "linear_attention",
                "rows": 2,
                "slots": [0, 2],
                "linear_attention_decode_path": "selected_c1_per_row_fallback",
                "linear_attention_segment_metadata": {"cu_seqlens": [0, 1, 2], "state_indices": [0, 2]},
                "linear_attention_row_state_map": [
                    {"row": 0, "slot": 0, "state_index": 0},
                    {"row": 1, "slot": 2, "state_index": 2},
                ],
                "full_attention_decode_path": "not_applicable",
                "native_caware_decode": False,
                "moe_decode_path": "selected_c1_per_row_linear_fallback",
            }
        ],
        "blockers": ["linear-attention decode forced to per-row diagnostic path"],
    }


def test_qwen35_resident_linear_batch_decode_uses_single_row_c1_path() -> None:
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.max_batch_size = 1
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.config = SimpleNamespace(hidden_size=8, layer_types=("linear_attention",), num_experts=4)
    session.batch_hidden = Tensor.from_handle(0x1000, (1, 8), DType.FP16, device)
    session.batch_next_hidden = Tensor.from_handle(0x2000, (1, 8), DType.FP16, device)
    conv = Tensor.from_handle(0x3000, (8, 4), DType.FP32, device)
    recurrent = Tensor.from_handle(0x4000, (2, 4, 4), DType.FP32, device)
    session.linear_states = {
        0: (
            conv,
            recurrent,
            DeviceBuffer(0x3000, conv.numel * conv.dtype.itemsize),
            DeviceBuffer(0x4000, recurrent.numel * recurrent.dtype.itemsize),
            None,
            None,
        )
    }
    session._batch_decode_segment_metadata = lambda *, rows, slots: (
        Tensor.from_handle(0xA000, (rows + 1,), DType.INT32, device),
        Tensor.from_handle(0xB000, (rows,), DType.INT64, device),
        (),
    )
    session._ensure_linear_decode_batch_scratch = lambda layer_id, rows: SimpleNamespace(
        attn_input=Tensor.from_handle(0x5000, (rows, 8), DType.FP16, device)
    )
    force_flags: list[bool] = []

    def fake_moe_scratch(layer_id, rows, *, force_selected_c1_moe=False):
        force_flags.append(bool(force_selected_c1_moe))
        return SimpleNamespace(residual=Tensor.from_handle(0x6000, (rows, 8), DType.FP16, device))

    session._ensure_moe_decode_batch_scratch = fake_moe_scratch

    class FakeRuntime:
        def __init__(self) -> None:
            self.copies: list[tuple[int, int, int, int]] = []

        def memcpy_async(self, dst, src, nbytes, kind, stream):
            self.copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.c1_calls = []
            self.batch_calls = []

        def run_linear_attention_moe_c1_layer_fp16(self, hidden, **kwargs):
            self.c1_calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9100, (1, 8), DType.FP16, device)

        def run_linear_attention_moe_decode_batch_layer_fp16(self, hidden, **kwargs):
            self.batch_calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9200, (1, 8), DType.FP16, device)

    runtime = FakeRuntime()
    state = FakeState()
    session.runtime = runtime
    session.states = [state]
    session.libraries = {}

    out = session._run_layers_batch_decode(rows=1, positions=(4,), slots=(0,), stream=5)

    assert out.ptr == 0x2000
    assert state.batch_calls == []
    assert len(state.c1_calls) == 1
    assert force_flags == [False]
    assert state.c1_calls[0][0].ptr == 0x1000
    assert runtime.copies == [(0x2000, 0x9100, session.hidden_nbytes, 5)]
    assert session.last_batch_decode_execution["linear_attention_segment_metadata"] == {
        "cu_seqlens": [0, 1],
        "state_indices": [0],
    }
    assert session.last_batch_decode_execution["layer_executions"][0]["linear_attention_decode_path"] == "single_row_c1"
    assert session.last_batch_decode_execution["moe_selected_c1_fallback_layers"] == 0
    assert session.last_batch_decode_execution["blockers"] == []


def test_qwen35_resident_linear_batch_decode_can_force_rows1_per_row_probe(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR", "1")
    device = Device("hip", 0)
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.device = device
    session.max_batch_size = 1
    session.hidden_nbytes = 8 * DType.FP16.itemsize
    session.config = SimpleNamespace(hidden_size=8, layer_types=("linear_attention",), num_experts=4)
    session.batch_hidden = Tensor.from_handle(0x1000, (1, 8), DType.FP16, device)
    session.batch_next_hidden = Tensor.from_handle(0x2000, (1, 8), DType.FP16, device)
    conv = Tensor.from_handle(0x3000, (8, 4), DType.FP32, device)
    recurrent = Tensor.from_handle(0x4000, (2, 4, 4), DType.FP32, device)
    session.linear_states = {
        0: (
            conv,
            recurrent,
            DeviceBuffer(0x3000, conv.numel * conv.dtype.itemsize),
            DeviceBuffer(0x4000, recurrent.numel * recurrent.dtype.itemsize),
            None,
            None,
        )
    }
    session._batch_decode_segment_metadata = lambda *, rows, slots: (
        Tensor.from_handle(0xA000, (rows + 1,), DType.INT32, device),
        Tensor.from_handle(0xB000, (rows,), DType.INT64, device),
        (),
    )
    linear_rows: list[int] = []
    moe_rows: list[tuple[int, bool]] = []
    session._ensure_linear_decode_batch_scratch = lambda layer_id, rows: linear_rows.append(rows) or SimpleNamespace(attn_input=Tensor.from_handle(0x5000, (rows, 8), DType.FP16, device))

    def fake_moe_scratch(layer_id, rows, *, force_selected_c1_moe=False):
        moe_rows.append((rows, bool(force_selected_c1_moe)))
        return SimpleNamespace(residual=Tensor.from_handle(0x6000, (rows, 8), DType.FP16, device))

    session._ensure_moe_decode_batch_scratch = fake_moe_scratch

    class FakeRuntime:
        def __init__(self) -> None:
            self.copies: list[tuple[int, int, int, int]] = []

        def memcpy_async(self, dst, src, nbytes, kind, stream):
            self.copies.append((int(dst), int(src), int(nbytes), int(stream)))

    class FakeState:
        def __init__(self) -> None:
            self.c1_calls = []
            self.batch_calls = []

        def run_linear_attention_moe_c1_layer_fp16(self, hidden, **kwargs):
            self.c1_calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9100, (1, 8), DType.FP16, device)

        def run_linear_attention_moe_decode_batch_layer_fp16(self, hidden, **kwargs):
            self.batch_calls.append((hidden, kwargs))
            return Tensor.from_handle(0x9200, (1, 8), DType.FP16, device)

    runtime = FakeRuntime()
    state = FakeState()
    session.runtime = runtime
    session.states = [state]
    session.libraries = {}

    out = session._run_layers_batch_decode(rows=1, positions=(4,), slots=(0,), stream=5)

    assert out.ptr == 0x2000
    assert state.batch_calls == []
    assert len(state.c1_calls) == 1
    assert linear_rows == [1]
    assert moe_rows == [(1, True)]
    assert state.c1_calls[0][0].ptr == 0x1000
    assert state.c1_calls[0][1]["conv_state"].ptr == 0x3000
    assert runtime.copies == [(0x2000, 0x9100, session.hidden_nbytes, 5)]
    assert session.last_batch_decode_execution["linear_attention_segment_metadata"] == {
        "cu_seqlens": [0, 1],
        "state_indices": [0],
    }
    assert session.last_batch_decode_execution["layer_executions"][0]["linear_attention_decode_path"] == (
        "selected_c1_per_row_fallback"
    )
    assert session.last_batch_decode_execution["blockers"] == [
        "linear-attention decode forced to per-row diagnostic path"
    ]


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
