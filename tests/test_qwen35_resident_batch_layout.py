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
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro import Qwen35ParoGroupedMoeScratch
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
    assert config.moe_grouped_device_gather is True
    assert PrefillConfig(attn_aotriton_min_tokens="1024").attn_aotriton_min_tokens == 1024
    assert PrefillConfig(moe_chunk_size="1024").moe_chunk_size == 1024

    with pytest.raises(ValueError, match="full_attn_query_chunk_size"):
        PrefillConfig(full_attn_query_chunk_size=-1)
    with pytest.raises(ValueError, match="moe_chunk_size"):
        PrefillConfig(moe_chunk_size=-1)
    with pytest.raises(ValueError, match="attn_aotriton_min_tokens"):
        PrefillConfig(attn_aotriton_min_tokens=-1)


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
