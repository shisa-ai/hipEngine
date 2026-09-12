from __future__ import annotations

from dataclasses import replace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.speculative import (
    DFlashDraftRequest,
    TargetVerifyBatch,
    TargetVerifyBufferOwner,
    TargetVerifyBufferSpec,
    TargetVerifyScratchSpec,
    compile_dflash_chain,
)


class FakeWorkspace:
    def __init__(self, device: Device) -> None:
        self.device = device
        self._next = 0x100000
        self.allocations: dict[str, Tensor] = {}

    def reserve_tensor(self, name: str, shape, dtype) -> Tensor:
        shape_tuple = tuple(int(dim) for dim in shape)
        parsed = DType.parse(dtype)
        current = self.allocations.get(name)
        if current is not None and current.shape == shape_tuple and current.dtype == parsed and current.device == self.device:
            return current
        nbytes = max(1, _numel(shape_tuple) * parsed.itemsize)
        ptr = self._next
        self._next += nbytes + 0x100
        tensor = Tensor.from_handle(ptr, shape_tuple, parsed, self.device)
        self.allocations[name] = tensor
        return tensor


def _numel(shape: tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def _spec(*, max_rows: int = 10, max_requests: int = 2, mode: str = "verify_chain") -> TargetVerifyBufferSpec:
    return TargetVerifyBufferSpec(
        backend="hip_gfx1151",
        bucket="chain4_b2",
        device=Device("hip", 0),
        max_rows=max_rows,
        max_requests=max_requests,
        mode=mode,
        hidden_tap_count=5,
        hidden_size=2048,
        hidden_tap_dtype=DType.BF16,
        scratch_specs=(
            TargetVerifyScratchSpec("layer0.conv_state", (max_rows, 16, 128), DType.FP32),
            TargetVerifyScratchSpec("layer0.recurrent_state", (max_rows, 2, 128), DType.FP32),
        ),
    )


def _target_batch() -> TargetVerifyBatch:
    draft = compile_dflash_chain(
        [
            DFlashDraftRequest(request_id=10, root_position=5, candidate_tokens=(101, 102, 103, 104)),
            DFlashDraftRequest(request_id=20, root_position=12, candidate_tokens=(201, 202, 203), active_count=3),
        ],
        candidate_budget=4,
    )
    return TargetVerifyBatch.from_draft(draft, root_tokens=(1000, 2000), root_positions=(5, 12))


def _small_target_batch() -> TargetVerifyBatch:
    draft = compile_dflash_chain(
        [DFlashDraftRequest(request_id=99, root_position=4, candidate_tokens=(301, 302))],
        candidate_budget=2,
    )
    return TargetVerifyBatch.from_draft(draft, root_tokens=(300,), root_positions=(4,))


def test_target_verify_buffer_owner_allocates_stable_bucket_buffers() -> None:
    workspace = FakeWorkspace(Device("hip", 0))
    spec = _spec()

    owner = TargetVerifyBufferOwner.allocate(spec, workspace=workspace)
    signature = owner.address_signature()

    assert owner.token_ids.shape == (10,)
    assert owner.positions.dtype == DType.INT32
    assert owner.active_mask.dtype == DType.BOOL
    assert owner.full_accept.dtype == DType.BOOL
    assert owner.committed_output_ids.shape == (2, 10)
    assert owner.committed_output_lengths.shape == (2,)
    assert owner.hidden_taps is not None
    assert owner.hidden_taps.shape == (5, 10, 2048)
    assert owner.scratch_tensors["layer0.conv_state"].shape == (10, 16, 128)

    rebound = TargetVerifyBufferOwner.allocate(spec, workspace=workspace)
    assert rebound.address_signature() == signature
    assert len(workspace.allocations) == len(signature)

    metadata = owner.compact_metadata()
    assert metadata["backend"] == "hip_gfx1151"
    assert metadata["device"] == "hip:0"
    assert metadata["bucket"] == "chain4_b2"
    assert metadata["mode"] == "verify_chain"
    assert metadata["max_rows"] == 10
    assert metadata["max_requests"] == 2
    assert metadata["stable_address_count"] == len(signature)
    assert metadata["total_nbytes"] == owner.total_nbytes
    assert "address_signature" not in metadata
    assert owner.compact_metadata(include_pointers=True)["address_signature"] == signature


def test_target_verify_buffer_owner_binds_exact_batch_views_without_reallocating() -> None:
    owner = TargetVerifyBufferOwner.allocate(_spec(), workspace=FakeWorkspace(Device("hip", 0)))
    target = _target_batch()

    buffers = owner.bind(target, transaction_id=17)

    assert buffers.transaction_id == 17
    assert buffers.rows == target.rows == 10
    assert buffers.request_ids == (10, 20)
    assert buffers.request_count == 2
    assert buffers.candidate_rows == 8
    assert buffers.candidate_counts == (4, 4)
    assert buffers.draft_depth == 4
    assert buffers.tree_shape == (0, 1, 2, 3, 0, 5, 6, 7)
    assert buffers.mode == "verify_chain"
    assert buffers.token_ids.ptr == owner.token_ids.ptr
    assert buffers.token_ids.shape == (10,)
    assert buffers.accepted_counts.ptr == owner.accepted_counts.ptr
    assert buffers.accepted_counts.shape == (2,)
    assert buffers.next_tokens is not None
    assert buffers.next_tokens.ptr == owner.next_tokens.ptr
    assert buffers.full_accept is not None
    assert buffers.full_accept.ptr == owner.full_accept.ptr
    assert buffers.committed_output_ids is not None
    assert buffers.committed_output_ids.ptr == owner.committed_output_ids.ptr
    assert buffers.committed_output_ids.shape == (2, 10)
    assert buffers.committed_output_ids.strides == (10, 1)
    assert buffers.committed_output_lengths is not None
    assert buffers.committed_output_lengths.ptr == owner.committed_output_lengths.ptr

    small = _small_target_batch()
    small_buffers = owner.bind(small)
    assert small_buffers.rows == 3
    assert small_buffers.request_ids == (99,)
    assert small_buffers.token_ids.ptr == owner.token_ids.ptr
    assert small_buffers.token_ids.shape == (3,)
    assert small_buffers.accepted_counts.shape == (1,)
    assert small_buffers.committed_output_ids is not None
    assert small_buffers.committed_output_ids.shape == (1, 3)
    assert small_buffers.committed_output_ids.strides == (10, 1)


def test_target_verify_buffer_owner_rejects_wrong_capacity_mode_or_workspace() -> None:
    target = _target_batch()
    workspace = FakeWorkspace(Device("hip", 0))

    too_few_rows = TargetVerifyBufferOwner.allocate(_spec(max_rows=9), workspace=workspace)
    with pytest.raises(ValueError, match="rows exceed"):
        too_few_rows.bind(target)

    too_few_requests = TargetVerifyBufferOwner.allocate(_spec(max_requests=1), workspace=workspace)
    with pytest.raises(ValueError, match="request count"):
        too_few_requests.bind(target)

    tree_owner = TargetVerifyBufferOwner.allocate(_spec(mode="verify_tree"), workspace=workspace)
    with pytest.raises(ValueError, match="mode"):
        tree_owner.bind(target)

    with pytest.raises(ValueError, match="workspace device"):
        TargetVerifyBufferOwner.allocate(_spec(), workspace=FakeWorkspace(Device("hip", 1)))


def test_target_verify_buffer_owner_validates_base_tensor_shapes_dtypes_and_scratch() -> None:
    owner = TargetVerifyBufferOwner.allocate(_spec(), workspace=FakeWorkspace(Device("hip", 0)))

    with pytest.raises(ValueError, match="token_ids tensor shape"):
        replace(owner, token_ids=Tensor.from_handle(0xA000, (9,), DType.INT32, Device("hip", 0)))

    with pytest.raises(ValueError, match="active_mask tensor dtype"):
        replace(owner, active_mask=Tensor.from_handle(0xA100, (10,), DType.INT32, Device("hip", 0)))

    with pytest.raises(ValueError, match="full_accept tensor dtype"):
        replace(owner, full_accept=Tensor.from_handle(0xA180, (2,), DType.INT32, Device("hip", 0)))

    with pytest.raises(ValueError, match="committed_output_ids tensor shape"):
        replace(owner, committed_output_ids=Tensor.from_handle(0xA190, (2, 9), DType.INT32, Device("hip", 0)))

    with pytest.raises(ValueError, match="hidden_taps tensor is required"):
        replace(owner, hidden_taps=None)

    bad_scratch = replace(owner.scratch[0], tensor=Tensor.from_handle(0xA200, (10, 16, 127), DType.FP32, Device("hip", 0)))
    with pytest.raises(ValueError, match="scratch.layer0.conv_state"):
        replace(owner, scratch=(bad_scratch, owner.scratch[1]))
