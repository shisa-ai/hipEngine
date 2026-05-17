"""Resident speculative target-verification buffer owners.

The classes here own stable tensor handles for fixed root+candidate verifier
buckets.  They intentionally separate allocation from execution: a runtime
workspace supplies stable device pointers, while ``bind()`` returns the existing
``TargetVerifyBuffers`` ABI views sized to a concrete ``TargetVerifyBatch``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol, Sequence

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.speculative.interfaces import TargetVerifyBatch, TargetVerifyBuffers


class TensorWorkspace(Protocol):
    """Minimal workspace API used by resident verifier buffer owners."""

    device: Device

    def reserve_tensor(self, name: str, shape: Sequence[int], dtype: str | DType) -> Tensor:
        """Return a stable tensor for ``name``/``shape``/``dtype``."""
        ...


@dataclass(frozen=True, slots=True)
class TargetVerifyScratchSpec:
    """Named scratch allocation attached to a verifier graph bucket."""

    name: str
    shape: tuple[int, ...]
    dtype: DType | str = DType.BF16

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scratch spec name must be non-empty")
        shape = tuple(int(dim) for dim in self.shape)
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError("scratch spec shape must contain positive dimensions")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", DType.parse(self.dtype))


@dataclass(frozen=True, slots=True)
class TargetVerifyBufferSpec:
    """Fixed-shape target-verifier bucket allocation spec."""

    backend: str
    bucket: str
    device: Device
    max_rows: int
    max_requests: int
    mode: str = "verify_chain"
    hidden_tap_count: int = 0
    hidden_size: int = 0
    hidden_tap_dtype: DType | str = DType.BF16
    metadata_dtype: DType | str = DType.INT32
    scratch_specs: tuple[TargetVerifyScratchSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.bucket:
            raise ValueError("bucket must be non-empty")
        if self.max_rows <= 0:
            raise ValueError("max_rows must be positive")
        if self.max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if self.mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("mode must be verify_chain or verify_tree")
        if self.hidden_tap_count < 0:
            raise ValueError("hidden_tap_count must be non-negative")
        if self.hidden_tap_count and self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive when hidden_tap_count is non-zero")
        if self.hidden_size < 0:
            raise ValueError("hidden_size must be non-negative")
        object.__setattr__(self, "hidden_tap_dtype", DType.parse(self.hidden_tap_dtype))
        metadata_dtype = DType.parse(self.metadata_dtype)
        if metadata_dtype not in {DType.INT32, DType.INT64}:
            raise ValueError("metadata_dtype must be int32 or int64")
        object.__setattr__(self, "metadata_dtype", metadata_dtype)
        scratch_specs = tuple(self.scratch_specs)
        if len({spec.name for spec in scratch_specs}) != len(scratch_specs):
            raise ValueError("scratch spec names must be unique")
        object.__setattr__(self, "scratch_specs", scratch_specs)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.backend, str(self.device), self.bucket, self.mode)

    @property
    def row_shape(self) -> tuple[int, ...]:
        return (self.max_rows,)

    @property
    def summary_shape(self) -> tuple[int, ...]:
        return (self.max_requests,)

    @property
    def committed_output_shape(self) -> tuple[int, ...]:
        return (self.max_requests, self.max_rows)

    @property
    def hidden_tap_shape(self) -> tuple[int, ...] | None:
        if self.hidden_tap_count == 0:
            return None
        return (self.hidden_tap_count, self.max_rows, self.hidden_size)


@dataclass(frozen=True, slots=True)
class TargetVerifyScratchHandle:
    name: str
    tensor: Tensor

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scratch handle name must be non-empty")


@dataclass(frozen=True, slots=True)
class TargetVerifyBufferOwner:
    """Stable device-buffer owner for one target-verifier bucket."""

    spec: TargetVerifyBufferSpec
    token_ids: Tensor
    positions: Tensor
    parent_rows: Tensor
    draft_depths: Tensor
    row_to_request: Tensor
    active_mask: Tensor
    target_top1: Tensor
    accepted_counts: Tensor
    commit_rows: Tensor
    commit_tokens: Tensor
    commit_positions: Tensor
    next_tokens: Tensor
    full_accept: Tensor
    committed_output_ids: Tensor
    committed_output_lengths: Tensor
    hidden_taps: Tensor | None = None
    scratch: tuple[TargetVerifyScratchHandle, ...] = ()

    def __post_init__(self) -> None:
        row_int = {
            "token_ids": self.token_ids,
            "positions": self.positions,
            "parent_rows": self.parent_rows,
            "draft_depths": self.draft_depths,
            "row_to_request": self.row_to_request,
            "target_top1": self.target_top1,
        }
        for name, tensor in row_int.items():
            self._validate_tensor(name, tensor, self.spec.row_shape, integer=True)
        self._validate_tensor("active_mask", self.active_mask, self.spec.row_shape, dtype=DType.BOOL)
        for name, tensor in self.summary_tensors.items():
            self._validate_tensor(name, tensor, self.spec.summary_shape, integer=True)
        self._validate_tensor("full_accept", self.full_accept, self.spec.summary_shape, dtype=DType.BOOL)
        self._validate_tensor("committed_output_ids", self.committed_output_ids, self.spec.committed_output_shape, integer=True)
        self._validate_tensor("committed_output_lengths", self.committed_output_lengths, self.spec.summary_shape, integer=True)
        hidden_shape = self.spec.hidden_tap_shape
        if hidden_shape is None:
            if self.hidden_taps is not None:
                raise ValueError("hidden_taps must be omitted when hidden_tap_count is zero")
        else:
            if self.hidden_taps is None:
                raise ValueError("hidden_taps tensor is required by the buffer spec")
            self._validate_tensor("hidden_taps", self.hidden_taps, hidden_shape, dtype=self.spec.hidden_tap_dtype)

        scratch_by_name = {handle.name: handle.tensor for handle in self.scratch}
        if len(scratch_by_name) != len(self.scratch):
            raise ValueError("scratch handle names must be unique")
        expected = {spec.name: spec for spec in self.spec.scratch_specs}
        if set(scratch_by_name) != set(expected):
            raise ValueError("scratch handles must match scratch spec names")
        for name, spec in expected.items():
            self._validate_tensor(f"scratch.{name}", scratch_by_name[name], spec.shape, dtype=spec.dtype)

    @classmethod
    def allocate(
        cls,
        spec: TargetVerifyBufferSpec,
        *,
        workspace: TensorWorkspace,
        prefix: str | None = None,
    ) -> "TargetVerifyBufferOwner":
        """Reserve stable tensors for ``spec`` from ``workspace``."""

        if workspace.device != spec.device:
            raise ValueError("workspace device must match target verify buffer spec")
        base = prefix or f"target_verify/{spec.backend}/{spec.device}/{spec.bucket}/{spec.mode}"

        def reserve(name: str, shape: Sequence[int], dtype: DType | str) -> Tensor:
            return workspace.reserve_tensor(f"{base}/{name}", shape, dtype)

        hidden_taps = None
        hidden_shape = spec.hidden_tap_shape
        if hidden_shape is not None:
            hidden_taps = reserve("hidden_taps", hidden_shape, spec.hidden_tap_dtype)

        scratch = tuple(
            TargetVerifyScratchHandle(
                name=scratch_spec.name,
                tensor=reserve(f"scratch/{scratch_spec.name}", scratch_spec.shape, scratch_spec.dtype),
            )
            for scratch_spec in spec.scratch_specs
        )

        return cls(
            spec=spec,
            token_ids=reserve("token_ids", spec.row_shape, spec.metadata_dtype),
            positions=reserve("positions", spec.row_shape, spec.metadata_dtype),
            parent_rows=reserve("parent_rows", spec.row_shape, spec.metadata_dtype),
            draft_depths=reserve("draft_depths", spec.row_shape, spec.metadata_dtype),
            row_to_request=reserve("row_to_request", spec.row_shape, spec.metadata_dtype),
            active_mask=reserve("active_mask", spec.row_shape, DType.BOOL),
            target_top1=reserve("target_top1", spec.row_shape, spec.metadata_dtype),
            accepted_counts=reserve("accepted_counts", spec.summary_shape, spec.metadata_dtype),
            commit_rows=reserve("commit_rows", spec.summary_shape, spec.metadata_dtype),
            commit_tokens=reserve("commit_tokens", spec.summary_shape, spec.metadata_dtype),
            commit_positions=reserve("commit_positions", spec.summary_shape, spec.metadata_dtype),
            next_tokens=reserve("next_tokens", spec.summary_shape, spec.metadata_dtype),
            full_accept=reserve("full_accept", spec.summary_shape, DType.BOOL),
            committed_output_ids=reserve("committed_output_ids", spec.committed_output_shape, spec.metadata_dtype),
            committed_output_lengths=reserve("committed_output_lengths", spec.summary_shape, spec.metadata_dtype),
            hidden_taps=hidden_taps,
            scratch=scratch,
        )

    @property
    def row_tensors(self) -> dict[str, Tensor]:
        return {
            "token_ids": self.token_ids,
            "positions": self.positions,
            "parent_rows": self.parent_rows,
            "draft_depths": self.draft_depths,
            "row_to_request": self.row_to_request,
            "active_mask": self.active_mask,
            "target_top1": self.target_top1,
        }

    @property
    def summary_tensors(self) -> dict[str, Tensor]:
        return {
            "accepted_counts": self.accepted_counts,
            "commit_rows": self.commit_rows,
            "commit_tokens": self.commit_tokens,
            "commit_positions": self.commit_positions,
            "next_tokens": self.next_tokens,
            "committed_output_lengths": self.committed_output_lengths,
        }

    @property
    def scratch_tensors(self) -> dict[str, Tensor]:
        return {handle.name: handle.tensor for handle in self.scratch}

    @property
    def total_nbytes(self) -> int:
        return sum(self._tensor_nbytes(tensor) for _, tensor in self._named_tensors())

    def bind(self, batch: TargetVerifyBatch, *, transaction_id: int | None = None) -> TargetVerifyBuffers:
        """Return exact-size ABI tensor views for ``batch`` without reallocating."""

        if batch.mode != self.spec.mode:
            raise ValueError("target verify batch mode must match buffer owner mode")
        if batch.rows > self.spec.max_rows:
            raise ValueError("target verify batch rows exceed buffer owner capacity")
        if len(batch.request_ids) > self.spec.max_requests:
            raise ValueError("target verify request count exceeds buffer owner capacity")
        row_count = batch.rows
        request_count = len(batch.request_ids)
        return TargetVerifyBuffers.for_batch(
            batch,
            token_ids=self._view_1d(self.token_ids, row_count),
            positions=self._view_1d(self.positions, row_count),
            parent_rows=self._view_1d(self.parent_rows, row_count),
            draft_depths=self._view_1d(self.draft_depths, row_count),
            row_to_request=self._view_1d(self.row_to_request, row_count),
            active_mask=self._view_1d(self.active_mask, row_count),
            target_top1=self._view_1d(self.target_top1, row_count),
            accepted_counts=self._view_1d(self.accepted_counts, request_count),
            commit_rows=self._view_1d(self.commit_rows, request_count),
            commit_tokens=self._view_1d(self.commit_tokens, request_count),
            commit_positions=self._view_1d(self.commit_positions, request_count),
            next_tokens=self._view_1d(self.next_tokens, request_count),
            full_accept=self._view_1d(self.full_accept, request_count),
            committed_output_ids=self._view_2d_prefix(self.committed_output_ids, request_count, row_count),
            committed_output_lengths=self._view_1d(self.committed_output_lengths, request_count),
            transaction_id=transaction_id,
        )

    def address_signature(self) -> dict[str, int]:
        """Return tensor base pointers keyed by semantic buffer name."""

        return {name: tensor.ptr for name, tensor in self._named_tensors()}

    def compact_metadata(self, *, include_pointers: bool = False) -> dict[str, object]:
        """Return compact benchmark-artifact metadata for this owner."""

        signature = self.address_signature()
        payload = "\n".join(f"{name}:{ptr}" for name, ptr in sorted(signature.items())).encode()
        metadata: dict[str, object] = {
            "backend": self.spec.backend,
            "device": str(self.spec.device),
            "bucket": self.spec.bucket,
            "mode": self.spec.mode,
            "max_rows": self.spec.max_rows,
            "max_requests": self.spec.max_requests,
            "hidden_tap_count": self.spec.hidden_tap_count,
            "hidden_size": self.spec.hidden_size,
            "total_nbytes": self.total_nbytes,
            "stable_address_count": len(signature),
            "address_digest_sha256": hashlib.sha256(payload).hexdigest(),
            "row_tensors": {name: self._tensor_metadata(tensor) for name, tensor in self.row_tensors.items()},
            "summary_tensors": {name: self._tensor_metadata(tensor) for name, tensor in self.summary_tensors.items()},
            "accept_tensors": {name: self._tensor_metadata(tensor) for name, tensor in self.accept_tensors.items()},
            "scratch_tensors": {name: self._tensor_metadata(tensor) for name, tensor in self.scratch_tensors.items()},
        }
        if self.hidden_taps is not None:
            metadata["hidden_taps"] = self._tensor_metadata(self.hidden_taps)
        if include_pointers:
            metadata["address_signature"] = signature
        return metadata

    @property
    def accept_tensors(self) -> dict[str, Tensor]:
        return {
            "full_accept": self.full_accept,
            "committed_output_ids": self.committed_output_ids,
        }

    def _named_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        named: list[tuple[str, Tensor]] = [
            *self.row_tensors.items(),
            *self.summary_tensors.items(),
            *self.accept_tensors.items(),
        ]
        if self.hidden_taps is not None:
            named.append(("hidden_taps", self.hidden_taps))
        named.extend((f"scratch.{handle.name}", handle.tensor) for handle in self.scratch)
        return tuple(named)

    def _validate_tensor(
        self,
        name: str,
        tensor: Tensor,
        shape: Sequence[int],
        *,
        dtype: DType | None = None,
        integer: bool = False,
    ) -> None:
        expected_shape = tuple(int(dim) for dim in shape)
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} tensor shape must be {expected_shape}, got {tensor.shape}")
        if tensor.device != self.spec.device:
            raise ValueError(f"{name} tensor must live on {self.spec.device}")
        if dtype is not None and tensor.dtype != dtype:
            raise ValueError(f"{name} tensor dtype must be {dtype.value}")
        if integer and tensor.dtype not in {DType.INT32, DType.INT64}:
            raise ValueError(f"{name} tensor dtype must be int32 or int64")

    @staticmethod
    def _view_1d(tensor: Tensor, length: int) -> Tensor:
        if tensor.ndim != 1:
            raise ValueError("only one-dimensional tensors can be narrowed as verifier ABI views")
        if length < 0 or length > tensor.shape[0]:
            raise ValueError("view length exceeds tensor capacity")
        return Tensor.from_handle(tensor.ptr, (length,), tensor.dtype, tensor.device)

    @staticmethod
    def _view_2d_prefix(tensor: Tensor, rows: int, cols: int) -> Tensor:
        if tensor.ndim != 2:
            raise ValueError("only two-dimensional tensors can be narrowed as verifier ABI views")
        if rows < 0 or cols < 0 or rows > tensor.shape[0] or cols > tensor.shape[1]:
            raise ValueError("view shape exceeds tensor capacity")
        return Tensor.from_handle(
            tensor.ptr,
            (rows, cols),
            tensor.dtype,
            tensor.device,
            strides=(tensor.shape[1], 1),
        )

    @staticmethod
    def _tensor_nbytes(tensor: Tensor) -> int:
        return tensor.numel * tensor.dtype.itemsize

    @staticmethod
    def _tensor_metadata(tensor: Tensor) -> dict[str, object]:
        return {
            "shape": list(tensor.shape),
            "dtype": tensor.dtype.value,
            "device": str(tensor.device),
            "nbytes": TargetVerifyBufferOwner._tensor_nbytes(tensor),
        }


__all__ = [
    "TargetVerifyBufferOwner",
    "TargetVerifyBufferSpec",
    "TargetVerifyScratchHandle",
    "TargetVerifyScratchSpec",
    "TensorWorkspace",
]
