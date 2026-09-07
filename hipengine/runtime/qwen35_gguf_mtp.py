"""Transactional GGUF NextN proposal and shared target-verifier integration."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Callable, Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.generation.batch_scheduler import ResidentBatchScheduler
from hipengine.generation.deadline import (
    GenerationDeadlineExceeded,
    generation_deadline_expired,
)
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import gguf_rmsnorm_bf16_f32_weight
from hipengine.kernels.hip_gfx1100.speculative.dflash_accept import build_dflash_accept
from hipengine.kernels.hip_gfx1100.speculative.dflash_commit import build_dflash_commit
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNDraftProvider
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.runtime.workspace import RuntimeWorkspace
from hipengine.speculative import (
    MtpBudgetCycleResult,
    MtpBudgetPolicy,
    MtpProposalContext,
    TargetAcceptSummary,
    TargetCommitPlan,
    TargetStateCommitBuffers,
    TargetVerifyBatch,
    TargetVerifyBufferOwner,
    TargetVerifyBufferSpec,
    TargetVerifyBuffers,
)

_GGUF_MTP_CANDIDATE_BUDGETS = (1, 2, 3, 4, 5, 6, 7)
_GGUF_MTP_TARGET_VERIFY_MODES = ("serial_exact", "native")
_GGUF_MTP_DRAFT_HIDDEN_VARIANTS = ("pre_output_norm", "post_output_norm")


@dataclass(frozen=True, slots=True)
class Qwen35GGUFDraftHiddenPolicy:
    """Immutable target-to-draft hidden convention for one decoder."""

    target_hidden_variant: str = "pre_output_norm"

    def __post_init__(self) -> None:
        variant = str(self.target_hidden_variant).strip()
        if variant not in _GGUF_MTP_DRAFT_HIDDEN_VARIANTS:
            raise ValueError(
                "target_hidden_variant must be 'pre_output_norm' or "
                "'post_output_norm'"
            )
        object.__setattr__(self, "target_hidden_variant", variant)

    @property
    def requires_target_output_norm(self) -> bool:
        return self.target_hidden_variant == "post_output_norm"

    def manifest(self) -> dict[str, str]:
        variant = self.target_hidden_variant
        return {
            "target_hidden_variant": variant,
            "prompt_target_hidden": variant,
            "proposal_target_hidden": variant,
            # Target transaction ownership remains strict and pre-output-norm.
            "target_commit_hidden": "pre_output_norm",
            # Chained draft depths keep the NextN head's own normalized output.
            "draft_chain_hidden": "nextn_post_output_norm",
        }

def _mtp_cycle_checkpoint(checkpoint: Callable[[], None] | None) -> None:
    """Observe cancellation/deadline/shutdown at one owned cycle boundary.

    RF3 lifecycle rule: this runs before any proposal/target mutation for the
    cycle, so an abort here leaves no in-flight transaction. During-cycle
    cancellation lets the owned cycle retire or roll back through its existing
    exception handler.
    """

    if checkpoint is not None:
        checkpoint()


def _mtp_lifecycle_phase(
    hook: Callable[[str], None] | None,
    phase: str,
) -> None:
    """Emit one deterministic lifecycle/fault-injection boundary."""

    if hook is not None:
        hook(str(phase))


def _mtp_should_rollback_transaction(
    *,
    target_committed: bool,
    transaction: Any,
) -> bool:
    """Return whether an exception still owns an uncommitted KV transaction."""

    return bool(
        not target_committed
        and not bool(getattr(transaction, "committed", False))
        and not bool(getattr(transaction, "rolled_back", False))
    )


def _deadline_checkpoint(deadline_at: float | None) -> Callable[[], None] | None:
    """Build a checkpoint that raises when an absolute monotonic deadline expires."""

    if deadline_at is None:
        return None
    boundary = float(deadline_at)

    def _check() -> None:
        if generation_deadline_expired(boundary):
            raise GenerationDeadlineExceeded(deadline_at=boundary)

    return _check


class _StreamingNextNPromptSink:
    """Request-owned shifted-hidden consumer for exact NextN prompt priming.

    The sink carries only the newest target trunk row across target-prefill
    chunks.  Every chunk appends ``token[start]`` against that carried row,
    appends the remaining tokens against the preceding rows in the current
    target chunk, then replaces the carried row with the chunk tail.  All
    copies and draft launches use the target stream, so target buffers may be
    reused as soon as :meth:`consume` returns without a host synchronization.
    """

    def __init__(
        self,
        *,
        request_id: int,
        prompt_tokens: Sequence[int],
        hidden_size: int,
        executor: Any,
        runtime: HipRuntime,
        checkpoint: Callable[[], None] | None,
        start_position: int = 0,
        initial_hidden: Tensor | None = None,
        transform_hidden_rows: Callable[[int, int, int], int] | None = None,
    ) -> None:
        tokens = tuple(int(token) for token in prompt_tokens)
        if not tokens:
            raise ValueError("streaming NextN prompt sink requires prompt tokens")
        if int(request_id) < 0:
            raise ValueError("streaming NextN prompt sink request_id must be non-negative")
        if int(hidden_size) <= 0:
            raise ValueError("streaming NextN prompt sink hidden_size must be positive")
        if int(start_position) < 0:
            raise ValueError("streaming NextN prompt sink start_position must be non-negative")
        if int(start_position) > 0 and initial_hidden is None:
            raise ValueError("warm streaming NextN prompt priming requires initial_hidden")
        if initial_hidden is not None and (
            initial_hidden.dtype != DType.BF16
            or initial_hidden.shape != (1, int(hidden_size))
        ):
            raise ValueError(
                "streaming NextN initial_hidden must be BF16 with shape (1, hidden_size)"
            )

        self.request_id = int(request_id)
        self.prompt_tokens = tokens
        self.total_rows = len(tokens)
        self.hidden_size = int(hidden_size)
        self.start_position = int(start_position)
        self.executor = executor
        self.runtime = runtime
        self.checkpoint = checkpoint
        self.transform_hidden_rows = transform_hidden_rows
        self.hidden_nbytes = self.hidden_size * DType.BF16.itemsize
        self._pending_hidden = malloc(self.hidden_nbytes, runtime=runtime)
        try:
            if initial_hidden is None:
                runtime.memset(self._pending_hidden.ptr, 0, self.hidden_nbytes)
            else:
                runtime.memcpy(
                    self._pending_hidden.ptr,
                    initial_hidden.ptr,
                    self.hidden_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
        except Exception:
            free(self._pending_hidden, runtime=runtime)
            raise
        self.consumed_rows = 0
        self._stream: int | None = None
        self._finished = False
        self._failed = False
        self._transferred = False
        self._closed = False

    @property
    def final_pending_hidden(self) -> Tensor:
        if self._transferred:
            raise RuntimeError("streaming NextN carried hidden row was already transferred")
        if self._closed:
            raise RuntimeError("streaming NextN prompt sink is closed")
        return Tensor.from_handle(
            self._pending_hidden.ptr,
            (1, self.hidden_size),
            DType.BF16,
            Device("hip", 0),
        )

    def _validate_owner(self, request_id: int) -> None:
        if int(request_id) != self.request_id:
            raise RuntimeError(
                "streaming NextN prompt sink owner mismatch: "
                f"expected request {self.request_id}, got {int(request_id)}"
            )

    def consume(
        self,
        *,
        request_id: int,
        chunk_start: int,
        hidden_ptr: int,
        rows: int,
        stream: int,
    ) -> None:
        """Append one contiguous target-hidden chunk to the draft timeline."""

        self._validate_owner(request_id)
        if self._closed:
            raise RuntimeError("streaming NextN prompt sink is closed")
        if self._failed:
            raise RuntimeError("streaming NextN prompt sink is failed")
        if self._finished:
            raise RuntimeError("streaming NextN prompt sink is already finished")
        start = int(chunk_start)
        count = int(rows)
        stream = int(stream)
        if start != self.consumed_rows:
            raise RuntimeError(
                "streaming NextN prompt chunks must be contiguous: "
                f"expected {self.consumed_rows}, got {start}"
            )
        if count <= 0 or start + count > self.total_rows:
            raise ValueError("streaming NextN prompt chunk is outside the prompt")
        if int(hidden_ptr) <= 0:
            raise ValueError("streaming NextN prompt chunk requires a device hidden pointer")
        if self._stream is None:
            self._stream = stream
        elif stream != self._stream:
            raise RuntimeError("streaming NextN prompt chunks must use one stream")
        if self.checkpoint is not None:
            self.checkpoint()

        tokens = self.prompt_tokens[start : start + count]
        try:
            draft_hidden_ptr = int(hidden_ptr)
            if self.transform_hidden_rows is not None:
                draft_hidden_ptr = int(
                    self.transform_hidden_rows(int(hidden_ptr), count, stream)
                )
                if draft_hidden_ptr <= 0:
                    raise RuntimeError(
                        "streaming NextN hidden transform returned a null pointer"
                    )
            self.executor.enqueue_prompt_rows(
                self.request_id,
                tokens[:1],
                position_start=self.start_position + start,
                target_hidden_base_ptr=self._pending_hidden.ptr,
                hidden_stride_bytes=self.hidden_nbytes,
                stream=stream,
            )
            if count > 1:
                self.executor.enqueue_prompt_rows(
                    self.request_id,
                    tokens[1:],
                    position_start=self.start_position + start + 1,
                    target_hidden_base_ptr=draft_hidden_ptr,
                    hidden_stride_bytes=self.hidden_nbytes,
                    stream=stream,
                )
            self.runtime.memcpy_async(
                self._pending_hidden.ptr,
                draft_hidden_ptr + (count - 1) * self.hidden_nbytes,
                self.hidden_nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                stream,
            )
        except Exception:
            self._failed = True
            raise
        self.consumed_rows += count

    def finish(self, *, request_id: int, total_rows: int, stream: int) -> None:
        """Validate complete request ownership at the target activation seam."""

        self._validate_owner(request_id)
        if self._closed:
            raise RuntimeError("streaming NextN prompt sink is closed")
        if self._failed:
            raise RuntimeError("streaming NextN prompt sink is failed")
        if int(total_rows) != self.total_rows or self.consumed_rows != self.total_rows:
            raise RuntimeError(
                "streaming NextN prompt sink finished before every row was consumed"
            )
        if self._stream is not None and int(stream) != self._stream:
            raise RuntimeError("streaming NextN prompt finish must use the chunk stream")
        self._finished = True

    def take_final_pending_buffer(self) -> DeviceBuffer:
        """Transfer the one carried BF16 row to the request-state owner."""

        if self._transferred:
            raise RuntimeError("streaming NextN carried hidden row was already transferred")
        if self._closed:
            raise RuntimeError("streaming NextN prompt sink is closed")
        if self._failed or not self._finished:
            raise RuntimeError("streaming NextN carried hidden row is not complete")
        self._transferred = True
        return self._pending_hidden

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._transferred:
            free(self._pending_hidden, runtime=self.runtime)


def _effective_target_verify_mode(
    requested: str,
    *,
    rows: int,
    backend: str | None = None,
    end_position: int | None = None,
) -> str:
    """Select only locally qualified native target rows before mutation."""

    selected = str(requested)
    # The native spec target-graph layer qualifies B1-B7 buckets (rows 2-8);
    # the cap derives from the declared candidate ladder, not a literal.
    native_max_rows = max(_GGUF_MTP_CANDIDATE_BUDGETS) + 1
    if selected == "native" and int(rows) > native_max_rows:
        return "serial_exact"
    if selected == "native" and backend is not None and end_position is not None:
        native_context_limit = int(
            backend_package_capability(
                str(backend),
                "GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT",
                int(end_position),
            )
        )
        if int(end_position) > native_context_limit:
            return "serial_exact"
    return selected


def _initial_state_only_journal_applies(
    requested: str,
    *,
    max_candidate_budget: int,
) -> bool:
    """Return whether every possible target row uses native session journals."""

    return (
        _effective_target_verify_mode(
            requested,
            rows=int(max_candidate_budget) + 1,
        )
        == "native"
    )


def _maybe_launch_device_proposal(
    verifier: Any,
    draft_provider: Any,
    proposal_context: MtpProposalContext,
    *,
    candidate_budget: int,
    remaining_decode: int,
    return_cycle_logits: bool,
) -> Any | None:
    """Launch a draft graph only when its cached target graph can retire it."""

    if bool(return_cycle_logits):
        return None
    device_ready = getattr(verifier, "device_proposal_ready", None)
    launch_device = getattr(draft_provider, "launch_device_proposal", None)
    if not callable(device_ready) or not callable(launch_device):
        return None
    if not device_ready(
        int(candidate_budget),
        remaining_decode=int(remaining_decode),
    ):
        return None
    return launch_device(
        proposal_context,
        candidate_budget=int(candidate_budget),
    )


@dataclass
class Qwen35GGUFVerifyGraphBucket:
    """Stable shared-ABI buffers for one scheduler verify shape.

    The correctness route is intentionally eager: ``captured`` remains false
    until a row-native GGUF chain forward can be captured without changing its
    serial target arithmetic. The scheduler still keys and reuses this bucket by
    the full ``BatchShapeKey`` contract rather than inventing a GGUF ABI.
    """

    key: Any
    owner: TargetVerifyBufferOwner
    remaining_decode: Tensor
    captured: bool = False
    replay_count: int = 0

    def to_json_dict(self) -> dict[str, object]:
        return {
            "mode": self.owner.spec.mode,
            "bucket": self.owner.spec.bucket,
            "max_rows": int(self.owner.spec.max_rows),
            "max_requests": int(self.owner.spec.max_requests),
            "captured": bool(self.captured),
            "replay_count": int(self.replay_count),
            "buffers": self.owner.compact_metadata(),
        }


@dataclass(frozen=True)
class Qwen35GGUFPreparedVerify:
    """Target rows and GPU accept summary awaiting transactional commit."""

    batch: TargetVerifyBatch
    buffers: TargetVerifyBuffers
    summary: TargetAcceptSummary
    target_top1: tuple[int, ...]
    target_logits: np.ndarray
    graph_bucket: Qwen35GGUFVerifyGraphBucket
    initial_position: int
    kv_journal_positions: tuple[int, ...]
    gpu_accept_match_cpu: bool
    target_verify_mode: str
    native_graph_submitted: bool = False
    native_graph_capture_ms: float = 0.0
    native_graph_submit_ms: float = 0.0
    native_graph_readback_ms: float = 0.0
    native_graph_fallback_reason: str | None = None
    native_device_accept_commit: bool = False
    native_proposal_target_chained: bool = False
    device_proposal_top1_values: tuple[float, ...] = ()
    device_state_commit_buffers: TargetStateCommitBuffers | None = None


@dataclass(frozen=True)
class Qwen35GGUFMTPGenerationResult:
    """One-request greedy GGUF generation result with MTP economics."""

    request_id: int
    token_ids: tuple[int, ...]
    candidate_budget: int
    accepted_counts: tuple[int, ...]
    target_forward_rows: int
    cycles: int
    prefill_seconds: float
    decode_seconds: float
    proposal_seconds: float
    verify_seconds: float
    gpu_accept_match_cpu: bool
    graph_stats: dict[str, object]
    cycle_records: tuple[dict[str, object], ...] = ()
    budget_policy_summary: dict[str, object] | None = None

    @property
    def accepted_draft_tokens(self) -> int:
        return sum(self.accepted_counts)

    @property
    def visible_tokens_per_cycle(self) -> float:
        if self.cycles <= 0:
            return 1.0
        return 1.0 + self.accepted_draft_tokens / self.cycles

    @property
    def decode_tok_s(self) -> float:
        return 0.0 if self.decode_seconds <= 0.0 else len(self.token_ids) / self.decode_seconds

    def to_json_dict(self) -> dict[str, object]:
        return {
            "request_id": int(self.request_id),
            "token_ids": list(self.token_ids),
            "candidate_budget": int(self.candidate_budget),
            "accepted_counts": list(self.accepted_counts),
            "accepted_draft_tokens": int(self.accepted_draft_tokens),
            "target_forward_rows": int(self.target_forward_rows),
            "cycles": int(self.cycles),
            "visible_tokens_per_cycle": float(self.visible_tokens_per_cycle),
            "prefill_seconds": float(self.prefill_seconds),
            "decode_seconds": float(self.decode_seconds),
            "proposal_seconds": float(self.proposal_seconds),
            "verify_seconds": float(self.verify_seconds),
            "decode_tok_s": float(self.decode_tok_s),
            "gpu_accept_match_cpu": bool(self.gpu_accept_match_cpu),
            "graph_stats": self.graph_stats,
            "cycle_records": list(self.cycle_records),
            "budget_policy": self.budget_policy_summary,
        }


@dataclass(frozen=True)
class _InitialStatePairCopy:
    """Fixed pointer tables for one-launch initial-state snapshot/rollback."""

    kernel: object
    live_table: DeviceBuffer
    snapshot_table: DeviceBuffer
    row_zero_i32: DeviceBuffer
    layer_count: int
    conv_row_nbytes: int
    recurrent_row_nbytes: int

    @property
    def buffers(self) -> tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer]:
        return self.live_table, self.snapshot_table, self.row_zero_i32


@dataclass
class _StateJournal:
    target: Qwen35GGUFResidentSession
    max_rows: int
    state_rows: tuple[tuple[DeviceBuffer, DeviceBuffer], ...]
    initial_hidden: DeviceBuffer
    row_hidden: DeviceBuffer
    initial_state_copy: _InitialStatePairCopy | None
    producer_capture_initial_state: bool
    initial_state_only: bool
    initial_state_captured: bool
    buffers: tuple[DeviceBuffer, ...]

    @classmethod
    def allocate(
        cls,
        target: Qwen35GGUFResidentSession,
        *,
        max_rows: int,
        producer_capture_initial_state: bool = False,
        initial_state_only: bool = False,
    ) -> "_StateJournal":
        owner = target._target_scratch_owner
        scratch = target.scratch
        if owner is None or scratch is None or target.runner is None:
            raise RuntimeError("GGUF target session is closed")
        runtime = target.runtime
        assert runtime is not None
        buffers: list[DeviceBuffer] = []
        conv_rows: list[tuple[int, DeviceBuffer, DeviceBuffer]] = []
        recurrent_rows: list[tuple[int, DeviceBuffer, DeviceBuffer]] = []
        producer_capture_active = False
        if producer_capture_initial_state:
            acquire = getattr(target, "_acquire_verify_initial_state_capture", None)
            captured = acquire() if callable(acquire) else None
            if captured is not None:
                conv_rows = list(captured[0])
                recurrent_rows = list(captured[1])
                producer_capture_active = True
        try:
            if not producer_capture_active:
                for family_rows, states in (
                    (conv_rows, scratch.layer_conv_states),
                    (recurrent_rows, scratch.layer_recurrent_states),
                ):
                    for layer_id, state in enumerate(states):
                        if state is None:
                            continue
                        snapshot_rows = 1 if initial_state_only else int(max_rows) + 1
                        snapshots = malloc(
                            snapshot_rows * int(state.nbytes),
                            runtime=runtime,
                        )
                        buffers.append(snapshots)
                        family_rows.append((int(layer_id), state, snapshots))
            hidden_nbytes = int(target.runner.hidden_size) * DType.BF16.itemsize
            initial_hidden = malloc(hidden_nbytes, runtime=runtime)
            buffers.append(initial_hidden)
            row_hidden = malloc(int(max_rows) * hidden_nbytes, runtime=runtime)
            buffers.append(row_hidden)
            initial_state_copy = cls._allocate_initial_state_copy(
                target,
                conv_rows=conv_rows,
                recurrent_rows=recurrent_rows,
                runtime=runtime,
            )
            if initial_state_copy is not None:
                buffers.extend(initial_state_copy.buffers)
        except BaseException:
            for buffer in reversed(buffers):
                free(buffer, runtime=runtime)
            if producer_capture_active:
                target._release_verify_initial_state_capture()
            raise
        return cls(
            target=target,
            max_rows=int(max_rows),
            state_rows=tuple(
                (state, snapshots)
                for _layer_id, state, snapshots in (*conv_rows, *recurrent_rows)
            ),
            initial_hidden=initial_hidden,
            row_hidden=row_hidden,
            initial_state_copy=initial_state_copy,
            producer_capture_initial_state=producer_capture_active,
            initial_state_only=bool(initial_state_only),
            initial_state_captured=False,
            buffers=tuple(buffers),
        )

    @staticmethod
    def _allocate_initial_state_copy(
        target: Qwen35GGUFResidentSession,
        *,
        conv_rows: list[tuple[int, DeviceBuffer, DeviceBuffer]],
        recurrent_rows: list[tuple[int, DeviceBuffer, DeviceBuffer]],
        runtime: HipRuntime,
    ) -> _InitialStatePairCopy | None:
        """Resolve and materialize the backend-owned pointer-table copy plan."""

        if not conv_rows or tuple(row[0] for row in conv_rows) != tuple(
            row[0] for row in recurrent_rows
        ):
            return None
        conv_row_nbytes = int(conv_rows[0][1].nbytes)
        recurrent_row_nbytes = int(recurrent_rows[0][1].nbytes)
        if any(int(state.nbytes) != conv_row_nbytes for _layer, state, _snapshots in conv_rows):
            return None
        if any(
            int(state.nbytes) != recurrent_row_nbytes
            for _layer, state, _snapshots in recurrent_rows
        ):
            return None
        key = KernelKey(target.backend, "linear_state_pair_copy", "f32", "chunked_i32")
        if not is_registered(key):
            return None
        kernel = resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        if kernel is None:
            return None
        live_host = np.asarray(
            [int(state.ptr) for _layer, state, _snapshots in (*conv_rows, *recurrent_rows)],
            dtype=np.uint64,
        )
        snapshot_host = np.asarray(
            [int(snapshots.ptr) for _layer, _state, snapshots in (*conv_rows, *recurrent_rows)],
            dtype=np.uint64,
        )
        row_zero_host = np.zeros((1,), dtype=np.int32)
        allocated: list[DeviceBuffer] = []
        try:
            live_table = malloc(live_host.nbytes, runtime=runtime)
            allocated.append(live_table)
            snapshot_table = malloc(snapshot_host.nbytes, runtime=runtime)
            allocated.append(snapshot_table)
            row_zero_i32 = malloc(row_zero_host.nbytes, runtime=runtime)
            allocated.append(row_zero_i32)
            for destination, source in (
                (live_table, live_host),
                (snapshot_table, snapshot_host),
                (row_zero_i32, row_zero_host),
            ):
                copy_host_to_device(
                    destination,
                    host_array_ptr(source),
                    source.nbytes,
                    runtime=runtime,
                )
        except BaseException:
            for buffer in reversed(allocated):
                free(buffer, runtime=runtime)
            raise
        return _InitialStatePairCopy(
            kernel=kernel,
            live_table=live_table,
            snapshot_table=snapshot_table,
            row_zero_i32=row_zero_i32,
            layer_count=len(conv_rows),
            conv_row_nbytes=conv_row_nbytes,
            recurrent_row_nbytes=recurrent_row_nbytes,
        )

    @property
    def hidden_nbytes(self) -> int:
        return int(self.initial_hidden.nbytes)

    def capture_initial(
        self,
        *,
        stream: int = 0,
        force_consumer_state: bool = False,
    ) -> None:
        self.initial_state_captured = False
        hidden = self.target.last_target_hidden
        self._copy_d2d(self.initial_hidden.ptr, hidden.ptr, self.hidden_nbytes, stream=stream)
        if self.producer_capture_initial_state and not bool(force_consumer_state):
            return
        if not self._copy_initial_state(restore=False, stream=stream):
            self._capture_state_index(0, stream=stream)
        self.initial_state_captured = True

    def mark_initial_state_captured(self) -> None:
        """Publish a fully retired producer-folded rollback snapshot."""

        if not self.producer_capture_initial_state:
            raise RuntimeError("initial state is not owned by producer capture")
        self.initial_state_captured = True

    def capture_row(self, row: int, *, stream: int = 0) -> None:
        if self.initial_state_only:
            raise RuntimeError("initial-state-only journal cannot capture serial rows")
        row = int(row)
        if row < 0 or row >= self.max_rows:
            raise ValueError("verify journal row outside capacity")
        hidden = self.target.last_target_hidden
        self._copy_d2d(
            self.row_hidden.ptr + row * self.hidden_nbytes,
            hidden.ptr,
            self.hidden_nbytes,
            stream=stream,
        )
        self._capture_state_index(row + 1, stream=stream)

    def capture_hidden_rows(self, hidden_rows: np.ndarray, *, stream: int = 0) -> None:
        """Stage exact BF16 trunk rows emitted by a native block forward."""

        if self.target.runner is None:
            raise RuntimeError("GGUF target session is closed")
        rows = np.ascontiguousarray(hidden_rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != int(self.target.runner.hidden_size):
            raise ValueError("native verifier trunk rows must have shape [rows, hidden_size]")
        if rows.shape[0] <= 0 or rows.shape[0] > self.max_rows:
            raise ValueError("native verifier trunk rows exceed journal capacity")
        hidden_bits = np.ascontiguousarray(float_array_to_bf16_bits(rows), dtype=np.uint16)
        copy_host_to_device(
            DeviceBuffer(self.row_hidden.ptr, hidden_bits.nbytes),
            host_array_ptr(hidden_bits),
            hidden_bits.nbytes,
            runtime=self.target.runtime,
        )

    def restore_initial(self, *, stream: int = 0) -> None:
        if self.initial_state_captured:
            if not self._copy_initial_state(restore=True, stream=stream):
                self._restore_state_index(0, stream=stream)
        self._restore_hidden(self.initial_hidden.ptr, stream=stream)

    def restore_row(self, row: int, *, stream: int = 0) -> None:
        if self.initial_state_only:
            raise RuntimeError("initial-state-only journal cannot restore serial rows")
        row = int(row)
        if row < 0 or row >= self.max_rows:
            raise ValueError("verify commit row outside journal capacity")
        self._restore_state_index(row + 1, stream=stream)
        self._restore_hidden(self.row_hidden.ptr + row * self.hidden_nbytes, stream=stream)

    def restore_native_row(self, row: int, *, position: int, stream: int = 0) -> None:
        """Commit a session-captured native state row and its exact trunk row."""

        row = int(row)
        if row < 0 or row >= self.max_rows:
            raise ValueError("verify commit row outside journal capacity")
        self.target._commit_verify_linear_state_row(
            row,
            position=int(position),
            stream=int(stream),
        )
        self._restore_hidden(self.row_hidden.ptr + row * self.hidden_nbytes, stream=stream)

    def hidden_rows_tensor(self, rows: int) -> Tensor:
        if self.target.runner is None:
            raise RuntimeError("GGUF target session is closed")
        return Tensor.from_handle(
            self.row_hidden.ptr,
            (1, int(rows), int(self.target.runner.hidden_size)),
            DType.BF16,
            Device("hip", 0),
        )

    def close(self) -> None:
        runtime = self.target.runtime
        try:
            for buffer in reversed(self.buffers):
                free(buffer, runtime=runtime)
        finally:
            if self.producer_capture_initial_state:
                self.target._release_verify_initial_state_capture()
                self.producer_capture_initial_state = False

    def _copy_initial_state(self, *, restore: bool, stream: int) -> bool:
        plan = self.initial_state_copy
        if plan is None:
            return False
        if self.target._dflash_commit_library is None:
            self.target._dflash_commit_library = build_dflash_commit(
                load=True,
                compiler_version=self.target.compiler_version,
                require_cached=self.target.require_cached_build,
            )
        table_half_nbytes = plan.layer_count * np.dtype(np.uint64).itemsize
        source_table = plan.snapshot_table if restore else plan.live_table
        destination_table = plan.live_table if restore else plan.snapshot_table
        plan.kernel(
            source_table.ptr,
            destination_table.ptr,
            plan.conv_row_nbytes,
            source_table.ptr + table_half_nbytes,
            destination_table.ptr + table_half_nbytes,
            plan.recurrent_row_nbytes,
            plan.row_zero_i32.ptr,
            plan.layer_count,
            stream=int(stream),
            library=self.target._dflash_commit_library,
            runtime=self.target.runtime,
        )
        return True

    def _capture_state_index(self, index: int, *, stream: int) -> None:
        for state, snapshots in self.state_rows:
            self._copy_d2d(
                snapshots.ptr + int(index) * int(state.nbytes),
                state.ptr,
                int(state.nbytes),
                stream=stream,
            )

    def _restore_state_index(self, index: int, *, stream: int) -> None:
        for state, snapshots in self.state_rows:
            self._copy_d2d(
                state.ptr,
                snapshots.ptr + int(index) * int(state.nbytes),
                int(state.nbytes),
                stream=stream,
            )

    def _restore_hidden(self, src_ptr: int, *, stream: int) -> None:
        hidden = self.target._hidden_a
        if hidden is None:
            raise RuntimeError("GGUF target hidden storage is closed")
        self._copy_d2d(hidden.ptr, int(src_ptr), self.hidden_nbytes, stream=stream)
        self.target._last_target_hidden_ptr = int(hidden.ptr)

    def _copy_d2d(self, dst: int, src: int, nbytes: int, *, stream: int) -> None:
        runtime = self.target.runtime
        if runtime is None:
            raise RuntimeError("GGUF target runtime is closed")
        runtime.memcpy_async(
            int(dst),
            int(src),
            int(nbytes),
            HipMemcpyKind.DEVICE_TO_DEVICE,
            int(stream),
        )


def _resolve_gguf_verifier_backend(
    target: Qwen35GGUFResidentSession,
    requested: str | None,
) -> str:
    """Use the target's concrete backend for every verifier registry lookup."""

    target_backend = str(target.backend)
    if target_backend == "auto":
        raise ValueError("GGUF target backend must be concrete before verifier setup")
    normalized = "auto" if requested is None else str(requested).strip()
    if normalized == "auto":
        return target_backend
    if normalized != target_backend:
        raise ValueError(
            f"verifier backend {normalized!r} does not match target backend "
            f"{target_backend!r}"
        )
    return normalized


class Qwen35GGUFTransactionalVerifier:
    """Shared-ABI chain verifier with journaled GGUF state/KV commit.

    Target rows execute either with retained exact c=1 arithmetic or the
    explicitly selected native row-attention/block-FFN path. Linear-attention
    state and the target-hidden tap are snapshotted per verify row. Full-attention
    K/V writes form an append journal in monotonically increasing positions;
    commit publishes only the selected prefix by resetting resident span
    position/context metadata, while rollback restores the pre-verify metadata.
    Rejected suffix cells are unreachable and overwritten on later appends.
    """

    def __init__(
        self,
        target: Qwen35GGUFResidentSession,
        *,
        max_candidate_budget: int = 3,
        backend: str | None = None,
        quant: str = "gguf_ud_q3_k_m",
        target_verify_mode: str = "serial_exact",
    ) -> None:
        if int(max_candidate_budget) not in _GGUF_MTP_CANDIDATE_BUDGETS:
            raise ValueError(
                "max_candidate_budget must be one of "
                f"{', '.join(str(value) for value in _GGUF_MTP_CANDIDATE_BUDGETS)}"
            )
        if target.runner is None or target.runtime is None:
            raise RuntimeError("GGUF target session is closed")
        selected_verify_mode = str(target_verify_mode).strip().lower().replace("-", "_")
        if selected_verify_mode not in _GGUF_MTP_TARGET_VERIFY_MODES:
            raise ValueError("target_verify_mode must be 'serial_exact' or 'native'")
        self.target = target
        self.max_candidate_budget = int(max_candidate_budget)
        self.backend = _resolve_gguf_verifier_backend(target, backend)
        self.quant = str(quant)
        self.target_verify_mode = selected_verify_mode
        self.workspace = RuntimeWorkspace(device=Device("hip", 0), runtime=target.runtime)
        self.journal = _StateJournal.allocate(
            target,
            max_rows=self.max_candidate_budget + 1,
            producer_capture_initial_state=(selected_verify_mode == "native"),
            initial_state_only=_initial_state_only_journal_applies(
                selected_verify_mode,
                max_candidate_budget=self.max_candidate_budget,
            ),
        )
        self._buckets: dict[object, Qwen35GGUFVerifyGraphBucket] = {}
        self._prepared: Qwen35GGUFPreparedVerify | None = None
        self._accept_kernel = resolve(
            backend=self.backend,
            layer="dflash_accept_chain",
            quant=self.quant,
            variant="i32",
        )
        self._accept_library = build_dflash_accept(
            load=True,
            compiler_version=target.compiler_version,
            require_cached=target.require_cached_build,
        )
        self.last_device_proposal_fallback_reason: str | None = None
        self.closed = False

    def graph_bucket(self, key: object, batch: TargetVerifyBatch) -> Qwen35GGUFVerifyGraphBucket:
        cached = self._buckets.get(key)
        if cached is not None:
            return cached
        rows = int(batch.rows)
        requests = len(batch.request_ids)
        spec = TargetVerifyBufferSpec(
            backend=self.backend,
            bucket=f"gguf-mtp-v{rows}-r{requests}",
            device=Device("hip", 0),
            max_rows=rows,
            max_requests=requests,
            mode=batch.mode,
        )
        owner = TargetVerifyBufferOwner.allocate(spec, workspace=self.workspace)
        remaining = self.workspace.reserve_tensor(
            f"target_verify/{self.backend}/{spec.bucket}/{spec.mode}/remaining_decode",
            (requests,),
            DType.INT32,
        )
        bucket = Qwen35GGUFVerifyGraphBucket(key=key, owner=owner, remaining_decode=remaining)
        self._buckets[key] = bucket
        return bucket

    def device_proposal_ready(
        self,
        candidate_budget: int,
        *,
        remaining_decode: int | None = None,
    ) -> bool:
        """Report cached target eligibility before launching any proposal work."""

        self.last_device_proposal_fallback_reason = None
        budget = int(candidate_budget)
        if self.closed:
            self.last_device_proposal_fallback_reason = "target_graph_verifier_closed"
            return False
        if self.target_verify_mode != "native":
            self.last_device_proposal_fallback_reason = "target_graph_verify_mode_miss"
            return False
        if budget not in _GGUF_MTP_CANDIDATE_BUDGETS:
            self.last_device_proposal_fallback_reason = "target_graph_budget_miss"
            return False
        graph = getattr(
            self.target,
            f"_native_spec_b{budget}_target_graph_n2",
            None,
        )
        if graph is None:
            self.last_device_proposal_fallback_reason = "target_graph_not_cached"
            return False
        eligibility = getattr(graph, "launch_ineligibility_reason", None)
        if not callable(eligibility):
            self.last_device_proposal_fallback_reason = "target_graph_admission_unavailable"
            return False
        reason = eligibility(
            self.target,
            position=int(self.target.position),
            rows=budget + 1,
            remaining_decode=(
                None if remaining_decode is None else int(remaining_decode)
            ),
            bulk_attention_mode="native",
            use_wmma_prefill=False,
            capture_linear_state_rows=True,
            capture_pre_output_norm_hidden=True,
            defer_linear_state_commit=True,
            device_accept_commit=True,
        )
        self.last_device_proposal_fallback_reason = reason
        if reason is not None:
            return False
        # The final short-context bucket row is exact for a host-materialized
        # proposal, but cross-stream device handoff can expose the proposal
        # sentinel before its result payload is reusable at that boundary.
        # Fail closed before launching proposal work; the established host
        # proposal can still use the target graph safely.
        context_limit = int(getattr(graph, "context_limit", 0))
        if (
            context_limit > 0
            and int(self.target.position) + budget + 1 >= context_limit
        ):
            self.last_device_proposal_fallback_reason = (
                "target_graph_proposal_handoff_boundary_miss"
            )
            return False
        return True

    def prepare(
        self,
        batch: TargetVerifyBatch,
        *,
        transaction_id: int,
        graph_bucket: Qwen35GGUFVerifyGraphBucket,
        remaining_decode: Sequence[int],
        return_logits: bool = False,
        stream: int = 0,
        device_proposal: Any | None = None,
        qualification_oracle: bool = True,
        allow_graph: bool = True,
    ) -> Qwen35GGUFPreparedVerify:
        if self.closed:
            raise RuntimeError("GGUF transactional verifier is closed")
        if self._prepared is not None:
            raise RuntimeError("a GGUF target verification transaction is already open")
        self._validate_chain(batch)
        budgets = tuple(int(value) for value in remaining_decode)
        if len(budgets) != len(batch.request_ids) or any(value < 0 for value in budgets):
            raise ValueError("remaining_decode must be non-negative and align with requests")
        if graph_bucket.owner.spec.mode != batch.mode or graph_bucket.owner.spec.max_rows < batch.rows:
            raise ValueError("verify graph bucket does not cover target batch")
        if int(self.target.position) != int(batch.positions[batch.root_rows[0]]):
            raise ValueError("target verify root position does not match resident cursor")
        if device_proposal is not None and (
            self.target_verify_mode != "native" or stream or return_logits
        ):
            raise ValueError(
                "device proposal handoff requires native no-logit verification on the session stream"
            )
        if device_proposal is not None and (
            int(getattr(device_proposal, "budget", -1)) + 1 != batch.rows
            or int(getattr(device_proposal, "request_id", -1)) != batch.request_ids[0]
        ):
            raise ValueError("device proposal identity does not match the target batch")

        initial_position = int(self.target.position)
        effective_verify_mode = _effective_target_verify_mode(
            self.target_verify_mode,
            rows=batch.rows,
            backend=self.backend,
            end_position=initial_position + int(batch.rows),
        )
        self.journal.capture_initial(
            stream=stream,
            force_consumer_state=(effective_verify_mode == "serial_exact"),
        )
        logits: list[np.ndarray] = []
        top1: list[int] = []
        native_graph_submitted = False
        native_graph_capture_ms = 0.0
        native_graph_submit_ms = 0.0
        native_graph_readback_ms = 0.0
        native_graph_fallback_reason = None
        native_device_accept_commit = False
        native_proposal_target_chained = False
        device_proposal_top1_values: tuple[float, ...] = ()
        device_state_commit_buffers: TargetStateCommitBuffers | None = None
        buffers: TargetVerifyBuffers | None = None
        gpu_summary: TargetAcceptSummary | None = None
        try:
            if effective_verify_mode == "native":
                native_kwargs = {
                    "bulk_attention_mode": "native",
                    "use_wmma_prefill": False,
                    "capture_linear_state_rows": True,
                    "capture_pre_output_norm_hidden": True,
                    "capture_lm_head_logits": return_logits,
                    "defer_linear_state_commit": True,
                }
                device_block = None
                if (
                    bool(allow_graph)
                    and not stream
                    and not return_logits
                    and budgets[0] >= batch.rows
                ):
                    from hipengine.runtime.gguf_native_spec_cycle import (
                        NativeSpecTargetGraphUnsupportedError,
                    )

                    try:
                        if device_proposal is None:
                            device_block = self.target.verify_target_block_native_cycle(
                                batch.tokens,
                                fallback=False,
                                cycle_id=int(graph_bucket.replay_count),
                                transaction_id=int(transaction_id),
                                request_id=int(batch.request_ids[0]),
                                device_accept_commit=True,
                                remaining_decode=int(budgets[0]),
                                **native_kwargs,
                            )
                        else:
                            device_block = self.target.verify_target_from_device_proposal(
                                device_proposal,
                                cycle_id=int(graph_bucket.replay_count),
                                transaction_id=int(transaction_id),
                                request_id=int(batch.request_ids[0]),
                                remaining_decode=int(budgets[0]),
                                compact_result=not bool(qualification_oracle),
                                **native_kwargs,
                            )
                    except NativeSpecTargetGraphUnsupportedError:
                        if device_proposal is not None:
                            raise
                        device_block = None
                if device_block is not None:
                    native_graph_submitted = bool(
                        self.target.last_native_spec_target_submitted
                    )
                    native_graph_capture_ms = float(
                        self.target.last_native_spec_target_capture_ms
                    )
                    native_graph_submit_ms = float(
                        self.target.last_native_spec_target_submit_ms
                    )
                    native_graph_readback_ms = float(
                        self.target.last_native_spec_target_readback_ms
                    )
                    native_graph_fallback_reason = (
                        self.target.last_native_spec_target_fallback_reason
                    )
                    if not bool(getattr(device_block, "device_accept_commit", False)):
                        raise RuntimeError("native GGUF N2 graph did not commit on device")
                    if int(device_block.start_position) != initial_position:
                        raise RuntimeError(
                            "native GGUF N2 graph changed the declared root position"
                        )
                    compact_result = bool(
                        getattr(device_block, "compact_result", False)
                    )
                    if device_proposal is not None:
                        if not compact_result:
                            from hipengine.runtime.gguf_native_spec_cycle import (
                                build_native_b2_target_batch,
                            )

                            batch = build_native_b2_target_batch(
                                device_block.input_token_ids,
                                start_position=initial_position,
                                request_id=int(batch.request_ids[0]),
                            )
                            self._validate_chain(batch)
                            device_proposal_top1_values = tuple(
                                float(value)
                                for value in device_block.proposal_top1_values
                            )
                            if (
                                len(device_proposal_top1_values)
                                != batch.candidate_count
                            ):
                                raise RuntimeError(
                                    "native GGUF N2 graph omitted proposal top-1 values"
                                )
                        native_proposal_target_chained = True
                    top1.extend(int(token) for token in device_block.target_top1)
                    if not compact_result and len(top1) != batch.rows:
                        raise RuntimeError("native GGUF N2 graph omitted target top-1 rows")
                    buffers = device_block.verify_buffers
                    device_state_commit_buffers = device_block.state_commit_buffers
                    if buffers is None or device_state_commit_buffers is None:
                        raise RuntimeError("native GGUF N2 graph omitted device buffer descriptors")
                    payload = {
                        "accepted_counts": (int(device_block.accepted_draft_tokens),),
                        "commit_rows": (int(device_block.commit_row),),
                        "commit_tokens": (int(device_block.commit_token),),
                        "commit_positions": (int(device_block.commit_position),),
                        "next_tokens": (int(device_block.next_token),),
                        "full_accept": (bool(device_block.full_accept),),
                    }
                    if compact_result:
                        accepted_count = int(device_block.accepted_draft_tokens)
                        gpu_summary = TargetAcceptSummary(
                            request_ids=batch.request_ids,
                            accepted_counts=(accepted_count,),
                            accepted_tokens=(
                                tuple(
                                    int(token)
                                    for token in device_block.token_ids[
                                        :accepted_count
                                    ]
                                ),
                            ),
                            commit_rows=(int(device_block.commit_row),),
                            commit_tokens=(int(device_block.commit_token),),
                            commit_positions=(int(device_block.commit_position),),
                            next_tokens=(int(device_block.next_token),),
                            full_accept=(bool(device_block.full_accept),),
                            candidate_counts=batch.candidate_counts,
                            transaction_id=int(transaction_id),
                            draft_depth=batch.draft_depth,
                            tree_shape=batch.tree_shape,
                            mode=batch.mode,
                        )
                    else:
                        gpu_summary = replace(
                            TargetAcceptSummary.from_gpu_payload(batch, payload),
                            transaction_id=int(transaction_id),
                        )
                    expected_visible = (
                        *gpu_summary.accepted_tokens[0],
                        gpu_summary.next_tokens[0],
                    )
                    if tuple(int(token) for token in device_block.token_ids) != expected_visible:
                        raise RuntimeError("native GGUF N2 graph returned inconsistent visible tokens")
                    if int(device_block.end_position) != initial_position + len(expected_visible):
                        raise RuntimeError("native GGUF N2 graph returned an inconsistent cursor")
                    if self.journal.producer_capture_initial_state:
                        self.journal.mark_initial_state_captured()
                    native_device_accept_commit = True
                else:
                    if stream or not bool(allow_graph):
                        native_graph_fallback_reason = (
                            "native target graph does not support caller-owned streams"
                            if stream
                            else "native target graph disabled by caller"
                        )
                        block = self.target.verify_target_block(
                            batch.tokens,
                            stream=stream,
                            **native_kwargs,
                        )
                    else:
                        block = self.target.verify_target_block_native_cycle(
                            batch.tokens,
                            fallback=True,
                            cycle_id=int(graph_bucket.replay_count),
                            transaction_id=int(transaction_id),
                            request_id=int(batch.request_ids[0]),
                            **native_kwargs,
                        )
                    if not stream:
                        native_graph_submitted = bool(
                            self.target.last_native_spec_target_submitted
                        )
                        native_graph_capture_ms = float(
                            self.target.last_native_spec_target_capture_ms
                        )
                        native_graph_submit_ms = float(
                            self.target.last_native_spec_target_submit_ms
                        )
                        native_graph_readback_ms = float(
                            self.target.last_native_spec_target_readback_ms
                        )
                        native_graph_fallback_reason = (
                            self.target.last_native_spec_target_fallback_reason
                        )
                    if block is None:
                        raise RuntimeError("native GGUF target verifier produced no host result")
                    if int(block.start_position) != initial_position:
                        raise RuntimeError(
                            "native GGUF target verifier changed the declared root position"
                        )
                    if self.journal.producer_capture_initial_state:
                        self.journal.mark_initial_state_captured()
                    if block.pre_output_norm_hidden is None:
                        raise RuntimeError(
                            "native GGUF target verifier did not capture trunk hidden rows"
                        )
                    top1.extend(int(token) for token in block.token_ids)
                    self.journal.capture_hidden_rows(
                        block.pre_output_norm_hidden,
                        stream=stream,
                    )
                    if return_logits:
                        if block.lm_head_logits_f32 is None:
                            raise RuntimeError(
                                "native GGUF target verifier did not capture requested logits"
                            )
                        logits.append(block.lm_head_logits_f32)
            else:
                for row, (token, position) in enumerate(zip(batch.tokens, batch.positions, strict=True)):
                    result = self.target.step(
                        int(token),
                        position=int(position),
                        return_logits=return_logits,
                        span_role=batch.mode,
                    )
                    top1.append(int(result.token_id))
                    if return_logits:
                        logits.append(result.logits)
                    self.journal.capture_row(row, stream=stream)

            if gpu_summary is None:
                buffers = graph_bucket.owner.bind(batch, transaction_id=int(transaction_id))
                self._write_verify_inputs(
                    buffers,
                    batch,
                    top1,
                    graph_bucket.remaining_decode,
                    budgets,
                )
                assert self._accept_kernel is not None
                self._accept_kernel(
                    buffers.token_ids.ptr,
                    buffers.positions.ptr,
                    buffers.parent_rows.ptr,
                    buffers.draft_depths.ptr,
                    buffers.active_mask.ptr,
                    buffers.target_top1.ptr,
                    graph_bucket.remaining_decode.ptr,
                    buffers.accepted_counts.ptr,
                    buffers.commit_rows.ptr,
                    buffers.commit_tokens.ptr,
                    buffers.commit_positions.ptr,
                    buffers.next_tokens.ptr if buffers.next_tokens is not None else 0,
                    buffers.full_accept.ptr if buffers.full_accept is not None else 0,
                    (
                        buffers.committed_output_ids.ptr
                        if buffers.committed_output_ids is not None
                        else 0
                    ),
                    (
                        buffers.committed_output_lengths.ptr
                        if buffers.committed_output_lengths is not None
                        else 0
                    ),
                    batch.rows,
                    len(batch.request_ids),
                    (
                        buffers.committed_output_ids.shape[1]
                        if buffers.committed_output_ids is not None
                        else batch.rows
                    ),
                    stream=stream,
                    library=self._accept_library,
                    runtime=self.target.runtime,
                )
                payload = self._read_accept_payload(buffers, stream=stream)
                gpu_summary = replace(
                    TargetAcceptSummary.from_gpu_payload(batch, payload),
                    transaction_id=int(transaction_id),
                )
            if buffers is None or gpu_summary is None:
                raise RuntimeError("GGUF target verifier omitted transaction buffers")
            gpu_match = True
            if bool(qualification_oracle):
                cpu_result = batch.accept_from_top1(
                    top1,
                    transaction_id=int(transaction_id),
                    remaining_decode=budgets,
                )
                cpu_summary = TargetAcceptSummary.from_accept_result(
                    batch, cpu_result
                )
                gpu_match = _summary_matches(gpu_summary, cpu_summary)
                if not gpu_match:
                    raise RuntimeError(
                        "GGUF GPU accept summary does not match the CPU oracle: "
                        f"gpu={gpu_summary!r} cpu={cpu_summary!r} "
                        f"tokens={batch.tokens!r} top1={tuple(top1)!r}"
                    )
            graph_bucket.replay_count += 1
            prepared = Qwen35GGUFPreparedVerify(
                batch=batch,
                buffers=buffers,
                summary=gpu_summary,
                target_top1=tuple(top1),
                target_logits=(
                    np.concatenate(logits, axis=0)
                    if logits
                    else np.empty((batch.rows, 0), dtype=np.float32)
                ),
                graph_bucket=graph_bucket,
                initial_position=initial_position,
                kv_journal_positions=tuple(int(position) for position in batch.positions),
                gpu_accept_match_cpu=gpu_match,
                target_verify_mode=effective_verify_mode,
                native_graph_submitted=native_graph_submitted,
                native_graph_capture_ms=native_graph_capture_ms,
                native_graph_submit_ms=native_graph_submit_ms,
                native_graph_readback_ms=native_graph_readback_ms,
                native_graph_fallback_reason=native_graph_fallback_reason,
                native_device_accept_commit=native_device_accept_commit,
                native_proposal_target_chained=native_proposal_target_chained,
                device_proposal_top1_values=device_proposal_top1_values,
                device_state_commit_buffers=device_state_commit_buffers,
            )
            self._prepared = prepared
            return prepared
        except Exception:
            self.journal.restore_initial(stream=stream)
            self._publish_position(initial_position, stream=stream)
            self._synchronize(stream)
            raise

    def commit(
        self,
        prepared: Qwen35GGUFPreparedVerify,
        plan: TargetCommitPlan,
        *,
        stream: int = 0,
    ) -> TargetStateCommitBuffers:
        self._require_open(prepared)
        if plan.transaction_id != prepared.buffers.transaction_id:
            raise ValueError("GGUF commit transaction_id must match prepared verify buffers")
        if plan.request_ids != prepared.batch.request_ids or plan.mode != prepared.batch.mode:
            raise ValueError("GGUF commit plan must match prepared target batch")
        expected = prepared.summary
        if (
            plan.accepted_counts != expected.accepted_counts
            or plan.commit_rows != expected.commit_rows
            or plan.commit_tokens != expected.commit_tokens
            or plan.commit_positions != expected.commit_positions
            or plan.next_tokens != expected.next_tokens
        ):
            raise ValueError("GGUF commit plan must match GPU target accept summary")
        selected_row = int(plan.commit_rows[0])
        next_position = int(plan.commit_positions[0]) + 1
        if prepared.native_device_accept_commit:
            if stream:
                raise ValueError("device-committed GGUF verify does not accept a commit stream")
            state_buffers = prepared.device_state_commit_buffers
            if state_buffers is None:
                raise RuntimeError("device-committed GGUF verify omitted state buffers")
            if (
                state_buffers.transaction_id != plan.transaction_id
                or state_buffers.request_ids != plan.request_ids
                or state_buffers.mode != plan.mode
            ):
                raise RuntimeError("device-committed GGUF state buffers drifted from the plan")
            if int(self.target.position) != next_position:
                raise RuntimeError("device-committed GGUF target cursor drifted before finalize")
            return state_buffers
        if prepared.target_verify_mode == "native":
            self.journal.restore_native_row(
                selected_row,
                position=next_position,
                stream=stream,
            )
        else:
            self.journal.restore_row(selected_row, stream=stream)
        self._publish_position(next_position, stream=stream)
        self._synchronize(stream)
        target_hidden = self.target.last_target_hidden
        hidden_dst = Tensor.from_handle(
            target_hidden.ptr,
            (1, 1, target_hidden.shape[1]),
            target_hidden.dtype,
            target_hidden.device,
        )
        return TargetStateCommitBuffers.for_plan(
            plan,
            accepted_counts=prepared.buffers.accepted_counts,
            commit_rows=prepared.buffers.commit_rows,
            commit_positions=prepared.buffers.commit_positions,
            hidden_taps_src=self.journal.hidden_rows_tensor(prepared.batch.rows),
            hidden_taps_dst=hidden_dst,
        )

    def rollback(self, prepared: Qwen35GGUFPreparedVerify, *, stream: int = 0) -> None:
        self._require_open(prepared)
        self.journal.restore_initial(stream=stream)
        self._publish_position(prepared.initial_position, stream=stream)
        self._synchronize(stream)
        self._prepared = None

    def finish(self, prepared: Qwen35GGUFPreparedVerify) -> None:
        self._require_open(prepared)
        self._prepared = None

    def close(self) -> None:
        if self.closed:
            return
        if self._prepared is not None:
            self.rollback(self._prepared)
        self.closed = True
        self.journal.close()
        self.workspace.free()
        self._buckets.clear()

    def _validate_chain(self, batch: TargetVerifyBatch) -> None:
        if batch.mode != "verify_chain":
            raise ValueError("GGUF transactional verifier supports verify_chain only")
        if len(batch.request_ids) != 1:
            raise ValueError("GGUF transactional verifier currently supports one request")
        if batch.rows > self.max_candidate_budget + 1:
            raise ValueError("target verify rows exceed GGUF verifier capacity")
        if batch.root_rows != (0,) or batch.candidate_rows != tuple(range(1, batch.rows)):
            raise ValueError("GGUF verify chain requires root-first contiguous rows")
        if batch.parent_rows != (-1, *tuple(range(batch.rows - 1))):
            raise ValueError("GGUF verify chain requires linear parent rows")
        if not all(batch.active_mask):
            raise ValueError("GGUF verify chain does not execute inactive padding rows")
        expected_positions = tuple(range(int(batch.positions[0]), int(batch.positions[0]) + batch.rows))
        if batch.positions != expected_positions:
            raise ValueError("GGUF verify chain positions must be contiguous")

    def _write_verify_inputs(
        self,
        buffers: TargetVerifyBuffers,
        batch: TargetVerifyBatch,
        top1: Sequence[int],
        remaining: Tensor,
        budgets: Sequence[int],
    ) -> None:
        _copy_array(buffers.token_ids, np.asarray(batch.tokens, dtype=np.int32), self.target.runtime)
        _copy_array(buffers.positions, np.asarray(batch.positions, dtype=np.int32), self.target.runtime)
        _copy_array(buffers.parent_rows, np.asarray(batch.parent_rows, dtype=np.int32), self.target.runtime)
        _copy_array(buffers.draft_depths, np.asarray(batch.draft_depths, dtype=np.int32), self.target.runtime)
        _copy_array(buffers.row_to_request, np.asarray(batch.row_to_request, dtype=np.int32), self.target.runtime)
        _copy_array(buffers.active_mask, np.asarray(batch.active_mask, dtype=np.uint8), self.target.runtime)
        _copy_array(buffers.target_top1, np.asarray(top1, dtype=np.int32), self.target.runtime)
        _copy_array(remaining, np.asarray(budgets, dtype=np.int32), self.target.runtime)

    def _read_accept_payload(
        self,
        buffers: TargetVerifyBuffers,
        *,
        stream: int,
    ) -> dict[str, tuple[int, ...] | tuple[bool, ...]]:
        self._synchronize(stream)
        return {
            "accepted_counts": _read_int32(buffers.accepted_counts, self.target.runtime),
            "commit_rows": _read_int32(buffers.commit_rows, self.target.runtime),
            "commit_tokens": _read_int32(buffers.commit_tokens, self.target.runtime),
            "commit_positions": _read_int32(buffers.commit_positions, self.target.runtime),
            "next_tokens": _read_int32_required(buffers.next_tokens, self.target.runtime),
            "full_accept": _read_bool_required(buffers.full_accept, self.target.runtime),
        }

    def _publish_position(self, next_position: int, *, stream: int) -> None:
        owner = self.target._target_scratch_owner
        if owner is None:
            raise RuntimeError("GGUF target scratch is closed")
        positions = list(owner.position_host.tolist())
        positions[0] = int(next_position)
        owner.set_full_attention_positions(tuple(positions), self.target.runtime)
        self.target._position = int(next_position)

    def _synchronize(self, stream: int) -> None:
        runtime = self.target.runtime
        if runtime is None:
            raise RuntimeError("GGUF target runtime is closed")
        if stream:
            runtime.stream_synchronize(int(stream))
        else:
            runtime.device_synchronize()

    def _require_open(self, prepared: Qwen35GGUFPreparedVerify) -> None:
        if self._prepared is not prepared:
            raise ValueError("prepared GGUF verification is not the open transaction")

    def __enter__(self) -> "Qwen35GGUFTransactionalVerifier":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class Qwen35GGUFMTPDecodeSession:
    """One-request GGUF MTP loop over the shared scheduler transaction ABI."""

    def __init__(
        self,
        target: Qwen35GGUFResidentSession,
        draft_provider: Qwen35GGUFNextNDraftProvider,
        *,
        candidate_budget: int = 1,
        quant: str = "gguf_ud_q3_k_m",
        target_verify_mode: str = "serial_exact",
        verifier: Qwen35GGUFTransactionalVerifier | None = None,
        owns_verifier: bool = True,
        draft_hidden_variant: str = "pre_output_norm",
    ) -> None:
        if int(candidate_budget) not in _GGUF_MTP_CANDIDATE_BUDGETS:
            raise ValueError(
                "candidate_budget must be one of "
                f"{', '.join(str(value) for value in _GGUF_MTP_CANDIDATE_BUDGETS)}"
            )
        selected_quant = str(quant).strip()
        if not selected_quant:
            raise ValueError("quant must be non-empty")
        self.target = target
        self.draft_provider = draft_provider
        self.candidate_budget = int(candidate_budget)
        self.quant = selected_quant
        self.draft_hidden_policy = Qwen35GGUFDraftHiddenPolicy(
            target_hidden_variant=str(draft_hidden_variant)
        )
        self.verifier = verifier or Qwen35GGUFTransactionalVerifier(
            target,
            max_candidate_budget=max(_GGUF_MTP_CANDIDATE_BUDGETS),
            quant=self.quant,
            target_verify_mode=target_verify_mode,
        )
        self.owns_verifier = bool(owns_verifier if verifier is not None else True)
        self._draft_post_norm_hidden: DeviceBuffer | None = None
        if self.draft_hidden_policy.requires_target_output_norm:
            if target.runner is None or target.runtime is None:
                if verifier is None:
                    self.verifier.close()
                raise RuntimeError("GGUF target session is closed")
            try:
                self._draft_post_norm_hidden = malloc(
                    target.runner.hidden_size * DType.BF16.itemsize,
                    runtime=target.runtime,
                )
            except Exception:
                if verifier is None:
                    self.verifier.close()
                raise

    @property
    def draft_hidden_manifest(self) -> dict[str, str]:
        policy = getattr(self, "draft_hidden_policy", Qwen35GGUFDraftHiddenPolicy())
        return policy.manifest()

    def _normalize_target_hidden_rows(
        self,
        src_ptr: int,
        dst_ptr: int,
        rows: int,
        *,
        stream: int,
    ) -> None:
        if self.target.runner is None or self.target.runner.weights is None:
            raise RuntimeError("GGUF target session is closed")
        gguf_rmsnorm_bf16_f32_weight(
            int(src_ptr),
            self.target.runner.weights.root("output_norm").allocation().tensor.ptr,
            int(dst_ptr),
            rows=int(rows),
            hidden_size=self.target.runner.hidden_size,
            eps=self.target.runner.weights.config.rms_norm_eps,
            stream=int(stream),
            runtime=self.target.runtime,
        )

    def _proposal_target_hidden(self) -> Tensor:
        hidden = self.target.last_target_hidden
        policy = getattr(self, "draft_hidden_policy", Qwen35GGUFDraftHiddenPolicy())
        if not policy.requires_target_output_norm:
            return hidden
        destination = self._draft_post_norm_hidden
        if destination is None or self.target.runner is None or self.target.runtime is None:
            raise RuntimeError("post-output-norm draft hidden storage is closed")
        self._normalize_target_hidden_rows(
            hidden.ptr,
            destination.ptr,
            1,
            stream=0,
        )
        # Proposal graphs own private streams. Retire the target normalization
        # before either an eager default-stream proposal or a graph D2D handoff.
        self.target.runtime.device_synchronize()
        return Tensor.from_handle(
            destination.ptr,
            (1, self.target.runner.hidden_size),
            DType.BF16,
            Device("hip", 0),
        )

    def generate(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_new_tokens: int,
        request_id: int = 0,
        return_cycle_logits: bool = False,
        use_bulk_prefill: bool | None = None,
        prefill_draft: bool = True,
        eos_token_id: int | None = None,
        stop_token_ids: Sequence[int] = (),
        checkpoint: Callable[[], None] | None = None,
        lifecycle_hook: Callable[[str], None] | None = None,
        budget_policy: MtpBudgetPolicy | None = None,
    ) -> Qwen35GGUFMTPGenerationResult:
        """Generate one MTP request, observing cancellation/deadline/shutdown.

        ``checkpoint`` is a no-arg callable polled at every streamed prompt
        chunk and cycle boundary before the corresponding draft/proposal
        mutation (RF3). It should raise
        ``GenerationCancelled`` / ``GenerationDeadlineExceeded`` (or any error)
        to stop at the next owned boundary. ``lifecycle_hook`` receives stable
        phase names for fault injection and direct lifecycle evidence; production
        cancellation continues to use the request-owned checkpoint/token.
        """
        prompt = tuple(int(token) for token in prompt_tokens)
        if not prompt:
            raise ValueError("prompt_tokens must be non-empty")
        if int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.target.reset()
        self.draft_provider.reset_request(int(request_id))
        scheduler = ResidentBatchScheduler(capacity=1)
        rid = scheduler.submit(prompt, max_new_tokens=int(max_new_tokens), request_id=int(request_id))
        if budget_policy is not None:
            budget_policy.start_request(
                request_id=rid,
                max_budget=self.candidate_budget,
            )
        scheduler.admit_pending()
        scheduler.next_prefill_work(chunk_size=len(prompt))

        prefill_started = time.perf_counter()
        if prefill_draft:
            first = self._prefill_target_and_draft(
                prompt,
                request_id=rid,
                use_bulk=use_bulk_prefill,
                checkpoint=checkpoint,
            )
        else:
            first = self.target.prefill(
                prompt,
                use_bulk=use_bulk_prefill,
                return_logits=False,
            )
        prefill_seconds = time.perf_counter() - prefill_started
        scheduler.record_generated(((rid, int(first.token_id)),))
        if eos_token_id is not None or stop_token_ids:
            scheduler.finish_request_at_stop(
                rid,
                eos_token_id=eos_token_id,
                stop_token_ids=stop_token_ids,
            )
        if rid in scheduler.completed:
            return Qwen35GGUFMTPGenerationResult(
                request_id=rid,
                token_ids=scheduler.completed[rid].generated_tokens,
                candidate_budget=self.candidate_budget,
                accepted_counts=(),
                target_forward_rows=0,
                cycles=0,
                prefill_seconds=prefill_seconds,
                decode_seconds=0.0,
                proposal_seconds=0.0,
                verify_seconds=0.0,
                gpu_accept_match_cpu=True,
                graph_stats=scheduler.graph_buckets.stats.to_json_dict(),
                budget_policy_summary=(
                    None if budget_policy is None else budget_policy.summary()
                ),
            )

        policy = self._register_kv_policy(rid)
        root = int(first.token_id)
        accepted_counts: list[int] = []
        records: list[dict[str, object]] = []
        proposal_seconds = 0.0
        verify_seconds = 0.0
        target_rows = 0
        gpu_match = True
        decode_started = time.perf_counter()
        while rid not in scheduler.completed:
            _mtp_cycle_checkpoint(checkpoint)
            cycle_started = time.perf_counter()
            cycle_phases: list[str] = []

            def lifecycle_phase(name: str) -> None:
                cycle_phases.append(str(name))
                _mtp_lifecycle_phase(lifecycle_hook, name)

            lifecycle_phase("before_proposal")
            request = scheduler.active_batch.requests[rid]
            remaining = int(request.remaining_decode)
            if remaining <= 0:
                break
            max_cycle_budget = _largest_budget_at_most(
                min(self.candidate_budget, remaining)
            )
            budget = max_cycle_budget
            if budget_policy is not None:
                budget = int(
                    budget_policy.choose_budget(
                        cycle=len(accepted_counts) + 1,
                        max_budget=max_cycle_budget,
                        remaining_decode=remaining,
                    )
                )
                if (
                    budget not in _GGUF_MTP_CANDIDATE_BUDGETS
                    or budget > max_cycle_budget
                    or budget > self.candidate_budget
                ):
                    raise ValueError(
                        "budget policy selected an unsupported or oversized budget"
                    )
            proposal_context = MtpProposalContext(
                request_ids=(rid,),
                root_tokens=(root,),
                root_positions=(int(self.target.position),),
                target_hidden=self._proposal_target_hidden(),
            )
            proposal_started = time.perf_counter()
            device_proposal = _maybe_launch_device_proposal(
                self.verifier,
                self.draft_provider,
                proposal_context,
                candidate_budget=budget,
                remaining_decode=remaining,
                return_cycle_logits=return_cycle_logits,
            )
            device_proposal_fallback_reason = (
                None
                if return_cycle_logits
                else getattr(
                    self.verifier,
                    "last_device_proposal_fallback_reason",
                    None,
                )
            )
            if device_proposal is None:
                draft = self.draft_provider.propose(
                    proposal_context,
                    candidate_budget=budget,
                    return_logits=return_cycle_logits,
                )
            else:
                placeholder = getattr(
                    self.draft_provider,
                    "placeholder_device_proposal",
                    None,
                )
                if not callable(placeholder):
                    raise RuntimeError(
                        "GGUF device proposal provider omitted its scheduler placeholder"
                    )
                draft = placeholder(device_proposal)
            proposal_seconds += time.perf_counter() - proposal_started
            lifecycle_phase("after_proposal_before_target")
            work = scheduler.next_speculative_verify_work(
                draft,
                root_tokens=(root,),
                root_positions=(int(self.target.position),),
            )
            config = self.target.runner.weights.config
            experts_per_token = int(getattr(config, "expert_used_count", 0) or 0)
            plan = scheduler.plan_speculative_verify(
                policy,
                work,
                lambda key, batch=work.target_batch: self.verifier.graph_bucket(key, batch),
                top_k=experts_per_token,
                experts_per_token=experts_per_token,
            )
            bucket = plan.graph
            if not isinstance(bucket, Qwen35GGUFVerifyGraphBucket):
                raise TypeError("GGUF speculative graph cache returned an incompatible bucket")
            prepared: Qwen35GGUFPreparedVerify | None = None
            target_transaction_committed = False
            prepared_top1: tuple[int, ...] = ()
            prepared_logits = np.empty((0, 0), dtype=np.float32)
            prepared_gpu_match = False
            prepared_verify_mode = ""
            prepared_native_graph_submitted = False
            prepared_native_graph_capture_ms = 0.0
            prepared_native_graph_submit_ms = 0.0
            prepared_native_graph_readback_ms = 0.0
            prepared_native_graph_fallback_reason = None
            prepared_native_device_accept_commit = False
            draft_tail_advanced = False
            try:
                verify_started = time.perf_counter()
                prepared = self.verifier.prepare(
                    work.target_batch,
                    transaction_id=plan.transaction.transaction_id,
                    graph_bucket=bucket,
                    remaining_decode=(remaining,),
                    return_logits=return_cycle_logits,
                    device_proposal=device_proposal,
                )
                verify_seconds += time.perf_counter() - verify_started
                lifecycle_phase("after_target_prepare")
                if device_proposal is not None:
                    if not prepared.native_proposal_target_chained:
                        raise RuntimeError(
                            "GGUF device proposal retired without target-chain ownership"
                        )
                    finish_device = getattr(
                        self.draft_provider,
                        "finish_device_proposal",
                        None,
                    )
                    if not callable(finish_device):
                        raise RuntimeError(
                            "GGUF device proposal provider omitted result materialization"
                        )
                    proposal_finish_started = time.perf_counter()
                    draft = finish_device(
                        device_proposal,
                        token_ids=tuple(
                            int(token) for token in prepared.batch.tokens[1:]
                        ),
                        top1_values=prepared.device_proposal_top1_values,
                    )
                    proposal_seconds += time.perf_counter() - proposal_finish_started
                    actual_work = scheduler.next_speculative_verify_work(
                        draft,
                        root_tokens=(root,),
                        root_positions=(int(prepared.initial_position),),
                    )
                    if actual_work.target_batch != prepared.batch:
                        raise RuntimeError(
                            "GGUF device proposal scheduler rows drifted after target retirement"
                        )
                    work = actual_work
                    plan = replace(
                        plan,
                        target_batch=actual_work.target_batch,
                        work_item=actual_work.work_item,
                    )
                target_rows += prepared.batch.rows
                prepared_top1 = prepared.target_top1
                prepared_logits = prepared.target_logits
                prepared_gpu_match = prepared.gpu_accept_match_cpu
                prepared_verify_mode = prepared.target_verify_mode
                prepared_native_graph_submitted = prepared.native_graph_submitted
                prepared_native_graph_capture_ms = prepared.native_graph_capture_ms
                prepared_native_graph_submit_ms = prepared.native_graph_submit_ms
                prepared_native_graph_readback_ms = prepared.native_graph_readback_ms
                prepared_native_graph_fallback_reason = (
                    prepared.native_graph_fallback_reason
                )
                prepared_native_device_accept_commit = (
                    prepared.native_device_accept_commit
                )
                buffer_plan = scheduler.bind_speculative_verify_buffers(plan, prepared.buffers)
                commit = scheduler.plan_speculative_commit(buffer_plan, prepared.summary)
                state_buffers = self.verifier.commit(prepared, commit.commit_plan)
                state_plan = scheduler.bind_speculative_commit_buffers(commit, state_buffers)
                committed_txn = scheduler.commit_speculative_kv_transaction(policy, state_plan)
                target_transaction_committed = True
                scheduler.finalize_speculative_accept(
                    committed_txn,
                    state_plan,
                    eos_token_id=eos_token_id,
                    stop_token_ids=stop_token_ids,
                )
                self.verifier.finish(prepared)
                prepared = None
                lifecycle_phase("after_target_commit")
            except Exception:
                if prepared is not None:
                    self.verifier.rollback(prepared)
                if _mtp_should_rollback_transaction(
                    target_committed=target_transaction_committed,
                    transaction=plan.transaction,
                ):
                    scheduler.rollback_speculative_kv_transaction(policy, plan)
                raise

            # Retire early at the model EOS or a request stop token: the whole
            # speculative cycle was already recorded, so trim generated tokens
            # to end exactly at the stop and complete through the reclaim path.
            if eos_token_id is not None or stop_token_ids:
                scheduler.finish_request_at_stop(
                    rid,
                    eos_token_id=eos_token_id,
                    stop_token_ids=stop_token_ids,
                )

            summary = commit.summary
            accepted = int(summary.accepted_counts[0])
            if rid not in scheduler.completed:
                proposal_update_started = time.perf_counter()
                draft_tail_advanced = (
                    self.draft_provider.advance_full_accept_tail(
                        rid,
                        accepted_count=accepted,
                    )
                    is not None
                )
                proposal_seconds += time.perf_counter() - proposal_update_started
            lifecycle_phase("after_draft_repair")
            cycle_wall_ms = (time.perf_counter() - cycle_started) * 1000.0
            if budget_policy is not None:
                budget_policy.record_cycle(
                    MtpBudgetCycleResult(
                        cycle=len(accepted_counts) + 1,
                        budget=budget,
                        accepted_count=accepted,
                        visible_tokens=accepted + 1,
                        cycle_wall_ms=cycle_wall_ms,
                        full_accept=bool(summary.full_accept[0]),
                    )
                )
            # Cancellation observed during an in-flight cycle is published only
            # after target commit and draft repair are both terminal. Raising
            # here suppresses the whole non-streaming response without leaving
            # an open transaction or an unrepaired draft cursor.
            _mtp_cycle_checkpoint(checkpoint)
            lifecycle_phase("before_output_publication")
            accepted_counts.append(accepted)
            gpu_match = gpu_match and prepared_gpu_match
            next_token = None if summary.next_tokens is None else summary.next_tokens[0]
            record: dict[str, object] = {
                "cycle": len(accepted_counts),
                "budget": budget,
                "cycle_wall_ms": cycle_wall_ms,
                "root_token": root,
                "root_position": int(work.target_batch.positions[0]),
                "draft_tokens": list(draft.candidate_tokens),
                "target_top1": list(prepared_top1),
                "accepted": accepted,
                "commit_row": int(summary.commit_rows[0]),
                "commit_position": int(summary.commit_positions[0]),
                "next_token": None if next_token is None else int(next_token),
                "span_role": work.target_batch.mode,
                "transaction_id": int(commit.commit_plan.transaction_id),
                "graph_bucket": bucket.owner.spec.bucket,
                "graph_replay_count": int(bucket.replay_count),
                "quant": self.quant,
                "experts_per_token": experts_per_token,
                "target_verify_mode": prepared_verify_mode,
                "target_native_graph_submitted": prepared_native_graph_submitted,
                "target_native_graph_capture_ms": prepared_native_graph_capture_ms,
                "target_native_graph_submit_ms": prepared_native_graph_submit_ms,
                "target_native_graph_readback_ms": prepared_native_graph_readback_ms,
                "target_native_graph_fallback_reason": prepared_native_graph_fallback_reason,
                "device_proposal_fallback_reason": device_proposal_fallback_reason,
                "target_native_device_accept_commit": prepared_native_device_accept_commit,
                "proposal_target_device_chained": bool(
                    device_proposal is not None
                    and prepared_native_device_accept_commit
                ),
                "draft_tail_advanced": draft_tail_advanced,
                "lifecycle_phases": tuple(cycle_phases),
            }
            # ``prepared`` is cleared only after the transaction is fully
            # committed, so use the summary-facing data in compact production
            # records and keep full row logits as an explicit test-only option.
            if return_cycle_logits:
                record["candidate_logits_recorded"] = True
                record["target_logits_shape"] = list(prepared_logits.shape)
            records.append(record)
            lifecycle_phase("after_output_publication")
            record["lifecycle_phases"] = tuple(cycle_phases)
            if rid in scheduler.completed:
                break
            if next_token is None:
                break
            root = int(next_token)

        decode_seconds = time.perf_counter() - decode_started
        if rid not in scheduler.completed:
            raise RuntimeError("GGUF MTP generation ended before scheduler completion")
        return Qwen35GGUFMTPGenerationResult(
            request_id=rid,
            token_ids=scheduler.completed[rid].generated_tokens,
            candidate_budget=self.candidate_budget,
            accepted_counts=tuple(accepted_counts),
            target_forward_rows=target_rows,
            cycles=len(accepted_counts),
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            proposal_seconds=proposal_seconds,
            verify_seconds=verify_seconds,
            gpu_accept_match_cpu=gpu_match,
            graph_stats=scheduler.graph_buckets.stats.to_json_dict(),
            cycle_records=tuple(records),
            budget_policy_summary=(
                None if budget_policy is None else budget_policy.summary()
            ),
        )

    def _prefill_target_and_draft(
        self,
        prompt: tuple[int, ...],
        *,
        request_id: int,
        use_bulk: bool | None,
        checkpoint: Callable[[], None] | None = None,
    ):
        """Stream target prompt chunks into the exact shifted NextN state.

        The target owns the public AR prefill policy and emits completed
        pre-output-norm trunk chunks before their hidden buffers are reused.
        The request sink pairs token 0 with one zero row and token ``i`` with
        target hidden ``i - 1``, retaining only the latest hidden row across
        chunk boundaries.  Draft state appends share the target stream and do
        not score the discarded prompt predictions.
        """

        if self.target.runner is None or self.target.runtime is None:
            raise RuntimeError("GGUF target session is closed")
        executor = self.draft_provider.executor
        policy = getattr(self, "draft_hidden_policy", Qwen35GGUFDraftHiddenPolicy())
        transformed_rows: DeviceBuffer | None = None
        transform_hidden_rows = None
        if policy.requires_target_output_norm:
            hidden_nbytes = int(self.target.runner.hidden_size) * DType.BF16.itemsize
            min_bulk_tokens = int(self.target.runner.weights.config.ssm_conv_kernel)
            run_bulk = len(prompt) >= min_bulk_tokens if use_bulk is None else bool(use_bulk)
            if run_bulk:
                hidden_buffer = self.target._prefill_hidden_a
                if hidden_buffer is None:
                    raise RuntimeError("GGUF target prefill hidden storage is closed")
                capacity = min(len(prompt), int(hidden_buffer.nbytes) // hidden_nbytes)
            else:
                capacity = 1
            transformed_rows = malloc(capacity * hidden_nbytes, runtime=self.target.runtime)

            def _transform(src_ptr: int, rows: int, stream: int) -> int:
                if transformed_rows is None or int(rows) > capacity:
                    raise RuntimeError("draft hidden transform exceeds its chunk capacity")
                self._normalize_target_hidden_rows(
                    src_ptr,
                    transformed_rows.ptr,
                    rows,
                    stream=stream,
                )
                return int(transformed_rows.ptr)

            transform_hidden_rows = _transform
        sink: _StreamingNextNPromptSink | None = None
        completed = False
        try:
            sink = _StreamingNextNPromptSink(
                request_id=int(request_id),
                prompt_tokens=prompt,
                hidden_size=int(self.target.runner.hidden_size),
                executor=executor,
                runtime=self.target.runtime,
                checkpoint=checkpoint,
                transform_hidden_rows=transform_hidden_rows,
            )
            result = self.target.prefill(
                prompt,
                use_bulk=use_bulk,
                return_logits=False,
                target_hidden_chunk_sink=sink,
                target_hidden_request_id=int(request_id),
            )
            completed = True
            return result
        finally:
            finish_prompt_priming = getattr(executor, "finish_prompt_priming", None)
            if callable(finish_prompt_priming):
                finish_prompt_priming(
                    int(request_id),
                    stream=0,
                    synchronize=not completed,
                )
            if sink is not None:
                sink.close()
            if transformed_rows is not None:
                free(transformed_rows, runtime=self.target.runtime)

    def _register_kv_policy(self, request_id: int) -> FixedPagedKVPolicy:
        owner = self.target._target_scratch_owner
        if owner is None:
            raise RuntimeError("GGUF target scratch is closed")
        storage_dtype = DType.parse(owner.kv_storage_dtype)
        scale_metadata = None
        if storage_dtype == DType.INT8_PER_TOKEN_HEAD:
            scale_metadata = next(
                (
                    metadata
                    for metadata in owner.full_kv_scale_metadata
                    if metadata is not None
                ),
                None,
            )
            if scale_metadata is None:
                raise RuntimeError(
                    "GGUF INT8 MTP target has no full-attention scale metadata"
                )
        policy = FixedPagedKVPolicy(
            block_size=int(owner.block_size),
            storage_dtype=storage_dtype,
        )
        policy.register(
            int(request_id),
            block_table=owner.block_table_tensor,
            live_counts=owner.context_tensor,
            max_live_count=int(self.target.position),
            capacity_tokens=int(owner.max_positions),
            scale_metadata=scale_metadata,
        )
        return policy

    def close(self) -> None:
        hidden = getattr(self, "_draft_post_norm_hidden", None)
        if hidden is not None:
            self._draft_post_norm_hidden = None
            free(hidden, runtime=self.target.runtime)
        if self.owns_verifier:
            self.verifier.close()

    def __enter__(self) -> "Qwen35GGUFMTPDecodeSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _copy_array(tensor: Tensor, array: np.ndarray, runtime: HipRuntime | None) -> None:
    contiguous = np.ascontiguousarray(array)
    if contiguous.nbytes != tensor.numel * tensor.dtype.itemsize:
        raise ValueError("host metadata bytes do not match target tensor")
    copy_host_to_device(
        DeviceBuffer(tensor.ptr, contiguous.nbytes),
        host_array_ptr(contiguous),
        contiguous.nbytes,
        runtime=runtime,
    )


def _read_int32(tensor: Tensor, runtime: HipRuntime | None) -> tuple[int, ...]:
    out = np.empty(tensor.shape, dtype=np.int32)
    copy_device_to_host(
        host_array_ptr(out),
        DeviceBuffer(tensor.ptr, out.nbytes),
        out.nbytes,
        runtime=runtime,
    )
    return tuple(int(value) for value in out.reshape(-1).tolist())


def _read_int32_required(tensor: Tensor | None, runtime: HipRuntime | None) -> tuple[int, ...]:
    if tensor is None:
        raise RuntimeError("GGUF GPU accept summary requires next_tokens")
    return _read_int32(tensor, runtime)


def _read_bool_required(tensor: Tensor | None, runtime: HipRuntime | None) -> tuple[bool, ...]:
    if tensor is None:
        raise RuntimeError("GGUF GPU accept summary requires full_accept")
    out = np.empty(tensor.shape, dtype=np.uint8)
    copy_device_to_host(
        host_array_ptr(out),
        DeviceBuffer(tensor.ptr, out.nbytes),
        out.nbytes,
        runtime=runtime,
    )
    return tuple(bool(value) for value in out.reshape(-1).tolist())


def _summary_matches(left: TargetAcceptSummary, right: TargetAcceptSummary) -> bool:
    return (
        left.request_ids == right.request_ids
        and left.accepted_counts == right.accepted_counts
        and left.accepted_tokens == right.accepted_tokens
        and left.commit_rows == right.commit_rows
        and left.commit_tokens == right.commit_tokens
        and left.commit_positions == right.commit_positions
        and left.full_accept == right.full_accept
        and left.next_tokens == right.next_tokens
        and left.transaction_id == right.transaction_id
        and left.mode == right.mode
    )


def _largest_budget_at_most(limit: int) -> int:
    allowed = [budget for budget in _GGUF_MTP_CANDIDATE_BUDGETS if budget <= int(limit)]
    if not allowed:
        raise ValueError("remaining decode budget cannot form an MTP candidate batch")
    return max(allowed)


__all__ = [
    "Qwen35GGUFMTPDecodeSession",
    "Qwen35GGUFMTPGenerationResult",
    "Qwen35GGUFPreparedVerify",
    "Qwen35GGUFTransactionalVerifier",
    "Qwen35GGUFVerifyGraphBucket",
]
