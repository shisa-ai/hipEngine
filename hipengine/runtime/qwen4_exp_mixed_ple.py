"""Mixed-row PLE uses shared projections and depth-ordered history updates.

Verification writes a provisional history slot per physical row. Each root
starts from its request's live history; later depths copy their parent's
history before the indexed convolution. Live history changes only on commit.
The first implementation gathers/scatters wave rows with synchronous device
copies. It does not implement a full target pass or a serving route.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Mapping, Sequence

from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.core.runtime import MemcpyKind
from hipengine.runtime.qwen4_exp_batch_ple import (
    Qwen4ExpBatchPleState,
    run_qwen4_exp_batch_ple_convolution,
    run_qwen4_exp_batch_ple_finish,
    run_qwen4_exp_batch_ple_prepare,
    validate_qwen4_exp_batch_ple_stage,
)
from hipengine.runtime.qwen4_exp_mixed_layout import Qwen4ExpMixedLayout
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpPLEScratch
from hipengine.speculative.interfaces import TargetCommitPlan


class _ScratchOwner:
    """Track each scratch buffer so a failed free never repeats earlier frees."""

    def __init__(self, rows, live):
        self.runtime = live.runtime
        self.buffers = []
        channels = live.channels
        sizes = (channels * 4, live.hidden * 4, channels * 4,
                 live.branches * 4, channels * 4, channels * 4,
                 channels * 4, channels * 2)
        try:
            for size in sizes:
                self.buffers.append(malloc(rows * size, runtime=self.runtime))
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup:
                error.add_note(f'PLE scratch allocation cleanup failed: {cleanup!r}')
            raise
        self.scratch = Qwen4ExpPLEScratch(*self.buffers, runtime=self.runtime)

    def close(self):
        while self.buffers:
            free(self.buffers[-1], runtime=self.runtime)
            self.buffers.pop()
        if hasattr(self, 'scratch'):
            self.scratch.closed = True


class Qwen4ExpMixedPleAdapter:
    """Own provisional PLE history and wave scratch for one target layout."""

    def __init__(self, layout: Qwen4ExpMixedLayout, live: Qwen4ExpBatchPleState,
                 *, owner_of: Callable[[int], int]) -> None:
        layout.require_verifier_compatible()
        live._require_open()
        self.layout, self.live, self.owner_of = layout, live, owner_of
        self.runtime = live.runtime
        self._pending_close = []
        self.closed = self.closing = False
        self.verified = self.committed = self.failed_commit = False
        self._validate_owners()
        try:
            self.history = Qwen4ExpBatchPleState.allocate(
                state_slots=layout.total_rows, branches=live.branches,
                hidden=live.hidden, conv_kernel=live.conv_kernel,
                dilation=live.dilation, runtime=self.runtime,
            )
            self._pending_close.append(self.history)
            # Indexed history admits one depth wave, not all verification rows.
            # Stateless projections still use the full-row scratch below.
            self.history.prepare(tuple(row.physical_row for row in layout.rows if row.depth == 0))
            self._root_active = self.history.active
            scratch = _ScratchOwner(layout.total_rows, live)
            self._pending_close.append(scratch)
            self.scratch = scratch.scratch
            wave = _ScratchOwner(layout.physical_request_width, live)
            self._pending_close.append(wave)
            self.wave = wave.scratch
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup:
                error.add_note(f'PLE adapter allocation cleanup failed: {cleanup!r}')
            raise

    def _validate_owners(self) -> None:
        self.live._require_open()
        for request in self.layout.requests:
            if request.slot >= self.live.state_slots:
                raise ValueError(f'PLE request slot {request.slot} exceeds the live owner')
            if self.owner_of(request.slot) != request.owner_token:
                raise ValueError(f'PLE request {request.request_id} has stale ownership')

    def _require_open(self) -> None:
        if self.closed or self.closing:
            raise RuntimeError('mixed PLE adapter is closed or closing')

    def _copy(self, dst: int, src: int, nbytes: int) -> None:
        if nbytes:
            self.runtime.memcpy(dst, src, nbytes, MemcpyKind.DEVICE_TO_DEVICE)

    def verify(self, residual: DeviceBuffer, embedding: DeviceBuffer,
               weights: Mapping, *, norm_key_ptr: int, norm_query_ptr: int,
               norm_conv_ptr: int, conv_weight_ptr: int) -> DeviceBuffer:
        """Compute all rows without changing the live request histories."""
        self._require_open()
        self._validate_owners()
        if self.committed or self.failed_commit:
            raise RuntimeError('a committed or failed mixed PLE adapter cannot verify again')
        rows, channels = self.layout.total_rows, self.live.channels
        if residual.nbytes < rows * channels * 2:
            raise ValueError('mixed PLE residual buffer is too small')
        if embedding.nbytes < rows * self.live.hidden * 4:
            raise ValueError('mixed PLE embedding buffer is too small')
        self.history.active = self._root_active
        wave_plan = validate_qwen4_exp_batch_ple_stage(
            weights, state=self.history, scratch=self.scratch,
            residual_ptr=residual.ptr, embedding_ptr=embedding.ptr,
            norm_key_ptr=norm_key_ptr, norm_query_ptr=norm_query_ptr,
            norm_conv_ptr=norm_conv_ptr, conv_weight_ptr=conv_weight_ptr,
            rows=self.layout.physical_request_width, branches=self.live.branches, hidden=self.live.hidden,
            conv_kernel=self.live.conv_kernel, dilation=self.live.dilation,
            runtime=self.runtime,
        )
        plan = replace(wave_plan, rows=rows)
        self.verified = False
        run_qwen4_exp_batch_ple_prepare(
            plan, residual.ptr, embedding.ptr, weights, scratch=self.scratch,
            norm_key_ptr=norm_key_ptr, norm_query_ptr=norm_query_ptr,
            norm_conv_ptr=norm_conv_ptr,
        )
        row_bytes = channels * 4
        history_bytes = self.live.history_bytes_per_slot
        for depth in range(self.layout.max_chain_rows):
            wave = tuple(row for row in self.layout.rows if row.depth == depth)
            for index, row in enumerate(wave):
                source = (self.live.history_slot_ptr(row.slot) if depth == 0 else
                          self.history.history_slot_ptr(self.layout.physical_row(row.request_id, depth - 1)))
                self._copy(self.history.history_slot_ptr(row.physical_row), source, history_bytes)
                self._copy(self.wave.normalized.ptr + index * row_bytes,
                           self.scratch.normalized.ptr + row.physical_row * row_bytes, row_bytes)
            self.history.prepare(tuple(row.physical_row for row in wave))
            wave_plan = replace(plan, rows=len(wave), indices_ptr=self.history.indices_ptr)
            run_qwen4_exp_batch_ple_convolution(
                wave_plan, state=self.history, scratch=self.wave,
                conv_weight_ptr=conv_weight_ptr,
            )
            for index, row in enumerate(wave):
                self._copy(self.scratch.conv.ptr + row.physical_row * row_bytes,
                           self.wave.conv.ptr + index * row_bytes, row_bytes)
        output = run_qwen4_exp_batch_ple_finish(plan, residual.ptr, scratch=self.scratch)
        self.verified = True
        return output

    def commit(self, accepted_counts: Sequence[int]) -> TargetCommitPlan:
        """Copy exactly each request's accepted-prefix history to its live slot."""
        self._require_open()
        if not self.verified or self.committed:
            raise RuntimeError('mixed PLE commit requires one uncommitted verification')
        plan = self.layout.commit_destinations(accepted_counts)
        self._validate_owners()
        # Every selected destination is validated before the first copy. A
        # device failure during these copies requires caller-owned containment.
        copies = tuple((self.live.history_slot_ptr(request.slot),
                        self.history.history_slot_ptr(row))
                       for request, row in zip(self.layout.requests, plan.commit_rows, strict=True))
        self.verified = False
        self.failed_commit = True
        for dst, src in copies:
            self._copy(dst, src, self.live.history_bytes_per_slot)
        self.committed = True
        self.failed_commit = False
        return plan

    def close(self) -> None:
        """Release only adapter-owned resources, retrying a failed close safely."""
        if self.closed:
            return
        self.closing = True
        while self._pending_close:
            self._pending_close[-1].close()
            self._pending_close.pop()
        self.closed = True
