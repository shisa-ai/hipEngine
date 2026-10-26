"""Request-owned row storage for a deterministic MTP subset proposal head."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer, copy_host_array_to_device, free, malloc
from hipengine.core.tensor import Tensor
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_F32, GGUF_OUTPUT_F32, launch_gguf_linear,
)
from hipengine.speculative.draft_vocab import (
    DEFAULT_EXTRA_CAPACITY, DEFAULT_PREFIX_SIZE, MTPDraftVocabulary,
)


class MTPSubsetHead:
    """Score a prefix in place and a request's extra rows in a reusable table.

    Only committed history changes the table. Captured full logits keep the
    full head; compact proposals use the table only after ``prepare`` admits
    their cycle. The target distribution and verifier are not changed here.
    """

    def __init__(self, weight, *, vocab_size, hidden_size, special_ids, runtime,
                 prefix_size=DEFAULT_PREFIX_SIZE, extra_capacity=DEFAULT_EXTRA_CAPACITY):
        self.weight = weight
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.runtime = runtime
        self.special_ids = tuple(special_ids)
        self.policy = None
        self.extra_buffer = self.extra_ids_buffer = self.extra_weight = None
        self._buffers = []
        self._unavailable = None
        self.reset()
        if self.vocab_size <= prefix_size:
            self._unavailable = 'full_vocabulary_within_prefix'
        elif weight.spec.layout != 'raw_gguf' or 'raw' not in weight.allocations:
            self._unavailable = 'head_layout_not_row_addressable'
        elif len(set(t for t in special_ids if t >= prefix_size)) > extra_capacity:
            self._unavailable = 'special_table_exceeds_capacity'
        if self._unavailable:
            return
        raw = weight.allocation('raw')
        self.row_bytes, remainder = divmod(raw.buffer.nbytes, self.vocab_size)
        if remainder or self.row_bytes <= 0:
            self._unavailable = 'head_rows_not_contiguous'
            return
        self.policy = MTPDraftVocabulary(self.vocab_size, prefix_size=prefix_size,
                                        extra_capacity=extra_capacity, special_ids=special_ids)
        # The existing pack8 projection keeps its full-head reduction on an
        # eight-row-aligned table. Padded outputs never enter argmax.
        padded_capacity = ((extra_capacity + 7) // 8) * 8
        needed = padded_capacity * self.row_bytes + extra_capacity * 8
        if needed > runtime.mem_get_info()[0]:
            self._unavailable = 'insufficient_memory_for_extra_rows'
            self.policy = None
            return
        try:
            if extra_capacity:
                self.extra_buffer = malloc(padded_capacity * self.row_bytes, runtime=runtime)
                self._buffers.append(self.extra_buffer)
                self.extra_ids_buffer = malloc(extra_capacity * 8, runtime=runtime)
                self._buffers.append(self.extra_ids_buffer)
                runtime.memset(self.extra_buffer.ptr, 0, self.extra_buffer.nbytes)
                view = replace(raw, buffer=self.extra_buffer, owns_buffer=False,
                               tensor=Tensor.from_handle(self.extra_buffer.ptr,
                                   (self.extra_buffer.nbytes,), DType.INT8, raw.tensor.device))
                self.extra_weight = replace(weight, allocations={'raw': view})
                self._upload(self.policy.extra_ids, 0)
        except BaseException:
            self.close()
            raise

    def reset(self):
        if self.policy is not None:
            self.policy.reset()
        self._active = False
        self._reason = 'request_not_prepared'
        self._subset_calls = self._full_calls = 0
        self._full_reasons = {}

    def _upload(self, ids, first):
        if not ids:
            return
        source = self.weight.allocation('raw').tensor.ptr
        for offset, token in enumerate(ids, first):
            self.runtime.memcpy(self.extra_buffer.ptr + offset * self.row_bytes,
                                source + token * self.row_bytes, self.row_bytes,
                                HipMemcpyKind.DEVICE_TO_DEVICE)
        host = np.asarray(ids, dtype=np.int64)
        copy_host_array_to_device(
            DeviceBuffer(self.extra_ids_buffer.ptr + first * 8, host.nbytes),
            host,
            host.nbytes,
            runtime=self.runtime,
        )

    def prepare(self, committed, *, anchor):
        """Select one cycle using only this request's append-only history."""
        self._active = False
        if self._unavailable:
            self._reason = self._unavailable
            return
        first = len(self.policy.extra_ids)
        added = self.policy.catch_up(committed)
        self._upload(added, first)
        count = len(self.policy.extra_ids)
        padded = ((count + 7) // 8) * 8
        if self.policy.prefix_size + padded >= self.vocab_size:
            self._reason = 'subset_not_smaller'
        elif not self.policy.fits(anchor):
            self._reason = 'recent_vocabulary_misses'
        else:
            self._active = True
            self._reason = None

    def project(self, x_ptr, out_ptr, *, capture_logits=False):
        """Return the number of valid scores written, excluding table padding."""
        common = dict(activation_dtype=GGUF_ACTIVATION_F32,
                      output_dtype=GGUF_OUTPUT_F32, runtime=self.runtime)
        if not self._active or capture_logits:
            reason = 'logits_requested' if capture_logits else self._reason
            self._full_calls += 1
            self._full_reasons[reason] = self._full_reasons.get(reason, 0) + 1
            launch_gguf_linear(self.weight, x_ptr, out_ptr, 1, self.hidden_size,
                               self.vocab_size, **common)
            return self.vocab_size
        prefix = self.policy.prefix_size
        count = len(self.policy.extra_ids)
        if prefix:
            launch_gguf_linear(self.weight, x_ptr, out_ptr, 1, self.hidden_size, prefix, **common)
        if count:
            # Narrow pack8 projections otherwise select 256 threads, changing
            # the reduction. The large full head uses the 128-thread parent.
            launch_gguf_linear(self.extra_weight, x_ptr, out_ptr + prefix * 4,
                               1, self.hidden_size, ((count + 7) // 8) * 8,
                               threads=128, **common)
        self._subset_calls += 1
        return prefix + count

    def remap(self, index_ptr):
        from hipengine.kernels.registry import resolve
        fn = resolve(backend=self.weight.backend, layer='mtp_draft_token_index', quant='w4_paro',
                     variant='remap_prefix_tail_i64')
        fn(index_ptr, 0 if self.extra_ids_buffer is None else self.extra_ids_buffer.ptr,
           self.policy.prefix_size, len(self.policy.extra_ids), runtime=self.runtime)

    def diagnostics(self):
        return {
            'mode': 'request_local_prefix_and_extras',
            'prefix_size': min(DEFAULT_PREFIX_SIZE, self.vocab_size) if self.policy is None else self.policy.prefix_size,
            'extra_rows': 0 if self.policy is None else len(self.policy.extra_ids),
            'scanned_tokens': 0 if self.policy is None else self.policy.scanned,
            'subset_calls': self._subset_calls, 'full_calls': self._full_calls,
            'full_reasons': dict(self._full_reasons),
            'sampling': 'deterministic_proposals_target_verified',
        }

    def close(self):
        for buffer in reversed(self._buffers):
            free(buffer, runtime=self.runtime)
        self._buffers.clear()
