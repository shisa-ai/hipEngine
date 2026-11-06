"""The qualification adapter maps request-local target rows, not result order."""
import ctypes
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import numpy as np

from hipengine.benchmark import device_chain_oracle as oracle
from hipengine.generation.qwen35_gguf_mtp2 import _device_chain_oracle_trace_rows
from hipengine.speculative.interfaces import DraftBatch, TargetVerifyBatch, TargetAcceptSummary


def test_oracle_maps_out_of_order_ragged_request_results(monkeypatch):
    draft = DraftBatch(request_ids=(10, 20), candidate_tokens=(101, 102, 201),
                       parent_positions=(5, 6, 8), draft_depths=(1, 2, 1),
                       row_to_request=(10, 10, 20), tree_parents=(-1, 0, -1),
                       active_mask=(True, True, True))
    batch = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 8))
    local = {10: np.asarray([101, 102, 777], dtype=np.int32),
             20: np.asarray([999, 888], dtype=np.int32)}
    top1 = [0] * batch.rows
    for index, request_id in enumerate(batch.request_ids):
        candidates = sorted((row for row in batch.candidate_rows
                             if batch.row_to_request[row] == request_id),
                            key=lambda row: batch.draft_depths[row])
        for row, value in zip((batch.root_rows[index], *candidates), local[request_id], strict=True):
            top1[row] = int(value)
    summary = TargetAcceptSummary.from_accept_result(batch,
        batch.accept_from_top1(top1, transaction_id=7, remaining_decode=(4, 4)))

    @dataclass(frozen=True)
    class Graph:
        candidate_tokens: tuple
        def to_draft_batch(self):
            return replace(draft, candidate_tokens=self.candidate_tokens)

    frontier = SimpleNamespace(candidate_graph=Graph(()), root_tokens=(100, 200),
                               root_positions=(5, 8), cycle_id=3)
    provider = SimpleNamespace(materialize_batch_device_proposal=lambda proposal:
        SimpleNamespace(active_mask=(True, True, True), draft_rows=3,
                        candidate_tokens=draft.candidate_tokens))
    rows = {request_id: SimpleNamespace(mtp2_device_chain_oracle_trace=[])
            for request_id in batch.request_ids}
    results = [SimpleNamespace(request_id=request_id, rows=len(local[request_id]),
                               target_top1=SimpleNamespace(ptr=request_id))
               for request_id in reversed(batch.request_ids)]
    def copy(destination, source, nbytes, **kwargs):
        ctypes.memmove(destination, local[source.ptr].ctypes.data, nbytes)
    monkeypatch.setattr(oracle, 'copy_device_to_host', copy)
    oracle.qualify_device_accept(SimpleNamespace(_row=rows.__getitem__), frontier,
        object(), results, summary, (4, 4), provider=provider, runtime=None,
        trace_rows=_device_chain_oracle_trace_rows)
    for invalid in (results[:1], [results[0], results[0]]):
        with pytest.raises(RuntimeError, match='target request IDs'):
            oracle.qualify_device_accept(SimpleNamespace(_row=rows.__getitem__), frontier,
                object(), invalid, summary, (4, 4), provider=provider, runtime=None,
                trace_rows=_device_chain_oracle_trace_rows)
    assert rows[10].mtp2_device_chain_oracle_trace[0]['accepted_count'] == 2
    assert rows[20].mtp2_device_chain_oracle_trace[0]['accepted_count'] == 0
    assert rows[10].mtp2_device_chain_oracle_trace[0]['target_top1'] == [101, 102, 777]
    assert rows[20].mtp2_device_chain_oracle_trace[0]['target_top1'] == [999, 888]
