"""Post-commit qualification preserves every accepted-prefix outcome."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.benchmark import device_chain_oracle as oracle
from hipengine.speculative.interfaces import DraftBatch, TargetVerifyBatch, TargetAcceptSummary


@pytest.mark.parametrize('failure', ('missing_graph', 'target_rows', 'trace_bound'))
def test_qualification_failure_and_trace_bounds(monkeypatch, failure):
    draft = DraftBatch(request_ids=(10,), candidate_tokens=(101,),
                       parent_positions=(5,), draft_depths=(1,),
                       row_to_request=(10,), tree_parents=(-1,), active_mask=(True,))
    batch = TargetVerifyBatch.from_draft(draft, root_tokens=(100,), root_positions=(5,))
    from dataclasses import make_dataclass
    Graph = make_dataclass('Graph', [('candidate_tokens', tuple)])
    Graph.to_draft_batch = lambda self: replace(draft, candidate_tokens=self.candidate_tokens)
    frontier = SimpleNamespace(candidate_graph=None if failure == 'missing_graph' else Graph(()),
                               root_tokens=(100,), root_positions=(5,), cycle_id=3)
    values = np.asarray([101, 777], dtype=np.int32)
    summary = TargetAcceptSummary.from_accept_result(batch,
        batch.accept_from_top1(tuple(values), transaction_id=7, remaining_decode=(4,)))
    provider = SimpleNamespace(materialize_batch_device_proposal=lambda proposal:
        SimpleNamespace(active_mask=(True,), draft_rows=1, candidate_tokens=(101,)))
    result = SimpleNamespace(rows=1 if failure == 'target_rows' else 2,
                             request_id=10, target_top1=SimpleNamespace(ptr=123))
    def copy(destination, source, nbytes, **kwargs):
        import ctypes
        ctypes.memmove(destination, values.ctypes.data, nbytes)
    monkeypatch.setattr(oracle, 'copy_device_to_host', copy)
    row = SimpleNamespace(mtp2_device_chain_oracle_trace=[{} for _ in range(64)])
    kwargs = dict(provider=provider, runtime=None,
                  trace_rows=lambda *args, **kwargs: ({'request_id': 10},))
    owner = SimpleNamespace(_row=lambda request_id: row)
    if failure == 'trace_bound':
        oracle.qualify_device_accept(owner, frontier, object(), [result], summary, (4,), **kwargs)
        assert len(row.mtp2_device_chain_oracle_trace) == 64
    else:
        message = 'lost its candidate graph' if failure == 'missing_graph' else 'target rows changed'
        with pytest.raises(RuntimeError, match=message):
            oracle.qualify_device_accept(owner, frontier, object(), [result], summary, (4,), **kwargs)
        assert len(row.mtp2_device_chain_oracle_trace) == 64


@pytest.mark.parametrize('accepted', (0, 1, 2, 3, 4))
def test_qualification_compares_each_prefix_and_rejects_mismatch(monkeypatch, accepted):
    draft = DraftBatch(request_ids=(10,), candidate_tokens=(101, 102, 103, 104),
                       parent_positions=(5, 6, 7, 8), draft_depths=(1, 2, 3, 4),
                       row_to_request=(10, 10, 10, 10), tree_parents=(-1, 0, 1, 2),
                       active_mask=(True, True, True, True))
    batch = TargetVerifyBatch.from_draft(draft, root_tokens=(100,), root_positions=(5,))
    candidate_rows = sorted(batch.candidate_rows, key=lambda row: batch.draft_depths[row])
    values = np.zeros(batch.rows, dtype=np.int32)
    sequence = tuple(101 + i if i < accepted else 999 for i in range(4)) + (777,)
    for row, value in zip((batch.root_rows[0], *candidate_rows), sequence, strict=True):
        values[row] = value
    summary = TargetAcceptSummary.from_accept_result(
        batch, batch.accept_from_top1(tuple(values), transaction_id=7, remaining_decode=(6,)))
    # replace() requires a dataclass; this tiny graph exposes only the adapter contract.
    from dataclasses import make_dataclass
    Graph = make_dataclass('Graph', [('candidate_tokens', tuple)])
    Graph.to_draft_batch = lambda self: replace(draft, candidate_tokens=self.candidate_tokens)
    frontier = SimpleNamespace(candidate_graph=Graph(()), root_tokens=(100,),
                               root_positions=(5,), cycle_id=3)
    provider = SimpleNamespace(materialize_batch_device_proposal=lambda proposal:
        SimpleNamespace(active_mask=(True,) * 4, draft_rows=4, candidate_tokens=(101, 102, 103, 104)))
    result = SimpleNamespace(rows=5, request_id=10, target_top1=SimpleNamespace(ptr=123))
    row = SimpleNamespace(mtp2_device_chain_oracle_trace=[])
    owner = SimpleNamespace(_row=lambda request_id: row)
    def copy(destination, source, nbytes, **kwargs):
        import ctypes
        ctypes.memmove(destination, values.ctypes.data, nbytes)
    monkeypatch.setattr(oracle, 'copy_device_to_host', copy)
    from hipengine.generation.qwen35_gguf_mtp2 import _device_chain_oracle_trace_rows
    kwargs = dict(provider=provider, runtime=None, trace_rows=_device_chain_oracle_trace_rows)
    assert oracle.qualify_device_accept(owner, frontier, object(), [result], summary, (6,), **kwargs) >= 0
    assert row.mtp2_device_chain_oracle_trace[0]['accepted_count'] == accepted
    with pytest.raises(RuntimeError, match='does not match CPU oracle'):
        oracle.qualify_device_accept(owner, frontier, object(), [result],
            replace(summary, next_tokens=(9999,)), (6,), **kwargs)
