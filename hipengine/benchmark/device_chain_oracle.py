"""Opt-in post-commit CPU comparison for the physical device accept chain."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.speculative.interfaces import TargetVerifyBatch, TargetAcceptSummary


def qualify_device_accept(owner, frontier, proposal, target_results, summary,
                          remaining_decode, *, provider, runtime, trace_rows):
    """Materialize diagnostic evidence only after the device chain commits."""
    started = time.perf_counter()
    materialized = provider.materialize_batch_device_proposal(proposal)
    enabled = materialized.active_mask or (True,) * materialized.draft_rows
    candidate_tokens = tuple(
        int(token) for token, active in zip(materialized.candidate_tokens, enabled, strict=True)
        if bool(active)
    )
    graph = frontier.candidate_graph
    if graph is None:
        raise RuntimeError('device-chain oracle lost its candidate graph')
    host_graph = replace(graph, candidate_tokens=candidate_tokens)
    batch = TargetVerifyBatch.from_draft(
        host_graph.to_draft_batch(), root_tokens=frontier.root_tokens,
        root_positions=frontier.root_positions,
    )
    top1 = [0] * batch.rows
    root_by_id = dict(zip(batch.request_ids, batch.root_rows, strict=True))
    candidate_by_id = {
        request_id: tuple(sorted(
            (row for row in batch.candidate_rows if batch.row_to_request[row] == request_id),
            key=lambda row: batch.draft_depths[row],
        )) for request_id in batch.request_ids
    }
    for result in target_results:
        values = np.empty((int(result.rows),), dtype=np.int32)
        copy_device_to_host(host_array_ptr(values),
                            DeviceBuffer(result.target_top1.ptr, values.nbytes),
                            values.nbytes, runtime=runtime)
        destination_rows = (root_by_id[int(result.request_id)],
                            *candidate_by_id[int(result.request_id)])
        if len(destination_rows) != values.size:
            raise RuntimeError('device-chain oracle target rows changed')
        for row, value in zip(destination_rows, values, strict=True):
            top1[row] = int(value)
    cpu_accept = batch.accept_from_top1(
        tuple(top1), transaction_id=summary.transaction_id,
        remaining_decode=tuple(int(value) for value in remaining_decode),
    )
    cpu_summary = replace(TargetAcceptSummary.from_accept_result(batch, cpu_accept),
                          transaction_id=summary.transaction_id)
    if any(getattr(summary, field) != getattr(cpu_summary, field) for field in (
        'request_ids', 'accepted_counts', 'accepted_tokens', 'commit_rows',
        'commit_tokens', 'commit_positions', 'next_tokens', 'full_accept',
    )):
        raise RuntimeError('physical GPU device-chain accept does not match CPU oracle')
    for trace in trace_rows(batch, top1, summary, cycle_id=frontier.cycle_id):
        row = owner._row(int(trace['request_id']))
        traces = getattr(row, 'mtp2_device_chain_oracle_trace', None)
        if traces is not None and len(traces) < 64:
            traces.append(trace)
    return time.perf_counter() - started
