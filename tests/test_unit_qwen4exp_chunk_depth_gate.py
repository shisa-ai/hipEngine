from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.qwen4exp_q8_repair_depth_gate import (
    observe_prefill_chunks, validate_chunk_allocation,
)


def allocation():
    return dict(
        schema=2, status="passed", source={"tracked_clean": True},
        chunk_size=2048, prepared_context=4352, prepared_runners=1,
        manifest_sha256="manifest", host={"machine_id": "host"},
        model_identity={"fingerprint": {"value": "weights"}, "revision": "rev"},
        allocation_margins={"scratch_margin_bytes": 1},
        memory_after_close={"current_allocated_bytes": 0},
        lazy_group_risk=[dict(runner_index=0, queues=[
            dict(owner=name, rows=2048, compact_rows=20480, output_width=2560,
                 nbytes=209715204)
            for name in ("gdn_prefill_scratch", "qsa_prefill_scratch")])])


def validate(packet):
    validate_chunk_allocation(
        packet, chunk=2048, context=4168, manifest="manifest",
        host={"machine_id": "host"},
        model={"fingerprint": {"value": "weights"}, "revision": "rev"})


def test_chunk_admission_requires_matching_prepared_evidence():
    validate(allocation())
    for key, value in (("schema", 1), ("status", "failed"), ("chunk_size", 1024),
                       ("prepared_context", 2051), ("manifest_sha256", "other"),
                       ("lazy_group_risk", [])):
        packet = deepcopy(allocation())
        packet[key] = value
        with pytest.raises(ValueError):
            validate(packet)
    packet = allocation()
    packet["allocation_margins"]["scratch_margin_bytes"] = -1
    with pytest.raises(ValueError):
        validate(packet)
    packet = allocation()
    packet["host"]["machine_id"] = "other"
    with pytest.raises(ValueError):
        validate(packet)


def test_chunk_trace_checks_actual_calls_and_restores_method():
    calls = []
    original = lambda tokens, **kw: calls.append((tokens, kw))
    runner = SimpleNamespace(_prefill_chunk=original)
    with observe_prefill_chunks(runner, tokens=5, size=3) as observed:
        runner._prefill_chunk([1, 2, 3], marker=True)
        runner._prefill_chunk([4, 5])
    assert observed == [3, 2]
    assert calls[0][1] == {"marker": True}
    assert runner._prefill_chunk is original
    with pytest.raises(ValueError, match="chunk coverage"):
        with observe_prefill_chunks(runner, tokens=5, size=3):
            runner._prefill_chunk([1, 2, 3, 4, 5])
    assert runner._prefill_chunk is original


def test_chunk_trace_does_not_leave_a_bound_method_override():
    class Runner:
        def _prefill_chunk(self, tokens):
            raise RuntimeError("original failure")

    runner = Runner()
    with pytest.raises(RuntimeError, match="original failure"):
        with observe_prefill_chunks(runner, tokens=3, size=3):
            runner._prefill_chunk([1, 2, 3])
    assert "_prefill_chunk" not in vars(runner)
