"""Public-device proof that the Qwen4 MTP step journal restores every owner.

``_advance_mtp`` captures the target's device transaction, the draft cursor,
the row's sampler state and its grammar, then either commits them together or
rolls all of them back.  ``test_unit_qwen4exp_mtp_advance_journal.py`` proves
the orchestration order on host doubles; this module proves the same rollback
against the real device owners, because a snapshot restored in the wrong order,
or dropped before the target commit, is only visible in the bytes.

The oracle reads what the transaction contract owns, not whole buffers.  A
failed step writes K/V, index keys and recurrent state past the cursor it is
about to restore, and those bytes stay in place: storage is append-only and
only the cursor moves back.  So every device comparison here is bounded by the
state's own live cursor -- ``context_host`` for K/V, ``count``/``pooled_count``
for the QSA index, the full buffer for the recurrent decode state, which has no
cursor -- and the paged index rows are gathered through
``physical_positions_host`` so a restored count with an unrestored block mapping
still fails.  Per-step scratch (the last hidden seed, logits staging) is
re-seeded by the next step and is not part of the transaction, so it is not
compared.

Two injections cover the two ways a step ends without committing:

* a cancellation that arrives after the target's accepted rows and the draft
  catch-up have been applied -- every owner must come back, the request must
  publish only the tokens it had before the step, and the runner must be able
  to start and commit another transaction;
* a failure while a later owner is being captured -- the owners captured
  before it must still be restored, no device transaction may be left leased,
  and the step iterator must be closed rather than left suspended.

The second injection is a step failure the engine cannot contain, so it stops
this engine; it is the last test in the module and the fixture is module
scoped.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from hipengine import LLM, SamplingParams
from hipengine.core.memory import (
    DeviceBuffer, copy_device_to_host, host_array_ptr, memory_stats,
)
from hipengine.generation.engine_loop import GenerationExecutionFailed
from hipengine.generation.qwen4_exp_serving_mtp import (
    Qwen4ExpServingMTP, _RequestRetired,
)
from hipengine.runtime.qwen4_exp_mtp import Qwen4ExpGGUFMTPDraftRunner
from tests.test_live_qwen4_exp_mtp_runner import _hip_available

pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")

GGUF_ROOT = Path.home() / "models/gguf"
TARGET_MODEL = GGUF_ROOT / "unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
DRAFT_MODEL = GGUF_ROOT / "Qwen3.8-Flash-Next-MTP-Q8_0/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
PROMPT_TOKEN = 9707
PROMPT_LENGTH = 37


@pytest.fixture(scope="module")
def llm():
    if not TARGET_MODEL.is_dir() or not DRAFT_MODEL.is_file():
        pytest.skip("real Flash-Next target and MTP Q8 sidecar are local-only")
    model = LLM(
        str(TARGET_MODEL), backend="hip_gfx1151", quant="gguf_ud_q4_k_xl",
        execution_profile="production", max_sequence_length=4096,
        max_active_requests=1, prefix_cache="off",
        speculative_provider="qwen4_exp_mtp", draft_model=str(DRAFT_MODEL),
        speculative_candidate_budget=3,
    )
    try:
        yield model
    finally:
        model.close()
        # A rollback that leaked a snapshot or a scratch allocation shows up
        # here rather than as a slow leak across requests.
        stats = memory_stats()
        assert stats["active_allocations"] == 0, stats
        assert stats["current_allocated_bytes"] == 0, stats


def _params():
    # Sampled rather than greedy so the row owns a host sampler state, which is
    # one of the owners the journal has to capture and restore.
    return SamplingParams(
        max_tokens=16, temperature=0.7, seed=17, ignore_eos=True,
        speculative_mtp=True, speculative_mtp_automatic=True,
    )


def _bytes(buffer, nbytes, runtime):
    count = int(nbytes)
    if count <= 0:
        return b""
    array = np.empty(count, dtype=np.uint8)
    copy_device_to_host(host_array_ptr(array), buffer, count, runtime=runtime)
    return array.tobytes()


def _rows(buffer, positions, row_nbytes, runtime):
    """Bytes of the rows a cursor names, in the order it names them."""
    values = []
    for position in positions:
        view = DeviceBuffer(int(buffer.ptr) + int(position) * row_nbytes, row_nbytes)
        values.append(_bytes(view, row_nbytes, runtime))
    return tuple(values)


def _attention_state(attention, runtime):
    """The K/V rows this request may read, and the cursor that bounds them."""
    live = int(attention.context_host[0])
    slots = int(attention.block_host.size) * int(attention.block_size)
    row_nbytes = 0 if slots <= 0 else attention.key_cache.nbytes // slots
    return (
        int(attention.position_host[0]),
        live,
        _bytes(attention.key_cache, live * row_nbytes, runtime),
        _bytes(attention.value_cache, live * row_nbytes, runtime),
    )


def _index_state(index, runtime):
    """The QSA index rows this request owns, gathered through its block map."""
    count = int(index.count)
    pooled = int(index.pooled_count)
    row_nbytes = index.index_dim * 4
    return (
        count,
        pooled,
        _rows(index.raw_keys, index.physical_positions_host[:count], row_nbytes, runtime),
        _bytes(index.pooled_keys, pooled * row_nbytes, runtime),
    )


def _target_state(runner):
    """Everything the target's device transaction owns, plus its paged cursors."""
    runtime = runner.runtime
    runtime.device_synchronize()
    return {
        "position": int(runner.position),
        "ple_hash_states": tuple(
            sorted(runner._ple_hash_states.items(), key=lambda item: int(item[0]))
        ),
        # The recurrent decode state has no cursor: all of it is live.
        "recurrent": {
            name: _bytes(buffer, buffer.nbytes, runtime)
            for name, buffer in runner.state.owned_buffers.items()
        },
        "attention": tuple(
            _attention_state(attention, runtime)
            for attention in runner.attention_states
        ),
        "index": tuple(
            _index_state(index, runtime) for index in runner.index_states
        ),
    }


def _draft_state(draft):
    """The draft cursor the journal snapshots, and the rows it bounds."""
    runtime = draft.runtime
    runtime.device_synchronize()
    return {
        "position": int(draft.position),
        "attention": _attention_state(draft.attention, runtime),
        "index": _index_state(draft.index, runtime),
    }


def _snapshot_state(runner):
    """The bytes of the real device snapshot the rollback restores from."""
    snapshot = runner._device_transaction_snapshot
    if snapshot is None:
        return None
    runtime = runner.runtime
    runtime.device_synchronize()
    return {
        name: _bytes(buffer, buffer.nbytes, runtime)
        for name, buffer in snapshot.buffers.items()
    }


class _CycleProbe:
    """Observe one MTP step, and optionally fail it at a chosen point."""

    def __init__(self):
        self.pending = False
        self.skip = 0
        self.record = None
        self.completed = []
        self.checks = 0
        self.inject = None
        self.fail_capture = False

    def watch(self, *, skip=0):
        """Observe the next step, after letting ``skip`` steps run normally."""
        self.pending = True
        self.skip = skip
        self.record = None
        self.checks = 0


def _instrument(monkeypatch, probe):
    """Wrap the step, its liveness probes and the draft's capture.

    Every injection is gated on ``probe.record``, so it can only fire inside the
    one step being observed -- never during prompt prefill, which also calls the
    liveness probe and the draft's snapshot.
    """
    advance_original = Qwen4ExpServingMTP._advance_mtp
    check_original = Qwen4ExpServingMTP._check_request_live
    snapshot_original = Qwen4ExpGGUFMTPDraftRunner.snapshot

    def check(self, row):
        if probe.record is not None:
            probe.checks += 1
            # The second probe of a step runs after the target's accepted rows
            # and the draft catch-up are applied and before the target commit.
            if probe.checks >= 2 and probe.inject is not None:
                inject, probe.inject = probe.inject, None
                inject()
        return check_original(self, row)

    def snapshot(self):
        if probe.record is not None and probe.fail_capture:
            probe.fail_capture = False
            raise RuntimeError("injected draft state capture failure")
        return snapshot_original(self)

    def advance(self, row, budget, *, plan=None):
        if not probe.pending:
            return advance_original(self, row, budget, plan=plan)
        if probe.skip > 0:
            probe.skip -= 1
            return advance_original(self, row, budget, plan=plan)
        probe.pending = False
        sampling_live = row.sampling_state
        record = probe.record = {
            "generated_before": tuple(row.generated_ids),
            "sampling_live": sampling_live,
            # The journal clones the sampler before the step mutates it, so
            # compare against the same pre-step copy rather than the live
            # object the step advances in place.
            "sampling_before": (
                None if sampling_live is None else sampling_live.clone()
            ),
            "grammar_before": row.grammar,
            "iterator": row.mtp_steps,
            "target_before": _target_state(row.runner),
            "draft_before": _draft_state(row.draft),
            "runner": row.runner,
            "row": row,
        }
        try:
            outcome = advance_original(self, row, budget, plan=plan)
        except BaseException:
            record["committed"] = False
            raise
        else:
            record["committed"] = outcome[0] is not None
            return outcome
        finally:
            record["target_after"] = _target_state(row.runner)
            record["draft_after"] = _draft_state(row.draft)
            record["snapshot_after"] = _snapshot_state(row.runner)
            record["sampling_after"] = row.sampling_state
            record["grammar_after"] = row.grammar
            record["lease_after"] = bool(row.runner._device_transaction_lease)
            record["pending_verify_block"] = row.runner._pending_verify_block
            probe.completed.append(record)
            probe.record = None

    monkeypatch.setattr(Qwen4ExpServingMTP, "_check_request_live", check)
    monkeypatch.setattr(Qwen4ExpServingMTP, "_advance_mtp", advance)
    monkeypatch.setattr(Qwen4ExpGGUFMTPDraftRunner, "snapshot", snapshot)
    return probe


def _retire():
    raise _RequestRetired()


def test_cancelled_step_restores_every_owner_and_the_runner_keeps_serving(llm, monkeypatch):
    probe = _instrument(monkeypatch, _CycleProbe())
    # One normal step first, so the cancellation lands on a request that has
    # already published accepted tokens rather than on its very first step.
    probe.watch(skip=1)
    probe.inject = _retire

    result = llm.generate_detailed([[PROMPT_TOKEN] * PROMPT_LENGTH], _params())[0]

    assert len(probe.completed) == 1, "the single-request step journal did not run"
    record = probe.completed[0]
    assert record["committed"] is False, "the injected step committed anyway"
    # Every captured owner came back, including the sampler state the rejected
    # step drew from.
    assert record["target_after"] == record["target_before"]
    assert record["draft_after"] == record["draft_before"]
    assert record["sampling_live"] is not None
    assert record["sampling_after"] is not record["sampling_live"]
    assert (
        record["sampling_after"]._rng.bit_generator.state
        == record["sampling_before"]._rng.bit_generator.state
    )
    assert record["sampling_after"].rng_draw_count == record["sampling_before"].rng_draw_count
    assert record["sampling_after"].step_index == record["sampling_before"].step_index
    # The live recurrent state is exactly the device snapshot the rollback
    # restored from, not merely equal to itself.
    assert record["snapshot_after"] == record["target_after"]["recurrent"]
    # No verify block is left pending and no transaction is left leased.
    assert record["pending_verify_block"] is None
    assert record["lease_after"] is False
    # The step iterator was closed rather than left suspended.
    assert record["iterator"] is not None
    assert inspect.getgeneratorstate(record["iterator"]) == inspect.GEN_CLOSED
    # The already-committed prefix is published unchanged and nothing the
    # rejected step produced is.
    assert record["generated_before"], "the injected step was the first one"
    assert tuple(result.generated_token_ids) == record["generated_before"]
    assert result.finish_details.cancelled

    # The released lease is re-usable, so the next request can take and commit
    # a transaction on the same runner.
    runner = record["runner"]
    runner.commit_device_transaction(runner.begin_device_transaction(reuse_snapshot=True))
    probe.watch()
    after = llm.generate_detailed([[PROMPT_TOKEN] * PROMPT_LENGTH], _params())[0]
    assert len(after.generated_token_ids) == 16
    assert probe.completed[-1]["committed"] is True


def test_capture_failure_cleans_up_earlier_owners_and_leaves_no_lease(llm, monkeypatch):
    """Runs last: a step failure the engine cannot contain stops this engine."""
    probe = _instrument(monkeypatch, _CycleProbe())
    probe.watch(skip=1)
    probe.fail_capture = True

    with pytest.raises(GenerationExecutionFailed) as raised:
        llm.generate_detailed([[PROMPT_TOKEN] * PROMPT_LENGTH], _params())

    assert "injected draft state capture failure" in str(raised.value)
    assert len(probe.completed) == 1
    record = probe.completed[0]
    # The sampler state was captured before the draft snapshot failed, so the
    # rollback had to restore it even though the step never ran.  Nothing had
    # advanced it yet, so the only trace of the restore is the rebind itself.
    assert record["sampling_live"] is not None
    assert record["sampling_after"] is not record["sampling_live"]
    # The failing capture was the draft's, so no device state was leased or
    # touched, and the target was never snapshotted.
    assert record["target_after"] == record["target_before"]
    assert record["draft_after"] == record["draft_before"]
    assert record["lease_after"] is False
    assert record["iterator"] is not None
    assert inspect.getgeneratorstate(record["iterator"]) == inspect.GEN_CLOSED
    # The runner is still usable: the reusable snapshot can be leased and
    # committed, so the failure did not strand the transaction protocol.
    runner = record["runner"]
    runner.commit_device_transaction(runner.begin_device_transaction(reuse_snapshot=True))
