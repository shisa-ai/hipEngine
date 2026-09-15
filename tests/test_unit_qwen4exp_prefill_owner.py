import json
from pathlib import Path

import pytest

from hipengine.loading.qwen4_exp_gguf import qwen4_exp_gguf_config_from_metadata
from hipengine.loading.qwen4_exp_scratch import qwen4_exp_scratch_breakdown
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpPrefillWorkspace
from scripts.qwen4exp_allocation_census import census, simulated_runner
from scripts.qwen4exp_chunk_workspace import capture_workspace, use_workspace
from tests.test_live_qwen4_exp_gguf_config import _info


def config():
    return qwen4_exp_gguf_config_from_metadata(_info())


def test_original_allocation_sequence_matches_frozen_census():
    path = Path(__file__).resolve().parents[1] / (
        "benchmarks/results/2026-09-15-chunk4096-accounting/artifact.json")
    frozen = json.loads(path.read_bytes())["captures"]["resume-scratch-census.json"]
    for row in frozen["records"]:
        actual = census(config(), context=row["context"], chunk=row["chunk"])
        assert actual["allocations"] == row["allocations"]
        assert actual["owner_bytes"] == row["owner_bytes"]


def test_extra_workspace_owns_no_decode_or_kv_state():
    cfg = config()
    with simulated_runner(cfg, context=256, chunk=16) as (runner, runtime):
        before = sum(runtime.live.values())
        primary = runner.gdn_prefill_scratch
        state = runner.state
        extra = runner._allocate_extra_prefill_workspace(32)
        extra.gdn_prefill_scratch.moe.ensure_group_risk_buffers(
            compact_rows=32 * cfg.expert_used_count,
            out_features_total=max(cfg.hidden_size, 2 * cfg.expert_feed_forward_length))
        extra.qsa_prefill_scratch.moe.ensure_group_risk_buffers(
            compact_rows=32 * cfg.expert_used_count,
            out_features_total=max(cfg.hidden_size, 2 * cfg.expert_feed_forward_length))
        plan = qwen4_exp_scratch_breakdown(cfg, context_tokens=256, prefill_chunk_size=32)
        keys = ("gdn_prefill_scratch", "qsa_prefill_scratch", "ple_prefill_scratch",
                "qsa_prefill_metadata", "_prefill_buffers")
        assert sum(runtime.live.values()) - before == sum(plan[key] for key in keys)
        with use_workspace(runner, capture_workspace(extra)):
            assert runner.prefill_chunk_size == 32
            assert runner.gdn_prefill_scratch is extra.gdn_prefill_scratch
            assert runner.state is state
        assert runner.prefill_chunk_size == 16
        assert runner.gdn_prefill_scratch is primary
        assert len(runner._prefill_workspaces) == 2
    assert extra.closed


def test_runner_close_does_not_free_foreign_borrowed_workspace():
    with simulated_runner(config(), context=256, chunk=16) as (runner, runtime):
        foreign = Qwen4ExpPrefillWorkspace.allocate_scratch(runner, 32)
        foreign.allocate_inputs()
        try:
            with use_workspace(runner, capture_workspace(foreign)):
                runner.close()
                assert runner.closed
                assert not foreign.closed
                assert runtime.live
        finally:
            foreign.close()
        assert not runtime.live


def test_extra_workspace_allocation_failure_closes_partial_owner(monkeypatch):
    with simulated_runner(config(), context=256, chunk=16) as (runner, runtime):
        before = dict(runtime.live)
        original = runtime.malloc
        count = 0

        def fail(nbytes):
            nonlocal count
            count += 1
            if count == 10:
                raise MemoryError("injected allocation failure")
            return original(nbytes)

        monkeypatch.setattr(runtime, "malloc", fail)
        with pytest.raises(MemoryError, match="injected"):
            runner._allocate_extra_prefill_workspace(32)
        assert runtime.live == before
        assert len(runner._prefill_workspaces) == 1


def test_extra_input_failure_closes_scratch_and_partial_inputs(monkeypatch):
    with simulated_runner(config(), context=256, chunk=16) as (runner, runtime):
        before = dict(runtime.live)
        allocate_inputs = Qwen4ExpPrefillWorkspace.allocate_inputs
        malloc = runtime.malloc

        def fail_inputs(workspace):
            count = 0

            def fail(nbytes):
                nonlocal count
                count += 1
                if count == 3:
                    raise MemoryError("input failure")
                return malloc(nbytes)

            runtime.malloc = fail
            try:
                return allocate_inputs(workspace)
            finally:
                runtime.malloc = malloc

        monkeypatch.setattr(Qwen4ExpPrefillWorkspace, "allocate_inputs", fail_inputs)
        with pytest.raises(MemoryError, match="input failure"):
            runner._allocate_extra_prefill_workspace(32)
        assert runtime.live == before
        assert len(runner._prefill_buffers) == 5
        assert len(runner._prefill_workspaces) == 1
