"""Runner-level ownership and call-site coverage for wide-F16 staging."""

import pytest

from hipengine.runtime.gguf_linear import (
    wide_f16_activation_session,
    wide_f16_activation_workspace,
    wide_f16_pair_diagnostic,
    wide_f16_pair_enabled,
    wide_f16_pair_session,
)
from scripts.qwen4exp_allocation_census import simulated_runner
from scripts.qwen4exp_chunk_workspace import capture_workspace, use_workspace
from tests.test_unit_qwen4exp_scratch_plan import config


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")


@pytest.mark.parametrize("chunk", [257, 1024, 2048, 4096])
def test_owner_is_bounded_reused_and_freed(enabled, chunk):
    from hipengine.loading.qwen4_exp_scratch import qwen4_exp_wide_f16_activation_nbytes

    cfg = config()
    with simulated_runner(cfg, context=4352, chunk=chunk) as (runner, runtime):
        before = dict(runtime.live)
        with runner._wide_f16_activation_context():
            owner = wide_f16_activation_workspace()
            assert owner is not None
            assert owner.nbytes == qwen4_exp_wide_f16_activation_nbytes(
                cfg, context_tokens=4352, prefill_chunk_size=chunk)
            assert owner.nbytes == 2 * chunk * 10_240
            assert runtime.live[owner.ptr] == owner.nbytes
        assert wide_f16_activation_workspace() is None
        assert len(runtime.live) == len(before) + 1
        runner.reset()
        with runner._wide_f16_activation_context():
            assert wide_f16_activation_workspace() == owner
        runner.close()
        assert not runtime.live
        with pytest.raises(RuntimeError):
            with runner._wide_f16_activation_context():
                pass


def test_disabled_context_does_not_allocate_and_shadows_outer(enabled, monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "0")
    with simulated_runner(config(), context=256, chunk=16) as (runner, runtime):
        before = dict(runtime.live)
        with wide_f16_activation_session(workspace_ptr=123, workspace_nbytes=456):
            outer = wide_f16_activation_workspace()
            with runner._wide_f16_activation_context():
                assert wide_f16_activation_workspace() is None
            assert wide_f16_activation_workspace() == outer
        assert runtime.live == before


def test_capacity_clamps_to_context(enabled):
    with simulated_runner(config(), context=256, chunk=4096) as (runner, runtime):
        with runner._wide_f16_activation_context():
            owner = wide_f16_activation_workspace()
            assert owner.nbytes == 2 * 256 * 10_240
            assert runtime.live[owner.ptr] == owner.nbytes


def test_default_allocates_qualified_workspace(enabled):
    with simulated_runner(config(), context=256, chunk=16) as (runner, runtime):
        before = dict(runtime.live)
        with runner._wide_f16_activation_context():
            owner = wide_f16_activation_workspace()
            assert owner is not None
            assert runtime.live[owner.ptr] == 2 * 16 * 10_240
        assert len(runtime.live) == len(before) + 1


def test_allocation_failure_does_not_publish_owner(enabled, monkeypatch):
    with simulated_runner(config(), context=256, chunk=16) as (runner, runtime):
        before = dict(runtime.live)
        def fail(_):
            raise MemoryError("staging allocation")
        monkeypatch.setattr(runtime, "malloc", fail)
        with pytest.raises(MemoryError, match="staging allocation"):
            with runner._wide_f16_activation_context():
                pass
        assert wide_f16_activation_workspace() is None
        assert runtime.live == before


def test_pair_scope_follows_the_runner_decision(enabled):
    """The profile publishes to the runner; the runner binds its own launches."""

    with simulated_runner(config(), context=256, chunk=16) as (runner, runtime):
        assert runner.dense_f16_pair_enabled is False
        with runner._wide_f16_activation_context():
            assert not wide_f16_pair_enabled()
        runner.dense_f16_pair_enabled = True
        with runner._wide_f16_activation_context():
            assert wide_f16_pair_enabled()
            with wide_f16_pair_diagnostic(False):
                assert not wide_f16_pair_enabled()
            assert wide_f16_pair_enabled()
        assert not wide_f16_pair_enabled()


def test_constructing_the_context_binds_nothing_until_it_is_entered(enabled):
    """A context that is built and dropped must not touch the enclosing state."""

    with simulated_runner(config(), context=256, chunk=16) as (runner, runtime):
        runner.dense_f16_pair_enabled = False
        with wide_f16_activation_session(
                workspace_ptr=11, workspace_nbytes=22), wide_f16_pair_session(True):
            outer = wide_f16_activation_workspace()
            pending = runner._wide_f16_activation_context()
            assert wide_f16_activation_workspace() == outer
            assert wide_f16_pair_enabled()
        del pending
        assert wide_f16_activation_workspace() is None
        assert not wide_f16_pair_enabled()


def test_disabled_context_binds_nothing_until_it_is_entered(enabled, monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "0")
    with simulated_runner(config(), context=256, chunk=16) as (runner, runtime):
        before = dict(runtime.live)
        with wide_f16_activation_session(
                workspace_ptr=11, workspace_nbytes=22), wide_f16_pair_session(True):
            outer = wide_f16_activation_workspace()
            pending = runner._wide_f16_activation_context()
            assert wide_f16_activation_workspace() == outer
            assert wide_f16_pair_enabled()
        del pending
        assert runtime.live == before


def test_an_exception_in_the_body_unwinds_workspace_and_pair(enabled):
    with simulated_runner(config(), context=256, chunk=16) as (runner, runtime):
        runner.dense_f16_pair_enabled = True
        with wide_f16_activation_session(
                workspace_ptr=11, workspace_nbytes=22), wide_f16_pair_session(False):
            outer = wide_f16_activation_workspace()
            with pytest.raises(RuntimeError, match="body failed"):
                with runner._wide_f16_activation_context():
                    owned = wide_f16_activation_workspace()
                    assert owned is not None and owned != outer
                    assert wide_f16_pair_enabled()
                    raise RuntimeError("body failed")
            assert wide_f16_activation_workspace() == outer
            assert not wide_f16_pair_enabled()
        assert wide_f16_activation_workspace() is None


def test_disabled_context_binds_parent_even_when_the_runner_bound_the_loan(
    enabled, monkeypatch,
):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "0")
    with simulated_runner(config(), context=256, chunk=16) as (runner, runtime):
        runner.dense_f16_pair_enabled = True
        with runner._wide_f16_activation_context():
            assert wide_f16_activation_workspace() is None
            assert not wide_f16_pair_enabled()
        assert not wide_f16_pair_enabled()


def test_pair_scope_restores_on_a_failing_prefill(enabled, monkeypatch):
    with simulated_runner(config(), context=256, chunk=32) as (runner, runtime):
        runner.dense_f16_pair_enabled = True
        monkeypatch.setattr(runner, "reset", lambda: None)

        def chunk(tokens, **kwargs):
            assert wide_f16_pair_enabled()
            raise RuntimeError("stop before head")

        monkeypatch.setattr(runner, "_prefill_chunk", chunk)
        with pytest.raises(RuntimeError, match="stop before head"):
            runner.prefill([1, 2, 3])
        assert not wide_f16_pair_enabled()


def test_real_prefill_call_site_binds_and_restores_on_failure(enabled, monkeypatch):
    with simulated_runner(config(), context=256, chunk=32) as (runner, runtime):
        monkeypatch.setattr(runner, "reset", lambda: None)
        calls = []
        def chunk(tokens, **kwargs):
            owner = wide_f16_activation_workspace()
            assert owner is not None
            assert runtime.live[owner.ptr] == owner.nbytes
            calls.append(owner)
            raise RuntimeError("stop before head")
        monkeypatch.setattr(runner, "_prefill_chunk", chunk)
        with wide_f16_activation_session(workspace_ptr=123, workspace_nbytes=456):
            outer = wide_f16_activation_workspace()
            with pytest.raises(RuntimeError, match="stop before head"):
                runner.prefill([1, 2, 3])
            assert wide_f16_activation_workspace() == outer
        assert len(calls) == 1

def test_borrowed_smaller_workspace_keeps_original_staging_capacity(enabled, monkeypatch):
    with simulated_runner(config(), context=256, chunk=32) as (runner, runtime):
        small = runner._allocate_extra_prefill_workspace(16)
        monkeypatch.setattr(runner, "reset", lambda: None)
        def chunk(tokens, **kwargs):
            from hipengine.loading.qwen4_exp_scratch import qwen4_exp_wide_f16_activation_nbytes
            assert runner.prefill_chunk_size == 16
            assert wide_f16_activation_workspace().nbytes == qwen4_exp_wide_f16_activation_nbytes(
                runner.config, context_tokens=256, prefill_chunk_size=32)
            raise RuntimeError("stop before head")
        monkeypatch.setattr(runner, "_prefill_chunk", chunk)
        with use_workspace(runner, capture_workspace(small)):
            with pytest.raises(RuntimeError, match="stop before head"):
                runner.prefill([1])
        assert runner.prefill_chunk_size == 32
        assert wide_f16_activation_workspace() is None
