"""The token64 dense leg is a prefill-local policy, not a process-wide switch."""

from contextvars import copy_context
import os

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear
from hipengine.runtime.gguf_linear import (
    q8_mmq_token64_diagnostic,
    q8_mmq_token64_enabled,
    q8_mmq_token64_session,
)
from scripts.qwen4exp_allocation_census import simulated_runner
from scripts.qwen4exp_halo_box_campaign_ab import mmq_token64_expected_calls, route_scope
from tests.test_unit_qwen4exp_scratch_plan import config

PARENT = "mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out"
TOKEN64 = "mmq128_token64_q8_1_d4x3_guarded_f32_f32_out"
PREFILL = "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"
PACKED = "mmq128_prepacked_vec4_q8_1_d4x3_guarded_f32_f32_out"
RETIRED_ENV = "HIPENGINE_QWEN4_EXP_MMQ_TOKEN64"


def parent_key(variant=PARENT):
    return KernelKey("hip_gfx1151", "linear", "gguf_q8_0", variant)


def select(monkeypatch, *, rows=1024, hidden=2560, outputs=12288, variant=PARENT,
           registered=True):
    monkeypatch.setattr(linear, "is_registered", lambda candidate: registered)
    return linear._q8_mmq_token64_key(
        parent_key(variant), rows=rows, hidden=hidden, outputs=outputs)


def test_the_unbound_default_is_the_raw_vec4_parent(monkeypatch):
    assert not q8_mmq_token64_enabled()
    assert select(monkeypatch) == parent_key()


@pytest.mark.parametrize("rows,remapped", [
    (1, False), (64, False), (128, False), (511, False),
    (512, True), (777, True), (1024, True), (4096, True),
])
def test_the_threshold_is_512_rows(monkeypatch, rows, remapped):
    with q8_mmq_token64_session():
        selected = select(monkeypatch, rows=rows)
    assert (selected.variant == TOKEN64) is remapped
    assert selected.variant in {PARENT, TOKEN64}


@pytest.mark.parametrize("hidden,outputs", [
    (2560, 12288), (2560, 10240), (10240, 320), (1280, 12288), (2560, 12289),
])
def test_only_the_measured_shape_is_remapped(monkeypatch, hidden, outputs):
    with q8_mmq_token64_session():
        selected = select(monkeypatch, hidden=hidden, outputs=outputs)
    assert (selected.variant == TOKEN64) is ((hidden, outputs) == (2560, 12288))


@pytest.mark.parametrize("variant", [PREFILL, PACKED])
def test_the_remap_requires_the_raw_vector_parent(monkeypatch, variant):
    """The Q8 raw-vector remap owns the parent variant this leg builds on."""

    with q8_mmq_token64_session():
        assert select(monkeypatch, variant=variant).variant == variant


def test_an_unregistered_candidate_keeps_the_parent(monkeypatch):
    with q8_mmq_token64_session():
        assert select(monkeypatch, registered=False) == parent_key()


def test_the_session_binds_and_restores_the_policy():
    with q8_mmq_token64_session():
        assert q8_mmq_token64_enabled()
    assert not q8_mmq_token64_enabled()
    with q8_mmq_token64_session(True):
        with q8_mmq_token64_session(False):
            assert not q8_mmq_token64_enabled()
        assert q8_mmq_token64_enabled()
    assert not q8_mmq_token64_enabled()


def test_an_exception_in_the_body_restores_the_enclosing_policy():
    with q8_mmq_token64_session(True):
        with pytest.raises(RuntimeError, match="chunk failed"):
            with q8_mmq_token64_session(False):
                assert not q8_mmq_token64_enabled()
                raise RuntimeError("chunk failed")
        assert q8_mmq_token64_enabled()
    assert not q8_mmq_token64_enabled()


def test_the_policy_does_not_cross_contexts():
    outside = copy_context()
    with q8_mmq_token64_session(True):
        assert q8_mmq_token64_enabled()
        assert not outside.run(q8_mmq_token64_enabled)
    assert not q8_mmq_token64_enabled()


def test_the_diagnostic_outranks_the_request_and_restores_it():
    with q8_mmq_token64_session(True):
        with q8_mmq_token64_diagnostic(False):
            assert not q8_mmq_token64_enabled()
        assert q8_mmq_token64_enabled()
        with q8_mmq_token64_diagnostic(None):
            assert q8_mmq_token64_enabled()
    with q8_mmq_token64_session(False):
        with q8_mmq_token64_diagnostic(True):
            assert q8_mmq_token64_enabled()
        assert not q8_mmq_token64_enabled()


def test_the_retired_switch_cannot_enable_the_remap(monkeypatch):
    """The switch is gone, not merely unused: no read may survive it."""

    monkeypatch.setenv(RETIRED_ENV, "1")
    assert not q8_mmq_token64_enabled()
    assert select(monkeypatch) == parent_key()
    with q8_mmq_token64_session(True):
        monkeypatch.setenv(RETIRED_ENV, "0")
        assert q8_mmq_token64_enabled()


def test_a_runner_binds_its_own_policy_and_interleaves_safely():
    with simulated_runner(config(), context=256, chunk=16) as (first, _):
        with simulated_runner(config(), context=256, chunk=16) as (second, _):
            assert first.dense_mmq_token64_enabled is False
            first.dense_mmq_token64_enabled = True
            with first._q8_mmq_prefill_context():
                assert q8_mmq_token64_enabled()
                with second._q8_mmq_prefill_context():
                    assert not q8_mmq_token64_enabled()
                assert q8_mmq_token64_enabled()
            assert not q8_mmq_token64_enabled()


def test_the_prefill_call_site_binds_the_runner_policy(monkeypatch):
    with simulated_runner(config(), context=256, chunk=32) as (runner, _):
        runner.dense_mmq_token64_enabled = True
        monkeypatch.setattr(runner, "reset", lambda: None)
        seen = []

        def chunk(tokens, **kwargs):
            seen.append(q8_mmq_token64_enabled())
            raise RuntimeError("stop before head")

        monkeypatch.setattr(runner, "_prefill_chunk", chunk)
        with pytest.raises(RuntimeError, match="stop before head"):
            runner.prefill_chunked([1, 2, 3])
        assert seen == [True]
        assert not q8_mmq_token64_enabled()


def test_a_runner_cloned_after_the_bind_inherits_the_policy():
    from types import SimpleNamespace

    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    generator = SimpleNamespace(dense_mmq_token64_enabled=True,
                                dense_f16_pair_enabled=True)
    clone = _inherit_lane_decisions(generator, SimpleNamespace())
    assert clone.dense_mmq_token64_enabled is True
    assert clone.dense_f16_pair_enabled is True
    unbound = _inherit_lane_decisions(SimpleNamespace(), SimpleNamespace())
    assert unbound.dense_mmq_token64_enabled is False
    assert unbound.dense_f16_pair_enabled is False


def test_the_campaign_arm_is_a_scoped_control_not_an_env_write():
    with route_scope("mmq-token64", "after"):
        assert q8_mmq_token64_enabled()
    with route_scope("mmq-token64", "before"):
        assert not q8_mmq_token64_enabled()
    with route_scope("q8-mmq-vec4", "after"):
        assert not q8_mmq_token64_enabled()
    assert RETIRED_ENV not in os.environ


@pytest.mark.parametrize("tokens,expected", [(511, 0), (512, 12), (1024, 12),
                                             (1535, 12), (1536, 24), (4096, 48)])
def test_current_chunk_engagement(tokens, expected):
    assert mmq_token64_expected_calls(tokens, 1024) == expected
