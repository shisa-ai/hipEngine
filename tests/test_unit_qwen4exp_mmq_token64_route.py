"""The token64 dense leg is a prefill-local policy, not a process-wide switch."""

from contextvars import copy_context
import os

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear
from hipengine.runtime.gguf_linear import (
    Q8MMQLaneSelection,
    q8_mmq_lane_diagnostic,
    q8_mmq_lane_selection,
    q8_mmq_lane_session,
    q8_mmq_token64_enabled,
)
from scripts.qwen4exp_allocation_census import simulated_runner
from scripts.qwen4exp_halo_box_campaign_ab import (
    MMQ_PREPACK_ENV,
    MMQ_PREFILL_ENV,
    mmq_token64_control_record,
    mmq_token64_expected_calls,
    route_scope,
)
from tests.test_unit_qwen4exp_mmq_raw_vector_route import ProvisioningRunner
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
    with q8_mmq_lane_session(token64=True):
        selected = select(monkeypatch, rows=rows)
    assert (selected.variant == TOKEN64) is remapped
    assert selected.variant in {PARENT, TOKEN64}


@pytest.mark.parametrize("hidden,outputs", [
    (2560, 12288), (2560, 10240), (10240, 320), (1280, 12288), (2560, 12289),
])
def test_only_the_measured_shape_is_remapped(monkeypatch, hidden, outputs):
    with q8_mmq_lane_session(token64=True):
        selected = select(monkeypatch, hidden=hidden, outputs=outputs)
    assert (selected.variant == TOKEN64) is ((hidden, outputs) == (2560, 12288))


@pytest.mark.parametrize("variant", [PREFILL, PACKED])
def test_the_remap_requires_the_raw_vector_parent(monkeypatch, variant):
    """The Q8 raw-vector remap owns the parent variant this leg builds on."""

    with q8_mmq_lane_session(token64=True, raw_vector=True):
        assert select(monkeypatch, variant=variant).variant == variant


def test_the_token64_leg_stays_inert_without_the_raw_vector_decision(monkeypatch):
    """Binding token64 alone may not promote the scalar parent it sits behind."""

    monkeypatch.setattr(linear, "is_registered", lambda candidate: True)
    scalar = parent_key(PREFILL)
    with q8_mmq_lane_session(token64=True, raw_vector=False):
        assert q8_mmq_lane_selection() == Q8MMQLaneSelection(token64=True)
        candidate = linear._q8_mmq_raw_vector_key(scalar, rows=1024)
        assert candidate == scalar
        assert linear._q8_mmq_token64_key(
            candidate, rows=1024, hidden=2560, outputs=12288) == scalar
    with q8_mmq_lane_session(token64=True, raw_vector=True):
        candidate = linear._q8_mmq_raw_vector_key(scalar, rows=1024)
        assert candidate.variant == PARENT
        assert linear._q8_mmq_token64_key(
            candidate, rows=1024, hidden=2560, outputs=12288).variant == TOKEN64


def test_an_unregistered_candidate_keeps_the_parent(monkeypatch):
    with q8_mmq_lane_session(token64=True, raw_vector=True):
        assert select(monkeypatch, registered=False) == parent_key()


def test_the_session_binds_and_restores_the_policy():
    with q8_mmq_lane_session(token64=True):
        assert q8_mmq_token64_enabled()
    assert not q8_mmq_token64_enabled()
    with q8_mmq_lane_session(token64=True):
        with q8_mmq_lane_session(token64=False):
            assert not q8_mmq_token64_enabled()
        assert q8_mmq_token64_enabled()
    assert not q8_mmq_token64_enabled()


def test_an_exception_in_the_body_restores_the_enclosing_policy():
    with q8_mmq_lane_session(token64=True):
        with pytest.raises(RuntimeError, match="chunk failed"):
            with q8_mmq_lane_session(token64=False):
                assert not q8_mmq_token64_enabled()
                raise RuntimeError("chunk failed")
        assert q8_mmq_token64_enabled()
    assert not q8_mmq_token64_enabled()


def test_the_policy_does_not_cross_contexts():
    outside = copy_context()
    with q8_mmq_lane_session(token64=True):
        assert q8_mmq_token64_enabled()
        assert not outside.run(q8_mmq_token64_enabled)
    assert not q8_mmq_token64_enabled()


def test_the_diagnostic_outranks_the_request_and_restores_it():
    with q8_mmq_lane_session(token64=True):
        with q8_mmq_lane_diagnostic(token64=False):
            assert not q8_mmq_token64_enabled()
        assert q8_mmq_token64_enabled()
        with q8_mmq_lane_diagnostic():
            assert q8_mmq_token64_enabled()
    with q8_mmq_lane_session(token64=False):
        with q8_mmq_lane_diagnostic(token64=True):
            assert q8_mmq_token64_enabled()
        assert not q8_mmq_token64_enabled()


def test_the_retired_switch_cannot_enable_the_remap(monkeypatch):
    """The switch is gone, not merely unused: no read may survive it."""

    monkeypatch.setenv(RETIRED_ENV, "1")
    assert not q8_mmq_token64_enabled()
    assert select(monkeypatch) == parent_key()
    with q8_mmq_lane_session(token64=True, raw_vector=True):
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
            runner.prefill([1, 2, 3])
        assert seen == [True]
        assert not q8_mmq_token64_enabled()


def test_a_runner_cloned_after_the_bind_inherits_the_policy():
    from types import SimpleNamespace

    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    generator = SimpleNamespace(dense_mmq_token64_enabled=True,
                                dense_f16_pair_enabled=True,
                                norm_f16_publication_enabled=True,
                                dense_wide_tail_variant="dense_wide256_padded_f16in_f32_f32_out")
    clone = _inherit_lane_decisions(generator, SimpleNamespace())
    assert clone.dense_mmq_token64_enabled is True
    assert clone.dense_f16_pair_enabled is True
    assert clone.dense_wide_tail_variant == generator.dense_wide_tail_variant
    assert clone.norm_f16_publication_enabled is True
    unbound = _inherit_lane_decisions(SimpleNamespace(), SimpleNamespace())
    assert unbound.dense_mmq_token64_enabled is False
    assert unbound.dense_f16_pair_enabled is False
    assert unbound.dense_wide_tail_variant is None
    assert unbound.norm_f16_publication_enabled is False


def test_the_campaign_arm_is_a_scoped_control_not_an_env_write():
    """Both arms name the raw-vec4 parent and admit the workspace it needs."""

    env = {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "1"}
    owner = ProvisioningRunner(env)
    with route_scope("mmq-token64", "after", owner, environment=env):
        selection = q8_mmq_lane_selection()
        assert env == {MMQ_PREFILL_ENV: "1", MMQ_PREPACK_ENV: "0"}
        assert owner.provisions == [{MMQ_PREFILL_ENV: "1", MMQ_PREPACK_ENV: "0"}]
    assert selection.token64 is True
    assert selection.raw_vector is True
    assert env == {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "1"}
    with route_scope("mmq-token64", "before", owner, environment=env):
        selection = q8_mmq_lane_selection()
        assert env == {MMQ_PREFILL_ENV: "1", MMQ_PREPACK_ENV: "0"}
    assert selection.token64 is False
    assert selection.raw_vector is True
    assert env == {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "1"}
    with q8_mmq_lane_diagnostic(vec4=True):
        assert not q8_mmq_token64_enabled()
    assert RETIRED_ENV not in os.environ


def test_the_campaign_arm_needs_the_owning_runner():
    """A runnerless arm would measure the scalar owner at zero calls."""

    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope("mmq-token64", "after")


def test_the_campaign_arm_refuses_a_runner_without_the_provisioner():
    from types import SimpleNamespace

    env = {}
    with pytest.raises(ValueError, match="MMQ provisioner"):
        with route_scope("mmq-token64", "after", SimpleNamespace(),
                         environment=env):
            pass
    assert env == {}


def test_the_campaign_arm_restores_the_switches_when_the_parent_fails():
    env = {MMQ_PREFILL_ENV: "0"}
    owner = ProvisioningRunner(env, workspace=False)
    with pytest.raises(ValueError, match="did not provision a workspace"):
        with route_scope("mmq-token64", "after", owner, environment=env):
            pass
    assert env == {MMQ_PREFILL_ENV: "0"}
    assert MMQ_PREPACK_ENV not in env
    assert not q8_mmq_token64_enabled()


def test_the_campaign_arm_records_the_parent_it_admits():
    env = {MMQ_PREFILL_ENV: "1", MMQ_PREPACK_ENV: "1"}
    owner = ProvisioningRunner(env)
    owner.dense_mmq_token64_enabled = True
    record = mmq_token64_control_record(owner, environment=env)
    assert record["runner_field"] == "dense_mmq_token64_enabled"
    assert record["parent_field"] == "dense_mmq_raw_vector_enabled"
    assert record["parent_bound_value"] is True
    assert record["before"] is False and record["after"] is True
    assert record["published_value"] is True
    assert record["prerequisite_environment"] == {
        MMQ_PREFILL_ENV: "1", MMQ_PREPACK_ENV: "0"}
    assert record["prerequisite_previous"] == {
        MMQ_PREFILL_ENV: "1", MMQ_PREPACK_ENV: "1"}
    assert "zero calls" in record["reason"]


@pytest.mark.parametrize("tokens,expected", [(511, 0), (512, 12), (1024, 12),
                                             (1535, 12), (1536, 24), (4096, 48)])
def test_current_chunk_engagement(tokens, expected):
    assert mmq_token64_expected_calls(tokens, 1024) == expected
