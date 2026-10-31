"""The prepacked-vec4 MMQ child is a prefill-local lane decision, not a switch."""

from contextvars import copy_context
from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear
from hipengine.runtime.gguf_linear import (
    Q8MMQLaneSelection,
    q8_mmq_lane_diagnostic,
    q8_mmq_lane_selection,
    q8_mmq_lane_session,
    q8_mmq_vec4_enabled,
)
from scripts.qwen4exp_allocation_census import simulated_runner
from scripts.qwen4exp_halo_box_campaign_ab import (
    MMQ_PREPACK_ENV,
    MMQ_PREFILL_ENV,
    _apply_mode,
    mmq_vector_control_record,
    mmq_vector_prerequisite,
    q8_mmq_vec4_expected_calls,
    route_scope,
)
from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode
from tests.test_unit_qwen4exp_scratch_plan import config

PREFILL = "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"
RAW_VEC4 = "mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out"
PACKED = "mmq128_prepacked_q8_1_d4x3_guarded_f32_f32_out"
PACKED_VEC4 = "mmq128_prepacked_vec4_q8_1_d4x3_guarded_f32_f32_out"
RETIRED_ENV = "HIPENGINE_QWEN4_EXP_Q8_MMQ_VEC4"


def key(variant=PACKED):
    return KernelKey("hip_gfx1151", "linear", "gguf_q8_0", variant)


def select(monkeypatch, *, rows=512, variant=PACKED, registered=True):
    monkeypatch.setattr(linear, "is_registered", lambda candidate: registered)
    return linear._q8_mmq_vec4_key(key(variant), rows=rows)


class ProvisioningRunner:
    """A runner stand-in whose provisioner honours the env the scope forces."""

    def __init__(self, environment, *, workspace=True, sidecars=True):
        self.environment = environment
        self._workspace = workspace
        self._sidecars = sidecars
        self._q8_mmq_buffers = ()
        self._q8_mmq_weight_sidecars = None
        self.q8_mmq_prefill_enabled = None
        self.dense_mmq_token64_enabled = False
        self.dense_mmq_raw_vector_enabled = True
        self.dense_mmq_vec4_enabled = True
        self.provisions = []

    def configure_mmq_prefill_resources(self):
        self.provisions.append({
            "q8_mmq_prefill_enabled": self.q8_mmq_prefill_enabled,
            MMQ_PREPACK_ENV: self.environment.get(MMQ_PREPACK_ENV),
        })
        if self.q8_mmq_prefill_enabled is True and self._workspace:
            self._q8_mmq_buffers = (object(),)
        if (
            self.q8_mmq_prefill_enabled is True
            and self.environment.get(MMQ_PREPACK_ENV) == "1"
            and self._sidecars
        ):
            self._q8_mmq_weight_sidecars = SimpleNamespace(
                mapping={(1, 2, 3): (4, 5)})


def test_the_unbound_default_is_the_scalar_parent(monkeypatch):
    assert not q8_mmq_vec4_enabled()
    assert q8_mmq_lane_selection() == Q8MMQLaneSelection()
    assert select(monkeypatch) == key()


@pytest.mark.parametrize("rows,remapped", [
    (1, False), (63, False), (64, True), (511, True), (512, True), (4096, True),
])
def test_the_threshold_is_64_rows(monkeypatch, rows, remapped):
    with q8_mmq_lane_session(vec4=True):
        selected = select(monkeypatch, rows=rows)
    assert (selected.variant == PACKED_VEC4) is remapped
    assert selected.variant in {PACKED, PACKED_VEC4}


@pytest.mark.parametrize("variant", [PREFILL, RAW_VEC4])
def test_only_the_prepacked_parent_is_remapped(monkeypatch, variant):
    """The raw-weight path owns the raw children; the mapping owns this one."""

    with q8_mmq_lane_session(vec4=True, raw_vector=True):
        assert select(monkeypatch, variant=variant).variant == variant


def test_an_unregistered_candidate_keeps_the_parent(monkeypatch):
    with q8_mmq_lane_session(vec4=True):
        assert select(monkeypatch, registered=False) == key()


def test_the_lane_binds_and_restores_the_decision():
    with q8_mmq_lane_session(vec4=True):
        assert q8_mmq_vec4_enabled()
        with q8_mmq_lane_session(vec4=False):
            assert not q8_mmq_vec4_enabled()
        assert q8_mmq_vec4_enabled()
    assert not q8_mmq_vec4_enabled()


def test_an_exception_in_the_body_restores_the_enclosing_lane():
    with q8_mmq_lane_session(vec4=True):
        with pytest.raises(RuntimeError, match="chunk failed"):
            with q8_mmq_lane_session(vec4=False):
                assert not q8_mmq_vec4_enabled()
                raise RuntimeError("chunk failed")
        assert q8_mmq_vec4_enabled()
    assert not q8_mmq_vec4_enabled()


def test_the_lane_does_not_cross_contexts():
    outside = copy_context()
    with q8_mmq_lane_session(vec4=True):
        assert q8_mmq_vec4_enabled()
        assert not outside.run(q8_mmq_vec4_enabled)
    assert not q8_mmq_vec4_enabled()


def test_the_diagnostic_outranks_the_request_per_field():
    with q8_mmq_lane_session(vec4=True, raw_vector=True):
        with q8_mmq_lane_diagnostic(vec4=False):
            selection = q8_mmq_lane_selection()
            assert selection.vec4 is False
            assert selection.raw_vector is True
        assert q8_mmq_vec4_enabled()
    with q8_mmq_lane_session():
        with q8_mmq_lane_diagnostic(vec4=True):
            assert q8_mmq_vec4_enabled()
        assert not q8_mmq_vec4_enabled()


def test_the_retired_switch_cannot_enable_the_remap(monkeypatch):
    """The switch is gone, not merely unused: no read may survive it."""

    monkeypatch.setenv(RETIRED_ENV, "1")
    assert not q8_mmq_vec4_enabled()
    assert select(monkeypatch) == key()
    with q8_mmq_lane_session(vec4=True):
        monkeypatch.setenv(RETIRED_ENV, "0")
        assert q8_mmq_vec4_enabled()
        assert select(monkeypatch).variant == PACKED_VEC4


def test_the_campaign_arm_writes_no_process_switch():
    for mode in ("before", "after"):
        env = {"other": "keep"}
        _apply_mode(mode, environment=env, route_package="q8-mmq-vec4")
        assert env == {"other": "keep"}, (mode, env)


def test_the_campaign_arm_binds_parent_then_candidate():
    env = {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "0"}
    owner = ProvisioningRunner(env)
    with route_scope("q8-mmq-vec4", "before", owner, environment=env):
        # The runner binds its published lane inside the harness scope, so the
        # arm's diagnostic composes over it.
        with q8_mmq_lane_session(token64=True, raw_vector=True, vec4=True):
            selection = q8_mmq_lane_selection()
            assert selection.vec4 is False
            # The raw child keeps the plan's published value.
            assert selection.raw_vector is True
            assert selection.token64 is True
        # The prerequisite admits the parent owner and prepares the mapping;
        # the ambient key is inert and unchanged.
        assert owner.q8_mmq_prefill_enabled is True
        assert env[MMQ_PREFILL_ENV] == "0"
        assert env[MMQ_PREPACK_ENV] == "1"
        assert owner._q8_mmq_buffers
        assert owner._q8_mmq_weight_sidecars.mapping
    assert env == {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "0"}
    assert owner.q8_mmq_prefill_enabled is None
    with route_scope("q8-mmq-vec4", "after", owner, environment=env):
        with q8_mmq_lane_session(raw_vector=True, vec4=False):
            selection = q8_mmq_lane_selection()
            assert selection.vec4 is True
            assert selection.raw_vector is True
    assert env == {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "0"}


def test_the_campaign_arm_needs_the_owning_runner():
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope("q8-mmq-vec4", "after")


def test_the_prerequisite_prepares_the_mapping_and_restores_the_switch():
    env = {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "0", "other": "keep"}
    owner = ProvisioningRunner(env)
    with mmq_vector_prerequisite(owner, "q8-mmq-vec4", environment=env) as record:
        assert owner.q8_mmq_prefill_enabled is True
        assert env[MMQ_PREFILL_ENV] == "0"
        assert env[MMQ_PREPACK_ENV] == "1"
        assert record["forced"][MMQ_PREPACK_ENV] == "1"
        assert record["previous"][MMQ_PREPACK_ENV] == "0"
        assert owner._q8_mmq_weight_sidecars.mapping
    assert env == {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "0", "other": "keep"}
    assert owner.q8_mmq_prefill_enabled is None


def test_the_prerequisite_fails_loudly_without_the_mapping():
    env = {MMQ_PREFILL_ENV: "0"}
    owner = ProvisioningRunner(env, sidecars=False)
    with pytest.raises(ValueError, match="prepacked MMQ mapping did not provision"):
        with mmq_vector_prerequisite(owner, "q8-mmq-vec4", environment=env):
            raise AssertionError("the arm must not run without the mapping")
    assert env == {MMQ_PREFILL_ENV: "0"}
    assert owner.q8_mmq_prefill_enabled is None


def test_the_control_record_names_the_field_and_the_prerequisite():
    env = {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "0"}
    owner = ProvisioningRunner(env)
    record = mmq_vector_control_record("q8-mmq-vec4", owner, environment=env)
    assert record["runner_field"] == "dense_mmq_vec4_enabled"
    assert record["before"] is False
    assert record["after"] is True
    assert record["published_value"] is True
    assert record["prerequisite_environment"] == {MMQ_PREPACK_ENV: "1"}
    assert record["prerequisite_owner"] == {
        "q8_mmq_prefill_enabled": True}


def test_the_state_gate_arm_writes_no_process_switch():
    env = {"other": "keep"}
    apply_state_gate_mode(None, "q8-mmq-vec4", "1", "", environment=env)
    assert env == {"other": "keep"}


def test_a_runner_binds_its_own_lane_and_interleaves_safely():
    with simulated_runner(config(), context=256, chunk=16) as (first, _):
        with simulated_runner(config(), context=256, chunk=16) as (second, _):
            assert first.dense_mmq_vec4_enabled is False
            first.dense_mmq_vec4_enabled = True
            with first._q8_mmq_prefill_context():
                assert q8_mmq_vec4_enabled()
                with second._q8_mmq_prefill_context():
                    assert not q8_mmq_vec4_enabled()
                assert q8_mmq_vec4_enabled()
            assert not q8_mmq_vec4_enabled()


def test_the_prefill_call_site_binds_the_runner_lane(monkeypatch):
    with simulated_runner(config(), context=256, chunk=32) as (runner, _):
        runner.dense_mmq_vec4_enabled = True
        monkeypatch.setattr(runner, "reset", lambda: None)
        seen = []

        def chunk(tokens, **kwargs):
            seen.append(q8_mmq_lane_selection())
            raise RuntimeError("stop before head")

        monkeypatch.setattr(runner, "_prefill_chunk", chunk)
        with pytest.raises(RuntimeError, match="stop before head"):
            runner.prefill([1, 2, 3])
        assert seen == [Q8MMQLaneSelection(vec4=True)]
        assert not q8_mmq_vec4_enabled()


@pytest.mark.parametrize("prompt,chunk,calls", [
    (63, 512, 0), (64, 512, 72), (512, 512, 72), (1024, 512, 144),
    (4096, 512, 576), (576, 512, 144), (512, 32, 0),
])
def test_vec4_call_counts(prompt, chunk, calls):
    assert q8_mmq_vec4_expected_calls(prompt, chunk) == calls
