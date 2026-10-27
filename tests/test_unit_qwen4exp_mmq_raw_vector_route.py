"""The raw-vec4 MMQ child is a prefill-local lane decision, not a process switch."""

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
    q8_mmq_raw_vector_enabled,
)
from scripts.qwen4exp_allocation_census import simulated_runner
from scripts.qwen4exp_halo_box_campaign_ab import (
    MMQ_PREPACK_ENV,
    MMQ_PREFILL_ENV,
    _apply_mode,
    mmq_vector_control_record,
    mmq_vector_prerequisite,
    q8_mmq_raw_vector_expected_calls,
    route_scope,
)
from scripts.qwen4exp_row4_state_gate import apply_state_gate_mode
from tests.test_unit_qwen4exp_scratch_plan import config

PREFILL = "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"
RAW_VEC4 = "mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out"
PACKED = "mmq128_prepacked_q8_1_d4x3_guarded_f32_f32_out"
PACKED_VEC4 = "mmq128_prepacked_vec4_q8_1_d4x3_guarded_f32_f32_out"
TOKEN64 = "mmq128_token64_q8_1_d4x3_guarded_f32_f32_out"
RETIRED_ENV = "HIPENGINE_QWEN4_EXP_Q8_MMQ_RAW_VECTOR"


def key(variant=PREFILL):
    return KernelKey("hip_gfx1151", "linear", "gguf_q8_0", variant)


def select(monkeypatch, *, rows=512, variant=PREFILL, registered=True):
    monkeypatch.setattr(linear, "is_registered", lambda candidate: registered)
    return linear._q8_mmq_raw_vector_key(key(variant), rows=rows)


class ProvisioningRunner:
    """A runner stand-in whose provisioner honours the env the scope forces."""

    def __init__(self, environment, *, workspace=True, sidecars=True):
        self.environment = environment
        self._workspace = workspace
        self._sidecars = sidecars
        self._q8_mmq_buffers = ()
        self._q8_mmq_weight_sidecars = None
        self.dense_mmq_token64_enabled = False
        self.dense_mmq_raw_vector_enabled = True
        self.dense_mmq_vec4_enabled = True
        self.provisions = []

    def configure_mmq_prefill_resources(self):
        self.provisions.append(dict(self.environment))
        if self.environment.get(MMQ_PREFILL_ENV) == "1" and self._workspace:
            self._q8_mmq_buffers = (object(),)
        if (
            self.environment.get(MMQ_PREFILL_ENV) == "1"
            and self.environment.get(MMQ_PREPACK_ENV) == "1"
            and self._sidecars
        ):
            self._q8_mmq_weight_sidecars = SimpleNamespace(
                mapping={(1, 2, 3): (4, 5)})


def test_the_unbound_default_is_the_scalar_parent(monkeypatch):
    assert not q8_mmq_raw_vector_enabled()
    assert q8_mmq_lane_selection() == Q8MMQLaneSelection()
    assert select(monkeypatch) == key()


@pytest.mark.parametrize("rows,remapped", [
    (1, False), (63, False), (64, True), (511, True), (512, True), (4096, True),
])
def test_the_threshold_is_64_rows(monkeypatch, rows, remapped):
    with q8_mmq_lane_session(raw_vector=True):
        selected = select(monkeypatch, rows=rows)
    assert (selected.variant == RAW_VEC4) is remapped
    assert selected.variant in {PREFILL, RAW_VEC4}


@pytest.mark.parametrize("variant", [PACKED, PACKED_VEC4, TOKEN64])
def test_only_the_raw_parent_is_remapped(monkeypatch, variant):
    """The prepacked and token64 children own their own parent variants."""

    with q8_mmq_lane_session(raw_vector=True, vec4=True, token64=True):
        assert select(monkeypatch, variant=variant).variant == variant


def test_an_unregistered_candidate_keeps_the_parent(monkeypatch):
    with q8_mmq_lane_session(raw_vector=True):
        assert select(monkeypatch, registered=False) == key()


def test_the_lane_binds_and_restores_every_decision():
    with q8_mmq_lane_session(raw_vector=True):
        assert q8_mmq_lane_selection() == Q8MMQLaneSelection(raw_vector=True)
        with q8_mmq_lane_session(token64=True, vec4=True):
            assert q8_mmq_lane_selection() == Q8MMQLaneSelection(
                token64=True, raw_vector=False, vec4=True)
        assert q8_mmq_lane_selection() == Q8MMQLaneSelection(raw_vector=True)
    assert q8_mmq_lane_selection() == Q8MMQLaneSelection()


def test_an_exception_in_the_body_restores_the_enclosing_lane():
    with q8_mmq_lane_session(raw_vector=True):
        with pytest.raises(RuntimeError, match="chunk failed"):
            with q8_mmq_lane_session(raw_vector=False):
                assert not q8_mmq_raw_vector_enabled()
                raise RuntimeError("chunk failed")
        assert q8_mmq_raw_vector_enabled()
    assert not q8_mmq_raw_vector_enabled()


def test_the_lane_does_not_cross_contexts():
    outside = copy_context()
    with q8_mmq_lane_session(raw_vector=True):
        assert q8_mmq_raw_vector_enabled()
        assert not outside.run(q8_mmq_raw_vector_enabled)
    assert not q8_mmq_raw_vector_enabled()


def test_the_diagnostic_outranks_the_request_per_field():
    with q8_mmq_lane_session(raw_vector=True, vec4=True):
        with q8_mmq_lane_diagnostic(raw_vector=False):
            selection = q8_mmq_lane_selection()
            assert selection.raw_vector is False
            # The fields the diagnostic does not name keep the request's value.
            assert selection.vec4 is True
            with q8_mmq_lane_diagnostic(vec4=False):
                selection = q8_mmq_lane_selection()
                assert selection.raw_vector is False
                assert selection.vec4 is False
            assert q8_mmq_lane_selection().vec4 is True
        assert q8_mmq_lane_selection() == Q8MMQLaneSelection(
            raw_vector=True, vec4=True)
        with q8_mmq_lane_diagnostic():
            assert q8_mmq_lane_selection().raw_vector is True
    with q8_mmq_lane_session():
        with q8_mmq_lane_diagnostic(raw_vector=True):
            assert q8_mmq_raw_vector_enabled()
        assert not q8_mmq_raw_vector_enabled()


def test_the_retired_switch_cannot_enable_the_remap(monkeypatch):
    """The switch is gone, not merely unused: no read may survive it."""

    monkeypatch.setenv(RETIRED_ENV, "1")
    assert not q8_mmq_raw_vector_enabled()
    assert select(monkeypatch) == key()
    with q8_mmq_lane_session(raw_vector=True):
        monkeypatch.setenv(RETIRED_ENV, "0")
        assert q8_mmq_raw_vector_enabled()
        assert select(monkeypatch).variant == RAW_VEC4


def test_the_campaign_arm_writes_no_process_switch():
    for mode in ("before", "after"):
        env = {"other": "keep"}
        _apply_mode(mode, environment=env, route_package="q8-mmq-raw-vector")
        assert env == {"other": "keep"}, (mode, env)


def test_the_campaign_arm_binds_parent_then_candidate():
    env = {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "1"}
    owner = ProvisioningRunner(env)
    with route_scope("q8-mmq-raw-vector", "before", owner, environment=env):
        # The runner binds its published lane inside the harness scope, so the
        # arm's diagnostic composes over it.
        with q8_mmq_lane_session(token64=True, raw_vector=True, vec4=True):
            selection = q8_mmq_lane_selection()
            assert selection.raw_vector is False
            # The prepacked sibling keeps the plan's published value; the
            # token64 sibling is pinned off because it remaps this arm's own
            # raw-vec4 parent.
            assert selection.vec4 is True
            assert selection.token64 is False
        # The prerequisite admits the parent and leaves no prepacked mapping.
        assert env[MMQ_PREFILL_ENV] == "1"
        assert env[MMQ_PREPACK_ENV] == "0"
        assert owner._q8_mmq_buffers
        assert owner._q8_mmq_weight_sidecars is None
    assert env == {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "1"}
    with route_scope("q8-mmq-raw-vector", "after", owner, environment=env):
        with q8_mmq_lane_session(token64=True, raw_vector=False, vec4=True):
            selection = q8_mmq_lane_selection()
            assert selection.raw_vector is True
            assert selection.vec4 is True
            assert selection.token64 is False
    assert env == {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "1"}


def test_the_campaign_arm_needs_the_owning_runner():
    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope("q8-mmq-raw-vector", "after")


def test_the_admitted_parent_reaches_the_raw_and_prepacked_children(monkeypatch):
    """The supported public diagnostic: admit the parent, then both children.

    This is the composition the runner performs for a production bind whose
    ``HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL`` was turned back on after the binder
    and whose resources were provisioned: the lane comes from the published
    runner fields, the arm overrides one field, and both children remain
    selectable on their own parent variants.
    """

    monkeypatch.setattr(linear, "is_registered", lambda candidate: True)
    env = {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "0"}
    owner = ProvisioningRunner(env)
    with route_scope("q8-mmq-raw-vector", "after", owner, environment=env):
        with q8_mmq_lane_session(
            token64=owner.dense_mmq_token64_enabled,
            raw_vector=owner.dense_mmq_raw_vector_enabled,
            vec4=owner.dense_mmq_vec4_enabled,
        ):
            assert linear._q8_mmq_raw_vector_key(
                key(PREFILL), rows=1024).variant == RAW_VEC4
            assert linear._q8_mmq_vec4_key(
                key(PACKED), rows=1024).variant == PACKED_VEC4


def test_the_prerequisite_admits_provisions_and_restores_the_parent():
    env = {MMQ_PREFILL_ENV: "0", "other": "keep"}
    owner = ProvisioningRunner(env)
    with mmq_vector_prerequisite(owner, "q8-mmq-raw-vector", environment=env) as record:
        assert env[MMQ_PREFILL_ENV] == "1"
        assert record["forced"][MMQ_PREFILL_ENV] == "1"
        assert record["previous"][MMQ_PREFILL_ENV] == "0"
        assert owner._q8_mmq_buffers
    assert env == {MMQ_PREFILL_ENV: "0", "other": "keep"}
    assert owner._q8_mmq_buffers, "the workspace stays runner-owned after the arm"


def test_the_prerequisite_fails_loudly_without_a_workspace():
    env = {MMQ_PREFILL_ENV: "0"}
    owner = ProvisioningRunner(env, workspace=False)
    with pytest.raises(ValueError, match="did not provision a workspace"):
        with mmq_vector_prerequisite(owner, "q8-mmq-raw-vector", environment=env):
            raise AssertionError("the arm must not run without the parent")
    assert env == {MMQ_PREFILL_ENV: "0"}


def test_the_control_record_names_the_field_and_the_prerequisite():
    env = {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "1"}
    owner = ProvisioningRunner(env)
    record = mmq_vector_control_record(
        "q8-mmq-raw-vector", owner, environment=env)
    assert record["runner_field"] == "dense_mmq_raw_vector_enabled"
    assert record["before"] is False
    assert record["after"] is True
    assert record["published_value"] is True
    assert record["prerequisite_environment"] == {
        MMQ_PREFILL_ENV: "1", MMQ_PREPACK_ENV: "0"}
    assert record["prerequisite_previous"] == {
        MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "1"}
    assert record["scope"] == "per measured run; restored on exit"


def test_the_raw_arm_pins_the_token64_leg_off_when_the_plan_enables_it(monkeypatch):
    """A production runner publishes token64; this arm may not inherit it.

    The token64 leg remaps a K2560/N12288 raw-vec4 parent onto its own owner,
    so an inherited decision would count a different kernel inside this arm's
    raw-vec4 window and would replace the scalar teacher the before arm
    measures.
    """

    monkeypatch.setattr(linear, "is_registered", lambda candidate: True)
    env = {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "0"}
    owner = ProvisioningRunner(env)
    owner.dense_mmq_token64_enabled = True
    matching = key(RAW_VEC4)

    def remapped():
        return linear._q8_mmq_token64_key(
            matching, rows=1024, hidden=2560, outputs=12288).variant

    with q8_mmq_lane_session(token64=True, raw_vector=True):
        # The plan's own lane would move the matching parent off raw vec4.
        assert remapped() == TOKEN64
    for mode in ("before", "after"):
        with route_scope("q8-mmq-raw-vector", mode, owner, environment=env):
            with q8_mmq_lane_session(
                token64=owner.dense_mmq_token64_enabled,
                raw_vector=owner.dense_mmq_raw_vector_enabled,
                vec4=owner.dense_mmq_vec4_enabled,
            ):
                assert remapped() == RAW_VEC4, mode
                assert select(monkeypatch, rows=1024).variant == (
                    RAW_VEC4 if mode == "after" else PREFILL), mode
    assert env == {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "0"}


def test_the_raw_arm_records_the_token64_pin_but_the_vec4_arm_does_not():
    env = {MMQ_PREFILL_ENV: "0", MMQ_PREPACK_ENV: "1"}
    owner = ProvisioningRunner(env)
    owner.dense_mmq_token64_enabled = True
    pin = mmq_vector_control_record(
        "q8-mmq-raw-vector", owner, environment=env)["sibling_pins"]
    assert pin["dense_mmq_token64_enabled"]["bound_value"] is False
    assert pin["dense_mmq_token64_enabled"]["published_value"] is True
    assert "token64" in pin["dense_mmq_token64_enabled"]["reason"]
    # The vec4 arm counts prepared weights, which the token64 leg never remaps,
    # so it keeps the plan's value and records no pin.
    assert "sibling_pins" not in mmq_vector_control_record(
        "q8-mmq-vec4", owner, environment=env)


def test_the_state_gate_arm_writes_no_process_switch():
    env = {"other": "keep"}
    apply_state_gate_mode(
        None, "q8-mmq-raw-vector", "1", "", environment=env)
    assert env == {"other": "keep"}


def test_a_runner_binds_its_own_lane_and_interleaves_safely():
    with simulated_runner(config(), context=256, chunk=16) as (first, _):
        with simulated_runner(config(), context=256, chunk=16) as (second, _):
            assert first.dense_mmq_raw_vector_enabled is False
            first.dense_mmq_raw_vector_enabled = True
            with first._q8_mmq_prefill_context():
                assert q8_mmq_raw_vector_enabled()
                with second._q8_mmq_prefill_context():
                    assert not q8_mmq_raw_vector_enabled()
                assert q8_mmq_raw_vector_enabled()
            assert not q8_mmq_raw_vector_enabled()


def test_the_prefill_call_site_binds_the_runner_lane(monkeypatch):
    with simulated_runner(config(), context=256, chunk=32) as (runner, _):
        runner.dense_mmq_raw_vector_enabled = True
        monkeypatch.setattr(runner, "reset", lambda: None)
        seen = []

        def chunk(tokens, **kwargs):
            seen.append(q8_mmq_lane_selection())
            raise RuntimeError("stop before head")

        monkeypatch.setattr(runner, "_prefill_chunk", chunk)
        with pytest.raises(RuntimeError, match="stop before head"):
            runner.prefill([1, 2, 3])
        assert seen == [Q8MMQLaneSelection(raw_vector=True)]
        assert not q8_mmq_raw_vector_enabled()


def test_a_runner_cloned_after_the_bind_inherits_the_lane():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    generator = SimpleNamespace(
        dense_mmq_raw_vector_enabled=True,
        dense_mmq_vec4_enabled=True,
        dense_mmq_token64_enabled=True,
        dense_f16_pair_enabled=True,
    )
    clone = _inherit_lane_decisions(generator, SimpleNamespace())
    assert clone.dense_mmq_raw_vector_enabled is True
    assert clone.dense_mmq_vec4_enabled is True
    assert clone.dense_mmq_token64_enabled is True
    assert clone.dense_f16_pair_enabled is True
    unbound = _inherit_lane_decisions(SimpleNamespace(), SimpleNamespace())
    assert unbound.dense_mmq_raw_vector_enabled is False
    assert unbound.dense_mmq_vec4_enabled is False


@pytest.mark.parametrize("prompt,chunk,expected", [
    (512, 512, 242), (1024, 512, 484), (4096, 512, 1936), (63, 512, 0),
    (576, 512, 484), (512, 32, 0),
])
def test_raw_vector_counts(prompt, chunk, expected):
    assert q8_mmq_raw_vector_expected_calls(prompt, chunk) == expected
