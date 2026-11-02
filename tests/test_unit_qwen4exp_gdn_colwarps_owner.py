"""The GDN columnwarp enablement gate is owner state, not a process key.

``HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL`` is the gate the production GDN
prefill default runs inside. The profile binder writes it, the token mixer
reads it at dispatch, and twelve scripts read or write it to build serial and
tiled arms. That makes a plan decision into process-wide state: an arm that
only writes the key measures whatever the binder happened to leave there, and
a caller-set value before construction is silently discarded.

The migration publishes the decision as ``gdn_colwarps_prefill`` on the owning
generator and runner, forwards it through the prefill chunk and the GDN layer,
and leaves the launch reading only that field:

* An unbound owner, the strict binder and every plan that falls back to it
  publish ``False``, so the serial strict owners run and the columnwarp parent
  is not entered.
* A production bind publishes ``True`` today; the value is derived from the
  arithmetic-recovery decision, so a recovery entry for this flag would
  publish ``False`` on the lane that pass covered — the production UD plan —
  and leave any other production plan open.
* The layer scope is separate owner state: enablement decides *whether* the
  branch runs, the layer list decides *which layers* it serves, and tile-16
  admission still outranks both inside the branch.
* The serial strict route, the peer owner and the register parent keep their
  precedence below the gate, and decode never reaches any of it.

What these tests pin:

* **Selection.** The gate is geometry plus the published decision; an unbound
  owner never reads the environment.
* **Scope.** ``gdn_colwarps_prefill(owner, enabled)`` binds a runner or a
  generator plus its runner, restores the published value on exit including
  after a raise, and stays per-owner.
* **Publication.** Production publishes on, strict publishes off, no bind
  writes the retired process key, the layer key is still written, a recovery
  entry controls the value, and a caller-set ambient value is reported once
  and stays inert.
* **Dispatch.** A real (faked-launch) token-mixer run selects the tiled branch,
  the columnwarp parent, the peer owner, the register parent or the serial
  owner from the published decision, keeps the layer scope distinct, never
  reads the retired key, and never consults the gate on a decode row.
* **Forwarding.** The owner reaches the mixer through the GDN layer and the
  prefill chunk path, the runner's default is off, and a serving clone inherits
  the published value rather than a diagnostic binding.

CPU only: no GPU, no kernels, no performance claim.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
import textwrap
import warnings
from types import SimpleNamespace

import pytest

from hipengine import execution_profiles as profiles_module
from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)
from hipengine.generation import qwen4_exp_profiles as profile_module
from hipengine.generation.qwen4_exp_profiles import (
    PRODUCTION_ARITHMETIC_RECOVERY_FLAGS,
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    profile_ignored_overrides,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime import qwen4_exp_runner as runtime

ENV = "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"
LAYERS_ENV = "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS"
DECODE_LAYERS_ENV = "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS"
MULTI = "qwen4exp_gdn_tiled16_multi_prefill"
UD = "gguf_ud_q4_k_xl"
Q4KM = "gguf_q4_k_m"
UD_LAYERS = ",".join(map(str, range(48)))
Q4KM_LAYERS = ",".join(map(str, range(27, 48)))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    clear_runtime_profile_registry_for_tests()
    for key in (ENV, LAYERS_ENV, DECODE_LAYERS_ENV):
        monkeypatch.delenv(key, raising=False)
    register_gfx1151_kernels(replace=True)
    yield
    clear_runtime_profile_registry_for_tests()


def _enable_historical_q4km_plan() -> None:
    key = profiles_module.RuntimeProfileKey(
        QWEN4_EXP_MODEL, QWEN4_EXP_BACKEND, Q4KM,
        ExecutionProfile.PRODUCTION,
    )
    plan = profiles_module._RUNTIME_PROFILE_PLANS[key]
    if plan.qualification_failure is None:
        return
    profiles_module.register_runtime_profile_plan(
        model=key.model, backend=key.backend, quant=key.quant,
        profile=key.profile,
        plan=dataclasses.replace(plan, qualification_failure=None), replace=True,
    )


def _resolve(profile: ExecutionProfile, *, quant: str = UD):
    register_qwen4_exp_gfx1151_profiles()
    if quant == Q4KM and profile is ExecutionProfile.PRODUCTION:
        _enable_historical_q4km_plan()
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=quant,
        profile=profile,
    )


def _bind(resolved):
    runner = SimpleNamespace(configure_mmq_prefill_resources=lambda: None)
    generator = SimpleNamespace(runner=runner)
    resolved.binder(generator, resolved)
    return generator, runner


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_an_unbound_owner_keeps_the_serial_route(monkeypatch):
    """No field, no gate: the ambient key cannot open the tiled branch."""

    monkeypatch.setenv(ENV, "1")

    assert runtime.resolve_gdn_colwarps_prefill(SimpleNamespace()) is False
    assert runtime.resolve_gdn_colwarps_prefill(
        SimpleNamespace(gdn_colwarps_prefill=None)) is False


def test_the_published_gate_is_the_d128_decision():
    assert runtime.qwen4_exp_gdn_colwarps_prefill_selected(
        head_dim=128, published=True) is True
    assert runtime.qwen4_exp_gdn_colwarps_prefill_selected(
        head_dim=128, published=False) is False


def test_a_non_d128_chunk_stays_serial_even_with_the_gate_open():
    assert runtime.qwen4_exp_gdn_colwarps_prefill_selected(
        head_dim=256, published=True) is False


def test_selection_does_not_read_the_environment(monkeypatch):
    class GuardedEnvironment(dict):
        def get(self, key, default=None):
            if key == ENV:
                raise AssertionError("the columnwarp gate is owner state now")
            return super().get(key, default)

    monkeypatch.setattr(runtime.os, "environ", GuardedEnvironment())

    assert runtime.resolve_gdn_colwarps_prefill(
        SimpleNamespace(gdn_colwarps_prefill=True)) is True
    assert runtime.qwen4_exp_gdn_colwarps_prefill_selected(
        head_dim=128, published=True) is True


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------


def test_the_scope_binds_a_generator_and_its_runner_and_restores():
    owner = SimpleNamespace(
        gdn_colwarps_prefill=True,
        runner=SimpleNamespace(gdn_colwarps_prefill=True),
    )

    with runtime.gdn_colwarps_prefill(owner, False):
        assert owner.gdn_colwarps_prefill is False
        assert owner.runner.gdn_colwarps_prefill is False

    assert owner.gdn_colwarps_prefill is True
    assert owner.runner.gdn_colwarps_prefill is True


def test_the_scope_binds_a_bare_runner():
    """Diagnostics call the scope with the runner they measure."""

    runner = SimpleNamespace(gdn_colwarps_prefill=True)

    with runtime.gdn_colwarps_prefill(runner, False):
        assert runner.gdn_colwarps_prefill is False
    assert runner.gdn_colwarps_prefill is True


def test_the_scope_keeps_the_published_gate_when_the_arm_names_nothing():
    owner = SimpleNamespace(
        gdn_colwarps_prefill=True, runner=SimpleNamespace())

    with runtime.gdn_colwarps_prefill(owner):
        assert owner.gdn_colwarps_prefill is True
    assert owner.gdn_colwarps_prefill is True


def test_the_scope_restores_after_failure():
    owner = SimpleNamespace(
        gdn_colwarps_prefill=True,
        runner=SimpleNamespace(gdn_colwarps_prefill=True),
    )

    with pytest.raises(RuntimeError, match="arm failed"):
        with runtime.gdn_colwarps_prefill(owner, False):
            assert owner.runner.gdn_colwarps_prefill is False
            raise RuntimeError("arm failed mid-run")

    assert owner.gdn_colwarps_prefill is True
    assert owner.runner.gdn_colwarps_prefill is True


def test_the_scope_is_per_owner():
    first = SimpleNamespace(
        gdn_colwarps_prefill=True,
        runner=SimpleNamespace(gdn_colwarps_prefill=True))
    second = SimpleNamespace(
        gdn_colwarps_prefill=True,
        runner=SimpleNamespace(gdn_colwarps_prefill=True))

    with runtime.gdn_colwarps_prefill(first, False):
        assert second.gdn_colwarps_prefill is True
        with runtime.gdn_colwarps_prefill(second, True):
            assert first.gdn_colwarps_prefill is False
            assert second.gdn_colwarps_prefill is True
        assert second.gdn_colwarps_prefill is True
    assert first.gdn_colwarps_prefill is True


def test_a_serving_clone_inherits_the_published_gate_not_a_binding():
    """The arm is per-runner: a clone built inside it keeps the plan's gate."""

    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    generator = SimpleNamespace(gdn_colwarps_prefill=True)
    measured = SimpleNamespace(gdn_colwarps_prefill=True)
    with runtime.gdn_colwarps_prefill(measured, False):
        clone = _inherit_lane_decisions(generator, SimpleNamespace())
        assert getattr(clone, "gdn_colwarps_prefill", False) is True
        assert measured.gdn_colwarps_prefill is False
    assert measured.gdn_colwarps_prefill is True


# --------------------------------------------------------------------------
# Publication
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile,quant,expected", [
    (ExecutionProfile.PRODUCTION, UD, True),
    (ExecutionProfile.PRODUCTION, Q4KM, True),
    (ExecutionProfile.STRICT, UD, False),
    (ExecutionProfile.STRICT, Q4KM, False),
])
def test_binder_publishes_the_plan_gate_and_writes_no_process_key(
        profile, quant, expected):
    generator, runner = _bind(_resolve(profile, quant=quant))

    assert generator.gdn_colwarps_prefill is expected
    assert runner.gdn_colwarps_prefill is expected
    assert ENV not in os.environ


@pytest.mark.parametrize("profile,quant,expected", [
    (ExecutionProfile.PRODUCTION, UD, UD_LAYERS),
    (ExecutionProfile.PRODUCTION, Q4KM, Q4KM_LAYERS),
    (ExecutionProfile.STRICT, UD, ""),
    (ExecutionProfile.STRICT, Q4KM, ""),
])
def test_the_layer_scope_is_distinct_owner_state(profile, quant, expected):
    """The independent scope and enablement both publish on the owner."""

    generator, runner = _bind(_resolve(profile, quant=quant))

    assert generator.gdn_colwarps_layers == runner.gdn_colwarps_layers == expected
    assert LAYERS_ENV not in os.environ
    assert DECODE_LAYERS_ENV not in os.environ


def test_the_recovery_sweep_still_owns_no_recorded_recovery_value():
    """The migration must not move this flag into the recovery set."""

    assert "GDN_COLWARPS_PREFILL" not in PRODUCTION_ARITHMETIC_RECOVERY_FLAGS


def test_a_recovery_entry_controls_the_published_gate(monkeypatch):
    """A recovery zero for this flag publishes off without a process key."""

    monkeypatch.setattr(
        profile_module, "PRODUCTION_ARITHMETIC_RECOVERY_FLAGS",
        (*PRODUCTION_ARITHMETIC_RECOVERY_FLAGS, "GDN_COLWARPS_PREFILL"),
    )

    generator, runner = _bind(_resolve(ExecutionProfile.PRODUCTION, quant=UD))

    assert generator.gdn_colwarps_prefill is False
    assert runner.gdn_colwarps_prefill is False
    assert ENV not in os.environ


def test_a_recovery_entry_closes_the_ud_lane_and_not_a_lifted_q4km_plan(
        monkeypatch):
    """The recovery pass this flag belonged to ran only on production UD.

    Preserving that scope matters now that the historical Q4_K_M production
    plan resolves: the tuple closes UD, and the Q4_K_M plan keeps the branch
    open instead of inheriting a recovery decision its lane never carried.
    """

    monkeypatch.setattr(
        profile_module, "PRODUCTION_ARITHMETIC_RECOVERY_FLAGS",
        (*PRODUCTION_ARITHMETIC_RECOVERY_FLAGS, "GDN_COLWARPS_PREFILL"),
    )

    ud, ud_runner = _bind(_resolve(ExecutionProfile.PRODUCTION, quant=UD))
    q4km, q4km_runner = _bind(
        _resolve(ExecutionProfile.PRODUCTION, quant=Q4KM))

    assert (ud.gdn_colwarps_prefill, ud_runner.gdn_colwarps_prefill) == (
        False, False)
    assert (q4km.gdn_colwarps_prefill, q4km_runner.gdn_colwarps_prefill) == (
        True, True)
    assert ENV not in os.environ


@pytest.mark.parametrize("production,quant,recovery,expected", [
    (True, UD, (), True),
    (True, UD, ("GDN_COLWARPS_PREFILL",), False),
    (True, Q4KM, ("GDN_COLWARPS_PREFILL",), True),
    (True, Q4KM, (), True),
    (False, UD, (), False),
    (False, Q4KM, (), False),
])
def test_the_recovery_scope_is_the_ud_lane_only(
        production, quant, recovery, expected):
    """Production opens the gate unless UD and the recovery tuple names it."""

    assert profile_module._gdn_colwarps_prefill_enabled(
        production=production, quant=quant, recovery_flags=recovery,
    ) is expected


def test_two_owners_keep_independent_gates_in_both_bind_orders():
    production = _resolve(ExecutionProfile.PRODUCTION, quant=UD)
    strict = _resolve(ExecutionProfile.STRICT, quant=UD)
    first = SimpleNamespace(
        runner=SimpleNamespace(configure_mmq_prefill_resources=lambda: None))
    second = SimpleNamespace(
        runner=SimpleNamespace(configure_mmq_prefill_resources=lambda: None))

    production.binder(first, production)
    strict.binder(second, strict)
    assert first.gdn_colwarps_prefill is True
    assert second.gdn_colwarps_prefill is False

    strict.binder(first, strict)
    assert first.gdn_colwarps_prefill is False
    assert second.gdn_colwarps_prefill is False
    production.binder(second, production)
    assert first.gdn_colwarps_prefill is False
    assert second.gdn_colwarps_prefill is True


def test_a_caller_set_gate_is_reported_on_each_owner_and_stays_inert(monkeypatch):
    monkeypatch.setenv(ENV, "1")

    generator, runner = _bind(_resolve(ExecutionProfile.STRICT, quant=UD))

    assert os.environ[ENV] == "1"
    assert generator.gdn_colwarps_prefill is False
    assert runner.gdn_colwarps_prefill is False
    assert profile_ignored_overrides(generator)[ENV] == ("1", "0")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        peer, _ = _bind(_resolve(ExecutionProfile.STRICT, quant=UD))
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert profile_ignored_overrides(peer) == profile_ignored_overrides(generator)


def test_a_matching_ambient_value_is_not_a_conflict(monkeypatch):
    monkeypatch.setenv(ENV, "1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        owner, _ = _bind(_resolve(ExecutionProfile.PRODUCTION, quant=UD))

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert ENV not in profile_ignored_overrides(owner)


# --------------------------------------------------------------------------
# Dispatch: a faked launch through the real token mixer
# --------------------------------------------------------------------------


def _weight(slot: str, ptr: int, *, quant: str = "gguf_q8_0"):
    return SimpleNamespace(
        backend="hip_gfx1151",
        spec=SimpleNamespace(quant_key=quant, slot_path=f"layers.40.{slot}"),
        allocation=lambda kind, _ptr=ptr: SimpleNamespace(
            tensor=SimpleNamespace(ptr=_ptr)),
    )


def _mixer_launch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    owner_kwargs: dict[str, object],
    rows: int = 512,
    head_dim: int = 128,
    published: bool | None = True,
    layers: str | None = None,
    guard_environment: bool = False,
) -> dict[str, object]:
    """Run the real GDN token mixer on fake buffers and record its owner.

    Every projection, conv and post-mixer launch is stubbed, so the run reaches
    exactly one recurrence owner: the tile-16 parent (gate open, tile-16
    admitted), the columnwarp parent (gate open, tile-16 pinned off), the peer
    owner, the register parent, or the direct serial leaf (gate closed). The
    retired process key is never set here; ``published`` is the owner decision
    the prefill chunk would forward.
    """

    if layers is not None:
        monkeypatch.setenv(LAYERS_ENV, layers)

    calls: list[str] = []
    resolved: list[dict[str, object]] = []
    for name in (
        "_launch_qwen4_exp_dense_pair",
        "launch_gguf_linear",
        "qwen35_linear_attn_conv_decode_f32",
    ):
        monkeypatch.setattr(runtime, name, lambda *a, **k: None)
    monkeypatch.setattr(
        runtime, "_qwen4_exp_gdn_decode_step",
        lambda *a, **k: (lambda *a2, **k2: calls.append("decode")),
    )
    monkeypatch.setattr(
        runtime, "qwen4_exp_gdn_prefill_columnwarps_f32",
        lambda *a, **k: calls.append("columnwarps"),
    )
    monkeypatch.setattr(
        runtime, "qwen4_exp_gdn_prefill_f32",
        lambda *a, **k: calls.append("serial"),
    )
    monkeypatch.setattr(
        runtime, "qwen4_exp_gdn_prefill_tiled16_f32",
        lambda *a, **k: calls.append("parent"),
    )
    monkeypatch.setattr(
        runtime, "qwen4_exp_gdn_peer_prefill_f32",
        lambda *a, **k: calls.append("peer"),
    )

    def resolve(**kwargs):
        resolved.append(dict(kwargs))
        if kwargs.get("layer") == "gdn_recurrence_norm_gate":
            return lambda *a, **k: calls.append(str(kwargs.get("variant")))
        return lambda *a, **k: None

    monkeypatch.setattr(runtime, "resolve", resolve)

    if guard_environment:
        class GuardedEnvironment(dict):
            def get(self, key, default=None):
                if key in {ENV, LAYERS_ENV}:
                    raise AssertionError(f"{key} is not a dispatch input")
                return super().get(key, default)

        monkeypatch.setattr(runtime.os, "environ", GuardedEnvironment())

    def buffer(ptr: int):
        return SimpleNamespace(ptr=ptr)

    scratch = SimpleNamespace(
        closed=False,
        runtime=object(),
        qkv=buffer(0x1000),
        gate=buffer(0x1100),
        alpha=buffer(0x1200),
        beta=buffer(0x1300),
        conv=buffer(0x1400),
        core=buffer(0x1500),
        output=buffer(0x1600),
    )
    weights = {
        "attn_qkv": _weight("attn_qkv", 0x11),
        "attn_gate": _weight("attn_gate", 0x12),
        "ssm_alpha": _weight("ssm_alpha", 0x13),
        "ssm_beta": _weight("ssm_beta", 0x14),
        "ssm_out": _weight("ssm_out", 0x15),
    }
    kwargs = dict(owner_kwargs)
    if published is not None:
        kwargs["gdn_colwarps_prefill"] = published

    runtime.run_qwen4_exp_gdn_token_mixer(
        0x100,
        weights,
        conv_weight_ptr=0x21,
        dt_bias_ptr=0x22,
        a_ptr=0x23,
        norm_weight_ptr=0x24,
        conv_state_ptr=0x25,
        recurrent_state_ptr=0x26,
        scratch=scratch,
        rows=rows,
        hidden=64,
        num_k_heads=16,
        num_v_heads=48,
        head_dim=head_dim,
        conv_kernel=4,
        stream=0,
        runtime=scratch.runtime,
        **kwargs,
    )

    return {
        "calls": calls,
        "resolved": resolved,
        "tile16_resolves": [
            entry["variant"]
            for entry in resolved
            if entry.get("layer") == "gdn_recurrence_norm_gate"
        ],
    }


def test_the_published_gate_reaches_the_tiled_branch(monkeypatch):
    record = _mixer_launch(monkeypatch, owner_kwargs={}, published=True)

    assert record["calls"] == ["parent"]


def test_the_published_gate_keeps_the_columnwarp_parent_reachable(monkeypatch):
    record = _mixer_launch(
        monkeypatch,
        owner_kwargs={"gdn_tile16_prefill": False},
        published=True,
    )

    assert record["calls"] == ["columnwarps"]


def test_a_closed_gate_runs_the_serial_owner(monkeypatch):
    record = _mixer_launch(monkeypatch, owner_kwargs={}, published=False)

    assert record["calls"] == ["serial"]


def test_an_unbound_owner_never_reads_the_retired_key(monkeypatch):
    """A stale ambient value cannot open the gate the plan did not publish."""

    monkeypatch.setenv(ENV, "1")
    record = _mixer_launch(
        monkeypatch, owner_kwargs={}, published=None, guard_environment=True)

    assert record["calls"] == ["serial"]


def test_a_non_d128_chunk_stays_serial_with_the_gate_open(monkeypatch):
    record = _mixer_launch(
        monkeypatch, owner_kwargs={}, head_dim=256, published=True)

    assert record["calls"] == ["serial"]


def test_the_layer_allowlist_still_selects_inside_the_gate(monkeypatch):
    inside = _mixer_launch(
        monkeypatch, owner_kwargs={}, published=True, layers="27-47")
    outside = _mixer_launch(
        monkeypatch, owner_kwargs={}, published=True, layers="27-34")

    assert inside["calls"] == ["parent"]
    assert outside["calls"] == ["serial"]


def test_an_empty_layer_scope_still_means_every_layer(monkeypatch):
    """The shared parser's empty-means-all reading is unchanged."""

    record = _mixer_launch(
        monkeypatch, owner_kwargs={}, published=True, layers="")

    assert record["calls"] == ["parent"]


def test_the_gate_outranks_the_peer_owner(monkeypatch):
    record = _mixer_launch(
        monkeypatch,
        owner_kwargs={"gdn_peer_prefill_layers": "35-47"},
        published=True,
    )

    assert record["calls"] == ["parent"]
    assert "peer" not in record["calls"]


def test_the_peer_owner_runs_below_a_closed_gate(monkeypatch):
    record = _mixer_launch(
        monkeypatch,
        owner_kwargs={"gdn_peer_prefill_layers": "35-47"},
        published=False,
    )

    assert record["calls"] == ["peer"]


def test_the_register_parent_stays_below_the_closed_gate(monkeypatch):
    record = _mixer_launch(
        monkeypatch,
        owner_kwargs={"gdn_register_prefill": True},
        published=False,
    )

    assert record["calls"] == [runtime.GDN_PREFILL_PARENT_VARIANT]


def test_the_serial_strict_leaves_stay_registered():
    for variant in (
        runtime.GDN_PREFILL_PARENT_VARIANT,
        runtime.GDN_PREFILL_WIDE_VARIANT,
        runtime.GDN_PREFILL_NARROW_VARIANT,
    ):
        assert is_registered(KernelKey(
            "hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state", variant))


def test_the_registered_tile16_owner_is_still_admitted_inside_the_gate(
        monkeypatch):
    record = _mixer_launch(
        monkeypatch,
        owner_kwargs={"gdn_tile16_variant": MULTI},
        published=True,
    )

    assert record["calls"] == [MULTI]


def test_a_decode_row_never_reads_the_gate(monkeypatch):
    """A c1 row runs the decode owner, so no captured segment can key on this."""

    record = _mixer_launch(
        monkeypatch, owner_kwargs={}, rows=1, published=True)

    assert record["calls"] == ["decode"]
    assert record["tile16_resolves"] == []


# --------------------------------------------------------------------------
# Forwarding
# --------------------------------------------------------------------------


def _layer(monkeypatch: pytest.MonkeyPatch, seen: list[object], **extra):
    runtime_owner = object()

    def gr_read(*args, **kwargs):
        return SimpleNamespace(normalized=None, gate=None,
                               mixed=SimpleNamespace(ptr=11),
                               inject_logits=SimpleNamespace(ptr=12))

    def mixer(*args, **kwargs):
        seen.append(kwargs.get("gdn_colwarps_prefill", "absent"))
        return SimpleNamespace(ptr=13)

    monkeypatch.setattr(runtime, "run_qwen4_exp_gr_read", gr_read)
    monkeypatch.setattr(runtime, "run_qwen4_exp_gdn_token_mixer", mixer)
    monkeypatch.setattr(
        runtime, "run_qwen4_exp_moe",
        lambda *a, **k: SimpleNamespace(output=SimpleNamespace(ptr=14)),
    )
    monkeypatch.setattr(runtime, "_qwen4_exp_gr_pending_write", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "_qwen4_exp_gr_defer_write", lambda *a, **k: False)
    monkeypatch.setattr(runtime, "qwen4_exp_gr_write_bf16_f32", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "qwen4_exp_gr_write_bf16_bf16", lambda *a, **k: None)
    output = SimpleNamespace(ptr=15)
    scratch = SimpleNamespace(
        closed=False, runtime=runtime_owner, attention_gr=SimpleNamespace(),
        gdn=SimpleNamespace(), after_attention=SimpleNamespace(ptr=16),
        ffn_gr=SimpleNamespace(), moe=SimpleNamespace(), output=output)
    weights = SimpleNamespace(
        attention_gr=SimpleNamespace(norm_weight_ptr=1, down=2, up=3, inject=4),
        mixer=SimpleNamespace(projections={}, conv_weight_ptr=5, dt_bias_ptr=6,
                              a_ptr=7, norm_weight_ptr=8),
        ffn_gr=SimpleNamespace(norm_weight_ptr=9, down=10, up=11, inject=12),
        moe=SimpleNamespace())
    runtime.run_qwen4_exp_gdn_layer(
        100, weights,
        conv_state_ptr=0, recurrent_state_ptr=0, scratch=scratch,
        rows=512, branches=1, hidden=2, low_rank=2, num_k_heads=16,
        num_v_heads=48, head_dim=128, conv_kernel=4, ffn=8,
        experts=1, top_k=1, runtime=runtime_owner,
        **extra,
    )


def test_the_layer_forwards_the_gate_to_the_mixer(monkeypatch):
    """The layer's only mixer call site carries the decision it was given."""

    seen: list[object] = []

    _layer(monkeypatch, seen, gdn_colwarps_prefill=True)
    _layer(monkeypatch, seen)

    assert seen == [True, False]


def test_the_runner_defaults_to_a_closed_gate():
    runner = object.__new__(runtime.Qwen4ExpGGUFResidentModelRunner)

    assert runner.gdn_colwarps_prefill is False


def test_the_prefill_chunk_forwards_the_runner_field():
    source = textwrap.dedent(
        inspect.getsource(
            runtime.Qwen4ExpGGUFResidentModelRunner._prefill_chunk_steps
        )
    )
    call = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_qwen4_exp_gdn_layer"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}

    assert isinstance(keywords["gdn_colwarps_prefill"], ast.Attribute)
    assert keywords["gdn_colwarps_prefill"].attr == "gdn_colwarps_prefill"


def test_the_launch_site_reads_the_owner_and_keeps_the_layer_scope():
    mixer = inspect.getsource(runtime.run_qwen4_exp_gdn_token_mixer)

    assert ENV not in mixer
    assert "qwen4_exp_gdn_colwarps_prefill_selected(" in mixer
    assert "qwen4_exp_gdn_prefill_columnwarps_f32" in mixer
    assert LAYERS_ENV not in mixer
    assert 'qwen4_exp_gdn_colwarps_layer_allowed(' in mixer
    assert 'published=gdn_colwarps_layers' in mixer
