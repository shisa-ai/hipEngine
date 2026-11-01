"""The GDN tile-16 registered leaf is owner state, not a process key.

``HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT`` names which registered leaf the
tile-16 GDN prefill branch resolves inside the columnwarp gate. It is read at
launch time by ``run_qwen4_exp_gdn_token_mixer``, written by the profile binder
in two places, and written or read by six diagnostics. That makes a process
value out of a plan decision: a stale ambient value can serve a request whose
plan published another leaf, and an arm that only writes the key measures the
plan instead of the leaf it names.

The migration moves the decision onto the owning generator and runner as
``gdn_tile16_variant`` and leaves the launch reading only that field:

* ``None`` or ``""`` keeps the directly imported
  ``qwen4_exp_gdn_prefill_tiled16_f32`` parent - the leaf every build resolved
  before the registered tile-16 names existed, and the value the strict binder
  publishes today.
* A name resolves that registered leaf, so the multi-column owner, the DPP
  owner and the plain ``qwen4exp_gdn_tiled16_prefill`` name all travel the same
  slot, and an unknown name still fails loudly at the registry.
* The columnwarp gate (``HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL``) and the
  serial strict owners below it are untouched: the owner only decides *inside*
  the gate, and a closed gate still runs the columnwarp parent and then the
  serial route.

What these tests pin:

* **Selection.** An unbound or empty owner takes the parent leaf and never the
  environment; an explicit name is returned unchanged.
* **Scope.** One owner scope binds a generator and its current runner, restores
  the published value on exit including after a raise, and stays per-owner. A
  serving clone built inside an arm inherits the published value, not the
  binding.
* **Publication.** Every bind publishes the value the retired key carried - the
  f16-route plan's tiled-GDN composition row on the UD production plan, the
  multi-column owner on the historical ``gguf_q4_k_m`` production plan, and the
  parent everywhere the strict binder runs - and no bind writes the process
  key. A caller-set value is reported once and stays inert.
* **Dispatch.** A real (faked-launch) token-mixer run reaches the leaf the owner
  names, keeps the multi-column, DPP, plain-name and parent routes distinct,
  never reads the retired key, and never consults the owner on a decode row -
  which is why a captured c1 segment cannot depend on it.
* **Forwarding.** The owner reaches the mixer through the GDN layer and the
  runner's prefill chunk path, and the runner's own default is the parent.

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
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    last_prebinder_conflicts,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime import qwen4_exp_runner as runtime

ENV = "HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"
COLWARPS_ENV = "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"
MULTI = "qwen4exp_gdn_tiled16_multi_prefill"
DPP = "qwen4exp_gdn_tiled16_dpp_prefill"
PLAIN = "qwen4exp_gdn_tiled16_prefill"
#: The UD production plan's tiled-GDN composition row. It is not a tile-16
#: owner, but it is the value the retired key carried on that plan, so the
#: migration preserves it verbatim.
UD_COMPOSITION_LEAF = "qwen4exp_gdn_rowsplit_prefill"
UD = "gguf_ud_q4_k_xl"
Q4KM = "gguf_q4_k_m"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    clear_runtime_profile_registry_for_tests()
    profile_module._owner_binder_discarded.clear()
    profile_module._last_prebinder_conflicts.clear()
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(COLWARPS_ENV, raising=False)
    yield
    profile_module._owner_binder_discarded.clear()
    profile_module._last_prebinder_conflicts.clear()
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
    register_gfx1151_kernels(replace=True)
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


def test_an_unbound_owner_keeps_the_directly_imported_tile16_parent(monkeypatch):
    """No field, no name: the parent is the leaf, not the ambient key."""

    monkeypatch.setenv(ENV, MULTI)

    assert runtime._qwen4_exp_gdn_tile16_variant(None) is None
    assert runtime.resolve_gdn_tile16_variant(SimpleNamespace()) is None
    assert runtime.resolve_gdn_tile16_variant(
        SimpleNamespace(gdn_tile16_variant=None)) is None


def test_an_empty_owner_selector_means_the_parent_route(monkeypatch):
    """The strict binder's former ``""`` is the parent, not an empty variant."""

    monkeypatch.setenv(ENV, MULTI)

    assert runtime._qwen4_exp_gdn_tile16_variant("") is None
    assert runtime.resolve_gdn_tile16_variant(
        SimpleNamespace(gdn_tile16_variant="")) is None


@pytest.mark.parametrize("variant", [MULTI, DPP, PLAIN])
def test_an_explicit_owner_name_is_returned_unchanged(monkeypatch, variant):
    """Selection never rewrites a name it does not recognise."""

    monkeypatch.setenv(ENV, DPP if variant != DPP else MULTI)

    assert runtime._qwen4_exp_gdn_tile16_variant(variant) == variant
    assert runtime.resolve_gdn_tile16_variant(
        SimpleNamespace(gdn_tile16_variant=variant)) == variant


def test_selection_does_not_read_the_environment(monkeypatch):
    class GuardedEnvironment(dict):
        def get(self, key, default=None):
            if key == ENV:
                raise AssertionError("the tile-16 key is owner state now")
            return super().get(key, default)

    monkeypatch.setattr(runtime.os, "environ", GuardedEnvironment())

    assert runtime._qwen4_exp_gdn_tile16_variant(None) is None
    assert runtime._qwen4_exp_gdn_tile16_variant(MULTI) == MULTI


def test_unknown_explicit_variant_still_fails_loudly_at_the_registry():
    bogus = "qwen4exp_gdn_tiled16_nope_prefill"

    # Selection passes the name through unchanged, so the registry is the only
    # place that decides whether an owner exists.
    assert runtime._qwen4_exp_gdn_tile16_variant(bogus) == bogus
    register_gfx1151_kernels(replace=True)
    assert not is_registered(
        KernelKey("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state", bogus))


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------


def test_the_scope_binds_the_owner_and_its_runner_and_restores():
    owner = SimpleNamespace(
        gdn_tile16_variant=MULTI,
        runner=SimpleNamespace(gdn_tile16_variant=MULTI),
    )

    with runtime.gdn_tile16_variant(owner, DPP):
        assert owner.gdn_tile16_variant == DPP
        assert owner.runner.gdn_tile16_variant == DPP

    assert owner.gdn_tile16_variant == MULTI
    assert owner.runner.gdn_tile16_variant == MULTI


def test_the_scope_keeps_the_published_owner_when_the_arm_names_none():
    owner = SimpleNamespace(gdn_tile16_variant=MULTI, runner=SimpleNamespace())

    with runtime.gdn_tile16_variant(owner):
        assert owner.gdn_tile16_variant == MULTI
    assert owner.gdn_tile16_variant == MULTI


def test_the_scope_restores_after_failure():
    owner = SimpleNamespace(
        gdn_tile16_variant=MULTI,
        runner=SimpleNamespace(gdn_tile16_variant=MULTI),
    )

    with pytest.raises(RuntimeError, match="arm failed"):
        with runtime.gdn_tile16_variant(owner, DPP):
            assert owner.runner.gdn_tile16_variant == DPP
            raise RuntimeError("arm failed mid-run")

    assert owner.gdn_tile16_variant == MULTI
    assert owner.runner.gdn_tile16_variant == MULTI


def test_the_scope_is_per_owner():
    first = SimpleNamespace(
        gdn_tile16_variant=MULTI, runner=SimpleNamespace(gdn_tile16_variant=MULTI))
    second = SimpleNamespace(
        gdn_tile16_variant=MULTI, runner=SimpleNamespace(gdn_tile16_variant=MULTI))

    with runtime.gdn_tile16_variant(first, DPP):
        assert second.gdn_tile16_variant == MULTI
        with runtime.gdn_tile16_variant(second, PLAIN):
            assert first.gdn_tile16_variant == DPP
            assert second.gdn_tile16_variant == PLAIN
        assert second.gdn_tile16_variant == MULTI
    assert first.gdn_tile16_variant == MULTI


def test_a_serving_clone_inherits_the_published_owner_not_a_diagnostic_binding():
    """The arm is per-runner: a clone built inside it keeps the plan's leaf."""

    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    generator = SimpleNamespace(gdn_tile16_variant=MULTI)
    measured = SimpleNamespace(gdn_tile16_variant=MULTI)
    with runtime.gdn_tile16_variant(measured, DPP):
        clone = _inherit_lane_decisions(generator, SimpleNamespace())
        assert getattr(clone, "gdn_tile16_variant", None) == MULTI
        assert measured.gdn_tile16_variant == DPP
    assert measured.gdn_tile16_variant == MULTI


# --------------------------------------------------------------------------
# Publication
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile,quant,expected", [
    # The f16-route plan publishes its tiled-GDN composition row, which is what
    # the retired key carried there; the columnwarp gate is closed on that plan,
    # so the value is inert and preserved verbatim.
    (ExecutionProfile.PRODUCTION, UD, UD_COMPOSITION_LEAF),
    (ExecutionProfile.PRODUCTION, Q4KM, MULTI),
    (ExecutionProfile.STRICT, UD, None),
    (ExecutionProfile.STRICT, Q4KM, None),
])
def test_binder_publishes_the_plan_owner_and_writes_no_process_key(
        profile, quant, expected):
    generator, runner = _bind(_resolve(profile, quant=quant))

    assert generator.gdn_tile16_variant == expected
    assert runner.gdn_tile16_variant == expected
    assert ENV not in os.environ


@pytest.mark.parametrize("quant,expected", [
    (UD, UD_COMPOSITION_LEAF),
    (Q4KM, MULTI),
])
def test_the_published_owner_is_a_registered_leaf(quant, expected):
    """A published name the registry does not carry would fail at first launch."""

    generator, _ = _bind(_resolve(ExecutionProfile.PRODUCTION, quant=quant))

    assert generator.gdn_tile16_variant == expected
    assert is_registered(KernelKey(
        "hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state", expected))


def test_two_owners_keep_independent_values_in_both_bind_orders():
    historical = _resolve(ExecutionProfile.PRODUCTION, quant=Q4KM)
    strict = _resolve(ExecutionProfile.STRICT, quant=UD)
    first = SimpleNamespace(
        runner=SimpleNamespace(configure_mmq_prefill_resources=lambda: None))
    second = SimpleNamespace(
        runner=SimpleNamespace(configure_mmq_prefill_resources=lambda: None))

    historical.binder(first, historical)
    strict.binder(second, strict)
    assert first.gdn_tile16_variant == MULTI
    assert first.runner.gdn_tile16_variant == MULTI
    assert second.gdn_tile16_variant is None
    assert second.runner.gdn_tile16_variant is None

    strict.binder(first, strict)
    assert first.gdn_tile16_variant is None
    assert second.gdn_tile16_variant is None
    historical.binder(second, historical)
    assert first.gdn_tile16_variant is None
    assert second.gdn_tile16_variant == MULTI


def test_a_caller_set_selector_is_reported_once_and_stays_inert(monkeypatch):
    monkeypatch.setenv(ENV, DPP)

    with pytest.warns(RuntimeWarning, match=ENV):
        generator, runner = _bind(_resolve(ExecutionProfile.PRODUCTION, quant=Q4KM))

    assert os.environ[ENV] == DPP
    assert generator.gdn_tile16_variant == MULTI
    assert runner.gdn_tile16_variant == MULTI
    assert last_prebinder_conflicts()[ENV] == (DPP, MULTI)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _bind(_resolve(ExecutionProfile.PRODUCTION, quant=Q4KM))
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_a_matching_ambient_value_is_not_a_conflict(monkeypatch):
    monkeypatch.setenv(ENV, MULTI)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _bind(_resolve(ExecutionProfile.PRODUCTION, quant=Q4KM))

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert last_prebinder_conflicts() == {}


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
    gate: bool = True,
    stale: str | None = None,
) -> dict[str, object]:
    """Run the real GDN token mixer on fake buffers and record its owner.

    Every projection, conv and post-mixer launch is stubbed, so the run reaches
    exactly one recurrence leaf: the columnwarp parent (gate closed), the serial
    strict owner (no tiled route), the directly imported tile-16 parent (no
    owner name) or the registered leaf the owner names. ``resolve`` is stubbed
    too, so the recorded call is the dispatch's own choice rather than a
    registry lookup.
    """

    monkeypatch.setenv(COLWARPS_ENV, "1" if gate else "0")
    if stale is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, stale)

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

    def resolve(**kwargs):
        resolved.append(dict(kwargs))
        if kwargs.get("layer") == "gdn_recurrence_norm_gate":
            return lambda *a, **k: calls.append(str(kwargs.get("variant")))
        return lambda *a, **k: None

    monkeypatch.setattr(runtime, "resolve", resolve)

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
        head_dim=128,
        conv_kernel=4,
        stream=0,
        runtime=scratch.runtime,
        **owner_kwargs,
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


@pytest.mark.parametrize("variant", [MULTI, DPP, PLAIN])
def test_the_bound_owner_is_the_leaf_the_launch_resolves(monkeypatch, variant):
    record = _mixer_launch(
        monkeypatch, owner_kwargs={"gdn_tile16_variant": variant})

    assert record["calls"] == [variant]
    assert record["tile16_resolves"] == [variant]


def test_the_empty_selector_launches_the_parent(monkeypatch):
    record = _mixer_launch(monkeypatch, owner_kwargs={"gdn_tile16_variant": ""})

    assert record["calls"] == ["parent"]
    assert record["tile16_resolves"] == []


def test_a_stale_selector_value_cannot_change_the_launch(monkeypatch):
    """The unbound owner takes the parent even with the retired key set."""

    record = _mixer_launch(monkeypatch, owner_kwargs={}, stale=DPP)

    assert record["calls"] == ["parent"]
    assert record["tile16_resolves"] == []


def test_the_columnwarp_parent_still_precedes_the_owner(monkeypatch):
    """The admission decision outranks the owner: a pinned-off arm keeps it.

    ``gdn_tile16_prefill=False`` is the diagnostic control that pins the
    columnwarp parent inside an open gate. Naming a tile-16 owner must not
    bypass it, or that arm would silently measure the leaf it meant to exclude.
    The name is still resolved eagerly, exactly as the retired key was read
    eagerly before the admission check; only the launch is gated.
    """

    record = _mixer_launch(
        monkeypatch,
        owner_kwargs={"gdn_tile16_variant": MULTI, "gdn_tile16_prefill": False},
    )

    assert record["calls"] == ["columnwarps"]
    assert MULTI not in record["calls"]


def test_the_serial_strict_route_stays_reachable_below_the_tile16_branch(
        monkeypatch):
    """The owner never opens the tiled route: a closed gate keeps serial."""

    record = _mixer_launch(
        monkeypatch, owner_kwargs={"gdn_tile16_variant": MULTI}, gate=False,
        rows=15)

    assert record["calls"] == ["serial"]
    assert record["tile16_resolves"] == []


def test_a_decode_row_never_reads_the_tile16_owner(monkeypatch):
    """A c1 row runs the decode owner, so no captured segment can key on this.

    The tile-16 owner only exists inside the multi-row prefill chunk path; a
    decode pass takes the decode branch first. That is why the c1 layer graph
    key does not carry it, and why no batch carrier needs it either.
    """

    record = _mixer_launch(
        monkeypatch, owner_kwargs={"gdn_tile16_variant": MULTI}, rows=1)

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
        seen.append(kwargs.get("gdn_tile16_variant", "absent"))
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


def test_the_layer_forwards_the_owner_to_the_mixer(monkeypatch):
    """The layer's only mixer call site carries the owner it was given."""

    seen: list[object] = []

    _layer(monkeypatch, seen, gdn_tile16_variant=MULTI)
    _layer(monkeypatch, seen)

    assert seen == [MULTI, None]


def test_the_runner_defaults_to_the_unbound_parent():
    runner = object.__new__(runtime.Qwen4ExpGGUFResidentModelRunner)

    assert runner.gdn_tile16_variant is None


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

    assert isinstance(keywords["gdn_tile16_variant"], ast.Attribute)
    assert keywords["gdn_tile16_variant"].attr == "gdn_tile16_variant"


def test_the_launch_site_reads_the_owner_and_keeps_the_parent_symbol():
    mixer = textwrap.dedent(inspect.getsource(runtime.run_qwen4_exp_gdn_token_mixer))

    assert ENV not in mixer
    assert "qwen4_exp_gdn_prefill_tiled16_f32" in mixer
    assert "_qwen4_exp_gdn_tile16_variant(gdn_tile16_variant)" in mixer
