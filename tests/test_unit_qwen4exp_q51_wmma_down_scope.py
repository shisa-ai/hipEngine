"""The Q5_1 F16 WMMA down owner is request-local runner state.

``HIPENGINE_QWEN4_EXP_Q5_1_WMMA`` was the process-wide transport for the
down-only MX-1 owner: the F16-WMMA compact grouped Q5_1 down projection that
sits below the exact grouped Q5_1 down chain. It is the owning runner's
``q51_wmma_down`` field now.

Only the transport moves, and the production admission survives: the dispatch
runs that owner whenever the production grouped MoE route owns the chunk (the
Q5K gate/up window, ``production_grouped_moe``), whatever the field says, and
otherwise when the plan-published field is on. The chain around it is untouched:
the routed-F16 route owns the whole down leg above the Q5_1 branch, the T16 owner
sits between it and this one, and the exact grouped Q5_1 down chain is tested
before it. Below it the strict rowbatch8 owner stays the fallback.

The owner is also a captured-segment decision: the MoE runs eager inside a
captured layer segment and decode (``rows == 1``) cannot reach
``production_grouped_moe``, so a segment captured with the field on bakes this
owner's launch in. The batch capture key therefore carries the field with the
other expert-down decisions, and the batch/mixed/clone carriers forward it.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation import qwen4_exp_ar_batch, qwen4_exp_gguf
from hipengine.generation.qwen4_exp_profiles import (
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import resolve
from hipengine.runtime import qwen4_exp_batch_runner as batch_module
from hipengine.runtime import qwen4_exp_mixed_target as mixed_module
from hipengine.runtime import qwen4_exp_runner as runner_module

RETIRED_ENV = "HIPENGINE_QWEN4_EXP_Q5_1_WMMA"
FIELD = "q51_wmma_down"
ROOT = Path(__file__).resolve().parents[1]

#: The owner this decision admits and the owner the dispatch falls through to
#: when it is closed (and the production grouped route does not own the chunk).
F16_DOWN_VARIANT = "selected_grouped_wmma_prefill_compact_bf16_bf16_out"
STRICT_DOWN_VARIANT = (
    "selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out")

#: The functions that carry the decision from a runner to the MoE dispatch.
MOE_FORWARDERS = ("run_qwen4_exp_moe", "_run_layer_ffn")


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RETIRED_ENV, raising=False)


@pytest.fixture
def profiles():
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    return None


# --- the decision -----------------------------------------------------------


def test_the_owner_decision_keeps_the_production_admission() -> None:
    """``production_grouped_moe`` admits the owner on its own; the field does not
    narrow it."""

    decide = runner_module._qwen4_exp_q51_wmma_down_enabled
    assert decide(production_grouped_moe=True, published=False) is True
    assert decide(production_grouped_moe=True, published=True) is True
    assert decide(production_grouped_moe=False, published=True) is True
    assert decide(production_grouped_moe=False, published=False) is False


def test_the_dispatch_reads_the_field_and_no_process_key() -> None:
    parameters = inspect.signature(runner_module.run_qwen4_exp_moe).parameters
    assert FIELD in parameters
    assert parameters[FIELD].default is False

    source = inspect.getsource(runner_module.run_qwen4_exp_moe)
    assert RETIRED_ENV not in source
    assert "_qwen4_exp_q51_wmma_down_enabled(" in source
    assert "production_grouped_moe, q51_wmma_down" in source
    # The branch keeps its place in the down-leg order: the T16 owner and the
    # exact grouped chain are tested first, and the strict owner is the
    # fallthrough below it.
    assert source.index("if t16_down:") < source.index("elif q5_wmma:")
    assert source.rindex("elif exact_grouped_down:") < source.index(
        "elif q5_wmma:")
    assert source.index("elif q5_wmma:") < source.index(
        "_run_qwen4_exp_q51_wmma_down(")
    assert source.index("_run_qwen4_exp_q51_wmma_down(") < source.index(
        STRICT_DOWN_VARIANT + "(")


def test_no_module_names_the_retired_key() -> None:
    offenders = [
        str(path)
        for root in (ROOT / "hipengine", ROOT / "scripts")
        for path in root.rglob("*.py")
        if RETIRED_ENV in path.read_text()
    ]
    assert offenders == []


# --- the scoped control -----------------------------------------------------


def test_the_runner_defaults_the_owner_off() -> None:
    runner = object.__new__(runner_module.Qwen4ExpGGUFResidentModelRunner)
    assert getattr(runner, FIELD) is False


def test_scope_binds_the_owner_and_restores() -> None:
    runner = SimpleNamespace(**{FIELD: False})

    with getattr(runner_module, FIELD)(runner, True) as prior:
        assert prior is False
        assert getattr(runner, FIELD) is True
        with getattr(runner_module, FIELD)(runner, False):
            assert getattr(runner, FIELD) is False
        assert getattr(runner, FIELD) is True
    assert getattr(runner, FIELD) is False


def test_scope_leaves_an_unset_binding_alone() -> None:
    runner = SimpleNamespace(**{FIELD: True})

    with getattr(runner_module, FIELD)(runner):
        assert getattr(runner, FIELD) is True
    assert getattr(runner, FIELD) is True


def test_scope_restores_after_an_exception() -> None:
    runner = SimpleNamespace(**{FIELD: False})

    with pytest.raises(RuntimeError, match="arm failed"):
        with getattr(runner_module, FIELD)(runner, True):
            raise RuntimeError("arm failed")
    assert getattr(runner, FIELD) is False


def test_scope_is_per_runner() -> None:
    first = SimpleNamespace(**{FIELD: False})
    second = SimpleNamespace(**{FIELD: False})

    with getattr(runner_module, FIELD)(first, True):
        assert getattr(second, FIELD) is False
        with getattr(runner_module, FIELD)(second, True):
            assert getattr(first, FIELD) is True
            assert getattr(second, FIELD) is True
        assert getattr(second, FIELD) is False
    assert getattr(first, FIELD) is False


# --- the plan publication ---------------------------------------------------


def _resolve(profile: ExecutionProfile):
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant="gguf_ud_q4_k_xl",
        profile=profile,
    )


@pytest.mark.parametrize(
    "profile", [ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION])
def test_the_binder_publishes_the_field_and_writes_no_process_key(
        profiles, profile) -> None:
    resolved = _resolve(profile)
    assert resolved.binder is not None
    generator = SimpleNamespace()
    resolved.binder(generator, resolved)

    assert getattr(generator, FIELD) is False
    assert RETIRED_ENV not in os.environ


def test_the_binder_reaches_a_runner_it_finds(profiles) -> None:
    resolved = _resolve(ExecutionProfile.PRODUCTION)
    runner = SimpleNamespace()
    resolved.binder(SimpleNamespace(runner=runner), resolved)

    assert getattr(runner, FIELD) is False


def test_a_cloned_runner_inherits_the_published_field() -> None:
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    clone = SimpleNamespace()
    _inherit_lane_decisions(SimpleNamespace(**{FIELD: True}), clone)
    assert getattr(clone, FIELD) is True
    # An unbound carrier publishes the plan's decision, never the other lane's.
    _inherit_lane_decisions(SimpleNamespace(), clone)
    assert getattr(clone, FIELD) is False


def test_the_batch_owner_publishes_the_generators_field() -> None:
    source = inspect.getsource(qwen4_exp_ar_batch)
    assert f"batch.{FIELD} = bool(" in source
    assert f'getattr(generator, "{FIELD}", False))' in source


# --- the capture key --------------------------------------------------------


def test_the_batch_capture_key_carries_the_field() -> None:
    """A captured decode segment bakes the owner in; its key must say so."""

    owner = SimpleNamespace(
        q51_down_m1=False,
        q51_fold128_prefill=False,
        q51_fold_pair_prefill=False,
        q51_exact_grouped_down=True,
        q51_out8=True,
        q4_exact_grouped_gate=True,
        q4_exact_grouped_all=True,
        q4_out4=True,
        forkb_grouped_down=False,
        q8_down_row4_prefill=False,
        q8_down_bundle_prefill=False,
        q5k_row4_prefill=False,
        **{FIELD: False},
    )
    assert batch_module.Qwen4ExpGGUFBatchRunner._expert_down_policy(owner) == (
        False, False, False, True, False, True, True, True, True, False, False,
        False, False)

    owner.q51_exact_grouped_down = True
    owner.q51_wmma_down = True
    assert batch_module.Qwen4ExpGGUFBatchRunner._expert_down_policy(owner) == (
        False, False, False, True, True, True, True, True, True, False, False,
        False, False)


# --- forwarding -------------------------------------------------------------


def _calls_named(source: str, names) -> list[ast.Call]:
    tree = ast.parse(textwrap.dedent(source))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in names
    ]


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


@pytest.mark.parametrize(
    "function",
    [
        runner_module.run_qwen4_exp_moe,
        runner_module.run_qwen4_exp_dense_qsa_layer,
        runner_module.run_qwen4_exp_qsa_prefill_layer,
        runner_module.run_qwen4_exp_gdn_layer,
        batch_module.run_qwen4_exp_batch_gdn_layer,
        batch_module.run_qwen4_exp_batch_qsa_layer,
        batch_module._run_layer_ffn,
    ],
)
def test_every_forwarder_carries_the_field(function) -> None:
    parameters = inspect.signature(function).parameters
    assert FIELD in parameters, function.__qualname__
    assert parameters[FIELD].default is False, function.__qualname__

    calls = _calls_named(inspect.getsource(function), MOE_FORWARDERS)
    assert calls, f"{function.__qualname__} forwards no MoE call"
    for call in calls:
        value = _keyword(call, FIELD)
        assert isinstance(value, ast.Name), (
            function.__qualname__, ast.dump(call))
        assert value.id == FIELD


def test_the_mixed_pass_reads_the_field_from_the_batch_owner() -> None:
    source = inspect.getsource(mixed_module)
    assert f"{FIELD}=getattr(" in source
    assert f'"{FIELD}", False)' in source


def _layer_scratch(runtime):
    return SimpleNamespace(
        closed=False, runtime=runtime,
        attention_gr=SimpleNamespace(), qsa=SimpleNamespace(),
        gdn=SimpleNamespace(),
        after_attention=SimpleNamespace(ptr=16), ffn_gr=SimpleNamespace(),
        moe=SimpleNamespace(), output=SimpleNamespace(ptr=15))


def _layer_weights():
    return SimpleNamespace(
        attention_gr=SimpleNamespace(norm_weight_ptr=1, down=2, up=3, inject=4),
        mixer=SimpleNamespace(),
        ffn_gr=SimpleNamespace(norm_weight_ptr=9, down=10, up=11, inject=12),
        moe=SimpleNamespace())


def _gr_read(*args, **kwargs):
    return SimpleNamespace(normalized=None, gate=None,
                           mixed=SimpleNamespace(ptr=11),
                           inject_logits=SimpleNamespace(ptr=12))


def _moe_seen(seen):
    def moe(*args, **kwargs):
        seen.append(kwargs.get(FIELD, "absent"))
        return SimpleNamespace(output=SimpleNamespace(ptr=14))
    return moe


def _stub_common(monkeypatch, seen):
    monkeypatch.setattr(runner_module, "run_qwen4_exp_gr_read", _gr_read)
    monkeypatch.setattr(runner_module, "run_qwen4_exp_moe", _moe_seen(seen))
    monkeypatch.setattr(
        runner_module, "_qwen4_exp_gr_pending_write", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_module, "qwen4_exp_gr_write_bf16_f32", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_module, "qwen4_exp_gr_write_bf16_bf16", lambda *a, **k: None)


def test_the_gdn_layer_forwards_the_field_to_run_moe(monkeypatch) -> None:
    seen = []
    runtime = object()
    _stub_common(monkeypatch, seen)
    monkeypatch.setattr(
        runner_module, "run_qwen4_exp_gdn_token_mixer",
        lambda *a, **k: SimpleNamespace(ptr=13))
    monkeypatch.setattr(
        runner_module, "_qwen4_exp_gr_defer_write", lambda *a, **k: False)
    weights = _layer_weights()
    weights.mixer = SimpleNamespace(projections={}, conv_weight_ptr=5,
                                    dt_bias_ptr=6, a_ptr=7, norm_weight_ptr=8)
    kwargs = dict(conv_state_ptr=0, recurrent_state_ptr=0,
                  scratch=_layer_scratch(runtime), rows=512, branches=1,
                  hidden=2, low_rank=2, num_k_heads=16, num_v_heads=48,
                  head_dim=128, conv_kernel=4, ffn=8, experts=1, top_k=1,
                  runtime=runtime)
    runner_module.run_qwen4_exp_gdn_layer(100, weights, **kwargs, **{FIELD: True})
    runner_module.run_qwen4_exp_gdn_layer(100, weights, **kwargs)
    assert seen == [True, False]


def test_the_dense_qsa_layer_forwards_the_field_to_run_moe(monkeypatch) -> None:
    seen = []
    runtime = object()
    _stub_common(monkeypatch, seen)
    monkeypatch.setattr(
        runner_module, "run_qwen4_exp_dense_qsa_token_mixer",
        lambda *a, **k: SimpleNamespace(ptr=13))
    kwargs = dict(attention_state=object(), scratch=_layer_scratch(runtime),
                  position=0, rows=512, branches=1, hidden=2, low_rank=2,
                  query_heads=24, kv_heads=2, head_dim=128, rotary_dim=64,
                  theta=1e6, ffn=8, experts=1, top_k=1, runtime=runtime)
    runner_module.run_qwen4_exp_dense_qsa_layer(
        100, _layer_weights(), **kwargs, **{FIELD: True})
    runner_module.run_qwen4_exp_dense_qsa_layer(100, _layer_weights(), **kwargs)
    assert seen == [True, False]


# --- the owners around it are preserved -------------------------------------


def test_the_owner_and_its_fallthrough_are_still_registered(profiles) -> None:
    for variant in (F16_DOWN_VARIANT, STRICT_DOWN_VARIANT):
        assert callable(resolve(
            backend=QWEN4_EXP_BACKEND, layer="moe_linear", quant="gguf_q5_1",
            variant=variant)), variant
    # The tile-map helper the down-only configuration needs is untouched.
    assert callable(runner_module._run_qwen4_exp_q51_wmma_down)


# --- the diagnostics --------------------------------------------------------


def _load_campaign():
    return __import__(
        "scripts.qwen4exp_halo_box_campaign_ab", fromlist=["route_scope"])


def test_the_campaign_closes_the_owner_on_the_runner() -> None:
    campaign = _load_campaign()
    assert RETIRED_ENV not in campaign.Q51_PAIR2_PREEMPTING_ENV
    assert campaign.Q51_PAIR2_PREEMPTING_FIELDS == (FIELD,)
    assert RETIRED_ENV not in campaign.PF13_PREEMPTING_ENV

    owner = SimpleNamespace(q51_exact_grouped_down=True, **{FIELD: True})
    with campaign.q51_pair2_scope(runner=owner, environment={}):
        assert getattr(owner, FIELD) is False
    assert getattr(owner, FIELD) is True

    record = campaign.q51_pair2_control_record(owner, environment={})
    assert record["preempting_owner_fields_closed"] == {FIELD: False}
    assert record["preempting_owner_fields_previous"] == {FIELD: True}


def test_the_pf13_arm_closes_the_owner_in_both_arms() -> None:
    campaign = _load_campaign()
    runner = SimpleNamespace(
        q51_exact_grouped_down=True,
        q4_exact_grouped_gate=True, q4_exact_grouped_all=True,
        q51_down_m1=True, forkb_grouped_down=True,
        q51_fold128_prefill=True, q51_fold_pair_prefill=True,
        q51_register_cache=True, q51_row_publish=True,
        q8_down_row4_prefill=True, q8_down_bundle_prefill=True,
        q8_down_register=True, **{FIELD: True})
    environment = {"HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID": "64"}

    for mode in ("before", "after"):
        with campaign.route_scope("pf13", mode, runner, environment=environment):
            assert getattr(runner, FIELD) is False, mode
        assert getattr(runner, FIELD) is True, mode

    control = campaign.pf13_control_record(runner, environment=environment)
    assert control["preempting_owner_fields"] == {
        "bound_value": False,
        "published_value": {FIELD: True},
    }


def test_the_out8_route_binds_the_owner_on_the_runner() -> None:
    from scripts import qwen4exp_r2e_route_ablation as module

    bindings, envs, off = module.RUNNER_ROUTE_OFF_WITH_ENV["q51_out8"]
    assert (FIELD, False) in bindings
    assert RETIRED_ENV not in envs
    assert off == "0"


def test_the_layer2_down_arms_bind_the_owner_on_the_runner() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "qwen4exp_layer2_profile_gate",
        ROOT / "scripts" / "qwen4exp_layer2_profile_gate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for name in ("f16_compact_down", "mx1_full_route"):
        candidate = module.CANDIDATES[name]
        assert candidate.q51_wmma_down is True, name
        assert RETIRED_ENV not in candidate.environment, name
        # The two owners that take the down leg ahead of the F16 owner stay
        # process keys; the production base binds the routed F16 route on.
        assert candidate.environment["HIPENGINE_QWEN4_EXP_ROUTED_F16"] == "0"
        assert candidate.environment["HIPENGINE_QWEN4_EXP_T16_DOWN"] == "0"

    import dataclasses

    fields = {
        field.name: field for field in dataclasses.fields(module.CandidateSpec)}
    assert fields[FIELD].default is None
    source = inspect.getsource(module)
    assert f"candidate_spec.{FIELD} is not None" in source
    assert f"enter_context({FIELD}(" in source
