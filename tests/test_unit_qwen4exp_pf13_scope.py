"""R0145: the PF-1 fork-b and Q5_1 down M1 selectors are runner-owned.

``HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN`` and
``HIPENGINE_QWEN4_EXP_PROFILE_Q5_1_DOWN_M1`` were the last two process-global
expert-down bridges. Both are request-local runner state now
(``runner.forkb_grouped_down`` / ``runner.q51_down_m1``): the binder publishes
the plan's decision, the MoE dispatch reads only that state, and a diagnostic
arm binds it around the measured prefill through the runner context managers.

The dispatch contracts below are source-level assertions because the selectors'
only effect is inside ``run_qwen4_exp_moe``; the branch arithmetic itself stays
covered by the device tests. The rest is behavioural: the binder publication,
the context-manager restore (including on an exception), and the campaign and
ablation diagnostics, which must bind the runner rather than a process key.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_profiles import (
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_runner

FORKB_ENV = "HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN"
Q5_M1_ENV = "HIPENGINE_QWEN4_EXP_PROFILE_Q5_1_DOWN_M1"
CONTROLS = (
    ("forkb_grouped_down", FORKB_ENV),
    ("q51_down_m1", Q5_M1_ENV),
)


def _resolve(profile: ExecutionProfile):
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant="gguf_ud_q4_k_xl",
        profile=profile,
    )


@pytest.fixture
def profiles(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(FORKB_ENV, raising=False)
    monkeypatch.delenv(Q5_M1_ENV, raising=False)
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    return None


# --- dispatch ownership -----------------------------------------------------


def test_the_dispatch_reads_both_decisions_from_parameters() -> None:
    parameters = inspect.signature(
        qwen4_exp_runner.run_qwen4_exp_moe).parameters

    for field, env in CONTROLS:
        assert field in parameters, field
        assert parameters[field].default is False, field
        assert env not in inspect.getsource(qwen4_exp_runner.run_qwen4_exp_moe)


def test_the_m1_variant_is_selected_by_the_runner_decision() -> None:
    source = inspect.getsource(qwen4_exp_runner.run_qwen4_exp_moe)

    assert (
        '"selected_grouped_prefill_compact_rowbatch8_out8_"\n'
        '                            "expertgrid64_m1_bf16_bf16_out"\n'
        "                            if q51_down_m1"
    ) in source
    # The preceding expertgrid64 owner stays the fallback, and the exact
    # expert-grid gate still decides whether that family is selected at all.
    assert "expertgrid64_bf16_bf16_out" in source
    assert 'HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID' in source


def test_the_forkb_branch_is_the_runner_decision() -> None:
    source = inspect.getsource(qwen4_exp_runner.run_qwen4_exp_moe)

    assert "elif forkb_grouped_down:" in source
    # The strict per-expert selected gemv stays registered below it.
    assert "Strict fallback" in source
    assert "selected_projection(" in source


@pytest.mark.parametrize("field,_env", CONTROLS)
def test_the_context_manager_restores_the_published_value(field, _env) -> None:
    runner = SimpleNamespace(**{field: True})
    manager = getattr(qwen4_exp_runner, field)

    with manager(runner, False) as prior:
        assert prior is True
        assert getattr(runner, field) is False
    assert getattr(runner, field) is True

    with pytest.raises(RuntimeError):
        with manager(runner, False):
            raise RuntimeError("measured run failed")
    assert getattr(runner, field) is True


@pytest.mark.parametrize("field,_env", CONTROLS)
def test_an_unset_scope_keeps_whatever_the_plan_published(field, _env) -> None:
    runner = SimpleNamespace(**{field: True})
    manager = getattr(qwen4_exp_runner, field)

    with manager(runner):
        assert getattr(runner, field) is True
    assert getattr(runner, field) is True


# --- binder publication -----------------------------------------------------


def test_the_production_binder_publishes_both_and_writes_no_process_key(
    profiles,
) -> None:
    production = _resolve(ExecutionProfile.PRODUCTION)
    assert production.binder is not None
    generator = SimpleNamespace()
    production.binder(generator, production)

    for field, env in CONTROLS:
        assert getattr(generator, field) is True, field
        assert env not in os.environ, env


def test_the_strict_binder_publishes_the_parents(profiles) -> None:
    strict = _resolve(ExecutionProfile.STRICT)
    assert strict.binder is not None
    generator = SimpleNamespace()
    strict.binder(generator, strict)

    for field, env in CONTROLS:
        assert getattr(generator, field) is False, field
        assert env not in os.environ, env


# --- diagnostics ------------------------------------------------------------


#: A shipped-lane environment: the grouped chain's gates are on, every owner
#: that preempts either leaf is published, and every replacement sibling is
#: published on the runner.
def _published_environment() -> dict[str, str]:
    return {
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4": "1",
        "HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL": "1",
        "HIPENGINE_QWEN4_EXP_ROUTED_F16": "1",
        "HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT": "1",
        "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN": "1",
        "HIPENGINE_QWEN4_EXP_Q8_0_GROUPED_WMMA": "0",
    }


def _published_runner() -> SimpleNamespace:
    return SimpleNamespace(
        q51_exact_grouped_down=True,
        q51_down_m1=True, forkb_grouped_down=True,
        q51_fold128_prefill=True, q51_fold_pair_prefill=True,
        q51_register_cache=True, q51_row_publish=True,
        q8_down_row4_prefill=True, q8_down_bundle_prefill=True,
        q8_down_register=True)


def test_the_pf13_arm_isolates_both_leaves_and_restores_everything() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import (
        PF13_FIELDS,
        PF13_PREEMPTING_ENV,
        PF13_SIBLING_FIELDS,
        Q51_PAIR2_PREEMPTING_ENV,
        Q8_DOWN_PREEMPTING_ENV,
        route_scope,
    )

    # Both chains' own preemptor sets are covered: the Q8_0 side includes the
    # WMMA down owner, which is what actually preempts fork-b on the shipped
    # lane, and the Q5_1 side includes the exact iu8 chain.
    assert set(Q51_PAIR2_PREEMPTING_ENV) <= set(PF13_PREEMPTING_ENV)
    assert set(Q8_DOWN_PREEMPTING_ENV) <= set(PF13_PREEMPTING_ENV)
    assert len(set(PF13_PREEMPTING_ENV)) == len(PF13_PREEMPTING_ENV)

    environment = _published_environment()
    before = dict(environment)
    runner = _published_runner()

    with route_scope("pf13", "before", runner, environment=environment):
        assert runner.q51_down_m1 is False
        assert runner.forkb_grouped_down is False
        # Every replacement sibling is held off, so the counted variants are the
        # leaves themselves and not a ladder rung that would replace them.
        for field in PF13_SIBLING_FIELDS:
            assert getattr(runner, field) is False, field
        # Every preempting owner is closed, including the pair2 selection the M1
        # leaf sits below and the two grouped Q8_0 owners below fork-b.
        assert all(environment[key] == "0" for key in PF13_PREEMPTING_ENV)
        # The chain's own gates are prerequisites, never forced: the Q5_1 chain
        # gate is runner state and the Q4 gate a process key.
        assert runner.q51_exact_grouped_down is True
        assert environment["HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4"] == "1"

    assert environment == before
    assert runner.q51_down_m1 is True
    assert runner.forkb_grouped_down is True
    for field in PF13_SIBLING_FIELDS:
        assert getattr(runner, field) is True, field

    with route_scope("pf13", "after", runner, environment=environment):
        assert runner.q51_down_m1 is True
        assert runner.forkb_grouped_down is True
        for field in PF13_SIBLING_FIELDS:
            assert getattr(runner, field) is False, field
        assert all(environment[key] == "0" for key in PF13_PREEMPTING_ENV)

    assert environment == before
    for field in PF13_SIBLING_FIELDS:
        assert getattr(runner, field) is True, field
    assert PF13_FIELDS == tuple(field for field, _env in CONTROLS)


def test_the_pf13_arm_restores_after_an_exception() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import (
        PF13_SIBLING_FIELDS,
        route_scope,
    )

    environment = _published_environment()
    before = dict(environment)
    runner = _published_runner()
    with pytest.raises(RuntimeError, match="measured run failed"):
        with route_scope("pf13", "after", runner, environment=environment):
            raise RuntimeError("measured run failed")
    assert environment == before
    assert runner.q51_down_m1 is True
    assert runner.forkb_grouped_down is True
    for field in PF13_SIBLING_FIELDS:
        assert getattr(runner, field) is True, field


def test_the_pf13_arm_refuses_a_closed_grouped_gate() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import (
        PF13_GATE_ENV,
        PF13_GATE_FIELD,
        route_scope,
    )

    # The Q5_1 chain gate is runner state: an arm that measures with it closed
    # would count zero on both leaves, so the scope refuses.
    runner = _published_runner()
    runner.q51_exact_grouped_down = False
    environment = _published_environment()
    with pytest.raises(ValueError, match=PF13_GATE_FIELD):
        with route_scope("pf13", "after", runner, environment=environment):
            pass
    assert runner.q51_down_m1 is True
    assert runner.forkb_grouped_down is True
    assert runner.q51_exact_grouped_down is False

    for key in PF13_GATE_ENV:
        environment = _published_environment()
        environment[key] = "0"
        runner = _published_runner()
        with pytest.raises(ValueError, match=key):
            with route_scope("pf13", "after", runner, environment=environment):
                pass
        # The refusal happens before anything is bound.
        assert runner.q51_down_m1 is True
        assert runner.forkb_grouped_down is True
        assert environment[key] == "0"


def test_the_pf13_arm_refuses_a_foreign_expert_grid_family() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import (
        PF13_EXPERT_GRID_ENV, route_scope)

    environment = _published_environment()
    environment[PF13_EXPERT_GRID_ENV] = "32"
    runner = _published_runner()
    with pytest.raises(ValueError, match=PF13_EXPERT_GRID_ENV):
        with route_scope("pf13", "after", runner, environment=environment):
            pass
    assert runner.q51_down_m1 is True


def test_the_pf13_arm_refuses_to_measure_without_its_runner() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope

    with pytest.raises(ValueError, match="needs the owning runner"):
        route_scope("pf13", "after", None, environment=_published_environment())


def test_the_pf13_control_records_every_forced_value_and_its_prior() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import (
        PF13_GATE_ENV,
        PF13_GATE_FIELD,
        PF13_LEAF_KEYS,
        PF13_PREEMPTING_ENV,
        PF13_SIBLING_FIELDS,
        pf13_control_record,
    )

    environment = _published_environment()
    runner = _published_runner()
    control = pf13_control_record(runner, environment=environment)

    assert control["runner_fields"] == {
        field: True for field, _env in CONTROLS}
    assert control["bound"] == {"before": False, "after": True}
    assert control["scope"] == "per measured run; restored on exit"
    assert control["replacement_siblings"] == {
        "bound_value": False,
        "published_value": {field: True for field in PF13_SIBLING_FIELDS},
    }
    assert control["preempting_owners"] == {
        "forced_value": "0",
        "previous": {
            key: environment.get(key) for key in PF13_PREEMPTING_ENV},
    }
    assert control["gates"] == {
        "runner_field": {
            "field": PF13_GATE_FIELD,
            "published_value": True,
        },
        "environment": {
            key: environment[key] for key in PF13_GATE_ENV},
    }
    assert control["expert_grid"] == {"published": None, "effective": "64"}
    assert control["counted_leaves"] == {
        name: {"layer": layer, "quant": quant, "variant": variant}
        for name, (layer, quant, variant) in PF13_LEAF_KEYS.items()
    }
    # Recording changes nothing.
    assert runner.q51_down_m1 is True
    assert environment == _published_environment()


def test_the_ablation_binds_the_runner_fields_instead_of_env_keys() -> None:
    from scripts import qwen4exp_r2e_route_ablation as ablation

    assert ablation.RUNNER_ROUTE_OFF == {
        "q51_down_m1": "q51_down_m1",
        "forkb_grouped_down": "forkb_grouped_down",
    }
    # Neither route may keep an env entry: the runner reads only its own state,
    # so an env write would ablate nothing and report the unmodified arm.
    for route in ablation.RUNNER_ROUTE_OFF:
        assert route not in ablation.ROUTES
        assert route not in ablation.RUNNER_ROUTE_MODES
    assert set(ablation.RUNNER_ROUTE_OFF) <= {
        route for route in ablation.RUNNER_ROUTE_OFF}


# --- measurement path: the real main() over CPU stand-ins -------------------
#
# These drive ``scripts/qwen4exp_halo_box_campaign_ab.py:main`` end to end for
# the ``pf13`` package. The device surface is replaced, so what is under test is
# the measurement path itself: the per-leaf counters are registered on the two
# named variants, both arms close the preemptors and hold the siblings off
# around the measured window only, the per-row expectation is derived from the
# artifact's quant split, and the counters are restored afterwards.

#: Two Q5_1 expert-down layers and three Q8_0 ones in the fake artifact; the
#: fixture case is 2048 tokens at chunk 1024, so each leaf launches once per
#: layer per chunk.
M1_LAUNCHES = 4
FORKB_LAUNCHES = 6
FIXTURE_CASE = {
    "id": "code-p2048",
    "category": "code",
    "prompt_tokens": 2048,
    "prompt_token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
    "prompt_token_ids_sha256": "prompt-code-p2048",
}


def _fake_resident() -> SimpleNamespace:
    weights = {}
    for index in range(2):
        weights[f"layers.{index}.expert_down"] = SimpleNamespace(
            spec=SimpleNamespace(quant_key="gguf_q5_1"))
    for index in range(2, 5):
        weights[f"layers.{index}.expert_down"] = SimpleNamespace(
            spec=SimpleNamespace(quant_key="gguf_q8_0"))
    return SimpleNamespace(device_weights=weights)


def _leaf_key(name: str):
    from hipengine.kernels.registry import KernelKey
    from scripts.qwen4exp_halo_box_campaign_ab import PF13_LEAF_KEYS

    layer, quant, variant = PF13_LEAF_KEYS[name]
    return KernelKey("hip_gfx1151", layer, quant, variant)


def _sample_row(case, repetition: int) -> dict:
    tokens = int(case["prompt_tokens"])
    return {
        "case_id": str(case["id"]),
        "category": str(case["category"]),
        "prompt_tokens": tokens,
        "prompt_token_ids_sha256": str(case["prompt_token_ids_sha256"]),
        "repetition": int(repetition),
        "prefill_ms": 10.0,
        "prefill_tok_s": 1000.0 * tokens / 10.0,
        "decode_ms": 20.0,
        "decode_transitions": 8,
        "decode_tok_s": 400.0,
        "client_wall_s": 0.03,
        "output_token_count": 9,
        "output_token_ids": [1] * 9,
        "output_token_ids_sha256": "same",
    }


def _install_campaign_fakes(monkeypatch, module, *, launches, case=None,
                            launch_unbound=False):
    """Patch everything ``main`` touches outside the measurement path itself."""

    from hipengine.core import memory as memory_module
    from hipengine.execution_profiles import ExecutionProfile  # noqa: F401
    from hipengine.generation import qwen4_exp_gguf as generator_module
    from hipengine.generation import qwen4_exp_profiles as profiles_module
    from hipengine.kernels import hip_gfx1151 as kernels_module
    from hipengine.kernels.hip_gfx1100.quant import (
        gguf_q5_k_q8_1_selected_prefill as q5k_module,
    )
    from hipengine import models as models_module
    from hipengine.loading import gguf as gguf_module
    from hipengine.kernels.registry import register, resolve

    # The shipped lane's process bindings, as the production binder leaves them.
    for key, value in _published_environment().items():
        monkeypatch.setenv(key, value)
    case = dict(case or FIXTURE_CASE)
    fixture = {"cases": [case], "decode_transitions": 8, "model": "fake"}
    stubs = {}
    for name in launches:
        key = _leaf_key(name)
        stub = lambda *a, **k: None  # noqa: E731 - identity is what we check
        register(key, stub, replace=True)
        stubs[name] = stub

    runner = SimpleNamespace(
        backend="hip_gfx1151",
        config=SimpleNamespace(expert_feed_forward_length=512),
        resident=_fake_resident(),
        prefill_chunk_size=1024,
        q51_exact_grouped_down=True,
        q51_down_m1=True,
        forkb_grouped_down=True,
        q51_fold128_prefill=True,
        q51_fold_pair_prefill=True,
        q51_register_cache=True,
        q51_row_publish=True,
        q8_down_row4_prefill=True,
        q8_down_bundle_prefill=True,
        q8_down_register=True,
    )
    generator = SimpleNamespace(runner=runner, close=lambda: None)
    resolved = SimpleNamespace(
        manifest_sha256="production-manifest",
        strict_manifest_sha256="strict-manifest",
        fell_back_to_strict=False,
        construct_generator=lambda factory: generator,
    )

    from scripts.qwen4exp_halo_box_campaign_ab import (
        PF13_PREEMPTING_ENV,
        PF13_SIBLING_FIELDS,
    )

    seen: list[dict] = []

    def sample(runner_, *, case, repetition, transitions):
        seen.append(
            {
                "m1": runner_.q51_down_m1,
                "forkb": runner_.forkb_grouped_down,
                "siblings": {
                    field: getattr(runner_, field)
                    for field in PF13_SIBLING_FIELDS
                },
                "preemptors": {
                    key: os.environ.get(key) for key in PF13_PREEMPTING_ENV
                },
                "gates": {
                    "q51_exact_grouped_down": runner_.q51_exact_grouped_down,
                    "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4": os.environ.get(
                        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4"),
                },
            }
        )
        if launch_unbound or (runner_.q51_down_m1 and runner_.forkb_grouped_down):
            for name, count in launches.items():
                key = _leaf_key(name)
                launch = resolve(backend=key.backend, layer=key.layer,
                                 quant=key.quant, variant=key.variant)
                for _ in range(count):
                    launch()
        return _sample_row(case, repetition)

    monkeypatch.setattr(module, "_git_metadata", lambda root: {
        "tracked_clean": True, "commit": "deadbeef", "dirty": False})
    monkeypatch.setattr(module, "_host_metadata", lambda: {})
    monkeypatch.setattr(module, "load_fixture", lambda path: (fixture, "sha"))
    monkeypatch.setattr(module, "_hipengine_case_sample", sample)
    monkeypatch.setattr(memory_module, "memory_stats", lambda: {})
    monkeypatch.setattr(
        "hipengine.execution_profiles.resolve_runtime_profile",
        lambda **kwargs: resolved)
    monkeypatch.setattr(kernels_module, "register_gfx1151_kernels",
                        lambda **kwargs: None)
    monkeypatch.setattr(profiles_module, "register_qwen4_exp_gfx1151_profiles",
                        lambda: None)
    monkeypatch.setattr(
        q5k_module, "register_gguf_q5_k_q8_1_selected_prefill_kernels",
        lambda **kwargs: None)
    monkeypatch.setattr(gguf_module, "discover_gguf_files",
                        lambda root: [Path("fake.gguf")])
    monkeypatch.setattr(gguf_module, "load_gguf_index",
                        lambda path: SimpleNamespace(architecture="fake"))
    monkeypatch.setattr(models_module, "resolve_model", lambda arch: None)
    monkeypatch.setattr(generator_module, "Qwen4ExpGGUFTextGenerator",
                        lambda **kwargs: SimpleNamespace())
    return SimpleNamespace(runner=runner, seen=seen, stubs=stubs)


def _run_pf13_main(monkeypatch, tmp_path, *, launches, case=None,
                   launch_unbound=False):
    from scripts import qwen4exp_halo_box_campaign_ab as module

    fakes = _install_campaign_fakes(
        monkeypatch, module, launches=launches, case=case,
        launch_unbound=launch_unbound)
    output = tmp_path / "pf13.json"
    code = module.main([
        "--model-root", str(tmp_path),
        "--output", str(output),
        "--route-package", "pf13",
        "--screen-only",
        "--repetitions-per-mode", "1",
        "--prefill-chunk-size", "1024",
    ])
    return code, json.loads(output.read_text()), fakes


def test_main_counts_both_leaves_and_records_the_isolation(
        monkeypatch, tmp_path) -> None:
    code, artifact, fakes = _run_pf13_main(
        monkeypatch, tmp_path,
        launches={"q51_down_m1": M1_LAUNCHES,
                  "forkb_grouped_down": FORKB_LAUNCHES})

    assert code == 0
    assert artifact["status"] == "completed"
    by_mode = {row["mode"]: row for row in artifact["samples"]}
    assert by_mode["after"]["leaf_calls"] == {
        "q51_down_m1": M1_LAUNCHES,
        "forkb_grouped_down": FORKB_LAUNCHES,
    }
    assert by_mode["before"]["leaf_calls"] == {
        "q51_down_m1": 0,
        "forkb_grouped_down": 0,
    }
    # The per-row count is the artifact's own quant split times the chunks that
    # clear the grouped guard, so the expectation and the observed launches are
    # derived independently.
    assert by_mode["after"]["candidate_calls"] == M1_LAUNCHES + FORKB_LAUNCHES
    assert by_mode["before"]["candidate_calls"] == 0

    control = artifact["selection_control"]
    assert control["bound"] == {"before": False, "after": True}
    assert control["scope"] == "per measured run; restored on exit"
    assert control["counted_leaves"] == {
        name: {"layer": layer, "quant": quant, "variant": variant}
        for name, (layer, quant, variant) in __import__(
            "scripts.qwen4exp_halo_box_campaign_ab",
            fromlist=["PF13_LEAF_KEYS"]).PF13_LEAF_KEYS.items()
    }
    assert control["replacement_siblings"]["bound_value"] is False
    assert control["preempting_owners"]["forced_value"] == "0"
    assert control["gates"] == {
        "runner_field": {
            "field": "q51_exact_grouped_down",
            "published_value": True,
        },
        "environment": {"HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4": "1"},
    }
    assert control["expert_grid"] == {"published": None, "effective": "64"}
    # The artifact states the two owners each arm measures.
    assert artifact["arms"]["after"] == {
        "q5_1_down": "selected_grouped_prefill_compact_rowbatch8_out8_"
                     "expertgrid64_m1_bf16_bf16_out",
        "q8_0_down": "selected_grouped_gemv_bf16_bf16_out",
    }
    assert artifact["arms"]["before"]["q8_0_down"] == (
        "selected_gemv_bf16_bf16_out")

    # Every measured call saw the isolation: both leaves bound by arm, every
    # sibling off, every preemptor closed, both gates untouched.
    assert fakes.seen
    for call in fakes.seen:
        assert all(value is False for value in call["siblings"].values())
        assert all(value == "0" for value in call["preemptors"].values())
        assert call["gates"] == {
            "q51_exact_grouped_down": True,
            "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4": "1",
        }
    assert any(call["m1"] and call["forkb"] for call in fakes.seen)
    assert any(not call["m1"] and not call["forkb"] for call in fakes.seen)

    # The counters and the runner bindings are restored after the run.
    for name, stub in fakes.stubs.items():
        key = _leaf_key(name)
        assert __import__("hipengine.kernels.registry", fromlist=["resolve"]) \
            .resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                     variant=key.variant) is stub
    assert fakes.runner.q51_down_m1 is True
    assert fakes.runner.forkb_grouped_down is True
    assert fakes.runner.q51_row_publish is True
    assert os.environ["HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL"] == "1"


def test_main_refuses_a_leaf_that_does_not_engage_the_enabled_arm(
        monkeypatch, tmp_path) -> None:
    with pytest.raises(
            AssertionError,
            match=r"pf13 q51_down_m1 calls 0 != 4 \(after arm\)"):
        _run_pf13_main(
            monkeypatch, tmp_path,
            launches={"q51_down_m1": 0, "forkb_grouped_down": FORKB_LAUNCHES})


def test_main_refuses_a_leaf_that_launches_in_the_parent_arm(
        monkeypatch, tmp_path) -> None:
    # A leaf that runs in both arms is the failure this guards: the package
    # would report a difference it does not have.
    with pytest.raises(
            AssertionError,
            match=r"pf13 q51_down_m1 calls 4 != 0 \(before arm\)"):
        _run_pf13_main(
            monkeypatch, tmp_path,
            launches={"q51_down_m1": M1_LAUNCHES,
                      "forkb_grouped_down": FORKB_LAUNCHES},
            launch_unbound=True)


def test_main_refuses_a_short_row_that_cannot_engage_a_leaf(
        monkeypatch, tmp_path) -> None:
    case = {**FIXTURE_CASE, "id": "code-p1", "prompt_tokens": 1}
    with pytest.raises(AssertionError, match="cannot be verified by a zero"):
        _run_pf13_main(
            monkeypatch, tmp_path,
            launches={"q51_down_m1": 0, "forkb_grouped_down": 0},
            case=case)
