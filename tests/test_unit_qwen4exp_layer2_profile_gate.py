from __future__ import annotations

import importlib.util
from contextlib import ExitStack
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_layer2_profile_gate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen4exp_layer2_profile_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_decode_dp4a_candidate_selects_the_kernel_it_counts() -> None:
    candidate = _load_script().CANDIDATES["decode_dp4a_linear"]
    assert candidate.q4_dp4a_decode_enabled is True
    assert candidate.q4_dp4a_decode_layers == frozenset(
        (0, 2, 5, 6, 8, 9, 10, 11) + tuple(range(13, 48)))
    assert "HIPENGINE_QWEN4_EXP_Q4_DP4A64" not in candidate.environment
    assert "HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_GATE_UP" not in candidate.environment
    assert candidate.moe_decode_warp == frozenset({"DOWN"})
    assert candidate.candidate_key[-1] == (
        "selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out")
    assert candidate.count_registered_dispatch


def test_production_moe_restore_candidate_binds_the_owner_not_a_process_key() -> None:
    module = _load_script()
    candidate = module.CANDIDATES["production_moe_restore"]
    assert candidate.production_moe_prefill_enabled is True
    assert "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL" not in candidate.environment
    assert module._candidate_named_profile_intact(candidate) is False


@pytest.mark.parametrize(
    ("override", "initial", "expected"),
    [
        (True, False, True),    # on
        (False, True, False),   # off overrides an inherited true
        (False, False, False),  # explicit false
        (None, True, True),     # preserve an inherited true
        (None, False, False),   # preserve an inherited false
    ],
)
def test_candidate_production_moe_owner_is_captured_while_bound(
    override, initial, expected
) -> None:
    module = _load_script()
    generator = SimpleNamespace(
        production_moe_prefill_enabled=initial,
        runner=SimpleNamespace(production_moe_prefill_enabled=initial),
    )
    spec = SimpleNamespace(production_moe_prefill_enabled=override)
    stack = ExitStack()
    try:
        actual = module._bind_candidate_production_moe_prefill(
            stack, generator, spec)
    finally:
        stack.close()

    assert actual is expected
    # The report value is captured before the scope is restored.
    assert generator.production_moe_prefill_enabled is initial
    assert generator.runner.production_moe_prefill_enabled is initial


@pytest.mark.parametrize(
    ("override", "resolved", "expected"),
    [
        (True, True, True),
        (False, False, False),
        (None, True, True),
    ],
)
def test_candidate_owner_provenance_reports_the_bound_decision(
    override, resolved, expected
) -> None:
    module = _load_script()
    fields = module._candidate_owner_provenance(
        SimpleNamespace(production_moe_prefill_enabled=override), resolved)

    assert fields == {
        "production_moe_prefill_enabled": expected,
        "production_moe_prefill_override": override,
    }


@pytest.mark.parametrize(
    ("environment", "qsa_wmma", "override", "expected"),
    [
        ({}, None, None, True),
        ({}, None, True, False),
        ({}, None, False, False),
        ({"HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "1"}, None, None, False),
        ({}, "on", None, False),
    ],
)
def test_candidate_named_profile_intact_rejects_owner_overrides(
    environment, qsa_wmma, override, expected
) -> None:
    module = _load_script()
    spec = SimpleNamespace(
        environment=environment,
        qsa_wmma_enabled=qsa_wmma,
        production_moe_prefill_enabled=override,
    )

    assert module._candidate_named_profile_intact(spec) is expected


def test_json_value_thaws_nested_immutable_profile_manifest() -> None:
    module = _load_script()
    value = MappingProxyType({"rows": (1, MappingProxyType({"variant": "x"}))})

    assert module._json_value(value) == {"rows": [1, {"variant": "x"}]}


def test_state_summary_hashes_owned_state_and_metadata() -> None:
    module = _load_script()
    snapshot = SimpleNamespace(
        position=7,
        decode_state=SimpleNamespace(
            buffers={
                "gdn_matrix": np.asarray([1.0, 2.0], dtype=np.float32).view(np.uint8),
                "residual": np.asarray([0x3F80, 0x4000], dtype=np.uint16).view(np.uint8),
            }
        ),
        ple_hash_states={0: SimpleNamespace(tokens=(1, 2), next_position=7)},
    )
    runner = SimpleNamespace(
        snapshot=lambda: snapshot,
        attention_states=(
            SimpleNamespace(position_host=np.asarray([6]), context_host=np.asarray([7])),
        ),
        index_states=(SimpleNamespace(count=7, pooled_count=1),),
    )

    first = module._state_summary(runner)
    second = module._state_summary(runner)

    assert first == second
    assert first["finite"] is True
    assert first["position"] == 7
    assert first["buffer_bytes"] == {"gdn_matrix": 8, "residual": 4}
    assert first["attention_positions"] == [[6, 7]]
    assert first["index_counts"] == [[7, 1]]


def test_state_repeat_gate_requires_candidate_repeatability_and_layout() -> None:
    module = _load_script()
    metadata = {"position": 7, "attention_positions": [[6, 7]], "index_counts": [[7, 1]]}
    strict = [{
        "prompt_id": "p", "state_sha256": "s", "layout_sha256": "l",
        "finite": True, **metadata,
    }]
    candidate = [[
        {"prompt_id": "p", "state_sha256": "c", "layout_sha256": "l", "finite": True, **metadata},
        {"prompt_id": "p", "state_sha256": "c", "layout_sha256": "l", "finite": True, **metadata},
        {"prompt_id": "p", "state_sha256": "c", "layout_sha256": "l", "finite": True, **metadata},
    ]]

    passed = module._state_repeat_gate(strict, candidate)
    assert passed["passed"] is True
    assert passed["prompts"][0]["strict_candidate_state_exact"] is False

    candidate[0][2] = {**candidate[0][2], "state_sha256": "different"}
    failed = module._state_repeat_gate(strict, candidate)
    assert failed["passed"] is False
    assert failed["mismatches"][0]["repeat_exact"] is False


def test_compact_state_gate_binds_strict_candidate_equality() -> None:
    module = _load_script()
    metadata = {
        "position": 7,
        "attention_positions": [[6, 7]],
        "index_counts": [[7, 1]],
    }
    strict = [{
        "prompt_id": "p", "state_sha256": "s", "layout_sha256": "l",
        "finite": True, **metadata,
    }]
    candidate = [[
        {
            "prompt_id": "p", "state_sha256": "c", "layout_sha256": "l",
            "finite": True, **metadata,
        }
        for _ in range(3)
    ]]

    gate = module._state_repeat_gate(strict, candidate)
    gate["all_strict_candidate_state_exact"] = all(
        row["strict_candidate_state_exact"] for row in gate["prompts"]
    )
    gate["passed"] = gate["passed"] and gate["all_strict_candidate_state_exact"]

    assert gate["passed"] is False


def test_compact_free_trajectory_reuses_device_token_and_skips_hidden_copy(
    monkeypatch,
) -> None:
    module = _load_script()

    class Runner:
        def __init__(self) -> None:
            self.calls = []

        def reset(self) -> None:
            return None

        def prefill(self, token_ids, **kwargs):
            self.calls.append(("prefill", tuple(token_ids), kwargs))
            return SimpleNamespace(token_id=7)

        def step(self, token_id, **kwargs):
            self.calls.append(("step", int(token_id), kwargs))
            return SimpleNamespace(token_id=int(token_id) + 1)

    runner = Runner()
    tokenizer = SimpleNamespace(decode=lambda ids, skip_special: str(list(ids)))
    monkeypatch.setattr(module, "_state_summary", lambda _runner: {"state_sha256": "s"})

    result = module._free_trajectory(
        runner, tokenizer, [1, 2], 3, compact_output=True
    )

    assert result["ids"] == [7, 8, 9]
    assert runner.calls == [
        (
            "prefill", (1, 2),
            {"capture_logits": False, "capture_target_hidden": False},
        ),
        (
            "step", 7,
            {
                "capture_logits": False,
                "capture_target_hidden": False,
                "token_id_resident": True,
            },
        ),
        (
            "step", 8,
            {
                "capture_logits": False,
                "capture_target_hidden": False,
                "token_id_resident": True,
            },
        ),
    ]


def test_device_argmax_candidate_is_t0_and_fail_closed() -> None:
    module = _load_script()
    candidate = module.CANDIDATES["device_argmax"]

    assert candidate.environment == {}
    assert candidate.classification == "T0"
    assert candidate.base_profile == "strict"
    assert candidate.candidate_key[-1] == "top1_i64"
    assert candidate.compact_output is True


def test_gdn_isolation_changes_only_gdn_over_strict() -> None:
    module = _load_script()
    candidate = module.CANDIDATES["gdn_isolated"]
    assert candidate.base_profile == "strict"
    assert candidate.classification == "diagnostic"
    assert all(key.startswith("HIPENGINE_QWEN4_EXP_GDN_")
               for key in candidate.environment)
    assert candidate.environment["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS"] == (
        ",".join(map(str, range(27, 48)))
    )
    # The wave-norm admission is request-local runner state, so the arm names
    # the leaf explicitly instead of carrying a process-wide switch.
    assert "HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM" not in candidate.environment
    assert candidate.gdn_prefill_variant == "qwen4exp_sigmoid_wave_norm_prefill"
    assert candidate.candidate_key[-1] == "qwen4exp_gdn_tiled16_dpp_prefill"


def test_q8_fallback_preserves_other_production_choices() -> None:
    candidate = _load_script().CANDIDATES["production_q8_fallback"]
    assert candidate.base_profile == "production"
    assert candidate.environment == {
        "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN": "0",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "0",
    }


def test_conservative_repair_disables_arithmetic_not_exact_owners() -> None:
    candidate = _load_script().CANDIDATES["production_conservative"]
    assert candidate.base_profile == "production"
    assert set(candidate.environment.values()) == {"0"}
    for suffix in ("Q8_MMQ_PREFILL", "Q8_0_SELECTED_WMMA_DOWN",
                   "QSA_ORDERED_DECODE", "GDN_COLWARPS_PREFILL"):
        assert candidate.environment["HIPENGINE_QWEN4_EXP_" + suffix] == "0"
    assert candidate.q4_dp4a_decode_enabled is False
    assert candidate.q4_dp4a_decode_layers == frozenset()
    assert "HIPENGINE_QWEN4_EXP_Q4_DP4A64" not in candidate.environment
    for suffix in ("Q4_IU8_EXACT", "Q51_IU8_EXACT", "GDN_REGISTER_PREFILL",
                   "GDN_WAVE_NORM", "PLE_MAPPING_ACCESS"):
        assert "HIPENGINE_QWEN4_EXP_" + suffix not in candidate.environment


def test_ordered_v2_candidates_bind_the_runner_field() -> None:
    module = _load_script()
    for name in ("q8_blockscale_guarded_restore", "q8_blockscale_guarded_ordered"):
        candidate = module.CANDIDATES[name]
        assert candidate.qsa_ordered_decode_v2 is True
        # The leaf is request-local, so the arm names the field, not a switch.
        assert "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2" not in candidate.environment
        # The v1 route it rides is still an ordinary environment decision.
        assert candidate.environment["HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE"] == "1"
    # The conservative arm keeps the ordered v1 parent its env zeroing selects.
    assert module.CANDIDATES["production_conservative"].qsa_ordered_decode_v2 is False
    # A candidate that does not name it keeps the base profile's decision.
    assert module.CANDIDATES["q8_blockscale"].qsa_ordered_decode_v2 is None


def test_q8_mmq_attn_gate_candidate_is_explicit_and_fail_closed() -> None:
    module = _load_script()
    candidate = module.CANDIDATES["q8_mmq_attn_gate"]

    assert candidate.environment == {"HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE": "1"}
    assert candidate.classification == "T2"
    assert candidate.base_profile == "production"
    assert candidate.candidate_key[-1] == (
        "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"
    )
    assert candidate.fallback_key[-1] == "coltile8_rowbatch4_f32_f32_out"


def test_gfx1151_registry_contains_layer2_candidate_and_strict_fallback() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve

    register_gfx1151_kernels(replace=True)

    assert resolve(
        backend="hip_gfx1151",
        layer="moe_linear",
        quant="gguf_q5_k",
        variant="selected_wmma_prefill_compact_bf16_bf16_out",
    ) is not None
    assert resolve(
        backend="hip_gfx1151",
        layer="linear",
        quant="gguf_q5_k",
        variant="selected_gemv_bf16_bf16_out",
    ) is not None
    assert resolve(
        backend="hip_gfx1151",
        layer="linear",
        quant="gguf_q8_0",
        variant="mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out",
    ) is not None


def test_checked_dense_q8_candidate_pins_checked_owner_and_parent_fallback() -> None:
    module = _load_script()
    candidate = module.CANDIDATES["dense_q8_checked_full_rows"]

    assert candidate.classification == "T0"
    assert candidate.base_profile == "production"
    assert candidate.environment == {
        "HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION": "1",
    }
    assert candidate.candidate_key == (
        "hip_gfx1151",
        "linear",
        "gguf_q8_0",
        "coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out",
    )
    assert candidate.fallback_key == (
        "hip_gfx1151",
        "linear",
        "gguf_q8_0",
        "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out",
    )
    assert candidate.count_registered_dispatch is True

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    register_gfx1151_kernels(replace=True)
    assert is_registered(KernelKey(*candidate.candidate_key))
    assert is_registered(KernelKey(*candidate.fallback_key))
    assert resolve(
        backend=candidate.candidate_key[0],
        layer=candidate.candidate_key[1],
        quant=candidate.candidate_key[2],
        variant=candidate.candidate_key[3],
    ) is not resolve(
        backend=candidate.fallback_key[0],
        layer=candidate.fallback_key[1],
        quant=candidate.fallback_key[2],
        variant=candidate.fallback_key[3],
    )


def test_task_gate_requires_repeatability_and_flags_cross_route_divergence() -> None:
    module = _load_script()
    strict = {"a": {"ids": [1, 2], "ids_sha256": "s", "text": "strict"}}
    candidate = {
        "a": [
            {"ids": [1, 3], "ids_sha256": "c", "text": "candidate"},
            {"ids": [1, 3], "ids_sha256": "c", "text": "candidate"},
        ]
    }

    result = module._task_gate(strict, candidate, categories={"a": "general_en"})

    assert result["candidate_repeat_exact"] is True
    assert result["strict_exact_count"] == 0
    assert result["status"] == "requires_review"
    assert result["divergences"][0]["id"] == "a"


def test_gr_staged_candidate_is_absent_after_its_gate_rejected_it() -> None:
    """HC-1's GR staged-f32 route failed its gate, so the gate must not offer it.

    The whole-route gate ran with the long-form suite and rejected the candidate
    (``benchmarks/results/2026-09-29-qwen4exp-hc1-gr-staged-gate/gate-final.json``:
    ``status: failed``, ``measurement_valid: true``, 1920 counted dispatches,
    ``mixed_ja_en`` mean KL 1.499e-3 against a 1e-3 limit). The ledger's
    disposition for that outcome is removal of the selector and its candidate,
    not default-off. A re-added spec would bind an env var the runner no longer
    reads, so it would measure the recovery plan and call it the candidate.
    """

    module = _load_script()

    assert "production_gr_up_staged" not in module.CANDIDATES
    assert "recovery_plus_gr_staged_f32" not in module._F16_ROUTE_ARMS
    # No surviving arm may bind the removed selector: an arm that sets an
    # unread variable is a control wearing a candidate's name.
    for name, spec in module.CANDIDATES.items():
        assert "HIPENGINE_QWEN4_EXP_GR_STAGED" not in spec.environment, name


def test_layer2_grouped_q5k_candidate_is_absent_after_its_gate_rejected_it() -> None:
    """R0146: the grouped-WMMA Q5_K candidate failed its complete profile gate.

    ``benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p1-layer2-grouped-
    profile-rejected.json`` records ``status: rejected_correctness_default_off``
    with the binding ``prefill_last``/``prefill_to_c1`` mean KL 0.0011791
    against a 0.001 limit, and its ``next`` is a new exact/T1 Q5_K dataflow -
    now ``HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT``. The selector and its candidate
    are removed rather than default-off: a surviving spec would bind a variable
    the runner no longer reads and measure the recovery plan as the candidate.
    """

    module = _load_script()

    assert "layer2_grouped_q5k" not in module.CANDIDATES
    assert "layer2_grouped_q5k_wmma" not in {
        spec.name for spec in module.CANDIDATES.values()
    }
    for name, spec in module.CANDIDATES.items():
        assert "HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL" not in spec.environment, name
    # A bare run must still qualify a retained candidate, never a retired one.
    default = module.build_parser().parse_args(
        ["--model-root", "/tmp", "--output", "/tmp/gate.json"]
    ).candidate
    assert default in module.CANDIDATES
    assert default == "production_baseline"


def test_f16_compact_down_candidate_pins_down_only_overrides() -> None:
    """MX-1's down half must be gated as a down-only change.

    The candidate selects the F16 down owner by closing the exact grouped Q5_1
    down chain on the runner, opening the Q5_1 WMMA route, and closing the two
    owners that take the down leg ahead of it: the routed F16 route (which the
    production base binds on and which owns both legs before the Q5_1 branch)
    and the T16 Q5_1 down owner. The gate/up leg is selected by the independent
    ``exact_grouped_q4_gate`` predicate, which stays on when only the down chain
    is closed because ``EXACT_GROUPED_Q4_ALL`` keeps its production default. A
    gate/up override here would silently turn the run into the already-refuted
    full route, so none of those keys may appear.
    """

    module = _load_script()
    candidate = module.CANDIDATES["f16_compact_down"]

    assert candidate.classification == "T1"
    assert candidate.base_profile == "production"
    assert candidate.scenario_id == "qwen4exp-f16-compact-down"
    assert candidate.environment == {}
    assert candidate.routed_f16_enabled is False
    # The F16 owner's own admission is runner state, so the arm admits it
    # through the spec's field instead of a process key.
    assert candidate.q51_wmma_down is True
    # The chain gate is runner state, so the arm closes it through the spec's
    # field instead of a process key.
    assert candidate.q51_exact_grouped_down is False
    # The 27-47 production grouped window is left at its production binding: it
    # selects its own gate/up owner, and overriding it would move that window
    # out from under the arm's declared "gate/up left at production" control.
    assert "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL" not in candidate.environment
    for gate_up_key in (
        "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT",
    ):
        assert gate_up_key not in candidate.environment, gate_up_key
    # The gate/up chain's gate and joint term are runner fields, so "left at
    # production" means the spec does not override them.
    assert candidate.q4_exact_grouped_gate is None
    assert candidate.q4_exact_grouped_all is None


def test_mx1_full_route_candidate_closes_the_leg_preemptors() -> None:
    """MX-1's two compact owners must be the ones the run actually launches.

    The arm claims the F16-WMMA compact gate/up *and* down owners. The routed
    F16 route owns both legs ahead of the Q5_1 branch and the production base
    binds it on, so the arm has to close it or it measures the routed
    route on both legs while counting zero compact dispatches.
    """

    module = _load_script()
    candidate = module.CANDIDATES["mx1_full_route"]

    assert candidate.classification == "T1"
    assert candidate.base_profile == "production"
    assert candidate.environment == {}
    assert candidate.routed_f16_enabled is False
    # The gate/up half is the Q4 chain's own field (closed below) and the
    # Q5_1 WMMA down owner is the Q5_1 side's, so neither is a process key.
    assert candidate.q51_wmma_down is True
    assert candidate.q51_exact_grouped_down is False
    # Both chain gates are runner state, so the arm closes them through the
    # spec's fields instead of process keys.
    assert candidate.q4_exact_grouped_gate is False
    # The production gate/up selector this arm moves is the Q4 exact gate; both
    # chain gates are runner state, and neither the `_ALL` joint term nor the
    # iu8 exact selectors are overridden.
    for key in (
        "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT",
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN",
        # The 27-47 production grouped window already selects both compact
        # owners; moving it would be a second change the arm does not claim.
        "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL",
    ):
        assert key not in candidate.environment, key


def test_f16_compact_down_candidate_engages_direct_alias_and_keeps_fallback() -> None:
    """The down candidate must count the alias the runner actually calls.

    The runner invokes ``qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out``
    through a direct module-attribute call, so a registry-only dispatch count
    would read zero no matter how often the route engaged. The declared direct
    target must be that same module attribute, and the registered strict
    fallback (the production iu8-risk down owner) must stay resolvable.
    """

    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        register_qwen4_exp_q5_1_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import is_registered, KernelKey, resolve

    # The source keys are registered by a module-level call that does not re-run
    # once the module is imported, and ``tests/conftest.py`` restores the
    # registry to its collection-time baseline after every test. Calling the two
    # registrars explicitly keeps this test independent of collection order.
    register_qwen4_exp_q5_1_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    module = _load_script()
    spec = module.CANDIDATES["f16_compact_down"]

    assert spec.candidate_key == (
        "hip_gfx1151", "moe_linear", "gguf_q5_1",
        "selected_grouped_wmma_prefill_compact_bf16_bf16_out",
    )
    assert spec.fallback_key == (
        "hip_gfx1151", "moe_linear", "gguf_q5_1",
        "selected_wmma_iu8_risk_prefill_bf16_bf16_out",
    )
    assert is_registered(KernelKey(*spec.candidate_key))
    assert is_registered(KernelKey(*spec.fallback_key))

    assert spec.direct_dispatch_target == (
        "hipengine.runtime.qwen4_exp_runner",
        "qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out",
    )
    assert spec.direct_dispatch_reference == (
        "hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1",
        "qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out",
    )
    assert spec.count_registered_dispatch is True
    # The counted ABI is (compact_rows, num_experts, in_features, out_features),
    # matching qwen4_exp_runner.py's direct call site.
    assert spec.dispatch_shape_positions == (6, 7, 8, 9)

    from importlib import import_module

    called = getattr(import_module(spec.direct_dispatch_target[0]),
                     spec.direct_dispatch_target[1])
    defined = getattr(import_module(spec.direct_dispatch_reference[0]),
                      spec.direct_dispatch_reference[1])
    assert called is defined
    assert resolve(backend=spec.candidate_key[0], layer=spec.candidate_key[1],
                   quant=spec.candidate_key[2], variant=spec.candidate_key[3]) is defined
