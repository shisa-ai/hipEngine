from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_layer2_profile_gate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen4exp_layer2_profile_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
                   "Q4_DP4A64", "QSA_ORDERED_DECODE", "GDN_COLWARPS_PREFILL"):
        assert candidate.environment["HIPENGINE_QWEN4_EXP_" + suffix] == "0"
    for suffix in ("Q4_IU8_EXACT", "Q51_IU8_EXACT", "GDN_REGISTER_PREFILL",
                   "GDN_WAVE_NORM", "PLE_MAPPING_ACCESS"):
        assert "HIPENGINE_QWEN4_EXP_" + suffix not in candidate.environment


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


def test_gr_up_staged_candidate_targets_the_registered_hc1_variant() -> None:
    """HC-1's gate candidate must name the variant the kernel module registers.

    The staged-f32 GR-up tile is default-off and not bit-exact with its exact
    parent, so the production gate is the only thing that can select it. A
    rename on either side would silently turn the gate into a control: the
    candidate key would fail to resolve, or the counted dispatch would no
    longer be the declared implementation.
    """

    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        register_gguf_k_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_gr_staged_f32 import (
        GR_STAGED_F32_VARIANT,
        register_gr_staged_f32_kernel,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import is_registered, KernelKey

    # Both source keys are registered by module-level calls that do not re-run
    # once the module is imported, and ``tests/conftest.py`` restores the
    # registry to its collection-time baseline after every test. Calling the
    # two module registrars explicitly is what a fresh process gets from the
    # imports, and it keeps this test independent of collection order.
    register_gguf_k_gemv_kernels(replace=True)
    register_gr_staged_f32_kernel(replace=True)
    register_gfx1151_kernels(replace=True)

    module = _load_script()
    spec = module.CANDIDATES["production_gr_up_staged"]

    assert spec.classification == "T1"
    # The variant name lives in the kernel module, not in this script, so a
    # rename there must fail here rather than silently retarget the gate.
    assert spec.candidate_key == (
        "hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0", GR_STAGED_F32_VARIANT,
    )
    assert spec.fallback_key == (
        "hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
        "coltile2_branch4_rowbatch4_f32_exact",
    )
    assert is_registered(KernelKey(*spec.candidate_key))
    assert is_registered(KernelKey(*spec.fallback_key))

    # The candidate must turn the competing GR routes off, or the gate would
    # measure whichever selector wins rather than the staged tile.
    assert spec.environment["HIPENGINE_QWEN4_EXP_GR_STAGED"] == "1"
    for competing in (
        "HIPENGINE_QWEN4_EXP_GR_WIDE",
        "HIPENGINE_QWEN4_EXP_GR_IU8",
        "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN",
    ):
        assert spec.environment[competing] == "0", competing

    # The counted dispatch must be the declared implementation. This is the
    # check that catches a rename in either module, and it is what the gate
    # relies on to prove the route engaged rather than merely being selected.
    from importlib import import_module

    called = getattr(import_module(spec.direct_dispatch_target[0]),
                     spec.direct_dispatch_target[1])
    defined = getattr(import_module(spec.direct_dispatch_reference[0]),
                      spec.direct_dispatch_reference[1])
    assert called is defined


def test_f16_compact_down_candidate_pins_down_only_overrides() -> None:
    """MX-1's down half must be gated as a down-only change.

    The candidate selects the F16 down owner by clearing the exact-grouped-down
    gate and opening the Q5_1 WMMA route. The gate/up leg is selected by the
    independent ``exact_grouped_q4_gate`` predicate, which stays on when only
    ``EXACT_GROUPED_DOWN`` is cleared because ``EXACT_GROUPED_Q4_ALL`` keeps its
    production default. A gate/up override here would silently turn the run into
    the already-refuted full route, so none of those keys may appear.
    """

    module = _load_script()
    candidate = module.CANDIDATES["f16_compact_down"]

    assert candidate.classification == "T1"
    assert candidate.base_profile == "production"
    assert candidate.scenario_id == "qwen4exp-f16-compact-down"
    assert candidate.environment == {
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN": "0",
        "HIPENGINE_QWEN4_EXP_Q5_1_WMMA": "1",
    }
    for gate_up_key in (
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4",
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4_ALL",
        "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT",
    ):
        assert gate_up_key not in candidate.environment, gate_up_key


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
