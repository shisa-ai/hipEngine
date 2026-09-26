from __future__ import annotations

import inspect

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner


def test_p10_x1_decode_repack_does_not_change_linear_attention_math() -> None:
    """T16 materialization must not conditionally switch linear-attention math.

    P10.X1 found that guarding the linear-attention ``ssm_out`` path on
    ``gguf_decode_repack_enabled()`` changed the activation contract from
    BF16-input Q8_0 GEMV to F32-input Q8T16 GEMV.  That was faster-looking but
    not equivalent enough for MoE routing.  The current route starts from F32
    and may use the separately selected GDN output cast; that selection remains
    independent of decode-repack. Decode repack selects weight layout / kernel
    implementation, not the math graph.
    """

    source = inspect.getsource(Qwen35GGUFFullStackRunner._run_linear_attention_attn_only)

    assert "gguf_decode_repack_enabled" not in source
    assert "scratch.recurrent_out.ptr" in source
    assert "ssm_out_activation_dtype = GGUF_ACTIVATION_F32" in source
    assert "output_cast = self._gdn_decode_output_cast_for_weight(ssm_out_weight)" in source
    assert "activation_dtype=ssm_out_activation_dtype" in source


def test_p10_x1_decode_repack_does_not_switch_full_attention_math() -> None:
    """The T16 flag must not choose a different full-attention decode graph.

    GGUF may now choose split-K full-attention decode by *context length*, but
    P10.X1 still forbids changing the graph merely because T16 decode-repack is
    enabled.
    """

    source = inspect.getsource(Qwen35GGUFFullStackRunner._run_full_attention_attn_only)

    assert "gguf_decode_repack_enabled" not in source
    assert "_use_gguf_full_attention_split_decode" in source
    assert "gguf_qwen35_head_rmsnorm_partial_rotary_position_key_bf16_f32_weight" not in source


# ---------------------------------------------------------------------------
# E3 (UD-GFX1151-OPTIMIZE2 C2/H3): admission/report truth must equal the
# production planner. Per-tensor eligibility (UD-U3, shipped default) keeps
# eligible tensors repacking on a raw-IQ carrier; only 'model-wide' mode
# vetoes the whole model, and only an explicit HIPENGINE_GGUF_DECODE_REPACK
# off disables repack.
# ---------------------------------------------------------------------------

import importlib.util as _ilu  # noqa: E402
from pathlib import Path  # noqa: E402

from hipengine.loading.qwen35_gguf_admission import (  # noqa: E402
    preflight_qwen35_gguf_artifact,
)
from hipengine.loading.qwen35_gguf_materialize import (  # noqa: E402
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_RAW_GGUF,
    plan_qwen35_gguf_materialization,
)
from hipengine.quant.gguf import GGMLQuantizationType  # noqa: E402


def _raw_iq_carrier_map():
    """Synthetic dense map whose attn_qkv carries a raw-IQ type (IQ3_XXS)."""

    spec = _ilu.spec_from_file_location(
        "_ud_admission_test_helpers",
        Path(__file__).with_name("test_live_gguf_ud_admission.py"),
    )
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._synthetic_model_map(
        attn_qkv_type=GGMLQuantizationType.IQ3_XXS
    )


def test_e3_admission_default_decode_repack_matches_planner(monkeypatch) -> None:
    """Under default policy admission must report what the planner plans.

    Production calls preflight_qwen35_gguf_artifact() with decode_repack=None
    and no repack_veto. The planner resolves None through the env default
    (on) and, in per-tensor eligibility mode, repacks every eligible tensor
    while vetoing only the raw-IQ ones per tensor. The admission plan flags
    must equal that truth, or route-audit surfaces record repack=OFF for
    files the runtime loads with T16 routes (E1's live-vs-audit divergence).
    """

    monkeypatch.delenv("HIPENGINE_UD_REPACK_ELIGIBILITY", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_REPACK", raising=False)
    model_map = _raw_iq_carrier_map()

    report = preflight_qwen35_gguf_artifact(model_map, backend="hip_gfx1100")
    plan = plan_qwen35_gguf_materialization(model_map)

    # Planner truth (already green today): per-tensor mode keeps the Q6_K
    # lm_head on T16 and leaves the raw-IQ attn_qkv on its raw layout.
    assert plan.root_specs["lm_head"].layout == LAYOUT_GGUF_Q6_K_T16
    layer0 = plan.layer_specs[0]
    attn_qkv = next(
        spec for name, spec in layer0.items() if name.endswith("attn_qkv")
    )
    assert attn_qkv.layout == LAYOUT_RAW_GGUF

    # Admission must agree: the model-level repack flag stays open on a
    # raw-IQ carrier under the shipped per-tensor default.
    assert report.plan_contract.decode_repack is True


def test_e3_admission_honors_modelwide_eligibility_mode(monkeypatch) -> None:
    """The documented rollback seam (model-wide) must veto the whole model."""

    monkeypatch.setenv("HIPENGINE_UD_REPACK_ELIGIBILITY", "model-wide")
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_REPACK", raising=False)
    model_map = _raw_iq_carrier_map()

    report = preflight_qwen35_gguf_artifact(model_map, backend="hip_gfx1100")
    plan = plan_qwen35_gguf_materialization(model_map)

    assert report.plan_contract.decode_repack is False
    assert plan.root_specs["lm_head"].layout == LAYOUT_RAW_GGUF


def test_e3_admission_honors_decode_repack_env_off(monkeypatch) -> None:
    """HIPENGINE_GGUF_DECODE_REPACK=0 keeps the model-level flag closed."""

    monkeypatch.delenv("HIPENGINE_UD_REPACK_ELIGIBILITY", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "0")
    model_map = _raw_iq_carrier_map()

    report = preflight_qwen35_gguf_artifact(model_map, backend="hip_gfx1100")

    assert report.plan_contract.decode_repack is False
