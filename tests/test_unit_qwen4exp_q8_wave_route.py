import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear
from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode


def test_wave_route_is_registry_checked_and_preserves_other_variants(monkeypatch):
    base = linear.GGUFLinearDispatch(
        KernelKey("hip_gfx1151", "linear", "gguf_q8_0", "coltile8_rowbatch4_f32_f32_out"), "raw"
    )
    select = linear._raw_k_wave_scale_dispatch
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    assert select(base, enabled=False) is base
    assert select(base, enabled=True).key.variant == "coltile8_rowbatch4_wave_scale_f32_f32_out"
    other = linear.GGUFLinearDispatch(KernelKey("hip_gfx1151", "linear", "gguf_q8_0", "mmq"), "raw")
    assert select(other, enabled=True) is other
    monkeypatch.setattr(linear, "is_registered", lambda key: False)
    assert select(base, enabled=True) is base


@pytest.mark.parametrize("mode,value", [("before", "0"), ("after", "1")])
def test_ab_changes_only_wave_flag(mode, value):
    env = {"other": "unchanged"}
    _apply_mode(mode, environment=env, route_package="q8-wave-scale")
    assert env == {"other": "unchanged", "HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE": value}


def test_pipelined_upgrade_is_registry_checked_and_leaf_only(monkeypatch):
    wave = linear.GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1151", "linear", "gguf_q8_0",
            "coltile8_rowbatch4_wave_scale_f32_f32_out",
        ),
        "raw",
    )
    upgrade = linear._raw_k_coltile_pipelined_dispatch
    # rows=1 (decode) is outside the checked full-row window, so this test
    # exercises the DPP/pipelined chain exactly as before that variant existed.
    shape = {"rows": 1, "in_features": 2560, "out_features": 10240}
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION", raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_DPP_CHECKED_FULL_ROWS", raising=False)
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    assert upgrade(wave, **shape).key.variant == (
        "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
    )
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION", "0")
    assert upgrade(wave, **shape).key.variant == (
        "coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out"
    )
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION", raising=False)
    monkeypatch.setattr(linear, "is_registered", lambda key: "_dpp_" not in key.variant)
    assert upgrade(wave, **shape).key.variant == (
        "coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out"
    )
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    # The unscaled exact parent and every other route keep their variant, so the
    # registered strict fallback chain and the dense-wide parents are untouched.
    for variant in (
        "coltile8_rowbatch4_f32_f32_out",
        "mmq128_token64_q8_1_d4x3_guarded_f32_f32_out",
        "prefill_f32_f32_out",
    ):
        other = linear.GGUFLinearDispatch(
            KernelKey("hip_gfx1151", "linear", "gguf_q8_0", variant), "raw"
        )
        assert upgrade(other, **shape) is other
    non_raw = linear.GGUFLinearDispatch(wave.key, "wmma_raw")
    assert upgrade(non_raw, **shape) is non_raw
    monkeypatch.setattr(linear, "is_registered", lambda key: False)
    assert upgrade(wave, **shape) is wave


def test_checked_full_rows_upgrade_is_shape_gated_and_env_opt_out(monkeypatch):
    wave = linear.GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1151", "linear", "gguf_q8_0",
            "coltile8_rowbatch4_wave_scale_f32_f32_out",
        ),
        "raw",
    )
    upgrade = linear._raw_k_coltile_pipelined_dispatch
    qualified = {"rows": 4096, "in_features": 2560, "out_features": 10240}
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION", raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_DPP_CHECKED_FULL_ROWS", raising=False)
    monkeypatch.setattr(linear, "is_registered", lambda key: True)

    # Qualified prefill shape: the checked full-row sibling claims the launch.
    assert upgrade(wave, **qualified).key.variant == (
        "coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out"
    )
    # The dedicated opt-out falls back to the plain DPP leaf on the same shape.
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DPP_CHECKED_FULL_ROWS", "0")
    assert upgrade(wave, **qualified).key.variant == (
        "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
    )
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_DPP_CHECKED_FULL_ROWS", raising=False)

    # The DPP family opt-out skips the checked sibling too: it is downstream of
    # the DPP leaf in the degradation chain.
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION", "0")
    assert upgrade(wave, **qualified).key.variant == (
        "coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out"
    )
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION", raising=False)

    # Unqualified shapes keep the plain DPP leaf: decode rows, sub-512 rows and
    # geometry outside the declared dense Q8_0 set.
    for shape in (
        {"rows": 1, "in_features": 2560, "out_features": 10240},
        {"rows": 508, "in_features": 2560, "out_features": 10240},
        {"rows": 4096, "in_features": 2048, "out_features": 10240},
    ):
        assert upgrade(wave, **shape).key.variant == (
            "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
        )

    # Registry guard: an unregistered checked key degrades to the DPP parent.
    monkeypatch.setattr(
        linear, "is_registered",
        lambda key: "checked_full_rows" not in key.variant,
    )
    assert upgrade(wave, **qualified).key.variant == (
        "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
    )
