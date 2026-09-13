"""The DPP experiment is separately built and never replaces its fallback."""

from hipengine.kernels.hip_gfx1100.linear_attn import qwen4_exp_gdn as gdn


def test_dpp_builder_uses_source_hashed_compile_flag_and_separate_family(monkeypatch):
    calls = []
    monkeypatch.setattr(gdn, "build_hip", lambda **kw: calls.append(kw) or "library")
    assert gdn.build_qwen4_exp_gdn_dpp(compiler_version="fixture", require_cached=True) == "library"
    call = calls[0]
    assert call["sources"] == [gdn._SOURCE]
    assert call["family"] == "qwen4_exp_gdn_dpp"
    assert call["extra_flags"] == ("-DHIPENGINE_QWEN4_GDN_DPP=1",)
    assert call["target_arch"] == "gfx1151"
    assert call["require_cached"] is True


def test_dpp_wrapper_preserves_explicit_parent_library_boundary(monkeypatch):
    calls = []
    monkeypatch.setattr(gdn, "build_qwen4_exp_gdn_dpp", lambda **kw: "dpp-library")
    monkeypatch.setattr(gdn, "qwen4_exp_gdn_prefill_tiled16_f32",
                        lambda *a, **kw: calls.append((a, kw)))
    gdn.qwen4_exp_gdn_prefill_tiled16_dpp_f32(1, 2, stream=3)
    assert calls == [((1, 2), {"stream": 3, "library": "dpp-library"})]
