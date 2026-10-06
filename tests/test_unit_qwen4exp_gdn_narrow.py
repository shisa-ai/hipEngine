"""CPU admission/registry contract for the diagnostic narrow-index GDN leaf."""
import pytest

from hipengine.kernels.hip_gfx1100.linear_attn import qwen4_exp_gdn as gdn
from hipengine.kernels.registry import resolve

INT32_MAX = 2**31 - 1


@pytest.mark.parametrize("shape", [
    (0, 16, 48, 128, 128),
    (1, 0, 48, 128, 128),
    (1, 16, 47, 128, 128),
    (1, 16, 48, 64, 128),
    (1, 16, 48, 128, 64),
    (INT32_MAX // 10240 + 1, 16, 48, 128, 128),
    (1, 1, INT32_MAX // 16384 + 1, 128, 128),
    (1, 2**63, 2**63, 128, 128),
])
def test_reject_invalid_geometry_before_build(monkeypatch, shape):
    def forbidden(**kwargs):
        pytest.fail("invalid narrow-index geometry reached HIP build")
    monkeypatch.setattr(gdn, "build_qwen4_exp_gdn", forbidden)
    with pytest.raises(ValueError):
        gdn.qwen4_exp_gdn_wave_norm_narrow_prefill_f32(*([1] * 9), *shape)


@pytest.mark.parametrize("tokens,hk,hv", [
    (1, 1, 1), (4096, 16, 48), (INT32_MAX // 10240, 16, 48),
    (1, 1, INT32_MAX // 16384),
])
def test_admitted_geometry_uses_diagnostic_symbol(monkeypatch, tokens, hk, hv):
    calls = []
    def fake_signed(library, symbol, argtypes, restype):
        assert symbol == "hipengine_qwen4_exp_gdn_wave_norm_narrow_prefill_f32"
        assert len(argtypes) == 15
        return lambda *args: calls.append(args) or 0
    monkeypatch.setattr(gdn, "signed_kernel_fn", fake_signed)
    gdn.qwen4_exp_gdn_wave_norm_narrow_prefill_f32(
        *([1] * 9), tokens, hk, hv, 128, 128, library=object(), runtime=object())
    assert calls[0][9:14] == (tokens, hk, hv, 128, 128)


def test_registry_retains_parent_and_strict():
    gdn.register_qwen4_exp_gdn_kernels()
    for variant, fn in (
        ("qwen4exp_sigmoid_wave_norm_narrow_prefill", gdn.qwen4_exp_gdn_wave_norm_narrow_prefill_f32),
        ("qwen4exp_sigmoid_wave_norm_prefill", gdn.qwen4_exp_gdn_wave_norm_prefill_f32),
        ("qwen4exp_sigmoid_strict_prefill", gdn.qwen4_exp_gdn_prefill_f32),
    ):
        assert resolve(backend="hip_gfx1100", layer="gdn_recurrence_norm_gate",
                       quant="f32_state", variant=variant) is fn
