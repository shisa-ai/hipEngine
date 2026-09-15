import pytest

from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as kernels


@pytest.mark.parametrize("variant", ["p4", "compensated"])
def test_corrections_are_explicit_gfx1151_variants(variant):
    register_gfx1151_kernels()
    name = f"iu8_{variant}_prefill_f32_f32_out"
    assert resolve(backend="hip_gfx1151", layer="linear", quant="gguf_q8_0", variant=name) is (
        getattr(kernels, f"gguf_q8_0_iu8_{variant}_prefill_f32_f32_t"))
    assert not is_registered(KernelKey("hip_gfx1100", "linear", "gguf_q8_0", name))


def test_correction_wrappers_preserve_the_raw_f32_abi(monkeypatch):
    calls = []
    monkeypatch.setattr(kernels, "_launch", lambda *args, **kwargs: calls.append((args, kwargs)))
    kernels.gguf_q8_0_iu8_p4_prefill_f32_f32_t(1, 2, 3, 17, 320, 130, stream=9)
    assert calls == [(("gguf_q8_0", "hipengine_gguf_q8_0_iu8_p4_prefill_f32_f32_t",
                       1, 2, 3, 17, 320, 130), {"stream": 9})]


def test_runtime_corrections_resolve_by_registry_without_changing_original(monkeypatch):
    from types import SimpleNamespace
    from hipengine.runtime import qwen4_exp_runner as runner
    register_gfx1151_kernels()
    weight = SimpleNamespace(backend="hip_gfx1151", spec=SimpleNamespace(quant_key="gguf_q8_0"))
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_GR_IU8_UP_VARIANT", raising=False)
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_GR_IU8_DOWN_VARIANT", raising=False)
    assert runner._qwen4_exp_gr_iu8_projection(weight, down=True) is kernels.gguf_q8_0_iu8_wmma_prefill_f32_f32
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_IU8_DOWN_VARIANT", "iu8_compensated_prefill_f32_f32_out")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_IU8_UP_VARIANT", "iu8_p4_prefill_f32_f32_out")
    assert runner._qwen4_exp_gr_iu8_projection(weight, down=True) is kernels.gguf_q8_0_iu8_compensated_prefill_f32_f32_t
    assert runner._qwen4_exp_gr_iu8_projection(weight, down=False) is kernels.gguf_q8_0_iu8_p4_prefill_f32_f32_t


def test_corrected_candidates_use_counted_registry_entries():
    from scripts.qwen4exp_layer2_profile_gate import CANDIDATES
    for name in ("production_gr_down_compensated", "production_gr_up_p4"):
        spec = CANDIDATES[name]
        assert spec.count_registered_dispatch
        assert spec.direct_dispatch_target is None
        assert spec.environment["HIPENGINE_QWEN4_EXP_Q8_IU8_WMM"] == "0"
        assert spec.environment["HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL"] == "0"
