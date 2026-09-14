from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_prefill as module
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import resolve
from scripts.qwen4exp_layer2_profile_gate import CANDIDATES
import pytest


def test_blockscale_is_registered_only_as_an_explicit_candidate():
    register_gfx1151_kernels(replace=True)
    assert resolve(backend="hip_gfx1151", layer="linear", quant="gguf_q8_0",
                   variant="selected_grouped_blockscale_prefill_bf16_bf16_out") is (
        module.gguf_q8_0_selected_grouped_blockscale_prefill_bf16_bf16_out)
    candidate = CANDIDATES["q8_blockscale"]
    assert candidate.count_registered_dispatch
    assert candidate.environment == {
        "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN": "1",
        "HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT":
            "selected_grouped_blockscale_prefill_bf16_bf16_out",
    }


def test_blockscale_wrapper_reuses_validated_raw_pointer_abi(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    module.gguf_q8_0_selected_grouped_blockscale_prefill_bf16_bf16_out(
        1, 2, 3, 4, 5, 6, 64, 4, 640, 320, 96, stream=7)
    assert calls[0][0] == (1, 2, 3, 4, 5, 6, 64, 4, 640, 320, 96)
    assert calls[0][1] == {
        "stream": 7,
        "_entry": "hipengine_gguf_q8_0_selected_grouped_blockscale_prefill_bf16_bf16_out",
    }


def test_guarded_wrapper_rejects_short_queue_before_build(monkeypatch):
    monkeypatch.setattr(module, "build_gguf_q8_0_prefill",
                        lambda: pytest.fail("must reject before compiler/runtime setup"))
    with pytest.raises(ValueError, match="risk capacity"):
        module.gguf_q8_0_selected_grouped_blockscale_guarded_prefill_bf16_bf16_out(
            1, 2, 3, 4, 5, 6, 64, 4, 640, 320, 96,
            risk_count_ptr=7, risk_indices_ptr=8, risk_capacity=64 * 320 - 1)


def test_quad_restoration_does_not_restore_other_approximate_routes():
    env = CANDIDATES["q8_blockscale_guarded_quad"].environment
    assert env["HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL"] == "page256"
    assert env["HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR"] == "quad"
    assert set(env) == {
        "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN",
        "HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT",
        "HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL",
        "HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR",
    }


def test_all_q8_codes_are_exact_bf16_without_rounding_bias():
    import numpy as np
    from hipengine.loading.materialize import float_array_to_bf16_bits

    values = np.arange(-128, 128, dtype=np.int32).astype(np.float32)
    shifted = (values.view(np.uint32) >> 16).astype(np.uint16)
    np.testing.assert_array_equal(shifted, float_array_to_bf16_bits(values))
