"""F16 route diagnostics must share production precondition validation."""
from types import SimpleNamespace
import pytest
from hipengine.runtime import laguna_f16_hipblaslt as f16
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession


def test_f16_runtime_setter_is_removed():
    assert not hasattr(LagunaGGUFResidentSession, "set_f16_prefill_mode")


def test_retained_validation_does_not_require_dense_metadata():
    assert f16.validate_laguna_f16_prefill_mode("hip_gfx1151", None, "retained") == "retained"


def test_blas_validation_rejects_missing_weights_and_invalid_mode():
    with pytest.raises(ValueError, match="weights"):
        f16.validate_laguna_f16_prefill_mode("hip_gfx1151", None, "hipblaslt_scaled")
    with pytest.raises(ValueError, match="unsupported"):
        f16.validate_laguna_f16_prefill_mode("hip_gfx1151", SimpleNamespace(), "invalid")


def test_layout_refusal_preserves_layer_identity():
    layer = SimpleNamespace(layer_id=17, weight=lambda slot: SimpleNamespace(
        spec=SimpleNamespace(layout="unsupported_layout")))
    with pytest.raises(ValueError, match="17.*attn_q.*unsupported_layout"):
        f16.validate_laguna_f16_prefill_mode(
            "hip_gfx1151", SimpleNamespace(layers=(layer,)), "hipblaslt_scaled",
        )
