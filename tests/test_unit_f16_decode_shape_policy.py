"""Shape-aware source-F16 automatic decode and explicit reference contracts."""

from types import SimpleNamespace

import pytest

from hipengine.runtime import f16_weight_linear as dispatch
from hipengine.kernels.hip_gfx1100.linear import laguna_f16_projection as primitive


@pytest.mark.parametrize("width,threads,expected", [
    (3072, 256, "fixedk_onebarrier"), (6144, 256, "fixedk_onebarrier"),
    (9216, 256, "fixedk_onebarrier"), (256, 256, "onebarrier"),
    (3071, 256, "onebarrier"), (3073, 256, "onebarrier"),
    (256, 64, "gemv"), (3072, 64, "gemv"),
])
@pytest.mark.parametrize("triple", [False, True])
def test_auto_decode_runs_declared_shape_or_generic_reference(monkeypatch, width, threads, expected, triple):
    monkeypatch.delenv("HIPENGINE_LAGUNA_F16_DECODE", raising=False)
    monkeypatch.setattr(dispatch, "backend_package_capability", lambda backend, name, default: name in {
        "LAGUNA_F16_DECODE_FIXEDK", "LAGUNA_F16_DECODE_ONEBARRIER",
    })
    calls = []

    class Library:
        def __getattr__(self, name):
            def launch(*args):
                calls.append((name, args))
                return 0
            return launch

    library = Library()
    families = {
        "linear": {
            "fixedk_onebarrier": primitive.laguna_f16w_fixedk_onebarrier_gemv_bf16_f32_out,
            "onebarrier": primitive.laguna_f16w_onebarrier_gemv_bf16_f32_out,
            "gemv": primitive.laguna_f16w_gemv_bf16_f32_out,
        },
        "linear_triple": {
            "fixedk_onebarrier": primitive.laguna_f16w_triple_fixedk_onebarrier_gemv_bf16_f32_out,
            "onebarrier": primitive.laguna_f16w_triple_onebarrier_gemv_bf16_f32_out,
            "gemv": primitive.laguna_f16w_triple_gemv_bf16_f32_out,
        },
    }

    def resolve(key):
        prefix = key.variant.removesuffix("bf16_f32_out").removesuffix("_") or "gemv"
        return families[key.layer][prefix]

    monkeypatch.setattr(dispatch, "_resolve", resolve)
    weight = SimpleNamespace(
        backend="hip_gfx1151", spec=SimpleNamespace(layout="dense_f16", quant_key="fp16"),
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
    )
    kwargs = dict(rows=1, in_features=width, threads=threads, stream=7,
                  libraries={"fp16_weight": library}, runtime=object())
    if triple:
        dispatch.launch_f16_weight_linear_triple(
            weight, weight, weight, 10, 20, 21, 22,
            out_a_features=64, out_b_features=64, out_c_features=64, **kwargs,
        )
    else:
        dispatch.launch_f16_weight_linear(weight, 10, 20, out_features=64, **kwargs)
    name, args = calls.pop()
    assert expected in name
    assert ("fixedk" in name) is (expected == "fixedk_onebarrier")
    assert ("onebarrier" in name) is (expected != "gemv")
    if triple:
        assert args == (10, 11, 11, 11, 20, 21, 22, 1, width, 64, 64, 64, threads, 7)
    else:
        assert args == (10, 11, 20, 1, width, 64, threads, 7)


@pytest.mark.parametrize("mode,width,threads,error", [
    ("fixedk", 256, 256, "fixed-K decode GEMV requires K3072"),
    ("fixedk", 3072, 64, "decode GEMV compatibility threads must be 256"),
    ("onebarrier", 256, 64, "decode GEMV compatibility threads must be 256"),
])
def test_explicit_decode_requests_keep_named_capability_errors(monkeypatch, mode, width, threads, error):
    monkeypatch.setenv("HIPENGINE_LAGUNA_F16_DECODE", mode)
    strategy = dispatch._decode_strategy(
        rows=1, activation_dtype="bf16", backend="hip_gfx1151",
        in_features=width, threads=threads,
    )
    expected = "fixedk_onebarrier" if mode == "fixedk" else mode
    assert strategy == expected
    launch = (
        primitive.laguna_f16w_fixedk_onebarrier_gemv_bf16_f32_out
        if mode == "fixedk" else primitive.laguna_f16w_onebarrier_gemv_bf16_f32_out
    )
    with pytest.raises(ValueError, match=error):
        launch(1, 2, 3, 1, width, 64, threads=threads)


@pytest.mark.parametrize("rows,dtype", [(1, "f32"), (2, "bf16"), (17, "bf16")])
def test_decode_shape_policy_does_not_capture_prefill_or_f32(monkeypatch, rows, dtype):
    monkeypatch.setenv("HIPENGINE_LAGUNA_F16_DECODE", "fixedk")
    assert dispatch._decode_strategy(
        rows=rows, activation_dtype=dtype, backend="hip_gfx1151", in_features=256,
    ) is None


def test_auto_decode_does_not_borrow_peer_capabilities(monkeypatch):
    monkeypatch.delenv("HIPENGINE_LAGUNA_F16_DECODE", raising=False)
    monkeypatch.setattr(dispatch, "backend_package_capability", lambda *a: False)
    assert dispatch._decode_strategy(
        rows=1, activation_dtype="bf16", backend="hip_gfx1100", in_features=3072,
    ) is None
