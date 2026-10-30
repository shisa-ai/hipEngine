import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear

PARENT = "dense_wide256_f16in_f32_f32_out"
LONG_K = "dense_wide128x128_wgp_f16in_f32_f32_out"


def _key(variant=PARENT):
    return KernelKey("hip_gfx1100", "linear", "gguf_q8_0", variant)


@pytest.mark.parametrize(
    "rows,in_features,out_features,expected",
    [
        (4096, 6144, 2560, LONG_K),
        (1024, 6144, 2560, LONG_K),
        (1023, 6144, 2560, PARENT),
        (1025, 6144, 2560, LONG_K),
        (1536, 6144, 2560, LONG_K),
        (512, 6144, 2560, PARENT),      # measured slower below 1024 rows
        (4096, 2560, 10240, PARENT),    # the builds tie at K=2560
        (4096, 10240, 320, PARENT),     # narrow output: CU mode wins
    ],
)
def test_long_k_rule_selects_by_shape(monkeypatch, rows, in_features, out_features, expected):
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_DENSE_WIDE_LONG_K", raising=False)
    key = linear._dense_wide_f16_long_k_key(
        _key(), rows=rows, in_features=in_features, out_features=out_features)
    assert key.variant == expected
    assert (key.backend, key.layer, key.quant) == ("hip_gfx1100", "linear", "gguf_q8_0")


@pytest.mark.parametrize("value", ["0", "false", "False", "1", "caller-sentinel"])
def test_long_k_rule_ignores_retired_key_and_preserves_other_parents(monkeypatch, value):
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_DENSE_WIDE_LONG_K", value)
    assert linear._dense_wide_f16_long_k_key(
        _key(), rows=4096, in_features=6144, out_features=2560).variant == LONG_K
    padded = "dense_wide256_padded_f16in_f32_f32_out"
    assert linear._dense_wide_f16_long_k_key(
        _key(padded), rows=4096, in_features=6144, out_features=2560).variant == padded


def test_long_k_rule_keeps_parent_when_sibling_is_unavailable(monkeypatch):
    attempted = []
    monkeypatch.setattr(linear, "is_registered", lambda key: False)
    monkeypatch.setattr(linear, "_ensure_linear_kernel_registered", attempted.append)
    parent = _key()
    assert linear._dense_wide_f16_long_k_key(
        parent, rows=1536, in_features=6144, out_features=2560) == parent
    assert attempted == [_key(LONG_K)]


def test_short_chunk_build_bound_is_overridable(monkeypatch):
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw

    monkeypatch.delenv("HIPENGINE_GGUF_Q8_DENSE_WIDE_WGP_MAX_ROWS", raising=False)
    assert dw._wgp_build_max_rows() == 1024
    assert dw._WGP_BUILD_MIN_ROWS == 257
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_DENSE_WIDE_WGP_MAX_ROWS", "0")
    assert dw._wgp_build_max_rows() == 0
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_DENSE_WIDE_WGP_MAX_ROWS", "2048")
    assert dw._wgp_build_max_rows() == 2048


@pytest.mark.parametrize(
    "rows,expected",
    [
        (8, "prefill_f32_f32_out"),
        (9, "dense_wide256_f32_f32_out"),
        (128, "dense_wide256_f32_f32_out"),
        (256, "dense_wide256_f32_f32_out"),
    ],
)
def test_qwen4exp_short_rows_admit_wide_route_only_in_profile(monkeypatch, rows, expected):
    from hipengine.runtime.gguf_linear import GGUFLinearDispatch

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_SHORT_ROWS", "1")
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS", raising=False)
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    dispatch = GGUFLinearDispatch(_key("prefill_f32_f32_out"), "raw")
    selected = linear._q8_dense_wide_dispatch(
        dispatch, rows=rows, in_features=320, out_features=2560, weight=None
    )
    assert selected.key.variant == expected


def test_qwen4exp_short_rows_stay_exact_without_profile_selector(monkeypatch):
    from hipengine.runtime.gguf_linear import GGUFLinearDispatch

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_SHORT_ROWS", raising=False)
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    dispatch = GGUFLinearDispatch(_key("prefill_f32_f32_out"), "raw")
    selected = linear._q8_dense_wide_dispatch(
        dispatch, rows=128, in_features=320, out_features=2560, weight=None
    )
    assert selected.key.variant == "prefill_f32_f32_out"



def test_pair_probe_and_staged_launch_resolve_the_same_long_k_sibling(monkeypatch):
    """The cold probe must return the key its own launch will compute.

    ``_wide_f16_consumer_key`` admits a loan by predicting the launch's key;
    the staged launcher then compares that prediction against the key it
    resolved. A probe that stops before the long-K rewrite predicts the parent
    and can never match, so a policy-qualified pair would silently never
    engage.
    """

    from types import SimpleNamespace

    from hipengine.runtime.gguf_linear import GGUFLinearDispatch

    weight = SimpleNamespace(
        spec=SimpleNamespace(slot_path="layers.24.attn_qkv"),
        allocation=lambda _: SimpleNamespace(
            tensor=SimpleNamespace(ptr=5 << 32),
            buffer=SimpleNamespace(nbytes=1 << 20),
        ),
    )
    monkeypatch.setattr(
        linear, "resolve_gguf_linear_dispatch",
        lambda *a, **kw: GGUFLinearDispatch(_key("dense_wide256_f32_f32_out"), "raw"),
    )
    for name in ("_q8_mmq_prefill_dispatch", "_raw_k_prefill_rowbatch_dispatch",
                 "_q8_dense_wide_dispatch", "_raw_k_staged_f32_dispatch"):
        monkeypatch.setattr(linear, name, lambda dispatch, **kw: dispatch)
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    monkeypatch.setattr(
        "hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide.f32_to_f16",
        lambda *a, **kw: None,
    )
    launched = []
    monkeypatch.setattr(
        linear, "resolve",
        lambda **kw: launched.append(kw) or (lambda *a, **k: None),
    )

    def launch():
        return linear._launch_dense_wide_f16_staged(
            None, weight, 2 << 32, 3 << 32, 4096, 6144, 2560, {"stream": 7},
            backend="hip_gfx1100", quant="gguf_q8_0", layer="linear",
            variant="dense_wide256_f32_f32_out", runtime=object(),
        )

    for env, expected in ((None, LONG_K), ("0", LONG_K)):
        if env is None:
            monkeypatch.delenv("HIPENGINE_GGUF_Q8_DENSE_WIDE_LONG_K", raising=False)
        else:
            monkeypatch.setenv("HIPENGINE_GGUF_Q8_DENSE_WIDE_LONG_K", env)
        launched.clear()
        with linear.wide_f16_activation_session(
            workspace_ptr=1 << 32, workspace_nbytes=4096 * 6144 * 2,
        ):
            assert launch()
        probe = linear._wide_f16_consumer_key(
            weight, rows=4096, in_features=6144, out_features=2560)
        assert probe.variant == expected
        assert launched[-1]["variant"] == probe.variant
