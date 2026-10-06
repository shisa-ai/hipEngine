"""DP-5 producer admission is registry-axis policy, not a quant/backend branch."""
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as runner
from hipengine.runtime import gguf_linear as linear
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

X, GAMMA, NORM, HALF, OUT, W = (i << 32 for i in range(1, 7))
RT = object()
COUNT = 4096 * 10240
ENV = "HIPENGINE_QWEN4_EXP_GR_NORM_F16_PUBLISH"
PADDED = "dense_wide256_padded_f16in_f32_f32_out"


def weight(backend="hip_gfx1151", quant="gguf_q8_0", role="layers.16.hc_attn_down"):
    return SimpleNamespace(backend=backend, spec=SimpleNamespace(quant_key=quant, slot_path=role,
        layout="raw_gguf", source=SimpleNamespace(shape=(320, 10240))),
        allocation=lambda _: SimpleNamespace(tensor=SimpleNamespace(ptr=W),
            buffer=SimpleNamespace(ptr=W, nbytes=320 * 10240 // 32 * 34)))


@pytest.fixture
def setup(monkeypatch):
    register_gfx1151_kernels(replace=True)
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(linear.WIDE_F16_ACTIVATION_ENV, "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS", ",".join(map(str, range(13,48))))
    monkeypatch.setenv(linear.DENSE_WIDE_TAIL_VARIANT_ENV, PADDED)
    calls = []
    monkeypatch.setattr(runner, "qwen4_exp_grouped_rmsnorm_bf16_f32",
                        lambda *a, **k: calls.append(("parent", a, k)))
    def resolve(**key):
        return lambda *a, **k: calls.append(("publisher", a, k))
    monkeypatch.setattr(runner, "resolve", resolve)
    def launch(*a, **k):
        assert linear._consume_wide_f16_publication(
            NORM, COUNT, stream=7, runtime=RT,
            consumer_key=linear.KernelKey("hip_gfx1151", "linear", "gguf_q8_0", PADDED)) == HALF
        calls.append(("down", a, k))
    monkeypatch.setattr(runner, "launch_gguf_linear", launch)
    return calls


def run(w=None, rows=4096, branches=4, hidden=2560, low_rank=320):
    return runner._run_qwen4_exp_gr_norm_down(X, GAMMA, w or weight(),
        SimpleNamespace(normalized=SimpleNamespace(ptr=NORM), low_rank=SimpleNamespace(ptr=OUT)),
        rows=rows, branches=branches, hidden=hidden, low_rank=low_rank, eps=1e-6,
        stream=7, runtime=RT)


def test_admitted_immediate_pair(setup):
    with linear.wide_f16_activation_session(workspace_ptr=HALF, workspace_nbytes=COUNT*2):
        run()
    assert [v[0] for v in setup] == ["publisher", "down"]
    assert setup[0][1][:4] == (X, GAMMA, NORM, HALF)
    assert linear._consume_wide_f16_publication(NORM, COUNT, stream=7, runtime=RT) == 0


@pytest.mark.parametrize("change", [
    {"rows":512}, {"rows":1024}, {"rows":4095}, {"rows":4097},
    {"branches":2}, {"hidden":2559}, {"low_rank":321},
    {"w":weight(backend="hip_gfx1100")}, {"w":weight(quant="gguf_q4_k")},
    {"w":weight(role="layers.12.hc_attn_down")},
    {"w":weight(role="layers.48.hc_ffn_down")},
    {"w":weight(role="layers.16.attn_qkv")},
])
def test_scope_miss_keeps_parent(setup, monkeypatch, change):
    monkeypatch.setattr(runner, "launch_gguf_linear", lambda *a, **k: setup.append(("down",a,k)))
    with linear.wide_f16_activation_session(workspace_ptr=HALF, workspace_nbytes=COUNT*2):
        run(**change)
    assert [v[0] for v in setup] == ["parent", "down"]


@pytest.mark.parametrize("env,value", [(ENV,"0"), (linear.WIDE_F16_ACTIVATION_ENV,"0"),
    ("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE","0"),
    ("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS", ",".join(map(str, range(17,48))))])
def test_disabled_keeps_parent(setup, monkeypatch, env, value):
    monkeypatch.setenv(env,value)
    monkeypatch.setattr(runner, "launch_gguf_linear", lambda *a, **k: setup.append(("down",a,k)))
    with linear.wide_f16_activation_session(workspace_ptr=HALF, workspace_nbytes=COUNT*2):
        run()
    assert [v[0] for v in setup] == ["parent", "down"]


@pytest.mark.parametrize("ptr,size", [(HALF,COUNT*2-1), (NORM,COUNT*2),
                                     (X,COUNT*2), (GAMMA,COUNT*2), (OUT,COUNT*2), (W,COUNT*2)])
def test_capacity_or_alias_keeps_parent_before_write(setup, monkeypatch, ptr, size):
    monkeypatch.setattr(runner, "launch_gguf_linear", lambda *a, **k: setup.append(("down",a,k)))
    with linear.wide_f16_activation_session(workspace_ptr=ptr, workspace_nbytes=size):
        run()
    assert [v[0] for v in setup] == ["parent", "down"]


def test_missing_registration_and_no_workspace_keep_parent(setup, monkeypatch):
    monkeypatch.setattr(linear, "is_registered", lambda key: key.layer != "gr_grouped_rmsnorm")
    monkeypatch.setattr(runner, "launch_gguf_linear", lambda *a, **k: setup.append(("down",a,k)))
    with linear.wide_f16_activation_session(workspace_ptr=HALF, workspace_nbytes=COUNT*2):
        run()
    run()
    assert [v[0] for v in setup] == ["parent", "down", "parent", "down"]
