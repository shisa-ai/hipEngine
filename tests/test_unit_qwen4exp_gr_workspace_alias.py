"""GR staging rejects overlapping workspace before choosing a direct route."""
from types import SimpleNamespace

import pytest

from hipengine.runtime import gguf_linear as linear
from hipengine.runtime import qwen4_exp_runner as runner

X, GAMMA, NORM, HALF, OUT, W = (i << 32 for i in range(1, 7))


@pytest.mark.parametrize("rows", [257, 566, 4096])
@pytest.mark.parametrize("workspace_ptr", [X, GAMMA, NORM, OUT, W, HALF])
def test_direct_and_fallback_preserve_workspace_alias_guards(monkeypatch, rows, workspace_ptr):
    calls = []
    weight = SimpleNamespace(
        backend="hip_gfx1151", spec=SimpleNamespace(quant_key="gguf_q8_0"),
        allocation=lambda _: SimpleNamespace(
            tensor=SimpleNamespace(ptr=W), buffer=SimpleNamespace(nbytes=320 * 10240 // 32 * 34)))
    scratch = SimpleNamespace(normalized=SimpleNamespace(ptr=NORM), low_rank=SimpleNamespace(ptr=OUT))
    monkeypatch.setattr(runner, "is_registered", lambda key: True)
    monkeypatch.setattr(runner, "_wide_f16_consumer_key", lambda *a, **k: object())
    monkeypatch.setattr(runner, "resolve", lambda **k: lambda *a, **kw: calls.append("direct_norm"))
    monkeypatch.setattr(runner, "qwen4_exp_grouped_rmsnorm_bf16_f32",
                        lambda *a, **k: calls.append("parent_norm"))
    monkeypatch.setattr(runner, "gguf_q8_0_dense_wide128x128_f16in_f32_f32_out",
                        lambda *a, **k: calls.append("direct_down"))

    def fallback(*a, **k):
        assert linear.wide_f16_activation_workspace() is None
        calls.append("parent_down")

    monkeypatch.setattr(runner, "launch_gguf_linear", fallback)
    with linear.wide_f16_activation_session(
            workspace_ptr=workspace_ptr, workspace_nbytes=rows * 10240 * 2):
        result = runner._run_qwen4_exp_gr_norm_down(
            X, GAMMA, weight, scratch, rows=rows, branches=4, hidden=2560,
            low_rank=320, eps=1e-6, stream=0, runtime=object())
        assert linear.wide_f16_activation_workspace().ptr == workspace_ptr
    assert result is False
    assert calls == (["direct_norm", "direct_down"] if workspace_ptr == HALF
                     else ["parent_norm", "parent_down"])
