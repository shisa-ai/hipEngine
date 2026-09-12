"""Decode-only GGUF layer chaining tests for MoE-tail + next RMSNorm."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime import qwen35_gguf_runner as qgr


class _Weight:
    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    def allocation(self):
        return SimpleNamespace(tensor=SimpleNamespace(ptr=self._ptr))


class _Layer:
    def __init__(self, norm_ptr: int) -> None:
        self._norm = _Weight(norm_ptr)

    def weight(self, name: str):
        assert name == "attn_norm"
        return self._norm


class _Weights:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            layer_types=(qgr.LINEAR_ATTENTION, qgr.FULL_ATTENTION, qgr.LINEAR_ATTENTION),
            is_moe=True,
            rms_norm_eps=1e-6,
        )
        self._layers = [_Layer(1000 + index) for index in range(3)]

    def layer(self, index: int):
        return self._layers[index]

    def root(self, name: str):
        assert name == "output_norm"
        return _Weight(2000)


def _session_and_calls():
    calls: list[tuple] = []
    weights = _Weights()

    def run_linear(layer_id, hidden_ptr, out_ptr, scratch, **kwargs):
        calls.append(("linear", layer_id, hidden_ptr, out_ptr, kwargs))

    def run_full(layer_id, hidden_ptr, out_ptr, scratch, **kwargs):
        calls.append(("full", layer_id, hidden_ptr, out_ptr, kwargs))

    runner = SimpleNamespace(
        weights=weights,
        hidden_size=16,
        _run_linear_attention_layer=run_linear,
        _run_full_attention_layer=run_full,
    )
    session = object.__new__(qgr.Qwen35GGUFResidentSession)
    session.runner = runner
    session.runtime = SimpleNamespace()
    session._hidden_a = SimpleNamespace(ptr=100)
    session._hidden_b = SimpleNamespace(ptr=200)
    session.scratch = SimpleNamespace(
        position_host=np.zeros((1,), dtype=np.int64),
        context_host=np.zeros((1,), dtype=np.int64),
        norm=SimpleNamespace(ptr=300),
    )
    return session, calls


def test_moe_tail_next_rms_env_defaults_on_with_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_MOE_TAIL_NEXT_RMS", raising=False)
    assert qgr._gguf_moe_tail_next_rms_enabled()
    monkeypatch.setenv("HIPENGINE_GGUF_MOE_TAIL_NEXT_RMS", "0")
    assert not qgr._gguf_moe_tail_next_rms_enabled()
    monkeypatch.setenv("HIPENGINE_GGUF_MOE_TAIL_NEXT_RMS", "1")
    assert qgr._gguf_moe_tail_next_rms_enabled()


def test_resident_decode_chains_norm_across_linear_full_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, calls = _session_and_calls()
    monkeypatch.setattr(qgr, "_gguf_moe_tail_next_rms_enabled", lambda: True)
    monkeypatch.setattr(
        qgr,
        "gguf_rmsnorm_bf16_f32_weight",
        lambda *args, **kwargs: calls.append(("output_norm", args, kwargs)),
    )

    result = session._run_current_hidden_to_final_hidden(position=7, max_context_len=64, stream=9)

    assert result == 300
    layer_calls = calls[:3]
    assert [(call[0], call[1]) for call in layer_calls] == [("linear", 0), ("full", 1), ("linear", 2)]
    assert layer_calls[0][4]["input_norm_ptr"] is None
    assert layer_calls[0][4]["next_norm_weight_ptr"] == 1001
    assert layer_calls[0][4]["next_norm_out_ptr"] == 300
    assert layer_calls[1][4]["input_norm_ptr"] == 300
    assert layer_calls[1][4]["next_norm_weight_ptr"] == 1002
    assert layer_calls[1][4]["next_norm_out_ptr"] == 300
    assert layer_calls[2][4]["input_norm_ptr"] == 300
    assert layer_calls[2][4]["next_norm_weight_ptr"] is None
    assert layer_calls[2][4]["next_norm_out_ptr"] is None
    assert layer_calls[1][4]["position"] == 7
    assert layer_calls[1][4]["max_context_len"] == 64
    assert calls[3][0] == "output_norm"
    assert calls[3][1][:3] == (200, 2000, 300)


def test_resident_decode_keeps_unfused_layer_inputs_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, calls = _session_and_calls()
    monkeypatch.setattr(qgr, "_gguf_moe_tail_next_rms_enabled", lambda: False)
    monkeypatch.setattr(qgr, "_gguf_moe_graph_enabled", lambda: False)
    monkeypatch.setattr(qgr, "gguf_rmsnorm_bf16_f32_weight", lambda *args, **kwargs: None)

    session._run_current_hidden_to_final_hidden(position=2, stream=4)

    for call in calls:
        kwargs = call[4]
        assert kwargs["input_norm_ptr"] is None
        assert kwargs["next_norm_weight_ptr"] is None
        assert kwargs["next_norm_out_ptr"] is None
