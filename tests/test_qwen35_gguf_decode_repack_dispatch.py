from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import hipengine.runtime.qwen35_gguf_runner as qwen_runtime
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner


class _Tensor:
    def __init__(self, ptr: int) -> None:
        self.ptr = ptr


class _Weight:
    def __init__(self, ptr: int) -> None:
        self._allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=ptr))

    def allocation(self):
        return self._allocation


class _Layer:
    def __init__(self) -> None:
        self._weights = {name: _Weight(0x1000 + index * 0x10) for index, name in enumerate(_WEIGHT_NAMES)}

    def weight(self, name: str) -> _Weight:
        return self._weights[name]


_WEIGHT_NAMES = (
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_q_norm",
    "attn_k_norm",
    "attn_output",
)


def _runner(*, is_moe: bool = True) -> Qwen35GGUFFullStackRunner:
    runner = object.__new__(Qwen35GGUFFullStackRunner)
    cfg = SimpleNamespace(
        is_moe=is_moe,
        rms_norm_eps=1.0e-6,
        head_count=16,
        head_count_kv=2,
        key_length=256,
        value_length=256,
        rope_dimension_count=64,
        hidden_size=2048,
    )
    layer = _Layer()
    runner.weights = SimpleNamespace(config=cfg, layer=lambda layer_id: layer)
    runner.runtime = object()
    return runner


def _scratch() -> SimpleNamespace:
    scratch = SimpleNamespace(
        position_host=np.array([7], dtype=np.int64),
        set_full_attention_position=lambda position, runtime: None,
        norm=_Tensor(0x2000),
        full_q=_Tensor(0x2010),
        full_k=_Tensor(0x2020),
        full_v=_Tensor(0x2030),
        full_query_raw=_Tensor(0x2040),
        full_gate=_Tensor(0x2050),
        full_key_raw=_Tensor(0x2060),
        cos_table=_Tensor(0x2070),
        sin_table=_Tensor(0x2080),
        position_tensor=_Tensor(0x2090),
        full_query=_Tensor(0x20A0),
        full_key=_Tensor(0x20B0),
        append_spans=object(),
        decode_spans=object(),
        block_size=256,
        max_positions=768,
        full_attn_context=_Tensor(0x20C0),
        full_attn_split_partial=_Tensor(0x20D0),
        full_attn_split_m=_Tensor(0x20E0),
        full_attn_split_l=_Tensor(0x20F0),
        full_attn_split_count=3,
        full_gated=_Tensor(0x2100),
    )
    key_cache = _Tensor(0x2110)
    value_cache = _Tensor(0x2120)
    scratch.full_cache = lambda layer_id: (key_cache, value_cache)
    return scratch


def _patch_full_attention_primitives(monkeypatch):
    calls: list[tuple[str, tuple, dict]] = []

    def record(name: str, *, returns=None):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return returns

        return fake

    monkeypatch.setattr(qwen_runtime, "gguf_rmsnorm_bf16_f32_weight", record("rmsnorm"))
    monkeypatch.setattr(qwen_runtime, "launch_gguf_linear_triple", record("qkv_triple", returns=True))
    monkeypatch.setattr(qwen_runtime, "launch_gguf_linear_pair", record("kv_pair", returns=True))
    monkeypatch.setattr(qwen_runtime, "launch_gguf_linear", record("linear"))
    monkeypatch.setattr(qwen_runtime, "qwen35_split_qgate_bf16", record("split_qgate"))
    monkeypatch.setattr(qwen_runtime, "bf16_to_f32", record("bf16_to_f32"))
    monkeypatch.setattr(
        qwen_runtime,
        "gguf_qwen35_head_rmsnorm_partial_rotary_position_key_bf16_f32_weight",
        record("rope_key_bf16"),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight",
        record("rope_key_f32"),
    )
    monkeypatch.setattr(qwen_runtime, "qwen35_write_paged_kv_mixed_value_bf16_spans", record("kv_write"))
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans",
        record("split_k_gqa_gate"),
    )
    monkeypatch.setattr(
        qwen_runtime,
        "qwen35_paged_full_attn_decode_context_bf16_spans",
        record("attention_context"),
    )
    monkeypatch.setattr(qwen_runtime, "qwen35_full_attn_gate_mul_bf16", record("attention_gate"))
    return calls


def test_qwen35moe_decode_repack_routes_full_attention_through_split_k_gqa_gate(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    runner = _runner(is_moe=True)
    scratch = _scratch()
    calls = _patch_full_attention_primitives(monkeypatch)

    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, scratch, position=7, stream=5)

    names = [name for name, _, _ in calls]
    assert "rope_key_bf16" in names
    assert "bf16_to_f32" not in names
    assert "split_k_gqa_gate" in names
    assert "attention_context" not in names
    assert "attention_gate" not in names

    split_args = next(args for name, args, _ in calls if name == "split_k_gqa_gate")
    assert split_args[:8] == (
        scratch.full_query.ptr,
        0x2110,
        0x2120,
        scratch.full_gate.ptr,
        scratch.full_gated.ptr,
        scratch.full_attn_split_partial.ptr,
        scratch.full_attn_split_m.ptr,
        scratch.full_attn_split_l.ptr,
    )
    assert split_args[9:18] == (256, scratch.full_attn_split_count, 256, 16, 2, 256, 256, 1, 256 ** -0.5)


def test_qwen35moe_without_decode_repack_keeps_unfused_full_attention_gate(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "0")
    runner = _runner(is_moe=True)
    scratch = _scratch()
    calls = _patch_full_attention_primitives(monkeypatch)

    runner._run_full_attention_attn_only(0, 0x3000, 0x4000, scratch, position=7, stream=5)

    names = [name for name, _, _ in calls]
    assert "rope_key_f32" in names
    assert "bf16_to_f32" in names
    assert "attention_context" in names
    assert "attention_gate" in names
    assert "split_k_gqa_gate" not in names

    context_args = next(args for name, args, _ in calls if name == "attention_context")
    gate_args = next(args for name, args, _ in calls if name == "attention_gate")
    assert context_args[:5] == (scratch.full_query.ptr, 0x2110, 0x2120, scratch.full_attn_context.ptr, scratch.decode_spans)
    assert context_args[5:11] == (scratch.max_positions, scratch.block_size, 16, 2, 256, 256 ** -0.5)
    assert gate_args[:4] == (scratch.full_attn_context.ptr, scratch.full_gate.ptr, scratch.full_gated.ptr, runner.q_width)
