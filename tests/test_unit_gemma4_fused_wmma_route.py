"""Fused T16 WMMA selection is a registered capability, not a quant branch."""
from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as experts
from hipengine.kernels.registry import KernelKey, register


@pytest.mark.parametrize("rows", [1, 257, 511, 512, 777, 1024])
def test_fused_wmma_capability_and_geometry(monkeypatch, rows):
    key = KernelKey("test_fused_backend", "moe_linear", "unknown_fused_layout",
                    "selected_dual_wmma_prefill_fused_bf16_bf16_out")
    owner = lambda *args, **kwargs: None
    register(key, owner)
    weight = SimpleNamespace(backend=key.backend, spec=SimpleNamespace(quant_key=key.quant))
    assert experts._fused_wmma_owner(weight, rows * 8, 2816, 704, 128) is (
        owner if rows * 8 >= 16 * 128 else None)
    assert experts._fused_wmma_owner(weight, rows * 8, 128, 704, 128) is None
    assert experts._fused_wmma_owner(weight, rows * 8, 2816, 703, 128) is None
    assert experts._fused_wmma_owner(10, rows * 8, 2816, 704, 128) is None


def test_forward_reuses_fused_plan_on_the_requested_stream(monkeypatch):
    events = []
    buffers = {}
    def buffer(name):
        return buffers.setdefault(name, SimpleNamespace(ptr=1000 * (len(buffers) + 1)))
    scratch = SimpleNamespace(tokens=512, top_k=8, hidden_size=2816,
                              intermediate=704, num_experts=128, buffer=buffer)
    weight = SimpleNamespace(spec=SimpleNamespace(layout="gguf_q4_k_t16_v1"),
                             allocation=lambda: SimpleNamespace(buffer=SimpleNamespace(ptr=90000)))
    monkeypatch.setattr(experts, "_prefill_mode", lambda: "auto")
    monkeypatch.setattr(experts, "_fused_wmma_owner", lambda *a: record("fused"))
    def record(name, result=None):
        def call(*args, **kwargs):
            assert kwargs["stream"] == 7
            events.append(name)
            return result
        return call
    monkeypatch.setattr(experts, "_build_wmma_tile_plan", record("map", 8192))
    def forbidden(*args, **kwargs):
        raise AssertionError("fused WMMA must not build or consume the MMQ plan")
    monkeypatch.setattr(experts, "_build_mmq_tile_plan", forbidden)
    monkeypatch.setattr(experts, "gemma4_project_experts_mmq_dual", forbidden)
    monkeypatch.setattr(experts, "qwen35_moe_group_compact_active", record("compact"))
    monkeypatch.setattr(experts, "qwen35_moe_gather_packed_hidden_lowp", record("gather"))
    monkeypatch.setattr(experts, "gemma4_gelu_tanh_mul_bf16", record("gelu"))
    monkeypatch.setattr(experts, "gemma4_project_experts_wmma", record("down", True))
    monkeypatch.setattr(experts, "gemma4_moe_lane_to_row_i32", record("lanes"))
    monkeypatch.setattr(experts, "gemma4_moe_weighted_accumulate_bf16", record("combine"))
    experts.gemma4_experts_forward_bf16(10, 20, 30, weight, 40, 50,
                                      scratch=scratch, rows=512, stream=7)
    assert events == ["compact", "map", "gather", "fused", "gelu", "down", "lanes", "combine"]
