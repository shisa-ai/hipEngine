"""Fused expert consumers use bounded device tile maps without host readback."""

from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as experts


@pytest.mark.parametrize("quant", ["gguf_q5_k", "gguf_q4_k"])
@pytest.mark.parametrize("rows", [8, 129, 2053])
def test_fused_tiles_do_not_synchronize_or_copy_to_host(monkeypatch, quant, rows):
    from hipengine.core import memory
    from hipengine.kernels.hip_gfx1100.moe import group_scatter
    from hipengine.kernels.hip_gfx1100.quant import (
        gguf_q4_k_t16_selected_prefill as q4,
        gguf_q5_k_q8_1_selected_prefill as q5,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("host synchronization/readback cannot be graph captured")

    monkeypatch.setattr(memory, "copy_device_to_host", forbidden)
    monkeypatch.setattr(group_scatter, "qwen35_moe_wmma_tile_map", lambda *a, **k: None)
    launches = []
    monkeypatch.setattr(q4, "build_gguf_q4_k_t16_selected_prefill", lambda **k: object())
    monkeypatch.setattr(q5, "build_gguf_q5_k_q8_1_selected_prefill", lambda **k: object())
    monkeypatch.setattr(q4, "gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_column_major_bf16_bf16_out",
                        lambda *a, **k: launches.append(a))
    monkeypatch.setattr(q5, "gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out",
                        lambda *a, **k: launches.append(a))
    monkeypatch.setattr(q5, "gguf_q5_k_selected_dual_sparse_exact_repair_bf16", lambda *a, **k: None)
    allocation = SimpleNamespace(buffer=SimpleNamespace(ptr=100))
    weight = SimpleNamespace(allocations={"t16_gate": allocation, "t16_up": allocation},
                             allocation=lambda name: allocation)
    scratch = SimpleNamespace(buffer=lambda name: SimpleNamespace(ptr=200, nbytes=1 << 24))
    runtime = SimpleNamespace(stream_synchronize=forbidden, memset_async=lambda *a: None)
    leaf = (experts._gemma4_project_experts_gate_up_wmma_iu8 if quant == "gguf_q5_k"
            else experts._gemma4_project_experts_gate_up_wmma_t16)
    assert leaf(weight, 1, 2, SimpleNamespace(ptr=3), rows, 128, 256, 704,
                scratch=scratch, stream=7, runtime=runtime)
    # The kernel reads a -1 sentinel for unused tiles; a launch bound depends
    # only on the compact-row count and expert geometry, never host tile totals.
    assert launches
    assert launches[0][-1] == ((rows + 15 * min(rows, 128)) // 16) * 16


def test_tile_bound_covers_all_small_expert_compositions():
    from itertools import product

    for counts in product(range(8), repeat=4):
        rows = sum(counts)
        if not rows:
            continue
        actual = sum(((count + 15) // 16) * 16 for count in counts)
        bound = experts._expert_tile_row_bound(rows, 4, 16, 100)
        assert actual <= bound


def test_tile_bound_refuses_an_insufficient_map():
    with pytest.raises(RuntimeError, match="capacity"):
        experts._expert_tile_row_bound(8, 128, 16, 7)
