from __future__ import annotations

import base64
import struct

from scripts.laguna_routing_replay import (
    DEFAULT_TILE_ROWS,
    _routing_distribution_summary,
    _synthetic_selected_by_layer,
)


def test_routing_distribution_reports_multi_tile_padding_once() -> None:
    selected = {1: (0, 1, 0, 2, 0, 1, 3, 4)}

    summary = _routing_distribution_summary(
        selected,
        rows=4,
        top_k=2,
        expert_count=5,
        tile_rows=(2, 4, 8),
    )

    assert summary["actual_lanes"] == 8
    assert summary["active_expert_groups"] == 5
    assert summary["group_size_histogram"] == {"1": 3, "2": 1, "3": 1}
    assert summary["max_expert_lanes"] == 3
    assert summary["tiles"]["2"] == {
        "tile_rows": 2,
        "padded_lanes": 12,
        "padding_lanes": 4,
        "padding_factor": 1.5,
        "padding_overhead_ratio": 0.5,
    }
    assert summary["tiles"]["4"]["padding_factor"] == 2.5
    assert summary["tiles"]["8"]["padding_factor"] == 5.0

    layer = summary["layers"]["1"]
    assert layer["active_experts"] == 5
    assert layer["max_expert_lanes"] == 3
    assert layer["per_expert_counts_encoding"] == (
        "uint16_le_dense_expert_id_order_base64"
    )
    assert struct.unpack(
        "<5H",
        base64.b64decode(layer["per_expert_counts_u16_le_base64"]),
    ) == (3, 2, 1, 1, 1)


def test_hot_and_zipf_synthetic_routes_are_deterministic_and_valid() -> None:
    assert DEFAULT_TILE_ROWS == (2, 4, 8, 16, 32)
    hot = _synthetic_selected_by_layer(
        rows=16,
        top_k=4,
        expert_count=32,
        sparse_layers=3,
        pattern="hot",
        seed=20260723,
    )
    repeated = _synthetic_selected_by_layer(
        rows=16,
        top_k=4,
        expert_count=32,
        sparse_layers=3,
        pattern="hot",
        seed=20260723,
    )
    zipf = _synthetic_selected_by_layer(
        rows=16,
        top_k=4,
        expert_count=32,
        sparse_layers=3,
        pattern="zipf",
        seed=20260723,
    )

    assert hot == repeated
    assert hot != zipf
    for selected_by_layer in (hot, zipf):
        assert set(selected_by_layer) == {1, 2, 3}
        for selected in selected_by_layer.values():
            assert len(selected) == 64
            assert all(0 <= expert < 32 for expert in selected)
            for row in range(16):
                token_experts = selected[row * 4 : (row + 1) * 4]
                assert len(set(token_experts)) == 4
