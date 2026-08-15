from __future__ import annotations

import pytest

from scripts import qwen35_08b_three_way as three_way


def test_rotated_order_balances_three_engines() -> None:
    orders = [three_way.rotated_order(block) for block in range(6)]

    assert orders[:3] == [
        ["hipengine", "llamacpp_hip", "llamacpp_vulkan"],
        ["llamacpp_hip", "llamacpp_vulkan", "hipengine"],
        ["llamacpp_vulkan", "hipengine", "llamacpp_hip"],
    ]
    for position in range(3):
        assert sorted(order[position] for order in orders) == sorted(
            list(three_way.ENGINES) * 2
        )


def test_normalize_hipengine_uses_full_core_and_public_trajectories() -> None:
    stats = {"median": 10.0}
    payload = {
        "prefill_tok_s": stats,
        "decode_tok_s": stats,
        "public_prefill_tok_s": stats,
        "public_decode_tok_s": stats,
        "timed_all_finite": True,
        "public_all_finite": True,
        "top1_all_finite": True,
        "public_top1_all_finite": True,
        "top1_repeat_exact": True,
        "public_top1_repeat_exact": True,
        "top1_ids": [7, 8],
        "public_top1_ids": [7, 9],
        "core_graph_nodes": [283],
        "public_graph_nodes": [286],
        "memory": {"owned_session_bytes": 1},
    }

    row = three_way.normalize_hipengine(payload)

    assert row["finite"] is True
    assert row["deterministic"] is True
    assert row["core_top1_sha256"] != row["public_top1_sha256"]


def test_normalize_llamacpp_uses_exact_fixture_counts() -> None:
    payload = {
        "engine": "llamacpp_hip",
        "prompt_tokens": 512,
        "forced_tokens": 128,
        "repetitions": 1,
        "prefill_ms": [100.0],
        "decode_ms": [800.0],
        "public_prefill_ms": [125.0],
        "public_decode_ms": [1000.0],
        "top1_ids": [7, 8],
        "public_top1_ids": [7, 9],
        "top1_deterministic": True,
        "public_top1_deterministic": True,
    }

    row = three_way.normalize_llamacpp(payload)

    assert row["prefill_tok_s"] == pytest.approx(5120.0)
    assert row["decode_tok_s"] == pytest.approx(160.0)
    assert row["public_prefill_tok_s"] == pytest.approx(4096.0)
    assert row["public_decode_tok_s"] == pytest.approx(128.0)
    assert row["deterministic"] is True
    assert row["core_top1_sha256"] != row["public_top1_sha256"]


def test_summarize_reports_ratios_and_cross_engine_correctness() -> None:
    shared_core = three_way._digest_ids([1, 2, 3])
    shared_public = three_way._digest_ids([4, 5, 6])
    blocks = []
    rates = {
        "hipengine": 4000.0,
        "llamacpp_hip": 5000.0,
        "llamacpp_vulkan": 6000.0,
    }
    for block in range(3):
        execution = []
        for order_index, engine in enumerate(three_way.rotated_order(block)):
            base = rates[engine] + block * 10.0
            execution.append(
                {
                    "engine": engine,
                    "order_index": order_index,
                    "prefill_tok_s": base,
                    "decode_tok_s": base / 40.0,
                    "public_prefill_tok_s": base - 20.0,
                    "public_decode_tok_s": base / 50.0,
                    "finite": True,
                    "deterministic": True,
                    "core_top1_sha256": shared_core,
                    "public_top1_sha256": shared_public,
                }
            )
        blocks.append(
            {"block": block, "order": three_way.rotated_order(block), "execution": execution}
        )

    result = three_way.summarize(blocks)

    comparison = result["comparisons"]["llamacpp_vulkan"]["prefill_tok_s"]
    assert comparison["hipengine_median"] == 4010.0
    assert comparison["peer_median"] == 6010.0
    assert comparison["hipengine_over_peer"] == pytest.approx(4010.0 / 6010.0)
    assert comparison["hipengine_paired_wins"] == 0
    assert result["correctness"]["all_finite"] is True
    assert result["correctness"]["all_deterministic"] is True
    assert result["correctness"]["cross_engine_core_top1_exact"] is True
    assert result["correctness"]["cross_engine_public_top1_exact"] is True
