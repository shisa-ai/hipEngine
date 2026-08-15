from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qwen38_int8_server_context_soak.py"
FIXTURE = ROOT / "benchmarks/prompts/qwen38-sharegpt-soak-v1.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("qwen38_int8_server_context_soak", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pool_pages_cover_all_near_capacity_requests() -> None:
    module = _load_module()
    assert module.pool_pages(114688, 1) == 448
    assert module.pool_pages(49152, 2) == 384
    assert module.pool_pages(8192, 8) == 256
    with pytest.raises(ValueError, match="positive"):
        module.pool_pages(0, 1)
    assert module.effective_resident_capacity(
        current_pool_pages=20,
        pages_per_request=5,
        offered_concurrency=8,
    ) == 4
    with pytest.raises(ValueError, match="integral"):
        module.effective_resident_capacity(
            current_pool_pages=21,
            pages_per_request=5,
            offered_concurrency=8,
        )


def test_pinned_sharegpt_fixture_is_complete_and_rotates_lanes() -> None:
    module = _load_module()
    fixture = module.load_prompt_fixture(FIXTURE)
    assert fixture.source_dataset == "Aeala/ShareGPT_Vicuna_unfiltered"
    assert fixture.source_commit == "8b0048ad6ae8c22f46a78c15559dec98feef5539"
    assert len(fixture.lanes) == 8
    assert all(len(lane.user_turns) >= 2 for lane in fixture.lanes)
    assert module.selected_lane_indices(8, concurrency=4, cycle=0) == (0, 1, 2, 3)
    assert module.selected_lane_indices(8, concurrency=4, cycle=1) == (4, 5, 6, 7)
    assert module.selected_lane_indices(8, concurrency=8, cycle=1) == tuple(range(8))


def test_extract_chat_response_requires_authoritative_exact_ids() -> None:
    module = _load_module()
    response = {
        "choices": [{"text": "ok", "finish_reason": "length", "hipengine": {}}],
        "usage": {"prompt_tokens": 4091, "completion_tokens": 4},
        "hipengine": {
            "token_accounting": {"choice_generated_token_ids": [[11, 12, 13, 14]]},
            "generation_shape": {
                "queue_group": {"id": "group-1", "request_count": 2},
                "backend_groups": [
                    {"input_rows": 2, "actual_group_rows": [2]}
                ],
            },
        },
    }
    row = module.extract_chat_response(
        response,
        expected_prompt_tokens=4091,
        expected_completion_tokens=4,
    )
    assert row["generated_token_ids"] == [11, 12, 13, 14]
    assert row["exact_accounting"] is True
    assert row["generation_shape"]["queue_group"]["request_count"] == 2
    shape = module.summarize_generation_shapes([row, row])
    assert shape["shape_count"] == 2
    assert shape["queue_group_count"] == 1
    assert shape["maximum_backend_group_rows"] == 2

    response["hipengine"]["token_accounting"] = {}
    with pytest.raises(ValueError, match="generated token IDs"):
        module.extract_chat_response(
            response,
            expected_prompt_tokens=4091,
            expected_completion_tokens=4,
        )


def test_safety_gate_requires_headroom_and_idle_ownership() -> None:
    module = _load_module()
    requests = [{"passed": True}, {"passed": True}]
    ownership = {
        "pending_requests": 0,
        "active_requests": 0,
        "stream_producers": 0,
        "model_active_requests": 0,
        "session_count": 0,
        "kv_refcounted_pages": 0,
        "kv_pinned_pages": 0,
        "graph_owners": 0,
        "workspace_owners": 0,
        "cache_resident_entries": 0,
        "cache_resident_pages": 0,
        "cache_resident_bytes": 0,
        "allowed_cache_bytes": 0,
    }
    accepted = module.evaluate_safety_gate(
        requests=requests,
        ready_after={"ready": True},
        ownership=ownership,
        total_vram_bytes=24 * 1024**3,
        peak_vram_bytes=(24 * 1024**3) - (640 * 1024**2),
        minimum_headroom_mib=512,
    )
    assert accepted["passed"] is True
    rejected = module.evaluate_safety_gate(
        requests=requests,
        ready_after={"ready": True},
        ownership=ownership | {"kv_refcounted_pages": 1},
        total_vram_bytes=24 * 1024**3,
        peak_vram_bytes=(24 * 1024**3) - (256 * 1024**2),
        minimum_headroom_mib=512,
    )
    assert rejected["passed"] is False
    assert rejected["checks"]["minimum_headroom"] is False
    assert rejected["checks"]["idle_ownership"] is False
