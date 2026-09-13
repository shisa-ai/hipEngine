from __future__ import annotations

from scripts.gguf_residency_g6 import _audit_plan, _build_artifact


def _plan_row(source: str, layout: str, allocations: list[str], nbytes: int = 100) -> dict:
    return {
        "slot_path": f"root.{source}",
        "source_name": source,
        "source_nbytes": nbytes,
        "quant_key": layout,
        "layout": layout,
        "allocation_names": allocations,
    }


def test_plan_audit_distinguishes_replacement_from_duplicate_sidecar() -> None:
    replacement_only = _audit_plan(
        [
            _plan_row("token", "raw_gguf", ["raw"]),
            _plan_row("expert", "gguf_q4_k_t16_v1", ["tiles"]),
        ]
    )
    duplicated = _audit_plan(
        [
            _plan_row("expert", "raw_gguf", ["raw"]),
            _plan_row("expert", "gguf_q4_k_t16_v1", ["tiles"]),
            _plan_row("head", "gguf_q6_k_t16_v1", ["tiles", "x8"]),
        ]
    )

    assert replacement_only["raw_plus_replacement_duplicate_count"] == 0
    assert replacement_only["optional_sidecar_count"] == 0
    assert duplicated["raw_plus_replacement_duplicate_count"] == 1
    assert duplicated["raw_plus_replacement_duplicates"][0]["source_name"] == "expert"
    assert duplicated["optional_sidecar_count"] == 1


def _snapshot(current: int, hip_used: int, breakdown: dict | None = None) -> dict:
    result = {
        "tracked": {"current_allocated_bytes": current, "peak_allocated_bytes": current},
        "hip": {"available": True, "used_bytes": hip_used},
    }
    if breakdown is not None:
        result["owned_session_breakdown"] = breakdown
    return result


def test_g6_artifact_accepts_complete_replacement_only_census(monkeypatch) -> None:
    monkeypatch.setattr(
        "hipengine.benchmark.provenance.validate_artifact_provenance",
        lambda payload, require_model=False: payload,
    )
    weights = {
        "total_bytes": 300,
        "allocation_count": 3,
        "by_layout_bytes": {"raw_gguf": 100, "gguf_q4_k_t16_v1": 200},
        "by_quant_key_bytes": {"gguf_q8_0": 100, "gguf_q4_k_t16_v1": 200},
        "by_allocation_name_bytes": {"raw": 100, "tiles": 200},
    }
    breakdown = {
        "total_bytes": 1000,
        "families": {
            "weights": weights,
            "decode_scratch": {
                "total_bytes": 400,
                "by_component_bytes": {
                    "full_attention_kv_cache": 250,
                    "full_attention_kv_scales": 0,
                },
            },
            "session_buffers": {"total_bytes": 300, "by_component_bytes": {}},
        },
    }
    source = {
        "model": "/models/example.gguf",
        "quant": "gguf_q4_k_m",
        "kv_storage_dtype": "bf16",
        "prompt_source": "repeated_token_id",
        "token_id": 9707,
        "prompt_length": 512,
        "decode_tokens": 128,
        "max_sequence_length": 641,
        "graph_replay_decode": True,
        "persistent_session": True,
        "argv": ["python3", "scripts/qwen35_gguf_bench.py"],
        "provenance": {
            "dirty": False,
            "resolved_backend": "hip_gfx1151",
            "target_arch": "gfx1151",
        },
        "runs": [
            {
                "measured": True,
                "effective_graph_replay_decode": True,
                "correctness_sanity": {"finite_final_logits": True, "final_token_id": 9707},
            }
        ],
        "persistent_session_memory": {
            "summary": {"tracked_peak_allocated_bytes": 1100},
            "snapshots": {
                "before_load": _snapshot(0, 10),
                "before_close": _snapshot(1000, 1010, breakdown),
                "after_graph_close": _snapshot(1000, 1000, breakdown),
                "after_close": _snapshot(0, 10),
            },
        },
    }
    g5 = {
        "status": "accepted",
        "performance_claim": True,
        "classification": {
            "decision": "promote_state_bound_graph_relaunch",
            "candidate_speedup_vs_eager": 1.001,
            "eager_median_ms_per_token": 20.3,
            "candidate_median_ms_per_token": 20.2,
        },
        "correctness": {
            "stable_key_relaunch": {
                "passed": True,
                "first_failing_launch": None,
                "third_and_later_launches_checked": 126,
                "comparisons": [{}] * 128,
            }
        },
        "provenance": {"hipengine_commit": "abc"},
    }

    artifact = _build_artifact(
        source,
        plan_rows=[
            _plan_row("token", "raw_gguf", ["raw"]),
            _plan_row("expert", "gguf_q4_k_t16_v1", ["tiles"]),
        ],
        g5=g5,
        source_sha256="source-sha",
        g5_sha256="g5-sha",
        source_path="/tmp/source.json",
        g5_path="benchmarks/results/g5.json",
        postprocess_command=["script"],
    )

    assert artifact["status"] == "accepted"
    assert artifact["classification"]["checks"]["no_default_raw_plus_replacement_duplicate"]
    assert artifact["allocation_census"]["kv_bytes"] == 250
    assert artifact["allocation_census"]["graph"]["hip_used_live_minus_closed_bytes"] == 10
    assert artifact["capacity_gate_24gib"]["owned_margin_bytes"] == 24 * (1 << 30) - 1000
