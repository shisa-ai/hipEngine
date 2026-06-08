from __future__ import annotations

from pathlib import Path

from scripts.qwen35_paro_bench import _memory_summary, _workload_summary


def test_qwen35_paro_bench_workload_summary_labels_c1_shape() -> None:
    workload = _workload_summary(
        model=Path("/models/qwen-paro"),
        prompt_length=512,
        decode_tokens=128,
        warmup_decode_tokens=4,
        max_layers=40,
        kv_policy_summary={"storage_dtype": "bf16"},
    )

    assert workload == {
        "shape": "c=1 prompt=512 decode=128",
        "model": "Qwen3.5-35B-A3B-PARO",
        "model_path": "/models/qwen-paro",
        "quant": "w4_paro",
        "prompt_tokens_per_request": 512,
        "prompt_tokens_aggregate": 512,
        "gen_tokens_per_request": 128,
        "gen_tokens_aggregate": 128,
        "warmup_decode_tokens": 4,
        "concurrency": 1,
        "prompt_lengths": [512],
        "max_layers": 40,
        "kv_policy": {"storage_dtype": "bf16"},
    }


def test_qwen35_paro_bench_memory_summary_embeds_kv_audit_and_peaks() -> None:
    snapshots = {
        "after_prefill": {
            "tracked": {"peak_allocated_bytes": 128, "current_allocated_bytes": 96},
            "hip": {"available": True, "used_bytes": 2048},
            "kv_memory_audit": {"passed": True, "kv_storage_dtype": "int8_per_token_head"},
        },
        "before_close": {
            "tracked": {"peak_allocated_bytes": 256, "current_allocated_bytes": 64},
            "hip": {"available": True, "used_bytes": 4096},
            "owned_session_bytes": 512,
            "kv_memory_audit": {
                "passed": True,
                "kv_storage_dtype": "int8_per_token_head",
                "retained_kv_payload_bytes_per_element": 1.0,
            },
        },
        "after_close": {
            "tracked": {"peak_allocated_bytes": 256, "current_allocated_bytes": 0},
            "hip": {"available": True, "used_bytes": 1024},
        },
    }

    summary = _memory_summary(snapshots)

    assert summary["tracked_peak_allocated_bytes"] == 256
    assert summary["hip_used_peak_sampled_bytes"] == 4096
    assert summary["kv_memory_audit"]["passed"] is True
    assert summary["kv_memory_audit"]["latest_label"] == "before_close"
    assert summary["kv_memory_audit"]["latest"]["retained_kv_payload_bytes_per_element"] == 1.0
    assert summary["kv_memory_audit"]["tracked_peak_allocated_bytes"] == 256
    assert summary["kv_memory_audit"]["hip_used_peak_sampled_bytes"] == 4096
