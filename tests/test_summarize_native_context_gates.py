import json
from copy import deepcopy

import pytest

from scripts.summarize_native_context_gates import summarize, validate_performance_pair


def test_state_summary_preserves_failed_attempt_and_actual_graph_coverage(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "status": "failed", "error": "rollback mismatch",
        "rows": [{"context": 1020, "budget": 3, "transport": "graph",
                  "graph": False, "mode": "native", "passed": True,
                  "prompt_ids": [1, 2], "following_step": True}],
    }))
    result = summarize(path)
    assert result["status"] == "failed"
    assert result["error"] == "rollback mismatch"
    assert result["cases"] == result["passed"] == result["following_step_cases"] == 1
    assert result["coverage"][0]["graph"] is False
    assert "prompt_ids" not in json.dumps(result)
    assert len(result["source_sha256"]) == 64


def test_e2e_summary_keeps_interference_and_hashes_prompts(tmp_path):
    path = tmp_path / "e2e.json"
    path.write_text(json.dumps({
        "status": "invalid_gpu_interference", "rows": {"true_ar": []},
        "context_gate": {"command": ["test"], "prompt_ids": {"code": [1, 2]},
                         "foreign_gpu_allocations": {"42": 10000000}},
        "summary": {"exact": False},
    }))
    result = summarize(path)
    assert result["command"] == ["test"]
    assert result["context_gate"]["foreign_gpu_allocations"] == {"42": 10000000}
    assert result["context_gate"]["prompt_fingerprints"]["code"]["tokens"] == 2
    assert "prompt_ids" not in result["context_gate"]
    assert result["performance_claim"] is False


def _performance_pair():
    from scripts.qwen36_dense_gguf_suite import FULL_PROMPT_IDS

    data = {
        "status": "complete_exact",
        "provenance": {key: "same" for key in (
            "host_name", "hipengine_commit", "resolved_backend", "target_arch",
            "device_name", "model_fingerprint", "quant", "kv_dtype",
            "rocm_version", "hipcc_version",
        )},
        "context_gate": {
            "identity": {"dirty": False}, "foreign_gpu_allocations": {},
            "fixed_prompt_length": None, "native_eager": False, "bulk_prefill": False,
            "native_context_limit": 95, "actual_verify_modes": {"serial_exact": 12},
            "graph_submissions": 5, "pci": "gpu1", "kfd_gpu_id": 33912,
            "environment": {"HIP_VISIBLE_DEVICES": "1"},
        },
        "workload": {"prompt_ids": list(FULL_PROMPT_IDS), "runs": 3, "warmup": True,
                     "candidate_budgets": [3], "max_new_tokens_visible": 129},
        "correctness": {"all_exact_greedy": True, "all_gpu_accept_match_cpu": True},
        "memory_after_close": {"active_allocations": 0},
        "summary": {"true_ar": {"full": {"request_count": 30}},
                    "mtp": {"3": {"full": {"request_count": 30, "decode_tok_s_weighted": 30}}}},
    }
    data["provenance"]["dirty"] = False
    after = deepcopy(data)
    after["context_gate"].update(native_context_limit=None, actual_verify_modes={"native": 20})
    after["summary"]["mtp"]["3"]["full"]["decode_tok_s_weighted"] = 60
    return data, after


def test_performance_pair_accepts_clean_matched_runs():
    assert validate_performance_pair(*_performance_pair())["mtp_change_percent"] == 100


@pytest.mark.parametrize("fault", ["dirty", "interference", "fallback", "shape", "host", "partial"])
def test_performance_pair_rejects_invalid_evidence(fault):
    before, after = _performance_pair()
    if fault == "dirty":
        after["provenance"]["dirty"] = True
    elif fault == "interference":
        after["context_gate"]["foreign_gpu_allocations"] = {"123": 10000000}
    elif fault == "fallback":
        after["context_gate"]["actual_verify_modes"]["serial_exact"] = 1
    elif fault == "shape":
        after["workload"]["max_new_tokens_visible"] = 25
    elif fault == "host":
        after["provenance"]["host_name"] = "other-host"
    else:
        after["summary"]["mtp"]["3"]["full"]["request_count"] = 29
    with pytest.raises(ValueError):
        validate_performance_pair(before, after)
