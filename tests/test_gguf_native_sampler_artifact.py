from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "benchmarks"
    / "results"
    / "2026-07-21-w7900-gguf-native-sampler-correctness.json"
)


def test_w7900_gguf_native_sampler_correctness_artifact_contract() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["schema"] == 1
    assert payload["kind"] == "gguf_native_sampler_correctness_gate"
    assert payload["status"] == "passed"
    assert payload["passed"] is True
    assert payload["correctness_claim"] is True
    assert payload["performance_claim"] is False
    assert payload["timing_claim"] is False
    assert payload["hardware"] == "AMD Radeon Pro W7900 (gfx1100)"
    assert payload["backend"] == "hip_gfx1100"
    assert payload["quant"] == "gguf_q4_k_m"

    workload = payload["workload"]
    assert workload["physical_rows"] == 4
    assert workload["prompt_tokens"] == 256
    assert workload["max_tokens"] == 4
    assert workload["top_k"] == 8
    assert workload["top_logprobs"] == 3
    assert set(workload["processors"]) == {
        "logit_bias",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "suppress_token_ids",
        "min_tokens",
    }

    repeat = payload["fixed_seed_repeat"]
    assert repeat["exact"] is True
    assert repeat["mismatches"] == []
    first_rows = repeat["first"]["rows"]
    second_rows = repeat["second"]["rows"]
    assert len(first_rows) == len(second_rows) == 4
    assert [row["generated_token_ids"] for row in first_rows] == [
        row["generated_token_ids"] for row in second_rows
    ]
    for row in (*first_rows, *second_rows):
        decode = row["decode_state"]
        assert decode["sampler_mode"] == "gpu_sample"
        assert decode["native_sampler_rows"] is True
        assert decode["full_vocab_logits_d2h"] is False
        assert decode["logits_d2h_bytes"] == 0
        assert "sampler_fallback_reason" not in decode
        assert len(row["token_logprobs"]) == len(row["generated_token_ids"])
        assert row["stream_token_logprob_count"] == 1
        assert all(sample["mode"] == "gpu_sample" for sample in row["samples"])

    oracle = payload["forced_host_oracle"]
    assert oracle["exact"] is True
    assert oracle["fallback_exact"] is True
    assert oracle["mismatches"] == []
    assert len(oracle["rows"]) == 4
    for row in oracle["rows"]:
        decode = row["decode_state"]
        assert decode["sampler_mode"] == "processed_argmax"
        assert decode["native_sampler_rows"] is False
        assert decode["sampler_fallback_reason"] == "processed_logits_required"
        assert decode["full_vocab_logits_d2h"] is True
        assert decode["logits_d2h_bytes"] > 0

    finish = payload["finish_routes"]
    assert finish["stop_exact"] is True
    assert finish["stop"]["finish_details"]["reason"] == "stop"
    assert finish["stop"]["finish_details"]["sampler_mode"] == "gpu_sample"
    assert finish["eos_exact"] is True
    assert finish["eos"]["finish_details"]["reason"] == "eos"
    assert finish["eos"]["finish_details"]["sampler_mode"] == "gpu_sample"

    telemetry = payload["telemetry"]
    assert telemetry["native_exact"] is True
    assert telemetry["batch_route_exact"] is True
    counts = telemetry["observability"]["routes"]["counts"]
    assert counts["native_sampler_batch_launches"] == 6
    assert counts["native_sampler_requests"] == 11
    assert counts["host_sampler_requests"] == 4
    assert counts["serial_decode_fallback_steps"] == 0

    ownership = payload["ownership"]
    assert ownership["exact"] is True
    assert ownership["active_request_ids"] == []
    assert ownership["available_sessions"] == 4
    assert ownership["final_pool"]["refcounted_pages"] == 0
    assert ownership["final_pool"]["pinned_pages"] == 0
    assert ownership["final_pool"]["cow_fork_events"] == 0

    assert "scripts/gguf_native_sampler_gate.py" in payload["command"]
