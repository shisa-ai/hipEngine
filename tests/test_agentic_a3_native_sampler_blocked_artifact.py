from __future__ import annotations

import hashlib
import json
from pathlib import Path


ARTIFACT = Path(
    "benchmarks/results/2026-07-22-w7900-agentic-a3-native-sampler-blocked.json"
)
CORRECTNESS = Path(
    "benchmarks/results/2026-07-21-w7900-gguf-native-sampler-correctness.json"
)
A1_CONTROL = Path(
    "benchmarks/results/2026-07-21-w7900-agentic-a1-repeated-baseline.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_agentic_a3_stops_before_timing_when_no_native_tool_route_is_valid() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["kind"] == "gfx1100_agentic_a3_native_sampler_blocked"
    assert payload["status"] == "blocked_fail_closed_tool_correctness"
    assert payload["passed"] is False
    assert payload["route_characterization_valid"] is True
    assert payload["measurement_valid"] is False
    assert payload["correctness_claim"] is False
    assert payload["performance_claim"] is False
    assert payload["timing_claim"] is False
    assert payload["source_clean_and_pushed"] is True
    assert payload["source_revision"] == "2f8f6bf19b00a7487965aa468df19d2271462d89"

    protocol = payload["protocol"]
    assert protocol["planned_families"] == [
        "small_repo",
        "growing_history",
        "medium_repo",
    ]
    assert protocol["planned_logical_concurrency"] == [1, 4, 8]
    assert protocol["preflight_transport"] == (
        "real_localhost_uvicorn_blocking_oracle_before_measured_sse"
    )
    assert protocol["sampling"] == {
        "enable_thinking": False,
        "frequency_penalty": 0.02,
        "logprobs": True,
        "max_tokens": 64,
        "min_p": 0.08,
        "presence_penalty": 0.1,
        "repetition_penalty": 1.05,
        "seed": 17,
        "temperature": 0.85,
        "top_k": 8,
        "top_logprobs": 3,
        "top_p": 0.82,
    }
    commands = payload["commands"]
    assert "HIPENGINE_QWEN35_NATIVE_SAMPLER=0" in commands["host_auto_server"]
    assert "HIPENGINE_QWEN35_NATIVE_SAMPLER=1" in commands["native_auto_server"]
    assert "HIPENGINE_QWEN35_NATIVE_SAMPLER=1" in commands["native_strict_server"]
    assert commands["host_auto_preflight"].endswith(
        "uv run python /tmp/a3_host_auto_all_diag.py"
    )
    assert commands["native_auto_preflight"].endswith(
        "uv run python /tmp/a3_retained_native_auto_hightemp.py"
    )

    linked = payload["linked_correctness_prerequisite"]
    assert linked["sha256"] == _sha256(CORRECTNESS)
    assert linked["fixed_seed_repeat_exact"] is True
    assert linked["cpu_reference_distribution_sanity"] is True
    assert linked["native_full_vocab_d2h_zero"] is True
    assert payload["retained_a1_guard"]["sha256"] == _sha256(A1_CONTROL)

    auto = payload["preflight"]["native_eligible_auto_tool"]
    host = auto["host"]
    native = auto["native"]
    assert host["turn0_repeat0"]["generated_token_ids"] == host["turn0_repeat1"][
        "generated_token_ids"
    ]
    assert native["turn0_repeat0"]["generated_token_ids"] == native[
        "turn0_repeat1"
    ]["generated_token_ids"]
    assert auto["turn0_host_native_generated_ids_equal"] is True
    assert host["turn0_repeat0"]["sampler"]["mode"] == "host_logits_sample"
    assert host["turn0_repeat0"]["sampler"]["full_vocab_logits_d2h"] is True
    assert native["turn0_repeat0"]["sampler"]["mode"] == "gpu_sample"
    assert native["turn0_repeat0"]["sampler"]["full_vocab_logits_d2h"] is False
    assert native["turn0_repeat0"]["sampler"]["logits_d2h_bytes_total"] == 0

    for route in (host, native):
        failed = route["turn1_failed"]
        assert failed["finish_reason"] == "length"
        assert failed["finish_details"]["reason"] == "invalid_tool_call"
        assert failed["generated_tokens"] == 64
        assert failed["tool_calls"] == []
    assert host["turn1_failed"]["sampler"]["logits_d2h_bytes_total"] == 64 * 993_280
    assert native["turn1_failed"]["sampler"]["logits_d2h_bytes_total"] == 0

    strict = payload["preflight"]["strict_forced_tool"]
    assert strict["turns_valid"] == strict["turns_total"] == 4
    assert strict["fixed_seed_repeat_exact"] is True
    assert strict["generated_tokens_across_two_repeats"] == 200
    assert strict["full_vocab_logits_d2h_bytes_across_two_repeats"] == 198_656_000
    assert strict["sampler_mode"] == "host_logits_sample"
    assert strict["fallback_reason"] == "native_gpu_unsupported_request"
    for turn in strict["turns"].values():
        first, second = turn["responses"]
        assert turn["fixed_seed_repeat_exact"] is True
        assert first["generated_token_ids"] == second["generated_token_ids"]
        assert first["tool_calls"] == second["tool_calls"]
        assert first["finish_reason"] == second["finish_reason"] == "tool_calls"
        for response in (first, second):
            assert response["sampler"]["mode"] == "host_logits_sample"
            assert response["sampler"]["native_sampler_rows"] is False
            assert response["sampler"]["fallback_reason"] == (
                "native_gpu_unsupported_request"
            )
            assert response["sampler"]["full_vocab_logits_d2h"] is True

    ownership = payload["preflight"]["final_ownership"]
    assert ownership["all_zero"] is True
    for server in ("host", "native", "strict_native_server"):
        assert all(value == 0 for value in ownership[server].values())

    disposition = payload["matrix_disposition"]
    assert disposition["full_host_native_matrix_started"] is False
    assert disposition["performance_conditions_measured"] == 0
    assert disposition["c1_c4_c8_timing_skipped"] is True
    assert disposition["inferred_timing_used"] is False
    assert disposition["active_sse_comparison_available"] is False
    assert disposition["tool_ready_comparison_available"] is False

    acceptance = payload["acceptance"]
    assert acceptance["frozen_sampled_tool_contract_passed"] is False
    assert acceptance["complete_matrix_measured"] is False
    assert acceptance["promote_native_sampler"] is False
    assert acceptance["default_native_sampler"] is False
    assert payload["decision"].startswith(
        "keep HIPENGINE_QWEN35_NATIVE_SAMPLER default-off"
    )
