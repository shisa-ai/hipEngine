"""Focused tests for scripts/server_f1_concurrency_bench.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "server_f1_concurrency_bench.py"
    module_name = "_server_f1_concurrency_bench_test_module"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


def test_extract_hipengine_response_retains_exact_ids_and_route() -> None:
    payload = {
        "usage": {"prompt_tokens": 512, "completion_tokens": 3},
        "choices": [
            {
                "finish_reason": "length",
                "hipengine": {
                    "generated_token_ids": [11, 12, 13],
                    "timing": {"prefill_ms": 8.0, "decode_batch_ms": 9.5},
                    "decode_state": {
                        "execution_path": "gguf_packed_ar_server_decode",
                        "serial_decode_fallback": False,
                        "native_caware_decode": True,
                    },
                },
            }
        ],
        "hipengine": {
            "token_accounting": {
                "choice_generated_token_ids": [[11, 12, 13]],
                "total_generated_tokens": 3,
            },
            "generation_shape": {
                "queue_group": {"request_count": 2, "prompt_rows": 2},
                "backend_groups": [
                    {"input_rows": 2, "actual_group_rows": [2], "max_actual_group_rows": 2}
                ],
            },
        },
    }

    record = SCRIPT.extract_response("hipengine", payload, prompt_tokens=512)

    assert record["generated_token_ids"] == [11, 12, 13]
    assert record["completion_tokens"] == 3
    assert record["prompt_tokens"] == 512
    assert record["backend_timing_ms"]["decode_batch_ms"] == 9.5
    assert record["execution_path"] == "gguf_packed_ar_server_decode"
    assert record["serial_decode_fallback"] is False
    assert record["native_caware_decode"] is True
    assert record["generation_shape"]["queue_group"]["request_count"] == 2


def test_greedy_requests_keep_one_compatible_seed() -> None:
    args = type("Args", (), {"served_model_name": "model", "decode_tokens": 2, "seed": 12345})()

    first = SCRIPT._request_payload(args, "hipengine", [1, 2], 0)
    eighth = SCRIPT._request_payload(args, "hipengine", [3, 4], 7)

    assert first["seed"] == eighth["seed"] == 12345


def test_generation_shape_requires_one_real_backend_group() -> None:
    good = [
        {
            "generation_shape": {
                "queue_group": {"id": "group-4", "request_count": 4, "prompt_rows": 4},
                "backend_groups": [
                    {"input_rows": 4, "actual_group_rows": [4], "max_actual_group_rows": 4}
                ],
            }
        }
        for _ in range(4)
    ]
    bad = [
        {
            "generation_shape": {
                "queue_group": {"id": "group-1", "request_count": 1, "prompt_rows": 1},
                "backend_groups": [
                    {"input_rows": 1, "actual_group_rows": [1], "max_actual_group_rows": 1}
                ],
            }
        }
        for _ in range(4)
    ]

    assert SCRIPT.generation_shape_proves_native_group(good, concurrency=4)["passed"] is True
    assert SCRIPT.generation_shape_proves_native_group(bad, concurrency=4)["passed"] is False


def test_extract_llamacpp_response_requires_returned_token_ids() -> None:
    payload = {
        "tokens": [21, 22],
        "tokens_evaluated": 512,
        "timings": {
            "prompt_n": 512,
            "prompt_ms": 7.0,
            "predicted_n": 2,
            "predicted_ms": 4.0,
            "predicted_per_second": 500.0,
        },
        "stop": True,
    }

    record = SCRIPT.extract_response("llamacpp-hip", payload, prompt_tokens=512)

    assert record["generated_token_ids"] == [21, 22]
    assert record["completion_tokens"] == 2
    assert record["prompt_tokens"] == 512
    assert record["backend_timing_ms"] == {"prompt_ms": 7.0, "predicted_ms": 4.0}
    assert record["backend_decode_tok_s"] == 500.0

    with pytest.raises(ValueError, match="exact generated token IDs"):
        SCRIPT.extract_response("llamacpp-vulkan", {"timings": {"predicted_n": 2}}, prompt_tokens=512)


def test_prometheus_parser_and_hipengine_latency_snapshot() -> None:
    text = """
# HELP hipengine_resident_request_latency_seconds test
hipengine_resident_request_latency_seconds{kind="time_to_first_token",quantile="0.5"} 1.25
hipengine_resident_request_latency_seconds{kind="time_to_first_token",quantile="0.95"} 1.5
hipengine_resident_request_latency_seconds_sum{kind="time_to_first_token"} 5.0
hipengine_resident_request_latency_seconds_count{kind="time_to_first_token"} 4
hipengine_resident_request_latency_max_seconds{kind="time_to_first_token"} 1.6
hipengine_resident_request_latency_seconds{kind="inter_token",quantile="0.5"} 0.02
hipengine_resident_request_latency_seconds{kind="inter_token",quantile="0.95"} 0.03
hipengine_resident_request_latency_seconds_sum{kind="inter_token"} 10.0
hipengine_resident_request_latency_seconds_count{kind="inter_token"} 500
hipengine_resident_request_latency_max_seconds{kind="inter_token"} 0.04
hipengine_resident_route_total{route="gguf_packed_ar_server_decode"} 128
hipengine_resident_fallback_total{reason="compatibility"} 0
hipengine_resident_bucket_info{active_mask="1100",last_work_kind="decode",policy="protect_decode"} 1
hipengine_resident_bucket_active_rows 2
"""
    samples = SCRIPT.parse_prometheus(text)
    latency = SCRIPT.hipengine_latency_snapshot(samples)

    assert latency["time_to_first_token"] == {
        "count": 4.0,
        "sum": 5.0,
        "max": 1.6,
        "p50": 1.25,
        "p95": 1.5,
    }
    assert latency["inter_token"]["count"] == 500.0
    assert SCRIPT.prometheus_value(samples, "hipengine_resident_bucket_active_rows") == 2.0
    bucket = SCRIPT.prometheus_sample(samples, "hipengine_resident_bucket_info")
    assert bucket["labels"]["active_mask"] == "1100"
    assert bucket["labels"]["last_work_kind"] == "decode"


def test_metric_summary_uses_nearest_rank_p95_and_variance() -> None:
    summary = SCRIPT.metric_summary([10.0, 12.0, 11.0])
    assert summary["samples"] == [10.0, 12.0, 11.0]
    assert summary["median"] == 11.0
    assert summary["p95"] == 12.0
    assert summary["min"] == 10.0
    assert summary["max"] == 12.0
    assert summary["stdev"] == pytest.approx(1.0)
    assert summary["stdev_pct_of_median"] == pytest.approx(100.0 / 11.0)


def test_oracle_join_delay_supports_resident_request_total_residual() -> None:
    delay = SCRIPT._oracle_join_delay_seconds(
        [
            {
                "backend_timing_ms": {
                    "prefill_ms": 60.0,
                    "request_total_ms": 100.0,
                }
            }
        ],
        join_after_tokens=1,
        expected_tokens=2,
    )
    assert delay == pytest.approx(0.1)


def test_model_fingerprint_supports_directory_checkpoints(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "weights.bin").write_bytes(b"weights")

    fingerprint = SCRIPT._model_fingerprint(model)

    assert fingerprint["path"] == str(model.resolve())
    assert fingerprint["path_type"] == "directory"
    assert fingerprint["file_count"] == 2
    assert fingerprint["size_bytes"] == 9
    assert fingerprint["revision"] is None


def test_paro_oracle_join_delay_uses_http_wall_without_prefill_breakdown() -> None:
    delay = SCRIPT._oracle_join_delay_seconds(
        [
            {
                "backend_timing_ms": {"request_total_ms": 600.0},
                "wall_seconds": 0.8,
            }
        ],
        join_after_tokens=2,
        expected_tokens=8,
    )
    assert delay == pytest.approx(0.72)


def test_hipengine_route_expectation_accepts_width1_and_native_or_serial_cn() -> None:
    assert SCRIPT._hipengine_route_expectation_passes(
        concurrency=1,
        expectation="native",
        serial_values=[True],
        native_values=[False],
        shape_passed=True,
        resident_capacity=1.0,
    )
    assert SCRIPT._hipengine_route_expectation_passes(
        concurrency=2,
        expectation="native",
        serial_values=[False, False],
        native_values=[True, True],
        shape_passed=True,
        resident_capacity=2.0,
    )
    assert SCRIPT._hipengine_route_expectation_passes(
        concurrency=8,
        expectation="serial",
        serial_values=[True] * 8,
        native_values=[False] * 8,
        shape_passed=False,
        resident_capacity=8.0,
    )
    assert not SCRIPT._hipengine_route_expectation_passes(
        concurrency=4,
        expectation="native",
        serial_values=[True] * 4,
        native_values=[False] * 4,
        shape_passed=False,
        resident_capacity=4.0,
    )


def test_parse_ldd_local_paths_only_retains_build_tree(tmp_path: Path) -> None:
    repo = tmp_path / "llama.cpp"
    local = repo / "build" / "bin" / "libllama.so"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"local")
    external = tmp_path / "libsystem.so"
    external.write_bytes(b"external")
    text = f"libllama.so => {local} (0x1)\nlibsystem.so => {external} (0x2)\n"

    assert SCRIPT.parse_ldd_local_paths(text, root=repo) == [local.resolve()]


def test_correctness_summary_matches_each_prompt_to_c1_oracle() -> None:
    prompt_a = [1, 1, 1]
    prompt_b = [1, 1, 2]
    oracle = {
        SCRIPT.token_ids_sha256(prompt_a): [7, 8],
        SCRIPT.token_ids_sha256(prompt_b): [9, 10],
    }
    records = [
        {"prompt_token_ids_sha256": SCRIPT.token_ids_sha256(prompt_a), "generated_token_ids": [7, 8]},
        {"prompt_token_ids_sha256": SCRIPT.token_ids_sha256(prompt_b), "generated_token_ids": [9, 10]},
    ]

    passed = SCRIPT.correctness_summary(records, oracle=oracle, expected_tokens=2)
    assert passed["passed"] is True
    assert passed["mismatch_count"] == 0
    assert passed["exact_rows"] == 2

    records[1]["generated_token_ids"] = [9, 11]
    failed = SCRIPT.correctness_summary(records, oracle=oracle, expected_tokens=2)
    assert failed["passed"] is False
    assert failed["mismatch_count"] == 1
    assert failed["mismatches"][0]["first_mismatch_index"] == 1
