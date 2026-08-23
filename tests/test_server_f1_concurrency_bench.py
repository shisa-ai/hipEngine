"""Focused tests for scripts/server_f1_concurrency_bench.py."""

from __future__ import annotations

import importlib.util
import json
import socket
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
                    "diagnostics": {
                        "kv_layout": {"persistent_bf16_mirror_bytes": 0}
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
    assert record["diagnostics"]["kv_layout"]["persistent_bf16_mirror_bytes"] == 0
    assert record["generation_shape"]["queue_group"]["request_count"] == 2


def test_greedy_requests_keep_one_compatible_seed() -> None:
    args = type(
        "Args",
        (),
        {
            "served_model_name": "model",
            "decode_tokens": 2,
            "seed": 12345,
            "hipengine_top_k": 0,
        },
    )()

    first = SCRIPT._request_payload(args, "hipengine", [1, 2], 0)
    eighth = SCRIPT._request_payload(args, "hipengine", [3, 4], 7)

    assert first["seed"] == eighth["seed"] == 12345
    assert first["top_k"] == eighth["top_k"] == 0


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


def test_generation_shape_accepts_grouped_c13_as_physical_c8_plus_c5() -> None:
    grouped = [
        {
            "generation_shape": {
                "queue_group": {
                    "id": "group-13",
                    "request_count": 13,
                    "prompt_rows": 13,
                    "item_index": index,
                    "item_prompt_offset": index,
                    "item_prompt_rows": 1,
                },
                "backend_groups": [
                    {
                        "input_rows": 13,
                        "actual_group_rows": [8, 5],
                        "max_actual_group_rows": 8,
                    }
                ],
            }
        }
        for index in range(13)
    ]

    summary = SCRIPT.generation_shape_proves_native_group(grouped, concurrency=13)

    assert summary["passed"] is True
    assert summary["queue_group_count"] == 1
    assert summary["backend_group_rows"] == [8, 5]
    assert summary["max_backend_group_rows"] == 8
    assert summary["native_false_records_expected"] == 0


def test_generation_shape_accepts_grouped_c32_as_four_physical_c8_groups() -> None:
    grouped = [
        {
            "generation_shape": {
                "queue_group": {
                    "id": "group-32",
                    "request_count": 32,
                    "prompt_rows": 32,
                    "item_index": index,
                    "item_prompt_offset": index,
                    "item_prompt_rows": 1,
                },
                "backend_groups": [
                    {
                        "input_rows": 32,
                        "actual_group_rows": [8, 8, 8, 8],
                        "max_actual_group_rows": 8,
                    }
                ],
            }
        }
        for index in range(32)
    ]

    summary = SCRIPT.generation_shape_proves_native_group(grouped, concurrency=32)

    assert summary["passed"] is True
    assert summary["queue_group_count"] == 1
    assert summary["backend_group_rows"] == [8, 8, 8, 8]
    assert summary["max_backend_group_rows"] == 8


def test_generation_shape_accepts_c13_as_complete_route_cap_four_groups() -> None:
    records = []
    for group_index, rows in enumerate((4, 4, 4, 1)):
        for item_index in range(rows):
            records.append(
                {
                    "generation_shape": {
                        "queue_group": {
                            "id": f"group-{group_index}",
                            "request_count": rows,
                            "prompt_rows": rows,
                            "item_index": item_index,
                            "item_prompt_offset": item_index,
                            "item_prompt_rows": 1,
                        },
                        "backend_groups": [
                            {
                                "input_rows": rows,
                                "actual_group_rows": [rows],
                                "max_actual_group_rows": rows,
                            }
                        ],
                    }
                }
            )

    summary = SCRIPT.generation_shape_proves_native_group(records, concurrency=13)

    assert summary["passed"] is True
    assert summary["shared_queue_group"] is False
    assert summary["queue_group_request_counts"] == [4, 4, 4, 1]
    assert summary["backend_group_rows"] == [4, 4, 4, 1]
    assert summary["native_false_records_expected"] == 1


def test_matched_concurrency_plan_accepts_arbitrary_logical_widths_through_c32() -> None:
    assert SCRIPT._validate_concurrency_plan(
        [1, 2, 4, 8, 13, 16, 17, 32],
        live_concurrency=17,
    ) == [1, 2, 4, 8, 13, 16, 17, 32]
    with pytest.raises(ValueError, match="must include c1"):
        SCRIPT._validate_concurrency_plan([2], live_concurrency=2)
    assert SCRIPT._validate_concurrency_plan(
        [2],
        live_concurrency=2,
        require_c1=False,
    ) == [2]
    with pytest.raises(ValueError, match="limited to logical c1-c32"):
        SCRIPT._validate_concurrency_plan([1, 2, 33], live_concurrency=2)


def test_live_admission_uses_sse_when_streaming_is_primary() -> None:
    streaming = type("Args", (), {"streaming_primary": True})()
    blocking = type("Args", (), {"streaming_primary": False})()

    assert SCRIPT._live_request_function(streaming) is SCRIPT._one_stream_request
    assert SCRIPT._live_request_function(blocking) is SCRIPT._one_request


def test_streaming_live_gate_requires_decode_interval_and_resident_overlap() -> None:
    args = type("Args", (), {"streaming_primary": True})()
    live = {
        "admission_during_first_request": True,
        "request_protocol": "streaming_sse",
        "join_during_observed_first_stream_decode": True,
        "resident_overlap_before_first_completion": True,
    }

    assert SCRIPT._live_admission_passes("hipengine", args, live) is True
    for field in (
        "join_during_observed_first_stream_decode",
        "resident_overlap_before_first_completion",
    ):
        rejected = live | {field: False}
        assert SCRIPT._live_admission_passes("hipengine", args, rejected) is False


def test_extract_stream_responses_records_client_ttft_itl_and_tokens() -> None:
    hip = SCRIPT.extract_stream_response(
        "hipengine",
        [
            (
                10.2,
                {
                    "choices": [
                        {
                            "text": "A",
                            "finish_reason": None,
                            "hipengine": {"tokens": {"delta_tokens": 1, "streamed_tokens": 1}},
                        }
                    ]
                },
            ),
            (
                10.5,
                {
                    "choices": [
                        {
                            "text": "B",
                            "finish_reason": None,
                            "hipengine": {"tokens": {"delta_tokens": 1, "streamed_tokens": 2}},
                        }
                    ]
                },
            ),
            (
                10.6,
                {
                    "choices": [
                        {
                            "text": "",
                            "finish_reason": "length",
                            "hipengine": {
                                "diagnostics": {
                                    "kv_layout": {
                                        "persistent_bf16_mirror_bytes": 0
                                    }
                                }
                            },
                        }
                    ]
                },
            ),
            (10.7, {"choices": [], "usage": {"prompt_tokens": 512, "completion_tokens": 2}}),
            (10.8, "[DONE]"),
        ],
        started_at=10.0,
        completed_at=10.8,
        prompt_tokens=512,
    )
    llama = SCRIPT.extract_stream_response(
        "llamacpp-hip",
        [
            (20.4, {"content": "A", "tokens": [21], "stop": False}),
            (20.7, {"content": "B", "tokens": [22], "stop": False}),
            (
                20.9,
                {
                    "content": "",
                    "stop": True,
                    "stop_type": "limit",
                    "timings": {"prompt_n": 512, "predicted_n": 2, "predicted_ms": 500.0},
                },
            ),
        ],
        started_at=20.0,
        completed_at=20.9,
        prompt_tokens=512,
    )

    assert hip["text"] == llama["text"] == "AB"
    assert hip["completion_tokens"] == llama["completion_tokens"] == 2
    assert hip["generated_token_ids"] is None
    assert llama["generated_token_ids"] == [21, 22]
    assert hip["client_ttft_seconds"] == pytest.approx(0.2)
    assert hip["client_inter_token_seconds"] == pytest.approx([0.3])
    assert hip["done_sentinel"] is True
    assert hip["diagnostics"]["kv_layout"]["persistent_bf16_mirror_bytes"] == 0
    assert llama["client_ttft_seconds"] == pytest.approx(0.4)
    assert llama["client_inter_token_seconds"] == pytest.approx([0.3])


def test_stream_route_summary_requires_native_nonserial_hipengine_cn() -> None:
    sample = {
        "records": [
            {
                "execution_path": "gguf_packed_ar_server_decode",
                "serial_decode_fallback": False,
                "native_caware_decode": True,
            }
            for _ in range(13)
        ]
    }

    summary = SCRIPT._stream_route_summary("hipengine", concurrency=13, samples=[sample])

    assert summary["passed"] is True
    sample["records"][0]["serial_decode_fallback"] = True
    assert SCRIPT._stream_route_summary(
        "hipengine", concurrency=13, samples=[sample]
    )["passed"] is False


def test_stream_route_summary_accepts_laguna_scheduler_owned_c1_model_steps() -> None:
    sample = {
        "records": [
            {
                "execution_path": "laguna_resident_scheduler_c1",
                "serial_decode_fallback": False,
                "native_caware_decode": False,
            }
            for _ in range(2)
        ]
    }

    summary = SCRIPT._stream_route_summary(
        "hipengine", concurrency=2, samples=[sample]
    )

    assert summary["passed"] is True
    assert summary["route_policy"] == "scheduler_native_model_c1"
    sample["records"][0]["serial_decode_fallback"] = True
    assert SCRIPT._stream_route_summary(
        "hipengine", concurrency=2, samples=[sample]
    )["passed"] is False


def test_stream_route_summary_accepts_paro_native_with_serial_c1_edges() -> None:
    native_record = {
        "execution_path": "paro_resident_native_width_decode",
        "serial_decode_fallback": False,
        "native_caware_decode": True,
    }
    native_with_serial_edge_record = {
        "execution_path": "paro_resident_native_width_decode",
        "serial_decode_fallback": True,
        "native_caware_decode": True,
    }
    serial_edge_record = {
        "execution_path": "paro_resident_serial_decode",
        "serial_decode_fallback": True,
        "native_caware_decode": True,
    }
    c1_record = serial_edge_record | {"native_caware_decode": False}
    sample = {
        "records": [
            native_record,
            native_with_serial_edge_record,
            serial_edge_record,
        ]
    }

    summary = SCRIPT._stream_route_summary(
        "hipengine", concurrency=4, samples=[sample]
    )

    assert summary["passed"] is True
    assert summary["route_policy"] == "paro_occupancy_adaptive"
    assert SCRIPT._stream_route_summary(
        "hipengine", concurrency=1, samples=[{"records": [c1_record]}]
    )["passed"] is True

    sample["records"][0]["native_caware_decode"] = False
    assert SCRIPT._stream_route_summary(
        "hipengine", concurrency=4, samples=[sample]
    )["passed"] is False


def test_stream_batch_summary_applies_per_request_slo_goodput() -> None:
    records = [
        {
            "completion_tokens": 2,
            "client_ttft_seconds": 0.5,
            "client_inter_token_seconds": [0.1],
            "wall_seconds": 1.5,
            "stream_exact": True,
            "stream_protocol_complete": True,
        },
        {
            "completion_tokens": 2,
            "client_ttft_seconds": 0.7,
            "client_inter_token_seconds": [0.2],
            "wall_seconds": 1.6,
            "stream_exact": True,
            "stream_protocol_complete": True,
        },
    ]

    summary = SCRIPT._stream_batch_summary(
        records,
        batch_wall_seconds=2.0,
        ttft_p95_limit=1.0,
        itl_p99_limit=0.25,
        e2e_p95_limit=2.0,
    )

    assert summary["passed"] is True
    assert summary["exact_generated_tok_s_aggregate"] == pytest.approx(2.0)
    assert summary["slo_goodput_tok_s_aggregate"] == pytest.approx(2.0)
    assert summary["slo"]["qualifying_requests"] == 2
    assert summary["latency_seconds"]["ttft"]["p95"] == pytest.approx(0.7)
    assert summary["latency_seconds"]["itl"]["p99"] == pytest.approx(0.2)


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
hipengine_resident_kv_int8_payload_bytes 32768
hipengine_resident_kv_bf16_payload_bytes 0
hipengine_resident_kv_scale_bytes 2048
hipengine_resident_kv_bf16_mirror_bytes 0
hipengine_resident_kv_total_bytes 34816
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
    poll = SCRIPT._compact_poll_state(samples, at_seconds=1.25)
    assert poll["kv_int8_payload_bytes"] == 32768.0
    assert poll["kv_bf16_payload_bytes"] == 0.0
    assert poll["kv_scale_bytes"] == 2048.0
    assert poll["kv_bf16_mirror_bytes"] == 0.0
    assert poll["kv_total_bytes"] == 34816.0


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


def test_hipengine_parser_locks_the_retained_prefill_decode_policy(tmp_path: Path) -> None:
    args = SCRIPT.build_parser().parse_args(
        ["--engine", "hipengine", "--json", str(tmp_path / "result.json")]
    )
    assert args.correctness_profile == "strict"
    assert args.production_correctness_artifact is None
    assert args.hipengine_prefill_decode_policy == "protect_ttft"
    assert args.hipengine_kv_storage == "bf16"
    assert args.hipengine_kv_scale_dtype == "fp16"
    assert args.hipengine_kv_scale_granularity == "per_token_head"
    assert args.batch_window_ms == 5.0
    assert args.hipengine_prefill_chunk_tokens is None
    assert args.same_server_oracle is False
    assert args.memory_sample_through_shutdown is False
    lifecycle_args = SCRIPT.build_parser().parse_args(
        [
            "--engine",
            "hipengine",
            "--memory-sample-through-shutdown",
            "--same-server-oracle",
            "--json",
            str(tmp_path / "lifecycle.json"),
        ]
    )
    assert lifecycle_args.memory_sample_through_shutdown is True
    assert lifecycle_args.same_server_oracle is True


def test_hipengine_command_separates_generation_window_from_prefill_chunk(
    tmp_path: Path,
) -> None:
    args = SCRIPT.build_parser().parse_args(
        [
            "--engine",
            "hipengine",
            "--generation-batch-window-ms",
            "5",
            "--hipengine-prefill-chunk-tokens",
            "256",
            "--json",
            str(tmp_path / "result.json"),
        ]
    )

    command, env, _cwd = SCRIPT._server_command_and_env(
        args,
        engine="hipengine",
        concurrency=8,
        port=19123,
    )

    assert command[command.index("--generation-batch-window-ms") + 1] == "5.0"
    assert env["HIPENGINE_MAX_PREFILL_CHUNK_TOKENS"] == "256"
    assert env["HIPENGINE_GGUF_AR_PACKED_DECODE"] == "1"


def test_hipengine_serial_route_disables_actual_packed_decode_selector(
    tmp_path: Path,
) -> None:
    args = SCRIPT.build_parser().parse_args(
        [
            "--engine",
            "hipengine",
            "--hipengine-route-expectation",
            "serial-c1-per-row",
            "--json",
            str(tmp_path / "result.json"),
        ]
    )
    _command, env, _cwd = SCRIPT._server_command_and_env(
        args,
        engine="hipengine",
        concurrency=8,
        port=19123,
    )
    assert env["HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE"] == "0"
    assert env["HIPENGINE_GGUF_AR_PACKED_DECODE"] == "0"


def test_hipengine_command_forwards_explicit_int8_kv_policy(tmp_path: Path) -> None:
    args = SCRIPT.build_parser().parse_args(
        [
            "--engine",
            "hipengine",
            "--hipengine-kv-storage",
            "int8_per_token_head",
            "--hipengine-kv-scale-dtype",
            "fp32",
            "--json",
            str(tmp_path / "result.json"),
        ]
    )

    command, env, _cwd = SCRIPT._server_command_and_env(
        args,
        engine="hipengine",
        concurrency=2,
        port=19123,
    )

    assert env["HIP_VISIBLE_DEVICES"] == str(args.gpu)
    assert "ROCR_VISIBLE_DEVICES" not in env
    assert command[command.index("--kv-storage") + 1] == "int8_per_token_head"
    assert command[command.index("--kv-scale-dtype") + 1] == "fp32"
    assert command[command.index("--kv-scale-granularity") + 1] == "per_token_head"


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
        serial_values=[False] * 6,
        native_values=[True] * 6,
        shape_passed=True,
        resident_capacity=2.0,
    )
    assert SCRIPT._hipengine_route_expectation_passes(
        concurrency=13,
        expectation="native",
        serial_values=[False] * 13,
        native_values=[True] * 12 + [False],
        shape_passed=True,
        resident_capacity=13.0,
        native_false_records_expected=1,
    )
    assert SCRIPT._hipengine_route_expectation_passes(
        concurrency=8,
        expectation="serial",
        serial_values=[True] * 8,
        native_values=[False] * 8,
        shape_passed=False,
        resident_capacity=8.0,
    )
    assert SCRIPT._hipengine_route_expectation_passes(
        concurrency=1,
        expectation="serial-c1-per-row",
        serial_values=[False] * 3,
        native_values=[False] * 3,
        shape_passed=True,
        resident_capacity=1.0,
        execution_paths=["gguf_packed_ar_server_decode"],
    )
    assert SCRIPT._hipengine_route_expectation_passes(
        concurrency=4,
        expectation="serial-c1-per-row",
        serial_values=[True] * 12,
        native_values=[False] * 12,
        shape_passed=True,
        resident_capacity=4.0,
        execution_paths=["gguf_packed_ar_server_decode"],
    )
    assert SCRIPT._hipengine_route_expectation_passes(
        concurrency=4,
        expectation="serial-c1-per-row",
        serial_values=[False, True, True, True] * 3,
        native_values=[False] * 12,
        shape_passed=True,
        resident_capacity=4.0,
        execution_paths=["gguf_packed_ar_server_decode"],
    )
    assert not SCRIPT._hipengine_route_expectation_passes(
        concurrency=4,
        expectation="serial-c1-per-row",
        serial_values=[False] * 12,
        native_values=[False] * 12,
        shape_passed=True,
        resident_capacity=4.0,
        execution_paths=["gguf_packed_ar_server_decode"],
    )
    assert SCRIPT._hipengine_route_expectation_passes(
        concurrency=2,
        expectation="scheduler-c1",
        serial_values=[False] * 6,
        native_values=[False] * 6,
        shape_passed=False,
        resident_capacity=2.0,
        execution_paths=["laguna_resident_scheduler_c1"],
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


def test_effective_server_environment_records_revalidation_axes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compiler_version = tmp_path / "hipcc-version.txt"
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")
    monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", "0")
    monkeypatch.setenv("HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", "1,2,3")
    monkeypatch.setenv("HIPENGINE_EXECUTION_PROFILE", "strict")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "5")
    args = SCRIPT.build_parser().parse_args(
        [
            "--engine",
            "hipengine",
            "--gpu",
            "7",
            "--gpu-max-hw-queues",
            "2",
            "--compiler-version-file",
            str(compiler_version),
            "--hipengine-prefill-decode-policy",
            "fair",
            "--concurrencies",
            "17",
            "--json",
            str(tmp_path / "result.json"),
        ]
    )

    environment = SCRIPT._effective_server_environment(args, engine="hipengine")

    assert environment["HIPENGINE_HIP_ARCH"] == "gfx1151"
    assert environment["HIPENGINE_GGUF_FP16_RECURRENT_STATE"] == "0"
    assert environment["HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS"] == "1,2,3"
    assert environment["HIPENGINE_EXECUTION_PROFILE"] == "strict"
    assert environment["GPU_MAX_HW_QUEUES"] == "2"
    assert environment["HIPENGINE_COMPILER_VERSION_FILE"] == str(compiler_version)
    assert environment["HIP_VISIBLE_DEVICES"] == "7"
    assert environment["ROCR_VISIBLE_DEVICES"] is None
    assert environment["HIPENGINE_PREFILL_DECODE_POLICY"] == "fair"
    assert environment["HIPENGINE_GGUF_AR_PACKED_DECODE"] == "1"
    assert environment["HIPENGINE_GPU_MAX_HW_QUEUES_POLICY"] == "explicit"


def test_unset_hardware_queue_policy_reaches_child_without_rocm_limit(
    tmp_path: Path,
) -> None:
    args = SCRIPT.build_parser().parse_args(
        [
            "--engine",
            "hipengine",
            "--gpu-max-hw-queues",
            "unset",
            "--work-dir",
            str(tmp_path),
            "--json",
            str(tmp_path / "result.json"),
        ]
    )

    _command, environment, _cwd = SCRIPT._server_command_and_env(
        args,
        engine="hipengine",
        concurrency=8,
        port=19108,
    )

    assert args.gpu_max_hw_queues is None
    assert "GPU_MAX_HW_QUEUES" not in environment
    assert environment["HIPENGINE_GPU_MAX_HW_QUEUES_POLICY"] == "runtime_default"
    assert environment["HIPENGINE_PROCESS_ENV_REPORT_PATH"] == str(
        tmp_path / "hipengine-process-env-19108.json"
    )


def test_production_correctness_treats_c1_id_equality_as_diagnostic() -> None:
    prompt = [1, 2, 3]
    prompt_hash = SCRIPT.token_ids_sha256(prompt)
    summary = SCRIPT.correctness_summary(
        [
            {
                "request_index": 0,
                "prompt_token_ids_sha256": prompt_hash,
                "generated_token_ids": [7, 99],
            }
        ],
        oracle={prompt_hash: [7, 8]},
        expected_tokens=2,
        profile="production",
    )

    assert summary["passed"] is True
    assert summary["control_passed"] is True
    assert summary["generated_id_equality_binding"] is False
    assert summary["generated_id_equality_passed"] is False
    assert summary["generated_id_mismatch_count"] == 1
    assert summary["mismatch_count"] == 0

    broken_control = SCRIPT.correctness_summary(
        [
            {
                "request_index": 0,
                "prompt_token_ids_sha256": prompt_hash,
                "generated_token_ids": [7],
            }
        ],
        oracle={prompt_hash: [7, 8]},
        expected_tokens=2,
        profile="production",
    )
    assert broken_control["passed"] is False
    assert broken_control["control_passed"] is False
    assert broken_control["control_mismatch_count"] == 1


def test_production_stream_goodput_uses_profile_correctness_not_exact_ids() -> None:
    rows = [
        {
            "request_index": 0,
            "completion_tokens": 2,
            "client_ttft_seconds": 0.5,
            "client_inter_token_seconds": [0.1],
            "wall_seconds": 1.5,
            "stream_exact": False,
            "stream_correctness_passed": True,
            "stream_protocol_complete": True,
        },
        {
            "request_index": 1,
            "completion_tokens": 2,
            "client_ttft_seconds": 0.7,
            "client_inter_token_seconds": [0.2],
            "wall_seconds": 1.6,
            "stream_exact": False,
            "stream_correctness_passed": True,
            "stream_protocol_complete": True,
        },
    ]

    summary = SCRIPT._stream_batch_summary(
        rows,
        batch_wall_seconds=2.0,
        ttft_p95_limit=1.0,
        itl_p99_limit=0.25,
        e2e_p95_limit=2.0,
    )

    assert summary["passed"] is True
    assert summary["exact_generated_tok_s_aggregate"] == 0.0
    assert summary["correctness_qualified_tok_s_aggregate"] == pytest.approx(2.0)
    assert summary["slo_goodput_tok_s_aggregate"] == pytest.approx(2.0)


def test_repeat_determinism_is_schedule_local_and_binding() -> None:
    base = {
        "request_index": 0,
        "prompt_token_ids_sha256": "prompt-a",
        "generated_token_ids": [4, 5],
    }
    passed = SCRIPT.repeat_determinism_summary(
        [{"records": [dict(base)]} for _ in range(3)]
    )
    assert passed["passed"] is True
    assert passed["runs"] == 3

    changed = [{"records": [dict(base)]} for _ in range(3)]
    changed[2]["records"][0]["generated_token_ids"] = [4, 6]
    failed = SCRIPT.repeat_determinism_summary(changed)
    assert failed["passed"] is False
    assert failed["mismatch_count"] == 1


def test_production_correctness_runtime_paths_cover_fp16_gate_dependency() -> None:
    assert (
        "scripts/execution_profile_gguf_fp16_state_gate.py"
        in SCRIPT._CORRECTNESS_RUNTIME_PATHS
    )


def test_production_correctness_bundle_is_fail_closed_and_matching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model-fixture")
    numerical = tmp_path / "numerical.json"
    numerical.write_text('{"passed":true}\n', encoding="utf-8")
    fingerprint = SCRIPT.collect_model_identity(model)["fingerprint"]
    head = str(
        SCRIPT._capture(["git", "rev-parse", "HEAD"], cwd=SCRIPT.REPO_ROOT)[
            "stdout"
        ]
    ).strip()
    bundle = {
        "schema_version": 1,
        "kind": "hipengine_server_production_correctness_bundle",
        "status": "passed",
        "correctness_profile": "production",
        "runtime_scope": "scoped_legacy_default_candidate",
        "profile_qualification_claim": False,
        "source_commit": head,
        "host": {"physical_host": socket.gethostname()},
        "configuration": {
            "backend": "hip_gfx1151",
            "model_fingerprint": fingerprint,
            "quant": "gguf_q4_k_m",
            "kv_storage": "bf16",
            "candidate_environment": {
                "HIPENGINE_GGUF_FP16_RECURRENT_STATE": "1"
            },
        },
        "generated_id_equality_binding": False,
        "gates": {
            "numerical": {
                "passed": True,
                "summary": {
                    "kl_mean": 0.0001,
                    "kl_p95": 0.0003,
                    "kl_p99": 0.001,
                    "kl_max": 0.01,
                    "top1_agreement": 0.999,
                },
                "scope_failures": [],
                "requires_outlier_review": False,
            },
            "repeat_determinism": {"passed": True, "runs": 3},
            "isolation": {"passed": True},
            "control_ownership": {"passed": True},
            "lifecycle": {"passed": True},
            "bf16_relative": {"passed": True},
            "task_quality": {"passed": True},
            "strict_fallback": {"passed": True, "registered": True},
        },
        "source_artifacts": [
            {
                "path": str(numerical),
                "sha256": SCRIPT.file_sha256(numerical),
            }
        ],
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", "1")
    args = SCRIPT.build_parser().parse_args(
        [
            "--engine",
            "hipengine",
            "--model",
            str(model),
            "--correctness-profile",
            "production",
            "--production-correctness-artifact",
            str(bundle_path),
            "--json",
            str(tmp_path / "result.json"),
        ]
    )

    contract = SCRIPT._resolve_correctness_contract(args, engine="hipengine")

    assert contract["profile"] == "production"
    assert contract["generated_id_equality_binding"] is False
    assert contract["bundle_sha256"] == SCRIPT.file_sha256(bundle_path)
    assert contract["public_profile_qualification_claim"] is False

    monkeypatch.delenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE")
    with pytest.raises(ValueError, match="explicit HIPENGINE_GGUF_FP16_RECURRENT_STATE=1"):
        SCRIPT._resolve_correctness_contract(args, engine="hipengine")
    monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", "1")

    bundle["gates"]["task_quality"]["passed"] = False
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="task_quality"):
        SCRIPT._resolve_correctness_contract(args, engine="hipengine")
