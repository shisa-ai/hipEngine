from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.stepfun_kv_trace_check import build_trace_check_report, main

PROMPT_KV = "hipengine_qwen35_write_paged_kv_mixed_value_bf16_prompt_spans"
DECODE_KV = "hipengine_qwen35_write_paged_kv_mixed_value_bf16_spans"
ATTN_CONTEXT = "hipengine_qwen35_paged_full_attn_decode_split_k_context_bf16_spans"
ATTN_REDUCE = "hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_f32"


def _kernel_names(layer_count: int) -> list[str]:
    names: list[str] = []
    for layer in range(layer_count):
        names.extend(
            [
                f"{PROMPT_KV} [layer={layer}]",
                f"{DECODE_KV} [layer={layer}]",
                f"{ATTN_CONTEXT} [layer={layer}]",
                f"{ATTN_REDUCE} [layer={layer}]",
            ]
        )
    return names


def test_stepfun_kv_trace_check_accepts_json_kernel_names(tmp_path: Path) -> None:
    trace = tmp_path / "kv-trace.json"
    trace.write_text(json.dumps({"kernel_names": _kernel_names(2)}))

    report = build_trace_check_report(trace, expected_layer_count=2)

    assert report["status"] == "passed"
    summary = report["trace_summary"]
    assert summary["ready"] is True
    assert summary["kernel_record_count"] == 8
    assert summary["expected_layer_count"] == 2
    assert summary["expected_min_kernel_launch_count"] == 8
    assert summary["missing_families"] == []
    assert summary["no_claim_policy"] == {
        "kv_kernel_trace_artifact_claim_allowed": True,
        "kv_backed_decode_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "KV trace validation only checks required kernel-family presence; "
            "token/logit correctness and benchmark gates remain separate."
        ),
    }
    assert [record["name"] for record in report["required_kernel_families"]] == [
        "prompt_kv_write",
        "decode_kv_write",
        "decode_attention_context",
        "decode_attention_gate_reduce",
    ]
    assert all(record["observed_count"] == 2 for record in report["required_kernel_families"])
    assert all(record["ready"] is True for record in report["required_kernel_families"])
    assert report["readiness_impact"] == {
        "kv_kernel_trace_artifact": True,
        "kv_backed_decode_ready": False,
        "e2e_inference_ready": False,
        "reason": (
            "This report can satisfy only the retained KV kernel trace artifact; "
            "the streaming decode loop and KV-backed next-token artifact must also pass."
        ),
    }
    assert len(report["report_sha256"]) == 64


def test_stepfun_kv_trace_check_reports_missing_csv_families(tmp_path: Path) -> None:
    trace = tmp_path / "kv-trace.csv"
    with trace.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Index", "Kernel_Name", "DurationNs"])
        writer.writeheader()
        writer.writerow({"Index": 0, "Kernel_Name": PROMPT_KV, "DurationNs": 100})
        writer.writerow({"Index": 1, "Kernel_Name": DECODE_KV, "DurationNs": 100})
        writer.writerow({"Index": 2, "Kernel_Name": ATTN_CONTEXT, "DurationNs": 100})

    report = build_trace_check_report(trace, expected_layer_count=2)

    assert report["status"] == "failed"
    summary = report["trace_summary"]
    assert summary["ready"] is False
    assert summary["trace_format"] == "csv"
    assert summary["missing_families"] == [
        "prompt_kv_write",
        "decode_kv_write",
        "decode_attention_context",
        "decode_attention_gate_reduce",
    ]
    records = {record["name"]: record for record in report["required_kernel_families"]}
    assert records["prompt_kv_write"]["observed_count"] == 1
    assert records["decode_attention_gate_reduce"]["observed_count"] == 0
    assert records["decode_attention_gate_reduce"]["ready"] is False
    assert report["readiness_impact"]["kv_kernel_trace_artifact"] is False


def test_stepfun_kv_trace_check_cli_compact_modes(tmp_path: Path) -> None:
    trace = tmp_path / "kv-trace.json"
    trace.write_text(json.dumps({"kernel_names": _kernel_names(1)}))
    summary_output = tmp_path / "summary.json"
    sha_output = tmp_path / "sha.json"
    status_output = tmp_path / "status.json"

    rc = main(
        [
            "--trace",
            str(trace),
            "--expected-layer-count",
            "1",
            "--summary-only",
            "--output",
            str(summary_output),
            "--pretty",
        ]
    )
    assert rc == 0
    summary = json.loads(summary_output.read_text())
    assert summary["status"] == "passed"
    assert summary["required_family_names"] == [
        "prompt_kv_write",
        "decode_kv_write",
        "decode_attention_context",
        "decode_attention_gate_reduce",
    ]

    rc = main(
        [
            "--trace",
            str(trace),
            "--expected-layer-count",
            "1",
            "--summary-only",
            "--sha-only",
            "--output",
            str(sha_output),
            "--pretty",
        ]
    )
    assert rc == 0
    assert len(json.loads(sha_output.read_text())) == 64

    trace.write_text(json.dumps({"kernel_names": [PROMPT_KV]}))
    rc = main(
        [
            "--trace",
            str(trace),
            "--expected-layer-count",
            "1",
            "--status-only",
            "--fail-on-missing",
            "--output",
            str(status_output),
            "--pretty",
        ]
    )
    assert rc == 2
    assert json.loads(status_output.read_text()) == "failed"
