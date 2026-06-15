from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.gguf_mtp_oracle_gate import DEFAULT_FIXTURE, run_oracle_gate


def test_gguf_mtp_oracle_gate_passes_committed_fixture() -> None:
    result = run_oracle_gate(DEFAULT_FIXTURE)

    assert result["kind"] == "gguf_mtp_oracle_gate"
    assert result["passed"] is True
    assert result["cpu_reference_kernel"] == [
        "cpu_reference",
        "mtp_nextn_layer",
        "w4_gguf",
        "qwen35_dense_logits",
    ]
    assert result["draft_topk_kernel"] == [
        "cpu_reference",
        "mtp_draft_topk",
        "w4_gguf",
        "full_vocab_d2h",
    ]
    assert result["metrics"] == {
        "max_kl": 0.0,
        "mean_kl": 0.0,
        "top1_agreement": 1.0,
        "top_k_match": True,
        "rows": 1,
        "vocab_size": 4,
    }
    assert result["actual_top1_token_ids"] == [2]
    assert result["expected_top1_token_ids"] == [2]
    assert result["actual_top_k_token_ids"] == [[2, 0, 3]]
    assert result["actual_top_k_logits"] == [[1.8975492715835571, 0.630980908870697, -1.3442893028259277]]
    assert result["draft_execution_plan"]["proposed_token_ids"] == [2]
    assert result["draft_execution_plan"]["proposal"]["top_k_token_ids"] == [[2, 0, 3]]
    assert result["draft_execution_plan"]["proposal"]["topk_kernel"] == [
        "cpu_reference",
        "mtp_draft_topk",
        "w4_gguf",
        "full_vocab_d2h",
    ]
    assert result["draft_execution_plan"]["kv_live_spans"] == {
        "spans_mode": "uniform",
        "storage_dtype": "bf16",
        "rows": 1,
        "block_size": 256,
        "logical_blocks": 1,
        "base_offsets": [[0]],
        "append_live_counts": [1],
        "decode_live_counts": [2],
        "token_positions": [1],
        "evict_mask": None,
    }
    assert result["draft_execution_plan"]["cpu_reference_kwargs"]["decode"] == {
        "kv_base_offsets": [[0]],
        "kv_live_counts": [2],
        "kv_token_positions": [1],
        "kv_evict_mask": None,
        "block_size": 256,
    }


def test_gguf_mtp_oracle_gate_fails_tampered_expected_logits(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_FIXTURE.read_text())
    payload["expected"]["logits"] = [[10.0, -10.0, -10.0, -10.0]]
    payload["expected"]["top_k_token_ids"] = [0, 1, 2]
    fixture = tmp_path / "tampered.json"
    fixture.write_text(json.dumps(payload))

    result = run_oracle_gate(fixture)

    assert result["passed"] is False
    assert result["metrics"]["top1_agreement"] == 0.0
    assert result["actual_top1_token_ids"] == [2]
    assert result["expected_top1_token_ids"] == [0]
    assert result["metrics"]["max_kl"] > 0.05


def test_gguf_mtp_oracle_gate_cli_writes_artifact_and_fails_on_fail(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_FIXTURE.read_text())
    payload["expected"]["logits"] = [[10.0, -10.0, -10.0, -10.0]]
    fixture = tmp_path / "tampered.json"
    out = tmp_path / "gate.json"
    fixture.write_text(json.dumps(payload))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/gguf_mtp_oracle_gate.py",
            "--fixture",
            str(fixture),
            "--out",
            str(out),
            "--fail-on-fail",
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 1
    artifact = json.loads(out.read_text())
    assert artifact["passed"] is False
    assert artifact["metrics"]["top1_agreement"] == 0.0
