from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.dispatch import (
    BatchSamplerMode,
    ProjectionDispatchCandidate,
    ProjectionDispatchEvidence,
    ProjectionKernelSelection,
    WorkKind,
    plan_batch_sampler_dispatch,
    plan_projection_dispatch,
)
from hipengine.generation import (
    CompactPromptSlab,
    EngineLoopConfig,
    GeneratedToken,
    GenerationRequest,
    GraphBucketCache,
    PerRowSamplingParams,
    ResidentBatchScheduler,
    ResidentEngineLoop,
    SubmitPollTextGenerator,
    SpeculativeCommitPlan,
    SpeculativeStateCommitPlan,
    SpeculativeVerifyBufferPlan,
    SpeculativeVerifyPlan,
    SpeculativeVerifyWork,
    add_engine_loop_config_args,
    engine_loop_config_from_args,
    engine_loop_config_from_env,
)
from hipengine.kvcache import ChunkedKVPool, FixedPagedKVPolicy
from hipengine.speculative import AcceptResult, DraftBatch, TargetAcceptSummary, TargetStateCommitBuffers, TargetVerifyBuffers
from scripts import qwen35_batch_c_sweep as c_sweep
from scripts import qwen35_batch_retained_bench as retained_bench
from scripts.qwen35_batch_artifact_schema import validate_cn_diagnostic_artifact_payload
from scripts.qwen35_batch_c_sweep import build_parser as build_c_sweep_parser, build_sweep_commands, run_sweep
from scripts.qwen35_batch_gguf_diagnostic import build_parser as build_gguf_diagnostic_parser, run as run_gguf_diagnostic
from scripts.qwen35_batch_hidden_bisect import (
    _first_hidden_mismatch,
    _parse_layer_limits,
    build_parser as build_hidden_bisect_parser,
    hidden_comparison,
    run as run_hidden_bisect,
)
from scripts.qwen35_batch_int8_diagnostic import build_parser as build_int8_diagnostic_parser, run as run_int8_diagnostic
from scripts.qwen35_batch_serial_bench import _load_prompt_slices, _summarize_samples


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


class _FakeSerialBridgeRunner:
    def __init__(self) -> None:
        self.prefills = []
        self.decodes = []
        self._counts: dict[int, int] = {}

    def prefill(self, work) -> None:
        self.prefills.append(work)

    def decode(self, work) -> tuple[GeneratedToken, ...]:
        self.decodes.append(work)
        tokens: list[GeneratedToken] = []
        for request_id in work.request_ids:
            count = self._counts.get(request_id, 0)
            self._counts[request_id] = count + 1
            tokens.append(GeneratedToken(request_id, 1000 + request_id * 10 + count))
        return tuple(tokens)


class _FakeTextGenerator:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> list[str]:
        self.requests.append(request)
        seeds = request.row_seeds or tuple(-1 for _ in request.prompts)
        return [f"generated:{prompt}:{seed}" for prompt, seed in zip(request.prompts, seeds, strict=True)]


def test_batch_c_sweep_dry_run_records_commands_and_artifacts(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    args = build_c_sweep_parser().parse_args(
        [
            "--dry-run",
            "--batch-sizes",
            "1,2",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--summary-json",
            str(summary_path),
            "--model",
            "/tmp/model",
            "--fixture",
            "/tmp/fixture.json",
            "--prompt-length",
            "16",
            "--decode-tokens",
            "2",
            "--warmup-decode-tokens",
            "1",
            "--max-layers",
            "3",
        ]
    )

    planned = build_sweep_commands(args)
    assert [(item.category, item.batch_size) for item in planned] == [
        ("primitive", 1),
        ("serial_bridge", 1),
        ("native_diagnostic", 1),
        ("primitive", 2),
        ("serial_bridge", 2),
        ("native_diagnostic", 2),
    ]

    summary = run_sweep(args)

    assert summary["status"] == "planned"
    assert summary["dry_run"] is True
    assert summary["options"] == {
        "stop_on_failure": True,
        "include_int8": False,
        "require_cached_build": False,
        "compiler_version_file": None,
    }
    assert summary_path.exists()
    persisted = json.loads(summary_path.read_text())
    assert persisted["options"] == summary["options"]
    assert persisted["command_count"] == 6
    assert persisted["completed_command_count"] == 6
    assert len(persisted["commands"]) == 6
    assert persisted["status_counts"] == {"planned": 6}
    assert persisted["category_status_counts"] == {
        "primitive": {"planned": 2},
        "serial_bridge": {"planned": 2},
        "native_diagnostic": {"planned": 2},
    }
    assert persisted["retained_precondition_counts"] == {}
    assert persisted["skipped_preconditions"] == []
    assert all(entry["status"] == "planned" for entry in persisted["commands"])
    assert all(entry["command"] for entry in persisted["commands"])
    assert all(entry["artifact_path"] for entry in persisted["commands"])
    assert all("git_dirty" in entry for entry in persisted["commands"])
    assert any("qwen35_batch_retained_bench.py" in entry["command"] for entry in persisted["commands"])
    assert any("qwen35_paro_bench.py" in entry["command"] for entry in persisted["commands"])
    retained_c2 = next(item for item in planned if item.category == "native_diagnostic" and item.batch_size == 2)
    assert "--c1-baseline-json" in retained_c2.argv
    assert str(tmp_path / "artifacts" / "native-baseline-c1.json") in retained_c2.argv
    assert "--serial-bridge-json" in retained_c2.argv
    assert str(tmp_path / "artifacts" / "serial-bridge-c2.json") in retained_c2.argv
    assert "--primitive-correctness-json" in retained_c2.argv
    assert str(tmp_path / "artifacts" / "primitive-c2.json") in retained_c2.argv


def test_batch_c_sweep_stops_and_counts_failed_command(tmp_path: Path, monkeypatch) -> None:
    args = build_c_sweep_parser().parse_args(
        [
            "--batch-sizes",
            "2",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--model",
            "/tmp/model",
            "--fixture",
            "/tmp/fixture.json",
            "--prompt-length",
            "16",
            "--decode-tokens",
            "2",
            "--warmup-decode-tokens",
            "1",
            "--max-layers",
            "3",
        ]
    )
    monkeypatch.setattr(c_sweep, "_git_state", lambda: {"commit": "test", "dirty": False, "status_short": []})

    class FakeProc:
        returncode = 42
        stdout = "primitive failed\n"

    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return FakeProc()

    monkeypatch.setattr(c_sweep.subprocess, "run", fake_run)

    summary = run_sweep(args)

    assert summary["status"] == "failed"
    assert summary["options"]["stop_on_failure"] is True
    assert summary["command_count"] == 3
    assert summary["completed_command_count"] == 1
    assert summary["status_counts"] == {"failed": 1}
    assert summary["category_status_counts"] == {"primitive": {"failed": 1}}
    assert summary["retained_precondition_counts"] == {}
    assert summary["skipped_preconditions"] == []
    assert len(summary["commands"]) == 1
    failed = summary["commands"][0]
    assert failed["category"] == "primitive"
    assert failed["returncode"] == 42
    assert failed["output_tail"] == "primitive failed\n"
    assert len(calls) == 1
    assert calls[0][1] == "scripts/qwen35_batch_correctness.py"


def test_batch_c_sweep_no_stop_counts_failed_and_skipped_rows(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "artifacts"
    args = build_c_sweep_parser().parse_args(
        [
            "--batch-sizes",
            "2",
            "--no-stop-on-failure",
            "--output-dir",
            str(output_dir),
            "--model",
            "/tmp/model",
            "--fixture",
            "/tmp/fixture.json",
            "--prompt-length",
            "16",
            "--decode-tokens",
            "2",
            "--warmup-decode-tokens",
            "1",
            "--max-layers",
            "3",
        ]
    )
    monkeypatch.setattr(c_sweep, "_git_state", lambda: {"commit": "test", "dirty": False, "status_short": []})

    class FakeProc:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout

    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if len(calls) == 1:
            return FakeProc(42, "primitive failed\n")
        return FakeProc(0, "serial passed\n")

    monkeypatch.setattr(c_sweep.subprocess, "run", fake_run)

    summary = run_sweep(args)

    assert summary["status"] == "failed"
    assert summary["options"]["stop_on_failure"] is False
    assert summary["command_count"] == 3
    assert summary["completed_command_count"] == 3
    assert summary["status_counts"] == {"failed": 1, "passed": 1, "skipped": 1}
    assert summary["category_status_counts"] == {
        "primitive": {"failed": 1},
        "serial_bridge": {"passed": 1},
        "native_diagnostic": {"skipped": 1},
    }
    assert summary["retained_precondition_counts"] == {
        "primitive_correctness": {"failed": 1},
        "c1_baseline": {"failed": 1},
        "serial_bridge": {"failed": 1},
    }
    assert [entry["status"] for entry in summary["commands"]] == ["failed", "passed", "skipped"]
    assert summary["skipped_preconditions"] == [
        {
            "category": "native_diagnostic",
            "batch_size": 2,
            "artifact_path": str(output_dir / "native-diagnostic-c2.json"),
            "kind": "primitive_correctness",
            "precondition_artifact_path": str(output_dir / "primitive-c2.json"),
            "reason": "primitive correctness artifact does not exist",
        }
    ]
    assert len(calls) == 2
    assert calls[0][1] == "scripts/qwen35_batch_correctness.py"
    assert calls[1][1] == "scripts/qwen35_batch_serial_bench.py"


def test_batch_c_sweep_skips_retained_when_primitive_artifact_missing(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "artifacts"
    summary_path = tmp_path / "summary.json"
    args = build_c_sweep_parser().parse_args(
        [
            "--batch-sizes",
            "2",
            "--output-dir",
            str(output_dir),
            "--summary-json",
            str(summary_path),
            "--model",
            "/tmp/model",
            "--fixture",
            "/tmp/fixture.json",
            "--prompt-length",
            "16",
            "--decode-tokens",
            "2",
            "--warmup-decode-tokens",
            "1",
            "--max-layers",
            "3",
        ]
    )
    monkeypatch.setattr(c_sweep, "_git_state", lambda: {"commit": "test", "dirty": False, "status_short": []})

    class FakeProc:
        returncode = 0
        stdout = "ok"

    monkeypatch.setattr(c_sweep.subprocess, "run", lambda *args, **kwargs: FakeProc())

    summary = run_sweep(args)

    assert summary["status"] == "blocked"
    assert summary["status_counts"] == {"passed": 2, "skipped": 1}
    assert summary["category_status_counts"] == {
        "primitive": {"passed": 1},
        "serial_bridge": {"passed": 1},
        "native_diagnostic": {"skipped": 1},
    }
    assert summary["retained_precondition_counts"] == {
        "primitive_correctness": {"failed": 1},
        "c1_baseline": {"failed": 1},
        "serial_bridge": {"failed": 1},
    }
    assert [entry["status"] for entry in summary["commands"]] == ["passed", "passed", "skipped"]
    skipped = summary["commands"][-1]
    assert skipped["category"] == "native_diagnostic"
    assert [item["kind"] for item in skipped["preconditions"]] == [
        "primitive_correctness",
        "c1_baseline",
        "serial_bridge",
    ]
    assert skipped["preconditions"][0]["passed"] is False
    assert skipped["precondition"] == skipped["preconditions"][0]
    assert skipped["precondition"]["kind"] == "primitive_correctness"
    assert skipped["precondition"]["passed"] is False
    assert skipped["precondition"]["reason"] == "primitive correctness artifact does not exist"
    assert summary["skipped_preconditions"] == [
        {
            "category": "native_diagnostic",
            "batch_size": 2,
            "artifact_path": str(output_dir / "native-diagnostic-c2.json"),
            "kind": "primitive_correctness",
            "precondition_artifact_path": str(output_dir / "primitive-c2.json"),
            "reason": "primitive correctness artifact does not exist",
        }
    ]
    persisted = json.loads(summary_path.read_text())
    persisted_skipped = persisted["commands"][-1]
    assert persisted_skipped["status"] == "skipped"
    assert persisted_skipped["preconditions"] == skipped["preconditions"]
    assert persisted_skipped["precondition"] == skipped["precondition"]
    assert persisted["retained_precondition_counts"] == summary["retained_precondition_counts"]
    assert persisted["skipped_preconditions"] == summary["skipped_preconditions"]


@pytest.mark.parametrize(
    ("missing_artifact", "expected_kind"),
    [("c1", "c1_baseline"), ("serial", "serial_bridge")],
)
def test_batch_c_sweep_skips_retained_when_scaling_reference_missing(
    tmp_path: Path,
    monkeypatch,
    missing_artifact: str,
    expected_kind: str,
) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / "primitive-c2.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "rows": 2,
                "passed": True,
                "append_key_mismatch": 0,
                "append_value_mismatch": 0,
                "attn_batch_vs_c1_max_abs": 0.0,
            }
        )
    )
    if missing_artifact != "c1":
        (output_dir / "native-baseline-c1.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "prompt_length": 16,
                    "decode_tokens": 2,
                    "throughput": {"warmed_decode_tok_s": 10.0},
                }
            )
        )
    if missing_artifact != "serial":
        (output_dir / "serial-bridge-c2.json").write_text(
            json.dumps(
                {
                    "schema": 2,
                    "status": "blocked",
                    "workload": {"concurrency": 2, "prompt_tokens_per_request": 16, "gen_tokens_per_request": 2},
                    "measurements": {"decode_tok_s_aggregate": 20.0, "decode_tok_s_per_request": 10.0},
                }
            )
        )
    args = build_c_sweep_parser().parse_args(
        [
            "--batch-sizes",
            "2",
            "--output-dir",
            str(output_dir),
            "--model",
            "/tmp/model",
            "--fixture",
            "/tmp/fixture.json",
            "--prompt-length",
            "16",
            "--decode-tokens",
            "2",
            "--warmup-decode-tokens",
            "1",
            "--max-layers",
            "3",
        ]
    )
    monkeypatch.setattr(c_sweep, "_git_state", lambda: {"commit": "test", "dirty": False, "status_short": []})

    class FakeProc:
        returncode = 0
        stdout = "ok"

    monkeypatch.setattr(c_sweep.subprocess, "run", lambda *args, **kwargs: FakeProc())

    summary = run_sweep(args)

    assert summary["status"] == "blocked"
    assert summary["status_counts"] == {"passed": 2, "skipped": 1}
    assert summary["category_status_counts"] == {
        "primitive": {"passed": 1},
        "serial_bridge": {"passed": 1},
        "native_diagnostic": {"skipped": 1},
    }
    expected_counts = {
        "primitive_correctness": {"passed": 1},
        "c1_baseline": {"failed" if missing_artifact == "c1" else "passed": 1},
        "serial_bridge": {"failed" if missing_artifact == "serial" else "passed": 1},
    }
    assert summary["retained_precondition_counts"] == expected_counts
    assert [entry["status"] for entry in summary["commands"]] == ["passed", "passed", "skipped"]
    skipped = summary["commands"][-1]
    assert [item["kind"] for item in skipped["preconditions"]] == [
        "primitive_correctness",
        "c1_baseline",
        "serial_bridge",
    ]
    assert skipped["preconditions"][0]["passed"] is True
    assert skipped["precondition"]["kind"] == expected_kind
    assert skipped["precondition"]["passed"] is False
    assert skipped["precondition"]["reason"] == "scaling reference artifact does not exist"
    assert summary["skipped_preconditions"] == [
        {
            "category": "native_diagnostic",
            "batch_size": 2,
            "artifact_path": str(output_dir / "native-diagnostic-c2.json"),
            "kind": expected_kind,
            "precondition_artifact_path": str(
                output_dir / ("native-baseline-c1.json" if missing_artifact == "c1" else "serial-bridge-c2.json")
            ),
            "reason": "scaling reference artifact does not exist",
        }
    ]


def test_batch_c_sweep_skips_retained_when_scaling_reference_shape_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / "primitive-c2.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "rows": 2,
                "passed": True,
                "append_key_mismatch": 0,
                "append_value_mismatch": 0,
                "attn_batch_vs_c1_max_abs": 0.0,
            }
        )
    )
    (output_dir / "native-baseline-c1.json").write_text(
        json.dumps({"schema": 1, "throughput": {"warmed_decode_tok_s": 10.0}})
    )
    (output_dir / "serial-bridge-c2.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "status": "blocked",
                "workload": {"concurrency": 2, "prompt_tokens_per_request": 16, "gen_tokens_per_request": 2},
                "measurements": {"decode_tok_s_aggregate": 20.0, "decode_tok_s_per_request": 10.0},
            }
        )
    )
    args = build_c_sweep_parser().parse_args(
        [
            "--batch-sizes",
            "2",
            "--output-dir",
            str(output_dir),
            "--model",
            "/tmp/model",
            "--fixture",
            "/tmp/fixture.json",
            "--prompt-length",
            "16",
            "--decode-tokens",
            "2",
            "--warmup-decode-tokens",
            "1",
            "--max-layers",
            "3",
        ]
    )
    monkeypatch.setattr(c_sweep, "_git_state", lambda: {"commit": "test", "dirty": False, "status_short": []})

    class FakeProc:
        returncode = 0
        stdout = "ok"

    monkeypatch.setattr(c_sweep.subprocess, "run", lambda *args, **kwargs: FakeProc())

    summary = run_sweep(args)

    assert summary["status"] == "blocked"
    assert summary["status_counts"] == {"passed": 2, "skipped": 1}
    skipped = summary["commands"][-1]
    assert skipped["status"] == "skipped"
    assert skipped["precondition"]["kind"] == "c1_baseline"
    assert skipped["precondition"]["reason"] == "prompt token count label is missing; decode token count label is missing"
    assert summary["retained_precondition_counts"] == {
        "primitive_correctness": {"passed": 1},
        "c1_baseline": {"failed": 1},
        "serial_bridge": {"passed": 1},
    }


def test_batch_c_sweep_skips_retained_when_scaling_reference_reason_is_non_null(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / "primitive-c2.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "rows": 2,
                "passed": True,
                "append_key_mismatch": 0,
                "append_value_mismatch": 0,
                "attn_batch_vs_c1_max_abs": 0.0,
            }
        )
    )
    (output_dir / "native-baseline-c1.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "prompt_length": 16,
                "decode_tokens": 2,
                "throughput": {"warmed_decode_tok_s": 10.0},
            }
        )
    )
    (output_dir / "serial-bridge-c2.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "status": "blocked",
                "reason": "decode throughput fields missing",
                "workload": {"concurrency": 2, "prompt_tokens_per_request": 16, "gen_tokens_per_request": 2},
                "measurements": {"decode_tok_s_aggregate": 20.0, "decode_tok_s_per_request": 10.0},
            }
        )
    )
    args = build_c_sweep_parser().parse_args(
        [
            "--batch-sizes",
            "2",
            "--output-dir",
            str(output_dir),
            "--model",
            "/tmp/model",
            "--fixture",
            "/tmp/fixture.json",
            "--prompt-length",
            "16",
            "--decode-tokens",
            "2",
            "--warmup-decode-tokens",
            "1",
            "--max-layers",
            "3",
        ]
    )
    monkeypatch.setattr(c_sweep, "_git_state", lambda: {"commit": "test", "dirty": False, "status_short": []})

    class FakeProc:
        returncode = 0
        stdout = "ok"

    monkeypatch.setattr(c_sweep.subprocess, "run", lambda *args, **kwargs: FakeProc())

    summary = run_sweep(args)

    assert summary["status"] == "blocked"
    assert summary["status_counts"] == {"passed": 2, "skipped": 1}
    skipped = summary["commands"][-1]
    assert skipped["status"] == "skipped"
    assert skipped["precondition"]["kind"] == "serial_bridge"
    assert skipped["precondition"]["reason"] == "scaling reference reason is non-null: decode throughput fields missing"
    assert summary["retained_precondition_counts"] == {
        "primitive_correctness": {"passed": 1},
        "c1_baseline": {"passed": 1},
        "serial_bridge": {"failed": 1},
    }


def test_batch_c_sweep_runs_retained_when_all_references_are_usable(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / "primitive-c2.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "rows": 2,
                "passed": True,
                "append_key_mismatch": 0,
                "append_value_mismatch": 0,
                "attn_batch_vs_c1_max_abs": 0.0,
            }
        )
    )
    (output_dir / "native-baseline-c1.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "prompt_length": 16,
                "decode_tokens": 2,
                "throughput": {"warmed_decode_tok_s": 10.0},
            }
        )
    )
    (output_dir / "serial-bridge-c2.json").write_text(
        json.dumps(
            {
                "schema": 2,
                "status": "blocked",
                "workload": {"concurrency": 2, "prompt_tokens_per_request": 16, "gen_tokens_per_request": 2},
                "measurements": {"decode_tok_s_aggregate": 20.0, "decode_tok_s_per_request": 10.0},
            }
        )
    )
    args = build_c_sweep_parser().parse_args(
        [
            "--batch-sizes",
            "2",
            "--output-dir",
            str(output_dir),
            "--model",
            "/tmp/model",
            "--fixture",
            "/tmp/fixture.json",
            "--prompt-length",
            "16",
            "--decode-tokens",
            "2",
            "--warmup-decode-tokens",
            "1",
            "--max-layers",
            "3",
        ]
    )
    monkeypatch.setattr(c_sweep, "_git_state", lambda: {"commit": "test", "dirty": False, "status_short": []})
    calls: list[list[str]] = []

    class FakeProc:
        returncode = 0
        stdout = "ok"

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return FakeProc()

    monkeypatch.setattr(c_sweep.subprocess, "run", fake_run)

    summary = run_sweep(args)

    assert summary["status"] == "passed"
    assert summary["command_count"] == 3
    assert summary["completed_command_count"] == 3
    assert summary["status_counts"] == {"passed": 3}
    assert summary["category_status_counts"] == {
        "primitive": {"passed": 1},
        "serial_bridge": {"passed": 1},
        "native_diagnostic": {"passed": 1},
    }
    assert summary["retained_precondition_counts"] == {
        "primitive_correctness": {"passed": 1},
        "c1_baseline": {"passed": 1},
        "serial_bridge": {"passed": 1},
    }
    assert summary["skipped_preconditions"] == []
    assert [entry["status"] for entry in summary["commands"]] == ["passed", "passed", "passed"]
    assert len(calls) == 3
    assert calls[-1][1] == "scripts/qwen35_batch_retained_bench.py"
    native = summary["commands"][-1]
    assert native["category"] == "native_diagnostic"
    assert [item["kind"] for item in native["preconditions"]] == [
        "primitive_correctness",
        "c1_baseline",
        "serial_bridge",
    ]
    assert all(item["passed"] is True for item in native["preconditions"])
    preconditions_by_kind = {item["kind"]: item for item in native["preconditions"]}
    assert preconditions_by_kind["c1_baseline"] == {
        "kind": "c1_baseline",
        "artifact_path": str(output_dir / "native-baseline-c1.json"),
        "passed": True,
        "reason": None,
        "reference_status": "loaded",
        "reference_reason": None,
        "workload_concurrency": 1,
        "prompt_tokens_per_request": 16,
        "gen_tokens_per_request": 2,
        "decode_tok_s_aggregate": 10.0,
        "decode_tok_s_per_request": 10.0,
    }
    assert preconditions_by_kind["serial_bridge"] == {
        "kind": "serial_bridge",
        "artifact_path": str(output_dir / "serial-bridge-c2.json"),
        "passed": True,
        "reason": None,
        "reference_status": "blocked",
        "reference_reason": None,
        "workload_concurrency": 2,
        "prompt_tokens_per_request": 16,
        "gen_tokens_per_request": 2,
        "decode_tok_s_aggregate": 20.0,
        "decode_tok_s_per_request": 10.0,
    }
    assert "precondition" not in native


def test_batch_c_sweep_can_plan_int8_blocked_diagnostics(tmp_path: Path) -> None:
    args = build_c_sweep_parser().parse_args(
        [
            "--dry-run",
            "--include-int8",
            "--batch-sizes",
            "1,2",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--model",
            "/tmp/model",
            "--fixture",
            "/tmp/fixture.json",
        ]
    )

    planned = build_sweep_commands(args)

    assert [(item.category, item.batch_size) for item in planned].count(("int8_native_diagnostic", 2)) == 1
    assert ("int8_native_diagnostic", 1) not in [(item.category, item.batch_size) for item in planned]
    int8 = next(item for item in planned if item.category == "int8_native_diagnostic")
    assert "scripts/qwen35_batch_int8_diagnostic.py" in int8.command
    assert "--rows 2" in int8.command
    assert int8.artifact_path.name == "int8-native-diagnostic-c2.json"

    summary = run_sweep(args)

    assert summary["status"] == "planned"
    assert summary["options"]["include_int8"] is True
    assert summary["command_count"] == 7
    assert summary["completed_command_count"] == 7
    assert summary["status_counts"] == {"planned": 7}
    assert summary["category_status_counts"] == {
        "primitive": {"planned": 2},
        "serial_bridge": {"planned": 2},
        "native_diagnostic": {"planned": 2},
        "int8_native_diagnostic": {"planned": 1},
    }
    assert summary["retained_precondition_counts"] == {}
    assert summary["skipped_preconditions"] == []


def test_projection_dispatch_keeps_c1_on_row_gemv_even_with_fast_candidate() -> None:
    row_gemv = ProjectionKernelSelection("linear", "w4_paro", "row_gemv")
    wmma = ProjectionDispatchCandidate(
        "wmma_caware",
        ProjectionKernelSelection("linear", "w4_paro", "wmma_caware"),
        min_rows=1,
        evidence=ProjectionDispatchEvidence(
            "benchmarks/results/projection-wmma-c1.json",
            aggregate_vs_row_gemv=3.0,
            per_request_vs_row_gemv=3.0,
        ),
    )

    decision = plan_projection_dispatch(rows=1, row_gemv=row_gemv, candidates=[wmma])

    assert decision.selection == row_gemv
    assert decision.selected_candidate == "row_gemv"
    assert decision.path == "row_gemv_c1"
    assert decision.throughput_claim_eligible is False
    assert decision.blockers == ()


def test_projection_dispatch_requires_accepted_cN_speedup_evidence() -> None:
    row_gemv = ProjectionKernelSelection("linear", "w4_paro", "row_gemv")
    missing = ProjectionDispatchCandidate(
        "mmq_missing",
        ProjectionKernelSelection("linear", "w4_paro", "mmq_caware"),
        min_rows=2,
    )
    rejected = ProjectionDispatchCandidate(
        "wmma_rejected",
        ProjectionKernelSelection("linear", "w4_paro", "wmma_caware"),
        min_rows=2,
        evidence=ProjectionDispatchEvidence(
            "benchmarks/results/rejected.json",
            aggregate_vs_row_gemv=1.5,
            per_request_vs_row_gemv=1.5,
            accepted=False,
        ),
    )
    too_slow = ProjectionDispatchCandidate(
        "gemm_too_slow",
        ProjectionKernelSelection("linear", "w4_paro", "gemm_caware"),
        min_rows=2,
        evidence=ProjectionDispatchEvidence(
            "benchmarks/results/slow.json",
            aggregate_vs_row_gemv=1.2,
            per_request_vs_row_gemv=0.99,
        ),
    )

    decision = plan_projection_dispatch(rows=4, row_gemv=row_gemv, candidates=[missing, rejected, too_slow])

    assert decision.selection == row_gemv
    assert decision.path == "row_gemv_until_caware_benchmark"
    assert decision.throughput_claim_eligible is False
    assert "mmq_missing: missing benchmark evidence" in decision.blockers
    assert "wmma_rejected: benchmark artifact was not accepted" in decision.blockers
    assert any("gemm_too_slow: per_request_vs_row_gemv" in blocker for blocker in decision.blockers)


def test_projection_dispatch_selects_best_evidence_green_cN_candidate() -> None:
    row_gemv = ProjectionKernelSelection("linear", "w4_paro", "row_gemv")
    mmq = ProjectionDispatchCandidate(
        "mmq_caware",
        ProjectionKernelSelection("linear", "w4_paro", "mmq_caware"),
        min_rows=2,
        evidence=ProjectionDispatchEvidence(
            "benchmarks/results/mmq-c4.json",
            aggregate_vs_row_gemv=1.20,
            per_request_vs_row_gemv=1.05,
        ),
    )
    wmma = ProjectionDispatchCandidate(
        "wmma_caware",
        ProjectionKernelSelection("linear", "w4_paro", "wmma_caware"),
        min_rows=4,
        max_rows=8,
        evidence=ProjectionDispatchEvidence(
            "benchmarks/results/wmma-c4.json",
            aggregate_vs_row_gemv=1.35,
            per_request_vs_row_gemv=1.10,
        ),
    )

    decision = plan_projection_dispatch(rows=4, row_gemv=row_gemv, candidates=[mmq, wmma])

    assert decision.selection == wmma.selection
    assert decision.selected_candidate == "wmma_caware"
    assert decision.path == "benchmark_accepted_caware_projection"
    assert decision.throughput_claim_eligible is True
    assert decision.evidence == wmma.evidence
    assert decision.to_json_dict()["evidence"]["aggregate_vs_row_gemv"] == 1.35


def test_batch_sampler_dispatch_requires_c2_equality_for_batched_lm_head() -> None:
    serial = plan_batch_sampler_dispatch(rows=2, requested_mode="serial_lm_head")
    assert serial.mode is BatchSamplerMode.SERIAL_LM_HEAD
    assert serial.native_row_aware_lm_head is False

    blocked = plan_batch_sampler_dispatch(rows=2, requested_mode="batched_lm_head")
    assert blocked.requested_mode is BatchSamplerMode.BATCHED_LM_HEAD
    assert blocked.mode is BatchSamplerMode.SERIAL_LM_HEAD
    assert blocked.native_row_aware_lm_head is False
    assert "batched LM-head requires green c>N generated-token equality evidence" in blocked.blockers
    assert "batched LM-head requires an equality artifact path" in blocked.blockers

    allowed = plan_batch_sampler_dispatch(
        rows=8,
        requested_mode="batched_lm_head",
        c2_equality_green=True,
        equality_artifact="benchmarks/results/qwen35-c8-eq.json",
    )
    assert allowed.mode is BatchSamplerMode.BATCHED_LM_HEAD
    assert allowed.native_row_aware_lm_head is True
    assert allowed.to_json_dict()["equality_artifact"] == "benchmarks/results/qwen35-c8-eq.json"

    with pytest.raises(ValueError, match="unknown batch sampler mode"):
        plan_batch_sampler_dispatch(rows=2, requested_mode="surprise")


def test_hidden_bisect_dry_run_records_layer_commands(tmp_path: Path) -> None:
    output = tmp_path / "hidden-bisect.json"
    args = build_hidden_bisect_parser().parse_args(
        [
            "--dry-run",
            "--model",
            "/tmp/model",
            "--fixture",
            "fixtures/qwen35_paro/parent_512_32_seed1234.json",
            "--prompt-length",
            "32",
            "--batch-size",
            "2",
            "--decode-tokens",
            "4",
            "--max-layers",
            "8",
            "--layer-limits",
            "1,4,8",
            "--json",
            str(output),
        ]
    )

    payload = run_hidden_bisect(args, ["--dry-run", "--layer-limits", "1,4,8"])

    assert payload["status"] == "planned"
    assert payload["mode"] == "qwen35_paro_native_hidden_bisect"
    assert payload["performance_claim"] is False
    assert payload["workload"]["native_compact_prefill"] is True
    assert payload["workload"]["native_caware_decode"] is True
    assert payload["workload"]["layer_limits"] == [1, 4, 8]
    assert len(payload["commands"]) == 3
    assert all("scripts/qwen35_batch_hidden_bisect.py" in command for command in payload["commands"])
    assert output.exists()


def test_hidden_bisect_helpers_find_first_hidden_mismatch() -> None:
    assert _parse_layer_limits("1,3-4", max_layers=4) == [1, 3, 4]
    same = np.array([[0x3C00, 0x4000]], dtype=np.uint16)
    changed = np.array([[0x3C00, 0x4200]], dtype=np.uint16)

    passed = hidden_comparison(same, same.copy(), atol=0.0)
    failed = hidden_comparison(changed, same, atol=0.0)

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["bit_mismatch"] == 1
    first = _first_hidden_mismatch(
        [
            {"layer_limit": 1, "steps": [{"decode_step": 0, "generated_index": 1, "rows": [{"row": 0, "hidden_comparison": passed}]}]},
            {
                "layer_limit": 2,
                "last_layer_index": 1,
                "last_layer_type": "linear_attention",
                "steps": [{"decode_step": 3, "generated_index": 4, "rows": [{"row": 1, "hidden_comparison": failed}]}],
            },
        ]
    )
    assert first == {
        "layer_limit": 2,
        "decode_step": 3,
        "generated_index": 4,
        "row": 1,
        "max_abs": failed["max_abs"],
        "bit_mismatch": 1,
        "last_layer_index": 1,
        "last_layer_type": "linear_attention",
    }


def test_gguf_cN_diagnostic_template_records_blocked_c2_command(tmp_path: Path) -> None:
    output = tmp_path / "gguf-c2.json"
    args = build_gguf_diagnostic_parser().parse_args(
        [
            "--fixture",
            "tests/fixtures/gguf/qwen35_0_8b_q4_k_m_e2e.json",
            "--rows",
            "2",
            "--quant",
            "gguf_q4_k_m",
            "--json",
            str(output),
        ]
    )

    payload = run_gguf_diagnostic(args)

    assert payload["status"] == "blocked"
    assert payload["mode"] == "gguf_cN_equality_template"
    assert payload["rows"] == 2
    assert payload["quant"] == "gguf_q4_k_m"
    assert "scripts/qwen35_batch_gguf_diagnostic.py" in payload["command"]
    assert "--rows 2" in payload["command"]
    assert len(payload["independent_c1_commands"]) == 2
    assert all("scripts/qwen35_gguf_e2e_correctness.py" in command for command in payload["independent_c1_commands"])
    assert any("native GGUF c>N" in reason for reason in payload["blockers"])


def test_int8_cN_diagnostic_template_records_blocked_c2_gate(tmp_path: Path) -> None:
    output = tmp_path / "int8-c2.json"
    args = build_int8_diagnostic_parser().parse_args(
        [
            "--fixture",
            "fixtures/qwen35_paro/parent_512_32_seed1234.json",
            "--rows",
            "2",
            "--prompt-length",
            "512",
            "--decode-tokens",
            "128",
            "--json",
            str(output),
        ]
    )

    payload = run_int8_diagnostic(args)

    assert payload["status"] == "blocked"
    assert payload["mode"] == "qwen35_paro_int8_cN_equality_template"
    assert payload["performance_claim"] is False
    assert payload["workload"]["kv_storage_dtype"] == "int8_per_token_head"
    assert payload["workload"]["native_compact_prefill"] is False
    assert payload["workload"]["native_caware_decode"] is False
    assert payload["execution"]["batch_execution"]["throughput_claim_eligible"] is False
    assert "--kv-storage int8_per_token_head" in payload["commands"]["future_generated_token_gate"]
    assert any("compact c>N native prefill" in reason for reason in payload["blockers"])
    validate_cn_diagnostic_artifact_payload(payload)


def test_submit_poll_text_generator_preserves_prompt_order_and_row_seeds() -> None:
    inner = _FakeTextGenerator()
    adapter = SubmitPollTextGenerator(inner)
    request = GenerationRequest(
        prompts=("one", "two", "three"),
        max_tokens=5,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
        seed=7,
        row_seeds=(70, 71, 72),
    )

    outputs = adapter.generate(request)

    assert outputs == ["generated:one:70", "generated:two:71", "generated:three:72"]
    assert [seen.prompts for seen in inner.requests] == [("one", "two", "three")]
    assert inner.requests[0] == request


def test_engine_loop_cli_env_defaults_match_docs() -> None:
    parser = argparse.ArgumentParser()
    add_engine_loop_config_args(parser, environ={})
    config = engine_loop_config_from_args(parser.parse_args([]))

    assert config == EngineLoopConfig()
    assert config.prefill_decode_policy == "protect_decode"
    assert config.kv_pool_initial_pages == 128
    assert config.kv_pool_low_water_pages == 128
    assert config.kv_pool_high_water_pages is None
    assert config.kv_pool_chunk_pages == 128
    assert config.kv_pool_idle_grace_seconds == 30.0

    docs = Path("docs/ENVS.md").read_text()
    for text in [
        "HIPENGINE_PREFILL_DECODE_POLICY",
        "HIPENGINE_KV_POOL_INITIAL_PAGES",
        "HIPENGINE_KV_POOL_LOW_WATER_PAGES",
        "HIPENGINE_KV_POOL_HIGH_WATER_PAGES",
        "HIPENGINE_KV_POOL_CHUNK_PAGES",
        "HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS",
        "protect_decode",
        "128",
        "30.0",
    ]:
        assert text in docs


def test_engine_loop_cli_env_overrides() -> None:
    env = {
        "HIPENGINE_PREFILL_DECODE_POLICY": "fair",
        "HIPENGINE_KV_POOL_INITIAL_PAGES": "16",
        "HIPENGINE_KV_POOL_LOW_WATER_PAGES": "8",
        "HIPENGINE_KV_POOL_HIGH_WATER_PAGES": "64",
        "HIPENGINE_KV_POOL_CHUNK_PAGES": "4",
        "HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS": "1.5",
    }
    env_config = engine_loop_config_from_env(env)
    assert env_config == EngineLoopConfig(
        prefill_decode_policy="fair",
        kv_pool_initial_pages=16,
        kv_pool_low_water_pages=8,
        kv_pool_high_water_pages=64,
        kv_pool_chunk_pages=4,
        kv_pool_idle_grace_seconds=1.5,
    )

    parser = argparse.ArgumentParser()
    add_engine_loop_config_args(parser, environ=env)
    cli_config = engine_loop_config_from_args(
        parser.parse_args(
            [
                "--prefill-decode-policy",
                "protect_ttft",
                "--kv-pool-initial-pages",
                "32",
                "--kv-pool-low-water-pages",
                "16",
                "--kv-pool-high-water-pages",
                "96",
                "--kv-pool-chunk-pages",
                "8",
                "--kv-pool-idle-grace-seconds",
                "2.5",
            ]
        )
    )
    assert cli_config.prefill_decode_policy == "protect_ttft"
    assert cli_config.kv_pool_initial_pages == 32
    assert cli_config.kv_pool_low_water_pages == 16
    assert cli_config.kv_pool_high_water_pages == 96
    assert cli_config.kv_pool_chunk_pages == 8
    assert cli_config.kv_pool_idle_grace_seconds == 2.5


def test_resident_scheduler_completion_observability_and_pool_counters() -> None:
    now = 100.0

    def clock() -> float:
        return now

    scheduler = ResidentBatchScheduler(capacity=1, context_bucket_size=4, clock=clock)
    r0 = scheduler.submit([1, 2, 3, 4, 5], max_new_tokens=1)
    now = 101.0
    assert scheduler.admit_pending() == (r0,)
    r1 = scheduler.submit([8], max_new_tokens=1)
    assert scheduler.admit_pending() == ()

    prefill = scheduler.next_prefill_work(chunk_size=8)
    assert prefill is not None
    scheduler.record_work_duration(prefill, 0.25)
    decode = scheduler.next_decode_work()
    assert decode is not None
    scheduler.record_work_duration(decode, 0.5)
    now = 102.0
    done = scheduler.record_generated([GeneratedToken(r0, 9)])[0]

    observed = done.to_json_dict()["observability"]
    assert observed["queue_seconds"] == 1.0
    assert observed["prefill_seconds"] == 0.25
    assert observed["decode_seconds"] == 0.5
    assert observed["kv_pages_owned"] == 1
    assert observed["kv_pages_peak"] == 1
    assert str(observed["bucket_key"]).startswith("decode:c=1:ctx=8")
    assert observed["admission_blocked_reason"] is None
    assert observed["finish_reason"] == "length"
    assert observed["completion_timestamp"] == 102.0

    now = 103.0
    assert scheduler.admit_pending() == (r1,)
    assert scheduler.next_prefill_work(chunk_size=8) is not None
    blocked_done = scheduler.cancel(r1)
    assert blocked_done is not None
    assert blocked_done.observability.admission_blocked_reason == "capacity"
    assert blocked_done.observability.finish_reason == "cancel"

    pool = ChunkedKVPool(page_bytes=4096, initial_pages=2, low_water_pages=1, chunk_pages=2)
    allocation = pool.allocate(3)
    pool.release(allocation.block_ids)
    counters = pool.stats.to_json_dict()
    for field in [
        "current_bytes",
        "high_water_observed_bytes",
        "grow_events",
        "grow_failures",
        "shrink_events",
        "free_pages",
        "refcounted_pages",
    ]:
        assert field in counters
    assert counters["current_bytes"] == pool.stats.current_bytes
    assert counters["grow_events"] == 1


def test_resident_engine_loop_submit_poll_cancel_and_reclaim() -> None:
    runner = _FakeSerialBridgeRunner()
    loop = ResidentEngineLoop(runner, capacity=2, prefill_chunk_size=8, context_bucket_size=4)
    r0 = loop.submit([10, 11], max_new_tokens=2)
    r1 = loop.submit([20], max_new_tokens=1)
    r2 = loop.submit([30], max_new_tokens=1)
    r3 = loop.submit([40], max_new_tokens=4)

    assert loop.cancel(r3) is True
    assert loop.cancel(9999) is False
    assert loop.completed[r3].finished is True
    assert loop.completed[r3].finish_reason == "cancel"
    assert loop.pending_count == 3

    events = loop.poll(max_ticks=8)

    assert [(work.kind, work.request_ids) for work in runner.prefills] == [
        (WorkKind.PREFILL, (r0,)),
        (WorkKind.PREFILL, (r2,)),
        (WorkKind.PREFILL, (r1,)),
    ]
    assert [(work.kind, work.request_ids) for work in runner.decodes] == [
        (WorkKind.DECODE, (r0,)),
        (WorkKind.DECODE, (r0,)),
        (WorkKind.DECODE, (r2,)),
        (WorkKind.DECODE, (r1,)),
    ]
    assert [event.request_id for event in events if event.kind == "completed"] == [r0, r2, r1]
    assert loop.pending_count == 0
    assert loop.active_count == 0
    assert set(loop.completed) == {r0, r1, r2, r3}
    assert loop.completed[r0].generated_tokens == (1000, 1001)
    assert loop.completed[r0].finish_reason == "length"
    assert loop.completed[r1].generated_tokens == (1010,)
    assert loop.completed[r1].finish_reason == "length"
    assert loop.completed[r2].generated_tokens == (1020,)
    assert loop.completed[r2].finish_reason == "length"

    active_cancel_loop = ResidentEngineLoop(_FakeSerialBridgeRunner(), capacity=1, prefill_chunk_size=8)
    active_id = active_cancel_loop.submit([50], max_new_tokens=4)
    assert active_cancel_loop.poll(max_ticks=1)[0].kind == "admitted"
    assert active_cancel_loop.active_count == 1
    assert active_cancel_loop.cancel(active_id) is True
    assert active_cancel_loop.active_count == 0
    assert active_cancel_loop.completed[active_id].finished is True
    assert active_cancel_loop.completed[active_id].finish_reason == "cancel"


def _prefilled_scheduler(*, max_new_tokens: int = 1) -> tuple[ResidentBatchScheduler, int]:
    scheduler = ResidentBatchScheduler(capacity=1, context_bucket_size=4)
    request_id = scheduler.submit([10], max_new_tokens=max_new_tokens)
    scheduler.admit_pending()
    scheduler.next_prefill_work(chunk_size=8)
    return scheduler, request_id


def test_resident_scheduler_unified_reclaim_finish_reasons() -> None:
    pending = ResidentBatchScheduler(capacity=1, context_bucket_size=4)
    pending_id = pending.submit([1], max_new_tokens=1)
    done = pending.cancel(pending_id)
    assert done is not None and done.finish_reason == "cancel"
    assert pending.pending_count == 0
    assert pending.cancel(pending_id) is None
    assert len(pending.completed) == 1

    disconnected, disconnected_id = _prefilled_scheduler(max_new_tokens=2)
    done = disconnected.disconnect(disconnected_id)
    assert done is not None and done.finish_reason == "disconnect"
    assert disconnected.active_count == 0
    assert disconnected.disconnect(disconnected_id) is None
    assert len(disconnected.completed) == 1

    timed_out, timeout_id = _prefilled_scheduler(max_new_tokens=2)
    done = timed_out.timeout(timeout_id)
    assert done is not None and done.finish_reason == "timeout"
    assert timed_out.active_count == 0
    assert timed_out.timeout(timeout_id) is None
    assert len(timed_out.completed) == 1

    eos, eos_id = _prefilled_scheduler(max_new_tokens=3)
    done_items = eos.record_generated([GeneratedToken(eos_id, 99, finished=True)])
    assert [item.finish_reason for item in done_items] == ["stop"]
    assert eos.active_count == 0
    assert eos.cancel(eos_id) is None
    assert len(eos.completed) == 1

    length, length_id = _prefilled_scheduler(max_new_tokens=1)
    done_items = length.record_generated([GeneratedToken(length_id, 100)])
    assert [item.finish_reason for item in done_items] == ["length"]
    assert length.active_count == 0
    assert length.cancel(length_id) is None
    assert len(length.completed) == 1

    with pytest.raises(ValueError, match="cancel reason"):
        pending.cancel(123, reason="stop")


def test_resident_scheduler_per_row_sampler_block_keeps_incompatible_rows_together() -> None:
    scheduler = ResidentBatchScheduler(capacity=3, context_bucket_size=4)
    r0 = scheduler.submit(
        [10],
        max_new_tokens=2,
        sampling=PerRowSamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=7, stop_tokens=(99,)),
    )
    r1 = scheduler.submit(
        [20],
        max_new_tokens=2,
        sampling=PerRowSamplingParams(temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.1, seed=7),
    )
    r2 = scheduler.submit(
        [30],
        max_new_tokens=2,
        sampling=PerRowSamplingParams(temperature=1.0, top_k=0, top_p=0.8, repetition_penalty=1.2, seed=99),
    )
    scheduler.admit_pending()
    for _ in range(3):
        assert scheduler.next_prefill_work(chunk_size=8) is not None

    decode = scheduler.next_decode_work()
    assert decode is not None
    assert decode.request_ids == (r0, r1, r2)

    block = scheduler.sampler_params_block(decode.request_ids)
    again = scheduler.sampler_params_block(decode.request_ids)

    assert block.request_ids == (r0, r1, r2)
    assert block.temperatures == (0.0, 0.7, 1.0)
    assert block.top_ks == (1, 40, 0)
    assert block.top_ps == (1.0, 0.9, 0.8)
    assert block.repetition_penalties == (1.0, 1.1, 1.2)
    assert block.stop_token_rows == ((99,), (), ())
    assert block.seeds == again.seeds
    assert len(set(block.seeds)) == 3
    assert block.params_for(r1).temperature == 0.7

    scheduler.record_generated([GeneratedToken(r0, 100, finished=True)])
    with pytest.raises(KeyError, match="sampler params"):
        scheduler.sampler_params_block((r0,))


def test_resident_scheduler_per_row_eos_reclaims_finished_rows_only() -> None:
    reclaimed: list[tuple[int, str]] = []
    policy = FixedPagedKVPolicy(block_size=16, total_capacity_tokens=96)

    def reclaim(done) -> None:
        reclaimed.append((done.request_id, done.finish_reason))
        policy.reclaim(done.request_id)

    scheduler = ResidentBatchScheduler(capacity=3, context_bucket_size=4, reclaim_callback=reclaim)
    r0 = scheduler.submit([10], max_new_tokens=3)
    r1 = scheduler.submit([20], max_new_tokens=2)
    r2 = scheduler.submit([30], max_new_tokens=3)
    scheduler.admit_pending()
    for request_id, ptr in [(r0, 0x1000), (r1, 0x2000), (r2, 0x3000)]:
        policy.register(
            request_id,
            block_table=_tensor(ptr, (1,), "int32"),
            live_counts=_tensor(ptr + 0x100, (1,), "int64"),
            max_live_count=1,
            capacity_tokens=16,
        )
    for _ in range(3):
        assert scheduler.next_prefill_work(chunk_size=8) is not None

    decode = scheduler.next_decode_work()
    assert decode is not None and decode.request_ids == (r0, r1, r2)
    completed = scheduler.record_generated(
        [
            GeneratedToken(r0, 100, finished=True),
            GeneratedToken(r1, 200),
            GeneratedToken(r2, 300),
        ]
    )

    assert [(item.request_id, item.finish_reason) for item in completed] == [(r0, "stop")]
    assert reclaimed == [(r0, "stop")]
    assert r0 not in policy.reservations
    assert set(policy.reservations) == {r1, r2}
    assert scheduler.active_batch.slot_to_request == (None, r1, r2)

    decode = scheduler.next_decode_work()
    assert decode is not None and decode.request_ids == (r1, r2)
    completed = scheduler.record_generated([GeneratedToken(r1, 201), GeneratedToken(r2, 301)])

    assert [(item.request_id, item.finish_reason) for item in completed] == [(r1, "length")]
    assert reclaimed == [(r0, "stop"), (r1, "length")]
    assert r1 not in policy.reservations
    assert set(policy.reservations) == {r2}

    decode = scheduler.next_decode_work()
    assert decode is not None and decode.request_ids == (r2,)
    completed = scheduler.record_generated([GeneratedToken(r2, 302, finished=True)])

    assert [(item.request_id, item.finish_reason) for item in completed] == [(r2, "stop")]
    assert reclaimed == [(r0, "stop"), (r1, "length"), (r2, "stop")]
    assert policy.reservations == {}
    assert scheduler.active_count == 0


def test_resident_engine_loop_prefill_decode_policies() -> None:
    protect_runner = _FakeSerialBridgeRunner()
    protect_loop = ResidentEngineLoop(
        protect_runner,
        capacity=2,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_decode",
    )
    r0 = protect_loop.submit([10], max_new_tokens=2)
    r1 = protect_loop.submit([20], max_new_tokens=1)
    protect_loop.poll(max_ticks=1)
    protect_loop.poll(max_ticks=1)
    assert [(work.kind, work.request_ids) for work in protect_runner.decodes] == [(WorkKind.DECODE, (r0,))]
    assert [(work.kind, work.request_ids) for work in protect_runner.prefills] == [(WorkKind.PREFILL, (r0,))]

    ttft_runner = _FakeSerialBridgeRunner()
    ttft_loop = ResidentEngineLoop(
        ttft_runner,
        capacity=2,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    r0 = ttft_loop.submit([10], max_new_tokens=2)
    r1 = ttft_loop.submit([20], max_new_tokens=1)
    ttft_loop.poll(max_ticks=1)
    ttft_loop.poll(max_ticks=1)
    assert ttft_runner.decodes == []
    assert [(work.kind, work.request_ids) for work in ttft_runner.prefills] == [
        (WorkKind.PREFILL, (r0,)),
        (WorkKind.PREFILL, (r1,)),
    ]

    fair_runner = _FakeSerialBridgeRunner()
    fair_loop = ResidentEngineLoop(
        fair_runner,
        capacity=2,
        prefill_chunk_size=8,
        prefill_decode_policy="fair",
    )
    r0 = fair_loop.submit([10], max_new_tokens=2)
    r1 = fair_loop.submit([20], max_new_tokens=1)
    fair_loop.poll(max_ticks=1)
    fair_loop.poll(max_ticks=1)
    fair_loop.poll(max_ticks=1)
    assert [(work.kind, work.request_ids) for work in fair_runner.decodes] == [(WorkKind.DECODE, (r0,))]
    assert [(work.kind, work.request_ids) for work in fair_runner.prefills] == [
        (WorkKind.PREFILL, (r0,)),
        (WorkKind.PREFILL, (r1,)),
    ]

    with pytest.raises(ValueError, match="prefill_decode_policy"):
        ResidentEngineLoop(_FakeSerialBridgeRunner(), capacity=1, prefill_decode_policy="unknown")


def test_resident_batch_scheduler_admits_compacts_and_routes_decode() -> None:
    scheduler = ResidentBatchScheduler(capacity=2, context_bucket_size=4)
    r0 = scheduler.submit([10, 11], max_new_tokens=2)
    r1 = scheduler.submit([20], max_new_tokens=1)
    r2 = scheduler.submit([30], max_new_tokens=1)

    assert (r0, r1, r2) == (0, 1, 2)
    assert scheduler.admit_pending() == (0, 1)
    assert scheduler.pending_count == 1
    assert scheduler.active_batch.slot_to_request == (0, 1)

    work = scheduler.next_prefill_work(chunk_size=8)
    assert work is not None
    assert work.kind is WorkKind.PREFILL
    assert work.request_ids == (0,)
    assert work.token_rows == ((10, 11),)

    work = scheduler.next_prefill_work(chunk_size=8)
    assert work is not None
    assert work.request_ids == (1,)
    assert work.token_rows == ((20,),)

    decode = scheduler.next_decode_work()
    assert decode is not None
    assert decode.kind is WorkKind.DECODE
    assert decode.request_ids == (0, 1)
    assert decode.row_to_request == (0, 1)

    completed = scheduler.record_generated([(1, 101, True)])
    assert [item.request_id for item in completed] == [1]
    assert scheduler.active_batch.slot_to_request == (0, None)

    assert scheduler.admit_pending() == (2,)
    assert scheduler.active_batch.slot_to_request == (0, 2)

    moves = scheduler.compact(order=(2, 0))
    assert [(move.request_id, move.old_slot, move.new_slot) for move in moves] == [(2, 1, 0), (0, 0, 1)]
    assert scheduler.active_batch.slot_to_request == (2, 0)


def test_resident_batch_scheduler_bucketizes_and_builds_compact_prefill_slabs() -> None:
    scheduler = ResidentBatchScheduler(capacity=3, context_bucket_size=4)
    r0 = scheduler.submit([10, 11, 12], max_new_tokens=1)
    r1 = scheduler.submit([20, 21, 22, 23, 24], max_new_tokens=1)
    r2 = scheduler.submit([30, 31], max_new_tokens=1)
    scheduler.admit_pending()

    buckets = scheduler.bucketize_by_block_count(chunk_size=8, block_size=4)

    assert [(bucket.block_count, bucket.request_ids) for bucket in buckets] == [
        (1, (r0, r2)),
        (2, (r1,)),
    ]

    slabs = scheduler.next_compact_prefill_slabs(chunk_size=8, block_size=4)

    assert len(slabs) == 2
    first = slabs[0]
    assert first.request_ids == (r0, r2)
    assert first.slot_ids == (0, 2)
    assert first.physical_slot_ids == (0, 2)
    assert first.token_ids == (10, 11, 12, 30, 31)
    assert first.positions == (0, 1, 2, 0, 1)
    assert first.append_counts == first.positions
    assert first.context_counts == (1, 2, 3, 1, 2)
    assert first.cu_seqlens_q == (0, 3, 5)
    assert first.cu_seqlens_k == (0, 3, 5)
    assert first.row_to_request == (r0, r0, r0, r2, r2)
    assert first.block_count == 1
    assert first.block_tables == ((0,), (0,), (0,), (0,), (0,))
    assert first.to_work_item().token_rows == ((10, 11, 12), (30, 31))

    second = slabs[1]
    assert second.request_ids == (r1,)
    assert second.slot_ids == (1,)
    assert second.token_ids == (20, 21, 22, 23, 24)
    assert second.cu_seqlens_q == (0, 5)
    assert second.block_count == 2
    assert second.block_tables == ((0, 1),) * 5
    assert scheduler.active_batch.requests[r0].remaining_prefill == 0
    assert scheduler.active_batch.requests[r1].remaining_prefill == 0
    assert scheduler.active_batch.requests[r2].remaining_prefill == 0


def test_resident_batch_scheduler_emits_speculative_verify_work() -> None:
    scheduler = ResidentBatchScheduler(capacity=2, context_bucket_size=4)
    r0 = scheduler.submit([10, 11], max_new_tokens=3)
    r1 = scheduler.submit([20], max_new_tokens=1)
    scheduler.admit_pending()
    scheduler.next_prefill_work(chunk_size=8)
    scheduler.next_prefill_work(chunk_size=8)
    draft = DraftBatch(
        request_ids=(r0, r1),
        candidate_tokens=(101, 102, 201),
        parent_positions=(1, 2, 0),
        draft_depths=(1, 2, 1),
        row_to_request=(r0, r0, r1),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )

    work = scheduler.next_speculative_verify_work(
        draft,
        root_tokens=(11, 20),
        root_positions=(1, 0),
    )

    assert isinstance(work, SpeculativeVerifyWork)
    assert work.target_batch.rows == 5
    assert work.target_batch.tokens == (11, 20, 101, 102, 201)
    assert work.target_batch.parent_rows == (-1, -1, 0, 2, 1)
    assert work.work_item.kind is WorkKind.VERIFY_TREE
    assert work.work_item.request_ids == (r0, r1)
    assert work.work_item.row_to_request == (r0, r0, r1)
    assert work.work_item.token_rows == ((101,), (102,), (201,))
    assert work.work_item.tree_parents == (0, 1, 0)

    key = scheduler.speculative_verify_shape_key(work, top_k=8, experts_per_token=8, replay_steps=2)
    assert key.mode is WorkKind.VERIFY_TREE
    assert key.active_c == 2
    assert key.context_bucket == 4
    assert key.active_mask == (True, True)
    assert key.top_k == 8
    assert key.experts_per_token == 8
    assert key.replay_steps == 2
    assert key.draft_depth == 2
    assert key.tree_shape == (0, 1, 0)
    graph = scheduler.get_or_create_speculative_verify_graph(
        work,
        lambda bucket: {"bucket": bucket},
        top_k=8,
        experts_per_token=8,
        replay_steps=2,
    )
    assert graph == {"bucket": key}
    assert scheduler.graph_buckets.stats.entries == 1
    assert scheduler.get_or_create_speculative_verify_graph(
        work,
        lambda bucket: {"unexpected": bucket},
        top_k=8,
        experts_per_token=8,
        replay_steps=2,
    ) is graph
    assert scheduler.graph_buckets.stats.hits == 1

    policy = FixedPagedKVPolicy()
    for request_id, ptr in [(r0, 0x1000), (r1, 0x2000)]:
        policy.register(
            request_id,
            block_table=_tensor(ptr, (4,), "int32"),
            live_counts=_tensor(ptr + 0x100, (1,), "int64"),
            max_live_count=4,
        )
    txn = scheduler.begin_speculative_verify_transaction(policy, work)
    assert txn.request_ids == (r0, r1)
    assert txn.draft_rows == 3
    assert txn.candidate_counts == (2, 1)
    assert txn.role == "verify_tree"
    plan = scheduler.plan_speculative_verify(
        policy,
        work,
        lambda bucket: {"unexpected": bucket},
        top_k=8,
        experts_per_token=8,
        replay_steps=2,
    )
    assert isinstance(plan, SpeculativeVerifyPlan)
    assert plan.target_batch is work.target_batch
    assert plan.work_item is work.work_item
    assert plan.transaction.request_ids == (r0, r1)
    assert plan.transaction.draft_rows == 3
    assert plan.transaction.candidate_counts == (2, 1)
    assert plan.shape_key == key
    assert plan.graph is graph
    assert scheduler.graph_buckets.stats.hits == 2
    rollback_plan = scheduler.plan_speculative_verify(
        policy,
        work,
        lambda bucket: {"unexpected_rollback": bucket},
        top_k=8,
        experts_per_token=8,
        replay_steps=2,
    )
    rolled_txn = scheduler.rollback_speculative_kv_transaction(policy, rollback_plan)
    assert rolled_txn.transaction_id == rollback_plan.transaction.transaction_id
    assert rolled_txn.request_ids == (r0, r1)
    assert rolled_txn.rolled_back
    assert not rolled_txn.committed
    assert scheduler.graph_buckets.stats.hits == 3
    buffers = TargetVerifyBuffers.for_batch(
        work.target_batch,
        token_ids=_tensor(0x3000, (work.target_batch.rows,), "int32"),
        positions=_tensor(0x3100, (work.target_batch.rows,), "int32"),
        parent_rows=_tensor(0x3200, (work.target_batch.rows,), "int32"),
        draft_depths=_tensor(0x3300, (work.target_batch.rows,), "int32"),
        row_to_request=_tensor(0x3400, (work.target_batch.rows,), "int32"),
        active_mask=_tensor(0x3500, (work.target_batch.rows,), "bool"),
        target_top1=_tensor(0x3600, (work.target_batch.rows,), "int32"),
        accepted_counts=_tensor(0x3700, (len(work.target_batch.request_ids),), "int32"),
        commit_rows=_tensor(0x3800, (len(work.target_batch.request_ids),), "int32"),
        commit_tokens=_tensor(0x3900, (len(work.target_batch.request_ids),), "int32"),
        commit_positions=_tensor(0x3A00, (len(work.target_batch.request_ids),), "int32"),
        transaction_id=plan.transaction.transaction_id,
    )
    buffer_plan = scheduler.bind_speculative_verify_buffers(plan, buffers)
    assert buffers.transaction_id == plan.transaction.transaction_id
    assert buffers.candidate_counts == work.target_batch.candidate_counts
    assert buffers.draft_depth == work.target_batch.draft_depth
    assert buffers.tree_shape == work.target_batch.tree_shape
    assert isinstance(buffer_plan, SpeculativeVerifyBufferPlan)
    assert buffer_plan.plan is plan
    assert buffer_plan.buffers is buffers
    wrong_verify_buffers = replace(buffers, transaction_id=plan.transaction.transaction_id + 1)
    with pytest.raises(ValueError, match="transaction_id"):
        scheduler.bind_speculative_verify_buffers(plan, wrong_verify_buffers)
    wrong_candidate_buffers = replace(buffers, candidate_counts=(1, 2))
    with pytest.raises(ValueError, match="candidate_counts"):
        scheduler.bind_speculative_verify_buffers(plan, wrong_candidate_buffers)
    wrong_depth_buffers = replace(buffers, draft_depth=work.target_batch.draft_depth + 1)
    with pytest.raises(ValueError, match="draft_depth"):
        scheduler.bind_speculative_verify_buffers(plan, wrong_depth_buffers)
    wrong_tree_buffers = replace(buffers, tree_shape=(0, 0, 1))
    with pytest.raises(ValueError, match="tree_shape"):
        scheduler.bind_speculative_verify_buffers(plan, wrong_tree_buffers)

    commit = scheduler.plan_speculative_commit_from_top1(buffer_plan, (101, 201, 102, 103, 202))
    assert isinstance(commit, SpeculativeCommitPlan)
    assert commit.verify_plan is buffer_plan
    summary = commit.summary
    assert summary.transaction_id == plan.transaction.transaction_id
    assert summary.accepted_tokens == ((101, 102), (201,))
    assert summary.next_tokens == (103, None)
    assert commit.commit_plan.transaction_id == plan.transaction.transaction_id
    assert commit.commit_plan.request_ids == (r0, r1)
    assert commit.commit_plan.accepted_counts == (2, 1)
    assert commit.commit_plan.commit_rows == (3, 4)
    assert commit.commit_plan.next_tokens == (103, None)
    assert commit.commit_plan.candidate_counts == (2, 1)
    assert commit.commit_plan.draft_depth == work.target_batch.draft_depth
    assert commit.commit_plan.tree_shape == work.target_batch.tree_shape
    assert commit.commit_plan.mode == "verify_tree"
    with pytest.raises(ValueError, match="target_top1"):
        scheduler.plan_speculative_commit_from_top1(buffer_plan, (101, 201))
    wrong_summary_txn = replace(summary, transaction_id=plan.transaction.transaction_id + 1)
    with pytest.raises(ValueError, match="transaction_id"):
        scheduler.plan_speculative_commit(buffer_plan, wrong_summary_txn)
    wrong_summary_depth = replace(summary, draft_depth=work.target_batch.draft_depth + 1)
    with pytest.raises(ValueError, match="draft_depth"):
        scheduler.plan_speculative_commit(buffer_plan, wrong_summary_depth)
    wrong_summary_tree = replace(summary, tree_shape=(0, 0, 1))
    with pytest.raises(ValueError, match="tree_shape"):
        scheduler.plan_speculative_commit(buffer_plan, wrong_summary_tree)
    state_buffers = TargetStateCommitBuffers.for_plan(
        commit.commit_plan,
        accepted_counts=_tensor(0x3B00, (len(work.target_batch.request_ids),), "int32"),
        commit_rows=_tensor(0x3C00, (len(work.target_batch.request_ids),), "int32"),
        commit_positions=_tensor(0x3D00, (len(work.target_batch.request_ids),), "int32"),
        linear_state_src=_tensor(0x3E00, (work.target_batch.rows, 4), "bf16"),
        linear_state_dst=_tensor(0x3F00, (len(work.target_batch.request_ids), 4), "bf16"),
        kv_rows_src=_tensor(0x4000, (work.target_batch.rows, 2, 4), "bf16"),
        kv_rows_dst=_tensor(0x4100, (sum(summary.accepted_counts), 2, 4), "bf16"),
    )
    state_plan = scheduler.bind_speculative_commit_buffers(commit, state_buffers)
    assert state_buffers.transaction_id == commit.commit_plan.transaction_id
    assert isinstance(state_plan, SpeculativeStateCommitPlan)
    assert state_plan.commit_plan is commit
    assert state_plan.buffers is state_buffers
    assert state_plan.buffers.device == buffer_plan.buffers.device
    assert state_plan.buffers.linear_state_src is not None
    assert state_plan.buffers.linear_state_src.shape[0] == work.target_batch.rows
    assert state_plan.buffers.kv_rows_dst is not None
    assert state_plan.buffers.kv_rows_dst.shape[0] == sum(summary.accepted_counts)
    assert state_plan.buffers.has_linear_state
    assert state_plan.buffers.has_kv_rows
    short_kv_dst_buffers = TargetStateCommitBuffers.for_plan(
        commit.commit_plan,
        accepted_counts=_tensor(0x4200, (len(work.target_batch.request_ids),), "int32"),
        commit_rows=_tensor(0x4300, (len(work.target_batch.request_ids),), "int32"),
        commit_positions=_tensor(0x4400, (len(work.target_batch.request_ids),), "int32"),
        kv_rows_src=_tensor(0x4500, (work.target_batch.rows, 2, 4), "bf16"),
        kv_rows_dst=_tensor(0x4600, (len(work.target_batch.request_ids), 2, 4), "bf16"),
    )
    with pytest.raises(ValueError, match="accepted token rows"):
        scheduler.bind_speculative_commit_buffers(commit, short_kv_dst_buffers)
    wrong_transaction_buffers = replace(state_buffers, transaction_id=commit.commit_plan.transaction_id + 1)
    with pytest.raises(ValueError, match="transaction_id"):
        scheduler.bind_speculative_commit_buffers(commit, wrong_transaction_buffers)
    other_device = Device("hip", 1)
    other_state_buffers = TargetStateCommitBuffers.for_plan(
        commit.commit_plan,
        accepted_counts=Tensor.from_handle(0x4700, (len(work.target_batch.request_ids),), "int32", other_device),
        commit_rows=Tensor.from_handle(0x4800, (len(work.target_batch.request_ids),), "int32", other_device),
        commit_positions=Tensor.from_handle(0x4900, (len(work.target_batch.request_ids),), "int32", other_device),
        linear_state_src=Tensor.from_handle(0x4A00, (work.target_batch.rows, 4), "bf16", other_device),
        linear_state_dst=Tensor.from_handle(0x4B00, (len(work.target_batch.request_ids), 4), "bf16", other_device),
    )
    with pytest.raises(ValueError, match="target verify device"):
        scheduler.bind_speculative_commit_buffers(commit, other_state_buffers)
    committed_txn = scheduler.commit_speculative_kv_transaction(policy, state_plan)
    assert committed_txn.transaction_id == plan.transaction.transaction_id
    assert committed_txn.request_ids == (r0, r1)
    assert committed_txn.accepted_counts == (2, 1)
    assert committed_txn.committed
    completed = scheduler.finalize_speculative_accept(committed_txn, state_plan)

    assert [item.request_id for item in completed] == [r0, r1]
    assert scheduler.completed[r0].generated_tokens == (101, 102, 103)
    assert scheduler.completed[r1].generated_tokens == (201,)
    assert scheduler.active_batch.slot_to_request == (None, None)


def test_resident_batch_scheduler_rejects_speculative_accept_over_budget() -> None:
    scheduler = ResidentBatchScheduler(capacity=1)
    r0 = scheduler.submit([10], max_new_tokens=1)
    scheduler.admit_pending()
    scheduler.next_prefill_work(chunk_size=8)
    draft = DraftBatch(
        request_ids=(r0,),
        candidate_tokens=(101, 102),
        parent_positions=(0, 1),
        draft_depths=(1, 2),
        row_to_request=(r0, r0),
    )
    work = scheduler.next_speculative_verify_work(draft, root_tokens=(10,), root_positions=(0,))
    summary = TargetAcceptSummary.from_accept_result(
        work.target_batch,
        AcceptResult(request_ids=(r0,), accepted_counts=(2,), accepted_tokens=((101, 102),)),
    )

    with pytest.raises(ValueError, match="remaining decode"):
        scheduler.record_speculative_accept(summary)

    next_token_over_budget_summary = TargetAcceptSummary.from_accept_result(
        work.target_batch,
        AcceptResult(request_ids=(r0,), accepted_counts=(1,), accepted_tokens=((101,),), next_tokens=(102,)),
        selected_candidate_rows=(1,),
    )
    with pytest.raises(ValueError, match="remaining decode"):
        scheduler.record_speculative_accept(next_token_over_budget_summary)


def test_resident_batch_scheduler_rejects_speculative_verify_before_prefill() -> None:
    scheduler = ResidentBatchScheduler(capacity=1)
    r0 = scheduler.submit([10, 11], max_new_tokens=3)
    scheduler.admit_pending()
    draft = DraftBatch(
        request_ids=(r0,),
        candidate_tokens=(101,),
        parent_positions=(1,),
        draft_depths=(1,),
        row_to_request=(r0,),
    )

    with pytest.raises(ValueError, match="completed prefill"):
        scheduler.next_speculative_verify_work(draft, root_tokens=(11,), root_positions=(1,))


def test_resident_batch_scheduler_shape_key_graph_bucket_and_completion() -> None:
    scheduler = ResidentBatchScheduler(capacity=4, context_bucket_size=4)
    r0 = scheduler.submit([1], max_new_tokens=1)
    r1 = scheduler.submit([2, 3, 4, 5], max_new_tokens=2)
    scheduler.admit_pending()
    scheduler.next_prefill_work(chunk_size=1)
    scheduler.next_prefill_work(chunk_size=4)

    key = scheduler.shape_key(mode=WorkKind.DECODE, top_k=8, experts_per_token=8, replay_steps=2)
    assert key.mode is WorkKind.DECODE
    assert key.active_c == 2
    assert key.context_bucket == 4
    assert key.active_mask == (True, True, False, False)
    assert key.top_k == 8
    assert key.experts_per_token == 8
    assert key.replay_steps == 2

    graph = scheduler.graph_buckets.get_or_create(key, lambda bucket: {"bucket": bucket})
    assert graph == {"bucket": key}
    assert scheduler.graph_buckets.stats.entries == 1
    assert scheduler.graph_buckets.stats.hits == 0
    assert scheduler.graph_buckets.stats.misses == 1
    assert scheduler.graph_buckets.get(key) is graph
    assert scheduler.graph_buckets.stats.hits == 1

    done = scheduler.record_generated([GeneratedToken(r0, 99)])
    assert [item.request_id for item in done] == [r0]
    assert scheduler.completed[r0].generated_tokens == (99,)
    assert not scheduler.completed[r0].prompt_tokens == ()

    scheduler.record_generated([(r1, 100), (r1, 101)])
    assert scheduler.completed[r1].generated_tokens == (100, 101)
    assert scheduler.active_count == 0


def test_graph_bucket_cache_clear_resets_entries_and_counters() -> None:
    cache = GraphBucketCache()
    scheduler = ResidentBatchScheduler(capacity=1, context_bucket_size=4)
    scheduler.submit([1], max_new_tokens=1)
    scheduler.admit_pending()
    scheduler.next_prefill_work(chunk_size=1)
    key = scheduler.shape_key(mode="decode")

    assert cache.get(key) is None
    cache.put(key, object())
    assert cache.stats.entries == 1
    cache.clear()
    assert cache.stats.entries == 0
    assert cache.stats.hits == 0
    assert cache.stats.misses == 0


def test_compact_prompt_slab_tracks_optional_physical_slots() -> None:
    slab = CompactPromptSlab.from_token_rows(
        request_ids=(10, 11),
        token_rows=((1,), (2, 3)),
        start_positions=(0, 4),
        block_count=1,
        slot_ids=(1, 0),
    )

    assert slab.slot_ids == (1, 0)
    assert slab.physical_slot_ids == (1, 0)

    legacy = CompactPromptSlab.from_token_rows(
        request_ids=(3,), token_rows=((4,),), start_positions=(0,), block_count=1
    )
    assert legacy.physical_slot_ids == (3,)

    with pytest.raises(ValueError, match="slot_ids"):
        CompactPromptSlab.from_token_rows(
            request_ids=(10, 11),
            token_rows=((1,), (2,)),
            start_positions=(0, 0),
            block_count=1,
            slot_ids=(0,),
        )


def test_compact_prompt_slab_validates_cu_seqlens_and_row_shapes() -> None:
    with pytest.raises(ValueError, match="cu_seqlens must end"):
        CompactPromptSlab(
            request_ids=(1,),
            token_ids=(10, 11),
            positions=(0, 1),
            cu_seqlens_q=(0, 1),
            cu_seqlens_k=(0, 2),
            row_to_request=(1, 1),
            block_tables=((0,), (0,)),
            append_counts=(0, 1),
            context_counts=(1, 2),
            token_rows=((10, 11),),
            block_count=1,
        )
    with pytest.raises(ValueError, match="block_tables rows"):
        CompactPromptSlab.from_token_rows(
            request_ids=(1,),
            token_rows=((10,),),
            start_positions=(0,),
            block_count=2,
            block_tables_by_request=((0,),),
        )


def test_resident_batch_scheduler_rejects_duplicate_ids_and_invalid_chunks() -> None:
    scheduler = ResidentBatchScheduler(capacity=1)
    scheduler.submit([1], max_new_tokens=1, request_id=7)
    with pytest.raises(ValueError, match="already exists"):
        scheduler.submit([2], max_new_tokens=1, request_id=7)
    scheduler.admit_pending()
    with pytest.raises(ValueError, match="chunk_size"):
        scheduler.next_prefill_work(chunk_size=0)


def test_qwen35_batch_serial_bench_helpers_summarize_and_slice(tmp_path) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"prompt_ids": list(range(12))}))

    assert _load_prompt_slices(fixture, prompt_length=3, batch_size=4) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ]
    with pytest.raises(ValueError, match="need at least"):
        _load_prompt_slices(fixture, prompt_length=4, batch_size=4)

    empty = _summarize_samples([])
    assert empty["median"] is None
    stats = _summarize_samples([3.0, 1.0, 2.0, 10.0])
    assert stats["samples"] == [3.0, 1.0, 2.0, 10.0]
    assert stats["median"] == 2.5
    assert stats["p95"] == 10.0
    assert stats["min"] == 1.0
    assert stats["max"] == 10.0
    assert stats["stdev"] > 0.0


def test_qwen35_batch_diagnostic_artifact_schema_requires_label_fields() -> None:
    payload = {
        "status": "blocked",
        "performance_claim": False,
        "workload": {
            "native_compact_prefill": True,
            "native_caware_decode": False,
        },
        "correctness": {"passed": True},
        "execution": {
            "batch_execution": {
                "native_compact_prefill": True,
                "native_caware_decode": False,
                "throughput_claim_eligible": False,
            }
        },
        "decision": {"accepted": False},
    }

    validate_cn_diagnostic_artifact_payload(payload)

    missing = dict(payload)
    missing["execution"] = {"batch_execution": {"native_compact_prefill": True}}

    with pytest.raises(ValueError, match="native_caware_decode"):
        validate_cn_diagnostic_artifact_payload(missing)


def test_qwen35_retained_scaling_comparison_uses_c1_and_serial_artifacts(tmp_path: Path) -> None:
    c1 = tmp_path / "native-baseline-c1.json"
    c1.write_text(
        json.dumps(
            {
                "run_tag": "c1",
                "prompt_length": 512,
                "decode_tokens": 128,
                "throughput": {"warmed_decode_tok_s": 5.0},
            }
        )
    )
    serial = tmp_path / "serial-bridge-c2.json"
    serial.write_text(
        json.dumps(
            {
                "run_tag": "serial-c2",
                "status": "blocked",
                "workload": {"concurrency": 2, "prompt_tokens_per_request": 512, "gen_tokens_per_request": 128},
                "measurements": {
                    "decode_tok_s_aggregate": 8.0,
                    "decode_tok_s_per_request": 4.0,
                },
            }
        )
    )
    args = argparse.Namespace(c1_baseline_json=c1, serial_bridge_json=serial)

    scaling = retained_bench._build_scaling_comparison(
        args,
        native_decode_tok_s_aggregate=16.0,
        native_decode_tok_s_per_request=8.0,
    )

    assert scaling["complete"] is True
    assert scaling["c1_baseline"]["status"] == "loaded"
    assert scaling["c1_baseline"]["reason"] is None
    assert scaling["c1_baseline"]["workload_concurrency"] == 1
    assert scaling["c1_baseline"]["prompt_tokens_per_request"] == 512
    assert scaling["c1_baseline"]["gen_tokens_per_request"] == 128
    assert scaling["c1_baseline"]["decode_tok_s_aggregate"] == 5.0
    assert scaling["serial_bridge_baseline"]["status"] == "blocked"
    assert scaling["serial_bridge_baseline"]["reason"] is None
    assert scaling["serial_bridge_baseline"]["workload_concurrency"] == 2
    assert scaling["serial_bridge_baseline"]["prompt_tokens_per_request"] == 512
    assert scaling["serial_bridge_baseline"]["gen_tokens_per_request"] == 128
    assert scaling["serial_bridge_baseline"]["decode_tok_s_per_request"] == 4.0
    assert scaling["ratios"] == {
        "aggregate_vs_c1": 16.0 / 5.0,
        "per_request_vs_c1": 8.0 / 5.0,
        "aggregate_vs_serial_bridge": 2.0,
        "per_request_vs_serial_bridge": 2.0,
    }


def test_qwen35_retained_primitive_correctness_reference_requires_same_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "primitive-c2.json"
    artifact.write_text(
        json.dumps(
            {
                "rows": 2,
                "passed": True,
                "append_key_mismatch": 0,
                "append_value_mismatch": 0,
                "attn_batch_vs_c1_max_abs": 0.0,
                "attn_batch_vs_numpy_max_abs": 5.0e-8,
            }
        )
    )

    passed = retained_bench._primitive_correctness_reference(artifact, rows=2)
    mismatched = retained_bench._primitive_correctness_reference(artifact, rows=4)
    missing = retained_bench._primitive_correctness_reference(None, rows=2)

    assert passed["passed"] is True
    assert passed["artifact_path"] == str(artifact)
    assert mismatched["passed"] is False
    assert "does not match batch_size=4" in mismatched["reason"]
    assert missing["status"] == "missing"


def test_qwen35_retained_payload_mirrors_fallback_native_decode_label(monkeypatch) -> None:
    monkeypatch.setattr(retained_bench, "_hardware_context", lambda: {"gpu": "test"})
    monkeypatch.setattr(retained_bench, "_software_context", lambda: {"python": "test"})
    args = argparse.Namespace(
        batch_size=2,
        prompt_length=512,
        decode_tokens=128,
        warmup_decode_tokens=0,
        max_layers=40,
        model="/tmp/model",
        kv_storage="bf16",
        kv_scale_dtype="fp16",
        kv_scale_granularity="per_token_head",
    )
    bench = {
        "load_seconds": 0.1,
        "prefill_seconds": 1.0,
        "warmup_seconds": 0.0,
        "decode_seconds": 2.0,
        "warmup_step_seconds": [],
        "decode_step_seconds": [0.25, 0.5],
        "seed_tokens": {
            "0": {"token_id": 10, "token_text": "a", "logit": 1.0},
            "1": {"token_id": 20, "token_text": "b", "logit": 1.0},
        },
        "generated_tokens": {"0": [], "1": []},
        "scheduler_metadata": {},
        "batch_execution": {
            "path": "scheduler_native_compact_batch",
            "scheduler_owned": True,
            "row_execution": "native_linear_batch_with_per_row_full_attention_fallback",
            "native_prefill_plan": {"full_layer_limit_native": True},
            "native_compact_prefill": True,
            "native_caware_decode": False,
            "throughput_claim_eligible": False,
            "blockers": ["full-attention decode used a per-row fallback"],
            "decode_execution": {
                "full_attention_decode_path": "per_row_splitk_fallback",
                "native_caware_decode": False,
            },
        },
        "completed": [],
        "request_observability": {},
        "finite_logits": True,
    }

    payload = retained_bench._build_payload(
        args,
        ["--batch-size", "2"],
        bench,
        [512, 512],
        {"passed": True, "skipped": False, "batch_sequences": [[10], [20]], "c1_sequences": [[10], [20]], "mismatches": []},
    )

    assert payload["status"] == "blocked"
    assert payload["performance_claim"] is False
    assert payload["workload"]["native_caware_decode"] is False
    assert payload["execution"]["batch_execution"]["native_caware_decode"] is False
    assert payload["execution"]["batch_execution"]["decode_execution"]["full_attention_decode_path"] == "per_row_splitk_fallback"


def test_qwen35_batch_diagnostic_artifact_schema_enforces_accepted_row_gates() -> None:
    accepted = {
        "status": "accepted",
        "performance_claim": True,
        "hardware": {"gpu": "AMD Radeon Pro W7900", "arch": "gfx1100"},
        "software": {"hipengine_commit": "abc1234", "hipengine_dirty": False},
        "commands": {
            "benchmark": "python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --json accepted.json",
            "correctness_reference": "inline generated-token equality vs independent c=1",
            "profiler": "rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-profile -- python3 scripts/qwen35_batch_retained_bench.py ...",
        },
        "profiler": {"status": "captured", "expected_kernels_present": True},
        "workload": {
            "concurrency": 2,
            "prompt_tokens_per_request": 512,
            "gen_tokens_per_request": 128,
            "native_compact_prefill": True,
            "native_caware_decode": True,
        },
        "correctness": {
            "passed": True,
            "generated_token_equality": {
                "passed": True,
                "skipped": False,
                "batch_sequences": [[10, 11], [20, 21]],
                "c1_sequences": [[10, 11], [20, 21]],
                "mismatches": [],
            },
            "primitive_batch_correctness": {
                "artifact_path": "benchmarks/results/primitive-c2.json",
                "rows": 2,
                "passed": True,
                "append_key_mismatch": 0,
                "append_value_mismatch": 0,
                "attn_batch_vs_c1_max_abs": 0.0,
            },
        },
        "execution": {
            "batch_execution": {
                "path": "scheduler_native_compact_batch",
                "row_execution": "native_compact_caware_layers",
                "native_compact_prefill": True,
                "native_caware_decode": True,
                "throughput_claim_eligible": True,
                "decode_execution": {
                    "full_attention_decode_path": "native_batch",
                    "native_caware_decode": True,
                    "sampler_execution": {"native_row_aware_lm_head": True},
                },
            }
        },
        "observability": {
            "admission_timestamps": {"0": 1.0, "1": 1.1},
            "completion_timestamps": {"0": 2.0, "1": 2.2},
            "request_latency_seconds": {"p50": 1.0, "p95": 1.25},
            "per_request": {
                "0": {
                    "queue_seconds": 0.1,
                    "prefill_seconds": 0.2,
                    "decode_seconds": 0.3,
                    "kv_pages_owned": 2,
                    "kv_pages_peak": 3,
                    "bucket_key": "decode:c=2:ctx=512:mask=11",
                    "admission_blocked_reason": None,
                    "finish_reason": "length",
                },
                "1": {
                    "queue_seconds": 0.15,
                    "prefill_seconds": 0.25,
                    "decode_seconds": 0.35,
                    "kv_pages_owned": 2,
                    "kv_pages_peak": 3,
                    "bucket_key": "decode:c=2:ctx=512:mask=11",
                    "admission_blocked_reason": None,
                    "finish_reason": "length",
                },
            },
        },
        "measurements": {
            "decode_seconds": 1.0,
            "decode_tok_s_aggregate": 100.0,
            "decode_tok_s_per_request": 50.0,
            "decode_step_seconds": {
                "samples": [0.1, 0.2, 0.3],
                "median": 0.2,
                "p95": 0.3,
                "min": 0.1,
                "max": 0.3,
                "stdev": 0.1,
            },
        },
        "scaling": {
            "complete": True,
            "native": {
                "decode_tok_s_aggregate": 100.0,
                "decode_tok_s_per_request": 50.0,
            },
            "c1_baseline": {
                "artifact_path": "benchmarks/results/c1.json",
                "status": "loaded",
                "reason": None,
                "workload_concurrency": 1,
                "prompt_tokens_per_request": 512,
                "gen_tokens_per_request": 128,
                "decode_tok_s_aggregate": 60.0,
                "decode_tok_s_per_request": 60.0,
            },
            "serial_bridge_baseline": {
                "artifact_path": "benchmarks/results/serial-c2.json",
                "status": "blocked",
                "reason": None,
                "workload_concurrency": 2,
                "prompt_tokens_per_request": 512,
                "gen_tokens_per_request": 128,
                "decode_tok_s_aggregate": 80.0,
                "decode_tok_s_per_request": 40.0,
            },
            "ratios": {
                "aggregate_vs_c1": 100.0 / 60.0,
                "per_request_vs_c1": 50.0 / 60.0,
                "aggregate_vs_serial_bridge": 1.25,
                "per_request_vs_serial_bridge": 1.25,
            },
        },
        "memory": {
            "dynamic_pool": {
                "evidence": "initial chunk sufficed",
                "grow_events": 0,
                "shrink_events": 0,
                "pool_counters": {
                    "current_bytes": 8192,
                    "high_water_observed_bytes": 8192,
                    "grow_events": 0,
                    "grow_failures": 0,
                    "shrink_events": 0,
                    "free_pages": 2,
                    "refcounted_pages": 0,
                },
            },
            "stable_block_id": {"passed": True, "audit": "debug check passed"},
            "prefix_sharing": {"enabled": False, "savings_bytes": 0},
        },
        "decision": {"accepted": True},
    }

    validate_cn_diagnostic_artifact_payload(accepted)

    missing_equality = json.loads(json.dumps(accepted))
    missing_equality["correctness"].pop("generated_token_equality")
    with pytest.raises(ValueError, match="generated_token_equality"):
        validate_cn_diagnostic_artifact_payload(missing_equality)

    skipped_equality = json.loads(json.dumps(accepted))
    skipped_equality["correctness"]["generated_token_equality"]["skipped"] = True
    with pytest.raises(ValueError, match="skipped must be false"):
        validate_cn_diagnostic_artifact_payload(skipped_equality)

    mismatch_equality = json.loads(json.dumps(accepted))
    mismatch_equality["correctness"]["generated_token_equality"]["mismatches"] = [{"row": 0}]
    with pytest.raises(ValueError, match="mismatches must be empty"):
        validate_cn_diagnostic_artifact_payload(mismatch_equality)

    mislabeled_equality = json.loads(json.dumps(accepted))
    mislabeled_equality["correctness"]["generated_token_equality"]["c1_sequences"][1][0] = 99
    with pytest.raises(ValueError, match="batch_sequences must equal c1_sequences"):
        validate_cn_diagnostic_artifact_payload(mislabeled_equality)

    short_equality = json.loads(json.dumps(accepted))
    short_equality["correctness"]["generated_token_equality"]["batch_sequences"] = [[10, 11]]
    with pytest.raises(ValueError, match="batch_sequences length must match workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(short_equality)

    missing_primitive = json.loads(json.dumps(accepted))
    missing_primitive["correctness"].pop("primitive_batch_correctness")
    with pytest.raises(ValueError, match="primitive_batch_correctness"):
        validate_cn_diagnostic_artifact_payload(missing_primitive)

    failed_primitive = json.loads(json.dumps(accepted))
    failed_primitive["correctness"]["primitive_batch_correctness"]["passed"] = False
    with pytest.raises(ValueError, match="primitive_batch_correctness.passed"):
        validate_cn_diagnostic_artifact_payload(failed_primitive)

    mismatched_primitive_rows = json.loads(json.dumps(accepted))
    mismatched_primitive_rows["correctness"]["primitive_batch_correctness"]["rows"] = 8
    with pytest.raises(ValueError, match="primitive_batch_correctness.rows must match workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(mismatched_primitive_rows)

    missing_workload_concurrency = json.loads(json.dumps(accepted))
    missing_workload_concurrency["workload"].pop("concurrency")
    with pytest.raises(ValueError, match="workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(missing_workload_concurrency)

    serial_bridge_execution = json.loads(json.dumps(accepted))
    serial_bridge_execution["execution"]["batch_execution"]["path"] = "scheduler_serial_slot_bridge"
    with pytest.raises(ValueError, match="serial bridge"):
        validate_cn_diagnostic_artifact_payload(serial_bridge_execution)

    fallback_execution = json.loads(json.dumps(accepted))
    fallback_execution["execution"]["batch_execution"]["row_execution"] = "native_linear_batch_with_per_row_full_attention_fallback"
    with pytest.raises(ValueError, match="serial or fallback"):
        validate_cn_diagnostic_artifact_payload(fallback_execution)

    non_native_decode = json.loads(json.dumps(accepted))
    non_native_decode["execution"]["batch_execution"]["native_caware_decode"] = False
    with pytest.raises(ValueError, match="native_caware_decode"):
        validate_cn_diagnostic_artifact_payload(non_native_decode)

    per_row_splitk = json.loads(json.dumps(accepted))
    per_row_splitk["execution"]["batch_execution"]["decode_execution"]["full_attention_decode_path"] = "per_row_splitk_fallback"
    with pytest.raises(ValueError, match="per-row fallback"):
        validate_cn_diagnostic_artifact_payload(per_row_splitk)

    serial_sampler = json.loads(json.dumps(accepted))
    serial_sampler["execution"]["batch_execution"]["decode_execution"]["sampler_execution"]["native_row_aware_lm_head"] = False
    with pytest.raises(ValueError, match="native_row_aware_lm_head"):
        validate_cn_diagnostic_artifact_payload(serial_sampler)

    missing_latency = dict(accepted)
    missing_latency["observability"] = {
        "admission_timestamps": {"0": 1.0},
        "completion_timestamps": {"0": 2.0},
    }
    with pytest.raises(ValueError, match="request_latency_seconds"):
        validate_cn_diagnostic_artifact_payload(missing_latency)

    missing_per_request = dict(accepted)
    missing_per_request["observability"] = {
        "admission_timestamps": {"0": 1.0, "1": 1.1},
        "completion_timestamps": {"0": 2.0, "1": 2.2},
        "request_latency_seconds": {"p50": 1.0, "p95": 1.25},
    }
    with pytest.raises(ValueError, match="per_request"):
        validate_cn_diagnostic_artifact_payload(missing_per_request)

    short_admission_timestamps = json.loads(json.dumps(accepted))
    short_admission_timestamps["observability"]["admission_timestamps"].pop("1")
    with pytest.raises(ValueError, match="admission_timestamps length must match workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(short_admission_timestamps)

    short_completion_timestamps = json.loads(json.dumps(accepted))
    short_completion_timestamps["observability"]["completion_timestamps"].pop("1")
    with pytest.raises(ValueError, match="completion_timestamps length must match workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(short_completion_timestamps)

    short_per_request = json.loads(json.dumps(accepted))
    short_per_request["observability"]["per_request"].pop("1")
    with pytest.raises(ValueError, match="per_request length must match workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(short_per_request)

    missing_pool = dict(accepted)
    missing_pool["memory"] = {
        "dynamic_pool": {"evidence": "initial chunk sufficed"},
        "prefix_sharing": {"enabled": False, "savings_bytes": 0},
    }
    with pytest.raises(ValueError, match="stable_block_id|pool_counters"):
        validate_cn_diagnostic_artifact_payload(missing_pool)

    missing_scaling = dict(accepted)
    missing_scaling.pop("scaling")
    with pytest.raises(ValueError, match="scaling"):
        validate_cn_diagnostic_artifact_payload(missing_scaling)

    for ratio_field in (
        "aggregate_vs_c1",
        "per_request_vs_c1",
        "aggregate_vs_serial_bridge",
        "per_request_vs_serial_bridge",
    ):
        missing_ratio = json.loads(json.dumps(accepted))
        missing_ratio["scaling"]["ratios"].pop(ratio_field)
        with pytest.raises(ValueError, match=ratio_field):
            validate_cn_diagnostic_artifact_payload(missing_ratio)

    inconsistent_ratio = json.loads(json.dumps(accepted))
    inconsistent_ratio["scaling"]["ratios"]["aggregate_vs_serial_bridge"] = 99.0
    with pytest.raises(ValueError, match="aggregate_vs_serial_bridge must match scaling throughput fields"):
        validate_cn_diagnostic_artifact_payload(inconsistent_ratio)

    missing_c1_status = json.loads(json.dumps(accepted))
    missing_c1_status["scaling"]["c1_baseline"].pop("status")
    with pytest.raises(ValueError, match="c1_baseline.status"):
        validate_cn_diagnostic_artifact_payload(missing_c1_status)

    failed_c1_status = json.loads(json.dumps(accepted))
    failed_c1_status["scaling"]["c1_baseline"]["status"] = "missing"
    with pytest.raises(ValueError, match="c1_baseline.status must be usable"):
        validate_cn_diagnostic_artifact_payload(failed_c1_status)

    missing_c1_concurrency = json.loads(json.dumps(accepted))
    missing_c1_concurrency["scaling"]["c1_baseline"].pop("workload_concurrency")
    with pytest.raises(ValueError, match="c1_baseline.workload_concurrency"):
        validate_cn_diagnostic_artifact_payload(missing_c1_concurrency)

    mismatched_c1_concurrency = json.loads(json.dumps(accepted))
    mismatched_c1_concurrency["scaling"]["c1_baseline"]["workload_concurrency"] = 2
    with pytest.raises(ValueError, match="c1_baseline.workload_concurrency must be 1"):
        validate_cn_diagnostic_artifact_payload(mismatched_c1_concurrency)

    missing_serial_concurrency = json.loads(json.dumps(accepted))
    missing_serial_concurrency["scaling"]["serial_bridge_baseline"].pop("workload_concurrency")
    with pytest.raises(ValueError, match="serial_bridge_baseline.workload_concurrency"):
        validate_cn_diagnostic_artifact_payload(missing_serial_concurrency)

    mismatched_serial_concurrency = json.loads(json.dumps(accepted))
    mismatched_serial_concurrency["scaling"]["serial_bridge_baseline"]["workload_concurrency"] = 8
    with pytest.raises(ValueError, match="serial_bridge_baseline.workload_concurrency must match workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(mismatched_serial_concurrency)

    mismatched_c1_prompt_shape = json.loads(json.dumps(accepted))
    mismatched_c1_prompt_shape["scaling"]["c1_baseline"]["prompt_tokens_per_request"] = 256
    with pytest.raises(ValueError, match="c1_baseline.prompt_tokens_per_request must match workload.prompt_tokens_per_request"):
        validate_cn_diagnostic_artifact_payload(mismatched_c1_prompt_shape)

    missing_serial_decode_shape = json.loads(json.dumps(accepted))
    missing_serial_decode_shape["scaling"]["serial_bridge_baseline"].pop("gen_tokens_per_request")
    with pytest.raises(ValueError, match="serial_bridge_baseline.gen_tokens_per_request"):
        validate_cn_diagnostic_artifact_payload(missing_serial_decode_shape)

    serial_baseline_reason = json.loads(json.dumps(accepted))
    serial_baseline_reason["scaling"]["serial_bridge_baseline"]["reason"] = "decode throughput fields missing"
    with pytest.raises(ValueError, match="serial_bridge_baseline.reason"):
        validate_cn_diagnostic_artifact_payload(serial_baseline_reason)

    missing_measurements = dict(accepted)
    missing_measurements.pop("measurements")
    with pytest.raises(ValueError, match="measurements"):
        validate_cn_diagnostic_artifact_payload(missing_measurements)

    missing_decode_rate = json.loads(json.dumps(accepted))
    missing_decode_rate["measurements"].pop("decode_tok_s_per_request")
    with pytest.raises(ValueError, match="decode_tok_s_per_request"):
        validate_cn_diagnostic_artifact_payload(missing_decode_rate)

    zero_decode_rate = json.loads(json.dumps(accepted))
    zero_decode_rate["measurements"]["decode_tok_s_aggregate"] = 0.0
    with pytest.raises(ValueError, match="measurements.decode_tok_s_aggregate must be positive numeric"):
        validate_cn_diagnostic_artifact_payload(zero_decode_rate)

    zero_baseline_rate = json.loads(json.dumps(accepted))
    zero_baseline_rate["scaling"]["c1_baseline"]["decode_tok_s_per_request"] = 0.0
    with pytest.raises(ValueError, match="c1_baseline.decode_tok_s_per_request must be positive numeric"):
        validate_cn_diagnostic_artifact_payload(zero_baseline_rate)

    empty_samples = json.loads(json.dumps(accepted))
    empty_samples["measurements"]["decode_step_seconds"]["samples"] = []
    with pytest.raises(ValueError, match="samples"):
        validate_cn_diagnostic_artifact_payload(empty_samples)

    zero_sample = json.loads(json.dumps(accepted))
    zero_sample["measurements"]["decode_step_seconds"]["samples"][0] = 0.0
    with pytest.raises(ValueError, match="samples must contain only positive numbers"):
        validate_cn_diagnostic_artifact_payload(zero_sample)

    missing_command = json.loads(json.dumps(accepted))
    missing_command["commands"]["benchmark"] = ""
    with pytest.raises(ValueError, match="commands.benchmark"):
        validate_cn_diagnostic_artifact_payload(missing_command)

    missing_correctness_command = json.loads(json.dumps(accepted))
    missing_correctness_command["commands"]["correctness_reference"] = ""
    with pytest.raises(ValueError, match="commands.correctness_reference"):
        validate_cn_diagnostic_artifact_payload(missing_correctness_command)

    missing_profiler = dict(accepted)
    missing_profiler.pop("profiler")
    with pytest.raises(ValueError, match="profiler"):
        validate_cn_diagnostic_artifact_payload(missing_profiler)

    missing_profiler_command = json.loads(json.dumps(accepted))
    missing_profiler_command["commands"]["profiler"] = None
    with pytest.raises(ValueError, match="commands.profiler"):
        validate_cn_diagnostic_artifact_payload(missing_profiler_command)

    not_captured_profiler = json.loads(json.dumps(accepted))
    not_captured_profiler["profiler"]["status"] = "not_captured"
    with pytest.raises(ValueError, match="profiler.status"):
        validate_cn_diagnostic_artifact_payload(not_captured_profiler)

    missing_expected_kernel = json.loads(json.dumps(accepted))
    missing_expected_kernel["profiler"]["expected_kernels_present"] = False
    with pytest.raises(ValueError, match="expected_kernels_present"):
        validate_cn_diagnostic_artifact_payload(missing_expected_kernel)

    empty_hardware = json.loads(json.dumps(accepted))
    empty_hardware["hardware"] = {}
    with pytest.raises(ValueError, match="hardware.gpu|hardware.arch"):
        validate_cn_diagnostic_artifact_payload(empty_hardware)

    for hardware_field in ("gpu", "arch"):
        missing_hardware_field = json.loads(json.dumps(accepted))
        missing_hardware_field["hardware"].pop(hardware_field)
        with pytest.raises(ValueError, match=f"hardware.{hardware_field}"):
            validate_cn_diagnostic_artifact_payload(missing_hardware_field)

    missing_dirty_state = json.loads(json.dumps(accepted))
    missing_dirty_state["software"].pop("hipengine_dirty")
    with pytest.raises(ValueError, match="hipengine_dirty"):
        validate_cn_diagnostic_artifact_payload(missing_dirty_state)

    incomplete_scaling = dict(accepted)
    incomplete_scaling["scaling"] = {"complete": False}
    with pytest.raises(ValueError, match="scaling.complete|scaling.native"):
        validate_cn_diagnostic_artifact_payload(incomplete_scaling)

    inconsistent = dict(accepted)
    inconsistent["status"] = "blocked"
    with pytest.raises(ValueError, match="status='accepted'"):
        validate_cn_diagnostic_artifact_payload(inconsistent)


def test_qwen35_batch_diagnostic_artifact_schema_rejects_missing_correctness() -> None:
    payload = {
        "status": "blocked",
        "performance_claim": False,
        "workload": {
            "native_compact_prefill": False,
            "native_caware_decode": False,
        },
        "execution": {
            "batch_execution": {
                "native_compact_prefill": False,
                "native_caware_decode": False,
                "throughput_claim_eligible": False,
            }
        },
        "decision": {"accepted": False},
    }

    with pytest.raises(ValueError, match="correctness"):
        validate_cn_diagnostic_artifact_payload(payload)
