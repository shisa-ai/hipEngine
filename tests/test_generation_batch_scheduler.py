from __future__ import annotations

import argparse
import json
import shlex
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
    plan_projection_dispatch_from_artifact,
    projection_dispatch_candidates_from_artifact,
    projection_dispatch_candidates_from_json,
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
from scripts.qwen35_batch_artifact_schema import (
    main as validate_cn_diagnostic_artifact_main,
    validate_cn_diagnostic_artifact_payload,
    validate_cn_diagnostic_rollup_evidence,
    validate_cn_diagnostic_validation_summary,
)
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


def _write_c_sweep_profiler_summary(
    output_dir: Path,
    *,
    rows: int = 2,
    model: str = "/tmp/model",
    fixture: str = "/tmp/fixture.json",
    warmup_decode_tokens: int = 8,
    max_layers: int = 40,
) -> None:
    profiler_path = output_dir / f"profiler-c{rows}.json"
    retained_path = output_dir / f"native-diagnostic-c{rows}.json"
    trace_dir = output_dir / f"profile-c{rows}"
    profiler_path.write_text(
        json.dumps(
            {
                "workload": {"concurrency": rows, "prompt_tokens_per_request": 16, "gen_tokens_per_request": 2},
                "profiler": {
                    "artifact_path": str(profiler_path),
                    "rows": rows,
                    "status": "captured",
                    "output_format": "csv",
                    "trace_dir": str(trace_dir),
                    "trace_files": [str(trace_dir / "hipengine_kernel_trace.csv")],
                    "trace_kernel_names": ["qwen35_batch_decode"],
                    "command": (
                        f"rocprofv3 --kernel-trace --output-format csv -d {trace_dir} -- python3 scripts/qwen35_batch_retained_bench.py "
                        f"--model {model} --fixture {fixture} --batch-size {rows} "
                        "--prompt-length 16 --decode-tokens 2 "
                        f"--warmup-decode-tokens {warmup_decode_tokens} --max-layers {max_layers} "
                        f"--json {retained_path} --c1-baseline-json {output_dir / 'native-baseline-c1.json'} "
                        f"--serial-bridge-json {output_dir / f'serial-bridge-c{rows}.json'} "
                        f"--primitive-correctness-json {output_dir / f'primitive-c{rows}.json'} "
                        f"--profiler-json {profiler_path}"
                    ),
                    "expected_kernels_present": True,
                    "expected_kernel_names": ["qwen35_batch_decode"],
                    "kernel_durations_ns": {"qwen35_batch_decode": 12345.0},
                    "total_kernel_duration_ns": 12345.0,
                    "kernel_duration_shares": {"qwen35_batch_decode": 1.0},
                    "kernel_duration_categories_ns": {
                        "attention": 0.0,
                        "moe": 0.0,
                        "projection": 0.0,
                        "sampling": 0.0,
                        "graph_replay": 0.0,
                        "other": 12345.0,
                    },
                    "kernel_duration_category_shares": {
                        "attention": 0.0,
                        "moe": 0.0,
                        "projection": 0.0,
                        "sampling": 0.0,
                        "graph_replay": 0.0,
                        "other": 1.0,
                    },
                    "cpu_side_total_seconds": 10.0,
                    "cpu_side_bottlenecks_seconds": {
                        "load": 1.0,
                        "prefill": 2.0,
                        "warmup_decode": 0.0,
                        "decode": 7.0,
                        "validation": 0.0,
                        "other": 0.0,
                    },
                    "cpu_side_bottleneck_shares": {
                        "load": 0.1,
                        "prefill": 0.2,
                        "warmup_decode": 0.0,
                        "decode": 0.7,
                        "validation": 0.0,
                        "other": 0.0,
                    },
                },
            }
        )
    )


def test_batch_c_sweep_profiler_precondition_rejects_mismatched_artifact_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    profiler_path = output_dir / "profiler-c2.json"
    profiler_path.write_text(
        json.dumps(
            {
                "workload": {"concurrency": 2, "prompt_tokens_per_request": 16, "gen_tokens_per_request": 2},
                "profiler": {
                    "artifact_path": str(output_dir / "profiler-c4.json"),
                    "rows": 2,
                    "status": "captured",
                    "output_format": "csv",
                    "trace_dir": str(output_dir / "profile-c2"),
                    "trace_files": [str(output_dir / "profile-c2" / "hipengine_kernel_trace.csv")],
                    "trace_kernel_names": ["qwen35_batch_decode"],
                    "command": (
                        f"rocprofv3 --kernel-trace --output-format csv -d {output_dir / 'profile-c2'} -- python3 scripts/qwen35_batch_retained_bench.py "
                        "--model /tmp/model --fixture /tmp/fixture.json --batch-size 2 "
                        "--prompt-length 16 --decode-tokens 2 "
                        f"--warmup-decode-tokens 8 --max-layers 40 --json {output_dir / 'native-diagnostic-c2.json'} "
                        f"--c1-baseline-json {output_dir / 'native-baseline-c1.json'} "
                        f"--serial-bridge-json {output_dir / 'serial-bridge-c2.json'} "
                        f"--primitive-correctness-json {output_dir / 'primitive-c2.json'} "
                        f"--profiler-json {profiler_path}"
                    ),
                    "expected_kernels_present": True,
                    "expected_kernel_names": ["qwen35_batch_decode"],
                    "kernel_durations_ns": {"qwen35_batch_decode": 12345.0},
                    "total_kernel_duration_ns": 12345.0,
                    "kernel_duration_shares": {"qwen35_batch_decode": 1.0},
                    "kernel_duration_categories_ns": {
                        "attention": 0.0,
                        "moe": 0.0,
                        "projection": 0.0,
                        "sampling": 0.0,
                        "graph_replay": 0.0,
                        "other": 12345.0,
                    },
                    "kernel_duration_category_shares": {
                        "attention": 0.0,
                        "moe": 0.0,
                        "projection": 0.0,
                        "sampling": 0.0,
                        "graph_replay": 0.0,
                        "other": 1.0,
                    },
                    "cpu_side_total_seconds": 10.0,
                    "cpu_side_bottlenecks_seconds": {
                        "load": 1.0,
                        "prefill": 2.0,
                        "warmup_decode": 0.0,
                        "decode": 7.0,
                        "validation": 0.0,
                        "other": 0.0,
                    },
                    "cpu_side_bottleneck_shares": {
                        "load": 0.1,
                        "prefill": 0.2,
                        "warmup_decode": 0.0,
                        "decode": 0.7,
                        "validation": 0.0,
                        "other": 0.0,
                    },
                },
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
        ]
    )
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "artifact_path does not match --profiler-json path",
    }


def test_batch_c_sweep_profiler_precondition_rejects_wrong_row_count(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"]["rows"] = 4
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "rows=4 does not match batch_size=2",
    }


def test_batch_c_sweep_profiler_precondition_rejects_wrong_shape(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["workload"]["prompt_tokens_per_request"] = 32
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "prompt_tokens_per_request=32 does not match prompt_length=16",
    }


def test_batch_c_sweep_profiler_precondition_rejects_missing_command(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"].pop("command")
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "profiler command is missing",
    }


def test_batch_c_sweep_profiler_precondition_rejects_missing_trace_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"].pop("trace_dir")
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "profiler.trace_dir is missing",
    }


def test_batch_c_sweep_profiler_precondition_rejects_missing_trace_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"].pop("trace_files")
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "profiler.trace_files is missing or empty",
    }


def test_batch_c_sweep_profiler_precondition_rejects_trace_files_outside_trace_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"]["trace_files"] = [str(output_dir / "other-profile" / "hipengine_kernel_trace.csv")]
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "profiler.trace_files contains a path outside profiler.trace_dir",
    }


def test_batch_c_sweep_profiler_precondition_rejects_trace_files_without_kernel_trace_csv(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"]["trace_files"] = [str(output_dir / "profile-c2" / "hipengine_api_trace.csv")]
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "profiler.trace_files does not include a kernel-trace CSV",
    }


def test_batch_c_sweep_profiler_precondition_synthesizes_trace_fields_from_csv(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
    trace_dir = output_dir / "profile-c2"
    trace_dir.mkdir()
    (trace_dir / "hipengine_kernel_trace.csv").write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp\n"
        "qwen35_batch_decode,0,100\n"
        "qwen35_batch_decode,100,150\n"
        "qwen35_batch_decode_wmma_caware,150,350\n"
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    for field in (
        "trace_kernel_names",
        "kernel_durations_ns",
        "total_kernel_duration_ns",
        "kernel_duration_shares",
        "kernel_duration_categories_ns",
        "kernel_duration_category_shares",
    ):
        payload["profiler"].pop(field)
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition["passed"] is True
    assert precondition["profiler_trace_kernel_names"] == ["qwen35_batch_decode", "qwen35_batch_decode_wmma_caware"]
    assert precondition["profiler_trace_synthesized_fields"] == [
        "trace_kernel_names",
        "kernel_durations_ns",
        "total_kernel_duration_ns",
        "kernel_duration_shares",
        "kernel_duration_categories_ns",
        "kernel_duration_category_shares",
    ]
    assert precondition["kernel_durations_ns"] == {"qwen35_batch_decode": 150.0, "qwen35_batch_decode_wmma_caware": 200.0}
    assert precondition["total_kernel_duration_ns"] == 350.0
    assert precondition["kernel_duration_categories_ns"] == {
        "attention": 0.0,
        "moe": 0.0,
        "projection": 200.0,
        "sampling": 0.0,
        "graph_replay": 0.0,
        "other": 150.0,
    }


def test_batch_c_sweep_profiler_precondition_rejects_missing_trace_kernel_names(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"].pop("trace_kernel_names")
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "profiler.trace_kernel_names is missing or empty",
    }


def test_batch_c_sweep_profiler_precondition_rejects_trace_kernel_names_missing_duration_key(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"]["trace_kernel_names"] = ["qwen35_batch_prefill"]
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "profiler.trace_kernel_names must include kernel_durations_ns keys",
    }


def test_batch_c_sweep_profiler_precondition_rejects_missing_command_trace_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"]["command"] = payload["profiler"]["command"].replace(f" -d {output_dir / 'profile-c2'}", "")
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "profiler command is missing -d <trace_dir>",
    }


def test_batch_c_sweep_profiler_precondition_rejects_missing_output_format(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"]["command"] = payload["profiler"]["command"].replace(" --output-format csv", "")
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "profiler command output-format=None does not match 'csv'",
    }


def test_batch_c_sweep_profiler_precondition_rejects_missing_structured_output_format(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"].pop("output_format")
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "profiler.output_format=None does not match 'csv'",
    }


def test_batch_c_sweep_profiler_precondition_rejects_wrong_structured_output_format(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"]["output_format"] = "json"
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "profiler.output_format='json' does not match 'csv'",
    }


def test_batch_c_sweep_profiler_precondition_rejects_wrong_workload_command(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
    args = build_c_sweep_parser().parse_args(
        [
            "--batch-sizes",
            "2",
            "--output-dir",
            str(output_dir),
            "--model",
            "/tmp/other-model",
            "--fixture",
            "/tmp/fixture.json",
            "--prompt-length",
            "16",
            "--decode-tokens",
            "2",
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "profiler command model='/tmp/model' does not match model=/tmp/other-model",
    }


def test_batch_c_sweep_profiler_precondition_rejects_serial_or_fallback_kernel_names(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"]["expected_kernel_names"].append("qwen35_per_row_fallback_decode")
    payload["profiler"]["kernel_durations_ns"]["qwen35_per_row_fallback_decode"] = 12345.0
    payload["profiler"]["total_kernel_duration_ns"] = 24690.0
    payload["profiler"]["kernel_duration_shares"] = {
        "qwen35_batch_decode": 0.5,
        "qwen35_per_row_fallback_decode": 0.5,
    }
    payload["profiler"]["kernel_duration_categories_ns"]["other"] = 24690.0
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": (
            "expected_kernel_names contains a serial/per-row/fallback kernel; "
            "kernel_durations_ns contains a serial/per-row/fallback kernel; "
            "kernel_duration_shares contains a serial/per-row/fallback kernel"
        ),
    }


def test_batch_c_sweep_profiler_precondition_rejects_fallback_kernel_share_key(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"]["kernel_duration_shares"] = {
        "qwen35_batch_decode": 1.0,
        "qwen35_per_row_fallback_decode": 0.0,
    }
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": (
            "kernel_duration_shares keys do not match kernel_durations_ns; "
            "kernel_duration_shares contains a serial/per-row/fallback kernel"
        ),
    }


def test_batch_c_sweep_profiler_precondition_rejects_wrong_retained_artifact_command(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"]["command"] = payload["profiler"]["command"].replace(
        str(output_dir / "native-diagnostic-c2.json"),
        str(output_dir / "native-diagnostic-c4.json"),
    )
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "profiler command --json path does not match retained artifact_path",
    }


def test_batch_c_sweep_profiler_precondition_rejects_wrong_reference_command(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"]["command"] = payload["profiler"]["command"].replace(
        str(output_dir / "serial-bridge-c2.json"),
        str(output_dir / "serial-bridge-c4.json"),
    )
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": (
            f"profiler command --serial-bridge-json='{output_dir / 'serial-bridge-c4.json'}' "
            f"does not match serial_bridge_json={output_dir / 'serial-bridge-c2.json'}"
        ),
    }


def test_batch_c_sweep_profiler_precondition_rejects_missing_cached_build_flags(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
    compiler_version_file = tmp_path / "hipcc-version.txt"
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
            "--compiler-version-file",
            str(compiler_version_file),
            "--require-cached-build",
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": (
            "profiler command compiler-version-file=None "
            f"does not match compiler_version_file={compiler_version_file}; "
            "profiler command is missing --require-cached-build"
        ),
    }


def test_batch_c_sweep_profiler_precondition_rejects_missing_kernel_shares(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"].pop("kernel_duration_shares")
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "kernel_duration_shares is missing or empty",
    }


def test_batch_c_sweep_profiler_precondition_rejects_missing_kernel_categories(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"].pop("kernel_duration_categories_ns")
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "kernel_duration_categories_ns is missing or empty",
    }


def test_batch_c_sweep_profiler_precondition_rejects_category_rows_mismatch(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"]["expected_kernel_names"] = ["qwen35_batch_decode", "qwen35_batch_decode_wmma_caware"]
    payload["profiler"]["trace_kernel_names"] = ["qwen35_batch_decode", "qwen35_batch_decode_wmma_caware"]
    payload["profiler"]["kernel_durations_ns"] = {
        "qwen35_batch_decode": 12345.0,
        "qwen35_batch_decode_wmma_caware": 2345.0,
    }
    payload["profiler"]["total_kernel_duration_ns"] = 14690.0
    payload["profiler"]["kernel_duration_shares"] = {
        "qwen35_batch_decode": 12345.0 / 14690.0,
        "qwen35_batch_decode_wmma_caware": 2345.0 / 14690.0,
    }
    payload["profiler"]["kernel_duration_categories_ns"] = {
        "attention": 0.0,
        "moe": 0.0,
        "projection": 0.0,
        "sampling": 0.0,
        "graph_replay": 0.0,
        "other": 14690.0,
    }
    payload["profiler"]["kernel_duration_category_shares"] = {
        "attention": 0.0,
        "moe": 0.0,
        "projection": 0.0,
        "sampling": 0.0,
        "graph_replay": 0.0,
        "other": 1.0,
    }
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "kernel_duration_categories_ns does not match categorized kernel_durations_ns",
    }


def test_batch_c_sweep_profiler_precondition_rejects_missing_cpu_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, rows=2)
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
        ]
    )
    profiler_path = output_dir / "profiler-c2.json"
    payload = json.loads(profiler_path.read_text())
    payload["profiler"].pop("cpu_side_bottlenecks_seconds")
    profiler_path.write_text(json.dumps(payload))
    native = next(command for command in build_sweep_commands(args) if command.category == "native_diagnostic")

    precondition = c_sweep._profiler_summary_precondition(native)

    assert precondition == {
        "kind": "profiler_summary",
        "artifact_path": str(profiler_path),
        "passed": False,
        "reason": "cpu_side_bottlenecks_seconds is missing or empty",
    }


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
    c_sweep.validate_sweep_summary(persisted)
    tampered_dropped_commands = json.loads(json.dumps(persisted))
    tampered_dropped_commands["commands"] = []
    tampered_dropped_commands["completed_command_count"] = 0
    with pytest.raises(ValueError, match="status_counts must match commands"):
        c_sweep.validate_sweep_summary(tampered_dropped_commands)
    tampered_dry_run_status = json.loads(json.dumps(persisted))
    tampered_dry_run_status["commands"][0]["status"] = "passed"
    with pytest.raises(ValueError, match=r"commands\[\]\.status must be planned for dry-run summaries"):
        c_sweep.validate_sweep_summary(tampered_dry_run_status)
    tampered_planned_duration = json.loads(json.dumps(persisted))
    tampered_planned_duration["commands"][0]["duration_seconds"] = 0.1
    with pytest.raises(ValueError, match=r"commands\[\]\.duration_seconds must be zero for planned rows"):
        c_sweep.validate_sweep_summary(tampered_planned_duration)
    tampered_planned_output = json.loads(json.dumps(persisted))
    tampered_planned_output["commands"][0]["output_tail"] = "dry-run output"
    with pytest.raises(ValueError, match=r"commands\[\]\.output_tail must be absent for planned rows"):
        c_sweep.validate_sweep_summary(tampered_planned_output)
    tampered_planned_condition = json.loads(json.dumps(persisted))
    tampered_planned_condition["commands"][0]["preconditions"] = [
        {"kind": "primitive_correctness", "passed": True}
    ]
    with pytest.raises(ValueError, match=r"commands\[\]\.conditions must be absent for planned rows"):
        c_sweep.validate_sweep_summary(tampered_planned_condition)
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
    assert "--profiler-json" in retained_c2.argv
    assert str(tmp_path / "artifacts" / "profiler-c2.json") in retained_c2.argv


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
    c_sweep.validate_sweep_summary(summary)
    tampered_failed_returncode = json.loads(json.dumps(summary))
    tampered_failed_returncode["commands"][0]["returncode"] = 0
    with pytest.raises(ValueError, match=r"commands\[\]\.status failed with returncode 0 requires a failed postcondition"):
        c_sweep.validate_sweep_summary(tampered_failed_returncode)
    tampered_failed_postconditions = json.loads(json.dumps(summary))
    tampered_failed_postconditions["commands"][0]["postconditions"] = [
        {"kind": "retained_profiler_synthesis", "passed": False}
    ]
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions must be absent for failed rows with nonzero returncode"):
        c_sweep.validate_sweep_summary(tampered_failed_postconditions)
    tampered_stop_on_failure = json.loads(json.dumps(summary))
    serial_command = build_sweep_commands(args)[1]
    tampered_stop_on_failure["commands"].append(
        {
            "category": serial_command.category,
            "batch_size": serial_command.batch_size,
            "command": serial_command.command,
            "argv": list(serial_command.argv),
            "artifact_path": str(serial_command.artifact_path),
            "git_dirty": False,
            "status": "failed",
            "returncode": 1,
            "duration_seconds": 0.1,
            "output_tail": "serial failed\n",
        }
    )
    tampered_stop_on_failure["completed_command_count"] = 2
    tampered_stop_on_failure["status_counts"] = {"failed": 2}
    tampered_stop_on_failure["category_status_counts"] = {"primitive": {"failed": 1}, "serial_bridge": {"failed": 1}}
    with pytest.raises(ValueError, match=r"commands\[\] failed/skipped row must be final"):
        c_sweep.validate_sweep_summary(tampered_stop_on_failure)
    assert len(calls) == 1
    assert calls[0][1] == "scripts/qwen35_batch_correctness.py"


def test_batch_c_sweep_no_stop_counts_failed_and_skipped_rows(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, warmup_decode_tokens=1, max_layers=3)
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
        "profiler_summary": {"passed": 1},
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
    output_dir.mkdir()
    _write_c_sweep_profiler_summary(output_dir, warmup_decode_tokens=1, max_layers=3)
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
        "profiler_summary": {"passed": 1},
    }
    assert [entry["status"] for entry in summary["commands"]] == ["passed", "passed", "skipped"]
    skipped = summary["commands"][-1]
    assert skipped["category"] == "native_diagnostic"
    assert [item["kind"] for item in skipped["preconditions"]] == [
        "primitive_correctness",
        "c1_baseline",
        "serial_bridge",
        "profiler_summary",
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
    c_sweep.validate_sweep_summary(persisted)
    persisted_skipped = persisted["commands"][-1]
    assert persisted_skipped["status"] == "skipped"
    assert persisted_skipped["preconditions"] == skipped["preconditions"]
    assert persisted_skipped["precondition"] == skipped["precondition"]
    assert persisted["retained_precondition_counts"] == summary["retained_precondition_counts"]
    assert persisted["skipped_preconditions"] == summary["skipped_preconditions"]
    tampered_precondition_counts = json.loads(json.dumps(persisted))
    tampered_precondition_counts["retained_precondition_counts"] = {}
    with pytest.raises(ValueError, match="retained_precondition_counts must match commands.preconditions"):
        c_sweep.validate_sweep_summary(tampered_precondition_counts)
    tampered_skipped_preconditions = json.loads(json.dumps(persisted))
    tampered_skipped_preconditions["skipped_preconditions"] = []
    with pytest.raises(ValueError, match="skipped_preconditions must match commands.preconditions"):
        c_sweep.validate_sweep_summary(tampered_skipped_preconditions)
    tampered_dropped_commands = json.loads(json.dumps(persisted))
    tampered_dropped_commands["commands"] = []
    tampered_dropped_commands["completed_command_count"] = 0
    with pytest.raises(ValueError, match="skipped_preconditions must match commands.preconditions"):
        c_sweep.validate_sweep_summary(tampered_dropped_commands)
    tampered_skipped_duration = json.loads(json.dumps(persisted))
    tampered_skipped_duration["commands"][-1]["duration_seconds"] = 0.1
    with pytest.raises(ValueError, match=r"commands\[\]\.duration_seconds must be zero for skipped rows"):
        c_sweep.validate_sweep_summary(tampered_skipped_duration)
    tampered_skipped_postcondition = json.loads(json.dumps(persisted))
    tampered_skipped_postcondition["commands"][-1]["postconditions"] = [
        {"kind": "retained_profiler_synthesis", "passed": True}
    ]
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions must be absent for skipped rows"):
        c_sweep.validate_sweep_summary(tampered_skipped_postcondition)
    tampered_skipped_output_tail = json.loads(json.dumps(persisted))
    tampered_skipped_output_tail["commands"][-1]["output_tail"] = "different failure reason"
    with pytest.raises(ValueError, match=r"commands\[\]\.output_tail must match skipped precondition reason"):
        c_sweep.validate_sweep_summary(tampered_skipped_output_tail)
    tampered_skipped_failed_precondition = json.loads(json.dumps(persisted))
    for precondition in tampered_skipped_failed_precondition["commands"][-1]["preconditions"]:
        precondition["passed"] = True
        precondition["reason"] = None
    tampered_skipped_failed_precondition["commands"][-1]["preconditions"][0].update(
        {
            "primitive_rows": 2,
            "append_key_mismatch": 0,
            "append_value_mismatch": 0,
            "attn_batch_vs_c1_max_abs": 0.0,
        }
    )
    tampered_skipped_failed_precondition["commands"][-1]["preconditions"][1].update(
        {
            "workload_concurrency": 1,
            "prompt_tokens_per_request": 16,
            "gen_tokens_per_request": 2,
            "decode_tok_s_aggregate": 10.0,
            "decode_tok_s_per_request": 10.0,
        }
    )
    tampered_skipped_failed_precondition["commands"][-1]["preconditions"][2].update(
        {
            "workload_concurrency": 2,
            "prompt_tokens_per_request": 16,
            "gen_tokens_per_request": 2,
            "decode_tok_s_aggregate": 20.0,
            "decode_tok_s_per_request": 10.0,
        }
    )
    with pytest.raises(ValueError, match=r"commands\[\]\.precondition must identify a failed precondition"):
        c_sweep.validate_sweep_summary(tampered_skipped_failed_precondition)
    tampered_singular_precondition = json.loads(json.dumps(persisted))
    tampered_singular_precondition["commands"][-1].pop("precondition")
    with pytest.raises(ValueError, match=r"commands\[\]\.precondition must match"):
        c_sweep.validate_sweep_summary(tampered_singular_precondition)


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
    _write_c_sweep_profiler_summary(output_dir, warmup_decode_tokens=1, max_layers=3)
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
        "profiler_summary": {"passed": 1},
    }
    assert summary["retained_precondition_counts"] == expected_counts
    assert [entry["status"] for entry in summary["commands"]] == ["passed", "passed", "skipped"]
    skipped = summary["commands"][-1]
    assert [item["kind"] for item in skipped["preconditions"]] == [
        "primitive_correctness",
        "c1_baseline",
        "serial_bridge",
        "profiler_summary",
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
    _write_c_sweep_profiler_summary(output_dir, warmup_decode_tokens=1, max_layers=3)
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
        "profiler_summary": {"passed": 1},
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
    _write_c_sweep_profiler_summary(output_dir, warmup_decode_tokens=1, max_layers=3)
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
        "profiler_summary": {"passed": 1},
    }


def test_batch_c_sweep_skips_retained_when_profiler_summary_missing(tmp_path: Path, monkeypatch) -> None:
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

    class FakeProc:
        returncode = 0
        stdout = "ok"

    monkeypatch.setattr(c_sweep.subprocess, "run", lambda *args, **kwargs: FakeProc())

    summary = run_sweep(args)

    assert summary["status"] == "blocked"
    assert summary["status_counts"] == {"passed": 2, "skipped": 1}
    skipped = summary["commands"][-1]
    assert skipped["status"] == "skipped"
    assert skipped["precondition"]["kind"] == "profiler_summary"
    assert skipped["precondition"]["reason"] == "profiler summary artifact does not exist"
    assert summary["retained_precondition_counts"] == {
        "primitive_correctness": {"passed": 1},
        "c1_baseline": {"passed": 1},
        "serial_bridge": {"passed": 1},
        "profiler_summary": {"failed": 1},
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
    _write_c_sweep_profiler_summary(output_dir, warmup_decode_tokens=1, max_layers=3)
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
    calls: list[list[str]] = []

    class FakeProc:
        returncode = 0
        stdout = "ok"

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if len(argv) > 1 and argv[1] == "scripts/qwen35_batch_retained_bench.py":
            (output_dir / "native-diagnostic-c2.json").write_text(
                json.dumps({"profiler": {"synthesized_fields": []}})
            )
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
        "profiler_summary": {"passed": 1},
    }
    assert summary["retained_postcondition_counts"] == {"retained_profiler_synthesis": {"passed": 1}}
    assert summary["skipped_preconditions"] == []
    assert summary["failed_postconditions"] == []
    assert [entry["status"] for entry in summary["commands"]] == ["passed", "passed", "passed"]
    assert len(calls) == 3
    assert calls[-1][1] == "scripts/qwen35_batch_retained_bench.py"
    native = summary["commands"][-1]
    assert native["category"] == "native_diagnostic"
    assert [item["kind"] for item in native["preconditions"]] == [
        "primitive_correctness",
        "c1_baseline",
        "serial_bridge",
        "profiler_summary",
    ]
    assert all(item["passed"] is True for item in native["preconditions"])
    assert native["postconditions"] == [
        {
            "kind": "retained_profiler_synthesis",
            "artifact_path": str(output_dir / "native-diagnostic-c2.json"),
            "profiler_precondition_artifact_path": str(output_dir / "profiler-c2.json"),
            "passed": True,
            "reason": None,
            "profiler_synthesized_fields": [],
            "profiler_precondition_synthesized_fields": [],
        }
    ]
    persisted = json.loads(summary_path.read_text())
    c_sweep.validate_sweep_summary(persisted)
    assert c_sweep.main(["--validate-summary-json", str(summary_path)]) == 0
    tampered_timestamp = json.loads(json.dumps(persisted))
    tampered_timestamp["timestamp"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="timestamp must be ISO-8601 parseable"):
        c_sweep.validate_sweep_summary(tampered_timestamp)
    tampered_naive_timestamp = json.loads(json.dumps(persisted))
    tampered_naive_timestamp["timestamp"] = "2026-05-28T00:00:00"
    with pytest.raises(ValueError, match="timestamp must include timezone"):
        c_sweep.validate_sweep_summary(tampered_naive_timestamp)
    tampered_batch_sizes = json.loads(json.dumps(persisted))
    tampered_batch_sizes["batch_sizes"] = []
    with pytest.raises(ValueError, match="batch_sizes must be a non-empty unique positive-int list"):
        c_sweep.validate_sweep_summary(tampered_batch_sizes)
    tampered_options = json.loads(json.dumps(persisted))
    tampered_options["options"]["stop_on_failure"] = "yes"
    with pytest.raises(ValueError, match="options.stop_on_failure must be a bool"):
        c_sweep.validate_sweep_summary(tampered_options)
    tampered_output_dir = json.loads(json.dumps(persisted))
    tampered_output_dir["output_dir"] = ""
    with pytest.raises(ValueError, match="output_dir must be a non-empty string"):
        c_sweep.validate_sweep_summary(tampered_output_dir)
    tampered_command_argv = json.loads(json.dumps(persisted))
    tampered_command_argv["commands"][-1]["argv"] = []
    with pytest.raises(ValueError, match=r"commands\[\]\.argv must be a non-empty string list"):
        c_sweep.validate_sweep_summary(tampered_command_argv)
    tampered_command_text = json.loads(json.dumps(persisted))
    tampered_command_text["commands"][-1]["command"] = "python3 scripts/qwen35_batch_retained_bench.py --tampered"
    with pytest.raises(ValueError, match=r"commands\[\]\.command must match"):
        c_sweep.validate_sweep_summary(tampered_command_text)
    tampered_artifact_path = json.loads(json.dumps(persisted))
    tampered_artifact_path["commands"][-1]["artifact_path"] = str(output_dir / "other-native-diagnostic-c2.json")
    with pytest.raises(ValueError, match=r"commands\[\]\.artifact_path must match commands\[\]\.argv --json"):
        c_sweep.validate_sweep_summary(tampered_artifact_path)
    tampered_artifact_output_dir = json.loads(json.dumps(persisted))
    primitive_row = tampered_artifact_output_dir["commands"][0]
    outside_artifact = str(tmp_path / "outside" / "primitive-c2.json")
    primitive_row["artifact_path"] = outside_artifact
    primitive_row["argv"][primitive_row["argv"].index("--json") + 1] = outside_artifact
    primitive_row["command"] = shlex.join(primitive_row["argv"])
    with pytest.raises(ValueError, match=r"commands\[\]\.artifact_path must be under output_dir"):
        c_sweep.validate_sweep_summary(tampered_artifact_output_dir)
    tampered_artifact_filename = json.loads(json.dumps(persisted))
    primitive_row = tampered_artifact_filename["commands"][0]
    wrong_filename = str(output_dir / "other-primitive-c2.json")
    primitive_row["artifact_path"] = wrong_filename
    primitive_row["argv"][primitive_row["argv"].index("--json") + 1] = wrong_filename
    primitive_row["command"] = shlex.join(primitive_row["argv"])
    with pytest.raises(ValueError, match=r"commands\[\]\.artifact_path must match category/batch-size filename"):
        c_sweep.validate_sweep_summary(tampered_artifact_filename)
    tampered_argv_batch_size = json.loads(json.dumps(persisted))
    retained_argv = tampered_argv_batch_size["commands"][-1]["argv"]
    retained_argv[retained_argv.index("--batch-size") + 1] = "3"
    tampered_argv_batch_size["commands"][-1]["command"] = shlex.join(retained_argv)
    with pytest.raises(ValueError, match=r"commands\[\]\.batch_size must match commands\[\]\.argv"):
        c_sweep.validate_sweep_summary(tampered_argv_batch_size)
    tampered_argv_shape = json.loads(json.dumps(persisted))
    retained_argv = tampered_argv_shape["commands"][-1]["argv"]
    retained_argv[retained_argv.index("--decode-tokens") + 1] = "two"
    tampered_argv_shape["commands"][-1]["command"] = shlex.join(retained_argv)
    with pytest.raises(ValueError, match=r"commands\[\]\.argv --decode-tokens must have an int value"):
        c_sweep.validate_sweep_summary(tampered_argv_shape)
    tampered_command_category = json.loads(json.dumps(persisted))
    retained_row = tampered_command_category["commands"][-1]
    retained_row["category"] = "serial_bridge"
    serial_artifact = str(output_dir / "serial-bridge-c2.json")
    retained_row["artifact_path"] = serial_artifact
    retained_row["argv"][retained_row["argv"].index("--json") + 1] = serial_artifact
    retained_row["command"] = shlex.join(retained_row["argv"])
    with pytest.raises(ValueError, match=r"commands\[\]\.category must match commands\[\]\.argv script"):
        c_sweep.validate_sweep_summary(tampered_command_category)
    tampered_returncode = json.loads(json.dumps(persisted))
    tampered_returncode["commands"][-1]["returncode"] = None
    with pytest.raises(ValueError, match=r"commands\[\]\.returncode must be an int"):
        c_sweep.validate_sweep_summary(tampered_returncode)
    tampered_passed_returncode = json.loads(json.dumps(persisted))
    tampered_passed_returncode["commands"][-1]["returncode"] = 1
    with pytest.raises(ValueError, match=r"commands\[\]\.status passed requires returncode 0"):
        c_sweep.validate_sweep_summary(tampered_passed_returncode)
    native_artifact_path = output_dir / "native-diagnostic-c2.json"
    native_artifact_payload = native_artifact_path.read_text()
    native_artifact_path.unlink()
    with pytest.raises(ValueError, match=r"commands\[\]\.artifact_path must exist for passed summary rows"):
        c_sweep.validate_sweep_summary(persisted)
    native_artifact_path.write_text(native_artifact_payload)
    primitive_artifact_path = output_dir / "primitive-c2.json"
    primitive_artifact_payload = primitive_artifact_path.read_text()
    primitive_artifact_path.unlink()
    with pytest.raises(ValueError, match=r"commands\[\]\.artifact_path must exist for passed summary rows"):
        c_sweep.validate_sweep_summary(persisted)
    primitive_artifact_path.write_text(primitive_artifact_payload)
    tampered_duration = json.loads(json.dumps(persisted))
    tampered_duration["commands"][-1]["duration_seconds"] = -1.0
    with pytest.raises(ValueError, match=r"commands\[\]\.duration_seconds must be a non-negative number"):
        c_sweep.validate_sweep_summary(tampered_duration)
    tampered_nonfinite_duration = json.loads(json.dumps(persisted))
    tampered_nonfinite_duration["commands"][-1]["duration_seconds"] = float("nan")
    with pytest.raises(ValueError, match=r"commands\[\]\.duration_seconds must be finite"):
        c_sweep.validate_sweep_summary(tampered_nonfinite_duration)
    tampered_output_tail = json.loads(json.dumps(persisted))
    tampered_output_tail["commands"][-1]["output_tail"] = ["ok"]
    with pytest.raises(ValueError, match=r"commands\[\]\.output_tail must be a string"):
        c_sweep.validate_sweep_summary(tampered_output_tail)
    tampered_long_output_tail = json.loads(json.dumps(persisted))
    tampered_long_output_tail["commands"][-1]["output_tail"] = "x" * 4001
    with pytest.raises(ValueError, match=r"commands\[\]\.output_tail must be no longer than 4000 characters"):
        c_sweep.validate_sweep_summary(tampered_long_output_tail)
    tampered_singular_precondition = json.loads(json.dumps(persisted))
    tampered_singular_precondition["commands"][-1]["precondition"] = tampered_singular_precondition["commands"][-1]["preconditions"][0]
    with pytest.raises(ValueError, match=r"commands\[\]\.precondition must be absent unless a precondition failed"):
        c_sweep.validate_sweep_summary(tampered_singular_precondition)
    tampered_stray_precondition = json.loads(json.dumps(persisted))
    tampered_stray_precondition["commands"][0]["precondition"] = {"kind": "primitive_correctness", "passed": False}
    with pytest.raises(ValueError, match=r"commands\[\]\.precondition must be absent unless preconditions include a failure"):
        c_sweep.validate_sweep_summary(tampered_stray_precondition)
    tampered_precondition_scope = json.loads(json.dumps(persisted))
    tampered_precondition_scope["commands"][0]["preconditions"] = [
        {"kind": "primitive_correctness", "passed": True}
    ]
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions are only valid for retained native diagnostic rows"):
        c_sweep.validate_sweep_summary(tampered_precondition_scope)
    tampered_singular_postcondition = json.loads(json.dumps(persisted))
    tampered_singular_postcondition["commands"][-1]["postcondition"] = tampered_singular_postcondition["commands"][-1]["postconditions"][0]
    with pytest.raises(ValueError, match=r"commands\[\]\.postcondition must be absent unless a postcondition failed"):
        c_sweep.validate_sweep_summary(tampered_singular_postcondition)
    tampered_stray_postcondition = json.loads(json.dumps(persisted))
    tampered_stray_postcondition["commands"][0]["postcondition"] = {"kind": "retained_profiler_synthesis", "passed": False}
    with pytest.raises(ValueError, match=r"commands\[\]\.postcondition must be absent unless postconditions include a failure"):
        c_sweep.validate_sweep_summary(tampered_stray_postcondition)
    tampered_postcondition_scope = json.loads(json.dumps(persisted))
    tampered_postcondition_scope["commands"][0]["postconditions"] = [
        {"kind": "retained_profiler_synthesis", "passed": True}
    ]
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions are only valid for retained native diagnostic rows"):
        c_sweep.validate_sweep_summary(tampered_postcondition_scope)
    tampered_missing_postconditions = json.loads(json.dumps(persisted))
    del tampered_missing_postconditions["commands"][-1]["postconditions"]
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions must include retained native postconditions for passed retained rows"):
        c_sweep.validate_sweep_summary(tampered_missing_postconditions)
    tampered_postcondition_kind = json.loads(json.dumps(persisted))
    tampered_postcondition_kind["commands"][-1]["postconditions"][0]["kind"] = "other_check"
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions must include retained native postcondition kinds"):
        c_sweep.validate_sweep_summary(tampered_postcondition_kind)
    tampered_postcondition_artifact_path = json.loads(json.dumps(persisted))
    tampered_postcondition_artifact_path["commands"][-1]["postconditions"][0]["artifact_path"] = str(output_dir / "other.json")
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions\[\]\.artifact_path must match commands\[\]\.artifact_path"):
        c_sweep.validate_sweep_summary(tampered_postcondition_artifact_path)
    tampered_postcondition_profiler_path = json.loads(json.dumps(persisted))
    tampered_postcondition_profiler_path["commands"][-1]["postconditions"][0]["profiler_precondition_artifact_path"] = str(output_dir / "other-profiler.json")
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions\[\]\.profiler_precondition_artifact_path must match profiler_summary precondition"):
        c_sweep.validate_sweep_summary(tampered_postcondition_profiler_path)
    tampered_passed_postcondition_reason = json.loads(json.dumps(persisted))
    tampered_passed_postcondition_reason["commands"][-1]["postconditions"][0]["reason"] = "unexpected"
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions\[\]\.reason must be null when passed"):
        c_sweep.validate_sweep_summary(tampered_passed_postcondition_reason)
    tampered_failed_postcondition_reason = json.loads(json.dumps(persisted))
    tampered_failed_postcondition_reason["commands"][-1]["postconditions"][0]["passed"] = False
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions\[\]\.reason must be a non-empty string when failed"):
        c_sweep.validate_sweep_summary(tampered_failed_postcondition_reason)
    tampered_postcondition_field_shape = json.loads(json.dumps(persisted))
    tampered_postcondition_field_shape["commands"][-1]["postconditions"][0]["profiler_synthesized_fields"] = [1]
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions\[\]\.profiler synthesized fields must be string lists when passed"):
        c_sweep.validate_sweep_summary(tampered_postcondition_field_shape)
    tampered_postcondition_field_match = json.loads(json.dumps(persisted))
    tampered_postcondition_field_match["commands"][-1]["postconditions"][0]["profiler_synthesized_fields"] = ["trace_kernel_names"]
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions\[\]\.profiler synthesized fields must match when passed"):
        c_sweep.validate_sweep_summary(tampered_postcondition_field_match)
    tampered_profiler_precondition_field_shape = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_field_shape["commands"][-1]["preconditions"][-1]["profiler_trace_synthesized_fields"] = [1]
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler_trace_synthesized_fields must be a string list when retained postcondition passed"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_field_shape)
    tampered_profiler_precondition_field_match = json.loads(json.dumps(persisted))
    postcondition = tampered_profiler_precondition_field_match["commands"][-1]["postconditions"][0]
    postcondition["profiler_synthesized_fields"] = ["trace_kernel_names"]
    postcondition["profiler_precondition_synthesized_fields"] = ["trace_kernel_names"]
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions\[\]\.profiler_precondition_synthesized_fields must match profiler_summary precondition"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_field_match)
    tampered_profiler_unknown_synthesized_field = json.loads(json.dumps(persisted))
    postcondition = tampered_profiler_unknown_synthesized_field["commands"][-1]["postconditions"][0]
    postcondition["profiler_synthesized_fields"] = ["trace_kernel_names", "edited_field"]
    postcondition["profiler_precondition_synthesized_fields"] = ["trace_kernel_names", "edited_field"]
    tampered_profiler_unknown_synthesized_field["commands"][-1]["preconditions"][-1]["profiler_trace_synthesized_fields"] = ["trace_kernel_names", "edited_field"]
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions\[\]\.profiler synthesized fields must be known trace-derived fields"):
        c_sweep.validate_sweep_summary(tampered_profiler_unknown_synthesized_field)
    tampered_profiler_duplicate_synthesized_field = json.loads(json.dumps(persisted))
    postcondition = tampered_profiler_duplicate_synthesized_field["commands"][-1]["postconditions"][0]
    postcondition["profiler_synthesized_fields"] = ["trace_kernel_names", "trace_kernel_names"]
    postcondition["profiler_precondition_synthesized_fields"] = ["trace_kernel_names", "trace_kernel_names"]
    tampered_profiler_duplicate_synthesized_field["commands"][-1]["preconditions"][-1]["profiler_trace_synthesized_fields"] = ["trace_kernel_names", "trace_kernel_names"]
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions\[\]\.profiler synthesized fields must be unique"):
        c_sweep.validate_sweep_summary(tampered_profiler_duplicate_synthesized_field)
    tampered_git_dirty = json.loads(json.dumps(persisted))
    tampered_git_dirty["commands"][-1]["git_dirty"] = True
    with pytest.raises(ValueError, match=r"commands\[\]\.git_dirty must match git.dirty"):
        c_sweep.validate_sweep_summary(tampered_git_dirty)
    tampered_git_status = json.loads(json.dumps(persisted))
    tampered_git_status["git"]["status_short"] = [1]
    with pytest.raises(ValueError, match="git.status_short must be a string list"):
        c_sweep.validate_sweep_summary(tampered_git_status)
    tampered_git_dirty_status = json.loads(json.dumps(persisted))
    tampered_git_dirty_status["git"]["status_short"] = ["?? uv.lock"]
    with pytest.raises(ValueError, match=r"git\.dirty must match bool\(git\.status_short\)"):
        c_sweep.validate_sweep_summary(tampered_git_dirty_status)
    tampered_completed_count = json.loads(json.dumps(persisted))
    tampered_completed_count["completed_command_count"] = 2
    with pytest.raises(ValueError, match=r"completed_command_count must match len\(commands\)"):
        c_sweep.validate_sweep_summary(tampered_completed_count)
    tampered_command_count = json.loads(json.dumps(persisted))
    tampered_command_count["command_count"] = 4
    with pytest.raises(ValueError, match=r"command_count must match batch_sizes/options\.include_int8"):
        c_sweep.validate_sweep_summary(tampered_command_count)
    tampered_command_order = json.loads(json.dumps(persisted))
    tampered_command_order["commands"][0], tampered_command_order["commands"][1] = (
        tampered_command_order["commands"][1],
        tampered_command_order["commands"][0],
    )
    with pytest.raises(ValueError, match=r"commands\[\] category/batch_size order must match batch_sizes/options\.include_int8"):
        c_sweep.validate_sweep_summary(tampered_command_order)
    tampered_status_counts = json.loads(json.dumps(persisted))
    tampered_status_counts["status_counts"] = {}
    with pytest.raises(ValueError, match="status_counts must match commands"):
        c_sweep.validate_sweep_summary(tampered_status_counts)
    invalid_summary_path = tmp_path / "invalid-summary.json"
    invalid_summary_path.write_text(json.dumps(tampered_status_counts))
    assert c_sweep.main(["--validate-summary-json", str(invalid_summary_path)]) == 1
    tampered_category_status_counts = json.loads(json.dumps(persisted))
    tampered_category_status_counts["category_status_counts"] = {}
    with pytest.raises(ValueError, match="category_status_counts must match commands"):
        c_sweep.validate_sweep_summary(tampered_category_status_counts)
    tampered_status = json.loads(json.dumps(persisted))
    tampered_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status must match commands"):
        c_sweep.validate_sweep_summary(tampered_status)
    tampered_executed_planned = json.loads(json.dumps(persisted))
    tampered_executed_planned["commands"][-1]["status"] = "planned"
    with pytest.raises(ValueError, match=r"commands\[\]\.status cannot be planned for executed summaries"):
        c_sweep.validate_sweep_summary(tampered_executed_planned)
    assert persisted["retained_postcondition_counts"] == summary["retained_postcondition_counts"]
    assert persisted["failed_postconditions"] == summary["failed_postconditions"]
    assert persisted["commands"][-1]["postconditions"] == native["postconditions"]
    tampered_postcondition_counts = json.loads(json.dumps(persisted))
    tampered_postcondition_counts["retained_postcondition_counts"] = {}
    with pytest.raises(ValueError, match="retained_postcondition_counts must match commands.postconditions"):
        c_sweep.validate_sweep_summary(tampered_postcondition_counts)
    tampered_precondition_schema = json.loads(json.dumps(persisted))
    tampered_precondition_schema["commands"][-1]["preconditions"][0]["passed"] = "yes"
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.passed must be a bool"):
        c_sweep.validate_sweep_summary(tampered_precondition_schema)
    tampered_passed_precondition_reason = json.loads(json.dumps(persisted))
    tampered_passed_precondition_reason["commands"][-1]["preconditions"][0]["reason"] = "unexpected"
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.reason must be null when passed"):
        c_sweep.validate_sweep_summary(tampered_passed_precondition_reason)
    tampered_failed_precondition_reason = json.loads(json.dumps(persisted))
    tampered_failed_precondition_reason["commands"][-1]["preconditions"][0]["passed"] = False
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.reason must be a non-empty string when failed"):
        c_sweep.validate_sweep_summary(tampered_failed_precondition_reason)
    tampered_passed_row_failed_precondition = json.loads(json.dumps(persisted))
    failed_gate = tampered_passed_row_failed_precondition["commands"][-1]["preconditions"][0]
    failed_gate["passed"] = False
    failed_gate["reason"] = "primitive correctness failed"
    tampered_passed_row_failed_precondition["commands"][-1]["precondition"] = dict(failed_gate)
    with pytest.raises(ValueError, match=r"commands\[\]\.status passed cannot include failed preconditions"):
        c_sweep.validate_sweep_summary(tampered_passed_row_failed_precondition)
    tampered_postcondition_schema = json.loads(json.dumps(persisted))
    tampered_postcondition_schema["commands"][-1]["postconditions"][0]["kind"] = ""
    with pytest.raises(ValueError, match=r"commands\[\]\.postconditions\[\]\.kind must be a non-empty string"):
        c_sweep.validate_sweep_summary(tampered_postcondition_schema)
    tampered_retained_gate_kinds = json.loads(json.dumps(persisted))
    tampered_retained_gate_kinds["commands"][-1]["preconditions"].pop()
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions must include retained native gate kinds"):
        c_sweep.validate_sweep_summary(tampered_retained_gate_kinds)
    tampered_precondition_artifact_path = json.loads(json.dumps(persisted))
    tampered_precondition_artifact_path["commands"][-1]["preconditions"][0]["artifact_path"] = str(output_dir / "other-primitive.json")
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.artifact_path must match retained native gate argv"):
        c_sweep.validate_sweep_summary(tampered_precondition_artifact_path)
    tampered_precondition_argv_path = json.loads(json.dumps(persisted))
    retained_argv = tampered_precondition_argv_path["commands"][-1]["argv"]
    retained_argv[retained_argv.index("--serial-bridge-json") + 1] = str(output_dir / "other-serial.json")
    tampered_precondition_argv_path["commands"][-1]["command"] = shlex.join(retained_argv)
    with pytest.raises(ValueError, match=r"commands\[\]\.argv retained native gate artifact paths must match output_dir filenames"):
        c_sweep.validate_sweep_summary(tampered_precondition_argv_path)
    tampered_missing_gate_argv = json.loads(json.dumps(persisted))
    retained_argv = tampered_missing_gate_argv["commands"][-1]["argv"]
    gate_index = retained_argv.index("--profiler-json")
    del retained_argv[gate_index : gate_index + 2]
    tampered_missing_gate_argv["commands"][-1]["command"] = shlex.join(retained_argv)
    with pytest.raises(ValueError, match=r"commands\[\]\.argv must include retained native gate artifact flags"):
        c_sweep.validate_sweep_summary(tampered_missing_gate_argv)
    tampered_gate_argv_filename = json.loads(json.dumps(persisted))
    retained_argv = tampered_gate_argv_filename["commands"][-1]["argv"]
    other_serial = str(output_dir / "other-serial.json")
    retained_argv[retained_argv.index("--serial-bridge-json") + 1] = other_serial
    tampered_gate_argv_filename["commands"][-1]["preconditions"][2]["artifact_path"] = other_serial
    tampered_gate_argv_filename["commands"][-1]["command"] = shlex.join(retained_argv)
    with pytest.raises(ValueError, match=r"commands\[\]\.argv retained native gate artifact paths must match output_dir filenames"):
        c_sweep.validate_sweep_summary(tampered_gate_argv_filename)
    tampered_scaling_precondition_concurrency = json.loads(json.dumps(persisted))
    tampered_scaling_precondition_concurrency["commands"][-1]["preconditions"][1]["workload_concurrency"] = 2
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.workload_concurrency must match retained scaling gate"):
        c_sweep.validate_sweep_summary(tampered_scaling_precondition_concurrency)
    tampered_scaling_precondition_shape = json.loads(json.dumps(persisted))
    tampered_scaling_precondition_shape["commands"][-1]["preconditions"][2]["prompt_tokens_per_request"] = 17
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.prompt_tokens_per_request must match retained command shape"):
        c_sweep.validate_sweep_summary(tampered_scaling_precondition_shape)
    tampered_scaling_precondition_rate = json.loads(json.dumps(persisted))
    tampered_scaling_precondition_rate["commands"][-1]["preconditions"][1]["decode_tok_s_aggregate"] = 0.0
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.decode rates must be positive numbers when passed"):
        c_sweep.validate_sweep_summary(tampered_scaling_precondition_rate)
    tampered_profiler_precondition_concurrency = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_concurrency["commands"][-1]["preconditions"][-1]["workload_concurrency"] = 1
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler workload_concurrency must match retained batch_size"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_concurrency)
    tampered_profiler_precondition_shape = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_shape["commands"][-1]["preconditions"][-1]["profiler_warmup_decode_tokens"] = 2
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler_warmup_decode_tokens must match retained command shape"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_shape)
    tampered_profiler_precondition_command = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_command["commands"][-1]["preconditions"][-1]["profiler_command"] = "python3 scripts/qwen35_batch_retained_bench.py --model /tmp/model --fixture /tmp/fixture.json"
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler_command must include rocprofv3 kernel trace retained bench when passed"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_command)
    tampered_profiler_precondition_model = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_model["commands"][-1]["preconditions"][-1]["profiler_model"] = "/tmp/other-model"
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler model must match retained command"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_model)
    tampered_profiler_precondition_fixture = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_fixture["commands"][-1]["preconditions"][-1]["profiler_fixture"] = "/tmp/other-fixture.json"
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler fixture must match retained command"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_fixture)
    tampered_profiler_precondition_json = json.loads(json.dumps(persisted))
    profiler_argv = shlex.split(tampered_profiler_precondition_json["commands"][-1]["preconditions"][-1]["profiler_command"])
    profiler_argv[profiler_argv.index("--json") + 1] = str(output_dir / "other-native.json")
    tampered_profiler_precondition_json["commands"][-1]["preconditions"][-1]["profiler_command"] = shlex.join(profiler_argv)
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler command --json must match retained artifact"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_json)
    tampered_profiler_precondition_profiler_json = json.loads(json.dumps(persisted))
    profiler_argv = shlex.split(tampered_profiler_precondition_profiler_json["commands"][-1]["preconditions"][-1]["profiler_command"])
    profiler_argv[profiler_argv.index("--profiler-json") + 1] = str(output_dir / "other-profiler.json")
    tampered_profiler_precondition_profiler_json["commands"][-1]["preconditions"][-1]["profiler_command"] = shlex.join(profiler_argv)
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler command --profiler-json must match profiler precondition artifact"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_profiler_json)
    tampered_profiler_precondition_gate_path = json.loads(json.dumps(persisted))
    profiler_argv = shlex.split(tampered_profiler_precondition_gate_path["commands"][-1]["preconditions"][-1]["profiler_command"])
    profiler_argv[profiler_argv.index("--serial-bridge-json") + 1] = str(output_dir / "other-serial.json")
    tampered_profiler_precondition_gate_path["commands"][-1]["preconditions"][-1]["profiler_command"] = shlex.join(profiler_argv)
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler command gate paths must match retained command"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_gate_path)
    tampered_profiler_precondition_retained_artifact = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_retained_artifact["commands"][-1]["preconditions"][-1]["retained_artifact_path"] = str(output_dir / "other-native.json")
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler retained_artifact_path must match retained artifact"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_retained_artifact)
    tampered_profiler_precondition_gate_artifact = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_gate_artifact["commands"][-1]["preconditions"][-1]["serial_bridge_artifact_path"] = str(output_dir / "other-serial.json")
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler gate artifact paths must match retained command"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_gate_artifact)
    tampered_profiler_precondition_compiler = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_compiler["commands"][-1]["preconditions"][-1]["profiler_compiler_version_file"] = str(output_dir / "other-hipcc-version.txt")
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler_compiler_version_file must match retained command"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_compiler)
    tampered_profiler_precondition_cached_build = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_cached_build["commands"][-1]["preconditions"][-1]["profiler_require_cached_build"] = True
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler_require_cached_build must match retained command"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_cached_build)
    tampered_profiler_precondition_status = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_status["commands"][-1]["preconditions"][-1]["profiler_status"] = "missing"
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler_status must be captured when passed"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_status)
    tampered_profiler_precondition_trace_dir = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_trace_dir["commands"][-1]["preconditions"][-1].pop("profiler_trace_dir")
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler_trace_dir must be a non-empty string when passed"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_trace_dir)
    tampered_profiler_precondition_trace_dir_command = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_trace_dir_command["commands"][-1]["preconditions"][-1]["profiler_trace_dir"] = str(output_dir / "other-profile-c2")
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler_trace_dir must match profiler command -d"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_trace_dir_command)
    tampered_profiler_precondition_trace_file_scope = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_trace_file_scope["commands"][-1]["preconditions"][-1]["profiler_trace_files"] = [str(output_dir / "other-profile-c2" / "hipengine_kernel_trace.csv")]
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler_trace_files must be under profiler_trace_dir when passed"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_trace_file_scope)
    tampered_profiler_precondition_trace_files = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_trace_files["commands"][-1]["preconditions"][-1]["profiler_trace_files"] = []
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler_trace_files must include a kernel-trace CSV when passed"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_trace_files)
    tampered_profiler_precondition_kernel_names = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_kernel_names["commands"][-1]["preconditions"][-1]["profiler_trace_kernel_names"] = ["serial_lm_head"]
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.profiler_trace_kernel_names must include native batch kernels only when passed"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_kernel_names)
    tampered_profiler_precondition_expected_kernels = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_expected_kernels["commands"][-1]["preconditions"][-1]["expected_kernel_names"] = ["serial_lm_head"]
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.expected_kernel_names must include native batch kernels only when profiler passed"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_expected_kernels)
    tampered_profiler_precondition_expected_trace = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_expected_trace["commands"][-1]["preconditions"][-1]["profiler_trace_kernel_names"] = ["qwen35_batch_other"]
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.expected_kernel_names must be present in profiler_trace_kernel_names"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_expected_trace)
    tampered_profiler_precondition_durations = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_durations["commands"][-1]["preconditions"][-1]["kernel_durations_ns"]["qwen35_batch_decode"] = 0.0
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.kernel_durations_ns must contain positive kernel durations when profiler passed"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_durations)
    tampered_profiler_precondition_missing_duration = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_missing_duration["commands"][-1]["preconditions"][-1]["kernel_durations_ns"] = {"qwen35_batch_other": 1.0}
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.kernel_durations_ns must include expected profiler kernels"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_missing_duration)
    tampered_profiler_precondition_duration_trace = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_duration_trace["commands"][-1]["preconditions"][-1]["kernel_durations_ns"]["qwen35_batch_other"] = 1.0
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.kernel_durations_ns keys must be present in profiler_trace_kernel_names"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_duration_trace)
    tampered_profiler_precondition_total_duration = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_total_duration["commands"][-1]["preconditions"][-1]["total_kernel_duration_ns"] = 0.0
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.total_kernel_duration_ns must be positive when profiler passed"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_total_duration)
    tampered_profiler_precondition_categories = json.loads(json.dumps(persisted))
    del tampered_profiler_precondition_categories["commands"][-1]["preconditions"][-1]["kernel_duration_categories_ns"]["other"]
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.kernel duration categories must include required non-negative categories when profiler passed"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_categories)
    tampered_profiler_precondition_cpu_total = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_cpu_total["commands"][-1]["preconditions"][-1]["cpu_side_total_seconds"] = 0.0
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.cpu_side_total_seconds must be positive when profiler passed"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_cpu_total)
    tampered_profiler_precondition_cpu_categories = json.loads(json.dumps(persisted))
    tampered_profiler_precondition_cpu_categories["commands"][-1]["preconditions"][-1]["cpu_side_bottlenecks_seconds"]["decode"] = -1.0
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.cpu-side bottlenecks must include required non-negative categories when profiler passed"):
        c_sweep.validate_sweep_summary(tampered_profiler_precondition_cpu_categories)
    tampered_primitive_precondition_rows = json.loads(json.dumps(persisted))
    tampered_primitive_precondition_rows["commands"][-1]["preconditions"][0]["primitive_rows"] = 3
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.primitive_rows must match retained batch_size"):
        c_sweep.validate_sweep_summary(tampered_primitive_precondition_rows)
    tampered_primitive_precondition_mismatch = json.loads(json.dumps(persisted))
    tampered_primitive_precondition_mismatch["commands"][-1]["preconditions"][0]["append_key_mismatch"] = 1
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.primitive append mismatches must be zero when passed"):
        c_sweep.validate_sweep_summary(tampered_primitive_precondition_mismatch)
    tampered_primitive_precondition_attn = json.loads(json.dumps(persisted))
    tampered_primitive_precondition_attn["commands"][-1]["preconditions"][0]["attn_batch_vs_c1_max_abs"] = 1e-3
    with pytest.raises(ValueError, match=r"commands\[\]\.preconditions\[\]\.attn_batch_vs_c1_max_abs must be at most 1e-6 when primitive passed"):
        c_sweep.validate_sweep_summary(tampered_primitive_precondition_attn)
    preconditions_by_kind = {item["kind"]: item for item in native["preconditions"]}
    assert preconditions_by_kind["primitive_correctness"] == {
        "kind": "primitive_correctness",
        "artifact_path": str(output_dir / "primitive-c2.json"),
        "passed": True,
        "reason": None,
        "primitive_rows": 2,
        "append_key_mismatch": 0,
        "append_value_mismatch": 0,
        "attn_batch_vs_c1_max_abs": 0.0,
    }
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
    assert preconditions_by_kind["profiler_summary"] == {
        "kind": "profiler_summary",
        "artifact_path": str(output_dir / "profiler-c2.json"),
        "passed": True,
        "reason": None,
        "profiler_status": "captured",
        "profiler_command": (
            f"rocprofv3 --kernel-trace --output-format csv -d {output_dir / 'profile-c2'} -- python3 scripts/qwen35_batch_retained_bench.py "
            "--model /tmp/model --fixture /tmp/fixture.json --batch-size 2 "
            "--prompt-length 16 --decode-tokens 2 --warmup-decode-tokens 1 --max-layers 3 "
            f"--json {output_dir / 'native-diagnostic-c2.json'} --c1-baseline-json {output_dir / 'native-baseline-c1.json'} "
            f"--serial-bridge-json {output_dir / 'serial-bridge-c2.json'} "
            f"--primitive-correctness-json {output_dir / 'primitive-c2.json'} --profiler-json {output_dir / 'profiler-c2.json'}"
        ),
        "profiler_output_format": "csv",
        "profiler_trace_dir": str(output_dir / "profile-c2"),
        "profiler_trace_files": [str(output_dir / "profile-c2" / "hipengine_kernel_trace.csv")],
        "profiler_trace_kernel_names": ["qwen35_batch_decode"],
        "profiler_trace_synthesized_fields": [],
        "retained_artifact_path": str(output_dir / "native-diagnostic-c2.json"),
        "c1_baseline_artifact_path": str(output_dir / "native-baseline-c1.json"),
        "serial_bridge_artifact_path": str(output_dir / "serial-bridge-c2.json"),
        "primitive_correctness_artifact_path": str(output_dir / "primitive-c2.json"),
        "profiler_compiler_version_file": None,
        "profiler_require_cached_build": False,
        "profiler_model": "/tmp/model",
        "profiler_fixture": "/tmp/fixture.json",
        "profiler_warmup_decode_tokens": 1,
        "profiler_max_layers": 3,
        "workload_concurrency": 2,
        "prompt_tokens_per_request": 16,
        "gen_tokens_per_request": 2,
        "expected_kernel_names": ["qwen35_batch_decode"],
        "kernel_durations_ns": {"qwen35_batch_decode": 12345.0},
        "total_kernel_duration_ns": 12345.0,
        "kernel_duration_shares": {"qwen35_batch_decode": 1.0},
        "kernel_duration_categories_ns": {
            "attention": 0.0,
            "moe": 0.0,
            "projection": 0.0,
            "sampling": 0.0,
            "graph_replay": 0.0,
            "other": 12345.0,
        },
        "kernel_duration_category_shares": {
            "attention": 0.0,
            "moe": 0.0,
            "projection": 0.0,
            "sampling": 0.0,
            "graph_replay": 0.0,
            "other": 1.0,
        },
        "cpu_side_total_seconds": 10.0,
        "cpu_side_bottlenecks_seconds": {
            "load": 1.0,
            "prefill": 2.0,
            "warmup_decode": 0.0,
            "decode": 7.0,
            "validation": 0.0,
            "other": 0.0,
        },
        "cpu_side_bottleneck_shares": {
            "load": 0.1,
            "prefill": 0.2,
            "warmup_decode": 0.0,
            "decode": 0.7,
            "validation": 0.0,
            "other": 0.0,
        },
    }
    assert "precondition" not in native


def test_batch_c_sweep_rejects_missing_retained_profiler_synthesis_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "missing-native-diagnostic-c2.json"
    profiler_path = tmp_path / "profiler-c2.json"
    command = c_sweep.SweepCommand(
        category="native_diagnostic",
        batch_size=2,
        artifact_path=artifact_path,
        argv=("python3", "scripts/qwen35_batch_retained_bench.py"),
    )
    postcondition = c_sweep._retained_profiler_synthesis_postcondition(
        command,
        [
            {
                "kind": "profiler_summary",
                "artifact_path": str(profiler_path),
                "passed": True,
                "profiler_trace_synthesized_fields": [],
            }
        ],
    )

    assert postcondition == {
        "kind": "retained_profiler_synthesis",
        "artifact_path": str(artifact_path),
        "profiler_precondition_artifact_path": str(profiler_path),
        "passed": False,
        "reason": "retained artifact was not written for profiler provenance cross-check",
    }


def test_batch_c_sweep_rejects_retained_profiler_synthesis_mismatch(tmp_path: Path) -> None:
    artifact_path = tmp_path / "native-diagnostic-c2.json"
    profiler_path = tmp_path / "profiler-c2.json"
    artifact_path.write_text(json.dumps({"profiler": {"synthesized_fields": ["trace_kernel_names"]}}))
    command = c_sweep.SweepCommand(
        category="native_diagnostic",
        batch_size=2,
        artifact_path=artifact_path,
        argv=("python3", "scripts/qwen35_batch_retained_bench.py"),
    )
    postcondition = c_sweep._retained_profiler_synthesis_postcondition(
        command,
        [
            {
                "kind": "profiler_summary",
                "artifact_path": str(profiler_path),
                "passed": True,
                "profiler_trace_synthesized_fields": [],
            }
        ],
    )

    assert postcondition == {
        "kind": "retained_profiler_synthesis",
        "artifact_path": str(artifact_path),
        "profiler_precondition_artifact_path": str(profiler_path),
        "passed": False,
        "reason": "retained artifact profiler.synthesized_fields does not match profiler precondition synthesized fields",
        "profiler_synthesized_fields": ["trace_kernel_names"],
        "profiler_precondition_synthesized_fields": [],
    }
    entries = [
        {
            "category": "native_diagnostic",
            "batch_size": 2,
            "artifact_path": str(artifact_path),
            "postconditions": [postcondition],
        }
    ]
    assert c_sweep._retained_postcondition_counts(entries) == {"retained_profiler_synthesis": {"failed": 1}}
    assert c_sweep._failed_postconditions(entries) == [
        {
            "category": "native_diagnostic",
            "batch_size": 2,
            "artifact_path": str(artifact_path),
            "kind": "retained_profiler_synthesis",
            "profiler_precondition_artifact_path": str(profiler_path),
            "reason": "retained artifact profiler.synthesized_fields does not match profiler precondition synthesized fields",
        }
    ]


def test_batch_c_sweep_fails_retained_row_on_profiler_synthesis_mismatch(tmp_path: Path, monkeypatch) -> None:
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
    _write_c_sweep_profiler_summary(output_dir, warmup_decode_tokens=1, max_layers=3)
    (output_dir / "native-baseline-c1.json").write_text(
        json.dumps({"schema": 1, "prompt_length": 16, "decode_tokens": 2, "throughput": {"warmed_decode_tok_s": 10.0}})
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

    def fake_run(argv, **kwargs):
        if len(argv) > 1 and argv[1] == "scripts/qwen35_batch_retained_bench.py":
            (output_dir / "native-diagnostic-c2.json").write_text(
                json.dumps({"profiler": {"synthesized_fields": ["trace_kernel_names"]}})
            )
        return FakeProc()

    monkeypatch.setattr(c_sweep.subprocess, "run", fake_run)

    summary = run_sweep(args)

    assert summary["status"] == "failed"
    assert summary["status_counts"] == {"passed": 2, "failed": 1}
    assert summary["retained_postcondition_counts"] == {"retained_profiler_synthesis": {"failed": 1}}
    assert summary["failed_postconditions"] == [
        {
            "category": "native_diagnostic",
            "batch_size": 2,
            "artifact_path": str(output_dir / "native-diagnostic-c2.json"),
            "kind": "retained_profiler_synthesis",
            "profiler_precondition_artifact_path": str(output_dir / "profiler-c2.json"),
            "reason": "retained artifact profiler.synthesized_fields does not match profiler precondition synthesized fields",
        }
    ]
    native = summary["commands"][-1]
    assert native["status"] == "failed"
    assert native["postcondition"] == native["postconditions"][0]
    assert native["output_tail"] == "retained artifact profiler.synthesized_fields does not match profiler precondition synthesized fields"
    c_sweep.validate_sweep_summary(summary)
    tampered_failed_postconditions = json.loads(json.dumps(summary))
    tampered_failed_postconditions["failed_postconditions"] = []
    with pytest.raises(ValueError, match="failed_postconditions must match commands.postconditions"):
        c_sweep.validate_sweep_summary(tampered_failed_postconditions)
    tampered_singular_postcondition = json.loads(json.dumps(summary))
    tampered_singular_postcondition["commands"][-1].pop("postcondition")
    with pytest.raises(ValueError, match=r"commands\[\]\.postcondition must match"):
        c_sweep.validate_sweep_summary(tampered_singular_postcondition)
    tampered_postcondition_output_tail = json.loads(json.dumps(summary))
    tampered_postcondition_output_tail["commands"][-1]["output_tail"] = "different postcondition failure"
    with pytest.raises(ValueError, match=r"commands\[\]\.output_tail must match failed postcondition reason"):
        c_sweep.validate_sweep_summary(tampered_postcondition_output_tail)


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


def test_projection_dispatch_evidence_loads_schema_checked_artifact_blocks() -> None:
    payload = {
        "artifact_path": "benchmarks/results/projection-wmma-c4.json",
        "aggregate_vs_row_gemv": 1.35,
        "per_request_vs_row_gemv": 1.10,
        "accepted": True,
    }

    evidence = ProjectionDispatchEvidence.from_json_dict(payload)

    assert evidence.to_json_dict() == payload
    rejected = ProjectionDispatchEvidence.from_json_dict({**payload, "accepted": False})
    assert rejected.accepted is False
    with pytest.raises(ValueError, match="aggregate_vs_row_gemv must be positive numeric"):
        ProjectionDispatchEvidence.from_json_dict({**payload, "aggregate_vs_row_gemv": 0.0})
    with pytest.raises(ValueError, match="accepted must be a bool"):
        ProjectionDispatchEvidence.from_json_dict({**payload, "accepted": "yes"})
    with pytest.raises(ValueError, match="artifact_path must be under benchmarks/results"):
        ProjectionDispatchEvidence.from_json_dict({**payload, "artifact_path": "/tmp/projection-wmma-c4.json"})
    with pytest.raises(ValueError, match="artifact_path must be under benchmarks/results"):
        ProjectionDispatchEvidence.from_json_dict({**payload, "artifact_path": "benchmarks/results/../tmp/projection-wmma-c4.json"})
    with pytest.raises(ValueError, match="accepted aggregate_vs_row_gemv must be > 1.0"):
        ProjectionDispatchEvidence.from_json_dict({**payload, "aggregate_vs_row_gemv": 1.0})
    with pytest.raises(ValueError, match="accepted per_request_vs_row_gemv must be > 1.0"):
        ProjectionDispatchEvidence.from_json_dict({**payload, "per_request_vs_row_gemv": 0.99})
    rejected_non_winning = ProjectionDispatchEvidence.from_json_dict(
        {**payload, "aggregate_vs_row_gemv": 0.95, "per_request_vs_row_gemv": 0.90, "accepted": False}
    )
    assert rejected_non_winning.accepted is False


def test_projection_dispatch_candidate_loads_schema_checked_artifact_blocks() -> None:
    payload = {
        "name": "wmma_caware",
        "selection": {"layer": "linear", "quant": "w4_paro", "variant": "wmma_caware"},
        "min_rows": 2,
        "max_rows": 8,
        "evidence": {
            "artifact_path": "benchmarks/results/projection-wmma-c4.json",
            "aggregate_vs_row_gemv": 1.35,
            "per_request_vs_row_gemv": 1.10,
            "accepted": True,
        },
    }

    candidate = ProjectionDispatchCandidate.from_json_dict(payload)

    assert candidate.to_json_dict() == payload
    assert candidate.applies_to(4)
    assert not candidate.applies_to(1)
    assert not candidate.applies_to(16)
    with pytest.raises(ValueError, match="selection.quant must be a non-empty string"):
        ProjectionDispatchCandidate.from_json_dict({**payload, "selection": {"layer": "linear", "variant": "wmma_caware"}})
    with pytest.raises(ValueError, match="max_rows must be >= min_rows"):
        ProjectionDispatchCandidate.from_json_dict({**payload, "min_rows": 8, "max_rows": 4})
    with pytest.raises(ValueError, match="aggregate_vs_row_gemv must be positive numeric"):
        ProjectionDispatchCandidate.from_json_dict(
            {**payload, "evidence": {**payload["evidence"], "aggregate_vs_row_gemv": 0.0}}
        )


def test_projection_dispatch_candidate_list_loads_ordered_artifact_blocks() -> None:
    first = {
        "name": "mmq_caware",
        "selection": {"layer": "linear", "quant": "w4_paro", "variant": "mmq_caware"},
        "min_rows": 2,
        "max_rows": 4,
        "evidence": {
            "artifact_path": "benchmarks/results/projection-mmq-c2.json",
            "aggregate_vs_row_gemv": 1.20,
            "per_request_vs_row_gemv": 1.05,
            "accepted": True,
        },
    }
    second = {
        "name": "wmma_caware",
        "selection": {"layer": "linear", "quant": "w4_paro", "variant": "wmma_caware"},
        "min_rows": 4,
        "max_rows": 8,
        "evidence": None,
    }

    candidates = projection_dispatch_candidates_from_json([first, second])

    assert [candidate.name for candidate in candidates] == ["mmq_caware", "wmma_caware"]
    assert candidates[0].to_json_dict() == first
    assert candidates[1].to_json_dict() == second
    with pytest.raises(ValueError, match="projection dispatch candidates must be a list"):
        projection_dispatch_candidates_from_json({"name": "mmq_caware"})
    with pytest.raises(ValueError, match=r"candidates\[1\] must be an object"):
        projection_dispatch_candidates_from_json([first, "not-a-candidate"])
    with pytest.raises(ValueError, match=r"candidates\[0\].*min_rows must be a positive int"):
        projection_dispatch_candidates_from_json([{**first, "min_rows": 0}])


def test_projection_dispatch_candidate_list_loads_from_artifact_payload() -> None:
    candidate = {
        "name": "wmma_caware",
        "selection": {"layer": "linear", "quant": "w4_paro", "variant": "wmma_caware"},
        "min_rows": 4,
        "max_rows": 8,
        "evidence": {
            "artifact_path": "benchmarks/results/projection-wmma-c4.json",
            "aggregate_vs_row_gemv": 1.35,
            "per_request_vs_row_gemv": 1.10,
            "accepted": True,
        },
    }
    artifact = {
        "status": "accepted",
        "projection_dispatch_candidates": [candidate],
    }

    loaded = projection_dispatch_candidates_from_artifact(artifact)

    assert len(loaded) == 1
    assert loaded[0].to_json_dict() == candidate
    assert projection_dispatch_candidates_from_artifact({"status": "blocked"}) == ()
    assert projection_dispatch_candidates_from_artifact({"projection_dispatch": [candidate]}, field="projection_dispatch")[0].name == "wmma_caware"
    with pytest.raises(ValueError, match="projection dispatch candidate field must be a non-empty string"):
        projection_dispatch_candidates_from_artifact(artifact, field="")
    with pytest.raises(ValueError, match="invalid projection_dispatch_candidates"):
        projection_dispatch_candidates_from_artifact({"projection_dispatch_candidates": ["not-a-candidate"]})


def test_projection_dispatch_plans_directly_from_artifact_payload() -> None:
    row_gemv = ProjectionKernelSelection("linear", "w4_paro", "row_gemv")
    candidate = {
        "name": "wmma_caware",
        "selection": {"layer": "linear", "quant": "w4_paro", "variant": "wmma_caware"},
        "min_rows": 4,
        "max_rows": 8,
        "evidence": {
            "artifact_path": "benchmarks/results/projection-wmma-c4.json",
            "aggregate_vs_row_gemv": 1.35,
            "per_request_vs_row_gemv": 1.10,
            "accepted": True,
        },
    }

    decision = plan_projection_dispatch_from_artifact(
        payload={"projection_dispatch_candidates": [candidate]},
        rows=4,
        row_gemv=row_gemv,
    )

    assert decision.selected_candidate == "wmma_caware"
    assert decision.path == "benchmark_accepted_caware_projection"
    assert decision.throughput_claim_eligible is True
    fallback = plan_projection_dispatch_from_artifact(payload={}, rows=4, row_gemv=row_gemv)
    assert fallback.selected_candidate == "row_gemv"
    assert fallback.throughput_claim_eligible is False
    assert "no c-aware projection candidate applies" in fallback.blockers[0]
    with pytest.raises(ValueError, match="invalid projection_dispatch_candidates"):
        plan_projection_dispatch_from_artifact(
            payload={"projection_dispatch_candidates": [{**candidate, "min_rows": 0}]},
            rows=4,
            row_gemv=row_gemv,
        )


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

    tmp_artifact = plan_batch_sampler_dispatch(
        rows=2,
        requested_mode="batched_lm_head",
        c2_equality_green=True,
        equality_artifact="/tmp/qwen35-c2-eq.json",
    )
    assert tmp_artifact.mode is BatchSamplerMode.SERIAL_LM_HEAD
    assert "batched LM-head equality artifact path must be under benchmarks/results" in tmp_artifact.blockers

    traversal_artifact = plan_batch_sampler_dispatch(
        rows=2,
        requested_mode="batched_lm_head",
        c2_equality_green=True,
        equality_artifact="benchmarks/results/../tmp/qwen35-c2-eq.json",
    )
    assert traversal_artifact.mode is BatchSamplerMode.SERIAL_LM_HEAD
    assert "batched LM-head equality artifact path must be under benchmarks/results" in traversal_artifact.blockers

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
    assert scheduler.graph_buckets.stats.miss_reasons == {"cache_absent": 1}
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

    assert cache.get(key, miss_reason="shape_changed") is None
    cache.put(key, object())
    cache.record_kernel_time_ns(5_000)
    cache.record_kernel_time_ns(50_000)
    assert cache.stats.entries == 1
    assert cache.stats.miss_reasons == {"shape_changed": 1}
    assert cache.stats.kernel_time_histogram_ns == {"le_10us": 1, "le_100us": 1}
    assert cache.stats.to_json_dict() == {
        "entries": 1,
        "hits": 0,
        "misses": 1,
        "miss_reasons": {"shape_changed": 1},
        "kernel_time_histogram_ns": {"le_100us": 1, "le_10us": 1},
    }
    with pytest.raises(ValueError, match="duration_ns"):
        cache.record_kernel_time_ns(-1)
    cache.clear()
    assert cache.stats.entries == 0
    assert cache.stats.hits == 0
    assert cache.stats.misses == 0
    assert cache.stats.miss_reasons == {}
    assert cache.stats.kernel_time_histogram_ns == {}


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


def test_qwen35_retained_records_decode_graph_bucket_metadata() -> None:
    scheduler = ResidentBatchScheduler(capacity=2, context_bucket_size=4)
    scheduler.submit([1, 2, 3], max_new_tokens=1)
    scheduler.submit([4, 5], max_new_tokens=1)
    scheduler.admit_pending()
    scheduler.next_compact_prefill_slabs(chunk_size=3, block_size=4)
    metadata: dict[str, object] = {}

    retained_bench._record_decode_graph_bucket_metadata(scheduler, metadata)

    assert metadata["decode_shape_key"] == {
        "mode": "decode",
        "active_c": 2,
        "context_bucket": 4,
        "active_mask": [True, True],
        "top_k": 0,
        "experts_per_token": 0,
        "replay_steps": 1,
        "draft_depth": 0,
        "tree_shape": [],
    }
    assert metadata["graph_bucket_stats"] == {
        "entries": 1,
        "hits": 1,
        "misses": 1,
        "miss_reasons": {"cache_absent": 1},
        "kernel_time_histogram_ns": {},
    }


def test_qwen35_retained_profiler_reference_loads_captured_summary(tmp_path: Path) -> None:
    profiler_path = tmp_path / "profiler-summary.json"
    trace_dir = tmp_path / "hipengine-profile-c2"
    trace_dir.mkdir()
    trace_csv = trace_dir / "hipengine_kernel_trace.csv"
    trace_csv.write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp\n"
        "qwen35_batch_decode,0,100\n"
        "qwen35_batch_decode,100,200\n"
    )
    profiler_path.write_text(
        json.dumps(
            {
                "profiler": {
                    "status": "captured",
                    "command": f"rocprofv3 --kernel-trace --output-format csv -d {trace_dir} -- python3 scripts/qwen35_batch_retained_bench.py",
                    "trace_files": [str(trace_csv)],
                    "expected_kernels_present": True,
                    "expected_kernel_names": ["qwen35_batch_decode"],
                    "kernel_durations_ns": {"qwen35_batch_decode": 12345.0},
                }
            }
        )
    )
    args = argparse.Namespace(profiler_json=profiler_path, profiler_command=None)

    loaded = retained_bench._profiler_reference(profiler_path)
    command = retained_bench._profiled_command(
        args,
        [
            "--batch-size",
            "2",
            "--prompt-length",
            "512",
            "--decode-tokens",
            "128",
            "--max-layers",
            "40",
            "--json",
            "benchmarks/results/native-c2.json",
            "--profiler-json",
            str(profiler_path),
        ],
    )

    assert loaded["status"] == "captured"
    assert loaded["output_format"] == "csv"
    assert loaded["trace_dir"] == str(trace_dir)
    assert loaded["trace_files"] == [str(trace_csv)]
    assert loaded["trace_kernel_names"] == ["qwen35_batch_decode"]
    assert loaded["synthesized_fields"] == [
        "trace_kernel_names",
        "total_kernel_duration_ns",
        "kernel_duration_shares",
        "kernel_duration_categories_ns",
        "kernel_duration_category_shares",
        "output_format",
        "trace_dir",
    ]
    assert loaded["expected_kernels_present"] is True
    assert loaded["total_kernel_duration_ns"] == 12345.0
    assert loaded["kernel_duration_shares"] == {"qwen35_batch_decode": 1.0}
    assert loaded["kernel_duration_categories_ns"] == {
        "attention": 0.0,
        "moe": 0.0,
        "projection": 0.0,
        "sampling": 0.0,
        "graph_replay": 0.0,
        "other": 12345.0,
    }
    assert loaded["kernel_duration_category_shares"] == {
        "attention": 0.0,
        "moe": 0.0,
        "projection": 0.0,
        "sampling": 0.0,
        "graph_replay": 0.0,
        "other": 1.0,
    }
    enriched = retained_bench._attach_profiler_cpu_side_bottlenecks(
        loaded,
        {"load_seconds": 1.0, "prefill_seconds": 2.0, "warmup_seconds": 0.0, "decode_seconds": 7.0},
    )
    assert enriched["cpu_side_total_seconds"] == 10.0
    assert enriched["cpu_side_bottlenecks_seconds"] == {
        "load": 1.0,
        "prefill": 2.0,
        "warmup_decode": 0.0,
        "decode": 7.0,
        "validation": 0.0,
        "other": 0.0,
    }
    assert enriched["cpu_side_bottleneck_shares"] == {
        "load": 0.1,
        "prefill": 0.2,
        "warmup_decode": 0.0,
        "decode": 0.7,
        "validation": 0.0,
        "other": 0.0,
    }
    assert loaded["artifact_path"] == str(profiler_path)
    assert command is not None
    assert command.startswith("rocprofv3 --kernel-trace")
    assert "scripts/qwen35_batch_retained_bench.py" in command
    assert "--json benchmarks/results/native-c2.json" in command
    assert retained_bench._profiler_reference(tmp_path / "missing.json")["status"] == "missing"


def test_qwen35_retained_profiler_reference_synthesizes_durations_from_trace_csv(tmp_path: Path) -> None:
    profiler_path = tmp_path / "profiler-summary.json"
    trace_csv = tmp_path / "hipengine_kernel_trace.csv"
    trace_csv.write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp\n"
        "qwen35_batch_decode,0,100\n"
        "qwen35_batch_decode,100,150\n"
        "qwen35_batch_decode_wmma_caware,150,350\n"
    )
    profiler_path.write_text(
        json.dumps(
            {
                "profiler": {
                    "status": "captured",
                    "trace_files": [str(trace_csv)],
                    "expected_kernels_present": True,
                    "expected_kernel_names": ["qwen35_batch_decode", "qwen35_batch_decode_wmma_caware"],
                }
            }
        )
    )

    loaded = retained_bench._profiler_reference(profiler_path)

    assert loaded["trace_kernel_names"] == ["qwen35_batch_decode", "qwen35_batch_decode_wmma_caware"]
    assert loaded["synthesized_fields"] == [
        "trace_kernel_names",
        "kernel_durations_ns",
        "total_kernel_duration_ns",
        "kernel_duration_shares",
        "kernel_duration_categories_ns",
        "kernel_duration_category_shares",
    ]
    assert loaded["kernel_durations_ns"] == {"qwen35_batch_decode": 150.0, "qwen35_batch_decode_wmma_caware": 200.0}
    assert loaded["total_kernel_duration_ns"] == 350.0
    assert loaded["kernel_duration_shares"] == {
        "qwen35_batch_decode": 150.0 / 350.0,
        "qwen35_batch_decode_wmma_caware": 200.0 / 350.0,
    }
    assert loaded["kernel_duration_categories_ns"] == {
        "attention": 0.0,
        "moe": 0.0,
        "projection": 200.0,
        "sampling": 0.0,
        "graph_replay": 0.0,
        "other": 150.0,
    }


def test_qwen35_retained_payload_mirrors_fallback_native_decode_label(monkeypatch) -> None:
    monkeypatch.setattr(retained_bench, "_hardware_context", lambda: {"gpu": "test"})
    monkeypatch.setattr(retained_bench, "_software_context", lambda: {"python": "test"})
    args = argparse.Namespace(
        batch_size=2,
        prompt_length=512,
        decode_tokens=128,
        warmup_decode_tokens=0,
        max_layers=40,
        json=None,
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
    assert payload["benchmark_rollup"] == {
        "artifact_path": None,
        "readme_path": "benchmarks/README.md",
        "changelog_path": "benchmarks/CHANGELOG.md",
    }
    assert "scripts/qwen35_batch_correctness.py" in payload["commands"]["correctness_reference"]
    assert "--rows 2" in payload["commands"]["correctness_reference"]


def test_qwen35_batch_diagnostic_artifact_schema_enforces_accepted_row_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = {
        "status": "accepted",
        "artifact_path": "benchmarks/results/accepted-c2.json",
        "performance_claim": True,
        "hardware": {
            "gpu": "AMD Radeon Pro W7900",
            "arch": "gfx1100",
            "rocminfo": {
                "command": "rocminfo | grep -E 'Name:|gfx' | head -4",
                "returncode": 0,
                "output": "Name: gfx1100",
            },
            "rocm_smi": {
                "command": "rocm-smi --showmeminfo vram --showuse --showtemp",
                "returncode": 0,
                "output": "GPU[0] VRAM Total Memory",
            },
        },
        "software": {
            "hipengine_commit": "0123456789abcdef0123456789abcdef01234567",
            "hipengine_dirty": False,
            "hipcc_version": "HIP version: 6.4.0",
        },
        "benchmark_rollup": {
            "artifact_path": "benchmarks/results/accepted-c2.json",
            "readme_path": "benchmarks/README.md",
            "changelog_path": "benchmarks/CHANGELOG.md",
        },
        "commands": {
            "environment": [
                "rocminfo | grep -E 'Name:|gfx' | head -4",
                "rocm-smi --showmeminfo vram --showuse --showtemp",
                "hipcc --version",
                "git rev-parse HEAD",
                "git diff --quiet",
            ],
            "benchmark": "python3 scripts/qwen35_batch_retained_bench.py --model /models/test-qwen35 --fixture fixtures/qwen35.json --batch-size 2 --prompt-length 512 --decode-tokens 128 --max-layers 40 --json benchmarks/results/accepted-c2.json --c1-baseline-json benchmarks/results/c1.json --serial-bridge-json benchmarks/results/serial-c2.json --primitive-correctness-json benchmarks/results/primitive-c2.json",
            "correctness_reference": "inline generated-token equality vs independent c=1 plus python3 scripts/qwen35_batch_correctness.py --rows 2 --json benchmarks/results/primitive-c2.json",
            "profiler": "rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-profile -- python3 scripts/qwen35_batch_retained_bench.py --model /models/test-qwen35 --fixture fixtures/qwen35.json --batch-size 2 --prompt-length 512 --decode-tokens 128 --max-layers 40 --compiler-version-file benchmarks/results/hipcc-version.txt --require-cached-build --json benchmarks/results/accepted-c2.json --c1-baseline-json benchmarks/results/c1.json --serial-bridge-json benchmarks/results/serial-c2.json --primitive-correctness-json benchmarks/results/primitive-c2.json --profiler-json benchmarks/results/profiler-c2.json",
        },
        "profiler": {
            "artifact_path": "benchmarks/results/profiler-c2.json",
            "status": "captured",
            "output_format": "csv",
            "trace_dir": "/tmp/hipengine-profile",
            "trace_files": ["/tmp/hipengine-profile/hipengine_kernel_trace.csv"],
            "trace_kernel_names": ["qwen35_batch_decode", "qwen35_batch_decode_wmma_caware"],
            "synthesized_fields": [],
            "expected_kernels_present": True,
            "expected_kernel_names": ["qwen35_batch_decode", "qwen35_batch_decode_wmma_caware"],
            "kernel_durations_ns": {
                "qwen35_batch_decode": 12345.0,
                "qwen35_batch_decode_wmma_caware": 2345.0,
            },
            "total_kernel_duration_ns": 14690.0,
            "kernel_duration_shares": {
                "qwen35_batch_decode": 12345.0 / 14690.0,
                "qwen35_batch_decode_wmma_caware": 2345.0 / 14690.0,
            },
            "kernel_duration_categories_ns": {
                "attention": 0.0,
                "moe": 0.0,
                "projection": 2345.0,
                "sampling": 0.0,
                "graph_replay": 0.0,
                "other": 12345.0,
            },
            "kernel_duration_category_shares": {
                "attention": 0.0,
                "moe": 0.0,
                "projection": 2345.0 / 14690.0,
                "sampling": 0.0,
                "graph_replay": 0.0,
                "other": 12345.0 / 14690.0,
            },
            "cpu_side_total_seconds": 10.0,
            "cpu_side_bottlenecks_seconds": {
                "load": 1.0,
                "prefill": 2.0,
                "warmup_decode": 0.0,
                "decode": 7.0,
                "validation": 0.0,
                "other": 0.0,
            },
            "cpu_side_bottleneck_shares": {
                "load": 0.1,
                "prefill": 0.2,
                "warmup_decode": 0.0,
                "decode": 0.7,
                "validation": 0.0,
                "other": 0.0,
            },
        },
        "workload": {
            "model": "Qwen3.5/3.6-35B-A3B-PARO",
            "quant": "w4_paro",
            "kv_storage_dtype": "bf16",
            "kv_policy": {"policy_class": "FixedPagedKVPolicy", "storage_dtype": "bf16"},
            "concurrency": 2,
            "prompt_tokens_per_request": 512,
            "prompt_tokens_aggregate": 1024,
            "prompt_lengths": [512, 512],
            "gen_tokens_per_request": 128,
            "gen_tokens_aggregate": 256,
            "warmup_decode_tokens": 8,
            "max_layers": 40,
            "scheduler_path": "scheduler_native_compact_batch",
            "native_compact_prefill": True,
            "native_caware_decode": True,
        },
        "correctness": {
            "passed": True,
            "generated_token_equality": {
                "passed": True,
                "skipped": False,
                "batch_sequences": [list(range(0, 137)), list(range(100, 237))],
                "c1_sequences": [list(range(0, 137)), list(range(100, 237))],
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
                "projection_dispatch": {
                    "rows": 2,
                    "selected_candidate": "wmma_caware",
                    "path": "benchmark_accepted_caware_projection",
                    "selection": {"layer": "linear", "quant": "w4_paro", "variant": "wmma_caware"},
                    "throughput_claim_eligible": True,
                    "blockers": [],
                    "evidence": {
                        "artifact_path": "benchmarks/results/projection-wmma-c2.json",
                        "aggregate_vs_row_gemv": 1.35,
                        "per_request_vs_row_gemv": 1.10,
                        "accepted": True,
                    },
                },
                "decode_execution": {
                    "full_attention_decode_path": "native_batch",
                    "native_caware_decode": True,
                    "sampler_execution": {
                        "requested_mode": "batched_lm_head",
                        "mode": "batched_lm_head",
                        "native_row_aware_lm_head": True,
                        "c2_equality_green": True,
                        "equality_artifact": "benchmarks/results/qwen35-c2-sampler-eq.json",
                        "blockers": [],
                    },
                },
            },
            "scheduler_metadata": {
                "decode_shape_key": {
                    "mode": "decode",
                    "active_c": 2,
                    "context_bucket": 512,
                    "active_mask": [True, True],
                    "top_k": 0,
                    "experts_per_token": 0,
                    "replay_steps": 1,
                    "draft_depth": 0,
                    "tree_shape": [],
                },
                "graph_bucket_stats": {
                    "entries": 1,
                    "hits": 1,
                    "misses": 1,
                    "miss_reasons": {"cache_absent": 1},
                    "kernel_time_histogram_ns": {},
                },
            },
            "seed_tokens": {"0": {"token_id": 0}, "1": {"token_id": 100}},
            "generated_tokens": {
                "0": [{"token_id": token} for token in range(9, 137)],
                "1": [{"token_id": token} for token in range(109, 237)],
            },
            "completed": [
                {
                    "request_id": 0,
                    "prompt_tokens": list(range(512)),
                    "generated_tokens": list(range(9, 137)),
                    "finished": True,
                    "finish_reason": "length",
                },
                {
                    "request_id": 1,
                    "prompt_tokens": list(range(512, 1024)),
                    "generated_tokens": list(range(109, 237)),
                    "finished": True,
                    "finish_reason": "length",
                },
            ],
        },
        "observability": {
            "admission_timestamps": {"0": 1.0, "1": 1.1},
            "completion_timestamps": {"0": 2.0, "1": 2.2},
            "request_latency_seconds": {"p50": 1.0, "p95": 1.25, "samples": [1.0, 1.1]},
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
        "projection_dispatch_candidates": [
            {
                "name": "wmma_caware",
                "selection": {"layer": "linear", "quant": "w4_paro", "variant": "wmma_caware"},
                "min_rows": 2,
                "max_rows": 8,
                "evidence": {
                    "artifact_path": "benchmarks/results/projection-wmma-c2.json",
                    "aggregate_vs_row_gemv": 1.35,
                    "per_request_vs_row_gemv": 1.10,
                    "accepted": True,
                },
            }
        ],
        "decision": {"accepted": True},
    }

    validate_cn_diagnostic_artifact_payload(accepted)

    projection_candidate = {
        "name": "wmma_caware",
        "selection": {"layer": "linear", "quant": "w4_paro", "variant": "wmma_caware"},
        "min_rows": 2,
        "max_rows": 8,
        "evidence": {
            "artifact_path": "benchmarks/results/projection-wmma-c2.json",
            "aggregate_vs_row_gemv": 1.35,
            "per_request_vs_row_gemv": 1.10,
            "accepted": True,
        },
    }
    with_projection_candidates = json.loads(json.dumps(accepted))
    with_projection_candidates["projection_dispatch_candidates"] = [projection_candidate]
    validate_cn_diagnostic_artifact_payload(with_projection_candidates)

    malformed_projection_candidates = json.loads(json.dumps(with_projection_candidates))
    malformed_projection_candidates["projection_dispatch_candidates"][0]["evidence"]["per_request_vs_row_gemv"] = 0.0
    with pytest.raises(ValueError, match="invalid projection_dispatch_candidates"):
        validate_cn_diagnostic_artifact_payload(malformed_projection_candidates)

    slow_projection_candidates = json.loads(json.dumps(with_projection_candidates))
    slow_projection_candidates["projection_dispatch_candidates"][0]["evidence"]["per_request_vs_row_gemv"] = 1.0
    with pytest.raises(ValueError, match="accepted per_request_vs_row_gemv must be > 1.0"):
        validate_cn_diagnostic_artifact_payload(slow_projection_candidates)

    rollup_root = tmp_path / "rollup-repo"
    (rollup_root / "benchmarks").mkdir(parents=True)
    (rollup_root / "benchmarks" / "README.md").write_text(
        f"retained row: `{accepted['artifact_path']}`\n",
        encoding="utf-8",
    )
    (rollup_root / "benchmarks" / "CHANGELOG.md").write_text(
        f"- retained c>N row; `{accepted['artifact_path']}`\n",
        encoding="utf-8",
    )
    artifact_file = rollup_root / "benchmarks" / "results" / "accepted-c2.json"
    artifact_file.parent.mkdir()
    artifact_file.write_text(json.dumps(accepted), encoding="utf-8")
    monkeypatch.chdir(rollup_root)
    validate_cn_diagnostic_rollup_evidence(accepted)
    summary_file = rollup_root / "benchmarks" / "results" / "accepted-c2-rollup-check.json"
    assert validate_cn_diagnostic_artifact_main(
        [str(artifact_file), "--rollup-evidence", "--summary-json", str(summary_file)]
    ) == 0
    summary = json.loads(summary_file.read_text())
    assert summary == {
        "schema": 1,
        "mode": "rollup_evidence",
        "passed": True,
        "artifact_json": str(artifact_file),
        "artifact_path": "benchmarks/results/accepted-c2.json",
        "status": "accepted",
        "performance_claim": True,
        "benchmark_rollup": accepted["benchmark_rollup"],
        "error": None,
    }
    validate_cn_diagnostic_validation_summary(summary)
    assert validate_cn_diagnostic_artifact_main([str(summary_file), "--validation-summary"]) == 0

    invalid_summary = dict(summary)
    invalid_summary["error"] = "unexpected warning"
    with pytest.raises(ValueError, match="summary.error must be null"):
        validate_cn_diagnostic_validation_summary(invalid_summary)
    invalid_summary_file = rollup_root / "benchmarks" / "results" / "accepted-c2-invalid-summary.json"
    invalid_summary_file.write_text(json.dumps(invalid_summary), encoding="utf-8")
    assert validate_cn_diagnostic_artifact_main([str(invalid_summary_file), "--validation-summary"]) == 1

    missing_rollup_artifact = json.loads(json.dumps(accepted))
    missing_rollup_artifact.pop("benchmark_rollup")
    missing_rollup_file = rollup_root / "benchmarks" / "results" / "accepted-c2-missing-rollup.json"
    missing_rollup_file.write_text(json.dumps(missing_rollup_artifact), encoding="utf-8")
    failed_summary_file = rollup_root / "benchmarks" / "results" / "accepted-c2-missing-rollup-check.json"
    assert validate_cn_diagnostic_artifact_main(
        [str(missing_rollup_file), "--rollup-evidence", "--summary-json", str(failed_summary_file)]
    ) == 1
    failed_summary = json.loads(failed_summary_file.read_text())
    assert failed_summary["passed"] is False
    assert failed_summary["mode"] == "rollup_evidence"
    assert failed_summary["artifact_json"] == str(missing_rollup_file)
    assert failed_summary["artifact_path"] == "benchmarks/results/accepted-c2.json"
    assert failed_summary["benchmark_rollup"] is None
    assert "benchmark_rollup must be an object" in failed_summary["error"]
    validate_cn_diagnostic_validation_summary(failed_summary)
    assert validate_cn_diagnostic_artifact_main([str(failed_summary_file), "--validation-summary"]) == 0

    invalid_failed_summary = dict(failed_summary)
    invalid_failed_summary["error"] = None
    with pytest.raises(ValueError, match="summary.error must be a non-empty string"):
        validate_cn_diagnostic_validation_summary(invalid_failed_summary)

    bad_summary_file = rollup_root / "tmp" / "accepted-c2-rollup-check.json"
    assert validate_cn_diagnostic_artifact_main(
        [str(artifact_file), "--rollup-evidence", "--summary-json", str(bad_summary_file)]
    ) == 1
    assert not bad_summary_file.exists()

    missing_rollup = json.loads(json.dumps(accepted))
    missing_rollup.pop("benchmark_rollup")
    with pytest.raises(ValueError, match="benchmark_rollup must be an object"):
        validate_cn_diagnostic_rollup_evidence(missing_rollup)

    wrong_rollup_artifact = json.loads(json.dumps(accepted))
    wrong_rollup_artifact["benchmark_rollup"]["artifact_path"] = "benchmarks/results/other-accepted-c2.json"
    with pytest.raises(ValueError, match="benchmark_rollup.artifact_path must match artifact_path"):
        validate_cn_diagnostic_rollup_evidence(wrong_rollup_artifact)

    wrong_rollup_path = json.loads(json.dumps(accepted))
    wrong_rollup_path["benchmark_rollup"]["readme_path"] = "README.md"
    with pytest.raises(ValueError, match="benchmark_rollup.readme_path must be benchmarks/README.md"):
        validate_cn_diagnostic_rollup_evidence(wrong_rollup_path)

    (rollup_root / "benchmarks" / "CHANGELOG.md").write_text(
        "- retained c>N row without an artifact link\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="benchmark_rollup.changelog_path must mention artifact_path"):
        validate_cn_diagnostic_rollup_evidence(accepted)
    (rollup_root / "benchmarks" / "CHANGELOG.md").write_text(
        f"- retained c>N row; `{accepted['artifact_path']}`\n",
        encoding="utf-8",
    )

    missing_artifact_path = json.loads(json.dumps(accepted))
    missing_artifact_path.pop("artifact_path")
    with pytest.raises(ValueError, match="artifact_path must be a non-empty string"):
        validate_cn_diagnostic_artifact_payload(missing_artifact_path)

    tmp_artifact_path = json.loads(json.dumps(accepted))
    tmp_artifact_path["artifact_path"] = "/tmp/accepted-c2.json"
    with pytest.raises(ValueError, match="artifact_path must be under benchmarks/results"):
        validate_cn_diagnostic_artifact_payload(tmp_artifact_path)

    mismatched_benchmark_artifact_path = json.loads(json.dumps(accepted))
    mismatched_benchmark_artifact_path["commands"]["benchmark"] = mismatched_benchmark_artifact_path["commands"]["benchmark"].replace("benchmarks/results/accepted-c2.json", "benchmarks/results/other-accepted-c2.json")
    with pytest.raises(ValueError, match="commands.benchmark --json path must match artifact_path"):
        validate_cn_diagnostic_artifact_payload(mismatched_benchmark_artifact_path)

    mismatched_profiler_artifact_path = json.loads(json.dumps(accepted))
    mismatched_profiler_artifact_path["commands"]["profiler"] = mismatched_profiler_artifact_path["commands"]["profiler"].replace("benchmarks/results/accepted-c2.json", "benchmarks/results/other-accepted-c2.json")
    with pytest.raises(ValueError, match="commands.profiler --json path must match artifact_path"):
        validate_cn_diagnostic_artifact_payload(mismatched_profiler_artifact_path)

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

    truncated_decode_equality = json.loads(json.dumps(accepted))
    truncated_decode_equality["correctness"]["generated_token_equality"]["batch_sequences"][0] = [10, 11]
    truncated_decode_equality["correctness"]["generated_token_equality"]["c1_sequences"][0] = [10, 11]
    with pytest.raises(ValueError, match=r"batch_sequences\[0\] length must match seed plus workload.warmup_decode_tokens plus workload.gen_tokens_per_request"):
        validate_cn_diagnostic_artifact_payload(truncated_decode_equality)

    missing_execution_seed_tokens = json.loads(json.dumps(accepted))
    missing_execution_seed_tokens["execution"].pop("seed_tokens")
    with pytest.raises(ValueError, match="execution.seed_tokens must be an object"):
        validate_cn_diagnostic_artifact_payload(missing_execution_seed_tokens)

    mismatched_execution_seed_tokens = json.loads(json.dumps(accepted))
    mismatched_execution_seed_tokens["execution"]["seed_tokens"]["0"]["token_id"] = 999
    with pytest.raises(ValueError, match="execution.seed_tokens.0 must match correctness.generated_token_equality.batch_sequences first token"):
        validate_cn_diagnostic_artifact_payload(mismatched_execution_seed_tokens)

    missing_execution_generated_tokens = json.loads(json.dumps(accepted))
    missing_execution_generated_tokens["execution"].pop("generated_tokens")
    with pytest.raises(ValueError, match="execution.generated_tokens must be an object"):
        validate_cn_diagnostic_artifact_payload(missing_execution_generated_tokens)

    mismatched_execution_generated_tokens = json.loads(json.dumps(accepted))
    mismatched_execution_generated_tokens["execution"]["generated_tokens"]["0"][0]["token_id"] = 999
    with pytest.raises(ValueError, match="execution.generated_tokens.0 must match correctness.generated_token_equality.batch_sequences suffix"):
        validate_cn_diagnostic_artifact_payload(mismatched_execution_generated_tokens)

    missing_completed_requests = json.loads(json.dumps(accepted))
    missing_completed_requests["execution"].pop("completed")
    with pytest.raises(ValueError, match="execution.completed must be a list"):
        validate_cn_diagnostic_artifact_payload(missing_completed_requests)

    mismatched_completed_tokens = json.loads(json.dumps(accepted))
    mismatched_completed_tokens["execution"]["completed"][0]["generated_tokens"][0] = 999
    with pytest.raises(ValueError, match="execution.completed request_id 0 generated_tokens must match execution.generated_tokens"):
        validate_cn_diagnostic_artifact_payload(mismatched_completed_tokens)

    missing_completed_prompt_tokens = json.loads(json.dumps(accepted))
    missing_completed_prompt_tokens["execution"]["completed"][0].pop("prompt_tokens")
    with pytest.raises(ValueError, match=r"execution.completed\[0\].prompt_tokens must be a list"):
        validate_cn_diagnostic_artifact_payload(missing_completed_prompt_tokens)

    mismatched_completed_prompt_length = json.loads(json.dumps(accepted))
    mismatched_completed_prompt_length["execution"]["completed"][0]["prompt_tokens"] = [1, 2]
    with pytest.raises(ValueError, match=r"execution.completed\[0\].prompt_tokens length must match workload.prompt_lengths"):
        validate_cn_diagnostic_artifact_payload(mismatched_completed_prompt_length)

    mismatched_completion_finish_reason = json.loads(json.dumps(accepted))
    mismatched_completion_finish_reason["execution"]["completed"][0]["finish_reason"] = "stop"
    with pytest.raises(ValueError, match="execution.completed request_id 0 finish_reason must match observability.per_request"):
        validate_cn_diagnostic_artifact_payload(mismatched_completion_finish_reason)

    missing_warmup_decode_tokens = json.loads(json.dumps(accepted))
    missing_warmup_decode_tokens["workload"].pop("warmup_decode_tokens")
    with pytest.raises(ValueError, match="warmup_decode_tokens must be a non-negative int"):
        validate_cn_diagnostic_artifact_payload(missing_warmup_decode_tokens)

    missing_primitive = json.loads(json.dumps(accepted))
    missing_primitive["correctness"].pop("primitive_batch_correctness")
    with pytest.raises(ValueError, match="primitive_batch_correctness"):
        validate_cn_diagnostic_artifact_payload(missing_primitive)

    failed_primitive = json.loads(json.dumps(accepted))
    failed_primitive["correctness"]["primitive_batch_correctness"]["passed"] = False
    with pytest.raises(ValueError, match="primitive_batch_correctness.passed"):
        validate_cn_diagnostic_artifact_payload(failed_primitive)

    primitive_tmp_artifact = json.loads(json.dumps(accepted))
    primitive_tmp_artifact["correctness"]["primitive_batch_correctness"]["artifact_path"] = "/tmp/primitive-c2.json"
    with pytest.raises(ValueError, match="primitive_batch_correctness.artifact_path must be under benchmarks/results"):
        validate_cn_diagnostic_artifact_payload(primitive_tmp_artifact)

    mismatched_primitive_rows = json.loads(json.dumps(accepted))
    mismatched_primitive_rows["correctness"]["primitive_batch_correctness"]["rows"] = 8
    with pytest.raises(ValueError, match="primitive_batch_correctness.rows must match workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(mismatched_primitive_rows)

    primitive_append_mismatch = json.loads(json.dumps(accepted))
    primitive_append_mismatch["correctness"]["primitive_batch_correctness"]["append_key_mismatch"] = 1
    with pytest.raises(ValueError, match="primitive_batch_correctness.append_key_mismatch must be 0"):
        validate_cn_diagnostic_artifact_payload(primitive_append_mismatch)

    primitive_attn_mismatch = json.loads(json.dumps(accepted))
    primitive_attn_mismatch["correctness"]["primitive_batch_correctness"]["attn_batch_vs_c1_max_abs"] = 0.25
    with pytest.raises(ValueError, match="primitive_batch_correctness.attn_batch_vs_c1_max_abs must be 0.0"):
        validate_cn_diagnostic_artifact_payload(primitive_attn_mismatch)

    missing_workload_concurrency = json.loads(json.dumps(accepted))
    missing_workload_concurrency["workload"].pop("concurrency")
    with pytest.raises(ValueError, match="workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(missing_workload_concurrency)

    missing_prompt_aggregate = json.loads(json.dumps(accepted))
    missing_prompt_aggregate["workload"].pop("prompt_tokens_aggregate")
    with pytest.raises(ValueError, match="workload.prompt_tokens_aggregate"):
        validate_cn_diagnostic_artifact_payload(missing_prompt_aggregate)

    mismatched_decode_aggregate = json.loads(json.dumps(accepted))
    mismatched_decode_aggregate["workload"]["gen_tokens_aggregate"] = 128
    with pytest.raises(ValueError, match="gen_tokens_aggregate must equal per-request tokens times workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(mismatched_decode_aggregate)

    missing_prompt_lengths = json.loads(json.dumps(accepted))
    missing_prompt_lengths["workload"].pop("prompt_lengths")
    with pytest.raises(ValueError, match="workload.prompt_lengths"):
        validate_cn_diagnostic_artifact_payload(missing_prompt_lengths)

    mismatched_prompt_lengths = json.loads(json.dumps(accepted))
    mismatched_prompt_lengths["workload"]["prompt_lengths"] = [512, 256]
    with pytest.raises(ValueError, match="prompt_lengths entries must match workload.prompt_tokens_per_request"):
        validate_cn_diagnostic_artifact_payload(mismatched_prompt_lengths)

    missing_max_layers = json.loads(json.dumps(accepted))
    missing_max_layers["workload"].pop("max_layers")
    with pytest.raises(ValueError, match="workload.max_layers"):
        validate_cn_diagnostic_artifact_payload(missing_max_layers)

    reduced_max_layers = json.loads(json.dumps(accepted))
    reduced_max_layers["workload"]["max_layers"] = 8
    with pytest.raises(ValueError, match="workload.max_layers must be 40"):
        validate_cn_diagnostic_artifact_payload(reduced_max_layers)

    for workload_label in ("model", "quant", "kv_storage_dtype"):
        missing_workload_label = json.loads(json.dumps(accepted))
        missing_workload_label["workload"].pop(workload_label)
        with pytest.raises(ValueError, match=f"workload.{workload_label}"):
            validate_cn_diagnostic_artifact_payload(missing_workload_label)

    missing_kv_policy = json.loads(json.dumps(accepted))
    missing_kv_policy["workload"].pop("kv_policy")
    with pytest.raises(ValueError, match="workload.kv_policy"):
        validate_cn_diagnostic_artifact_payload(missing_kv_policy)

    mismatched_kv_policy = json.loads(json.dumps(accepted))
    mismatched_kv_policy["workload"]["kv_policy"]["storage_dtype"] = "int8_per_token_head"
    with pytest.raises(ValueError, match="kv_policy.storage_dtype must match workload.kv_storage_dtype"):
        validate_cn_diagnostic_artifact_payload(mismatched_kv_policy)

    serial_bridge_execution = json.loads(json.dumps(accepted))
    serial_bridge_execution["execution"]["batch_execution"]["path"] = "scheduler_serial_slot_bridge"
    with pytest.raises(ValueError, match="serial bridge"):
        validate_cn_diagnostic_artifact_payload(serial_bridge_execution)

    missing_scheduler_path = json.loads(json.dumps(accepted))
    missing_scheduler_path["workload"].pop("scheduler_path")
    with pytest.raises(ValueError, match="workload.scheduler_path"):
        validate_cn_diagnostic_artifact_payload(missing_scheduler_path)

    mismatched_scheduler_path = json.loads(json.dumps(accepted))
    mismatched_scheduler_path["workload"]["scheduler_path"] = "scheduler_native_other"
    with pytest.raises(ValueError, match="scheduler_path must match execution.batch_execution.path"):
        validate_cn_diagnostic_artifact_payload(mismatched_scheduler_path)

    fallback_execution = json.loads(json.dumps(accepted))
    fallback_execution["execution"]["batch_execution"]["row_execution"] = "native_linear_batch_with_per_row_full_attention_fallback"
    with pytest.raises(ValueError, match="serial or fallback"):
        validate_cn_diagnostic_artifact_payload(fallback_execution)

    non_native_decode = json.loads(json.dumps(accepted))
    non_native_decode["execution"]["batch_execution"]["native_caware_decode"] = False
    with pytest.raises(ValueError, match="native_caware_decode"):
        validate_cn_diagnostic_artifact_payload(non_native_decode)

    missing_decode_execution = json.loads(json.dumps(accepted))
    missing_decode_execution["execution"]["batch_execution"].pop("decode_execution")
    with pytest.raises(ValueError, match="decode_execution must be an object"):
        validate_cn_diagnostic_artifact_payload(missing_decode_execution)

    per_row_splitk = json.loads(json.dumps(accepted))
    per_row_splitk["execution"]["batch_execution"]["decode_execution"]["full_attention_decode_path"] = "per_row_splitk_fallback"
    with pytest.raises(ValueError, match="full_attention_decode_path must be native_batch"):
        validate_cn_diagnostic_artifact_payload(per_row_splitk)

    non_native_decode_execution = json.loads(json.dumps(accepted))
    non_native_decode_execution["execution"]["batch_execution"]["decode_execution"]["native_caware_decode"] = False
    with pytest.raises(ValueError, match="decode_execution.native_caware_decode must be true"):
        validate_cn_diagnostic_artifact_payload(non_native_decode_execution)

    missing_sampler_execution = json.loads(json.dumps(accepted))
    missing_sampler_execution["execution"]["batch_execution"]["decode_execution"].pop("sampler_execution")
    with pytest.raises(ValueError, match="sampler_execution must be an object"):
        validate_cn_diagnostic_artifact_payload(missing_sampler_execution)

    serial_sampler = json.loads(json.dumps(accepted))
    serial_sampler["execution"]["batch_execution"]["decode_execution"]["sampler_execution"]["native_row_aware_lm_head"] = False
    with pytest.raises(ValueError, match="native_row_aware_lm_head"):
        validate_cn_diagnostic_artifact_payload(serial_sampler)

    serial_sampler_mode = json.loads(json.dumps(accepted))
    serial_sampler_mode["execution"]["batch_execution"]["decode_execution"]["sampler_execution"]["mode"] = "serial_lm_head"
    with pytest.raises(ValueError, match="sampler_execution.mode must be batched_lm_head"):
        validate_cn_diagnostic_artifact_payload(serial_sampler_mode)

    missing_sampler_equality = json.loads(json.dumps(accepted))
    missing_sampler_equality["execution"]["batch_execution"]["decode_execution"]["sampler_execution"]["c2_equality_green"] = False
    with pytest.raises(ValueError, match="sampler_execution.c2_equality_green must be true"):
        validate_cn_diagnostic_artifact_payload(missing_sampler_equality)

    tmp_sampler_artifact = json.loads(json.dumps(accepted))
    tmp_sampler_artifact["execution"]["batch_execution"]["decode_execution"]["sampler_execution"]["equality_artifact"] = "/tmp/qwen35-c2-sampler-eq.json"
    with pytest.raises(ValueError, match="sampler_execution.equality_artifact must be under benchmarks/results"):
        validate_cn_diagnostic_artifact_payload(tmp_sampler_artifact)

    blocked_sampler = json.loads(json.dumps(accepted))
    blocked_sampler["execution"]["batch_execution"]["decode_execution"]["sampler_execution"]["blockers"] = ["missing retained sampler evidence"]
    with pytest.raises(ValueError, match="sampler_execution.blockers must be empty"):
        validate_cn_diagnostic_artifact_payload(blocked_sampler)

    missing_projection_dispatch = json.loads(json.dumps(accepted))
    missing_projection_dispatch["execution"]["batch_execution"].pop("projection_dispatch")
    with pytest.raises(ValueError, match="projection_dispatch must be an object"):
        validate_cn_diagnostic_artifact_payload(missing_projection_dispatch)

    missing_projection_candidate_list = json.loads(json.dumps(accepted))
    missing_projection_candidate_list.pop("projection_dispatch_candidates")
    with pytest.raises(ValueError, match="projection_dispatch_candidates must include selected projection candidate"):
        validate_cn_diagnostic_artifact_payload(missing_projection_candidate_list)

    unlisted_projection_candidate = json.loads(json.dumps(accepted))
    unlisted_projection_candidate["projection_dispatch_candidates"][0]["name"] = "other_caware"
    with pytest.raises(ValueError, match="projection_dispatch_candidates must include selected_candidate"):
        validate_cn_diagnostic_artifact_payload(unlisted_projection_candidate)

    row_gemv_projection_dispatch = json.loads(json.dumps(accepted))
    row_gemv_projection_dispatch["execution"]["batch_execution"]["projection_dispatch"].update(
        {
            "selected_candidate": "row_gemv",
            "path": "row_gemv_until_caware_benchmark",
            "selection": {"layer": "linear", "quant": "w4_paro", "variant": "row_gemv"},
            "throughput_claim_eligible": False,
            "blockers": ["no c-aware projection candidate applies to this row count"],
            "evidence": None,
        }
    )
    with pytest.raises(ValueError, match="projection_dispatch.path must be benchmark_accepted_caware_projection"):
        validate_cn_diagnostic_artifact_payload(row_gemv_projection_dispatch)

    missing_decode_shape_key = json.loads(json.dumps(accepted))
    missing_decode_shape_key["execution"]["scheduler_metadata"].pop("decode_shape_key")
    with pytest.raises(ValueError, match="execution.scheduler_metadata.decode_shape_key"):
        validate_cn_diagnostic_artifact_payload(missing_decode_shape_key)

    mismatched_decode_shape_key = json.loads(json.dumps(accepted))
    mismatched_decode_shape_key["execution"]["scheduler_metadata"]["decode_shape_key"]["active_c"] = 8
    with pytest.raises(ValueError, match="decode_shape_key.active_c must match workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(mismatched_decode_shape_key)

    missing_decode_context_bucket = json.loads(json.dumps(accepted))
    missing_decode_context_bucket["execution"]["scheduler_metadata"]["decode_shape_key"].pop("context_bucket")
    with pytest.raises(ValueError, match="decode_shape_key.context_bucket must be a positive int"):
        validate_cn_diagnostic_artifact_payload(missing_decode_context_bucket)

    invalid_decode_top_k = json.loads(json.dumps(accepted))
    invalid_decode_top_k["execution"]["scheduler_metadata"]["decode_shape_key"]["top_k"] = -1
    with pytest.raises(ValueError, match="decode_shape_key.top_k must be a non-negative int"):
        validate_cn_diagnostic_artifact_payload(invalid_decode_top_k)

    invalid_decode_replay_steps = json.loads(json.dumps(accepted))
    invalid_decode_replay_steps["execution"]["scheduler_metadata"]["decode_shape_key"]["replay_steps"] = 0
    with pytest.raises(ValueError, match="decode_shape_key.replay_steps must be a positive int"):
        validate_cn_diagnostic_artifact_payload(invalid_decode_replay_steps)

    invalid_decode_tree_shape = json.loads(json.dumps(accepted))
    invalid_decode_tree_shape["execution"]["scheduler_metadata"]["decode_shape_key"]["tree_shape"] = [2, -1]
    with pytest.raises(ValueError, match="decode_shape_key.tree_shape must be a list of non-negative ints"):
        validate_cn_diagnostic_artifact_payload(invalid_decode_tree_shape)

    missing_graph_bucket_stats = json.loads(json.dumps(accepted))
    missing_graph_bucket_stats["execution"]["scheduler_metadata"].pop("graph_bucket_stats")
    with pytest.raises(ValueError, match="execution.scheduler_metadata.graph_bucket_stats"):
        validate_cn_diagnostic_artifact_payload(missing_graph_bucket_stats)

    negative_graph_bucket_hits = json.loads(json.dumps(accepted))
    negative_graph_bucket_hits["execution"]["scheduler_metadata"]["graph_bucket_stats"]["hits"] = -1
    with pytest.raises(ValueError, match="graph_bucket_stats.hits must be a non-negative int"):
        validate_cn_diagnostic_artifact_payload(negative_graph_bucket_hits)

    empty_graph_bucket_cache = json.loads(json.dumps(accepted))
    empty_graph_bucket_cache["execution"]["scheduler_metadata"]["graph_bucket_stats"]["entries"] = 0
    with pytest.raises(ValueError, match="graph_bucket_stats.entries must be positive"):
        validate_cn_diagnostic_artifact_payload(empty_graph_bucket_cache)

    missing_graph_bucket_miss_reasons = json.loads(json.dumps(accepted))
    missing_graph_bucket_miss_reasons["execution"]["scheduler_metadata"]["graph_bucket_stats"].pop("miss_reasons")
    with pytest.raises(ValueError, match="graph_bucket_stats.miss_reasons"):
        validate_cn_diagnostic_artifact_payload(missing_graph_bucket_miss_reasons)

    mismatched_graph_bucket_miss_reasons = json.loads(json.dumps(accepted))
    mismatched_graph_bucket_miss_reasons["execution"]["scheduler_metadata"]["graph_bucket_stats"]["miss_reasons"] = {"cache_absent": 2}
    with pytest.raises(ValueError, match="miss_reasons counts must sum to misses"):
        validate_cn_diagnostic_artifact_payload(mismatched_graph_bucket_miss_reasons)

    invalid_graph_bucket_histogram = json.loads(json.dumps(accepted))
    invalid_graph_bucket_histogram["execution"]["scheduler_metadata"]["graph_bucket_stats"]["kernel_time_histogram_ns"] = {"le_10us": -1}
    with pytest.raises(ValueError, match="kernel_time_histogram_ns.le_10us must be a non-negative int"):
        validate_cn_diagnostic_artifact_payload(invalid_graph_bucket_histogram)

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
        "request_latency_seconds": {"p50": 1.0, "p95": 1.25, "samples": [1.0, 1.1]},
    }
    with pytest.raises(ValueError, match="per_request"):
        validate_cn_diagnostic_artifact_payload(missing_per_request)

    missing_latency_samples = json.loads(json.dumps(accepted))
    missing_latency_samples["observability"]["request_latency_seconds"].pop("samples")
    with pytest.raises(ValueError, match="request_latency_seconds.samples"):
        validate_cn_diagnostic_artifact_payload(missing_latency_samples)

    short_latency_samples = json.loads(json.dumps(accepted))
    short_latency_samples["observability"]["request_latency_seconds"]["samples"] = [1.0]
    with pytest.raises(ValueError, match="request_latency_seconds.samples length must match workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(short_latency_samples)

    zero_latency_sample = json.loads(json.dumps(accepted))
    zero_latency_sample["observability"]["request_latency_seconds"]["samples"][0] = 0.0
    with pytest.raises(ValueError, match="request_latency_seconds.samples must contain only positive numbers"):
        validate_cn_diagnostic_artifact_payload(zero_latency_sample)

    short_admission_timestamps = json.loads(json.dumps(accepted))
    short_admission_timestamps["observability"]["admission_timestamps"].pop("1")
    with pytest.raises(ValueError, match="admission_timestamps length must match workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(short_admission_timestamps)

    short_completion_timestamps = json.loads(json.dumps(accepted))
    short_completion_timestamps["observability"]["completion_timestamps"].pop("1")
    with pytest.raises(ValueError, match="completion_timestamps length must match workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(short_completion_timestamps)

    mismatched_observability_keys = json.loads(json.dumps(accepted))
    mismatched_observability_keys["observability"]["completion_timestamps"].pop("1")
    mismatched_observability_keys["observability"]["completion_timestamps"]["2"] = 2.3
    with pytest.raises(ValueError, match="completion_timestamps keys must match observability.per_request keys"):
        validate_cn_diagnostic_artifact_payload(mismatched_observability_keys)

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

    non_winning_c1_aggregate = json.loads(json.dumps(accepted))
    non_winning_c1_aggregate["scaling"]["c1_baseline"]["decode_tok_s_aggregate"] = 100.0
    non_winning_c1_aggregate["scaling"]["c1_baseline"]["decode_tok_s_per_request"] = 100.0
    non_winning_c1_aggregate["scaling"]["ratios"]["aggregate_vs_c1"] = 1.0
    non_winning_c1_aggregate["scaling"]["ratios"]["per_request_vs_c1"] = 0.5
    with pytest.raises(ValueError, match="aggregate_vs_c1 must be > 1.0"):
        validate_cn_diagnostic_artifact_payload(non_winning_c1_aggregate)

    non_winning_serial_aggregate = json.loads(json.dumps(accepted))
    non_winning_serial_aggregate["scaling"]["serial_bridge_baseline"]["decode_tok_s_aggregate"] = 100.0
    non_winning_serial_aggregate["scaling"]["serial_bridge_baseline"]["decode_tok_s_per_request"] = 50.0
    non_winning_serial_aggregate["scaling"]["ratios"]["aggregate_vs_serial_bridge"] = 1.0
    non_winning_serial_aggregate["scaling"]["ratios"]["per_request_vs_serial_bridge"] = 1.0
    with pytest.raises(ValueError, match="aggregate_vs_serial_bridge must be > 1.0"):
        validate_cn_diagnostic_artifact_payload(non_winning_serial_aggregate)

    inconsistent_native_rate = json.loads(json.dumps(accepted))
    inconsistent_native_rate["scaling"]["native"]["decode_tok_s_aggregate"] = 99.0
    with pytest.raises(ValueError, match="scaling.native.decode_tok_s_aggregate must match measurements.decode_tok_s_aggregate"):
        validate_cn_diagnostic_artifact_payload(inconsistent_native_rate)

    inconsistent_measurement_rate = json.loads(json.dumps(accepted))
    inconsistent_measurement_rate["measurements"]["decode_tok_s_per_request"] = 40.0
    inconsistent_measurement_rate["scaling"]["native"]["decode_tok_s_per_request"] = 40.0
    with pytest.raises(ValueError, match="measurements.decode_tok_s_aggregate must match decode_tok_s_per_request times concurrency"):
        validate_cn_diagnostic_artifact_payload(inconsistent_measurement_rate)

    inconsistent_serial_rate = json.loads(json.dumps(accepted))
    inconsistent_serial_rate["scaling"]["serial_bridge_baseline"]["decode_tok_s_per_request"] = 30.0
    with pytest.raises(ValueError, match="serial_bridge_baseline.decode_tok_s_aggregate must match decode_tok_s_per_request times concurrency"):
        validate_cn_diagnostic_artifact_payload(inconsistent_serial_rate)

    missing_c1_status = json.loads(json.dumps(accepted))
    missing_c1_status["scaling"]["c1_baseline"].pop("status")
    with pytest.raises(ValueError, match="c1_baseline.status"):
        validate_cn_diagnostic_artifact_payload(missing_c1_status)

    tmp_c1_artifact = json.loads(json.dumps(accepted))
    tmp_c1_artifact["scaling"]["c1_baseline"]["artifact_path"] = "/tmp/c1.json"
    with pytest.raises(ValueError, match="scaling.c1_baseline.artifact_path must be under benchmarks/results"):
        validate_cn_diagnostic_artifact_payload(tmp_c1_artifact)

    failed_c1_status = json.loads(json.dumps(accepted))
    failed_c1_status["scaling"]["c1_baseline"]["status"] = "missing"
    with pytest.raises(ValueError, match="c1_baseline.status must be usable"):
        validate_cn_diagnostic_artifact_payload(failed_c1_status)

    rejected_c1_status = json.loads(json.dumps(accepted))
    rejected_c1_status["scaling"]["c1_baseline"]["status"] = "rejected_correctness"
    with pytest.raises(ValueError, match="c1_baseline.status must be usable"):
        validate_cn_diagnostic_artifact_payload(rejected_c1_status)

    failed_serial_status = json.loads(json.dumps(accepted))
    failed_serial_status["scaling"]["serial_bridge_baseline"]["status"] = "failed"
    with pytest.raises(ValueError, match="serial_bridge_baseline.status must be usable"):
        validate_cn_diagnostic_artifact_payload(failed_serial_status)

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

    missing_environment_commands = json.loads(json.dumps(accepted))
    missing_environment_commands["commands"].pop("environment")
    with pytest.raises(ValueError, match="commands.environment"):
        validate_cn_diagnostic_artifact_payload(missing_environment_commands)

    incomplete_environment_commands = json.loads(json.dumps(accepted))
    incomplete_environment_commands["commands"]["environment"] = ["rocminfo", "hipcc --version"]
    with pytest.raises(ValueError, match="commands.environment must include rocm-smi"):
        validate_cn_diagnostic_artifact_payload(incomplete_environment_commands)

    missing_git_environment_command = json.loads(json.dumps(accepted))
    missing_git_environment_command["commands"]["environment"].remove("git diff --quiet")
    with pytest.raises(ValueError, match="commands.environment must include git diff --quiet"):
        validate_cn_diagnostic_artifact_payload(missing_git_environment_command)

    wrong_benchmark_command = json.loads(json.dumps(accepted))
    wrong_benchmark_command["commands"]["benchmark"] = "python3 scripts/qwen35_batch_serial_bench.py --batch-size 2"
    with pytest.raises(ValueError, match="commands.benchmark must reference scripts/qwen35_batch_retained_bench.py"):
        validate_cn_diagnostic_artifact_payload(wrong_benchmark_command)

    missing_benchmark_model = json.loads(json.dumps(accepted))
    missing_benchmark_model["commands"]["benchmark"] = "python3 scripts/qwen35_batch_retained_bench.py --fixture fixtures/qwen35.json --batch-size 2 --prompt-length 512 --decode-tokens 128 --max-layers 40 --json benchmarks/results/accepted-c2.json"
    with pytest.raises(ValueError, match="commands.benchmark must include --model"):
        validate_cn_diagnostic_artifact_payload(missing_benchmark_model)

    missing_benchmark_fixture = json.loads(json.dumps(accepted))
    missing_benchmark_fixture["commands"]["benchmark"] = "python3 scripts/qwen35_batch_retained_bench.py --model /models/test-qwen35 --batch-size 2 --prompt-length 512 --decode-tokens 128 --max-layers 40 --json benchmarks/results/accepted-c2.json"
    with pytest.raises(ValueError, match="commands.benchmark must include --fixture"):
        validate_cn_diagnostic_artifact_payload(missing_benchmark_fixture)

    missing_benchmark_json = json.loads(json.dumps(accepted))
    missing_benchmark_json["commands"]["benchmark"] = "python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --prompt-length 512 --decode-tokens 128 --max-layers 40"
    with pytest.raises(ValueError, match="commands.benchmark must include --json"):
        validate_cn_diagnostic_artifact_payload(missing_benchmark_json)

    tmp_benchmark_json = json.loads(json.dumps(accepted))
    tmp_benchmark_json["commands"]["benchmark"] = "python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --prompt-length 512 --decode-tokens 128 --max-layers 40 --json /tmp/accepted-c2.json"
    with pytest.raises(ValueError, match="commands.benchmark --json path must be under benchmarks/results"):
        validate_cn_diagnostic_artifact_payload(tmp_benchmark_json)

    missing_benchmark_c1_baseline = json.loads(json.dumps(accepted))
    missing_benchmark_c1_baseline["commands"]["benchmark"] = missing_benchmark_c1_baseline["commands"]["benchmark"].replace(" --c1-baseline-json benchmarks/results/c1.json", "")
    with pytest.raises(ValueError, match="commands.benchmark must include --c1-baseline-json"):
        validate_cn_diagnostic_artifact_payload(missing_benchmark_c1_baseline)

    mismatched_benchmark_serial_baseline = json.loads(json.dumps(accepted))
    mismatched_benchmark_serial_baseline["commands"]["benchmark"] = mismatched_benchmark_serial_baseline["commands"]["benchmark"].replace("benchmarks/results/serial-c2.json", "benchmarks/results/other-serial-c2.json")
    with pytest.raises(ValueError, match="commands.benchmark --serial-bridge-json path must match scaling.serial_bridge_baseline.artifact_path"):
        validate_cn_diagnostic_artifact_payload(mismatched_benchmark_serial_baseline)

    missing_benchmark_batch_size = json.loads(json.dumps(accepted))
    missing_benchmark_batch_size["commands"]["benchmark"] = "python3 scripts/qwen35_batch_retained_bench.py --prompt-length 512 --decode-tokens 128 --max-layers 40 --json accepted.json"
    with pytest.raises(ValueError, match="commands.benchmark must include --batch-size"):
        validate_cn_diagnostic_artifact_payload(missing_benchmark_batch_size)

    wrong_benchmark_batch_size = json.loads(json.dumps(accepted))
    wrong_benchmark_batch_size["commands"]["benchmark"] = "python3 scripts/qwen35_batch_retained_bench.py --batch-size 8 --prompt-length 512 --decode-tokens 128 --max-layers 40 --json accepted.json"
    with pytest.raises(ValueError, match="commands.benchmark --batch-size must match workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(wrong_benchmark_batch_size)

    missing_benchmark_prompt_length = json.loads(json.dumps(accepted))
    missing_benchmark_prompt_length["commands"]["benchmark"] = "python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --decode-tokens 128 --max-layers 40 --json accepted.json"
    with pytest.raises(ValueError, match="commands.benchmark must include --prompt-length"):
        validate_cn_diagnostic_artifact_payload(missing_benchmark_prompt_length)

    wrong_benchmark_prompt_length = json.loads(json.dumps(accepted))
    wrong_benchmark_prompt_length["commands"]["benchmark"] = "python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --prompt-length 128 --decode-tokens 128 --max-layers 40 --json accepted.json"
    with pytest.raises(ValueError, match="commands.benchmark --prompt-length must match workload.prompt_tokens_per_request"):
        validate_cn_diagnostic_artifact_payload(wrong_benchmark_prompt_length)

    missing_benchmark_decode_tokens = json.loads(json.dumps(accepted))
    missing_benchmark_decode_tokens["commands"]["benchmark"] = "python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --prompt-length 512 --max-layers 40 --json accepted.json"
    with pytest.raises(ValueError, match="commands.benchmark must include --decode-tokens"):
        validate_cn_diagnostic_artifact_payload(missing_benchmark_decode_tokens)

    wrong_benchmark_decode_tokens = json.loads(json.dumps(accepted))
    wrong_benchmark_decode_tokens["commands"]["benchmark"] = "python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --prompt-length 512 --decode-tokens 32 --max-layers 40 --json accepted.json"
    with pytest.raises(ValueError, match="commands.benchmark --decode-tokens must match workload.gen_tokens_per_request"):
        validate_cn_diagnostic_artifact_payload(wrong_benchmark_decode_tokens)

    missing_benchmark_max_layers = json.loads(json.dumps(accepted))
    missing_benchmark_max_layers["commands"]["benchmark"] = "python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --prompt-length 512 --decode-tokens 128 --json accepted.json"
    with pytest.raises(ValueError, match="commands.benchmark must include --max-layers"):
        validate_cn_diagnostic_artifact_payload(missing_benchmark_max_layers)

    wrong_benchmark_max_layers = json.loads(json.dumps(accepted))
    wrong_benchmark_max_layers["commands"]["benchmark"] = "python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --prompt-length 512 --decode-tokens 128 --max-layers 4 --json accepted.json"
    with pytest.raises(ValueError, match="commands.benchmark --max-layers must match workload.max_layers"):
        validate_cn_diagnostic_artifact_payload(wrong_benchmark_max_layers)

    missing_correctness_command = json.loads(json.dumps(accepted))
    missing_correctness_command["commands"]["correctness_reference"] = ""
    with pytest.raises(ValueError, match="commands.correctness_reference"):
        validate_cn_diagnostic_artifact_payload(missing_correctness_command)

    wrong_correctness_command = json.loads(json.dumps(accepted))
    wrong_correctness_command["commands"]["correctness_reference"] = "inline generated-token equality only"
    with pytest.raises(ValueError, match="commands.correctness_reference must reference scripts/qwen35_batch_correctness.py"):
        validate_cn_diagnostic_artifact_payload(wrong_correctness_command)

    missing_correctness_rows = json.loads(json.dumps(accepted))
    missing_correctness_rows["commands"]["correctness_reference"] = "python3 scripts/qwen35_batch_correctness.py --json primitive-c2.json"
    with pytest.raises(ValueError, match="commands.correctness_reference must include --rows"):
        validate_cn_diagnostic_artifact_payload(missing_correctness_rows)

    wrong_correctness_rows = json.loads(json.dumps(accepted))
    wrong_correctness_rows["commands"]["correctness_reference"] = "python3 scripts/qwen35_batch_correctness.py --rows 8 --json primitive-c8.json"
    with pytest.raises(ValueError, match="commands.correctness_reference --rows must match workload.concurrency"):
        validate_cn_diagnostic_artifact_payload(wrong_correctness_rows)

    missing_correctness_json = json.loads(json.dumps(accepted))
    missing_correctness_json["commands"]["correctness_reference"] = "python3 scripts/qwen35_batch_correctness.py --rows 2"
    with pytest.raises(ValueError, match="commands.correctness_reference must include --json"):
        validate_cn_diagnostic_artifact_payload(missing_correctness_json)

    mismatched_correctness_json = json.loads(json.dumps(accepted))
    mismatched_correctness_json["commands"]["correctness_reference"] = "python3 scripts/qwen35_batch_correctness.py --rows 2 --json benchmarks/results/other-primitive-c2.json"
    with pytest.raises(ValueError, match="commands.correctness_reference --json path must match correctness.primitive_batch_correctness.artifact_path"):
        validate_cn_diagnostic_artifact_payload(mismatched_correctness_json)

    missing_profiler = dict(accepted)
    missing_profiler.pop("profiler")
    with pytest.raises(ValueError, match="profiler"):
        validate_cn_diagnostic_artifact_payload(missing_profiler)

    missing_profiler_artifact = json.loads(json.dumps(accepted))
    missing_profiler_artifact["profiler"].pop("artifact_path")
    with pytest.raises(ValueError, match="profiler.artifact_path must be a non-empty string"):
        validate_cn_diagnostic_artifact_payload(missing_profiler_artifact)

    tmp_profiler_artifact = json.loads(json.dumps(accepted))
    tmp_profiler_artifact["profiler"]["artifact_path"] = "/tmp/profiler-c2.json"
    with pytest.raises(ValueError, match="profiler.artifact_path must be under benchmarks/results"):
        validate_cn_diagnostic_artifact_payload(tmp_profiler_artifact)

    missing_profiler_command = json.loads(json.dumps(accepted))
    missing_profiler_command["commands"]["profiler"] = None
    with pytest.raises(ValueError, match="commands.profiler"):
        validate_cn_diagnostic_artifact_payload(missing_profiler_command)

    profiler_without_kernel_trace = json.loads(json.dumps(accepted))
    profiler_without_kernel_trace["commands"]["profiler"] = "python3 scripts/qwen35_batch_retained_bench.py --json accepted.json"
    with pytest.raises(ValueError, match="commands.profiler must include rocprofv3 --kernel-trace"):
        validate_cn_diagnostic_artifact_payload(profiler_without_kernel_trace)

    profiler_without_output_format = json.loads(json.dumps(accepted))
    profiler_without_output_format["commands"]["profiler"] = profiler_without_output_format["commands"]["profiler"].replace(
        " --output-format csv",
        "",
    )
    with pytest.raises(ValueError, match="commands.profiler must include --output-format csv"):
        validate_cn_diagnostic_artifact_payload(profiler_without_output_format)

    profiler_mismatched_output_format = json.loads(json.dumps(accepted))
    profiler_mismatched_output_format["commands"]["profiler"] = profiler_mismatched_output_format["commands"]["profiler"].replace(
        " --output-format csv",
        " --output-format json",
    )
    with pytest.raises(ValueError, match="profiler.output_format must match commands.profiler --output-format"):
        validate_cn_diagnostic_artifact_payload(profiler_mismatched_output_format)

    profiler_without_trace_dir = json.loads(json.dumps(accepted))
    profiler_without_trace_dir["commands"]["profiler"] = profiler_without_trace_dir["commands"]["profiler"].replace(
        " -d /tmp/hipengine-profile",
        "",
    )
    with pytest.raises(ValueError, match="commands.profiler must include -d"):
        validate_cn_diagnostic_artifact_payload(profiler_without_trace_dir)

    profiler_missing_trace_dir = json.loads(json.dumps(accepted))
    profiler_missing_trace_dir["profiler"].pop("trace_dir")
    with pytest.raises(ValueError, match="profiler.trace_dir must be a non-empty string"):
        validate_cn_diagnostic_artifact_payload(profiler_missing_trace_dir)

    profiler_mismatched_trace_dir = json.loads(json.dumps(accepted))
    profiler_mismatched_trace_dir["profiler"]["trace_dir"] = "/tmp/other-profile"
    with pytest.raises(ValueError, match="profiler.trace_dir must match commands.profiler -d"):
        validate_cn_diagnostic_artifact_payload(profiler_mismatched_trace_dir)

    profiler_missing_trace_files = json.loads(json.dumps(accepted))
    profiler_missing_trace_files["profiler"].pop("trace_files")
    with pytest.raises(ValueError, match="profiler.trace_files must be a non-empty string list"):
        validate_cn_diagnostic_artifact_payload(profiler_missing_trace_files)

    profiler_trace_file_outside_trace_dir = json.loads(json.dumps(accepted))
    profiler_trace_file_outside_trace_dir["profiler"]["trace_files"] = ["/tmp/other-profile/hipengine_kernel_trace.csv"]
    with pytest.raises(ValueError, match="profiler.trace_files must be under profiler.trace_dir"):
        validate_cn_diagnostic_artifact_payload(profiler_trace_file_outside_trace_dir)

    profiler_trace_file_without_kernel_trace_csv = json.loads(json.dumps(accepted))
    profiler_trace_file_without_kernel_trace_csv["profiler"]["trace_files"] = ["/tmp/hipengine-profile/hipengine_api_trace.csv"]
    with pytest.raises(ValueError, match="profiler.trace_files must include a kernel-trace CSV path"):
        validate_cn_diagnostic_artifact_payload(profiler_trace_file_without_kernel_trace_csv)

    profiler_missing_synthesized_fields = json.loads(json.dumps(accepted))
    profiler_missing_synthesized_fields["profiler"].pop("synthesized_fields")
    with pytest.raises(ValueError, match="profiler.synthesized_fields must be a string list"):
        validate_cn_diagnostic_artifact_payload(profiler_missing_synthesized_fields)

    profiler_duplicate_synthesized_fields = json.loads(json.dumps(accepted))
    profiler_duplicate_synthesized_fields["profiler"]["synthesized_fields"] = ["trace_kernel_names", "trace_kernel_names"]
    with pytest.raises(ValueError, match="profiler.synthesized_fields must not contain duplicates"):
        validate_cn_diagnostic_artifact_payload(profiler_duplicate_synthesized_fields)

    profiler_unknown_synthesized_fields = json.loads(json.dumps(accepted))
    profiler_unknown_synthesized_fields["profiler"]["synthesized_fields"] = ["trace_kernel_names", "unexpected_field"]
    with pytest.raises(ValueError, match="profiler.synthesized_fields must only name known synthesized profiler fields"):
        validate_cn_diagnostic_artifact_payload(profiler_unknown_synthesized_fields)

    profiler_missing_trace_kernel_names = json.loads(json.dumps(accepted))
    profiler_missing_trace_kernel_names["profiler"].pop("trace_kernel_names")
    with pytest.raises(ValueError, match="profiler.trace_kernel_names must be a non-empty string list"):
        validate_cn_diagnostic_artifact_payload(profiler_missing_trace_kernel_names)

    profiler_trace_kernel_names_missing_duration = json.loads(json.dumps(accepted))
    profiler_trace_kernel_names_missing_duration["profiler"]["trace_kernel_names"] = ["qwen35_batch_prefill"]
    with pytest.raises(ValueError, match="profiler.trace_kernel_names must include profiler.kernel_durations_ns keys"):
        validate_cn_diagnostic_artifact_payload(profiler_trace_kernel_names_missing_duration)

    profiler_wrong_target = json.loads(json.dumps(accepted))
    profiler_wrong_target["commands"]["profiler"] = "rocprofv3 --kernel-trace -- python3 scripts/qwen35_batch_serial_bench.py --batch-size 2"
    with pytest.raises(ValueError, match="commands.profiler must target scripts/qwen35_batch_retained_bench.py"):
        validate_cn_diagnostic_artifact_payload(profiler_wrong_target)

    missing_profiler_model = json.loads(json.dumps(accepted))
    missing_profiler_model["commands"]["profiler"] = "rocprofv3 --kernel-trace -- python3 scripts/qwen35_batch_retained_bench.py --fixture fixtures/qwen35.json --batch-size 2 --prompt-length 512 --decode-tokens 128 --max-layers 40 --compiler-version-file benchmarks/results/hipcc-version.txt --require-cached-build --json benchmarks/results/accepted-c2.json --profiler-json benchmarks/results/profiler-c2.json"
    with pytest.raises(ValueError, match="commands.profiler must include --model"):
        validate_cn_diagnostic_artifact_payload(missing_profiler_model)

    missing_profiler_json = json.loads(json.dumps(accepted))
    missing_profiler_json["commands"]["profiler"] = "rocprofv3 --kernel-trace -- python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --prompt-length 512 --decode-tokens 128 --max-layers 40"
    with pytest.raises(ValueError, match="commands.profiler must include --json"):
        validate_cn_diagnostic_artifact_payload(missing_profiler_json)

    tmp_profiler_json = json.loads(json.dumps(accepted))
    tmp_profiler_json["commands"]["profiler"] = "rocprofv3 --kernel-trace -- python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --prompt-length 512 --decode-tokens 128 --max-layers 40 --json /tmp/accepted-c2.json"
    with pytest.raises(ValueError, match="commands.profiler --json path must be under benchmarks/results"):
        validate_cn_diagnostic_artifact_payload(tmp_profiler_json)

    missing_profiler_json_reference = json.loads(json.dumps(accepted))
    missing_profiler_json_reference["commands"]["profiler"] = "rocprofv3 --kernel-trace -- python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --prompt-length 512 --decode-tokens 128 --max-layers 40 --json benchmarks/results/accepted-c2.json"
    with pytest.raises(ValueError, match="commands.profiler must include --profiler-json"):
        validate_cn_diagnostic_artifact_payload(missing_profiler_json_reference)

    mismatched_profiler_primitive = json.loads(json.dumps(accepted))
    mismatched_profiler_primitive["commands"]["profiler"] = mismatched_profiler_primitive["commands"]["profiler"].replace("--primitive-correctness-json benchmarks/results/primitive-c2.json", "--primitive-correctness-json benchmarks/results/other-primitive-c2.json")
    with pytest.raises(ValueError, match="commands.profiler --primitive-correctness-json path must match correctness.primitive_batch_correctness.artifact_path"):
        validate_cn_diagnostic_artifact_payload(mismatched_profiler_primitive)

    mismatched_profiler_json_reference = json.loads(json.dumps(accepted))
    mismatched_profiler_json_reference["commands"]["profiler"] = "rocprofv3 --kernel-trace -- python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --prompt-length 512 --decode-tokens 128 --max-layers 40 --compiler-version-file benchmarks/results/hipcc-version.txt --require-cached-build --json benchmarks/results/accepted-c2.json --profiler-json benchmarks/results/other-profiler-c2.json"
    with pytest.raises(ValueError, match="commands.profiler --profiler-json path must match profiler.artifact_path"):
        validate_cn_diagnostic_artifact_payload(mismatched_profiler_json_reference)

    profiler_without_require_cached = json.loads(json.dumps(accepted))
    profiler_without_require_cached["commands"]["profiler"] = profiler_without_require_cached["commands"]["profiler"].replace(" --require-cached-build", "")
    with pytest.raises(ValueError, match="commands.profiler must include --require-cached-build"):
        validate_cn_diagnostic_artifact_payload(profiler_without_require_cached)

    profiler_without_compiler_version = json.loads(json.dumps(accepted))
    profiler_without_compiler_version["commands"]["profiler"] = profiler_without_compiler_version["commands"]["profiler"].replace(" --compiler-version-file benchmarks/results/hipcc-version.txt", "")
    with pytest.raises(ValueError, match="commands.profiler must include --compiler-version-file"):
        validate_cn_diagnostic_artifact_payload(profiler_without_compiler_version)

    profiler_missing_batch_size = json.loads(json.dumps(accepted))
    profiler_missing_batch_size["commands"]["profiler"] = "rocprofv3 --kernel-trace -- python3 scripts/qwen35_batch_retained_bench.py --prompt-length 512 --decode-tokens 128 --max-layers 40"
    with pytest.raises(ValueError, match="commands.profiler must include --batch-size"):
        validate_cn_diagnostic_artifact_payload(profiler_missing_batch_size)

    profiler_wrong_decode_tokens = json.loads(json.dumps(accepted))
    profiler_wrong_decode_tokens["commands"]["profiler"] = "rocprofv3 --kernel-trace -- python3 scripts/qwen35_batch_retained_bench.py --batch-size 2 --prompt-length 512 --decode-tokens 32 --max-layers 40"
    with pytest.raises(ValueError, match="commands.profiler --decode-tokens must match workload.gen_tokens_per_request"):
        validate_cn_diagnostic_artifact_payload(profiler_wrong_decode_tokens)

    not_captured_profiler = json.loads(json.dumps(accepted))
    not_captured_profiler["profiler"]["status"] = "not_captured"
    with pytest.raises(ValueError, match="profiler.status"):
        validate_cn_diagnostic_artifact_payload(not_captured_profiler)

    missing_profiler_output_format = json.loads(json.dumps(accepted))
    missing_profiler_output_format["profiler"].pop("output_format")
    with pytest.raises(ValueError, match="profiler.output_format must be 'csv'"):
        validate_cn_diagnostic_artifact_payload(missing_profiler_output_format)

    wrong_profiler_output_format = json.loads(json.dumps(accepted))
    wrong_profiler_output_format["profiler"]["output_format"] = "json"
    with pytest.raises(ValueError, match="profiler.output_format must be 'csv'"):
        validate_cn_diagnostic_artifact_payload(wrong_profiler_output_format)

    missing_expected_kernel = json.loads(json.dumps(accepted))
    missing_expected_kernel["profiler"]["expected_kernels_present"] = False
    with pytest.raises(ValueError, match="expected_kernels_present"):
        validate_cn_diagnostic_artifact_payload(missing_expected_kernel)

    missing_expected_kernel_names = json.loads(json.dumps(accepted))
    missing_expected_kernel_names["profiler"].pop("expected_kernel_names")
    with pytest.raises(ValueError, match="expected_kernel_names"):
        validate_cn_diagnostic_artifact_payload(missing_expected_kernel_names)

    non_batch_expected_kernel = json.loads(json.dumps(accepted))
    non_batch_expected_kernel["profiler"]["expected_kernel_names"] = ["qwen35_decode"]
    non_batch_expected_kernel["profiler"]["kernel_durations_ns"] = {"qwen35_decode": 12345.0}
    with pytest.raises(ValueError, match="expected_kernel_names must include at least one native batch kernel"):
        validate_cn_diagnostic_artifact_payload(non_batch_expected_kernel)

    missing_projection_profiler_kernel = json.loads(json.dumps(accepted))
    missing_projection_profiler_kernel["profiler"]["expected_kernel_names"] = ["qwen35_batch_decode"]
    missing_projection_profiler_kernel["profiler"]["kernel_durations_ns"] = {"qwen35_batch_decode": 12345.0}
    with pytest.raises(ValueError, match="profiler kernel names must include selected projection_dispatch candidate or variant"):
        validate_cn_diagnostic_artifact_payload(missing_projection_profiler_kernel)

    fallback_expected_kernel = json.loads(json.dumps(accepted))
    fallback_expected_kernel["profiler"]["expected_kernel_names"] = ["qwen35_batch_decode", "qwen35_per_row_fallback_decode"]
    fallback_expected_kernel["profiler"]["kernel_durations_ns"] = {
        "qwen35_batch_decode": 12345.0,
        "qwen35_per_row_fallback_decode": 12345.0,
    }
    with pytest.raises(ValueError, match="expected_kernel_names must not include serial/per-row/fallback"):
        validate_cn_diagnostic_artifact_payload(fallback_expected_kernel)

    missing_kernel_durations = json.loads(json.dumps(accepted))
    missing_kernel_durations["profiler"].pop("kernel_durations_ns")
    with pytest.raises(ValueError, match="kernel_durations_ns"):
        validate_cn_diagnostic_artifact_payload(missing_kernel_durations)

    missing_profiler_total = json.loads(json.dumps(accepted))
    missing_profiler_total["profiler"].pop("total_kernel_duration_ns")
    with pytest.raises(ValueError, match="total_kernel_duration_ns must be positive numeric"):
        validate_cn_diagnostic_artifact_payload(missing_profiler_total)

    mismatched_profiler_total = json.loads(json.dumps(accepted))
    mismatched_profiler_total["profiler"]["total_kernel_duration_ns"] = 1.0
    with pytest.raises(ValueError, match=r"total_kernel_duration_ns must match sum\(profiler.kernel_durations_ns\)"):
        validate_cn_diagnostic_artifact_payload(mismatched_profiler_total)

    missing_profiler_shares = json.loads(json.dumps(accepted))
    missing_profiler_shares["profiler"].pop("kernel_duration_shares")
    with pytest.raises(ValueError, match="kernel_duration_shares must be a non-empty object"):
        validate_cn_diagnostic_artifact_payload(missing_profiler_shares)

    mismatched_profiler_share = json.loads(json.dumps(accepted))
    mismatched_profiler_share["profiler"]["kernel_duration_shares"]["qwen35_batch_decode"] = 0.25
    with pytest.raises(ValueError, match="qwen35_batch_decode must match profiler.kernel_durations_ns/kernel total"):
        validate_cn_diagnostic_artifact_payload(mismatched_profiler_share)

    missing_duration_categories = json.loads(json.dumps(accepted))
    missing_duration_categories["profiler"].pop("kernel_duration_categories_ns")
    with pytest.raises(ValueError, match="kernel_duration_categories_ns must be a non-empty object"):
        validate_cn_diagnostic_artifact_payload(missing_duration_categories)

    mismatched_duration_category = json.loads(json.dumps(accepted))
    mismatched_duration_category["profiler"]["kernel_duration_categories_ns"]["projection"] = 1.0
    with pytest.raises(ValueError, match="kernel_duration_category_shares.projection must match"):
        validate_cn_diagnostic_artifact_payload(mismatched_duration_category)

    miscategorized_duration_categories = json.loads(json.dumps(accepted))
    miscategorized_duration_categories["profiler"]["kernel_duration_categories_ns"] = {
        "attention": 0.0,
        "moe": 0.0,
        "projection": 0.0,
        "sampling": 0.0,
        "graph_replay": 0.0,
        "other": 14690.0,
    }
    miscategorized_duration_categories["profiler"]["kernel_duration_category_shares"] = {
        "attention": 0.0,
        "moe": 0.0,
        "projection": 0.0,
        "sampling": 0.0,
        "graph_replay": 0.0,
        "other": 1.0,
    }
    with pytest.raises(ValueError, match="kernel_duration_categories_ns must match categorized"):
        validate_cn_diagnostic_artifact_payload(miscategorized_duration_categories)

    missing_cpu_bottlenecks = json.loads(json.dumps(accepted))
    missing_cpu_bottlenecks["profiler"].pop("cpu_side_bottlenecks_seconds")
    with pytest.raises(ValueError, match="cpu_side_bottlenecks_seconds must be a non-empty object"):
        validate_cn_diagnostic_artifact_payload(missing_cpu_bottlenecks)

    mismatched_cpu_bottleneck_share = json.loads(json.dumps(accepted))
    mismatched_cpu_bottleneck_share["profiler"]["cpu_side_bottleneck_shares"]["decode"] = 0.25
    with pytest.raises(ValueError, match="cpu_side_bottleneck_shares.decode must match"):
        validate_cn_diagnostic_artifact_payload(mismatched_cpu_bottleneck_share)

    fallback_kernel_duration = json.loads(json.dumps(accepted))
    fallback_kernel_duration["profiler"]["kernel_durations_ns"]["qwen35_per_row_fallback_decode"] = 12345.0
    with pytest.raises(ValueError, match="kernel_durations_ns must not include serial/per-row/fallback"):
        validate_cn_diagnostic_artifact_payload(fallback_kernel_duration)

    zero_extra_kernel_duration = json.loads(json.dumps(accepted))
    zero_extra_kernel_duration["profiler"]["kernel_durations_ns"]["qwen35_batch_prefill"] = 0.0
    with pytest.raises(ValueError, match="qwen35_batch_prefill must be positive numeric"):
        validate_cn_diagnostic_artifact_payload(zero_extra_kernel_duration)

    zero_kernel_duration = json.loads(json.dumps(accepted))
    zero_kernel_duration["profiler"]["kernel_durations_ns"]["qwen35_batch_decode"] = 0.0
    with pytest.raises(ValueError, match="qwen35_batch_decode must be positive numeric"):
        validate_cn_diagnostic_artifact_payload(zero_kernel_duration)

    empty_hardware = json.loads(json.dumps(accepted))
    empty_hardware["hardware"] = {}
    with pytest.raises(ValueError, match="hardware.gpu|hardware.arch"):
        validate_cn_diagnostic_artifact_payload(empty_hardware)

    for hardware_field in ("gpu", "arch"):
        missing_hardware_field = json.loads(json.dumps(accepted))
        missing_hardware_field["hardware"].pop(hardware_field)
        with pytest.raises(ValueError, match=f"hardware.{hardware_field}"):
            validate_cn_diagnostic_artifact_payload(missing_hardware_field)

    missing_rocminfo = json.loads(json.dumps(accepted))
    missing_rocminfo["hardware"].pop("rocminfo")
    with pytest.raises(ValueError, match="hardware.rocminfo"):
        validate_cn_diagnostic_artifact_payload(missing_rocminfo)

    failed_rocm_smi = json.loads(json.dumps(accepted))
    failed_rocm_smi["hardware"]["rocm_smi"]["returncode"] = 1
    with pytest.raises(ValueError, match="hardware.rocm_smi.returncode must be 0"):
        validate_cn_diagnostic_artifact_payload(failed_rocm_smi)

    wrong_rocminfo_command = json.loads(json.dumps(accepted))
    wrong_rocminfo_command["hardware"]["rocminfo"]["command"] = "cat /tmp/hw.txt"
    with pytest.raises(ValueError, match="hardware.rocminfo.command must include rocminfo"):
        validate_cn_diagnostic_artifact_payload(wrong_rocminfo_command)

    wrong_rocm_smi_command = json.loads(json.dumps(accepted))
    wrong_rocm_smi_command["hardware"]["rocm_smi"]["command"] = "cat /tmp/hw.txt"
    with pytest.raises(ValueError, match="hardware.rocm_smi.command must include rocm-smi"):
        validate_cn_diagnostic_artifact_payload(wrong_rocm_smi_command)

    mismatched_rocminfo_arch = json.loads(json.dumps(accepted))
    mismatched_rocminfo_arch["hardware"]["rocminfo"]["output"] = "Name: gfx0000"
    with pytest.raises(ValueError, match="hardware.rocminfo.output must include hardware.arch"):
        validate_cn_diagnostic_artifact_payload(mismatched_rocminfo_arch)

    short_commit = json.loads(json.dumps(accepted))
    short_commit["software"]["hipengine_commit"] = "abc1234"
    with pytest.raises(ValueError, match="software.hipengine_commit must be a full commit hash"):
        validate_cn_diagnostic_artifact_payload(short_commit)

    missing_dirty_state = json.loads(json.dumps(accepted))
    missing_dirty_state["software"].pop("hipengine_dirty")
    with pytest.raises(ValueError, match="hipengine_dirty"):
        validate_cn_diagnostic_artifact_payload(missing_dirty_state)

    dirty_state = json.loads(json.dumps(accepted))
    dirty_state["software"]["hipengine_dirty"] = True
    with pytest.raises(ValueError, match="software.hipengine_dirty must be false"):
        validate_cn_diagnostic_artifact_payload(dirty_state)

    missing_hipcc_version = json.loads(json.dumps(accepted))
    missing_hipcc_version["software"].pop("hipcc_version")
    with pytest.raises(ValueError, match="software.hipcc_version"):
        validate_cn_diagnostic_artifact_payload(missing_hipcc_version)

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
