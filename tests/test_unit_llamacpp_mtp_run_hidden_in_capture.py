from __future__ import annotations

import json
import os
import struct
from pathlib import Path

from scripts.llamacpp_mtp_run_hidden_in_capture import (
    compare_capture,
    compare_numeric_reference,
    parse_prompt_tokens,
    run_hidden_in_capture,
    sha256_bytes,
    summarize_capture,
    summarize_floats,
)


def test_run_hidden_in_capture_matches_expected_hash(tmp_path: Path) -> None:
    values = [1.0, -2.0, 3.5, 4.25]
    raw = struct.pack("<4f", *values)
    expected = sha256_bytes(raw)
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, raw))
    reference_path = _write_reference_arrays(tmp_path, values)

    artifact = run_hidden_in_capture(
        compile_artifact_path=compile_artifact,
        model_path=tmp_path / "model.gguf",
        prompt_tokens="1,2,3,4",
        layer=3,
        position=2,
        expected_sha256=expected,
        output_prefix=tmp_path / "capture" / "layer3-pos2",
        reference_arrays_path=reference_path,
        all_rows=True,
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
    )

    assert artifact["status"] == "matched"
    assert artifact["run"]["returncode"] == 0
    assert artifact["prompt_tokens"] == [1, 2, 3, 4]
    assert artifact["comparison"]["matches_expected"] is True
    assert artifact["capture"]["float_count"] == 4
    assert artifact["capture"]["samples"] == [1.0, -2.0, 3.5, 4.25]
    assert artifact["capture"]["metadata"]["prompt_token_source"] == "token_ids"
    assert artifact["numeric_delta"]["available"] is True
    assert artifact["numeric_delta"]["max_abs_diff"] == 0.0
    assert artifact["numeric_delta"]["all_rows_scan"]["available"] is True
    assert artifact["numeric_delta"]["all_rows_scan"]["best_by_rmse"]["row"] == 0
    json.dumps(artifact)


def test_run_hidden_in_capture_reports_mismatch(tmp_path: Path) -> None:
    raw = struct.pack("<2f", 0.25, 0.5)
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, raw))

    artifact = run_hidden_in_capture(
        compile_artifact_path=compile_artifact,
        model_path=tmp_path / "model.gguf",
        prompt_tokens="7,8",
        layer=3,
        position=1,
        expected_sha256="0" * 64,
        output_prefix=tmp_path / "capture" / "layer3-pos1",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
    )

    assert artifact["status"] == "mismatched"
    assert artifact["comparison"]["comparable"] is True
    assert artifact["comparison"]["matches_expected"] is False
    assert "inspect_llamacpp_vs_hipengine" in artifact["next_action"]


def test_run_hidden_in_capture_reports_failure_without_output(tmp_path: Path) -> None:
    compile_artifact = _write_compile_artifact(tmp_path, _write_failing_harness(tmp_path))

    artifact = run_hidden_in_capture(
        compile_artifact_path=compile_artifact,
        model_path=tmp_path / "model.gguf",
        prompt_tokens="1",
        layer=3,
        position=0,
        expected_sha256="0" * 64,
        output_prefix=tmp_path / "capture" / "layer3-pos0",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
    )

    assert artifact["status"] == "capture_failed"
    assert artifact["run"]["returncode"] == 23
    assert artifact["capture"]["binary_exists"] is False
    assert "intentional failure" in artifact["run"]["stderr_tail"]


def test_summarize_capture_reads_metadata_stats_and_top_abs(tmp_path: Path) -> None:
    binary = tmp_path / "row.f32"
    meta = tmp_path / "row.json"
    values = [0.0, -4.0, 2.0, 1.0]
    binary.write_bytes(struct.pack("<4f", *values))
    meta.write_text(json.dumps({"n_embd": 4}))

    summary = summarize_capture(binary_path=binary, meta_path=meta)

    assert summary["binary_exists"] is True
    assert summary["metadata"]["n_embd"] == 4
    assert summary["stats"]["count"] == 4
    assert summary["top_abs"][0]["index"] == 1
    assert summary["top_abs"][0]["value"] == -4.0


def test_compare_numeric_reference_reports_delta(tmp_path: Path) -> None:
    binary = tmp_path / "row.f32"
    binary.write_bytes(struct.pack("<3f", 1.0, 3.0, 6.0))
    reference_path = _write_reference_arrays(tmp_path, [1.0, 2.0, 4.0])
    capture = {"binary_exists": True, "binary_path": str(binary), "sha256": "actual"}

    delta = compare_numeric_reference(
        capture,
        reference_arrays_path=reference_path,
        reference_key="hidden_in_f32",
    )

    assert delta["available"] is True
    assert delta["shape_match"] is True
    assert delta["max_abs_diff"] == 2.0
    assert delta["mean_abs_diff"] == 1.0
    assert delta["top_abs_diff"][0]["index"] == 2


def test_compare_capture_and_prompt_token_parser() -> None:
    assert parse_prompt_tokens("1,2,3") == [1, 2, 3]
    assert compare_capture({"sha256": "abc"}, expected_sha256="abc")["matches_expected"]
    assert not compare_capture({}, expected_sha256="abc")["comparable"]
    stats = summarize_floats([3.0, 4.0])
    assert stats["l2"] == 5.0


def _write_reference_arrays(tmp_path: Path, values: list[float]) -> Path:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"arrays": {"hidden_in_f32": values}}))
    return path


def _write_compile_artifact(tmp_path: Path, executable: Path) -> Path:
    artifact = {
        "outputs": {"executable": str(executable)},
        "lib_dir": str(tmp_path / "lib"),
    }
    (tmp_path / "lib").mkdir(exist_ok=True)
    path = tmp_path / "compile.json"
    path.write_text(json.dumps(artifact))
    return path


def _write_fake_harness(tmp_path: Path, raw: bytes) -> Path:
    exe = tmp_path / "fake-harness"
    values_literal = repr(raw)
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "prefix = pathlib.Path(args[args.index('--output-prefix') + 1])\n"
        "prefix.parent.mkdir(parents=True, exist_ok=True)\n"
        f"(prefix.with_suffix('.f32')).write_bytes({values_literal})\n"
        "all_rows = '--all-rows' in args\n"
        "if all_rows:\n"
        f"    (prefix.with_suffix('.all.f32')).write_bytes({values_literal} * 2)\n"
        "meta = {\n"
        "    'kind': 'llamacpp_hidden_in_capture',\n"
        "    'prompt_token_source': 'token_ids',\n"
        "    'binary': str(prefix.with_suffix('.f32')),\n"
        "    'n_embd': 4,\n"
        "}\n"
        "if all_rows:\n"
        "    meta['all_rows_binary'] = str(prefix.with_suffix('.all.f32'))\n"
        "(prefix.with_suffix('.json')).write_text(json.dumps(meta))\n"
        "print('{\"captured_hidden_in\":true}')\n"
    )
    exe.chmod(0o755)
    return exe


def _write_failing_harness(tmp_path: Path) -> Path:
    exe = tmp_path / "failing-harness"
    exe.write_text("#!/bin/sh\necho intentional failure >&2\nexit 23\n")
    exe.chmod(0o755)
    return exe


def _prepend_path(path: Path) -> str:
    return str(path) + os.pathsep + os.environ.get("PATH", "")
