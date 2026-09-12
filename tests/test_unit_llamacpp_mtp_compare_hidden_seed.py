from __future__ import annotations

import json
import os
import struct
from pathlib import Path

from scripts.llamacpp_mtp_compare_hidden_seed import (
    compare_capture_vectors,
    compare_hidden_seed,
    parse_prompt_tokens,
    redact_values,
)
from scripts.llamacpp_mtp_run_hidden_in_capture import pack_float32, sha256_bytes


def test_compare_hidden_seed_matches_with_fake_harness(tmp_path: Path) -> None:
    values = [1.0, -2.0, 3.5, 4.25]
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, values))

    artifact = compare_hidden_seed(
        compile_artifact_path=compile_artifact,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(1, 2, 3),
        position=2,
        output_prefix=tmp_path / "capture" / "seed-pos2",
        all_rows=True,
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture(values),
    )

    assert artifact["status"] == "matched"
    assert artifact["llamacpp_run"]["returncode"] == 0
    assert artifact["llamacpp_capture"]["float_count"] == 4
    assert artifact["hipengine_capture"]["summary"]["sha256"] == sha256_bytes(
        pack_float32(values)
    )
    assert artifact["numeric_delta"]["exact_match"] is True
    assert artifact["numeric_delta"]["all_rows_scan"]["matches"][0]["row"] == 0
    assert artifact["next_action"] == "promote_fp32_hidden_seed_oracle_to_nextn_parity_gate"
    json.dumps(artifact)


def test_compare_hidden_seed_reports_mismatch(tmp_path: Path) -> None:
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, [1.0, 2.0]))

    artifact = compare_hidden_seed(
        compile_artifact_path=compile_artifact,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(7, 8),
        position=1,
        output_prefix=tmp_path / "capture" / "seed-pos1",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture([1.5, 2.0]),
    )

    assert artifact["status"] == "mismatched"
    assert artifact["numeric_delta"]["shape_match"] is True
    assert artifact["numeric_delta"]["max_abs_diff"] == 0.5
    assert artifact["next_action"] == (
        "inspect_upstream_bf16_activation_propagation_before_fp32_seed"
    )


def test_compare_hidden_seed_reports_llamacpp_failure(tmp_path: Path) -> None:
    compile_artifact = _write_compile_artifact(tmp_path, _write_failing_harness(tmp_path))

    artifact = compare_hidden_seed(
        compile_artifact_path=compile_artifact,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(1,),
        position=0,
        output_prefix=tmp_path / "capture" / "seed-pos0",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture([1.0]),
    )

    assert artifact["status"] == "llamacpp_capture_failed"
    assert artifact["llamacpp_run"]["returncode"] == 23
    assert artifact["llamacpp_capture"]["binary_exists"] is False
    assert artifact["next_action"] == "inspect_llamacpp_hidden_seed_capture_logs"


def test_compare_hidden_seed_requires_final_prompt_position(tmp_path: Path) -> None:
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, [1.0]))

    try:
        compare_hidden_seed(
            compile_artifact_path=compile_artifact,
            model_path=tmp_path / "model.gguf",
            prompt_tokens=(1, 2, 3),
            position=1,
            output_prefix=tmp_path / "capture" / "seed-pos1",
            hip_capture_fn=_hip_capture([1.0]),
        )
    except ValueError as exc:
        assert "final prompt token" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")


def test_compare_capture_vectors_handles_shape_mismatch(tmp_path: Path) -> None:
    binary = tmp_path / "row.f32"
    binary.write_bytes(struct.pack("<2f", 1.0, 2.0))

    delta = compare_capture_vectors(
        llamacpp_capture={"binary_exists": True, "binary_path": str(binary)},
        hipengine_capture={"status": "captured", "values": [1.0]},
        exact_atol=0.0,
    )

    assert delta["shape_match"] is False
    assert delta["llamacpp_count"] == 2
    assert delta["hipengine_count"] == 1


def test_redact_values_and_prompt_parser() -> None:
    redacted = redact_values({"status": "captured", "values": [3.0, 4.0]})

    assert "values" not in redacted
    assert redacted["summary"]["stats"]["l2"] == 5.0
    assert parse_prompt_tokens("1, 2,3") == (1, 2, 3)


def _write_compile_artifact(tmp_path: Path, executable: Path) -> Path:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(exist_ok=True)
    artifact = {
        "outputs": {"executable": str(executable)},
        "lib_dir": str(lib_dir),
    }
    path = tmp_path / "compile.json"
    path.write_text(json.dumps(artifact))
    return path


def _write_fake_harness(tmp_path: Path, values: list[float]) -> Path:
    exe = tmp_path / "fake-hidden-seed"
    raw_literal = repr(pack_float32(values))
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "prefix = pathlib.Path(args[args.index('--output-prefix') + 1])\n"
        "prefix.parent.mkdir(parents=True, exist_ok=True)\n"
        f"raw = {raw_literal}\n"
        "prefix.with_suffix('.f32').write_bytes(raw)\n"
        "if '--all-rows' in args:\n"
        "    prefix.with_suffix('.all.f32').write_bytes(raw * 2)\n"
        "meta = {\n"
        "    'kind': 'llamacpp_hidden_seed_capture',\n"
        "    'tap': 'h_nextn_post_output_norm',\n"
        "    'prompt_token_source': 'token_ids',\n"
        "    'binary': str(prefix.with_suffix('.f32')),\n"
        f"    'n_embd': {len(values)},\n"
        "}\n"
        "if '--all-rows' in args:\n"
        "    meta['all_rows_binary'] = str(prefix.with_suffix('.all.f32'))\n"
        "prefix.with_suffix('.json').write_text(json.dumps(meta))\n"
        "print('{\"captured_hidden_seed\":true}')\n"
    )
    exe.chmod(0o755)
    return exe


def _write_failing_harness(tmp_path: Path) -> Path:
    exe = tmp_path / "failing-hidden-seed"
    exe.write_text("#!/bin/sh\necho intentional failure >&2\nexit 23\n")
    exe.chmod(0o755)
    return exe


def _hip_capture(values: list[float]):
    def capture(_model: Path, prompt_tokens: tuple[int, ...], position: int, _max_seq: int | None):
        return {
            "status": "captured",
            "position": int(position),
            "token_id": int(prompt_tokens[position]),
            "next_token_id": 0,
            "contract": {"dtype": "FP32", "ready_for_mtp": True},
            "dtype": "FP32",
            "values": [float(value) for value in values],
        }

    return capture


def _prepend_path(path: Path) -> str:
    return str(path) + os.pathsep + os.environ.get("PATH", "")
