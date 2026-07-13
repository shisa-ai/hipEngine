from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.llamacpp_kv_matched_context import (
    _build_command,
    _run_command,
    _validate_cpp_payload,
)


def _payload() -> dict:
    return {
        "mode": "llamacpp_kv_matched_context",
        "prompt_length": 8,
        "decode_steps": 2,
        "positions": 3,
        "reference": {
            "finite_logits": True,
            "top1_ids": [10, 11, 12],
            "decode_input_ids": [10, 11],
        },
        "candidate": {
            "finite_logits": True,
            "top1_ids": [10, 20, 12],
            "decode_input_ids": [10, 11],
        },
        "matched_context": {
            "kl": [0.0, 0.03, 0.06],
            "mean_kl": 0.03,
            "max_kl": 0.06,
            "top1_matches": [True, False, True],
            "top1_agreement": 2.0 / 3.0,
        },
    }


def test_validate_cpp_payload_accepts_reference_teacher_forcing() -> None:
    _validate_cpp_payload(_payload(), prompt_length=8, decode_steps=2)


def test_validate_cpp_payload_rejects_candidate_history_drift() -> None:
    payload = _payload()
    payload["candidate"]["decode_input_ids"] = [10, 20]

    with pytest.raises(ValueError, match="reference token history"):
        _validate_cpp_payload(payload, prompt_length=8, decode_steps=2)


def test_validate_cpp_payload_rejects_inconsistent_metric_summary() -> None:
    payload = _payload()
    payload["matched_context"]["mean_kl"] = 0.5

    with pytest.raises(ValueError, match="mean KL"):
        _validate_cpp_payload(payload, prompt_length=8, decode_steps=2)


def test_build_command_uses_public_headers_and_measured_library_directory(tmp_path: Path) -> None:
    source = tmp_path / "llama.cpp"
    build = tmp_path / "build"
    output = tmp_path / "out" / "harness"

    command = _build_command(
        compiler="g++",
        source=tmp_path / "harness.cpp",
        llama_source=source,
        llama_build=build,
        output=output,
    )

    assert command[0] == "g++"
    assert f"-I{source / 'include'}" in command
    assert f"-I{source / 'ggml' / 'include'}" in command
    assert f"-L{build / 'bin'}" in command
    assert "-lllama" in command
    assert command[-1] == str(output)


def test_run_command_encodes_exact_matched_workload() -> None:
    args = argparse.Namespace(
        model=Path("/models/model.gguf"),
        prompt_token_id=9707,
        prompt_length=131072,
        decode_steps=16,
        ctx_size=0,
        batch_size=4096,
        ubatch_size=512,
        n_gpu_layers=99,
        threads=16,
        reference_cache="f16",
        candidate_cache="q8_0",
        flash_attn=True,
        kl_threshold=0.05,
        top1_threshold=0.90,
        reference_logits_bin=Path("/tmp/reference.bin"),
    )

    command = _run_command(args, binary=Path("/tmp/harness"), cpp_json=Path("/tmp/result.json"))

    assert command[command.index("--prompt-length") + 1] == "131072"
    assert command[command.index("--decode-steps") + 1] == "16"
    assert command[command.index("--ctx-size") + 1] == "131089"
    assert command[command.index("--reference-cache") + 1] == "f16"
    assert command[command.index("--candidate-cache") + 1] == "q8_0"
    assert command[command.index("--reference-logits-bin") + 1] == "/tmp/reference.bin"
