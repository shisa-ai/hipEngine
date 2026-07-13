from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.llamacpp_kv_matched_context import (
    _build_command,
    _prompt_metadata,
    _run_command,
    _validate_cpp_payload,
)


def _payload() -> dict:
    return {
        "mode": "llamacpp_kv_matched_context",
        "prompt_length": 8,
        "decode_steps": 2,
        "positions": 3,
        "prompt": {
            "mode": "repeated_token",
            "token_count": 8,
        },
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
    _validate_cpp_payload(
        _payload(),
        prompt_length=8,
        decode_steps=2,
        prompt_mode="repeated_token",
    )


def test_validate_cpp_payload_rejects_wrong_prompt_mode() -> None:
    with pytest.raises(ValueError, match="prompt mode"):
        _validate_cpp_payload(
            _payload(),
            prompt_length=8,
            decode_steps=2,
            prompt_mode="token_file",
        )


def test_prompt_metadata_hashes_exact_int32_token_stream(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.tokens"
    prompt.write_text("1 2\n3 2\n", encoding="utf-8")
    args = argparse.Namespace(
        prompt_token_file=prompt,
        prompt_token_id=9707,
        prompt_length=4,
    )

    metadata = _prompt_metadata(args)

    assert metadata == {
        "mode": "token_file",
        "token_file": str(prompt),
        "token_count": 4,
        "distinct_tokens": 3,
        "prefix_token_ids_sample": [1, 2, 3, 2],
        "token_ids_int32_le_sha256": "a8aaa835a9d64a57862dbac5fdcc0704bc4284fff4f36f1c73833de117b4cab3",
        "token_file_fingerprint": {
            "algorithm": "sha256-full-v1",
            "size_bytes": 8,
            "sampled_bytes": 8,
            "value": "360be5a7c6834dab4e51ff69bc6fbe82fbbdf04ffda90bc3947727a458087ce5",
        },
    }


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
        prompt_token_file=Path("/tmp/mixed.tokens"),
        prompt_length=131072,
        decode_steps=16,
        ctx_size=0,
        batch_size=4096,
        ubatch_size=512,
        n_gpu_layers=99,
        threads=16,
        reference_cache="f16",
        candidate_cache="q8_0",
        candidate_cache_k="q8_0",
        candidate_cache_v="f16",
        flash_attn=True,
        kl_threshold=0.05,
        top1_threshold=0.90,
        reference_logits_bin=Path("/tmp/reference.bin"),
    )

    command = _run_command(args, binary=Path("/tmp/harness"), cpp_json=Path("/tmp/result.json"))

    assert command[command.index("--prompt-length") + 1] == "131072"
    assert command[command.index("--decode-steps") + 1] == "16"
    assert command[command.index("--ctx-size") + 1] == "131089"
    assert command[command.index("--prompt-token-file") + 1] == "/tmp/mixed.tokens"
    assert command[command.index("--reference-cache") + 1] == "f16"
    assert command[command.index("--candidate-cache") + 1] == "q8_0"
    assert command[command.index("--candidate-cache-k") + 1] == "q8_0"
    assert command[command.index("--candidate-cache-v") + 1] == "f16"
    assert command[command.index("--reference-logits-bin") + 1] == "/tmp/reference.bin"
