from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.laguna_llamacpp_ar_bench import (
    EXPECTED_PROMPT_COUNT,
    MATCHED_PROMPT_SHA256,
    _aggregate,
    _build_server_command,
    _completion_payload,
    _load_hipengine_reference,
    _response_row,
    _sha256_file,
    _verify_source_archive,
)


def test_poolside_response_row_uses_native_timing_ownership() -> None:
    row = _response_row(
        prompt={
            "id": "p",
            "category": "code",
            "prompt_tokens": 10,
            "token_ids_sha256": "prompt-hash",
        },
        horizon=4,
        repetition=1,
        response={
            "tokens": [1, 2, 3, 4],
            "tokens_predicted": 4,
            "stop_type": "limit",
            "timings": {
                "prompt_n": 10,
                "prompt_ms": 500.0,
                "predicted_n": 4,
                "predicted_ms": 200.0,
            },
        },
        wall_seconds=0.8,
        hipengine_ids=[1, 2, 9, 4],
    )

    assert row["valid_token_count"] is True
    assert row["valid_prompt_count"] is True
    assert row["prompt_tok_s"] == pytest.approx(20.0)
    assert row["predicted_tok_s"] == pytest.approx(20.0)
    assert row["timed_decode_transitions"] == 3
    assert row["transition_normalized_tok_s"] == pytest.approx(15.0)
    assert row["wall_output_tok_s"] == pytest.approx(5.0)
    assert row["matches_hipengine"] is False
    assert row["matching_hipengine_prefix_tokens"] == 2


def test_streaming_transport_keeps_native_timing_authoritative() -> None:
    assert _completion_payload((1, 2, 3), 4)["stream"] is True
    row = _response_row(
        prompt={
            "id": "p",
            "category": "code",
            "prompt_tokens": 3,
            "token_ids_sha256": "prompt-hash",
        },
        horizon=4,
        repetition=0,
        response={
            "tokens": [10, 11, 12],
            "timings": {
                "prompt_n": 3,
                "prompt_ms": 10.0,
                "predicted_n": 4,
                "predicted_ms": 200.0,
            },
        },
        wall_seconds=0.3,
        hipengine_ids=[10, 11, 12, 13],
    )

    assert row["valid_token_count"] is True
    assert row["valid_native_predicted_count"] is True
    assert row["returned_token_array_complete"] is False
    assert row["timed_decode_transitions"] == 3
    assert row["transition_normalized_tok_s"] == pytest.approx(15.0)
    assert row["matches_hipengine"] is False


def test_poolside_aggregate_is_weighted_and_fail_closed() -> None:
    rows = [
        {
            "prompt_n": 10,
            "prompt_seconds": 1.0,
            "predicted_n": 4,
            "timed_decode_transitions": 3,
            "predicted_seconds": 0.2,
            "wall_seconds": 1.3,
            "valid_token_count": True,
            "valid_native_predicted_count": True,
            "returned_token_array_complete": True,
            "valid_prompt_count": True,
            "matches_hipengine": True,
        },
        {
            "prompt_n": 20,
            "prompt_seconds": 1.0,
            "predicted_n": 4,
            "timed_decode_transitions": 3,
            "predicted_seconds": 0.3,
            "wall_seconds": 1.5,
            "valid_token_count": True,
            "valid_native_predicted_count": True,
            "returned_token_array_complete": True,
            "valid_prompt_count": True,
            "matches_hipengine": False,
        },
    ]

    aggregate = _aggregate(rows)

    assert aggregate["prompt_tok_s"] == pytest.approx(15.0)
    assert aggregate["predicted_tok_s"] == pytest.approx(16.0)
    assert aggregate["timed_decode_transitions"] == 6
    assert aggregate["transition_normalized_tok_s"] == pytest.approx(12.0)
    assert aggregate["wall_output_tok_s"] == pytest.approx(8 / 2.8)
    assert aggregate["valid_token_counts"] is True
    assert aggregate["valid_prompt_counts"] is True
    assert aggregate["hipengine_exact_runs"] == 1

    rows[1]["valid_token_count"] = False
    assert _aggregate(rows)["valid_token_counts"] is False


def test_matched_server_command_pins_hip_bf16_and_flash_attention() -> None:
    args = SimpleNamespace(
        server_bin=Path("/tmp/build/bin/llama-server"),
        model=Path("/models/laguna.gguf"),
        host="127.0.0.1",
        port=18084,
        context_length=4096,
        gpu_layers=999,
        cache_type_k="bf16",
        cache_type_v="bf16",
        flash_attention="on",
        mmap=False,
        repack=True,
        skip_chat_parsing=True,
    )

    command = _build_server_command(args)

    assert command[:3] == [
        "/tmp/build/bin/llama-server",
        "-m",
        "/models/laguna.gguf",
    ]
    assert command[command.index("-ctk") + 1] == "bf16"
    assert command[command.index("-ctv") + 1] == "bf16"
    assert command[command.index("-fa") + 1] == "on"
    assert "--no-mmap" in command
    assert "--no-repack" not in command
    assert "--skip-chat-parsing" in command


def test_source_archive_verification_is_fail_closed(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    source = tmp_path / "source"
    reference.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=reference, check=True)
    subprocess.run(("git", "config", "user.email", "test@example.com"), cwd=reference, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=reference, check=True)
    (reference / "source.txt").write_text("exact\n", encoding="utf-8")
    subprocess.run(("git", "add", "source.txt"), cwd=reference, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=reference, check=True)
    revision = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=reference, text=True).strip()
    source.mkdir()
    payload = subprocess.check_output(("git", "show", f"{revision}:source.txt"), cwd=reference)
    (source / "source.txt").write_bytes(payload)

    state = _verify_source_archive(source, reference, revision)

    assert state["archive_matches_revision"] is True
    assert state["revision"] == revision
    assert state["tracked_files"] == 1

    (source / "source.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source archive mismatch"):
        _verify_source_archive(source, reference, revision)


def test_source_archive_verification_accepts_only_declared_patch(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    source = tmp_path / "source"
    reference.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=reference, check=True)
    subprocess.run(("git", "config", "user.email", "test@example.com"), cwd=reference, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=reference, check=True)
    (reference / "source.txt").write_text("before\n", encoding="utf-8")
    subprocess.run(("git", "add", "source.txt"), cwd=reference, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=reference, check=True)
    revision = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=reference, text=True).strip()
    source.mkdir()
    (source / "source.txt").write_text("after\n", encoding="utf-8")
    patch = tmp_path / "measurement.patch"
    patch.write_text(
        "diff --git a/source.txt b/source.txt\n"
        "index 90be1c3..89eaf97 100644\n"
        "--- a/source.txt\n"
        "+++ b/source.txt\n"
        "@@ -1 +1 @@\n"
        "-before\n"
        "+after\n",
        encoding="utf-8",
    )

    state = _verify_source_archive(source, reference, revision, patches=[patch])

    assert state["archive_matches_revision_plus_patches"] is True
    assert state["patches"] == [
        {"path": str(patch.resolve()), "sha256": _sha256_file(patch)}
    ]

    (source / "source.txt").write_text("undeclared\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source archive mismatch"):
        _verify_source_archive(source, reference, revision, patches=[patch])


def test_hipengine_reference_matches_transition_count_and_repetitions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hipengine.json"
    prompts = [
        {"id": "p0", "category": "code"},
        {"id": "p1", "category": "general_en"},
    ]
    rows = []
    for repetition in range(2):
        for prompt_index, prompt in enumerate(prompts):
            token = prompt_index + 1
            rows.append(
                {
                    "prompt_id": prompt["id"],
                    "mode": "bulk",
                    "repetition": repetition,
                    "checkpoints": {
                        "16": {
                            "generated_token_ids": [token] * 16,
                            "decode_forward_calls": 15,
                            "decode_seconds": 0.25,
                        },
                        "32": {
                            "generated_token_ids": [token] * 32,
                            "decode_forward_calls": 31,
                            "decode_seconds": 0.5,
                        },
                    },
                }
            )
    path.write_text(
        json.dumps(
            {
                "pass": True,
                "model": {"sha256": "model"},
                "repo": {"revision": "revision"},
                "protocol": {
                    "prompt_suite_sha256": "prompts",
                    "prompt_count": 2,
                    "context_length": 4096,
                    "output_horizons": [16, 32],
                },
                "prompt_runs": rows,
            }
        ),
        encoding="utf-8",
    )

    reference, oracle = _load_hipengine_reference(
        [path],
        model_sha256="model",
        prompt_sha256="prompts",
        prompts=prompts,
        context_length=4096,
        horizons=(16, 32),
        repetitions=2,
    )

    assert reference["horizons"]["16"] == {
        "runs": 4,
        "timed_decode_transitions": 60,
        "decode_seconds": 1.0,
        "transition_tok_s": 60.0,
    }
    assert reference["horizons"]["32"]["timed_decode_transitions"] == 124
    assert oracle[("p0", 16)] == [1] * 16


def test_tracked_matched_prompt_suite_is_complete() -> None:
    path = Path("benchmarks/prompts/laguna-target-ar-code-general-ja-heldout.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == EXPECTED_PROMPT_COUNT == 18
    assert {row["category"] for row in rows} == {
        "code",
        "general_en",
        "general_ja",
        "mixed_ja_en",
    }
    assert _sha256_file(path) == MATCHED_PROMPT_SHA256
