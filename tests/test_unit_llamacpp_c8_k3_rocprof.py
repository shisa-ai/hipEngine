from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from scripts.llamacpp_c8_k3_rocprof import (
    build_rocprof_command,
    build_server_command,
    read_prompt,
    request_payload,
    summarize_response,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        server=tmp_path / "llama-server",
        source=tmp_path / "llama.cpp",
        model=tmp_path / "model.gguf",
        host="127.0.0.1",
        port=18123,
        ctx_size=8192,
        batch_size=2048,
        ubatch_size=512,
        gpu_layers=999,
        width=8,
        threads=16,
        draft_max=3,
        collection_delay_s=20.0,
        collection_duration_s=12.0,
        rocprofv3="rocprofv3",
    )


def test_read_prompt_selects_requested_suite_row(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "first",
                        "category": "code",
                        "messages": [{"role": "user", "content": "alpha"}],
                    }
                ),
                json.dumps(
                    {
                        "id": "second",
                        "category": "general_en",
                        "messages": [{"role": "user", "content": "beta"}],
                    }
                ),
            ]
        )
        + "\n"
    )

    prompt = read_prompt(prompt_file, "second")

    assert prompt == {
        "id": "second",
        "category": "general_en",
        "rendered": "<|im_start|>user\nbeta<|im_end|>\n<|im_start|>assistant\n",
    }
    with pytest.raises(ValueError, match="does not contain"):
        read_prompt(prompt_file, "missing")


def test_profile_commands_keep_client_outside_profiled_server(tmp_path: Path) -> None:
    args = _args(tmp_path)

    server = build_server_command(args)
    profile = build_rocprof_command(args, trace_dir=tmp_path / "trace", server_command=server)

    assert server[-8:] == [
        "-ctv",
        "f16",
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "3",
        "--spec-draft-p-min",
        "0.0",
    ]
    assert server[server.index("-np") + 1] == "8"
    assert profile[:7] == [
        "rocprofv3",
        "--kernel-trace",
        "--hip-runtime-trace",
        "--memory-copy-trace",
        "--output-format",
        "csv",
        "--collection-period",
    ]
    assert profile[7] == "20.0:12.0:1"
    assert profile[profile.index("--") + 1 :] == server


def test_request_and_response_summary_preserve_acceptance_inputs() -> None:
    payload = request_payload("hello", n_predict=24)
    response = {
        "content": "answer",
        "tokens_predicted": 24,
        "tokens_evaluated": 36,
        "timings": {"draft_n": 19, "draft_n_accepted": 16},
    }

    row = summarize_response(response, lane=3, start=0.1, end=1.2)

    assert payload["n_predict"] == 24
    assert payload["temperature"] == 0.0
    assert payload["cache_prompt"] is False
    assert row["lane"] == 3
    assert row["tokens_predicted"] == 24
    assert row["timings"] == {"draft_n": 19, "draft_n_accepted": 16}
    assert row["content_sha256"] == hashlib.sha256(b"answer").hexdigest()
