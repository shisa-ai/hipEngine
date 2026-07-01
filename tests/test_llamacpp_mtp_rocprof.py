from __future__ import annotations

import argparse
from pathlib import Path

from scripts import llamacpp_mtp_rocprof as rocprof


def _args(**overrides):
    base = {
        "server_bin": Path("/tmp/llama-server"),
        "model": Path("/models/qwen.gguf"),
        "gpu_layers": 99,
        "flash_attn": "on",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "ctx_size": 8192,
        "host": "127.0.0.1",
        "port": 8019,
        "alias": "qwen",
        "reasoning": "off",
        "draft_max": 2,
        "server_extra_arg": [],
        "token_id": 9707,
        "prompt_tokens": 32,
        "token_repeat": False,
        "prompt": "Write add.",
        "max_tokens": 16,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": 12345,
        "rocprofv3": "rocprofv3",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_llamacpp_mtp_rocprof_server_command_enables_mtp() -> None:
    cmd = rocprof.build_server_command(_args(server_extra_arg=["--no-host"]))

    assert cmd[:3] == ["/tmp/llama-server", "-m", "/models/qwen.gguf"]
    assert "--spec-type" in cmd
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert "--spec-draft-n-max" in cmd
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "2"
    assert "--reasoning" in cmd
    assert cmd[cmd.index("--reasoning") + 1] == "off"
    assert cmd[-1] == "--no-host"


def test_llamacpp_mtp_rocprof_completion_payload_is_bounded() -> None:
    payload = rocprof.build_completion_payload(_args())

    assert payload["prompt"] == "Write add."
    assert payload["n_predict"] == 16
    assert payload["temperature"] == 0.0
    assert payload["top_k"] == 1
    assert payload["cache_prompt"] is False
    assert payload["stream"] is False


def test_llamacpp_mtp_rocprof_completion_payload_token_repeat() -> None:
    payload = rocprof.build_completion_payload(_args(token_repeat=True, token_id=42, prompt_tokens=4))

    assert payload["prompt"] == [42, 42, 42, 42]
