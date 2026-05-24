from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


TOOL_PATH = Path("scripts/mtp-bench.py")


def _load_tool():
    spec = importlib.util.spec_from_file_location("hipengine_mtp_bench_tool", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_record_from_llamacpp_timing_payload_matches_pr_columns() -> None:
    tool = _load_tool()

    record = tool.record_from_response(
        "code_python",
        {
            "usage": {"completion_tokens": 192},
            "timings": {
                "predicted_per_second": 303.7,
                "draft_n": 177,
                "draft_n_accepted": 131,
            },
        },
        wall_s=0.75,
    )

    assert record == {
        "name": "code_python",
        "wall_s": 0.75,
        "predicted_n": 192,
        "predicted_per_second": 303.7,
        "draft_n": 177,
        "draft_n_accepted": 131,
        "accept_rate": 0.7401,
    }
    assert tool.format_result_line(record) == (
        "  code_python        pred= 192 draft= 177 acc= 131 rate=0.740 tok/s=303.7"
    )


def test_cli_lists_llamacpp_prompt_suite() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--list-prompts"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    lines = completed.stdout.strip().splitlines()
    assert len(lines) == 9
    assert lines[0].startswith("code_python\t")
    assert lines[-1].startswith("long_code_review\t")


def test_hipengine_current_mode_wraps_existing_prompt_suite_command(tmp_path: Path) -> None:
    tool = _load_tool()
    out = tmp_path / "current.json"
    args = tool.build_parser().parse_args(
        [
            "--mode",
            "hipengine-current",
            "--prompt-names",
            "code_python,translation",
            "--limit",
            "1",
            "--max-tokens",
            "64",
            "--candidate-budgets",
            "2,3",
            "--prompt-render",
            "qwen_chat_thinking_off",
            "--runs",
            "2",
            "--backend",
            "hip_gfx1100",
            "--hip-arch",
            "gfx1100",
            "--out",
            str(out),
            "--dry-run",
        ]
    )

    cmd = tool.build_hipengine_current_command(args)

    assert cmd[:2] == [sys.executable, "scripts/mtp_prompt_suite_economics.py"]
    assert "--decode-tokens" in cmd
    assert cmd[cmd.index("--decode-tokens") + 1] == "64"
    assert cmd[cmd.index("--candidate-budgets") + 1] == "2,3"
    assert cmd[cmd.index("--prompt-render") + 1] == "qwen_chat_thinking_off"
    assert cmd[cmd.index("--prompt-names") + 1] == "code_python,translation"
    assert cmd[cmd.index("--out") + 1] == str(out)
    assert cmd[-1] == "--dry-run"


def test_print_payload_preserves_gist_defaults_and_allows_overrides() -> None:
    default_payload = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--prompt-names", "translation", "--print-payload"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    default_body = json.loads(default_payload.stdout)

    assert default_body["model"] == "llama"
    assert default_body["max_tokens"] == 192
    assert default_body["seed"] == 42
    assert "temperature" not in default_body
    assert "cache_prompt" not in default_body
    assert default_body["messages"] == [
        {"role": "user", "content": "Translate to French: 'The quick brown fox jumps over the lazy dog.'"}
    ]

    override_payload = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--prompt-names",
            "translation",
            "--print-payload",
            "--temperature",
            "0",
            "--no-cache-prompt",
            "--extra-payload",
            '{"metadata":{"bench":"mtp"}}',
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    override_body = json.loads(override_payload.stdout)

    assert override_body["temperature"] == 0.0
    assert override_body["cache_prompt"] is False
    assert override_body["metadata"] == {"bench": "mtp"}
