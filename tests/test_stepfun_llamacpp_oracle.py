from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_llamacpp_oracle import main


def _write_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "prompt": "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n",
                "prompt_length": 23,
                "next_token_id": 369,
                "next_token_text": " |",
                "next_token_logit": 19.158626556396484,
                "top_tokens": [
                    {"rank": 1, "token_id": 369, "token_text": " |", "logit": 19.158626556396484}
                ],
            }
        )
    )


def test_stepfun_llamacpp_oracle_dry_run_builds_command(
    capsys,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    _write_artifact(artifact)
    llama_cli = tmp_path / "llama-cli"
    llama_cli.write_text("#!/usr/bin/env bash\necho 'version: test (deadbeef)'\n")
    llama_cli.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")

    rc = main(
        [
            "--artifact",
            str(artifact),
            "--llama-cli",
            str(llama_cli),
            "--model",
            str(model),
            "--pretty",
        ]
    )

    assert rc == 0
    output = capsys.readouterr()
    assert output.err == ""
    payload = json.loads(output.out)
    assert payload["status"] == "planned"
    assert payload["llama_cpp_version"] == "version: test (deadbeef)"
    assert payload["expected_next_token_id"] == 369
    assert payload["expected_next_token_text"] == " |"
    command = payload["command"]
    assert command[:2] == [str(llama_cli), "--model"]
    assert str(model) in command
    assert "--prompt" in command
    assert artifact.read_text()  # fixture sanity
    assert "--predict" in command
    assert command[command.index("--predict") + 1] == "1"
    assert command[command.index("--temp") + 1] == "0"
    assert command[command.index("--top-k") + 1] == "1"
    assert command[command.index("--top-p") + 1] == "1"
    assert command[command.index("--min-p") + 1] == "0"
    assert command[command.index("--repeat-penalty") + 1] == "1"
    assert "--no-display-prompt" in command
    assert "--simple-io" in command
    assert "--log-disable" in command
    assert payload["diagnostic_logs"] is False
    assert payload["comparison_policy"]["expected_text_field"] == "expected_next_token_text"
    assert "llama.cpp one-token run" in payload["note"]


def test_stepfun_llamacpp_oracle_execute_compares_stdout(
    capsys,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    _write_artifact(artifact)
    llama_cli = tmp_path / "llama-cli"
    llama_cli.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  echo 'version: test (deadbeef)'\n"
        "else\n"
        "  printf ' |'\n"
        "fi\n"
    )
    llama_cli.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")

    rc = main(
        [
            "--artifact",
            str(artifact),
            "--llama-cli",
            str(llama_cli),
            "--model",
            str(model),
            "--execute",
            "--pretty",
        ]
    )

    assert rc == 0
    output = capsys.readouterr()
    assert output.err == ""
    payload = json.loads(output.out)
    assert payload["status"] == "executed"
    assert payload["returncode"] == 0
    assert payload["stdout"] == " |"
    assert payload["stderr"] == ""
    assert payload["generated_text"] == " |"
    assert payload["text_matches_expected_exact"] is True
    assert payload["text_matches_expected_stripped"] is True
    assert payload["oracle_blocker_kind"] is None
    assert payload["step35_supported"] is None


def test_stepfun_llamacpp_oracle_execute_structures_step35_blocker(
    capsys,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    _write_artifact(artifact)
    llama_cli = tmp_path / "llama-cli"
    llama_cli.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  echo 'version: test (deadbeef)'\n"
        "else\n"
        "  echo \"llama_model_load: error loading model: unknown model architecture: 'step35'\" >&2\n"
        "  exit 1\n"
        "fi\n"
    )
    llama_cli.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")

    rc = main(
        [
            "--artifact",
            str(artifact),
            "--llama-cli",
            str(llama_cli),
            "--model",
            str(model),
            "--execute",
            "--diagnostic-logs",
            "--pretty",
        ]
    )

    assert rc == 0
    output = capsys.readouterr()
    assert output.err == ""
    payload = json.loads(output.out)
    assert payload["status"] == "executed"
    assert payload["returncode"] == 1
    assert payload["generated_text"] == ""
    assert payload["text_matches_expected_exact"] is False
    assert payload["oracle_blocker_kind"] == "llama_cpp_missing_step35_architecture"
    assert payload["step35_supported"] is False


def test_stepfun_llamacpp_oracle_diagnostic_logs_omit_log_disable(
    capsys,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    _write_artifact(artifact)
    llama_cli = tmp_path / "llama-cli"
    llama_cli.write_text("#!/usr/bin/env bash\necho 'version: test (deadbeef)'\n")
    llama_cli.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")

    rc = main(
        [
            "--artifact",
            str(artifact),
            "--llama-cli",
            str(llama_cli),
            "--model",
            str(model),
            "--diagnostic-logs",
            "--pretty",
        ]
    )

    assert rc == 0
    output = capsys.readouterr()
    assert output.err == ""
    payload = json.loads(output.out)
    assert payload["status"] == "planned"
    assert payload["diagnostic_logs"] is True
    assert "--log-disable" not in payload["command"]
