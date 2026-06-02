from __future__ import annotations

import json
import os
from pathlib import Path

from scripts import stepfun_llamacpp_oracle

main = stepfun_llamacpp_oracle.main


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


def test_stepfun_llamacpp_oracle_emit_json_replaces_output_atomically(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "oracle.json"
    output.write_text('{"status":"old"}\n')
    observed: dict[str, object] = {}
    real_replace = os.replace

    def spy_replace(src: object, dst: object) -> None:
        observed["destination_before_replace"] = output.read_text()
        observed["temp_payload"] = Path(src).read_text()
        real_replace(src, dst)

    monkeypatch.setattr(stepfun_llamacpp_oracle.os, "replace", spy_replace)

    stepfun_llamacpp_oracle._emit_json(
        {"status": "running", "partial_artifact": True},
        pretty=True,
        output=output,
    )

    assert observed == {
        "destination_before_replace": '{"status":"old"}\n',
        "temp_payload": '{\n  "partial_artifact": true,\n  "status": "running"\n}\n',
    }
    assert json.loads(output.read_text()) == {
        "partial_artifact": True,
        "status": "running",
    }
    assert not list(tmp_path.glob(".oracle.json.*.tmp"))


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
    assert payload["extra_llama_args"] == []
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
    assert payload["elapsed_s"] >= 0.0
    assert payload["stdout"] == " |"
    assert payload["stderr"] == ""
    assert payload["generated_text"] == " |"
    assert payload["text_matches_expected_exact"] is True
    assert payload["text_matches_expected_stripped"] is True
    assert payload["oracle_blocker_kind"] is None
    assert payload["step35_supported"] is None


def test_stepfun_llamacpp_oracle_execute_writes_partial_output_before_launch(
    capsys,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    _write_artifact(artifact)
    output_path = tmp_path / "oracle-output.json"
    marker_path = tmp_path / "partial-marker.json"
    llama_cli = tmp_path / "llama-cli"
    llama_cli.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  echo 'version: test (deadbeef)'\n"
        "else\n"
        f"  python3 - {str(output_path)!r} {str(marker_path)!r} <<'PY'\n"
        "import json, sys\n"
        "output_path, marker_path = sys.argv[1:3]\n"
        "payload = json.load(open(output_path))\n"
        "json.dump({\n"
        "    'status': payload.get('status'),\n"
        "    'partial_artifact': payload.get('partial_artifact'),\n"
        "    'oracle_blocker_kind': payload.get('oracle_blocker_kind'),\n"
        "    'partial_output_path': payload.get('partial_output_path'),\n"
        "}, open(marker_path, 'w'), sort_keys=True)\n"
        "PY\n"
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
            "--output",
            str(output_path),
            "--pretty",
        ]
    )

    assert rc == 0
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
    marker = json.loads(marker_path.read_text())
    assert marker == {
        "status": "running",
        "partial_artifact": True,
        "oracle_blocker_kind": "llama_cpp_oracle_in_progress",
        "partial_output_path": str(output_path),
    }
    payload = json.loads(output_path.read_text())
    assert payload["status"] == "executed"
    assert payload["partial_output_written_before_launch"] is True
    assert payload["partial_output_path"] == str(output_path)
    assert payload["generated_text"] == " |"
    assert payload["text_matches_expected_exact"] is True


def test_stepfun_llamacpp_oracle_timeout_overwrites_partial_output(
    capsys,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    _write_artifact(artifact)
    output_path = tmp_path / "oracle-timeout-output.json"
    marker_path = tmp_path / "timeout-partial-marker.json"
    llama_cli = tmp_path / "llama-cli"
    llama_cli.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  echo 'version: test (deadbeef)'\n"
        "else\n"
        f"  python3 - {str(output_path)!r} {str(marker_path)!r} <<'PY'\n"
        "import json, sys\n"
        "output_path, marker_path = sys.argv[1:3]\n"
        "payload = json.load(open(output_path))\n"
        "json.dump({\n"
        "    'status': payload.get('status'),\n"
        "    'partial_artifact': payload.get('partial_artifact'),\n"
        "    'oracle_blocker_kind': payload.get('oracle_blocker_kind'),\n"
        "    'partial_output_path': payload.get('partial_output_path'),\n"
        "}, open(marker_path, 'w'), sort_keys=True)\n"
        "PY\n"
        "  sleep 2\n"
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
            "--timeout-s",
            "0.2",
            "--output",
            str(output_path),
            "--pretty",
        ]
    )

    assert rc == 0
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
    marker = json.loads(marker_path.read_text())
    assert marker == {
        "status": "running",
        "partial_artifact": True,
        "oracle_blocker_kind": "llama_cpp_oracle_in_progress",
        "partial_output_path": str(output_path),
    }
    payload = json.loads(output_path.read_text())
    assert payload["status"] == "timeout"
    assert payload["timeout_s"] == 0.2
    assert payload["elapsed_s"] >= 0.0
    assert payload["partial_output_written_before_launch"] is True
    assert payload["partial_output_path"] == str(output_path)
    assert payload["oracle_blocker_kind"] == "llama_cpp_oracle_timeout"
    assert payload["oracle_blocker_detail"] == (
        "llama.cpp oracle timed out before producing a comparable token"
    )
    assert payload["timeout_termination"] == {
        "timeout_reached": True,
        "timeout_s": 0.2,
        "process_group_started": True,
        "termination_method": "os.killpg",
        "termination_signal": "SIGKILL",
        "termination_signal_number": 9,
        "termination_path": "killpg_sigkill_then_communicate",
        "communicate_after_signal_timeout_s": 10.0,
        "process_exited_before_signal": False,
        "fallback_proc_kill_used": False,
    }
    assert payload["generated_text"] == ""
    assert payload["text_matches_expected_exact"] is False
    assert payload["text_matches_expected_stripped"] is False


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
    assert payload["elapsed_s"] >= 0.0
    assert payload["generated_text"] == ""
    assert payload["text_matches_expected_exact"] is False
    assert payload["oracle_blocker_kind"] == "llama_cpp_missing_step35_architecture"
    assert payload["step35_supported"] is False


def test_stepfun_llamacpp_oracle_appends_extra_args(capsys, tmp_path: Path) -> None:
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
            "--llama-arg=--device",
            "--llama-arg=none",
            "--llama-arg=--gpu-layers",
            "--llama-arg=0",
            "--pretty",
        ]
    )

    assert rc == 0
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert payload["extra_llama_args"] == ["--device", "none", "--gpu-layers", "0"]
    assert payload["command"][-5:] == ["--device", "none", "--gpu-layers", "0", "--log-disable"]


def test_stepfun_llamacpp_oracle_timeout_is_structured(capsys, tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    _write_artifact(artifact)
    llama_cli = tmp_path / "llama-cli"
    llama_cli.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  echo 'version: test (deadbeef)'\n"
        "else\n"
        "  sleep 2\n"
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
            "--timeout-s",
            "0.1",
            "--pretty",
        ]
    )

    assert rc == 0
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert payload["status"] == "timeout"
    assert payload["elapsed_s"] >= 0.0
    assert payload["oracle_blocker_kind"] == "llama_cpp_oracle_timeout"
    assert payload["step35_supported"] is None
    assert payload["text_matches_expected_exact"] is False


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
