from __future__ import annotations

import subprocess

import numpy as np
import pytest

from scripts.qwen4_exp_compare_logits import (
    _arguments,
    _run_llama_debug,
    compare_logits,
)


def test_qwen4_exp_compare_logits_accepts_prompt_file(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    assert _arguments(["model", "--prompt-file", str(prompt_file)]).prompt_file == prompt_file


def test_qwen4_exp_compare_logits_reports_kl_top1_and_errors() -> None:
    teacher = np.array([0.0, 2.0, -1.0], dtype=np.float32)
    actual = np.array([0.1, 1.8, -0.8], dtype=np.float32)

    report = compare_logits(teacher, actual)

    assert report["teacher_top1"] == 1
    assert report["hipengine_top1"] == 1
    assert report["top1_agreement"] is True
    assert 0.0 < report["kl_teacher_to_hipengine"] < 0.05
    assert report["mean_absolute_logit_error"] == pytest.approx(1.0 / 6.0)
    assert report["max_absolute_logit_error"] == pytest.approx(0.2)


def test_qwen4_exp_llama_debug_matches_bf16_kv_contract(tmp_path, monkeypatch) -> None:
    captured: list[list[str]] = []

    def run(command, **_kwargs):
        captured.append(command)
        np.asarray([1.0, 2.0], dtype=np.float32).tofile(tmp_path / "llamacpp-test.bin")
        np.asarray([3], dtype=np.int32).tofile(tmp_path / "llamacpp-test-tokens.bin")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("scripts.qwen4_exp_compare_logits.subprocess.run", run)

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt", encoding="utf-8")
    _run_llama_debug(
        tmp_path / "llama-debug",
        tmp_path / "model.gguf",
        "prompt",
        tmp_path,
        2052,
        llama_batch=2052,
        prompt_file=prompt_file,
    )

    assert captured
    command = captured[0]
    assert command[command.index("-ctk") + 1] == "bf16"
    assert command[command.index("-ctv") + 1] == "bf16"
    assert command[command.index("-b") + 1] == "2052"
    assert command[command.index("-f") + 1] == str(prompt_file)
    assert "-p" not in command


def test_qwen4_exp_llama_debug_replaces_invalid_diagnostic_bytes(tmp_path) -> None:
    executable = tmp_path / "fake-llama-debug"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "os.write(1, b'stdout-\\xff')\n"
        "os.write(2, b'stderr-\\xfe')\n"
        "raise SystemExit(9)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    with pytest.raises(RuntimeError, match="exit code 9") as failure:
        _run_llama_debug(executable, tmp_path / "model.gguf", "prompt", tmp_path, 16)

    assert "stdout-\ufffd" in str(failure.value)
    assert "stderr-\ufffd" in str(failure.value)


def test_qwen4_exp_compare_logits_rejects_shape_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="same 1D shape"):
        compare_logits(np.zeros(2, dtype=np.float32), np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        compare_logits(
            np.array([0.0, np.inf], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )
