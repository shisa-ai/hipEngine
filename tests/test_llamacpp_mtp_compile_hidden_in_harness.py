from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.llamacpp_mtp_compile_hidden_in_harness import (
    build_compile_command,
    compile_hidden_in_harness,
    hidden_in_capture_harness_source,
    hidden_in_harness_source,
    validate_headers,
)


def test_hidden_in_harness_source_links_extension_symbols() -> None:
    source = hidden_in_harness_source()

    assert '#include "llama.h"' in source
    assert '#include "llama-ext.h"' in source
    assert "llama_set_embeddings_layer_inp" in source
    assert "llama_get_embeddings_layer_inp" in source
    assert "linked_layer_input_api" in source


def test_capture_harness_source_loads_decodes_and_writes_hidden_in() -> None:
    source = hidden_in_capture_harness_source()

    required = [
        "llama_model_load_from_file",
        "llama_model_get_vocab",
        "llama_tokenize",
        "llama_init_from_model",
        "llama_set_embeddings_layer_inp",
        "llama_decode",
        "batch.n_tokens = n_tokens",
        "llama_get_embeddings_layer_inp",
        "llama_model_n_embd",
        "parse_prompt_tokens",
        "--prompt-tokens IDS",
        "prompt_token_source",
        "--all-rows",
        "all_rows_binary",
        "captured_hidden_in",
    ]
    for needle in required:
        assert needle in source
    assert "--model MODEL.gguf" in source
    assert "--output-prefix PATH" in source
    assert "provide exactly one of --prompt or --prompt-tokens" in source


def test_capture_harness_prompt_token_sequence_matches_checkpoint() -> None:
    source = hidden_in_capture_harness_source()
    prompt_tokens = [
        248045,
        846,
        198,
        7734,
        264,
        2716,
        40719,
        13,
        248046,
        198,
        248045,
        74455,
        198,
        248068,
        271,
        248069,
        271,
    ]
    csv = ",".join(str(token) for token in prompt_tokens)

    assert len(prompt_tokens) == 17
    assert prompt_tokens[16] == 271
    assert str(prompt_tokens[0]) in csv
    assert "prompt_tokens_csv" in source


def test_build_compile_command_has_expected_include_and_link_flags(tmp_path: Path) -> None:
    command = build_compile_command(
        compiler="c++",
        source_path=tmp_path / "probe.cpp",
        exe_path=tmp_path / "probe",
        source_dir=tmp_path / "source",
        lib_dir=tmp_path / "build" / "bin",
    )

    assert "-std=c++17" in command
    assert str(tmp_path / "source" / "include") in command
    assert str(tmp_path / "source" / "src") in command
    assert str(tmp_path / "source" / "ggml" / "include") in command
    assert str(tmp_path / "build" / "bin") in command
    assert f"-Wl,-rpath,{tmp_path / 'build' / 'bin'}" in command
    assert "-lllama" in command
    assert "-llama" not in command


def test_compile_hidden_in_harness_with_fake_compiler(tmp_path: Path) -> None:
    build_result_path, source_dir, build_dir = _write_build_result(tmp_path)
    compiler = _write_fake_compiler(tmp_path, fail=False)

    artifact = compile_hidden_in_harness(
        build_result_path=build_result_path,
        output_dir=tmp_path / "harness",
        compiler=str(compiler),
        timeout_seconds=30,
        env={"PATH": _prepend_path(compiler.parent)},
    )

    assert artifact["status"] == "compiled"
    assert artifact["source_dir"] == str(source_dir)
    assert artifact["build_dir"] == str(build_dir)
    assert artifact["compile"]["returncode"] == 0
    assert artifact["probe_run"]["returncode"] == 0
    assert "linked_layer_input_api" in artifact["probe_run"]["stdout_tail"]
    assert artifact["outputs"]["executable_exists"] is True
    assert "extend_harness" in artifact["next_action"]
    json.dumps(artifact)


def test_compile_capture_harness_skips_probe_run(tmp_path: Path) -> None:
    build_result_path, _, _ = _write_build_result(tmp_path)
    compiler = _write_fake_compiler(tmp_path, fail=False)

    artifact = compile_hidden_in_harness(
        build_result_path=build_result_path,
        output_dir=tmp_path / "harness",
        compiler=str(compiler),
        harness_kind="capture",
        timeout_seconds=30,
        env={"PATH": _prepend_path(compiler.parent)},
    )

    assert artifact["status"] == "compiled"
    assert artifact["harness_kind"] == "capture"
    assert artifact["probe_run"]["returncode"] is None
    assert "compile_only_capture_harness" in artifact["probe_run"]["stderr_tail"]
    assert artifact["next_action"] == (
        "run_hidden_in_capture_harness_with_model_prompt_layer_position"
    )


def test_compile_hidden_in_harness_reports_compile_failure(tmp_path: Path) -> None:
    build_result_path, _, _ = _write_build_result(tmp_path)
    compiler = _write_fake_compiler(tmp_path, fail=True)

    artifact = compile_hidden_in_harness(
        build_result_path=build_result_path,
        output_dir=tmp_path / "harness",
        compiler=str(compiler),
        timeout_seconds=30,
        env={"PATH": _prepend_path(compiler.parent)},
    )

    assert artifact["status"] == "compile_failed"
    assert artifact["compile"]["returncode"] == 17
    assert artifact["probe_run"]["returncode"] is None
    assert "compile failed intentionally" in artifact["compile"]["stderr_tail"]


def test_validate_headers_reports_expected_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "include").mkdir(parents=True)
    (source / "src").mkdir()
    (source / "ggml" / "include").mkdir(parents=True)
    (source / "include" / "llama.h").write_text("// llama\n")
    (source / "src" / "llama-ext.h").write_text("// ext\n")

    validation = validate_headers(source)

    assert validation["llama_h"]["exists"] is True
    assert validation["llama_ext_h"]["exists"] is True
    assert validation["ggml_include"]["exists"] is True


def _write_build_result(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_dir = tmp_path / "source"
    build_dir = tmp_path / "build"
    lib_dir = build_dir / "bin"
    (source_dir / "include").mkdir(parents=True)
    (source_dir / "src").mkdir()
    (source_dir / "ggml" / "include").mkdir(parents=True)
    (source_dir / "include" / "llama.h").write_text("// llama\n")
    (source_dir / "src" / "llama-ext.h").write_text("// ext\n")
    lib_dir.mkdir(parents=True)
    (lib_dir / "libllama.so").write_text("fake lib\n")
    build_result = {
        "status": "built",
        "source_dir": str(source_dir),
        "build_dir": str(build_dir),
        "outputs": {"libraries": [str(lib_dir / "libllama.so")]},
    }
    path = tmp_path / "build-result.json"
    path.write_text(json.dumps(build_result))
    return path, source_dir, build_dir


def _write_fake_compiler(tmp_path: Path, *, fail: bool) -> Path:
    tool_dir = tmp_path / "tools"
    tool_dir.mkdir()
    compiler = tool_dir / "fake-c++"
    compiler.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, stat, sys\n"
        "args = sys.argv[1:]\n"
        f"fail = {str(fail)}\n"
        "if fail:\n"
        "    print('compile failed intentionally', file=sys.stderr)\n"
        "    raise SystemExit(17)\n"
        "out = pathlib.Path(args[args.index('-o') + 1])\n"
        "out.write_text(\"#!/bin/sh\\necho '{\\\"linked_layer_input_api\\\":true,"
        "\\\"layer_id\\\":3}'\\n\")\n"
        "out.chmod(out.stat().st_mode | stat.S_IXUSR)\n"
        "print('fake compile ok')\n"
    )
    compiler.chmod(0o755)
    return compiler


def _prepend_path(path: Path) -> str:
    return str(path) + os.pathsep + os.environ.get("PATH", "")
