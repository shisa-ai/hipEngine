from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.llamacpp_mtp_compile_hidden_in_harness import build_compile_command
from scripts.llamacpp_mtp_compile_hidden_seed_harness import (
    compile_hidden_seed_harness,
    hidden_seed_capture_harness_source,
    hidden_seed_probe_source,
)


def test_hidden_seed_probe_source_links_nextn_symbols() -> None:
    source = hidden_seed_probe_source()

    assert '#include "llama.h"' in source
    assert '#include "llama-ext.h"' in source
    assert "llama_set_embeddings_nextn" in source
    assert "llama_get_embeddings_nextn" in source
    assert "llama_get_embeddings_nextn_ith" in source
    assert "linked_nextn_api" in source


def test_hidden_seed_capture_source_decodes_and_writes_nextn_row() -> None:
    source = hidden_seed_capture_harness_source()

    required = [
        "llama_model_load_from_file",
        "llama_model_get_vocab",
        "llama_tokenize",
        "llama_init_from_model",
        "llama_set_embeddings_nextn(ctx, true, false)",
        "llama_decode",
        "batch.n_tokens = n_tokens",
        "llama_get_embeddings_nextn_ith",
        "llama_get_embeddings_nextn(ctx)",
        "llama_model_n_embd_out",
        "parse_prompt_tokens",
        "--prompt-tokens IDS",
        "prompt_token_source",
        "h_nextn_post_output_norm",
        "row_index_semantics",
        "raw_prompt_position",
        "--all-rows",
        "all_rows_binary",
        "captured_hidden_seed",
    ]
    for needle in required:
        assert needle in source
    assert "provide exactly one of --prompt or --prompt-tokens" in source


def test_hidden_seed_capture_prompt_contract_matches_oracle() -> None:
    source = hidden_seed_capture_harness_source()
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

    assert len(prompt_tokens) == 17
    assert prompt_tokens[16] == 271
    assert "position = 16" in source
    assert "llama_model_n_embd_out" in source


def test_compile_hidden_seed_probe_with_fake_compiler(tmp_path: Path) -> None:
    build_result_path, source_dir, build_dir = _write_build_result(tmp_path)
    compiler = _write_fake_compiler(tmp_path, fail=False)

    artifact = compile_hidden_seed_harness(
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
    assert "linked_nextn_api" in artifact["probe_run"]["stdout_tail"]
    assert artifact["outputs"]["executable_exists"] is True
    assert artifact["next_action"] == "compile_hidden_seed_capture_harness"
    json.dumps(artifact)


def test_compile_hidden_seed_capture_skips_probe_run(tmp_path: Path) -> None:
    build_result_path, _, _ = _write_build_result(tmp_path)
    compiler = _write_fake_compiler(tmp_path, fail=False)

    artifact = compile_hidden_seed_harness(
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
        "run_hidden_seed_capture_harness_with_model_prompt_position"
    )


def test_compile_hidden_seed_harness_reports_compile_failure(tmp_path: Path) -> None:
    build_result_path, _, _ = _write_build_result(tmp_path)
    compiler = _write_fake_compiler(tmp_path, fail=True)

    artifact = compile_hidden_seed_harness(
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


def test_hidden_seed_compile_command_still_links_llama(tmp_path: Path) -> None:
    command = build_compile_command(
        compiler="c++",
        source_path=tmp_path / "probe.cpp",
        exe_path=tmp_path / "probe",
        source_dir=tmp_path / "source",
        lib_dir=tmp_path / "build" / "bin",
    )

    assert "-std=c++17" in command
    assert "-lllama" in command
    assert "-llama" not in command


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
        "import pathlib, stat, sys\n"
        "args = sys.argv[1:]\n"
        f"fail = {str(fail)}\n"
        "if fail:\n"
        "    print('compile failed intentionally', file=sys.stderr)\n"
        "    raise SystemExit(17)\n"
        "out = pathlib.Path(args[args.index('-o') + 1])\n"
        "out.write_text(\"#!/bin/sh\\necho '{\\\"linked_nextn_api\\\":true}'\\n\")\n"
        "out.chmod(out.stat().st_mode | stat.S_IXUSR)\n"
        "print('fake compile ok')\n"
    )
    compiler.chmod(0o755)
    return compiler


def _prepend_path(path: Path) -> str:
    return str(path) + os.pathsep + os.environ.get("PATH", "")
