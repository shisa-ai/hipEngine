from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.llamacpp_mtp_build_input_embed_harness import (
    apply_input_embed_patch,
    build_input_embed_harness,
)
from scripts.llamacpp_mtp_build_pre_output_norm_harness import (
    apply_post_output_preserve_patch,
)

INPUT_OLD = (
    "    inpL = build_inp_embd(model.tok_embd);\n\n"
    "    cb(inpL, \"model.input_embed\", -1);\n\n"
    "    auto * inp = build_inp_mem_hybrid();"
)
INPUT_NEW = (
    "    inpL = build_inp_embd(model.tok_embd);\n\n"
    "    cb(inpL, \"model.input_embed\", -1);\n"
    "    cb(inpL, \"h_nextn_input_embed\", -1);\n"
    "    res->t_h_nextn = inpL;\n\n"
    "    auto * inp = build_inp_mem_hybrid();"
)
POST_OUTPUT_OLD = (
    "    cb(cur, \"h_nextn\", -1);\n"
    "    res->t_h_nextn = cur;\n\n"
    "    if (!cparams.embeddings_nextn_masked && inp_out_ids) {"
)


def test_apply_input_embed_patch_and_preserve_are_idempotent(tmp_path: Path) -> None:
    source = _write_source_tree(tmp_path / "src")

    first = apply_input_embed_patch(
        source_dir=source,
        old_text=INPUT_OLD,
        new_text=INPUT_NEW,
    )
    second = apply_input_embed_patch(
        source_dir=source,
        old_text=INPUT_OLD,
        new_text=INPUT_NEW,
    )
    preserve = apply_post_output_preserve_patch(source_dir=source)

    text = (source / "src" / "models" / "qwen35moe.cpp").read_text()
    assert first["status"] == "applied"
    assert first["capture_label"] == "h_nextn_input_embed"
    assert second["status"] == "already_applied"
    assert preserve["status"] == "applied"
    assert text.count("h_nextn_input_embed") == 1
    assert text.count("h_nextn_post_output_norm") == 1
    assert text.count("h_nextn_layer_out") == 0
    assert text.count("h_nextn_pre_output_norm") == 0
    assert text.count("res->t_h_nextn =") == 1


def test_build_input_embed_harness_with_fake_tools(tmp_path: Path) -> None:
    source = _write_source_tree(tmp_path / "base-src")
    tools = tmp_path / "tools"
    tools.mkdir()
    _write_fake_cmake(tools / "cmake")
    compiler = _write_fake_compiler(tools / "fake-c++")
    base_build = _write_base_build(tmp_path, source)
    plan = _write_plan(tmp_path)

    artifact = build_input_embed_harness(
        plan_path=plan,
        base_build_path=base_build,
        output_path=tmp_path / "summary.json",
        patched_build_result_path=tmp_path / "patched-build.json",
        harness_compile_path=tmp_path / "harness-compile.json",
        target_source_dir=tmp_path / "patched-src",
        target_build_dir=tmp_path / "patched-build",
        log_dir=tmp_path / "logs",
        harness_dir=tmp_path / "harness",
        compiler=str(compiler),
        jobs=2,
        timeout_seconds=30,
        clean=True,
        env={"PATH": str(tools) + os.pathsep + os.environ.get("PATH", "")},
    )

    assert artifact["status"] == "ready"
    assert artifact["patch"]["applied"] is True
    assert artifact["patch"]["post_output_preserve"]["applied"] is True
    assert artifact["patch"]["layer_capture_patch_count"] == 0
    assert artifact["patch"]["final_pre_output_patch_count"] == 0
    assert artifact["commands"]["configure"]["returncode"] == 0
    assert artifact["commands"]["build"]["returncode"] == 0
    assert artifact["harness_compile"]["status"] == "compiled"
    assert artifact["libraries"][0].endswith("libllama.so")
    assert artifact["next_action"] == "run_input_embed_harness_and_compare_hipengine_hidden_in"
    assert (tmp_path / "patched-build.json").exists()
    assert (tmp_path / "harness-compile.json").exists()
    json.dumps(artifact)


def _write_source_tree(path: Path) -> Path:
    (path / "src" / "models").mkdir(parents=True)
    (path / "include").mkdir()
    (path / "ggml" / "include").mkdir(parents=True)
    (path / "include" / "llama.h").write_text("// llama\n")
    (path / "src" / "llama-ext.h").write_text("// ext\n")
    (path / "src" / "models" / "qwen35moe.cpp").write_text(
        INPUT_OLD + "\n\n" + POST_OUTPUT_OLD + "\n"
    )
    return path


def _write_base_build(tmp_path: Path, source: Path) -> Path:
    build_dir = tmp_path / "base-build"
    build_dir.mkdir()
    artifact = {
        "source_dir": str(source),
        "build_dir": str(build_dir),
        "commands": {
            "configure": {
                "command": ["cmake", "-S", str(source), "-B", str(build_dir)]
            },
            "build": {
                "command": ["cmake", "--build", str(build_dir), "--target", "llama"]
            },
        },
    }
    path = tmp_path / "base-build.json"
    path.write_text(json.dumps(artifact))
    return path


def _write_plan(tmp_path: Path) -> Path:
    artifact = {
        "llamacpp_input_patch": {
            "input_capture_old_text": INPUT_OLD,
            "input_capture_new_text": INPUT_NEW,
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(artifact))
    return path


def _write_fake_cmake(path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if '-S' in args:\n"
        "    build = pathlib.Path(args[args.index('-B') + 1])\n"
        "    (build / 'bin').mkdir(parents=True, exist_ok=True)\n"
        "    (build / 'CMakeCache.txt').write_text('fake cache\\n')\n"
        "    raise SystemExit(0)\n"
        "if '--build' in args:\n"
        "    build = pathlib.Path(args[args.index('--build') + 1])\n"
        "    (build / 'bin').mkdir(parents=True, exist_ok=True)\n"
        "    for name in ['libllama.so', 'libllama.so.0', 'libllama.so.0.0.0']:\n"
        "        (build / 'bin' / name).write_text('fake lib\\n')\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(2)\n"
    )
    path.chmod(0o755)
    return path


def _write_fake_compiler(path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, stat, sys\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('-o') + 1])\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_text(\"#!/bin/sh\\necho fake capture\\n\")\n"
        "out.chmod(out.stat().st_mode | stat.S_IXUSR)\n"
    )
    path.chmod(0o755)
    return path
