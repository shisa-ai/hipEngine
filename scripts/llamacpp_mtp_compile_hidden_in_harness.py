#!/usr/bin/env python3
"""Compile llama.cpp hidden-in probe/capture harnesses.

The link probe is intentionally small: it includes llama.h and llama-ext.h,
takes the addresses of llama_set/get_embeddings_layer_inp so the extension
symbols must link, and prints a JSON-ish readiness line.

The capture harness is compile-ready for the next step: it accepts model/prompt
arguments, loads llama.cpp, decodes the prompt, and writes one layer-input row.
This helper only compiles it by default; it does not run the full model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

DEFAULT_BUILD_RESULT = Path(
    "benchmarks/results/mtp-gguf-iter289-llamacpp-build-result-amdclang.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter290-llamacpp-hidden-in-harness-compile.json"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/hipengine-llamacpp-mtp-iter290-hidden-in-harness")
HARNESS_NAMES = {
    "link-probe": "llamacpp_hidden_in_probe",
    "capture": "llamacpp_hidden_in_capture",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-result", type=Path, default=DEFAULT_BUILD_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler")
    parser.add_argument(
        "--harness-kind",
        choices=sorted(HARNESS_NAMES),
        default="link-probe",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--iteration", type=int, default=290)
    args = parser.parse_args()

    artifact = compile_hidden_in_harness(
        build_result_path=args.build_result,
        output_dir=args.output_dir,
        compiler=args.compiler,
        harness_kind=args.harness_kind,
        timeout_seconds=args.timeout_seconds,
        env=os.environ,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "compiler": artifact["compiler"],
                "harness_kind": artifact["harness_kind"],
                "executable": artifact["outputs"]["executable"],
                "probe_rc": artifact["probe_run"]["returncode"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def compile_hidden_in_harness(
    *,
    build_result_path: Path,
    output_dir: Path,
    compiler: str | None = None,
    harness_kind: str = "link-probe",
    timeout_seconds: int = 120,
    env: Mapping[str, str] | None = None,
    iteration: int = 290,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    build_result = json.loads(build_result_path.read_text())
    source_dir = Path(build_result["source_dir"])
    build_dir = Path(build_result["build_dir"])
    lib_dir = choose_lib_dir(build_result)
    if harness_kind not in HARNESS_NAMES:
        raise ValueError(f"unsupported harness kind: {harness_kind}")
    output_dir.mkdir(parents=True, exist_ok=True)
    harness_name = HARNESS_NAMES[harness_kind]
    source_path = output_dir / f"{harness_name}.cpp"
    exe_path = output_dir / harness_name
    source_path.write_text(select_harness_source(harness_kind))

    selected_compiler = compiler or choose_compiler(env_map)
    header_validation = validate_headers(source_dir)
    command = build_compile_command(
        compiler=selected_compiler,
        source_path=source_path,
        exe_path=exe_path,
        source_dir=source_dir,
        lib_dir=lib_dir,
    )
    compile_result = run_logged(
        command,
        cwd=Path.cwd(),
        env=env_map,
        stdout_path=output_dir / "compile.stdout.log",
        stderr_path=output_dir / "compile.stderr.log",
        timeout_seconds=timeout_seconds,
    )
    if compile_result["returncode"] == 0 and harness_kind == "link-probe":
        probe_run = run_probe(exe_path, lib_dir=lib_dir, env=env_map, timeout_seconds=30)
    elif compile_result["returncode"] == 0:
        probe_run = skipped_command([str(exe_path)], "compile_only_capture_harness")
    else:
        probe_run = skipped_command([str(exe_path)], "compile_failed")
    outputs = {
        "source": str(source_path),
        "executable": str(exe_path),
        "executable_exists": exe_path.exists(),
        "executable_bytes": exe_path.stat().st_size if exe_path.exists() else 0,
    }
    status = "compiled" if compile_result["returncode"] == 0 else "compile_failed"
    if (
        status == "compiled"
        and harness_kind == "link-probe"
        and probe_run["returncode"] != 0
    ):
        status = "probe_failed"
    return {
        "schema": 1,
        "kind": "llamacpp_hidden_in_harness_compile",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "harness_kind": harness_kind,
        "build_result_path": str(build_result_path),
        "source_dir": str(source_dir),
        "build_dir": str(build_dir),
        "lib_dir": str(lib_dir),
        "compiler": selected_compiler,
        "header_validation": header_validation,
        "link_symbols": [
            "llama_set_embeddings_layer_inp",
            "llama_get_embeddings_layer_inp",
        ],
        "compile": compile_result,
        "probe_run": probe_run,
        "outputs": outputs,
        "external_checkout_modified": False,
        "next_action": next_action(status, harness_kind),
    }


def select_harness_source(harness_kind: str) -> str:
    if harness_kind == "link-probe":
        return hidden_in_harness_source()
    if harness_kind == "capture":
        return hidden_in_capture_harness_source()
    raise ValueError(f"unsupported harness kind: {harness_kind}")


def hidden_in_harness_source() -> str:
    return r'''#include "llama.h"
#include "llama-ext.h"

#include <cstdio>
#include <cstdint>

int main() {
    auto set_layer_input = &llama_set_embeddings_layer_inp;
    auto get_layer_input = &llama_get_embeddings_layer_inp;
    if (set_layer_input == nullptr || get_layer_input == nullptr) {
        return 2;
    }
    std::printf("{\"linked_layer_input_api\":true,\"layer_id\":3}\n");
    return 0;
}
'''


def hidden_in_capture_harness_source() -> str:
    return r'''#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

struct Args {
    std::string model;
    std::string prompt;
    std::string prompt_tokens_csv;
    std::string output_prefix;
    int layer = 3;
    int position = 16;
    int n_gpu_layers = 999;
    int n_threads = 8;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model MODEL.gguf (--prompt TEXT | --prompt-tokens IDS) "
        "--layer 3 --position 16 --output-prefix PATH [--n-gpu-layers N]\n",
        argv0);
}

static bool parse_int(const char * text, int * out) {
    char * end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    if (value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = (int) value;
    return true;
}

static bool parse_args(int argc, char ** argv, Args * args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (key == "--model") {
            const char * value = need_value("--model");
            if (!value) return false;
            args->model = value;
        } else if (key == "--prompt") {
            const char * value = need_value("--prompt");
            if (!value) return false;
            args->prompt = value;
        } else if (key == "--prompt-tokens") {
            const char * value = need_value("--prompt-tokens");
            if (!value) return false;
            args->prompt_tokens_csv = value;
        } else if (key == "--output-prefix") {
            const char * value = need_value("--output-prefix");
            if (!value) return false;
            args->output_prefix = value;
        } else if (key == "--layer") {
            const char * value = need_value("--layer");
            if (!value || !parse_int(value, &args->layer)) return false;
        } else if (key == "--position") {
            const char * value = need_value("--position");
            if (!value || !parse_int(value, &args->position)) return false;
        } else if (key == "--n-gpu-layers") {
            const char * value = need_value("--n-gpu-layers");
            if (!value || !parse_int(value, &args->n_gpu_layers)) return false;
        } else if (key == "--threads") {
            const char * value = need_value("--threads");
            if (!value || !parse_int(value, &args->n_threads)) return false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", key.c_str());
            return false;
        }
    }
    const bool has_prompt = !args->prompt.empty();
    const bool has_prompt_tokens = !args->prompt_tokens_csv.empty();
    if (has_prompt == has_prompt_tokens) {
        std::fprintf(stderr, "provide exactly one of --prompt or --prompt-tokens\n");
        return false;
    }
    return !args->model.empty() && !args->output_prefix.empty() &&
        args->layer >= 0 && args->position >= 0;
}

static bool parse_prompt_tokens(
        const std::string & csv,
        std::vector<llama_token> * tokens) {
    tokens->clear();
    size_t begin = 0;
    while (begin <= csv.size()) {
        const size_t comma = csv.find(',', begin);
        const size_t end = comma == std::string::npos ? csv.size() : comma;
        const std::string item = csv.substr(begin, end - begin);
        int value = 0;
        if (item.empty() || !parse_int(item.c_str(), &value)) {
            return false;
        }
        tokens->push_back((llama_token) value);
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return !tokens->empty();
}

static int tokenize_prompt(
        const llama_vocab * vocab,
        const std::string & prompt,
        std::vector<llama_token> * tokens) {
    int32_t n = llama_tokenize(
        vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, true);
    if (n == 0 || n == INT32_MIN) {
        return -1;
    }
    if (n < 0) {
        n = -n;
    }
    tokens->assign((size_t) n, 0);
    const int32_t got = llama_tokenize(
        vocab, prompt.c_str(), (int32_t) prompt.size(),
        tokens->data(), n, true, true);
    if (got < 0 || got > n) {
        return -2;
    }
    tokens->resize((size_t) got);
    return got;
}

static bool write_binary(const std::string & path, const float * data, size_t count) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char *>(data), (std::streamsize) (count * sizeof(float)));
    return (bool) out;
}

static bool write_meta(
        const std::string & path,
        const Args & args,
        int n_tokens,
        int n_embd,
        const std::string & bin_path) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << "{\n";
    out << "  \"kind\": \"llamacpp_hidden_in_capture\",\n";
    out << "  \"model\": \"" << args.model << "\",\n";
    out << "  \"prompt_token_source\": \""
        << (args.prompt_tokens_csv.empty() ? "text" : "token_ids") << "\",\n";
    out << "  \"prompt_token_count\": " << n_tokens << ",\n";
    out << "  \"layer\": " << args.layer << ",\n";
    out << "  \"position\": " << args.position << ",\n";
    out << "  \"n_embd\": " << n_embd << ",\n";
    out << "  \"dtype\": \"float32\",\n";
    out << "  \"binary\": \"" << bin_path << "\"\n";
    out << "}\n";
    return (bool) out;
}

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        usage(argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", args.model.c_str());
        llama_backend_free();
        return 3;
    }

    std::vector<llama_token> tokens;
    if (!args.prompt_tokens_csv.empty()) {
        if (!parse_prompt_tokens(args.prompt_tokens_csv, &tokens)) {
            std::fprintf(stderr, "failed to parse --prompt-tokens\n");
            llama_model_free(model);
            llama_backend_free();
            return 4;
        }
    } else {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int tokenized = tokenize_prompt(vocab, args.prompt, &tokens);
        if (tokenized <= 0) {
            std::fprintf(stderr, "failed to tokenize prompt\n");
            llama_model_free(model);
            llama_backend_free();
            return 4;
        }
    }
    const int n_tokens = (int) tokens.size();
    if (args.position >= n_tokens) {
        std::fprintf(stderr, "position %d out of range for %d tokens\n",
            args.position, n_tokens);
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) std::max(128, n_tokens + 1);
    cparams.n_batch = (uint32_t) n_tokens;
    cparams.n_ubatch = (uint32_t) n_tokens;
    cparams.n_threads = args.n_threads;
    cparams.n_threads_batch = args.n_threads;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create llama context\n");
        llama_model_free(model);
        llama_backend_free();
        return 6;
    }

    llama_set_embeddings_layer_inp(ctx, (uint32_t) args.layer, true);
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[(size_t) i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == args.position ? 1 : 0;
    }
    const int decode_rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (decode_rc != 0) {
        std::fprintf(stderr, "llama_decode failed: %d\n", decode_rc);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 7;
    }

    float * layer_input = llama_get_embeddings_layer_inp(ctx, (uint32_t) args.layer);
    if (!layer_input) {
        std::fprintf(stderr, "llama_get_embeddings_layer_inp returned null\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 8;
    }
    const int n_embd = llama_model_n_embd(model);
    const float * row = layer_input + (size_t) args.position * (size_t) n_embd;
    const std::string bin_path = args.output_prefix + ".f32";
    const std::string meta_path = args.output_prefix + ".json";
    if (!write_binary(bin_path, row, (size_t) n_embd)) {
        std::fprintf(stderr, "failed to write %s\n", bin_path.c_str());
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 9;
    }
    if (!write_meta(meta_path, args, n_tokens, n_embd, bin_path)) {
        std::fprintf(stderr, "failed to write %s\n", meta_path.c_str());
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 10;
    }

    std::printf(
        "{\"captured_hidden_in\":true,\"layer\":%d,\"position\":%d,"
        "\"n_embd\":%d,\"binary\":\"%s\"}\n",
        args.layer, args.position, n_embd, bin_path.c_str());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
'''


def validate_headers(source_dir: Path) -> dict[str, Any]:
    return {
        "llama_h": {
            "path": str(source_dir / "include" / "llama.h"),
            "exists": (source_dir / "include" / "llama.h").exists(),
        },
        "llama_ext_h": {
            "path": str(source_dir / "src" / "llama-ext.h"),
            "exists": (source_dir / "src" / "llama-ext.h").exists(),
        },
        "ggml_include": {
            "path": str(source_dir / "ggml" / "include"),
            "exists": (source_dir / "ggml" / "include").exists(),
        },
    }


def choose_lib_dir(build_result: dict[str, Any]) -> Path:
    libraries = build_result.get("outputs", {}).get("libraries", [])
    for library in libraries:
        path = Path(library)
        if path.name == "libllama.so":
            return path.parent
    if libraries:
        return Path(libraries[0]).parent
    return Path(build_result["build_dir"]) / "bin"


def choose_compiler(env: Mapping[str, str]) -> str:
    path = env.get("PATH")
    for name in ("amdclang++", "clang++", "c++", "g++"):
        resolved = shutil.which(name, path=path)
        if resolved:
            return resolved
    return "c++"


def build_compile_command(
    *,
    compiler: str,
    source_path: Path,
    exe_path: Path,
    source_dir: Path,
    lib_dir: Path,
) -> list[str]:
    return [
        compiler,
        "-std=c++17",
        "-O2",
        "-I",
        str(source_dir / "include"),
        "-I",
        str(source_dir / "src"),
        "-I",
        str(source_dir / "ggml" / "include"),
        str(source_path),
        "-L",
        str(lib_dir),
        f"-Wl,-rpath,{lib_dir}",
        "-Wl,--enable-new-dtags",
        "-lllama",
        "-o",
        str(exe_path),
    ]


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(env),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = decode_timeout_stream(exc.stdout)
        stderr = decode_timeout_stream(exc.stderr)
        stderr += f"\nTIMEOUT after {timeout_seconds}s"
        timed_out = True
    elapsed = time.monotonic() - start
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    return {
        "command": command,
        "command_shell": subprocess.list2cmdline(command),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 3),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_tail": tail_text(stdout),
        "stderr_tail": tail_text(stderr),
    }


def run_probe(
    exe_path: Path,
    *,
    lib_dir: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    run_env = dict(env)
    old_ld_path = run_env.get("LD_LIBRARY_PATH", "")
    run_env["LD_LIBRARY_PATH"] = str(lib_dir) + (os.pathsep + old_ld_path if old_ld_path else "")
    return run_logged(
        [str(exe_path)],
        cwd=Path.cwd(),
        env=run_env,
        stdout_path=exe_path.with_suffix(".stdout.log"),
        stderr_path=exe_path.with_suffix(".stderr.log"),
        timeout_seconds=timeout_seconds,
    )


def skipped_command(command: list[str], reason: str) -> dict[str, Any]:
    return {
        "command": command,
        "command_shell": subprocess.list2cmdline(command),
        "returncode": None,
        "timed_out": False,
        "elapsed_seconds": 0.0,
        "stdout_log": None,
        "stderr_log": None,
        "stdout_tail": "",
        "stderr_tail": reason,
    }


def next_action(status: str, harness_kind: str) -> str:
    if status == "compiled" and harness_kind == "capture":
        return "run_hidden_in_capture_harness_with_model_prompt_layer_position"
    if status == "compiled":
        return "extend_harness_to_load_model_decode_prompt_and_dump_hidden_in"
    if status == "probe_failed":
        return "inspect_harness_runtime_linker_or_rocm_library_path"
    return "inspect_hidden_in_harness_compile_logs"


def decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def tail_text(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


if __name__ == "__main__":
    main()
