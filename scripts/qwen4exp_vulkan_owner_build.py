"""Build metadata-only Qwen4Exp/Vulkan profiling libraries in an owned directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = "b212548e0ddbf0a14e5a1d81b6ffcf8e4d098faf"
SCOPES = {
    "graph::graph(const llama_model & model, const llm_graph_params & params) :": (
        "boundary",
        True,
    ),
    "graph::build_hc_mix(": ("gr_read", False),
    "graph::build_hc_combine(": ("boundary", False),
    "graph::build_qkvz(": ("gdn", False),
    "graph::build_norm_gated(": ("gdn", False),
    "graph::build_qsa_top_k(": ("qsa", False),
    "graph::build_attn_qsa(": ("qsa", False),
    "graph::build_layer_attn(": ("qsa", False),
    "graph::build_layer_attn_linear(": ("gdn", False),
    "graph::build_layer_ffn(": ("moe", False),
    "graph::build_inp_ple(": ("ple", False),
    "graph::build_ple(": ("ple", False),
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"source anchor must occur once: {old[:80]!r}")
    return text.replace(old, new, 1)


def instrument_model(text):
    for suffix, (owner, root) in SCOPES.items():
        needle = "llama_model_qwen4exp::" + suffix
        if text.count(needle) != 1:
            raise ValueError(f"model anchor must occur once: {needle}")
        pos = text.index("{", text.index(needle)) + 1
        text = (
            text[:pos] + f'\n    hipengine_profile::owner_scope he_owner(ctx0, "{owner}", '
            f"{'true' if root else 'false'});\n" + text[pos:]
        )
    return text


def instrument_vulkan(text):
    text = replace_once(
        text,
        "    std::vector<ggml_tensor *> query_nodes;",
        "    std::vector<ggml_tensor *> query_nodes;\n"
        "    std::vector<std::string> query_owner_members;",
    )
    text = replace_once(
        text,
        "            ctx->query_nodes.resize(ctx->num_queries);",
        "            ctx->query_nodes.resize(ctx->num_queries);\n"
        "            ctx->query_owner_members.resize(ctx->num_queries);",
    )
    anchor = "                ctx->query_nodes[ctx->query_idx] = cgraph->nodes[i];"
    text = replace_once(
        text,
        anchor,
        anchor
        + """
                std::string members;
                if (std::getenv("HIPENGINE_VK_OWNER_TRACE")) {
                    members = " HE_NODES=";
                    for (int j = i; j <= i + (int)ctx->num_additional_fused_ops; ++j) {
                        char ptr[32];
                        std::snprintf(ptr, sizeof(ptr), "%p", (void *)cgraph->nodes[j]);
                        if (j != i) members += ",";
                        members += ptr;
                    }
                }
                ctx->query_owner_members[ctx->query_idx] = members;
""",
    )
    text = replace_once(
        text,
        "void log_timing(const ggml_tensor * node, const char *fusion_name, uint64_t time) {",
        "void log_timing(const ggml_tensor * node, const char *fusion_name, uint64_t time,\n"
        '                    const std::string & members = "") {',
    )
    text = replace_once(
        text,
        "        std::string name = get_node_fusion_name(node, fusion_name, &n_flops);",
        "        std::string name = get_node_fusion_name(node, fusion_name, &n_flops) + members;",
    )
    text = replace_once(
        text,
        "ctx->perf_logger->log_timing(node, name, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));",
        "ctx->perf_logger->log_timing(node, name, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod), ctx->query_owner_members[i]);",
    )
    return text


def compile_command(record, source, output, header=None):
    command = shlex.split(record["command"])
    if any(arg in {"-MD", "-MMD", "-MF", "-MT", "-MQ"} for arg in command):
        raise ValueError("unexpected dependency-output flags")
    command[command.index("-o") + 1] = str(output)
    command[command.index("-c") + 1] = str(source)
    command += ["-I" + str(Path(record["file"]).parent)]
    if header:
        command += ["-include", str(header)]
    return command


def link_command(text, old_object, new_object, output):
    command = [arg for arg in shlex.split(text) if not arg.startswith("-Wl,--dependency-file=")]
    if command.count(old_object) != 1:
        raise ValueError(f"link object not unique: {old_object}")
    command[command.index(old_object)] = str(new_object)
    command[command.index("-o") + 1] = str(output)
    return command


def build(source_root, build_root, output):
    source_root, build_root, output = map(Path.resolve, (source_root, build_root, output))
    if output == source_root or source_root in output.parents or build_root in output.parents:
        raise ValueError("owned output must be outside reference source/build")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    if head != PIN:
        raise ValueError(f"expected pinned halo-box {PIN}, got {head}")
    if subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=source_root):
        raise ValueError("reference source must be clean")
    output.mkdir(parents=True, exist_ok=False)
    records = json.loads((build_root / "compile_commands.json").read_text())
    manifest = {
        "schema": 1,
        "kind": "qwen4exp_vulkan_owner_build",
        "upstream_commit": head,
        "source_root": str(source_root),
        "build_root": str(build_root),
        "output": str(output),
        "commands": [],
        "files": {},
    }
    specs = [
        (
            "src/models/qwen4exp.cpp",
            instrument_model,
            "src",
            "llama",
            "CMakeFiles/llama.dir/models/qwen4exp.cpp.o",
            "libllama.so.0.3.0",
        ),
        (
            "ggml/src/ggml-vulkan/ggml-vulkan.cpp",
            instrument_vulkan,
            "ggml/src/ggml-vulkan",
            "ggml-vulkan",
            "CMakeFiles/ggml-vulkan.dir/ggml-vulkan.cpp.o",
            "libggml-vulkan.so.0.22.0",
        ),
    ]
    for rel, transform, cwd_rel, target, old_object, library in specs:
        original = source_root / rel
        record = next(r for r in records if Path(r["file"]).resolve() == original)
        copy = output / original.name
        original_hash = digest(original)
        copy.write_text(transform(original.read_text()))
        obj = output / (original.name + ".o")
        header = ROOT / "scripts/profiling/qwen4exp_owner_scope.hpp" if target == "llama" else None
        command = compile_command(record, copy, obj, header)
        subprocess.run(command, cwd=record["directory"], check=True)
        link_text = (build_root / cwd_rel / f"CMakeFiles/{target}.dir/link.txt").read_text()
        link = link_command(link_text, old_object, obj, output / library)
        subprocess.run(link, cwd=build_root / cwd_rel, check=True)
        soname = library.split(".so.")[0] + ".so.0"
        (output / soname).symlink_to(library)
        manifest["commands"] += [command, link]
        manifest["files"][str(original)] = original_hash
        manifest["files"][str(copy)] = digest(copy)
        manifest["files"][str(output / library)] = digest(output / library)
        if digest(original) != original_hash:
            raise RuntimeError("reference source changed during build")
    server = build_root / "bin/llama-server"
    shutil.copy2(server, output / "llama-server")
    for library in (build_root / "bin").glob("lib*.so*"):
        target = output / library.name
        if not target.exists():
            if library.name == "libllama.so":
                target.symlink_to("libllama.so.0")
            elif library.name == "libggml-vulkan.so":
                target.symlink_to("libggml-vulkan.so.0")
            else:
                target.symlink_to(library.resolve())
    manifest["files"][str(server)] = digest(server)
    manifest["owner_header_sha256"] = digest(ROOT / "scripts/profiling/qwen4exp_owner_scope.hpp")
    (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--build-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    build(args.source_root, args.build_root, args.output)


if __name__ == "__main__":
    main()
