#!/usr/bin/env python3
"""Generate a local llama.cpp patch for early linear-attention tensor traces.

The external llama.cpp checkout is treated as a reference.  This helper emits a
unified diff that extends the existing local MTP tensor trace instrumentation so
the Qwen35MoE target graph can expose early linear-attention taps before
``ssm_out``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any

DEFAULT_LLAMA_CPP_ROOT = Path("/home/lhl/llama.cpp/llama.cpp-hip")
DEFAULT_PATCH_OUTPUT = Path(
    "benchmarks/results/2026-07-02-llamacpp-linear-attn-trace-allowlist.patch"
)
DEFAULT_JSON_OUTPUT = Path(
    "benchmarks/results/2026-07-02-llamacpp-linear-attn-trace-allowlist.json"
)
GRAPH_RELATIVE_PATH = Path("src/llama-graph.cpp")
CONTEXT_RELATIVE_PATH = Path("src/llama-context.cpp")

TRACE_PREFIX_LINES = (
    '             std::strncmp(name, "linear_attn_qkv_mixed_", 22) == 0 ||\n',
    '             std::strncmp(name, "z_", 2) == 0 ||\n',
    '             std::strncmp(name, "beta_", 5) == 0 ||\n',
    '             std::strncmp(name, "alpha_", 6) == 0 ||\n',
    '             std::strncmp(name, "a_softplus_", 11) == 0 ||\n',
    '             std::strncmp(name, "gate_", 5) == 0 ||\n',
    '             std::strncmp(name, "conv_output_raw_", 16) == 0 ||\n',
    '             std::strncmp(name, "conv_output_silu_", 17) == 0 ||\n',
    '             std::strncmp(name, "q_conv_", 7) == 0 ||\n',
    '             std::strncmp(name, "k_conv_", 7) == 0 ||\n',
    '             std::strncmp(name, "v_conv_", 7) == 0 ||\n',
)

RENAME_LINES = (
    '             std::strcmp(name, "linear_attn_qkv_mixed") == 0 ||\n',
    '             std::strcmp(name, "z") == 0 ||\n',
    '             std::strcmp(name, "beta") == 0 ||\n',
    '             std::strcmp(name, "beta_sigmoid") == 0 ||\n',
    '             std::strcmp(name, "alpha") == 0 ||\n',
    '             std::strcmp(name, "a_softplus") == 0 ||\n',
    '             std::strcmp(name, "gate") == 0 ||\n',
    '             std::strcmp(name, "conv_output_raw") == 0 ||\n',
    '             std::strcmp(name, "conv_output_silu") == 0 ||\n',
    '             std::strcmp(name, "q_conv") == 0 ||\n',
    '             std::strcmp(name, "k_conv") == 0 ||\n',
    '             std::strcmp(name, "v_conv") == 0 ||\n',
    '             std::strcmp(name, "q_conv_predelta") == 0 ||\n',
    '             std::strcmp(name, "k_conv_predelta") == 0 ||\n',
    '             std::strcmp(name, "v_conv_predelta") == 0 ||\n',
)

TOKEN_DIM_LINES = (
    '    if (label.rfind("beta_", 0) == 0 ||\n',
    '            label.rfind("q_conv_", 0) == 0 ||\n',
    '            label.rfind("k_conv_", 0) == 0 ||\n',
    '            label.rfind("v_conv_", 0) == 0) {\n',
    "        return 2;\n",
    "    }\n",
)

GRAPH_PREFIX_ANCHOR = '            (std::strncmp(name, "attn_norm_", 10) == 0 ||\n'
GRAPH_TARGET_PREFIX_ANCHOR = '         std::strncmp(name, "attn_norm_", 10) == 0 ||\n'
GRAPH_RENAME_ANCHOR = '            (std::strcmp(name, "attn_norm") == 0 ||\n'
CONTEXT_TOKEN_DIM_ANCHOR = '    if (label.rfind("ffn_moe_gate_up_", 0) == 0 ||\n'


@dataclass(frozen=True)
class SourcePatch:
    relative_path: Path
    original_text: str
    patched_text: str

    @property
    def changed(self) -> bool:
        return self.original_text != self.patched_text


@dataclass(frozen=True)
class PatchBuildResult:
    status: str
    graph: SourcePatch
    context: SourcePatch
    reason: str | None = None

    @property
    def changed(self) -> bool:
        return self.graph.changed or self.context.changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llamacpp-root", type=Path, default=DEFAULT_LLAMA_CPP_ROOT)
    parser.add_argument("--patch-output", type=Path, default=DEFAULT_PATCH_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--iteration", type=int, default=1)
    args = parser.parse_args()

    artifact = build_patch_artifact(
        llamacpp_root=args.llamacpp_root,
        patch_output=args.patch_output,
        iteration=args.iteration,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "patch_output": artifact["patch_output"],
                "patch_sha256": artifact["patch_sha256"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_patch_artifact(
    *, llamacpp_root: Path, patch_output: Path, iteration: int = 1
) -> dict[str, Any]:
    root = llamacpp_root.resolve()
    graph_path = root / GRAPH_RELATIVE_PATH
    context_path = root / CONTEXT_RELATIVE_PATH
    graph_text = graph_path.read_text() if graph_path.exists() else ""
    context_text = context_path.read_text() if context_path.exists() else ""
    result = build_linear_attn_trace_patch_text(
        graph_text=graph_text,
        context_text=context_text,
    )
    diff_text = render_combined_diff(result)
    patch_output.parent.mkdir(parents=True, exist_ok=True)
    patch_output.write_text(diff_text)
    patch_sha256 = hashlib.sha256(diff_text.encode()).hexdigest()
    return {
        "schema": 1,
        "kind": "llamacpp_linear_attn_trace_allowlist_patch",
        "date": "2026-07-02",
        "iteration": int(iteration),
        "status": result.status,
        "reason": result.reason,
        "llamacpp_root": str(root),
        "source_exists": {
            str(GRAPH_RELATIVE_PATH): graph_path.exists(),
            str(CONTEXT_RELATIVE_PATH): context_path.exists(),
        },
        "reference_basis": {
            "observed_llamacpp_commit": _git_rev_parse(root),
            "source_is_read_only_reference": True,
            "external_checkout_modified": False,
        },
        "targets": [str(GRAPH_RELATIVE_PATH), str(CONTEXT_RELATIVE_PATH)],
        "patch_output": str(patch_output),
        "patch_sha256": patch_sha256,
        "patch_bytes": len(diff_text.encode()),
        "validation": summarize_patch_validation(result, diff_text),
        "trace_labels_enabled": {
            "projection_inputs": [
                "linear_attn_qkv_mixed_0",
                "z_0",
                "alpha_0",
                "beta_0",
            ],
            "conv_gdn_inputs_outputs": [
                "conv_output_raw_0",
                "conv_output_silu_0",
                "q_conv_0",
                "k_conv_0",
                "v_conv_0",
                "q_conv_predelta_0",
                "k_conv_predelta_0",
                "v_conv_predelta_0",
            ],
        },
        "next_action": next_action(result),
    }


def build_linear_attn_trace_patch_text(*, graph_text: str, context_text: str) -> PatchBuildResult:
    graph_status = audit_graph_trace_support(graph_text)
    context_status = audit_context_trace_support(context_text)
    if graph_status["ready"] and context_status["ready"]:
        return PatchBuildResult(
            status="already_wired",
            graph=SourcePatch(GRAPH_RELATIVE_PATH, graph_text, graph_text),
            context=SourcePatch(CONTEXT_RELATIVE_PATH, context_text, context_text),
            reason="llama.cpp trace allowlist already includes early linear-attention labels",
        )

    graph_patched = patch_graph_trace_support(graph_text)
    context_patched = patch_context_trace_support(context_text)
    if graph_patched == graph_text and not graph_status["ready"]:
        return PatchBuildResult(
            status="graph_anchor_missing",
            graph=SourcePatch(GRAPH_RELATIVE_PATH, graph_text, graph_text),
            context=SourcePatch(CONTEXT_RELATIVE_PATH, context_text, context_patched),
            reason="expected llama-graph.cpp trace anchors were not found",
        )
    if context_patched == context_text and not context_status["ready"]:
        return PatchBuildResult(
            status="context_anchor_missing",
            graph=SourcePatch(GRAPH_RELATIVE_PATH, graph_text, graph_patched),
            context=SourcePatch(CONTEXT_RELATIVE_PATH, context_text, context_text),
            reason="expected llama-context.cpp token-dimension anchor was not found",
        )
    return PatchBuildResult(
        status="patch_ready",
        graph=SourcePatch(GRAPH_RELATIVE_PATH, graph_text, graph_patched),
        context=SourcePatch(CONTEXT_RELATIVE_PATH, context_text, context_patched),
    )


def audit_graph_trace_support(text: str) -> dict[str, Any]:
    wanted = [
        "linear_attn_qkv_mixed_",
        "conv_output_silu_",
        "q_conv_",
        "v_conv_",
    ]
    renames = [
        '"linear_attn_qkv_mixed"',
        '"conv_output_silu"',
        '"q_conv_predelta"',
        '"v_conv_predelta"',
    ]
    prefix_counts = {item: text.count(item) for item in wanted}
    return {
        "ready": all(count >= 2 for count in prefix_counts.values())
        and all(item in text for item in renames),
        "wanted_prefix_counts": prefix_counts,
        "rename_labels_present": {item: item in text for item in renames},
    }


def audit_context_trace_support(text: str) -> dict[str, Any]:
    wanted = ['label.rfind("beta_", 0)', 'label.rfind("q_conv_", 0)']
    return {
        "ready": all(item in text for item in wanted),
        "token_dim_prefixes_present": {item: item in text for item in wanted},
    }


def patch_graph_trace_support(text: str) -> str:
    patched = text
    if patched.count("linear_attn_qkv_mixed_") < 2:
        wants_block = GRAPH_PREFIX_ANCHOR + "".join(TRACE_PREFIX_LINES)
        patched = replace_n(patched, GRAPH_PREFIX_ANCHOR, wants_block, count=1)
        target_lines = "".join(line.replace("             ", "         ", 1) for line in TRACE_PREFIX_LINES)
        target_block = GRAPH_TARGET_PREFIX_ANCHOR + target_lines
        patched = replace_n(patched, GRAPH_TARGET_PREFIX_ANCHOR, target_block, count=1)
    if 'std::strcmp(name, "linear_attn_qkv_mixed")' not in patched:
        block = GRAPH_RENAME_ANCHOR + "".join(RENAME_LINES)
        patched = replace_n(patched, GRAPH_RENAME_ANCHOR, block, count=1)
    return patched


def patch_context_trace_support(text: str) -> str:
    if 'label.rfind("q_conv_", 0)' in text and 'label.rfind("beta_", 0)' in text:
        return text
    block = "".join(TOKEN_DIM_LINES) + CONTEXT_TOKEN_DIM_ANCHOR
    return replace_n(text, CONTEXT_TOKEN_DIM_ANCHOR, block, count=1)


def replace_n(text: str, old: str, new: str, *, count: int) -> str:
    if old not in text:
        return text
    return text.replace(old, new, count)


def render_combined_diff(result: PatchBuildResult) -> str:
    parts = [
        render_unified_diff(
            result.graph.original_text,
            result.graph.patched_text,
            relative_path=result.graph.relative_path,
        ),
        render_unified_diff(
            result.context.original_text,
            result.context.patched_text,
            relative_path=result.context.relative_path,
        ),
    ]
    return "".join(parts)


def render_unified_diff(
    original_text: str, patched_text: str, *, relative_path: Path
) -> str:
    if original_text == patched_text:
        return ""
    return "".join(
        unified_diff(
            original_text.splitlines(keepends=True),
            patched_text.splitlines(keepends=True),
            fromfile=f"a/{relative_path.as_posix()}",
            tofile=f"b/{relative_path.as_posix()}",
            n=1,
        )
    )


def summarize_patch_validation(result: PatchBuildResult, diff_text: str) -> dict[str, Any]:
    patched_graph = result.graph.patched_text
    patched_context = result.context.patched_text
    return {
        "changed": result.changed,
        "diff_has_graph_target": f"b/{GRAPH_RELATIVE_PATH.as_posix()}" in diff_text,
        "diff_has_context_target": f"b/{CONTEXT_RELATIVE_PATH.as_posix()}" in diff_text,
        "graph_allows_projection_input": patched_graph.count("linear_attn_qkv_mixed_") >= 2,
        "graph_allows_conv_output": patched_graph.count("conv_output_silu_") >= 2,
        "graph_renames_qkv": 'std::strcmp(name, "linear_attn_qkv_mixed")' in patched_graph,
        "graph_renames_conv_views": 'std::strcmp(name, "q_conv_predelta")' in patched_graph,
        "context_beta_uses_token_dim_2": 'label.rfind("beta_", 0)' in patched_context,
        "context_qkv_views_use_token_dim_2": 'label.rfind("q_conv_", 0)' in patched_context,
        "external_checkout_modified": False,
    }


def next_action(result: PatchBuildResult) -> str:
    if result.status == "patch_ready":
        return "apply_patch_to_temporary_llamacpp_trace_tree_and_capture_layer0_pre_ssm_labels"
    if result.status == "already_wired":
        return "capture_layer0_pre_ssm_labels_with_existing_llamacpp_trace_tree"
    return "refresh_llamacpp_trace_anchor_before_generating_patch"


def _git_rev_parse(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
