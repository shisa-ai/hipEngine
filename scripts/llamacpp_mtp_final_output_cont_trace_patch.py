#!/usr/bin/env python3
"""Generate a local llama.cpp patch for a contiguous final_output trace tap."""

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
    "benchmarks/results/2026-07-02-llamacpp-final-output-cont-trace.patch"
)
DEFAULT_JSON_OUTPUT = Path(
    "benchmarks/results/2026-07-02-llamacpp-final-output-cont-trace.json"
)
GRAPH_RELATIVE_PATH = Path("src/llama-graph.cpp")
QWEN35MOE_RELATIVE_PATH = Path("src/models/qwen35moe.cpp")

GRAPH_RENAME_ANCHOR = '             std::strcmp(name, "final_output") == 0 ||\n'
GRAPH_RENAME_INSERT = '             std::strcmp(name, "final_output_cont") == 0 ||\n'
GRAPH_TRACE_WANTS_ANCHOR = '             std::strncmp(name, "final_output_", 13) == 0 ||\n'
GRAPH_TRACE_WANTS_INSERT = '             std::strncmp(name, "final_output_cont_", 18) == 0 ||\n'
GRAPH_ADD_TENSOR_ANCHOR = '         std::strncmp(name, "final_output_", 13) == 0 ||\n'
GRAPH_ADD_TENSOR_INSERT = '         std::strncmp(name, "final_output_cont_", 18) == 0 ||\n'

FINAL_OUTPUT_ANCHOR = (
    "    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, "
    "head_v_dim * num_v_heads, n_seq_tokens, n_seqs);\n"
    '    cb(final_output, "final_output", il);\n'
)
FINAL_OUTPUT_INSERT = (
    "\n"
    "    ggml_tensor * final_output_cont = ggml_cont_3d(ctx0, final_output,\n"
    "            head_v_dim * num_v_heads, n_seq_tokens, n_seqs);\n"
    '    cb(final_output_cont, "final_output_cont", il);\n'
)
SSM_OUT_FINAL_OUTPUT_ANCHOR = (
    "    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);\n"
)
SSM_OUT_FINAL_OUTPUT_CONT = (
    "    cur = build_lora_mm(model.layers[il].ssm_out, final_output_cont, model.layers[il].ssm_out_s);\n"
)


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
    qwen35moe: SourcePatch
    reason: str | None = None

    @property
    def changed(self) -> bool:
        return self.graph.changed or self.qwen35moe.changed


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
    qwen35moe_path = root / QWEN35MOE_RELATIVE_PATH
    graph_text = graph_path.read_text() if graph_path.exists() else ""
    qwen35moe_text = qwen35moe_path.read_text() if qwen35moe_path.exists() else ""
    result = build_final_output_cont_trace_patch_text(
        graph_text=graph_text,
        qwen35moe_text=qwen35moe_text,
    )
    diff_text = render_combined_diff(result)
    patch_output.parent.mkdir(parents=True, exist_ok=True)
    patch_output.write_text(diff_text)
    patch_sha256 = hashlib.sha256(diff_text.encode()).hexdigest()
    return {
        "schema": 1,
        "kind": "llamacpp_final_output_cont_trace_patch",
        "date": "2026-07-02",
        "iteration": int(iteration),
        "status": result.status,
        "reason": result.reason,
        "llamacpp_root": str(root),
        "source_exists": {
            str(GRAPH_RELATIVE_PATH): graph_path.exists(),
            str(QWEN35MOE_RELATIVE_PATH): qwen35moe_path.exists(),
        },
        "reference_basis": {
            "observed_llamacpp_commit": _git_rev_parse(root),
            "source_is_read_only_reference": True,
            "external_checkout_modified": False,
        },
        "targets": [str(GRAPH_RELATIVE_PATH), str(QWEN35MOE_RELATIVE_PATH)],
        "patch_output": str(patch_output),
        "patch_sha256": patch_sha256,
        "patch_bytes": len(diff_text.encode()),
        "validation": summarize_patch_validation(result, diff_text),
        "trace_labels_enabled": ["final_output_cont_0"],
        "next_action": next_action(result),
    }


def build_final_output_cont_trace_patch_text(
    *, graph_text: str, qwen35moe_text: str
) -> PatchBuildResult:
    graph_ready = audit_graph_trace_support(graph_text)["ready"]
    qwen35moe_ready = audit_qwen35moe_trace_support(qwen35moe_text)["ready"]
    if graph_ready and qwen35moe_ready:
        return PatchBuildResult(
            status="already_wired",
            graph=SourcePatch(GRAPH_RELATIVE_PATH, graph_text, graph_text),
            qwen35moe=SourcePatch(QWEN35MOE_RELATIVE_PATH, qwen35moe_text, qwen35moe_text),
            reason="llama.cpp already exposes final_output_cont trace tap",
        )

    graph_patched = patch_graph_trace_support(graph_text)
    qwen35moe_patched = patch_qwen35moe_trace_support(qwen35moe_text)
    if not audit_graph_trace_support(graph_patched)["ready"] and not graph_ready:
        return PatchBuildResult(
            status="graph_anchor_missing",
            graph=SourcePatch(GRAPH_RELATIVE_PATH, graph_text, graph_patched),
            qwen35moe=SourcePatch(
                QWEN35MOE_RELATIVE_PATH, qwen35moe_text, qwen35moe_patched
            ),
            reason="expected llama-graph.cpp final_output rename anchor was not found",
        )
    if not audit_qwen35moe_trace_support(qwen35moe_patched)["ready"] and not qwen35moe_ready:
        return PatchBuildResult(
            status="qwen35moe_anchor_missing",
            graph=SourcePatch(GRAPH_RELATIVE_PATH, graph_text, graph_patched),
            qwen35moe=SourcePatch(QWEN35MOE_RELATIVE_PATH, qwen35moe_text, qwen35moe_text),
            reason="expected qwen35moe final_output anchor was not found",
        )
    return PatchBuildResult(
        status="patch_ready",
        graph=SourcePatch(GRAPH_RELATIVE_PATH, graph_text, graph_patched),
        qwen35moe=SourcePatch(QWEN35MOE_RELATIVE_PATH, qwen35moe_text, qwen35moe_patched),
    )


def audit_graph_trace_support(text: str) -> dict[str, Any]:
    has_trace_prefix = has_exact_line(text, GRAPH_TRACE_WANTS_INSERT)
    has_add_tensor_prefix = has_exact_line(text, GRAPH_ADD_TENSOR_INSERT)
    return {
        "ready": (
            'std::strcmp(name, "final_output_cont")' in text
            and has_trace_prefix
            and has_add_tensor_prefix
        ),
        "final_output_cont_rename_present": 'std::strcmp(name, "final_output_cont")' in text,
        "final_output_cont_trace_wants_prefix_present": has_trace_prefix,
        "final_output_cont_add_tensor_prefix_present": has_add_tensor_prefix,
    }


def audit_qwen35moe_trace_support(text: str) -> dict[str, Any]:
    uses_cont_input = has_exact_line(text, SSM_OUT_FINAL_OUTPUT_CONT)
    return {
        "ready": (
            "final_output_cont" in text
            and 'cb(final_output_cont, "final_output_cont", il)' in text
            and uses_cont_input
        ),
        "cont_tensor_present": "final_output_cont" in text,
        "ssm_out_uses_final_output_cont": uses_cont_input,
    }


def patch_graph_trace_support(text: str) -> str:
    patched = text
    if 'std::strcmp(name, "final_output_cont")' not in patched:
        patched = replace_line_n(
            patched,
            GRAPH_RENAME_ANCHOR,
            GRAPH_RENAME_INSERT + GRAPH_RENAME_ANCHOR,
            count=1,
        )
    if not has_exact_line(patched, GRAPH_TRACE_WANTS_INSERT):
        patched = replace_line_n(
            patched,
            GRAPH_TRACE_WANTS_ANCHOR,
            GRAPH_TRACE_WANTS_INSERT + GRAPH_TRACE_WANTS_ANCHOR,
            count=1,
        )
    if not has_exact_line(patched, GRAPH_ADD_TENSOR_INSERT):
        patched = replace_line_n(
            patched,
            GRAPH_ADD_TENSOR_ANCHOR,
            GRAPH_ADD_TENSOR_INSERT + GRAPH_ADD_TENSOR_ANCHOR,
            count=1,
        )
    return patched


def patch_qwen35moe_trace_support(text: str) -> str:
    patched = text
    if "final_output_cont" not in patched:
        patched = replace_n(
            patched, FINAL_OUTPUT_ANCHOR, FINAL_OUTPUT_ANCHOR + FINAL_OUTPUT_INSERT, count=1
        )
    if not has_exact_line(patched, SSM_OUT_FINAL_OUTPUT_CONT):
        patched = replace_line_n(
            patched,
            SSM_OUT_FINAL_OUTPUT_ANCHOR,
            SSM_OUT_FINAL_OUTPUT_CONT,
            count=1,
        )
    return patched


def replace_n(text: str, old: str, new: str, *, count: int) -> str:
    if old not in text:
        return text
    return text.replace(old, new, count)


def replace_line_n(text: str, old: str, new: str, *, count: int) -> str:
    lines = text.splitlines(keepends=True)
    replaced = 0
    output: list[str] = []
    for line in lines:
        if line == old and replaced < count:
            output.append(new)
            replaced += 1
        else:
            output.append(line)
    return "".join(output)


def has_exact_line(text: str, line: str) -> bool:
    return line in text.splitlines(keepends=True)


def render_combined_diff(result: PatchBuildResult) -> str:
    parts = [
        render_unified_diff(
            result.graph.original_text,
            result.graph.patched_text,
            relative_path=result.graph.relative_path,
        ),
        render_unified_diff(
            result.qwen35moe.original_text,
            result.qwen35moe.patched_text,
            relative_path=result.qwen35moe.relative_path,
        ),
    ]
    return normalize_blank_context_lines("".join(parts))


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


def normalize_blank_context_lines(diff_text: str) -> str:
    return "".join("-\n+\n" if line == " \n" else line for line in diff_text.splitlines(True))


def summarize_patch_validation(result: PatchBuildResult, diff_text: str) -> dict[str, Any]:
    graph = result.graph.patched_text
    qwen35moe = result.qwen35moe.patched_text
    return {
        "changed": result.changed,
        "diff_has_graph_target": f"b/{GRAPH_RELATIVE_PATH.as_posix()}" in diff_text,
        "diff_has_qwen35moe_target": f"b/{QWEN35MOE_RELATIVE_PATH.as_posix()}" in diff_text,
        "graph_renames_final_output_cont": 'std::strcmp(name, "final_output_cont")' in graph,
        "graph_wants_final_output_cont_prefix": has_exact_line(
            graph, GRAPH_TRACE_WANTS_INSERT
        ),
        "graph_adds_final_output_cont_prefix": has_exact_line(
            graph, GRAPH_ADD_TENSOR_INSERT
        ),
        "qwen35moe_emits_final_output_cont": 'cb(final_output_cont, "final_output_cont", il)' in qwen35moe,
        "qwen35moe_uses_cont_3d": "ggml_cont_3d(ctx0, final_output" in qwen35moe,
        "qwen35moe_ssm_out_uses_final_output_cont": has_exact_line(
            qwen35moe, SSM_OUT_FINAL_OUTPUT_CONT
        ),
        "external_checkout_modified": False,
    }


def next_action(result: PatchBuildResult) -> str:
    if result.status == "patch_ready":
        return "apply_patch_to_temporary_llamacpp_trace_tree_and_capture_final_output_cont"
    if result.status == "already_wired":
        return "capture_final_output_cont_with_existing_llamacpp_trace_tree"
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
