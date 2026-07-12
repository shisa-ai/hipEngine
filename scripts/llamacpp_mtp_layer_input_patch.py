#!/usr/bin/env python3
"""Generate the minimal llama.cpp Qwen35MoE layer-input tap patch.

This does not modify the external llama.cpp checkout.  It emits a unified diff
that wires `res->t_layer_inp[il] = inpL;` in Qwen35MoE's main layer loop so the
existing llama.cpp layer-input extraction API can capture hidden-in checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any

DEFAULT_LLAMA_CPP_ROOT = Path("/home/lhl/llama.cpp/llama.cpp-hip")
DEFAULT_PATCH_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter284-llamacpp-qwen35moe-layer-input.patch"
)
DEFAULT_JSON_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter284-llamacpp-qwen35moe-layer-input-patch.json"
)
TARGET_RELATIVE_PATH = Path("src/models/qwen35moe.cpp")
REFERENCE_COMMIT = "6e9007ae61f4e994c27484759caac6ef2aa32b30"
LAYER_INPUT_ASSIGNMENT = "        res->t_layer_inp[il] = inpL;"
ANCHOR_COMMENT = (
    "    // MTP/NextN layers are loaded as extra decoder blocks but not executed "
    "in the main pass.\n"
)
ANCHOR_BLOCK = (
    ANCHOR_COMMENT
    + "    for (int il = 0; il < n_layer; ++il) {\n"
    + "        ggml_tensor * inpSA = inpL;\n"
)
PATCHED_BLOCK = (
    ANCHOR_COMMENT
    + "    for (int il = 0; il < n_layer; ++il) {\n"
    + "        res->t_layer_inp[il] = inpL;\n"
    + "        ggml_tensor * inpSA = inpL;\n"
)


@dataclass(frozen=True)
class PatchBuildResult:
    status: str
    original_text: str
    patched_text: str
    insertion_line: int | None
    reason: str | None = None

    @property
    def changed(self) -> bool:
        return self.original_text != self.patched_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llamacpp-root", type=Path, default=DEFAULT_LLAMA_CPP_ROOT)
    parser.add_argument("--patch-output", type=Path, default=DEFAULT_PATCH_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--iteration", type=int, default=284)
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
                "target": artifact["target_relative_path"],
                "patch_output": artifact["patch_output"],
                "patch_sha256": artifact["patch_sha256"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_patch_artifact(
    *, llamacpp_root: Path, patch_output: Path, iteration: int = 284
) -> dict[str, Any]:
    source_path = llamacpp_root / TARGET_RELATIVE_PATH
    source_text = source_path.read_text() if source_path.exists() else ""
    result = build_layer_input_patch_text(source_text)
    diff_text = render_unified_diff(
        result.original_text,
        result.patched_text,
        relative_path=TARGET_RELATIVE_PATH,
    )
    patch_output.parent.mkdir(parents=True, exist_ok=True)
    patch_output.write_text(diff_text)
    patch_sha256 = hashlib.sha256(diff_text.encode()).hexdigest()
    return {
        "schema": 1,
        "kind": "llamacpp_qwen35moe_layer_input_patch",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": result.status,
        "reason": result.reason,
        "llamacpp_root": str(llamacpp_root),
        "target_relative_path": str(TARGET_RELATIVE_PATH),
        "source_exists": source_path.exists(),
        "reference_basis": {
            "expected_llamacpp_commit": REFERENCE_COMMIT,
            "source_is_read_only_reference": True,
            "external_checkout_modified": False,
        },
        "patch_output": str(patch_output),
        "patch_sha256": patch_sha256,
        "patch_bytes": len(diff_text.encode()),
        "insertion_line": result.insertion_line,
        "validation": summarize_patch_validation(result, diff_text),
        "next_action": next_action(result),
    }


def build_layer_input_patch_text(source_text: str) -> PatchBuildResult:
    if LAYER_INPUT_ASSIGNMENT in source_text:
        return PatchBuildResult(
            status="already_wired",
            original_text=source_text,
            patched_text=source_text,
            insertion_line=_find_assignment_line(source_text),
            reason="target source already contains res->t_layer_inp[il] = inpL;",
        )
    if ANCHOR_BLOCK not in source_text:
        return PatchBuildResult(
            status="anchor_missing",
            original_text=source_text,
            patched_text=source_text,
            insertion_line=None,
            reason="expected Qwen35MoE main layer-loop anchor was not found",
        )
    patched = source_text.replace(ANCHOR_BLOCK, PATCHED_BLOCK, 1)
    return PatchBuildResult(
        status="patch_ready",
        original_text=source_text,
        patched_text=patched,
        insertion_line=_find_line(source_text, "        ggml_tensor * inpSA = inpL;"),
    )


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
            n=5,
        )
    )


def summarize_patch_validation(result: PatchBuildResult, diff_text: str) -> dict[str, Any]:
    inserted_count = result.patched_text.count(LAYER_INPUT_ASSIGNMENT)
    original_count = result.original_text.count(LAYER_INPUT_ASSIGNMENT)
    return {
        "changed": result.changed,
        "single_assignment_added": inserted_count == original_count + 1,
        "anchor_present_before_patch": ANCHOR_BLOCK in result.original_text,
        "anchor_absent_after_patch": ANCHOR_BLOCK not in result.patched_text,
        "patched_block_present": PATCHED_BLOCK in result.patched_text,
        "diff_has_expected_target": f"b/{TARGET_RELATIVE_PATH.as_posix()}" in diff_text,
        "diff_has_expected_assignment": LAYER_INPUT_ASSIGNMENT in diff_text,
    }


def next_action(result: PatchBuildResult) -> str:
    if result.status == "patch_ready":
        return "apply_patch_to_temporary_llamacpp_checkout_and_capture_hidden_in"
    if result.status == "already_wired":
        return "capture_hidden_in_with_existing_llamacpp_layer_input_api"
    return "refresh_qwen35moe_anchor_before_generating_patch"


def _find_assignment_line(text: str) -> int | None:
    return _find_line(text, LAYER_INPUT_ASSIGNMENT)


def _find_line(text: str, needle: str) -> int | None:
    index = text.find(needle)
    if index < 0:
        return None
    return text.count("\n", 0, index) + 1


if __name__ == "__main__":
    main()
