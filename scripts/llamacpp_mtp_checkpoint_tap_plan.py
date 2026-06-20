#!/usr/bin/env python3
"""Plan the llama.cpp source tap needed for numeric MTP checkpoint parity.

The existing llama.cpp draft trace is token/probability-only.  This helper reads
local read-only llama.cpp source and emits a stable JSON artifact that says which
Qwen35/Qwen35MoE tensors are already named, which extraction APIs already exist,
and which minimal source patch is needed before capturing numeric layer arrays.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_LLAMA_CPP_ROOT = Path("/home/lhl/llama.cpp/llama.cpp-hip")
DEFAULT_CHECKPOINT = Path(
    "benchmarks/results/mtp-gguf-iter281-layer3-actual-checkpoint-summary.json"
)
DEFAULT_TRACE_INVENTORY = Path(
    "benchmarks/results/mtp-gguf-iter282-llamacpp-trace-inventory.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter283-llamacpp-checkpoint-tap-plan.json"
)
REFERENCE_COMMIT = "6e9007ae61f4e994c27484759caac6ef2aa32b30"

_CB_RE = re.compile(r"cb\s*\([^;]*?\"(?P<name>[^\"]+)\"\s*,\s*(?P<layer>[^);]+)\)", re.S)
_LAYER_INPUT_ASSIGNMENT_RE = re.compile(
    r"res\s*->\s*t_layer_inp\s*\[\s*il\s*\]\s*=\s*inpL\s*;"
)

CHECKPOINT_KEY_TO_LLAMA_CALLBACKS = {
    "hidden_in_f32": ("layer_input",),
    "attn_out_f32": ("attn_output",),
    "residual_f32": ("attn_residual",),
    "post_norm_f32": ("attn_post_norm",),
    "ffn_or_moe_down_f32": ("ffn_out", "ffn_moe_out"),
    "moe_shared_out_f32": ("ffn_shexp_gated",),
    "moe_shared_gate_f32": ("shared_expert_gate_sigmoid",),
    "layer_out_f32": ("l_out",),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llamacpp-root", type=Path, default=DEFAULT_LLAMA_CPP_ROOT)
    parser.add_argument("--checkpoint-summary", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--trace-inventory", type=Path, default=DEFAULT_TRACE_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=283)
    args = parser.parse_args()

    artifact = build_tap_plan_artifact(
        llamacpp_root=args.llamacpp_root,
        checkpoint_summary_path=args.checkpoint_summary,
        trace_inventory_path=args.trace_inventory,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "preferred_arch_source": artifact["preferred_arch_source"],
                "hidden_in_status": artifact["checkpoint_mapping"][0]["status"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_tap_plan_artifact(
    *,
    llamacpp_root: Path,
    checkpoint_summary_path: Path,
    trace_inventory_path: Path | None = None,
    iteration: int = 283,
) -> dict[str, Any]:
    checkpoint = _read_json_if_exists(checkpoint_summary_path) or {}
    trace_inventory = _read_json_if_exists(trace_inventory_path)
    source_inventory = analyze_source_tree(llamacpp_root)
    checkpoint_target = summarize_checkpoint_target(checkpoint)
    preferred_arch = choose_preferred_arch(checkpoint_target, source_inventory)
    arch_inventory = source_inventory["architectures"].get(preferred_arch, {})
    extraction_support = summarize_extraction_support(source_inventory, preferred_arch)
    mapping = build_checkpoint_mapping(checkpoint_target, arch_inventory)
    return {
        "schema": 1,
        "kind": "llamacpp_mtp_numeric_checkpoint_tap_plan",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": "tap_plan_ready",
        "llamacpp_root": str(llamacpp_root),
        "reference_basis": {
            "expected_llamacpp_commit": REFERENCE_COMMIT,
            "source_is_read_only_reference": True,
        },
        "checkpoint_summary_path": str(checkpoint_summary_path),
        "trace_inventory_path": None if trace_inventory_path is None else str(trace_inventory_path),
        "checkpoint_target": checkpoint_target,
        "trace_inventory_summary": summarize_trace_inventory(trace_inventory),
        "preferred_arch_source": preferred_arch,
        "source_inventory": source_inventory,
        "extraction_support": extraction_support,
        "checkpoint_mapping": mapping,
        "patch_plan": build_patch_plan(preferred_arch, checkpoint_target, extraction_support),
        "next_action": next_action(mapping, extraction_support),
        "conclusion": conclusion(preferred_arch, mapping, extraction_support),
    }


def analyze_source_tree(llamacpp_root: Path) -> dict[str, Any]:
    qwen35 = _analyze_arch_source(
        llamacpp_root / "src" / "models" / "qwen35.cpp",
        arch="qwen35",
    )
    qwen35moe = _analyze_arch_source(
        llamacpp_root / "src" / "models" / "qwen35moe.cpp",
        arch="qwen35moe",
    )
    return {
        "architectures": {"qwen35": qwen35, "qwen35moe": qwen35moe},
        "support_files": {
            "llama_ext_header": _summarize_file_symbols(
                llamacpp_root / "src" / "llama-ext.h",
                (
                    "llama_set_embeddings_layer_inp",
                    "llama_get_embeddings_layer_inp",
                    "llama_set_embeddings_nextn",
                    "llama_get_embeddings_nextn",
                ),
            ),
            "llama_context": _summarize_file_symbols(
                llamacpp_root / "src" / "llama-context.cpp",
                (
                    "set_embeddings_layer_inp",
                    "get_embeddings_layer_inp",
                    "extract_layer_inputs",
                    "ggml_backend_tensor_get_async",
                    "graph_get_cb",
                ),
            ),
            "llama_graph": _summarize_file_symbols(
                llamacpp_root / "src" / "llama-graph.cpp",
                ("t_layer_inp", "ggml_set_output(t_layer_inp", "t_h_nextn"),
            ),
            "common_speculative": _summarize_file_symbols(
                llamacpp_root / "common" / "speculative.cpp",
                ("llama_set_embeddings_layer_inp", "llama_get_embeddings_layer_inp"),
            ),
        },
    }


def _analyze_arch_source(path: Path, *, arch: str) -> dict[str, Any]:
    text = path.read_text(errors="replace") if path.exists() else ""
    main_text, mtp_text = _split_main_and_mtp_sections(text)
    main_callbacks = _extract_callbacks(main_text)
    mtp_callbacks = _extract_callbacks(mtp_text)
    return {
        "arch": arch,
        "path": str(path),
        "exists": path.exists(),
        "has_mtp_graph": "LLM_GRAPH_TYPE_DECODER_MTP" in text and "graph_mtp" in text,
        "has_main_layer_loop": "for (int il = 0; il < n_layer; ++il)" in text,
        "has_layer_input_assignment": bool(_LAYER_INPUT_ASSIGNMENT_RE.search(text)),
        "layer_input_assignment_line": _find_line_for_regex(text, _LAYER_INPUT_ASSIGNMENT_RE),
        "main_callback_count": len(main_callbacks),
        "mtp_callback_count": len(mtp_callbacks),
        "main_callbacks": main_callbacks,
        "mtp_callbacks": mtp_callbacks,
        "anchor_lines": {
            "main_layer_loop": _find_line(text, "for (int il = 0; il < n_layer; ++il)"),
            "attn_output_cb": _find_line(text, 'cb(cur, "attn_output", il)'),
            "ffn_moe_out_cb": _find_line(text, 'cb(moe_out, "ffn_moe_out", il)'),
            "ffn_shexp_gated_cb": _find_line(text, 'cb(ffn_shexp, "ffn_shexp_gated", il)'),
            "post_moe_cb": _find_line(text, 'cb(cur, "post_moe", il)'),
            "l_out_cb": _find_line(text, 'cb(cur, "l_out", il)'),
            "h_nextn_assignment": _find_line(text, "res->t_h_nextn"),
            "mtp_graph": _find_line(text, "graph_mtp"),
        },
    }


def _split_main_and_mtp_sections(text: str) -> tuple[str, str]:
    marker = "// LLM_GRAPH_TYPE_DECODER_MTP"
    if marker not in text:
        return text, ""
    main, mtp = text.split(marker, 1)
    return main, marker + mtp


def _extract_callbacks(text: str) -> list[dict[str, Any]]:
    callbacks: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for match in _CB_RE.finditer(text):
        name = match.group("name")
        layer_expr = " ".join(match.group("layer").strip().split())
        key = (name, layer_expr)
        if key in seen:
            continue
        seen.add(key)
        callbacks.append(
            {
                "name": name,
                "layer_expr": layer_expr,
                "line": _line_from_index(text, match.start()),
            }
        )
    return callbacks


def _summarize_file_symbols(path: Path, symbols: tuple[str, ...]) -> dict[str, Any]:
    text = path.read_text(errors="replace") if path.exists() else ""
    return {
        "path": str(path),
        "exists": path.exists(),
        "symbols": {
            symbol: {"present": symbol in text, "line": _find_line(text, symbol)}
            for symbol in symbols
        },
    }


def summarize_checkpoint_target(checkpoint: dict[str, Any]) -> dict[str, Any]:
    capture = checkpoint.get("capture") or {}
    arrays = checkpoint.get("arrays") or {}
    array_hashes = {
        key: value.get("sha256")
        for key, value in arrays.items()
        if isinstance(value, dict) and "sha256" in value
    }
    return {
        "model": capture.get("model"),
        "position": capture.get("position"),
        "token_id": capture.get("token_id"),
        "layer_id": capture.get("layer_id"),
        "layer_type": capture.get("layer_type"),
        "run_preceding_layers": capture.get("run_preceding_layers"),
        "preceding_layer_count": capture.get("preceding_layer_count"),
        "hidden_size": capture.get("hidden_size"),
        "array_keys": sorted(arrays),
        "array_hashes": array_hashes,
        "has_moe_arrays": any(key.startswith("moe_") for key in arrays),
    }


def choose_preferred_arch(
    checkpoint_target: dict[str, Any], source_inventory: dict[str, Any]
) -> str:
    model = str(checkpoint_target.get("model") or "").lower()
    wants_moe = checkpoint_target.get("has_moe_arrays") or "a3b" in model or "moe" in model
    arches = source_inventory.get("architectures") or {}
    if wants_moe and arches.get("qwen35moe", {}).get("exists"):
        return "qwen35moe"
    if arches.get("qwen35", {}).get("exists"):
        return "qwen35"
    return "qwen35moe" if wants_moe else "qwen35"


def summarize_extraction_support(
    source_inventory: dict[str, Any], preferred_arch: str
) -> dict[str, Any]:
    support_files = source_inventory.get("support_files") or {}
    arch = (source_inventory.get("architectures") or {}).get(preferred_arch) or {}
    ext = support_files.get("llama_ext_header", {}).get("symbols", {})
    context = support_files.get("llama_context", {}).get("symbols", {})
    graph = support_files.get("llama_graph", {}).get("symbols", {})
    common = support_files.get("common_speculative", {}).get("symbols", {})
    layer_api_present = all(
        _symbol_present(symbols, name)
        for symbols, name in (
            (ext, "llama_set_embeddings_layer_inp"),
            (ext, "llama_get_embeddings_layer_inp"),
            (context, "set_embeddings_layer_inp"),
            (context, "extract_layer_inputs"),
            (graph, "t_layer_inp"),
        )
    )
    return {
        "existing_layer_input_api_present": layer_api_present,
        "common_speculative_uses_layer_input_api": all(
            _symbol_present(common, name)
            for name in ("llama_set_embeddings_layer_inp", "llama_get_embeddings_layer_inp")
        ),
        "preferred_arch_wires_layer_input": bool(arch.get("has_layer_input_assignment")),
        "needs_preferred_arch_layer_input_patch": layer_api_present
        and not bool(arch.get("has_layer_input_assignment")),
        "graph_callback_names_tensors_only": _symbol_present(context, "graph_get_cb"),
        "backend_tensor_copy_available": _symbol_present(
            context, "ggml_backend_tensor_get_async"
        ),
    }


def build_checkpoint_mapping(
    checkpoint_target: dict[str, Any], arch_inventory: dict[str, Any]
) -> list[dict[str, Any]]:
    layer_id = checkpoint_target.get("layer_id")
    callbacks = {item["name"]: item for item in arch_inventory.get("main_callbacks", [])}
    has_layer_input = bool(arch_inventory.get("has_layer_input_assignment"))
    mappings: list[dict[str, Any]] = []
    for key in _ordered_checkpoint_keys(checkpoint_target.get("array_keys", [])):
        candidates = CHECKPOINT_KEY_TO_LLAMA_CALLBACKS.get(key)
        if not candidates:
            continue
        if key == "hidden_in_f32":
            status = (
                "existing_layer_input_api_ready"
                if has_layer_input
                else "source_patch_needed_for_layer_input"
            )
            mappings.append(
                {
                    "hipengine_key": key,
                    "llamacpp_tensor": f"layer_inp-{layer_id}",
                    "source_anchor": "res->t_layer_inp[il] = inpL before layer body",
                    "status": status,
                    "sha256": checkpoint_target.get("array_hashes", {}).get(key),
                }
            )
            continue
        callback = next((name for name in candidates if name in callbacks), None)
        if callback is None:
            status = "callback_name_missing"
            tensor = None
            line = None
        else:
            status = "named_callback_present_but_output_copy_patch_needed"
            tensor = f"{callback}-{layer_id}"
            line = callbacks[callback]["line"]
        mappings.append(
            {
                "hipengine_key": key,
                "llamacpp_tensor": tensor,
                "candidate_callbacks": list(candidates),
                "callback_line": line,
                "status": status,
                "sha256": checkpoint_target.get("array_hashes", {}).get(key),
            }
        )
    return mappings


def _ordered_checkpoint_keys(keys: list[str]) -> list[str]:
    known = [key for key in CHECKPOINT_KEY_TO_LLAMA_CALLBACKS if key in keys]
    unknown = [key for key in keys if key not in CHECKPOINT_KEY_TO_LLAMA_CALLBACKS]
    return known + unknown


def build_patch_plan(
    preferred_arch: str,
    checkpoint_target: dict[str, Any],
    extraction_support: dict[str, Any],
) -> list[dict[str, Any]]:
    layer_id = checkpoint_target.get("layer_id")
    position = checkpoint_target.get("position")
    arch_file = f"src/models/{preferred_arch}.cpp"
    return [
        {
            "step": 1,
            "scope": arch_file,
            "action": "wire_existing_layer_input_output",
            "details": (
                "Inside the main `for (int il = 0; il < n_layer; ++il)` loop, "
                "insert `res->t_layer_inp[il] = inpL;` before `ggml_tensor * "
                "inpSA = inpL;` so the existing llama-ext layer-input API can "
                f"return hipEngine hidden_in_f32 for layer {layer_id}."
            ),
            "needed": extraction_support["needs_preferred_arch_layer_input_patch"],
        },
        {
            "step": 2,
            "scope": "src/llama-graph.* and src/llama-context.cpp",
            "action": "add_temporary_named_checkpoint_outputs",
            "details": (
                "Add a debug-only selected tensor output/copy path for named graph "
                "callbacks such as attn_output, attn_residual, attn_post_norm, "
                "ffn_moe_out, ffn_shexp_gated, shared_expert_gate_sigmoid, and "
                "l_out. Use ggml_set_output plus ggml_backend_tensor_get_async, "
                "mirroring extract_layer_inputs, rather than relying on log text."
            ),
            "needed": True,
        },
        {
            "step": 3,
            "scope": "trace harness / artifact parser",
            "action": "capture_prompt_layer_checkpoint",
            "details": (
                f"Run the greeting request, capture layer {layer_id} row at prompt "
                f"position {position}, and emit float32 arrays plus sha256 hashes "
                "with the same keys as the hipEngine compact checkpoint summary."
            ),
            "needed": True,
        },
    ]


def summarize_trace_inventory(trace_inventory: dict[str, Any] | None) -> dict[str, Any] | None:
    if trace_inventory is None:
        return None
    coverage = trace_inventory.get("trace_coverage") or {}
    alignment = trace_inventory.get("alignment") or {}
    return {
        "status": trace_inventory.get("status"),
        "prompt_tokens_match": alignment.get("prompt_tokens_match"),
        "has_numeric_layer_checkpoints": coverage.get("has_numeric_layer_checkpoints"),
        "supports_token_draft_alignment": coverage.get("supports_token_draft_alignment"),
        "next_action": trace_inventory.get("next_action"),
    }


def next_action(mapping: list[dict[str, Any]], extraction_support: dict[str, Any]) -> str:
    if any(item["status"] == "source_patch_needed_for_layer_input" for item in mapping):
        return "prepare_llamacpp_qwen35moe_layer_input_patch_then_capture_hidden_in"
    if any(
        item["status"] == "named_callback_present_but_output_copy_patch_needed"
        for item in mapping
    ):
        return "add_llamacpp_named_tensor_output_tap_then_capture_numeric_checkpoint"
    if not extraction_support["backend_tensor_copy_available"]:
        return "locate_backend_tensor_copy_api_before_checkpoint_capture"
    return "capture_llamacpp_numeric_checkpoint_and_compare_hashes"


def conclusion(
    preferred_arch: str, mapping: list[dict[str, Any]], extraction_support: dict[str, Any]
) -> str:
    hidden_status = next(
        (item["status"] for item in mapping if item["hipengine_key"] == "hidden_in_f32"),
        "missing",
    )
    named_count = sum(
        item["status"] == "named_callback_present_but_output_copy_patch_needed"
        for item in mapping
    )
    if hidden_status == "source_patch_needed_for_layer_input":
        return (
            f"llama.cpp already has layer-input extraction APIs, but {preferred_arch} "
            "does not wire res->t_layer_inp in its main graph. Patch that first; "
            f"then add output copies for {named_count} named callbacks to align the "
            "hipEngine layer checkpoint numerically."
        )
    if extraction_support["existing_layer_input_api_present"]:
        return (
            "The hidden-in tap can use the existing layer-input API; remaining "
            f"arrays need debug output copies for {named_count} named callbacks."
        )
    return "The needed callback names were inventoried, but extraction APIs need source work."


def _symbol_present(symbols: dict[str, Any], name: str) -> bool:
    return bool((symbols.get(name) or {}).get("present"))


def _read_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _find_line(text: str, needle: str) -> int | None:
    index = text.find(needle)
    if index < 0:
        return None
    return _line_from_index(text, index)


def _find_line_for_regex(text: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.search(text)
    if match is None:
        return None
    return _line_from_index(text, match.start())


def _line_from_index(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


if __name__ == "__main__":
    main()
