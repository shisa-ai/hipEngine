#!/usr/bin/env python3
"""Audit hipEngine GGUF layer-capture graph/materialization divergence candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader  # noqa: E402
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map  # noqa: E402
from hipengine.loading.qwen35_gguf_materialize import (  # noqa: E402
    audit_qwen35_gguf_precision_contractions,
    plan_qwen35_gguf_materialization,
)

DEFAULT_RUNNER = Path("hipengine/runtime/qwen35_gguf_runner.py")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_CAPTURE = Path(
    "benchmarks/results/mtp-gguf-iter280-layer3-full-attn-actual-routing-full-arrays.json"
)
DEFAULT_TAP_COMPARE = Path(
    "benchmarks/results/mtp-gguf-iter296-llamacpp-tap-placement-compare.json"
)
DEFAULT_LAYER_SWEEP = Path(
    "benchmarks/results/mtp-gguf-iter295-llamacpp-hidden-in-layer-sweep.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter297-hipengine-capture-path-audit.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--tap-compare", type=Path, default=DEFAULT_TAP_COMPARE)
    parser.add_argument("--layer-sweep", type=Path, default=DEFAULT_LAYER_SWEEP)
    parser.add_argument("--layers", default="0-3")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=297)
    args = parser.parse_args()

    artifact = build_capture_path_audit(
        runner_path=args.runner,
        model_path=args.model,
        capture_path=args.capture,
        tap_compare_path=args.tap_compare,
        layer_sweep_path=args.layer_sweep,
        layers=parse_layers(args.layers),
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "capture_embedding_setup": artifact["source_audit"]["facts"][
                    "capture_sets_embedding"
                ],
                "hidden_tap_dtype": artifact["source_audit"]["facts"]["hidden_tap_dtype"],
                "precision_contractions": artifact["precision_audit"]["count"],
                "conclusion": artifact["conclusion"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_capture_path_audit(
    *,
    runner_path: Path,
    model_path: Path,
    capture_path: Path,
    tap_compare_path: Path,
    layer_sweep_path: Path,
    layers: Iterable[int],
    iteration: int = 297,
) -> dict[str, Any]:
    runner_text = runner_path.read_text()
    capture = read_json(capture_path)
    tap_compare = read_json(tap_compare_path)
    layer_sweep = read_json(layer_sweep_path)
    source_audit = audit_runner_source(runner_text)
    precision = audit_precision_contractions_for_layers(model_path, layers=tuple(layers))
    evidence = summarize_mismatch_evidence(
        capture=capture,
        tap_compare=tap_compare,
        layer_sweep=layer_sweep,
    )
    conclusion = conclude(source_audit=source_audit, precision=precision, evidence=evidence)
    return {
        "schema": 1,
        "kind": "hipengine_gguf_capture_path_audit",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": "audited",
        "runner_path": str(runner_path),
        "model_path": str(model_path),
        "capture_path": str(capture_path),
        "tap_compare_path": str(tap_compare_path),
        "layer_sweep_path": str(layer_sweep_path),
        "source_audit": source_audit,
        "precision_audit": precision,
        "mismatch_evidence": evidence,
        "conclusion": conclusion,
        "external_checkout_modified": False,
        "next_action": next_action(conclusion),
    }


def audit_runner_source(text: str) -> dict[str, Any]:
    capture_body = extract_function_body(text, "capture_attention_layer")
    set_token_body = extract_function_body(text, "_set_token_id_device")
    embedding_body = extract_function_body(text, "_set_token_embedding_from_ptr")
    current_body = extract_function_body(text, "_run_current_hidden_to_final_hidden")
    facts = {
        "capture_calls_set_token_id_device": "self._set_token_id_device" in capture_body,
        "set_token_id_launches_embedding": "self._set_token_embedding_from_ptr" in set_token_body,
        "embedding_launches_gguf_embedding": "launch_gguf_embedding" in embedding_body,
        "normal_decode_uses_hidden_a_after_embedding": "src = self._hidden_a" in current_body,
        "capture_uses_hidden_a_after_embedding": "src = self._hidden_a" in capture_body,
        "capture_replays_preceding_layers": "for prev_layer_id" in capture_body,
        "capture_full_attention_passes_position": "position=position" in capture_body,
        "capture_hidden_tap_copies_target_src_ptr": "target_src_ptr" in capture_body
        and "hidden_in_f32=_copy_bf16_ptr_to_host_f32" in capture_body,
        "hidden_tap_dtype": "bf16_to_host_f32",
    }
    facts["capture_sets_embedding"] = all(
        (
            facts["capture_calls_set_token_id_device"],
            facts["set_token_id_launches_embedding"],
            facts["embedding_launches_gguf_embedding"],
        )
    )
    return {
        "facts": facts,
        "anchors": {
            "capture_attention_layer": find_line(text, "def capture_attention_layer"),
            "capture_set_token_id": find_line(text, "self._set_token_id_device(int(token_id)"),
            "capture_src_hidden_a": find_line(text, "src = self._hidden_a"),
            "preceding_layer_loop": find_line(text, "for prev_layer_id, prev_layer_type"),
            "hidden_in_copy": find_line(text, "hidden_in_f32=_copy_bf16_ptr_to_host_f32"),
            "set_token_embedding": find_line(text, "def _set_token_embedding_from_ptr"),
            "launch_gguf_embedding": find_line(text, "launch_gguf_embedding("),
        },
    }


def audit_precision_contractions_for_layers(
    model_path: Path, *, layers: tuple[int, ...]
) -> dict[str, Any]:
    if not model_path.exists():
        return {"available": False, "reason": "model_missing", "count": 0, "records": []}
    reader = GGUFReader(model_path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    plan = plan_qwen35_gguf_materialization(model_map)
    records = []
    wanted_prefixes = {f"layers.{layer}." for layer in layers}
    for item in audit_qwen35_gguf_precision_contractions(plan):
        if not any(item.slot_path.startswith(prefix) for prefix in wanted_prefixes):
            continue
        records.append(
            {
                "slot_path": item.slot_path,
                "source_name": item.source_name,
                "source_type": item.source_type,
                "resident_layout": item.resident_layout,
                "resident_quant_key": item.resident_quant_key,
                "llama_cpp_contract": item.llama_cpp_contract,
                "hipengine_contract": item.hipengine_contract,
            }
        )
    return {
        "available": True,
        "layers": list(layers),
        "count": len(records),
        "records": records,
    }


def summarize_mismatch_evidence(
    *, capture: dict[str, Any], tap_compare: dict[str, Any], layer_sweep: dict[str, Any]
) -> dict[str, Any]:
    best_tap = (tap_compare.get("ranking") or {}).get("best_same_width") or {}
    best_layer = (layer_sweep.get("ranking") or {}).get("best_selected") or {}
    capture_summary = capture.get("capture_summary") or {}
    return {
        "capture_layer_id": capture.get("layer_id"),
        "capture_position": capture.get("position"),
        "capture_run_preceding_layers": capture.get("run_preceding_layers"),
        "capture_preceding_layer_count": capture_summary.get("preceding_layer_count"),
        "tap_compare_conclusion": tap_compare.get("conclusion"),
        "tap_compare_best_key": best_tap.get("key"),
        "tap_compare_best_rmse": best_tap.get("rmse"),
        "layer_sweep_conclusion": layer_sweep.get("conclusion"),
        "layer_sweep_best_layer": best_layer.get("layer"),
        "layer_sweep_best_rmse": best_layer.get("rmse"),
    }


def conclude(
    *, source_audit: dict[str, Any], precision: dict[str, Any], evidence: dict[str, Any]
) -> str:
    facts = source_audit["facts"]
    if not facts["capture_sets_embedding"]:
        return "capture_embedding_setup_suspect"
    if evidence.get("tap_compare_best_key") != "hidden_in_f32":
        return "tap_placement_suspect"
    if evidence.get("layer_sweep_best_layer") != evidence.get("capture_layer_id"):
        return "layer_numbering_suspect"
    if precision.get("count", 0) > 0:
        return "precision_contractions_or_preceding_layer_math_suspect"
    return "preceding_layer_math_or_materialization_suspect"


def next_action(conclusion: str) -> str:
    if conclusion == "capture_embedding_setup_suspect":
        return "fix_or_instrument_capture_attention_layer_embedding_setup"
    if conclusion == "tap_placement_suspect":
        return "capture_llamacpp_closest_tap_and_compare_directly"
    if conclusion == "layer_numbering_suspect":
        return "align_hipengine_and_llamacpp_layer_ids"
    if conclusion == "precision_contractions_or_preceding_layer_math_suspect":
        return "run_earliest_layer_hidden_in_sweep_and_audit_precision_contractors"
    return "run_earliest_layer_hidden_in_sweep_between_hipengine_and_llamacpp"


def extract_function_body(text: str, name: str) -> str:
    pattern = re.compile(rf"^(?P<indent>\s*)def {re.escape(name)}\b.*$", re.M)
    match = pattern.search(text)
    if match is None:
        return ""
    start = match.start()
    indent = len(match.group("indent"))
    rest = text[match.end() :]
    end = len(text)
    for next_match in re.finditer(r"^\s*def \w+\b|^\s*@", rest, re.M):
        line = next_match.group(0)
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= indent:
            end = match.end() + next_match.start()
            break
    return text[start:end]


def find_line(text: str, needle: str) -> int | None:
    index = text.find(needle)
    if index < 0:
        return None
    return text.count("\n", 0, index) + 1


def parse_layers(spec: str) -> tuple[int, ...]:
    layers: list[int] = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            layers.extend(range(start, end + step, step))
        else:
            layers.append(int(item))
    return tuple(layers)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
