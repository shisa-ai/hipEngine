#!/usr/bin/env python3
"""Emit the StepFun same-prompt llama.cpp logits helper contract.

The contract records the retained StepFun prompt/token IDs, the llama.cpp source
hooks that can produce final prompt logits, and the concrete source/CLI behavior
needed to close the current special-token-safe logits capture blocker. It is
handoff evidence only: it does not edit the external llama.cpp tree and does not
claim oracle, KV, e2e, or performance readiness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stepfun_correctness_status as status_mod
from scripts import stepfun_llamacpp_logits_preflight as preflight_mod
from scripts import stepfun_llamacpp_logits_probe as probe_mod

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-helper-contract.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"
DEFAULT_LLAMA_CPP_ROOT = Path("/home/lhl/llama.cpp/llama.cpp-vulkan")
DEFAULT_PROBE_ARTIFACT = probe_mod.DEFAULT_OUTPUT
DEFAULT_PREFLIGHT_ARTIFACT = preflight_mod.DEFAULT_OUTPUT

SOURCE_HOOKS = (
    {
        "key": "debug_current_tokenization",
        "path": "examples/debug/debug.cpp",
        "marker": "common_tokenize(ctx, params.prompt, add_bos)",
        "contract_role": "current llama-debug prompt tokenization",
        "contract_status": "insufficient",
        "required_behavior": "tokenize retained prompt with parse_special=true or consume explicit retained token IDs",
        "required_code_shape": "common_tokenize(ctx, params.prompt, add_bos, true)",
    },
    {
        "key": "debug_logits_read",
        "path": "examples/debug/debug.cpp",
        "marker": "llama_get_logits_ith(ctx, tokens.size() - 1)",
        "contract_role": "final prompt-token logits readback",
        "contract_status": "usable",
        "required_behavior": "read logits for the final retained prompt token after prompt decode",
        "required_code_shape": "llama_get_logits_ith(ctx, tokens.size() - 1)",
    },
    {
        "key": "debug_save_logits_gate",
        "path": "examples/debug/debug.cpp",
        "marker": "params.save_logits",
        "contract_role": "existing save-logits control path",
        "contract_status": "usable",
        "required_behavior": "write final logits and prompt-token metadata to retained raw outputs",
        "required_code_shape": "--save-logits / --logits-output-dir path",
    },
    {
        "key": "common_tokenize_parse_special_signature",
        "path": "common/common.h",
        "marker": "bool   parse_special = false);",
        "contract_role": "public common-tokenize parse-special parameter",
        "contract_status": "usable",
        "required_behavior": "pass parse_special=true for literal chat/control token markers in the retained prompt",
        "required_code_shape": "common_tokenize(..., add_special, true)",
    },
    {
        "key": "common_tokenize_llama_tokenize_forward",
        "path": "common/common.cpp",
        "marker": "llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special)",
        "contract_role": "parse-special forwarding to llama_tokenize",
        "contract_status": "usable",
        "required_behavior": "forward parse_special=true to llama_tokenize",
        "required_code_shape": "llama_tokenize(..., add_special, parse_special)",
    },
    {
        "key": "arg_special_output_only",
        "path": "common/arg.cpp",
        "marker": '{"-sp", "--special"}',
        "contract_role": "existing --special flag is output-only for supported examples",
        "contract_status": "insufficient",
        "required_behavior": "do not confuse output-special rendering with prompt special-token parsing",
        "required_code_shape": "add a parse-special/token-ids control to the logits helper instead",
    },
    {
        "key": "arg_parse_special_imatrix_only",
        "path": "common/arg.cpp",
        "marker": '{"--parse-special"}',
        "contract_role": "existing parse-special flag is scoped to imatrix args in this checkout",
        "contract_status": "insufficient",
        "required_behavior": "expose parse-special behavior for the logits helper or accept explicit token IDs",
        "required_code_shape": "helper-specific --parse-special or --token-ids input",
    },
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp-root", type=Path, default=DEFAULT_LLAMA_CPP_ROOT)
    parser.add_argument("--prompt-artifact", type=Path, default=status_mod.DEFAULT_PROMPT_ARTIFACT)
    parser.add_argument("--probe-artifact", type=Path, default=DEFAULT_PROBE_ARTIFACT)
    parser.add_argument("--preflight-artifact", type=Path, default=DEFAULT_PREFLIGHT_ARTIFACT)
    parser.add_argument("--artifact-date", default=DEFAULT_ARTIFACT_DATE)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON output atomically to this path instead of stdout.",
    )
    parser.add_argument(
        "--default-output",
        action="store_true",
        help=f"Write to the canonical artifact path: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--status-only", action="store_true", help="Emit only status.")
    parser.add_argument("--implementation-ready-only", action="store_true")
    parser.add_argument("--missing-evidence-only", action="store_true")
    parser.add_argument("--required-code-shapes-only", action="store_true")
    parser.add_argument("--sha-only", action="store_true")
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _artifact_ref(path: Path) -> dict[str, object]:
    payload = _load_json_object(path)
    if payload is None:
        return {
            "path": str(path),
            "exists": False,
            "artifact_kind": None,
            "status": None,
            "sha256": None,
        }
    return {
        "path": str(path),
        "exists": True,
        "artifact_kind": payload.get("artifact_kind"),
        "status": payload.get("status"),
        "ready": payload.get("ready"),
        "missing_evidence": payload.get("missing_evidence"),
        "sha256": status_mod._stable_json_sha256(payload),
    }


def _find_marker(path: Path, marker: str) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "line": None, "line_text": None, "marker_present": False}
    for line_no, line_text in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        if marker in line_text:
            return {
                "exists": True,
                "line": line_no,
                "line_text": line_text.strip(),
                "marker_present": True,
            }
    return {"exists": True, "line": None, "line_text": None, "marker_present": False}


def _source_hook_record(llama_cpp_root: Path, hook: dict[str, str]) -> dict[str, object]:
    source_path = llama_cpp_root / hook["path"]
    marker_record = _find_marker(source_path, hook["marker"])
    return {
        "key": hook["key"],
        "path": hook["path"],
        "absolute_path": str(source_path),
        "marker": hook["marker"],
        **marker_record,
        "contract_role": hook["contract_role"],
        "contract_status": hook["contract_status"],
        "required_behavior": hook["required_behavior"],
        "required_code_shape": hook["required_code_shape"],
    }


def _prompt_contract(prompt_payload: dict[str, object] | None) -> dict[str, object]:
    payload = prompt_payload or {}
    input_ids = payload.get("input_ids") if isinstance(payload.get("input_ids"), list) else []
    return {
        "prompt": payload.get("prompt"),
        "prompt_length": payload.get("prompt_length"),
        "input_id_count": len(input_ids),
        "input_ids": input_ids,
        "expected_next_token_id": payload.get("next_token_id"),
        "expected_next_token_text": payload.get("next_token_text"),
        "expected_next_token_logit": payload.get("next_token_logit"),
        "generated_first_token_id": probe_mod.DEFAULT_GENERATED_TOKEN_ID,
        "generated_first_token_text": probe_mod.DEFAULT_GENERATED_TOKEN_TEXT,
    }


def build_llamacpp_logits_helper_contract(
    *,
    llama_cpp_root: Path = DEFAULT_LLAMA_CPP_ROOT,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    probe_artifact: Path = DEFAULT_PROBE_ARTIFACT,
    preflight_artifact: Path = DEFAULT_PREFLIGHT_ARTIFACT,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the same-prompt logits helper implementation contract."""

    prompt_payload = _load_json_object(prompt_artifact)
    source_hooks = [_source_hook_record(llama_cpp_root, hook) for hook in SOURCE_HOOKS]
    marker_failures = [record["key"] for record in source_hooks if not record["marker_present"]]
    prompt_ready = bool(prompt_payload and isinstance(prompt_payload.get("input_ids"), list))
    source_hooks_ready = not marker_failures
    implementation_ready = prompt_ready and source_hooks_ready
    missing_evidence = [
        "same_prompt_logits_helper_built",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]
    if not implementation_ready:
        missing_evidence.append("same_prompt_logits_helper_contract_source_hooks_present")
    if not prompt_ready:
        missing_evidence.append("retained_prompt_input_ids_present")
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_llamacpp_logits_helper_contract",
        "date": artifact_date,
        "status": "blocked",
        "implementation_ready": implementation_ready,
        "llama_cpp_root": str(llama_cpp_root),
        "prompt_artifact": _artifact_ref(prompt_artifact),
        "probe_artifact": _artifact_ref(probe_artifact),
        "preflight_artifact": _artifact_ref(preflight_artifact),
        "prompt_contract": _prompt_contract(prompt_payload),
        "source_hooks_ready": source_hooks_ready,
        "source_hook_marker_failures": marker_failures,
        "source_hooks": source_hooks,
        "required_code_shapes": [record["required_code_shape"] for record in source_hooks],
        "helper_contract": {
            "must_match_retained_input_ids": True,
            "accepted_prompt_paths": [
                "tokenize prompt text with parse_special=true and add_bos=false for this retained artifact",
                "or bypass text tokenization by accepting the retained input_ids explicitly",
            ],
            "decode_contract": "decode all retained prompt tokens with logits enabled for the final token",
            "logits_contract": (
                "write or return final-token logits for the full GGUF vocabulary, then compare "
                "expected token 369 against generated token 671"
            ),
            "raw_output_contract": list(probe_mod._raw_output_paths(probe_mod.DEFAULT_MODEL, probe_mod.DEFAULT_RAW_OUTPUT_DIR).keys()),
        },
        "recommended_commands": [
            "patch or build a llama.cpp logits helper that calls common_tokenize(..., parse_special=true) or accepts explicit token IDs",
            "cmake --build /home/lhl/llama.cpp/llama.cpp-vulkan/build-vulkan-release --target llama-debug -j 8",
            "python3 scripts/stepfun_llamacpp_logits_probe.py --execute --default-output --pretty",
            "python3 scripts/stepfun_llamacpp_logits_preflight.py --default-output --pretty",
        ],
        "missing_evidence": missing_evidence,
        "blocked_reason": (
            "same-prompt-capable llama.cpp logits helper has not been built and captured"
        ),
        "next_action": (
            "build a logits helper that tokenizes with parse_special=true or accepts retained input_ids, "
            "then rerun scripts/stepfun_llamacpp_logits_probe.py --execute --default-output --pretty"
        ),
        "blocked_gates": ["oracle_parity", "kv_backed_decode", "e2e_inference"],
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This contract only identifies the source hooks for same-prompt logits capture; "
                "generated_text_matches_target remains unresolved."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_llamacpp_logits_helper_contract(
        llama_cpp_root=args.llama_cpp_root,
        prompt_artifact=args.prompt_artifact,
        probe_artifact=args.probe_artifact,
        preflight_artifact=args.preflight_artifact,
        artifact_date=args.artifact_date,
    )
    if args.status_only:
        payload: object = report["status"]
    elif args.implementation_ready_only:
        payload = report["implementation_ready"]
    elif args.missing_evidence_only:
        payload = report["missing_evidence"]
    elif args.required_code_shapes_only:
        payload = report["required_code_shapes"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(report)
    else:
        payload = report
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
