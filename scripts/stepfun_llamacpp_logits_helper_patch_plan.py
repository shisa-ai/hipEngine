#!/usr/bin/env python3
"""Emit the StepFun llama.cpp retained-token logits helper patch plan.

This script does not modify the external llama.cpp checkout. It mechanically
checks that the source anchors needed for a small helper patch are present, then
emits the exact target files, code-shape edits, build command, and in-tree probe
command needed to close the retained-token logits blocker.
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
from scripts import stepfun_llamacpp_logits_helper_contract as contract_mod
from scripts import stepfun_llamacpp_logits_probe as probe_mod

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-helper-patch-plan.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"
DEFAULT_BUILD_DIR = Path("/home/lhl/llama.cpp/llama.cpp-vulkan/build-vulkan-release")

SOURCE_ANCHORS = (
    {
        "key": "debug_include_regex",
        "path": "examples/debug/debug.cpp",
        "marker": "#include <regex>",
        "role": "insert <sstream> for CSV token-id parsing near existing standard includes",
    },
    {
        "key": "debug_output_data_constructor",
        "path": "examples/debug/debug.cpp",
        "marker": "output_data(llama_context * ctx, const llama_model * model, const common_params & params) {",
        "role": "thread already-decoded prompt tokens into output_data without re-tokenizing",
    },
    {
        "key": "debug_output_data_current_tokenize",
        "path": "examples/debug/debug.cpp",
        "marker": "tokens = common_tokenize(ctx, params.prompt, add_bos);",
        "role": "replace output_data re-tokenization with caller-provided retained tokens",
    },
    {
        "key": "debug_run_signature",
        "path": "examples/debug/debug.cpp",
        "marker": "static bool run(llama_context * ctx, const common_params & params) {",
        "role": "pass helper-local tokenization options into run()",
    },
    {
        "key": "debug_run_current_tokenize",
        "path": "examples/debug/debug.cpp",
        "marker": "std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);",
        "role": "select retained --token-ids or parse-special prompt tokenization before decode",
    },
    {
        "key": "debug_save_output_data_call",
        "path": "examples/debug/debug.cpp",
        "marker": "output_data output {ctx, model, params};",
        "role": "save logits and prompt-token metadata for the exact decoded token vector",
    },
    {
        "key": "debug_common_params_parse",
        "path": "examples/debug/debug.cpp",
        "marker": "if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DEBUG, print_usage)) {",
        "role": "strip helper-local --token-ids/--parse-special before common_params_parse",
    },
    {
        "key": "debug_run_call",
        "path": "examples/debug/debug.cpp",
        "marker": "if (!run(ctx, params)) {",
        "role": "pass helper-local tokenization options to run()",
    },
    {
        "key": "common_tokenize_parse_special_signature",
        "path": "common/common.h",
        "marker": "bool   parse_special = false);",
        "role": "confirm parse_special is already supported by common_tokenize",
    },
)

PATCH_STEPS = (
    {
        "key": "add_helper_local_cli_state",
        "path": "examples/debug/debug.cpp",
        "anchors": ["debug_include_regex", "debug_common_params_parse"],
        "edit_kind": "insert helper-local argv pre-parser",
        "proposed_code_shapes": [
            "struct stepfun_debug_extra_params { bool parse_special = false; std::vector<llama_token> token_ids; };",
            "accept --token-ids comma-separated llama_token IDs and remove it plus its value before common_params_parse",
            "accept --parse-special as a debug-helper-local prompt tokenization flag and remove it before common_params_parse",
            "include the helper flags in debug.cpp print_usage so llama-debug --help advertises them",
        ],
        "why": "common/arg.cpp scopes --parse-special to imatrix in this checkout; a debug-local pre-parser avoids broad common-arg behavior changes.",
    },
    {
        "key": "decode_retained_tokens",
        "path": "examples/debug/debug.cpp",
        "anchors": ["debug_run_signature", "debug_run_current_tokenize"],
        "edit_kind": "replace prompt tokenization selection",
        "proposed_code_shapes": [
            "const bool add_bos = extra.token_ids.empty() ? llama_vocab_get_add_bos(vocab) : false;",
            "std::vector<llama_token> tokens = extra.token_ids.empty() ? common_tokenize(ctx, params.prompt, add_bos, extra.parse_special) : extra.token_ids;",
            "reject empty explicit token-id lists before llama_decode",
        ],
        "why": "the retained StepFun prompt artifact already has the exact special-token IDs; bypassing text tokenization removes the current oracle evidence ambiguity.",
    },
    {
        "key": "save_exact_prompt_tokens",
        "path": "examples/debug/debug.cpp",
        "anchors": ["debug_output_data_constructor", "debug_output_data_current_tokenize", "debug_save_output_data_call"],
        "edit_kind": "thread decoded tokens into output_data",
        "proposed_code_shapes": [
            "output_data(llama_context * ctx, const llama_model * model, const common_params & params, const std::vector<llama_token> & input_tokens)",
            "tokens = input_tokens;",
            "output_data output {ctx, model, params, tokens};",
        ],
        "why": "logits and saved prompt-token metadata must describe the same retained tokens that were decoded before llama_get_logits_ith(ctx, tokens.size() - 1).",
    },
    {
        "key": "wire_main_to_helper_args",
        "path": "examples/debug/debug.cpp",
        "anchors": ["debug_common_params_parse", "debug_run_call"],
        "edit_kind": "thread helper-local params through main",
        "proposed_code_shapes": [
            "stepfun_debug_extra_params stepfun_extra;",
            "auto filtered_argv = stepfun_filter_debug_args(argc, argv, stepfun_extra);",
            "common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_DEBUG, print_usage)",
            "run(ctx, params, stepfun_extra)",
        ],
        "why": "the future helper binary should keep normal llama-debug behavior except for the explicit retained-token oracle path.",
    },
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp-root", type=Path, default=contract_mod.DEFAULT_LLAMA_CPP_ROOT)
    parser.add_argument("--prompt-artifact", type=Path, default=status_mod.DEFAULT_PROMPT_ARTIFACT)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
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
    parser.add_argument("--anchors-ready-only", action="store_true", help="Emit only the anchor readiness boolean.")
    parser.add_argument("--missing-evidence-only", action="store_true", help="Emit only missing evidence.")
    parser.add_argument("--patch-steps-only", action="store_true", help="Emit only patch steps.")
    parser.add_argument("--sha-only", action="store_true", help="Emit stable SHA-256 of the selected payload.")
    parser.add_argument(
        "--verify-patch-plan",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted llama.cpp logits helper patch-plan artifact with current "
            f"source-anchor/prompt recipe metadata. If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-patch-plan, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-patch-plan, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-patch-plan, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


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


def _source_anchor_record(llama_cpp_root: Path, anchor: dict[str, str]) -> dict[str, object]:
    source_path = llama_cpp_root / anchor["path"]
    marker_record = _find_marker(source_path, anchor["marker"])
    return {
        "key": anchor["key"],
        "path": anchor["path"],
        "absolute_path": str(source_path),
        "marker": anchor["marker"],
        "role": anchor["role"],
        **marker_record,
    }


def _prompt_contract(prompt_payload: dict[str, object] | None) -> dict[str, object]:
    payload = prompt_payload or {}
    input_ids = payload.get("input_ids") if isinstance(payload.get("input_ids"), list) else []
    token_ids_csv = ",".join(str(item) for item in input_ids if isinstance(item, int))
    return {
        "prompt_artifact_present": prompt_payload is not None,
        "prompt": payload.get("prompt"),
        "prompt_length": payload.get("prompt_length"),
        "input_id_count": len(input_ids),
        "input_ids": input_ids,
        "token_ids_csv": token_ids_csv,
        "expected_next_token_id": payload.get("next_token_id"),
        "expected_next_token_text": payload.get("next_token_text"),
    }


def build_llamacpp_logits_helper_patch_plan(
    *,
    llama_cpp_root: Path = contract_mod.DEFAULT_LLAMA_CPP_ROOT,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    build_dir: Path = DEFAULT_BUILD_DIR,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return a retained-token helper patch plan without editing llama.cpp."""

    prompt_payload = _load_json_object(prompt_artifact)
    prompt_contract = _prompt_contract(prompt_payload)
    source_anchors = [_source_anchor_record(llama_cpp_root, anchor) for anchor in SOURCE_ANCHORS]
    missing_anchors = [record["key"] for record in source_anchors if not record["marker_present"]]
    prompt_ready = bool(prompt_contract["input_ids"])
    anchors_ready = not missing_anchors
    token_ids_csv = str(prompt_contract["token_ids_csv"])
    build_command = f"cmake --build {build_dir} --target llama-debug -j"
    probe_command = (
        "python3 scripts/stepfun_llamacpp_logits_probe.py "
        "--prompt-token-source retained-input-ids --execute --default-output --pretty"
    )
    missing_evidence = []
    if not anchors_ready:
        missing_evidence.append("llama_cpp_helper_patch_source_anchors_present")
    if not prompt_ready:
        missing_evidence.append("retained_prompt_input_ids_present")
    missing_evidence.extend(
        [
            "llama_cpp_token_ids_helper_patch_applied",
            "same_prompt_logits_helper_built",
            "llama_cpp_same_prompt_logits_artifact_present",
        ]
    )
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_llamacpp_logits_helper_patch_plan",
        "date": artifact_date,
        "status": "blocked",
        "ready": False,
        "llama_cpp_root": str(llama_cpp_root),
        "prompt_artifact": str(prompt_artifact),
        "prompt_contract": prompt_contract,
        "anchors_ready": anchors_ready,
        "missing_anchors": missing_anchors,
        "source_anchors": source_anchors,
        "patch_steps": list(PATCH_STEPS),
        "patch_application_policy": {
            "edits_external_tree": False,
            "reason": (
                "hipEngine evidence records the exact helper patch plan, but this script does not "
                "modify the external llama.cpp checkout."
            ),
            "allowed_next_step": "apply the reviewed patch manually in the llama.cpp checkout, rebuild llama-debug, then rerun the retained-token probe",
        },
        "build_command": build_command,
        "probe_command_after_build": probe_command,
        "expected_probe_token_ids_argument": token_ids_csv,
        "expected_probe_command_shape": [
            "--prompt-token-source retained-input-ids",
            "--token-ids",
            token_ids_csv,
            "--save-logits",
            "--logits-output-dir",
        ],
        "missing_evidence": missing_evidence,
        "blocked_reason": "retained-token llama.cpp helper patch is planned but not applied, rebuilt, or captured",
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This artifact is a patch recipe only; generated_text_matches_target remains unresolved "
                "until the retained-token helper captures same-prompt logits and oracle comparison passes."
            ),
        },
    }


def verify_llamacpp_logits_helper_patch_plan(
    patch_plan_artifact: Path,
    *,
    current_report: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted llama.cpp logits helper patch plan with current metadata."""

    persisted_payload = _load_json_object(patch_plan_artifact)
    persisted: dict[str, object] = (
        persisted_payload if persisted_payload is not None else {"exists": False}
    )
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_report)
    failures: list[dict[str, object]] = []
    if persisted != current_report:
        failures.append(
            {
                "name": "llamacpp_logits_helper_patch_plan_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted llama.cpp logits helper patch-plan artifact differs "
                    "from current source-anchor or retained-prompt recipe metadata."
                ),
            }
        )
    all_match = not failures
    persisted_prompt = persisted.get("prompt_contract")
    current_prompt = current_report.get("prompt_contract")
    return {
        "schema_version": 1,
        "artifact_path": str(patch_plan_artifact),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_artifact_sha256": persisted_sha256,
        "current_artifact_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "verification_failure_count": len(failures),
        "persisted_status": persisted.get("status"),
        "current_status": current_report.get("status"),
        "persisted_anchors_ready": persisted.get("anchors_ready"),
        "current_anchors_ready": current_report.get("anchors_ready"),
        "persisted_missing_anchors": persisted.get("missing_anchors"),
        "current_missing_anchors": current_report.get("missing_anchors"),
        "persisted_missing_evidence": persisted.get("missing_evidence"),
        "current_missing_evidence": current_report.get("missing_evidence"),
        "persisted_patch_step_keys": [
            step.get("key")
            for step in persisted.get("patch_steps", [])
            if isinstance(step, dict)
        ],
        "current_patch_step_keys": [
            step.get("key")
            for step in current_report.get("patch_steps", [])
            if isinstance(step, dict)
        ],
        "persisted_expected_probe_token_ids_argument": persisted.get(
            "expected_probe_token_ids_argument"
        ),
        "current_expected_probe_token_ids_argument": current_report.get(
            "expected_probe_token_ids_argument"
        ),
        "persisted_expected_probe_command_shape": persisted.get(
            "expected_probe_command_shape"
        ),
        "current_expected_probe_command_shape": current_report.get(
            "expected_probe_command_shape"
        ),
        "persisted_prompt_input_ids": (
            persisted_prompt.get("input_ids") if isinstance(persisted_prompt, dict) else None
        ),
        "current_prompt_input_ids": (
            current_prompt.get("input_ids") if isinstance(current_prompt, dict) else None
        ),
        "persisted_build_command": persisted.get("build_command"),
        "current_build_command": current_report.get("build_command"),
        "persisted_probe_command_after_build": persisted.get("probe_command_after_build"),
        "current_probe_command_after_build": current_report.get("probe_command_after_build"),
    }


def _select_payload(report: dict[str, object], args: argparse.Namespace) -> object:
    if args.status_only:
        return report["status"]
    if args.anchors_ready_only:
        return report["anchors_ready"]
    if args.missing_evidence_only:
        return report["missing_evidence"]
    if args.patch_steps_only:
        return report["patch_steps"]
    return report


def _write_json(payload: object, *, output: Path | None, pretty: bool) -> None:
    text = json.dumps(payload, indent=2 if pretty else None, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(output)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = DEFAULT_OUTPUT if args.default_output else args.output
    report = build_llamacpp_logits_helper_patch_plan(
        llama_cpp_root=args.llama_cpp_root,
        prompt_artifact=args.prompt_artifact,
        build_dir=args.build_dir,
        artifact_date=args.artifact_date,
    )
    if args.verify_patch_plan is not None:
        verification = verify_llamacpp_logits_helper_patch_plan(
            args.verify_patch_plan,
            current_report=report,
        )
        if args.verification_status_only:
            payload: object = verification["status"]
        elif args.verification_failures_only:
            payload = verification["verification_failures"]
        elif args.verification_sha_only:
            payload = status_mod._stable_json_sha256(verification)
        else:
            payload = verification
        if args.sha_only:
            payload = status_mod._stable_json_sha256(payload)
        _write_json(payload, output=output, pretty=args.pretty)
        return (
            0
            if verification["all_match"] is True
            else status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
        )
    payload = _select_payload(report, args)
    if args.sha_only:
        payload = status_mod._stable_json_sha256(payload)
    _write_json(payload, output=output, pretty=args.pretty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
