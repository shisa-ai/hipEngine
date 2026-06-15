#!/usr/bin/env python3
"""Generate and dry-run the StepFun retained-token llama-debug patch.

The script is read-only by default: it transforms the current external
llama.cpp debug helper in memory, emits a unified diff, and optionally runs
`git apply --check` against the external checkout without applying it. The
result is a compact artifact that proves the reviewed patch still applies before
someone edits/rebuilds the external llama.cpp tree.
"""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stepfun_correctness_status as status_mod
from scripts import stepfun_llamacpp_logits_helper_contract as contract_mod
from scripts import stepfun_llamacpp_logits_helper_patch_plan as patch_plan_mod

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-helper-patch-dry-run.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"
DEBUG_RELATIVE_PATH = Path("examples/debug/debug.cpp")

INCLUDE_OLD = "#include <regex>\n"
INCLUDE_NEW = "#include <regex>\n#include <sstream>\n"

USAGE_OLD = """          Save logits/embeddings:\n\n          {prog} -m model.gguf -p \"Hello my name is\" --save-logits\n\n          Add --embedding to save embeddings)\" \"\\n\";\n"""
USAGE_NEW = """          Save logits/embeddings:\n\n          {prog} -m model.gguf -p \"Hello my name is\" --save-logits\n\n          StepFun retained-token oracle helper:\n\n          {prog} -m model.gguf -p \"<|im_start|>...\" --save-logits --token-ids 0,128006,...\n\n          Use --token-ids to decode an explicit comma-separated prompt token list.\n          Use --parse-special to tokenize -p with special-token parsing enabled.\n\n          Add --embedding to save embeddings)\" \"\\n\";\n"""

HELPER_INSERT_ANCHOR = """static bool has_pooling(llama_context * ctx) {\n"""
HELPER_INSERT = r'''
struct stepfun_debug_extra_params {
    bool parse_special = false;
    std::vector<llama_token> token_ids;
};

static std::vector<llama_token> stepfun_parse_token_ids_csv(const std::string & value) {
    std::vector<llama_token> result;
    std::stringstream stream(value);
    std::string item;

    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            throw std::runtime_error("empty token id in --token-ids");
        }
        std::size_t consumed = 0;
        const long parsed = std::stol(item, &consumed, 10);
        if (consumed != item.size()) {
            throw std::runtime_error("invalid token id in --token-ids: " + item);
        }
        result.push_back(static_cast<llama_token>(parsed));
    }

    if (result.empty()) {
        throw std::runtime_error("--token-ids requires at least one token id");
    }

    return result;
}

static std::vector<std::string> stepfun_filter_debug_args(int argc, char ** argv, stepfun_debug_extra_params & extra) {
    std::vector<std::string> filtered;
    filtered.reserve(argc);

    const std::string token_ids_prefix = "--token-ids=";
    for (int i = 0; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--parse-special") {
            extra.parse_special = true;
            continue;
        }
        if (arg == "--token-ids") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--token-ids requires a comma-separated value");
            }
            extra.token_ids = stepfun_parse_token_ids_csv(argv[++i]);
            continue;
        }
        if (arg.rfind(token_ids_prefix, 0) == 0) {
            extra.token_ids = stepfun_parse_token_ids_csv(arg.substr(token_ids_prefix.size()));
            continue;
        }
        filtered.push_back(arg);
    }

    return filtered;
}

static std::vector<char *> stepfun_argv_pointers(std::vector<std::string> & args) {
    std::vector<char *> ptrs;
    ptrs.reserve(args.size());
    for (std::string & arg : args) {
        ptrs.push_back(arg.data());
    }
    return ptrs;
}

'''

OUTPUT_CTOR_OLD = """    output_data(llama_context * ctx, const llama_model * model, const common_params & params) {\n        const llama_vocab * vocab = llama_model_get_vocab(model);\n        const bool add_bos = llama_vocab_get_add_bos(vocab);\n\n        tokens = common_tokenize(ctx, params.prompt, add_bos);\n        prompt = params.prompt;\n"""
OUTPUT_CTOR_NEW = """    output_data(llama_context * ctx, const llama_model * model, const common_params & params, const std::vector<llama_token> & input_tokens) {\n        const llama_vocab * vocab = llama_model_get_vocab(model);\n\n        tokens = input_tokens;\n        prompt = params.prompt;\n"""

RUN_SIGNATURE_OLD = "static bool run(llama_context * ctx, const common_params & params) {\n"
RUN_SIGNATURE_NEW = "static bool run(llama_context * ctx, const common_params & params, const stepfun_debug_extra_params & extra) {\n"

RUN_TOKENIZE_OLD = """    const bool add_bos = llama_vocab_get_add_bos(vocab);\n\n    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);\n"""
RUN_TOKENIZE_NEW = """    const bool add_bos = extra.token_ids.empty() ? llama_vocab_get_add_bos(vocab) : false;\n\n    std::vector<llama_token> tokens = extra.token_ids.empty()\n        ? common_tokenize(ctx, params.prompt, add_bos, extra.parse_special)\n        : extra.token_ids;\n"""

OUTPUT_CALL_OLD = "            output_data output {ctx, model, params};\n"
OUTPUT_CALL_NEW = "            output_data output {ctx, model, params, tokens};\n"

MAIN_PARSE_OLD = """    common_init();\n\n    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DEBUG, print_usage)) {\n        return 1;\n    }\n"""
MAIN_PARSE_NEW = """    common_init();\n\n    stepfun_debug_extra_params stepfun_extra;\n    std::vector<std::string> filtered_args;\n    try {\n        filtered_args = stepfun_filter_debug_args(argc, argv, stepfun_extra);\n    } catch (const std::exception & e) {\n        LOG_ERR(\"%s : error parsing StepFun debug helper args: %s\\n\", __func__, e.what());\n        return 1;\n    }\n    std::vector<char *> filtered_argv = stepfun_argv_pointers(filtered_args);\n\n    if (!common_params_parse(static_cast<int>(filtered_argv.size()), filtered_argv.data(), params, LLAMA_EXAMPLE_DEBUG, print_usage)) {\n        return 1;\n    }\n"""

RUN_CALL_OLD = "    if (!run(ctx, params)) {\n"
RUN_CALL_NEW = "    if (!run(ctx, params, stepfun_extra)) {\n"

TRANSFORMS = (
    ("include_sstream", INCLUDE_OLD, INCLUDE_NEW),
    ("usage_helper_flags", USAGE_OLD, USAGE_NEW),
    ("helper_local_arg_parser", HELPER_INSERT_ANCHOR, HELPER_INSERT + HELPER_INSERT_ANCHOR),
    ("output_data_exact_tokens", OUTPUT_CTOR_OLD, OUTPUT_CTOR_NEW),
    ("run_signature_extra", RUN_SIGNATURE_OLD, RUN_SIGNATURE_NEW),
    ("run_retained_token_decode", RUN_TOKENIZE_OLD, RUN_TOKENIZE_NEW),
    ("save_output_exact_tokens", OUTPUT_CALL_OLD, OUTPUT_CALL_NEW),
    ("main_filter_helper_args", MAIN_PARSE_OLD, MAIN_PARSE_NEW),
    ("main_run_extra", RUN_CALL_OLD, RUN_CALL_NEW),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp-root", type=Path, default=contract_mod.DEFAULT_LLAMA_CPP_ROOT)
    parser.add_argument("--artifact-date", default=DEFAULT_ARTIFACT_DATE)
    parser.add_argument("--skip-apply-check", action="store_true", help="Do not run git apply --check.")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON output atomically to this path.")
    parser.add_argument("--default-output", action="store_true", help=f"Write to {DEFAULT_OUTPUT}.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--status-only", action="store_true", help="Emit only status.")
    parser.add_argument("--patch-ready-only", action="store_true", help="Emit only patch readiness boolean.")
    parser.add_argument("--apply-check-status-only", action="store_true", help="Emit only git apply check status.")
    parser.add_argument("--missing-evidence-only", action="store_true", help="Emit only missing evidence.")
    parser.add_argument("--patch-sha-only", action="store_true", help="Emit only the unified diff SHA-256.")
    parser.add_argument("--patch-only", action="store_true", help="Emit only the generated unified diff text.")
    parser.add_argument("--sha-only", action="store_true", help="Emit stable SHA-256 of the selected JSON payload.")
    return parser.parse_args(argv)


def _apply_transform(text: str, key: str, old: str, new: str) -> tuple[str, dict[str, object]]:
    count = text.count(old)
    if count != 1:
        return text, {"key": key, "applied": False, "match_count": count}
    return text.replace(old, new, 1), {"key": key, "applied": True, "match_count": count}


def _generate_patch(original: str, transformed: str, relative_path: Path) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            transformed.splitlines(keepends=True),
            fromfile=f"a/{relative_path.as_posix()}",
            tofile=f"b/{relative_path.as_posix()}",
        )
    )


def _git_apply_check(llama_cpp_root: Path, patch_text: str) -> dict[str, object]:
    completed = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=llama_cpp_root,
        input=patch_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def build_llamacpp_logits_helper_patch_dry_run(
    *,
    llama_cpp_root: Path = contract_mod.DEFAULT_LLAMA_CPP_ROOT,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
    skip_apply_check: bool = False,
) -> dict[str, object]:
    """Return a read-only dry-run report for the retained-token helper patch."""

    debug_path = llama_cpp_root / DEBUG_RELATIVE_PATH
    source_exists = debug_path.exists()
    original = debug_path.read_text() if source_exists else ""
    transformed = original
    transform_results: list[dict[str, object]] = []
    for key, old, new in TRANSFORMS:
        transformed, result = _apply_transform(transformed, key, old, new)
        transform_results.append(result)
    missing_transforms = [result["key"] for result in transform_results if not result["applied"]]
    patch_text = _generate_patch(original, transformed, DEBUG_RELATIVE_PATH) if source_exists else ""
    patch_sha256 = status_mod._stable_json_sha256(patch_text)
    apply_check = {"status": "skipped", "returncode": None, "stdout": "", "stderr": ""}
    if source_exists and patch_text and not missing_transforms and not skip_apply_check:
        apply_check = _git_apply_check(llama_cpp_root, patch_text)
    patch_ready = source_exists and bool(patch_text) and not missing_transforms and apply_check["status"] in {"passed", "skipped"}
    missing_evidence: list[str] = []
    if not source_exists:
        missing_evidence.append("llama_cpp_debug_source_present")
    if missing_transforms:
        missing_evidence.append("llama_cpp_token_ids_helper_patch_transforms_apply")
    if apply_check["status"] == "failed":
        missing_evidence.append("llama_cpp_token_ids_helper_patch_git_apply_check_passes")
    missing_evidence.extend(
        [
            "llama_cpp_token_ids_helper_patch_applied",
            "same_prompt_logits_helper_built",
            "llama_cpp_same_prompt_logits_artifact_present",
        ]
    )
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_llamacpp_logits_helper_patch_dry_run",
        "date": artifact_date,
        "status": "blocked",
        "ready": False,
        "llama_cpp_root": str(llama_cpp_root),
        "target_file": DEBUG_RELATIVE_PATH.as_posix(),
        "source_exists": source_exists,
        "transform_results": transform_results,
        "missing_transforms": missing_transforms,
        "patch_ready": patch_ready,
        "patch_line_count": len(patch_text.splitlines()),
        "patch_sha256": patch_sha256,
        "git_apply_check": apply_check,
        "patch_application_policy": {
            "edits_external_tree": False,
            "dry_run_only": True,
            "reason": "This artifact proves the patch applies but does not modify /home/lhl/llama.cpp.",
        },
        "patch_summary": {
            "adds_debug_local_token_ids_flag": True,
            "adds_debug_local_parse_special_flag": True,
            "decodes_retained_token_ids_without_retokenizing": True,
            "threads_exact_tokens_to_saved_prompt_metadata": True,
        },
        "helper_binary_after_build": str(patch_plan_mod.DEFAULT_BUILD_DIR.joinpath("bin/llama-debug")),
        "build_command": f"cmake --build {patch_plan_mod.DEFAULT_BUILD_DIR} --target llama-debug -j",
        "probe_command_after_build": (
            "python3 scripts/stepfun_llamacpp_logits_probe.py "
            "--prompt-token-source retained-input-ids --execute --default-output --pretty"
        ),
        "missing_evidence": missing_evidence,
        "blocked_reason": "retained-token helper patch dry-run is ready but the external patch/build/capture has not been performed",
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": "Dry-run patch applicability is not same-prompt logits evidence.",
        },
    }


def _select_payload(report: dict[str, object], patch_text: str, args: argparse.Namespace) -> object:
    if args.patch_only:
        return patch_text
    if args.status_only:
        return report["status"]
    if args.patch_ready_only:
        return report["patch_ready"]
    if args.apply_check_status_only:
        return report["git_apply_check"]["status"]
    if args.missing_evidence_only:
        return report["missing_evidence"]
    if args.patch_sha_only:
        return report["patch_sha256"]
    return report


def _write_payload(payload: object, *, output: Path | None, pretty: bool, raw_text: bool = False) -> None:
    if raw_text:
        text = str(payload)
    else:
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
    report = build_llamacpp_logits_helper_patch_dry_run(
        llama_cpp_root=args.llama_cpp_root,
        artifact_date=args.artifact_date,
        skip_apply_check=args.skip_apply_check,
    )
    debug_path = args.llama_cpp_root / DEBUG_RELATIVE_PATH
    patch_text = ""
    if debug_path.exists() and not report["missing_transforms"]:
        original = debug_path.read_text()
        transformed = original
        for key, old, new in TRANSFORMS:
            transformed, _ = _apply_transform(transformed, key, old, new)
        patch_text = _generate_patch(original, transformed, DEBUG_RELATIVE_PATH)
    payload = _select_payload(report, patch_text, args)
    raw_text = args.patch_only and not args.sha_only
    if args.sha_only:
        payload = status_mod._stable_json_sha256(payload)
        raw_text = False
    _write_payload(payload, output=output, pretty=args.pretty, raw_text=raw_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
