#!/usr/bin/env python3
"""Plan the llama.cpp post-output_norm FP32 hidden-seed oracle capture."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SOURCE_DIR = Path("/home/lhl/llama.cpp/llama.cpp-hip")
DEFAULT_RUNNER = Path("hipengine/runtime/qwen35_gguf_runner.py")
DEFAULT_DOC = Path("docs/MTP-gguf.md")
DEFAULT_DECISION = Path(
    "benchmarks/results/mtp-gguf-iter300-hidden-precision-decision-audit.json"
)
DEFAULT_OUTPUT = Path(
    "benchmarks/results/mtp-gguf-iter301-llamacpp-hidden-seed-oracle-plan.json"
)
DEFAULT_PROMPT_TOKENS = (
    "248045,846,198,7734,264,2716,40719,13,248046,198,248045,74455,198,"
    "248068,271,248069,271"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--prompt-tokens", default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--position", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=301)
    args = parser.parse_args()

    artifact = build_hidden_seed_oracle_plan(
        source_dir=args.source_dir,
        runner_path=args.runner,
        doc_path=args.doc,
        decision_path=args.decision,
        prompt_tokens=parse_prompt_tokens(args.prompt_tokens),
        position=args.position,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "conclusion": artifact["conclusion"],
                "llamacpp_api_ready": artifact["llamacpp_oracle_api"]["ready"],
                "hipengine_api_ready": artifact["hipengine_fp32_seed_api"]["ready"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_hidden_seed_oracle_plan(
    *,
    source_dir: Path,
    runner_path: Path,
    doc_path: Path,
    decision_path: Path,
    prompt_tokens: tuple[int, ...],
    position: int,
    iteration: int = 301,
) -> dict[str, Any]:
    source_files = load_llamacpp_sources(source_dir)
    runner_text = runner_path.read_text()
    doc_text = doc_path.read_text()
    decision = read_json(decision_path)
    llama_api = audit_llamacpp_nextn_api(source_files)
    qwen_graph = audit_qwen35moe_h_nextn(source_files["qwen35moe.cpp"])
    context_extract = audit_llamacpp_context_extraction(source_files["llama-context.cpp"])
    hipengine_api = audit_hipengine_fp32_seed_api(runner_text)
    doc_contract = audit_doc_contract(doc_text)
    contract = build_comparison_contract(
        decision=decision,
        prompt_tokens=prompt_tokens,
        position=position,
    )
    readiness = decide_readiness(
        llama_api=llama_api,
        qwen_graph=qwen_graph,
        context_extract=context_extract,
        hipengine_api=hipengine_api,
        doc_contract=doc_contract,
        decision=decision,
    )
    return {
        "schema": 1,
        "kind": "llamacpp_mtp_hidden_seed_oracle_plan",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": readiness["status"],
        "source_dir": str(source_dir),
        "runner_path": str(runner_path),
        "doc_path": str(doc_path),
        "decision_path": str(decision_path),
        "llamacpp_oracle_api": llama_api,
        "llamacpp_qwen35moe_h_nextn": qwen_graph,
        "llamacpp_context_extraction": context_extract,
        "hipengine_fp32_seed_api": hipengine_api,
        "doc_contract": doc_contract,
        "comparison_contract": contract,
        "harness_plan": build_harness_plan(contract),
        "readiness": readiness,
        "conclusion": readiness["conclusion"],
        "external_checkout_modified": False,
        "next_action": readiness["next_action"],
    }


def load_llamacpp_sources(source_dir: Path) -> dict[str, str]:
    paths = {
        "llama-ext.h": source_dir / "src" / "llama-ext.h",
        "llama-context.h": source_dir / "src" / "llama-context.h",
        "llama-context.cpp": source_dir / "src" / "llama-context.cpp",
        "qwen35moe.cpp": source_dir / "src" / "models" / "qwen35moe.cpp",
    }
    return {name: path.read_text() for name, path in paths.items()}


def audit_llamacpp_nextn_api(sources: Mapping[str, str]) -> dict[str, Any]:
    ext_h = sources["llama-ext.h"]
    context_h = sources["llama-context.h"]
    context_cpp = sources["llama-context.cpp"]
    facts = {
        "declares_set_embeddings_nextn": has(ext_h, "llama_set_embeddings_nextn"),
        "declares_get_embeddings_nextn": has(ext_h, "llama_get_embeddings_nextn("),
        "declares_get_embeddings_nextn_ith": has(ext_h, "llama_get_embeddings_nextn_ith"),
        "context_has_embd_nextn_buffer": has(context_h, "buffer_view<float> embd_nextn"),
        "implements_set_embeddings_nextn": has(
            context_cpp,
            "void llama_set_embeddings_nextn",
        ),
        "implements_get_embeddings_nextn_ith": has(
            context_cpp,
            "float * llama_get_embeddings_nextn_ith",
        ),
    }
    ready = all(facts.values())
    return {
        "ready": ready,
        "facts": facts,
        "anchors": {
            "llama_ext_set": find_line(ext_h, "llama_set_embeddings_nextn"),
            "llama_ext_get_ith": find_line(ext_h, "llama_get_embeddings_nextn_ith"),
            "context_embd_nextn": find_line(context_h, "buffer_view<float> embd_nextn"),
            "context_get_ith_impl": find_line(
                context_cpp,
                "float * llama_get_embeddings_nextn_ith",
            ),
        },
    }


def audit_qwen35moe_h_nextn(text: str) -> dict[str, Any]:
    trunk_region = slice_region(text, "// post-norm hidden state", "// LM head")
    facts = {
        "trunk_builds_output_norm_before_h_nextn": has(
            trunk_region,
            "cur = build_norm(cur, model.output_norm",
        ),
        "trunk_labels_h_nextn": has(trunk_region, 'cb(cur, "h_nextn", -1)'),
        "trunk_sets_t_h_nextn": has(trunk_region, "res->t_h_nextn = cur"),
        "mtp_graph_also_sets_h_nextn": has(text, "res->t_h_nextn= cur")
        or has(text, "res->t_h_nextn = cur"),
    }
    return {
        "ready": all(
            facts[key]
            for key in (
                "trunk_builds_output_norm_before_h_nextn",
                "trunk_labels_h_nextn",
                "trunk_sets_t_h_nextn",
            )
        ),
        "facts": facts,
        "anchors": {
            "trunk_output_norm": find_line(
                text,
                "cur = build_norm(cur, model.output_norm",
            ),
            "trunk_h_nextn_label": find_line(text, 'cb(cur, "h_nextn", -1)'),
            "trunk_t_h_nextn": find_line(text, "res->t_h_nextn = cur"),
        },
    }


def audit_llamacpp_context_extraction(text: str) -> dict[str, Any]:
    facts = {
        "encode_reads_t_h_nextn_when_enabled": has(
            text,
            "cparams.embeddings_nextn ? res->get_h_nextn() : nullptr",
        ),
        "decode_reads_t_h_nextn_when_enabled": has(
            text,
            "cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr",
        ),
        "copies_t_h_nextn_to_embd_nextn": has(
            text,
            "ggml_backend_tensor_get_async(backend_h, t_h_nextn",
        ),
        "unmasked_get_ith_uses_raw_position": has(
            text,
            "unmasked: nextn rows are stored densely",
        )
        and has(text, "return embd_nextn.data + (size_t) i * n_embd"),
        "unmasked_buffer_sized_by_batch": has(
            text,
            "embd_nextn.size = (size_t) n_embd_out * n_batch",
        ),
    }
    return {
        "ready": all(facts.values()),
        "facts": facts,
        "anchors": {
            "encode_t_h_nextn": find_line(
                text,
                "cparams.embeddings_nextn ? res->get_h_nextn() : nullptr",
            ),
            "copy_async": find_line(
                text,
                "ggml_backend_tensor_get_async(backend_h, t_h_nextn",
            ),
            "unmasked_get_ith": find_line(
                text,
                "unmasked: nextn rows are stored densely",
            ),
            "unmasked_size": find_line(
                text,
                "embd_nextn.size = (size_t) n_embd_out * n_batch",
            ),
        },
    }


def audit_hipengine_fp32_seed_api(text: str) -> dict[str, Any]:
    facts = {
        "prefill_accepts_capture_hidden_seed_fp32": has(
            text,
            "capture_hidden_seed_fp32: bool = False",
        )
        and has(text, "def prefill("),
        "step_accepts_capture_hidden_seed_fp32": has(text, "def step(")
        and has(text, "capture_hidden_seed_fp32: bool = False"),
        "fp32_seed_ptr_guard_exists": has(text, "def fp32_hidden_seed_ptr")
        and has(text, "ready_for_mtp"),
        "mtp_draft_seed_uses_fp32_ptr": has(text, "def mtp_draft_seed")
        and has(text, "hidden_ptr=self.fp32_hidden_seed_ptr()"),
        "output_norm_can_capture_fp32": has(text, "gguf_rmsnorm_bf16_f32_weight_out_f32"),
    }
    return {
        "ready": all(facts.values()),
        "facts": facts,
        "anchors": {
            "prefill_capture_arg": find_line(text, "capture_hidden_seed_fp32: bool = False"),
            "fp32_seed_ptr": find_line(text, "def fp32_hidden_seed_ptr"),
            "mtp_draft_seed": find_line(text, "def mtp_draft_seed"),
            "output_norm_out_f32": find_line(text, "gguf_rmsnorm_bf16_f32_weight_out_f32"),
        },
    }


def audit_doc_contract(text: str) -> dict[str, Any]:
    facts = {
        "requires_post_output_norm": has(text, "POST output-norm hidden"),
        "requires_fp32": has(text, "GGML_TYPE_F32"),
        "mentions_embd_nextn_copy": has(text, "ggml_backend_tensor_get_async"),
    }
    return {
        "ready": all(facts.values()),
        "facts": facts,
        "anchors": {
            "post_output_norm": find_line(text, "POST output-norm hidden"),
            "ggml_type_f32": find_line(text, "GGML_TYPE_F32"),
            "backend_tensor_get_async": find_line(text, "ggml_backend_tensor_get_async"),
        },
    }


def build_comparison_contract(
    *, decision: Mapping[str, Any], prompt_tokens: tuple[int, ...], position: int
) -> dict[str, Any]:
    token_id = prompt_tokens[int(position)]
    numeric = decision.get("numeric_evidence") or {}
    return {
        "prompt_tokens": list(prompt_tokens),
        "position": int(position),
        "token_id_at_position": int(token_id),
        "hidden_width": 2048,
        "llamacpp_capture_api": (
            "llama_set_embeddings_nextn(ctx, true, false) + "
            "llama_get_embeddings_nextn_ith(ctx, position)"
        ),
        "llamacpp_row_semantics": (
            "unmasked embeddings_nextn row indexed by raw prompt token position"
        ),
        "hipengine_capture_api": (
            "prefill(prompt_tokens, capture_hidden_seed_fp32=True) then "
            "fp32_hidden_seed_ptr/mtp_draft_seed"
        ),
        "expected_known_layer0_embedding_rmse_if_bf16_seed_leaks": numeric.get(
            "earliest_layer0_rmse"
        ),
        "must_compare": [
            "sha256 of float32 row",
            "shape == hidden_width",
            "max_abs_diff / mean_abs_diff / RMSE",
            "top_abs_diff indices",
        ],
    }


def build_harness_plan(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_template": "extend hidden-in capture harness with embeddings_nextn mode",
        "required_includes": ["llama.h", "llama-ext.h"],
        "required_symbols": [
            "llama_set_embeddings_nextn",
            "llama_get_embeddings_nextn_ith",
        ],
        "runtime_steps": [
            "load model and create llama_context",
            "call llama_set_embeddings_nextn(ctx, true, false) before decode",
            "decode the exact prompt-token sequence",
            "read float* row = llama_get_embeddings_nextn_ith(ctx, position)",
            "write row as little-endian float32 plus metadata",
            "capture hipEngine fp32 seed for the same prompt with capture_hidden_seed_fp32=True",
            "compare llama.cpp row to hipEngine seed row numerically",
        ],
        "output_prefix_hint": f"/tmp/hipengine-llamacpp-mtp-hidden-seed/pos{contract['position']}",
    }


def decide_readiness(
    *,
    llama_api: Mapping[str, Any],
    qwen_graph: Mapping[str, Any],
    context_extract: Mapping[str, Any],
    hipengine_api: Mapping[str, Any],
    doc_contract: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, str]:
    required_ready = all(
        bool(item.get("ready"))
        for item in (llama_api, qwen_graph, context_extract, hipengine_api, doc_contract)
    )
    prior_decision_ok = decision.get("conclusion") == (
        "fp32_seed_target_exists_but_activation_lane_is_bf16"
    )
    if required_ready and prior_decision_ok:
        return {
            "status": "ready",
            "conclusion": "llamacpp_nextn_embedding_oracle_capture_ready",
            "next_action": "compile_and_run_llamacpp_nextn_hidden_seed_capture_harness",
        }
    return {
        "status": "blocked",
        "conclusion": "hidden_seed_oracle_capture_plan_missing_required_fact",
        "next_action": "inspect_missing_oracle_plan_facts",
    }


def parse_prompt_tokens(text: str) -> tuple[int, ...]:
    tokens = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not tokens:
        raise ValueError("prompt token list is empty")
    return tokens


def slice_region(text: str, start_needle: str, end_needle: str) -> str:
    start = text.find(start_needle)
    if start < 0:
        return ""
    end = text.find(end_needle, start)
    return text[start : end if end >= 0 else len(text)]


def has(text: str, needle: str) -> bool:
    return needle in text


def find_line(text: str, needle: str) -> int | None:
    index = text.find(needle)
    if index < 0:
        return None
    return text.count("\n", 0, index) + 1


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
