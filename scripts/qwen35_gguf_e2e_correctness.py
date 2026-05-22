#!/usr/bin/env python3
"""True GGUF LLM.generate() E2E correctness gate.

This script is intentionally a public-API gate: it calls ``hipengine.LLM.generate``
with local GGUF fixtures and compares the generated completion against fixture
text/token oracles.  It also records GGUF intake, tokenizer, optional external
``llama-tokenize`` oracle, and finite-logit evidence so a single compact JSON
artifact can serve as the correctness handoff for dense Qwen3.5 and qwen35moe
GGUF bring-up.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FIXTURE = REPO_ROOT / "tests/fixtures/gguf/qwen35_0_8b_q4_k_m_e2e.json"
DEFAULT_LLAMA_TOKENIZE = Path("/home/lhl/llama.cpp/llama.cpp-hip-therock/build/bin/llama-tokenize")


def _load_fixture(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text())
    required = {
        "model",
        "prompt",
        "prompt_ids",
        "sampling",
        "expected_generated_text",
        "expected_generated_token_ids",
        "acceptance",
    }
    missing = sorted(required - set(fixture))
    if missing:
        raise ValueError(f"fixture {path} missing required keys: {', '.join(missing)}")
    return fixture


def _tokenize_text(*, model: Path, text: str, llama_tokenize: Path) -> list[int]:
    if not llama_tokenize.is_file():
        raise FileNotFoundError(f"llama-tokenize binary not found: {llama_tokenize}")
    completed = subprocess.run(
        [
            str(llama_tokenize),
            "-m",
            str(model),
            "-p",
            text,
            "--ids",
            "--log-disable",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    parsed = ast.literal_eval(completed.stdout.strip())
    if not isinstance(parsed, list) or not all(isinstance(item, int) for item in parsed):
        raise ValueError(f"unexpected llama-tokenize output: {completed.stdout!r}")
    return parsed


def _model_intake(model_path: Path) -> dict[str, Any]:
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
    from hipengine.quant.gguf import dequantization_supported

    info = GGUFReader(model_path).info
    model_map = build_qwen35_gguf_tensor_map(info)
    type_counts = Counter(tensor.ggml_type_name for tensor in info.tensors)
    unsupported_dequant = sorted(
        {tensor.ggml_type_name for tensor in info.tensors if not dequantization_supported(tensor.ggml_type)}
    )
    return {
        "path": str(info.path),
        "version": info.version,
        "architecture": info.architecture,
        "file_type": info.file_type,
        "file_type_name": info.file_type_name,
        "tensor_count": info.tensor_count,
        "total_tensor_nbytes": info.total_tensor_nbytes,
        "tensor_count_by_type": dict(sorted(type_counts.items())),
        "unsupported_dequant_types": unsupported_dequant,
        "map_validation_passed": model_map.validation.passed,
        "map_missing": model_map.validation.missing,
        "map_unexpected": model_map.validation.unexpected,
        "root_lm_head": model_map.root("lm_head").name,
        "config": {
            "architecture": model_map.config.architecture,
            "block_count": model_map.config.block_count,
            "hidden_size": model_map.config.hidden_size,
            "vocab_size": model_map.config.vocab_size,
            "context_length": model_map.config.context_length,
            "full_attention_interval": model_map.config.full_attention_interval,
            "expert_count": model_map.config.expert_count,
            "expert_used_count": model_map.config.expert_used_count,
            "expert_feed_forward_length": model_map.config.expert_feed_forward_length,
            "expert_shared_feed_forward_length": model_map.config.expert_shared_feed_forward_length,
        },
        "passed": model_map.validation.passed and not unsupported_dequant,
    }


def _internal_tokenizer_check(model_path: Path, fixture: dict[str, Any], outputs: list[str]) -> dict[str, Any]:
    from hipengine.loading import load_gguf_index
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(model_path))
    expected_prompt_ids = [int(item) for item in fixture["prompt_ids"]]
    expected_generated_ids = [int(item) for item in fixture["expected_generated_token_ids"]]
    prompt_ids = tokenizer.encode(str(fixture["prompt"]))
    expected_text_ids = tokenizer.encode(str(fixture["expected_generated_text"]))
    output_ids = tokenizer.encode(outputs[0]) if outputs else None
    return {
        "tokenizer_model": fixture.get("model", {}).get("architecture"),
        "prompt_ids": prompt_ids,
        "prompt_ids_match": prompt_ids == expected_prompt_ids,
        "prompt_roundtrip": tokenizer.decode(prompt_ids),
        "prompt_roundtrip_match": tokenizer.decode(prompt_ids) == fixture["prompt"],
        "expected_text_ids": expected_text_ids,
        "expected_text_ids_match": expected_text_ids == expected_generated_ids,
        "output_ids": output_ids,
        "output_ids_match": bool(outputs) and output_ids == expected_generated_ids,
        "passed": (
            prompt_ids == expected_prompt_ids
            and tokenizer.decode(prompt_ids) == fixture["prompt"]
            and expected_text_ids == expected_generated_ids
            and (not outputs or output_ids == expected_generated_ids)
        ),
    }


def _external_tokenizer_check(
    *,
    model_path: Path,
    fixture: dict[str, Any],
    outputs: list[str],
    llama_tokenize: Path,
) -> dict[str, Any]:
    expected_prompt_ids = [int(item) for item in fixture["prompt_ids"]]
    expected_generated_ids = [int(item) for item in fixture["expected_generated_token_ids"]]
    prompt_ids = _tokenize_text(model=model_path, text=str(fixture["prompt"]), llama_tokenize=llama_tokenize)
    output_ids = _tokenize_text(model=model_path, text=outputs[0], llama_tokenize=llama_tokenize) if outputs else None
    return {
        "oracle": str(llama_tokenize),
        "prompt_ids": prompt_ids,
        "prompt_ids_match": prompt_ids == expected_prompt_ids,
        "generated_token_ids": output_ids,
        "expected_token_ids_match": bool(outputs) and output_ids == expected_generated_ids,
        "passed": prompt_ids == expected_prompt_ids and bool(outputs) and output_ids == expected_generated_ids,
    }


def _finite_logits_check(model_path: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    from hipengine.core.memory import memory_stats
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    prompt_ids = [int(item) for item in fixture["prompt_ids"]]
    expected_first_id = int(fixture["expected_generated_token_ids"][0])
    with Qwen35GGUFResidentSession(model_path) as session:
        result = session.prefill(prompt_ids, return_logits=True)
        logits = result.logits
        finite = bool(logits.size and np.all(np.isfinite(logits)))
        argmax_id = int(np.argmax(logits.reshape(-1))) if finite else None
        return {
            "prompt_ids": prompt_ids,
            "shape": list(logits.shape),
            "dtype": str(logits.dtype),
            "finite": finite,
            "argmax_token_id": int(result.token_id),
            "argmax_recomputed_token_id": argmax_id,
            "argmax_logit": float(result.logit),
            "expected_first_token_id": expected_first_id,
            "expected_first_token_match": int(result.token_id) == expected_first_id,
            "min_logit": float(np.min(logits)) if finite else None,
            "max_logit": float(np.max(logits)) if finite else None,
            "memory_stats": memory_stats(),
            "passed": finite and int(result.token_id) == expected_first_id and argmax_id == int(result.token_id),
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture = _load_fixture(args.fixture)
    model_path = Path(args.model or fixture["model"]["path"])
    acceptance = fixture["acceptance"]
    sampling = fixture["sampling"]
    quant = args.quant or acceptance["quant"]
    backend = args.backend or acceptance["backend"]
    repeat = int(args.repeat or acceptance.get("repeat", 2))
    max_tokens = int(args.max_new_tokens or sampling["max_new_tokens"])
    finite_logits_required = bool(args.finite_logits_check or acceptance.get("finite_logits_required", False))

    torch_preloaded = "torch" in sys.modules
    from hipengine.core.memory import reset_memory_stats

    reset_memory_stats()
    errors: list[str] = []
    intake: dict[str, Any] | None = None
    if not args.skip_intake_check:
        try:
            intake = _model_intake(model_path)
        except Exception as exc:
            errors.append(f"intake:{type(exc).__name__}: {exc}")

    from hipengine import LLM, SamplingParams

    outputs: list[str] = []
    for _ in range(repeat):
        try:
            llm = LLM(str(model_path), backend=backend, quant=quant)
            generated = llm.generate(
                fixture["prompt"],
                SamplingParams(
                    max_tokens=max_tokens,
                    temperature=float(sampling["temperature"]),
                    top_p=float(sampling["top_p"]),
                    ignore_eos=bool(sampling["ignore_eos"]),
                ),
            )
            if len(generated) != 1:
                raise RuntimeError(f"expected one generated string, got {len(generated)}")
            outputs.append(generated[0])
        except Exception as exc:  # report structured failure for the gate
            errors.append(f"generate:{type(exc).__name__}: {exc}")
            break

    expected_text = str(fixture["expected_generated_text"])
    deterministic = bool(outputs) and all(output == outputs[0] for output in outputs)
    expected_text_match = bool(outputs) and all(output == expected_text for output in outputs)
    torch_loaded_by_generate = (not torch_preloaded) and "torch" in sys.modules

    expected_token_ids = [int(item) for item in fixture["expected_generated_token_ids"]]
    internal_tokenizer = None
    try:
        internal_tokenizer = _internal_tokenizer_check(model_path, fixture, outputs)
    except Exception as exc:
        errors.append(f"internal_tokenizer:{type(exc).__name__}: {exc}")

    external_tokenizer = None
    tokenization_error = None
    if not args.skip_tokenize_check:
        try:
            external_tokenizer = _external_tokenizer_check(
                model_path=model_path,
                fixture=fixture,
                outputs=outputs,
                llama_tokenize=args.llama_tokenize,
            )
        except Exception as exc:
            tokenization_error = f"{type(exc).__name__}: {exc}"
    elif outputs:
        external_tokenizer = {
            "skipped": True,
            "expected_token_ids_match": True,
            "passed": True,
        }

    finite_logits = None
    if finite_logits_required:
        try:
            finite_logits = _finite_logits_check(model_path, fixture)
        except Exception as exc:
            errors.append(f"finite_logits:{type(exc).__name__}: {exc}")
            finite_logits = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    intake_passed = True if args.skip_intake_check else bool(intake and intake.get("passed"))
    internal_tokenizer_passed = bool(internal_tokenizer and internal_tokenizer.get("passed"))
    external_tokenizer_passed = bool(external_tokenizer and external_tokenizer.get("passed"))
    finite_logits_passed = True if not finite_logits_required else bool(finite_logits and finite_logits.get("passed"))
    token_ids = None if external_tokenizer is None else external_tokenizer.get("generated_token_ids")
    token_ids_match = external_tokenizer_passed

    passed = (
        not errors
        and deterministic
        and expected_text_match
        and internal_tokenizer_passed
        and external_tokenizer_passed
        and intake_passed
        and finite_logits_passed
        and not torch_loaded_by_generate
    )
    return {
        "schema": 2,
        "mode": "gguf_true_llm_generate_e2e_correctness",
        "model": str(model_path),
        "backend": backend,
        "quant": quant,
        "fixture": str(args.fixture),
        "prompt": fixture["prompt"],
        "prompt_ids": fixture["prompt_ids"],
        "max_new_tokens": max_tokens,
        "repeat": repeat,
        "expected_generated_text": expected_text,
        "outputs": outputs,
        "deterministic": deterministic,
        "expected_text_match": expected_text_match,
        "expected_generated_token_ids": expected_token_ids,
        "generated_token_ids": token_ids,
        "expected_token_ids_match": token_ids_match,
        "intake": intake,
        "internal_tokenizer": internal_tokenizer,
        "external_tokenizer": external_tokenizer,
        "tokenization_error": tokenization_error,
        "finite_logits_required": finite_logits_required,
        "finite_logits": finite_logits,
        "torch_preloaded": torch_preloaded,
        "torch_loaded_by_generate": torch_loaded_by_generate,
        "errors": errors,
        "passed": passed,
        "notes": [
            "Hard gate for GGUF E2E: this must call hipengine.LLM.generate(), not a lower-level runner.",
            "Passing requires deterministic repeated generation, expected text/token match, GGUF intake, "
            "internal tokenizer parity, external llama-tokenize parity unless skipped, finite logits when "
            "required by the fixture, and no torch import by the generate path.",
            "The qwen35moe 35B-A3B fixture is a narrow deterministic bring-up smoke; throughput and "
            "stronger model-quality comparisons are separate benchmark/correctness tasks.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model", default="", help="Override fixture model path")
    parser.add_argument("--backend", default="", help="Override fixture backend")
    parser.add_argument("--quant", default="", help="Override fixture quant key")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--llama-tokenize", type=Path, default=DEFAULT_LLAMA_TOKENIZE)
    parser.add_argument("--skip-tokenize-check", action="store_true")
    parser.add_argument("--skip-intake-check", action="store_true")
    parser.add_argument("--finite-logits-check", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json is not None:
        args.json.write_text(payload + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
