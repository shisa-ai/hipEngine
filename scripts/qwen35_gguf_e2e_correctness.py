#!/usr/bin/env python3
"""True GGUF LLM.generate() E2E correctness gate.

This script is intentionally a public-API gate: it calls ``hipengine.LLM.generate``
with a local Qwen3.5-0.8B GGUF fixture and compares the generated completion
against the fixture text/token oracle. Passing the Q4_K_M, Q8_0, Q4_1, and
UD-Q4_K_XL fixtures is the hard acceptance gate for local GGUF E2E quant
coverage.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def _tokenize_completion(*, model: Path, text: str, llama_tokenize: Path) -> list[int]:
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture = _load_fixture(args.fixture)
    model_path = Path(args.model or fixture["model"]["path"])
    acceptance = fixture["acceptance"]
    sampling = fixture["sampling"]
    quant = args.quant or acceptance["quant"]
    backend = args.backend or acceptance["backend"]
    repeat = int(args.repeat or acceptance.get("repeat", 2))
    max_tokens = int(args.max_new_tokens or sampling["max_new_tokens"])

    torch_preloaded = "torch" in sys.modules
    from hipengine import LLM, SamplingParams

    outputs: list[str] = []
    errors: list[str] = []
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
            errors.append(f"{type(exc).__name__}: {exc}")
            break

    expected_text = str(fixture["expected_generated_text"])
    deterministic = bool(outputs) and all(output == outputs[0] for output in outputs)
    expected_text_match = bool(outputs) and all(output == expected_text for output in outputs)
    torch_loaded_by_generate = (not torch_preloaded) and "torch" in sys.modules

    token_ids: list[int] | None = None
    expected_token_ids = [int(item) for item in fixture["expected_generated_token_ids"]]
    token_ids_match = False
    tokenization_error = None
    if outputs and not args.skip_tokenize_check:
        try:
            token_ids = _tokenize_completion(
                model=model_path,
                text=outputs[0],
                llama_tokenize=args.llama_tokenize,
            )
            token_ids_match = token_ids == expected_token_ids
        except Exception as exc:
            tokenization_error = f"{type(exc).__name__}: {exc}"
    elif outputs:
        token_ids_match = True

    passed = (
        not errors
        and deterministic
        and expected_text_match
        and token_ids_match
        and not torch_loaded_by_generate
    )
    return {
        "schema": 1,
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
        "tokenization_error": tokenization_error,
        "torch_preloaded": torch_preloaded,
        "torch_loaded_by_generate": torch_loaded_by_generate,
        "errors": errors,
        "passed": passed,
        "notes": [
            "Hard gate for GGUF E2E: this must call hipengine.LLM.generate(), not a lower-level runner.",
            "Passing requires deterministic repeated generation, expected text/token match, "
            "and no torch import by the generate path.",
            "Finite-logit and kernel-trace evidence are recorded by the GGUF runner/profile "
            "tasks once the public API path is wired.",
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
