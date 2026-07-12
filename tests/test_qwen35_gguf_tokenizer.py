from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFReader
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.compare_prompt_token_inventories import compare_prompt_token_inventories
from scripts.gguf_prompt_token_inventory import (
    build_prompt_token_inventory,
    load_prompt_suite,
    select_prompts,
    sha256_token_ids,
)

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
MOE_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
D32_PROMPTS = Path("benchmarks/fixtures/llamacpp_mtp_bench_prompts.json")
LLAMACPP_HIP_D32_TOKEN_FIXTURE = Path(
    "benchmarks/fixtures/llamacpp_hip_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json"
)


def _tokenizer() -> Qwen35GGUFTokenizer:
    if not MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MODEL}")
    return Qwen35GGUFTokenizer.from_gguf_info(GGUFReader(MODEL).info)


def test_qwen35_gguf_tokenizer_matches_e2e_fixture() -> None:
    tokenizer = _tokenizer()

    assert tokenizer.encode("The answer is") == [760, 4087, 369]
    assert tokenizer.decode([220, 16, 13, 271]) == " 1.\n\n"
    assert tokenizer.encode(" 1.\n\n") == [220, 16, 13, 271]
    assert tokenizer.decode([760, 4087, 369, 220, 16, 13, 271]) == "The answer is 1.\n\n"


def test_qwen35moe_gguf_tokenizer_matches_smoke_fixture() -> None:
    if not MOE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MOE_MODEL}")
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(GGUFReader(MOE_MODEL).info)

    assert tokenizer.encode("Hello") == [9419]
    assert tokenizer.decode([9419]) == "Hello"
    assert tokenizer.encode("izio.") == [43482, 13]
    assert tokenizer.decode([43482, 13]) == "izio."
    assert tokenizer.decode([9419, 43482, 13]) == "Helloizio."


def test_qwen35_gguf_tokenizer_round_trips_common_ascii_prompts() -> None:
    tokenizer = _tokenizer()

    examples = [
        "Hello",
        "AMD GPUs are",
        "The answer is 1.",
        "line one\nline two",
    ]
    for text in examples:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_gguf_prompt_token_inventory_records_raw_token_ids() -> None:
    tokenizer = _tokenizer()
    suite = {
        "schema": 1,
        "prompts": [
            {"name": "answer", "prompt": "The answer is"},
            {"name": "skip", "prompt": "not selected"},
        ],
    }

    prompts = select_prompts(suite, names_csv="answer")
    inventory = build_prompt_token_inventory(
        tokenizer=tokenizer,
        prompts=prompts,
        model=MODEL,
        prompts_file="synthetic-prompts.json",
        tokenizer_model="gpt2",
        tokenizer_pre="qwen35",
    )

    assert inventory["kind"] == "hipengine_gguf_prompt_token_inventory"
    assert inventory["prompt_render"] == "raw"
    assert inventory["tokenization"] == "hipengine.gguf.qwen35.byte_bpe_approx"
    assert inventory["warning"].startswith("This records hipEngine GGUF tokenizer output only")
    row = inventory["prompts"][0]
    assert row["name"] == "answer"
    assert row["token_ids"] == [760, 4087, 369]
    assert row["token_ids_sha256"] == sha256_token_ids([760, 4087, 369])
    assert row["roundtrip_ok"]


def test_qwen35_gguf_tokenizer_decodes_special_tokens() -> None:
    tokenizer = _tokenizer()

    assert tokenizer.decode([248046]) == "<|im_end|>"
    assert tokenizer.decode([248046], skip_special=True) == ""
    assert tokenizer.eos_token_id == 248046
    assert tokenizer.padding_token_id == 248055


def test_qwen35moe_gguf_tokenizer_matches_committed_llamacpp_d32_fixture() -> None:
    if not MOE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MOE_MODEL}")
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(GGUFReader(MOE_MODEL).info)
    suite = load_prompt_suite(D32_PROMPTS)
    prompts = select_prompts(suite)
    metadata = GGUFReader(MOE_MODEL).info.metadata
    inventory = build_prompt_token_inventory(
        tokenizer=tokenizer,
        prompts=prompts,
        model=MOE_MODEL,
        prompts_file=D32_PROMPTS,
        tokenizer_model=str(metadata.get("tokenizer.ggml.model")),
        tokenizer_pre=str(metadata.get("tokenizer.ggml.pre")),
    )
    llama_inventory = json.loads(LLAMACPP_HIP_D32_TOKEN_FIXTURE.read_text())

    comparison = compare_prompt_token_inventories(
        inventory,
        llama_inventory,
        left_label="hipengine",
        right_label="llamacpp",
    )

    assert comparison["all_match"]
    assert comparison["compared_prompts"] == 9
    assert comparison["mismatches"] == []


def test_qwen35_gguf_tokenizer_does_not_import_torch() -> None:
    torch_preloaded = "torch" in sys.modules
    tokenizer = _tokenizer()

    assert tokenizer.encode("The answer is") == [760, 4087, 369]
    assert ("torch" in sys.modules) is torch_preloaded
