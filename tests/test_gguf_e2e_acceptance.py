from __future__ import annotations

import json
from pathlib import Path

import pytest


QWEN35MOE_FIXTURE = Path("tests/fixtures/gguf/qwen36_35b_a3b_q4km_smoke.json")
QWEN35MOE_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
QWEN35MOE_Q3_FIXTURE = Path("tests/fixtures/gguf/qwen36_35b_a3b_ud_q3km_smoke.json")
QWEN35MOE_Q3_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf")

FIXTURES = {
    "gguf_q4_k_m": (
        Path("tests/fixtures/gguf/qwen35_0_8b_q4_k_m_e2e.json"),
        "/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf",
        " 1.\n\n",
        [220, 16, 13, 271],
        {
            "gguf_q4_k",
            "gguf_q5_k_dense_bf16_fallback",
            "gguf_q6_k",
            "gguf_q6_k_dense_bf16_fallback",
            "gguf_q8_0",
        },
    ),
    "gguf_q8_0": (
        Path("tests/fixtures/gguf/qwen35_0_8b_q8_0_e2e.json"),
        "/models/gguf/Qwen3.5-0.8B-Q8_0.gguf",
        " 1.\n",
        [220, 16, 13, 198],
        {"gguf_q8_0"},
    ),
    "gguf_q4_1": (
        Path("tests/fixtures/gguf/qwen35_0_8b_q4_1_e2e.json"),
        "/models/gguf/Qwen3.5-0.8B-Q4_1.gguf",
        " 1.\n",
        [220, 16, 13, 198],
        {
            "gguf_q4_1_dense_bf16_fallback",
            "gguf_q5_k_dense_bf16_fallback",
            "gguf_q6_k",
            "gguf_q6_k_dense_bf16_fallback",
            "gguf_q8_0",
        },
    ),
    "gguf_ud_q4_k_xl": (
        Path("tests/fixtures/gguf/qwen35_0_8b_ud_q4_k_xl_e2e.json"),
        "/models/gguf/Qwen3.5-0.8B-UD-Q4_K_XL.gguf",
        " 1.\n",
        [220, 16, 13, 198],
        {
            "fp16_dense_bf16_fallback",
            "gguf_iq4_xs_dense_bf16_fallback",
            "gguf_q4_k",
            "gguf_q5_k_dense_bf16_fallback",
            "gguf_q6_k",
            "gguf_q6_k_dense_bf16_fallback",
            "gguf_q8_0",
        },
    ),
}


@pytest.mark.parametrize("quant", sorted(FIXTURES))
def test_qwen35_gguf_e2e_fixture_declares_public_api_gate(quant: str) -> None:
    fixture_path, model_path, expected_text, expected_ids, required_kernels = FIXTURES[quant]
    fixture = json.loads(fixture_path.read_text())

    assert fixture["model"]["path"] == model_path
    assert fixture["model"]["quant"] == quant
    assert fixture["prompt"] == "The answer is"
    assert fixture["prompt_ids"] == [760, 4087, 369]
    assert fixture["sampling"] == {
        "max_new_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": False,
    }
    assert fixture["expected_generated_text"] == expected_text
    assert fixture["expected_generated_token_ids"] == expected_ids

    acceptance = fixture["acceptance"]
    assert acceptance["public_api"] == "hipengine.LLM.generate"
    assert acceptance["backend"] == "hip_gfx1100"
    assert acceptance["quant"] == quant
    assert acceptance["torch_hot_path_allowed"] is False
    assert acceptance["deterministic_required"] is True
    assert acceptance["expected_text_match_required"] is True
    assert acceptance["expected_token_ids_match_required"] is True
    assert acceptance["finite_logits_required"] is True
    assert set(acceptance["required_kernel_families"]) == required_kernels


def test_qwen35moe_gguf_e2e_fixture_declares_public_api_gate() -> None:
    fixture = json.loads(QWEN35MOE_FIXTURE.read_text())

    assert fixture["schema_version"] == 1
    assert fixture["model"] == {
        "path": str(QWEN35MOE_MODEL),
        "quant": "gguf_q4_k_m",
        "architecture": "qwen35moe",
    }
    assert fixture["prompt"] == "Hello"
    assert fixture["prompt_ids"] == [9419]
    assert fixture["sampling"] == {
        "max_new_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
    }
    assert fixture["expected_generated_text"] == "izio."
    assert fixture["expected_generated_token_ids"] == [43482, 13]

    acceptance = fixture["acceptance"]
    assert acceptance["public_api"] == "hipengine.LLM.generate"
    assert acceptance["backend"] == "hip_gfx1100"
    assert acceptance["quant"] == "gguf_q4_k_m"
    assert acceptance["torch_hot_path_allowed"] is False
    assert acceptance["deterministic_required"] is True
    assert acceptance["expected_text_match_required"] is True
    assert acceptance["expected_token_ids_match_required"] is True
    assert acceptance["finite_logits_required"] is True
    assert acceptance["external_token_oracle"] == "llama-tokenize"
    assert set(acceptance["required_kernel_families"]) == {
        "gguf_q4_k",
        "gguf_q5_k",
        "gguf_q6_k",
        "gguf_q8_0",
        "qwen35_router",
        "paro_combine",
    }


def test_qwen35moe_ud_q3km_e2e_fixture_pins_llamacpp_first_token() -> None:
    fixture = json.loads(QWEN35MOE_Q3_FIXTURE.read_text())

    assert fixture["schema_version"] == 1
    assert fixture["model"] == {
        "path": str(QWEN35MOE_Q3_MODEL),
        "file_size_bytes": 17104402720,
        "sha256": "8966dd0cd8c543c4228490a2a8b0e0814fc4f1e6a8e199ceed4de6754ae7b8e1",
        "quant": "gguf_ud_q3_k_m",
        "architecture": "qwen35moe",
    }
    assert fixture["oracle"]["commit"] == "1ebf790cda38d827559548f67b0469189690cc8c"
    assert fixture["oracle"]["scope"] == "first_generated_token"
    assert fixture["prompt"] == "Hello"
    assert fixture["prompt_ids"] == [9419]
    assert fixture["sampling"] == {
        "max_new_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
    }
    assert fixture["expected_generated_text"] == ","
    assert fixture["expected_generated_token_ids"] == [11]
    assert fixture["decode_graph"] == {
        "oracle": "resident eager direct decode",
        "max_new_tokens": 3,
        "expected_generated_text": ",, a",
        "expected_generated_token_ids": [11, 11, 264],
    }

    acceptance = fixture["acceptance"]
    assert acceptance["public_api"] == "hipengine.LLM.generate"
    assert acceptance["backend"] == "hip_gfx1100"
    assert acceptance["quant"] == "gguf_ud_q3_k_m"
    assert acceptance["torch_hot_path_allowed"] is False
    assert acceptance["deterministic_required"] is True
    assert acceptance["expected_text_match_required"] is True
    assert acceptance["expected_token_ids_match_required"] is True
    assert acceptance["finite_logits_required"] is True
    assert acceptance["external_token_oracle"] == "llama.cpp + llama-tokenize"
    assert acceptance["bulk_prefill_attention_mode"] == "native"
    assert set(acceptance["required_kernel_families"]) == {
        "gguf_iq3_xxs",
        "gguf_iq4_xs",
        "gguf_q6_k",
        "gguf_q8_0",
        "qwen35_router",
        "paro_combine",
    }
