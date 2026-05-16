from __future__ import annotations

import json
from pathlib import Path


FIXTURE = Path("tests/fixtures/gguf/qwen35_0_8b_q4_k_m_e2e.json")


def test_qwen35_gguf_e2e_fixture_declares_public_api_gate() -> None:
    fixture = json.loads(FIXTURE.read_text())

    assert fixture["model"]["path"] == "/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"
    assert fixture["model"]["quant"] == "gguf_q4_k_m"
    assert fixture["prompt"] == "The answer is"
    assert fixture["prompt_ids"] == [760, 4087, 369]
    assert fixture["sampling"] == {
        "max_new_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": False,
    }
    assert fixture["expected_generated_text"] == " 1.\n\n"
    assert fixture["expected_generated_token_ids"] == [220, 16, 13, 271]

    acceptance = fixture["acceptance"]
    assert acceptance["public_api"] == "hipengine.LLM.generate"
    assert acceptance["backend"] == "hip_gfx1100"
    assert acceptance["quant"] == "gguf_q4_k_m"
    assert acceptance["torch_hot_path_allowed"] is False
    assert acceptance["deterministic_required"] is True
    assert acceptance["expected_text_match_required"] is True
    assert acceptance["expected_token_ids_match_required"] is True
    assert set(acceptance["required_kernel_families"]) == {
        "gguf_q4_k",
        "gguf_q5_k",
        "gguf_q6_k",
        "gguf_q8_0",
    }
