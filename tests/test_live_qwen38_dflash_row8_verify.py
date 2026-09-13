"""Current DFlash-shaped R8 verifier regression for the historical AR divergence."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
PROMPTS = Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl")
BLOCK_INPUTS = (248068, 198, 760, 1156, 6587, 264, 12654, 43072)
TAP_LAYERS = (0, 15, 31, 47, 63)
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"missing {MODEL}")


def _render_lru_prompt() -> str:
    payload = next(
        json.loads(line)
        for line in PROMPTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("id") == "code_lru_cache"
    )
    rendered = [
        f"<|im_start|>{'system' if row['role'] == 'developer' else row['role']}\n"
        f"{row['content']}<|im_end|>"
        for row in payload["messages"]
    ]
    rendered.append("<|im_start|>assistant\n")
    return "\n".join(rendered)


def test_qwen38_dflash_row8_verify_and_selected_commit_match_serial_ar(
    monkeypatch: pytest.MonkeyPatch,
    hip_test_target_arch: str,
) -> None:
    if hip_test_target_arch != "gfx1151":
        pytest.skip("Qwen3.8 row8 regression is gfx1151-only")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is unavailable")
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")

    from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
        register_qwen35_paged_attn_decode_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading import load_gguf_index
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    register_qwen35_paged_attn_decode_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(MODEL))
    prompt_ids = tuple(int(token) for token in tokenizer.encode(_render_lru_prompt()))

    with Qwen35GGUFResidentSession(
        MODEL,
        backend="hip_gfx1151",
        max_sequence_length=128,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as serial:
        native = Qwen35GGUFResidentSession(
            MODEL,
            backend="hip_gfx1151",
            shared_runner=serial.runner,
            max_sequence_length=128,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        )
        try:
            serial_root = serial.prefill(prompt_ids, use_bulk=True, return_logits=False)
            native_root = native.prefill(prompt_ids, use_bulk=True, return_logits=False)
            assert int(serial_root.token_id) == int(native_root.token_id) == BLOCK_INPUTS[0]

            serial_result = serial.verify_target_block_serial_exact(
                BLOCK_INPUTS,
                capture_linear_state_rows=True,
                capture_pre_output_norm_hidden=True,
                capture_layer_output_hidden=TAP_LAYERS,
            )
            native_result = native.verify_target_block(
                BLOCK_INPUTS,
                bulk_attention_mode="native",
                use_wmma_prefill=False,
                capture_linear_state_rows=True,
                capture_pre_output_norm_hidden=True,
                capture_layer_output_hidden=TAP_LAYERS,
                defer_linear_state_commit=True,
            )

            assert native_result.token_ids == serial_result.token_ids
            np.testing.assert_array_equal(native_result.hidden_seeds, serial_result.hidden_seeds)
            np.testing.assert_array_equal(
                native_result.pre_output_norm_hidden,
                serial_result.pre_output_norm_hidden,
            )
            assert native_result.layer_output_hidden is not None
            assert serial_result.layer_output_hidden is not None
            for layer_id in TAP_LAYERS:
                np.testing.assert_array_equal(
                    native_result.layer_output_hidden[layer_id],
                    serial_result.layer_output_hidden[layer_id],
                )

            native._commit_verify_linear_state_row(
                7,
                position=int(native_result.start_position) + 8,
            )
            serial_next = serial.step(int(serial_result.token_ids[-1]), return_logits=False)
            native_next = native.step(int(native_result.token_ids[-1]), return_logits=False)
            assert int(native_next.token_id) == int(serial_next.token_id) == 1714
        finally:
            native.close()
