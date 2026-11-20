"""Public MTP with a tokenizer-compatible map across weight quantizations."""
from pathlib import Path

import pytest

from tests._rocm_guard import hip_runtime_available


@pytest.mark.parametrize("filename", ["Qwen3.8-27B-Q4_K_S.gguf", "Qwen3.8-27B-UD-Q4_K_M.gguf"])
def test_public_mtp_uses_tokenizer_compatible_hot_vocab(filename, hip_test_target_arch):
    if not hip_runtime_available() or hip_test_target_arch != "gfx1151":
        pytest.skip("requires gfx1151 HIP runtime")
    model = Path("/models/gguf") / filename
    if not model.is_file():
        pytest.skip(f"missing {model}")
    from hipengine import LLM, SamplingParams

    llm = LLM(str(model), backend="hip_gfx1151", execution_profile="strict",
              max_sequence_length=512, max_active_requests=1,
              speculative_candidate_budget=3, mtp_hot_vocab="auto")
    try:
        params = SamplingParams(temperature=0, max_tokens=17, ignore_eos=True)
        for prompt in ("Explain why the sky is blue in one sentence.", "日本語で猫について教えてください。"):
            ar = llm.generate_detailed([prompt], params)[0]
            mtp = llm.generate_speculative_mtp_detailed([prompt], params)[0]
            assert len(mtp.generated_token_ids) == params.max_tokens
            assert mtp.generated_token_ids == ar.generated_token_ids
            owner = llm._get_text_generator()
            weights = tuple(owner._shared_mtp_nextn_weights.values())
            assert weights and any(w.hot_vocab is not None for w in weights)
    finally:
        llm.close()
