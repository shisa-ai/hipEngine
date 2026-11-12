"""Real LLM requests for target-session reuse (explicit target and drafter paths)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests._rocm_guard import hip_runtime_available
# Register at collection so pytest's per-test registry restore retains the
# lazily imported graph launcher for both parametrized provider requests.
from hipengine.speculative import native_cycle_graph  # noqa: F401


@pytest.mark.parametrize("method", ["eagle4", "dspine"])
def test_public_feature_provider_reuses_target_and_matches_ar(method):
    if not hip_runtime_available():
        pytest.skip("HIP runtime unavailable")
    model = os.environ.get("HIPENGINE_TEST_QWEN38_GGUF")
    drafter_env = {
        "eagle4": "HIPENGINE_TEST_EAGLE4_DRAFTER",
        "dspine": "HIPENGINE_TEST_DSPINE_DRAFTER",
    }[method]
    drafter = os.environ.get(drafter_env)
    if not model or not drafter or not Path(model).is_file() or not Path(drafter).is_dir():
        pytest.skip(f"set HIPENGINE_TEST_QWEN38_GGUF and {drafter_env}")
    from hipengine import LLM, SamplingParams

    llm = LLM(model, backend="hip_gfx1151", speculative_provider=method, draft_model=drafter,
              max_sequence_length=256, max_active_requests=1)
    params = SamplingParams(temperature=0, max_tokens=17, ignore_eos=True)
    try:
        for i, prompt in enumerate((
            "Explain why the sky is blue in one sentence.",
            "日本語で短く、猫について教えてください。",
        )):
            ar = llm.generate_detailed([prompt], params)[0]
            output = llm.generate_speculative_detailed([prompt], params)[0]
            assert len(output.generated_token_ids) == params.max_tokens
            assert output.generated_token_ids == ar.generated_token_ids
            assert output.telemetry.decode_state.execution_path.startswith(f"qwen35_gguf_{method}_")
            diag = output.telemetry.diagnostics
            assert diag["target_session_reused"] is (i > 0)
            assert diag["graph_verify_cycles"] == diag["draft_forwards"] > 0
    finally:
        llm.close()
