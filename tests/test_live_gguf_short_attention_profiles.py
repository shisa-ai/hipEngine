"""Public strict attention must not inherit the production thread override."""
import os
from pathlib import Path

import pytest

from tests._rocm_guard import hip_runtime_available
from hipengine.speculative import native_cycle_graph  # noqa: F401


def test_strict_public_long_output_matches_mtp_with_hostile_thread_environment(monkeypatch):
    if not hip_runtime_available():
        pytest.skip("HIP runtime unavailable")
    model = os.environ.get("HIPENGINE_TEST_QWEN38_GGUF")
    if not model or not Path(model).is_file():
        pytest.skip("set HIPENGINE_TEST_QWEN38_GGUF")
    from hipengine import LLM, SamplingParams
    from hipengine.kernels.registry import resolve
    from scripts.gguf_mtp_category_bench import load_prompt_rows

    rows = load_prompt_rows(Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"))
    prompt = next(row["prompt"] for row in rows if row["id"] == "code_merge_intervals")
    monkeypatch.setenv("HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS", "1024")
    llm = LLM(model, backend="hip_gfx1151", execution_profile="strict",
              max_sequence_length=2048, max_active_requests=1,
              prefix_cache="off", speculative_candidate_budget=3)
    params = SamplingParams(temperature=0, max_tokens=273, ignore_eos=True)
    try:
        ar = llm.generate_detailed([prompt], params)[0]
        mtp = llm.generate_speculative_mtp_detailed([prompt], params)[0]
        assert len(ar.generated_token_ids) == len(mtp.generated_token_ids) == params.max_tokens
        assert ar.generated_token_ids == mtp.generated_token_ids
        assert mtp.telemetry.diagnostics["specdec2_mtp2"]["cycles"] > 0
        runner = llm._get_text_generator()._get_shared_runner()
        assert runner.target_short_c1_attn_variant == "bf16_context_batch_c1_exact_spans"
        assert runner._gguf_full_attn_decode_short_batch_fn_cache is resolve(
            backend="hip_gfx1151", layer="paged_attn_decode", quant="w4_paro",
            variant="bf16_context_batch_c1_exact_spans",
        )
        assert os.environ["HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS"] == "1024"
    finally:
        llm.close()
