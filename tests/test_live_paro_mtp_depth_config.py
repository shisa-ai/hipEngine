"""User-configured PARO candidate depths reach the public generation route."""
from contextlib import closing
import os
from pathlib import Path

import pytest

from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from test_live_paro_mtp_rollback import _hip_available
from test_live_paro_mtp_graph import _routes

pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is unavailable")


@pytest.mark.parametrize("depth", [None, 1, 2, 3, 4])
def test_constructor_candidate_depth_runs_or_refuses_by_capability(depth):
    model = Path(os.environ.get(
        "HIPENGINE_TEST_PARO_MTP_MODEL",
        "/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16",
    ))
    if not model.is_dir():
        pytest.skip("packed PARO model with BF16 MTP weights is unavailable")
    before = memory_stats()
    with closing(LLM(str(model), backend="auto", quant="w4_paro",
                     max_active_requests=1, max_sequence_length=512,
                     prefix_cache="off", speculative_candidate_budget=depth)) as llm:
        prompt = [9707] * 37
        params = SamplingParams(max_tokens=17, temperature=0, ignore_eos=True,
                                speculative_mtp=True, kv_storage="bf16")
        if depth == 4:
            with pytest.raises(NotImplementedError, match="mtp_candidate_depth_unsupported"):
                llm.generate_detailed([prompt], params)
            plain = llm.generate_detailed([prompt], SamplingParams(
                max_tokens=3, temperature=0, ignore_eos=True, kv_storage="bf16",
            ))[0]
            assert len(plain.generated_token_ids) == 3
        else:
            reference = None
            for _ in range(2):
                result = llm.generate_detailed([prompt], params)[0]
                assert len(result.generated_token_ids) == 17
                routes = _routes(llm.live_loop_snapshot())
                route = routes["recent_completed"][-1]
                assert route["specdec2_mtp2_used"]
                assert max(route["specdec2_mtp2_candidate_counts"]) == (2 if depth is None else depth)
                assert set(route["specdec2_mtp2_execution_routes"]) == {"eager"}
                assert not any(k.startswith("specdec2_mtp2_recovered")
                               for k in routes["fallback_reasons"])
                value = (result.generated_token_ids,
                         route["specdec2_mtp2_candidate_counts"],
                         route["specdec2_mtp2_accepted_counts"])
                if reference is not None:
                    assert value == reference
                reference = value
    after = memory_stats()
    assert after["active_allocations"] == before["active_allocations"]
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
