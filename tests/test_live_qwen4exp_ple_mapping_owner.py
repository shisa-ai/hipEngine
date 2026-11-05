"""Public PLE mapping selection and explicit normal rollback preserve output."""
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason="requires ROCm and Qwen4Exp GGUF")


def test_public_mapping_policy_and_cold_unique_rows(monkeypatch):
    from hipengine import LLM, SamplingParams
    from hipengine.core.memory import memory_stats
    from hipengine.core.hip import get_hip_runtime

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_PLE_MAPPING_ACCESS", "normal")
    before = memory_stats()
    captures = []
    _install_logit_capture(monkeypatch, captures)
    llm = LLM(str(MODEL), backend="hip_gfx1151", execution_profile="production",
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache="off")
    try:
        owner = _unwrap(llm._get_text_generator())
        table = owner._resident.ple_table
        assert table._mapping_access_mode == "random"
        indices = np.random.default_rng(41).choice(table.semantic_rows, size=1025, replace=False)
        expected = table.gather_rows(indices)
        results = []
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        for mode in ("normal", "random", "random"):
            assert table.configure_mapping_access(mode)
            table.advise_cache("cold")
            assert table._mapping_access_mode == mode
            np.testing.assert_array_equal(table.gather_rows(indices), expected)
            captures.clear()
            output = llm.generate([list(range(100, 173))], params)
            results.append((output, [x.copy() for x in captures]))
        for output, rows in results[1:]:
            assert output == results[0][0]
            assert rows and len(rows) == len(results[0][1])
            for actual, expected_row in zip(rows, results[0][1], strict=True):
                np.testing.assert_array_equal(actual, expected_row)
    finally:
        llm.close()
    get_hip_runtime().device_synchronize()
    after = memory_stats()
    for key in ("active_allocations", "current_allocated_bytes"):
        assert after[key] == before[key]
