"""Public Q5 risk dispatch observes explicit owners under opposing process input."""
from contextlib import ExitStack
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason="requires ROCm and Qwen4Exp GGUF")


def test_public_q5_risk_owner(monkeypatch):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.kernels.registry import KernelKey, register, resolve
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats

    before = memory_stats()
    captures, calls = [], {"q5k": [], "q51": []}
    _install_logit_capture(monkeypatch, captures)
    originals = []
    llm = LLM(str(MODEL), backend="hip_gfx1151", execution_profile="production",
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache="off")
    try:
        for family, quant, variant in (
            ("q5k", "gguf_q5_k", "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"),
            ("q51", "gguf_q5_1", "selected_wmma_iu8_risk_prefill_bf16_bf16_out"),
        ):
            key = KernelKey("hip_gfx1151", "moe_linear", quant, variant)
            original = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
            originals.append((key, original))
            def counted(*args, _family=family, _original=original, **kwargs):
                calls[_family].append(args[10] if _family == "q5k" else args[9])
                return _original(*args, **kwargs)
            register(key, counted, replace=True)
        owner = _unwrap(llm._get_text_generator())
        serving = owner.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        runners = list(serving._all_runners)
        if all(owner.runner is not r for r in runners):
            runners.append(owner.runner)
        with ExitStack() as stack:
            for target in [owner, *runners]:
                stack.enter_context(runtime.q5k_iu8_risk_mult(target, 24.0))
                stack.enter_context(runtime.q51_iu8_risk_mult(target, 32.0))
                stack.enter_context(runtime.routed_f16_enabled(target, False))
            params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
            outputs, logits = [], []
            for ambient in ("1", "invalid"):
                monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q5_K_IU8_RISK_MULT", ambient)
                monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q51_IU8_RISK_MULT", ambient)
                captures.clear()
                for values in calls.values():
                    values.clear()
                outputs.append(llm.generate([list(range(100, 173))], params))
                assert calls["q5k"] and set(calls["q5k"]) == {24.0}
                assert calls["q51"] and set(calls["q51"]) == {32.0}
                logits.append([x.copy() for x in captures])
            assert outputs[0] == outputs[1]
            assert logits[0] and len(logits[0]) == len(logits[1])
            for a, b in zip(*logits, strict=True):
                assert np.isfinite(a).all()
                np.testing.assert_array_equal(a, b)
    finally:
        for key, original in originals:
            register(key, original, replace=True)
        llm.close()
    get_hip_runtime().device_synchronize()
    after = memory_stats()
    for key in ("active_allocations", "current_allocated_bytes"):
        assert after[key] == before[key]
