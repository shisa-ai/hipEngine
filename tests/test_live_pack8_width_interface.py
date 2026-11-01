"""Public Pack8 comparison widths are read per launch, not cached per dispatch."""
import os

import numpy as np
import pytest

from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()),
    reason='requires ROCm and the Flash-Next UD-Q4_K_XL artifact',
)


@pytest.mark.parametrize('length', [1, 73])
def test_public_pack8_width_comparison_restores_and_repeats(monkeypatch, length):
    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as kernels
    from hipengine.runtime import gguf_linear as linear

    key = linear._PACK8_GEMV_WIDE_ENV
    prior = os.environ.get(key)
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    def open_llm():
        result = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
                     max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
        try:
            generator = _unwrap(result._get_text_generator())
            serving = generator.create_resident_model_runner(capacity=1)
            serving.prepare(max_sequence_length=CONTEXT)
        except BaseException:
            result.close()
            raise
        return result

    llm = open_llm() if length == 1 else None
    launches = []
    captures = []
    real = kernels._launch

    def launch(quant, symbol, x, w, out, rows, k, n, **kwargs):
        launches.append((quant, symbol, rows, k, n, kwargs.get('threads', 128)))
        return real(quant, symbol, x, w, out, rows, k, n, **kwargs)

    monkeypatch.setattr(kernels, '_launch', launch)
    _install_logit_capture(monkeypatch, captures)
    try:
        def arm(value):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
            # Decode graphs retain captured launch parameters. Separate gate
            # arms own separate sessions; do not infer a new launch from replay.
            active = llm if llm is not None else open_llm()
            try:
                def generate():
                    launches.clear()
                    captures.clear()
                    output = active.generate_detailed(
                        [[1] * length], SamplingParams(max_tokens=3, temperature=0,
                                                      ignore_eos=True, speculative_mtp=False))
                    assert len(output[0].generated_token_ids) == 3
                    assert captures and all(np.isfinite(row).all() for row in captures)
                    return (list(launches), [row.copy() for row in captures],
                            tuple(output[0].generated_token_ids))

                first = generate()
                if length != 1:
                    cached = generate()
                    assert first[2] == cached[2]
                    assert len(first[1]) == len(cached[1])
                    for expected, observed in zip(first[1], cached[1], strict=True):
                        np.testing.assert_array_equal(expected, observed)
                return first
            finally:
                if llm is None:
                    active.close()

        if length == 1:
            arm('0')  # exclude first-use graph construction from measured arms
        narrow = arm('0')
        wide = arm('1')
        default = arm(None)
        repeat = arm('0')

        def eligible(result):
            return [row for row in result[0] if row[2] == 1 and row[4] <= 640
                    and 'pack8_gemv_' in row[1]
                    and 'pack8_gemv_decode_' not in row[1]]

        for result, threads in ((narrow, 128), (wide, 256), (default, 256),
                                (repeat, 128)):
            rows = eligible(result)
            assert rows, 'no eligible Pack8 launch reached the registered wrapper'
            assert all(row[-1] == threads for row in rows), (threads, rows[:4])
            outside = [row for row in result[0] if row[2] != 1 or row[4] > 640]
            assert outside and all(row[-1] == 128 for row in outside), outside[:4]
        assert [(row[:-1]) for row in eligible(narrow)] == [
            row[:-1] for row in eligible(wide)]
        for reference, actual in ((narrow, repeat), (wide, default)):
            assert reference[2] == actual[2]
            assert len(reference[1]) == len(actual[1])
            for expected, observed in zip(reference[1], actual[1], strict=True):
                np.testing.assert_array_equal(expected, observed)
        # Equality across widths is not required: changing the reduction tree
        # is the purpose of this maintained numerical comparison interface.
    finally:
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior
        if llm is not None:
            llm.close()
    assert os.environ.get(key) == prior
    runtime.device_synchronize()
    after = memory_stats()
    for name in ('current_allocated_bytes', 'active_allocations'):
        assert after[name] == before[name]
