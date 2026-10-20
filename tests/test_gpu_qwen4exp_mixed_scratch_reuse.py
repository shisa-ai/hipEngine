"""Mixed target scratch reuse across growing and shrinking real device passes."""
import ctypes

import numpy as np
import pytest

try:
    ctypes.CDLL('libamdhip64.so')
    HIP_AVAILABLE = True
except OSError:
    HIP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not HIP_AVAILABLE, reason='HIP runtime is not available')

from hipengine.runtime.qwen4_exp_mixed_layout import Qwen4ExpMixedRequest, compile_mixed_layout
from hipengine.runtime.qwen4_exp_mixed_target import Qwen4ExpMixedTargetPass
from tests.test_gpu_qwen4exp_batch_runner import model, _batch, _c1, LOGIT_ATOL, LOGIT_RTOL


def test_mixed_scratch_growth_shrink_and_repeat_preserve_scores(model):
    batch = _batch(model)
    references = [_c1(model), _c1(model)]
    prompts = ((2, 5, 3), (7, 4))
    slots = (2, 0)
    saved = []
    previous_scratch = None
    target = None
    try:
        owners = tuple(batch.admit(slot) for slot in slots)
        for slot, owner, runner, prompt in zip(slots, owners, references, prompts, strict=True):
            runner.prefill(prompt)
            batch.import_prefilled(slot, owner, source=runner)
        positions = tuple(batch._positions)
        for depth in (0, 4, 1, 3, 0, 4):
            requests = (
                Qwen4ExpMixedRequest(10, slots[0], owners[0], len(prompts[0]), 11,
                                    (13, 15, 17, 19)[:depth]),
                Qwen4ExpMixedRequest(20, slots[1], owners[1], len(prompts[1]), 21),
            )
            layout = compile_mixed_layout(requests)
            target = Qwen4ExpMixedTargetPass(batch, layout)
            scratch = target.scratch
            if previous_scratch is not None and previous_scratch.capacity_rows >= layout.total_rows:
                assert scratch is previous_scratch
            result = target.verify()
            for request, runner, prompt in zip(requests, references, prompts, strict=True):
                runner.prefill(prompt)
                for row in layout.rows_for_request(request.request_id):
                    expected = runner.step(row.token)
                    np.testing.assert_allclose(result.logits[row.physical_row], expected.logits,
                                               atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
            for old_depth, old_result, old_logits, old_hidden in saved:
                np.testing.assert_array_equal(old_result.logits, old_logits)
                np.testing.assert_array_equal(old_result.hidden_seeds, old_hidden)
                if old_depth == depth:
                    np.testing.assert_array_equal(result.logits, old_logits)
                    np.testing.assert_array_equal(result.hidden_seeds, old_hidden)
            saved.append((depth, result, result.logits.copy(), result.hidden_seeds.copy()))
            target.close()
            target = None
            assert not scratch.closed
            assert scratch.borrowers == 0
            assert tuple(batch._positions) == positions
            previous_scratch = scratch
        batch.close()
        assert previous_scratch.closed
    finally:
        if target is not None:
            target.close()
        batch.close()
        for runner in references:
            runner.close()
