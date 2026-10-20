"""Synthetic full-model shared target work and selected-prefix publication."""
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
from tests.test_gpu_qwen4exp_batch_runner import (
    model, two_qsa_model, _batch, _c1, _download, LOGIT_ATOL, LOGIT_RTOL,
)


@pytest.mark.parametrize('accepted', [0, 1, 2, 3, 4])
@pytest.mark.parametrize('permuted', [False, True])
def test_two_attention_layers_depth_four_and_abort_restore_visibility(two_qsa_model, accepted, permuted):
    model = two_qsa_model
    batch = _batch(model)
    references = [_c1(model), _c1(model)]
    prompts = ((2, 5, 3), (7, 4))
    slots = (2, 0)
    target = None
    try:
        owners = tuple(batch.admit(slot) for slot in slots)
        for slot, owner, runner, prompt in zip(slots, owners, references, prompts, strict=True):
            runner.prefill(prompt)
            batch.import_prefilled(slot, owner, source=runner)
        requests = (Qwen4ExpMixedRequest(10, 2, owners[0], 3, 11, (13, 15, 17, 19)),
                    Qwen4ExpMixedRequest(20, 0, owners[1], 2, 21))
        layout = compile_mixed_layout(requests, physical_row_order=(5, 0, 1, 2, 3, 4) if permuted else None)
        before_positions = tuple(batch._positions)
        before_cursors = tuple(index.cursor_host.copy() for index in batch.index_states)
        before_gdn = tuple((_download(state.recurrent_state, (state.recurrent_state.nbytes,), np.uint8, batch.runtime),
                            _download(state.conv_state, (state.conv_state.nbytes,), np.uint8, batch.runtime))
                           for state in batch.gdn_states)
        before_ple = _download(batch.ple_state.history, (batch.ple_state.history.nbytes,), np.uint8, batch.runtime)
        target = Qwen4ExpMixedTargetPass(batch, layout)
        target.verify()
        target.abort()
        target.close()
        target = None
        assert tuple(batch._positions) == before_positions
        for index, cursor in zip(batch.index_states, before_cursors, strict=True):
            np.testing.assert_array_equal(index.cursor_host, cursor)
            assert index.pending_rollback is None
        for state, (recurrent, history) in zip(batch.gdn_states, before_gdn, strict=True):
            np.testing.assert_array_equal(_download(state.recurrent_state, recurrent.shape, np.uint8, batch.runtime), recurrent)
            np.testing.assert_array_equal(_download(state.conv_state, history.shape, np.uint8, batch.runtime), history)
        np.testing.assert_array_equal(_download(batch.ple_state.history, before_ple.shape,
                                               np.uint8, batch.runtime), before_ple)
        target = Qwen4ExpMixedTargetPass(batch, layout)
        scores = target.verify()
        target.commit((accepted, 0))
        for request, runner in zip(requests, references, strict=True):
            for row in layout.rows_for_request(request.request_id):
                expected = runner.step(row.token)
                np.testing.assert_allclose(scores.logits[row.physical_row], expected.logits,
                                           atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
        for slot, owner, runner, prompt, request, depth in zip(slots, owners, references, prompts,
                                                              requests, (accepted, 0), strict=True):
            runner.prefill(prompt)
            for row in layout.rows_for_request(request.request_id)[:depth + 1]:
                runner.step(row.token)
            expected = runner.step(23)
            actual = batch.step((23,), slots=(slot,), owners=(owner,))
            np.testing.assert_allclose(actual.logits[0], expected.logits, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
    finally:
        if target is not None:
            target.close()
        batch.close()
        for runner in references:
            runner.close()


@pytest.mark.parametrize('accepted', [0, 1, 2])
@pytest.mark.parametrize('permuted', [False, True])
def test_shared_target_scores_and_selected_prefix_match_request_oracles(model, accepted, permuted):
    batch = _batch(model)
    references = [_c1(model), _c1(model)]
    target_pass = None
    try:
        owners = [batch.admit(slot) for slot in (2, 0)]
        for slot, owner, runner, prompt in zip((2, 0), owners, references, ((2, 5, 3), (7, 4)), strict=True):
            runner.prefill(prompt)
            batch.import_prefilled(slot, owner, source=runner)
        requests = (Qwen4ExpMixedRequest(10, 2, owners[0], 3, 11, (13, 15)),
                    Qwen4ExpMixedRequest(20, 0, owners[1], 2, 17))
        layout = compile_mixed_layout(requests, physical_row_order=(3, 0, 1, 2) if permuted else None)
        before = tuple(batch._positions)
        target_pass = Qwen4ExpMixedTargetPass(batch, layout)
        result = target_pass.verify()
        assert tuple(batch._positions) == before
        for request, runner in zip(requests, references, strict=True):
            for row in layout.rows_for_request(request.request_id):
                ref = runner.step(row.token)
                np.testing.assert_allclose(result.logits[row.physical_row], ref.logits,
                                           atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
        target_pass.commit((accepted, 0))
        assert tuple(batch._positions) == (3, 0, 4 + accepted)
        for slot, owner, runner, request, depth in zip((2, 0), owners, references, requests, (accepted, 0), strict=True):
            # Rebuild only the selected prefix, not the rejected scored suffix.
            runner.prefill((2, 5, 3) if slot == 2 else (7, 4))
            for row in layout.rows_for_request(request.request_id)[:depth + 1]:
                runner.step(row.token)
            ref = runner.step(19)
            actual = batch.step((19,), slots=(slot,), owners=(owner,))
            np.testing.assert_allclose(actual.logits[0], ref.logits, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
    finally:
        if target_pass is not None:
            target_pass.close()
        batch.close()
        for runner in references:
            runner.close()
