"""EOS selection must restore the same live state as the AR prefix."""

import pytest
import numpy as np

from tests.test_live_gguf_int8_mtp import (
    _MODEL, _hip_available, _assert_live_state, sessions, restore_kernel_registrations,
)
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFTransactionalVerifier
from hipengine.speculative import DraftBatch, TargetVerifyBatch, TargetCommitPlan
from hipengine.speculative.streaming import limit_chain_accept_eos


pytestmark = pytest.mark.skipif(
    not _hip_available() or not _MODEL.is_file(), reason="HIP or GGUF model unavailable",
)


@pytest.mark.parametrize("eos_index", [0, 1, 2, 3])
@pytest.mark.parametrize("graph", [False, True])
def test_eos_recommit_matches_ar_state_and_cursor(sessions, eos_index, graph):
    target, reference = sessions
    prompt = (9707, 11, 220, 264)
    root = int(target.prefill(prompt, use_bulk=False, return_logits=False).token_id)
    reference.prefill(prompt, use_bulk=False, return_logits=False)
    predictions = []
    token = root
    for _ in range(4):
        token = int(reference.step(token, return_logits=False).token_id)
        predictions.append(token)
    eos = predictions[eos_index]
    boundary = predictions.index(eos)
    batch = TargetVerifyBatch.from_draft(
        DraftBatch(
            request_ids=(17,), candidate_tokens=tuple(predictions[:3]),
            parent_positions=(4, 5, 6), draft_depths=(1, 2, 3),
            row_to_request=(17, 17, 17), mode="verify_chain",
        ),
        root_tokens=(root,), root_positions=(4,),
    )
    with Qwen35GGUFTransactionalVerifier(
        target, max_candidate_budget=3, quant="gguf_q4_k_m", target_verify_mode="native",
    ) as verifier:
        prepared = verifier.prepare(
            batch, transaction_id=1, graph_bucket=verifier.graph_bucket("eos", batch),
            remaining_decode=(4,), allow_graph=graph,
        )
        if graph:
            assert prepared.native_device_accept_commit
        summary, finished = limit_chain_accept_eos(batch, prepared.summary, eos_token_id=eos)
        assert finished
        if summary is not prepared.summary:
            verifier.adopt_terminal_summary(prepared, summary)
        plan = TargetCommitPlan(
            transaction_id=1, request_ids=batch.request_ids,
            accepted_counts=summary.accepted_counts, commit_rows=summary.commit_rows,
            commit_tokens=summary.commit_tokens, commit_positions=summary.commit_positions,
            next_tokens=summary.next_tokens, candidate_counts=batch.candidate_counts,
            draft_depth=batch.draft_depth, tree_shape=batch.tree_shape, mode=batch.mode,
        )
        verifier.commit(prepared, plan)
        verifier.finish(prepared)
    reference.prefill(prompt, use_bulk=False, return_logits=False)
    reference.step(root, return_logits=False)
    for token in predictions[:boundary]:
        reference.step(token, return_logits=False)
    _assert_live_state(target, reference)
    assert target.position == 5 + boundary
    np.testing.assert_array_equal(
        target.step(eos, return_logits=True).logits,
        reference.step(eos, return_logits=True).logits,
    )
