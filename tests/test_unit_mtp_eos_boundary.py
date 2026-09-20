from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.speculative import DraftBatch, TargetAcceptSummary, TargetVerifyBatch
from hipengine.speculative import streaming
from hipengine.runtime import qwen35_gguf_mtp as runtime
from hipengine.generation.qwen35_gguf_mtp2 import _row_eos_token_id
from hipengine.speculative.sampling import SparseDistribution, sampled_accept_from_distributions


def _batch(candidates=(5, 2, 7)):
    draft = DraftBatch(
        request_ids=(17,), candidate_tokens=candidates,
        parent_positions=(10, 11, 12), draft_depths=(1, 2, 3),
        row_to_request=(17, 17, 17), mode="verify_chain",
    )
    return TargetVerifyBatch.from_draft(draft, root_tokens=(100,), root_positions=(10,))


def _summary(batch, top1):
    return TargetAcceptSummary.from_accept_result(
        batch, batch.accept_from_top1(top1, transaction_id=1, remaining_decode=(4,)),
    )


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_eos_is_the_final_visible_token_not_a_consumed_candidate(index):
    predictions = [5, 6, 7, 8]
    predictions[index] = 2
    batch = _batch(tuple(predictions[:3]))
    original = _summary(batch, predictions)
    summary, finished = streaming.limit_chain_accept_eos(
        batch, original, eos_token_id=2, generated_tokens=9,
    )
    assert finished
    assert summary.accepted_counts == (index,)
    assert summary.accepted_tokens == (tuple(predictions[:index]),)
    assert summary.next_tokens == (2,)
    assert summary.commit_rows == (index,)
    assert summary.commit_positions == (10 + index,)
    assert summary.commit_tokens == ((100 if index == 0 else predictions[index - 1]),)


@pytest.mark.parametrize("ignore,minimum,finished", [(True, 0, False), (False, 20, False), (False, 11, True)])
def test_eos_limit_respects_ignore_and_min_tokens(ignore, minimum, finished):
    batch = _batch()
    original = _summary(batch, (5, 2, 7, 8))
    summary, actual = streaming.limit_chain_accept_eos(
        batch, original, eos_token_id=2, generated_tokens=9,
        min_tokens=minimum, ignore_eos=ignore,
    )
    assert actual is finished
    if not finished:
        assert summary is original


def test_unreachable_draft_eos_does_not_finish_rejected_chain():
    batch = _batch()
    original = _summary(batch, (9, 2, 7, 8))
    summary, finished = streaming.limit_chain_accept_eos(batch, original, eos_token_id=2)
    assert summary is original
    assert not finished


def test_eos_resolution_uses_request_sampling_or_tokenizer_metadata():
    row = SimpleNamespace(request=SimpleNamespace(eos_token_id=None))
    tokenizer = SimpleNamespace(eos_token_id=2)
    assert _row_eos_token_id(row, tokenizer) == 2
    row.sampling_request = SimpleNamespace(eos_token_id=3)
    assert _row_eos_token_id(row, tokenizer) == 3


def test_host_sampled_accept_does_not_draw_rng_after_eos():
    batch = _batch()
    targets = tuple(SparseDistribution.point_mass(token) for token in (5, 2, 7, 8))
    draws = []
    result = sampled_accept_from_distributions(
        batch, targets, targets,
        draws=lambda: draws.append(0.5) or 0.5, eos_token_ids=(2,),
    )
    assert result.accepted_tokens == ((5,),)
    assert result.next_tokens == (2,)
    assert result.selected_candidate_rows == (1,)
    assert len(draws) == 2


def test_graph_terminal_recommit_restages_summary_and_preserves_hidden_rows(monkeypatch):
    batch = _batch()
    original = _summary(batch, (5, 2, 7, 8))
    summary, _ = streaming.limit_chain_accept_eos(batch, original, eos_token_id=2)
    tensors = {
        name: SimpleNamespace(name=name) for name in (
            "accepted_counts", "commit_rows", "commit_tokens", "commit_positions",
            "next_tokens", "full_accept", "committed_output_lengths",
        )
    }
    tensors["committed_output_ids"] = SimpleNamespace(name="committed_output_ids", shape=(1, 4))
    prepared = SimpleNamespace(
        batch=batch, summary=original, native_device_accept_commit=True,
        device_state_commit_buffers=SimpleNamespace(hidden_taps_src=SimpleNamespace(ptr=0x2000)),
        buffers=SimpleNamespace(**tensors),
    )
    verifier = object.__new__(runtime.Qwen35GGUFTransactionalVerifier)
    verifier._prepared = prepared
    verifier.target = SimpleNamespace(runtime=object())
    copies = []
    verifier.journal = SimpleNamespace(
        row_hidden=SimpleNamespace(ptr=0x1000), hidden_nbytes=16,
        _copy_d2d=lambda *args, **kwargs: copies.append(args),
    )
    writes = {}
    monkeypatch.setattr(runtime, "_copy_array", lambda tensor, array, rt: writes.update({tensor.name: array.copy()}))
    verifier.adopt_terminal_summary(prepared, summary)
    assert not prepared.native_device_accept_commit
    assert prepared.summary is summary
    assert copies == [(0x1000, 0x2000, 64)]
    assert writes["accepted_counts"].tolist() == [1]
    assert writes["commit_positions"].tolist() == [11]
    assert writes["next_tokens"].tolist() == [2]
    np.testing.assert_array_equal(writes["committed_output_ids"], [[100, 5, -1, -1]])


def test_terminal_cursor_publication_updates_only_the_owned_slot():
    published = []
    target = SimpleNamespace(
        _target_scratch_owner=SimpleNamespace(
            position_host=np.asarray([99, 12]),
            set_full_attention_positions=lambda *args: pytest.fail("must not rewrite neighbor metadata"),
        ),
        scratch=SimpleNamespace(
            position_host=np.asarray([12]),
            set_full_attention_positions=lambda positions, runtime: published.append(positions),
        ),
        runtime=object(), _position=12,
    )
    verifier = object.__new__(runtime.Qwen35GGUFTransactionalVerifier)
    verifier.target = target
    verifier._publish_position(11, stream=0)
    assert published == [(11,)]
    assert target._position == 11
    assert target._target_scratch_owner.position_host.tolist() == [99, 12]
