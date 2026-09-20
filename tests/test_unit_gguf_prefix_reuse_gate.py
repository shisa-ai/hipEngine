from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.gguf_prefix_reuse_gate import (
    _compare_states,
    _lifecycle_exact,
    _logical_page_segments,
    _production_metadata_exact,
    build_parser,
)


def test_logical_page_segments_follow_noncontiguous_block_table_order() -> None:
    allocation = SimpleNamespace(
        block_ids=(8, 9, 11),
        chunk_start_block_id=8,
    )

    assert _logical_page_segments(
        allocation,
        position=513,
        row_nbytes=4,
    ) == (
        (0, 1024),
        (1024, 1024),
        (3072, 4),
    )

    with pytest.raises(ValueError, match="does not cover"):
        _logical_page_segments(allocation, position=769, row_nbytes=4)


def test_compare_states_reports_exact_component_and_layer() -> None:
    reference = {
        "position": 257,
        "linear": [{"layer": 0, "conv": "a", "recurrent": "b"}],
        "kv": [{"layer": 3, "key": "c", "value": "d", "checked_nbytes": 8}],
    }
    exact = {
        "position": 257,
        "linear": [{"layer": 0, "conv": "a", "recurrent": "b"}],
        "kv": [{"layer": 3, "key": "c", "value": "d", "checked_nbytes": 8}],
    }
    changed = {
        "position": 257,
        "linear": [{"layer": 0, "conv": "a", "recurrent": "changed"}],
        "kv": [{"layer": 3, "key": "c", "value": "d", "checked_nbytes": 8}],
    }

    assert _compare_states(exact, reference) == []
    assert _compare_states(changed, reference) == [
        {
            "component": "linear",
            "layer": 0,
            "part": "recurrent",
            "candidate": "changed",
            "reference": "b",
        }
    ]


def test_workspace_lease_pages_are_not_prefix_lifecycle_leaks() -> None:
    assert _lifecycle_exact(
        "active",
        source_refcount_before_release=2,
        source_refcount_after_release=1,
        shared_refcount_after_admission=2,
        shared_refcount_after_continuation_release=0,
        final_refcounted_pages=32,
        final_pinned_pages=32,
        workspace_lease_pages=32,
        source_session_reset=True,
        snapshot_evicted=False,
    )


def test_completed_source_lifecycle_and_metadata_fail_closed() -> None:
    assert _lifecycle_exact(
        "completed",
        source_refcount_before_release=1,
        source_refcount_after_release=1,
        shared_refcount_after_admission=2,
        shared_refcount_after_continuation_release=1,
        final_refcounted_pages=0,
        source_session_reset=True,
        snapshot_evicted=True,
    )
    assert not _lifecycle_exact(
        "completed",
        source_refcount_before_release=1,
        source_refcount_after_release=0,
        shared_refcount_after_admission=1,
        shared_refcount_after_continuation_release=0,
        final_refcounted_pages=0,
        source_session_reset=True,
        snapshot_evicted=True,
    )
    assert _production_metadata_exact(
        "completed",
        boundary=256,
        reused_tokens=256,
        source_request_id=None,
        source_id=1001,
        clone_bytes=384,
        snapshot_hit=True,
    )
    assert not _production_metadata_exact(
        "completed",
        boundary=256,
        reused_tokens=256,
        source_request_id=1001,
        source_id=1001,
        clone_bytes=384,
        snapshot_hit=False,
    )

    args = build_parser().parse_args(
        [
            "--source-lifecycle",
            "completed",
            "--sampler-mode",
            "processed_argmax",
            "--forced-token-id",
            "811",
        ]
    )
    assert args.source_lifecycle == "completed"
    assert args.sampler_mode == "processed_argmax"
    assert args.forced_token_id == 811


def test_only_the_production_shaped_comparison_gates_the_contract() -> None:
    """The binding terms reproduce the production prefill shape.

    The one-shot-prefix and serial routes prefill the same tokens through a
    different shape (a single batched prefix call, or per-token decode steps,
    instead of a batched suffix prefill).  Their batched arithmetic is not
    bit-equal to the production shape by construction: on the Japanese suite
    prompt the two single-call routes disagree with each other (174267 vs
    96026) while the candidate and the native chunked oracle agree (271).  Those
    comparisons stay in the payload as diagnostics and must not gate.
    """

    from scripts.gguf_prefix_reuse_gate import _PRODUCTION_GATE_TERMS, gate_passed

    terms = {name: True for name in _PRODUCTION_GATE_TERMS}
    assert gate_passed(terms) is True

    for diagnostic in (
        "semantic_boundary_exact",
        "initial_state_exact",
        "final_state_exact",
        "output_exact",
        "trajectory_exact",
    ):
        assert diagnostic not in _PRODUCTION_GATE_TERMS
        assert gate_passed({**terms, diagnostic: False}) is True

    for binding in _PRODUCTION_GATE_TERMS:
        assert gate_passed({**terms, binding: False}) is False

    # The candidate's continuation token can be processor-forced while the
    # native oracle samples freely (`mixed_ja_en`: candidate 9709 versus oracle
    # 248046 with bit-identical states), so that comparison is diagnostic.
    assert "scheduler_output_exact" not in _PRODUCTION_GATE_TERMS
    assert gate_passed({**terms, "scheduler_output_exact": False}) is True

    incomplete = {k: v for k, v in terms.items() if k != "scheduler_state_exact"}
    with pytest.raises(KeyError):
        gate_passed(incomplete)


def test_per_step_diagnostics_match_the_aggregate_metrics() -> None:
    """The recorded per-step KL/top-1 must reproduce the aggregate exactly."""

    import numpy as np

    from hipengine.benchmark.correctness import evaluate_logits
    from scripts.gguf_prefix_reuse_gate import _per_step_softmax

    rng = np.random.default_rng(0)
    reference = rng.normal(size=(4, 97)).astype(np.float64)
    candidate = rng.normal(size=(4, 97)).astype(np.float64)
    metrics = evaluate_logits(reference, candidate)

    reference_p = _per_step_softmax(reference)
    kl = np.sum(
        reference_p * (np.log(reference_p) - np.log(_per_step_softmax(candidate))),
        axis=-1,
    )
    top1 = np.argmax(reference, axis=-1) == np.argmax(candidate, axis=-1)

    assert float(np.mean(kl)) == pytest.approx(metrics.kl_mean, abs=1e-12)
    assert float(np.max(kl)) == pytest.approx(metrics.kl_max, abs=1e-12)
    assert float(np.mean(top1)) == pytest.approx(metrics.top1_agreement, abs=1e-12)


def test_margin_diagnosis_reports_rank_and_near_tie_margin() -> None:
    """A flipped argmax on a near-tie must be visible as a tiny margin."""

    import numpy as np

    from scripts.gguf_prefix_reuse_gate import _per_step_margins

    reference = np.array(
        [
            [10.0, 9.999, 0.0, -1.0],  # near tie: candidate flips it
            [1.0, 0.0, -1.0, -2.0],  # agreement
            [10.0, 0.0, -1.0, -2.0],  # candidate has a different winner
        ]
    )
    candidate = np.array(
        [
            [9.998, 10.0, 0.0, -1.0],
            [1.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, -1.0, -2.0],
        ]
    )

    diagnosis = _per_step_margins(reference, candidate, top_k=2)

    assert diagnosis["reference_top1_rank_in_candidate"] == [1, 0, 1]

    # Captured rows arrive as (batch=1, vocab), so the caller flattens the
    # stacked (steps, 1, vocab) tensor; the helper must be called on 2-D input
    # and must not silently index a singleton batch axis instead of the vocab.
    stacked = np.stack([reference, reference], axis=1)  # (3, 2, 4)
    flattened = stacked.reshape(-1, stacked.shape[-1])
    assert flattened.shape == (6, 4)
    assert len(_per_step_margins(flattened, flattened)["reference_margin"]) == 6
    assert diagnosis["reference_margin"][0] == pytest.approx(0.001, abs=1e-9)
    assert diagnosis["top_k_overlap"] == [2, 2, 2]
    assert diagnosis["top_k"] == 2


def test_reference_suffix_uses_batched_recomputation_and_keeps_serial_diagnostic():
    from scripts import gguf_prefix_reuse_gate as gate

    class Session:
        position = 1024

        def prefill_batch_native(self, prompts, **kwargs):
            assert prompts == [(7, 8, 9)]
            assert kwargs["sessions"] == [self]
            assert kwargs["full_prompt_lengths"] == [1027]
            assert kwargs["return_logits"]
            self.position += len(prompts[0])
            return [SimpleNamespace(token_id=17, logits=[1.0, 2.0])]

        def step(self, token, *, return_logits):
            self.position += 1
            return SimpleNamespace(token_id=token, logits=[3.0] if return_logits else [])

    session = Session()
    result = gate._prefill_reference_suffix(
        session, (7, 8, 9), full_prompt_length=1027, return_logits=True,
    )
    assert session.position == 1027
    assert result.token_id == 17
    session.position = 1024
    serial = gate._prefill_reference_suffix(
        session, (7, 8, 9), full_prompt_length=1027, return_logits=True, serial=True,
    )
    assert session.position == 1027
    assert serial.token_id == 9
    assert serial.logits == [3.0]


def test_prefix_numerical_gate_rejects_mean_drift_below_outer_smoke_limit():
    import numpy as np
    from scripts import gguf_prefix_reuse_gate as gate

    reference = np.tile([2.0, 0.0], (128, 1))
    candidate = np.tile([1.75, 0.0], (128, 1))
    result = gate._profile_metrics(reference, candidate, category="code", lifecycle="active")
    assert result["summary"]["kl_max"] < 0.05
    assert result["summary"]["top1_agreement"] == 1.0
    assert result["summary"]["kl_mean"] > 0.001
    assert not result["hard_gates_passed"]
    exact = gate._profile_metrics(reference, reference, category="code", lifecycle="completed")
    assert exact["hard_gates_passed"]


def test_profile_metrics_uses_shared_teacher_labels_not_local_argmax():
    import numpy as np
    from scripts import gguf_prefix_reuse_gate as gate

    result = gate._profile_metrics(
        np.array([[2.0, 0.0]]), np.array([[0.0, 2.0]]),
        category="code", lifecycle="active", teacher_token_ids=[1],
    )
    assert result["summary"]["strict_teacher_nll_mean"] == pytest.approx(2.126928011)
    assert result["summary"]["candidate_teacher_nll_mean"] == pytest.approx(0.126928011)


@pytest.mark.parametrize(
    ("length", "segments"),
    [(1, [1]), (17, [17]), (255, [255]), (256, [256]), (257, [257]), (258, [256, 2])],
)
def test_reference_suffix_matches_checkpoint_segment_boundaries(length, segments):
    from scripts import gguf_prefix_reuse_gate as gate

    calls = []

    class Session:
        position = 512

        def step(self, token, *, return_logits):
            calls.append(1)
            self.position += 1
            return SimpleNamespace(token_id=token)

        def prefill_batch_native(self, prompts, **kwargs):
            calls.append(len(prompts[0]))
            self.position += len(prompts[0])
            return [SimpleNamespace(token_id=prompts[0][-1])]

    session = Session()
    result = gate._prefill_reference_suffix(
        session, tuple(range(length)), full_prompt_length=512 + length, return_logits=False,
    )
    assert calls == segments
    assert session.position == 512 + length
    assert result.token_id == length - 1
