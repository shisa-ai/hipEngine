"""Peer-fairness summary: fresh rows emit the construction-time chunk field.

R0115 renamed the recorded session-chunk field from the post-set name to
``session_chunks_after_construction``, because the generator constructor now
owns the chunk and nothing re-asserts it afterwards. Archived artifacts still
carry the old name, so the summary reads the legacy key as a fallback and always
emits the new one.
"""

from scripts import qwen4exp_prefill_peer_fairness as fairness


def _row(**overrides):
    row = {
        "arm": "g1024_l1024",
        "status": "measured",
        "peer_id": 1,
        "prefill_id": 2,
        "peer_submit_t": 0.0,
        "prefill_submit_t": 0.5,
        "generator_chunk_requested": 1024,
        "loop_cap_requested": 1024,
        "prefill_prompt_tokens": 4,
        "tokens": [
            {"t": 1.0, "request_id": 1},
            {"t": 1.1, "request_id": 1},
            {"t": 2.0, "request_id": 2},
        ],
        "trace": {
            "loop_prefill_batch": [{"t": 1.5, "total": 4}],
            "session_prefill": [
                {"n_tokens": 4, "prefill_chunk_size": 1024,
                 "workspace_chunks": [1024]},
            ],
        },
        "work": [{"kind": "prefill"}, {"kind": "decode"}],
        "env": {"HIPENGINE_MAX_ACTIVE_REQUESTS": "2"},
        "host": {"machine_id": "test"},
        "elapsed_seconds": 3.0,
    }
    row.update(overrides)
    return row


def test_fresh_rows_summarize_the_construction_field():
    summary = fairness._summarize_arm(
        _row(session_chunks_after_construction=[4096]))
    assert summary["session_chunks_after_construction"] == [4096]
    assert "session_chunks_after_set" not in summary


def test_archived_rows_summarize_their_legacy_post_set_field():
    # The archived child traces predate the rename; they must still summarize,
    # under the field a reader of the new schema looks for.
    summary = fairness._summarize_arm(_row(session_chunks_after_set=[1024]))
    assert summary["session_chunks_after_construction"] == [1024]
    assert "session_chunks_after_set" not in summary


def test_the_fresh_field_wins_over_a_stale_legacy_one():
    summary = fairness._summarize_arm(_row(
        session_chunks_after_construction=[4096],
        session_chunks_after_set=[1024]))
    assert summary["session_chunks_after_construction"] == [4096]


def test_a_row_with_neither_field_summarizes_as_unrecorded():
    assert fairness._summarize_arm(_row())[
        "session_chunks_after_construction"] is None
