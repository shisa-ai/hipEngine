from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_draft_trace import parse_llamacpp_mtp_draft_trace


TRACE_FIXTURE = Path("benchmarks/fixtures/llamacpp_mtp_explain_concept_draft_trace.json")


def test_parse_llamacpp_mtp_draft_trace_handles_multiline_pieces_and_acceptance() -> None:
    text = """
slot update_slots: id  3 | task 0 | new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = 17
 - seq_id 3, draft candidate   0, pos   0:   8068 (   0.476) 'Spec'
 - seq_id 3, draft candidate   1, pos   0:     16 (   0.168) '1'
 - seq_id 3, draft candidate   2, pos   0: 248068 (   0.106) '<think>'
common_speculative_draft: called impl draft-mtp, hist size = 17, call_count = 1, gen = 1
slot update_slots: id  3 | task 0 | accepted 0/1 draft tokens, new n_tokens = 18
 - seq_id 3, draft candidate   0, pos   0:    271 (   0.950) '

'
 - seq_id 3, draft candidate   1, pos   0:    198 (   0.050) '
'
 - seq_id 3, draft candidate   2, pos   0:     13 (   0.000) '.'
common_speculative_draft: called impl draft-mtp, hist size = 18, call_count = 2, gen = 1
slot update_slots: id  3 | task 0 | accepted 0/1 draft tokens, new n_tokens = 19
slot print_timing: id  3 | task 0 | draft acceptance = 0.00000 (    0 accepted /     2 generated)
"""

    trace = parse_llamacpp_mtp_draft_trace(text, source_log="synthetic")

    assert trace["prompt_tokens"] == 17
    assert trace["summary"] == {
        "draft_call_count": 2,
        "candidate_count": 6,
        "observed_top_k": 3,
        "draft_n": 2,
        "draft_n_accepted": 0,
        "draft_acceptance": 0.0,
    }
    assert trace["calls"][0]["candidates"][0] == {
        "line": 3,
        "seq_id": 3,
        "rank": 0,
        "pos": 0,
        "token_id": 8068,
        "prob": 0.476,
        "piece": "Spec",
    }
    assert trace["calls"][1]["candidates"][0]["piece"] == "\n\n"
    assert trace["calls"][1]["candidates"][1]["piece"] == "\n"
    assert trace["calls"][0]["accepted"] == 0
    assert trace["calls"][1]["accepted"] == 0


def test_committed_llamacpp_mtp_draft_trace_fixture_pins_short_prompt_topk() -> None:
    fixture = json.loads(TRACE_FIXTURE.read_text())

    assert fixture["schema"] == 1
    assert fixture["kind"] == "llamacpp_mtp_draft_candidate_trace"
    assert fixture["metadata"]["model"].endswith("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    assert fixture["metadata"]["request"]["prompt_name"] == "explain_concept"
    assert fixture["metadata"]["request"]["n_predict"] == 4
    assert fixture["metadata"]["request"]["temperature"] == 0.0
    server_command = fixture["metadata"]["server_command"]
    assert "--spec-type" in server_command
    assert "draft-mtp" in server_command
    assert "--spec-draft-n-max" in server_command
    assert "--no-spec-draft-backend-sampling" in server_command
    assert "--log-verbosity" in server_command
    assert fixture["prompt_tokens"] == 17
    assert fixture["summary"] == {
        "candidate_count": 6,
        "draft_acceptance": 0.0,
        "draft_call_count": 2,
        "draft_n": 2,
        "draft_n_accepted": 0,
        "observed_top_k": 3,
    }
    first, second = fixture["calls"]
    assert first["hist_size"] == 17
    assert [(row["rank"], row["token_id"], row["prob"], row["piece"]) for row in first["candidates"]] == [
        (0, 8068, 0.476, "Spec"),
        (1, 16, 0.168, "1"),
        (2, 248068, 0.106, "<think>"),
    ]
    assert second["hist_size"] == 18
    assert [(row["rank"], row["token_id"], row["prob"], row["piece"]) for row in second["candidates"]] == [
        (0, 271, 0.95, "\n\n"),
        (1, 198, 0.05, "\n"),
        (2, 13, 0.0, "."),
    ]
