"""The pair harness records per-row draft acceptance, from the extension.

The extension block is where the served row reports its own MTP accounting
(``speculative_mtp.accepted_draft_tokens``); the usage block carries the same
count under the OpenAI key when a server populates it. Both are read so a
category suite row can state its acceptance instead of only its coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gguf_mtp_server_pair_bench as bench  # noqa: E402


def test_route_summary_reads_acceptance_from_the_extension():
    extension = {
        "speculative_mtp": {
            "used": True,
            "draft_cycles": 37,
            "accepted_draft_tokens": 96,
            "output_accounting": {"mtp_output_tokens": 127, "mtp_coverage": 0.99},
        }
    }
    summary = bench.route_summary(extension, None)
    assert summary["accepted_draft_tokens"] == 96
    assert summary["speculative_cycles"] == 37
    assert summary["mtp_coverage"] == 0.99


def test_route_summary_falls_back_to_the_usage_block():
    extension = {"speculative_mtp": {"used": True, "draft_cycles": 37}}
    usage = {
        "completion_tokens": 128,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 96,
            "rejected_prediction_tokens": 15,
        },
    }
    summary = bench.route_summary(extension, usage)
    assert summary["accepted_draft_tokens"] == 96
    assert summary["rejected_draft_tokens"] == 15


def test_route_summary_tolerates_a_missing_usage_block():
    summary = bench.route_summary({}, None)
    assert summary["accepted_draft_tokens"] is None
    assert summary["rejected_draft_tokens"] is None


def test_a_stored_prompt_row_keeps_the_acceptance_keys():
    """The row a category rollup reads must carry acceptance, not only coverage."""

    assert {"accepted_draft_tokens", "rejected_draft_tokens"} <= set(bench.ROUTE_ROW_KEYS)
    assert "mtp_coverage" in bench.ROUTE_ROW_KEYS
