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


def _result_with(*routes):
    return {
        "shapes": {
            str(512 * (index + 1)): {
                "route": {"effective_route": route},
                "runs": [{"effective_route": route}],
            }
            for index, route in enumerate(routes)
        }
    }


def test_an_arm_that_ran_its_declared_route_is_accepted():
    result = _result_with("speculative_mtp", "speculative_mtp")
    assert bench.route_mismatches(result, expected="speculative_mtp") == []


def test_a_control_arm_that_ran_mtp_is_named_as_a_mismatch():
    """The recorded footgun: an AR control that silently ran the MTP route.

    Every row was byte-identical to the MTP arm, so the pair read as a clean
    1.00x ratio. Only the route each run reported said the arm had not run
    what its name claimed.
    """

    result = _result_with("speculative_mtp", "speculative_mtp")

    mismatches = bench.route_mismatches(result, expected="default")

    assert len(mismatches) == 4  # one route summary and one run per shape
    assert all("speculative_mtp" in item for item in mismatches)


def test_a_partly_degraded_arm_is_still_a_mismatch():
    """One shape falling back to AR is enough to invalidate the arm."""

    result = _result_with("speculative_mtp", "default")

    mismatches = bench.route_mismatches(result, expected="speculative_mtp")

    assert len(mismatches) == 2
    assert all(item.startswith("shapes.1024") for item in mismatches)


def test_a_missing_route_is_a_mismatch_rather_than_a_pass():
    """A run that reports no route cannot confirm the arm's contract."""

    result = {"shapes": {"512": {"runs": [{}]}}}

    mismatches = bench.route_mismatches(result, expected="default")

    assert mismatches == ["shapes.512[0]=None"]


def test_prompt_rows_are_checked_too():
    result = {
        "prompts": {
            "code-1": {
                "route": {"effective_route": "default"},
                "runs": [{"effective_route": "default"}],
            }
        }
    }

    assert bench.route_mismatches(result, expected="default") == []
    assert len(bench.route_mismatches(result, expected="speculative_mtp")) == 2


def test_the_cli_offers_an_arm_contract_flag():
    """The intent is declared on the command line, not in the environment.

    An environment-declared arm loses its intent when a launcher clears the
    environment, which is exactly how the AR control became an MTP duplicate.
    """

    import inspect

    source = inspect.getsource(bench.main)
    assert "--expect-route" in source
