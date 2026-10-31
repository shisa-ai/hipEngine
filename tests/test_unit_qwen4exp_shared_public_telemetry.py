"""Public summary counts on shared target rows; CPU only."""
from types import SimpleNamespace

import pytest

from hipengine.generation.qwen4_exp_serving_mixed import (
    Qwen4ExpServingMixed,
    _MixedCommit,
)
from hipengine.server.api import _mtp_response_summary


def _mixed_row(*, commits, generated_ids, mtp_budget=3, **overrides):
    """One delivered mixed row: a real commit ledger plus its legacy aggregates.

    The legacy counters are derived from the ledger the way the serving pass
    writes them -- every committed cycle, delivered or not -- so a row trimmed
    by ``reclaim`` carries aggregates that describe more output than it
    delivered.
    """

    row = SimpleNamespace(
        prompt_ids=(1, 2),
        generated_ids=list(generated_ids),
        mtp_budget=mtp_budget,
        mixed_proposed=sum(commit.depth for commit in commits),
        mixed_accepted=sum(commit.accepted for commit in commits),
        mixed_cycles=len(commits),
        mixed_widths={2: 3},
        mixed_target_rows=18,
        mixed_commits=list(commits),
    )
    for name, value in overrides.items():
        setattr(row, name, value)
    return row


def _commit(*, position, tokens, accepted, depth=3):
    return _MixedCommit(
        position=position,
        tokens=tokens,
        accepted=accepted,
        depth=depth,
        reason="speculative_qualified",
    )


def test_shared_mtp_public_counts_come_from_the_delivered_commit_ledger():
    # ``reclaim`` trimmed this row to eight delivered ids, so the fourth
    # committed depth-3 cycle was never delivered. The block replays the ledger
    # onto that prefix; the legacy aggregates still describe all four cycles.
    row = _mixed_row(
        generated_ids=[3] * 8,
        commits=[
            _commit(position=1, tokens=3, accepted=2),
            _commit(position=4, tokens=3, accepted=2),
            # Cut by the output horizon: this cycle delivered only the target's
            # own token, so it accepted no draft inside the delivered prefix.
            _commit(position=7, tokens=1, accepted=0),
            _commit(position=8, tokens=3, accepted=2),
        ],
    )

    telemetry = Qwen4ExpServingMixed._mixed_telemetry(object(), row)
    block = telemetry.diagnostics["specdec2_mtp2"]
    assert block["cycles"] == 3
    assert block["generated_draft_tokens"] == 9
    assert block["accepted_draft_tokens"] == 4
    assert block["selected_depth_histogram"] == {"3": 3}
    assert block["mtp_output_tokens"] == 7
    # The legacy aggregates still claim the undelivered cycle.
    assert row.mixed_proposed == 12
    assert row.mixed_accepted == 6
    assert row.mixed_cycles == 4

    result = _mtp_response_summary(
        "speculative_mtp", [SimpleNamespace(telemetry=telemetry)]
    )

    # Counts come from the block, never from the legacy aggregates.
    assert result["draft_tokens"] == 9
    assert result["accepted_draft_tokens"] == 4
    assert result["rejected_draft_tokens"] == 5
    assert result["draft_cycles"] == 3
    assert result["acceptance_rate"] == 4 / 9
    assert result["mtp_output_tokens"] == 7
    assert result["ar_output_tokens"] == 1
    assert result["output_accounting"]["completion_tokens"] == 8
    assert result["output_accounting"]["mtp_coverage"] == 7 / 8
    assert result["output_accounting"]["reconciled"] is True
    assert result["output_accounting"]["reconciled_reasons"] == []


def test_shared_ar_peer_does_not_claim_speculation():
    # A peer shares the target pass but never carried speculative intent, so it
    # publishes no block and reports no speculation.
    row = _mixed_row(
        generated_ids=[3] * 8,
        mtp_budget=0,
        commits=[],
        mixed_widths={},
        mixed_target_rows=0,
    )
    telemetry = Qwen4ExpServingMixed._mixed_telemetry(object(), row)
    assert "specdec2_mtp2" not in telemetry.diagnostics

    result = _mtp_response_summary("default", [SimpleNamespace(telemetry=telemetry)])

    assert result["used"] is False
    assert result["draft_tokens"] == 0
    assert result["draft_cycles"] == 0
    assert "output_accounting" not in result


@pytest.mark.parametrize(('mixed_mtp', 'widths', 'expected'), [
    (True, {}, 'mixed'),
    (False, {2: 1}, 'mixed'),
    (False, {}, 'ar'),
])
def test_terminal_single_token_publication_preserves_mixed_telemetry(
    mixed_mtp, widths, expected,
):
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpResidentServingRunner
    row = SimpleNamespace(
        request_id=7, generated_ids=[11, 12], token_logprobs=[],
        mixed_mtp=mixed_mtp, mixed_widths=widths,
    )
    finish = SimpleNamespace(reason='length')
    from hipengine.generation.registry import GenerationTelemetry

    owner = SimpleNamespace(
        _visible_ids=lambda row, finish: row.generated_ids,
        _stream_text_delta=lambda row, visible, final: 'done',
        _ar_batch_telemetry=lambda row: GenerationTelemetry(decode_state={}, diagnostics={'source': 'ar'}),
        _mixed_telemetry=lambda row: GenerationTelemetry(decode_state={}, diagnostics={'source': 'mixed'}),
        _with_prefix_telemetry=lambda row, telemetry: telemetry,
    )
    published = Qwen4ExpResidentServingRunner._published_token(owner, row, 12, finish)
    assert published.stream_chunk.telemetry.diagnostics['source'] == expected
    assert published.stream_chunk.generated_token_ids == (11, 12)
