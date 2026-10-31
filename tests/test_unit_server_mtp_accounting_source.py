"""Only request-local execution accounting can attribute speculative output."""
from types import SimpleNamespace

import pytest

from hipengine.server.api import _mtp_draft_counts


@pytest.mark.parametrize('telemetry', [
    SimpleNamespace(diagnostics={'speculative_provider': 'gemma4_mtp',
        'proposed_draft_tokens': 7, 'accepted_draft_tokens': 4,
        'cycles': [{'accepted': 2}, {'accepted': 2}]}),
    SimpleNamespace(diagnostics={}, timing={'mtp_generated_draft_tokens': 7,
        'mtp_accepted_draft_tokens': 4, 'mtp_cycles_count': 2},
        timing_owner=True, timing_scope='choice'),
])
def test_legacy_only_payload_cannot_supply_draft_counts(telemetry):
    assert _mtp_draft_counts(SimpleNamespace(telemetry=telemetry)) is None


def test_execution_block_counts_are_independent_of_batch_timing_ownership():
    block = {'generated_draft_tokens': 7, 'accepted_draft_tokens': 4, 'cycles': 2}
    detail = SimpleNamespace(telemetry=SimpleNamespace(
        diagnostics={'specdec2_mtp2': block}, timing_owner=False,
        timing_scope='batch', timing={'mtp_generated_draft_tokens': 999}))
    assert _mtp_draft_counts(detail) == (4, 7, 2)
