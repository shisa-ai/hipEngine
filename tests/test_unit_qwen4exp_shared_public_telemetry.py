"""Regression for public summary counts on shared target rows; CPU only."""
from types import SimpleNamespace
from hipengine.generation.qwen4_exp_serving_mixed import Qwen4ExpServingMixed
from hipengine.server.api import _mtp_response_summary


def test_shared_mtp_public_counts_match_committed_diagnostics():
    row=SimpleNamespace(prompt_ids=(1,2),generated_ids=[3]*8,mtp_budget=4,
        mixed_proposed=12,mixed_accepted=6,mixed_cycles=3,mixed_widths={2:3},mixed_target_rows=18)
    telemetry=Qwen4ExpServingMixed._mixed_telemetry(object(),row)
    result=_mtp_response_summary('speculative_mtp',[SimpleNamespace(telemetry=telemetry)])
    assert result['draft_tokens']==12
    assert result['accepted_draft_tokens']==6
    assert result['draft_cycles']==3
    assert result['acceptance_rate']==0.5


def test_shared_ar_peer_does_not_claim_speculation():
    row=SimpleNamespace(prompt_ids=(1,2),generated_ids=[3]*8,mtp_budget=0,
        mixed_proposed=0,mixed_accepted=0,mixed_cycles=0,mixed_widths={2:3},mixed_target_rows=3)
    telemetry=Qwen4ExpServingMixed._mixed_telemetry(object(),row)
    result=_mtp_response_summary('default',[SimpleNamespace(telemetry=telemetry)])
    assert result['used'] is False
    assert result['draft_tokens']==0
