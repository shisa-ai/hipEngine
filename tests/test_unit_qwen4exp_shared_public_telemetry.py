"""Regression for public summary counts on shared target rows; CPU only."""
from types import SimpleNamespace
import pytest
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
