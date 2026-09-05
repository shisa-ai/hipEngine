from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.runtime.qwen35_paro_runner import Qwen35ParoBulkVerifyResult, Qwen35ParoResidentSession
from hipengine.speculative import mtp_native, paro_mtp_profiles
from scripts import mtp_chain_e2e_smoke


@pytest.fixture(autouse=True)
def _isolate_route_environment(monkeypatch: pytest.MonkeyPatch):
    """Prevent the profile binder's process-env contract from leaking across tests."""

    for name in (
        paro_mtp_profiles.PARO_MTP_CONTRACT_ENV,
        paro_mtp_profiles.PARO_MTP_ROUTE_ENV,
        paro_mtp_profiles.PARO_MTP_CHAIN_ATTN_MODE_ENV,
        paro_mtp_profiles.GDN_EXACT_ENV,
        paro_mtp_profiles.LINEAR_EXACT_ENV,
        paro_mtp_profiles.MOE_EXACT_ENV,
        paro_mtp_profiles.FULL_ATTN_EXACT_SUFFIX_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def _step_result() -> mtp_native.NativeMtpStepResult:
    return mtp_native.NativeMtpStepResult(token=17, logit=1.25, topk_experts=(), topk_logits=())


def test_step_hidden_taps_exposes_final_normalized_capture() -> None:
    parameters = inspect.signature(Qwen35ParoResidentSession.step_with_hidden_taps).parameters
    assert "capture_final_hidden_bf16" in parameters


def test_bulk_verify_result_carries_selected_final_hidden_pointer() -> None:
    result = Qwen35ParoBulkVerifyResult(
        target_top1=(17, 23),
        target_top1_values=(1.0, 0.5),
        accepted_count=0,
        accepted_tokens=(),
        commit_row=0,
        commit_token=11,
        commit_position=7,
        next_token=17,
        full_accept=False,
        finite_logits=True,
        gpu_accept_match_cpu=True,
        rows=2,
        selected_target_hidden_ptr=0x123400,
    )
    assert result.selected_target_hidden_ptr == 0x123400
    payload = result.to_json_dict()
    assert payload["selected_target_hidden_available"] is True
    assert "selected_target_hidden_ptr" not in payload


def test_proposer_target_hidden_reseed_forwards_borrowed_pointer() -> None:
    proposer = mtp_native.NativeMtpChainProposer.__new__(mtp_native.NativeMtpChainProposer)
    calls: list[dict[str, object]] = []

    def fake_advance(**kwargs):
        calls.append(kwargs)
        return _step_result()

    proposer.advance = fake_advance  # type: ignore[method-assign]
    result = proposer.advance_with_target_hidden(
        input_token=19,
        target_hidden_ptr=0xABC000,
        position=33,
        need_result=False,
        read_expert_topk=False,
        read_lm_head_value=False,
        stream=7,
    )

    assert result == _step_result()
    assert calls == [
        {
            "input_token": 19,
            "target_hidden_ptr": 0xABC000,
            "position": 33,
            "need_result": False,
            "read_token_id": True,
            "read_expert_topk": False,
            "read_lm_head_value": False,
            "stream": 7,
        }
    ]


def test_w8a16_target_head_binding_is_full_vocab_and_borrowed() -> None:
    head_cls = getattr(mtp_native, "NativeMtpW8A16Head")
    head = head_cls(
        weight_int8_ptr=0x1000,
        scale_f32_ptr=0x2000,
        vocab_size=248320,
        threads=256,
    )
    assert head.vocab_size == 248320
    assert head.weight_int8_ptr == 0x1000
    assert head.scale_f32_ptr == 0x2000


def test_borrowed_w8_head_rejects_closed_owner() -> None:
    import pytest

    owner = SimpleNamespace(closed=False)
    head = mtp_native.NativeMtpW8A16Head(
        weight_int8_ptr=0x1000,
        scale_f32_ptr=0x2000,
        vocab_size=248320,
        owner=owner,
    )
    head.validate_live()
    owner.closed = True
    with pytest.raises(RuntimeError, match="owner is closed"):
        head.validate_live()


def test_target_contract_scope_fails_closed_outside_b1_graph_off_chain() -> None:
    validate = mtp_chain_e2e_smoke._validate_proposer_target_contract_scope
    validate(
        enabled=True,
        candidate_budget=1,
        graph_mode="off",
        tree_mode="chain",
        confidence_threshold=0.0,
        draft_p_min=0.0,
        ar_fallback_zero_streak=0,
        overlap_verify_commit_proposer=False,
    )

    import pytest

    with pytest.raises(ValueError, match="B=1 only"):
        validate(
            enabled=True,
            candidate_budget=2,
            graph_mode="off",
            tree_mode="chain",
            confidence_threshold=0.0,
            draft_p_min=0.0,
            ar_fallback_zero_streak=0,
            overlap_verify_commit_proposer=False,
        )
    with pytest.raises(ValueError, match="graph_mode=off"):
        validate(
            enabled=True,
            candidate_budget=1,
            graph_mode="auto",
            tree_mode="chain",
            confidence_threshold=0.0,
            draft_p_min=0.0,
            ar_fallback_zero_streak=0,
            overlap_verify_commit_proposer=False,
        )


def test_omitted_route_defaults_to_registered_fast_production(monkeypatch) -> None:
    args = SimpleNamespace(execution_profile=None, chain_attn_mode=None)
    monkeypatch.delenv("HIPENGINE_MTP_PROPOSER_TARGET_CONTRACT", raising=False)

    mtp_chain_e2e_smoke._apply_paro_execution_profile(args)

    assert args.execution_profile == "production"
    assert args.chain_attn_mode == "decode_batched"
    assert len(args.execution_profile_manifest_sha256) == 64
    assert __import__("os").environ["HIPENGINE_MTP_PROPOSER_TARGET_CONTRACT"] == "1"


def test_explicit_strict_profile_selects_c1_loop() -> None:
    args = SimpleNamespace(execution_profile="strict", chain_attn_mode=None)
    mtp_chain_e2e_smoke._apply_paro_execution_profile(args)
    assert args.chain_attn_mode == "c1_loop"


def test_manual_chain_mode_preserves_legacy_diagnostic_route() -> None:
    args = SimpleNamespace(execution_profile=None, chain_attn_mode="batched")
    mtp_chain_e2e_smoke._apply_paro_execution_profile(args)
    assert args.execution_profile is None
    assert args.chain_attn_mode == "batched"


def test_cycle_reseed_helper_uses_selected_target_hidden() -> None:
    calls: list[dict[str, object]] = []

    class FakeProposer:
        position = 41

        def advance_with_target_hidden(self, **kwargs):
            calls.append(kwargs)
            return _step_result()

    verify = SimpleNamespace(selected_target_hidden_ptr=0xDEAD00)
    result = mtp_chain_e2e_smoke._advance_proposer_from_selected_target(
        FakeProposer(),
        verify=verify,
        input_token=29,
        need_result=True,
        read_expert_topk=False,
        read_lm_head_value=False,
        stream=3,
    )

    assert result == _step_result()
    assert calls == [
        {
            "input_token": 29,
            "target_hidden_ptr": 0xDEAD00,
            "position": 42,
            "need_result": True,
            "read_expert_topk": False,
            "read_lm_head_value": False,
            "stream": 3,
        }
    ]
