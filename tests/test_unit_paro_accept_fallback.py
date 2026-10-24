"""Acceptance fallback must commit the oracle row, not stale GPU metadata."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.core.dtype import DType
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession
from hipengine.speculative import MtpDraftRequest, TargetVerifyBatch, compile_mtp_chain


@pytest.mark.parametrize("mode", ["verify_chain", "verify_tree"])
@pytest.mark.parametrize("accept_mode", ["validate", "off", "on"])
@pytest.mark.parametrize("gpu_matches", [False, True])
def test_bulk_commit_uses_authoritative_accept_row(
    monkeypatch, mode, accept_mode, gpu_matches, native_commit=False, depth=1, cpu_row=0,
):
    monkeypatch.setenv("HIPENGINE_VERIFY_GPU_ACCEPT", accept_mode)
    tokens = (10, 20, 30, 40)[:depth + 1]
    cpu_top1 = tuple(tokens[i + 1] if i < cpu_row else 99 - i * 11
                     for i in range(depth + 1))
    batch = TargetVerifyBatch.from_draft(
        compile_mtp_chain(
            [MtpDraftRequest(request_id=0, root_position=7,
                             candidate_tokens=tokens[1:], active_count=depth)],
            candidate_budget=depth,
        ),
        root_tokens=(tokens[0],), root_positions=(7,),
    )
    batch = replace(batch, mode=mode)
    # Exercise recovery in either direction, including an interior/full prefix.
    gpu_row = cpu_row if gpu_matches else (cpu_row + 1) % (depth + 1)
    payload = dict(
        accepted_counts=(gpu_row,), commit_rows=(gpu_row,),
        commit_tokens=(tokens[gpu_row],), commit_positions=(7 + gpu_row,),
        next_tokens=(cpu_top1[gpu_row],), full_accept=(gpu_row == depth,),
        committed_output_lengths=(gpu_row + 1,),
    )
    state = {"row": None, "device_selector": None}
    positions = []
    def commit(selected_row, *, commit_row_ptr, **kwargs):
        # Model the actual fused kernel's pointer precedence.
        state["row"] = gpu_row if commit_row_ptr is not None else selected_row
        state["device_selector"] = commit_row_ptr
    def noop(*args, **kwargs):
        return None
    session = SimpleNamespace(
        closed=False, max_batch_size=4, config=SimpleNamespace(hidden_size=2),
        runtime=SimpleNamespace(stream_synchronize=noop),
        _check_slot=noop, _check_position=noop, _write_verify_chain_metadata=noop,
        _should_use_chain_tloop_linear_verify=lambda *a, **k: True,
        _launch_verify_chain_forward_accept=noop,
        _read_verify_accept_payload=lambda *a, **k: payload,
        _verify_gpu_accept_enabled=lambda: accept_mode != "off",
        _read_verify_top1=lambda rows: (cpu_top1, (1.0,) * (depth + 1)),
        _gpu_accept_payload_matches=Qwen35ParoResidentSession._gpu_accept_payload_matches,
        _commit_bulk_linear_states=commit,
        _verify_accept_updates_position_enabled=lambda batch: True,
        _record_slot_position_host=lambda position, **kwargs: positions.append(("host", position)),
        _set_slot_position=lambda position, **kwargs: positions.append(("device", position)),
        _verify_capture_staging_tensor=lambda **kwargs: capture,
        _run_verify_graph_or_direct=lambda *a, **k: {
            "status": "replayed", "native_spec_target_commit": native_commit,
        },
        _canonicalize_decode_scratch=noop,
        _commit_tree_full_attention_kv=noop,
        _commit_tree_capture_hidden_concat=noop,
        verify_commit_rows=SimpleNamespace(ptr=0x9000),
        batch_norm_out_bf16=SimpleNamespace(ptr=0x1000), hidden_nbytes=4,
    )
    capture = SimpleNamespace(dtype=DType.BF16, ndim=2, shape=(depth + 1, 0))
    verify = (
        Qwen35ParoResidentSession.verify_chain_bulk_and_commit
        if mode == "verify_chain"
        else Qwen35ParoResidentSession.verify_tree_bulk_and_commit
    )
    kwargs = dict(
        base_slot=0, capture_layer_ids=(), capture_hidden_concat=capture,
        capture_row_start=0, graph_mode="auto" if native_commit else "off",
    )
    if native_commit:
        with pytest.raises(RuntimeError, match="does not match the CPU oracle"):
            verify(session, batch, **kwargs)
        assert state["row"] is None
        assert positions == []
        return
    result = verify(session, batch, **kwargs)
    expected = gpu_row if accept_mode == "on" else cpu_row
    assert result.commit_row == expected
    assert result.commit_position == 7 + expected
    assert result.commit_token == tokens[expected]
    assert result.accepted_count == expected
    assert result.accepted_tokens == tokens[1:expected + 1]
    assert result.next_token == cpu_top1[expected]
    assert result.full_accept == (expected == depth)
    # Trusted GPU mode did not read a CPU oracle or inspect even top-1 values.
    assert result.gpu_accept_match_cpu == (None if accept_mode == "on" else gpu_matches)
    assert result.finite_logits is (None if accept_mode == "on" else True)
    assert result.to_json_dict()["finite_logits_scope"] == (
        "not_checked" if accept_mode == "on" else "top1_values"
    )
    assert state["row"] == expected, "canonical Conv/GDN must match selected hidden and metadata"
    cursor_owner = "host" if (accept_mode == "on" or gpu_matches) else "device"
    assert positions == [(cursor_owner, 7 + expected)]
    if mode == "verify_chain":
        assert result.selected_target_hidden_ptr == 0x1000 + expected * 4
    if accept_mode != "on" and not gpu_matches:
        assert state["device_selector"] is None
        assert not result.gpu_accept_match_cpu
    else:
        assert state["device_selector"] == 0x9000


@pytest.mark.parametrize("mode", ["verify_chain", "verify_tree"])
@pytest.mark.parametrize("accept_mode", ["validate", "off"])
@pytest.mark.parametrize("gpu_matches", [False, True])
@pytest.mark.parametrize("cpu_row", [0, 1, 2, 3])
def test_depth_three_commit_uses_every_authoritative_row(
    monkeypatch, mode, accept_mode, gpu_matches, cpu_row,
):
    test_bulk_commit_uses_authoritative_accept_row(
        monkeypatch, mode, accept_mode, gpu_matches, depth=3, cpu_row=cpu_row,
    )


@pytest.mark.parametrize("accept_mode", ["validate", "off"])
def test_graph_owned_disagreement_cannot_report_success(monkeypatch, accept_mode):
    test_bulk_commit_uses_authoritative_accept_row(
        monkeypatch, "verify_chain", accept_mode, False, native_commit=True,
    )
