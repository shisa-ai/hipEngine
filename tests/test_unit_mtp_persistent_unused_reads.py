"""Persistent PARO harness read/result/snapshot ownership without HIP."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import mtp_chain_e2e_smoke as bench


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
@pytest.mark.parametrize("canonicalize", [None, "0", "false", "1"])
@pytest.mark.parametrize("budget,target_contract", [(1, True), (3, False)])
def test_persistent_cycles_skip_only_unused_results(monkeypatch, retired, canonicalize, budget, target_contract):
    if canonicalize is None:
        monkeypatch.delenv("HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY", canonicalize)
    if retired is None:
        monkeypatch.delenv("HIPENGINE_MTP_PROPOSER_SKIP_UNUSED_READS", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_MTP_PROPOSER_SKIP_UNUSED_READS", retired)
    monkeypatch.setenv("HIPENGINE_MTP_PROPOSER_TARGET_CONTRACT", str(int(target_contract)))
    monkeypatch.setenv("HIPENGINE_MTP_OVERLAP_VERIFY_COMMIT_PROPOSER", "0")
    advances, saves, restores, priming, accepted = [], [], [], [], []
    buffers, freed = [], []

    class Proposer:
        draft_vocab = 1000

        def __init__(self, *args, **kwargs):
            self.position = 1
            self.cache_len = 1
            self.current = SimpleNamespace(token=20)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def prefill_from_target_hidden_rows(self, *args, **kwargs):
            priming.append(kwargs)

        def save_state(self, slot):
            snapshot = (slot, self.position, self.cache_len, self.current.token)
            saves.append(snapshot)
            return snapshot

        def restore_state(self, snapshot, **kwargs):
            restores.append(snapshot)
            _, self.position, self.cache_len, token = snapshot
            self.current = SimpleNamespace(token=token)

        def advance_with_previous_hidden(self, **kwargs):
            advances.append(kwargs)
            self.position = kwargs["position"]
            self.cache_len += 1
            if kwargs.get("need_result", True):
                self.current = SimpleNamespace(token=int(kwargs["input_token"]) + 1)
            return self.current

        advance_with_target_hidden = advance_with_previous_hidden

    class Session:
        config = SimpleNamespace(hidden_size=4)
        layer_limit = 1
        runtime = object()
        compiler_version = None
        lm_head_weight = SimpleNamespace(tensor=SimpleNamespace(ptr=0x1000))
        lm_head_scale = SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000))
        vocab_size = 1000
        lm_head_threads = 256

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def step_with_hidden_taps(self, *args, **kwargs):
            return SimpleNamespace(token_id=10)

        def verify_chain_bulk_and_commit(self, batch, **kwargs):
            assert kwargs["canonicalize_after"] is False
            count = (0, min(1, budget), budget)[len(accepted) % 3]
            accepted.append(count)
            return SimpleNamespace(
                accepted_count=count, next_token=30 + len(accepted),
                target_top1=tuple(range(30, 30 + budget + 1)),
                target_top1_values=(1.0,) * (budget + 1),
                graph=None, gpu_accept_match_cpu=True,
                selected_target_hidden_ptr=0x3000,
            )

    def allocate(nbytes, **kwargs):
        result = SimpleNamespace(ptr=0x4000, nbytes=nbytes)
        buffers.append(result)
        return result

    monkeypatch.setattr(bench, "Qwen35ParoNextTokenRunner", lambda *a, **k: None)
    monkeypatch.setattr(bench, "Qwen35ParoResidentSession", Session)
    monkeypatch.setattr(bench, "NativeMtpChainProposer", Proposer)
    monkeypatch.setattr(bench, "malloc", allocate)
    monkeypatch.setattr(bench, "free", lambda buf, **kwargs: freed.append(buf))
    monkeypatch.setattr(bench, "_RoctxProfilerWindow", lambda **kwargs: SimpleNamespace(range_push=lambda *a: None, range_pop=lambda: None))

    generated, result = bench._run_spec_persistent_device(
        Path("model"), (7,), decode_tokens=14, candidate_budget=budget,
        backend="hip_gfx1100", chain_attn_mode="c1_loop",
    )
    assert len(generated) == 14
    assert result["canonicalize_after_verify"] is False
    assert result["proposer_skip_unused_reads"] is True
    assert priming[0]["read_expert_topk"] is False
    assert priming[0]["read_lm_head_value"] is False
    assert advances
    assert all(row["read_expert_topk"] is False for row in advances)
    assert all(row["read_lm_head_value"] is False for row in advances)
    assert any(row.get("need_result") is False for row in advances)
    if budget > 1:
        assert restores  # rejected/partially accepted cycles restore a saved prefix
    assert set(accepted) == {0, min(1, budget), budget}
    assert result["proposal_snapshot_saves"] == len(saves)
    if budget > 1:
        assert result["proposal_snapshot_skips"] > 0
        assert all(row[0] != budget - 1 for row in saves)
    assert freed == buffers
