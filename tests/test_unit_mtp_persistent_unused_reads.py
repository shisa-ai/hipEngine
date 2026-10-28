"""Persistent PARO harness read/result/snapshot ownership without HIP."""
from __future__ import annotations

from pathlib import Path
import inspect
from types import SimpleNamespace

import pytest

from scripts import mtp_chain_e2e_smoke as bench


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
@pytest.mark.parametrize("canonicalize", [None, "0", "false", "1"])
@pytest.mark.parametrize("overlap", [None, "0", "false", "1"])
@pytest.mark.parametrize("budget,target_contract,tree_mode,fallback", [
    (1, True, "chain", 0), (3, False, "chain", 0), (3, False, "branching_topk", 0),
    (3, False, "chain", 1), (3, False, "chain", 2),
])
def test_persistent_cycles_skip_only_unused_results(monkeypatch, retired, canonicalize, overlap, budget, target_contract, tree_mode, fallback):
    if overlap is None:
        monkeypatch.delenv("HIPENGINE_MTP_OVERLAP_VERIFY_COMMIT_PROPOSER", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_MTP_OVERLAP_VERIFY_COMMIT_PROPOSER", overlap)
    if canonicalize is None:
        monkeypatch.delenv("HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY", canonicalize)
    if retired is None:
        monkeypatch.delenv("HIPENGINE_MTP_PROPOSER_SKIP_UNUSED_READS", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_MTP_PROPOSER_SKIP_UNUSED_READS", retired)
    monkeypatch.setenv("HIPENGINE_MTP_PROPOSER_TARGET_CONTRACT", str(int(target_contract)))
    advances, saves, restores, priming, accepted = [], [], [], [], []
    buffers, freed, ar_steps = [], [], []

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

        def vocab_topk(self, *, k):
            return list(range(20, 20 + k)), [10.0] + [0.0] * (k - 1)

        def save_state(self, slot):
            snapshot = (slot, self.position, self.cache_len, self.current.token)
            saves.append(snapshot)
            return snapshot

        def restore_state(self, snapshot, **kwargs):
            assert kwargs.get("stream", 0) == 0
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

        def step(self, token, **kwargs):
            ar_steps.append((token, kwargs))
            return SimpleNamespace(token_id=50 + len(ar_steps))

        def verify_chain_bulk_and_commit(self, batch, **kwargs):
            assert kwargs["canonicalize_after"] is False
            assert kwargs.get("synchronize_after_commit", True) is True
            count = (0, min(1, budget), budget)[len(accepted) % 3]
            accepted.append(count)
            return SimpleNamespace(
                accepted_count=count, accepted_tokens=tuple(range(40, 40 + count)),
                next_token=30 + len(accepted),
                target_top1=tuple(range(30, 30 + budget + 1)),
                target_top1_values=(1.0,) * (budget + 1),
                graph=None, gpu_accept_match_cpu=True,
                selected_target_hidden_ptr=0x3000,
            )

        verify_tree_bulk_and_commit = verify_chain_bulk_and_commit

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
        backend="hip_gfx1100", chain_attn_mode="c1_loop", tree_mode=tree_mode,
        confidence_threshold=0.5 if tree_mode == "branching_topk" else 0.0,
        ar_fallback_zero_streak=1 if fallback else 0,
        ar_fallback_until_end=fallback == 2,
        acceptance_diagnostics=True,
    )
    assert len(generated) == 14
    assert result["overlap_verify_commit_proposer"] is False
    assert result["canonicalize_after_verify"] is False
    assert result["proposer_skip_unused_reads"] is True
    assert priming[0]["read_expert_topk"] is False
    assert priming[0]["read_lm_head_value"] is False
    assert advances
    assert all(row["read_expert_topk"] is False for row in advances)
    assert all(row["read_lm_head_value"] is False for row in advances)
    assert all(row.get("stream", 0) == 0 for row in advances)
    if fallback != 2:
        assert any(row.get("need_result") is False for row in advances)
    if budget > 1 and fallback != 2:
        assert restores  # rejected/partially accepted cycles restore a saved prefix
    assert set(accepted) == ({0} if fallback == 2 else {0, min(1, budget), budget})
    assert result["proposal_snapshot_saves"] == len(saves)
    if budget > 1 and tree_mode == "chain" and fallback != 2:
        assert result["proposal_snapshot_skips"] > 0
        assert all(row[0] != budget - 1 for row in saves)
    assert result["ar_fallback_cycles"] == len(ar_steps)
    assert bool(ar_steps) is bool(fallback)
    if fallback:
        rows = [row for row in result["proposal_trace_sample"] if row.get("policy") == "ar_fallback"]
        assert rows and all(row["fallback_reason"] == "ar_fallback_zero_accept_streak" for row in rows)
        assert all(row["mtp_verify_cycles_before_fallback"] > 0 for row in rows)
        if fallback == 2:
            assert len(accepted) == 1
    assert freed == buffers


def test_forced_cycle_drift_switch_is_absent_from_api_and_cli(monkeypatch, capsys):
    assert "ar_fallback_after_mtp_cycles" not in inspect.signature(bench._run_spec_persistent_device).parameters
    monkeypatch.setattr("sys.argv", ["mtp_chain_e2e_smoke", "--help"])
    with pytest.raises(SystemExit) as result:
        bench.main()
    assert result.value.code == 0
    assert "--ar-fallback-after-mtp-cycles" not in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["mtp_chain_e2e_smoke", "--ar-fallback-after-mtp-cycles", "2"])
    with pytest.raises(SystemExit) as result:
        bench.main()
    assert result.value.code == 2
    assert "unrecognized arguments: --ar-fallback-after-mtp-cycles 2" in capsys.readouterr().err
