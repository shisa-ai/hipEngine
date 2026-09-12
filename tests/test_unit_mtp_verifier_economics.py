from __future__ import annotations

import json
from types import SimpleNamespace

from scripts import mtp_verifier_economics as tool


def test_verifier_economics_forwards_registered_profile_to_smoke(
    tmp_path, monkeypatch
) -> None:
    captured: list[str] = []

    def fake_run(command, **_kwargs):
        captured.extend(str(item) for item in command)
        output = command[command.index("--json") + 1]
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "status": "passed",
                    "exact_ar_match": True,
                    "decode_tokens": 2,
                    "candidate_budget": 1,
                    "ar": {"decode_seconds": 0.02, "decode_tok_s": 100.0},
                    "mtp": {
                        "accepted_lengths": [1],
                        "active_budgets": [1],
                        "decode_seconds": 0.01,
                        "decode_tok_s": 200.0,
                        "verify_seconds": 0.008,
                        "target_prefill_seconds": 0.03,
                        "proposal_prefill_seconds": 0.01,
                        "proposal_decode_update_seconds": 0.002,
                        "cycle_marker_ns": [
                            {"start_perf_ns": 1, "end_perf_ns": 10_000_001}
                        ],
                    },
                },
                handle,
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    args = SimpleNamespace(
        model=tmp_path / "model",
        decode_tokens=2,
        proposal_impl="persistent_device",
        backend="hip_gfx1100",
        hip_arch="gfx1100",
        execution_profile="production",
        chain_attn_mode="decode_batched",
        graph_mode="off",
        active_budget_cap=0,
        acceptance_diagnostics=False,
        confidence_threshold=0.0,
        ar_fallback_zero_streak=0,
        ar_fallback_after_mtp_cycles=0,
        ar_fallback_tokens=1,
        ar_fallback_until_end=False,
        small_batch_decode_threshold=7,
        verify_gpu_accept=None,
        llama_target_cycle_cost=2.0,
    )

    _smoke, metrics = tool._run_one(
        args,
        budget=1,
        run_idx=1,
        prompt_tokens="1,2",
        raw_root=tmp_path,
    )

    assert captured[captured.index("--execution-profile") + 1] == "production"
    assert metrics["exact_ar_match"] is True
    assert metrics["target_prefill_seconds"] == 0.03
    assert metrics["proposal_prefill_seconds"] == 0.01
