from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts import gguf_sh_m1_screen as screen


def _fingerprint(value: str = "same") -> dict[str, object]:
    return {"nbytes": 4, "blake2b_128": value, "finite": True}


def _state_child(query_rows: int, *, changed: bool = False) -> dict[str, object]:
    fingerprint = _fingerprint("changed" if changed else "same")
    checkpoint = {
        "position": 4096,
        "current_token_id": 9707,
        "predicted_token_id": 9707,
        "finite": True,
        "hidden_seed": fingerprint,
        "layer_outputs": [],
        "linear_states": [],
        "kv_states": [],
    }
    return {
        "kind": screen.STATE_KIND,
        "schema_version": screen.SCHEMA_VERSION,
        "mode": screen.mode_for_query_rows(query_rows),
        "query_rows": query_rows,
        "resolved_backend": "hip_gfx1151",
        "target_arch": "gfx1151",
        "prefill_chunk_sizes": screen.chunk_sizes(query_rows),
        "bulk_prefill_scratch_rows": query_rows,
        "bulk_prefill_scratch_capacity": 65792,
        "contexts": [
            {
                "prompt_length": 4096,
                "finite": True,
                "prefill_logits": fingerprint,
                "prefill_state": checkpoint,
                "trajectory": [
                    {
                        "transition": 0,
                        "input_token_id": 9707,
                        "predicted_token_id": 9707,
                        "logits": fingerprint,
                    }
                ],
                "final_state": checkpoint,
            }
        ],
        "provenance": {"dirty": False},
    }


def _benchmark_leg(query_rows: int, *, context: int = 4096) -> dict[str, object]:
    chunks = screen.chunk_sizes(query_rows)
    expected_scratch = min(4352, max(1024, query_rows))
    breakdown = {
        "families": {
            "session_buffers": {
                "bulk_prefill_scratch_rows": expected_scratch,
                "bulk_prefill_scratch_census": {
                    "rows": expected_scratch,
                    "physical_owner_bytes": int(1.75 * (1 << 30)),
                },
            }
        }
    }
    run = {
        "resolved_backend": "hip_gfx1151",
        "target_arch": "gfx1151",
        "prefill_chunk_sizes": {**chunks, "attn_aotriton_min_tokens": 512, "auto_tune": True, "chunk_tune_min_tokens": 1025},
        "effective_use_wmma_prefill": True,
        "effective_use_gemv_decode": True,
        "correctness_sanity": {"final_token_id": 9707, "finite_final_logits": True},
        "memory_snapshots": {
            "after_load": {"owned_session_breakdown": breakdown},
        },
    }
    tracked = 23.0 if query_rows == 4096 else 21.7
    prefill = 1400.0 if query_rows == 4096 else 1395.0
    decode = 55.0 if query_rows == 4096 else 54.8
    return {
        "persistent_session": True,
        "warmup_runs": 1,
        "measured_runs": 3,
        "prompt_length": context,
        "decode_tokens": 128,
        "warmup_decode_tokens": 1,
        "resolved_backend": "hip_gfx1151",
        "target_arch": "gfx1151",
        "requested_prefill_chunk_sizes": chunks,
        "prefill_chunk_sizes_all": [run["prefill_chunk_sizes"]] * 4,
        "effective_use_wmma_prefill_all": [True] * 4,
        "effective_use_gemv_decode_all": [True] * 4,
        "runs": [run] * 4,
        "summary": {
            "prefill_tok_s": {"median": prefill},
            "decode_tok_s": {"median": decode},
            "tracked_peak_allocated_gib": {"median": tracked},
            "owned_session_peak_gib": {"median": tracked},
            "finite_final_logits_all": True,
            "final_token_ids": [9707, 9707, 9707],
        },
        "persistent_session_memory": {
            "summary": {"tracked_current_allocated_bytes_after_close": 0},
            "snapshots": {
                "before_load": {"tracked": {"current_allocated_bytes": 0}},
                "after_close": {"tracked": {"current_allocated_bytes": 0}},
            },
        },
        "provenance": {"dirty": False},
    }


def test_benchmark_command_sets_every_chunk_surface_explicitly(tmp_path: Path) -> None:
    command = screen.build_benchmark_command(
        python="python3",
        model=Path("/models/model.gguf"),
        prompt_length=4096,
        query_rows=1024,
        decode_tokens=128,
        warmup_decode_tokens=1,
        warmup_runs=1,
        measured_runs=3,
        compiler_version_file=Path("/tmp/hipcc.txt"),
        output=tmp_path / "candidate.json",
    )

    joined = " ".join(command)
    assert "--persistent-session" in command
    assert "--require-cached-build" in command
    for flag in (
        "--prefill-linear-chunk-size 1024",
        "--prefill-moe-chunk-size 1024",
        "--prefill-full-attn-query-chunk-size 1024",
        "--prefill-full-attn-post-chunk-size 1024",
        "--prefill-full-attn-rope-chunk-size 1024",
    ):
        assert flag in joined


def test_validate_benchmark_leg_fails_closed_on_resolved_scratch_rows() -> None:
    payload = _benchmark_leg(1024)
    screen.validate_benchmark_leg(
        payload,
        query_rows=1024,
        prompt_length=4096,
        decode_tokens=128,
        warmup_decode_tokens=1,
        warmup_runs=1,
        measured_runs=3,
    )

    broken = copy.deepcopy(payload)
    broken["runs"][0]["memory_snapshots"]["after_load"]["owned_session_breakdown"]["families"]["session_buffers"]["bulk_prefill_scratch_rows"] = 4096
    with pytest.raises(screen.ScreenError, match="scratch rows"):
        screen.validate_benchmark_leg(
            broken,
            query_rows=1024,
            prompt_length=4096,
            decode_tokens=128,
            warmup_decode_tokens=1,
            warmup_runs=1,
            measured_runs=3,
        )


def test_prefill_hidden_seed_replay_populates_chunked_outer_checkpoint() -> None:
    calls: list[tuple[int, int, bool]] = []

    class Session:
        _last_target_hidden_ptr = 0x1000
        scratch = type("Scratch", (), {"norm": type("Buffer", (), {"ptr": 0x2000})()})()
        runtime = type("Runtime", (), {"device_synchronize": lambda self: None})()
        ready = False

        def fp32_hidden_seed_contract(self):
            return type("Contract", (), {"ready_for_mtp": self.ready})()

        def _run_output_norm_hidden(self, src_ptr, out_ptr, *, capture_hidden_seed_fp32):
            calls.append((src_ptr, out_ptr, capture_hidden_seed_fp32))
            self.ready = True

    session = Session()
    replayed = screen.ensure_prefill_hidden_seed_capture(session)

    assert replayed is True
    assert calls == [(0x1000, 0x2000, True)]
    assert screen.ensure_prefill_hidden_seed_capture(session) is False


def test_compare_state_children_requires_byte_exact_prefill_trajectory_and_state() -> None:
    baseline = _state_child(4096)
    candidate = _state_child(1024)

    comparison = screen.compare_state_children(
        baseline,
        candidate,
        expected_contexts=(4096,),
    )
    assert comparison["passed"] is True
    assert comparison["contexts"][0]["prefill_logits_exact"] is True
    assert comparison["contexts"][0]["final_state_exact"] is True

    changed = _state_child(1024, changed=True)
    rejected = screen.compare_state_children(
        baseline,
        changed,
        expected_contexts=(4096,),
    )
    assert rejected["passed"] is False
    assert rejected["contexts"][0]["prefill_logits_exact"] is False


def test_classify_promotes_only_exact_one_gib_pareto_point() -> None:
    baseline = _benchmark_leg(4096)
    candidate = _benchmark_leg(1024)
    row = screen.summarize_context(
        prompt_length=4096,
        baseline=baseline,
        candidate=candidate,
        baseline_gtt={"peak_gib": 23.5},
        candidate_gtt={"peak_gib": 22.1},
    )
    state = screen.compare_state_children(
        _state_child(4096),
        _state_child(1024),
        expected_contexts=(4096,),
    )

    decision = screen.classify_screen(
        [row],
        state_comparison=state,
        provenance={"dirty": False},
        long_context_min=4096,
        min_tracked_savings_gib=1.0,
        max_prefill_loss_pct=1.0,
        max_decode_loss_pct=1.0,
    )
    assert decision["status"] == "promote_q1024"
    assert decision["measurement_valid"] is True
    assert decision["promotion_passed"] is True

    regressed = copy.deepcopy(row)
    regressed["comparison"]["prefill_loss_pct"] = 1.01
    rejected = screen.classify_screen(
        [regressed],
        state_comparison=state,
        provenance={"dirty": False},
        long_context_min=4096,
        min_tracked_savings_gib=1.0,
        max_prefill_loss_pct=1.0,
        max_decode_loss_pct=1.0,
    )
    assert rejected["status"] == "reject_prefill_regression"
    assert rejected["measurement_valid"] is True
    assert rejected["promotion_passed"] is False
