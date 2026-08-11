from __future__ import annotations

import argparse
import json
import time

from hipengine.util.amdgpu_vram import AmdgpuCard
from scripts import llamacpp_mtp_bench as bench


def test_llamacpp_mtp_natural_summary_reports_accepted_per_output() -> None:
    rows = [
        {
            "timings": {
                "predicted_n": 10,
                "predicted_ms": 100.0,
                "predicted_per_second": 100.0,
                "draft_n": 8,
                "draft_n_accepted": 4,
            }
        },
        {
            "timings": {
                "predicted_n": 5,
                "predicted_ms": 50.0,
                "predicted_per_second": 100.0,
                "draft_n": 2,
                "draft_n_accepted": 1,
            }
        },
    ]

    summary = bench._summarize_rows(rows)

    assert summary["draft_acceptance"] == 0.5
    assert summary["accepted_per_output"] == 5 / 15
    assert summary["first_output_tokens_untimed"] == 2
    assert summary["timed_decode_transitions"] == 13
    assert summary["transition_normalized_predicted_per_second"] == 13 / 0.15
    assert summary["aggregate_decode_transition_per_second"] == 13 / 0.15
    assert summary["denominators"] == {
        "draft_acceptance": "draft_n_accepted / draft_n",
        "accepted_per_output": "draft_n_accepted / predicted_n",
        "native_predicted_per_second": "predicted_n / predicted_ms",
        "transition_normalized_predicted_per_second": (
            "(predicted_n - one prompt-produced first output token per request) / predicted_ms"
        ),
    }
    assert summary["timing_boundary"]["cross_engine_rule"] == (
        "request N+1 outputs and report N timed transitions per prompt"
    )


def test_llamacpp_mtp_natural_summary_reports_concurrent_client_wall() -> None:
    rows = [
        {
            "wall_s": 0.8,
            "timings": {
                "predicted_n": 24,
                "predicted_ms": 300.0,
                "predicted_per_second": 80.0,
                "draft_n": 16,
                "draft_n_accepted": 12,
            },
        },
        {
            "wall_s": 0.9,
            "timings": {
                "predicted_n": 24,
                "predicted_ms": 320.0,
                "predicted_per_second": 75.0,
                "draft_n": 18,
                "draft_n_accepted": 13,
            },
        },
    ]

    summary = bench._summarize_rows(rows, client_wall_s=0.95, concurrency=2, aggregate_decode_ms=320.0)

    assert summary["concurrency"] == 2
    assert summary["aggregate_decode_ms_total"] == 320.0
    assert summary["aggregate_decode_predicted_per_second"] == 150.0
    assert summary["timed_decode_transitions"] == 46
    assert summary["aggregate_decode_transition_per_second"] == 143.75
    assert summary["client_wall_s_total"] == 0.95
    assert summary["request_wall_s_total"] == 1.7000000000000002
    assert summary["client_aggregate_predicted_per_second"] == 48 / 0.95


def test_llamacpp_mtp_token_repeat_summary_reports_accepted_per_output() -> None:
    rows = [
        {"tokens_predicted": 10, "predicted_ms": 100.0, "draft_n": 8, "draft_n_accepted": 4},
        {"tokens_predicted": 6, "predicted_ms": 60.0, "draft_n": 4, "draft_n_accepted": 2},
    ]

    summary = bench._summarize_token_repeat(rows)

    assert summary["draft_acceptance"] == 0.5
    assert summary["accepted_per_output"] == 6 / 16
    assert summary["denominators"] == {
        "draft_acceptance": "draft_n_accepted / draft_n",
        "accepted_per_output": "draft_n_accepted / tokens_predicted",
    }


def test_llamacpp_mtp_artifact_summary_and_text_include_accepted_per_output() -> None:
    artifact = {
        "runs": {
            "base": {
                "protocols": {
                    "natural": {
                        "summary": {
                            "predicted_per_second_weighted": 50.0,
                            "aggregate_decode_predicted_per_second": 100.0,
                            "aggregate_decode_transition_per_second": 96.0,
                            "client_aggregate_predicted_per_second": 45.0,
                            "accepted_per_output": None,
                        }
                    },
                    "token_repeat": {
                        "summary": {
                            "weighted_predicted_per_second": 40.0,
                            "accepted_per_output": None,
                        }
                    },
                }
            },
            "mtp": {
                "protocols": {
                    "natural": {
                        "summary": {
                            "predicted_per_second_weighted": 75.0,
                            "aggregate_decode_predicted_per_second": 180.0,
                            "aggregate_decode_transition_per_second": 168.0,
                            "client_aggregate_predicted_per_second": 90.0,
                            "draft_acceptance": 0.5,
                            "accepted_per_output": 0.25,
                        }
                    },
                    "token_repeat": {
                        "summary": {
                            "weighted_predicted_per_second": 80.0,
                            "draft_acceptance": 0.75,
                            "accepted_per_output": 0.5,
                        }
                    },
                }
            },
        }
    }

    artifact["summary"] = bench._summarize_artifact(artifact)
    text = bench._summary_text(artifact)

    assert artifact["summary"]["natural"]["mtp_accepted_per_output"] == 0.25
    assert artifact["summary"]["natural"]["aggregate_decode_speedup"] == 1.8
    assert artifact["summary"]["natural"]["transition_normalized_speedup"] == 1.75
    assert artifact["summary"]["natural"]["client_aggregate_speedup"] == 2.0
    assert artifact["summary"]["token_repeat"]["mtp_accepted_per_output"] == 0.5
    assert "agg_decode_speedup=1.800x" in text
    assert "transition_speedup=1.750x" in text
    assert "client_speedup=2.000x" in text
    assert "accepted/output=0.250" in text
    assert "accepted/output=0.500" in text


def test_llamacpp_mtp_row_helpers_handle_missing_denominators() -> None:
    assert bench._accepted_per_output({"predicted_n": 0, "draft_n_accepted": 1}) is None
    assert bench._summarize_rows([{"timings": {}}])["accepted_per_output"] is None
    assert bench._summarize_token_repeat([{}])["accepted_per_output"] is None


def test_llamacpp_mtp_stage_timing_summary_excludes_first_task(tmp_path) -> None:
    path = tmp_path / "llama-stage.jsonl"
    rows = [
        {
            "task_id": 0,
            "visible_output_tokens": 1,
            "accepted_draft_tokens": 0,
            "generated_draft_tokens": 1,
            "cycle_wall_ms": 10.0,
            "target_verify_layer_passes": 1,
            "target_verify_rows_evaluated": 2,
            "target_verify_discarded_rows": 1,
            "stage_timings_ms": {"draft_initial": 1.0, "target_block_verify_total": 9.0},
        },
        {
            "task_id": 1,
            "visible_output_tokens": 2,
            "accepted_draft_tokens": 1,
            "generated_draft_tokens": 2,
            "draft_token_ids": [10, 11],
            "sampled_token_ids": [10, 99],
            "accepted_token_ids": [10],
            "output_token_ids": [10, 99],
            "bonus_token_id": 99,
            "rejected_draft_token_id": 11,
            "cycle_wall_ms": 30.0,
            "target_verify_layer_passes": 1,
            "target_verify_rows_evaluated": 3,
            "target_verify_discarded_rows": 1,
            "stage_timings_ms": {
                "draft_initial": 4.0,
                "mtp_context_replay_append": 20.0,
                "target_block_forward": 2.0,
                "target_block_verify_total": 22.0,
            },
        },
        {
            "task_id": 1,
            "visible_output_tokens": 1,
            "accepted_draft_tokens": 0,
            "generated_draft_tokens": 1,
            "cycle_wall_ms": 15.0,
            "target_verify_layer_passes": 1,
            "target_verify_rows_evaluated": 2,
            "target_verify_discarded_rows": 1,
            "stage_timings_ms": {
                "draft_initial": 2.0,
                "target_block_forward": 10.0,
                "target_block_verify_total": 10.0,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = bench._summarize_stage_timings(path)
    measured = summary["measured_excluding_first_task"]

    assert summary["rows_total"] == 3
    assert summary["rows_measured"] == 2
    assert summary["warmup_task_id_excluded"] == 0
    assert measured["total_output_tokens"] == 3
    assert measured["accepted_per_output"] == 1 / 3
    assert measured["draft_acceptance"] == 1 / 3
    assert measured["cycle_wall_ms_per_output"] == 15.0
    assert measured["target_verify_layer_passes_per_output"] == 2 / 3
    assert measured["target_verify_rows_per_output"] == 5 / 3
    assert measured["cycle_histograms"] == {
        "generated_draft_tokens": {"1": 1, "2": 1},
        "accepted_draft_tokens": {"0": 1, "1": 1},
        "visible_output_tokens": {"1": 1, "2": 1},
        "target_verify_layer_passes": {"1": 2},
        "target_verify_rows_evaluated": {"2": 1, "3": 1},
        "target_verify_block_rows": {"0": 2},
        "target_verify_discarded_rows": {"1": 2},
        "target_verify_rows_minus_visible_output": {"1": 2},
    }
    assert measured["stage_timing_totals_ms"] == {
        "draft_initial": 6.0,
        "mtp_context_replay_append": 20.0,
        "target_block_forward": 12.0,
        "target_block_verify_total": 32.0,
    }
    assert measured["stage_timing_per_output_ms"]["target_block_verify_total"] == 32 / 3
    assert measured["token_trace_rows"] == 1
    assert measured["proposal_trace_sample"] == [
        {
            "task_id": 1,
            "cycle": None,
            "checkpoint_restore": False,
            "generated_draft_tokens": 2,
            "accepted_draft_tokens": 1,
            "visible_output_tokens": 2,
            "draft_token_ids": [10, 11],
            "sampled_token_ids": [10, 99],
            "accepted_token_ids": [10],
            "output_token_ids": [10, 99],
            "bonus_token_id": 99,
            "rejected_draft_token_id": 11,
        }
    ]


def test_llamacpp_mtp_server_command_passes_extra_args_after_mtp_flags() -> None:
    args = argparse.Namespace(
        server_bin="/tmp/llama-server",
        model="/tmp/model.gguf",
        gpu_layers=99,
        flash_attn="on",
        cache_type_k="f16",
        cache_type_v="f16",
        ctx_size=8192,
        concurrency=4,
        host="127.0.0.1",
        port=8013,
        alias="qwen",
        draft_max=2,
        server_extra_arg=["--reasoning", "off"],
    )

    cmd = bench._server_command(args, "mtp")

    assert cmd[cmd.index("-c") + 1] == str(8192 * 4)
    assert cmd[cmd.index("-np") + 1] == "4"
    assert cmd[-6:] == [
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "2",
        "--reasoning",
        "off",
    ]


def test_llamacpp_config_records_memory_sampling() -> None:
    args = argparse.Namespace(
        server_bin="llama-server",
        model="model.gguf",
        alias="model",
        host="127.0.0.1",
        port=8011,
        ctx_size=8192,
        concurrency=4,
        gpu_layers=99,
        flash_attn="on",
        cache_type_k="f16",
        cache_type_v="f16",
        draft_max=2,
        protocol="natural",
        prompts=bench.DEFAULT_PROMPTS,
        max_tokens=25,
        seed=12345,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        min_p=0.0,
        token_id=9707,
        shapes=["512/128"],
        server_extra_arg=[],
        stage_timings_jsonl=None,
        stage_token_trace=False,
        sample_memory=True,
        poll=5.0,
        memory_domain="vram",
        card_name="card0",
        pci_id=None,
        card_index=None,
    )

    config = bench._config_json(args)

    assert config["ctx_size_per_sequence"] == 8192
    assert config["server_ctx_size"] == 32768
    assert config["concurrency"] == 4
    assert config["memory_sampling"] == {
        "enabled": True,
        "poll_ms": 5.0,
        "memory_domain": "vram",
        "card_name": "card0",
        "pci_id": None,
        "card_index": None,
    }


def _fake_amdgpu_card(tmp_path, *, used_bytes: int = 100) -> AmdgpuCard:
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "mem_info_vram_total").write_text("1000\n")
    (device / "mem_info_vram_used").write_text(f"{used_bytes}\n")
    return AmdgpuCard(
        card_name="card0",
        pci_id="0000:10:00.0",
        sysfs_path=device,
        vram_total_bytes=1000,
    )


def test_server_memory_recorder_reports_process_and_phase_fake_sysfs(tmp_path) -> None:
    card = _fake_amdgpu_card(tmp_path)
    used_path = card.vram_used_path
    recorder = bench._ServerMemoryRecorder(
        card,
        interval_ms=1.0,
        memory_domain="vram",
    )

    recorder.start_process()
    recorder.start_phase("startup")
    used_path.write_text("400\n")
    time.sleep(0.01)
    recorder.stop_phase("startup")
    recorder.start_phase("teardown")
    used_path.write_text("100\n")
    time.sleep(0.005)
    recorder.stop_phase("teardown")
    recorder.stop_process()
    payload = recorder.to_dict()

    assert payload["card"]["card_name"] == "card0"
    assert payload["card"]["pci_id"] == "0000:10:00.0"
    assert payload["process"]["baseline_bytes"] == 100
    assert payload["process"]["peak_bytes"] == 400
    assert payload["process"]["final_bytes"] == 100
    assert payload["process"]["peak_delta_bytes"] == 300
    assert payload["process"]["samples_count"] >= 3
    assert payload["phases"]["startup"]["peak_bytes"] == 400
    assert payload["phases"]["startup"]["scope"] == (
        "before llama-server Popen through successful health readiness"
    )
    assert payload["phases"]["teardown"]["final_bytes"] == 100


def test_run_server_mode_samples_fake_subprocess_lifecycle(tmp_path, monkeypatch) -> None:
    card = _fake_amdgpu_card(tmp_path)
    used_path = card.vram_used_path
    recorder = bench._ServerMemoryRecorder(
        card,
        interval_ms=1.0,
        memory_domain="vram",
    )
    popen_calls = []

    class FakeProcess:
        pass

    def fake_popen(command, *, stdout, stderr, env):
        popen_calls.append((command, stdout, stderr, env))
        return FakeProcess()

    def fake_health(_host, _port, _timeout):
        used_path.write_text("300\n")
        time.sleep(0.005)

    def fake_natural(_args):
        used_path.write_text("500\n")
        time.sleep(0.005)
        return {"summary": {}}

    def fake_terminate(_process):
        used_path.write_text("100\n")
        time.sleep(0.005)

    monkeypatch.setattr(bench.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(bench, "_wait_for_health", fake_health)
    monkeypatch.setattr(bench, "_run_natural", fake_natural)
    monkeypatch.setattr(bench, "_terminate", fake_terminate)
    args = argparse.Namespace(
        server_bin="/tmp/llama.cpp/build/bin/llama-server",
        model="model.gguf",
        alias="model",
        host="127.0.0.1",
        port=8011,
        ctx_size=8192,
        concurrency=1,
        gpu_layers=99,
        flash_attn="on",
        cache_type_k="f16",
        cache_type_v="f16",
        draft_max=2,
        server_extra_arg=[],
        stage_timings_jsonl=None,
        stage_token_trace=False,
        server_start_timeout=5.0,
    )

    payload = bench._run_server_mode(
        args=args,
        mode="base",
        protocols=["natural"],
        logs_dir=tmp_path,
        memory_recorder=recorder,
    )

    assert len(popen_calls) == 1
    assert payload["protocols"]["natural"] == {"summary": {}}
    assert payload["memory"]["process"]["baseline_bytes"] == 100
    assert payload["memory"]["process"]["peak_bytes"] == 500
    assert payload["memory"]["process"]["final_bytes"] == 100
    assert payload["memory"]["phases"]["startup"]["peak_bytes"] >= 300
    assert payload["memory"]["phases"]["natural"]["peak_bytes"] == 500
    assert payload["memory"]["phases"]["teardown"]["final_bytes"] == 100
    assert payload["memory"]["scope"]["process"] == (
        "before llama-server Popen through post-termination sysfs sample"
    )
