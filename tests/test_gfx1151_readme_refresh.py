from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.util.amdgpu_vram import AmdgpuCard, VramSampler
from scripts.llamacpp_bench_with_peak import parse_args
from scripts.merge_readme_sweep_components import (
    STANDARD_WORKLOADS,
    _merge_component_payloads,
)
from scripts.qwen35_readme_sweep import (
    _acquire_paro_readme_graph,
    _gguf_session_identity,
    _measured_graph_replay_requested,
    _summarize_runs,
)


def _fake_card(tmp_path: Path) -> AmdgpuCard:
    device = tmp_path / "device"
    device.mkdir()
    (device / "mem_info_vram_total").write_text(str(512 << 20), encoding="utf-8")
    (device / "mem_info_vram_used").write_text(str(128 << 20), encoding="utf-8")
    (device / "mem_info_gtt_total").write_text(str(120 << 30), encoding="utf-8")
    (device / "mem_info_gtt_used").write_text(str(20 << 30), encoding="utf-8")
    return AmdgpuCard(
        card_name="card1",
        pci_id="0000:c1:00.0",
        sysfs_path=device,
        vram_total_bytes=512 << 20,
    )


def test_amdgpu_sampler_can_measure_gtt_for_uma(tmp_path: Path) -> None:
    card = _fake_card(tmp_path)
    sampler = VramSampler(card, interval_ms=1, memory_domain="gtt")

    sampler.start()
    (card.sysfs_path / "mem_info_gtt_used").write_text(
        str(21 << 30), encoding="utf-8"
    )
    sampler.stop()
    result = sampler.result()

    assert result.memory_domain == "gtt"
    assert result.memory_total_bytes == 120 << 30
    assert result.peak_bytes == 21 << 30
    assert result.to_dict()["memory_domain"] == "gtt"


def test_llamacpp_peak_parser_accepts_gtt_domain(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--llama-bench",
            "/bin/true",
            "--model",
            str(tmp_path / "model.gguf"),
            "--memory-domain",
            "gtt",
        ]
    )

    assert args.memory_domain == "gtt"


def test_readme_sweep_summary_includes_sampled_device_peak() -> None:
    summary = _summarize_runs(
        [
            {
                "throughput": {"prefill_tok_s": 10.0, "decode_tok_s": 20.0},
                "memory": {
                    "tracked_peak_allocated_gib": 21.0,
                    "hip_used_peak_sampled_gib": 21.5,
                },
                "correctness_sanity": {"final_token_id": 7},
            },
            {
                "throughput": {"prefill_tok_s": 12.0, "decode_tok_s": 22.0},
                "memory": {
                    "tracked_peak_allocated_gib": 22.0,
                    "hip_used_peak_sampled_gib": 22.5,
                },
                "correctness_sanity": {"final_token_id": 7},
            },
        ]
    )

    assert summary["hip_used_peak_sampled_gib"]["median"] == pytest.approx(22.0)


def test_gguf_sweep_snapshots_identity_before_session_close() -> None:
    session = SimpleNamespace(
        backend="hip_gfx1151",
        runner=SimpleNamespace(target_arch="gfx1151"),
    )

    identity = _gguf_session_identity(session)
    session.runner = None

    assert identity == ("hip_gfx1151", "gfx1151")


def test_paro_sweep_uses_fresh_measured_graphs() -> None:
    graph = object()
    calls: list[dict[str, int]] = []

    class Session:
        def capture_decode_graph(self, **kwargs):
            calls.append(kwargs)
            return graph

    first, first_reused, _ = _acquire_paro_readme_graph(
        session=Session(),
        graph_holder=None,
        position=516,
        steps_per_replay=1,
        max_replay_steps=128,
    )
    second, second_reused, second_capture_seconds = _acquire_paro_readme_graph(
        session=Session(),
        graph_holder=None,
        position=516,
        steps_per_replay=1,
        max_replay_steps=128,
    )

    assert first is second is graph
    assert first_reused is False
    assert second_reused is False
    assert second_capture_seconds >= 0.0
    assert calls == [
        {
            "position": 516,
            "steps_per_replay": 1,
            "max_replay_steps": 128,
            "record_steps": 0,
        },
        {
            "position": 516,
            "steps_per_replay": 1,
            "max_replay_steps": 128,
            "record_steps": 0,
        }
    ]
    assert _measured_graph_replay_requested(requested=True, measured=False) is False
    assert _measured_graph_replay_requested(requested=True, measured=True) is True


def _provenance() -> dict[str, object]:
    return {
        "kind": "hipengine_artifact_provenance",
        "schema_version": 1,
        "collected_at": "2026-07-11T00:00:00+00:00",
        "repo_root": "/repo",
        "hipengine_commit": "a" * 40,
        "git_branch": None,
        "staged_dirty": False,
        "unstaged_dirty": False,
        "untracked_dirty": False,
        "untracked_count": 0,
        "dirty": False,
        "configured_backend": "hip_gfx1151",
        "resolved_backend": "hip_gfx1151",
        "target_arch": "gfx1151",
        "device_name": "Radeon 8060S Graphics",
        "model_path": "/model",
        "model_revision": "revision",
        "model_fingerprint": {
            "algorithm": "sha256-full-v1",
            "value": "b" * 64,
            "size_bytes": 1,
            "sampled_bytes": 1,
            "exists": True,
            "path_type": "directory",
        },
        "quant": "w4_paro",
        "kv_dtype": "bf16",
        "command": ["python3", "sweep.py"],
        "environment": {"HIPENGINE_HIP_ARCH": "gfx1151"},
        "rocm_version": "7.13",
        "hipcc_version": "HIP 7.13",
        "build_profile": "readme_resident_sweep",
        "timing_protocol": "test",
        "warmups": 2,
        "repetitions": 5,
        "profiler": {"enabled": False},
    }


def test_component_rollup_promotes_six_clean_right_sized_workloads(
    tmp_path: Path,
) -> None:
    components = []
    provenance = _provenance()
    for index, workload in enumerate(STANDARD_WORKLOADS, start=1):
        runs = [
            {
                "measured": True,
                "correctness_sanity": {
                    "finite_final_logit": True,
                    "final_token_id": index,
                },
            }
            for _ in range(5)
        ]
        payload = {
            "engine": "paro",
            "model": "/model",
            "quant": "w4_paro",
            "workloads": [workload],
            "provenance": provenance,
            "summary_by_workload": {
                workload: {
                    "prefill_tok_s": {
                        "count": 5,
                        "median": 100.0,
                        "stdev": 1.0,
                    },
                    "decode_tok_s": {
                        "count": 5,
                        "median": 50.0,
                        "stdev": 0.1,
                    },
                    "final_token_ids": [index] * 5,
                    "final_token_ids_stable": True,
                }
            },
            "runs_by_workload": {workload: runs},
            "max_sequence_length": index * 1024,
            "persistent_session_load_seconds": 1.0,
            "persistent_session_memory": {"summary": {"peak": index}},
            "extra": {"backend": "hip_gfx1151", "target_arch": "gfx1151"},
        }
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        components.append((path, payload))

    output = _merge_component_payloads(
        components,
        engine="paro",
        provenance=provenance,
    )

    assert output["status"] == "accepted_topline"
    assert output["performance_claim"] is True
    assert output["correctness"]["passed"] is True
    assert list(output["summary_by_workload"]) == list(STANDARD_WORKLOADS)
    assert output["persistent_session_memory_by_workload"]["128K/128"] == {
        "summary": {"peak": 6}
    }


def test_gfx1151_readme_refresh_wrapper_encodes_retained_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts/run_gfx1151_readme_refresh.sh"
    text = script.read_text(encoding="utf-8")

    assert "HIPENGINE_HIP_ARCH=gfx1151" in text
    assert "--backend hip_gfx1151" in text
    assert "512/128 1K/128 4K/128 32K/128 64K/128 128K/128" in text
    assert "--warmup-runs 2 --measured-runs 5" in text
    assert "--memory-domain gtt" in text
    assert "merge_readme_sweep_components.py" in text
    assert "llamacpp-hip" in text
    assert "llamacpp-vulkan" in text

    result = subprocess.run(
        ["bash", str(script), "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "hipengine" in result.stdout
    assert "llamacpp" in result.stdout
    assert "all" in result.stdout
