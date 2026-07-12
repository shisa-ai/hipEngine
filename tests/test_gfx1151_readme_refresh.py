from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.util.amdgpu_vram import AmdgpuCard, VramSampler
from scripts.assemble_gfx1151_readme_topline import (
    _assemble_topline,
    _render_markdown,
)
from scripts.llamacpp_bench_with_peak import parse_args
from scripts.merge_readme_sweep_components import (
    STANDARD_WORKLOADS,
    _finalize_rollup,
    _finite_final_logit_passed,
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


def test_component_rollup_accepts_paro_and_gguf_finite_logit_keys() -> None:
    assert _finite_final_logit_passed({"finite_final_logit": True}) is True
    assert _finite_final_logit_passed({"finite_final_logits": True}) is True
    assert _finite_final_logit_passed({"finite_final_logit": False}) is False
    assert _finite_final_logit_passed({}) is False


def test_rollup_preserves_measured_provenance_and_separates_assembly() -> None:
    measured = _provenance()
    output = {"status": "accepted_topline", "performance_claim": True, "provenance": measured}
    assembly = {**measured, "hipengine_commit": "d" * 40, "dirty": False}

    finalized = _finalize_rollup(output, assembly_provenance=assembly)

    assert finalized["provenance"]["hipengine_commit"] == "a" * 40
    assert finalized["rollup_assembly_provenance"]["hipengine_commit"] == "d" * 40
    assert finalized["performance_claim"] is True

    dirty = _finalize_rollup(output, assembly_provenance={**assembly, "dirty": True})
    assert dirty["status"] == "rejected_topline_gate"
    assert dirty["performance_claim"] is False


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
    assert "assemble_gfx1151_readme_topline.py" in text
    assert 'memory.pop("kv_memory_audit", None)' in text
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
    assert "summary" in result.stdout
    assert "all" in result.stdout


def _fake_hipengine_rollup(*, engine: str, quant: str) -> dict[str, object]:
    provenance = _provenance()
    provenance["quant"] = quant
    if engine == "gguf":
        provenance["model_path"] = "/model.gguf"
        provenance["model_revision"] = None
        provenance["model_fingerprint"] = {
            "algorithm": "sha256-full-v1",
            "value": "c" * 64,
            "size_bytes": 2,
            "sampled_bytes": 2,
            "exists": True,
            "path_type": "file",
        }
    return {
        "schema": 1,
        "kind": "gfx1151_readme_model_sweep_rollup",
        "status": "accepted_topline",
        "performance_claim": True,
        "engine": engine,
        "quant": quant,
        "workloads": list(STANDARD_WORKLOADS),
        "warmup_runs": 2,
        "measured_runs": 5,
        "summary_by_workload": {
            workload: {
                "prefill_tok_s": {"count": 5, "median": 1000.0 + index, "stdev": 1.0},
                "decode_tok_s": {"count": 5, "median": 60.0 + index, "stdev": 0.1},
                "tracked_peak_allocated_gib": {
                    "count": 5,
                    "median": 20.0 + index,
                    "stdev": 0.0,
                },
                "final_token_ids_stable": True,
            }
            for index, workload in enumerate(STANDARD_WORKLOADS)
        },
        "correctness": {
            "passed": True,
            "all_measured_final_logits_finite": True,
            "all_workload_final_ids_stable": True,
            "all_workload_variance_gates_passed": True,
            "all_component_provenance_clean": True,
        },
        "provenance": provenance,
    }


def _fake_llamacpp_artifact(*, backend: str) -> dict[str, object]:
    provenance = _provenance()
    provenance.update(
        {
            "configured_backend": f"llamacpp_{backend}",
            "resolved_backend": "vulkan" if backend == "vulkan" else "hip_gfx1151",
            "model_path": "/model.gguf",
            "model_revision": None,
            "model_fingerprint": {
                "algorithm": "sha256-full-v1",
                "value": "c" * 64,
                "size_bytes": 2,
                "sampled_bytes": 2,
                "exists": True,
                "path_type": "file",
            },
            "quant": "gguf_q4_k_m",
            "kv_dtype": "f16/f16",
            "warmups": 1,
            "repetitions": 5,
        }
    )
    rows = []
    phase_records = []
    for index, workload in enumerate(STANDARD_WORKLOADS):
        prefill = 900.0 + index
        decode = 50.0 + index
        peak = 22.0 + index
        rows.append(
            {
                "workload": workload,
                "prefill_tok_s": prefill,
                "decode_tok_s": decode,
                "peak_vram_gib": peak,
            }
        )
        for phase, center in (("prefill", prefill), ("decode", decode)):
            phase_records.append(
                {
                    "phase": phase,
                    "workload": workload,
                    "returncode": 0,
                    "tok_s": center,
                    "vram": {"memory_domain": "gtt", "peak_gib": peak},
                    "llamacpp_record": {
                        "avg_ts": center,
                        "stddev_ts": 0.1,
                        "samples_ts": [center - 0.2, center - 0.1, center, center + 0.1, center + 0.2],
                        "build_commit": "d" * 9,
                        "build_number": 9999,
                        "gpu_info": "Radeon 8060S Graphics",
                    },
                }
            )
    return {
        "schema": 1,
        "status": "diagnostic_retained",
        "performance_claim": False,
        "backend": f"llamacpp_{backend}",
        "provenance": provenance,
        "build_commit": "d" * 9,
        "build_number": 9999,
        "gpu_info": "Radeon 8060S Graphics",
        "common_args": {
            "ngl": 99,
            "flash_attn": 1,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "repetitions": 5,
            "no_warmup": False,
        },
        "workloads_requested": list(STANDARD_WORKLOADS),
        "memory_domain": "gtt",
        "poll_ms": 10.0,
        "rows": rows,
        "phase_records": phase_records,
    }


def test_topline_assembler_promotes_only_complete_stable_four_engine_matrix(
    tmp_path: Path,
) -> None:
    sources = {
        "hipengine_paro": (tmp_path / "paro.json", _fake_hipengine_rollup(engine="paro", quant="w4_paro")),
        "hipengine_gguf": (tmp_path / "gguf.json", _fake_hipengine_rollup(engine="gguf", quant="gguf_q4_k_m")),
        "llamacpp_hip": (tmp_path / "llama-hip.json", _fake_llamacpp_artifact(backend="hip")),
        "llamacpp_vulkan": (tmp_path / "llama-vulkan.json", _fake_llamacpp_artifact(backend="vulkan")),
    }
    for path, payload in sources.values():
        path.write_text(json.dumps(payload), encoding="utf-8")

    output = _assemble_topline(sources)
    markdown = _render_markdown(output)

    assert output["status"] == "accepted_topline"
    assert output["performance_claim"] is True
    assert output["measured_hipengine_commit"] == "a" * 40
    assert output["gates"]["llamacpp_all_phase_variance_passed"] is True
    assert output["tables"]["decode_tok_s"][0]["llamacpp_vulkan"] == pytest.approx(50.0)
    assert output["tables"]["peak_gib"][-1]["hipengine_gguf"] == pytest.approx(25.0)
    assert "#### Prefill tok/s" in markdown
    assert "| 128K/128 |" in markdown

    bad = json.loads(json.dumps(sources["llamacpp_vulkan"][1]))
    bad["phase_records"][0]["llamacpp_record"]["samples_ts"][-1] *= 2
    bad_sources = dict(sources)
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    bad_sources["llamacpp_vulkan"] = (bad_path, bad)
    bad_output = _assemble_topline(bad_sources)
    assert bad_output["status"] == "rejected_topline_gate"
    assert bad_output["performance_claim"] is False
    assert bad_output["gates"]["llamacpp_all_phase_variance_passed"] is False
