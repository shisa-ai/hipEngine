from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.util.amdgpu_vram import AmdgpuCard, VramSampler
from scripts.llamacpp_bench_with_peak import parse_args
from scripts.qwen35_readme_sweep import (
    _acquire_paro_readme_graph,
    _gguf_session_identity,
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


def test_paro_sweep_reuses_one_graph_per_shape() -> None:
    graph = object()
    calls: list[dict[str, int]] = []

    class Session:
        def capture_decode_graph(self, **kwargs):
            calls.append(kwargs)
            return graph

    holder: dict[str, object] = {}
    first, first_reused, _ = _acquire_paro_readme_graph(
        session=Session(),
        graph_holder=holder,
        position=516,
        steps_per_replay=1,
        max_replay_steps=128,
    )
    second, second_reused, second_capture_seconds = _acquire_paro_readme_graph(
        session=Session(),
        graph_holder=holder,
        position=516,
        steps_per_replay=1,
        max_replay_steps=128,
    )

    assert first is second is graph
    assert first_reused is False
    assert second_reused is True
    assert second_capture_seconds == 0.0
    assert calls == [
        {
            "position": 516,
            "steps_per_replay": 1,
            "max_replay_steps": 128,
            "record_steps": 0,
        }
    ]


def test_gfx1151_readme_refresh_wrapper_encodes_retained_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts/run_gfx1151_readme_refresh.sh"
    text = script.read_text(encoding="utf-8")

    assert "HIPENGINE_HIP_ARCH=gfx1151" in text
    assert "--backend hip_gfx1151" in text
    assert "512/128 1K/128 4K/128 32K/128 64K/128 128K/128" in text
    assert "--warmup-runs 2 --measured-runs 5" in text
    assert "--memory-domain gtt" in text
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
