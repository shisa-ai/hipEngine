"""Command construction and provenance for the Gemma 4 scoreboard snapshot.

The snapshot advertises ``--gpu 0`` as the W7900 lane, but the campaign bench
validates ``--expect-gpu`` against the name of logical device 0 and defaults
to the XTX name - so a W7900 run died with "expected it to contain 'RX 7900
XTX'" instead of measuring. The expectation has to follow the selected lane.

Provenance now routes through ``scripts.gpu_smi`` so the recorded tool is the
one shipped by the active ROCm environment, and the artifact names it.
"""

from __future__ import annotations

from scripts import gemma4_scoreboard_snapshot as snapshot
from scripts import gpu_smi
from scripts.gemma4_scoreboard_snapshot import expect_gpu_for


def test_expect_gpu_follows_the_selected_lane():
    assert expect_gpu_for(0) == "W7900"
    assert expect_gpu_for(1) == "RX 7900 XTX"


def test_provenance_records_which_tool_described_the_device(monkeypatch):
    """ROCm 10.1.0 dropped rocm-smi, so the artifact must say what it used."""

    monkeypatch.setattr(snapshot, "_git", lambda *args: "")
    monkeypatch.setattr(snapshot, "_now", lambda: "2026-10-19T00:00:00Z")
    monkeypatch.setattr(snapshot, "_run", lambda command, **kwargs: "HIP version: 7.16.26385\n")
    monkeypatch.setattr(
        snapshot.gpu_smi,
        "capture_text",
        lambda **kwargs: {
            "command": "/prefix/bin/amd-smi static -a -b",
            "returncode": 0,
            "output": "GPU: 0",
            "tool": "amd-smi",
            "tool_path": "/prefix/bin/amd-smi",
            "rocm_root": "/prefix",
        },
    )
    monkeypatch.setattr(snapshot.gpu_smi, "product_name", lambda gpu, **kwargs: "AMD Radeon 8060S")

    prov = snapshot.provenance(0)

    assert prov["gpu_name"] == "AMD Radeon 8060S"
    assert prov["gpu_telemetry"]["tool"] == "amd-smi"
    assert prov["gpu_telemetry"]["tool_path"] == "/prefix/bin/amd-smi"
    assert prov["hipcc_version_first_line"] == "HIP version: 7.16.26385"
    assert prov["physical_gpu_index"] == 0


def test_provenance_asks_for_an_amd_smi_equivalent_of_the_rocm_smi_call(monkeypatch):
    """The capture must carry both invocations so either stack works."""

    captured: dict[str, object] = {}

    def fake_capture(**kwargs):
        captured.update(kwargs)
        return {"tool": "amd-smi", "output": "", "returncode": 0, "tool_path": "", "command": ""}

    monkeypatch.setattr(snapshot, "_git", lambda *args: "")
    monkeypatch.setattr(snapshot, "_now", lambda: "now")
    monkeypatch.setattr(snapshot, "_run", lambda command, **kwargs: "")
    monkeypatch.setattr(snapshot.gpu_smi, "capture_text", fake_capture)
    monkeypatch.setattr(snapshot.gpu_smi, "product_name", lambda gpu, **kwargs: "")

    snapshot.provenance(1)

    assert captured["rocm_args"] == ["--showid", "--showproductname"]
    assert captured["amd_args"] == ["static", "-a", "-b"]
    assert captured["device"] == 1


def test_provenance_fails_rather_than_borrowing_another_environments_tool(monkeypatch):
    monkeypatch.setattr(snapshot, "_git", lambda *args: "")
    monkeypatch.setattr(snapshot, "_now", lambda: "now")
    monkeypatch.setattr(snapshot, "_run", lambda command, **kwargs: "")

    def refuse(**kwargs):
        raise gpu_smi.GpuToolUnavailable("no supported GPU telemetry tool")

    monkeypatch.setattr(snapshot.gpu_smi, "capture_text", refuse)

    try:
        snapshot.provenance(0)
    except gpu_smi.GpuToolUnavailable:
        return
    raise AssertionError("provenance must propagate a missing telemetry tool")
