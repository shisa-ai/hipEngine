"""Command construction for the Gemma 4 scoreboard snapshot.

The snapshot advertises ``--gpu 0`` as the W7900 lane, but the campaign bench
validates ``--expect-gpu`` against the name of logical device 0 and defaults
to the XTX name - so a W7900 run died with "expected it to contain 'RX 7900
XTX'" instead of measuring. The expectation has to follow the selected lane.
"""

from __future__ import annotations

from scripts import gemma4_scoreboard_snapshot as snapshot
from scripts.gemma4_scoreboard_snapshot import expect_gpu_for


class _Done:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _install(monkeypatch, outputs, *, which):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        value = outputs.get(command[0], "")
        if isinstance(value, _Done):
            return value
        return _Done(stdout=value)

    monkeypatch.setattr(snapshot.subprocess, "run", fake_run)
    monkeypatch.setattr(snapshot.shutil, "which", which)
    monkeypatch.setattr(snapshot, "_gpu_name", lambda gpu: "AMD Radeon 8060S")
    monkeypatch.setattr(snapshot, "_git", lambda *args: "")
    return calls


def test_expect_gpu_follows_the_selected_lane():
    assert expect_gpu_for(0) == "W7900"
    assert expect_gpu_for(1) == "RX 7900 XTX"


def test_provenance_records_a_missing_rocm_smi_without_failing(monkeypatch):
    """ROCm 10.1.0 dropped rocm-smi; a snapshot must still capture provenance."""

    calls = _install(
        monkeypatch,
        {"hipcc": "HIP version: 7.16.26385\n", "amd-smi": "GPU: 0\n"},
        which=lambda name: None if name == "rocm-smi" else f"/prefix/bin/{name}",
    )

    prov = snapshot.provenance(0)

    assert prov["rocm_smi"] == []
    assert prov["rocm_smi_available"] is False
    assert prov["rocm_smi_path"] == ""
    assert prov["amd_smi_available"] is True
    assert prov["amd_smi_path"] == "/prefix/bin/amd-smi"
    assert prov["amd_smi"] == ["GPU: 0"]
    assert prov["hipcc_version_first_line"] == "HIP version: 7.16.26385"
    assert ["rocm-smi", "--showid", "--showproductname"] not in calls


def test_provenance_records_where_a_rocm_smi_binary_came_from(monkeypatch):
    """A stale /opt/rocm can still provide rocm-smi; the path must be visible."""

    _install(
        monkeypatch,
        {
            "hipcc": "HIP version: 7.16\n",
            "rocm-smi": "GPU[0] : Device Name: fake\n",
            "amd-smi": "GPU: 0\n",
        },
        which=lambda name: (
            "/opt/rocm/bin/rocm-smi" if name == "rocm-smi" else f"/prefix/bin/{name}"
        ),
    )

    prov = snapshot.provenance(0)

    assert prov["rocm_smi_available"] is True
    assert prov["rocm_smi_path"] == "/opt/rocm/bin/rocm-smi"
    assert prov["rocm_smi"] == ["GPU[0] : Device Name: fake"]


def test_provenance_marks_a_failing_tool_unavailable(monkeypatch):
    _install(
        monkeypatch,
        {
            "hipcc": "HIP version: 7.16\n",
            "rocm-smi": _Done(returncode=1, stderr="unsupported"),
            "amd-smi": "GPU: 0\n",
        },
        which=lambda name: f"/prefix/bin/{name}",
    )

    prov = snapshot.provenance(0)

    assert prov["rocm_smi_available"] is False
    assert prov["rocm_smi"] == []
    assert prov["rocm_smi_path"] == "/prefix/bin/rocm-smi"
