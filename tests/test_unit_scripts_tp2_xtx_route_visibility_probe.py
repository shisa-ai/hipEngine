"""CPU tests for the XTX visibility probe: mapping logic only, no HIP runtime."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load("tp2_xtx_route_visibility_probe", SCRIPTS / "tp2_xtx_route_visibility_probe.py")


class _Completed:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def test_main_maps_hip_visible_devices_1_to_the_xtx(probe, monkeypatch, tmp_path, capsys):
    def fake_run(command, *, env, **kwargs):
        if env.get("HIP_VISIBLE_DEVICES") == "1" or env.get("ROCR_VISIBLE_DEVICES") == "1":
            payload = {
                "count": 1,
                "devices": [
                    {
                        "index": 0,
                        "name": "AMD Radeon RX 7900 XTX",
                        "uuid": "cc4d02090dc9c3ff",
                        "uuid_hex": "cc4d02090dc9c3ff",
                        "pci_bus_id": "0000:10:00.0",
                    }
                ],
            }
        elif env.get("HIP_VISIBLE_DEVICES") == "0":
            payload = {
                "count": 1,
                "devices": [
                    {
                        "index": 0,
                        "name": "AMD Radeon Pro W7900",
                        "uuid": "e282895b62c2b295",
                        "uuid_hex": "e282895b62c2b295",
                        "pci_bus_id": "0000:0d:00.0",
                    }
                ],
            }
        else:
            payload = {
                "count": 2,
                "devices": [
                    {"index": 0, "name": "AMD Radeon Pro W7900", "uuid": "a", "uuid_hex": "a", "pci_bus_id": "0000:0d:00.0"},
                    {"index": 1, "name": "AMD Radeon RX 7900 XTX", "uuid": "b", "uuid_hex": "b", "pci_bus_id": "0000:10:00.0"},
                ],
            }
        return _Completed(json.dumps(payload))

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    out = tmp_path / "visibility.json"
    code = probe.main(["--json", str(out)])
    capsys.readouterr()
    assert code == 0
    artifact = json.loads(out.read_text())
    assert artifact["hip_visible_devices_1_logical0_is_xtx"] is True
    assert len(artifact["combinations"]) == 5


def test_main_reports_false_when_logical0_is_not_the_xtx(probe, monkeypatch, tmp_path, capsys):
    def fake_run(command, *, env, **kwargs):
        payload = {
            "count": 1,
            "devices": [
                {"index": 0, "name": "AMD Radeon Pro W7900", "uuid": "a", "uuid_hex": "a", "pci_bus_id": "0000:0d:00.0"}
            ],
        }
        return _Completed(json.dumps(payload))

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    out = tmp_path / "visibility.json"
    probe.main(["--json", str(out)])
    capsys.readouterr()
    artifact = json.loads(out.read_text())
    assert artifact["hip_visible_devices_1_logical0_is_xtx"] is False


def test_enumeration_failure_is_recorded_not_raised(probe, monkeypatch):
    monkeypatch.setattr(
        probe.subprocess, "run", lambda command, **kwargs: _Completed("", returncode=1, stderr="boom")
    )
    observed = probe.enumerate_devices({"HIP_VISIBLE_DEVICES": "1"})
    assert observed["error"] == "rc=1" and "boom" in observed["stderr"]
