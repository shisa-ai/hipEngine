from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
UPDATER = REPO_ROOT / "scripts" / "update-therock-torch.sh"


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    command_env = os.environ.copy()
    command_env["PYTHON"] = sys.executable
    if env:
        command_env.update(env)
    return subprocess.run(
        ["bash", str(UPDATER), *args],
        cwd=REPO_ROOT,
        env=command_env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_device_auto_detection_prefers_configured_arch() -> None:
    result = _run(
        "--detect-device-only",
        "--json",
        env={"HIPENGINE_HIP_ARCH": "gfx1151"},
    )
    assert result.returncode == 0, result.stderr
    assert '"device": "gfx1151"' in result.stdout
    assert '"device_source": "HIPENGINE_HIP_ARCH"' in result.stdout


def test_explicit_device_overrides_configured_arch() -> None:
    result = _run(
        "--device",
        "gfx1100",
        "--detect-device-only",
        "--json",
        env={"HIPENGINE_HIP_ARCH": "gfx1151"},
    )
    assert result.returncode == 0, result.stderr
    assert '"device": "gfx1100"' in result.stdout
    assert '"device_source": "explicit --device"' in result.stdout


def test_device_auto_detection_uses_rocminfo(tmp_path: Path) -> None:
    rocminfo = tmp_path / "rocminfo"
    rocminfo.write_text(
        "#!/bin/sh\nprintf '%s\\n' '  Name: gfx1151' '  Name: amdgcn-amd-amdhsa--gfx1151'\n",
        encoding="utf-8",
    )
    rocminfo.chmod(0o755)
    result = _run(
        "--detect-device-only",
        "--json",
        env={
            "HIPENGINE_HIP_ARCH": "",
            "PATH": f"{tmp_path}:/usr/bin:/bin",
        },
    )
    assert result.returncode == 0, result.stderr
    assert '"device": "gfx1151"' in result.stdout
    assert '"device_source": "rocminfo"' in result.stdout
