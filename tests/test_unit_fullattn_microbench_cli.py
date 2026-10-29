"""The full-attention experiment CLI validates without loading HIP."""
from pathlib import Path
import ctypes
import runpy
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qwen4exp_fullattn_prefill_microbench.py"


@pytest.mark.parametrize(
    "args,code,message",
    [
        (["--help"], 0, "--qk-variants"),
        (["--qk-variants", "0,1,2"], 2, "--qk-variants requires --replay"),
    ],
)
def test_cli_does_not_load_hip_before_help_or_validation(monkeypatch, capsys, args, code, message):
    original = ctypes.CDLL

    def no_hip(name, *a, **kw):
        if "amdhip" in str(name):
            pytest.fail("CLI loaded HIP before parsing/validation")
        return original(name, *a, **kw)

    monkeypatch.setattr(ctypes, "CDLL", no_hip)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *args])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exit_info.value.code == code
    output = capsys.readouterr()
    assert message in output.out + output.err
