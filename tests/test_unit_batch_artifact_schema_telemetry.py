"""GPU telemetry tool acceptance in the c>N artifact schema.

ROCm 10.1.0 removed ROCm SMI, so a c>N artifact produced on a current stack
declares ``amd-smi`` where older artifacts declare ``rocm-smi``. The schema has
to accept either while still refusing an artifact that never recorded telemetry,
and it must keep reading pre-existing artifacts that carry no ``tool`` field.
"""

from __future__ import annotations

from scripts.qwen35_batch_artifact_schema import (
    _GPU_TELEMETRY_COMMAND_FRAGMENTS,
    _REQUIRED_ACCEPTED_ENVIRONMENT_TELEMETRY_COMMANDS,
    _validate_capture_context,
)

AMD_CAPTURE = {
    "command": "/prefix/bin/amd-smi metric --mem-usage --usage --temperature",
    "returncode": 0,
    "output": "GPU: 0\n    TOTAL_VRAM: 512 MB",
    "tool": "amd-smi",
}

ROCM_CAPTURE = {
    "command": "/prefix/bin/rocm-smi --showmeminfo vram --showuse --showtemp",
    "returncode": 0,
    "output": "GPU[0] : VRAM Total Memory (B): 536870912",
}


def _fragments_ok(command: str, tool: str) -> list[str]:
    errors: list[str] = []
    _validate_capture_context("hardware.rocm_smi", {"command": command, "returncode": 0, "output": "x"}, errors, command_fragment=tool)
    return errors


def test_both_tools_have_declared_command_fragments():
    assert _GPU_TELEMETRY_COMMAND_FRAGMENTS["rocm-smi"] == (
        "rocm-smi",
        "--showmeminfo",
        "vram",
        "--showuse",
        "--showtemp",
    )
    assert _GPU_TELEMETRY_COMMAND_FRAGMENTS["amd-smi"] == (
        "amd-smi",
        "metric",
        "--mem-usage",
        "--usage",
        "--temperature",
    )


def test_environment_requires_one_telemetry_invocation_from_either_tool():
    commands = set(_REQUIRED_ACCEPTED_ENVIRONMENT_TELEMETRY_COMMANDS.values())
    assert commands == {
        "rocm-smi --showmeminfo vram --showuse --showtemp",
        "amd-smi metric --mem-usage --usage --temperature",
    }


def test_an_amd_smi_capture_satisfies_its_own_fragments():
    assert _fragments_ok(AMD_CAPTURE["command"], "amd-smi") == []


def test_a_rocm_smi_capture_satisfies_its_own_fragments():
    assert _fragments_ok(ROCM_CAPTURE["command"], "rocm-smi") == []


def test_a_capture_that_declares_the_wrong_tool_is_rejected():
    """A record naming amd-smi but running rocm-smi is a provenance defect."""

    errors = _fragments_ok(ROCM_CAPTURE["command"], "amd-smi")

    assert errors, "a mismatched tool/command pair must not validate"
