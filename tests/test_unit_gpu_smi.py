"""GPU telemetry resolution and normalization.

The load-bearing property is provenance: a tool must belong to the ROCm
environment this process is measuring. A ``rocm-smi`` from ``/opt/rocm`` (a
system ROCm 7.2.3 on the gfx1151 host) reports SMI-LIB 7.8.0 and has nothing to
do with a ROCm 10.1.0 prefix, so resolution must refuse it rather than record
it as if it described the stack under test.

Fixtures below are real tool output captured from the 10.1.0 ``amd-smi``
(27.1.0) and the 10.0.0 backup prefix's ``rocm-smi`` (4.0.0).
"""

from __future__ import annotations

import json
import stat

import pytest

from scripts import gpu_smi

AMD_STATIC = {
    "gpu_data": [
        {
            "gpu": 0,
            "asic": {
                "market_name": "AMD Radeon 8060S Graphics",
                "device_id": "0x1586",
                "target_graphics_version": "gfx1151",
                "num_compute_units": 40,
            },
            "driver": {
                "name": "amdgpu",
                "version": "N/A",
                "os_kernel_version": "7.3.0-rc2-1-mainline",
            },
        }
    ]
}

AMD_MEM = {
    "gpu_data": [
        {
            "gpu": 0,
            "mem_usage": {
                "total_vram": {"value": 512, "unit": "MB"},
                "used_vram": {"value": 360, "unit": "MB"},
                "free_vram": {"value": 152, "unit": "MB"},
            },
        }
    ]
}

AMD_METRIC = {
    "gpu_data": [
        {
            "gpu": 0,
            "power": {
                "socket_power": {"value": 65, "unit": "W"},
                "apu_average_socket_power": {"value": 64.66, "unit": "W"},
            },
            "clock": {
                "gfx_0": {"clk": "N/A"},
                "mem_0": {"clk": "N/A"},
                "apu_average_gfxclk_frequency": {"value": 1089, "unit": "MHz"},
            },
            "temperature": {"edge": {"value": 48, "unit": "C"}},
        }
    ]
}

ROCM_PRODUCT = {
    "card0": {
        "Card Series": "AMD Radeon 8060S Graphics",
        "Card Model": "0x1586",
        "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
        "GFX Version": "gfx1151",
    },
    "system": {"Driver version": "7.3.0-rc2-1-mainline"},
}

ROCM_MEM = {
    "card0": {
        "VRAM Total Memory (B)": "536870912",
        "VRAM Total Used Memory (B)": "377163776",
    }
}

ROCM_METRIC = {
    "card0": {
        "Temperature (Sensor edge) (C)": "50.0",
        "mclk clock speed:": "(1000Mhz)",
        "sclk clock speed:": "(851Mhz)",
        "Current Socket Graphics Package Power (W)": "64.607",
    }
}


def _make_root(tmp_path, *tools: str):
    """Build a fake ROCm root containing the named ``bin/`` tools."""

    root = tmp_path / "rocm"
    (root / "bin").mkdir(parents=True, exist_ok=True)
    for name in tools:
        tool = root / "bin" / name
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return root


@pytest.fixture(autouse=True)
def _clear_rocm_env(monkeypatch):
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("ROCM_HOME", raising=False)


def test_prefers_amd_smi_when_the_environment_ships_both(tmp_path, monkeypatch):
    root = _make_root(tmp_path, "amd-smi", "rocm-smi")
    monkeypatch.setenv("ROCM_PATH", str(root))

    tool = gpu_smi.resolve_gpu_tool()

    assert tool.name == "amd-smi"
    assert tool.path == (root / "bin" / "amd-smi").resolve()
    assert tool.root == root.resolve()


def test_falls_back_to_rocm_smi_on_an_older_stack(tmp_path, monkeypatch):
    root = _make_root(tmp_path, "rocm-smi")
    monkeypatch.setenv("ROCM_PATH", str(root))

    tool = gpu_smi.resolve_gpu_tool()

    assert tool.name == "rocm-smi"


def test_refuses_a_tool_that_only_exists_on_path(tmp_path, monkeypatch):
    """A binary from another install must never be used, even though it runs."""

    root = _make_root(tmp_path)
    elsewhere = tmp_path / "system-rocm" / "bin"
    elsewhere.mkdir(parents=True)
    foreign = elsewhere / "rocm-smi"
    foreign.write_text("#!/bin/sh\necho 'GPU[0] : Card Series: wrong stack'\n")
    foreign.chmod(foreign.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setenv("ROCM_PATH", str(root))
    monkeypatch.setenv("PATH", f"{elsewhere}:{tmp_path}")

    with pytest.raises(gpu_smi.GpuToolUnavailable) as excinfo:
        gpu_smi.resolve_gpu_tool()

    message = str(excinfo.value)
    assert "no supported GPU telemetry tool" in message
    assert str(root / "bin" / "amd-smi") in message
    assert "PATH" in message


def test_try_resolve_reports_the_reason_instead_of_raising(tmp_path, monkeypatch):
    root = _make_root(tmp_path)
    monkeypatch.setenv("ROCM_PATH", str(root))

    tool, reason = gpu_smi.try_resolve_gpu_tool()

    assert tool is None
    assert reason is not None and "no supported GPU telemetry tool" in reason


def test_rocm_root_comes_from_the_running_interpreter(monkeypatch, tmp_path):
    """Without ROCM_PATH the root follows this process's own environment."""

    root = _make_root(tmp_path, "amd-smi")

    class _Completed:
        returncode = 0
        stdout = str(root) + "\n"
        stderr = ""

    monkeypatch.setattr(gpu_smi.subprocess, "run", lambda *a, **k: _Completed())

    assert gpu_smi.active_rocm_root() == root.resolve()


def test_rocm_root_failure_names_the_interpreter(monkeypatch):
    class _Completed:
        returncode = 1
        stdout = ""
        stderr = "No module named rocm_sdk"

    monkeypatch.setattr(gpu_smi.subprocess, "run", lambda *a, **k: _Completed())

    with pytest.raises(gpu_smi.GpuToolUnavailable) as excinfo:
        gpu_smi.active_rocm_root()

    assert "rocm_sdk" in str(excinfo.value)


def _stub_invoke(monkeypatch, responses):
    calls: list[list[str]] = []

    def fake_invoke(tool, args, *, timeout):
        calls.append([tool.name, *args])
        for matcher, payload in responses.items():
            if matcher in args:
                if isinstance(payload, tuple):
                    return payload
                return 0, json.dumps(payload)
        return 1, ""

    monkeypatch.setattr(gpu_smi, "_invoke", fake_invoke)
    return calls


def test_capture_text_uses_each_tool_s_own_arguments(tmp_path, monkeypatch):
    root = _make_root(tmp_path, "amd-smi")
    monkeypatch.setenv("ROCM_PATH", str(root))
    _stub_invoke(monkeypatch, {"--mem-usage": (0, "")})

    record = gpu_smi.capture_text(
        rocm_args=["--showmeminfo", "vram", "--showuse"],
        amd_args=["metric", "--mem-usage"],
    )

    assert record["tool"] == "amd-smi"
    assert "metric --mem-usage" in record["command"]
    assert record["returncode"] == 0


def test_capture_text_records_which_tool_produced_it(tmp_path, monkeypatch):
    root = _make_root(tmp_path, "rocm-smi")
    monkeypatch.setenv("ROCM_PATH", str(root))
    _stub_invoke(monkeypatch, {"--showmeminfo": (0, "GPU[0] : VRAM Total")})

    record = gpu_smi.capture_text(rocm_args=["--showmeminfo", "vram"])

    assert record["tool"] == "rocm-smi"
    assert record["tool_path"] == str((root / "bin" / "rocm-smi").resolve())
    assert "VRAM" in record["output"]


@pytest.mark.parametrize("tool_name", ["amd-smi", "rocm-smi"])
def test_device_identity_has_one_shape_for_either_tool(tmp_path, monkeypatch, tool_name):
    root = _make_root(tmp_path, tool_name)
    monkeypatch.setenv("ROCM_PATH", str(root))
    responses = (
        {"static": AMD_STATIC} if tool_name == "amd-smi" else {"--showproductname": ROCM_PRODUCT}
    )
    _stub_invoke(monkeypatch, responses)

    identity = gpu_smi.device_identity()

    assert identity is not None
    assert identity["product_name"] == "AMD Radeon 8060S Graphics"
    assert identity["gfx_target"] == "gfx1151"
    assert identity["driver_version"] == "7.3.0-rc2-1-mainline"
    assert identity["source"] == tool_name


@pytest.mark.parametrize("tool_name", ["amd-smi", "rocm-smi"])
def test_memory_usage_normalizes_to_bytes(tmp_path, monkeypatch, tool_name):
    root = _make_root(tmp_path, tool_name)
    monkeypatch.setenv("ROCM_PATH", str(root))
    responses = (
        {"--mem-usage": AMD_MEM} if tool_name == "amd-smi" else {"--showmeminfo": ROCM_MEM}
    )
    _stub_invoke(monkeypatch, responses)

    usage = gpu_smi.memory_usage()

    assert usage is not None
    assert usage["vram_total_bytes"] == 512 * 1024**2
    assert usage["vram_used_bytes"] is not None
    assert usage["vram_free_bytes"] is not None
    assert usage["source"] == tool_name


def test_memory_usage_derives_free_bytes_for_rocm_smi(tmp_path, monkeypatch):
    root = _make_root(tmp_path, "rocm-smi")
    monkeypatch.setenv("ROCM_PATH", str(root))
    _stub_invoke(monkeypatch, {"--showmeminfo": ROCM_MEM})

    usage = gpu_smi.memory_usage()

    assert usage is not None
    assert usage["vram_total_bytes"] == 536870912
    assert usage["vram_used_bytes"] == 377163776
    assert usage["vram_free_bytes"] == 536870912 - 377163776


@pytest.mark.parametrize("tool_name", ["amd-smi", "rocm-smi"])
def test_metrics_normalizes_clocks_power_and_temperature(tmp_path, monkeypatch, tool_name):
    root = _make_root(tmp_path, tool_name)
    monkeypatch.setenv("ROCM_PATH", str(root))
    responses = (
        {"--clock": AMD_METRIC} if tool_name == "amd-smi" else {"--showclocks": ROCM_METRIC}
    )
    _stub_invoke(monkeypatch, responses)

    values = gpu_smi.metrics()

    assert values is not None
    assert values["socket_power_w"] is not None
    assert values["edge_temp_c"] is not None
    assert values["source"] == tool_name


def test_amd_smi_metrics_fall_back_to_apu_clocks(tmp_path, monkeypatch):
    """Discrete gfx clocks read N/A on an APU; the APU averages carry the value."""

    root = _make_root(tmp_path, "amd-smi")
    monkeypatch.setenv("ROCM_PATH", str(root))
    _stub_invoke(monkeypatch, {"--clock": AMD_METRIC})

    values = gpu_smi.metrics()

    assert values is not None
    assert values["sclk_mhz"] == 1089.0
    assert values["edge_temp_c"] == 48.0


def test_accessors_raise_when_the_environment_has_no_tool(tmp_path, monkeypatch):
    """Absence must be loud: never "no data" while another install answers."""

    root = _make_root(tmp_path)
    monkeypatch.setenv("ROCM_PATH", str(root))

    for accessor in (gpu_smi.product_name, gpu_smi.device_identity, gpu_smi.memory_usage, gpu_smi.metrics):
        with pytest.raises(gpu_smi.GpuToolUnavailable):
            accessor()

    with pytest.raises(gpu_smi.GpuToolUnavailable):
        gpu_smi.capture_text(rocm_args=["--showmeminfo", "vram"])

    with pytest.raises(gpu_smi.GpuToolUnavailable):
        gpu_smi.command_line(rocm_args=["--showmeminfo", "vram"])
