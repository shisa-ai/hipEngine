"""Environment-anchored GPU telemetry for hipEngine scripts and benchmarks.

hipEngine records GPU identity and memory state next to benchmark results. Two
CLIs can supply it:

* ``amd-smi`` -- AMD SMI, the supported tool since ROCm 10.1.0 stopped shipping
  ROCM SMI.
* ``rocm-smi`` -- ROCM SMI, shipped through ROCm 10.0.0 and still present on the
  retained W7900 ROCm 7.13 lane.

Both are resolved from the *active* ROCm root only. A tool discovered anywhere
else -- ``/opt/rocm``, another conda prefix, a ``PATH`` shim -- reports on a
different software stack, so recording its output silently corrupts provenance.
Resolution therefore never consults ``PATH``. When the active environment has
neither tool, resolution raises :class:`GpuToolUnavailable` and names what it
looked for; it does not degrade to some other installation.

Normalized accessors return one shape for either tool, so callers never branch
on which one is installed. :func:`capture_text` exists for artifact fields that
record raw tool output and must keep their original text form.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

#: Preference order. AMD SMI is the supported tool; ROCM SMI is the fallback for
#: stacks that predate its removal.
TOOL_PREFERENCE: tuple[str, ...] = ("amd-smi", "rocm-smi")

AMD_SMI = "amd-smi"
ROCM_SMI = "rocm-smi"


class GpuToolUnavailable(RuntimeError):
    """No supported GPU telemetry tool exists in the active ROCm environment."""


@dataclass(frozen=True)
class GpuTool:
    """A GPU telemetry CLI that belongs to the active ROCm environment."""

    name: str
    path: Path
    root: Path

    @property
    def is_amd_smi(self) -> bool:
        return self.name == AMD_SMI

    def describe(self) -> dict[str, str]:
        return {"tool": self.name, "tool_path": str(self.path), "rocm_root": str(self.root)}


def active_rocm_root() -> Path:
    """Root of the ROCm installation belonging to this process's environment.

    ``ROCM_PATH``/``ROCM_HOME`` win when set, because they are an explicit
    declaration of which stack the caller means. Otherwise the running
    interpreter is asked, which anchors the answer to the environment actually
    executing the script rather than to whatever is first on ``PATH``.
    """

    for variable in ("ROCM_PATH", "ROCM_HOME"):
        declared = os.environ.get(variable)
        if declared and (Path(declared) / "bin").is_dir():
            return Path(declared).resolve()

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "rocm_sdk", "path", "--root"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuToolUnavailable(
            f"cannot resolve the active ROCm root: {sys.executable} -m rocm_sdk "
            f"failed with {type(exc).__name__}: {exc}"
        ) from exc
    root = (completed.stdout or "").strip()
    if completed.returncode != 0 or not root:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        raise GpuToolUnavailable(
            f"cannot resolve the active ROCm root from {sys.executable}: "
            f"`rocm_sdk path --root` exited {completed.returncode}. "
            f"Is this interpreter a TheRock ROCm environment? {' '.join(detail)}"
        )
    return Path(root).resolve()


def resolve_gpu_tool(preference: Sequence[str] = TOOL_PREFERENCE) -> GpuTool:
    """Return the GPU telemetry tool shipped by the active ROCm environment.

    Raises :class:`GpuToolUnavailable` when the environment provides none of the
    candidates. It never falls back to ``PATH``: a tool from another install
    would describe a stack this process is not measuring.
    """

    root = active_rocm_root()
    searched: list[str] = []
    for name in preference:
        candidate = root / "bin" / name
        searched.append(str(candidate))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return GpuTool(name=name, path=candidate.resolve(), root=root)

    raise GpuToolUnavailable(
        "the active ROCm environment ships no supported GPU telemetry tool "
        f"(root={root}); looked for: {', '.join(searched)}. "
        "Refusing to use a tool resolved from PATH, because it would report on a "
        "different ROCm installation. Install or activate an environment that "
        "provides amd-smi (ROCm 10.1.0 and newer) or rocm-smi (older stacks)."
    )


def try_resolve_gpu_tool(
    preference: Sequence[str] = TOOL_PREFERENCE,
) -> tuple[GpuTool | None, str | None]:
    """Resolve without raising.

    Returns ``(tool, None)`` on success and ``(None, reason)`` when the active
    environment provides no supported tool. Callers that legitimately record
    absence (environment inventories) use this; artifact producers should call
    :func:`resolve_gpu_tool` and fail loudly.
    """

    try:
        return resolve_gpu_tool(preference), None
    except GpuToolUnavailable as exc:
        return None, str(exc)


def _invoke(tool: GpuTool, args: Sequence[str], *, timeout: float) -> tuple[int, str]:
    completed = subprocess.run(
        [str(tool.path), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return int(completed.returncode), output


def capture_text(
    *,
    rocm_args: Sequence[str],
    amd_args: Sequence[str] | None = None,
    timeout: float = 10.0,
    device: int | None = None,
) -> dict[str, Any]:
    """Capture raw tool output for an artifact field.

    ``rocm_args`` and ``amd_args`` are the equivalent invocation for each tool;
    ``amd_args`` defaults to ``rocm_args`` when the two CLIs accept the same
    flags. The returned record keeps the ``command``/``returncode``/``output``
    shape existing artifacts use and adds which tool produced it.
    """

    tool = resolve_gpu_tool()
    args = list(rocm_args if not tool.is_amd_smi else (amd_args or rocm_args))
    if device is not None and tool.is_amd_smi and "--gpu" not in args:
        args.extend(["--gpu", str(device)])
    returncode, output = _invoke(tool, args, timeout=timeout)
    return {
        "command": shlex.join([str(tool.path), *args]),
        "returncode": returncode,
        "output": output.strip(),
        **tool.describe(),
    }


def command_argv(
    *,
    rocm_args: Sequence[str],
    amd_args: Sequence[str] | None = None,
    preference: Sequence[str] = TOOL_PREFERENCE,
) -> list[str]:
    """Full argv for this capture, with the tool's absolute path.

    Use this when handing the invocation to an existing runner: resolving the
    tool here keeps the run anchored to the active environment instead of
    letting a subprocess search ``PATH`` and find a different install.

    ``preference`` narrows the candidates, for callers that record one specific
    tool (an environment inventory listing ``amd-smi`` separately from
    ``rocm-smi``). It still resolves inside the active environment only.
    """

    tool = resolve_gpu_tool(preference)
    args = rocm_args if not tool.is_amd_smi else (amd_args or rocm_args)
    return [str(tool.path), *args]


def command_line(
    *, rocm_args: Sequence[str], amd_args: Sequence[str] | None = None
) -> str:
    """The invocation an artifact should declare for this capture.

    Artifacts record which commands produced them, so the declaration has to
    follow the tool actually installed rather than naming ``rocm-smi`` on a stack
    that no longer ships it.
    """

    tool = resolve_gpu_tool()
    args = rocm_args if not tool.is_amd_smi else (amd_args or rocm_args)
    return shlex.join([tool.name, *args])


def _parse_json(output: str) -> Any | None:
    text = output.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some rocm-smi builds prefix a banner line before the JSON document.
        start = text.find("{")
        if start <= 0:
            return None
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            return None


def _capture_json(
    tool: GpuTool, args: Sequence[str], *, timeout: float
) -> Any | None:
    returncode, output = _invoke(tool, args, timeout=timeout)
    if returncode != 0:
        return None
    return _parse_json(output)


def _amd_entry(payload: Any, device: int) -> dict[str, Any] | None:
    """Select one GPU from an ``amd-smi`` ``gpu_data`` document."""

    if not isinstance(payload, dict):
        return None
    entries = payload.get("gpu_data")
    if not isinstance(entries, list) or not entries:
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("gpu") == device:
            return entry
    first = entries[0]
    return first if isinstance(first, dict) else None


def _rocm_card(payload: Any, device: int) -> dict[str, Any] | None:
    """Select one card from a ``rocm-smi`` ``cardN`` document."""

    if not isinstance(payload, dict):
        return None
    card = payload.get(f"card{device}")
    if isinstance(card, dict):
        return card
    for key, value in payload.items():
        if key.startswith("card") and isinstance(value, dict):
            return value
    return None


def _amd_quantity(value: Any) -> tuple[float | None, str | None]:
    """Unwrap an ``amd-smi`` ``{"value": N, "unit": "U"}`` quantity."""

    if not isinstance(value, dict):
        return None, None
    raw = value.get("value")
    unit = value.get("unit")
    if isinstance(raw, (int, float)):
        return float(raw), unit if isinstance(unit, str) else None
    return None, unit if isinstance(unit, str) else None


def _to_bytes(value: float | None, unit: str | None) -> int | None:
    if value is None:
        return None
    scale = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }.get((unit or "B").upper())
    if scale is None:
        return None
    return int(value * scale)


def _first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _int_or_none(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    return int(match.group(0)) if match else None


def device_identity(device: int = 0, *, timeout: float = 10.0) -> dict[str, Any] | None:
    """Product name, device id, GFX target, and driver for one GPU.

    Returns ``None`` only when the tool runs but its output cannot be parsed.
    A missing tool raises :class:`GpuToolUnavailable`, so absence is never
    confused with "no data".
    """

    tool = resolve_gpu_tool()

    if tool.is_amd_smi:
        payload = _capture_json(tool, ["static", "-a", "-d", "--json"], timeout=timeout)
        entry = _amd_entry(payload, device)
        if entry is None:
            return None
        asic = entry.get("asic") or {}
        driver = entry.get("driver") or {}
        return {
            "product_name": asic.get("market_name"),
            "device_id": asic.get("device_id"),
            "gfx_target": asic.get("target_graphics_version"),
            "driver_version": driver.get("os_kernel_version") or driver.get("version"),
            "source": tool.name,
            "tool_path": str(tool.path),
        }

    payload = _capture_json(
        tool, ["--showproductname", "--showdriverversion", "--json"], timeout=timeout
    )
    card = _rocm_card(payload, device)
    if card is None:
        return None
    system = payload.get("system") if isinstance(payload, dict) else None
    return {
        "product_name": card.get("Card Series"),
        "device_id": card.get("Card Model"),
        "gfx_target": card.get("GFX Version"),
        "driver_version": (system or {}).get("Driver version"),
        "source": tool.name,
        "tool_path": str(tool.path),
    }


def product_name(device: int = 0, *, timeout: float = 10.0) -> str | None:
    """Physical product name of one GPU, or ``None`` when it cannot be parsed."""

    identity = device_identity(device, timeout=timeout)
    if not identity:
        return None
    name = identity.get("product_name")
    return name if isinstance(name, str) and name.strip() else None


def memory_usage(device: int = 0, *, timeout: float = 15.0) -> dict[str, Any] | None:
    """VRAM total/used/free in bytes for one GPU.

    A phase measured while another process still holds VRAM is not a throughput
    result, so free VRAM is recorded to let a caller reject such a phase.
    """

    tool = resolve_gpu_tool()

    if tool.is_amd_smi:
        payload = _capture_json(tool, ["metric", "--mem-usage", "--json"], timeout=timeout)
        entry = _amd_entry(payload, device)
        if entry is None:
            return None
        usage = entry.get("mem_usage") or {}
        total, total_unit = _amd_quantity(usage.get("total_vram"))
        used, used_unit = _amd_quantity(usage.get("used_vram"))
        free, free_unit = _amd_quantity(usage.get("free_vram"))
        return {
            "vram_total_bytes": _to_bytes(total, total_unit),
            "vram_used_bytes": _to_bytes(used, used_unit),
            "vram_free_bytes": _to_bytes(free, free_unit),
            "source": tool.name,
            "tool_path": str(tool.path),
        }

    payload = _capture_json(tool, ["--showmeminfo", "vram", "--json"], timeout=timeout)
    card = _rocm_card(payload, device)
    if card is None:
        return None
    total_text = card.get("VRAM Total Memory (B)")
    used_text = card.get("VRAM Total Used Memory (B)")
    total_bytes = _int_or_none(str(total_text)) if total_text is not None else None
    used_bytes = _int_or_none(str(used_text)) if used_text is not None else None
    free_bytes = (
        total_bytes - used_bytes
        if total_bytes is not None and used_bytes is not None
        else None
    )
    return {
        "vram_total_bytes": total_bytes,
        "vram_used_bytes": used_bytes,
        "vram_free_bytes": free_bytes,
        "source": tool.name,
        "tool_path": str(tool.path),
    }


def metrics(device: int = 0, *, timeout: float = 15.0) -> dict[str, Any] | None:
    """Clocks, socket power, and edge temperature for one GPU.

    Clock and power reporting differs between discrete and APU parts, so each
    field is filled from the most specific source the tool provides and left
    ``None`` when the tool reports ``N/A``.
    """

    tool = resolve_gpu_tool()

    if tool.is_amd_smi:
        payload = _capture_json(
            tool, ["metric", "--clock", "--power", "--temperature", "--json"], timeout=timeout
        )
        entry = _amd_entry(payload, device)
        if entry is None:
            return None
        clock = entry.get("clock") or {}
        power = entry.get("power") or {}
        temperature = entry.get("temperature") or {}

        def _clock(*keys: str) -> float | None:
            for key in keys:
                value = clock.get(key)
                if isinstance(value, dict):
                    # Discrete parts report {"clk": "1500Mhz"}; APU averages
                    # report {"value": 1089, "unit": "MHz"}.
                    for field in ("clk", "value"):
                        if field in value:
                            parsed = _first_number(str(value.get(field)))
                            if parsed is not None:
                                return parsed
                elif isinstance(value, (int, float)):
                    return float(value)
            return None

        power_w, _ = _amd_quantity(power.get("socket_power"))
        if power_w is None:
            power_w, _ = _amd_quantity(power.get("apu_average_socket_power"))
        temp_c, _ = _amd_quantity(temperature.get("edge"))
        return {
            "sclk_mhz": _clock("gfx_0", "apu_average_gfxclk_frequency"),
            "mclk_mhz": _clock("mem_0", "apu_average_uclk_frequency"),
            "socket_power_w": power_w,
            "edge_temp_c": temp_c,
            "source": tool.name,
            "tool_path": str(tool.path),
        }

    payload = _capture_json(
        tool, ["--showclocks", "--showpower", "--showtemp", "--json"], timeout=timeout
    )
    card = _rocm_card(payload, device)
    if card is None:
        return None

    def _text_number(key: str) -> float | None:
        raw = card.get(key)
        return _first_number(str(raw)) if raw is not None else None

    return {
        "sclk_mhz": _text_number("sclk clock speed:"),
        "mclk_mhz": _text_number("mclk clock speed:"),
        "socket_power_w": _text_number("Current Socket Graphics Package Power (W)"),
        "edge_temp_c": _text_number("Temperature (Sensor edge) (C)"),
        "source": tool.name,
        "tool_path": str(tool.path),
    }
