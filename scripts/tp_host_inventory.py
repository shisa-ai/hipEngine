#!/usr/bin/env python3
"""Record the physical host and PCIe topology for the TP2 campaign.

Packet 0 of docs/QWEN38-27B-GFX1100-TP2.md requires the host identity, GPU
UUID/PCI bus IDs and rank mapping, ROCm/driver/RCCL/compiler versions, NUMA
placement, CPU affinity, display load, clocks/power/temperature, and the
negotiated PCIe generation/width under load, plus IOMMU/ACS state, before any
performance claim is made. This script collects all of that without changing
security settings and writes one compact JSON artifact.

Static topology only. Peer access, device-to-device copy behavior and
collective latency live in scripts/tp_collective_bench.py.

Usage:
    python3 scripts/tp_host_inventory.py --json benchmarks/results/tp2_host_inventory.json
    python3 scripts/tp_host_inventory.py --devices 0,1
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DRM_ROOT = Path("/sys/class/drm")
PCI_DEVICES_ROOT = Path("/sys/bus/pci/devices")
HWMON_ROOT = Path("/sys/class/hwmon")

_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Pure parsing helpers (CPU-testable without a GPU)
# ---------------------------------------------------------------------------


def parse_device_list(text: str) -> list[int]:
    """Parse an ordered ``--devices`` list such as ``0,1`` into indices."""

    indices: list[int] = []
    for chunk in str(text).replace(" ", "").split(","):
        if not chunk:
            raise ValueError("device list contains an empty entry")
        if not re.fullmatch(r"\d+", chunk):
            raise ValueError(f"device entry {chunk!r} is not a non-negative integer")
        index = int(chunk)
        if index in indices:
            raise ValueError(f"device index {index} appears more than once")
        indices.append(index)
    if not indices:
        raise ValueError("device list must contain at least one index")
    return indices


def parse_pcie_link_speed(text: str) -> float | None:
    """Parse a sysfs PCIe speed string (``16.0 GT/s PCIe``) into GT/s."""

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GT/s", str(text))
    if match is None:
        return None
    return float(match.group(1))


def parse_pcie_link_width(text: str) -> int | None:
    match = re.search(r"(\d+)", str(text))
    if match is None:
        return None
    return int(match.group(1))


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def read_int(path: Path) -> int | None:
    raw = read_text(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def pcie_link_snapshot(device_path: Path) -> dict[str, Any]:
    """Return negotiated and maximum PCIe link state for one PCI device."""

    snapshot: dict[str, Any] = {}
    for key, filename in (
        ("current_speed", "current_link_speed"),
        ("max_speed", "max_link_speed"),
    ):
        raw = read_text(device_path / filename)
        snapshot[key] = raw
        snapshot[f"{key}_gts"] = parse_pcie_link_speed(raw) if raw else None
    for key, filename in (
        ("current_width", "current_link_width"),
        ("max_width", "max_link_width"),
    ):
        raw = read_text(device_path / filename)
        snapshot[key] = raw
        snapshot[f"{key}_lanes"] = parse_pcie_link_width(raw) if raw else None
    snapshot["downgraded"] = bool(
        snapshot.get("current_speed_gts")
        and snapshot.get("max_speed_gts")
        and snapshot["current_speed_gts"] < snapshot["max_speed_gts"]
    ) or bool(
        snapshot.get("current_width_lanes")
        and snapshot.get("max_width_lanes")
        and snapshot["current_width_lanes"] < snapshot["max_width_lanes"]
    )
    return snapshot


def parse_lspci_tree(text: str) -> list[dict[str, Any]]:
    """Parse ``lspci -tv`` output into a flat bridge/device listing.

    Each line is indented to encode the hierarchy; we keep the depth, every
    bracketed bus range on the line, and the final endpoint slot so two GPUs
    under one upstream bridge (or on separate root ports) are visible without
    needing a graph library.
    """

    entries: list[dict[str, Any]] = []
    for raw_line in str(text).splitlines():
        if not raw_line.strip():
            continue
        depth = (len(raw_line) - len(raw_line.lstrip(" +-|\\"))) // 2
        line = raw_line.strip(" +-|\\")
        bus_ranges = re.findall(r"\[([0-9a-fA-F]{2}(?:-[0-9a-fA-F]{2})?)\]", line)
        bridge = re.search(r"([0-9a-fA-F]{2})\.([0-9])-\[", line)
        slots = re.findall(r"(?:^|[\]\-])([0-9a-fA-F]{2}\.[0-9])(?=\s|\-)", line)
        entries.append(
            {
                "depth": depth,
                "bridge": f"{bridge.group(1)}.{bridge.group(2)}" if bridge else None,
                "bus_ranges": bus_ranges,
                "device": slots[-1] if slots else None,
                "text": line,
            }
        )
    return entries


def parse_acs_entries(lspci_verbose: str, slots: Sequence[str]) -> dict[str, Any]:
    """Extract ACS capability/control for the requested slots from ``lspci -vv``.

    ``SrcValid+``, ``TransBlk+``, ``ReqRedir+``, ``CmpltRedir+``, or
    ``UpstreamFwd+`` in ``ACSCtl`` force peer TLPs up to the root complex and
    therefore block peer-to-peer DMA between endpoints below that bridge.
    """

    result: dict[str, Any] = {}
    wanted = {str(slot) for slot in slots}
    current_slot: str | None = None
    for raw_line in str(lspci_verbose).splitlines():
        header = re.match(r"^([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9])\s", raw_line)
        if header:
            current_slot = header.group(1)
        if current_slot is None or current_slot not in wanted:
            continue
        stripped = raw_line.strip()
        is_cap = stripped.startswith("ACSCap:")
        is_ctl = stripped.startswith("ACSCtl:")
        if "Access Control Services" in stripped:
            block = result.setdefault(current_slot, {"present": True, "capability_lines": [], "control_lines": [], "raw": []})
            block["present"] = True
            block["raw"].append(stripped)
        elif is_cap or is_ctl:
            block = result.setdefault(current_slot, {"present": True, "capability_lines": [], "control_lines": [], "raw": []})
            body = stripped.split(":", 1)[1]
            block["capability_lines" if is_cap else "control_lines"].append(body.strip())
            block["raw"].append(stripped)
    for slot, block in result.items():
        enabled: list[str] = []
        for line in block.get("control_lines", []):
            for token in line.split():
                if token.endswith("+") and token[:-1] in _ACS_BLOCKING_BITS:
                    enabled.append(token[:-1])
        block["blocking_bits_enabled"] = sorted(set(enabled))
        block["blocks_peer_dma"] = bool(enabled)
    return result


_ACS_BLOCKING_BITS = {"SrcValid", "TransBlk", "ReqRedir", "CmpltRedir", "UpstreamFwd", "DirectTrans"}


def merged_has_acs(blocks: Sequence[str]) -> bool:
    return any("Access Control Services" in block for block in blocks)


def collect_acs_state(slots: Sequence[str]) -> dict[str, Any]:
    """Read ACS capability/control for the given slots via ``lspci -vv``.

    ACS redirection on an intermediate bridge blocks peer-to-peer DMA between
    endpoints below it, so its state is part of the P2P screen. Unprivileged
    reads return ``Capabilities: <access denied>``; that is recorded rather
    than reported as "no ACS".
    """

    result: dict[str, Any] = {}
    if not slots:
        return result
    # ``lspci -s`` takes a single slot in this build, so query each one.
    merged: list[str] = []
    available = False
    access_denied = False
    prefix: list[str] = []
    for slot in dict.fromkeys(str(entry) for entry in slots):
        verbose = _run(prefix + ["lspci", "-vv", "-s", slot])
        if not verbose.get("available"):
            continue
        available = True
        stdout = verbose.get("stdout") or ""
        if "<access denied>" in stdout:
            access_denied = True
        merged.append(stdout)
    if access_denied and not merged_has_acs(merged):
        # PCIe extended capabilities need root. Retry once through a
        # non-interactive sudo when the host allows it; never prompt.
        probe = _run(["sudo", "-n", "true"])
        if probe.get("available") and probe.get("returncode") == 0:
            prefix = ["sudo", "-n"]
            merged = []
            access_denied = False
            result["read_via"] = "sudo -n"
            for slot in dict.fromkeys(str(entry) for entry in slots):
                verbose = _run(prefix + ["lspci", "-vv", "-s", slot])
                if not verbose.get("available"):
                    continue
                stdout = verbose.get("stdout") or ""
                if "<access denied>" in stdout:
                    access_denied = True
                merged.append(stdout)
    result["available"] = available
    result["access_denied"] = access_denied
    if access_denied:
        result["note"] = "ACS capability space needs root; values are unknown, not absent"
    result["entries"] = parse_acs_entries("\n".join(merged), list(dict.fromkeys(str(entry) for entry in slots)))
    return result


def collect_hwmon_diagnostics(device_path: Path | None) -> dict[str, Any]:
    """Root-free power/temperature/frequency snapshot from the DRM hwmon node."""

    if device_path is None:
        return {}
    hwmon_root = device_path / "hwmon"
    hwmon_dirs = sorted(hwmon_root.glob("hwmon*")) if hwmon_root.is_dir() else []
    if not hwmon_dirs:
        return {}
    root = hwmon_dirs[0]
    snapshot: dict[str, Any] = {"path": str(root)}
    power = read_int(root / "power1_average")
    if power is not None:
        snapshot["power_average_uw"] = power
    cap = read_int(root / "power1_cap")
    if cap is not None:
        snapshot["power_cap_uw"] = cap
    for index in (1, 2):
        label = read_text(root / f"temp{index}_label")
        value = read_int(root / f"temp{index}_input")
        if value is not None:
            key = (label or f"temp{index}").strip().lower().replace(" ", "_")
            snapshot[f"temp_{key}_c"] = value / 1000.0
    for index in (1, 2):
        label = read_text(root / f"freq{index}_label")
        value = read_int(root / f"freq{index}_input")
        if value is not None:
            key = (label or f"freq{index}").strip().lower().replace(" ", "_")
            snapshot[f"freq_{key}_hz"] = value
    return snapshot


def collect_pcie_dpm_table(device_path: Path | None) -> dict[str, Any]:
    """Snapshot ``pp_dpm_pcie``.

    This is the driver's DPM capability table - the link widths it may
    downshift to - not the live negotiated link state. On this host the table
    lists ``16.0GT/s, x8`` while ``current_link_width`` and ``lspci`` LnkSta
    both report x16, so the table must never be reported as the live width.
    """

    if device_path is None:
        return {}
    text = read_text(device_path / "pp_dpm_pcie")
    if not text:
        return {}
    levels: list[dict[str, Any]] = []
    active: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        level, body = stripped.split(":", 1)
        is_active = body.rstrip().endswith("*")
        if is_active:
            active = level.strip()
        speed = re.search(r"([0-9]+(?:\.[0-9]+)?)GT/s", body)
        width = re.search(r"x(\d+)", body)
        levels.append(
            {
                "level": level.strip(),
                "speed_gts": float(speed.group(1)) if speed else None,
                "width_lanes": int(width.group(1)) if width else None,
                "active": is_active,
            }
        )
    return {
        "note": "driver DPM capability table; not the negotiated link state",
        "levels": levels,
        "active_level": active,
    }


def card_sysfs_path(pci_bus_id: str | None) -> Path | None:
    """Resolve a HIP PCI bus id (``0000:c3:00.0``) to its sysfs device path."""

    if not pci_bus_id:
        return None
    normalized = pci_bus_id.strip().lower()
    if not normalized.startswith("0000:"):
        normalized = f"0000:{normalized}"
    path = PCI_DEVICES_ROOT / normalized
    return path if path.is_dir() else None


def card_device_path(pci_bus_id: str | None) -> Path | None:
    """Resolve a DRM card device directory (with PCIe link/hwmon attributes)."""

    if not pci_bus_id:
        return None
    normalized = pci_bus_id.strip().lower()
    if not normalized.startswith("0000:"):
        normalized = f"0000:{normalized}"
    for card in sorted(DRM_ROOT.glob("card[0-9]*")):
        device = card / "device"
        if not device.exists():
            continue
        try:
            resolved = device.resolve().name.lower()
        except OSError:
            continue
        if resolved == normalized:
            return device
    return card_sysfs_path(pci_bus_id)


# ---------------------------------------------------------------------------
# Hardware probes
# ---------------------------------------------------------------------------


def _run(command: Sequence[str], timeout: float = 10.0) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": list(command), "error": "executable not found"}
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "command": list(command), "error": str(error)}
    return {
        "available": True,
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr[-2000:],
    }


def collect_software_versions() -> dict[str, Any]:
    software: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    rocm_version = read_text(Path("/opt/rocm/.info/version"))
    if rocm_version:
        software["rocm_version"] = rocm_version
    try:
        rccl = ctypes.CDLL("librccl.so")
    except OSError as error:
        software["rccl_version"] = None
        software["rccl_error"] = str(error)
    else:
        version = ctypes.c_int()
        rccl.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        rccl.ncclGetVersion.restype = ctypes.c_int
        status = int(rccl.ncclGetVersion(ctypes.byref(version)))
        software["rccl_version"] = int(version.value) if status == 0 else None
        software["rccl_version_status"] = status
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    hip_version = ctypes.c_int()
    version_function = getattr(runtime.library, "hipRuntimeGetVersion", None)
    if version_function is None:
        software["hip_runtime_version"] = None
    else:
        version_function.argtypes = [ctypes.POINTER(ctypes.c_int)]
        version_function.restype = ctypes.c_int
        status = int(version_function(ctypes.byref(hip_version)))
        software["hip_runtime_version"] = int(hip_version.value) if status == 0 else None
        software["hip_runtime_version_status"] = status
    hipcc = _run(["hipcc", "--version"])
    if hipcc.get("available"):
        software["hipcc"] = {
            "returncode": hipcc.get("returncode"),
            "version_line": (hipcc.get("stdout") or "").splitlines()[0] if hipcc.get("stdout") else None,
        }
    amdgpu = read_text(Path("/sys/module/amdgpu/version"))
    if amdgpu:
        software["amdgpu_driver_version"] = amdgpu
    return software


def parse_iommu_state(
    *,
    iommu_groups_root: Path = Path("/sys/kernel/iommu_groups"),
    cmdline: str | None = None,
) -> dict[str, Any]:
    """Summarize IOMMU enablement and the groups the kernel created."""

    groups = sorted(path.name for path in iommu_groups_root.glob("*")) if iommu_groups_root.is_dir() else []
    if cmdline is None:
        cmdline = read_text(Path("/proc/cmdline")) or ""
    tokens = [token for token in str(cmdline).split() if "iommu" in token.lower()]
    return {
        "enabled": bool(groups),
        "group_count": len(groups),
        "groups": groups,
        "kernel_cmdline_iommu": tokens,
    }


def parse_rocm_smi_json(payload: Any) -> dict[str, Any]:
    """Normalize ``rocm-smi --json`` output into a per-card diagnostics map.

    rocm-smi keys cards either by index or by PCI bus id depending on version;
    we index both so the caller can join by either. Only ``clock speed`` fields
    are used for sclk/mclk - the sibling ``clock level`` fields would otherwise
    overwrite them with a DPM level number.
    """

    cards: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return cards
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        card: dict[str, Any] = {}
        for field, raw in value.items():
            lowered = str(field).lower()
            is_speed = "speed" in lowered
            if "temperature" in lowered and "edge" in lowered:
                card["temperature_edge_c"] = _parse_float(raw)
            elif "average" in lowered and "power" in lowered:
                card["power_average_w"] = _parse_float(raw)
            elif "power" in lowered and ("cap" in lowered or "max power" in lowered):
                card["power_cap_w"] = _parse_float(raw)
            elif is_speed and ("sclk" in lowered or "gfx_clock" in lowered):
                card["sclk_mhz"] = _parse_float(raw)
                card["sclk_raw"] = str(raw)
            elif is_speed and ("mclk" in lowered or "mem_clock" in lowered):
                card["mclk_mhz"] = _parse_float(raw)
                card["mclk_raw"] = str(raw)
            elif "pcie clock level" in lowered:
                card["pcie_clock_level"] = str(raw)
                speed = re.search(r"([0-9]+(?:\.[0-9]+)?)GT/s", str(raw))
                width = re.search(r"x(\d+)", str(raw))
                if speed:
                    card["pcie_speed_gts"] = float(speed.group(1))
                if width:
                    card["pcie_width_lanes"] = int(width.group(1))
            elif "card series" in lowered or "card model" in lowered:
                card["model"] = str(raw)
            elif "vbios" in lowered:
                card["vbios"] = str(raw)
            elif "pci bus" in lowered:
                card["pci_bus"] = str(raw)
        if card:
            cards[str(key)] = card
    return cards


def _parse_float(raw: Any) -> float | None:
    if isinstance(raw, (int, float)):
        return float(raw)
    match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", str(raw))
    if match is None:
        return None
    return float(match.group(0))


def summarize_host(
    *,
    hostname: str,
    uname: dict[str, str],
    cpu_model: str | None,
    cpu_count: int,
    affinity: Sequence[int],
    mem_total_bytes: int | None,
    numa_nodes: Sequence[int],
) -> dict[str, Any]:
    return {
        "hostname": hostname,
        "kernel": dict(uname),
        "cpu_model": cpu_model,
        "cpu_count": int(cpu_count),
        "process_cpu_affinity": [int(cpu) for cpu in affinity],
        "mem_total_bytes": mem_total_bytes,
        "numa_nodes": [int(node) for node in numa_nodes],
    }


def summarize_device(
    *,
    rank: int,
    index: int,
    name: str,
    arch: str | None,
    uuid: str | None,
    uuid_hex: str | None,
    pci_bus_id: str | None,
    vram_total_bytes: int | None,
    vram_free_bytes: int | None,
    numa_node: int | None,
    pcie: dict[str, Any] | None,
    smi: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "rank": int(rank),
        "hip_index": int(index),
        "name": name,
        "arch": arch,
        "uuid": uuid,
        "uuid_hex": uuid_hex,
        "pci_bus_id": pci_bus_id,
        "numa_node": numa_node,
        "vram_total_bytes": vram_total_bytes,
        "vram_free_bytes": vram_free_bytes,
        "pcie": pcie or {},
        "rocm_smi": smi or {},
    }


def collect_devices(indices: Sequence[int]) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    from hipengine.core.device import Device, scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.backends import detect_hip_target_arches

    runtime = get_hip_runtime()
    arches = tuple(dict.fromkeys(detect_hip_target_arches()))
    smi_cards: dict[str, Any] = {}
    smi = _run(["rocm-smi", "--showtemp", "--showpower", "--showclocks", "--showproductname", "--showbus", "--json"])
    if smi.get("available") and smi.get("returncode") == 0 and smi.get("stdout"):
        try:
            smi_cards = parse_rocm_smi_json(json.loads(smi["stdout"]))
        except json.JSONDecodeError as error:
            errors.append(f"rocm-smi JSON parse failed: {error}")

    devices: list[dict[str, Any]] = []
    for rank, index in enumerate(indices):
        entry: dict[str, Any] = {"rank": rank, "hip_index": int(index)}
        try:
            with scoped_current_device(runtime, int(index)):
                info = runtime.device_info(int(index))
                free_bytes, total_bytes = runtime.mem_get_info()
        except Exception as error:  # noqa: BLE001 - record and continue screening
            errors.append(f"device {index} identity probe failed: {error!r}")
            devices.append(entry)
            continue
        sysfs = card_sysfs_path(info.pci_bus_id)
        device_path = card_device_path(info.pci_bus_id)
        numa_node = read_int(sysfs / "numa_node") if sysfs else None
        pcie = pcie_link_snapshot(device_path) if device_path else {}
        smi_entry = None
        for key, value in smi_cards.items():
            pci_field = str(value.get("pci_bus", "")).lower().replace("0000:", "")
            if key == str(index) or (pci_field and pci_field in info.pci_bus_id.lower()):
                smi_entry = value
                break
        entry.update(
            summarize_device(
                rank=rank,
                index=int(index),
                name=info.name,
                arch=arches[rank] if rank < len(arches) else (arches[0] if arches else None),
                uuid=info.uuid,
                uuid_hex=info.uuid_hex,
                pci_bus_id=info.pci_bus_id,
                vram_total_bytes=int(total_bytes),
                vram_free_bytes=int(free_bytes),
                numa_node=numa_node,
                pcie=pcie,
                smi=smi_entry,
            )
        )
        entry["driver"] = {
            "amdgpu_version": read_text(Path("/sys/module/amdgpu/version")),
            "vbios": read_text(sysfs / "vbios_version") if sysfs else None,
        }
        entry["hwmon"] = collect_hwmon_diagnostics(device_path)
        entry["pp_dpm_pcie"] = collect_pcie_dpm_table(device_path)
        devices.append(entry)
    return devices, errors


def collect_topology(devices: Sequence[dict[str, Any]]) -> dict[str, Any]:
    topology: dict[str, Any] = {}
    lspci_tree = _run(["lspci", "-tv"])
    if lspci_tree.get("available") and lspci_tree.get("stdout"):
        topology["lspci_tree"] = parse_lspci_tree(lspci_tree["stdout"])
    slots = [str(device.get("pci_bus_id", "")).split(":", 1)[-1] for device in devices if device.get("pci_bus_id")]
    bridge_slots: list[str] = []
    for device in devices:
        path = card_sysfs_path(str(device.get("pci_bus_id", "")))
        if path is None:
            continue
        try:
            path = path.resolve()
        except OSError:
            pass
        for parent in path.parents:
            name = parent.name
            if not name.startswith("0000:"):
                continue
            slot = name.split(":", 1)[-1]
            if slot and slot not in bridge_slots:
                bridge_slots.append(slot)
    topology["acs"] = collect_acs_state(slots + bridge_slots)
    topology["iommu"] = parse_iommu_state()
    return topology


def collect_display_load(
    *,
    proc_root: Path = Path("/proc"),
    drm_root: Path = Path("/dev/dri"),
) -> dict[str, Any]:
    """Which processes hold a DRM device open, per render/card node.

    A compositor or a browser on the same GPU perturbs latency measurements, so
    the screen records the openers instead of assuming an idle desktop. Only the
    device nodes are recorded; no process details beyond name/pid.
    """

    devices = (
        sorted(
            path.name
            for path in drm_root.glob("*")
            if path.name.startswith(("card", "renderD")) and path.is_char_device()
        )
        if drm_root.is_dir()
        else []
    )
    openers: dict[str, list[dict[str, Any]]] = {name: [] for name in devices}
    if not devices:
        return {
            "devices": [],
            "openers": {},
            "session": os.environ.get("XDG_SESSION_TYPE") or os.environ.get("WAYLAND_DISPLAY") or None,
            "busy": False,
        }
    targets = {str(drm_root / name): name for name in devices}
    for entry in sorted(proc_root.glob("[0-9]*")):
        try:
            pid = int(entry.name)
        except ValueError:
            continue
        fd_dir = entry / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except (PermissionError, FileNotFoundError, OSError):
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except (PermissionError, FileNotFoundError, OSError):
                continue
            name = targets.get(target)
            if name is None:
                continue
            if pid == os.getpid():
                continue
            openers[name].append({"pid": pid, "comm": read_text(entry / "comm") or "?"})
            break
    session = os.environ.get("XDG_SESSION_TYPE") or os.environ.get("WAYLAND_DISPLAY") or None
    return {
        "devices": devices,
        "openers": openers,
        "session": session,
        "busy": any(entries for entries in openers.values()),
    }


def collect_host() -> dict[str, Any]:
    uname = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "node": platform.node(),
    }
    cpu_model = None
    for line in (read_text(Path("/proc/cpuinfo")) or "").splitlines():
        if line.lower().startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    mem_total = None
    for line in (read_text(Path("/proc/meminfo")) or "").splitlines():
        if line.startswith("MemTotal:"):
            mem_total = int(line.split()[1]) * 1024
            break
    numa_nodes: list[int] = []
    for node in Path("/sys/devices/system/node").glob("node[0-9]*"):
        numa_nodes.append(int(node.name.replace("node", "")))
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else []
    return summarize_host(
        hostname=socket.gethostname(),
        uname=uname,
        cpu_model=cpu_model,
        cpu_count=os.cpu_count() or 0,
        affinity=affinity,
        mem_total_bytes=mem_total,
        numa_nodes=numa_nodes,
    )


def build_inventory(indices: Sequence[int]) -> dict[str, Any]:
    devices, errors = collect_devices(indices)
    return {
        "kind": "tp_host_inventory",
        "schema_version": _SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device_order": [int(index) for index in indices],
        "host": collect_host(),
        "software": collect_software_versions(),
        "topology": collect_topology(devices),
        "display": collect_display_load(),
        "devices": devices,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", default="0,1", help="Ordered HIP device indices defining rank order (default: 0,1)")
    parser.add_argument("--json", type=Path, default=None, help="Write the inventory JSON here (default: stdout)")
    args = parser.parse_args()

    indices = parse_device_list(args.devices)
    inventory = build_inventory(indices)
    payload = json.dumps(inventory, indent=2, sort_keys=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
