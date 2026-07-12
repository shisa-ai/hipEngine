#!/usr/bin/env python3
"""Collect environment metadata for hipEngine microbenchmarks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance


DEFAULT_MAX_OUTPUT_CHARS = 20000
_MICRO_PROVENANCE_ENV_KEYS = (
    "HIPENGINE_BACKEND",
    "HIPENGINE_HIP_ARCH",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "VK_ICD_FILENAMES",
    "RADV_PERFTEST",
)


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def run_command(args: list[str], *, timeout_s: float, max_output_chars: int, cwd: Path | None = None) -> dict[str, Any]:
    executable = shutil.which(args[0])
    record: dict[str, Any] = {
        "args": args,
        "available": executable is not None,
        "path": executable,
        "returncode": None,
        "timed_out": False,
        "stdout": "",
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
    }
    if executable is None:
        return record
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        record["timed_out"] = True
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        record["stdout"], record["stdout_truncated"] = _truncate(stdout, max_output_chars)
        record["stderr"], record["stderr_truncated"] = _truncate(stderr, max_output_chars)
        return record
    record["returncode"] = completed.returncode
    record["stdout"], record["stdout_truncated"] = _truncate(completed.stdout, max_output_chars)
    record["stderr"], record["stderr_truncated"] = _truncate(completed.stderr, max_output_chars)
    return record


def _git_text(repo_root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip()


def collect_git(repo_root: Path) -> dict[str, Any]:
    status_short = _git_text(repo_root, ["status", "--short"]).splitlines()
    return {
        "root": str(repo_root),
        "branch": _git_text(repo_root, ["branch", "--show-current"]),
        "commit": _git_text(repo_root, ["rev-parse", "HEAD"]),
        "commit_short": _git_text(repo_root, ["rev-parse", "--short", "HEAD"]),
        "dirty": bool(status_short),
        "status_short": status_short,
    }


def _first_matching_lines(text: str, needles: tuple[str, ...], *, limit: int = 80) -> list[str]:
    matches: list[str] = []
    for line in text.splitlines():
        if any(needle in line for needle in needles):
            matches.append(line.strip())
            if len(matches) >= limit:
                break
    return matches


def parse_device_summaries(commands: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rocminfo = commands.get("rocminfo", {})
    vulkaninfo = commands.get("vulkaninfo_summary", {})
    lspci = commands.get("lspci", {})
    rocm_smi = commands.get("rocm_smi", {})
    rocm_smi_metrics = commands.get("rocm_smi_metrics", {})
    amd_smi_static = commands.get("amd_smi_static", {})
    amd_smi_metric = commands.get("amd_smi_metric", {})
    inxi = commands.get("inxi_system", {})
    ryzenadj = commands.get("ryzenadj_sudo_info", {}) or commands.get("ryzenadj_info", {})
    return {
        "rocminfo_name_gfx_lines": _first_matching_lines(
            str(rocminfo.get("stdout", "")),
            ("Name:", "gfx"),
        ),
        "vulkan_summary_lines": _first_matching_lines(
            str(vulkaninfo.get("stdout", "")),
            ("GPU", "deviceName", "driver", "apiVersion", "Mesa", "RADV"),
        ),
        "lspci_display_lines": _first_matching_lines(
            str(lspci.get("stdout", "")),
            ("VGA", "Display", "3D controller", "AMD", "ATI"),
        ),
        "rocm_smi_lines": _first_matching_lines(
            str(rocm_smi.get("stdout", "")),
            ("GPU", "Card", "Driver", "VBIOS", "Device", "ASIC"),
        ),
        "rocm_smi_metric_lines": _first_matching_lines(
            str(rocm_smi_metrics.get("stdout", "")),
            ("Driver", "Device", "VBIOS", "Temperature", "clock", "Power", "GPU use", "Memory", "GFX"),
        ),
        "amd_smi_static_lines": _first_matching_lines(
            str(amd_smi_static.get("stdout", "")),
            ("MARKET_NAME", "DEVICE_ID", "NUM_COMPUTE_UNITS", "TARGET_GRAPHICS_VERSION"),
        ),
        "amd_smi_metric_lines": _first_matching_lines(
            str(amd_smi_metric.get("stdout", "")),
            ("EDGE", "SOCKET_POWER", "CLOCK", "PERF_LEVEL", "TOTAL_VRAM", "USED_VRAM", "TOTAL_GTT"),
        ),
        "inxi_cpu_gpu_lines": _first_matching_lines(
            str(inxi.get("stdout", "")),
            ("Kernel", "Distro", "Machine", "Firmware", "CPU:", "Info", "Speed", "Graphics", "Device-1", "driver", "OpenGL", "Vulkan"),
        ),
        "ryzenadj_lines": _first_matching_lines(
            str(ryzenadj.get("stdout", "")),
            ("CPU Family", "SMU", "Version", "PM Table", "STAPM", "PPT", "THM", "STT", "CCLK"),
        ),
    }


def parse_software_summaries(commands: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hipcc = commands.get("hipcc_version", {})
    amdclang = commands.get("amdclang_version", {})
    glslc = commands.get("glslc_version", {})
    vulkaninfo = commands.get("vulkaninfo_summary", {})
    uname = commands.get("uname", {})
    proc_version = commands.get("proc_version", {})
    pacman = commands.get("pacman_versions", {})
    conda_rocm = commands.get("conda_rocm_packages", {})
    modinfo = commands.get("modinfo_amdgpu", {})
    return {
        "kernel_lines": _first_matching_lines(
            "\n".join((str(uname.get("stdout", "")), str(proc_version.get("stdout", "")))),
            ("Linux", "version", "clang", "gcc"),
        ),
        "hipcc_lines": _first_matching_lines(
            str(hipcc.get("stdout", "")),
            ("HIP version", "AMD clang version", "ROCm", "InstalledDir"),
        ),
        "amdclang_lines": _first_matching_lines(
            str(amdclang.get("stdout", "")),
            ("AMD clang version", "InstalledDir"),
        ),
        "glslc_lines": _first_matching_lines(
            str(glslc.get("stdout", "")),
            ("202", "Target", "LLVM version", "Vulkan", "SPIR-V"),
        ),
        "vulkan_driver_lines": _first_matching_lines(
            str(vulkaninfo.get("stdout", "")),
            ("Vulkan Instance Version", "apiVersion", "driverVersion", "driverInfo", "driverName"),
        ),
        "arch_package_lines": _first_matching_lines(
            "\n".join((str(pacman.get("stdout", "")), str(pacman.get("stderr", "")))),
            ("linux", "firmware", "ucode", "mesa", "vulkan", "shaderc", "glslang", "llvm", "clang", "rocm", "hip"),
            limit=120,
        ),
        "conda_rocm_lines": _first_matching_lines(
            str(conda_rocm.get("stdout", "")),
            ("rocm", "hip", "llvm", "therock", "vulkan", "shader", "mesa", "clang", "torch", "triton"),
            limit=120,
        ),
        "amdgpu_module_lines": _first_matching_lines(
            str(modinfo.get("stdout", "")),
            ("filename:", "firmware:", "version:", "srcversion:", "vermagic:"),
            limit=120,
        ),
    }


def collect_environment(
    *,
    repo_root: Path,
    include_device_probes: bool,
    include_privileged: bool,
    timeout_s: float,
    max_output_chars: int,
) -> dict[str, Any]:
    commands: dict[str, dict[str, Any]] = {
        "uname": run_command(["uname", "-a"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "proc_version": run_command(["cat", "/proc/version"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "git_version": run_command(["git", "--version"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "hipcc_version": run_command(["hipcc", "--version"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "amdclang_version": run_command(["amdclang++", "--version"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "clang_version": run_command(["clang", "--version"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "glslc_version": run_command(["glslc", "--version"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "llvm_objdump_version": run_command(["llvm-objdump", "--version"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "llvm_readobj_version": run_command(["llvm-readobj", "--version"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "modinfo_amdgpu": run_command(["modinfo", "amdgpu"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "lsmod": run_command(["lsmod"], timeout_s=timeout_s, max_output_chars=max_output_chars),
        "amdgpu_module_state": run_command(
            [
                "bash",
                "-lc",
                "for f in /sys/module/amdgpu/initstate /sys/module/amdgpu/refcnt /sys/module/amdgpu/srcversion; do "
                "printf '### %s\\n' \"$f\"; cat \"$f\" 2>&1; done",
            ],
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
        ),
        "amdgpu_parameters": run_command(
            [
                "bash",
                "-lc",
                "for f in /sys/module/amdgpu/parameters/*; do printf '%s=' \"${f##*/}\"; cat \"$f\"; done 2>/dev/null | sort",
            ],
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
        ),
        "amdgpu_firmware_debugfs": run_command(
            [
                "bash",
                "-lc",
                "for f in /sys/kernel/debug/dri/*/amdgpu_firmware_info; do printf '### %s\\n' \"$f\"; cat \"$f\"; done",
            ],
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
        ),
        "dmesg_amdgpu_firmware": run_command(
            [
                "bash",
                "-lc",
                "dmesg | grep -Ei 'amdgpu|firmware|ucode|gfx|smu' | tail -200",
            ],
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
        ),
        "pacman_versions": run_command(
            [
                "pacman",
                "-Q",
                "linux-cachyos",
                "linux-firmware",
                "amd-ucode",
                "mesa",
                "vulkan-radeon",
                "vulkan-tools",
                "shaderc",
                "glslang",
                "llvm",
                "clang",
                "rocm-core",
                "rocm-hip-runtime",
                "hip-runtime-amd",
                "rocm-smi-lib",
            ],
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
        ),
        "conda_rocm_packages": run_command(
            [
                "bash",
                "-lc",
                "conda list 2>/dev/null | grep -Ei 'rocm|hip|llvm|therock|vulkan|shader|mesa|clang|torch|triton' || true",
            ],
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
        ),
        "python_rocm_sdk_packages": run_command(
            [
                sys.executable,
                "-c",
                (
                    "import importlib.metadata as m\n"
                    "for n in ['rocm','rocm-sdk-core','rocm-sdk-devel','rocm-sdk-libraries-gfx1151','_rocm_sdk_devel']:\n"
                    "    try:\n"
                    "        d=m.distribution(n); print(n, d.version, d.locate_file(''))\n"
                    "    except Exception as e:\n"
                    "        print(n, 'missing', type(e).__name__)\n"
                ),
            ],
            timeout_s=timeout_s,
            max_output_chars=max_output_chars,
        ),
    }
    if include_device_probes:
        commands.update(
            {
                "rocminfo": run_command(["rocminfo"], timeout_s=timeout_s, max_output_chars=max_output_chars),
                "rocm_smi": run_command(
                    ["rocm-smi", "--showproductname", "--showdriverversion"],
                    timeout_s=timeout_s,
                    max_output_chars=max_output_chars,
                ),
                "rocm_smi_metrics": run_command(
                    [
                        "rocm-smi",
                        "--showproductname",
                        "--showdriverversion",
                        "--showvbios",
                        "--showid",
                        "--showtemp",
                        "--showclocks",
                        "--showpower",
                        "--showuse",
                        "--showmeminfo",
                        "vram",
                        "--showmeminfo",
                        "vis_vram",
                    ],
                    timeout_s=timeout_s,
                    max_output_chars=max_output_chars,
                ),
                "amd_smi_static": run_command(["amd-smi", "static", "-a"], timeout_s=timeout_s, max_output_chars=max_output_chars),
                "amd_smi_metric": run_command(["amd-smi", "metric"], timeout_s=timeout_s, max_output_chars=max_output_chars),
                "vulkaninfo_summary": run_command(
                    ["vulkaninfo", "--summary"],
                    timeout_s=timeout_s,
                    max_output_chars=max_output_chars,
                ),
                "lspci": run_command(["lspci", "-nn"], timeout_s=timeout_s, max_output_chars=max_output_chars),
                "inxi_system": run_command(
                    ["inxi", "-c", "0", "-C", "-G", "-S", "-M", "-xx", "--filter"],
                    timeout_s=timeout_s,
                    max_output_chars=max_output_chars,
                ),
                "ryzenadj_info": run_command(["ryzenadj", "-i"], timeout_s=timeout_s, max_output_chars=max_output_chars),
            }
        )
        if include_privileged:
            commands["ryzenadj_sudo_info"] = run_command(
                ["sudo", "-n", "ryzenadj", "-i"],
                timeout_s=timeout_s,
                max_output_chars=max_output_chars,
            )

    hipcc_command = commands.get("hipcc_version", {})
    hipcc_version = (
        str(hipcc_command.get("stdout") or "").strip()
        if hipcc_command.get("returncode") == 0
        else None
    )
    provenance = collect_artifact_provenance(
        repo_root=repo_root,
        configured_backend=os.environ.get("HIPENGINE_BACKEND", "auto"),
        detected_arches=None if include_device_probes else (),
        command=tuple(sys.argv),
        environment={key: os.environ.get(key) for key in _MICRO_PROVENANCE_ENV_KEYS},
        build_profile="micro_environment",
        timing_protocol=None,
        rocm_version=None,
        hipcc_version=hipcc_version,
    )
    return {
        "schema_version": 1,
        "kind": "hipengine_micro_environment",
        "collected_at_unix": time.time(),
        "collector": {
            "path": str(Path(__file__).resolve()),
            "argv": sys.argv,
            "max_output_chars": max_output_chars,
            "timeout_s": timeout_s,
            "include_device_probes": include_device_probes,
            "include_privileged": include_privileged,
        },
        "repo": collect_git(repo_root),
        "provenance": provenance,
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "uname": platform.uname()._asdict(),
            "env": {
                "PATH": os.environ.get("PATH", ""),
                "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
                "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
                "VK_ICD_FILENAMES": os.environ.get("VK_ICD_FILENAMES"),
                "RADV_PERFTEST": os.environ.get("RADV_PERFTEST"),
            },
        },
        "commands": commands,
        "devices": parse_device_summaries(commands),
        "software": parse_software_summaries(commands),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root to record git metadata from")
    parser.add_argument("--out", type=Path, help="Write JSON to this path instead of stdout")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON with sorted keys")
    parser.add_argument("--skip-device-probes", action="store_true", help="Skip rocminfo, rocm-smi, vulkaninfo, and lspci")
    parser.add_argument(
        "--include-privileged",
        action="store_true",
        help="Also run fast-failing privileged probes such as sudo -n ryzenadj -i",
    )
    parser.add_argument("--timeout-s", type=float, default=8.0, help="Per-command timeout")
    parser.add_argument("--max-output-chars", type=int, default=DEFAULT_MAX_OUTPUT_CHARS, help="Truncate each command stream to this many chars")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    data = collect_environment(
        repo_root=repo_root,
        include_device_probes=not args.skip_device_probes,
        include_privileged=args.include_privileged,
        timeout_s=args.timeout_s,
        max_output_chars=args.max_output_chars,
    )
    indent = 2 if args.pretty else None
    text = json.dumps(data, indent=indent, sort_keys=args.pretty)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
