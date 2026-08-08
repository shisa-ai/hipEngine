#!/usr/bin/env python3
"""Guarded ROCm#6529 lifecycle runbook and local evidence collector.

Without ``--execute`` this script only prints a plan. Execution defaults to the
safe queue/resource recreate-without-submit arm. Native AQL/PM4 submit plus
resource recreation requires three explicit choices: ``--submit-recreate``,
``--ack-reset-risk``, and the exact approval token. Raw journals, full GPU
addresses, process metadata, and devcoredumps stay outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root_text = str(REPO_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != _repo_root_text]
sys.path.insert(0, _repo_root_text)

from hipengine.benchmark.provenance import collect_repo_state  # noqa: E402
from hipengine.core.pm4.native_build import plan_pm4_native_build  # noqa: E402
from hipengine.kernels.backends import hip_target_arch_environment  # noqa: E402
from hipengine.kernels.hip_gfx1100.smoke import plan_smoke_add_build  # noqa: E402

Transport = Literal["aql", "pm4"]
AllocationMode = Literal["hip", "hsa"]
BufferMode = Literal["reuse", "recreate"]
DESTRUCTIVE_APPROVAL_TOKEN = "ROCM-6529-RESET-RISK-APPROVED"
_ISSUE_URL = "https://github.com/ROCm/ROCm/issues/6529"
_ARCH_PATTERN = re.compile(r"\bgfx[0-9a-fA-F]+\b")


@dataclass(frozen=True, slots=True)
class IssueCapturePlan:
    repo_root: Path
    output_dir: Path | None
    compiler_version_file: Path | None
    python_executable: str
    transport: Transport = "pm4"
    cycles: int = 8
    submit_recreate: bool = False
    allocation_mode: AllocationMode = "hip"
    buffer_mode: BufferMode = "reuse"
    timestamps: bool = False
    quarantine_generations: int = 0
    hip_visible_devices: str = "0"
    rocr_visible_devices: str = "0"
    process_timeout_seconds: float = 900.0
    settle_seconds: float = 10.0
    devcoredump_reader: Literal["direct", "sudo"] = "sudo"
    execute: bool = False
    reset_risk_acknowledged: bool = False
    approval_token: str | None = None

    @property
    def destructive(self) -> bool:
        return self.submit_recreate

    def validated(self) -> "IssueCapturePlan":
        root = self.repo_root.expanduser().resolve()
        output = None if self.output_dir is None else self.output_dir.expanduser().resolve()
        compiler = (
            None
            if self.compiler_version_file is None
            else self.compiler_version_file.expanduser().resolve()
        )
        if self.transport not in {"aql", "pm4"}:
            raise ValueError("transport must be aql or pm4")
        if not 1 <= self.cycles <= 1_000_000:
            raise ValueError("cycles must be in 1..1000000")
        if self.allocation_mode not in {"hip", "hsa"}:
            raise ValueError("allocation-mode must be hip or hsa")
        if self.buffer_mode not in {"reuse", "recreate"}:
            raise ValueError("buffer-mode must be reuse or recreate")
        if self.timestamps and self.transport != "pm4":
            raise ValueError("timestamps require transport=pm4")
        if not 0 <= self.quarantine_generations <= 4096:
            raise ValueError("quarantine-generations must be in 0..4096")
        if self.process_timeout_seconds <= 0 or self.settle_seconds < 0:
            raise ValueError("capture timeouts must be positive/non-negative")
        if self.devcoredump_reader not in {"direct", "sudo"}:
            raise ValueError("devcoredump-reader must be direct or sudo")
        if not self.hip_visible_devices.strip() or not self.rocr_visible_devices.strip():
            raise ValueError("HIP/ROCR visibility values must be non-empty")
        if self.submit_recreate and not self.reset_risk_acknowledged:
            raise ValueError("submit-recreate requires --ack-reset-risk")
        if self.execute and self.submit_recreate and self.approval_token != DESTRUCTIVE_APPROVAL_TOKEN:
            raise ValueError(
                "destructive execution requires --approval-token "
                f"{DESTRUCTIVE_APPROVAL_TOKEN} after separate operator approval"
            )
        if self.execute and output is None:
            raise ValueError("--execute requires --output-dir")
        if self.execute and compiler is None:
            raise ValueError("--execute requires --compiler-version-file and cached builds")
        if output is not None and (output == root or root in output.parents):
            raise ValueError("raw issue evidence must be written outside the repository")
        if self.allocation_mode == "hsa" and self.buffer_mode != "recreate":
            # Public-HSA allocations are context-owned, so queue recreation must
            # recreate those buffers. Mixed HIP allocation is the isolation arm
            # that can retain input/output addresses across queue generations.
            return IssueCapturePlan(
                **{
                    **asdict(self),
                    "repo_root": root,
                    "output_dir": output,
                    "compiler_version_file": compiler,
                    "buffer_mode": "recreate",
                }
            )
        return IssueCapturePlan(
            **{
                **asdict(self),
                "repo_root": root,
                "output_dir": output,
                "compiler_version_file": compiler,
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--python", default=sys.executable, dest="python_executable")
    parser.add_argument("--transport", choices=("aql", "pm4"), default="pm4")
    parser.add_argument("--cycles", type=int)
    parser.add_argument(
        "--submit-recreate",
        action="store_true",
        help="submit through native AQL/PM4 before recreating packet resources (reset risk)",
    )
    parser.add_argument("--allocation-mode", choices=("hip", "hsa"), default="hip")
    parser.add_argument("--buffer-mode", choices=("reuse", "recreate"), default="reuse")
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--quarantine-generations", type=int, default=0)
    parser.add_argument(
        "--hip-visible-devices",
        default=os.environ.get("HIP_VISIBLE_DEVICES", "0"),
    )
    parser.add_argument(
        "--rocr-visible-devices",
        default=os.environ.get("ROCR_VISIBLE_DEVICES", "0"),
    )
    parser.add_argument("--process-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--settle-seconds", type=float, default=10.0)
    parser.add_argument(
        "--devcoredump-reader",
        choices=("direct", "sudo"),
        default="sudo",
        help="read a transient raw devcoredump directly or through non-interactive sudo",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ack-reset-risk", action="store_true")
    parser.add_argument("--approval-token")
    return parser


def plan_from_args(args: argparse.Namespace) -> IssueCapturePlan:
    cycles = args.cycles
    if cycles is None:
        cycles = 1 if args.submit_recreate else 8
    return IssueCapturePlan(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        compiler_version_file=args.compiler_version_file,
        python_executable=args.python_executable,
        transport=args.transport,
        cycles=cycles,
        submit_recreate=args.submit_recreate,
        allocation_mode=args.allocation_mode,
        buffer_mode=args.buffer_mode,
        timestamps=args.timestamps,
        quarantine_generations=args.quarantine_generations,
        hip_visible_devices=args.hip_visible_devices,
        rocr_visible_devices=args.rocr_visible_devices,
        process_timeout_seconds=args.process_timeout_seconds,
        settle_seconds=args.settle_seconds,
        devcoredump_reader=args.devcoredump_reader,
        execute=args.execute,
        reset_risk_acknowledged=args.ack_reset_risk,
        approval_token=args.approval_token,
    ).validated()


def _evidence_path(plan: IssueCapturePlan, name: str) -> Path:
    if plan.output_dir is None:
        return Path("<OUTPUT_DIR>") / name
    return plan.output_dir / name


def build_reproducer_invocation(plan: IssueCapturePlan) -> list[str]:
    command = [
        plan.python_executable,
        str(plan.repo_root / "scripts" / "pm4_lifecycle_repro.py"),
        "--transport",
        plan.transport,
        "--cycles",
        str(plan.cycles),
        "--queue-mode",
        "recreate",
        "--resource-mode",
        "recreate",
        "--allocation-mode",
        plan.allocation_mode,
        "--buffer-mode",
        plan.buffer_mode,
        "--quarantine-generations",
        str(plan.quarantine_generations),
        "--timeout-seconds",
        "5",
        "--json",
        str(_evidence_path(plan, "reproducer.json")),
        "--journal-jsonl",
        str(_evidence_path(plan, "lifecycle-events.jsonl")),
        "--submit" if plan.submit_recreate else "--no-submit",
    ]
    if plan.timestamps:
        command.append("--timestamps")
    if plan.compiler_version_file is not None:
        command.extend(
            [
                "--compiler-version-file",
                str(plan.compiler_version_file),
                "--require-cached-build",
            ]
        )
    if plan.submit_recreate:
        command.append("--ack-reset-risk")
    return command


def _child_environment(plan: IssueCapturePlan) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "HIP_VISIBLE_DEVICES": plan.hip_visible_devices,
            "ROCR_VISIBLE_DEVICES": plan.rocr_visible_devices,
            "GPU_MAX_HW_QUEUES": "1",
            "HIPENGINE_HIP_ARCH": "gfx1100",
            "PYTHONPATH": str(plan.repo_root),
        }
    )
    return env


def discover_devcoredump_data(
    devcoredump_root: Path = Path("/sys/class/devcoredump"),
    drm_root: Path = Path("/sys/class/drm"),
) -> list[Path]:
    candidates = list(devcoredump_root.glob("devcd*/data"))
    candidates.extend(drm_root.glob("card*/device/devcoredump/data"))
    return sorted(dict.fromkeys(path for path in candidates if path.exists()), key=str)


def classify_kernel_journal(text: str) -> dict[str, Any]:
    lowered = text.lower()
    address_zero = "address 0x0000000000000000" in lowered or "at 0x0" in lowered
    status = "0x00801431" in lowered
    sqc_data = "sqc (data)" in lowered
    client_10 = "client 10" in lowered or "(0xa)" in lowered
    issue_tuple = bool(address_zero and status and sqc_data and client_10)
    remove_queue = "remove_queue" in lowered and (
        "failed" in lowered or "failed to respond" in lowered
    )
    gpu_reset = "gpu reset begin" in lowered or "mode1 reset" in lowered or "mode2 reset" in lowered
    vram_lost = "vram is lost" in lowered
    if issue_tuple:
        classification = "reproduced_issue_6529_signature"
    elif any((remove_queue, gpu_reset, vram_lost)):
        classification = "other_amdgpu_recovery_event"
    else:
        classification = "no_issue_6529_signature_observed"
    return {
        "classification": classification,
        "issue_6529_fault_tuple": issue_tuple,
        "address_zero": address_zero,
        "fault_status_0x00801431": status,
        "sqc_data": sqc_data,
        "client_10": client_10,
        "remove_queue_failure": remove_queue,
        "gpu_reset": gpu_reset,
        "vram_lost": vram_lost,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _command_capture(
    output_dir: Path,
    name: str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout,
        )
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except (OSError, subprocess.SubprocessError) as exc:
        returncode = 127
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}\n"
    stdout_path = output_dir / f"{name}.stdout.txt"
    stderr_path = output_dir / f"{name}.stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "command": command,
        "returncode": returncode,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
    }


def _hip_device_probe(
    output_dir: Path,
    env: dict[str, str],
    python_executable: str,
) -> dict[str, Any]:
    probe = r'''
import ctypes
import json
hip = ctypes.CDLL("libamdhip64.so")
device = ctypes.c_int()
name = ctypes.create_string_buffer(256)
bdf = ctypes.create_string_buffer(32)
runtime_version = ctypes.c_int()
driver_version = ctypes.c_int()
checks = [
    ("hipGetDevice", hip.hipGetDevice(ctypes.byref(device))),
    ("hipDeviceGetName", hip.hipDeviceGetName(name, len(name), device.value)),
    ("hipDeviceGetPCIBusId", hip.hipDeviceGetPCIBusId(bdf, len(bdf), device.value)),
    ("hipRuntimeGetVersion", hip.hipRuntimeGetVersion(ctypes.byref(runtime_version))),
    ("hipDriverGetVersion", hip.hipDriverGetVersion(ctypes.byref(driver_version))),
]
failed = [(label, int(status)) for label, status in checks if int(status) != 0]
if failed:
    raise SystemExit(f"HIP probe failed: {failed}")
print(json.dumps({
    "hip_ordinal": int(device.value),
    "device_name": name.value.decode("utf-8", errors="replace"),
    "pci_bdf": bdf.value.decode("ascii"),
    "hip_runtime_version": int(runtime_version.value),
    "hip_driver_version": int(driver_version.value),
}, sort_keys=True))
'''
    capture = _command_capture(
        output_dir,
        "hip-device",
        [python_executable, "-c", probe],
        env=env,
        timeout=30,
    )
    if capture["returncode"] != 0:
        error = (output_dir / capture["stderr"]).read_text(encoding="utf-8").strip()
        raise RuntimeError(f"selected HIP device probe failed: {error}")
    try:
        payload = json.loads(
            (output_dir / capture["stdout"]).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"selected HIP device probe returned invalid JSON: {exc}") from exc
    if not payload.get("pci_bdf") or not payload.get("device_name"):
        raise RuntimeError("selected HIP device probe omitted PCI BDF or device name")
    return {"capture": capture, **payload}


def _capture_host_files(output_dir: Path) -> dict[str, Any]:
    text_files: dict[str, Any] = {}
    candidates = [
        Path("/proc/cmdline"),
        Path("/sys/module/amdgpu/srcversion"),
        Path("/sys/module/amdgpu/version"),
    ]
    candidates.extend(sorted(Path("/sys/module/amdgpu/parameters").glob("*")))
    for path in candidates:
        try:
            text_files[str(path)] = path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
        except OSError as exc:
            text_files[str(path)] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    firmware: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in (Path("/usr/lib/firmware/amdgpu"), Path("/lib/firmware/amdgpu")):
        for path in sorted(root.glob("gc_11_0_0_mes*.bin*")):
            resolved = str(path.resolve())
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            try:
                firmware.append(
                    {
                        "path": str(path),
                        "resolved_path": resolved,
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
            except OSError as exc:
                firmware.append(
                    {
                        "path": str(path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    payload = {"text_files": text_files, "mes_firmware": firmware}
    _write_json(output_dir / "host-files.json", payload)
    return payload


def _journal_cursor() -> str:
    result = subprocess.run(
        ["journalctl", "-k", "-n", "0", "--show-cursor", "--no-pager"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"kernel journal is unreadable: {result.stderr.strip()}")
    match = re.search(r"-- cursor: (\S+)", result.stdout)
    if match is None:
        raise RuntimeError("journalctl did not return a kernel-journal cursor")
    return match.group(1)


def _check_cached_builds(plan: IssueCapturePlan) -> dict[str, str]:
    assert plan.compiler_version_file is not None
    try:
        compiler_version = plan.compiler_version_file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"compiler version file is unreadable: {exc}") from exc
    if not compiler_version:
        raise RuntimeError("compiler version file is empty")
    with hip_target_arch_environment("gfx1100"):
        smoke = plan_smoke_add_build(compiler_version=compiler_version)
        native = plan_pm4_native_build(
            compiler_version=compiler_version, target_arch="gfx1100"
        )
    missing = [str(artifact.output_path) for artifact in (smoke, native) if not artifact.output_path.is_file()]
    if missing:
        raise RuntimeError(
            "required lifecycle builds are not cached; prepare them outside the capture window: "
            + ", ".join(missing)
        )
    return {
        "compiler_version_sha256": hashlib.sha256(compiler_version.encode("utf-8")).hexdigest(),
        "smoke_add": str(smoke.output_path),
        "pm4_native": str(native.output_path),
    }


def _devcoredump_enabled() -> bool:
    disabled = Path("/sys/class/devcoredump/disabled")
    try:
        return disabled.read_text(encoding="ascii").strip() == "0"
    except OSError:
        return False


class _DevcoredumpWatcher(threading.Thread):
    def __init__(self, output_dir: Path, *, reader: Literal["direct", "sudo"]) -> None:
        super().__init__(name="pm4-devcoredump-watcher", daemon=True)
        self.output_dir = output_dir
        self.reader = reader
        self.stop_event = threading.Event()
        self.seen: set[str] = set()
        self.records: list[dict[str, Any]] = []

    def _save_index(self) -> None:
        _write_json(self.output_dir / "devcoredump-index.json", self.records)

    def _capture(self, source: Path) -> None:
        key = str(source.resolve())
        if key in self.seen:
            return
        self.seen.add(key)
        index = len(self.records)
        destination = self.output_dir / f"devcoredump-{index:03d}.bin"
        record: dict[str, Any] = {
            "source": str(source),
            "resolved_source": key,
            "destination": destination.name,
            "reader": self.reader,
            "capture_started_at": datetime.now(timezone.utc).isoformat(),
            "status": "started",
        }
        self.records.append(record)
        self._save_index()
        try:
            with destination.open("xb") as output_handle:
                if self.reader == "sudo":
                    captured = subprocess.run(
                        ["sudo", "-n", "cat", "--", str(source)],
                        check=False,
                        stdout=output_handle,
                        stderr=subprocess.PIPE,
                        timeout=120,
                    )
                    if captured.returncode != 0:
                        error = captured.stderr.decode("utf-8", errors="replace").strip()
                        raise RuntimeError(
                            f"privileged devcoredump read failed ({captured.returncode}): {error}"
                        )
                else:
                    with source.open("rb", buffering=0) as input_handle:
                        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            record.update(
                {
                    "status": "captured",
                    "size_bytes": destination.stat().st_size,
                    "sha256": _sha256_file(destination),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        finally:
            record["capture_finished_at"] = datetime.now(timezone.utc).isoformat()
            self._save_index()

    def run(self) -> None:
        while not self.stop_event.is_set():
            for source in discover_devcoredump_data():
                self._capture(source)
            self.stop_event.wait(0.05)

    def stop(self) -> None:
        self.stop_event.set()


def _preflight(plan: IssueCapturePlan, output_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    if plan.repo_root != REPO_ROOT:
        raise RuntimeError(
            f"capture source root {plan.repo_root} does not match this script's root {REPO_ROOT}"
        )
    source = collect_repo_state(plan.repo_root)
    if source["dirty"] and plan.destructive:
        raise RuntimeError("destructive issue capture requires clean source")
    if not _devcoredump_enabled():
        raise RuntimeError("/sys/class/devcoredump is unavailable or disabled")
    existing_dumps = discover_devcoredump_data()
    if existing_dumps:
        raise RuntimeError(f"stale devcoredump nodes must be cleared first: {existing_dumps}")
    cursor = _journal_cursor()
    if plan.devcoredump_reader == "sudo":
        sudo = subprocess.run(
            ["sudo", "-n", "true"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        if sudo.returncode != 0:
            error = sudo.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                "non-interactive sudo is required for raw devcoredump capture; "
                f"configure it or select --devcoredump-reader direct: {error}"
            )
    cached_builds = _check_cached_builds(plan)
    arch_probe = _command_capture(output_dir, "amdgpu-arch", ["amdgpu-arch"], env=env)
    arch_text = (output_dir / arch_probe["stdout"]).read_text(encoding="utf-8")
    arches = list(dict.fromkeys(match.group(0).lower() for match in _ARCH_PATTERN.finditer(arch_text)))
    if arch_probe["returncode"] != 0 or arches != ["gfx1100"]:
        raise RuntimeError(
            f"selected visibility must expose only gfx1100 targets, observed {arches}"
        )
    hip_device = _hip_device_probe(output_dir, env, plan.python_executable)
    return {
        "source": source,
        "journal_cursor_before": cursor,
        "devcoredump_enabled": True,
        "devcoredump_reader": plan.devcoredump_reader,
        "existing_devcoredumps": [],
        "cached_builds": cached_builds,
        "visible_arches": arches,
        "hip_device": hip_device,
        "environment": {
            key: env.get(key)
            for key in (
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "GPU_MAX_HW_QUEUES",
                "HIPENGINE_HIP_ARCH",
                "PYTHONPATH",
            )
        },
    }


def _plan_payload(plan: IssueCapturePlan) -> dict[str, Any]:
    payload = asdict(plan)
    for field in ("repo_root", "output_dir", "compiler_version_file"):
        value = payload[field]
        payload[field] = None if value is None else str(value)
    payload["destructive"] = plan.destructive
    payload["issue"] = _ISSUE_URL
    payload["reproducer_command"] = build_reproducer_invocation(plan)
    payload["approval_token_required_for_destructive_execute"] = DESTRUCTIVE_APPROVAL_TOKEN
    payload["raw_evidence_publishable"] = False
    return payload


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass


def execute_capture(plan: IssueCapturePlan) -> dict[str, Any]:
    if not plan.execute or plan.output_dir is None:
        raise ValueError("execute_capture requires an executable plan and output directory")
    output_dir = plan.output_dir
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    env = _child_environment(plan)
    started_at = datetime.now(timezone.utc).isoformat()
    _write_json(output_dir / "plan.json", _plan_payload(plan))
    try:
        preflight = _preflight(plan, output_dir, env)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "kind": "hipengine_pm4_issue_capture",
            "status": "preflight_failed",
            "started_at": started_at,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "plan": _plan_payload(plan),
        }
        _write_json(output_dir / "manifest.json", failure)
        raise
    _write_json(output_dir / "preflight.json", preflight)

    system_commands = {
        "uname": ["uname", "-a"],
        "rocminfo": ["rocminfo"],
        "hipcc-version": ["hipcc", "--version"],
        "lspci": ["lspci", "-nnk"],
        "amdgpu-modinfo": ["modinfo", "amdgpu"],
        "kernel-command-line": ["cat", "/proc/cmdline"],
    }
    system_capture = {
        name: _command_capture(output_dir, name, command, env=env, timeout=60)
        for name, command in system_commands.items()
    }
    host_files = _capture_host_files(output_dir)

    journal_follow_path = output_dir / "kernel-journal-follow.txt"
    journal_follow_error_path = output_dir / "kernel-journal-follow.stderr.txt"
    journal_follow_out = journal_follow_path.open("x", encoding="utf-8")
    journal_follow_err = journal_follow_error_path.open("x", encoding="utf-8")
    journal_process: subprocess.Popen[Any] | None = None
    watcher = _DevcoredumpWatcher(
        output_dir,
        reader=plan.devcoredump_reader,
    )
    watcher_started = False
    watcher_joined = False
    command = build_reproducer_invocation(plan)
    child_stdout_path = output_dir / "reproducer.stdout.txt"
    child_stderr_path = output_dir / "reproducer.stderr.txt"
    child_started_at = datetime.now(timezone.utc).isoformat()
    returncode: int | None = None
    timed_out = False
    execution_error: dict[str, str] | None = None
    try:
        journal_process = subprocess.Popen(
            [
                "journalctl",
                "-k",
                "--after-cursor",
                preflight["journal_cursor_before"],
                "--follow",
                "--no-pager",
                "-o",
                "short-precise",
            ],
            stdout=journal_follow_out,
            stderr=journal_follow_err,
            text=True,
            start_new_session=True,
        )
        watcher.start()
        watcher_started = True
        with child_stdout_path.open("x", encoding="utf-8") as child_out, child_stderr_path.open(
            "x", encoding="utf-8"
        ) as child_err:
            process = subprocess.Popen(
                command,
                cwd=plan.repo_root,
                env=env,
                stdout=child_out,
                stderr=child_err,
                text=True,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=plan.process_timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_group(process)
                returncode = process.poll()
    except Exception as exc:
        execution_error = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if watcher_started and plan.settle_seconds:
            time.sleep(plan.settle_seconds)
        if watcher_started:
            watcher.stop()
            watcher.join(timeout=120)
            watcher_joined = not watcher.is_alive()
        if journal_process is not None:
            _terminate_process_group(journal_process)
        journal_follow_out.close()
        journal_follow_err.close()

    journal_after = _command_capture(
        output_dir,
        "kernel-journal-after-cursor",
        [
            "journalctl",
            "-k",
            "--after-cursor",
            preflight["journal_cursor_before"],
            "--no-pager",
            "-o",
            "short-precise",
        ],
        timeout=60,
    )
    coredump_metadata = _command_capture(
        output_dir,
        "process-coredump-metadata",
        ["coredumpctl", "list", "--since", child_started_at, "--no-pager"],
        timeout=30,
    )
    journal_text = ""
    for path in (journal_follow_path, output_dir / journal_after["stdout"]):
        try:
            journal_text += path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    classification = classify_kernel_journal(journal_text)

    reproducer = None
    reproducer_path = output_dir / "reproducer.json"
    if reproducer_path.is_file():
        try:
            reproducer = json.loads(reproducer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            reproducer = None
    if classification["issue_6529_fault_tuple"]:
        status = "reproduced_issue_6529_signature"
    elif execution_error is not None:
        status = "capture_execution_failed"
    elif timed_out:
        status = "reproducer_process_timeout"
    elif returncode == 0 and reproducer is not None and reproducer.get("status") == "pass":
        status = "completed_without_issue_6529_signature"
    else:
        status = "reproducer_failed_without_issue_6529_signature"

    active_devcoredumps = {
        str(record.get("destination"))
        for record in watcher.records
        if record.get("status") == "started"
    }
    files: list[dict[str, Any]] = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if (
            not path.is_file()
            or path.name == "manifest.json"
            or path.name in active_devcoredumps
        ):
            continue
        files.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "hipengine_pm4_issue_capture",
        "issue": _ISSUE_URL,
        "status": status,
        "destructive": plan.destructive,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "host": {"node": platform.node(), "platform": platform.platform()},
        "plan": _plan_payload(plan),
        "preflight": preflight,
        "child": {
            "command": command,
            "started_at": child_started_at,
            "returncode": returncode,
            "timed_out": timed_out,
            "execution_error": execution_error,
            "parsed_reproducer_status": None if reproducer is None else reproducer.get("status"),
        },
        "kernel_journal": classification,
        "devcoredumps": {
            "watcher_joined": watcher_joined,
            "records": watcher.records,
        },
        "system_capture": system_capture,
        "host_files": host_files,
        "journal_capture": journal_after,
        "process_coredump_metadata": coredump_metadata,
        "files": files,
        "publication": {
            "raw_evidence_publishable": False,
            "reason": "contains full GPU addresses, process metadata, journals, and possible raw devcoredumps",
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        plan = plan_from_args(parser.parse_args(argv))
    except ValueError as exc:
        parser.error(str(exc))
    if not plan.execute:
        print(json.dumps(_plan_payload(plan), sort_keys=True, indent=2))
        return 0
    try:
        manifest = execute_capture(plan)
    except Exception as exc:
        print(f"capture preflight/execution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0 if manifest["status"] == "completed_without_issue_6529_signature" else 1


if __name__ == "__main__":
    raise SystemExit(main())
