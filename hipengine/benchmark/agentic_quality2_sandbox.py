"""Bubblewrap sandbox for untrusted AGENTIC-QUALITY2 Python functions."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class SandboxLimits:
    """Hard per-test limits for one generated-code sandbox process."""

    wall_seconds: float = 2.0
    cpu_seconds: int = 1
    memory_bytes: int = 128 << 20
    file_bytes: int = 64 << 10
    output_bytes: int = 16 << 10
    processes: int = 1
    open_files: int = 32

    def __post_init__(self) -> None:
        values = (
            self.wall_seconds,
            self.cpu_seconds,
            self.memory_bytes,
            self.file_bytes,
            self.output_bytes,
            self.processes,
            self.open_files,
        )
        if any(isinstance(value, bool) or float(value) <= 0 for value in values):
            raise ValueError("sandbox limits must be positive")


# Expected outputs never enter this process. The host invokes a fresh namespace
# for each hidden input and compares the returned JSON value outside the sandbox.
_RUNNER = r"""from __future__ import annotations
import importlib.util
import json

with open("/input/request.json", "r", encoding="utf-8") as handle:
    request = json.load(handle)
spec = importlib.util.spec_from_file_location("candidate", "/input/source.py")
if spec is None or spec.loader is None:
    raise RuntimeError("candidate module could not be loaded")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
function = getattr(module, request["entry_point"], None)
if not callable(function):
    raise RuntimeError(f"entry point {request['entry_point']!r} is not callable")
try:
    observed = function(*request["args"], **request["kwargs"])
    json.dumps(observed, ensure_ascii=False)
    result = {"returned": True, "observed": observed, "failure": None}
except BaseException as exc:
    result = {
        "returned": False,
        "observed": None,
        "failure": {"kind": "exception", "type": type(exc).__name__},
    }
print("AQ2_RESULT:" + json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
"""


class AgenticQuality2Sandbox:
    """Run generated functions with no ambient host/network/device access."""

    def __init__(
        self,
        *,
        bwrap_path: str | Path | None = None,
        python_path: str | Path = "/usr/bin/python3",
        prlimit_path: str | Path = "/usr/bin/prlimit",
        limits: SandboxLimits | None = None,
    ) -> None:
        discovered = shutil.which("bwrap") if bwrap_path is None else str(bwrap_path)
        self.bwrap_path = None if discovered is None else Path(discovered)
        self.python_path = Path(python_path)
        self.prlimit_path = Path(prlimit_path)
        self.limits = limits or SandboxLimits()

    @staticmethod
    def _validate_source_imports(source: str, allowed_imports: Sequence[str]) -> str | None:
        try:
            tree = ast.parse(str(source), mode="exec")
        except SyntaxError as exc:
            return f"syntax error: {exc.msg}"
        allowed = {str(value).split(".", 1)[0] for value in allowed_imports}
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [str(node.module or "").split(".", 1)[0]]
            denied = [name for name in names if name not in allowed]
            if denied:
                return f"undeclared imports: {sorted(denied)}"
        return None

    @staticmethod
    def _read_bounded(path: Path, limit: int) -> tuple[str, bool]:
        with path.open("rb") as handle:
            payload = handle.read(int(limit) + 1)
        truncated = len(payload) > int(limit)
        payload = payload[: int(limit)]
        return payload.decode("utf-8", errors="replace"), truncated

    @staticmethod
    def _sha256_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _truncate_text(value: str, limit: int) -> tuple[str, bool]:
        payload = str(value).encode("utf-8")
        truncated = len(payload) > int(limit)
        return payload[: int(limit)].decode("utf-8", errors="ignore"), truncated

    def _blocked(self, reason: str) -> dict[str, Any]:
        return {
            "status": "blocked_sandbox",
            "reason": str(reason),
            "tests_attempted": 0,
            "tests_passed": 0,
            "failure": None,
            "stdout": "",
            "stderr": "",
            "stdout_sha256": self._sha256_text(""),
            "stderr_sha256": self._sha256_text(""),
            "output_truncated": False,
            "network_isolated": False,
            "filesystem_isolated": False,
            "device_isolated": False,
            "environment_cleared": False,
            "hidden_expected_exposed": False,
            "process_group_killed": False,
            "scratch_cleaned": True,
        }

    def _base_command(self, source: Path, request: Path, runner: Path) -> list[str]:
        limits = self.limits
        command = [
            str(self.bwrap_path),
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib",
            "/lib",
        ]
        if Path("/lib64").exists():
            command.extend(("--ro-bind", "/lib64", "/lib64"))
        command.extend(
            (
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--tmpfs",
                "/work",
                "--dir",
                "/input",
                "--ro-bind",
                str(source),
                "/input/source.py",
                "--ro-bind",
                str(request),
                "/input/request.json",
                "--ro-bind",
                str(runner),
                "/input/runner.py",
                "--chdir",
                "/work",
                str(self.prlimit_path),
                f"--cpu={int(limits.cpu_seconds)}:{int(limits.cpu_seconds)}",
                f"--as={int(limits.memory_bytes)}:{int(limits.memory_bytes)}",
                f"--fsize={int(limits.file_bytes)}:{int(limits.file_bytes)}",
                f"--nproc={int(limits.processes)}:{int(limits.processes)}",
                f"--nofile={int(limits.open_files)}:{int(limits.open_files)}",
                "--core=0:0",
                "--",
                str(self.python_path),
                "-I",
                "-S",
                "/input/runner.py",
            )
        )
        return command

    def _run_one(
        self,
        *,
        command: Sequence[str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> dict[str, Any]:
        timed_out = False
        process_group_killed = False
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
                close_fds=True,
            )
            try:
                returncode = process.wait(timeout=float(self.limits.wall_seconds))
            except subprocess.TimeoutExpired:
                timed_out = True
                process_group_killed = True
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=2.0)
        stdout, stdout_truncated = self._read_bounded(
            stdout_path,
            int(self.limits.output_bytes),
        )
        stderr, stderr_truncated = self._read_bounded(
            stderr_path,
            int(self.limits.output_bytes),
        )
        result_payload: dict[str, Any] | None = None
        visible_stdout: list[str] = []
        for line in stdout.splitlines():
            if line.startswith("AQ2_RESULT:"):
                try:
                    candidate = json.loads(line[len("AQ2_RESULT:") :])
                except json.JSONDecodeError:
                    candidate = None
                if isinstance(candidate, dict):
                    result_payload = candidate
            else:
                visible_stdout.append(line)
        stdout = "\n".join(visible_stdout)
        if visible_stdout:
            stdout += "\n"
        stdout, visible_stdout_truncated = self._truncate_text(
            stdout,
            int(self.limits.output_bytes),
        )
        stderr, visible_stderr_truncated = self._truncate_text(
            stderr,
            int(self.limits.output_bytes),
        )
        return {
            "timed_out": timed_out,
            "process_group_killed": process_group_killed,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": bool(stdout_truncated or visible_stdout_truncated),
            "stderr_truncated": bool(stderr_truncated or visible_stderr_truncated),
            "result": result_payload,
        }

    def run_code_case(
        self,
        *,
        source: str,
        entry_point: str,
        hidden_tests: Sequence[dict[str, Any]],
        scratch_root: str | Path,
        allowed_imports: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Run hidden inputs separately; expected values remain host-only."""

        if self.bwrap_path is None or not self.bwrap_path.is_file():
            return self._blocked("bubblewrap is unavailable")
        if not self.python_path.is_file() or not self.prlimit_path.is_file():
            return self._blocked("sandbox interpreter or prlimit is unavailable")
        source_error = self._validate_source_imports(source, allowed_imports)
        if source_error is not None:
            result = self._blocked(source_error)
            result.update(
                {
                    "status": "failed",
                    "network_isolated": True,
                    "filesystem_isolated": True,
                    "device_isolated": True,
                    "environment_cleared": True,
                }
            )
            return result
        root = Path(scratch_root)
        root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="aq2-sandbox-", dir=root))
        source_path = temporary / "source.py"
        runner_path = temporary / "runner.py"
        source_path.write_text(str(source), encoding="utf-8")
        runner_path.write_text(_RUNNER, encoding="utf-8")
        tests_attempted = 0
        tests_passed = 0
        status = "passed"
        reason: str | None = None
        failure: dict[str, Any] | None = None
        process_group_killed = False
        returncode: int | None = 0
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        output_truncated = False
        try:
            for index, hidden in enumerate(hidden_tests):
                # Deliberately omit expected from the sandbox-visible request.
                request_path = temporary / f"request-{index}.json"
                request_path.write_text(
                    json.dumps(
                        {
                            "entry_point": str(entry_point),
                            "args": list(hidden["args"]),
                            "kwargs": dict(hidden["kwargs"]),
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                stdout_path = temporary / f"stdout-{index}.log"
                stderr_path = temporary / f"stderr-{index}.log"
                execution = self._run_one(
                    command=self._base_command(source_path, request_path, runner_path),
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                tests_attempted += 1
                process_group_killed = bool(
                    process_group_killed or execution["process_group_killed"]
                )
                returncode = int(execution["returncode"])
                stdout_parts.append(str(execution["stdout"]))
                stderr_parts.append(str(execution["stderr"]))
                output_truncated = bool(
                    output_truncated
                    or execution["stdout_truncated"]
                    or execution["stderr_truncated"]
                )
                result_payload = execution["result"]
                if execution["timed_out"]:
                    status, reason = "timeout", "wall timeout"
                    failure = {"index": index, "kind": "timeout"}
                    break
                if returncode != 0 or not isinstance(result_payload, dict):
                    status, reason = "failed", f"sandbox process exited {returncode}"
                    failure = {"index": index, "kind": "process_failure"}
                    break
                if result_payload.get("returned") is not True:
                    status, reason = "failed", "candidate raised an exception"
                    failure = {
                        "index": index,
                        **dict(result_payload.get("failure") or {"kind": "exception"}),
                    }
                    break
                if result_payload.get("observed") != hidden["expected"]:
                    status, reason = "failed", "hidden test failed"
                    failure = {"index": index, "kind": "wrong_result"}
                    break
                tests_passed += 1
            stdout, stdout_truncated = self._truncate_text(
                "".join(stdout_parts),
                int(self.limits.output_bytes),
            )
            stderr, stderr_truncated = self._truncate_text(
                "".join(stderr_parts),
                int(self.limits.output_bytes),
            )
            output_truncated = bool(output_truncated or stdout_truncated or stderr_truncated)
            return {
                "status": status,
                "reason": reason,
                "tests_attempted": tests_attempted,
                "tests_passed": tests_passed,
                "failure": failure,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_sha256": self._sha256_text(stdout),
                "stderr_sha256": self._sha256_text(stderr),
                "output_truncated": output_truncated,
                "network_isolated": True,
                "filesystem_isolated": True,
                "device_isolated": True,
                "environment_cleared": True,
                "hidden_expected_exposed": False,
                "process_group_killed": process_group_killed,
                "scratch_cleaned": True,
                "limits": {
                    "scope": "per_hidden_test",
                    "wall_seconds": float(self.limits.wall_seconds),
                    "cpu_seconds": int(self.limits.cpu_seconds),
                    "memory_bytes": int(self.limits.memory_bytes),
                    "file_bytes": int(self.limits.file_bytes),
                    "output_bytes": int(self.limits.output_bytes),
                    "processes": int(self.limits.processes),
                    "open_files": int(self.limits.open_files),
                },
            }
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


__all__ = ["AgenticQuality2Sandbox", "SandboxLimits"]
