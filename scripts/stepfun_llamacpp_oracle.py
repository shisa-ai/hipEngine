#!/usr/bin/env python3
"""Build or run a llama.cpp one-token oracle command for StepFun artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

DEFAULT_LLAMA_CLI = Path("/home/lhl/ai/llama.cpp-cpu/llama-cli")
DEFAULT_MODEL = Path("/data/models/gguf/Step-3.7-flash-Q3_K_L-00001-of-00003.gguf")
DEFAULT_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-all45-prompt-smoke.json"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--llama-cli", type=Path, default=DEFAULT_LLAMA_CLI)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--n-predict", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--llama-arg",
        action="append",
        default=None,
        help="Additional argument appended to llama-cli; repeat for each token (use --llama-arg=--flag for flags).",
    )
    parser.add_argument("--execute", action="store_true", help="Run llama-cli instead of only emitting the plan.")
    parser.add_argument(
        "--diagnostic-logs",
        action="store_true",
        help="Do not pass --log-disable, so llama.cpp load errors are captured in stderr.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write JSON output to this path instead of stdout.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args(argv)


def _llama_version(llama_cli: Path) -> str | None:
    if not llama_cli.exists():
        return None
    try:
        completed = subprocess.run(
            [str(llama_cli), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"unavailable: {type(exc).__name__}: {exc}"
    return (completed.stdout + completed.stderr).strip() or None


def _write_text_atomic(output: Path, text: str) -> None:
    """Atomically write text by replacing the destination with a flushed temp file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, output)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _emit_json(result: dict[str, object], *, pretty: bool, output: Path | None) -> None:
    text = json.dumps(result, indent=2 if pretty else None, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    _write_text_atomic(output, text)


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _comparison_fields(generated_text: str, expected_text: object) -> dict[str, object]:
    if not isinstance(expected_text, str):
        return {
            "generated_text": generated_text,
            "text_matches_expected_exact": None,
            "text_matches_expected_stripped": None,
        }
    return {
        "generated_text": generated_text,
        "text_matches_expected_exact": generated_text == expected_text,
        "text_matches_expected_stripped": generated_text.strip() == expected_text.strip(),
    }


def _blocker_fields(stderr: str) -> dict[str, object]:
    if "unknown model architecture: 'step35'" in stderr:
        return {
            "oracle_blocker_kind": "llama_cpp_missing_step35_architecture",
            "oracle_blocker_detail": "local llama.cpp build reports unknown model architecture: 'step35'",
            "step35_supported": False,
        }
    return {
        "oracle_blocker_kind": None,
        "oracle_blocker_detail": None,
        "step35_supported": None,
    }


def _partial_execution_result(
    result: dict[str, object],
    *,
    output: Path,
    timeout_s: float,
) -> dict[str, object]:
    """Return the pre-launch artifact written before llama-cli execution."""

    partial = dict(result)
    partial.update(
        {
            "status": "running",
            "timeout_s": timeout_s,
            "partial_artifact": True,
            "partial_artifact_reason": (
                "stepfun_llamacpp_oracle.py wrote this structured handoff artifact before "
                "launching llama-cli; the helper overwrites it with executed/timeout JSON "
                "when the child process completes or reaches timeout_s."
            ),
            "partial_artifact_overwrite_policy": "overwrite_on_execute_or_timeout",
            "partial_output_path": str(output),
            "stdout": "",
            "stderr": "",
            "generated_text": "",
            "text_matches_expected_exact": None,
            "text_matches_expected_stripped": None,
            "oracle_blocker_kind": "llama_cpp_oracle_in_progress",
            "oracle_blocker_detail": (
                "llama-cli execution has started but no comparable token or timeout result "
                "has been recorded yet"
            ),
            "step35_supported": None,
        }
    )
    return partial


class _SupervisorSignal(Exception):
    """Raised when an outer supervisor asks the oracle helper to stop."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:  # pragma: no cover - defensive for non-standard platforms
        return f"SIGNAL_{signum}"


def _terminate_process_group(
    proc: subprocess.Popen[bytes],
    termination: dict[str, object],
) -> tuple[str, str, dict[str, object]]:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:  # pragma: no cover - process exited during timeout handling
        termination["process_exited_before_signal"] = True
        termination["termination_path"] = "process_exited_before_killpg"
    try:
        stdout, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive fallback
        termination["fallback_proc_kill_used"] = True
        termination["termination_path"] = "killpg_sigkill_then_proc_kill"
        proc.kill()
        stdout, stderr = proc.communicate()
    return _as_text(stdout), _as_text(stderr), termination


def _run_with_timeout(
    command: list[str],
    timeout_s: float,
) -> tuple[str, int | None, str, str, float, dict[str, object] | None]:
    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    previous_handlers: dict[int, signal.Handlers] = {}

    def raise_supervisor_signal(signum: int, _frame: object) -> None:
        raise _SupervisorSignal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, raise_supervisor_signal)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            return (
                "executed",
                proc.returncode,
                _as_text(stdout),
                _as_text(stderr),
                time.perf_counter() - started,
                None,
            )
        except _SupervisorSignal as exc:
            for signum in previous_handlers:
                signal.signal(signum, signal.SIG_IGN)
            termination = {
                "timeout_reached": False,
                "timeout_s": timeout_s,
                "supervisor_signal_received": True,
                "supervisor_signal": _signal_name(exc.signum),
                "supervisor_signal_number": int(exc.signum),
                "process_group_started": True,
                "termination_method": "os.killpg",
                "termination_signal": "SIGKILL",
                "termination_signal_number": int(signal.SIGKILL),
                "termination_path": "supervisor_signal_killpg_then_communicate",
                "communicate_after_signal_timeout_s": 10.0,
                "process_exited_before_signal": False,
                "fallback_proc_kill_used": False,
            }
            stdout_text, stderr_text, termination = _terminate_process_group(proc, termination)
            return (
                "timeout",
                None,
                stdout_text,
                stderr_text,
                time.perf_counter() - started,
                termination,
            )
        except subprocess.TimeoutExpired as exc:
            termination = {
                "timeout_reached": True,
                "timeout_s": timeout_s,
                "process_group_started": True,
                "termination_method": "os.killpg",
                "termination_signal": "SIGKILL",
                "termination_signal_number": int(signal.SIGKILL),
                "termination_path": "killpg_sigkill_then_communicate",
                "communicate_after_signal_timeout_s": 10.0,
                "process_exited_before_signal": False,
                "fallback_proc_kill_used": False,
            }
            stdout_text, stderr_text, termination = _terminate_process_group(proc, termination)
            return (
                "timeout",
                None,
                _as_text(exc.stdout) + stdout_text,
                _as_text(exc.stderr) + stderr_text,
                time.perf_counter() - started,
                termination,
            )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact = json.loads(args.artifact.read_text())
    prompt = artifact["prompt"]
    command = [
        str(args.llama_cli),
        "--model",
        str(args.model),
        "--prompt",
        prompt,
        "--predict",
        str(args.n_predict),
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1",
        "--min-p",
        "0",
        "--repeat-penalty",
        "1",
        "--seed",
        "0",
        "--no-display-prompt",
        "--simple-io",
    ]
    if args.llama_arg:
        command.extend(args.llama_arg)
    if not args.diagnostic_logs:
        command.append("--log-disable")
    result: dict[str, object] = {
        "status": "planned",
        "artifact": str(args.artifact),
        "llama_cli": str(args.llama_cli),
        "llama_cpp_version": _llama_version(args.llama_cli),
        "model": str(args.model),
        "n_predict": args.n_predict,
        "diagnostic_logs": bool(args.diagnostic_logs),
        "extra_llama_args": list(args.llama_arg or ()),
        "command": command,
        "command_shell": shlex.join(command),
        "prompt": prompt,
        "prompt_length": artifact.get("prompt_length"),
        "expected_next_token_id": artifact.get("next_token_id"),
        "expected_next_token_text": artifact.get("next_token_text"),
        "expected_next_token_logit": artifact.get("next_token_logit"),
        "expected_top_tokens": artifact.get("top_tokens"),
        "comparison_policy": {
            "generated_text_source": "llama-cli stdout with --no-display-prompt --simple-io",
            "exact_text_match_field": "text_matches_expected_exact",
            "stripped_text_match_field": "text_matches_expected_stripped",
            "expected_text_field": "expected_next_token_text",
        },
        "note": (
            "Dry-run oracle plan by default. Use --execute only when the machine can afford a llama.cpp "
            "one-token run over the StepFun Q3_K_L GGUF shards; compare output/tokenization manually or with "
            "a follow-up parser before claiming parity."
        ),
    }
    if args.execute:
        if args.output is not None:
            _emit_json(
                _partial_execution_result(result, output=args.output, timeout_s=args.timeout_s),
                pretty=args.pretty,
                output=args.output,
            )
        status, returncode, stdout, stderr, elapsed_s, timeout_termination = _run_with_timeout(
            command, args.timeout_s
        )
        partial_output_fields = {
            "partial_output_written_before_launch": args.output is not None,
            "partial_output_path": str(args.output) if args.output is not None else None,
        }
        if status == "timeout":
            blocker = _blocker_fields(stderr)
            if blocker["oracle_blocker_kind"] is None:
                blocker = {
                    "oracle_blocker_kind": "llama_cpp_oracle_timeout",
                    "oracle_blocker_detail": "llama.cpp oracle timed out before producing a comparable token",
                    "step35_supported": None,
                }
            result.update(
                {
                    "status": "timeout",
                    "timeout_s": args.timeout_s,
                    "elapsed_s": elapsed_s,
                    "stdout": stdout,
                    "stderr": stderr,
                    "timeout_termination": timeout_termination,
                    **_comparison_fields(stdout, result.get("expected_next_token_text")),
                    **blocker,
                    **partial_output_fields,
                }
            )
        else:
            result.update(
                {
                    "status": "executed",
                    "returncode": returncode,
                    "elapsed_s": elapsed_s,
                    "stdout": stdout,
                    "stderr": stderr,
                    **_comparison_fields(stdout, result.get("expected_next_token_text")),
                    **_blocker_fields(stderr),
                    **partial_output_fields,
                }
            )
    _emit_json(result, pretty=args.pretty, output=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
