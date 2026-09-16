"""Deadline-enforced parent for fresh-process TP2/TP1 teacher controls.

One fresh subprocess per case (``scripts/tp2_teacher_child.py``), each of which
builds AND executes its session on its own main thread. This parent:

* enforces a wall deadline and always reaps the child (SIGTERM, then SIGKILL);
  it never closes a session while a child call is blocked, because it never
  touches the child's session at all;
* streams and retains the child's flushed progress, including partial output
  collected before a timeout, and records which phase (build vs execution) a
  timeout landed in;
* records the exact token prefix every case used, requires the requested
  length, and checks prefix consistency over the shared common length so a
  legitimate 16 -> 24 prefix progression is not rejected;
* snapshots host CPU load, this process's nice level, and per-GPU busy percent
  before each case, and can wait for an idle gate before starting. Missing GPU
  telemetry is *not* treated as idle.

A case is PASS only when the child exited 0 with no timeout and no failure,
reported CHILD_OK and CHILD_CLEANUP_OK, and produced exactly the requested
number of finite calls at the requested positions.

Never kill other jobs, never reset GPUs, never change firmware.

Usage::

    # three identical fresh 16-position XTX controls
    python scripts/tp2_teacher_bisect_parent.py \
        --case positions=16,devices=1,mode=tp1,schedule=eager,calls=1 \
        --repeat-per-case 3 --timeout 420 --idle-gate \
        --json /tmp/tp2_controls_16.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHILD = REPO_ROOT / "scripts" / "tp2_teacher_child.py"

_TOKEN_RE = re.compile(r"^CHILD_TOKENS (\[.*\])$")
_BUILD_RE = re.compile(r"^CHILD_BUILD_OK elapsed_s=([0-9.]+)")
_CALL_RE = re.compile(
    r"^CHILD_CALL call=(\d+) positions=(\d+) wall_s=([0-9.]+) "
    r"finite=(True|False) argmax_last=(-?\d+)"
)
_OK_RE = re.compile(r"^CHILD_OK calls=(\d+) positions=(\d+) total_wall_s=([0-9.]+)")
_FAIL_RE = re.compile(r"^CHILD_FAIL (.*)$")
_ALLOWED_KEYS = {"positions", "devices", "mode", "schedule", "calls", "reset_between"}


# -- host state -------------------------------------------------------------


def _parse_gpu_csv(text: str) -> list[int] | None:
    """Parse ``rocm-smi --showuse --csv`` into per-GPU busy percents."""

    percents: list[int] = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 2:
            continue
        # Header row is "device,GPU use (%)"; data rows are "card0,54" (the
        # CSV value has no percent sign).
        value = cells[1].rstrip("%").strip()
        if value.isdigit():
            percents.append(int(value))
    return percents or None


def gpu_busy_percent() -> list[int] | None:
    """Per-GPU busy percent from rocm-smi, or None when unavailable."""

    try:
        out = subprocess.run(
            ["rocm-smi", "--showuse", "--csv"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return _parse_gpu_csv(out.stdout)


def host_snapshot() -> dict[str, object]:
    load1, load5, load15 = os.getloadavg()
    gpus = gpu_busy_percent()
    return {
        "nice": os.nice(0),
        "loadavg": [round(load1, 3), round(load5, 3), round(load15, 3)],
        "gpu_busy_percent": gpus,
        "gpu_telemetry_available": gpus is not None,
    }


def wait_for_idle(
    *,
    max_load: float,
    max_gpu: float,
    timeout_s: float,
    poll_s: float = 5.0,
) -> tuple[bool, dict[str, object]]:
    """Wait (bounded) for a low-load, low-GPU window; return (idle, snapshot).

    Idle requires *both* a low load and available GPU telemetry showing every
    card below ``max_gpu``. Missing telemetry cannot prove idle, so it is not
    treated as idle.
    """

    deadline = time.perf_counter() + timeout_s
    snap = host_snapshot()
    while True:
        load1 = float(snap["loadavg"][0])
        gpus = snap["gpu_busy_percent"]
        gpu_ok = isinstance(gpus, list) and max(gpus) <= max_gpu
        if load1 <= max_load and gpu_ok:
            return True, snap
        if time.perf_counter() >= deadline:
            return False, snap
        time.sleep(poll_s)
        snap = host_snapshot()


# -- case parsing -----------------------------------------------------------


def parse_case(text: str) -> dict[str, object]:
    """Parse a comma-separated ``key=value`` case.

    Values may themselves contain commas (``devices=0,1``) because the split is
    only before a token that looks like ``key=``. Unknown or malformed keys are
    rejected rather than silently kept.
    """

    case: dict[str, object] = {}
    items = re.split(r",(?=[A-Za-z_][A-Za-z0-9_]*=)", text)
    for item in items:
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"malformed case item {item!r}; expected key=value")
        key, _, value = item.partition("=")
        key = key.strip()
        value = value.strip()
        if key not in _ALLOWED_KEYS:
            raise ValueError(f"unknown case key {key!r}; allowed={sorted(_ALLOWED_KEYS)}")
        if key in {"positions", "calls"}:
            if not re.fullmatch(r"-?\d+", value):
                raise ValueError(f"{key} must be an integer, got {value!r}")
            case[key] = int(value)
        elif key == "reset_between":
            if value.lower() not in {"0", "1", "true", "false", "yes", "no"}:
                raise ValueError(f"reset_between must be boolean, got {value!r}")
            case[key] = value.lower() in {"1", "true", "yes"}
        else:
            case[key] = value
    return case


# -- case execution ---------------------------------------------------------


def _parse(lines: list[str]) -> dict[str, object]:
    tokens = None
    build_s = None
    calls: list[dict[str, object]] = []
    ok = None
    fail = None
    exec_started = False
    cleanup_ok = False
    cleanup_fail = None
    for line in lines:
        m = _TOKEN_RE.match(line)
        if m:
            tokens = json.loads(m.group(1))
            continue
        m = _BUILD_RE.match(line)
        if m:
            build_s = float(m.group(1))
            continue
        if line.startswith("CHILD_EXEC_START"):
            exec_started = True
            continue
        m = _CALL_RE.match(line)
        if m:
            calls.append(
                {
                    "call": int(m.group(1)),
                    "positions": int(m.group(2)),
                    "wall_s": float(m.group(3)),
                    "finite": m.group(4) == "True",
                    "argmax_last": int(m.group(5)),
                }
            )
            continue
        m = _OK_RE.match(line)
        if m:
            ok = {
                "calls": int(m.group(1)),
                "positions": int(m.group(2)),
                "total_wall_s": float(m.group(3)),
            }
            continue
        m = _FAIL_RE.match(line)
        if m:
            fail = m.group(1)
            continue
        if line == "CHILD_CLEANUP_OK":
            cleanup_ok = True
        elif line.startswith("CHILD_CLEANUP_FAIL"):
            cleanup_fail = line
    return {
        "tokens": tokens,
        "build_s": build_s,
        "exec_started": exec_started,
        "calls": calls,
        "ok": ok,
        "fail": fail,
        "cleanup_ok": cleanup_ok,
        "cleanup_fail": cleanup_fail,
    }


def evaluate_case(
    parsed: dict[str, object],
    *,
    timed_out: bool,
    returncode: int | None,
    expected_calls: int,
    positions: int,
) -> tuple[str, list[str]]:
    """Fail-closed per-case verdict. Returns ``(verdict, reasons)``."""

    reasons: list[str] = []
    if timed_out:
        phase = "execution" if parsed["exec_started"] else "build"
        return f"TIMEOUT[{phase}]", [f"timeout in {phase}"]
    if returncode != 0:
        reasons.append(f"returncode={returncode}")
    if parsed["fail"] is not None:
        reasons.append(f"child failure: {parsed['fail']}")
    if parsed["ok"] is None:
        reasons.append("no CHILD_OK")
    else:
        if int(parsed["ok"]["calls"]) != expected_calls:  # type: ignore[index]
            reasons.append(
                f"ok.calls={parsed['ok']['calls']} expected {expected_calls}"  # type: ignore[index]
            )
        if int(parsed["ok"]["positions"]) != positions:  # type: ignore[index]
            reasons.append(
                f"ok.positions={parsed['ok']['positions']} expected {positions}"  # type: ignore[index]
            )
    if parsed["cleanup_fail"] is not None:
        reasons.append("cleanup failed")
    if not parsed["cleanup_ok"]:
        reasons.append("no CHILD_CLEANUP_OK")

    calls = parsed["calls"]
    if len(calls) != expected_calls:  # type: ignore[arg-type]
        reasons.append(f"call count={len(calls)} expected {expected_calls}")  # type: ignore[arg-type]
    else:
        for index, call in enumerate(calls):  # type: ignore[arg-type]
            if int(call["call"]) != index:
                reasons.append(f"call index {call['call']} != {index}")
            if int(call["positions"]) != positions:
                reasons.append(f"call {index} positions={call['positions']} expected {positions}")
            if not call["finite"]:
                reasons.append(f"call {index} non-finite")

    if reasons:
        return f"FAIL[{' | '.join(reasons)}]", reasons
    return "PASS", []


def prefix_status(
    tokens: list[int] | None,
    *,
    positions: int,
    reference: list[int] | None,
) -> tuple[bool, str]:
    """Check requested length and consistency with the reference prefix.

    Returns ``(consistent, detail)``. A shorter prefix is consistent when it
    equals the reference's leading slice; only a mismatch over the shared
    common length is a failure.
    """

    if tokens is None:
        return False, "no CHILD_TOKENS"
    if len(tokens) != positions:
        return False, f"token count {len(tokens)} != requested {positions}"
    if reference is None:
        return True, "reference"
    common = min(len(tokens), len(reference))
    if tokens[:common] != reference[:common]:
        return False, f"prefix mismatch over {common} tokens"
    return True, "consistent"


def supervise(
    cmd: list[str],
    *,
    timeout: float,
    grace: float,
) -> tuple[list[str], bool, int | None, float]:
    """Run ``cmd``, stream its output, enforce the deadline, and reap it.

    Returns ``(lines, timed_out, returncode, wall_s)``. On timeout the child is
    SIGTERM'd, given ``grace`` seconds, then SIGKILL'd and unconditionally
    reaped. Output collected before the kill is retained.
    """

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []

    def _reader() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            lines.append(line)
            print(f"    [child] {line}", flush=True)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        print(f"  DEADLINE {timeout:.0f}s reached; SIGTERM and reap", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            print(f"  grace {grace:.0f}s expired; SIGKILL", flush=True)
            proc.kill()
            proc.wait()  # unconditional reap
    thread.join(timeout=10.0)
    return lines, timed_out, proc.returncode, time.perf_counter() - t0


def run_case(
    *,
    positions: int,
    devices: str,
    mode: str,
    schedule: str | None,
    calls: int,
    reset_between: bool,
    timeout: float,
    grace: float,
    idle_gate: bool,
    require_idle: bool,
    max_load: float,
    max_gpu: float,
    idle_timeout: float,
) -> dict[str, object]:
    snapshot = host_snapshot()
    idle = None
    if idle_gate:
        idle, snapshot = wait_for_idle(
            max_load=max_load, max_gpu=max_gpu, timeout_s=idle_timeout
        )
        if not idle and require_idle:
            print(
                f"CONTENDED: no idle window within {idle_timeout:.0f}s "
                f"(load={snapshot['loadavg'][0]}, gpu={snapshot['gpu_busy_percent']}, "
                f"telemetry={snapshot['gpu_telemetry_available']})",
                flush=True,
            )
            return {
                "case": {
                    "positions": positions,
                    "devices": devices,
                    "mode": mode,
                    "schedule": schedule,
                    "calls": calls,
                    "reset_between": reset_between,
                },
                "verdict": "CONTENDED",
                "snapshot": snapshot,
                "idle": idle,
                "lines": [],
            }

    cmd = [
        sys.executable,
        "-u",
        str(CHILD),
        "--positions",
        str(positions),
        "--devices",
        devices,
        "--mode",
        mode,
        "--calls",
        str(calls),
    ]
    if schedule:
        cmd += ["--schedule", schedule]
    if reset_between:
        cmd.append("--reset-between")

    label = f"positions={positions} devices={devices} mode={mode} calls={calls}"
    print(f"RUN {label} schedule={schedule} reset_between={reset_between}", flush=True)
    print(
        f"  snapshot nice={snapshot['nice']} load={snapshot['loadavg']} "
        f"gpu={snapshot['gpu_busy_percent']} idle={idle}",
        flush=True,
    )

    lines, timed_out, returncode, wall = supervise(cmd, timeout=timeout, grace=grace)
    parsed = _parse(lines)
    verdict, reasons = evaluate_case(
        parsed,
        timed_out=timed_out,
        returncode=returncode,
        expected_calls=calls,
        positions=positions,
    )

    result = {
        "case": {
            "positions": positions,
            "devices": devices,
            "mode": mode,
            "schedule": schedule,
            "calls": calls,
            "reset_between": reset_between,
        },
        "verdict": verdict,
        "reasons": reasons,
        "wall_s": wall,
        "snapshot": snapshot,
        "idle": idle,
        "timed_out": timed_out,
        "returncode": returncode,
        "tokens": parsed["tokens"],
        "build_s": parsed["build_s"],
        "exec_started": parsed["exec_started"],
        "calls": parsed["calls"],
        "ok": parsed["ok"],
        "fail": parsed["fail"],
        "cleanup_ok": parsed["cleanup_ok"],
        "lines": lines,
    }
    print(
        f"  {verdict} wall={wall:.1f}s build_s={parsed['build_s']} "
        f"calls_done={len(parsed['calls'])}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="comma key=value case, e.g. positions=16,devices=1,mode=tp1,calls=1",
    )
    parser.add_argument("--repeat-per-case", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument("--grace", type=float, default=15.0)
    parser.add_argument("--idle-gate", action="store_true")
    parser.add_argument(
        "--require-idle",
        action="store_true",
        help="abort a case with CONTENDED instead of running it contended "
        "(implies --idle-gate)",
    )
    parser.add_argument("--max-load", type=float, default=2.0)
    parser.add_argument("--max-gpu", type=float, default=5.0)
    parser.add_argument("--idle-timeout", type=float, default=300.0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    if args.repeat_per_case < 1:
        parser.error(f"--repeat-per-case must be >= 1, got {args.repeat_per_case}")
    if args.timeout <= 0:
        parser.error(f"--timeout must be > 0, got {args.timeout}")
    if args.grace < 0:
        parser.error(f"--grace must be >= 0, got {args.grace}")
    idle_gate = args.idle_gate or args.require_idle

    try:
        cases = [parse_case(text) for text in args.case]
    except ValueError as error:
        parser.error(str(error))
    if not cases:
        cases = [
            {
                "positions": 16,
                "devices": "1",
                "mode": "tp1",
                "schedule": "eager",
                "calls": 1,
            }
        ]

    results: list[dict[str, object]] = []
    reference_tokens: list[int] | None = None
    for case in cases:
        for _ in range(args.repeat_per_case):
            result = run_case(
                positions=int(case.get("positions", 16)),
                devices=str(case.get("devices", "1")),
                mode=str(case.get("mode", "tp1")),
                schedule=case.get("schedule"),  # type: ignore[arg-type]
                calls=int(case.get("calls", 1)),
                reset_between=bool(case.get("reset_between", False)),
                timeout=args.timeout,
                grace=args.grace,
                idle_gate=idle_gate,
                require_idle=args.require_idle,
                max_load=args.max_load,
                max_gpu=args.max_gpu,
                idle_timeout=args.idle_timeout,
            )
            tokens = result.get("tokens")
            consistent, detail = prefix_status(
                list(tokens) if tokens is not None else None,  # type: ignore[arg-type]
                positions=int(case.get("positions", 16)),
                reference=reference_tokens,
            )
            result["prefix_consistent"] = consistent
            result["prefix_detail"] = detail
            if reference_tokens is None and consistent and tokens is not None:
                reference_tokens = list(tokens)  # type: ignore[arg-type]
            results.append(result)

    summary = {
        "reference_tokens": reference_tokens,
        "results": results,
    }
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=1) + "\n")
        print(f"artifact: {out}", flush=True)

    passed = sum(1 for r in results if r.get("verdict") == "PASS")
    prefixes_ok = all(r.get("prefix_consistent", False) for r in results)
    print(
        f"summary: {passed}/{len(results)} PASS (prefixes consistent={prefixes_ok})",
        flush=True,
    )
    return 0 if passed == len(results) and prefixes_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
