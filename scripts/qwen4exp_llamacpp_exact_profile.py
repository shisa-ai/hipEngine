#!/usr/bin/env python3
"""Profile exact-token llama.cpp prefill and one cached decode transition.

A wrapper starts llama-server as its child, waits while the controller warms the
exact cases, and then ``exec``s ``rocprofv3``. Preserving that parent-child
relationship satisfies normal Linux ptrace policy without changing host
security settings. One direct-attach session records only measured requests.
Client monotonic bounds identify each request in the shared trace.

Prefill requests evaluate the committed token array and sample one output.
Decode requests append that sampled root token and profile cached evaluation of
only the appended token, matching hipEngine live ``prompt_tokens + 1``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import (  # noqa: E402
    DEFAULT_FIXTURE,
    _git_metadata,
    _host_metadata,
    _post_json,
    _wait_for_health,
    load_fixture,
    sha256_path,
    token_ids_sha256,
)


def _select_case(fixture: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    matches = [row for row in fixture.get("cases", ()) if str(row.get("id")) == case_id]
    if len(matches) != 1:
        raise ValueError(f"fixture must contain exactly one case {case_id!r}")
    return dict(matches[0])


def _completion_payload(
    token_ids: Sequence[int], *, n_predict: int, cache_prompt: bool
) -> dict[str, Any]:
    return {
        "prompt": [int(token_id) for token_id in token_ids],
        "n_predict": int(n_predict),
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": 12345,
        "ignore_eos": True,
        "cache_prompt": bool(cache_prompt),
        "stream": False,
        "return_tokens": True,
    }


def _decode_prompt(case: Mapping[str, Any], prefill_response: Mapping[str, Any]) -> list[int]:
    output = prefill_response.get("tokens")
    if not isinstance(output, list) or len(output) != 1:
        raise ValueError("decode warmup must return exactly one output token")
    return [
        *(int(token_id) for token_id in case["prompt_token_ids"]),
        int(output[0]),
    ]


def _response_summary(response: Mapping[str, Any]) -> dict[str, Any]:
    timings = response.get("timings")
    if not isinstance(timings, Mapping):
        raise ValueError("llama.cpp response omitted timings")
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError("llama.cpp response omitted output token IDs")
    return {
        "prompt_n": int(timings.get("prompt_n") or 0),
        "prompt_ms": float(timings.get("prompt_ms") or 0.0),
        "predicted_n": int(timings.get("predicted_n") or 0),
        "predicted_ms": float(timings.get("predicted_ms") or 0.0),
        "output_token_ids": [int(token_id) for token_id in tokens],
        "output_token_ids_sha256": token_ids_sha256(tokens),
        "stop_type": response.get("stop_type"),
        "truncated": response.get("truncated"),
    }


def _server_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["ROCP_TOOL_ATTACH"] = "1"
    return environment


def _git_diff_sha256(path: Path) -> str | None:
    try:
        payload = subprocess.check_output(["git", "diff", "--binary"], cwd=path)
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(payload).hexdigest()


def _trace_hashes(trace_dir: Path) -> list[dict[str, Any]]:
    rows = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
        for path in sorted(trace_dir.rglob("*.csv"))
    ]
    if not any("kernel_trace" in Path(row["path"]).name for row in rows):
        raise RuntimeError(f"profiler did not emit a kernel trace under {trace_dir}")
    return rows


def _wait_for_pid(path: Path, process: subprocess.Popen[Any], timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text().strip():
            return int(path.read_text().strip())
        if process.poll() is not None:
            raise RuntimeError(
                f"server/profiler wrapper exited early with code {process.returncode}"
            )
        time.sleep(0.05)
    raise TimeoutError(f"server PID file did not appear within {timeout}s")


def _stop_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _request(
    args: argparse.Namespace, payload: Mapping[str, Any]
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    started = time.perf_counter()
    response = _post_json(
        args.host,
        args.port,
        "/completion",
        payload,
        args.request_timeout,
    )
    return {
        "payload": dict(payload),
        "request_monotonic_ns": {
            "start": started_ns,
            "end": time.monotonic_ns(),
        },
        "client_wall_s": time.perf_counter() - started,
        "response": _response_summary(response),
        "raw_response": response,
    }


def _wrapper_script(
    *,
    server_command: Sequence[str],
    server_log: Path,
    pid_file: Path,
    profiler_command: Sequence[str],
) -> str:
    return "\n".join(
        (
            "set -euo pipefail",
            f"{shlex.join(server_command)} >{shlex.quote(str(server_log))} 2>&1 &",
            'server_pid="$!"',
            f"printf '%s\\n' \"$server_pid\" >{shlex.quote(str(pid_file))}",
            "IFS= read -r _",
            f"exec {shlex.join(profiler_command[:2])} \"$server_pid\" "
            + shlex.join(profiler_command[3:]),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--engine-label", default="upstream_patched_hip_f1793c1c4")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18115)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--attach-settle-seconds", type=float, default=4.0)
    parser.add_argument("--attach-exit-timeout", type=float, default=120.0)
    parser.add_argument("--rocprof-bin", type=Path, default=Path("rocprofv3"))
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    fixture, fixture_sha256 = load_fixture(args.fixture)
    cases = [_select_case(fixture, case_id) for case_id in args.case_id]
    for path, description in (
        (args.server_bin, "server binary"),
        (args.source_root, "source root"),
        (args.model, "model"),
    ):
        if not path.exists():
            raise ValueError(f"{description} does not exist: {path}")
    if args.trace_root.exists():
        raise FileExistsError(f"refusing to overwrite trace root {args.trace_root}")
    args.trace_root.mkdir(parents=True)
    server_log_path = args.server_log or args.trace_root / "llama-server.log"
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_file = args.trace_root / "server.pid"
    profiler_log_path = args.trace_root / "rocprof-attach.log"
    server_command = [
        str(args.server_bin.resolve()),
        "-m",
        str(args.model.resolve()),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--parallel",
        "1",
        "--no-webui",
        *args.server_arg,
    ]
    profiler_command = [
        str(args.rocprof_bin),
        "--pid",
        "SERVER_PID",
        "--attach-children=false",
        "--attach-sync-output",
        "--kernel-trace",
        "--hip-trace",
        "--memory-copy-trace",
        "--memory-allocation-trace",
        "--output-format",
        "csv",
        "--output-directory",
        str(args.trace_root),
        "--output-file",
        "llama-exact",
    ]
    wrapper_script = _wrapper_script(
        server_command=server_command,
        server_log=server_log_path,
        pid_file=pid_file,
        profiler_command=profiler_command,
    )
    artifact: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_llamacpp_exact_profile",
        "status": "running",
        "measurement_class": "diagnostic_profiled_not_performance",
        "command": list(command),
        "host": _host_metadata(),
        "controller_source": _git_metadata(ROOT),
        "engine": args.engine_label,
        "fixture": str(args.fixture),
        "fixture_sha256": fixture_sha256,
        "model": str(args.model.resolve()),
        "server": {
            "binary": str(args.server_bin.resolve()),
            "binary_sha256": sha256_path(args.server_bin),
            "source": _git_metadata(args.source_root),
            "source_diff_sha256": _git_diff_sha256(args.source_root),
            "command": server_command,
            "attach_registration": "ROCP_TOOL_ATTACH=1",
            "ptrace_ownership": "wrapper starts server child then execs rocprofv3",
        },
        "profiler": {
            "command_template": profiler_command,
            "wrapper_script": wrapper_script,
            "log": str(profiler_log_path),
        },
        "protocol": {
            "prefill": "exact prompt, cache_prompt=false, n_predict=1",
            "decode": (
                "append the measured prefill root token, cache_prompt=true, "
                "profile evaluation of that one appended token"
            ),
            "temperature": 0.0,
            "top_k": 1,
            "seed": 12345,
            "attach_settle_seconds": float(args.attach_settle_seconds),
            "request_selection": "client CLOCK_MONOTONIC bounds into shared trace",
        },
        "warmups": [],
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")

    wrapper: subprocess.Popen[Any] | None = None
    server_pid: int | None = None
    profiler_started = False
    with profiler_log_path.open("wb") as profiler_log:
        try:
            wrapper = subprocess.Popen(
                ["bash", "-c", wrapper_script],
                stdin=subprocess.PIPE,
                stdout=profiler_log,
                stderr=subprocess.STDOUT,
                env=_server_environment(),
            )
            server_pid = _wait_for_pid(pid_file, wrapper, args.startup_timeout)
            artifact["server"]["pid"] = server_pid
            _wait_for_health(args.host, args.port, args.startup_timeout)
            for case in cases:
                prompt = [int(token_id) for token_id in case["prompt_token_ids"]]
                warmup = _request(
                    args,
                    _completion_payload(prompt, n_predict=1, cache_prompt=False),
                )
                artifact["warmups"].append(
                    {
                        "case_id": str(case["id"]),
                        "response": warmup["response"],
                    }
                )
            assert wrapper.stdin is not None
            wrapper.stdin.write(b"start\n")
            wrapper.stdin.flush()
            profiler_started = True
            time.sleep(float(args.attach_settle_seconds))
            if wrapper.poll() is not None:
                raise RuntimeError(
                    f"rocprofv3 attach exited before requests with code {wrapper.returncode}"
                )
            for case in cases:
                prompt = [int(token_id) for token_id in case["prompt_token_ids"]]
                prefill = _request(
                    args,
                    _completion_payload(prompt, n_predict=1, cache_prompt=False),
                )
                if int(prefill["response"]["prompt_n"]) != int(case["prompt_tokens"]):
                    raise RuntimeError(
                        f"{case['id']} prefill evaluated {prefill['response']['prompt_n']} "
                        f"tokens, expected {case['prompt_tokens']}"
                    )
                decode_prompt = _decode_prompt(case, prefill["raw_response"])
                decode = _request(
                    args,
                    _completion_payload(
                        decode_prompt, n_predict=1, cache_prompt=True
                    ),
                )
                if int(decode["response"]["prompt_n"]) > 2:
                    raise RuntimeError(
                        f"{case['id']} cached decode evaluated "
                        f"{decode['response']['prompt_n']} prompt tokens"
                    )
                prefill.pop("raw_response")
                decode.pop("raw_response")
                artifact["cases"].append(
                    {
                        "id": str(case["id"]),
                        "category": str(case["category"]),
                        "prompt_tokens": int(case["prompt_tokens"]),
                        "prompt_token_ids_sha256": str(
                            case["prompt_token_ids_sha256"]
                        ),
                        "prefill": prefill,
                        "decode_live_count": len(decode_prompt),
                        "decode_prompt_token_ids_sha256": token_ids_sha256(
                            decode_prompt
                        ),
                        "decode": decode,
                    }
                )
                args.output.write_text(json.dumps(artifact, indent=2) + "\n")
                print(
                    f"case={case['id']} prefill_prompt_n="
                    f"{prefill['response']['prompt_n']} decode_prompt_n="
                    f"{decode['response']['prompt_n']}",
                    flush=True,
                )
            wrapper.stdin.write(b"stop\n")
            wrapper.stdin.flush()
            wrapper.stdin.close()
            wrapper.wait(timeout=args.attach_exit_timeout)
            if wrapper.returncode != 0:
                raise RuntimeError(
                    f"rocprofv3 attach failed with code {wrapper.returncode}"
                )
            artifact["profiler"]["returncode"] = wrapper.returncode
            artifact["profiler"]["log_sha256"] = sha256_path(profiler_log_path)
            artifact["profiler"]["files"] = _trace_hashes(args.trace_root)
            artifact["status"] = "completed"
        except Exception as exc:
            artifact["status"] = "failed"
            artifact["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if wrapper is not None and wrapper.poll() is None:
                if profiler_started and wrapper.stdin is not None and not wrapper.stdin.closed:
                    try:
                        wrapper.stdin.write(b"stop\n")
                        wrapper.stdin.flush()
                        wrapper.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                try:
                    wrapper.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    wrapper.terminate()
                    try:
                        wrapper.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        wrapper.kill()
                        wrapper.wait(timeout=10)
            if server_pid is not None:
                _stop_pid(server_pid)
            profiler_log.flush()
            artifact["server"]["termination"] = "SIGTERM_then_SIGKILL_if_needed"
            artifact["server"]["log"] = str(server_log_path)
            if server_log_path.is_file():
                artifact["server"]["log_sha256"] = sha256_path(server_log_path)
            if profiler_log_path.is_file():
                artifact["profiler"]["log_sha256"] = sha256_path(profiler_log_path)
            args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args, command=[Path(sys.argv[0]).name, *sys.argv[1:]])
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
