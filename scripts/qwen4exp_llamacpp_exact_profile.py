#!/usr/bin/env python3
"""Profile exact-token llama.cpp prefill and one cached decode transition.

The server runs normally. ``rocprofv3`` attaches directly to the already-loaded
server for each request, so model load, warmup, and the Python controller are
outside the trace. Prefill requests evaluate the exact committed token array and
sample one output. Decode requests first cache that prompt, append its sampled
root token, and then profile evaluation of only the appended token. The latter
matches hipEngine's live ``prompt_tokens + 1`` transition without profiling a
nested launcher process or relying on repeated-token ``llama-bench`` input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    _terminate,
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


def _decode_prompt(case: Mapping[str, Any], warm_response: Mapping[str, Any]) -> list[int]:
    output = warm_response.get("tokens")
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
    rows = []
    for path in sorted(trace_dir.rglob("*.csv")):
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    if not any("kernel_trace" in Path(row["path"]).name for row in rows):
        raise RuntimeError(f"profiler did not emit a kernel trace under {trace_dir}")
    return rows


def _profile_request(
    *,
    args: argparse.Namespace,
    server_pid: int,
    label: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    trace_dir = args.trace_root / label
    if trace_dir.exists():
        raise FileExistsError(f"refusing to overwrite trace directory {trace_dir}")
    trace_dir.mkdir(parents=True)
    profiler_log_path = trace_dir / "rocprof-attach.log"
    profiler_command = [
        str(args.rocprof_bin),
        "--pid",
        str(server_pid),
        "--attach-children=false",
        "--attach-sync-output",
        "--kernel-trace",
        "--hip-trace",
        "--memory-copy-trace",
        "--memory-allocation-trace",
        "--output-format",
        "csv",
        "--output-directory",
        str(trace_dir),
        "--output-file",
        label,
    ]
    with profiler_log_path.open("wb") as profiler_log:
        profiler = subprocess.Popen(
            profiler_command,
            stdin=subprocess.PIPE,
            stdout=profiler_log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        try:
            time.sleep(float(args.attach_settle_seconds))
            if profiler.poll() is not None:
                raise RuntimeError(
                    f"rocprofv3 attach exited before request with code {profiler.returncode}"
                )
            started_ns = time.monotonic_ns()
            started = time.perf_counter()
            response = _post_json(
                args.host,
                args.port,
                "/completion",
                payload,
                args.request_timeout,
            )
            client_wall_s = time.perf_counter() - started
            finished_ns = time.monotonic_ns()
            assert profiler.stdin is not None
            profiler.stdin.write(b"\n")
            profiler.stdin.flush()
            profiler.stdin.close()
            profiler.wait(timeout=args.attach_exit_timeout)
        except Exception:
            if profiler.poll() is None:
                profiler.terminate()
                try:
                    profiler.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    profiler.kill()
                    profiler.wait(timeout=10)
            raise
    if profiler.returncode != 0:
        raise RuntimeError(f"rocprofv3 attach failed with code {profiler.returncode}")
    return {
        "label": label,
        "payload": dict(payload),
        "request_monotonic_ns": {"start": started_ns, "end": finished_ns},
        "client_wall_s": client_wall_s,
        "response": _response_summary(response),
        "profiler": {
            "command": profiler_command,
            "returncode": profiler.returncode,
            "log": str(profiler_log_path),
            "log_sha256": sha256_path(profiler_log_path),
            "trace_dir": str(trace_dir),
            "files": _trace_hashes(trace_dir),
        },
    }


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
    args.trace_root.mkdir(parents=True, exist_ok=True)
    server_log_path = args.server_log or args.trace_root / "llama-server.log"
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
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
        },
        "protocol": {
            "prefill": "exact prompt, cache_prompt=false, n_predict=1",
            "decode": (
                "warm exact prompt with cache_prompt=true, append its sampled root token, "
                "then profile cached evaluation of that one appended token"
            ),
            "temperature": 0.0,
            "top_k": 1,
            "seed": 12345,
            "attach_settle_seconds": float(args.attach_settle_seconds),
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    with server_log_path.open("wb") as server_log:
        server = subprocess.Popen(
            server_command,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=_server_environment(),
        )
        try:
            _wait_for_health(args.host, args.port, args.startup_timeout)
            for case in cases:
                prompt = [int(token_id) for token_id in case["prompt_token_ids"]]
                warm_prefill = _post_json(
                    args.host,
                    args.port,
                    "/completion",
                    _completion_payload(prompt, n_predict=1, cache_prompt=False),
                    args.request_timeout,
                )
                prefill = _profile_request(
                    args=args,
                    server_pid=server.pid,
                    label=f"{case['id']}-prefill",
                    payload=_completion_payload(
                        prompt, n_predict=1, cache_prompt=False
                    ),
                )
                if int(prefill["response"]["prompt_n"]) != int(case["prompt_tokens"]):
                    raise RuntimeError(
                        f"{case['id']} prefill evaluated {prefill['response']['prompt_n']} "
                        f"tokens, expected {case['prompt_tokens']}"
                    )
                warm_decode = _post_json(
                    args.host,
                    args.port,
                    "/completion",
                    _completion_payload(prompt, n_predict=1, cache_prompt=True),
                    args.request_timeout,
                )
                decode_prompt = _decode_prompt(case, warm_decode)
                decode = _profile_request(
                    args=args,
                    server_pid=server.pid,
                    label=f"{case['id']}-decode-live{len(decode_prompt)}",
                    payload=_completion_payload(
                        decode_prompt, n_predict=1, cache_prompt=True
                    ),
                )
                if int(decode["response"]["prompt_n"]) > 2:
                    raise RuntimeError(
                        f"{case['id']} cached decode evaluated "
                        f"{decode['response']['prompt_n']} prompt tokens"
                    )
                artifact["cases"].append(
                    {
                        "id": str(case["id"]),
                        "category": str(case["category"]),
                        "prompt_tokens": int(case["prompt_tokens"]),
                        "prompt_token_ids_sha256": str(
                            case["prompt_token_ids_sha256"]
                        ),
                        "warm_prefill": _response_summary(warm_prefill),
                        "prefill": prefill,
                        "warm_decode": _response_summary(warm_decode),
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
            artifact["status"] = "completed"
        except Exception as exc:
            artifact["status"] = "failed"
            artifact["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _terminate(server)
            server_log.flush()
            artifact["server"]["returncode"] = server.returncode
            artifact["server"]["log"] = str(server_log_path)
            artifact["server"]["log_sha256"] = sha256_path(server_log_path)
            args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args, command=[Path(sys.argv[0]).name, *sys.argv[1:]])
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
