#!/usr/bin/env python3
"""Run one llama.cpp-family server against the pinned UD-Q4_K_XL GGUF and time prefill.

Every arm is a different engine, so the comparison is only meaningful if the
file, the KV dtype, the attention mode, the context, the batch sizes, the
threads, the case set and the host are identical. Those are all arguments here
and all recorded in the output, together with the server binary's hash and, when
the source tree is a git checkout, its commit and dirty state.

Requests are prefill-only (``n_predict=1``) and sent as exact token ids from the
canonical fixture, so the measured window is prompt processing and nothing else.
``prompt_ms`` from the server's own timings is the authoritative rate; the
client wall is recorded beside it for the reader who wants to see the overhead.

``--profile`` wraps the server in ``rocprofv3`` and buckets kernel time by name.
That is attribution, not a rate: the profiled run is slower and its
``prompt_ms`` is not comparable to an unprofiled one.

Nothing here evaluates correctness. It records the generated token id so a
result that is obviously not the model can be spotted, and says so.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL/"
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
)
DEFAULT_FIXTURE = (
    REPO_ROOT / "benchmarks" / "fixtures" / "qwen4exp_canonical_ar_p512_p1024_p4096.json"
)

ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("moe_routed", r"routed|moe_|mul_mat_q_routed|expert"),
    ("attention", r"fattn|flash_attn|attn|softmax|ssm|conv|gdn|delta"),
    ("quantize_pack", r"quantize_row|quantize_mmq|quantize_q8|im2col|dequantize"),
    ("elementwise_norm", r"norm|silu|gelu|add|mul_|cpy|scale|sqrt|tanh|sigmoid|clamp|rope"),
    ("gemm_plain", r"mul_mat|gemm|mmq|wmma|rocblas|Cijk"),
)


def _role_for(name: str) -> str:
    lowered = name.lower()
    for role, pattern in ROLE_PATTERNS:
        if re.search(pattern, lowered):
            return role
    return "other"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _source_state(tree: Path | None) -> dict:
    if tree is None:
        return {}
    state: dict = {"path": str(tree)}
    if not (tree / ".git").exists():
        return state
    try:
        state["head"] = subprocess.check_output(
            ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
        ).strip()
        state["describe"] = subprocess.check_output(
            ["git", "-C", str(tree), "log", "-1", "--format=%h %ad %s", "--date=short"],
            text=True,
        ).strip()
        state["dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", str(tree), "status", "--porcelain"], text=True
            ).strip()
        )
        state["remote"] = subprocess.check_output(
            ["git", "-C", str(tree), "remote", "get-url", "origin"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        pass
    return state


def _wait_healthy(base: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=3) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(1.0)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help=(
            "Unmeasured requests per case before the measured ones. The first "
            "request on a shape pays graph capture and weight-touch costs that "
            "a steady-state rate must not include."
        ),
    )
    parser.add_argument("--port", type=int, default=18211)
    parser.add_argument("--kv-dtype", default="bf16", choices=("bf16", "f16", "q8_0", "f32"))
    parser.add_argument("--context", type=int, default=4352)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--ubatch", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--source-tree", type=Path, default=None)
    parser.add_argument(
        "--server-arg",
        action="append",
        default=None,
        help=(
            "Extra argument passed to the server verbatim; repeatable. Used for "
            "delimitation, e.g. --server-arg --no-warmup so a profiled trace "
            "contains only the measured prefills and not the server's own "
            "warmup run."
        ),
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--marker-trace",
        action="store_true",
        help="Also pass --marker-trace to rocprofv3 (harmless when the engine emits no markers).",
    )
    parser.add_argument("--rocprof-bin", default="rocprofv3")
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.server.is_file():
        raise SystemExit(f"missing server binary: {args.server}")
    if not args.model.is_file():
        raise SystemExit(f"missing model: {args.model}")
    fixture = json.loads(args.fixture.read_text())
    cases = [
        case for case in fixture["cases"]
        if args.case_id is None or case["id"] in args.case_id
    ]
    if not cases:
        raise SystemExit("no cases selected")
    needed = max(int(case["prompt_tokens"]) for case in cases) + 1
    if args.context < needed:
        raise SystemExit(
            f"--context {args.context} is below the largest case ({needed})"
        )

    trace_root = args.trace_root.resolve()
    trace_root.mkdir(parents=True, exist_ok=True)
    server_args = [
        str(args.server), "-m", str(args.model),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--parallel", "1", "--no-webui", "-ngl", "999", "-fa", "on",
        "-ctk", args.kv_dtype, "-ctv", args.kv_dtype, "-c", str(args.context),
        "-b", str(args.batch), "-ub", str(args.ubatch), "-t", str(args.threads),
        *(args.server_arg or []),
    ]
    profiler_args = [args.rocprof_bin, "--kernel-trace"]
    if args.marker_trace:
        profiler_args.append("--marker-trace")
    profiler_args += ["--output-format", "csv",
         "-d", str(trace_root / "trace"), "--", *server_args]
    launched = profiler_args if args.profile else list(server_args)
    report: dict = {
        "schema": 1,
        "kind": "llamacpp_family_prefill_comparator",
        "label": args.label,
        "performance_claim": not args.profile,
        "measurement_class": (
            "diagnostic_profiled_not_performance" if args.profile
            else "comparator_unprofiled"
        ),
        "numerics_evaluated": False,
        "server": str(args.server),
        "server_sha256": _sha256_file(args.server),
        "server_args": server_args,
        "server_arg_extra": list(args.server_arg or []),
        "marker_trace": bool(args.marker_trace),
        "source": _source_state(args.source_tree),
        "model": str(args.model),
        "model_sha256_first_shard": _sha256_file(args.model),
        "kv_dtype": args.kv_dtype,
        "flash_attention": "on",
        "context": args.context,
        "batch": args.batch,
        "ubatch": args.ubatch,
        "threads": args.threads,
        "fixture": str(args.fixture),
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "cases": [],
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n")

    server_log = trace_root / "server.log"
    started = time.monotonic()
    with server_log.open("wb") as log_handle:
        process = subprocess.Popen(
            launched, stdout=log_handle, stderr=subprocess.STDOUT
        )
    try:
        base = f"http://127.0.0.1:{args.port}"
        if not _wait_healthy(base, args.startup_timeout):
            raise SystemExit(
                f"{args.label}: server did not become healthy; see {server_log}"
            )
        report["startup_seconds"] = time.monotonic() - started

        def _request(case, measured: bool, rep: int | None = None) -> dict:
            payload = json.dumps({
                "prompt": [int(t) for t in case["prompt_token_ids"]],
                "n_predict": 1, "temperature": 0.0, "top_k": 1,
                "top_p": 1.0, "min_p": 0.0, "seed": 12345,
                "ignore_eos": True, "cache_prompt": False,
                "stream": False, "return_tokens": True,
            }).encode()
            request = urllib.request.Request(
                base + "/completion", data=payload,
                headers={"Content-Type": "application/json"},
            )
            client_start = time.perf_counter()
            with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
                body = json.loads(response.read())
            timings = body.get("timings", {})
            row = {
                "prompt_ms": timings.get("prompt_ms"),
                "prompt_tokens": timings.get("prompt_n"),
                "prompt_tok_s": timings.get("prompt_per_second"),
                "client_wall_ms": (time.perf_counter() - client_start) * 1e3,
                "generated_token_id": (body.get("tokens") or [None])[0],
            }
            if measured:
                print(
                    f"{args.label} {case['id']} rep{rep}: "
                    f"prompt_ms={row['prompt_ms']} tok_s={row['prompt_tok_s']}",
                    flush=True,
                )
            return row

        for case in cases:
            for warmup in range(args.warmups):
                row = _request(case, measured=False, rep=warmup)
                print(
                    f"{args.label} {case['id']} warmup{warmup}: "
                    f"prompt_ms={row['prompt_ms']}",
                    flush=True,
                )
            rows = []
            for rep in range(args.repetitions):
                row = _request(case, measured=True, rep=rep)
                row["rep"] = rep
                rows.append(row)
            measured_tok_s = sorted(
                r["prompt_tok_s"] for r in rows if r.get("prompt_tok_s")
            )
            case_row = {
                "id": case["id"],
                "category": case["category"],
                "prompt_tokens": case["prompt_tokens"],
                "prompt_token_ids_sha256": case["prompt_token_ids_sha256"],
                "repetitions": rows,
            }
            if measured_tok_s:
                median = measured_tok_s[len(measured_tok_s) // 2]
                case_row["median_prompt_tok_s"] = median
                # A request that stalls (page-cache eviction, swap) is an
                # environment event, not an engine property. Flag it rather
                # than letting it sit silently in a mean.
                case_row["stalled_reps"] = [
                    r["rep"] for r in rows
                    if r.get("prompt_tok_s") and r["prompt_tok_s"] < 0.7 * median
                ]
            report["cases"].append(case_row)
            args.output.write_text(json.dumps(report, indent=1) + "\n")
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)

    measured = [
        row for case in report["cases"] for row in case["repetitions"]
        if row.get("prompt_tok_s")
    ]
    if measured:
        report["prompt_tok_s_mean"] = sum(r["prompt_tok_s"] for r in measured) / len(measured)
        report["prompt_tok_s_min"] = min(r["prompt_tok_s"] for r in measured)
        report["prompt_tok_s_max"] = max(r["prompt_tok_s"] for r in measured)
        report["measured_samples"] = len(measured)
        by_shape: dict[int, list[float]] = defaultdict(list)
        for case in report["cases"]:
            for row in case["repetitions"]:
                if row.get("prompt_tok_s"):
                    by_shape[int(case["prompt_tokens"])].append(row["prompt_tok_s"])
        report["median_prompt_tok_s_by_shape"] = {
            shape: sorted(values)[len(values) // 2]
            for shape, values in sorted(by_shape.items())
        }
        report["total_stalled_reps"] = sum(
            len(case.get("stalled_reps", [])) for case in report["cases"]
        )

    if args.profile:
        kernel_csvs = sorted((trace_root / "trace").rglob("*kernel_trace*.csv"))
        if not kernel_csvs:
            raise SystemExit("no kernel trace csv produced")
        roles: dict[str, float] = defaultdict(float)
        kernels: dict[str, float] = defaultdict(float)
        total_ns = 0
        with kernel_csvs[-1].open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = row.get("Name") or row.get("Kernel_Name") or ""
                duration = row.get("Duration") or row.get("Duration_ns")
                if duration is None and name and row.get("Start_Timestamp") and row.get("End_Timestamp"):
                    duration = int(float(row["End_Timestamp"])) - int(float(row["Start_Timestamp"]))
                if not name or not duration:
                    continue
                ns = int(float(duration))
                roles[_role_for(name)] += ns / 1e6
                kernels[name] += ns / 1e6
                total_ns += ns
        report["role_ms"] = {k: round(v, 3) for k, v in sorted(roles.items(), key=lambda kv: -kv[1])}
        report["device_total_ms"] = round(total_ns / 1e6, 3)
        report["top_kernels_ms"] = {
            k: round(v, 3)
            for k, v in sorted(kernels.items(), key=lambda kv: -kv[1])[:40]
        }

    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(
        f"{args.label}: {report.get('measured_samples', 0)} samples, "
        f"mean {report.get('prompt_tok_s_mean')} tok/s, "
        f"by-shape {report.get('median_prompt_tok_s_by_shape')}, "
        f"stalled {report.get('total_stalled_reps')} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
