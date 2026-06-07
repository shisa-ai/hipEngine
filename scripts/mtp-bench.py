#!/usr/bin/env python3
"""llama.cpp-compatible MTP prompt-suite benchmark.

This is a protocol-compatible hipEngine copy of the ad-hoc ``mtp-bench.py``
script used in llama.cpp MTP PR discussions, including PR #23287.  It keeps the
same default prompt suite and request shape so we can run the same benchmark
against a llama.cpp server and a hipEngine OpenAI-compatible server.  It can
also wrap hipEngine's existing prompt-suite verifier-economics harness via
``--mode hipengine-current`` so old/current diagnostic artifacts and new
llama.cpp-compatible server numbers share one entry point.

Defaults intentionally mirror the upstream gist as of raw revision
``0bee1e2b88904e62670d0df1cf0991883b0815d7``:

* POST ``/v1/chat/completions``
* ``model="llama"``
* one user message per prompt
* ``max_tokens=192``
* ``seed=42``
* no explicit temperature/top_p/cache_prompt unless requested by CLI flags

Source reference:
https://gist.github.com/am17an/228edfb84ed082aa88e3865d6fa27090
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib import error, request

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks" / "fixtures" / "llamacpp_mtp_bench_prompts.json"
DEFAULT_ENDPOINT = "/v1/chat/completions"
DEFAULT_ENGINE_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")
DEFAULT_HIPENGINE_RAW_ROOT = Path("/tmp/hipengine-mtp-llamacpp-prompt-suite-economics")
SOURCE_GIST = "https://gist.github.com/am17an/228edfb84ed082aa88e3865d6fa27090"
SOURCE_RAW = (
    "https://gist.githubusercontent.com/am17an/228edfb84ed082aa88e3865d6fa27090/raw/"
    "0bee1e2b88904e62670d0df1cf0991883b0815d7/mtp-bench.py"
)


class BenchError(RuntimeError):
    """Raised for benchmark setup or response-shape errors."""


def load_prompt_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    prompts = suite.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise BenchError(f"{path} does not contain a non-empty 'prompts' list")

    seen: set[str] = set()
    for item in prompts:
        if not isinstance(item, dict):
            raise BenchError(f"invalid prompt entry in {path}: {item!r}")
        name = str(item.get("name") or "")
        prompt = str(item.get("prompt") or "")
        if not name or not prompt:
            raise BenchError(f"prompt entries require non-empty name and prompt: {item!r}")
        if name in seen:
            raise BenchError(f"duplicate prompt name in {path}: {name}")
        seen.add(name)
    return suite


def split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def select_prompts(
    suite: dict[str, Any],
    *,
    names_csv: str | None = None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    prompts = [{"name": str(p["name"]), "prompt": str(p["prompt"])} for p in suite["prompts"]]
    names = split_csv(names_csv)
    if names:
        by_name = {p["name"]: p for p in prompts}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise BenchError(f"unknown prompt name(s): {', '.join(missing)}")
        prompts = [by_name[name] for name in names]
    if limit is not None:
        prompts = prompts[: max(0, limit)]
    if not prompts:
        raise BenchError("prompt selection is empty")
    return prompts


def make_payload(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.cache_prompt is not None:
        payload["cache_prompt"] = args.cache_prompt
    if args.ignore_eos:
        payload["ignore_eos"] = True
    if args.extra_payload:
        try:
            extra = json.loads(args.extra_payload)
        except json.JSONDecodeError as exc:
            raise BenchError(f"--extra-payload must be a JSON object: {exc}") from exc
        if not isinstance(extra, dict):
            raise BenchError("--extra-payload must decode to a JSON object")
        payload.update(extra)
    return payload


def post_json(url: str, payload: dict[str, Any], *, timeout: float, api_key: str | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            data = response.read()
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise BenchError(f"HTTP {exc.code} from {url}: {body}") from exc
    except error.URLError as exc:
        raise BenchError(f"failed to POST {url}: {exc}") from exc
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise BenchError(f"expected JSON object response from {url}, got {type(parsed).__name__}")
    return parsed


def first_number(*values: Any) -> int | float | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def record_from_response(name: str, response: dict[str, Any], wall_s: float) -> dict[str, Any]:
    usage = response.get("usage") or {}
    timings = response.get("timings") or {}
    if not isinstance(usage, dict):
        usage = {}
    if not isinstance(timings, dict):
        timings = {}

    predicted_n = first_number(usage.get("completion_tokens"), timings.get("predicted_n"))
    predicted_per_second = first_number(timings.get("predicted_per_second"))
    if predicted_per_second is None and predicted_n is not None and wall_s > 0:
        predicted_per_second = float(predicted_n) / wall_s
    if predicted_per_second is None:
        predicted_per_second = 0.0

    draft_n = first_number(timings.get("draft_n"), 0) or 0
    draft_n_accepted = first_number(timings.get("draft_n_accepted"), 0) or 0

    record = {
        "name": name,
        "wall_s": round(wall_s, 3),
        "predicted_n": int(predicted_n) if predicted_n is not None else 0,
        "predicted_per_second": round(float(predicted_per_second), 2),
        "draft_n": int(draft_n),
        "draft_n_accepted": int(draft_n_accepted),
    }
    record["accept_rate"] = (
        round(record["draft_n_accepted"] / record["draft_n"], 4) if record["draft_n"] else None
    )
    return record


def format_result_line(record: dict[str, Any]) -> str:
    accept_rate = f"{record['accept_rate']:.3f}" if record["accept_rate"] is not None else "n/a"
    return (
        f"  {record['name']:<18} pred={record['predicted_n']:>4} "
        f"draft={record['draft_n']:>4} acc={record['draft_n_accepted']:>4} "
        f"rate={accept_rate} tok/s={record['predicted_per_second']:.1f}"
    )


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_draft = sum(int(x.get("draft_n") or 0) for x in results)
    total_accepted = sum(int(x.get("draft_n_accepted") or 0) for x in results)
    total_predicted = sum(int(x.get("predicted_n") or 0) for x in results)
    total_wall = sum(float(x.get("wall_s") or 0.0) for x in results)
    return {
        "n_requests": len(results),
        "total_predicted": total_predicted,
        "total_draft": total_draft,
        "total_draft_accepted": total_accepted,
        "aggregate_accept_rate": round(total_accepted / total_draft, 4) if total_draft else None,
        "wall_s_total": round(total_wall, 2),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    suite = load_prompt_suite(args.prompts_file)
    prompts = select_prompts(suite, names_csv=args.prompt_names, limit=args.limit)
    base_url = args.url.rstrip("/")
    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    url = f"{base_url}{endpoint}"

    out: dict[str, Any] = {"results": []}
    for prompt in prompts:
        payload = make_payload(prompt["prompt"], args)
        if args.print_payload:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            continue
        start = time.perf_counter()
        response = post_json(url, payload, timeout=args.timeout, api_key=args.api_key)
        wall_s = time.perf_counter() - start
        record = record_from_response(prompt["name"], response, wall_s)
        out["results"].append(record)
        print(format_result_line(record))

    if args.print_payload:
        return out

    out["aggregate"] = aggregate(out["results"])
    print("\nAggregate:", json.dumps(out["aggregate"], indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print("Wrote", args.out)
    return out


def default_hipengine_out_path() -> Path:
    return (
        REPO_ROOT
        / "benchmarks"
        / "results"
        / f"{date.today().isoformat()}-hipengine-mtp-llamacpp-prompt-suite-economics.json"
    )


def hipengine_output_path(args: argparse.Namespace) -> Path:
    return args.out if args.out is not None else default_hipengine_out_path()


def build_hipengine_current_command(args: argparse.Namespace) -> list[str]:
    """Build the existing hipEngine prompt-suite economics command.

    This deliberately shells out to ``scripts/mtp_prompt_suite_economics.py`` so
    the JSON remains compatible with the artifacts we already use for MTP
    verifier economics, while this tool owns the shared llama.cpp prompt-suite
    entry point.
    """

    cmd = [
        sys.executable,
        "scripts/mtp_prompt_suite_economics.py",
        "--model",
        str(args.engine_model),
        "--prompts-file",
        str(args.prompts_file),
        "--prompt-render",
        str(args.prompt_render),
        "--decode-tokens",
        str(args.max_tokens),
        "--candidate-budgets",
        str(args.candidate_budgets),
        "--runs",
        str(args.runs),
        "--proposal-impl",
        str(args.proposal_impl),
        "--backend",
        str(args.backend),
        "--hip-arch",
        str(args.hip_arch),
        "--chain-attn-mode",
        str(args.chain_attn_mode),
        "--graph-mode",
        str(args.graph_mode),
        "--raw-root",
        str(args.raw_root),
        "--out",
        str(hipengine_output_path(args)),
    ]
    if args.prompt_names:
        cmd += ["--prompt-names", str(args.prompt_names)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.small_batch_decode_threshold is not None:
        cmd += ["--small-batch-decode-threshold", str(args.small_batch_decode_threshold)]
    if args.verify_gpu_accept is not None:
        cmd += ["--verify-gpu-accept", str(args.verify_gpu_accept)]
    if args.llama_target_cycle_cost is not None:
        cmd += ["--llama-target-cycle-cost", str(args.llama_target_cycle_cost)]
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def quote_command(cmd: list[str]) -> str:
    return " ".join(_shell_quote(part) for part in cmd)


def _shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "@%_+=:,./-" for ch in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def run_hipengine_current(args: argparse.Namespace) -> None:
    cmd = build_hipengine_current_command(args)
    if args.print_command:
        print(quote_command(cmd))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    print("[hipengine-current] running existing prompt-suite economics:")
    print("  " + quote_command(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True)
    if completed.returncode != 0:
        raise BenchError(f"hipEngine current economics exited with status {completed.returncode}")

    out_path = hipengine_output_path(args)
    if not args.dry_run and out_path.exists():
        print_hipengine_summary(out_path)


def print_hipengine_summary(path: Path) -> None:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    aggregate_by_budget = artifact.get("aggregate_by_budget") or {}
    if not aggregate_by_budget:
        return
    print("\nHipEngine current economics summary:")
    for budget, row in sorted(aggregate_by_budget.items(), key=lambda item: int(item[0])):
        prompts = row.get("prompts")
        exact = row.get("all_exact_ar_match")
        cycle = row.get("cycle_cost_ar_tokens_mean_across_prompts_mean")
        visible = row.get("avg_visible_tokens_per_cycle_mean_across_prompts_mean")
        observed = row.get("observed_cycle_speedup_vs_ar_mean_across_prompts_mean")
        actual = row.get("actual_decode_speedup_vs_ar_mean_across_prompts_mean")
        accept = row.get("acceptance_rate_mean_across_prompts_mean")
        print(
            f"  B={budget:<3} prompts={prompts} exact={exact} "
            f"cycle_cost={format_optional(cycle)} AR-tok "
            f"visible/cycle={format_optional(visible)} "
            f"cycle_speedup={format_optional(observed)}x "
            f"actual_mtp/ar={format_optional(actual)}x "
            f"accept={format_optional(accept)}"
        )


def format_optional(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def diff(path_a: Path, path_b: Path) -> None:
    data_a = json.loads(path_a.read_text(encoding="utf-8"))
    data_b = json.loads(path_b.read_text(encoding="utf-8"))
    print(f"{'metric':<24} {'A':>14} {'B':>14} {'delta':>10}")
    for key in (
        "aggregate_accept_rate",
        "total_predicted",
        "total_draft",
        "total_draft_accepted",
        "wall_s_total",
    ):
        val_a = data_a["aggregate"].get(key)
        val_b = data_b["aggregate"].get(key)
        if val_a is None or val_b is None:
            print(f"{key:<24} {str(val_a):>14} {str(val_b):>14}")
            continue
        delta = val_b - val_a
        delta_str = f"{delta:>+10.4f}" if isinstance(delta, float) else f"{delta:>+10}"
        print(f"{key:<24} {val_a:>14} {val_b:>14} {delta_str}")

    by_a = {row["name"]: row for row in data_a.get("results", [])}
    print("\n{:<20} {:>8} {:>8} {:>8}".format("prompt", "A", "B", "delta"))
    for row_b in data_b.get("results", []):
        row_a = by_a.get(row_b["name"]) or {}
        accept_a = row_a.get("accept_rate") or 0
        accept_b = row_b.get("accept_rate") or 0
        print(f"{row_b['name']:<20} {accept_a:>8.3f} {accept_b:>8.3f} {accept_b - accept_a:>+8.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("server", "hipengine-current"),
        default="server",
        help="server = llama.cpp/OpenAI-compatible requests; hipengine-current = existing verifier economics wrapper",
    )

    # Shared prompt/output controls.
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-names", help="comma-separated prompt names to run")
    parser.add_argument("--limit", type=int, help="run only the first N selected prompts")
    parser.add_argument("--list-prompts", action="store_true", help="list selected prompt names and exit")
    parser.add_argument("--max-tokens", type=int, default=192, help="server max_tokens / hipEngine decode_tokens")
    parser.add_argument("--out", type=Path, help="write JSON results")
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("A", "B"), help="diff two server-mode JSON outputs and exit")

    # llama.cpp / OpenAI-compatible server mode.
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="OpenAI-compatible server base URL")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="completion endpoint path")
    parser.add_argument("--model", default="llama", help="model field sent in each server request")
    parser.add_argument("--seed", type=int, default=42, help="seed field sent in each server request")
    parser.add_argument("--temperature", type=float, default=None, help="optional temperature field; omitted by default for gist parity")
    parser.add_argument("--top-p", type=float, default=None, help="optional top_p field; omitted by default for gist parity")
    parser.add_argument("--ignore-eos", action="store_true", help="send ignore_eos=true")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--cache-prompt", dest="cache_prompt", action="store_true", default=None, help="send cache_prompt=true")
    cache_group.add_argument("--no-cache-prompt", dest="cache_prompt", action="store_false", help="send cache_prompt=false")
    parser.add_argument("--extra-payload", help="JSON object merged into each server request payload")
    parser.add_argument("--api-key", help="Bearer token for servers requiring OpenAI-style auth")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request timeout in seconds")
    parser.add_argument("--print-payload", action="store_true", help="print server request payloads instead of posting")

    # hipEngine current verifier-economics mode.  Defaults are the W7900/gfx1100
    # path used by current local M12 artifacts.
    parser.add_argument("--engine-model", type=Path, default=DEFAULT_ENGINE_MODEL, help="hipEngine model path for hipengine-current mode")
    parser.add_argument("--candidate-budgets", default="3", help="comma-separated candidate budgets for hipengine-current mode")
    parser.add_argument("--runs", type=int, default=1, help="runs per prompt/budget for hipengine-current mode")
    parser.add_argument(
        "--prompt-render",
        choices=("raw", "qwen_chat_thinking_off", "qwen_chat_thinking_on"),
        default="raw",
        help="hipengine-current prompt rendering before tokenization",
    )
    parser.add_argument(
        "--proposal-impl",
        choices=("persistent_device", "persistent_device_b1", "reload_d2h"),
        default="persistent_device",
    )
    parser.add_argument("--backend", default="hip_gfx1100", help="hipEngine backend for hipengine-current mode")
    parser.add_argument("--hip-arch", default="gfx1100", help="HIP arch for hipengine-current mode")
    parser.add_argument("--chain-attn-mode", choices=("c1_loop", "batched", "decode_batched"), default="batched")
    parser.add_argument("--graph-mode", choices=("off", "auto", "validate"), default="off")
    parser.add_argument("--small-batch-decode-threshold", type=int, default=7)
    parser.add_argument("--verify-gpu-accept", default=None)
    parser.add_argument("--llama-target-cycle-cost", type=float, default=2.0)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_HIPENGINE_RAW_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="pass --dry-run to hipengine-current mode")
    parser.add_argument("--print-command", action="store_true", help="print hipengine-current command and exit")

    parser.epilog = (
        "Examples:\n"
        "  python3 scripts/mtp-bench.py --url http://127.0.0.1:8080 --out llama-mtp.json\n"
        "  python3 scripts/mtp-bench.py --diff llama-master.json llama-pr23287.json\n"
        "  python3 scripts/mtp-bench.py --temperature 0 --no-cache-prompt\n"
        "  python3 scripts/mtp-bench.py --mode hipengine-current --candidate-budgets 3 --runs 3 --out hipengine-current.json\n\n"
        f"Source gist: {SOURCE_GIST}\nRaw revision: {SOURCE_RAW}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.diff:
            diff(*args.diff)
            return 0
        suite = load_prompt_suite(args.prompts_file)
        prompts = select_prompts(suite, names_csv=args.prompt_names, limit=args.limit)
        if args.list_prompts:
            for prompt in prompts:
                print(f"{prompt['name']}\t{len(prompt['prompt'])} chars")
            return 0
        if args.mode == "hipengine-current":
            run_hipengine_current(args)
        else:
            run(args)
        return 0
    except BenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
