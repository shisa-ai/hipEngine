#!/usr/bin/env python3
"""Serve the Inkling 2 GGUF and time one and several concurrent chat completions.

Starts ``python -m hipengine.server`` from this repository in a child process, waits for
``/ready`` and records the profile the server resolved. After one warm request it posts one
request ``--reps`` times, then ``--streams`` identical requests together. It reports wall seconds
and output tokens per request, and the output tokens per second of the concurrent round.

The server needs ``--kv-storage auto`` for this model, so the check passes it. Greedy, prefix cache
and drafting off. Wall time is measured at the client from request start to the complete response,
so it holds prefill, decode and the HTTP round trip. This is the product-surface control for the
profile default and for serving changes; ``scripts/inkling2_decode_bench.py`` times the engine.

    python scripts/inkling2_serve_check.py MODEL --backend hip_gfx1100 --output out.json
    python scripts/inkling2_serve_check.py MODEL --backend hip_gfx1100 --execution-profile strict
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KIND = "hipengine_inkling2_serve_check"
PROTOCOL = "wall_clock_http_nonstream_v1"
PROMPT = "Explain in three short sentences why the sky looks blue."


def environment() -> dict[str, object]:
    return {name: value for name, value in sorted(os.environ.items()) if name.startswith("HIPENGINE_")}


def provenance(model: Path, args) -> dict:
    from hipengine.benchmark.provenance import collect_artifact_provenance

    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    hipcc_version = None
    if version_file and Path(version_file).is_file():
        hipcc_version = Path(version_file).read_text(encoding="utf-8")
    return collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=args.backend,
        model_path=model,
        quant=None,
        kv_dtype="fp32",
        command=[sys.argv[0], *sys.argv[1:]],
        environment=environment(),
        timing_protocol=PROTOCOL,
        warmups=1,
        repetitions=args.reps,
        hipcc_version=hipcc_version,
    )


def _request(base: str, method: str, path: str, body=None, timeout: float = 900.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def completion(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": False}
    started = time.perf_counter()
    reply = _request(base, "POST", "/v1/chat/completions", body)
    seconds = time.perf_counter() - started
    usage = reply.get("usage", {})
    return {"seconds": seconds, "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"), "text": reply["choices"][0]["message"]["content"]}


def wait_until_ready(base: str, log_path: Path, seconds: float = 600.0) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        try:
            readiness = _request(base, "GET", "/ready", timeout=30)
            if readiness.get("ready") is True or readiness.get("status") in ("ok", "ready"):
                return
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"the server did not become ready in {seconds:.0f} s; its log is {log_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--execution-profile", default=None,
                        help="profile the server is started with; omitted takes the shipped default")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--streams", type=int, default=4, help="requests posted together in the last round")
    parser.add_argument("--reps", type=int, default=3, help="single requests after the warm one")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--output", type=Path, default=None, help="write the summary JSON here")
    args = parser.parse_args(argv)
    model = args.model.expanduser()
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    base = f"http://127.0.0.1:{args.port}"
    log_path = (args.output.with_suffix(".server.log") if args.output is not None else Path("inkling2-serve.server.log"))
    command = [sys.executable, "-B", "-m", "hipengine.server", "--model", str(model), "--backend", args.backend,
               "--port", str(args.port), "--host", "127.0.0.1", "--log-level", "warning", "--served-model-name", "inkling",
               "--kv-storage", "auto", "--max-context-tokens", "4096", "--max-active-requests", str(args.streams),
               "--prefix-cache", "off", "--speculative-mtp-serving", "off"]
    if args.execution_profile:
        command += ["--execution-profile", args.execution_profile]
    child_env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1")
    with log_path.open("w") as log:
        server = subprocess.Popen(command, cwd=REPO_ROOT, env=child_env, stdout=log, stderr=log)
        try:
            wait_until_ready(base, log_path)
            served = _request(base, "GET", "/v1/models")["data"][0]
            completion(base, served["id"], args.prompt, 8)
            single = [completion(base, served["id"], args.prompt, args.max_tokens) for _ in range(args.reps)]
            results = [None] * args.streams

            def one(index: int) -> None:
                results[index] = completion(base, served["id"], args.prompt, args.max_tokens)

            threads = [threading.Thread(target=one, args=(index,)) for index in range(args.streams)]
            started = time.perf_counter()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            wall = time.perf_counter() - started
        finally:
            server.terminate()
            try:
                server.wait(60)
            except subprocess.TimeoutExpired:
                server.kill()
    tokens = sum(row["completion_tokens"] for row in results)
    result = {
        "kind": KIND,
        "tool": "scripts/inkling2_serve_check.py",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "model": {"path": str(model), "bytes": model.stat().st_size},
        "backend": args.backend,
        "requested_profile": args.execution_profile,
        "execution_profile": served["hipengine"]["execution_profile"],
        "context": served["hipengine"].get("context"),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "reps": args.reps,
        "streams": args.streams,
        "timing_protocol": PROTOCOL,
        "single": single,
        "concurrent": {"wall_seconds": wall, "requests": results, "output_tokens": tokens,
                       "aggregate_tok_s": tokens / wall if wall else None},
        "environment": environment(),
        "provenance": provenance(model, args),
    }
    text = json.dumps(result, indent=1, ensure_ascii=False)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
