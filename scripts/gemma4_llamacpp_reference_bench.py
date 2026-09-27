"""G0 same-artifact llama.cpp HIP reference for the Gemma 4 campaign.

Runs the exact frozen campaign prompt ids (and the identical decoded text) on
llama.cpp's ``llama-server`` on the same GPU, context, KV dtype, greedy
sampling, and fixed-length contract as ``gemma4_campaign_bench.py``, then
reports the same phase accounting so the engine gap is measurable rather than
assumed. The server launch flags mirror ``scripts/llamacpp_raw_suite_bench.py``
(the repository's canonical external-llama.cpp timing path), and the request
body mirrors its greedy, uncached ``/completion`` calls.

Comparability, stated once:

- Prompt ids are byte-identical to the hipEngine harness (built with the same
  ``exact_prompt_ids``), and a text probe proves llama.cpp tokenizes the
  decoded text back to the same count before any timing row is accepted.
- llama.cpp ``predicted_ms`` spans prompt-end to last token, which includes the
  first greedy sample; the hipEngine ``decode_s`` starts after it. The
  difference is one host sample (~sub-millisecond) inside a second-plus phase
  and is recorded here rather than silently adjusted.
- llama.cpp runs at its shipped configuration for this artifact (flash
  attention on, graph-free decode); hipEngine runs its own shipped path. Each
  engine is measured at its product defaults.

Unit tests for the response accounting live in
``tests/test_unit_gemma4_llamacpp_reference.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.request import Request, urlopen

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gemma4_campaign_bench import (  # noqa: E402
    DEFAULT_ARTIFACT,
    exact_prompt_ids,
)

# The pinned llama.cpp HIP comparator. These were hard-coded paths under
# /mnt/nvme1 until that mount moved, which left the campaign's ground-truth
# build unreachable from the scripts that name it. ~/llama.cpp is a container
# directory rather than a checkout, so the comparator is the llama.cpp-hip tree
# inside it. Override with --server/--source when measuring another build.
DEFAULT_SERVER = Path("~/llama.cpp/llama.cpp-hip/build-hip/bin/llama-server").expanduser()
DEFAULT_SOURCE = Path("~/llama.cpp/llama.cpp-hip").expanduser()
DEFAULT_OUT = Path("/tmp/gemma4_llamacpp_reference.json")
DEFAULT_PORT = 18793


def reference_row(response: dict[str, Any], prompt_ids: Sequence[int], outputs: int) -> dict[str, Any]:
    """Validate one /completion response and convert it to campaign phases."""

    if response.get("truncated"):
        raise ValueError("response reports a truncated prompt")
    if response.get("tokens_evaluated") != len(prompt_ids):
        raise ValueError(
            f"tokens_evaluated={response.get('tokens_evaluated')} does not match "
            f"prompt of {len(prompt_ids)} ids"
        )
    tokens = response.get("tokens", [])
    if len(tokens) != outputs:
        raise ValueError(f"response carried {len(tokens)} output tokens, expected {outputs}")
    timings = response.get("timings") or {}
    if timings.get("predicted_n") != outputs:
        raise ValueError(
            f"timings.predicted_n={timings.get('predicted_n')} does not equal {outputs}"
        )
    prompt_ms = float(timings.get("prompt_ms") or 0.0)
    predicted_ms = float(timings.get("predicted_ms") or 0.0)
    if not prompt_ms > 0 or not predicted_ms > 0:
        raise ValueError(f"non-positive timing: prompt_ms={prompt_ms}, predicted_ms={predicted_ms}")
    decode_forwards = outputs - 1
    decode_s = predicted_ms / 1000.0
    prefill_s = prompt_ms / 1000.0
    return {
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": outputs,
        "decode_forwards": decode_forwards,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tps": decode_forwards / decode_s,
        "prefill_tps": len(prompt_ids) / prefill_s,
    }


def _post(base: str, body: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    request = Request(
        base + "/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _tokenize(base: str, text: str) -> list[int]:
    request = Request(
        base + "/tokenize",
        data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120.0) as response:
        return [int(token) for token in json.load(response)["tokens"]]


def tokenizer_probe(llama_tokens: Sequence[int], prompt_ids: Sequence[int]) -> dict[str, Any]:
    """Compare llama.cpp's text tokenization against the campaign ids.

    llama.cpp prepends the model's BOS for text prompts while hipEngine's raw
    ``encode`` does not, so both spellings are accepted and recorded; the
    timing rows themselves feed the exact id array and need no normalization.
    """

    tokens = list(llama_tokens)
    expected = list(prompt_ids)
    if tokens == expected:
        return {"match": True, "bos": None, "tokens": len(tokens)}
    if len(tokens) == len(expected) + 1 and tokens[1:] == expected:
        return {"match": True, "bos": tokens[0], "tokens": len(tokens)}
    divergence = next(
        (i for i, (a, b) in enumerate(zip(tokens, expected)) if a != b),
        min(len(tokens), len(expected)),
    )
    return {
        "match": False,
        "tokens": len(tokens),
        "expected": len(expected),
        "first_divergence": divergence,
    }


def _completion_body(prompt: Any, outputs: int) -> dict[str, Any]:
    return dict(
        prompt=prompt,
        n_predict=outputs,
        temperature=0,
        top_k=1,
        seed=12345,
        ignore_eos=True,
        cache_prompt=False,
        stream=False,
        return_tokens=True,
    )


def _wait_health(base: str, process: subprocess.Popen[bytes], log_path: Path, budget: float = 600.0) -> None:
    deadline = time.monotonic() + budget
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited {process.returncode}; see {log_path}")
        try:
            with urlopen(base + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(f"llama-server startup exceeded {budget}s; see {log_path}")
        time.sleep(0.5)


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prompt", type=int, default=1024)
    parser.add_argument("--output", type=int, default=128)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--kv", default="bf16")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--pci", default=None,
                        help="PCI id of the card whose idleness gates the run; by default the "
                             "only amdgpu card is used, and several cards fail loudly")
    parser.add_argument("--idle-limit-mib", type=int, default=128)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--label", default="g0-llamacpp-reference")
    args = parser.parse_args(argv)

    if args.prompt < 1 or args.output < 2:
        parser.error("--prompt >= 1 and --output >= 2 required")
    if args.prompt + args.output > args.context:
        parser.error("prompt + output must fit --context")

    from hipengine.loading.gguf import GGUFReader
    from hipengine.tokenization.gguf import Gemma4GGUFTokenizer
    from hipengine.util.amdgpu_vram import select_card

    tokenizer = Gemma4GGUFTokenizer.from_gguf_info(GGUFReader(args.artifact).info)
    prompt_ids = exact_prompt_ids(lambda text: tokenizer.encode(text), args.prompt)
    prompt_text = tokenizer.decode(prompt_ids)
    roundtrip = list(tokenizer.encode(prompt_text))
    if roundtrip != prompt_ids:
        print("ERROR: gemma tokenizer round-trip diverged; comparator not comparable", file=sys.stderr)
        return 2

    card = select_card(pci_id=args.pci)
    idle_used = int(card.vram_used_path.read_text())
    if idle_used > args.idle_limit_mib * 1024 * 1024:
        print(
            f"ERROR: GPU {card.pci_id} is not idle ({idle_used / 1024 / 1024:.0f} MiB used, "
            f"limit {args.idle_limit_mib} MiB); refusing to measure",
            file=sys.stderr,
        )
        return 2

    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            print(f"ERROR: port {args.port} already occupied", file=sys.stderr)
            return 2

    source_commit = ""
    try:
        source_commit = subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    server_command = [
        str(args.server), "-m", str(args.artifact), "-ngl", "99", "-fa", "on",
        "-ctk", args.kv, "-ctv", args.kv, "-c", str(args.context), "-np", "1",
        "-b", "4096", "-ub", "1024", "--host", "127.0.0.1",
        "--port", str(args.port), "--no-cache-prompt", "--fit", "off",
    ]

    print(f"[llamacpp_reference] launching: {shlex.join(server_command)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".log")
    base = f"http://127.0.0.1:{args.port}"
    started_at = datetime.now(timezone.utc).isoformat()
    samples: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    text_probe: dict[str, Any]
    process: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT)
        _wait_health(base, process, log_path)

        llama_tokens = _tokenize(base, prompt_text)
        text_probe = tokenizer_probe(llama_tokens, prompt_ids)
        text_probe["llama_tokens"] = len(llama_tokens)
        text_probe["expected"] = len(prompt_ids)
        if not text_probe["match"]:
            print(
                f"ERROR: llama.cpp tokenization diverges from campaign ids at "
                f"offset {text_probe.get('first_divergence')} "
                f"({text_probe['tokens']} vs {text_probe['expected']} tokens); "
                "comparator refused",
                file=sys.stderr,
            )
            return 2
        print(f"[llamacpp_reference] text tokenizer parity: {text_probe}", flush=True)

        for index in range(args.warmup):
            row = reference_row(_post(base, _completion_body(prompt_ids, args.output)), prompt_ids, args.output)
            row["index"] = index
            warmups.append(row)
            print(
                f"[llamacpp_reference] warmup {index}: prefill {row['prefill_s']:.2f}s, "
                f"decode {row['decode_s']:.2f}s ({row['decode_tps']:.2f} tok/s)",
                flush=True,
            )
        for index in range(args.samples):
            row = reference_row(_post(base, _completion_body(prompt_ids, args.output)), prompt_ids, args.output)
            row["index"] = index
            samples.append(row)
            print(
                f"[llamacpp_reference] sample {index}: prefill {row['prefill_s']:.2f}s "
                f"({row['prefill_tps']:.0f} tok/s), decode {row['decode_s']:.2f}s "
                f"({row['decode_tps']:.2f} tok/s)",
                flush=True,
            )
    finally:
        if process is not None:
            _stop(process)

    stats = {
        "samples": len(samples),
        "decode_tps": _median([row["decode_tps"] for row in samples]),
        "decode_tps_min": min(row["decode_tps"] for row in samples),
        "decode_tps_max": max(row["decode_tps"] for row in samples),
        "prefill_tps": _median([row["prefill_tps"] for row in samples]),
        "prefill_s": _median([row["prefill_s"] for row in samples]),
        "decode_s": _median([row["decode_s"] for row in samples]),
        "prompt_tokens": args.prompt,
        "generated_tokens": args.output,
        "decode_forwards_per_sample": args.output - 1,
    }
    artifact = {
        "schema": 1,
        "label": args.label,
        "status": "ok",
        "performance_claim": False,
        "created_at": started_at,
        "command": shlex.join([sys.executable, *sys.argv]),
        "server_command": server_command,
        "llamacpp_source": str(args.source),
        "llamacpp_commit": source_commit,
        "artifact": str(args.artifact),
        "artifact_bytes": args.artifact.stat().st_size,
        "prompt_ids_sha256": hashlib.sha256(
            b"".join(int(t).to_bytes(4, "little") for t in prompt_ids)
        ).hexdigest(),
        "env": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("HIP", "ROCR", "HSA"))},
        "kv": args.kv,
        "context": args.context,
        "text_probe": text_probe,
        "text_probe_method": "POST /tokenize on the decoded campaign text; exact id-list comparison, BOS-prefix spelling accepted and recorded",
        "conventions": {
            "decode_tps": "(outputs - 1) / predicted_ms; predicted_ms also spans the first greedy sample",
            "prefill_tps": "prompt_tokens / prompt_ms",
            "greedy": "temperature=0, top_k=1, seed=12345, ignore_eos=true, cache_prompt=false",
        },
        "warmups": warmups,
        "samples": samples,
        "stats": stats,
    }
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"llamacpp_decode_tps={stats['decode_tps']:.4f}")
    print(f"llamacpp_prefill_tps={stats['prefill_tps']:.2f}")
    print(f"artifact={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())