#!/usr/bin/env python3
"""External llama-server timing with exact hipEngine category-suite token IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.loading import load_gguf_index  # noqa: E402
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer  # noqa: E402
from hipengine.util.amdgpu_vram import VramSampler, select_card  # noqa: E402
from scripts.gguf_mtp_bench import build_chat_prompt  # noqa: E402
from scripts.gguf_mtp_category_bench import load_prompt_rows  # noqa: E402
from scripts.llamacpp_mtp_bench import _terminate  # noqa: E402


def require_idle_memory(used_bytes: int, limit_mib: int) -> None:
    if used_bytes > limit_mib * (1 << 20):
        raise RuntimeError(f"GPU not idle: {used_bytes / (1 << 20):.1f} MiB exceeds {limit_mib} MiB")


def completion_row(response: dict, prompt: list[int], outputs: int) -> dict:
    ids = response.get("tokens", [])
    timings = response["timings"]
    if len(ids) != outputs or timings["predicted_n"] != outputs:
        raise ValueError("incomplete output token accounting")
    if response.get("truncated") or response["tokens_evaluated"] != len(prompt):
        raise ValueError("prompt truncated or not fully evaluated")
    if timings["predicted_ms"] <= 0 or timings["prompt_ms"] <= 0:
        raise ValueError("non-positive timing")
    return {
        "prompt_tokens": len(prompt),
        "prompt_ids": prompt,
        "output_ids": ids,
        "timings": timings,
        "decode_transitions": outputs - 1,
        "decode_tok_s": (outputs - 1) * 1000 / timings["predicted_ms"],
        "prefill_tok_s": len(prompt) * 1000 / timings["prompt_ms"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl")
    parser.add_argument("--mode", choices=("ar", "mtp", "adaptive"), default="ar")
    parser.add_argument("--outputs", type=int, default=25)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--repeat-lengths", default="")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--kv", default="bf16")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--ubatch", type=int, default=1024)
    parser.add_argument("--port", type=int, default=18791)
    parser.add_argument("--pci", default="0000:10:00.0")
    parser.add_argument("--idle-limit-mib", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.outputs < 2 or args.repetitions < 1:
        parser.error("outputs >= 2 and repetitions >= 1 required")
    lengths = [int(x) for x in args.repeat_lengths.split(",") if x]
    if lengths:
        prompts = [{"id": f"repeat-{n}", "category": "synthetic", "tokens": [9707] * n} for n in lengths]
    else:
        tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(args.model))
        prompts = [
            dict(id=p["id"], category=p["category"],
                 tokens=list(build_chat_prompt(tokenizer, p["prompt"], reasoning="off")))
            for p in load_prompt_rows(args.prompts)
        ]
    if any(len(p["tokens"]) + args.outputs > args.context for p in prompts):
        parser.error("context must contain each prompt plus outputs")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            parser.error("port already occupied")
    command = [
        str(args.server), "-m", str(args.model), "-ngl", "99", "-fa", "on",
        "-ctk", args.kv, "-ctv", args.kv, "-c", str(args.context), "-np", "1",
        "-b", str(args.batch), "-ub", str(args.ubatch), "--host", "127.0.0.1",
        "--port", str(args.port), "--no-cache-prompt", "--fit", "off",
    ]
    if args.mode != "ar":
        command += ["--spec-type", "draft-mtp-adaptive" if args.mode == "adaptive" else "draft-mtp",
                    "--spec-draft-n-max", "3"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")
    payload = {
        "schema": 1, "status": "running", "performance_claim": False,
        "host": socket.gethostname(), "command": sys.argv, "server_command": command,
        "source_commit": subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
        "environment": {k: v for k, v in os.environ.items() if k.startswith(("HIP", "ROCR", "GGML", "GPU_MAX"))},
        "model": str(args.model), "model_bytes": args.model.stat().st_size,
        "prompt_file_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "mode": args.mode, "kv": args.kv, "rows": [],
        "correctness_scope": "complete prompt/output accounting; AR/MTP ID comparison performed separately",
    }
    card = select_card(pci_id=args.pci)
    require_idle_memory(int(card.vram_used_path.read_text()), args.idle_limit_mib)
    sampler = VramSampler(card=card, interval_ms=20)
    process = None
    base = f"http://127.0.0.1:{args.port}"
    sampler.start()
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"server exited {process.returncode}; see {log_path}")
            try:
                with urlopen(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("server startup")
            time.sleep(0.5)

        def request(tokens: list[int], outputs: int) -> dict:
            body = dict(prompt=tokens, n_predict=outputs, temperature=0, top_k=1,
                        seed=12345, ignore_eos=True, cache_prompt=False, stream=False,
                        return_tokens=True)
            req = Request(base + "/completion", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=3600) as response:
                return completion_row(json.load(response), tokens, outputs)

        request(prompts[0]["tokens"][:32], 8)
        for repetition in range(args.repetitions):
            for prompt in prompts:
                row = request(prompt["tokens"], args.outputs)
                row.update(id=prompt["id"], category=prompt["category"], repetition=repetition)
                if lengths:
                    row.pop("prompt_ids")
                    row["prompt_token_id"] = 9707
                payload["rows"].append(row)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
                print(f'{row["id"]} pp={row["prefill_tok_s"]:.3f} tg={row["decode_tok_s"]:.3f}', flush=True)
        payload["status"] = "complete"
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = repr(exc)
        raise
    finally:
        if process is not None:
            _terminate(process)
        sampler.stop()
        payload["memory"] = sampler.result().to_dict()
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
