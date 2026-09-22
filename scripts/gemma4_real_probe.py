"""Run hipEngine's streaming Gemma 4 forward on the real artifact, chat-templated.

Writes results incrementally so a long run can be inspected while it proceeds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from jinja2 import Environment

from hipengine.kernels.cpu_reference.gemma4_streaming import (
    Gemma4GGUFStreamingWeights,
    gemma4_streaming_forward,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.tokenization.gguf import Gemma4GGUFTokenizer

ARTIFACT = Path("/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf")
OUT = Path("/tmp/gemma4_real_run.jsonl")


def render(template: str, messages: list[dict[str, str]]) -> str:
    def raise_helper(message: str):
        raise AssertionError(message)

    environment = Environment(trim_blocks=True, lstrip_blocks=True, autoescape=False)
    environment.globals["raise_exception"] = raise_helper
    return environment.from_string(template).render(
        messages=messages,
        bos_token="<bos>",
        eos_token="<eos>",
        add_generation_prompt=True,
    )


def main() -> int:
    reader = GGUFReader(ARTIFACT)
    streaming = Gemma4GGUFStreamingWeights(reader)
    tokenizer = Gemma4GGUFTokenizer.from_gguf_info(reader.info)
    started = time.time()
    embed = streaming.embed_tokens()
    norm = streaming.final_norm()
    print(f"loaded in {time.time() - started:.0f}s", flush=True)

    probes = [
        ("chat-factual", [{"role": "user", "content": "What is the capital of France?"}]),
        ("chat-math", [{"role": "user", "content": "What is 2 + 2?"}]),
        ("chat-open", [{"role": "user", "content": "Say hello in one short sentence."}]),
    ]
    results = []
    for name, messages in probes:
        prompt = render(tokenizer.chat_template, messages)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        started = time.time()
        logits = gemma4_streaming_forward(streaming, ids, embed_tokens=embed, final_norm=norm)
        elapsed = time.time() - started
        top = np.argsort(-logits[-1])[:12]
        record = {
            "probe": name,
            "prompt": prompt,
            "prompt_tokens": len(ids),
            "seconds": round(elapsed, 1),
            "top": [{"id": int(index), "text": tokenizer.decode([int(index)])} for index in top],
        }
        results.append(record)
        with OUT.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"[{name}] {len(ids)} tokens, {elapsed:.0f}s, top: "
            + ", ".join(repr(item["text"]) for item in record["top"]),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
