"""Compare hipEngine's first-token choice against llama.cpp on one prompt.

llama.cpp's ``--jinja`` path renders the Gemma 4 chat template with
``enable_thinking=True``, which the template expands into a 23-token prompt
beginning with a ``<|turn>system\\n<|think|>`` turn. This script renders the
same prompt, runs hipEngine's streaming forward on it, and reports the top-k
next-token distribution so the two implementations can be compared on an
identical token sequence.
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gemma4_campaign_bench import resolve_artifact  # noqa: E402

ARTIFACT = resolve_artifact()
OUT = Path("/tmp/gemma4_llamacpp_compare.jsonl")
PROMPTS = [
    ("thinking-on", "Say hello in one short sentence."),
    ("thinking-on-factual", "What is the capital of France?"),
    ("thinking-on-math", "What is 2 + 2?"),
]


def render(template: str, content: str, *, enable_thinking: bool) -> str:
    environment = Environment(trim_blocks=True, lstrip_blocks=True, autoescape=False)
    environment.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(
        AssertionError(message)
    )
    return environment.from_string(template).render(
        messages=[{"role": "user", "content": content}],
        bos_token="<bos>",
        eos_token="<eos>",
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def main() -> int:
    reader = GGUFReader(ARTIFACT)
    streaming = Gemma4GGUFStreamingWeights(reader)
    tokenizer = Gemma4GGUFTokenizer.from_gguf_info(reader.info)
    embed = streaming.embed_tokens()
    norm = streaming.final_norm()

    for name, content in PROMPTS:
        prompt = render(tokenizer.chat_template, content, enable_thinking=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        started = time.time()
        logits = gemma4_streaming_forward(streaming, ids, embed_tokens=embed, final_norm=norm)
        elapsed = time.time() - started
        order = np.argsort(-logits[-1])[:12]
        record = {
            "probe": name,
            "prompt": prompt,
            "prompt_tokens": len(ids),
            "seconds": round(elapsed, 1),
            "top": [{"id": int(index), "text": tokenizer.decode([int(index)])} for index in order],
        }
        with OUT.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"[{name}] {len(ids)} tokens, {elapsed:.0f}s\n  prompt {prompt!r}\n  top: "
            + ", ".join(f"{item['id']}:{item['text']!r}" for item in record["top"]),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
