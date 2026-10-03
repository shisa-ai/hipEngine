"""Run the real UD-Q4_K_XL artifact through the public ``LLM`` surface on GPU.

This is the end-to-end check that the serving path works on the artifact a user
actually has, not on a synthetic fixture: ``hipengine.LLM(model=<gguf>)``
resolves the Gemma 4 generator, loads the quantized blocks to the device, and
decodes greedily.

The prompt is the one llama.cpp's ``--jinja`` path renders for
``enable_thinking=True``, so the two implementations run on an identical
23-token sequence and their first sampled token is directly comparable.
llama.cpp's greedy output for that sequence is ``<|channel>thought``; the
tokenizer has no single token for that string, so the first token it emits is
``<|channel>`` (id 100), followed by ``thought``.

The context limit defaults to the generator's own. The runner sizes its
per-layer prefill scratch from the widest block it forwards rather than from the
context, so a prompt wider than that block is forwarded as consecutive blocks and
the context no longer decides whether the artifact fits on the device.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import hipengine
from hipengine.llm import SamplingParams

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gemma4_campaign_bench import resolve_artifact  # noqa: E402

ARTIFACT = resolve_artifact()
OUT = Path("/tmp/gemma4_real_generate.jsonl")

# llama.cpp printed this exact sequence for the prompt below.
LLAMA_CPP_PROMPT_IDS = (
    2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 37889, 29104, 528,
    886, 2822, 13315, 236761, 106, 107, 105, 4368, 107,
)
LLAMA_CPP_FIRST_TOKENS = (100, 45518)  # '<|channel>', 'thought'
PROMPTS = [
    ("thinking-on", "Say hello in one short sentence."),
    ("thinking-on-factual", "What is the capital of France?"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--long-probe-words",
        type=int,
        default=0,
        help="add a probe whose user message is this many words, to force "
        "prefill across more than one block (0 disables)",
    )
    arguments = parser.parse_args()

    probes = list(PROMPTS)
    if arguments.long_probe_words > 0:
        # Deliberately bland and repetitive: the point is the block count, not
        # the answer.
        sentence = "The quick brown fox jumps over the lazy dog. "
        words = (sentence * (arguments.long_probe_words // len(sentence.split()) + 1)).split()
        probes.append(("thinking-on-long", " ".join(words[: arguments.long_probe_words])))

    llm = hipengine.LLM(
        model=str(arguments.artifact),
        backend="hip_gfx1100",
        max_sequence_length=arguments.context,
    )
    try:
        generator = llm._get_text_generator()
        inner = getattr(generator, "_inner", generator)
        runner = inner._ensure_runner()
        print(
            f"generator={type(inner).__name__} backend={llm.resolved_backend} "
            f"quant={llm.resolved_quant} context={inner.context_length} "
            f"max_block={runner.max_block}",
            flush=True,
        )

        for name, content in probes:
            prompt_ids = inner.tokenize_chat(content, enable_thinking=True)
            if name == "thinking-on":
                print(
                    f"[{name}] prompt ids match llama.cpp: "
                    f"{prompt_ids == LLAMA_CPP_PROMPT_IDS} ({len(prompt_ids)} tokens)",
                    flush=True,
                )

            started = time.perf_counter()
            outputs = llm.generate_detailed(
                prompt_ids, SamplingParams(max_tokens=arguments.max_tokens)
            )
            elapsed = time.perf_counter() - started
            output = outputs[0]
            generated = tuple(output.generated_token_ids)
            record = {
                "probe": name,
                "prompt_tokens": len(prompt_ids),
                "generated_token_ids": list(generated),
                "generated_text": output.text,
                "finish_reason": output.finish_details.reason,
                "seconds": round(elapsed, 1),
                "tokens_per_second": round(len(generated) / max(elapsed, 1e-9), 2),
            }
            with arguments.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"[{name}] {len(prompt_ids)} prompt tokens, {len(generated)} generated "
                f"in {elapsed:.1f}s ({record['tokens_per_second']} tok/s)\n"
                f"  ids:  {generated}\n"
                f"  text: {output.text!r}",
                flush=True,
            )
            if name == "thinking-on":
                match = generated[: len(LLAMA_CPP_FIRST_TOKENS)] == LLAMA_CPP_FIRST_TOKENS
                print(f"[{name}] first tokens match llama.cpp: {match}", flush=True)
    finally:
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
