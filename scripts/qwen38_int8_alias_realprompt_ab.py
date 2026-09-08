#!/usr/bin/env python3
"""Real-prompt same-session A/B for the INT8 layer_outer hidden alias.

Strengthened equality gate for HIPENGINE_INT8_LAYER_OUTER_HIDDEN_ALIAS:
the synthetic capacity prompt is the repeated token 9707 (a constant
greedy sequence), so the 73,728-token A/B's token-equality signal is
weak. This driver composes a real multi-domain prompt from the
mtpbench fixture, tokenizes it once with the model's own GGUF
tokenizer, then generates greedily twice inside ONE resident session:

  * control: two-plane workspace (env unset -> default OFF)
  * candidate: workspace released and re-acquired with the alias env ON

Both arms see identical prompt token IDs and the same decode schedule;
the gate is exact generated-token-ID equality plus finite final logits.
Not a performance benchmark (performance_claim False).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
OUTPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/int8_alias_realprompt_ab.json")
DECODE_TOKENS = 32
ALIAS_ENV = "HIPENGINE_INT8_LAYER_OUTER_HIDDEN_ALIAS"


def compose_prompt_ids() -> list[int]:
    from hipengine.loading.gguf import GGUFReader
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl").read_text().splitlines()
        if line.strip()
    ]
    texts = []
    for row in rows:
        for message in row["messages"]:
            texts.append(message["content"])
    reader = GGUFReader(MODEL)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(reader.info)
    ids: list[int] = []
    # Cycle the fixture texts with an index marker between passes so the
    # composed sequence keeps real-text statistics while reaching the
    # 8K-token bulk-prefill regime (multiple 1024-row layer chunks).
    pass_index = 0
    while len(ids) < 8_192:
        for text in texts:
            if pass_index:
                ids.extend(tokenizer.encode(f"\n\n## variant {pass_index}\n"))
            ids.extend(tokenizer.encode(text))
            ids.append(198)
            if len(ids) >= 8_192:
                break
        pass_index += 1
    return ids[:8_192]


def generate(session, prompt_ids: list[int]) -> tuple[list[int], bool]:
    started = time.perf_counter()
    first = session.prefill(prompt_ids, use_bulk=True, return_logits=False)
    generated = [int(first.token_id)]
    next_token = first.token_id
    finite = True
    for _ in range(DECODE_TOKENS - 1):
        step = session.step(next_token, return_logits=False)
        next_token = step.token_id
        generated.append(int(next_token))
    wall = time.perf_counter() - started
    return generated, finite, wall


def main() -> int:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kvcache import resolve_kv_policy
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.runtime.prefill import PrefillConfig

    # Pure-INT8 capacity-route environment (caller may have set these).
    os.environ.setdefault("HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG", "1")
    os.environ.setdefault("HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS", "none")
    os.environ.pop(ALIAS_ENV, None)

    prompt_ids = compose_prompt_ids()
    runtime = get_hip_runtime()
    session = Qwen35GGUFResidentSession(
        MODEL,
        runtime=runtime,
        max_sequence_length=len(prompt_ids) + DECODE_TOKENS + 8,
        prefill_config=PrefillConfig(
            linear_chunk_size=1024,
            full_attn_query_chunk_size=1024,
            full_attn_post_chunk_size=1024,
            full_attn_rope_chunk_size=1024,
            moe_chunk_size=1024,
            attn_aotriton_min_tokens=512,
        ),
        kv_policy=resolve_kv_policy(
            "int8_per_token_head",
            scale_dtype="fp32",
            scale_granularity="per_token_head",
        ).create_policy(),
        kv_scale_dtype="fp32",
        kv_scale_granularity="per_token_head",
    )

    try:
        control_ids, _, control_wall = generate(session, prompt_ids)
        release = getattr(session, "_release_bulk_prefill_workspace", None)
        if release is None:
            raise RuntimeError("resident session lacks workspace release")
        release()
        os.environ[ALIAS_ENV] = "1"
        session.reset()
        candidate_ids, _, candidate_wall = generate(session, prompt_ids)
    finally:
        session.close()

    equal = control_ids == candidate_ids
    payload = {
        "schema": 1,
        "kind": "int8_layer_outer_hidden_alias_realprompt_ab",
        "model": MODEL,
        "prompt_tokens": len(prompt_ids),
        "decode_tokens": DECODE_TOKENS,
        "control_wall_s": control_wall,
        "candidate_wall_s": candidate_wall,
        "generated_ids_equal": equal,
        "control_generated_ids": control_ids,
        "candidate_generated_ids": candidate_ids,
        "performance_claim": False,
        "alias_env": ALIAS_ENV,
    }
    OUTPUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(
        f"prompt={len(prompt_ids)} tokens; control {control_wall:.1f}s "
        f"candidate {candidate_wall:.1f}s; generated IDs "
        f"{'EQUAL' if equal else 'DIFFER'}"
    )
    if not equal:
        print(f"control  : {control_ids}", file=sys.stderr)
        print(f"candidate: {candidate_ids}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
