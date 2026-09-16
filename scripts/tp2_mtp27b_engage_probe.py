"""Engagement probe: resident NextN MTP cycle on the Qwen3.8-27B dense GGUF.

Drives the in-tree resident cycle (Qwen35GGUFResidentSession target +
Qwen35GGUFNextNDraftProvider + Qwen35GGUFMTPDecodeSession) on this host's
gfx1100 lane. Uses sequential prefill (the qualified path) - the runner's
bulk prefill route NaNs on this lane (worklog ee20cd) and is not required
for MTP engagement. Reports the first measured cycle acceptance stats here.

Usage::

    python scripts/tp2_mtp27b_engage_probe.py [MAX_NEW_TOKENS]
"""

import faulthandler
import json
import sys
import time

# Self-dump the Python stack every 5 minutes: names the loop we stall in
# without needing ptrace permissions (entry 08fcef).
faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)

sys.path.insert(0, "/home/lhl/hipEngine-main")

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
MAX_NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 16

from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFMTPDecodeSession
from hipengine.runtime.qwen35_gguf_nextn import (
    Qwen35GGUFNextNDraftProvider,
    borrow_qwen35_gguf_nextn_fallback_weights,
)
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from hipengine.loading import load_gguf_index

tok = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(MODEL))
prompt_text = "Write a short explanation of what a binary search tree is.\n<|im_start|>assistant\n"
prompt = [int(t) for t in tok.encode(prompt_text)]
print(f"prompt tokens: {len(prompt)}", flush=True)

t0 = time.perf_counter()
with Qwen35GGUFResidentSession(
    MODEL, max_sequence_length=256, require_cached_build=False, max_batch_size=1,
    use_wmma_prefill=True, use_gemv_decode=True,
) as target:
    target.select_prefill_quant("gguf_q4_k_m")
    # AR reference on the clean sequential path
    target.reset()
    probe = target.prefill(prompt, use_bulk=False, return_logits=False)
    ar_ids = [int(probe.token_id)]
    for _ in range(MAX_NEW - 1):
        ar_ids.append(int(target.step(ar_ids[-1], return_logits=False).token_id))
    print(f"AR reference ({MAX_NEW} tokens, {time.perf_counter() - t0:.1f}s): "
          f"{tok.decode(ar_ids)!r}", flush=True)

    provider = Qwen35GGUFNextNDraftProvider.from_model(
        MODEL,
        max_positions=256,
        max_requests=1,
        runtime=target.runtime,
        require_cached_build=False,
        borrowed_fallback_weights=borrow_qwen35_gguf_nextn_fallback_weights(target),
    )
    t1 = time.perf_counter()
    decoder = Qwen35GGUFMTPDecodeSession(
        target,
        provider,
        candidate_budget=3,
        quant="gguf_q4_k_m",
        target_verify_mode="native",
    )
    try:
        result = decoder.generate(
            prompt,
            max_new_tokens=MAX_NEW,
            request_id=1,
            return_cycle_logits=True,
            use_bulk_prefill=False,
        )
        ids = list(result.token_ids) if hasattr(result, "token_ids") else None
        if ids is None and isinstance(result, (list, tuple)):
            ids = list(result)
        print(f"MTP decode wall: {time.perf_counter() - t1:.1f}s", flush=True)
        if ids:
            text = tok.decode([int(t) for t in ids if 0 <= int(t) < 248320])
            print(f"MTP output ({len(ids)} tokens): {text!r}", flush=True)
            exact = [int(t) for t in ids][: len(ar_ids)] == ar_ids
            print(f"greedy-exact vs AR prefix: {exact}", flush=True)
        cycles = getattr(result, "cycles", None)
        if cycles:
            print(f"cycles: {len(cycles)}; summary of first cycle keys: "
                  f"{sorted(cycles[0].keys()) if isinstance(cycles[0], dict) else type(cycles[0])}",
                  flush=True)
            json.dump(
                cycles if isinstance(cycles[0], dict) else [str(c) for c in cycles],
                open("/tmp/mtp27b_cycles.json", "w"), default=str,
            )
    finally:
        decoder.close()
print(f"total wall: {time.perf_counter() - t0:.1f}s", flush=True)
