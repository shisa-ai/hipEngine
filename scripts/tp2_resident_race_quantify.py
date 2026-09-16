"""Quantify the resident-prefill race: stall rate unwrapped vs mask rate wrapped.

Runs the same 8-token sequential prefill on a fresh bare ResidentSession
N times per arm; each attempt gets a 60 s budget in a worker thread so a
stall is counted and the process moves on. Prints per-arm stall counts.

Usage::

    python scripts/tp2_resident_race_quantify.py [N_PER_ARM]
"""

import faulthandler
import sys
import threading
import time

faulthandler.dump_traceback_later(600, repeat=True, file=sys.stderr)
sys.path.insert(0, "/home/lhl/hipEngine-main")

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3

from hipengine.loading import load_gguf_index
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

tok = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(MODEL))
ids = tok.encode("The quick brown fox jumps over the lazy dog. Continue:")[:8]


def one_attempt(wrapped: bool) -> str:
    outcome: dict[str, str] = {}

    def run():
        try:
            session = Qwen35GGUFResidentSession(
                MODEL, max_sequence_length=256, require_cached_build=False,
                max_batch_size=1,
            )
            with session:
                if wrapped:
                    rt = session.runtime
                    for name in ("_run_full_attention_attn_only", "_run_post_attention_ffn"):
                        original = getattr(session.runner, name)

                        def make_wrapper(original=original):
                            def wrapper(*args, **kwargs):
                                return original(*args, **kwargs)
                            return wrapper

                        setattr(session.runner, name, make_wrapper())
                probe = session.prefill(ids, use_bulk=False, return_logits=False)
                outcome["ok"] = str(int(probe.token_id))
        except BaseException as error:  # noqa: BLE001
            outcome["error"] = f"{type(error).__name__}: {str(error)[:100]}"

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=60)
    return outcome.get("ok", "STALL" if worker.is_alive() else outcome.get("error", "?"))


for wrapped in (False, True):
    results = []
    for i in range(N):
        t0 = time.perf_counter()
        result = one_attempt(wrapped)
        results.append(result)
        print(f"wrapped={wrapped} run={i}: {result} ({time.perf_counter() - t0:.1f}s)",
              flush=True)
    stalls = sum(1 for r in results if r == "STALL")
    print(f"ARM wrapped={wrapped}: stalls {stalls}/{N}", flush=True)
print("quantification complete", flush=True)
