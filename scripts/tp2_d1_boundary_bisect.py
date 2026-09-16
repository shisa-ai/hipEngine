"""Bisect the d1 (XTX) harness teacher stall by chat-row length.

Builds ONE tp1 session on device 1 and sweeps rendered chat rows of
increasing length (via padding the user text), 3 rows per length, with a
per-call watchdog thread that reports a stall and exits the process (a
livelocked readback poisons the device until exit).

Usage::

    python scripts/tp2_d1_boundary_bisect.py
"""

import faulthandler
import sys
import threading
import time

faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
sys.path.insert(0, "/home/lhl/hipEngine-main")

import numpy as np

import hipengine.loading as _loading
from hipengine.distributed.tp2_generate import MlpTP2GenerationSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
LENGTHS = (16, 24, 32, 40, 48, 56, 64)

tok = Qwen35GGUFTokenizer.from_gguf_info(_loading.load_gguf_index(MODEL))

BASE_TEXT = ("Write a Python function that reverses a list and explain its "
             "complexity in detail with examples and edge cases considered.")


def render(tokens_target: int) -> tuple[int, ...]:
    prefix = tuple(int(t) for t in tok.encode("<|im_start|>user\n"))
    suffix = tuple(int(t) for t in tok.encode("<|im_end|>\n<|im_start|>assistant\n"))
    filler = tuple(int(t) for t in tok.encode(BASE_TEXT))
    body_len = tokens_target - len(prefix) - len(suffix)
    if body_len <= 0:
        raise SystemExit(f"target {tokens_target} too small")
    body = filler
    while len(body) < body_len:
        body = body + filler
    return prefix + body[:body_len] + suffix


session = MlpTP2GenerationSession(MODEL, devices=(1,), mode="tp1")
print("d1 session built", flush=True)

for target in LENGTHS:
    tokens = render(target)
    for repeat in range(3):
        outcome: dict[str, str] = {}

        def run():
            try:
                logits = session.teacher_forced_logits(tokens)
                outcome["ok"] = f"rows={logits.shape[0]}"
            except BaseException as error:  # noqa: BLE001
                outcome["error"] = f"{type(error).__name__}: {str(error)[:80]}"

        worker = threading.Thread(target=run, daemon=True)
        t0 = time.perf_counter()
        worker.start()
        worker.join(timeout=40)
        verdict = ("OK " + outcome.get("ok", "")) if "ok" in outcome else (
            outcome.get("error", "STALL"))
        print(f"len={len(tokens)} target={target} repeat={repeat}: {verdict} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
        if "ok" not in outcome:
            print(f"STALL at length {len(tokens)}; exiting (device poisoned until exit)",
                  flush=True)
            session.close()
            sys.exit(2)
print("no boundary found up to", LENGTHS[-1], flush=True)
session.close()
