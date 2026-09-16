"""Corrected d1-teacher-stall bisect: fresh subprocess per case, main-thread session.

Per lead review: one child process per case; the child constructs AND
executes the session on its main thread; the parent enforces the timeout
and terminates/reaps the child (never closes the session while blocked).
Cases use exact common-prefix token sequences of one fixed sequence, and
record token IDs.

Child usage: tp2_teacher_bisect_child.py <N_POSITIONS> <DEVICES>
Parent usage: tp2_teacher_bisect_parent.py [N_PER_CASE] [CASES...]
"""

import subprocess
import sys
import time

CHILD = r'''
import sys, time
sys.path.insert(0, "/home/lhl/hipEngine-main")
n_positions = int(sys.argv[1])
devices = tuple(int(d) for d in sys.argv[2].split(","))
mode = "tp2" if len(devices) > 1 else "tp1"
import hipengine.loading as _loading
from hipengine.distributed.tp2_generate import MlpTP2GenerationSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
tok = Qwen35GGUFTokenizer.from_gguf_info(_loading.load_gguf_index(MODEL))
FULL = tuple(int(t) for t in tok.encode(
    "<|im_start|>user\nWrite a Python function that reverses a list and explain "
    "its complexity in detail with examples and edge cases considered."
    "<|im_end|>\n<|im_start|>assistant\n"))
tokens = FULL[:n_positions]
print(f"CHILD tokens={list(tokens)}", flush=True)
session = MlpTP2GenerationSession(MODEL, devices=devices, mode=mode)
t0 = time.perf_counter()
logits = session.teacher_forced_logits(tokens)
print(f"CHILD OK positions={logits.shape[0]} wall={time.perf_counter() - t0:.1f}s "
      f"argmax_last={int(__import__('numpy').argmax(logits[-1]))}", flush=True)
'''

def run_case(n_positions: int, devices: str, timeout: float = 240.0) -> str:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-u", "-c", CHILD, str(n_positions), devices],
            capture_output=True, text=True, timeout=timeout,
        )
        lines = [l for l in proc.stdout.splitlines() if l.startswith("CHILD ")]
        verdict = lines[-1] if lines else f"NO-OUTPUT rc={proc.returncode}"
        if "CHILD OK" not in proc.stdout:
            verdict += " | STALL/timeout" if proc.returncode != 0 else ""
    except subprocess.TimeoutExpired:
        verdict = f"TIMEOUT after {timeout}s (killed and reaped)"
    print(f"n={n_positions} devices={devices}: {verdict} ({time.perf_counter() - t0:.0f}s)",
          flush=True)
    return verdict

if __name__ == "__main__":
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    # Controls first: fresh-process 16 and 24 on the XTX-as-device-1.
    for n in (16, 24):
        for _ in range(n_per):
            run_case(n, "1")
