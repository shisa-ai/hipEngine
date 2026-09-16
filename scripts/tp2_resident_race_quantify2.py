"""Per-subprocess race quantification: one prefill attempt per child process.

The parent spawns N children per arm; each child runs ONE 8-token sequential
prefill on a fresh bare ResidentSession (arm = none | passthrough | events)
and exits — a stalled child is killed by timeout, which reclaims the device.
The parent tallies stall rates per arm.

Usage::

    python scripts/tp2_resident_race_quantify2.py [N_PER_ARM]
"""

import subprocess
import sys
import time

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ARMS = ("none", "passthrough", "events")

CHILD = r'''
import faulthandler, sys, threading, time
faulthandler.dump_traceback_later(45, repeat=True, file=sys.stderr)
sys.path.insert(0, "/home/lhl/hipEngine-main")
arm = sys.argv[1]
from hipengine.loading import load_gguf_index
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
tok = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(MODEL))
ids = tok.encode("The quick brown fox jumps over the lazy dog. Continue:")[:8]
done = {}

def run():
    try:
        with Qwen35GGUFResidentSession(
            MODEL, max_sequence_length=256, require_cached_build=False,
            max_batch_size=1,
        ) as session:
            if arm in ("passthrough", "events"):
                rt = session.runtime
                originals = {}
                for name in ("_run_linear_attention_layer",
                             "_run_full_attention_attn_only",
                             "_run_post_attention_ffn"):
                    originals[name] = getattr(session.runner, name)
                for name, original in originals.items():
                    if arm == "passthrough":
                        def make(original=original):
                            def wrapper(*a, **k):
                                return original(*a, **k)
                            return wrapper
                    else:
                        def make(original=original):
                            def wrapper(*a, **k):
                                stream = k.get("stream", 0)
                                start, stop = rt.event_create(), rt.event_create()
                                rt.event_record(start, stream)
                                result = original(*a, **k)
                                rt.event_record(stop, stream)
                                rt.event_destroy(start)
                                rt.event_destroy(stop)
                                return result
                            return wrapper
                    setattr(session.runner, name, make())
            probe = session.prefill(ids, use_bulk=False, return_logits=False)
            done["token"] = int(probe.token_id)
    except BaseException as error:
        done["error"] = f"{type(error).__name__}: {str(error)[:80]}"

worker = threading.Thread(target=run, daemon=True)
worker.start()
worker.join(timeout=45)
if "token" in done:
    print(f"OK token={done['token']}")
elif "error" in done:
    print(f"ERROR {done['error']}")
else:
    print("STALL")
'''

for arm in ARMS:
    stalls = 0
    for i in range(N):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-u", "-c", CHILD, arm],
            capture_output=True, text=True, timeout=150,
        )
        verdict = "STALL" if "STALL" in proc.stdout else (
            "OK" if "OK" in proc.stdout else f"? {proc.stdout.strip()[:60]} "
            f"{proc.stderr.strip().splitlines()[-1][:60] if proc.stderr.strip() else ''}")
        if verdict == "STALL":
            stalls += 1
        print(f"{arm} run={i}: {verdict} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"ARM {arm}: stalls {stalls}/{N}", flush=True)
print("quantification complete", flush=True)
