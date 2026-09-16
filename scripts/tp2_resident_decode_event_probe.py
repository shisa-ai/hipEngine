"""Name the livelocked kernel: per-primitive HIP events on the resident decode.

Monkeypatches ``Qwen35GGUFFullStackRunner._run_linear_attention_layer`` and
``._run_full_attention_layer`` to bracket each call with an event pair, then
runs one bare-resident sequential prefill token. After ~20 s every event is
queried; the first layer whose stop event never completed names the hanging
kernel family. Exits immediately after the verdict (a livelocked kernel
poisons the device until process exit reclaims it).

Usage::

    python scripts/tp2_resident_decode_event_probe.py
"""

import faulthandler
import sys
import threading
import time

faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
sys.path.insert(0, "/home/lhl/hipEngine-main")

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"

from hipengine.core.device import scoped_current_device
from hipengine.core.hip import get_hip_runtime
from hipengine.core.runtime import MemcpyKind  # noqa: F401
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    register_qwen35_paged_attn_decode_kernels,
)
from hipengine.loading import load_gguf_index
from hipengine.runtime.qwen35_gguf_runner import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
)
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

register_qwen35_paged_attn_decode_kernels(replace=True)

rt = get_hip_runtime()
events: list[tuple[str, object, object]] = []

def wrap(runner: Qwen35GGUFFullStackRunner) -> None:
    for name in (
        "_run_linear_attention_layer",
        "_run_full_attention_layer",
        "_run_full_attention_attn_only",
        "_run_post_attention_ffn",
    ):
        original = getattr(runner, name)
        kind = name.split("_run_")[1]

        def make_wrapper(original=original, kind=kind):
            def wrapper(layer_id, *args, **kwargs):
                start, stop = rt.event_create(), rt.event_create()
                stream = kwargs.get("stream", 0)
                with scoped_current_device(rt, 0):
                    rt.event_record(start, stream)
                original(layer_id, *args, **kwargs)
                rt.event_record(stop, stream)
                events.append((f"{kind}:{layer_id}", start, stop))
            return wrapper

        setattr(runner, name, make_wrapper())

tok = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(MODEL))
ids = tok.encode("The quick brown fox jumps over the lazy dog. Continue:")

with Qwen35GGUFResidentSession(
    MODEL, max_sequence_length=256, require_cached_build=False, max_batch_size=1,
) as session:
    wrap(session.runner)
    outcome: dict[str, object] = {}

    def run_prefill():
        t = time.perf_counter()
        try:
            probe = session.prefill(ids[:8], use_bulk=False, return_logits=False)
            outcome["done"] = (time.perf_counter() - t, int(probe.token_id))
        except BaseException as error:  # noqa: BLE001
            outcome["error"] = f"{type(error).__name__}: {str(error)[:150]}"

    worker = threading.Thread(target=run_prefill, daemon=True)
    worker.start()
    worker.join(timeout=30)
    if "done" in outcome:
        print(f"prefill completed in {outcome['done'][0]:.1f}s token={outcome['done'][1]}", flush=True)
    elif "error" in outcome:
        print(f"prefill raised: {outcome['error']}", flush=True)
    else:
        print("prefill still blocked after 30s; querying per-layer events...", flush=True)
    with scoped_current_device(rt, 0):
        for label, start, stop in list(events):
            print(f"{label}: started={rt.event_query(start)} "
                  f"completed={rt.event_query(stop)}", flush=True)
    print("verdict printed; exiting", flush=True)
