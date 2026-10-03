#!/usr/bin/env python3
"""Decode/throughput comparison on the gufo cached-prefix protocol.

depth d = untimed cached prefix; timed request = a new TURN-token prefill plus
DEC decode steps, exactly gufo's pp2048/tg128 shape.  Configs:
  bf16      -> BF16 KV, eager
  int8      -> int8_per_token_head fp16, no-mirror (diagnostic capability), eager
  int8graph -> same as int8 but the C1 decode graph is admitted
  bf16graph -> BF16 KV with the C1 decode graph

Run: rt_protocol.py <config> <out.json> [depths] [reps]
"""
import ctypes, json, os, statistics, sys, threading, time
sys.path.insert(0, "/home/lhl/hipEngine")
ctypes.CDLL("libamdhip64.so")

from hipengine.kvcache import resolve_kv_policy
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from scripts.qwen35_native_mixed_kv_suite import _diagnostic_capability_mapping
from scripts.qwen35_gguf_bench import (
    _memory_snapshot,
    _rearm_reused_decode_graph,
    _backend_min_replay_steps,
)

MODEL = "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
COMPILER = "/tmp/reprof/compiler.txt"
TURN = 2048
DEC = 128
TOKEN = 9707

config = sys.argv[1]
out_path = sys.argv[2]
DEPTHS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else "0,4096,16384".split(","))]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 2

use_graph = config.endswith("graph")
storage = "bf16" if config.startswith("bf16") else "int8_per_token_head"

busy, stop = [], threading.Event()


def sample():
    while not stop.is_set():
        try:
            busy.append(int(open("/sys/class/drm/card1/device/gpu_busy_percent").read().strip()))
        except Exception:
            pass
        time.sleep(0.05)


threading.Thread(target=sample, daemon=True).start()

policy = resolve_kv_policy(storage, scale_dtype="fp16")
cap = max(DEPTHS) + TURN + DEC + 64
kwargs = dict(
    max_sequence_length=cap,
    backend="hip_gfx1151",
    use_wmma_prefill=True,
    use_gemv_decode=True,
    kv_policy=policy.create_policy(),
    kv_scale_dtype=policy.scale_dtype,
    kv_scale_granularity=policy.scale_granularity,
    compiler_version=open(COMPILER).read().strip(),
    require_cached_build=True,
)
if storage != "bf16":
    kwargs["kv_capability"] = _diagnostic_capability_mapping()

out = {
    "config": config,
    "storage": storage,
    "use_graph_requested": use_graph,
    "turn_tokens": TURN,
    "decode_tokens": DEC,
    "reps": REPS,
    "depths": {},
}
with Qwen35GGUFResidentSession(MODEL, **kwargs) as s:
    out["kv_storage_dtype"] = str(s.kv_storage_dtype)
    out["kv_storage_layout"] = str(getattr(s, "kv_storage_layout", None))
    out["kv_scale_granularity"] = str(getattr(s, "kv_scale_granularity", None))
    out["int8_kv_no_mirror_qualified"] = bool(getattr(s, "int8_kv_no_mirror_qualified", False))
    out["decode_graph_min_replay_steps"] = _backend_min_replay_steps(s)
    out["bf16_mirror_layer_indices"] = list(getattr(s, "bf16_mirror_layer_indices", ()))
    out["int8_bf16_full_attention_layer_indices"] = list(
        getattr(s, "int8_bf16_full_attention_layer_indices", ())
    )
    out["scratch_max_positions"] = int(s.scratch.max_positions)
    out["after_load_memory"] = _memory_snapshot("after_load", s.runtime, s)

    for depth in DEPTHS:
        pp, tg, graph_used = [], [], False
        graph = None
        for rep in range(REPS + 1):
            s.reset(stream=0)
            s.runtime.device_synchronize()
            if depth:
                s.prefill([TOKEN] * depth, use_bulk=True, return_logits=False)
                s.runtime.device_synchronize()
            t0 = time.perf_counter()
            s.prefill([TOKEN] * TURN, use_bulk=True, return_logits=False)
            s.runtime.device_synchronize()
            p = time.perf_counter() - t0
            if use_graph:
                if graph is None:
                    graph = s.capture_decode_graph(
                        position=s.position,
                        steps_per_replay=1,
                        max_replay_steps=DEC,
                        record_steps=0,
                    )
                else:
                    _rearm_reused_decode_graph(s, graph, s.runtime)
                graph_used = True
                t1 = time.perf_counter()
                graph.replay(DEC)
                s.runtime.device_synchronize()
                d = time.perf_counter() - t1
                final = graph.read_sample()
            else:
                tok = TOKEN
                t1 = time.perf_counter()
                for i in range(DEC):
                    tok = s.step(tok, return_logits=(i == DEC - 1)).token_id
                s.runtime.device_synchronize()
                d = time.perf_counter() - t1
                final = tok
            if rep:
                pp.append(TURN / p)
                tg.append(DEC / d)
            print(
                f"  depth={depth} rep{rep}{' (warm)' if rep == 0 else ''} "
                f"pp={TURN / p:.1f} tok/s tg={DEC / d:.2f} tok/s",
                flush=True,
            )
        if graph is not None:
            graph.close()
            graph = None
        out["depths"][str(depth)] = {
            "context_tokens": depth + TURN,
            "pp_tok_s": round(statistics.median(pp), 1),
            "tg_tok_s": round(statistics.median(tg), 2),
            "pp_samples": [round(x, 1) for x in pp],
            "tg_samples": [round(x, 2) for x in tg],
            "graph_used": graph_used,
        }
    out["after_run_memory"] = _memory_snapshot("after_run", s.runtime, s)

stop.set()
nz = [b for b in busy if b > 0]
out["gpu_busy_mean"] = round(statistics.mean(nz), 2) if nz else None
open(out_path, "w").write(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
