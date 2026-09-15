"""TP2 per-phase layer attribution via eager-schedule stream events.

Diagnostic: drives the production eager schedule's exact per-layer phases
(attention-only leaves, add+norm, shard MLP chain with its staged exchange,
residual add) with three stream events per layer per rank, enqueued in
stream order without intermediate syncs - so the device timeline keeps its
pipeline and each span is the true device execution of that phase.

Attribution target: the 22.7 ms/step layer floor measured under the graphed
schedule, split into attention (replicated) vs MLP-shard + exchange.

Usage::

    python scripts/tp2_layer_phase_attribution.py MODEL.gguf [STEPS]
"""

import sys
import time

sys.path.insert(0, "/home/lhl/hipEngine-main")
import numpy as np

from hipengine.core.device import scoped_current_device
from hipengine.core.memory import copy_host_to_device
from hipengine.distributed.tp2_generate import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    MlpTP2GenerationSession,
)
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import gguf_bf16_add

model = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 8

session = MlpTP2GenerationSession(
    model, devices=(0, 1), mode="tp2", max_sequence_length=256, schedule="eager"
)
runtime = session.runtime
rng = np.random.default_rng(11)
prompt = [int(i) for i in rng.integers(1000, 50000, size=64)]
session.generate(prompt, max_new_tokens=4, eos_token_id=None)
runtime.device_synchronize()

devices = session.devices
layer_types = session._config.layer_types
group = session._shard_group

# Two events per phase boundary per layer per rank: attn_start, attn_stop
# (= mlp_start), mlp_stop. Recorded in stream order; queried once after the
# final device sync.
EVENTS_PER_LAYER = 3
token = int(np.argmax(session.teacher_forced_logits(prompt)[-1]))

t0 = time.perf_counter()
phase_totals = {"attn": 0.0, "mlp": 0.0}
sampled_ok = True
for offset in range(steps):
    pos = offset
    pools = {}
    for device in devices:
        with scoped_current_device(runtime, device):
            pools[device] = [
                runtime.event_create()
                for _ in range(len(layer_types) * EVENTS_PER_LAYER + 1)
            ]
    for device in devices:
        with scoped_current_device(runtime, device):
            pools[device] = [
                runtime.event_create()
                for _ in range(len(layer_types) * EVENTS_PER_LAYER + 1)
            ]
    for device in devices:
        # metadata: token H2D, pinned refresh, embedding (exact production)
        hidden_ptrs = session._enqueue_embedding(token, pos, {})
        stream = session._rank_stream(device)
        with scoped_current_device(runtime, device):
            runtime.event_record(pools[device][0], stream)
    for layer_id, layer_type in enumerate(layer_types):
        for device in devices:
            src, _dst = hidden_ptrs[device]
            stream = session._rank_stream(device)
            pool = pools[device]
            base = layer_id * EVENTS_PER_LAYER
            with scoped_current_device(runtime, device):
                runtime.event_record(pool[base], stream)
                runner = session._runners[device]
                scratch = session._scratches[device]
                attn_out = scratch.attn_out.ptr
                if layer_type == LINEAR_ATTENTION:
                    runner._run_linear_attention_attn_only(
                        layer_id, src, attn_out, scratch, stream=stream
                    )
                elif layer_type == FULL_ATTENTION:
                    runner._run_full_attention_attn_only(
                        layer_id, src, attn_out, scratch, position=pos, stream=stream
                    )
                else:
                    raise SystemExit(f"unsupported layer type {layer_type!r}")
                runtime.event_record(pool[base + 1], stream)
        # add+norm and the shard MLP chain with its staged exchange, exactly
        # as the eager schedule enqueues them (both ranks, rank-skew free).
        for device in devices:
            src, _dst = hidden_ptrs[device]
            stream = session._rank_stream(device)
            with scoped_current_device(runtime, device):
                runner = session._runners[device]
                scratch = session._scratches[device]
                session._add_norm_kernel(runner)(
                    src,
                    scratch.attn_out.ptr,
                    runner.weights.layer(layer_id)
                    .weight("post_attention_norm")
                    .allocation()
                    .tensor.ptr,
                    scratch.post_norm.ptr,
                    scratch.residual.ptr,
                    1,
                    runner.hidden_size,
                    runner.weights.config.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
        if session.mode == "tp2":
            outputs = group.forward(
                layer_id,
                {
                    device: session._scratches[device].post_norm.ptr
                    for device in devices
                },
            )
        for device in devices:
            src, dst = hidden_ptrs[device]
            stream = session._rank_stream(device)
            with scoped_current_device(runtime, device):
                scratch = session._scratches[device]
                gguf_bf16_add(
                    scratch.residual.ptr,
                    outputs[device],
                    dst,
                    session.hidden_size,
                    stream=stream,
                    runtime=runtime,
                )
        for device in devices:
            stream = session._rank_stream(device)
            pool = pools[device]
            with scoped_current_device(runtime, device):
                runtime.event_record(pool[base + 2], stream)
        # swap: the summed output becomes the next layer's input
        for device in devices:
            src, dst = hidden_ptrs[device]
            hidden_ptrs[device] = (dst, src)
    # finish (exact production tail)
    logits = session._finish_step({})
    token = int(np.argmax(logits))
    runtime.device_synchronize()
    for device in devices:
        pool = pools[device]
        with scoped_current_device(runtime, device):
            for layer_id in range(len(layer_types)):
                base = layer_id * EVENTS_PER_LAYER
                phase_totals["attn"] += runtime.event_elapsed_time_ms(
                    pool[base], pool[base + 1]
                )
                phase_totals["mlp"] += runtime.event_elapsed_time_ms(
                    pool[base + 1], pool[base + 2]
                )
        for e in pool:
            runtime.event_destroy(e)

wall = time.perf_counter() - t0
print(f"wall: {wall / steps * 1e3:.3f} ms/step over {steps} steps (eager schedule)")
for phase, total in phase_totals.items():
    per_layer = total / steps / len(layer_types)
    per_step = total / steps
    print(f"{phase}: {per_step:.3f} ms/step ({per_layer * 1e3:.1f} us/layer summed over ranks)")
session.close()
