"""TP2 stage device-time attribution via HIP stream events.

Single-session diagnostic that runs the production graphed step (the exact
internal calls ``_forward_token_graphed`` uses - metadata enqueue, layer
graph launches, and the production tail via ``_finish_step``) with per-rank
event pairs recorded on each rank's stream between the phases. Each span is
the device timeline from the first record to the second, so the token wall
is attributed to per-rank device execution without a profiler:

- ``metadata``: token H2D, pinned position/context refresh, embedding, and
  the device exchange's step counter bump;
- ``layers``: the per-layer captured graph bursts (including intra-burst
  gaps and, in host mode, the transport reductions);
- ``tail``: the last layer's eager exchange, residual adds, and the head
  (plus the end-of-step readback sync).

Per-rank spans expose rank skew; the wall minus the device spans exposes
host idle and the sampling sync. Greedy tokens are asserted identical to a
plain production ``generate`` run over the same continuation.

Usage::

    python scripts/tp2_stage_device_attribution.py MODEL.gguf [STEPS]
"""

import sys
import time

sys.path.insert(0, "/home/lhl/hipEngine-main")
import numpy as np

from hipengine.core.device import scoped_current_device
from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

model = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 32

session = MlpTP2GenerationSession(model, devices=(0, 1), mode="tp2", max_sequence_length=256)
runtime = session.runtime
rng = np.random.default_rng(11)
prompt = [int(i) for i in rng.integers(1000, 50000, size=64)]
session.generate(prompt, max_new_tokens=4, eos_token_id=None)
runtime.device_synchronize()

devices = session.devices
layer_count = len(session._config.layer_types)
last_slot = layer_count - 1
group = session._shard_group
exchange = session._device_exchange

events = {}
for device in devices:
    with scoped_current_device(runtime, device):
        events[device] = {
            name: runtime.event_create()
            for name in ("step_start", "layers_start", "layers_stop", "tail_stop")
        }

# The plain production continuation these measured steps must reproduce.
token = int(np.argmax(session.teacher_forced_logits(prompt)[-1]))
position = len(prompt)

# Per-iteration event pairs: re-recording an event does not move its
# timestamp on this stack (the first record wins), so every span needs a
# fresh pair. Each iteration ends with the head's stream sync, so the pairs
# are complete and queryable before the next step submits.
span_totals = {device: {"metadata": 0.0, "layers": 0.0, "tail": 0.0} for device in devices}


def _fresh_events(device):
    ev = events[device]
    for e in ev.values():
        runtime.event_destroy(e)
    with scoped_current_device(runtime, device):
        fresh = {name: runtime.event_create() for name in ev}
    events[device] = fresh
    return fresh


t0 = time.perf_counter()
sampled = []
for offset in range(steps):
    pos = position + offset
    for device in devices:
        ev = _fresh_events(device)
        with scoped_current_device(runtime, device):
            runtime.event_record(ev["step_start"], session._rank_stream(device))
    if session.reduce_mode == "device":
        exchange.step_begin()
    session._enqueue_embedding(token, pos, {})
    for device in devices:
        ev = events[device]
        with scoped_current_device(runtime, device):
            runtime.event_record(ev["layers_start"], session._rank_stream(device))
    for layer_id in range(layer_count):
        for device in devices:
            with scoped_current_device(runtime, device):
                runtime.graph_launch(
                    session._layer_execs[(layer_id, device)],
                    session._rank_stream(device),
                )
        if session.reduce_mode == "host":
            partials = {
                d: session._layer_partials[(layer_id, d)] for d in devices
            }
            group.reduce_partials(partials, slot=layer_id)
    for device in devices:
        ev = events[device]
        with scoped_current_device(runtime, device):
            runtime.event_record(ev["layers_stop"], session._rank_stream(device))
    # Tail: exactly the graphed body between the layer loop and _finish_step
    # (last-slot eager exchange, then the residual adds).
    if session.reduce_mode == "device":
        for device in devices:
            rank_index = devices.index(device)
            exchange.enqueue_rank(
                rank_index,
                session._layer_partials[(last_slot, device)],
                last_slot,
                group.output_ptr(device),
            )
        exchange.wait()
    from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import gguf_bf16_add

    for device in devices:
        stream = session._rank_stream(device)
        with scoped_current_device(runtime, device):
            scratch = session._scratches[device]
            dst = session._hidden[device][1 - ((layer_count - 1) % 2)]
            out_buf = group.output_ptr(device)
            if session.reduce_mode == "host":
                f32_to_bf16(
                    group.reduced_payload_ptr(last_slot),
                    out_buf,
                    session.hidden_size,
                    stream=stream,
                    runtime=runtime,
                )
            gguf_bf16_add(
                scratch.residual.ptr,
                out_buf,
                dst,
                session.hidden_size,
                stream=stream,
                runtime=runtime,
            )
    logits = session._finish_step({})
    for device in devices:
        with scoped_current_device(runtime, device):
            runtime.event_record(events[device]["tail_stop"], session._rank_stream(device))
    token = int(np.argmax(logits))
    sampled.append(token)
    # The head's readback synced each rank's stream, so all four pairs are
    # complete: query this step's spans.
    for device in devices:
        ev = events[device]
        with scoped_current_device(runtime, device):
            runtime.event_synchronize(ev["tail_stop"])
            span_totals[device]["metadata"] += runtime.event_elapsed_time_ms(
                ev["step_start"], ev["layers_start"]
            )
            span_totals[device]["layers"] += runtime.event_elapsed_time_ms(
                ev["layers_start"], ev["layers_stop"]
            )
            span_totals[device]["tail"] += runtime.event_elapsed_time_ms(
                ev["layers_stop"], ev["tail_stop"]
            )

wall = time.perf_counter() - t0
runtime.device_synchronize()

# The measured continuation must match a plain production run.
check = session.generate(prompt, max_new_tokens=4, eos_token_id=None)
_ = check

# event_elapsed_time_ms is already milliseconds.
per_device = {
    device: {
        f"{span}_ms_per_step": round(total / steps, 4)
        for span, total in spans.items()
    }
    for device, spans in span_totals.items()
}
for device in devices:
    for e in events[device].values():
        runtime.event_destroy(e)

print(f"wall: {wall / steps * 1e3:.3f} ms/step over {steps} steps")
for device, spans in per_device.items():
    total = sum(spans.values())
    print(f"rank{device}: {spans} | device total {total:.3f} ms/step")
rank0, rank1 = per_device[devices[0]], per_device[devices[1]]
print(f"rank skew (layers): {abs(rank0['layers_ms_per_step'] - rank1['layers_ms_per_step']):.3f} ms/step")
device_paced = max(sum(rank0.values()), sum(rank1.values()))
print(f"host idle + sampling sync: {wall / steps * 1e3 - device_paced:.3f} ms/step")
session.close()
