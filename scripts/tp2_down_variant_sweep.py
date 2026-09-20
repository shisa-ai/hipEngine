"""Down-shard GEMV variant sweep at the TP2 shard shape.

Standalone microbench over one real down shard materialized from the
artifact (the down projections are Q6_K in this Q4_K_M artifact: layout
``gguf_q6_k_t16_qmicro_planar_v1``, 36.5 MB per rank): time the registered
decode variants at the exact TP2 shape (rows=1, in=4352 per rank, out=5120,
bf16 activation, bf16 partial) with HIP events, and check bit-parity of
every variant's output against the incumbent.

This is a diagnostic; promotion goes through the registry/policy path with
the full correctness gates, not by editing this script.

Usage::

    python scripts/tp2_down_variant_sweep.py MODEL.gguf [ITERATIONS]
"""
import pathlib

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from hipengine.core.device import scoped_current_device
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device
from hipengine.core.runtime import MemcpyKind
from hipengine.distributed.shard_weights import materialize_mlp_shards
from hipengine.runtime.gguf_linear import launch_gguf_linear

model = sys.argv[1]
iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 200

rt = get_hip_runtime()
device = 0
per_rank_ffn = 4352
hidden = 5120

# The full runtime import chain registers the q6 t16 consumers.
import hipengine.distributed.tp2_generate  # noqa: F401,E402

shards = materialize_mlp_shards(model, world_size=2, layer_ids=[0])
payload = shards[0].rank_payloads(device)["ffn_down"]
print(f"down shard: layout={payload.layout} quant={payload.quant_key} "
      f"shape={payload.local_shape} bytes={payload.payload.nbytes}")

VARIANTS = (
    None,  # the incumbent dispatch (use_gemv_decode=True, bf16 out)
    "t16_gemv_rowtile_bf16_bf16_out",
    "t16_gemv_rowtile_col8_bf16_bf16_out",
    "t16_gemv_rowtile_col8_grouped_rows6_bf16_bf16_out",
    "t16_gemv_rowtile_col8_grouped_rows8_bf16_bf16_out",
    "t16_q8_1_dp4a_gemv_bf16_bf16_out",
    "t16_q8_1_dp4a_gemv_grouped_bf16_bf16_out",
    "f16_rocblas_t16_qmicro_planar_bf16_bf16_out",
)

class _RawPtr:
    def __init__(self, ptr, nbytes):
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)

with scoped_current_device(rt, device):
    stream = rt.stream_create()
    from hipengine.distributed.shard_exec import upload_shard_weight

    weight = upload_shard_weight(
        rt, device=device, name="tiles", layout=payload.layout,
        quant_key=payload.quant_key, payload=payload.payload,
    )
    rng = np.random.default_rng(5)
    act = (rng.standard_normal(per_rank_ffn) * 0.3).astype(np.float32)
    act_bits = (np.frombuffer(act.tobytes(), dtype=np.uint32) >> 16).astype(np.uint16)
    act_dev = rt.malloc(per_rank_ffn * 2)
    copy_host_to_device(_RawPtr(act_dev, per_rank_ffn * 2), act_bits.ctypes.data,
                        per_rank_ffn * 2, runtime=rt)
    partial_dev = rt.malloc(hidden * 2)
    start, stop = rt.event_create(), rt.event_create()

    outputs = {}
    for variant in VARIANTS:
        label = variant or "incumbent"
        try:
            for _ in range(8):
                launch_gguf_linear(
                    weight, act_dev, partial_dev, 1, per_rank_ffn, hidden,
                    use_gemv_decode=True, stream=stream, runtime=rt,
                    registered_variant=variant,
                )
            rt.stream_synchronize(stream)
            rt.event_record(start, stream)
            for _ in range(iterations):
                launch_gguf_linear(
                    weight, act_dev, partial_dev, 1, per_rank_ffn, hidden,
                    use_gemv_decode=True, stream=stream, runtime=rt,
                    registered_variant=variant,
                )
            rt.event_record(stop, stream)
            rt.event_synchronize(stop)
            ms = rt.event_elapsed_time_ms(start, stop) / iterations
            out = np.empty(hidden, dtype="<u2")
            rt.memcpy(out.ctypes.data, partial_dev, hidden * 2, MemcpyKind.DEVICE_TO_HOST)
            outputs[label] = (ms, out)
            gb = (payload.payload.nbytes + per_rank_ffn * 2 + hidden * 2) / 1e9
            print(f"{label:44s} {ms * 1e3:8.2f} us/launch  ~{gb / (ms * 1e-3):7.1f} GB/s",
                  flush=True)
        except Exception as error:  # noqa: BLE001 - sweep continues
            print(f"{label:44s} FAILED: {type(error).__name__}: {str(error)[:90]}",
                  flush=True)

    if "incumbent" in outputs:
        incumbent = outputs["incumbent"][1]
        for label, (_ms, out) in outputs.items():
            ndiff = int(np.count_nonzero(out != incumbent))
            print(f"parity {label:44s} ndiff={ndiff}")

    rt.event_destroy(start); rt.event_destroy(stop)
    rt.free(act_dev); rt.free(partial_dev); rt.stream_destroy(stream)
