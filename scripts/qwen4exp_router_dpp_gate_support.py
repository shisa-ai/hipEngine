"""Private-buffer natural-input checker for the ordered router candidate.

Only correctness harnesses install this hook. It never overwrites live model
outputs and its synchronized copies are excluded from performance evidence.
"""
from __future__ import annotations

import hashlib

import numpy as np

from hipengine.core.memory import copy_device_to_host, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.moe import router


def assert_router_outputs_exact(parent, candidate, repeat) -> dict[str, str]:
    """Compare raw bytes, including logits, int64 expert IDs and F32 routing."""
    hashes = {}
    for name, arrays in zip(("logits", "expert_ids", "routing"),
                            zip(parent, candidate, repeat, strict=True), strict=True):
        left, right, again = arrays
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"{name}: candidate shape/dtype mismatch")
        if left.shape != again.shape or left.dtype != again.dtype:
            raise AssertionError(f"{name}: repeat shape/dtype mismatch")
        if not np.isfinite(right).all():
            raise AssertionError(f"{name}: nonfinite candidate")
        if left.tobytes() != right.tobytes():
            raise AssertionError(f"{name}: candidate differs from parent bits")
        if right.tobytes() != again.tobytes():
            raise AssertionError(f"{name}: repeat differs from candidate bits")
        hashes[name + "_sha256"] = hashlib.sha256(right.tobytes()).hexdigest()
    return hashes


class NaturalRouterCheck:
    """Run parent and candidate-repeat into private buffers on each live call."""

    def __init__(self):
        self.captures = []

    def check(self, hidden_ptr, weight_ptr, live_logits_ptr, rows, hidden, experts,
              *, runtime, stream=0, **kwargs):
        rows, hidden, experts = int(rows), int(hidden), int(experts)
        top_k = 10
        output_arrays = [np.empty((rows, experts), np.float32),
                         np.empty((rows, top_k), np.int64),
                         np.empty((rows, top_k), np.float32)]
        buffers = []
        try:
            for _ in range(3):
                group = []
                buffers.append(group)
                for array in output_arrays:
                    group.append(malloc(array.nbytes, runtime=runtime))
            # The candidate has already written live logits. Parent and repeat
            # use separate outputs, never model scratch. All kernels use the
            # same request stream, so copies follow both producer and selector.
            parent, candidate, repeat = buffers
            router.qwen35_router_logits_f32_f32w_token_tile4_dense_exact(
                hidden_ptr, weight_ptr, parent[0].ptr, rows, hidden, experts,
                runtime=runtime, stream=stream)
            router.qwen35_router_logits_f32_f32w_token_tile4_dpp_exact(
                hidden_ptr, weight_ptr, repeat[0].ptr, rows, hidden, experts,
                runtime=runtime, stream=stream)
            from hipengine.core.runtime import MemcpyKind
            runtime.device_synchronize()
            runtime.memcpy(candidate[0].ptr, live_logits_ptr, output_arrays[0].nbytes,
                           MemcpyKind.DEVICE_TO_DEVICE)
            for group in buffers:
                router.qwen35_router_select(group[0].ptr, group[1].ptr, group[2].ptr,
                                           rows, experts, experts, top_k,
                                           runtime=runtime, stream=stream)
            runtime.device_synchronize()
            outputs = []
            for group in buffers:
                host = [np.empty_like(a) for a in output_arrays]
                for array, buffer in zip(host, group, strict=True):
                    copy_device_to_host(host_array_ptr(array), buffer, array.nbytes,
                                        runtime=runtime)
                outputs.append(host)
            hashes = assert_router_outputs_exact(*outputs)
            self.captures.append({"ordinal": len(self.captures), "rows": rows,
                                  "hidden": hidden, "experts": experts,
                                  "expert_id_dtype": "int64", "parent_repeat_exact": True,
                                  **hashes})
        finally:
            for group in reversed(buffers):
                for buffer in reversed(group):
                    free(buffer, runtime=runtime)
