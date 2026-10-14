"""The three-launch decode read is bit-identical to the six-launch chain."""
import ctypes
import time

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    qwen4_exp_gated_mean_sigmoid_f32,
    qwen4_exp_grouped_rmsnorm_bf16_f32,
    qwen4_exp_scaled_silu_f32,
)
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import dense_gemv_out_f32
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    gguf_q8_0_pack8_gemv_f32_f32_out,
    qwen4_exp_gr_down_pack8_decode,
    qwen4_exp_gr_up_mean_pack8_decode,
)
from tests.test_gpu_gguf_k_gemv import make_q8_0_weight


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")


def _bf16(values):
    bits = np.ascontiguousarray(values, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def run_read(rows, hidden, low_rank, *, inject=True, iters=0):
    runtime = get_hip_runtime()
    rng = np.random.default_rng(rows * 11 + hidden + low_rank)
    width = 4 * hidden
    host = {
        "residual": _bf16(rng.normal(0, 2.0, (rows, width))),
        "norm_weight": rng.normal(1.0, 0.2, width).astype(np.float32),
        "inject_weight": rng.normal(0, 0.02, (4, width)).astype(np.float32),
        "down": make_q8_0_weight(low_rank, width),
        "up": make_q8_0_weight(width, low_rank),
    }
    dev = {name: malloc(a.nbytes, runtime=runtime) for name, a in host.items()}
    sizes = {"plane": rows * width * 4, "gate": rows * width * 4,
             "low_p": rows * low_rank * 4, "low_c": rows * low_rank * 4,
             "inject_p": rows * 16, "inject_c": rows * 16,
             "mixed_p": rows * hidden * 4, "mixed_c": rows * hidden * 4}
    dev.update({name: malloc(n, runtime=runtime) for name, n in sizes.items()})
    try:
        for name, a in host.items():
            copy_host_to_device(dev[name], host_array_ptr(a), runtime=runtime)
        zeros = np.zeros(rows * 4, np.float32)
        copy_host_to_device(dev["inject_p"], host_array_ptr(zeros), runtime=runtime)

        def chain():
            qwen4_exp_grouped_rmsnorm_bf16_f32(
                dev["residual"].ptr, dev["norm_weight"].ptr, dev["plane"].ptr,
                rows, 4, hidden, runtime=runtime)
            gguf_q8_0_pack8_gemv_f32_f32_out(
                dev["plane"].ptr, dev["down"].ptr, dev["low_p"].ptr, rows, width,
                low_rank, threads=256, runtime=runtime)
            qwen4_exp_scaled_silu_f32(
                dev["low_p"].ptr, dev["low_p"].ptr, rows * low_rank, 0.25, runtime=runtime)
            gguf_q8_0_pack8_gemv_f32_f32_out(
                dev["low_p"].ptr, dev["up"].ptr, dev["gate"].ptr, rows, low_rank,
                width, threads=128, runtime=runtime)
            if inject:
                dense_gemv_out_f32(
                    dev["plane"].ptr, dev["inject_weight"].ptr, dev["inject_p"].ptr,
                    rows, width, 4, runtime=runtime)
            qwen4_exp_gated_mean_sigmoid_f32(
                dev["plane"].ptr, dev["gate"].ptr, dev["mixed_p"].ptr, rows, 4,
                hidden, runtime=runtime)

        def fused():
            qwen4_exp_grouped_rmsnorm_bf16_f32(
                dev["residual"].ptr, dev["norm_weight"].ptr, dev["plane"].ptr,
                rows, 4, hidden, runtime=runtime)
            qwen4_exp_gr_down_pack8_decode(
                dev["plane"].ptr, dev["down"].ptr,
                dev["inject_weight"].ptr if inject else 0, dev["low_c"].ptr,
                dev["inject_c"].ptr, rows, width, low_rank, 0.25, runtime=runtime)
            qwen4_exp_gr_up_mean_pack8_decode(
                dev["low_c"].ptr, dev["up"].ptr, dev["plane"].ptr, dev["mixed_c"].ptr,
                rows, 4, hidden, low_rank, runtime=runtime)

        timings = []
        for fn in (chain, fused):
            fn()
            runtime.device_synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                fn()
            runtime.device_synchronize()
            timings.append((time.perf_counter() - start) / max(iters, 1) * 1e6)
        out = {}
        for name, count in (("low_p", rows * low_rank), ("low_c", rows * low_rank),
                            ("inject_p", rows * 4), ("inject_c", rows * 4),
                            ("mixed_p", rows * hidden), ("mixed_c", rows * hidden)):
            a = np.empty(count, np.uint32)
            copy_device_to_host(host_array_ptr(a), dev[name], runtime=runtime)
            out[name] = a
        return out, timings
    finally:
        for buf in dev.values():
            free(buf, runtime=runtime)


@pytest.mark.parametrize(
    "rows,hidden,low_rank,inject",
    [(1, 2560, 320, True), (1, 2560, 320, False), (3, 2560, 320, True), (2, 512, 64, True)],
)
def test_decode_read_is_bit_identical_to_the_chain(rows, hidden, low_rank, inject):
    out, _ = run_read(rows, hidden, low_rank, inject=inject)
    np.testing.assert_array_equal(out["low_c"], out["low_p"])
    np.testing.assert_array_equal(out["inject_c"], out["inject_p"])
    np.testing.assert_array_equal(out["mixed_c"], out["mixed_p"])
    assert np.isfinite(out["mixed_c"].view(np.float32)).all()


if __name__ == "__main__":
    out, t = run_read(1, 2560, 320, iters=300)
    same = all(np.array_equal(out[a + "_c"], out[a + "_p"]) for a in ("low", "inject", "mixed"))
    print("identical", same, "chain %.1f us, three launches %.1f us (warm weights)" % tuple(t))
