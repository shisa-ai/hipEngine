"""The hyper-connection read without the FP32 norm plane keeps every output bit."""
import ctypes
import time

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    qwen4_exp_gr_inject_bf16_f32,
    qwen4_exp_grouped_rmsnorm_bf16_f16_scale,
    qwen4_exp_grouped_rmsnorm_bf16_f32_f16,
)
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import dense_gemv_out_f32
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw
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


def run_chains(rows, k, hidden, *, iters=0):
    runtime = get_hip_runtime()
    rng = np.random.default_rng(rows * 3 + hidden)
    width = 4 * hidden
    residual = _bf16(rng.normal(0, 2.0, (rows, width)))
    norm_weight = rng.normal(1.0, 0.2, width).astype(np.float32)
    inject_weight = rng.normal(0, 0.02, (4, width)).astype(np.float32)
    x = rng.normal(0, 0.3, (rows, k)).astype(np.float16)
    up = make_q8_0_weight(width, k)
    host = {"residual": residual, "norm_weight": norm_weight,
            "inject_weight": inject_weight, "x": x, "up": up}
    dev = {name: malloc(a.nbytes, runtime=runtime) for name, a in host.items()}
    out_sizes = {"image": rows * hidden * 2, "image_ref": rows * hidden * 2,
                 "plane": rows * width * 4, "half_p": rows * width * 2,
                 "half_c": rows * width * 2, "inv": rows * 4 * 4,
                 "mixed_p": rows * hidden * 4, "mixed_c": rows * hidden * 4,
                 "inject_p": rows * 4 * 4, "inject_c": rows * 4 * 4}
    dev.update({name: malloc(n, runtime=runtime) for name, n in out_sizes.items()})
    try:
        for name, a in host.items():
            copy_host_to_device(dev[name], host_array_ptr(a), runtime=runtime)

        def parent():
            qwen4_exp_grouped_rmsnorm_bf16_f32_f16(
                dev["residual"].ptr, dev["norm_weight"].ptr, dev["plane"].ptr,
                dev["half_p"].ptr, rows, 4, hidden, runtime=runtime)
            dw.qwen4_exp_gr_up_mean_f16in_f32_out(
                dev["x"].ptr, dev["up"].ptr, dev["plane"].ptr, dev["mixed_p"].ptr,
                rows, k, 4, hidden, runtime=runtime)
            dense_gemv_out_f32(
                dev["plane"].ptr, dev["inject_weight"].ptr, dev["inject_p"].ptr,
                rows, width, 4, runtime=runtime)

        def candidate():
            qwen4_exp_grouped_rmsnorm_bf16_f16_scale(
                dev["residual"].ptr, dev["norm_weight"].ptr, dev["half_c"].ptr,
                dev["inv"].ptr, rows, 4, hidden, runtime=runtime)
            dw.qwen4_exp_gr_up_mean_residual_f16in_f32_out(
                dev["x"].ptr, dev["up"].ptr, dev["residual"].ptr, dev["inv"].ptr,
                dev["norm_weight"].ptr, dev["mixed_c"].ptr, rows, k, 4, hidden,
                out_f16_ptr=dev["image"].ptr, runtime=runtime)
            qwen4_exp_gr_inject_bf16_f32(
                dev["residual"].ptr, dev["norm_weight"].ptr, dev["inv"].ptr,
                dev["inject_weight"].ptr, dev["inject_c"].ptr, rows, 4, hidden,
                runtime=runtime)

        timings = []
        for fn in (parent, candidate):
            fn()
            runtime.device_synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                fn()
            runtime.device_synchronize()
            timings.append((time.perf_counter() - start) / max(iters, 1) * 1e3)
        dw.f32_to_f16(dev["mixed_c"].ptr, dev["image_ref"].ptr, rows * hidden,
                      runtime=runtime)
        runtime.device_synchronize()
        result = {}
        for name, dtype in (("image", np.uint16), ("image_ref", np.uint16),
                            ("half_p", np.uint16), ("half_c", np.uint16),
                            ("mixed_p", np.uint32), ("mixed_c", np.uint32),
                            ("inject_p", np.uint32), ("inject_c", np.uint32)):
            a = np.empty(out_sizes[name] // np.dtype(dtype).itemsize, dtype)
            copy_device_to_host(host_array_ptr(a), dev[name], runtime=runtime)
            result[name] = a
        return result, timings
    finally:
        for buf in dev.values():
            free(buf, runtime=runtime)


@pytest.mark.parametrize("rows,k,hidden", [(512, 320, 2560), (300, 64, 512)])
def test_read_chain_without_the_f32_plane_is_bit_identical(rows, k, hidden):
    result, _ = run_chains(rows, k, hidden)
    np.testing.assert_array_equal(result["half_c"], result["half_p"])
    np.testing.assert_array_equal(result["mixed_c"], result["mixed_p"])
    np.testing.assert_array_equal(result["inject_c"], result["inject_p"])
    # The F16 image is the standalone cast of the FP32 output it mirrors.
    np.testing.assert_array_equal(result["image"], result["image_ref"])
    assert np.isfinite(result["mixed_c"].view(np.float32)).all()


if __name__ == "__main__":
    r, t = run_chains(4096, 320, 2560, iters=20)
    print("identical", all(np.array_equal(r[a + "_c"], r[a + "_p"])
                           for a in ("half", "mixed", "inject")),
          "parent %.3f ms candidate %.3f ms" % tuple(t))


@pytest.mark.parametrize("rows,hidden", [(512, 2560), (300, 512), (3, 2560)])
def test_norm_with_inject_matches_the_two_launches(rows, hidden):
    from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
        qwen4_exp_grouped_rmsnorm_bf16_f16_scale_inject,
    )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(rows + hidden)
    width = 4 * hidden
    residual = _bf16(rng.normal(0, 2.0, (rows, width)))
    norm_weight = rng.normal(1.0, 0.2, width).astype(np.float32)
    inject_weight = rng.normal(0, 0.02, (4, width)).astype(np.float32)
    dev = {name: malloc(a.nbytes, runtime=runtime) for name, a in (
        ("residual", residual), ("norm_weight", norm_weight),
        ("inject_weight", inject_weight))}
    sizes = {"half_p": rows * width * 2, "half_c": rows * width * 2,
             "inv_p": rows * 16, "inv_c": rows * 16,
             "inject_p": rows * 16, "inject_c": rows * 16}
    dev.update({name: malloc(n, runtime=runtime) for name, n in sizes.items()})
    try:
        for name, a in (("residual", residual), ("norm_weight", norm_weight),
                        ("inject_weight", inject_weight)):
            copy_host_to_device(dev[name], host_array_ptr(a), runtime=runtime)
        qwen4_exp_grouped_rmsnorm_bf16_f16_scale(
            dev["residual"].ptr, dev["norm_weight"].ptr, dev["half_p"].ptr,
            dev["inv_p"].ptr, rows, 4, hidden, runtime=runtime)
        qwen4_exp_gr_inject_bf16_f32(
            dev["residual"].ptr, dev["norm_weight"].ptr, dev["inv_p"].ptr,
            dev["inject_weight"].ptr, dev["inject_p"].ptr, rows, 4, hidden,
            runtime=runtime)
        qwen4_exp_grouped_rmsnorm_bf16_f16_scale_inject(
            dev["residual"].ptr, dev["norm_weight"].ptr, dev["inject_weight"].ptr,
            dev["half_c"].ptr, dev["inv_c"].ptr, dev["inject_c"].ptr, rows, 4,
            hidden, runtime=runtime)
        runtime.device_synchronize()
        got = {}
        for name, dtype in (("half_p", np.uint16), ("half_c", np.uint16),
                            ("inv_p", np.uint32), ("inv_c", np.uint32),
                            ("inject_p", np.uint32), ("inject_c", np.uint32)):
            a = np.empty(sizes[name] // np.dtype(dtype).itemsize, dtype)
            copy_device_to_host(host_array_ptr(a), dev[name], runtime=runtime)
            got[name] = a
    finally:
        for buf in dev.values():
            free(buf, runtime=runtime)
    np.testing.assert_array_equal(got["half_c"], got["half_p"])
    np.testing.assert_array_equal(got["inv_c"], got["inv_p"])
    np.testing.assert_array_equal(got["inject_c"], got["inject_p"])


@pytest.mark.parametrize("rows,hidden", [(1, 2560), (37, 512)])
def test_write_with_bf16_block_output_matches_widening_first(rows, hidden):
    from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
        qwen4_exp_gr_write_bf16_bf16,
        qwen4_exp_gr_write_bf16_f32,
    )

    runtime = get_hip_runtime()
    rng = np.random.default_rng(rows * 5 + hidden)
    residual = _bf16(rng.normal(0, 2.0, (rows, 4 * hidden)))
    block = _bf16(rng.normal(0, 1.0, (rows, hidden)))
    widened = (block.astype(np.uint32) << 16).view(np.float32)
    inject = rng.normal(0, 1.0, (rows, 4)).astype(np.float32)
    arrays = {"residual": residual, "block": block, "widened": widened, "inject": inject}
    dev = {name: malloc(a.nbytes, runtime=runtime) for name, a in arrays.items()}
    dev["out_p"] = malloc(residual.nbytes, runtime=runtime)
    dev["out_c"] = malloc(residual.nbytes, runtime=runtime)
    try:
        for name, a in arrays.items():
            copy_host_to_device(dev[name], host_array_ptr(a), runtime=runtime)
        qwen4_exp_gr_write_bf16_f32(dev["residual"].ptr, dev["widened"].ptr,
                                    dev["inject"].ptr, dev["out_p"].ptr, rows, 4,
                                    hidden, runtime=runtime)
        qwen4_exp_gr_write_bf16_bf16(dev["residual"].ptr, dev["block"].ptr,
                                     dev["inject"].ptr, dev["out_c"].ptr, rows, 4,
                                     hidden, runtime=runtime)
        runtime.device_synchronize()
        got = [np.empty_like(residual), np.empty_like(residual)]
        copy_device_to_host(host_array_ptr(got[0]), dev["out_p"], runtime=runtime)
        copy_device_to_host(host_array_ptr(got[1]), dev["out_c"], runtime=runtime)
    finally:
        for buf in dev.values():
            free(buf, runtime=runtime)
    np.testing.assert_array_equal(got[1], got[0])
