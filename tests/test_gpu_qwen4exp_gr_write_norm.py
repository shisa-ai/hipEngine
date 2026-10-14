"""The fused hyper-connection write + norm + inject keeps every output bit.

``qwen4_exp_gr_write_norm_inject`` must equal the residual write
(``qwen4_exp_gr_write_bf16_f32`` / ``..._bf16_bf16``) followed by
``qwen4_exp_grouped_rmsnorm_bf16_f16_scale_inject`` on the written residual:
the residual, the F16 normalized plane, ``inv_rms`` and the inject logits.
"""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    qwen4_exp_gr_write_bf16_bf16,
    qwen4_exp_gr_write_bf16_f32,
    qwen4_exp_gr_write_norm_inject,
    qwen4_exp_grouped_rmsnorm_bf16_f16_scale_inject,
)


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


@pytest.mark.parametrize("block_bf16", [False, True])
@pytest.mark.parametrize("rows,hidden", [(37, 2560), (3, 512)])
def test_fused_write_norm_inject_matches_two_launches(block_bf16, rows, hidden):
    runtime = get_hip_runtime()
    rng = np.random.default_rng(rows + hidden + int(block_bf16))
    width = 4 * hidden
    host = {
        "residual": _bf16(rng.normal(0, 2.0, (rows, width))),
        "block": (_bf16(rng.normal(0, 1.0, (rows, hidden))) if block_bf16
                  else rng.normal(0, 1.0, (rows, hidden)).astype(np.float32)),
        "logits": rng.normal(0, 3.0, (rows, 4)).astype(np.float32),
        "weight": rng.normal(1.0, 0.2, width).astype(np.float32),
        "inject_weight": rng.normal(0, 0.02, (4, width)).astype(np.float32),
    }
    sizes = {"res_p": rows * width * 2, "res_f": rows * width * 2,
             "half_p": rows * width * 2, "half_f": rows * width * 2,
             "inv_p": rows * 16, "inv_f": rows * 16,
             "inj_p": rows * 16, "inj_f": rows * 16}
    dev = {name: malloc(a.nbytes, runtime=runtime) for name, a in host.items()}
    dev.update({name: malloc(n, runtime=runtime) for name, n in sizes.items()})
    try:
        for name, a in host.items():
            copy_host_to_device(dev[name], host_array_ptr(a), runtime=runtime)
        write = qwen4_exp_gr_write_bf16_bf16 if block_bf16 else qwen4_exp_gr_write_bf16_f32
        write(dev["residual"].ptr, dev["block"].ptr, dev["logits"].ptr,
              dev["res_p"].ptr, rows, 4, hidden)
        qwen4_exp_grouped_rmsnorm_bf16_f16_scale_inject(
            dev["res_p"].ptr, dev["weight"].ptr, dev["inject_weight"].ptr,
            dev["half_p"].ptr, dev["inv_p"].ptr, dev["inj_p"].ptr, rows, 4, hidden)
        qwen4_exp_gr_write_norm_inject(
            dev["residual"].ptr, dev["block"].ptr, dev["logits"].ptr,
            dev["res_f"].ptr, dev["weight"].ptr, dev["inject_weight"].ptr,
            dev["half_f"].ptr, dev["inv_f"].ptr, dev["inj_f"].ptr, rows, 4, hidden,
            block_bf16=block_bf16)
        runtime.device_synchronize()
        out = {}
        for name, n in sizes.items():
            buf = np.empty(n, np.uint8)
            copy_device_to_host(host_array_ptr(buf), dev[name], runtime=runtime)
            out[name] = buf
    finally:
        for buf in dev.values():
            free(buf, runtime=runtime)
    for name in ("res", "half", "inv", "inj"):
        assert np.array_equal(out[name + "_p"], out[name + "_f"]), name
