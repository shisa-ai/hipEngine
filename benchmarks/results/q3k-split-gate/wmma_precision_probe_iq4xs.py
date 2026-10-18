"""Isolate the W4A16 route's intrinsic noise beyond the bf16 output floor.

Three-way comparison on a real IQ4_XS tensor at 512 rows:
  - strict GEMV output (bf16)          : the reference owner
  - W4A16 WMMA output (bf16)           : the candidate route
  - exact f64 dot rounded to bf16      : the ground truth
ULP distances in bf16 space tell whether the WMMA path adds noise beyond
operand rounding (fp16 weights, 1.3e-4) + output rounding (2e-3 ULP).
"""
import sys
sys.path.insert(0, '/home/lhl/hipEngine-ud')
import numpy as np
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (copy_device_to_host, copy_host_to_device,
                                   free, host_array_ptr, malloc)
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import build_gguf_iq_dense, launch as gemv
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_wmma_prefill as w4a16
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf import dequantize_gguf_data, GGMLQuantizationType

reader = GGUFReader('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
t = next(x for x in reader.info.tensors
         if x.ggml_type_name == 'IQ4_XS' and x.name == 'blk.0.ffn_gate.weight')
n, k = int(t.shape[0]), int(t.shape[1])
raw = reader.tensor_data(t.name)
w_exact = dequantize_gguf_data(raw, GGMLQuantizationType.IQ4_XS).astype(np.float64)

rng = np.random.default_rng(11)
x32 = rng.normal(0, 0.1, (512, k)).astype(np.float32)
bf16 = lambda a: ((a.view(np.uint32) + 0x7fff + ((a.view(np.uint32) >> 16) & 1)) >> 16).astype(np.uint16)
x_b16 = bf16(x32)
x_f64 = (x_b16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

# exact ground truth: f64 dot of bf16 activations with exact dequantized weights
exact = x_f64 @ w_exact.T          # (512, n) f64
ex_b16 = bf16(exact.astype(np.float32))

# the fp16-operand simulation: weights rounded to fp16, exact f64 dot
w_f16 = w_exact.astype(np.float16).astype(np.float64)
sim_f16w = x_f64 @ w_f16.T
sim_b16 = bf16(sim_f16w.astype(np.float32))

hip = get_hip_runtime()
lib = build_gguf_iq_dense(compiler_version=open('/tmp/ud-hipcc-version.txt').read())
wlib = w4a16.build_gguf_iq_wmma_prefill(load=True, compiler_version=open('/tmp/ud-hipcc-version.txt').read())
strict_out = np.zeros((512, n), dtype=np.uint16)
w4a16_out = np.zeros((512, n), dtype=np.uint16)
bufs = []
try:
    raw_b = np.frombuffer(raw, dtype=np.uint8)
    w_b = malloc(raw_b.nbytes); bufs.append(w_b); copy_host_to_device(w_b, host_array_ptr(raw_b), raw_b.nbytes)
    x_b = malloc(x_b16.nbytes); bufs.append(x_b); copy_host_to_device(x_b, host_array_ptr(x_b16), x_b16.nbytes)
    s_b = malloc(strict_out.nbytes); bufs.append(s_b)
    c_b = malloc(w4a16_out.nbytes); bufs.append(c_b)
    gemv(x_b.ptr, w_b.ptr, s_b.ptr, 512, k, n, quant='gguf_iq4_xs', output='bf16', library=lib)
    w4a16.launch(x_b.ptr, w_b.ptr, c_b.ptr, 512, k, n, quant='gguf_iq4_xs', library=wlib)
    hip.device_synchronize()
    copy_device_to_host(host_array_ptr(strict_out), s_b, strict_out.nbytes)
    copy_device_to_host(host_array_ptr(w4a16_out), c_b, w4a16_out.nbytes)
finally:
    for b in reversed(bufs): free(b)

def ulps(a, b):
    """bf16 ULP distance between two uint16-coded arrays (same sign region)."""
    ai = a.astype(np.int32); bi = b.astype(np.int32)
    return np.abs(ai - bi)

ref = ex_b16
print("vs exact-f64-dot-rounded-to-bf16 (bf16 ULP distances):")
for name, out in (("strict GEMV", strict_out), ("W4A16 WMMA", w4a16_out),
                  ("fp16-weight sim (f64 dot)", sim_b16)):
    u = ulps(out, ref)
    frac = lambda d: float((u >= d).mean())
    print(f"  {name:<26} mean {u.mean():7.4f}  ==0 {frac(1):.4f}  >=1 {frac(1):.4f}  >=2 {frac(2):.4f}  >=4 {frac(4):.4f}  >=8 {frac(8):.4f}")
u = ulps(w4a16_out, strict_out)
print(f"  {'W4A16 vs strict':<26} mean {u.mean():7.4f}  >=1 {float((u>=1).mean()):.4f}  >=2 {float((u>=2).mean()):.4f}")
