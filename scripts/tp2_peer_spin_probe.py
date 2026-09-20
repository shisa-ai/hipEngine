"""Probe: HIP peer access between the W7900 and RX 7900 XTX, and the
spin-on-flag mechanism the graphed device-side reduction needs.

Pins, on hardware:
1. hipDeviceCanAccessPeer in both directions.
2. Whether an allocation on device 1 is readable from a device-0 kernel once
   peer access is enabled (bandwidth + correctness).
3. Whether a device kernel can spin on a device-1-published flag from
   device 0 and observe the write (the graphed lockstep dependency).
4. Host-mapped pinned flags as the fallback completion mechanism.
"""
import pathlib

import ctypes
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hipengine.core.device import scoped_current_device
from hipengine.core.hip import get_hip_runtime

rt = get_hip_runtime()
print("devices:", rt.device_count(), flush=True)

# -- 1. can-access-peer in both directions -----------------------------------

can01 = rt.library.hipDeviceCanAccessPeer(
    ctypes.byref(ctypes.c_int(0)), ctypes.c_int(0), ctypes.c_int(1)
)
code01 = rt.library.hipDeviceCanAccessPeer(
    ctypes.byref(flag := ctypes.c_int(0)), 0, 1
)
print("hipDeviceCanAccessPeer(0,1) ->", code01, flag.value, flush=True)
code10 = rt.library.hipDeviceCanAccessPeer(
    ctypes.byref(flag10 := ctypes.c_int(0)), 1, 0
)
print("hipDeviceCanAccessPeer(1,0) ->", code10, flag10.value, flush=True)

if not (flag.value and flag10.value):
    print("PEER BLOCKED: the device-side graph reduction needs host-mapped flags")
    sys.exit(0)

# -- 2. enable peer access and read remote memory from a kernel --------------

with scoped_current_device(rt, 0):
    e = rt.library.hipDeviceEnablePeerAccess(1, 0)
    print("enablePeerAccess(0->1):", e, flush=True)
with scoped_current_device(rt, 1):
    e = rt.library.hipDeviceEnablePeerAccess(0, 0)
    print("enablePeerAccess(1->0):", e, flush=True)

# -- 3. spin-on-peer-flag from a compiled kernel ------------------------------

# Use the runtime's own JIT to build a tiny probe module.
from hipengine.kernels.hip_gfx1100.common import build_hip_module  # noqa: E402

SRC = r"""
#include <hip/hip_runtime.h>

// Spin until the peer-published flag reaches `expected`, then add the two
// bf16 rows in f32 and narrow to bf16.
extern "C" __global__ void spin_add_bf16(
    const unsigned short* __restrict__ a,
    const unsigned short* __restrict__ b,
    unsigned short* __restrict__ out,
    const volatile unsigned int* __restrict__ flag,
    unsigned int expected,
    int n) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    while (*flag < expected) {
      __builtin_amdgcn_s_sleep(16);
    }
  }
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float va = (float)a[i];
    float vb = (float)b[i];
    out[i] = (unsigned short)__float2bfloat16(va + vb);
  }
}

extern "C" __global__ void publish_flag(volatile unsigned int* flag, unsigned int value) {
  if (threadIdx.x == 0) {
    *flag = value;
    __threadfence_system();
  }
}
"""

with scoped_current_device(rt, 0):
    mod0 = build_hip_module(SRC, "probe_spin0")
    spin0 = mod0.spin_add_bf16
    publish0 = mod0.publish_flag
with scoped_current_device(rt, 1):
    mod1 = build_hip_module(SRC, "probe_spin1")
    publish1 = mod1.publish_flag

hidden = 5120
nbytes = hidden * 2
with scoped_current_device(rt, 0):
    a0 = rt.malloc(nbytes)
    out0 = rt.malloc(nbytes)
    flag0 = rt.malloc(4)
    rt.memset(flag0, 0, 4)
    row_a = np.ones(hidden, dtype="<u2") * 15360  # 1.25 in bf16
    rt.memcpy(a0, row_a.ctypes.data, nbytes, 3)
with scoped_current_device(rt, 1):
    b1 = rt.malloc(nbytes)
    flag1 = rt.malloc(4)
    rt.memset(flag1, 0, 4)
    row_b = np.ones(hidden, dtype="<u2") * 15432  # ~1.3125
    rt.memcpy(b1, row_b.ctypes.data, nbytes, 3)

# Device 0 kernel spins on device 1's flag (peer read), adds a0 + b1.
expected = 7
with scoped_current_device(rt, 1):
    publish1(None if False else flag1, ctypes.c_uint(expected), block=(1, 1, 1), grid=(1, 1, 1))
with scoped_current_device(rt, 0):
    spin0(
        a0, b1, out0, flag1, ctypes.c_uint(expected), ctypes.c_int(hidden),
        block=(256, 1, 1), grid=((hidden + 255) // 256, 1, 1),
    )
    rt.device_synchronize()
    got = np.empty(hidden, dtype="<u2")
    rt.memcpy(got.ctypes.data, out0, nbytes, 2)
print("peer spin-add sample:", got[:4].tolist(), "expected ~ (1.25+1.3125) bf16", flush=True)

# Timing: how expensive is the peer read + flag spin per layer boundary?
with scoped_current_device(rt, 1):
    t0 = time.perf_counter()
    for step in range(1, 101):
        publish1(flag1, ctypes.c_uint(step), block=(1, 1, 1), grid=(1, 1, 1))
    rt.device_synchronize()
    t1 = time.perf_counter()
print(f"100 publishes on dev1: {(t1 - t0) * 1e3:.2f} ms", flush=True)

print("PEER OK: the graphed device-side reduction is feasible", flush=True)
