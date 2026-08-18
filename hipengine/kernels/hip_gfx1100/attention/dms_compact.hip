// Compact DMS streaming pack / compact decode kernel family (C2-7 device port).
//
// Reference: FastDMS c602b0e (fastdms/engine/dms.py, models/qwen3.py) and the
// hipEngine host composition (hipengine/kvcache/dms.py). The borrowed-neuron
// DMS convention stores one scalar eviction decision in the last channel of
// the first query head of each GQA group; extraction runs on pre-RoPE Q and
// zeroes the channel before attention. K/V pack (U2) runs on post-RoPE K/V.
//
// U1: decision extraction. The CPU reference compares the BF16 decision value
// against NumPy float64 scalars (float32 array upcast), so the device side
// also compares in double for bit-exact parity.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float dms_bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__global__ void dms_extract_decision_kernel(
    uint16_t* __restrict__ q,
    uint8_t* __restrict__ evict,
    const double threshold,
    const double alpha_scale,
    const double alpha_offset,
    const int64_t tokens,
    const int64_t q_heads,
    const int64_t kv_heads,
    const int64_t head_dim,
    const int positive_scale) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count = tokens * kv_heads;
  if (idx >= count) {
    return;
  }
  const int64_t token = idx / kv_heads;
  const int64_t kv_head = idx - token * kv_heads;
  const int64_t q_head = kv_head * (q_heads / kv_heads);
  const int64_t channel =
      (token * q_heads + q_head) * head_dim + (head_dim - 1);
  const float decision = dms_bf16_bits_to_float(q[channel]);
  uint8_t keep_evict;
  if (positive_scale) {
    keep_evict = (static_cast<double>(decision) > threshold) ? 1 : 0;
  } else {
    keep_evict =
        (static_cast<double>(decision) * alpha_scale - alpha_offset > 0.0) ? 1 : 0;
  }
  evict[idx] = keep_evict;
  q[channel] = 0;
}

}  // namespace

extern "C" int hipengine_dms_extract_decision_bf16(
    uint16_t* q,
    uint8_t* evict,
    double alpha_scale,
    double alpha_offset,
    int64_t tokens,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    hipStream_t stream) {
  if (tokens <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (q_heads % kv_heads != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (alpha_scale == 0.0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int positive_scale = alpha_scale > 0.0 ? 1 : 0;
  const double threshold = alpha_offset / alpha_scale;
  const int64_t count = tokens * kv_heads;
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  hipLaunchKernelGGL(
      dms_extract_decision_kernel, dim3(blocks), dim3(threads), 0, stream,
      q,
      evict,
      threshold,
      alpha_scale,
      alpha_offset,
      tokens,
      q_heads,
      kv_heads,
      head_dim,
      positive_scale);
  return static_cast<int>(hipGetLastError());
}
