// Compact DMS streaming pack / compact decode kernel family (C2-7 device port).
//
// Reference: FastDMS c602b0e (fastdms/engine/dms.py, models/qwen3.py) and the
// hipEngine host composition (hipengine/kvcache/dms.py). The borrowed-neuron
// DMS convention stores one scalar eviction decision in the last channel of
// the first query head of each GQA group; extraction runs on pre-RoPE Q and
// zeroes the channel before attention. K/V pack (U2) runs on post-RoPE K/V.

#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int kDmsPackThreads = 256;

namespace {

__device__ inline float dms_bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

// U1: decision extraction. The CPU reference compares the BF16 decision value
// against NumPy float64 scalars (float32 array upcast), so the device side
// also compares in double for bit-exact parity.

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

// U2: streaming pack. Per (row, head) the surviving prompt tokens
// (~evict | inside the retention window) are scattered in token order into
// the reserved extent; inputs are never retained (no dense shadow). The
// extent capacity must cover the worst case (row token count) — admission
// reserves it (see DMSCompactBackend.estimate).

__global__ void dms_streaming_pack_kernel(
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v,
    const uint8_t* __restrict__ evict,
    const int32_t* __restrict__ base_offsets,
    const int32_t* __restrict__ range_capacity,
    int32_t* __restrict__ live_counts,
    const int32_t* __restrict__ row_starts,
    const int32_t* __restrict__ row_tokens,
    uint16_t* __restrict__ k_slot,
    uint16_t* __restrict__ v_slot,
    int32_t* __restrict__ token_positions,
    uint8_t* __restrict__ slot_evict,
    const int32_t rows,
    const int32_t heads,
    const int32_t dim,
    const int32_t window_size) {
  const int32_t gid = static_cast<int32_t>(blockIdx.x);
  const int32_t r = gid / heads;
  const int32_t h = gid - r * heads;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t tokens = row_tokens[r];

  __shared__ int32_t s_token_base;
  __shared__ int32_t s_count[kDmsPackThreads];
  __shared__ int32_t s_prefix[kDmsPackThreads];

  if (tid == 0) {
    int32_t base = 0;
    for (int32_t i = 0; i < r; i++) {
      base += row_tokens[i];
    }
    s_token_base = base;
  }
  __syncthreads();
  const int32_t token_base = s_token_base;
  const int32_t current = tokens - 1;
  const int64_t ev_row = static_cast<int64_t>(token_base) * heads + h;

  int32_t my_count = 0;
  if (tokens > 0) {
    for (int32_t t = tid; t < tokens; t += kDmsPackThreads) {
      const int32_t e = evict[ev_row + static_cast<int64_t>(t) * heads];
      if (e == 0 || current - t <= window_size) {
        my_count++;
      }
    }
  }
  s_count[tid] = my_count;
  __syncthreads();
  if (tid == 0) {
    int32_t acc = 0;
    for (int32_t i = 0; i < kDmsPackThreads; i++) {
      s_prefix[i] = acc;
      acc += s_count[i];
    }
    live_counts[static_cast<int64_t>(r) * heads + h] = acc;
  }
  __syncthreads();

  int32_t rank = s_prefix[tid];
  const int32_t base = base_offsets[static_cast<int64_t>(r) * heads + h];
  const int32_t capacity = range_capacity[static_cast<int64_t>(r) * heads + h];
  const uint16_t* k_src =
      k + (static_cast<int64_t>(token_base) * heads + h) * dim;
  const uint16_t* v_src =
      v + (static_cast<int64_t>(token_base) * heads + h) * dim;
  if (tokens > 0) {
    for (int32_t t = tid; t < tokens; t += kDmsPackThreads) {
      const int32_t e = evict[ev_row + static_cast<int64_t>(t) * heads];
      if (e != 0 && current - t > window_size) {
        continue;
      }
      const int32_t dest = base + rank;
      if (dest - base >= capacity) {
        return;  // admission invariant violated; fail closed, no OOB write
      }
      rank++;
      const uint16_t* ks = k_src + static_cast<int64_t>(t) * heads * dim;
      const uint16_t* vs = v_src + static_cast<int64_t>(t) * heads * dim;
      uint16_t* kd = k_slot + static_cast<int64_t>(dest) * dim;
      uint16_t* vd = v_slot + static_cast<int64_t>(dest) * dim;
      for (int32_t d = 0; d < dim; d++) {
        kd[d] = ks[d];
        vd[d] = vs[d];
      }
      token_positions[dest] = row_starts[r] + t;
      slot_evict[dest] = static_cast<uint8_t>(e);
    }
  }
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

extern "C" int hipengine_dms_streaming_pack_bf16(
    const uint16_t* k,
    const uint16_t* v,
    const uint8_t* evict,
    const int32_t* base_offsets,
    const int32_t* range_capacity,
    int32_t* live_counts,
    const int32_t* row_starts,
    const int32_t* row_tokens,
    uint16_t* k_slot,
    uint16_t* v_slot,
    int32_t* token_positions,
    uint8_t* slot_evict,
    int32_t rows,
    int32_t heads,
    int32_t dim,
    int32_t window_size,
    hipStream_t stream) {
  if (rows <= 0 || heads <= 0 || dim <= 0 || window_size < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int blocks = rows * heads;
  hipLaunchKernelGGL(
      dms_streaming_pack_kernel, dim3(blocks), dim3(kDmsPackThreads), 0, stream,
      k,
      v,
      evict,
      base_offsets,
      range_capacity,
      live_counts,
      row_starts,
      row_tokens,
      k_slot,
      v_slot,
      token_positions,
      slot_evict,
      rows,
      heads,
      dim,
      window_size);
  return static_cast<int>(hipGetLastError());
}
