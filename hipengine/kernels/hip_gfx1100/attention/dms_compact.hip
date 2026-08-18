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

// U3: append/decode. Strict parent walk of
// DMSCompactBackend.append_decode: keep = ~evict | p - pos <= window,
// recomputed over the whole live extent, then the new row is appended.
// The parent's overflow victim loop can never actually drop a row (every
// evictable+out-of-window row has already failed `keep`), so when the
// combined length exceeds capacity the parent raises MemoryError without
// mutating state — the kernel mirrors that with status=1 and no writes.
//
// Under the admission steady state (no evicted out-of-window rows in the
// extent, decode advances one position per step) the recompute reduces to
// dropping at most the single row at position p - window - 1 when it is
// evicted (the same incremental invariant FastDMS uses). The full
// recompute is implemented directly so the kernel stays bit-exact against
// the parent for any legal input state, including corrupted steady states
// with several expiring rows.
//
// One block per (row, head). The extent (<= window + 2 rows) can exceed one
// 256-slot tile, so keep flags are scanned in 256-slot chunks: pass 1
// counts keepers per chunk (Hillis-Steele prefix over per-slot flags)
// before any write so overflow fails closed; pass 2 repeats the scan and
// compacts keepers in the same strided order; the new row lands at
// base + keepers. Within a chunk the exclusive rank of a kept slot is
// chunk_rank + inclusive_prefix - 1 <= its slot index, so writes are
// self or backward and never clobber a slot another thread still reads.
__global__ void dms_append_decode_kernel(
    const uint16_t* __restrict__ k_new,
    const uint16_t* __restrict__ v_new,
    const uint8_t* __restrict__ evict_new,
    const int32_t* __restrict__ row_positions,
    const int32_t* __restrict__ base_offsets,
    const int32_t* __restrict__ range_capacity,
    int32_t* __restrict__ live_counts,
    uint16_t* __restrict__ k_slot,
    uint16_t* __restrict__ v_slot,
    int32_t* __restrict__ token_positions,
    uint8_t* __restrict__ slot_evict,
    int32_t* __restrict__ status,
    const int32_t rows,
    const int32_t heads,
    const int32_t dim,
    const int32_t window_size) {
  const int32_t gid = static_cast<int32_t>(blockIdx.x);
  const int32_t r = gid / heads;
  const int32_t h = gid - r * heads;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int64_t rowhead = static_cast<int64_t>(r) * heads + h;
  const int32_t live = live_counts[rowhead];
  const int32_t base = base_offsets[rowhead];
  const int32_t cap = range_capacity[rowhead];
  const int32_t p = row_positions[r];

  __shared__ int32_t s_flag[kDmsPackThreads];
  __shared__ int32_t s_chunk_total;

  // Pass 1: count keepers (read-only; overflow must fail closed).
  int32_t keepers = 0;
  for (int32_t chunk0 = 0; chunk0 < live; chunk0 += kDmsPackThreads) {
    const int32_t in_chunk = tid < (live - chunk0);
    s_flag[tid] = 0;
    if (in_chunk) {
      const int32_t pos = token_positions[base + chunk0 + tid];
      const uint8_t e = slot_evict[base + chunk0 + tid];
      if (e == 0 || p - pos <= window_size) {
        s_flag[tid] = 1;
      }
    }
    __syncthreads();
    for (int32_t off = 1; off < kDmsPackThreads; off <<= 1) {
      const int32_t delta = (tid >= off) ? s_flag[tid - off] : 0;
      __syncthreads();
      s_flag[tid] += delta;
      __syncthreads();
    }
    s_chunk_total = s_flag[kDmsPackThreads - 1];
    __syncthreads();
    keepers += s_chunk_total;
  }
  const int32_t final_live = keepers + 1;
  if (final_live > cap) {
    // Host parent raises MemoryError without mutating state.
    status[rowhead] = 1;
    return;
  }

  // Pass 2: compact keepers in the same order, then append the new row.
  int32_t chunk_rank = 0;
  for (int32_t chunk0 = 0; chunk0 < live; chunk0 += kDmsPackThreads) {
    const int32_t in_chunk = tid < (live - chunk0);
    s_flag[tid] = 0;
    if (in_chunk) {
      const int32_t pos = token_positions[base + chunk0 + tid];
      const uint8_t e = slot_evict[base + chunk0 + tid];
      if (e == 0 || p - pos <= window_size) {
        s_flag[tid] = 1;
      }
    }
    __syncthreads();
    for (int32_t off = 1; off < kDmsPackThreads; off <<= 1) {
      const int32_t delta = (tid >= off) ? s_flag[tid - off] : 0;
      __syncthreads();
      s_flag[tid] += delta;
      __syncthreads();
    }
    if (in_chunk && s_flag[tid] > (tid > 0 ? s_flag[tid - 1] : 0)) {
      // This slot is kept: exclusive rank = chunk_rank + inclusive - 1.
      const int32_t dst = base + chunk_rank + s_flag[tid] - 1;
      const int32_t src = base + chunk0 + tid;
      if (dst != src) {
        const uint16_t* sk = k_slot + static_cast<int64_t>(src) * dim;
        const uint16_t* sv = v_slot + static_cast<int64_t>(src) * dim;
        uint16_t* dk = k_slot + static_cast<int64_t>(dst) * dim;
        uint16_t* dv = v_slot + static_cast<int64_t>(dst) * dim;
        for (int32_t d = 0; d < dim; d++) {
          dk[d] = sk[d];
          dv[d] = sv[d];
        }
        token_positions[dst] = token_positions[src];
        slot_evict[dst] = slot_evict[src];
      }
    }
    s_chunk_total = s_flag[kDmsPackThreads - 1];
    __syncthreads();
    chunk_rank += s_chunk_total;
    __syncthreads();
  }
  const int32_t append_slot = base + keepers;
  const uint16_t* kn = k_new + static_cast<int64_t>(h) * dim;
  const uint16_t* vn = v_new + static_cast<int64_t>(h) * dim;
  uint16_t* dk = k_slot + static_cast<int64_t>(append_slot) * dim;
  uint16_t* dv = v_slot + static_cast<int64_t>(append_slot) * dim;
  for (int32_t d = 0; d < dim; d++) {
    dk[d] = kn[d];
    dv[d] = vn[d];
  }
  token_positions[append_slot] = p;
  slot_evict[append_slot] = evict_new[h];
  live_counts[rowhead] = final_live;
  status[rowhead] = 0;
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

extern "C" int hipengine_dms_append_decode_bf16(
    const uint16_t* k_new,
    const uint16_t* v_new,
    const uint8_t* evict_new,
    const int32_t* row_positions,
    const int32_t* base_offsets,
    const int32_t* range_capacity,
    int32_t* live_counts,
    uint16_t* k_slot,
    uint16_t* v_slot,
    int32_t* token_positions,
    uint8_t* slot_evict,
    int32_t* status,
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
      dms_append_decode_kernel, dim3(blocks), dim3(kDmsPackThreads), 0, stream,
      k_new,
      v_new,
      evict_new,
      row_positions,
      base_offsets,
      range_capacity,
      live_counts,
      k_slot,
      v_slot,
      token_positions,
      slot_evict,
      status,
      rows,
      heads,
      dim,
      window_size);
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
