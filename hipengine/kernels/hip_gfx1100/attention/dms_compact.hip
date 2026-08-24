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

// Schema-v2 external decision source. One workgroup owns one (token, KV-head)
// dot product over the declared BF16 normalized hidden stage and resident BF16
// sidecar weights. The fixed reduction tree is deterministic; ordinary Q is not
// read or modified. FP32 logits are optional serving telemetry/calibration
// output, while the uint8 decision plane is consumed directly by pack/append.
__global__ void dms_external_linear_decision_kernel(
    const uint16_t* __restrict__ hidden,
    const uint16_t* __restrict__ weight,
    const uint16_t* __restrict__ bias,
    float* __restrict__ logits,
    uint8_t* __restrict__ evict,
    const double alpha_scale,
    const double alpha_offset,
    const int32_t tokens,
    const int32_t hidden_size,
    const int32_t kv_heads) {
  const int32_t gid = static_cast<int32_t>(blockIdx.x);
  const int32_t token = gid / kv_heads;
  const int32_t head = gid - token * kv_heads;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const uint16_t* x = hidden + static_cast<int64_t>(token) * hidden_size;
  const uint16_t* w = weight + static_cast<int64_t>(head) * hidden_size;
  float acc = 0.0f;
  for (int32_t index = tid; index < hidden_size; index += kDmsPackThreads) {
    acc = fmaf(
        dms_bf16_bits_to_float(x[index]),
        dms_bf16_bits_to_float(w[index]),
        acc);
  }
  __shared__ float partial[kDmsPackThreads];
  partial[tid] = acc;
  __syncthreads();
  for (int32_t stride = kDmsPackThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const float score = partial[0] + dms_bf16_bits_to_float(bias[head]);
    logits[static_cast<int64_t>(token) * kv_heads + head] = score;
    evict[static_cast<int64_t>(token) * kv_heads + head] =
        (static_cast<double>(score) * alpha_scale - alpha_offset > 0.0) ? 1 : 0;
  }
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
  __shared__ int32_t s_flag[kDmsPackThreads];
  __shared__ int32_t s_chunk_total;

  if (tid == 0) {
    int32_t tbase = 0;
    for (int32_t i = 0; i < r; i++) {
      tbase += row_tokens[i];
    }
    s_token_base = tbase;
  }
  __syncthreads();
  const int32_t token_base = s_token_base;
  const int32_t current = tokens - 1;
  const int64_t ev_row = static_cast<int64_t>(token_base) * heads + h;
  const int32_t base = base_offsets[static_cast<int64_t>(r) * heads + h];
  const int32_t capacity = range_capacity[static_cast<int64_t>(r) * heads + h];
  const uint16_t* k_src =
      k + (static_cast<int64_t>(token_base) * heads + h) * dim;
  const uint16_t* v_src =
      v + (static_cast<int64_t>(token_base) * heads + h) * dim;

  // Pass 1: count keepers in 256-token chunks (read-only). A lane-prefix
  // of per-thread counts is only a true global rank while the row fits one
  // tile, so the count is chunked the same way the writes are: the
  // inclusive prefix within a chunk plus a running cross-chunk rank is the
  // global output rank for any prompt length. Counting before any write
  // lets an over-capacity row fail closed without mutating the extent.
  int32_t total = 0;
  for (int32_t chunk0 = 0; chunk0 < tokens; chunk0 += kDmsPackThreads) {
    const int32_t in_chunk = tid < (tokens - chunk0);
    s_flag[tid] = 0;
    if (in_chunk) {
      const int32_t t = chunk0 + tid;
      const int32_t e = evict[ev_row + static_cast<int64_t>(t) * heads];
      if (e == 0 || current - t <= window_size) {
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
    total += s_chunk_total;
  }
  if (total > capacity) {
    // Admission invariant violated; fail closed, no extent mutation.
    return;
  }

  // Pass 2: scatter keepers in token order (input K/V and the slot extent
  // are disjoint, so the chunk-wise scan+write is race-free).
  int32_t chunk_rank = 0;
  for (int32_t chunk0 = 0; chunk0 < tokens; chunk0 += kDmsPackThreads) {
    const int32_t in_chunk = tid < (tokens - chunk0);
    s_flag[tid] = 0;
    if (in_chunk) {
      const int32_t t = chunk0 + tid;
      const int32_t e = evict[ev_row + static_cast<int64_t>(t) * heads];
      if (e == 0 || current - t <= window_size) {
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
      // This token is kept: exclusive rank = chunk_rank + inclusive - 1.
      const int32_t t = chunk0 + tid;
      const int32_t dest = base + chunk_rank + s_flag[tid] - 1;
      const int32_t e = evict[ev_row + static_cast<int64_t>(t) * heads];
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
    s_chunk_total = s_flag[kDmsPackThreads - 1];
    __syncthreads();
    chunk_rank += s_chunk_total;
    __syncthreads();
  }
  live_counts[static_cast<int64_t>(r) * heads + h] = total;
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

// U4: compact decode attention. GQA decode over the dense-in-extent DMS
// storage written by U2/U3: one workgroup per (row, q_head), K/V rows read
// from the slot-major extent at base_offsets + [0, live). Numerics mirror
// the paged decode kernel (qwen35_full_attn_decode_impl, ported from
// nano-vllm-amd@59195ed): fp32 score accumulation, tree max, expf softmax
// with a floored denominator, fp32 P@V output. live == 0 writes zeros
// (the host oracle rejects live=0, so zero is the safe neutral); live == 1
// is bit-exact against the reference (softmax of one element is exactly 1).
__global__ void dms_compact_attn_decode_kernel(
    const float* __restrict__ q,
    const uint16_t* __restrict__ k_slot,
    const uint16_t* __restrict__ v_slot,
    const int32_t* __restrict__ base_offsets,
    const int32_t* __restrict__ live_counts,
    float* __restrict__ out,
    const int32_t rows,
    const int32_t q_heads,
    const int32_t kv_heads,
    const int32_t dim,
    const float scale,
    const int32_t score_capacity) {
  const int32_t gid = static_cast<int32_t>(blockIdx.x);
  const int32_t r = gid / q_heads;
  const int32_t g = gid - r * q_heads;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t group = q_heads / kv_heads;
  const int32_t h = g / group;
  const int64_t rowhead = static_cast<int64_t>(r) * kv_heads + h;
  const int32_t live = live_counts[rowhead];
  const int32_t base = base_offsets[rowhead];
  const int64_t q_off = (static_cast<int64_t>(r) * q_heads + g) * dim;
  float* o = out + q_off;
  if (live <= 0) {
    for (int32_t d = tid; d < dim; d += kDmsPackThreads) {
      o[d] = 0.0f;
    }
    return;
  }
  const uint16_t* ks = k_slot + static_cast<int64_t>(base) * dim;
  const uint16_t* vs = v_slot + static_cast<int64_t>(base) * dim;

  extern __shared__ float shared[];
  float* scores = shared;
  float* partial = shared + score_capacity;

  float local_max = -3.4028234663852886e38f;
  for (int32_t t = tid; t < live; t += kDmsPackThreads) {
    float acc = 0.0f;
    const uint16_t* krow = ks + static_cast<int64_t>(t) * dim;
    for (int32_t d = 0; d < dim; ++d) {
      acc += q[q_off + d] * dms_bf16_bits_to_float(krow[d]);
    }
    const float score = acc * scale;
    scores[t] = score;
    local_max = fmaxf(local_max, score);
  }
  partial[tid] = local_max;
  __syncthreads();
  for (int32_t stride = kDmsPackThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = partial[0];

  float local_sum = 0.0f;
  for (int32_t t = tid; t < live; t += kDmsPackThreads) {
    local_sum += expf(scores[t] - max_score);
  }
  partial[tid] = local_sum;
  __syncthreads();
  for (int32_t stride = kDmsPackThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_denom = 1.0f / fmaxf(partial[0], 1.0e-20f);

  for (int32_t t = tid; t < live; t += kDmsPackThreads) {
    scores[t] = expf(scores[t] - max_score) * inv_denom;
  }
  __syncthreads();
  for (int32_t d = tid; d < dim; d += kDmsPackThreads) {
    float acc = 0.0f;
    for (int32_t t = 0; t < live; ++t) {
      acc += scores[t] * dms_bf16_bits_to_float(vs[static_cast<int64_t>(t) * dim + d]);
    }
    o[d] = acc;
  }
}

// Long-context compact decode uses bounded 256-row split producers plus one
// deterministic reducer. Unlike the fixture body above, LDS is constant in the
// live count: the generic producer uses two 256-float arrays (2 KiB), the
// gfx1151 group-6 producer uses bounded score/query/reduction slabs (12.5 KiB
// before allocation alignment), and the reducer uses one 256-float array
// (1 KiB).
constexpr int32_t kDmsMaxGqaGroup = 8;
#ifndef HIPENGINE_DMS_ENABLE_WAVE_GROUP6
#define HIPENGINE_DMS_ENABLE_WAVE_GROUP6 0
#endif

// Qwen3.8's 24Q/4KV/D256 path scores one compact token per wave. This mirrors
// the qualified dense GQA producer's K-side work distribution while retaining
// compact contiguous per-head extents and the existing split output ABI.
template <int32_t QPerKv>
__global__ void dms_compact_attn_splitk_group6_wave_producer_kernel(
    const float* __restrict__ q,
    const uint16_t* __restrict__ k_slot,
    const uint16_t* __restrict__ v_slot,
    const int32_t* __restrict__ base_offsets,
    const int32_t* __restrict__ live_counts,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    const int32_t rows,
    const int32_t q_heads,
    const int32_t kv_heads,
    const int32_t dim,
    const float scale,
    const int32_t chunk_size,
    const int32_t num_splits) {
  constexpr int32_t kHeadDim = 256;
  constexpr int32_t kWaveSize = 32;
  constexpr int32_t kNumWaves = kDmsPackThreads / kWaveSize;
  const int32_t gid = static_cast<int32_t>(blockIdx.x);
  const int32_t split = gid % num_splits;
  const int32_t rowh = gid / num_splits;
  const int32_t r = rowh / kv_heads;
  const int32_t h = rowh - r * kv_heads;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t lane = tid & (kWaveSize - 1);
  const int32_t wave = tid / kWaveSize;
  const int32_t rowq_base = r * q_heads + h * QPerKv;
  const int64_t rowhead = static_cast<int64_t>(r) * kv_heads + h;
  const int32_t live = live_counts[rowhead];
  const int32_t begin = split * chunk_size;
  const int32_t count = max(0, min(chunk_size, live - begin));
  const int32_t base = base_offsets[rowhead] + begin;
  __shared__ float scores[QPerKv][kDmsPackThreads];
  __shared__ float wave_partials[QPerKv][kNumWaves];
  __shared__ float block_max[QPerKv];
  __shared__ float q_shared[QPerKv][kHeadDim];

  if (count <= 0) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      const int64_t stat_index =
          static_cast<int64_t>(rowq_base + j) * num_splits + split;
      float* part = partial_out + stat_index * dim;
      for (int32_t d = tid; d < dim; d += kDmsPackThreads) {
        part[d] = 0.0f;
      }
      if (tid == 0) {
        partial_m[stat_index] = -3.4028234663852886e38f;
        partial_l[stat_index] = 0.0f;
      }
    }
    return;
  }

#pragma unroll
  for (int32_t j = 0; j < QPerKv; ++j) {
    q_shared[j][tid] = q[static_cast<int64_t>(rowq_base + j) * dim + tid];
  }
  __syncthreads();

  float local_max[QPerKv];
#pragma unroll
  for (int32_t j = 0; j < QPerKv; ++j) {
    local_max[j] = -3.4028234663852886e38f;
  }
  for (int32_t t = wave; t < count; t += kNumWaves) {
    const uint16_t* krow =
        k_slot + (static_cast<int64_t>(base) + t) * kHeadDim;
    float dots[QPerKv] = {};
    for (int32_t d = lane * 4; d < kHeadDim; d += kWaveSize * 4) {
      const float k0 = dms_bf16_bits_to_float(krow[d + 0]);
      const float k1 = dms_bf16_bits_to_float(krow[d + 1]);
      const float k2 = dms_bf16_bits_to_float(krow[d + 2]);
      const float k3 = dms_bf16_bits_to_float(krow[d + 3]);
#pragma unroll
      for (int32_t j = 0; j < QPerKv; ++j) {
        dots[j] += q_shared[j][d + 0] * k0;
        dots[j] += q_shared[j][d + 1] * k1;
        dots[j] += q_shared[j][d + 2] * k2;
        dots[j] += q_shared[j][d + 3] * k3;
      }
    }
    for (int32_t offset = kWaveSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
      for (int32_t j = 0; j < QPerKv; ++j) {
        dots[j] += __shfl_down(dots[j], offset);
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int32_t j = 0; j < QPerKv; ++j) {
        const float score = dots[j] * scale;
        scores[j][t] = score;
        local_max[j] = fmaxf(local_max[j], score);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      wave_partials[j][wave] = local_max[j];
    }
  }
  __syncthreads();

  if (tid < kWaveSize) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      float value = tid < kNumWaves
          ? wave_partials[j][tid]
          : -3.4028234663852886e38f;
      for (int32_t offset = kWaveSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down(value, offset));
      }
      if (tid == 0) {
        wave_partials[j][0] = value;
        block_max[j] = value;
      }
    }
  }
  __syncthreads();

  float local_sum[QPerKv] = {};
#pragma unroll
  for (int32_t j = 0; j < QPerKv; ++j) {
    const float maximum = block_max[j];
    for (int32_t t = tid; t < count; t += kDmsPackThreads) {
      const float value = expf(scores[j][t] - maximum);
      scores[j][t] = value;
      local_sum[j] += value;
    }
  }
  for (int32_t offset = kWaveSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      local_sum[j] += __shfl_down(local_sum[j], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      wave_partials[j][wave] = local_sum[j];
    }
  }
  __syncthreads();

  if (tid < kWaveSize) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      float value = tid < kNumWaves ? wave_partials[j][tid] : 0.0f;
      for (int32_t offset = kWaveSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset);
      }
      if (tid == 0) {
        wave_partials[j][0] = value;
      }
    }
  }
  __syncthreads();

  if (tid < kHeadDim) {
    float values[QPerKv] = {};
    const uint16_t* vs = v_slot + static_cast<int64_t>(base) * kHeadDim;
    for (int32_t t = 0; t < count; ++t) {
      const float value =
          dms_bf16_bits_to_float(vs[static_cast<int64_t>(t) * kHeadDim + tid]);
#pragma unroll
      for (int32_t j = 0; j < QPerKv; ++j) {
        values[j] += scores[j][t] * value;
      }
    }
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      const int64_t stat_index =
          static_cast<int64_t>(rowq_base + j) * num_splits + split;
      partial_out[stat_index * kHeadDim + tid] = values[j];
    }
  }
  if (tid == 0) {
#pragma unroll
    for (int32_t j = 0; j < QPerKv; ++j) {
      const int64_t stat_index =
          static_cast<int64_t>(rowq_base + j) * num_splits + split;
      partial_m[stat_index] = block_max[j];
      partial_l[stat_index] = wave_partials[j][0];
    }
  }
}

__global__ void dms_compact_attn_splitk_grouped_producer_kernel(
    const float* __restrict__ q,
    const uint16_t* __restrict__ k_slot,
    const uint16_t* __restrict__ v_slot,
    const int32_t* __restrict__ base_offsets,
    const int32_t* __restrict__ live_counts,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    const int32_t rows,
    const int32_t q_heads,
    const int32_t kv_heads,
    const int32_t dim,
    const float scale,
    const int32_t chunk_size,
    const int32_t num_splits) {
  const int32_t gid = static_cast<int32_t>(blockIdx.x);
  const int32_t split = gid % num_splits;
  const int32_t rowh = gid / num_splits;
  const int32_t r = rowh / kv_heads;
  const int32_t h = rowh - r * kv_heads;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t group = q_heads / kv_heads;
  const int32_t query_head_base = h * group;
  const int32_t rowq_base = r * q_heads + query_head_base;
  const int64_t rowhead = static_cast<int64_t>(r) * kv_heads + h;
  const int32_t live = live_counts[rowhead];
  const int32_t begin = split * chunk_size;
  const int32_t count = max(0, min(chunk_size, live - begin));
  const int32_t base = base_offsets[rowhead] + begin;
  __shared__ float probabilities[kDmsMaxGqaGroup][kDmsPackThreads];
  __shared__ float reduction[kDmsPackThreads];

  if (count <= 0) {
#pragma unroll
    for (int32_t j = 0; j < kDmsMaxGqaGroup; ++j) {
      if (j < group) {
        const int64_t stat_index =
            static_cast<int64_t>(rowq_base + j) * num_splits + split;
        float* part = partial_out + stat_index * dim;
        for (int32_t d = tid; d < dim; d += kDmsPackThreads) {
          part[d] = 0.0f;
        }
        if (tid == 0) {
          partial_m[stat_index] = -3.4028234663852886e38f;
          partial_l[stat_index] = 0.0f;
        }
      }
    }
    return;
  }

  float dots[kDmsMaxGqaGroup] = {};
  if (tid < count) {
    const uint16_t* krow =
        k_slot + (static_cast<int64_t>(base) + tid) * dim;
    for (int32_t d = 0; d < dim; ++d) {
      const float key = dms_bf16_bits_to_float(krow[d]);
#pragma unroll
      for (int32_t j = 0; j < kDmsMaxGqaGroup; ++j) {
        if (j < group) {
          dots[j] += q[static_cast<int64_t>(rowq_base + j) * dim + d] * key;
        }
      }
    }
  }

#pragma unroll
  for (int32_t j = 0; j < kDmsMaxGqaGroup; ++j) {
    if (j < group) {
      const float score =
          tid < count ? dots[j] * scale : -3.4028234663852886e38f;
      probabilities[j][tid] = score;
      reduction[tid] = score;
      __syncthreads();
      for (int32_t stride = kDmsPackThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
        }
        __syncthreads();
      }
      const float block_m = reduction[0];
      const float probability = tid < count ? expf(score - block_m) : 0.0f;
      probabilities[j][tid] = probability;
      reduction[tid] = probability;
      __syncthreads();
      for (int32_t stride = kDmsPackThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
      }
      if (tid == 0) {
        const int64_t stat_index =
            static_cast<int64_t>(rowq_base + j) * num_splits + split;
        partial_m[stat_index] = block_m;
        partial_l[stat_index] = reduction[0];
      }
      __syncthreads();
    }
  }

  const uint16_t* vs = v_slot + static_cast<int64_t>(base) * dim;
  for (int32_t d = tid; d < dim; d += kDmsPackThreads) {
    float values[kDmsMaxGqaGroup] = {};
    for (int32_t t = 0; t < count; ++t) {
      const float value =
          dms_bf16_bits_to_float(vs[static_cast<int64_t>(t) * dim + d]);
#pragma unroll
      for (int32_t j = 0; j < kDmsMaxGqaGroup; ++j) {
        if (j < group) {
          values[j] += probabilities[j][t] * value;
        }
      }
    }
#pragma unroll
    for (int32_t j = 0; j < kDmsMaxGqaGroup; ++j) {
      if (j < group) {
        const int64_t stat_index =
            static_cast<int64_t>(rowq_base + j) * num_splits + split;
        partial_out[stat_index * dim + d] = values[j];
      }
    }
  }
}

__global__ void dms_compact_attn_splitk_producer_kernel(
    const float* __restrict__ q,
    const uint16_t* __restrict__ k_slot,
    const uint16_t* __restrict__ v_slot,
    const int32_t* __restrict__ base_offsets,
    const int32_t* __restrict__ live_counts,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    const int32_t rows,
    const int32_t q_heads,
    const int32_t kv_heads,
    const int32_t dim,
    const float scale,
    const int32_t chunk_size,
    const int32_t num_splits) {
  const int32_t gid = static_cast<int32_t>(blockIdx.x);
  const int32_t split = gid % num_splits;
  const int32_t rowq = gid / num_splits;
  const int32_t r = rowq / q_heads;
  const int32_t g = rowq - r * q_heads;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t group = q_heads / kv_heads;
  const int32_t h = g / group;
  const int64_t rowhead = static_cast<int64_t>(r) * kv_heads + h;
  const int32_t live = live_counts[rowhead];
  const int32_t begin = split * chunk_size;
  const int32_t count = max(0, min(chunk_size, live - begin));
  const int64_t stat_index = static_cast<int64_t>(rowq) * num_splits + split;
  float* part = partial_out + stat_index * dim;
  if (count <= 0) {
    for (int32_t d = tid; d < dim; d += kDmsPackThreads) {
      part[d] = 0.0f;
    }
    if (tid == 0) {
      partial_m[stat_index] = -3.4028234663852886e38f;
      partial_l[stat_index] = 0.0f;
    }
    return;
  }

  const int32_t base = base_offsets[rowhead] + begin;
  const uint16_t* ks = k_slot + static_cast<int64_t>(base) * dim;
  const uint16_t* vs = v_slot + static_cast<int64_t>(base) * dim;
  const int64_t q_off = static_cast<int64_t>(rowq) * dim;
  __shared__ float scores[kDmsPackThreads];
  __shared__ float reduction[kDmsPackThreads];

  float score = -3.4028234663852886e38f;
  if (tid < count) {
    float dot = 0.0f;
    const uint16_t* krow = ks + static_cast<int64_t>(tid) * dim;
    for (int32_t d = 0; d < dim; ++d) {
      dot += q[q_off + d] * dms_bf16_bits_to_float(krow[d]);
    }
    score = dot * scale;
  }
  scores[tid] = score;
  reduction[tid] = score;
  __syncthreads();
  for (int32_t stride = kDmsPackThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  const float block_m = reduction[0];
  const float probability = tid < count ? expf(score - block_m) : 0.0f;
  scores[tid] = probability;
  reduction[tid] = probability;
  __syncthreads();
  for (int32_t stride = kDmsPackThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    partial_m[stat_index] = block_m;
    partial_l[stat_index] = reduction[0];
  }
  for (int32_t d = tid; d < dim; d += kDmsPackThreads) {
    float value = 0.0f;
    for (int32_t t = 0; t < count; ++t) {
      value += scores[t] *
          dms_bf16_bits_to_float(vs[static_cast<int64_t>(t) * dim + d]);
    }
    part[d] = value;
  }
}

__global__ void dms_compact_attn_splitk_reduce_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    float* __restrict__ out,
    const int32_t rows,
    const int32_t q_heads,
    const int32_t dim,
    const int32_t num_splits) {
  const int32_t rowq = static_cast<int32_t>(blockIdx.x);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int64_t stat_base = static_cast<int64_t>(rowq) * num_splits;
  __shared__ float reduction[kDmsPackThreads];
  float local_m = -3.4028234663852886e38f;
  for (int32_t split = tid; split < num_splits; split += kDmsPackThreads) {
    if (partial_l[stat_base + split] > 0.0f) {
      local_m = fmaxf(local_m, partial_m[stat_base + split]);
    }
  }
  reduction[tid] = local_m;
  __syncthreads();
  for (int32_t stride = kDmsPackThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  const float global_m = reduction[0];
  float local_l = 0.0f;
  for (int32_t split = tid; split < num_splits; split += kDmsPackThreads) {
    const float split_l = partial_l[stat_base + split];
    if (split_l > 0.0f) {
      local_l += split_l * expf(partial_m[stat_base + split] - global_m);
    }
  }
  reduction[tid] = local_l;
  __syncthreads();
  for (int32_t stride = kDmsPackThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  const float denom = reduction[0];
  float* destination = out + static_cast<int64_t>(rowq) * dim;
  for (int32_t d = tid; d < dim; d += kDmsPackThreads) {
    float value = 0.0f;
    if (denom > 0.0f) {
      for (int32_t split = 0; split < num_splits; ++split) {
        const float split_l = partial_l[stat_base + split];
        if (split_l > 0.0f) {
          const float factor = expf(partial_m[stat_base + split] - global_m);
          value += partial_out[(stat_base + split) * dim + d] * factor;
        }
      }
      value /= denom;
    }
    destination[d] = value;
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

extern "C" int hipengine_dms_external_linear_decision_bf16(
    const uint16_t* hidden,
    const uint16_t* weight,
    const uint16_t* bias,
    float* logits,
    uint8_t* evict,
    double alpha_scale,
    double alpha_offset,
    int32_t tokens,
    int32_t hidden_size,
    int32_t kv_heads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || kv_heads <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (alpha_scale == 0.0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int32_t blocks = tokens * kv_heads;
  hipLaunchKernelGGL(
      dms_external_linear_decision_kernel,
      dim3(blocks),
      dim3(kDmsPackThreads),
      0,
      stream,
      hidden,
      weight,
      bias,
      logits,
      evict,
      alpha_scale,
      alpha_offset,
      tokens,
      hidden_size,
      kv_heads);
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

extern "C" int hipengine_dms_compact_attn_decode_splitk_producer_bf16(
    const float* q,
    const uint16_t* k_slot,
    const uint16_t* v_slot,
    const int32_t* base_offsets,
    const int32_t* live_counts,
    float* partial_out,
    float* partial_m,
    float* partial_l,
    int32_t rows,
    int32_t q_heads,
    int32_t kv_heads,
    int32_t dim,
    float scale,
    int32_t chunk_size,
    int32_t num_splits,
    hipStream_t stream) {
  if (rows <= 0 || q_heads <= 0 || kv_heads <= 0 || dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (q_heads % kv_heads != 0 || chunk_size != kDmsPackThreads) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (!(scale > 0.0f) || num_splits <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int32_t group = q_heads / kv_heads;
  const bool grouped = group <= kDmsMaxGqaGroup;
  const bool wave_group6 = HIPENGINE_DMS_ENABLE_WAVE_GROUP6 != 0 &&
      group == 6 && q_heads == 24 && kv_heads == 4 && dim == 256;
  const int32_t blocks =
      rows * (grouped ? kv_heads : q_heads) * num_splits;
  if (wave_group6) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dms_compact_attn_splitk_group6_wave_producer_kernel<6>),
        dim3(blocks),
        dim3(kDmsPackThreads),
        0,
        stream,
        q,
        k_slot,
        v_slot,
        base_offsets,
        live_counts,
        partial_out,
        partial_m,
        partial_l,
        rows,
        q_heads,
        kv_heads,
        dim,
        scale,
        chunk_size,
        num_splits);
  } else if (grouped) {
    hipLaunchKernelGGL(
        dms_compact_attn_splitk_grouped_producer_kernel,
        dim3(blocks),
        dim3(kDmsPackThreads),
        0,
        stream,
        q,
        k_slot,
        v_slot,
        base_offsets,
        live_counts,
        partial_out,
        partial_m,
        partial_l,
        rows,
        q_heads,
        kv_heads,
        dim,
        scale,
        chunk_size,
        num_splits);
  } else {
    hipLaunchKernelGGL(
        dms_compact_attn_splitk_producer_kernel,
        dim3(blocks),
        dim3(kDmsPackThreads),
        0,
        stream,
        q,
        k_slot,
        v_slot,
        base_offsets,
        live_counts,
        partial_out,
        partial_m,
        partial_l,
        rows,
        q_heads,
        kv_heads,
        dim,
        scale,
        chunk_size,
        num_splits);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dms_compact_attn_decode_splitk_reduce_bf16(
    const float* partial_out,
    const float* partial_m,
    const float* partial_l,
    float* out,
    int32_t rows,
    int32_t q_heads,
    int32_t dim,
    int32_t num_splits,
    hipStream_t stream) {
  if (rows <= 0 || q_heads <= 0 || dim <= 0 || num_splits <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int32_t blocks = rows * q_heads;
  hipLaunchKernelGGL(
      dms_compact_attn_splitk_reduce_kernel,
      dim3(blocks),
      dim3(kDmsPackThreads),
      0,
      stream,
      partial_out,
      partial_m,
      partial_l,
      out,
      rows,
      q_heads,
      dim,
      num_splits);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dms_compact_attn_decode_splitk_bf16(
    const float* q,
    const uint16_t* k_slot,
    const uint16_t* v_slot,
    const int32_t* base_offsets,
    const int32_t* live_counts,
    float* partial_out,
    float* partial_m,
    float* partial_l,
    float* out,
    int32_t rows,
    int32_t q_heads,
    int32_t kv_heads,
    int32_t dim,
    float scale,
    int32_t chunk_size,
    int32_t num_splits,
    hipStream_t stream) {
  if (rows <= 0 || q_heads <= 0 || kv_heads <= 0 || dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (q_heads % kv_heads != 0 || chunk_size != kDmsPackThreads) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (!(scale > 0.0f) || num_splits <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int32_t group = q_heads / kv_heads;
  const bool grouped = group <= kDmsMaxGqaGroup;
  const bool wave_group6 = HIPENGINE_DMS_ENABLE_WAVE_GROUP6 != 0 &&
      group == 6 && q_heads == 24 && kv_heads == 4 && dim == 256;
  const int32_t producer_blocks =
      rows * (grouped ? kv_heads : q_heads) * num_splits;
  if (wave_group6) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(dms_compact_attn_splitk_group6_wave_producer_kernel<6>),
        dim3(producer_blocks),
        dim3(kDmsPackThreads),
        0,
        stream,
        q,
        k_slot,
        v_slot,
        base_offsets,
        live_counts,
        partial_out,
        partial_m,
        partial_l,
        rows,
        q_heads,
        kv_heads,
        dim,
        scale,
        chunk_size,
        num_splits);
  } else if (grouped) {
    hipLaunchKernelGGL(
        dms_compact_attn_splitk_grouped_producer_kernel,
        dim3(producer_blocks),
        dim3(kDmsPackThreads),
        0,
        stream,
        q,
        k_slot,
        v_slot,
        base_offsets,
        live_counts,
        partial_out,
        partial_m,
        partial_l,
        rows,
        q_heads,
        kv_heads,
        dim,
        scale,
        chunk_size,
        num_splits);
  } else {
    hipLaunchKernelGGL(
        dms_compact_attn_splitk_producer_kernel,
        dim3(producer_blocks),
        dim3(kDmsPackThreads),
        0,
        stream,
        q,
        k_slot,
        v_slot,
        base_offsets,
        live_counts,
        partial_out,
        partial_m,
        partial_l,
        rows,
        q_heads,
        kv_heads,
        dim,
        scale,
        chunk_size,
        num_splits);
  }
  hipError_t error = hipGetLastError();
  if (error != hipSuccess) {
    return static_cast<int>(error);
  }
  const int32_t reducer_blocks = rows * q_heads;
  hipLaunchKernelGGL(
      dms_compact_attn_splitk_reduce_kernel,
      dim3(reducer_blocks),
      dim3(kDmsPackThreads),
      0,
      stream,
      partial_out,
      partial_m,
      partial_l,
      out,
      rows,
      q_heads,
      dim,
      num_splits);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_dms_compact_attn_decode_bf16(
    const float* q,
    const uint16_t* k_slot,
    const uint16_t* v_slot,
    const int32_t* base_offsets,
    const int32_t* live_counts,
    float* out,
    int32_t rows,
    int32_t q_heads,
    int32_t kv_heads,
    int32_t dim,
    float scale,
    int32_t score_capacity,
    hipStream_t stream) {
  if (rows <= 0 || q_heads <= 0 || kv_heads <= 0 || dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (q_heads % kv_heads != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (!(scale > 0.0f) || score_capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int blocks = rows * q_heads;
  const size_t shared_bytes =
      static_cast<size_t>(2 * score_capacity) * sizeof(float);
  hipLaunchKernelGGL(
      dms_compact_attn_decode_kernel, dim3(blocks), dim3(kDmsPackThreads),
      shared_bytes, stream, q,
      k_slot,
      v_slot,
      base_offsets,
      live_counts,
      out,
      rows,
      q_heads,
      kv_heads,
      dim,
      scale,
      score_capacity);
  return static_cast<int>(hipGetLastError());
}
