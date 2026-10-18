// Gemma 4 direct per-token/head INT8 paged attention decode consumer (gfx1100).
//
// This is the strict consumer for the BF16-source per-token/head INT8 KV writer
// (`paged_kv_write.hip`). It reads the exact representation the writer stores --
// an INT8 payload plus a per-token/per-KV-head FP16 or FP32 scale -- and
// reconstructs K/V as `float32(int8) * float32(scale)`. It is a decode consumer:
// one query row per (query head, KV head group), one block per query head.
//
// It is deliberately not fused with the BF16 Gemma 4 attention family. The BF16
// kernels are unchanged; this file adds the INT8-specific path beside them.
//
// Two Gemma 4 specifics carry over from the BF16 family:
//
//   * `scale` is always 1.0. Gemma 4 folds the softmax scaling into the query
//     norm weight, so the kernel multiplies by whatever it is handed.
//   * The keep decision is span-native, not an implied causal test. A logical
//     slot is visible when it is not evicted, its absolute `token_positions`
//     entry is non-negative and no later than the query's `row_positions`, and
//     -- on a sliding layer -- strictly inside `sliding_window`. A non-positive
//     `sliding_window` is a global (causal-only) layer. When the span carries no
//     position/eviction tensors the dense fill is uniform: slot `j` sits at
//     position `j` and the query sits at `context_len - 1`.
//
// GQA maps query head `h` to KV head `h / (num_q_heads / num_kv_heads)`.
//
// Failure policy (loud, never a silent success):
//
//   * An empty span (`live_counts[0] == 0`) writes an all-zero row.
//   * A negative or `> capacity` count writes a NaN row: an invalid count is
//     not a truncated success prefix. The host wrapper reports the same
//     condition before launch.
//   * A *visible* slot whose page-table id is out of `[0, cache_blocks)`, or
//     whose logit is non-finite, writes a NaN row. A masked slot never
//     participates, so a poisoned masked K/V scale cannot leak into the result.
//   * A visible slot whose reconstructed V is non-finite writes a NaN row too,
//     independent of its softmax weight: an underflowed (zero) weight must not
//     hide a poisoned visible value. The failure is head-local (only the query
//     heads mapped to the poisoned KV head fail).

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kGemma4Int8AttentionThreads = 256;
constexpr int kGemma4Int8WarpSize = 32;
// `blockIdx.y` selects the query row and HIP's gridDim.y maximum is 65535, so a
// larger row count must be refused here rather than silently truncated by the
// `unsigned int` cast (rows = 2**32 + 1 would launch only row 0).
constexpr int64_t kGemma4Int8PrefillMaxRows = 65535;

__device__ inline float gemma4_int8_nan() { return __int_as_float(0x7fc00000); }

template <typename scale_t>
__device__ inline float gemma4_int8_scale_to_float(scale_t value) {
  return static_cast<float>(value);
}

__device__ inline void gemma4_int8_write_row_at(
    float* __restrict__ out_row, int64_t head_dim, float value) {
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    out_row[dim] = value;
  }
}

__device__ inline void gemma4_int8_write_row(
    float* __restrict__ out, int64_t q_head, int64_t head_dim, float value) {
  gemma4_int8_write_row_at(out + q_head * head_dim, head_dim, value);
}

__device__ inline float gemma4_int8_dot(
    const float* __restrict__ q,
    const int8_t* __restrict__ k_row,
    float k_scale,
    int64_t head_dim) {
  // One warp per key slot, so each lane covers a strided slice of the whole row.
  const int64_t lane = threadIdx.x % warpSize;
  float acc = 0.0f;
  for (int64_t dim = lane; dim < head_dim; dim += warpSize) {
    acc += q[dim] * (static_cast<float>(k_row[dim]) * k_scale);
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    acc += __shfl_down(acc, offset);
  }
  return acc;
}

__device__ inline bool gemma4_int8_slot_visible(
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    int64_t slot,
    int64_t query_position,
    int64_t sliding_window) {
  if (evict_mask != nullptr && evict_mask[slot] != 0) {
    return false;
  }
  const int64_t position = token_positions != nullptr ? token_positions[slot] : slot;
  if (position < 0 || position > query_position) {
    return false;
  }
  if (sliding_window > 0 && position <= query_position - sliding_window) {
    return false;
  }
  return true;
}

template <typename scale_t>
__global__ __launch_bounds__(kGemma4Int8AttentionThreads) void
gemma4_attention_decode_int8_per_token_head_kernel(
    const float* __restrict__ query,          // [num_q_heads, head_dim]
    const int8_t* __restrict__ key_cache,     // [cache_blocks, block_size, Hkv, D]
    const int8_t* __restrict__ value_cache,
    const scale_t* __restrict__ k_scale,      // [cache_blocks, block_size, Hkv]
    const scale_t* __restrict__ v_scale,
    float* __restrict__ out,                  // [num_q_heads, head_dim]
    const int32_t* __restrict__ base_offsets,  // [block_table_len]
    const int64_t* __restrict__ live_counts,   // [1]
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    int64_t capacity,
    int64_t cache_blocks,
    int64_t block_table_len,
    int64_t block_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t sliding_window,
    float scale) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  if (q_head >= num_q_heads) {
    return;
  }
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;

  const int64_t raw_count = live_counts[0];
  if (raw_count < 0 || raw_count > capacity) {
    // Invalid count: a negative malformed count or an overflow past the scratch
    // bound. Loud NaN, never a truncated success prefix.
    gemma4_int8_write_row(out, q_head, head_dim, gemma4_int8_nan());
    return;
  }
  const int64_t context_len = raw_count;
  if (context_len == 0) {
    // Supported empty span: no keys to attend to, and no stale output left.
    gemma4_int8_write_row(out, q_head, head_dim, 0.0f);
    return;
  }
  const int64_t query_position =
      row_positions != nullptr ? row_positions[0] : context_len - 1;

  extern __shared__ float shared[];
  float* scores = shared;                    // [capacity]
  float* q_shared = scores + capacity;       // [head_dim]
  float* warp_buf = q_shared + head_dim;     // [num_warps]
  int* warp_flags = reinterpret_cast<int*>(warp_buf + blockDim.x / warpSize);  // [num_warps]

  const int64_t num_warps = blockDim.x / warpSize;
  const int64_t lane_id = threadIdx.x % warpSize;
  const int64_t warp_id = threadIdx.x / warpSize;

  const float* q_row = query + q_head * head_dim;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    q_shared[dim] = q_row[dim];
  }
  __syncthreads();

  // Pass 1: per-key logits and their running maximum. A masked slot is -inf.
  // A visible slot whose page id is out of range or whose logit is non-finite
  // sets the failure flag instead of being silently treated as masked.
  float local_max = -INFINITY;
  int local_nonfinite = 0;
  for (int64_t slot = warp_id; slot < context_len; slot += num_warps) {
    const bool visible = gemma4_int8_slot_visible(
        token_positions, evict_mask, slot, query_position, sliding_window);
    float score = -INFINITY;
    bool ok = false;
    if (visible) {
      const int64_t logical_block = slot / block_size;
      if (logical_block < block_table_len) {
        const int64_t physical_block = static_cast<int64_t>(base_offsets[logical_block]);
        if (physical_block >= 0 && physical_block < cache_blocks) {
          const int64_t block_offset = slot - logical_block * block_size;
          const int64_t physical_slot = physical_block * block_size + block_offset;
          const int64_t scale_offset = physical_slot * num_kv_heads + kv_head;
          const float k_s = gemma4_int8_scale_to_float(k_scale[scale_offset]);
          const int8_t* k_row = key_cache + scale_offset * head_dim;
          score = gemma4_int8_dot(q_shared, k_row, k_s, head_dim) * scale;
          // A visible non-finite reconstructed V is a failure independent of its
          // softmax weight: an underflowed (zero) weight must not hide it.
          const float v_s = gemma4_int8_scale_to_float(v_scale[scale_offset]);
          const int8_t* v_row = value_cache + scale_offset * head_dim;
          for (int64_t dim = lane_id; dim < head_dim; dim += warpSize) {
            if (!isfinite(static_cast<float>(v_row[dim]) * v_s)) {
              local_nonfinite = 1;
            }
          }
          ok = true;
        }
      }
    }
    if (lane_id == 0) {
      scores[slot] = score;
      if (visible) {
        if (ok && isfinite(score)) {
          local_max = fmaxf(local_max, score);
        } else {
          local_nonfinite = 1;
        }
      }
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    local_nonfinite |= __shfl_down(local_nonfinite, offset);
  }
  if (lane_id == 0) {
    warp_buf[warp_id] = local_max;
    warp_flags[warp_id] = local_nonfinite;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_max = warp_buf[0];
    int block_nonfinite = warp_flags[0];
    for (int64_t warp = 1; warp < num_warps; ++warp) {
      block_max = fmaxf(block_max, warp_buf[warp]);
      block_nonfinite |= warp_flags[warp];
    }
    warp_buf[0] = block_max;
    warp_flags[0] = block_nonfinite;
  }
  __syncthreads();
  if (warp_flags[0] != 0) {
    gemma4_int8_write_row(out, q_head, head_dim, gemma4_int8_nan());
    return;
  }
  const float row_max = warp_buf[0];

  // Pass 2: denominator. Masked slots stay zero even when the whole row is
  // masked (row_max is -inf), because a non-finite score is never exponentiated.
  float local_sum = 0.0f;
  for (int64_t slot = warp_id; slot < context_len; slot += num_warps) {
    if (lane_id == 0) {
      const float score = scores[slot];
      const float weight = isfinite(score) ? expf(score - row_max) : 0.0f;
      scores[slot] = weight;
      local_sum += weight;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_sum += __shfl_down(local_sum, offset);
  }
  if (lane_id == 0) {
    warp_buf[warp_id] = local_sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int64_t warp = 0; warp < num_warps; ++warp) {
      total += warp_buf[warp];
    }
    warp_buf[0] = 1.0f / fmaxf(total, 1.0e-20f);
  }
  __syncthreads();
  const float inv_denom = warp_buf[0];

  // Pass 3: weighted value sum. Weights are normalised in `scores`. A masked
  // slot has weight 0 and is skipped, so a poisoned masked V scale never
  // participates; a visible non-finite V propagates as NaN.
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t slot = 0; slot < context_len; ++slot) {
      const float weight = scores[slot] * inv_denom;
      if (!(weight > 0.0f)) {
        continue;
      }
      const int64_t logical_block = slot / block_size;
      const int64_t physical_block = static_cast<int64_t>(base_offsets[logical_block]);
      const int64_t block_offset = slot - logical_block * block_size;
      const int64_t physical_slot = physical_block * block_size + block_offset;
      const int64_t scale_offset = physical_slot * num_kv_heads + kv_head;
      const float v_s = gemma4_int8_scale_to_float(v_scale[scale_offset]);
      acc += weight * (static_cast<float>(value_cache[scale_offset * head_dim + dim]) * v_s);
    }
    out[q_head * head_dim + dim] = acc;
  }
}

// Multi-row prefill sibling of the decode consumer above.
//
// Semantics, declared explicitly. Unsupported forms are refused by the host
// wrapper before any build or launch; this kernel assumes a validated launch.
//
//   * ONE SHARED LOGICAL KV PREFIX. Every query row reads the same logical slot
//     sequence through one shared 1-D page table (`base_offsets`). A row does
//     not have an independent page table; a per-row page table would be a
//     separate prefix and is refused. This is deliberately distinct from a
//     multi-row decode span, which carries a row-major page table.
//   * Each query row carries its own live count (`live_counts[row]`, the row's
//     causal prefix length) and its own absolute position (`row_positions[row]`).
//     A zero count is a supported empty row (zeros); a negative or
//     over-capacity count is a NaN row; a row whose position leaves no visible
//     slot is all-masked and returns zeros.
//   * One block per (query row, query head): `blockIdx.x` is the query head and
//     `blockIdx.y` is the row. Shared scratch is bounded by `capacity`.
//
// The failure policy is the decode consumer's, scoped per (row, head): a
// visible out-of-range page id, a visible non-finite logit, or a visible
// non-finite reconstructed V writes a NaN row for that block only. A masked
// slot never participates, so a poisoned masked K/V scale cannot leak.
template <typename scale_t>
__global__ __launch_bounds__(kGemma4Int8AttentionThreads) void
gemma4_attention_prefill_int8_per_token_head_kernel(
    const float* __restrict__ query,          // [rows, num_q_heads, head_dim]
    const int8_t* __restrict__ key_cache,     // [cache_blocks, block_size, Hkv, D]
    const int8_t* __restrict__ value_cache,
    const scale_t* __restrict__ k_scale,      // [cache_blocks, block_size, Hkv]
    const scale_t* __restrict__ v_scale,
    float* __restrict__ out,                  // [rows, num_q_heads, head_dim]
    const int32_t* __restrict__ base_offsets,  // [block_table_len] shared prefix
    const int64_t* __restrict__ live_counts,   // [rows]
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,  // [rows]
    int64_t rows,
    int64_t capacity,
    int64_t cache_blocks,
    int64_t block_table_len,
    int64_t block_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t sliding_window,
    float scale) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (q_head >= num_q_heads || row >= rows) {
    return;
  }
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  float* const out_row = out + (row * num_q_heads + q_head) * head_dim;

  const int64_t raw_count = live_counts[row];
  if (raw_count < 0 || raw_count > capacity) {
    gemma4_int8_write_row_at(out_row, head_dim, gemma4_int8_nan());
    return;
  }
  const int64_t context_len = raw_count;
  if (context_len == 0) {
    gemma4_int8_write_row_at(out_row, head_dim, 0.0f);
    return;
  }
  const int64_t query_position =
      row_positions != nullptr ? row_positions[row] : context_len - 1;

  extern __shared__ float shared[];
  float* scores = shared;                    // [capacity]
  float* q_shared = scores + capacity;       // [head_dim]
  float* warp_buf = q_shared + head_dim;     // [num_warps]
  int* warp_flags = reinterpret_cast<int*>(warp_buf + blockDim.x / warpSize);  // [num_warps]

  const int64_t num_warps = blockDim.x / warpSize;
  const int64_t lane_id = threadIdx.x % warpSize;
  const int64_t warp_id = threadIdx.x / warpSize;

  const float* q_row = query + (row * num_q_heads + q_head) * head_dim;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    q_shared[dim] = q_row[dim];
  }
  __syncthreads();

  // Pass 1: per-key logits and their running maximum. A masked slot is -inf.
  // A visible slot whose page id is out of range or whose logit is non-finite
  // sets the failure flag instead of being silently treated as masked.
  float local_max = -INFINITY;
  int local_nonfinite = 0;
  for (int64_t slot = warp_id; slot < context_len; slot += num_warps) {
    const bool visible = gemma4_int8_slot_visible(
        token_positions, evict_mask, slot, query_position, sliding_window);
    float score = -INFINITY;
    bool ok = false;
    if (visible) {
      const int64_t logical_block = slot / block_size;
      if (logical_block < block_table_len) {
        const int64_t physical_block = static_cast<int64_t>(base_offsets[logical_block]);
        if (physical_block >= 0 && physical_block < cache_blocks) {
          const int64_t block_offset = slot - logical_block * block_size;
          const int64_t physical_slot = physical_block * block_size + block_offset;
          const int64_t scale_offset = physical_slot * num_kv_heads + kv_head;
          const float k_s = gemma4_int8_scale_to_float(k_scale[scale_offset]);
          const int8_t* k_row = key_cache + scale_offset * head_dim;
          score = gemma4_int8_dot(q_shared, k_row, k_s, head_dim) * scale;
          // A visible non-finite reconstructed V is a failure independent of its
          // softmax weight: an underflowed (zero) weight must not hide it.
          const float v_s = gemma4_int8_scale_to_float(v_scale[scale_offset]);
          const int8_t* v_row = value_cache + scale_offset * head_dim;
          for (int64_t dim = lane_id; dim < head_dim; dim += warpSize) {
            if (!isfinite(static_cast<float>(v_row[dim]) * v_s)) {
              local_nonfinite = 1;
            }
          }
          ok = true;
        }
      }
    }
    if (lane_id == 0) {
      scores[slot] = score;
      if (visible) {
        if (ok && isfinite(score)) {
          local_max = fmaxf(local_max, score);
        } else {
          local_nonfinite = 1;
        }
      }
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    local_nonfinite |= __shfl_down(local_nonfinite, offset);
  }
  if (lane_id == 0) {
    warp_buf[warp_id] = local_max;
    warp_flags[warp_id] = local_nonfinite;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_max = warp_buf[0];
    int block_nonfinite = warp_flags[0];
    for (int64_t warp = 1; warp < num_warps; ++warp) {
      block_max = fmaxf(block_max, warp_buf[warp]);
      block_nonfinite |= warp_flags[warp];
    }
    warp_buf[0] = block_max;
    warp_flags[0] = block_nonfinite;
  }
  __syncthreads();
  if (warp_flags[0] != 0) {
    gemma4_int8_write_row_at(out_row, head_dim, gemma4_int8_nan());
    return;
  }
  const float row_max = warp_buf[0];

  // Pass 2: denominator. Masked slots stay zero even when the whole row is
  // masked (row_max is -inf), because a non-finite score is never exponentiated.
  float local_sum = 0.0f;
  for (int64_t slot = warp_id; slot < context_len; slot += num_warps) {
    if (lane_id == 0) {
      const float score = scores[slot];
      const float weight = isfinite(score) ? expf(score - row_max) : 0.0f;
      scores[slot] = weight;
      local_sum += weight;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_sum += __shfl_down(local_sum, offset);
  }
  if (lane_id == 0) {
    warp_buf[warp_id] = local_sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int64_t warp = 0; warp < num_warps; ++warp) {
      total += warp_buf[warp];
    }
    warp_buf[0] = 1.0f / fmaxf(total, 1.0e-20f);
  }
  __syncthreads();
  const float inv_denom = warp_buf[0];

  // Pass 3: weighted value sum. Weights are normalised in `scores`. A masked
  // slot has weight 0 and is skipped, so a poisoned masked V scale never
  // participates; a visible non-finite V propagates as NaN.
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t slot = 0; slot < context_len; ++slot) {
      const float weight = scores[slot] * inv_denom;
      if (!(weight > 0.0f)) {
        continue;
      }
      const int64_t logical_block = slot / block_size;
      const int64_t physical_block = static_cast<int64_t>(base_offsets[logical_block]);
      const int64_t block_offset = slot - logical_block * block_size;
      const int64_t physical_slot = physical_block * block_size + block_offset;
      const int64_t scale_offset = physical_slot * num_kv_heads + kv_head;
      const float v_s = gemma4_int8_scale_to_float(v_scale[scale_offset]);
      acc += weight * (static_cast<float>(value_cache[scale_offset * head_dim + dim]) * v_s);
    }
    out_row[dim] = acc;
  }
}

template <typename scale_t>
int launch_gemma4_attention_decode_int8_per_token_head(
    const float* query,
    const int8_t* key_cache,
    const int8_t* value_cache,
    const scale_t* k_scale,
    const scale_t* v_scale,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t cache_blocks,
    int64_t block_table_len,
    int64_t block_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t sliding_window,
    float scale,
    hipStream_t stream) {
  if (capacity <= 0 || cache_blocks <= 0 || block_size <= 0 || block_table_len <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      num_q_heads % num_kv_heads != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const size_t shared_bytes =
      static_cast<size_t>(capacity + head_dim + 2 * (kGemma4Int8AttentionThreads /
                                                      kGemma4Int8WarpSize)) *
      sizeof(float);
  if (shared_bytes > 64 * 1024) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(gemma4_attention_decode_int8_per_token_head_kernel<scale_t>),
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(static_cast<unsigned int>(kGemma4Int8AttentionThreads)),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      k_scale,
      v_scale,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      cache_blocks,
      block_table_len,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      sliding_window,
      scale);
  return static_cast<int>(hipGetLastError());
}

template <typename scale_t>
int launch_gemma4_attention_prefill_int8_per_token_head(
    const float* query,
    const int8_t* key_cache,
    const int8_t* value_cache,
    const scale_t* k_scale,
    const scale_t* v_scale,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t cache_blocks,
    int64_t block_table_len,
    int64_t block_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t sliding_window,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 || rows > kGemma4Int8PrefillMaxRows || capacity <= 0 ||
      cache_blocks <= 0 || block_size <= 0 ||
      block_table_len <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      num_q_heads % num_kv_heads != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const size_t shared_bytes =
      static_cast<size_t>(capacity + head_dim + 2 * (kGemma4Int8AttentionThreads /
                                                      kGemma4Int8WarpSize)) *
      sizeof(float);
  if (shared_bytes > 64 * 1024) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(gemma4_attention_prefill_int8_per_token_head_kernel<scale_t>),
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(kGemma4Int8AttentionThreads)),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      k_scale,
      v_scale,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      rows,
      capacity,
      cache_blocks,
      block_table_len,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      sliding_window,
      scale);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gemma4_attention_decode_int8_per_token_head_scale_f32_spans(
    const float* query,
    const int8_t* key_cache,
    const int8_t* value_cache,
    const float* k_scale,
    const float* v_scale,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t cache_blocks,
    int64_t block_table_len,
    int64_t block_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t sliding_window,
    float scale,
    hipStream_t stream) {
  return launch_gemma4_attention_decode_int8_per_token_head<float>(
      query,
      key_cache,
      value_cache,
      k_scale,
      v_scale,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      cache_blocks,
      block_table_len,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      sliding_window,
      scale,
      stream);
}

extern "C" int hipengine_gemma4_attention_decode_int8_per_token_head_scale_fp16_spans(
    const float* query,
    const int8_t* key_cache,
    const int8_t* value_cache,
    const _Float16* k_scale,
    const _Float16* v_scale,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t cache_blocks,
    int64_t block_table_len,
    int64_t block_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t sliding_window,
    float scale,
    hipStream_t stream) {
  return launch_gemma4_attention_decode_int8_per_token_head<_Float16>(
      query,
      key_cache,
      value_cache,
      k_scale,
      v_scale,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      cache_blocks,
      block_table_len,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      sliding_window,
      scale,
      stream);
}

extern "C" int hipengine_gemma4_attention_prefill_int8_per_token_head_scale_f32_spans(
    const float* query,
    const int8_t* key_cache,
    const int8_t* value_cache,
    const float* k_scale,
    const float* v_scale,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t cache_blocks,
    int64_t block_table_len,
    int64_t block_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t sliding_window,
    float scale,
    hipStream_t stream) {
  return launch_gemma4_attention_prefill_int8_per_token_head<float>(
      query,
      key_cache,
      value_cache,
      k_scale,
      v_scale,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      rows,
      capacity,
      cache_blocks,
      block_table_len,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      sliding_window,
      scale,
      stream);
}

extern "C" int hipengine_gemma4_attention_prefill_int8_per_token_head_scale_fp16_spans(
    const float* query,
    const int8_t* key_cache,
    const int8_t* value_cache,
    const _Float16* k_scale,
    const _Float16* v_scale,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t cache_blocks,
    int64_t block_table_len,
    int64_t block_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t sliding_window,
    float scale,
    hipStream_t stream) {
  return launch_gemma4_attention_prefill_int8_per_token_head<_Float16>(
      query,
      key_cache,
      value_cache,
      k_scale,
      v_scale,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      rows,
      capacity,
      cache_blocks,
      block_table_len,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      sliding_window,
      scale,
      stream);
}
