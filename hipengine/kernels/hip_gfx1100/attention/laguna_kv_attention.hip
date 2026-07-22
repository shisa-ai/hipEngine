// hipEngine-native Laguna global/SWA KV and attention.
// Every body consumes the complete KVLiveSpans ABI: physical offsets, live
// count, absolute slot positions, eviction mask, and absolute query position.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

__device__ inline float laguna_bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t laguna_float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__device__ inline int64_t laguna_global_physical_slot(
    const int32_t* base_offsets,
    int64_t logical_slot,
    int64_t block_size) {
  const int64_t logical_block = logical_slot / block_size;
  const int64_t block_offset = logical_slot - logical_block * block_size;
  return static_cast<int64_t>(base_offsets[logical_block]) * block_size + block_offset;
}

__global__ void laguna_global_write_kv_f32_bf16_kernel(
    const float* key,
    const float* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    int64_t* live_counts,
    int64_t* token_positions,
    uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t position = row_positions[0];
  if (position < 0 || position >= capacity ||
      position / block_size >= block_table_len) {
    return;
  }
  const int64_t physical_slot =
      laguna_global_physical_slot(base_offsets, position, block_size);
  if (physical_slot < 0 || physical_slot >= block_table_len * block_size) {
    return;
  }

  const int64_t width = num_kv_heads * head_dim;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < width;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t cache_offset = physical_slot * width + idx;
    key_cache[cache_offset] = laguna_float_to_bf16_bits(key[idx]);
    value_cache[cache_offset] = laguna_float_to_bf16_bits(value[idx]);
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const bool slot_was_live =
        evict_mask[position] == 0 && token_positions[position] >= 0;
    token_positions[position] = position;
    evict_mask[position] = 0;
    const int64_t old_count = live_counts[0];
    const int64_t position_count = position + 1 < capacity ? position + 1 : capacity;
    const int64_t candidate_count = slot_was_live ? old_count : position_count;
    live_counts[0] = old_count > candidate_count ? old_count : candidate_count;
  }
}

__global__ void laguna_global_attention_decode_bf16_kernel(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t block_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t context_len = live_counts[0] < capacity ? live_counts[0] : capacity;
  const int64_t query_position = row_positions[0];
  if (context_len <= 0 || query_position < 0) {
    return;
  }

  extern __shared__ float shared[];
  float* scores = shared;
  const int num_warps = blockDim.x / warpSize;
  float* warp_buf = scores + capacity;
  float* q_shared = warp_buf + num_warps;
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  if (q_head >= num_q_heads) {
    return;
  }
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const int64_t q_offset = q_head * head_dim;
  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    q_shared[dim] = query[q_offset + dim];
  }
  __syncthreads();

  float local_max = -3.4028234663852886e38f;
  for (int64_t token = warp_id; token < context_len; token += num_warps) {
    const int64_t key_position = token_positions[token];
    const bool visible = evict_mask[token] == 0 &&
                         key_position >= 0 && key_position <= query_position;
    float acc = 0.0f;
    if (visible) {
      const int64_t physical_slot =
          laguna_global_physical_slot(base_offsets, token, block_size);
      const int64_t k_offset =
          (physical_slot * num_kv_heads + kv_head) * head_dim;
      for (int64_t dim = lane_id * 4; dim < head_dim; dim += warpSize * 4) {
        acc += q_shared[dim + 0] *
               laguna_bf16_bits_to_float(key_cache[k_offset + dim + 0]);
        acc += q_shared[dim + 1] *
               laguna_bf16_bits_to_float(key_cache[k_offset + dim + 1]);
        acc += q_shared[dim + 2] *
               laguna_bf16_bits_to_float(key_cache[k_offset + dim + 2]);
        acc += q_shared[dim + 3] *
               laguna_bf16_bits_to_float(key_cache[k_offset + dim + 3]);
      }
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      acc += __shfl_down(acc, offset);
    }
    if (lane_id == 0) {
      const float score = visible ? acc * scale : -3.4028234663852886e38f;
      scores[token] = score;
      local_max = fmaxf(local_max, score);
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_max = fmaxf(local_max, __shfl_down(local_max, offset));
  }
  if (lane_id == 0) {
    warp_buf[warp_id] = local_max;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float max_score = warp_buf[0];
    for (int warp = 1; warp < num_warps; ++warp) {
      max_score = fmaxf(max_score, warp_buf[warp]);
    }
    warp_buf[0] = max_score;
  }
  __syncthreads();
  const float max_score = warp_buf[0];

  float local_sum = 0.0f;
  for (int64_t token = warp_id; token < context_len; token += num_warps) {
    if (lane_id == 0) {
      const bool visible = evict_mask[token] == 0 &&
                           token_positions[token] >= 0 &&
                           token_positions[token] <= query_position;
      const float weight = visible ? expf(scores[token] - max_score) : 0.0f;
      scores[token] = weight;
      local_sum += weight;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down(local_sum, offset);
  }
  if (lane_id == 0) {
    warp_buf[warp_id] = local_sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float total_sum = warp_buf[0];
    for (int warp = 1; warp < num_warps; ++warp) {
      total_sum += warp_buf[warp];
    }
    warp_buf[0] = 1.0f / fmaxf(total_sum, 1.0e-20f);
  }
  __syncthreads();
  const float inv_denom = warp_buf[0];

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < context_len; ++token) {
      const float weight = scores[token] * inv_denom;
      if (!(weight > 0.0f)) {
        continue;
      }
      const int64_t physical_slot =
          laguna_global_physical_slot(base_offsets, token, block_size);
      const int64_t v_offset =
          (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
      acc += weight * laguna_bf16_bits_to_float(value_cache[v_offset]);
    }
    out[q_offset + dim] = acc;
  }
}

__global__ void laguna_swa_write_kv_f32_bf16_kernel(
    const float* key,
    const float* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    int64_t* live_counts,
    int64_t* token_positions,
    uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t position = row_positions[0];
  if (position < 0) {
    return;
  }
  const int64_t logical_slot = position % capacity;
  const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
  if (physical_slot < 0 || physical_slot >= capacity) {
    return;
  }

  const int64_t width = num_kv_heads * head_dim;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < width;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t cache_offset = physical_slot * width + idx;
    key_cache[cache_offset] = laguna_float_to_bf16_bits(key[idx]);
    value_cache[cache_offset] = laguna_float_to_bf16_bits(value[idx]);
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const bool slot_was_live =
        evict_mask[logical_slot] == 0 && token_positions[logical_slot] >= 0;
    token_positions[logical_slot] = position;
    evict_mask[logical_slot] = 0;
    const int64_t old_count = live_counts[0];
    const int64_t position_count = position + 1 < capacity ? position + 1 : capacity;
    const int64_t candidate_count = slot_was_live ? old_count : position_count;
    live_counts[0] = old_count > candidate_count ? old_count : candidate_count;
  }
}

__global__ __launch_bounds__(128) void laguna_swa_attention_decode_bf16_kernel(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t sliding_window,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t dim = static_cast<int64_t>(threadIdx.x);
  if (q_head >= num_q_heads || dim >= head_dim) {
    return;
  }

  const int64_t live_count = live_counts[0];
  const int64_t query_position = row_positions[0];
  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  const int64_t q_offset = q_head * head_dim;
  const int64_t scan_slots = live_count < capacity ? live_count : capacity;

  __shared__ float reduction[128];
  __shared__ float shared_max;
  __shared__ float shared_denom;
  __shared__ float shared_weight;

  if (threadIdx.x == 0) {
    shared_max = -3.4028234663852886e38f;
    shared_denom = 0.0f;
  }
  __syncthreads();

  if (live_count <= 0 || query_position < 0) {
    out[q_offset + dim] = 0.0f;
    return;
  }

  for (int64_t logical_slot = 0; logical_slot < scan_slots; ++logical_slot) {
    const int64_t key_position = token_positions[logical_slot];
    const bool visible = evict_mask[logical_slot] == 0 &&
                         key_position >= 0 &&
                         key_position <= query_position &&
                         key_position > query_position - sliding_window;
    if (!visible) {
      continue;
    }
    const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    const bool physical_valid = physical_slot >= 0 && physical_slot < capacity;
    const int64_t cache_offset =
        (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
    reduction[threadIdx.x] = physical_valid
                                 ? query[q_offset + dim] *
                                       laguna_bf16_bits_to_float(key_cache[cache_offset])
                                 : 0.0f;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0 && physical_valid) {
      shared_max = fmaxf(shared_max, reduction[0] * scale);
    }
    __syncthreads();
  }

  float output_acc = 0.0f;
  for (int64_t logical_slot = 0; logical_slot < scan_slots; ++logical_slot) {
    const int64_t key_position = token_positions[logical_slot];
    const bool visible = evict_mask[logical_slot] == 0 &&
                         key_position >= 0 &&
                         key_position <= query_position &&
                         key_position > query_position - sliding_window;
    if (!visible) {
      continue;
    }
    const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    const bool physical_valid = physical_slot >= 0 && physical_slot < capacity;
    const int64_t cache_offset =
        (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
    reduction[threadIdx.x] = physical_valid
                                 ? query[q_offset + dim] *
                                       laguna_bf16_bits_to_float(key_cache[cache_offset])
                                 : 0.0f;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      shared_weight = physical_valid
                          ? expf(reduction[0] * scale - shared_max)
                          : 0.0f;
      shared_denom += shared_weight;
    }
    __syncthreads();
    if (physical_valid) {
      output_acc += shared_weight *
                    laguna_bf16_bits_to_float(value_cache[cache_offset]);
    }
    __syncthreads();
  }

  const float denom = fmaxf(shared_denom, 1.0e-20f);
  out[q_offset + dim] = output_acc / denom;
}

}  // namespace

extern "C" int hipengine_laguna_global_write_kv_f32_bf16_spans(
    const float* key,
    const float* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    int64_t* live_counts,
    int64_t* token_positions,
    uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    hipStream_t stream) {
  if (capacity <= 0 || block_size != 256 || block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len ||
      num_kv_heads <= 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t width = num_kv_heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((width + threads - 1) / threads);
  hipLaunchKernelGGL(
      laguna_global_write_kv_f32_bf16_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      key,
      value,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      block_size,
      block_table_len,
      num_kv_heads,
      head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_global_attention_decode_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (capacity <= 0 || block_size != 256 || block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int warps = threads / 32;
  const size_t shared_bytes =
      static_cast<size_t>(capacity + warps + head_dim) * sizeof(float);
  hipLaunchKernelGGL(
      laguna_global_attention_decode_bf16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_swa_write_kv_f32_bf16_spans(
    const float* key,
    const float* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    int64_t* live_counts,
    int64_t* token_positions,
    uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t num_kv_heads,
    int64_t head_dim,
    hipStream_t stream) {
  if (capacity <= 0 || num_kv_heads <= 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t width = num_kv_heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((width + threads - 1) / threads);
  hipLaunchKernelGGL(
      laguna_swa_write_kv_f32_bf16_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      key,
      value,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      num_kv_heads,
      head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_swa_attention_decode_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t sliding_window,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (capacity <= 0 || sliding_window <= 0 || sliding_window > capacity ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_swa_attention_decode_bf16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(128),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      sliding_window,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}
