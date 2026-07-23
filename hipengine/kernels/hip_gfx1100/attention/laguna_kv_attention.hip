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

__device__ __forceinline__ float laguna_ordered_mul_f32(float lhs, float rhs) {
  float product;
  asm volatile(
      "v_mul_f32_e32 %[product], %[lhs], %[rhs]"
      : [product] "=v"(product)
      : [lhs] "v"(lhs), [rhs] "v"(rhs));
  return product;
}

__device__ __forceinline__ float laguna_ordered_add_f32(float lhs, float rhs) {
  float result;
  asm volatile(
      "v_add_f32_e32 %[result], %[lhs], %[rhs]"
      : [result] "=v"(result)
      : [lhs] "v"(lhs), [rhs] "v"(rhs));
  return result;
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

__global__ void laguna_global_write_kv_rows_f32_bf16_kernel(
    const float* key,
    const float* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    int64_t* live_counts,
    int64_t* token_positions,
    uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= rows) return;
  const int64_t position = row_positions[0] + row;
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
  const int64_t row_offset = row * width;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < width;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t cache_offset = physical_slot * width + idx;
    key_cache[cache_offset] = laguna_float_to_bf16_bits(key[row_offset + idx]);
    value_cache[cache_offset] = laguna_float_to_bf16_bits(value[row_offset + idx]);
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    token_positions[position] = position;
    evict_mask[position] = 0;
    if (row == rows - 1) {
      const int64_t old_count = live_counts[0];
      const int64_t position_count = position + 1 < capacity ? position + 1 : capacity;
      live_counts[0] = old_count > position_count ? old_count : position_count;
    }
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

__global__ void laguna_global_attention_prefill_bf16_kernel(
    const float* query,
    const float* current_key,
    const float* current_value,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t block_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t query_row = static_cast<int64_t>(blockIdx.y);
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  if (query_row >= rows || q_head >= num_q_heads) return;
  const int64_t start_position = row_positions[0];
  const int64_t query_position = start_position + query_row;
  const int64_t prior_count = live_counts[0] < capacity ? live_counts[0] : capacity;
  const int64_t position_count = query_position + 1 < capacity ? query_position + 1 : capacity;
  const int64_t context_len = prior_count > position_count ? prior_count : position_count;
  if (context_len <= 0 || start_position < 0 || query_position < start_position) return;

  extern __shared__ float shared[];
  float* scores = shared;
  const int num_warps = blockDim.x / warpSize;
  float* warp_buf = scores + capacity;
  float* q_shared = warp_buf + num_warps;
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const int64_t q_offset = (query_row * num_q_heads + q_head) * head_dim;
  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    q_shared[dim] = query[q_offset + dim];
  }
  __syncthreads();

  float local_max = -3.4028234663852886e38f;
  for (int64_t token = warp_id; token < context_len; token += num_warps) {
    const int64_t current_row = token - start_position;
    const bool from_current = current_row >= 0 && current_row <= query_row && current_row < rows;
    const int64_t key_position = from_current ? token : token_positions[token];
    const bool visible = from_current ||
        (evict_mask[token] == 0 && key_position >= 0 && key_position <= query_position);
    float acc = 0.0f;
    if (visible) {
      const int64_t physical_slot = from_current
          ? -1
          : laguna_global_physical_slot(base_offsets, token, block_size);
      const int64_t k_offset = from_current
          ? (current_row * num_kv_heads + kv_head) * head_dim
          : (physical_slot * num_kv_heads + kv_head) * head_dim;
      for (int64_t dim = lane_id * 4; dim < head_dim; dim += warpSize * 4) {
        const float k0 = from_current
            ? laguna_bf16_bits_to_float(laguna_float_to_bf16_bits(current_key[k_offset + dim + 0]))
            : laguna_bf16_bits_to_float(key_cache[k_offset + dim + 0]);
        const float k1 = from_current
            ? laguna_bf16_bits_to_float(laguna_float_to_bf16_bits(current_key[k_offset + dim + 1]))
            : laguna_bf16_bits_to_float(key_cache[k_offset + dim + 1]);
        const float k2 = from_current
            ? laguna_bf16_bits_to_float(laguna_float_to_bf16_bits(current_key[k_offset + dim + 2]))
            : laguna_bf16_bits_to_float(key_cache[k_offset + dim + 2]);
        const float k3 = from_current
            ? laguna_bf16_bits_to_float(laguna_float_to_bf16_bits(current_key[k_offset + dim + 3]))
            : laguna_bf16_bits_to_float(key_cache[k_offset + dim + 3]);
        acc += q_shared[dim + 0] * k0;
        acc += q_shared[dim + 1] * k1;
        acc += q_shared[dim + 2] * k2;
        acc += q_shared[dim + 3] * k3;
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
  if (lane_id == 0) warp_buf[warp_id] = local_max;
  __syncthreads();
  if (threadIdx.x == 0) {
    float max_score = warp_buf[0];
    for (int warp = 1; warp < num_warps; ++warp) max_score = fmaxf(max_score, warp_buf[warp]);
    warp_buf[0] = max_score;
  }
  __syncthreads();
  const float max_score = warp_buf[0];

  float local_sum = 0.0f;
  for (int64_t token = warp_id; token < context_len; token += num_warps) {
    if (lane_id == 0) {
      const int64_t current_row = token - start_position;
      const bool from_current = current_row >= 0 && current_row <= query_row && current_row < rows;
      const bool visible = from_current ||
          (evict_mask[token] == 0 && token_positions[token] >= 0 &&
           token_positions[token] <= query_position);
      const float weight = visible ? expf(scores[token] - max_score) : 0.0f;
      scores[token] = weight;
      local_sum += weight;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down(local_sum, offset);
  }
  if (lane_id == 0) warp_buf[warp_id] = local_sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total_sum = warp_buf[0];
    for (int warp = 1; warp < num_warps; ++warp) total_sum += warp_buf[warp];
    warp_buf[0] = 1.0f / fmaxf(total_sum, 1.0e-20f);
  }
  __syncthreads();
  const float inv_denom = warp_buf[0];

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < context_len; ++token) {
      const float weight = scores[token] * inv_denom;
      if (!(weight > 0.0f)) continue;
      const int64_t current_row = token - start_position;
      const bool from_current = current_row >= 0 && current_row <= query_row && current_row < rows;
      const int64_t physical_slot = from_current
          ? -1
          : laguna_global_physical_slot(base_offsets, token, block_size);
      const int64_t v_offset = from_current
          ? (current_row * num_kv_heads + kv_head) * head_dim + dim
          : (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
      const float v = from_current
          ? laguna_bf16_bits_to_float(laguna_float_to_bf16_bits(current_value[v_offset]))
          : laguna_bf16_bits_to_float(value_cache[v_offset]);
      acc += weight * v;
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

__global__ void laguna_swa_write_kv_rows_f32_bf16_kernel(
    const float* key,
    const float* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    int64_t* live_counts,
    int64_t* token_positions,
    uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= rows) return;
  const int64_t position = row_positions[0] + row;
  if (position < 0) return;
  const int64_t logical_slot = position % capacity;
  const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
  if (physical_slot < 0 || physical_slot >= capacity) return;

  const int64_t width = num_kv_heads * head_dim;
  const int64_t row_offset = row * width;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < width;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t cache_offset = physical_slot * width + idx;
    key_cache[cache_offset] = laguna_float_to_bf16_bits(key[row_offset + idx]);
    value_cache[cache_offset] = laguna_float_to_bf16_bits(value[row_offset + idx]);
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    token_positions[logical_slot] = position;
    evict_mask[logical_slot] = 0;
    if (row == rows - 1) {
      const int64_t old_count = live_counts[0];
      const int64_t position_count = position + 1 < capacity ? position + 1 : capacity;
      live_counts[0] = old_count > position_count ? old_count : position_count;
    }
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

__global__ __launch_bounds__(128) void laguna_swa_attention_prefill_bf16_kernel(
    const float* query,
    const float* current_key,
    const float* current_value,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t sliding_window,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t query_row = static_cast<int64_t>(blockIdx.y);
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t dim = static_cast<int64_t>(threadIdx.x);
  if (query_row >= rows || q_head >= num_q_heads || dim >= head_dim) return;

  const int64_t start_position = row_positions[0];
  const int64_t query_position = start_position + query_row;
  const int64_t prior_live_count = live_counts[0];
  const int64_t query_live_count = query_position + 1 < capacity ? query_position + 1 : capacity;
  const int64_t live_count = prior_live_count > query_live_count ? prior_live_count : query_live_count;
  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  const int64_t q_offset = (query_row * num_q_heads + q_head) * head_dim;
  const int64_t scan_slots = live_count < capacity ? live_count : capacity;
  const int64_t start_slot = start_position % capacity;

  __shared__ float reduction[128];
  __shared__ float shared_max;
  __shared__ float shared_denom;
  __shared__ float shared_weight;

  if (threadIdx.x == 0) {
    shared_max = -3.4028234663852886e38f;
    shared_denom = 0.0f;
  }
  __syncthreads();
  if (live_count <= 0 || start_position < 0 || query_position < start_position) {
    out[q_offset + dim] = 0.0f;
    return;
  }

  for (int64_t logical_slot = 0; logical_slot < scan_slots; ++logical_slot) {
    int64_t current_row = logical_slot - start_slot;
    if (current_row < 0) current_row += capacity;
    const bool from_current = current_row <= query_row && current_row < rows;
    const int64_t key_position = from_current
        ? start_position + current_row
        : token_positions[logical_slot];
    const bool visible = (from_current || evict_mask[logical_slot] == 0) &&
                         key_position >= 0 && key_position <= query_position &&
                         key_position > query_position - sliding_window;
    if (!visible) continue;
    const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    const bool physical_valid = from_current || (physical_slot >= 0 && physical_slot < capacity);
    const int64_t k_offset = from_current
        ? (current_row * num_kv_heads + kv_head) * head_dim + dim
        : (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
    const float k = from_current
        ? laguna_bf16_bits_to_float(laguna_float_to_bf16_bits(current_key[k_offset]))
        : (physical_valid ? laguna_bf16_bits_to_float(key_cache[k_offset]) : 0.0f);
    reduction[threadIdx.x] = physical_valid ? query[q_offset + dim] * k : 0.0f;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0 && physical_valid) shared_max = fmaxf(shared_max, reduction[0] * scale);
    __syncthreads();
  }

  float output_acc = 0.0f;
  for (int64_t logical_slot = 0; logical_slot < scan_slots; ++logical_slot) {
    int64_t current_row = logical_slot - start_slot;
    if (current_row < 0) current_row += capacity;
    const bool from_current = current_row <= query_row && current_row < rows;
    const int64_t key_position = from_current
        ? start_position + current_row
        : token_positions[logical_slot];
    const bool visible = (from_current || evict_mask[logical_slot] == 0) &&
                         key_position >= 0 && key_position <= query_position &&
                         key_position > query_position - sliding_window;
    if (!visible) continue;
    const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    const bool physical_valid = from_current || (physical_slot >= 0 && physical_slot < capacity);
    const int64_t kv_offset = from_current
        ? (current_row * num_kv_heads + kv_head) * head_dim + dim
        : (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
    const float k = from_current
        ? laguna_bf16_bits_to_float(laguna_float_to_bf16_bits(current_key[kv_offset]))
        : (physical_valid ? laguna_bf16_bits_to_float(key_cache[kv_offset]) : 0.0f);
    reduction[threadIdx.x] = physical_valid ? query[q_offset + dim] * k : 0.0f;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      shared_weight = physical_valid ? expf(reduction[0] * scale - shared_max) : 0.0f;
      shared_denom += shared_weight;
    }
    __syncthreads();
    if (physical_valid) {
      const float v = from_current
          ? laguna_bf16_bits_to_float(laguna_float_to_bf16_bits(current_value[kv_offset]))
          : laguna_bf16_bits_to_float(value_cache[kv_offset]);
      output_acc += shared_weight * v;
    }
    __syncthreads();
  }
  out[q_offset + dim] = output_acc / fmaxf(shared_denom, 1.0e-20f);
}

__device__ __forceinline__ float laguna_wave32_sum_128_exact(
    float product_0,
    float product_32,
    float product_64,
    float product_96) {
  // Match the baseline 128-thread shared tree: stride 64 pairs 0+64 and
  // 32+96, stride 32 combines those pairs, then one wave finishes 16..1.
  float sum = laguna_ordered_add_f32(
      laguna_ordered_add_f32(product_0, product_64),
      laguna_ordered_add_f32(product_32, product_96));
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum = laguna_ordered_add_f32(sum, __shfl_down(sum, offset));
  }
  return sum;
}

__global__ __launch_bounds__(128) void laguna_swa_attention_decode_token4_exact_bf16_kernel(
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
  const int64_t lane = dim & 31;
  const int wave = static_cast<int>(dim >> 5);
  if (q_head >= num_q_heads || dim >= head_dim) return;

  const int64_t live_count = live_counts[0];
  const int64_t query_position = row_positions[0];
  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  const int64_t q_offset = q_head * head_dim;
  const int64_t scan_slots = live_count < capacity ? live_count : capacity;
  if (live_count <= 0 || query_position < 0) {
    out[q_offset + dim] = 0.0f;
    return;
  }

  // One block still owns one query head, so denominator and output additions
  // remain in baseline logical-slot order. Four waves compute four independent
  // 128-D dots concurrently with the exact stride64..1 reduction tree. Scores
  // and physical slots stay in LDS, eliminating the baseline's second K/dot
  // pass while retaining its BF16 loads and softmax arithmetic boundaries.
  extern __shared__ uint8_t shared_raw[];
  float* scores = reinterpret_cast<float*>(shared_raw);
  int32_t* physical_slots = reinterpret_cast<int32_t*>(scores + capacity);
  float* batch_weights = reinterpret_cast<float*>(physical_slots + capacity);
  float* shared_max = batch_weights + 4;
  float* shared_denom = shared_max + 1;

  float q_values[4];
#pragma unroll
  for (int part = 0; part < 4; ++part) {
    q_values[part] = query[q_offset + lane + part * 32];
  }
  if (dim == 0) {
    *shared_max = -3.4028234663852886e38f;
    *shared_denom = 0.0f;
  }
  __syncthreads();

  for (int64_t slot_base = 0; slot_base < scan_slots; slot_base += 4) {
    const int64_t logical_slot = slot_base + wave;
    bool physical_valid = false;
    int64_t physical_slot = -1;
    if (logical_slot < scan_slots) {
      const int64_t key_position = token_positions[logical_slot];
      const bool visible = evict_mask[logical_slot] == 0 &&
                           key_position >= 0 &&
                           key_position <= query_position &&
                           key_position > query_position - sliding_window;
      if (visible) {
        physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
        physical_valid = physical_slot >= 0 && physical_slot < capacity;
      }
    }

    float products[4] = {};
    if (physical_valid) {
      const int64_t key_base =
          (physical_slot * num_kv_heads + kv_head) * head_dim;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const int64_t key_dim = lane + part * 32;
        products[part] = laguna_ordered_mul_f32(
            q_values[part],
            laguna_bf16_bits_to_float(key_cache[key_base + key_dim]));
      }
    }
    const float dot = laguna_wave32_sum_128_exact(
        products[0], products[1], products[2], products[3]);
    if (lane == 0 && logical_slot < scan_slots) {
      // Store the unscaled dot. The baseline max pass rounds ``dot * scale``
      // independently, while its weight pass may contract ``dot * scale -
      // max``. Replaying those source expressions below preserves both paths.
      scores[logical_slot] =
          physical_valid ? dot : -3.4028234663852886e38f;
      physical_slots[logical_slot] =
          physical_valid ? static_cast<int32_t>(physical_slot) : -1;
    }
    __syncthreads();
    if (dim == 0) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int64_t slot = slot_base + item;
        if (slot < scan_slots && physical_slots[slot] >= 0) {
          *shared_max = fmaxf(*shared_max, scores[slot] * scale);
        }
      }
    }
    __syncthreads();
  }

  float output_acc = 0.0f;
  for (int64_t slot_base = 0; slot_base < scan_slots; slot_base += 4) {
    if (dim == 0) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int64_t slot = slot_base + item;
        const float weight =
            slot < scan_slots && physical_slots[slot] >= 0
                ? expf(scores[slot] * scale - *shared_max)
                : 0.0f;
        batch_weights[item] = weight;
        if (slot < scan_slots && physical_slots[slot] >= 0) {
          *shared_denom = laguna_ordered_add_f32(*shared_denom, weight);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int64_t slot = slot_base + item;
      if (slot < scan_slots) {
        const int32_t physical_slot = physical_slots[slot];
        if (physical_slot >= 0) {
          const int64_t value_offset =
              (static_cast<int64_t>(physical_slot) * num_kv_heads + kv_head) *
                  head_dim +
              dim;
          output_acc = fmaf(
              batch_weights[item],
              laguna_bf16_bits_to_float(value_cache[value_offset]),
              output_acc);
        }
      }
    }
    __syncthreads();
  }

  out[q_offset + dim] = output_acc / fmaxf(*shared_denom, 1.0e-20f);
}

__global__ __launch_bounds__(256) void laguna_swa_attention_decode_token8_exact_bf16_kernel(
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
  const int64_t thread = static_cast<int64_t>(threadIdx.x);
  const int64_t lane = thread & 31;
  const int wave = static_cast<int>(thread >> 5);
  const bool output_thread = thread < head_dim;
  if (q_head >= num_q_heads) return;

  const int64_t live_count = live_counts[0];
  const int64_t query_position = row_positions[0];
  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  const int64_t q_offset = q_head * head_dim;
  const int64_t scan_slots = live_count < capacity ? live_count : capacity;
  if (live_count <= 0 || query_position < 0) {
    if (output_thread) out[q_offset + thread] = 0.0f;
    return;
  }

  // Eight waves contract score/max and weight/value batches without changing
  // any per-slot arithmetic. Threads 128..255 stay value-arithmetic-idle but
  // must participate in every barrier so the lower 128 dimensions preserve
  // the token4 logical-slot FMA order exactly.
  extern __shared__ uint8_t shared_raw[];
  float* scores = reinterpret_cast<float*>(shared_raw);
  int32_t* physical_slots = reinterpret_cast<int32_t*>(scores + capacity);
  float* batch_weights = reinterpret_cast<float*>(physical_slots + capacity);
  float* shared_max = batch_weights + 8;
  float* shared_denom = shared_max + 1;

  float q_values[4];
#pragma unroll
  for (int part = 0; part < 4; ++part) {
    q_values[part] = query[q_offset + lane + part * 32];
  }
  if (thread == 0) {
    *shared_max = -3.4028234663852886e38f;
    *shared_denom = 0.0f;
  }
  __syncthreads();

  for (int64_t slot_base = 0; slot_base < scan_slots; slot_base += 8) {
    const int64_t logical_slot = slot_base + wave;
    bool physical_valid = false;
    int64_t physical_slot = -1;
    if (logical_slot < scan_slots) {
      const int64_t key_position = token_positions[logical_slot];
      const bool visible = evict_mask[logical_slot] == 0 &&
                           key_position >= 0 &&
                           key_position <= query_position &&
                           key_position > query_position - sliding_window;
      if (visible) {
        physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
        physical_valid = physical_slot >= 0 && physical_slot < capacity;
      }
    }

    float products[4] = {};
    if (physical_valid) {
      const int64_t key_base =
          (physical_slot * num_kv_heads + kv_head) * head_dim;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const int64_t key_dim = lane + part * 32;
        products[part] = laguna_ordered_mul_f32(
            q_values[part],
            laguna_bf16_bits_to_float(key_cache[key_base + key_dim]));
      }
    }
    const float dot = laguna_wave32_sum_128_exact(
        products[0], products[1], products[2], products[3]);
    if (lane == 0 && logical_slot < scan_slots) {
      scores[logical_slot] =
          physical_valid ? dot : -3.4028234663852886e38f;
      physical_slots[logical_slot] =
          physical_valid ? static_cast<int32_t>(physical_slot) : -1;
    }
    __syncthreads();
    if (thread == 0) {
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        const int64_t slot = slot_base + item;
        if (slot < scan_slots && physical_slots[slot] >= 0) {
          *shared_max = fmaxf(*shared_max, scores[slot] * scale);
        }
      }
    }
    __syncthreads();
  }

  float output_acc = 0.0f;
  for (int64_t slot_base = 0; slot_base < scan_slots; slot_base += 8) {
    if (thread == 0) {
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        const int64_t slot = slot_base + item;
        const float weight =
            slot < scan_slots && physical_slots[slot] >= 0
                ? expf(scores[slot] * scale - *shared_max)
                : 0.0f;
        batch_weights[item] = weight;
        if (slot < scan_slots && physical_slots[slot] >= 0) {
          *shared_denom = laguna_ordered_add_f32(*shared_denom, weight);
        }
      }
    }
    __syncthreads();
    if (output_thread) {
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        const int64_t slot = slot_base + item;
        if (slot < scan_slots) {
          const int32_t physical_slot = physical_slots[slot];
          if (physical_slot >= 0) {
            const int64_t value_offset =
                (static_cast<int64_t>(physical_slot) * num_kv_heads + kv_head) *
                    head_dim +
                thread;
            output_acc = fmaf(
                batch_weights[item],
                laguna_bf16_bits_to_float(value_cache[value_offset]),
                output_acc);
          }
        }
      }
    }
    __syncthreads();
  }

  if (output_thread) {
    out[q_offset + thread] =
        output_acc / fmaxf(*shared_denom, 1.0e-20f);
  }
}

__global__ __launch_bounds__(32) void laguna_swa_attention_prefill_wave32_exact_bf16_kernel(
    const float* query,
    const float* current_key,
    const float* current_value,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t sliding_window,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t query_row = static_cast<int64_t>(blockIdx.y);
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t lane = static_cast<int64_t>(threadIdx.x);
  if (query_row >= rows || q_head >= num_q_heads || lane >= 32) return;

  const int64_t start_position = row_positions[0];
  const int64_t query_position = start_position + query_row;
  const int64_t prior_live_count = live_counts[0];
  const int64_t query_live_count =
      query_position + 1 < capacity ? query_position + 1 : capacity;
  const int64_t live_count =
      prior_live_count > query_live_count ? prior_live_count : query_live_count;
  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  const int64_t q_offset = (query_row * num_q_heads + q_head) * head_dim;
  const int64_t scan_slots = live_count < capacity ? live_count : capacity;
  const int64_t start_slot = start_position % capacity;
  if (live_count <= 0 || start_position < 0 || query_position < start_position) {
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      out[q_offset + lane + part * 32] = 0.0f;
    }
    return;
  }

  float max_score = -3.4028234663852886e38f;
  for (int64_t logical_slot = 0; logical_slot < scan_slots; ++logical_slot) {
    int64_t current_row = logical_slot - start_slot;
    if (current_row < 0) current_row += capacity;
    const bool from_current = current_row <= query_row && current_row < rows;
    const int64_t key_position = from_current
        ? start_position + current_row
        : token_positions[logical_slot];
    const bool visible = (from_current || evict_mask[logical_slot] == 0) &&
                         key_position >= 0 && key_position <= query_position &&
                         key_position > query_position - sliding_window;
    if (!visible) continue;
    const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    const bool physical_valid =
        from_current || (physical_slot >= 0 && physical_slot < capacity);
    const int64_t key_base = from_current
        ? (current_row * num_kv_heads + kv_head) * head_dim
        : (physical_slot * num_kv_heads + kv_head) * head_dim;
    float products[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int64_t dim = lane + part * 32;
      const float key_value = from_current
          ? laguna_bf16_bits_to_float(
                laguna_float_to_bf16_bits(current_key[key_base + dim]))
          : (physical_valid
                 ? laguna_bf16_bits_to_float(key_cache[key_base + dim])
                 : 0.0f);
      products[part] = physical_valid
          ? laguna_ordered_mul_f32(query[q_offset + dim], key_value)
          : 0.0f;
    }
    const float dot = laguna_wave32_sum_128_exact(
        products[0], products[1], products[2], products[3]);
    if (lane == 0 && physical_valid) {
      max_score = fmaxf(max_score, dot * scale);
    }
  }
  max_score = __shfl(max_score, 0);

  float denominator = 0.0f;
  float output_acc[4] = {};
  for (int64_t logical_slot = 0; logical_slot < scan_slots; ++logical_slot) {
    int64_t current_row = logical_slot - start_slot;
    if (current_row < 0) current_row += capacity;
    const bool from_current = current_row <= query_row && current_row < rows;
    const int64_t key_position = from_current
        ? start_position + current_row
        : token_positions[logical_slot];
    const bool visible = (from_current || evict_mask[logical_slot] == 0) &&
                         key_position >= 0 && key_position <= query_position &&
                         key_position > query_position - sliding_window;
    if (!visible) continue;
    const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    const bool physical_valid =
        from_current || (physical_slot >= 0 && physical_slot < capacity);
    const int64_t kv_base = from_current
        ? (current_row * num_kv_heads + kv_head) * head_dim
        : (physical_slot * num_kv_heads + kv_head) * head_dim;
    float products[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int64_t dim = lane + part * 32;
      const float key_value = from_current
          ? laguna_bf16_bits_to_float(
                laguna_float_to_bf16_bits(current_key[kv_base + dim]))
          : (physical_valid
                 ? laguna_bf16_bits_to_float(key_cache[kv_base + dim])
                 : 0.0f);
      products[part] = physical_valid
          ? laguna_ordered_mul_f32(query[q_offset + dim], key_value)
          : 0.0f;
    }
    const float dot = laguna_wave32_sum_128_exact(
        products[0], products[1], products[2], products[3]);
    float weight = 0.0f;
    if (lane == 0) {
      weight = physical_valid ? expf(dot * scale - max_score) : 0.0f;
      denominator += weight;
    }
    weight = __shfl(weight, 0);
    if (physical_valid) {
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const int64_t dim = lane + part * 32;
        const float value = from_current
            ? laguna_bf16_bits_to_float(
                  laguna_float_to_bf16_bits(current_value[kv_base + dim]))
            : laguna_bf16_bits_to_float(value_cache[kv_base + dim]);
        output_acc[part] += weight * value;
      }
    }
  }
  denominator = __shfl(denominator, 0);
  const float safe_denominator = fmaxf(denominator, 1.0e-20f);
#pragma unroll
  for (int part = 0; part < 4; ++part) {
    out[q_offset + lane + part * 32] = output_acc[part] / safe_denominator;
  }
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

extern "C" int hipengine_laguna_global_write_kv_rows_f32_bf16_spans(
    const float* key,
    const float* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    int64_t* live_counts,
    int64_t* token_positions,
    uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    hipStream_t stream) {
  if (rows <= 0 || rows > capacity || capacity <= 0 || block_size != 256 ||
      block_table_len <= 0 || (capacity + block_size - 1) / block_size > block_table_len ||
      num_kv_heads <= 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t width = num_kv_heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((width + threads - 1) / threads);
  hipLaunchKernelGGL(
      laguna_global_write_kv_rows_f32_bf16_kernel,
      dim3(static_cast<unsigned int>(blocks), static_cast<unsigned int>(rows)),
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
      rows,
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

extern "C" int hipengine_laguna_global_attention_prefill_bf16_spans(
    const float* query,
    const float* current_key,
    const float* current_value,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 || rows > capacity || capacity <= 0 || block_size != 256 ||
      block_table_len <= 0 || (capacity + block_size - 1) / block_size > block_table_len ||
      num_q_heads <= 0 || num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int warps = threads / 32;
  const size_t shared_bytes = static_cast<size_t>(capacity + warps + head_dim) * sizeof(float);
  hipLaunchKernelGGL(
      laguna_global_attention_prefill_bf16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)),
      dim3(threads),
      shared_bytes,
      stream,
      query,
      current_key,
      current_value,
      key_cache,
      value_cache,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      rows,
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

extern "C" int hipengine_laguna_swa_write_kv_rows_f32_bf16_spans(
    const float* key,
    const float* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    int64_t* live_counts,
    int64_t* token_positions,
    uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t num_kv_heads,
    int64_t head_dim,
    hipStream_t stream) {
  if (rows <= 0 || rows > capacity || capacity <= 0 || num_kv_heads <= 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t width = num_kv_heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((width + threads - 1) / threads);
  hipLaunchKernelGGL(
      laguna_swa_write_kv_rows_f32_bf16_kernel,
      dim3(static_cast<unsigned int>(blocks), static_cast<unsigned int>(rows)),
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
      rows,
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

extern "C" int hipengine_laguna_swa_attention_decode_token4_exact_bf16_spans(
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
  const size_t shared_bytes =
      static_cast<size_t>(capacity) * (sizeof(float) + sizeof(int32_t)) +
      6 * sizeof(float);
  hipLaunchKernelGGL(
      laguna_swa_attention_decode_token4_exact_bf16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(128),
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
      sliding_window,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_swa_attention_decode_token8_exact_bf16_spans(
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
  const size_t shared_bytes =
      static_cast<size_t>(capacity) * (sizeof(float) + sizeof(int32_t)) +
      10 * sizeof(float);
  hipLaunchKernelGGL(
      laguna_swa_attention_decode_token8_exact_bf16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(256),
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
      sliding_window,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_swa_attention_prefill_wave32_exact_bf16_spans(
    const float* query,
    const float* current_key,
    const float* current_value,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t sliding_window,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 || rows > capacity || capacity <= 0 || sliding_window <= 0 ||
      sliding_window > capacity || num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_swa_attention_prefill_wave32_exact_bf16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)),
      dim3(32),
      0,
      stream,
      query,
      current_key,
      current_value,
      key_cache,
      value_cache,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      rows,
      capacity,
      sliding_window,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_swa_attention_prefill_bf16_spans(
    const float* query,
    const float* current_key,
    const float* current_value,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t sliding_window,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 || rows > capacity || capacity <= 0 || sliding_window <= 0 ||
      sliding_window > capacity || num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_swa_attention_prefill_bf16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)),
      dim3(128),
      0,
      stream,
      query,
      current_key,
      current_value,
      key_cache,
      value_cache,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      rows,
      capacity,
      sliding_window,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}
