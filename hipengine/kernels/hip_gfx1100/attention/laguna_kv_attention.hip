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

__device__ inline float laguna_softplus_f32(float value) {
  return value > 20.0f ? value : log1pf(expf(value));
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

__device__ __forceinline__ void laguna_head_rmsnorm_partial_rotary_f32_weight_row(
    const float* __restrict__ src,
    const float* __restrict__ weight,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ dst,
    float eps,
    int64_t head_dim,
    int64_t rotary_dim,
    float* partial) {
  const int64_t tid = threadIdx.x;
  float sum = 0.0f;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float value = src[dim];
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  const int64_t half_rotary = rotary_dim / 2;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float value = src[dim] * inv_rms * weight[dim];
    float out_value = value;
    if (dim < rotary_dim) {
      const int64_t pair_dim = dim < half_rotary ? dim + half_rotary : dim - half_rotary;
      const float paired = src[pair_dim] * inv_rms * weight[pair_dim];
      const float rotated = dim < half_rotary ? -paired : paired;
      out_value = value * cos[dim] + rotated * sin[dim];
    }
    dst[dim] = out_value;
  }
}

__device__ __forceinline__ void laguna_head_rmsnorm_partial_rotary_write_kv_f32_weight_row(
    const float* __restrict__ src,
    const float* __restrict__ value_row,
    const float* __restrict__ weight,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ dst,
    uint16_t* __restrict__ key_cache,
    uint16_t* __restrict__ value_cache,
    float eps,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t cache_offset,
    bool cache_valid,
    float* partial) {
  const int64_t tid = threadIdx.x;
  float sum = 0.0f;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float src_value = src[dim];
    sum += src_value * src_value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  const int64_t half_rotary = rotary_dim / 2;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float value = src[dim] * inv_rms * weight[dim];
    float out_value = value;
    if (dim < rotary_dim) {
      const int64_t pair_dim = dim < half_rotary ? dim + half_rotary : dim - half_rotary;
      const float paired = src[pair_dim] * inv_rms * weight[pair_dim];
      const float rotated = dim < half_rotary ? -paired : paired;
      out_value = value * cos[dim] + rotated * sin[dim];
    }
    dst[dim] = out_value;
    if (cache_valid) {
      key_cache[cache_offset + dim] = laguna_float_to_bf16_bits(out_value);
      value_cache[cache_offset + dim] = laguna_float_to_bf16_bits(value_row[dim]);
    }
  }
}

template <bool WriteKV>
__device__ __forceinline__ void laguna_head_rmsnorm_partial_rotary_wave0_tree_f32_weight_row(
    const float* __restrict__ src,
    const float* __restrict__ value_row,
    const float* __restrict__ weight,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ dst,
    uint16_t* __restrict__ key_cache,
    uint16_t* __restrict__ value_cache,
    float eps,
    int64_t rotary_dim,
    int64_t cache_offset,
    bool cache_valid,
    float* partial) {
  const int64_t tid = threadIdx.x;
  constexpr int64_t head_dim = 128;
  constexpr int64_t threads = 256;
  float sum = 0.0f;
  for (int64_t dim = tid; dim < head_dim; dim += threads) {
    const float src_value = src[dim];
    sum += src_value * src_value;
  }
  partial[tid] = sum;
  __syncthreads();
  float total;
  if (tid < 32) {
    const float p0 = partial[tid] + partial[tid + 128];
    const float p64 = partial[tid + 64] + partial[tid + 192];
    const float left = p0 + p64;
    const float p32 = partial[tid + 32] + partial[tid + 160];
    const float p96 = partial[tid + 96] + partial[tid + 224];
    const float right = p32 + p96;
    total = left + right;
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
      total += __shfl_down(total, stride, 32);
    }
    if (tid == 0) {
      partial[0] = total;
    }
  }
  __syncthreads();
  total = partial[0];
  const float inv_rms = rsqrtf(total / 128.0f + eps);
  const int64_t half_rotary = rotary_dim / 2;
  for (int64_t dim = tid; dim < head_dim; dim += threads) {
    const float value = src[dim] * inv_rms * weight[dim];
    float out_value = value;
    if (dim < rotary_dim) {
      const int64_t pair_dim = dim < half_rotary ? dim + half_rotary : dim - half_rotary;
      const float paired = src[pair_dim] * inv_rms * weight[pair_dim];
      const float rotated = dim < half_rotary ? -paired : paired;
      out_value = value * cos[dim] + rotated * sin[dim];
    }
    dst[dim] = out_value;
    if constexpr (WriteKV) {
      if (cache_valid) {
        key_cache[cache_offset + dim] = laguna_float_to_bf16_bits(out_value);
        value_cache[cache_offset + dim] = laguna_float_to_bf16_bits(value_row[dim]);
      }
    }
  }
}

template <bool GlobalSpans>
__global__ __launch_bounds__(256) void laguna_head_rmsnorm_partial_rotary_write_kv_f32_bf16_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ query_out,
    float* __restrict__ key_out,
    uint16_t* __restrict__ key_cache,
    uint16_t* __restrict__ value_cache,
    const int32_t* __restrict__ base_offsets,
    int64_t* __restrict__ live_counts,
    int64_t* __restrict__ token_positions,
    uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    float eps,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions) {
  const int64_t position = row_positions[0];
  if (position < 0 || position >= max_positions) return;
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= num_q_heads + num_kv_heads) return;

  int64_t logical_slot = -1;
  int64_t physical_slot = -1;
  bool cache_valid = false;
  if constexpr (GlobalSpans) {
    if (position < capacity && position / block_size < block_table_len) {
      logical_slot = position;
      physical_slot = laguna_global_physical_slot(base_offsets, position, block_size);
      cache_valid = physical_slot >= 0 && physical_slot < block_table_len * block_size;
    }
  } else {
    logical_slot = position % capacity;
    physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    cache_valid = physical_slot >= 0 && physical_slot < capacity;
  }

  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const int64_t offset = local_head * head_dim;
  const float* cos = cos_table + position * rotary_dim;
  const float* sin = sin_table + position * rotary_dim;
  extern __shared__ float partial[];
  if (is_query) {
    laguna_head_rmsnorm_partial_rotary_f32_weight_row(
        query + offset,
        q_weight,
        cos,
        sin,
        query_out + offset,
        eps,
        head_dim,
        rotary_dim,
        partial);
    return;
  }

  const int64_t cache_offset = (physical_slot * num_kv_heads + local_head) * head_dim;
  laguna_head_rmsnorm_partial_rotary_write_kv_f32_weight_row(
      key + offset,
      value + offset,
      k_weight,
      cos,
      sin,
      key_out + offset,
      key_cache,
      value_cache,
      eps,
      head_dim,
      rotary_dim,
      cache_offset,
      cache_valid,
      partial);

  if (local_head == 0 && threadIdx.x == 0 && cache_valid) {
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

__global__ __launch_bounds__(256) void laguna_global_head_rmsnorm_partial_rotary_write_kv_wave0_tree_f32_bf16_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ query_out,
    float* __restrict__ key_out,
    uint16_t* __restrict__ key_cache,
    uint16_t* __restrict__ value_cache,
    const int32_t* __restrict__ base_offsets,
    int64_t* __restrict__ live_counts,
    int64_t* __restrict__ token_positions,
    uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    float eps,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    int64_t max_positions) {
  const int64_t position = row_positions[0];
  if (position < 0 || position >= max_positions) return;
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  if (head >= num_q_heads + num_kv_heads) return;

  int64_t logical_slot = -1;
  int64_t physical_slot = -1;
  bool cache_valid = false;
  if (position < capacity && position / block_size < block_table_len) {
    logical_slot = position;
    physical_slot = laguna_global_physical_slot(base_offsets, position, block_size);
    cache_valid = physical_slot >= 0 && physical_slot < block_table_len * block_size;
  }

  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  constexpr int64_t head_dim = 128;
  const int64_t offset = local_head * head_dim;
  const float* cos = cos_table + position * rotary_dim;
  const float* sin = sin_table + position * rotary_dim;
  extern __shared__ float partial[];
  if (is_query) {
    laguna_head_rmsnorm_partial_rotary_wave0_tree_f32_weight_row<false>(
        query + offset,
        nullptr,
        q_weight,
        cos,
        sin,
        query_out + offset,
        nullptr,
        nullptr,
        eps,
        rotary_dim,
        0,
        false,
        partial);
    return;
  }

  const int64_t cache_offset = (physical_slot * num_kv_heads + local_head) * head_dim;
  laguna_head_rmsnorm_partial_rotary_wave0_tree_f32_weight_row<true>(
      key + offset,
      value + offset,
      k_weight,
      cos,
      sin,
      key_out + offset,
      key_cache,
      value_cache,
      eps,
      rotary_dim,
      cache_offset,
      cache_valid,
      partial);

  if (local_head == 0 && threadIdx.x == 0 && cache_valid) {
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

__global__ void laguna_global_attention_decode_single_page_bf16_kernel(
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
  if (context_len <= 0 || context_len > block_size || query_position < 0) {
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
          static_cast<int64_t>(base_offsets[0]) * block_size + token;
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
          static_cast<int64_t>(base_offsets[0]) * block_size + token;
      const int64_t v_offset =
          (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
      acc += weight * laguna_bf16_bits_to_float(value_cache[v_offset]);
    }
    out[q_offset + dim] = acc;
  }
}

__global__ void laguna_global_attention_decode_single_page_softplus_gate_bf16_kernel(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const float* gate,
    uint16_t* gated_out,
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
  if (context_len <= 0 || context_len > block_size || query_position < 0) {
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
          static_cast<int64_t>(base_offsets[0]) * block_size + token;
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
          static_cast<int64_t>(base_offsets[0]) * block_size + token;
      const int64_t v_offset =
          (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
      acc += weight * laguna_bf16_bits_to_float(value_cache[v_offset]);
    }
    const float context_value = acc;
    out[q_offset + dim] = context_value;
    gated_out[q_offset + dim] = laguna_float_to_bf16_bits(
        context_value * laguna_softplus_f32(gate[q_head]));
  }
}

constexpr int LAGUNA_SPLIT_SCORE_TILE = 16;

// Exact split topology: independent wave32 score blocks expose token-level
// parallelism, then a local256 reducer replays the retained global softmax and
// value accumulation order. Scratch is caller-owned and reusable.
__global__ __launch_bounds__(32) void laguna_global_attention_split_exact_scores_kernel(
    const float* query,
    const uint16_t* key_cache,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t block_size,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t logical_slot = static_cast<int64_t>(blockIdx.y);
  if (q_head >= num_q_heads || logical_slot >= scan_slots) return;

  const int64_t context_len = live_counts[0] < capacity ? live_counts[0] : capacity;
  const int64_t query_position = row_positions[0];
  const int64_t scratch_idx = q_head * scratch_stride + logical_slot;
  const int64_t key_position =
      logical_slot < context_len ? token_positions[logical_slot] : -1;
  const bool visible = logical_slot < context_len && query_position >= 0 &&
                       evict_mask[logical_slot] == 0 && key_position >= 0 &&
                       key_position <= query_position;
  int64_t physical_slot = -1;
  if (visible) {
    physical_slot =
        laguna_global_physical_slot(base_offsets, logical_slot, block_size);
  }

  const int lane = threadIdx.x;
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const int64_t q_offset = q_head * head_dim;
  float acc = 0.0f;
  if (physical_slot >= 0) {
    const int64_t k_offset =
        (physical_slot * num_kv_heads + kv_head) * head_dim + lane * 4;
    const int64_t q_lane = q_offset + lane * 4;
    acc += query[q_lane + 0] * laguna_bf16_bits_to_float(key_cache[k_offset + 0]);
    acc += query[q_lane + 1] * laguna_bf16_bits_to_float(key_cache[k_offset + 1]);
    acc += query[q_lane + 2] * laguna_bf16_bits_to_float(key_cache[k_offset + 2]);
    acc += query[q_lane + 3] * laguna_bf16_bits_to_float(key_cache[k_offset + 3]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down(acc, offset);
  }
  if (lane == 0) {
    score_scratch[scratch_idx] =
        physical_slot >= 0 ? acc * scale : -3.4028234663852886e38f;
    physical_scratch[scratch_idx] = static_cast<int32_t>(physical_slot);
  }
}

__global__ __launch_bounds__(256) void laguna_global_attention_split_exact_reduce_kernel(
    const uint16_t* value_cache,
    float* score_scratch,
    const int32_t* physical_scratch,
    float* out,
    const int64_t* live_counts,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  if (q_head >= num_q_heads) return;
  int64_t context_len = live_counts[0] < capacity ? live_counts[0] : capacity;
  context_len = context_len < scan_slots ? context_len : scan_slots;
  const int64_t query_position = row_positions[0];
  if (context_len <= 0 || query_position < 0) return;

  constexpr int NUM_WARPS = 8;
  __shared__ float warp_buf[NUM_WARPS];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t scratch_base = q_head * scratch_stride;

  float local_max = -3.4028234663852886e38f;
  for (int64_t token = warp; token < context_len; token += NUM_WARPS) {
    if (lane == 0) {
      local_max = fmaxf(local_max, score_scratch[scratch_base + token]);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down(local_max, offset));
  }
  if (lane == 0) warp_buf[warp] = local_max;
  __syncthreads();
  if (threadIdx.x == 0) {
    float max_score = warp_buf[0];
#pragma unroll
    for (int item = 1; item < NUM_WARPS; ++item) {
      max_score = fmaxf(max_score, warp_buf[item]);
    }
    warp_buf[0] = max_score;
  }
  __syncthreads();
  const float max_score = warp_buf[0];

  float local_sum = 0.0f;
  for (int64_t token = warp; token < context_len; token += NUM_WARPS) {
    if (lane == 0) {
      const int64_t idx = scratch_base + token;
      const bool visible = physical_scratch[idx] >= 0;
      const float weight = visible ? expf(score_scratch[idx] - max_score) : 0.0f;
      score_scratch[idx] = weight;
      local_sum += weight;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down(local_sum, offset);
  }
  if (lane == 0) warp_buf[warp] = local_sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total_sum = warp_buf[0];
#pragma unroll
    for (int item = 1; item < NUM_WARPS; ++item) total_sum += warp_buf[item];
    warp_buf[0] = 1.0f / fmaxf(total_sum, 1.0e-20f);
  }
  __syncthreads();
  const float inv_denom = warp_buf[0];
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const int64_t q_offset = q_head * head_dim;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < context_len; ++token) {
      const int64_t idx = scratch_base + token;
      const float weight = score_scratch[idx] * inv_denom;
      if (!(weight > 0.0f)) continue;
      const int64_t physical_slot = physical_scratch[idx];
      const int64_t value_offset =
          (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
      acc += weight * laguna_bf16_bits_to_float(value_cache[value_offset]);
    }
    out[q_offset + dim] = acc;
  }
}

__global__ __launch_bounds__(256) void laguna_global_attention_split_exact_gated_reduce_kernel(
    const uint16_t* value_cache,
    float* score_scratch,
    const int32_t* physical_scratch,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    const int64_t* live_counts,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  if (q_head >= num_q_heads) return;
  int64_t context_len = live_counts[0] < capacity ? live_counts[0] : capacity;
  context_len = context_len < scan_slots ? context_len : scan_slots;
  const int64_t query_position = row_positions[0];
  if (context_len <= 0 || query_position < 0) return;

  constexpr int NUM_WARPS = 8;
  __shared__ float warp_buf[NUM_WARPS];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t scratch_base = q_head * scratch_stride;

  float local_max = -3.4028234663852886e38f;
  for (int64_t token = warp; token < context_len; token += NUM_WARPS) {
    if (lane == 0) {
      local_max = fmaxf(local_max, score_scratch[scratch_base + token]);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down(local_max, offset));
  }
  if (lane == 0) warp_buf[warp] = local_max;
  __syncthreads();
  if (threadIdx.x == 0) {
    float max_score = warp_buf[0];
#pragma unroll
    for (int item = 1; item < NUM_WARPS; ++item) {
      max_score = fmaxf(max_score, warp_buf[item]);
    }
    warp_buf[0] = max_score;
  }
  __syncthreads();
  const float max_score = warp_buf[0];

  float local_sum = 0.0f;
  for (int64_t token = warp; token < context_len; token += NUM_WARPS) {
    if (lane == 0) {
      const int64_t idx = scratch_base + token;
      const bool visible = physical_scratch[idx] >= 0;
      const float weight = visible ? expf(score_scratch[idx] - max_score) : 0.0f;
      score_scratch[idx] = weight;
      local_sum += weight;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down(local_sum, offset);
  }
  if (lane == 0) warp_buf[warp] = local_sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total_sum = warp_buf[0];
#pragma unroll
    for (int item = 1; item < NUM_WARPS; ++item) total_sum += warp_buf[item];
    warp_buf[0] = 1.0f / fmaxf(total_sum, 1.0e-20f);
  }
  __syncthreads();
  const float inv_denom = warp_buf[0];
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const int64_t q_offset = q_head * head_dim;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < context_len; ++token) {
      const int64_t idx = scratch_base + token;
      const float weight = score_scratch[idx] * inv_denom;
      if (!(weight > 0.0f)) continue;
      const int64_t physical_slot = physical_scratch[idx];
      const int64_t value_offset =
          (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
      acc += weight * laguna_bf16_bits_to_float(value_cache[value_offset]);
    }
    const int64_t output_idx = q_offset + dim;
    out[output_idx] = acc;
    gated_out[output_idx] = laguna_float_to_bf16_bits(
        acc * laguna_softplus_f32(gate[q_head]));
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

template <bool kDenseInitialMetadata = false>
__global__ void laguna_global_attention_prefill_cached_exact_bf16_kernel(
    const float* query,
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
    int64_t expected_start_position,
    float scale) {
  const int64_t query_row = static_cast<int64_t>(blockIdx.y);
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  if (query_row >= rows || q_head >= num_q_heads) return;
  const int64_t start_position = row_positions[0];
  const int64_t appended_count =
      live_counts[0] < capacity ? live_counts[0] : capacity;
  if (start_position != expected_start_position ||
      appended_count < start_position + rows) {
    return;
  }
  const int64_t query_position = start_position + query_row;
  const int64_t context_len =
      query_position + 1 < capacity ? query_position + 1 : capacity;
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
    bool visible;
    int64_t physical_slot;
    if constexpr (kDenseInitialMetadata) {
      visible = true;
      physical_slot = token;
    } else {
      const int64_t current_row = token - start_position;
      const bool from_current =
          current_row >= 0 && current_row <= query_row && current_row < rows;
      const int64_t key_position =
          from_current ? token : token_positions[token];
      visible = from_current ||
          (evict_mask[token] == 0 && key_position >= 0 &&
           key_position <= query_position);
      physical_slot =
          laguna_global_physical_slot(base_offsets, token, block_size);
    }
    float acc = 0.0f;
    if (visible) {
      const int64_t k_offset =
          (physical_slot * num_kv_heads + kv_head) * head_dim;
      for (int64_t dim = lane_id * 4; dim < head_dim; dim += warpSize * 4) {
        const float k0 =
            laguna_bf16_bits_to_float(key_cache[k_offset + dim + 0]);
        const float k1 =
            laguna_bf16_bits_to_float(key_cache[k_offset + dim + 1]);
        const float k2 =
            laguna_bf16_bits_to_float(key_cache[k_offset + dim + 2]);
        const float k3 =
            laguna_bf16_bits_to_float(key_cache[k_offset + dim + 3]);
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
      bool visible;
      if constexpr (kDenseInitialMetadata) {
        visible = true;
      } else {
        const int64_t current_row = token - start_position;
        const bool from_current =
            current_row >= 0 && current_row <= query_row && current_row < rows;
        visible = from_current ||
            (evict_mask[token] == 0 && token_positions[token] >= 0 &&
             token_positions[token] <= query_position);
      }
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
      const int64_t physical_slot = kDenseInitialMetadata
          ? token
          : laguna_global_physical_slot(base_offsets, token, block_size);
      const int64_t v_offset =
          (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
      const float v = laguna_bf16_bits_to_float(value_cache[v_offset]);
      acc += weight * v;
    }
    out[q_offset + dim] = acc;
  }
}

// H6N preserves H6A's local256 exact association while sizing the dynamic
// score arena to the strict first-fill bound rather than the C4096 KV ABI.
__global__ void laguna_global_attention_prefill_dense_initial_fixed512_cached_exact_bf16_kernel(
    const float* query,
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
    int64_t expected_start_position,
    float scale) {
  constexpr int64_t kScoreCapacity = 512;
  const int64_t query_row = static_cast<int64_t>(blockIdx.y);
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  if (query_row >= rows || q_head >= num_q_heads) return;
  const int64_t start_position = row_positions[0];
  const int64_t appended_count =
      live_counts[0] < capacity ? live_counts[0] : capacity;
  if (start_position != expected_start_position ||
      appended_count < start_position + rows) {
    return;
  }
  const int64_t query_position = start_position + query_row;
  const int64_t context_len =
      query_position + 1 < capacity ? query_position + 1 : capacity;
  if (context_len <= 0 || context_len > kScoreCapacity || start_position < 0 ||
      query_position < start_position) {
    return;
  }

  (void)base_offsets;
  (void)token_positions;
  (void)evict_mask;
  (void)block_size;
  extern __shared__ float shared[];
  float* scores = shared;
  const int num_warps = blockDim.x / warpSize;
  float* warp_buf = scores + kScoreCapacity;
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
    const bool visible = true;
    const int64_t physical_slot = token;
    float acc = 0.0f;
    const int64_t k_offset =
        (physical_slot * num_kv_heads + kv_head) * head_dim;
    for (int64_t dim = lane_id * 4; dim < head_dim; dim += warpSize * 4) {
      const float k0 =
          laguna_bf16_bits_to_float(key_cache[k_offset + dim + 0]);
      const float k1 =
          laguna_bf16_bits_to_float(key_cache[k_offset + dim + 1]);
      const float k2 =
          laguna_bf16_bits_to_float(key_cache[k_offset + dim + 2]);
      const float k3 =
          laguna_bf16_bits_to_float(key_cache[k_offset + dim + 3]);
      acc += q_shared[dim + 0] * k0;
      acc += q_shared[dim + 1] * k1;
      acc += q_shared[dim + 2] * k2;
      acc += q_shared[dim + 3] * k3;
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
      const float weight = expf(scores[token] - max_score);
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
      if (!(weight > 0.0f)) continue;
      const int64_t physical_slot = token;
      const int64_t v_offset =
          (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
      const float v = laguna_bf16_bits_to_float(value_cache[v_offset]);
      acc += weight * v;
    }
    out[q_offset + dim] = acc;
  }
}

// H6Z shares each dense-initial global K/V vector across four query rows while
// retaining H6N's contiguous-four product order, wave32 tree, token-mod-8
// denominator partials, serial partial0..7 sum, and token-ordered PV updates.
__global__ __launch_bounds__(32) void laguna_global_attention_prefill_qrow4_dense_initial_global_score_weight_replay_exact_bf16_kernel(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    float4* score_scratch_records,
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
    int64_t expected_start_position,
    float scale) {
  constexpr int kQueryRows = 4;
  constexpr int kScoreCapacity = 512;
  constexpr float kLowest = -3.4028234663852886e38f;

  const int64_t query_row_base =
      static_cast<int64_t>(blockIdx.y) * kQueryRows;
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t lane = static_cast<int64_t>(threadIdx.x);
  if (query_row_base >= rows || q_head >= num_q_heads || lane >= 32) return;

  const int64_t start_position = row_positions[0];
  const int64_t appended_count =
      live_counts[0] < capacity ? live_counts[0] : capacity;
  if (rows != 128 || capacity != 4096 || block_size != 256 ||
      num_q_heads != 48 || num_kv_heads != 8 || head_dim != 128 ||
      start_position != expected_start_position ||
      (start_position != 256 && start_position != 384) ||
      appended_count < start_position + rows) {
    return;
  }
  (void)base_offsets;
  (void)token_positions;
  (void)evict_mask;

  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  const int64_t workgroup_index =
      static_cast<int64_t>(blockIdx.y) * num_q_heads + q_head;
  float4* score_records =
      score_scratch_records + workgroup_index * kScoreCapacity;
  const int64_t context_len =
      start_position + query_row_base + kQueryRows;

  float max_scores[kQueryRows];
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    max_scores[row_index] = kLowest;
  }
  {
    float query_values[kQueryRows][4];
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      const int64_t query_base =
          ((query_row_base + row_index) * num_q_heads + q_head) * head_dim;
      const int64_t dim_base = lane * 4;
      query_values[row_index][0] = query[query_base + dim_base + 0];
      query_values[row_index][1] = query[query_base + dim_base + 1];
      query_values[row_index][2] = query[query_base + dim_base + 2];
      query_values[row_index][3] = query[query_base + dim_base + 3];
    }

    for (int64_t logical_slot = 0;
         logical_slot < context_len;
         ++logical_slot) {
      const int64_t cache_base =
          (logical_slot * num_kv_heads + kv_head) * head_dim + lane * 4;
      const uint16_t* key_vector = &key_cache[cache_base];
      float cached_keys[4];
      cached_keys[0] = laguna_bf16_bits_to_float(key_vector[0]);
      cached_keys[1] = laguna_bf16_bits_to_float(key_vector[1]);
      cached_keys[2] = laguna_bf16_bits_to_float(key_vector[2]);
      cached_keys[3] = laguna_bf16_bits_to_float(key_vector[3]);
      float score_values[kQueryRows] = {};
#pragma unroll
      for (int row_index = 0; row_index < kQueryRows; ++row_index) {
        float acc = 0.0f;
        acc += query_values[row_index][0] * cached_keys[0];
        acc += query_values[row_index][1] * cached_keys[1];
        acc += query_values[row_index][2] * cached_keys[2];
        acc += query_values[row_index][3] * cached_keys[3];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
          acc += __shfl_down(acc, offset);
        }
        if (lane == 0) {
          const bool visible =
              logical_slot <= start_position + query_row_base + row_index;
          const float score = visible ? acc * scale : kLowest;
          score_values[row_index] = score;
          if (visible) {
            max_scores[row_index] = fmaxf(max_scores[row_index], score);
          }
        }
      }
      if (lane == 0) {
        float4 score_record = make_float4(
            score_values[0],
            score_values[1],
            score_values[2],
            score_values[3]);
        score_records[logical_slot] = score_record;
      }
    }
  }
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    max_scores[row_index] = __shfl(max_scores[row_index], 0);
  }

  float denominator_partials[kQueryRows] = {};
  if (lane < 8) {
    for (int64_t logical_slot = lane;
         logical_slot < context_len;
         logical_slot += 8) {
      float4 score_record = score_records[logical_slot];
      const float score_values[kQueryRows] = {
          score_record.x,
          score_record.y,
          score_record.z,
          score_record.w,
      };
      float weights[kQueryRows];
#pragma unroll
      for (int row_index = 0; row_index < kQueryRows; ++row_index) {
        const bool visible =
            logical_slot <= start_position + query_row_base + row_index;
        const float weight =
            visible ? expf(score_values[row_index] - max_scores[row_index])
                    : 0.0f;
        weights[row_index] = weight;
        denominator_partials[row_index] += weight;
      }
      float4 weight_record = make_float4(
          weights[0], weights[1], weights[2], weights[3]);
      score_records[logical_slot] = weight_record;
    }
  }

  float safe_denominators[kQueryRows];
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    float total_sum = __shfl(denominator_partials[row_index], 0);
#pragma unroll
    for (int source_lane = 1; source_lane < 8; ++source_lane) {
      total_sum += __shfl(denominator_partials[row_index], source_lane);
    }
    safe_denominators[row_index] = 1.0f / fmaxf(total_sum, 1.0e-20f);
  }

  float output_acc[kQueryRows][4] = {};
  for (int64_t logical_slot = 0;
       logical_slot < context_len;
       ++logical_slot) {
    float4 weight_record = score_records[logical_slot];
    const int64_t value_base =
        (logical_slot * num_kv_heads + kv_head) * head_dim + lane * 4;
    const uint16_t* value_vector = &value_cache[value_base];
    float cached_values[4];
    cached_values[0] = laguna_bf16_bits_to_float(value_vector[0]);
    cached_values[1] = laguna_bf16_bits_to_float(value_vector[1]);
    cached_values[2] = laguna_bf16_bits_to_float(value_vector[2]);
    cached_values[3] = laguna_bf16_bits_to_float(value_vector[3]);
    float normalized_weights[kQueryRows] = {
        laguna_ordered_mul_f32(weight_record.x, safe_denominators[0]),
        laguna_ordered_mul_f32(weight_record.y, safe_denominators[1]),
        laguna_ordered_mul_f32(weight_record.z, safe_denominators[2]),
        laguna_ordered_mul_f32(weight_record.w, safe_denominators[3]),
    };
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      const float normalized_weight = normalized_weights[row_index];
      if (!(normalized_weight > 0.0f)) continue;
      output_acc[row_index][0] += normalized_weight * cached_values[0];
      output_acc[row_index][1] += normalized_weight * cached_values[1];
      output_acc[row_index][2] += normalized_weight * cached_values[2];
      output_acc[row_index][3] += normalized_weight * cached_values[3];
    }
  }

#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    const int64_t output_base =
        ((query_row_base + row_index) * num_q_heads + q_head) * head_dim +
        lane * 4;
    out[output_base + 0] = output_acc[row_index][0];
    out[output_base + 1] = output_acc[row_index][1];
    out[output_base + 2] = output_acc[row_index][2];
    out[output_base + 3] = output_acc[row_index][3];
  }
}

template <
    int kQueryRows,
    bool kCachedOnly = false,
    bool kCachedMetadata = false,
    bool kDenseInitialMetadata = false>
__global__ __launch_bounds__(32) void laguna_global_attention_prefill_qrows_online_bf16_kernel(
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
  constexpr float kLowest = -3.4028234663852886e38f;
  const int64_t query_row_base =
      static_cast<int64_t>(blockIdx.y) * kQueryRows;
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t lane = static_cast<int64_t>(threadIdx.x);
  if (query_row_base >= rows || q_head >= num_q_heads || lane >= 32) return;

  const int64_t start_position = row_positions[0];
  if (start_position < 0) return;
  if (kDenseInitialMetadata) {
    const int64_t final_position = start_position + rows - 1;
    if (final_position >= capacity ||
        token_positions[0] != 0 ||
        token_positions[final_position] != final_position ||
        evict_mask[0] != 0 ||
        evict_mask[final_position] != 0) {
      return;
    }
  }
  const int64_t prior_count =
      live_counts[0] < capacity ? live_counts[0] : capacity;
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;

  bool row_valid[kQueryRows];
  int64_t query_rows[kQueryRows];
  int64_t query_positions[kQueryRows];
  int64_t context_lens[kQueryRows];
  float query_values[kQueryRows][4];
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    const int64_t query_row = query_row_base + row_index;
    const int64_t query_position = start_position + query_row;
    const int64_t position_count = query_position + 1 < capacity
        ? query_position + 1
        : capacity;
    row_valid[row_index] = query_row < rows;
    query_rows[row_index] = query_row;
    query_positions[row_index] = query_position;
    context_lens[row_index] = row_valid[row_index]
        ? (prior_count > position_count ? prior_count : position_count)
        : 0;
    const int64_t query_base =
        (query_row * num_q_heads + q_head) * head_dim + lane * 4;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      query_values[row_index][part] = row_valid[row_index]
          ? query[query_base + part]
          : 0.0f;
    }
  }
  int64_t max_context_len = 0;
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    if (context_lens[row_index] > max_context_len) {
      max_context_len = context_lens[row_index];
    }
  }

  float running_max[kQueryRows];
  float denominator[kQueryRows] = {};
  float output_acc[kQueryRows][4] = {};
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    running_max[row_index] = kLowest;
  }
  for (int64_t token = 0; token < max_context_len; ++token) {
    const int64_t current_row =
        kCachedMetadata ? -1 : token - start_position;
    const bool has_current = current_row >= 0 && current_row < rows;
    bool visible[kQueryRows];
    bool any_visible = false;
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      if (kCachedMetadata) {
        const int64_t key_position = kDenseInitialMetadata
            ? token
            : token_positions[token];
        visible[row_index] = row_valid[row_index] &&
            token < context_lens[row_index] &&
            (kDenseInitialMetadata || evict_mask[token] == 0) &&
            key_position >= 0 &&
            key_position <= query_positions[row_index];
      } else {
        const bool from_current =
            has_current && current_row <= query_rows[row_index];
        visible[row_index] = row_valid[row_index] &&
            token < context_lens[row_index] &&
            (from_current ||
             (!has_current && evict_mask[token] == 0 &&
              token_positions[token] >= 0 &&
              token_positions[token] <= query_positions[row_index]));
      }
      any_visible = any_visible || visible[row_index];
    }
    if (!any_visible) continue;

    const int64_t physical_slot =
        laguna_global_physical_slot(base_offsets, token, block_size);
    const int64_t current_base =
        (current_row * num_kv_heads + kv_head) * head_dim + lane * 4;
    const int64_t cache_base =
        (physical_slot * num_kv_heads + kv_head) * head_dim + lane * 4;
    float key_values[4];
    float value_values[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      key_values[part] = kCachedOnly
          ? laguna_bf16_bits_to_float(key_cache[cache_base + part])
          : has_current
          ? laguna_bf16_bits_to_float(
                laguna_float_to_bf16_bits(current_key[current_base + part]))
          : laguna_bf16_bits_to_float(key_cache[cache_base + part]);
      value_values[part] = kCachedOnly
          ? laguna_bf16_bits_to_float(value_cache[cache_base + part])
          : has_current
          ? laguna_bf16_bits_to_float(
                laguna_float_to_bf16_bits(current_value[current_base + part]))
          : laguna_bf16_bits_to_float(value_cache[cache_base + part]);
    }

#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      if (!visible[row_index]) continue;
      float dot = 0.0f;
      dot += query_values[row_index][0] * key_values[0];
      dot += query_values[row_index][1] * key_values[1];
      dot += query_values[row_index][2] * key_values[2];
      dot += query_values[row_index][3] * key_values[3];
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down(dot, offset);
      }
      const float score = __shfl(dot, 0) * scale;
      if (score > running_max[row_index]) {
        const float old_scale = running_max[row_index] == kLowest
            ? 0.0f
            : expf(running_max[row_index] - score);
        denominator[row_index] = denominator[row_index] * old_scale + 1.0f;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          output_acc[row_index][part] =
              output_acc[row_index][part] * old_scale + value_values[part];
        }
        running_max[row_index] = score;
      } else {
        const float weight = expf(score - running_max[row_index]);
        denominator[row_index] += weight;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          output_acc[row_index][part] += weight * value_values[part];
        }
      }
    }
  }

#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    if (!row_valid[row_index]) continue;
    const float inv_denom = 1.0f / fmaxf(denominator[row_index], 1.0e-20f);
    const int64_t out_base =
        (query_rows[row_index] * num_q_heads + q_head) * head_dim + lane * 4;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      out[out_base + part] = output_acc[row_index][part] * inv_denom;
    }
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

__global__ __launch_bounds__(32) void laguna_swa_attention_split_exact_scores_kernel(
    const float* query,
    const uint16_t* key_cache,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t sliding_window,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t logical_slot = static_cast<int64_t>(blockIdx.y);
  if (q_head >= num_q_heads || logical_slot >= scan_slots) return;

  const int64_t live_count = live_counts[0];
  const int64_t context_len = live_count < capacity ? live_count : capacity;
  const int64_t query_position = row_positions[0];
  const int64_t key_position =
      logical_slot < context_len ? token_positions[logical_slot] : -1;
  const bool visible = logical_slot < context_len && query_position >= 0 &&
                       evict_mask[logical_slot] == 0 && key_position >= 0 &&
                       key_position <= query_position &&
                       key_position > query_position - sliding_window;
  int64_t physical_slot = -1;
  if (visible) {
    physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    if (physical_slot < 0 || physical_slot >= capacity) physical_slot = -1;
  }

  const int lane = threadIdx.x;
  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  const int64_t q_offset = q_head * head_dim;
  float products[4] = {};
  if (physical_slot >= 0) {
    const int64_t key_base =
        (physical_slot * num_kv_heads + kv_head) * head_dim;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int64_t dim = lane + part * 32;
      products[part] = laguna_ordered_mul_f32(
          query[q_offset + dim],
          laguna_bf16_bits_to_float(key_cache[key_base + dim]));
    }
  }
  const float dot = laguna_wave32_sum_128_exact(
      products[0], products[1], products[2], products[3]);
  if (lane == 0) {
    const int64_t idx = q_head * scratch_stride + logical_slot;
    score_scratch[idx] =
        physical_slot >= 0 ? dot : -3.4028234663852886e38f;
    physical_scratch[idx] = static_cast<int32_t>(physical_slot);
  }
}

// Exact SWA score tile. It retains the token4 kernel's strided dimensions,
// ordered products, and reconstructed 128-thread
// reduction tree for each logical slot before the unchanged exact reducer.
__global__ __launch_bounds__(256)
void laguna_swa_attention_split_tile16_exact_scores_kernel(
    const float* query,
    const uint16_t* key_cache,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t sliding_window,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t tile_base =
      static_cast<int64_t>(blockIdx.y) * LAGUNA_SPLIT_SCORE_TILE;
  if (q_head >= num_q_heads || tile_base >= scan_slots) return;

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int64_t live_count = live_counts[0];
  const int64_t context_len =
      live_count < capacity ? live_count : capacity;
  const int64_t query_position = row_positions[0];
  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  const int64_t q_offset = q_head * head_dim;
  float q_values[4];
#pragma unroll
  for (int part = 0; part < 4; ++part) {
    q_values[part] = query[q_offset + lane + part * 32];
  }

  for (int item = wave; item < LAGUNA_SPLIT_SCORE_TILE; item += 8) {
    const int64_t logical_slot = tile_base + item;
    const int64_t key_position =
        logical_slot < context_len && logical_slot < scan_slots
            ? token_positions[logical_slot]
            : -1;
    const bool visible = logical_slot < context_len &&
                         logical_slot < scan_slots && query_position >= 0 &&
                         evict_mask[logical_slot] == 0 && key_position >= 0 &&
                         key_position <= query_position &&
                         key_position > query_position - sliding_window;
    int64_t physical_slot = -1;
    if (visible) {
      physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
      if (physical_slot < 0 || physical_slot >= capacity) physical_slot = -1;
    }

    float products[4] = {};
    if (physical_slot >= 0) {
      const int64_t key_base =
          (physical_slot * num_kv_heads + kv_head) * head_dim;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const int64_t dim = lane + part * 32;
        products[part] = laguna_ordered_mul_f32(
            q_values[part],
            laguna_bf16_bits_to_float(key_cache[key_base + dim]));
      }
    }
    const float dot = laguna_wave32_sum_128_exact(
        products[0], products[1], products[2], products[3]);
    if (lane == 0 && logical_slot < scan_slots) {
      const int64_t idx = q_head * scratch_stride + logical_slot;
      score_scratch[idx] =
          physical_slot >= 0 ? dot : -3.4028234663852886e38f;
      physical_scratch[idx] = static_cast<int32_t>(physical_slot);
    }
  }
}

__global__ __launch_bounds__(128) void laguna_swa_attention_split_exact_reduce_kernel(
    const uint16_t* value_cache,
    const float* score_scratch,
    const int32_t* physical_scratch,
    float* out,
    const int64_t* live_counts,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t dim = static_cast<int64_t>(threadIdx.x);
  if (q_head >= num_q_heads || dim >= head_dim) return;
  const int64_t live_count = live_counts[0];
  const int64_t query_position = row_positions[0];
  int64_t context_len = live_count < capacity ? live_count : capacity;
  context_len = context_len < scan_slots ? context_len : scan_slots;
  const int64_t q_offset = q_head * head_dim;
  if (live_count <= 0 || query_position < 0) {
    out[q_offset + dim] = 0.0f;
    return;
  }

  __shared__ float batch_weights[4];
  __shared__ float shared_max;
  __shared__ float shared_denom;
  if (dim == 0) {
    shared_max = -3.4028234663852886e38f;
    shared_denom = 0.0f;
  }
  __syncthreads();
  const int64_t scratch_base = q_head * scratch_stride;
  for (int64_t slot_base = 0; slot_base < context_len; slot_base += 4) {
    if (dim == 0) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int64_t slot = slot_base + item;
        if (slot < context_len && physical_scratch[scratch_base + slot] >= 0) {
          shared_max = fmaxf(
              shared_max, score_scratch[scratch_base + slot] * scale);
        }
      }
    }
    __syncthreads();
  }

  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  float output_acc = 0.0f;
  for (int64_t slot_base = 0; slot_base < context_len; slot_base += 4) {
    if (dim == 0) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int64_t slot = slot_base + item;
        const int64_t idx = scratch_base + slot;
        const float weight =
            slot < context_len && physical_scratch[idx] >= 0
                ? expf(score_scratch[idx] * scale - shared_max)
                : 0.0f;
        batch_weights[item] = weight;
        if (slot < context_len && physical_scratch[idx] >= 0) {
          shared_denom = laguna_ordered_add_f32(shared_denom, weight);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int64_t slot = slot_base + item;
      if (slot < context_len) {
        const int64_t physical_slot =
            physical_scratch[scratch_base + slot];
        if (physical_slot >= 0) {
          const int64_t value_offset =
              (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
          output_acc = fmaf(
              batch_weights[item],
              laguna_bf16_bits_to_float(value_cache[value_offset]),
              output_acc);
        }
      }
    }
    __syncthreads();
  }
  out[q_offset + dim] =
      output_acc / fmaxf(shared_denom, 1.0e-20f);
}

__global__ __launch_bounds__(128) void laguna_swa_attention_split_exact_gated_reduce_kernel(
    const uint16_t* value_cache,
    const float* score_scratch,
    const int32_t* physical_scratch,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    const int64_t* live_counts,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t dim = static_cast<int64_t>(threadIdx.x);
  if (q_head >= num_q_heads || dim >= head_dim) return;
  const int64_t live_count = live_counts[0];
  const int64_t query_position = row_positions[0];
  int64_t context_len = live_count < capacity ? live_count : capacity;
  context_len = context_len < scan_slots ? context_len : scan_slots;
  const int64_t q_offset = q_head * head_dim;
  if (live_count <= 0 || query_position < 0) {
    out[q_offset + dim] = 0.0f;
    gated_out[q_offset + dim] = 0;
    return;
  }

  __shared__ float batch_weights[4];
  __shared__ float shared_max;
  __shared__ float shared_denom;
  if (dim == 0) {
    shared_max = -3.4028234663852886e38f;
    shared_denom = 0.0f;
  }
  __syncthreads();
  const int64_t scratch_base = q_head * scratch_stride;
  for (int64_t slot_base = 0; slot_base < context_len; slot_base += 4) {
    if (dim == 0) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int64_t slot = slot_base + item;
        if (slot < context_len && physical_scratch[scratch_base + slot] >= 0) {
          shared_max = fmaxf(
              shared_max, score_scratch[scratch_base + slot] * scale);
        }
      }
    }
    __syncthreads();
  }

  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  float output_acc = 0.0f;
  for (int64_t slot_base = 0; slot_base < context_len; slot_base += 4) {
    if (dim == 0) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int64_t slot = slot_base + item;
        const int64_t idx = scratch_base + slot;
        const float weight =
            slot < context_len && physical_scratch[idx] >= 0
                ? expf(score_scratch[idx] * scale - shared_max)
                : 0.0f;
        batch_weights[item] = weight;
        if (slot < context_len && physical_scratch[idx] >= 0) {
          shared_denom = laguna_ordered_add_f32(shared_denom, weight);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int64_t slot = slot_base + item;
      if (slot < context_len) {
        const int64_t physical_slot = physical_scratch[scratch_base + slot];
        if (physical_slot >= 0) {
          const int64_t value_offset =
              (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
          output_acc = fmaf(
              batch_weights[item],
              laguna_bf16_bits_to_float(value_cache[value_offset]),
              output_acc);
        }
      }
    }
    __syncthreads();
  }
  const float context_value =
      output_acc / fmaxf(shared_denom, 1.0e-20f);
  const int64_t output_idx = q_offset + dim;
  out[output_idx] = context_value;
  gated_out[output_idx] = laguna_float_to_bf16_bits(
      context_value * laguna_softplus_f32(gate[q_head]));
}

// Exact wave-private split reducer. Each wave leader replays the retained
// scalar maximum/denominator order independently and broadcasts each batch of
// four weights within its wave. This trades fourfold scalar score/exp work for
// removal of all cross-wave barriers and reducer LDS.
__global__ __launch_bounds__(128)
void laguna_swa_attention_split_exact_gated_wave_local_reduce_kernel(
    const uint16_t* value_cache,
    const float* score_scratch,
    const int32_t* physical_scratch,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    const int64_t* live_counts,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t dim = static_cast<int64_t>(threadIdx.x);
  if (q_head >= num_q_heads || dim >= head_dim) return;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int64_t live_count = live_counts[0];
  const int64_t query_position = row_positions[0];
  int64_t context_len = live_count < capacity ? live_count : capacity;
  context_len = context_len < scan_slots ? context_len : scan_slots;
  const int64_t q_offset = q_head * head_dim;
  if (live_count <= 0 || query_position < 0) {
    out[q_offset + dim] = 0.0f;
    gated_out[q_offset + dim] = 0;
    return;
  }

  const int64_t scratch_base = q_head * scratch_stride;
  float wave_max = -3.4028234663852886e38f;
  if (lane == 0) {
    for (int64_t slot_base = 0; slot_base < context_len; slot_base += 4) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int64_t slot = slot_base + item;
        if (slot < context_len && physical_scratch[scratch_base + slot] >= 0) {
          wave_max = fmaxf(
              wave_max, score_scratch[scratch_base + slot] * scale);
        }
      }
    }
  }
  wave_max = __shfl(wave_max, 0, 32);

  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  float wave_denom = 0.0f;
  float output_acc = 0.0f;
  for (int64_t slot_base = 0; slot_base < context_len; slot_base += 4) {
    float weights[4] = {};
    if (lane == 0) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int64_t slot = slot_base + item;
        const int64_t idx = scratch_base + slot;
        const bool visible =
            slot < context_len && physical_scratch[idx] >= 0;
        const float weight = visible
            ? expf(score_scratch[idx] * scale - wave_max)
            : 0.0f;
        weights[item] = weight;
        if (visible) {
          wave_denom = laguna_ordered_add_f32(wave_denom, weight);
        }
      }
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const float weight = __shfl(weights[item], 0, 32);
      const int64_t slot = slot_base + item;
      if (slot < context_len) {
        const int64_t physical_slot = physical_scratch[scratch_base + slot];
        if (physical_slot >= 0) {
          const int64_t value_offset =
              (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
          output_acc = fmaf(
              weight,
              laguna_bf16_bits_to_float(value_cache[value_offset]),
              output_acc);
        }
      }
    }
  }
  wave_denom = __shfl(wave_denom, 0, 32);
  const float context_value = output_acc / fmaxf(wave_denom, 1.0e-20f);
  const int64_t output_idx = q_offset + dim;
  out[output_idx] = context_value;
  gated_out[output_idx] = laguna_float_to_bf16_bits(
      context_value * laguna_softplus_f32(gate[q_head]));
}

// Exact local64 sibling of the retained wave-local reducer. One block still
// owns one query head, but each thread carries two adjacent dimensions. This
// retains each dimension's slot-order FMA/divide/gate/store sequence while two
// wave leaders replay the scalar softmax state and packed loads provide both
// BF16 source values.
__global__ __launch_bounds__(64)
void laguna_swa_attention_split_exact_gated_wave_local_dim2_reduce_kernel(
    const uint16_t* value_cache,
    const float* score_scratch,
    const int32_t* physical_scratch,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    const int64_t* live_counts,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t dim0 = static_cast<int64_t>(threadIdx.x) * 2;
  const int64_t dim1 = dim0 + 1;
  if (q_head >= num_q_heads || dim1 >= head_dim) return;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int64_t live_count = live_counts[0];
  const int64_t query_position = row_positions[0];
  int64_t context_len = live_count < capacity ? live_count : capacity;
  context_len = context_len < scan_slots ? context_len : scan_slots;
  const int64_t q_offset = q_head * head_dim;
  if (live_count <= 0 || query_position < 0) {
    out[q_offset + dim0] = 0.0f;
    out[q_offset + dim1] = 0.0f;
    gated_out[q_offset + dim0] = 0;
    gated_out[q_offset + dim1] = 0;
    return;
  }

  const int64_t scratch_base = q_head * scratch_stride;
  float wave_max = -3.4028234663852886e38f;
  if (lane == 0) {
    for (int64_t slot_base = 0; slot_base < context_len; slot_base += 4) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int64_t slot = slot_base + item;
        if (slot < context_len && physical_scratch[scratch_base + slot] >= 0) {
          wave_max = fmaxf(
              wave_max, score_scratch[scratch_base + slot] * scale);
        }
      }
    }
  }
  wave_max = __shfl(wave_max, 0, 32);

  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  float wave_denom = 0.0f;
  float output_acc0 = 0.0f;
  float output_acc1 = 0.0f;
  for (int64_t slot_base = 0; slot_base < context_len; slot_base += 4) {
    float weights[4] = {};
    if (lane == 0) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int64_t slot = slot_base + item;
        const int64_t idx = scratch_base + slot;
        const bool visible =
            slot < context_len && physical_scratch[idx] >= 0;
        const float weight = visible
            ? expf(score_scratch[idx] * scale - wave_max)
            : 0.0f;
        weights[item] = weight;
        if (visible) {
          wave_denom = laguna_ordered_add_f32(wave_denom, weight);
        }
      }
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const float weight = __shfl(weights[item], 0, 32);
      const int64_t slot = slot_base + item;
      if (slot < context_len) {
        const int64_t physical_slot = physical_scratch[scratch_base + slot];
        if (physical_slot >= 0) {
          const int64_t value_offset =
              (physical_slot * num_kv_heads + kv_head) * head_dim + dim0;
          const uint32_t packed =
              *reinterpret_cast<const uint32_t*>(value_cache + value_offset);
          const uint16_t value0 = static_cast<uint16_t>(packed & 0xffffu);
          const uint16_t value1 = static_cast<uint16_t>(packed >> 16);
          output_acc0 = fmaf(
              weight, laguna_bf16_bits_to_float(value0), output_acc0);
          output_acc1 = fmaf(
              weight, laguna_bf16_bits_to_float(value1), output_acc1);
        }
      }
    }
  }
  wave_denom = __shfl(wave_denom, 0, 32);
  const float safe_denom = fmaxf(wave_denom, 1.0e-20f);
  const float context_value0 = output_acc0 / safe_denom;
  const float context_value1 = output_acc1 / safe_denom;
  const float gate_value = laguna_softplus_f32(gate[q_head]);
  const int64_t output_idx0 = q_offset + dim0;
  const int64_t output_idx1 = q_offset + dim1;
  out[output_idx0] = context_value0;
  out[output_idx1] = context_value1;
  gated_out[output_idx0] =
      laguna_float_to_bf16_bits(context_value0 * gate_value);
  gated_out[output_idx1] =
      laguna_float_to_bf16_bits(context_value1 * gate_value);
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

template <
    int kQueryRows,
    bool kQualifiedSourceLoads = false,
    bool kDenseFirstFill = false>
__global__ __launch_bounds__(32) void laguna_swa_attention_prefill_qrows_exact_bf16_kernel(
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
  const int64_t query_row_base =
      static_cast<int64_t>(blockIdx.y) * kQueryRows;
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t lane = static_cast<int64_t>(threadIdx.x);
  if (query_row_base >= rows || q_head >= num_q_heads || lane >= 32) return;

  const int64_t start_position = row_positions[0];
  if constexpr (kDenseFirstFill) {
    static_assert(kQualifiedSourceLoads);
    if (rows != 128 || capacity != 512 || sliding_window != 512 ||
        num_q_heads != 72 || num_kv_heads != 8 || head_dim != 128 ||
        (start_position != 256 && start_position != 384) ||
        start_position + rows > capacity) {
      return;
    }
  }
  const int64_t prior_live_count =
      kDenseFirstFill ? start_position : live_counts[0];
  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  const int64_t start_slot =
      kDenseFirstFill ? start_position : start_position % capacity;
  if (prior_live_count <= 0 && start_position < 0) {
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      const int64_t query_row = query_row_base + row_index;
      if (query_row >= rows) continue;
      const int64_t q_offset = (query_row * num_q_heads + q_head) * head_dim;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        out[q_offset + lane + part * 32] = 0.0f;
      }
    }
    return;
  }

  float max_scores[kQueryRows];
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    max_scores[row_index] = -3.4028234663852886e38f;
  }
  const int64_t final_query_position =
      start_position + (rows - 1 < query_row_base + kQueryRows - 1
                            ? rows - 1
                            : query_row_base + kQueryRows - 1);
  const int64_t final_query_live_count =
      final_query_position + 1 < capacity ? final_query_position + 1 : capacity;
  const int64_t live_count =
      prior_live_count > final_query_live_count
          ? prior_live_count
          : final_query_live_count;
  const int64_t scan_slots = live_count < capacity ? live_count : capacity;

  for (int64_t logical_slot = 0; logical_slot < scan_slots; ++logical_slot) {
    int64_t current_row = logical_slot - start_slot;
    if (!kDenseFirstFill && current_row < 0) current_row += capacity;
    const bool has_current = current_row >= 0 && current_row < rows;
    const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    const bool cache_valid = physical_slot >= 0 && physical_slot < capacity;
    bool qualified_from_current[kQueryRows] = {};
    bool qualified_visible[kQueryRows] = {};
    bool need_current = false;
    bool need_cached = false;
    if constexpr (kQualifiedSourceLoads) {
      bool any_visible = false;
#pragma unroll
      for (int row_index = 0; row_index < kQueryRows; ++row_index) {
        const int64_t query_row = query_row_base + row_index;
        if (query_row >= rows) continue;
        const int64_t query_position = start_position + query_row;
        const bool from_current = has_current && current_row <= query_row;
        const int64_t key_position = kDenseFirstFill
            ? logical_slot
            : from_current
            ? start_position + current_row
            : token_positions[logical_slot];
        const bool source_valid = from_current ||
            (cache_valid && (!kDenseFirstFill || logical_slot < start_position));
        const bool visible = source_valid &&
                             (kDenseFirstFill || from_current ||
                              evict_mask[logical_slot] == 0) &&
                             key_position >= 0 && key_position <= query_position &&
                             key_position > query_position - sliding_window;
        qualified_from_current[row_index] = from_current;
        qualified_visible[row_index] = visible;
        any_visible = any_visible || visible;
        need_current = need_current || (visible && from_current);
        need_cached = need_cached || (visible && !from_current);
      }
      if (!any_visible) continue;
    }
    const int64_t current_base =
        (current_row * num_kv_heads + kv_head) * head_dim;
    const int64_t cache_base =
        (physical_slot * num_kv_heads + kv_head) * head_dim;
    float current_keys[4];
    float cached_keys[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int64_t dim = lane + part * 32;
      current_keys[part] = (kQualifiedSourceLoads ? need_current : has_current)
          ? laguna_bf16_bits_to_float(
                laguna_float_to_bf16_bits(current_key[current_base + dim]))
          : 0.0f;
      cached_keys[part] = (kQualifiedSourceLoads ? need_cached : cache_valid)
          ? laguna_bf16_bits_to_float(key_cache[cache_base + dim])
          : 0.0f;
    }
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      const int64_t query_row = query_row_base + row_index;
      if (query_row >= rows) continue;
      const int64_t query_position = start_position + query_row;
      const bool from_current = kQualifiedSourceLoads
          ? qualified_from_current[row_index]
          : has_current && current_row <= query_row;
      const int64_t key_position = kDenseFirstFill
          ? logical_slot
          : from_current
          ? start_position + current_row
          : token_positions[logical_slot];
      const bool source_valid = from_current ||
          (cache_valid && (!kDenseFirstFill || logical_slot < start_position));
      const bool visible = kQualifiedSourceLoads
          ? qualified_visible[row_index]
          : (from_current || evict_mask[logical_slot] == 0) &&
                key_position >= 0 && key_position <= query_position &&
                key_position > query_position - sliding_window;
      if (!visible) continue;
      const int64_t q_offset = (query_row * num_q_heads + q_head) * head_dim;
      float products[4];
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const int64_t dim = lane + part * 32;
        const float key_value =
            from_current ? current_keys[part] : cached_keys[part];
        products[part] = source_valid
            ? laguna_ordered_mul_f32(query[q_offset + dim], key_value)
            : 0.0f;
      }
      const float dot = laguna_wave32_sum_128_exact(
          products[0], products[1], products[2], products[3]);
      if (lane == 0 && source_valid) {
        max_scores[row_index] = fmaxf(max_scores[row_index], dot * scale);
      }
    }
  }
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    max_scores[row_index] = __shfl(max_scores[row_index], 0);
  }

  float denominators[kQueryRows] = {};
  float output_acc[kQueryRows][4] = {};
  for (int64_t logical_slot = 0; logical_slot < scan_slots; ++logical_slot) {
    int64_t current_row = logical_slot - start_slot;
    if (!kDenseFirstFill && current_row < 0) current_row += capacity;
    const bool has_current = current_row >= 0 && current_row < rows;
    const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    const bool cache_valid = physical_slot >= 0 && physical_slot < capacity;
    bool qualified_from_current[kQueryRows] = {};
    bool qualified_visible[kQueryRows] = {};
    bool need_current = false;
    bool need_cached = false;
    if constexpr (kQualifiedSourceLoads) {
      bool any_visible = false;
#pragma unroll
      for (int row_index = 0; row_index < kQueryRows; ++row_index) {
        const int64_t query_row = query_row_base + row_index;
        if (query_row >= rows) continue;
        const int64_t query_position = start_position + query_row;
        const bool from_current = has_current && current_row <= query_row;
        const int64_t key_position = kDenseFirstFill
            ? logical_slot
            : from_current
            ? start_position + current_row
            : token_positions[logical_slot];
        const bool source_valid = from_current ||
            (cache_valid && (!kDenseFirstFill || logical_slot < start_position));
        const bool visible = source_valid &&
                             (kDenseFirstFill || from_current ||
                              evict_mask[logical_slot] == 0) &&
                             key_position >= 0 && key_position <= query_position &&
                             key_position > query_position - sliding_window;
        qualified_from_current[row_index] = from_current;
        qualified_visible[row_index] = visible;
        any_visible = any_visible || visible;
        need_current = need_current || (visible && from_current);
        need_cached = need_cached || (visible && !from_current);
      }
      if (!any_visible) continue;
    }
    const int64_t current_base =
        (current_row * num_kv_heads + kv_head) * head_dim;
    const int64_t cache_base =
        (physical_slot * num_kv_heads + kv_head) * head_dim;
    float current_keys[4];
    float cached_keys[4];
    float current_values[4];
    float cached_values[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int64_t dim = lane + part * 32;
      current_keys[part] = (kQualifiedSourceLoads ? need_current : has_current)
          ? laguna_bf16_bits_to_float(
                laguna_float_to_bf16_bits(current_key[current_base + dim]))
          : 0.0f;
      cached_keys[part] = (kQualifiedSourceLoads ? need_cached : cache_valid)
          ? laguna_bf16_bits_to_float(key_cache[cache_base + dim])
          : 0.0f;
      current_values[part] = (kQualifiedSourceLoads ? need_current : has_current)
          ? laguna_bf16_bits_to_float(
                laguna_float_to_bf16_bits(current_value[current_base + dim]))
          : 0.0f;
      cached_values[part] = (kQualifiedSourceLoads ? need_cached : cache_valid)
          ? laguna_bf16_bits_to_float(value_cache[cache_base + dim])
          : 0.0f;
    }
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      const int64_t query_row = query_row_base + row_index;
      if (query_row >= rows) continue;
      const int64_t query_position = start_position + query_row;
      const bool from_current = kQualifiedSourceLoads
          ? qualified_from_current[row_index]
          : has_current && current_row <= query_row;
      const int64_t key_position = kDenseFirstFill
          ? logical_slot
          : from_current
          ? start_position + current_row
          : token_positions[logical_slot];
      const bool source_valid = from_current ||
          (cache_valid && (!kDenseFirstFill || logical_slot < start_position));
      const bool visible = kQualifiedSourceLoads
          ? qualified_visible[row_index]
          : (from_current || evict_mask[logical_slot] == 0) &&
                key_position >= 0 && key_position <= query_position &&
                key_position > query_position - sliding_window;
      if (!visible) continue;
      const int64_t q_offset = (query_row * num_q_heads + q_head) * head_dim;
      float products[4];
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const int64_t dim = lane + part * 32;
        const float key_value =
            from_current ? current_keys[part] : cached_keys[part];
        products[part] = source_valid
            ? laguna_ordered_mul_f32(query[q_offset + dim], key_value)
            : 0.0f;
      }
      const float dot = laguna_wave32_sum_128_exact(
          products[0], products[1], products[2], products[3]);
      float weight = 0.0f;
      if (lane == 0) {
        weight = source_valid ? expf(dot * scale - max_scores[row_index]) : 0.0f;
        denominators[row_index] += weight;
      }
      weight = __shfl(weight, 0);
      if (source_valid) {
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          const float value =
              from_current ? current_values[part] : cached_values[part];
          output_acc[row_index][part] += weight * value;
        }
      }
    }
  }

#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    const int64_t query_row = query_row_base + row_index;
    if (query_row >= rows) continue;
    denominators[row_index] = __shfl(denominators[row_index], 0);
    const float safe_denominator = fmaxf(denominators[row_index], 1.0e-20f);
    const int64_t q_offset = (query_row * num_q_heads + q_head) * head_dim;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      out[q_offset + lane + part * 32] =
          output_acc[row_index][part] / safe_denominator;
    }
  }
}

template <bool kGlobalLayout, bool kDenseInitialMetadata = false>
__global__ __launch_bounds__(32) void laguna_attention_prefill_qrow4_cached_exact_bf16_kernel(
    const float* query,
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
    int64_t block_size_or_window,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t expected_start_position,
    float scale) {
  constexpr int kQueryRows = 4;
  constexpr float kLowest = -3.4028234663852886e38f;
  const int64_t query_row_base =
      static_cast<int64_t>(blockIdx.y) * kQueryRows;
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t lane = static_cast<int64_t>(threadIdx.x);
  if (query_row_base >= rows || q_head >= num_q_heads || lane >= 32) return;

  const int64_t start_position = row_positions[0];
  if (rows != 128 || start_position != expected_start_position ||
      (start_position != 0 && start_position != 128 &&
       start_position != 256 && start_position != 384)) {
    return;
  }
  const int64_t prior_live_count = live_counts[0];
  const int64_t scan_slots =
      prior_live_count < capacity ? prior_live_count : capacity;
  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;

  if constexpr (kGlobalLayout) {
    // Reconstruct the production local256 global-attention association. Its dot
    // owner is one wave with four contiguous dimensions per lane; its softmax
    // denominator is eight token-strided partial sums followed by serial
    // wave-0..7 accumulation; and PV multiplies normalized weights before the
    // token-ordered value sum. Preappend changes only the K/V source.
    constexpr int64_t kMaxFirstFillContext = 512;
    __shared__ float scores[kQueryRows][kMaxFirstFillContext];
    bool global_row_valid[kQueryRows];
    int64_t global_query_rows[kQueryRows];
    int64_t global_query_positions[kQueryRows];
    int64_t global_context_lens[kQueryRows];
    float contiguous_query_values[kQueryRows][4];
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      const int64_t query_row = query_row_base + row_index;
      const int64_t query_position = start_position + query_row;
      global_row_valid[row_index] = query_row < rows;
      global_query_rows[row_index] = query_row;
      global_query_positions[row_index] = query_position;
      global_context_lens[row_index] = global_row_valid[row_index]
          ? query_position + 1
          : 0;
      const int64_t query_base =
          (query_row * num_q_heads + q_head) * head_dim + lane * 4;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        contiguous_query_values[row_index][part] =
            global_row_valid[row_index] ? query[query_base + part] : 0.0f;
      }
    }
    const int64_t valid_group_rows =
        rows - query_row_base < kQueryRows ? rows - query_row_base : kQueryRows;
    const int64_t max_context_len =
        global_context_lens[valid_group_rows - 1];

    float global_max_scores[kQueryRows];
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      global_max_scores[row_index] = kLowest;
    }
    for (int64_t logical_slot = 0;
         logical_slot < max_context_len;
         ++logical_slot) {
      const int64_t key_position = token_positions[logical_slot];
      const int64_t physical_slot = laguna_global_physical_slot(
          base_offsets, logical_slot, block_size_or_window);
      const bool physical_valid =
          physical_slot >= 0 && physical_slot < capacity;
      const int64_t key_base =
          (physical_slot * num_kv_heads + kv_head) * head_dim + lane * 4;
      float contiguous_keys[4];
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        contiguous_keys[part] = physical_valid
            ? laguna_bf16_bits_to_float(key_cache[key_base + part])
            : 0.0f;
      }
#pragma unroll
      for (int row_index = 0; row_index < kQueryRows; ++row_index) {
        const bool visible = global_row_valid[row_index] &&
            logical_slot < global_context_lens[row_index] && physical_valid &&
            evict_mask[logical_slot] == 0 && key_position >= 0 &&
            key_position <= global_query_positions[row_index];
        float dot = 0.0f;
        if (visible) {
#pragma unroll
          for (int part = 0; part < 4; ++part) {
            dot += contiguous_query_values[row_index][part] *
                contiguous_keys[part];
          }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
          dot += __shfl_down(dot, offset);
        }
        if (lane == 0) {
          const float score = visible ? dot * scale : kLowest;
          scores[row_index][logical_slot] = score;
          global_max_scores[row_index] =
              fmaxf(global_max_scores[row_index], score);
        }
      }
    }
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      global_max_scores[row_index] = __shfl(global_max_scores[row_index], 0);
    }
    __syncthreads();

    float denominator_partials[kQueryRows] = {};
    if (lane < 8) {
#pragma unroll
      for (int row_index = 0; row_index < kQueryRows; ++row_index) {
        float local_sum = 0.0f;
        for (int64_t logical_slot = lane;
             logical_slot < global_context_lens[row_index];
             logical_slot += 8) {
          const int64_t key_position = token_positions[logical_slot];
          const bool visible = evict_mask[logical_slot] == 0 &&
              key_position >= 0 &&
              key_position <= global_query_positions[row_index];
          const float weight = visible
              ? expf(scores[row_index][logical_slot] -
                     global_max_scores[row_index])
              : 0.0f;
          scores[row_index][logical_slot] = weight;
          local_sum += weight;
        }
        denominator_partials[row_index] = local_sum;
      }
    }
    __syncthreads();

    float inverse_denominators[kQueryRows] = {};
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      float total_sum = __shfl(denominator_partials[row_index], 0);
#pragma unroll
      for (int warp = 1; warp < 8; ++warp) {
        total_sum += __shfl(denominator_partials[row_index], warp);
      }
      if (lane == 0) {
        inverse_denominators[row_index] =
            1.0f / fmaxf(total_sum, 1.0e-20f);
      }
      inverse_denominators[row_index] =
          __shfl(inverse_denominators[row_index], 0);
    }

#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      if (!global_row_valid[row_index]) continue;
      const int64_t out_base =
          (global_query_rows[row_index] * num_q_heads + q_head) * head_dim;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const int64_t dim = lane + part * 32;
        float acc = 0.0f;
        for (int64_t logical_slot = 0;
             logical_slot < global_context_lens[row_index];
             ++logical_slot) {
          const float weight =
              scores[row_index][logical_slot] * inverse_denominators[row_index];
          if (!(weight > 0.0f)) continue;
          const int64_t physical_slot = laguna_global_physical_slot(
              base_offsets, logical_slot, block_size_or_window);
          const int64_t value_offset =
              (physical_slot * num_kv_heads + kv_head) * head_dim + dim;
          acc += weight *
              laguna_bf16_bits_to_float(value_cache[value_offset]);
        }
        out[out_base + dim] = acc;
      }
    }
    return;
  }

  if constexpr (kDenseInitialMetadata) {
    // A complete preappended first fill has identity logical/physical slots,
    // identity token positions, and no eviction. Preserve H5R's two-pass
    // qrow4 arithmetic while compiling every per-token span lookup away.
    if (prior_live_count < start_position + rows) return;
    float dense_query_values[kQueryRows][4];
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      const int64_t query_row = query_row_base + row_index;
      const int64_t query_base =
          (query_row * num_q_heads + q_head) * head_dim;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        dense_query_values[row_index][part] =
            query[query_base + lane + part * 32];
      }
    }
    const int64_t dense_context_len =
        start_position + query_row_base + kQueryRows;
    float dense_max_scores[kQueryRows];
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      dense_max_scores[row_index] = kLowest;
    }
    for (int64_t logical_slot = 0;
         logical_slot < dense_context_len;
         ++logical_slot) {
      const int64_t cache_base =
          (logical_slot * num_kv_heads + kv_head) * head_dim;
      float cached_keys[4];
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        cached_keys[part] = laguna_bf16_bits_to_float(
            key_cache[cache_base + lane + part * 32]);
      }
#pragma unroll
      for (int row_index = 0; row_index < kQueryRows; ++row_index) {
        if (logical_slot > start_position + query_row_base + row_index) continue;
        float products[4];
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          products[part] = laguna_ordered_mul_f32(
              dense_query_values[row_index][part], cached_keys[part]);
        }
        const float dot = laguna_wave32_sum_128_exact(
            products[0], products[1], products[2], products[3]);
        if (lane == 0) {
          dense_max_scores[row_index] =
              fmaxf(dense_max_scores[row_index], dot * scale);
        }
      }
    }
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      dense_max_scores[row_index] = __shfl(dense_max_scores[row_index], 0);
    }

    float dense_denominators[kQueryRows] = {};
    float dense_output_acc[kQueryRows][4] = {};
    for (int64_t logical_slot = 0;
         logical_slot < dense_context_len;
         ++logical_slot) {
      const int64_t cache_base =
          (logical_slot * num_kv_heads + kv_head) * head_dim;
      float cached_keys[4];
      float cached_values[4];
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const int64_t cache_index = cache_base + lane + part * 32;
        cached_keys[part] = laguna_bf16_bits_to_float(key_cache[cache_index]);
        cached_values[part] = laguna_bf16_bits_to_float(value_cache[cache_index]);
      }
#pragma unroll
      for (int row_index = 0; row_index < kQueryRows; ++row_index) {
        if (logical_slot > start_position + query_row_base + row_index) continue;
        float products[4];
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          products[part] = laguna_ordered_mul_f32(
              dense_query_values[row_index][part], cached_keys[part]);
        }
        const float dot = laguna_wave32_sum_128_exact(
            products[0], products[1], products[2], products[3]);
        float weight = 0.0f;
        if (lane == 0) {
          weight = expf(dot * scale - dense_max_scores[row_index]);
          dense_denominators[row_index] += weight;
        }
        weight = __shfl(weight, 0);
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          dense_output_acc[row_index][part] +=
              weight * cached_values[part];
        }
      }
    }

#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      dense_denominators[row_index] =
          __shfl(dense_denominators[row_index], 0);
      const float safe_denominator =
          fmaxf(dense_denominators[row_index], 1.0e-20f);
      const int64_t out_base =
          ((query_row_base + row_index) * num_q_heads + q_head) * head_dim;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        out[out_base + lane + part * 32] =
            dense_output_acc[row_index][part] / safe_denominator;
      }
    }
    return;
  }

  bool row_valid[kQueryRows];
  int64_t query_rows[kQueryRows];
  int64_t query_positions[kQueryRows];
  float query_values[kQueryRows][4];
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    const int64_t query_row = query_row_base + row_index;
    row_valid[row_index] = query_row < rows;
    query_rows[row_index] = query_row;
    query_positions[row_index] = start_position + query_row;
    const int64_t query_base =
        (query_row * num_q_heads + q_head) * head_dim;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      query_values[row_index][part] = row_valid[row_index]
          ? query[query_base + lane + part * 32]
          : 0.0f;
    }
  }

  float max_scores[kQueryRows];
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    max_scores[row_index] = kLowest;
  }
  for (int64_t logical_slot = 0; logical_slot < scan_slots; ++logical_slot) {
    const int64_t key_position = token_positions[logical_slot];
    const int64_t physical_slot = kGlobalLayout
        ? laguna_global_physical_slot(
              base_offsets, logical_slot, block_size_or_window)
        : static_cast<int64_t>(base_offsets[logical_slot]);
    const bool physical_valid =
        physical_slot >= 0 && physical_slot < capacity;
    bool visible[kQueryRows];
    bool any_visible = false;
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      const int64_t query_position = query_positions[row_index];
      visible[row_index] = row_valid[row_index] && physical_valid &&
          evict_mask[logical_slot] == 0 && key_position >= 0 &&
          key_position <= query_position &&
          (kGlobalLayout ||
           key_position > query_position - block_size_or_window);
      any_visible = any_visible || visible[row_index];
    }
    if (!any_visible) continue;

    const int64_t cache_base =
        (physical_slot * num_kv_heads + kv_head) * head_dim;
    float cached_keys[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      cached_keys[part] = laguna_bf16_bits_to_float(
          key_cache[cache_base + lane + part * 32]);
    }
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      if (!visible[row_index]) continue;
      float products[4];
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        products[part] = laguna_ordered_mul_f32(
            query_values[row_index][part], cached_keys[part]);
      }
      const float dot = laguna_wave32_sum_128_exact(
          products[0], products[1], products[2], products[3]);
      if (lane == 0) {
        max_scores[row_index] = fmaxf(max_scores[row_index], dot * scale);
      }
    }
  }
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    max_scores[row_index] = __shfl(max_scores[row_index], 0);
  }

  float denominators[kQueryRows] = {};
  float output_acc[kQueryRows][4] = {};
  for (int64_t logical_slot = 0; logical_slot < scan_slots; ++logical_slot) {
    const int64_t key_position = token_positions[logical_slot];
    const int64_t physical_slot = kGlobalLayout
        ? laguna_global_physical_slot(
              base_offsets, logical_slot, block_size_or_window)
        : static_cast<int64_t>(base_offsets[logical_slot]);
    const bool physical_valid =
        physical_slot >= 0 && physical_slot < capacity;
    bool visible[kQueryRows];
    bool any_visible = false;
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      const int64_t query_position = query_positions[row_index];
      visible[row_index] = row_valid[row_index] && physical_valid &&
          evict_mask[logical_slot] == 0 && key_position >= 0 &&
          key_position <= query_position &&
          (kGlobalLayout ||
           key_position > query_position - block_size_or_window);
      any_visible = any_visible || visible[row_index];
    }
    if (!any_visible) continue;

    const int64_t cache_base =
        (physical_slot * num_kv_heads + kv_head) * head_dim;
    float cached_keys[4];
    float cached_values[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int64_t cache_index = cache_base + lane + part * 32;
      cached_keys[part] = laguna_bf16_bits_to_float(key_cache[cache_index]);
      cached_values[part] = laguna_bf16_bits_to_float(value_cache[cache_index]);
    }
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      if (!visible[row_index]) continue;
      float products[4];
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        products[part] = laguna_ordered_mul_f32(
            query_values[row_index][part], cached_keys[part]);
      }
      const float dot = laguna_wave32_sum_128_exact(
          products[0], products[1], products[2], products[3]);
      float weight = 0.0f;
      if (lane == 0) {
        weight = expf(dot * scale - max_scores[row_index]);
        denominators[row_index] += weight;
      }
      weight = __shfl(weight, 0);
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        output_acc[row_index][part] += weight * cached_values[part];
      }
    }
  }

#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    if (!row_valid[row_index]) continue;
    denominators[row_index] = __shfl(denominators[row_index], 0);
    const float safe_denominator =
        fmaxf(denominators[row_index], 1.0e-20f);
    const int64_t out_base =
        (query_rows[row_index] * num_q_heads + q_head) * head_dim;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      out[out_base + lane + part * 32] =
          output_acc[row_index][part] / safe_denominator;
    }
  }
}

__global__ __launch_bounds__(32) void laguna_swa_attention_prefill_qrow4_dense_initial_global_score_replay_exact_bf16_kernel(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    float4* score_records,
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
    int64_t expected_start_position,
    float scale) {
  constexpr int kQueryRows = 4;
  constexpr int kCapacity = 512;
  constexpr float kLowest = -3.4028234663852886e38f;

  const int64_t query_row_base =
      static_cast<int64_t>(blockIdx.y) * kQueryRows;
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t lane = static_cast<int64_t>(threadIdx.x);
  if (query_row_base >= rows || q_head >= num_q_heads || lane >= 32) return;

  const int64_t start_position = row_positions[0];
  if (rows != 128 || capacity != kCapacity || sliding_window != kCapacity ||
      num_q_heads != 72 || num_kv_heads != 8 || head_dim != 128 ||
      start_position != expected_start_position ||
      (start_position != 256 && start_position != 384) ||
      live_counts[0] < start_position + rows) {
    return;
  }
  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  const int64_t workgroup_index =
      static_cast<int64_t>(blockIdx.y) * num_q_heads + q_head;
  float4* workgroup_score_records =
      score_records + workgroup_index * kCapacity;

  float query_values[kQueryRows][4];
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    const int64_t query_row = query_row_base + row_index;
    const int64_t query_base =
        (query_row * num_q_heads + q_head) * head_dim;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      query_values[row_index][part] =
          query[query_base + lane + part * 32];
    }
  }

  const int64_t context_len =
      start_position + query_row_base + kQueryRows;
  float max_scores[kQueryRows];
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    max_scores[row_index] = kLowest;
  }
  for (int64_t logical_slot = 0;
       logical_slot < context_len;
       ++logical_slot) {
    const int64_t cache_base =
        (logical_slot * num_kv_heads + kv_head) * head_dim;
    float cached_keys[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      cached_keys[part] = laguna_bf16_bits_to_float(
          key_cache[cache_base + lane + part * 32]);
    }
    float replay_values[kQueryRows] = {};
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      if (logical_slot > start_position + query_row_base + row_index) continue;
      float products[4];
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        products[part] = laguna_ordered_mul_f32(
            query_values[row_index][part], cached_keys[part]);
      }
      const float dot = laguna_wave32_sum_128_exact(
          products[0], products[1], products[2], products[3]);
      if (lane == 0) {
        replay_values[row_index] = dot;
        max_scores[row_index] = fmaxf(max_scores[row_index], dot * scale);
      }
    }
    if (lane == 0) {
      workgroup_score_records[logical_slot] = make_float4(
          replay_values[0],
          replay_values[1],
          replay_values[2],
          replay_values[3]);
    }
  }
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    max_scores[row_index] = __shfl(max_scores[row_index], 0);
  }

  float denominators[kQueryRows] = {};
  float output_acc[kQueryRows][4] = {};
  for (int64_t logical_slot = 0;
       logical_slot < context_len;
       ++logical_slot) {
    float4 replay_record = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (lane == 0) {
      replay_record = workgroup_score_records[logical_slot];
    }
    const float replay_dots[kQueryRows] = {
        replay_record.x,
        replay_record.y,
        replay_record.z,
        replay_record.w,
    };
    const int64_t cache_base =
        (logical_slot * num_kv_heads + kv_head) * head_dim;
    float cached_values[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      cached_values[part] = laguna_bf16_bits_to_float(
          value_cache[cache_base + lane + part * 32]);
    }
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      if (logical_slot > start_position + query_row_base + row_index) continue;
      float weight = 0.0f;
      if (lane == 0) {
        const float dot = replay_dots[row_index];
        weight = expf(dot * scale - max_scores[row_index]);
        denominators[row_index] += weight;
      }
      weight = __shfl(weight, 0);
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        output_acc[row_index][part] += weight * cached_values[part];
      }
    }
  }

#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    denominators[row_index] = __shfl(denominators[row_index], 0);
    const float safe_denominator =
        fmaxf(denominators[row_index], 1.0e-20f);
    const int64_t out_base =
        ((query_row_base + row_index) * num_q_heads + q_head) * head_dim;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      out[out_base + lane + part * 32] =
          output_acc[row_index][part] / safe_denominator;
    }
  }
}

template <
    int kQueryRows,
    bool kQualifiedSourceLoads = false,
    bool kCachedOnly = false,
    bool kCachedMetadata = false,
    bool kDenseInitialMetadata = false>
__global__ __launch_bounds__(32) void laguna_swa_attention_prefill_qrows_online_bf16_kernel(
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
  constexpr float kLowest = -3.4028234663852886e38f;
  const int64_t query_row_base =
      static_cast<int64_t>(blockIdx.y) * kQueryRows;
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t lane = static_cast<int64_t>(threadIdx.x);
  if (query_row_base >= rows || q_head >= num_q_heads || lane >= 32) return;

  const int64_t start_position = row_positions[0];
  if (start_position < 0) {
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      const int64_t query_row = query_row_base + row_index;
      if (query_row >= rows) continue;
      const int64_t out_base =
          (query_row * num_q_heads + q_head) * head_dim;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        out[out_base + lane + part * 32] = 0.0f;
      }
    }
    return;
  }
  if (kDenseInitialMetadata) {
    const int64_t final_position = start_position + rows - 1;
    if (final_position >= capacity ||
        token_positions[0] != 0 ||
        token_positions[final_position] != final_position ||
        evict_mask[0] != 0 ||
        evict_mask[final_position] != 0) {
      return;
    }
  }

  const int64_t prior_live_count = live_counts[0];
  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  const int64_t start_slot = start_position % capacity;
  bool row_valid[kQueryRows];
  int64_t query_rows[kQueryRows];
  int64_t query_positions[kQueryRows];
  float query_values[kQueryRows][4];
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    const int64_t query_row = query_row_base + row_index;
    row_valid[row_index] = query_row < rows;
    query_rows[row_index] = query_row;
    query_positions[row_index] = start_position + query_row;
    const int64_t query_base =
        (query_row * num_q_heads + q_head) * head_dim;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      query_values[row_index][part] = row_valid[row_index]
          ? query[query_base + lane + part * 32]
          : 0.0f;
    }
  }

  const int64_t valid_group_rows =
      rows - query_row_base < kQueryRows ? rows - query_row_base : kQueryRows;
  const int64_t final_query_position =
      query_positions[valid_group_rows - 1];
  const int64_t final_query_live_count =
      final_query_position + 1 < capacity ? final_query_position + 1 : capacity;
  const int64_t live_count =
      prior_live_count > final_query_live_count
          ? prior_live_count
          : final_query_live_count;
  const int64_t scan_slots = live_count < capacity ? live_count : capacity;
  float running_max[kQueryRows];
  float denominators[kQueryRows] = {};
  float output_acc[kQueryRows][4] = {};
#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    running_max[row_index] = kLowest;
  }

  for (int64_t logical_slot = 0; logical_slot < scan_slots; ++logical_slot) {
    int64_t current_row = kCachedMetadata
        ? -1
        : logical_slot - start_slot;
    if (!kCachedMetadata && current_row < 0) current_row += capacity;
    const bool has_current = current_row >= 0 && current_row < rows;
    const int64_t physical_slot =
        static_cast<int64_t>(base_offsets[logical_slot]);
    const bool cache_valid = physical_slot >= 0 && physical_slot < capacity;
    bool from_current[kQueryRows];
    bool visible[kQueryRows];
    bool any_visible = false;
    bool need_current = false;
    bool need_cached = false;
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      const int64_t query_position = query_positions[row_index];
      from_current[row_index] =
          !kCachedMetadata &&
          row_valid[row_index] && has_current &&
          current_row <= query_rows[row_index];
      const int64_t key_position = kDenseInitialMetadata
          ? logical_slot
          : kCachedMetadata
          ? token_positions[logical_slot]
          : from_current[row_index]
          ? start_position + current_row
          : token_positions[logical_slot];
      const bool source_valid = kCachedMetadata
          ? cache_valid
          : from_current[row_index] || cache_valid;
      visible[row_index] = row_valid[row_index] && source_valid &&
          (kDenseInitialMetadata ||
           !(kCachedMetadata || !from_current[row_index]) ||
           evict_mask[logical_slot] == 0) &&
          key_position >= 0 && key_position <= query_position &&
          key_position > query_position - sliding_window;
      any_visible = any_visible || visible[row_index];
      need_current =
          need_current || (visible[row_index] && from_current[row_index]);
      need_cached =
          need_cached || (visible[row_index] && !from_current[row_index]);
    }
    if (!any_visible) continue;

    const int64_t current_base =
        (current_row * num_kv_heads + kv_head) * head_dim;
    const int64_t cache_base =
        (physical_slot * num_kv_heads + kv_head) * head_dim;
    float current_keys[4];
    float cached_keys[4];
    float current_values[4];
    float cached_values[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int64_t dim = lane + part * 32;
      current_keys[part] =
          !kCachedOnly &&
              (kQualifiedSourceLoads ? need_current : has_current)
          ? laguna_bf16_bits_to_float(
                laguna_float_to_bf16_bits(current_key[current_base + dim]))
          : 0.0f;
      current_values[part] =
          !kCachedOnly &&
              (kQualifiedSourceLoads ? need_current : has_current)
          ? laguna_bf16_bits_to_float(
                laguna_float_to_bf16_bits(current_value[current_base + dim]))
          : 0.0f;
      cached_keys[part] =
          (kCachedOnly ||
           (kQualifiedSourceLoads ? need_cached : cache_valid))
          ? laguna_bf16_bits_to_float(key_cache[cache_base + dim])
          : 0.0f;
      cached_values[part] =
          (kCachedOnly ||
           (kQualifiedSourceLoads ? need_cached : cache_valid))
          ? laguna_bf16_bits_to_float(value_cache[cache_base + dim])
          : 0.0f;
    }

#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      if (!visible[row_index]) continue;
      float products[4];
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const float key_value = !kCachedOnly && from_current[row_index]
            ? current_keys[part]
            : cached_keys[part];
        products[part] = laguna_ordered_mul_f32(
            query_values[row_index][part], key_value);
      }
      const float dot = laguna_wave32_sum_128_exact(
          products[0], products[1], products[2], products[3]);
      float rescale = 1.0f;
      float weight = 0.0f;
      if (lane == 0) {
        const float score = dot * scale;
        if (score > running_max[row_index]) {
          rescale = running_max[row_index] == kLowest
              ? 0.0f
              : expf(running_max[row_index] - score);
          denominators[row_index] =
              denominators[row_index] * rescale + 1.0f;
          running_max[row_index] = score;
          weight = 1.0f;
        } else {
          weight = expf(score - running_max[row_index]);
          denominators[row_index] += weight;
        }
      }
      rescale = __shfl(rescale, 0);
      weight = __shfl(weight, 0);
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const float value = !kCachedOnly && from_current[row_index]
            ? current_values[part]
            : cached_values[part];
        output_acc[row_index][part] =
            output_acc[row_index][part] * rescale + weight * value;
      }
    }
  }

#pragma unroll
  for (int row_index = 0; row_index < kQueryRows; ++row_index) {
    if (!row_valid[row_index]) continue;
    const float denominator = __shfl(denominators[row_index], 0);
    const float inv_denom = 1.0f / fmaxf(denominator, 1.0e-20f);
    const int64_t out_base =
        (query_rows[row_index] * num_q_heads + q_head) * head_dim;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      out[out_base + lane + part * 32] =
          output_acc[row_index][part] * inv_denom;
    }
  }
}

template <bool kGlobalLayout>
__global__ void laguna_dense_initial_cache_bf16_to_f32_kernel(
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    float* __restrict__ key_f32,
    float* __restrict__ value_f32,
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    int64_t context,
    int64_t capacity,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim) {
  if (live_counts[0] < context) return;
  const int64_t width = num_kv_heads * head_dim;
  const int64_t total = context * width;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t logical_slot = index / width;
    if (token_positions[logical_slot] != logical_slot ||
        evict_mask[logical_slot] != 0) {
      continue;
    }
    int64_t physical_slot;
    if constexpr (kGlobalLayout) {
      const int64_t logical_block = logical_slot / block_size;
      const int64_t block_offset =
          logical_slot - logical_block * block_size;
      physical_slot =
          static_cast<int64_t>(base_offsets[logical_block]) * block_size +
          block_offset;
    } else {
      physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    }
    if (physical_slot < 0 || physical_slot >= capacity) continue;
    const int64_t element = index - logical_slot * width;
    const int64_t cache_index = physical_slot * width + element;
    key_f32[index] = laguna_bf16_bits_to_float(key_cache[cache_index]);
    value_f32[index] = laguna_bf16_bits_to_float(value_cache[cache_index]);
  }
}

template <bool kGlobalLayout>
__global__ void laguna_dense_initial_cache_block_bf16_to_f32_kernel(
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    float* __restrict__ key_f32,
    float* __restrict__ value_f32,
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    int64_t logical_start,
    int64_t count,
    int64_t context,
    int64_t capacity,
    int64_t block_size,
    int64_t num_kv_heads,
    int64_t head_dim) {
  if (live_counts[0] < context) return;
  const int64_t width = num_kv_heads * head_dim;
  const int64_t total = count * width;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t relative_slot = index / width;
    const int64_t logical_slot = logical_start + relative_slot;
    if (token_positions[logical_slot] != logical_slot ||
        evict_mask[logical_slot] != 0) {
      continue;
    }
    int64_t physical_slot;
    if constexpr (kGlobalLayout) {
      const int64_t logical_block = logical_slot / block_size;
      const int64_t block_offset =
          logical_slot - logical_block * block_size;
      physical_slot =
          static_cast<int64_t>(base_offsets[logical_block]) * block_size +
          block_offset;
    } else {
      physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    }
    if (physical_slot < 0 || physical_slot >= capacity) continue;
    const int64_t element = index - relative_slot * width;
    const int64_t cache_index = physical_slot * width + element;
    key_f32[index] = laguna_bf16_bits_to_float(key_cache[cache_index]);
    value_f32[index] = laguna_bf16_bits_to_float(value_cache[cache_index]);
  }
}

__global__ void
laguna_dense_initial_contiguous_cache_block_bf16_to_f32_kernel(
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    float* __restrict__ key_f32,
    float* __restrict__ value_f32,
    const int64_t* __restrict__ live_counts,
    int64_t logical_start,
    int64_t count,
    int64_t context,
    int64_t num_kv_heads,
    int64_t head_dim) {
  if (live_counts[0] < context) return;
  const int64_t width = num_kv_heads * head_dim;
  const int64_t total = count * width;
  const int64_t cache_base = logical_start * width;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t cache_index = cache_base + index;
    key_f32[index] = laguna_bf16_bits_to_float(key_cache[cache_index]);
    value_f32[index] = laguna_bf16_bits_to_float(value_cache[cache_index]);
  }
}

__global__ void laguna_swa_union_bf16_to_f32_kernel(
    const float* __restrict__ current_key,
    const float* __restrict__ current_value,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    float* __restrict__ key_f32,
    float* __restrict__ value_f32,
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t sliding_window,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t start_position) {
  const int64_t union_context = sliding_window + rows - 1;
  if (live_counts[0] < sliding_window ||
      row_positions[0] != start_position) {
    return;
  }
  const int64_t width = num_kv_heads * head_dim;
  const int64_t total = union_context * width;
  const int64_t union_start = start_position - sliding_window + 1;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t union_slot = index / width;
    const int64_t element = index - union_slot * width;
    const int64_t key_position = union_start + union_slot;
    float key_value = 0.0f;
    float value_value = 0.0f;
    if (key_position >= start_position) {
      const int64_t current_row = key_position - start_position;
      if (current_row >= 0 && current_row < rows) {
        const int64_t current_index = current_row * width + element;
        key_value = laguna_bf16_bits_to_float(
            laguna_float_to_bf16_bits(current_key[current_index]));
        value_value = laguna_bf16_bits_to_float(
            laguna_float_to_bf16_bits(current_value[current_index]));
      }
    } else {
      int64_t logical_slot = key_position % capacity;
      if (logical_slot < 0) logical_slot += capacity;
      const int64_t physical_slot =
          static_cast<int64_t>(base_offsets[logical_slot]);
      if (token_positions[logical_slot] == key_position &&
          evict_mask[logical_slot] == 0 &&
          physical_slot >= 0 && physical_slot < capacity) {
        const int64_t cache_index = physical_slot * width + element;
        key_value = laguna_bf16_bits_to_float(key_cache[cache_index]);
        value_value = laguna_bf16_bits_to_float(value_cache[cache_index]);
      }
    }
    key_f32[index] = key_value;
    value_f32[index] = value_value;
  }
}

template <bool kToHeadMajor>
__global__ void laguna_dense_initial_query_head_transpose_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t rows,
    int64_t num_q_heads,
    int64_t head_dim) {
  const int64_t row_stride = num_q_heads * head_dim;
  const int64_t total = rows * row_stride;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t row = index / row_stride;
    const int64_t within_row = index - row * row_stride;
    const int64_t q_head = within_row / head_dim;
    const int64_t element = within_row - q_head * head_dim;
    const int64_t head_major =
        (q_head * rows + row) * head_dim + element;
    if constexpr (kToHeadMajor) {
      output[head_major] = input[index];
    } else {
      output[index] = input[head_major];
    }
  }
}

__device__ __forceinline__ float laguna_block_max_256(
    float value,
    float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down(value, offset));
  }
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < 8 ? shared[lane] : -3.4028234663852886e38f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value = fmaxf(value, __shfl_down(value, offset));
    }
    if (lane == 0) shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ float laguna_block_sum_256(
    float value,
    float* shared) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < 8 ? shared[lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down(value, offset);
    }
    if (lane == 0) shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__launch_bounds__(256, 2)
__global__ void laguna_dense_initial_causal_softmax_f32_kernel(
    float* __restrict__ scores,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    int64_t rows,
    int64_t context,
    int64_t num_q_heads,
    int64_t start_position,
    float scale) {
  const int64_t group = static_cast<int64_t>(blockIdx.x);
  const int64_t row = group % rows;
  const int64_t q_head = group / rows;
  if (q_head >= num_q_heads || live_counts[0] < context ||
      row_positions[0] != start_position ||
      token_positions[0] != 0 ||
      token_positions[context - 1] != context - 1 ||
      evict_mask[0] != 0 || evict_mask[context - 1] != 0) {
    return;
  }
  const int64_t visible = start_position + row + 1 < context
      ? start_position + row + 1
      : context;
  const int64_t base = (q_head * rows + row) * context;
  const int64_t token0 = threadIdx.x;
  const int64_t token1 = token0 + blockDim.x;
  const float lowest = -3.4028234663852886e38f;
  float local_max = lowest;
  if (token0 < visible) {
    local_max = scores[base + token0] * scale;
  }
  if (token1 < visible) {
    local_max = fmaxf(local_max, scores[base + token1] * scale);
  }
  __shared__ float reduction[8];
  const float maximum = laguna_block_max_256(local_max, reduction);
  float weight0 = 0.0f;
  float weight1 = 0.0f;
  if (token0 < visible) {
    weight0 = expf(scores[base + token0] * scale - maximum);
    scores[base + token0] = weight0;
  } else if (token0 < context) {
    scores[base + token0] = 0.0f;
  }
  if (token1 < visible) {
    weight1 = expf(scores[base + token1] * scale - maximum);
    scores[base + token1] = weight1;
  } else if (token1 < context) {
    scores[base + token1] = 0.0f;
  }
  const float denominator =
      laguna_block_sum_256(weight0 + weight1, reduction);
  const float inverse = 1.0f / fmaxf(denominator, 1.0e-20f);
  if (token0 < visible) scores[base + token0] *= inverse;
  if (token1 < visible) scores[base + token1] *= inverse;
}

__launch_bounds__(256, 2)
__global__ void laguna_dense_initial_causal_softmax_wave_rows_f32_kernel(
    float* __restrict__ scores,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    int64_t rows,
    int64_t context,
    int64_t num_q_heads,
    int64_t start_position,
    float scale) {
  constexpr int kWave = 32;
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t group = static_cast<int64_t>(blockIdx.x);
  const int64_t row = group % rows;
  const int64_t q_head = group / rows;
  if (q_head >= num_q_heads || live_counts[0] < context ||
      row_positions[0] != start_position ||
      token_positions[0] != 0 ||
      token_positions[context - 1] != context - 1 ||
      evict_mask[0] != 0 || evict_mask[context - 1] != 0) {
    return;
  }
  const int64_t visible = start_position + row + 1 < context
      ? start_position + row + 1
      : context;
  const int64_t base = (q_head * rows + row) * context;
  const float lowest = -3.4028234663852886e38f;
  float local_max = lowest;
  for (int64_t token = lane; token < visible; token += kWave) {
    local_max = fmaxf(local_max, scores[base + token] * scale);
  }
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) {
    local_max = fmaxf(
        local_max,
        __shfl_down(local_max, offset, kWave));
  }
  const float maximum = __shfl(local_max, 0, kWave);
  float local_sum = 0.0f;
  for (int64_t token = lane; token < context; token += kWave) {
    const float weight = token < visible
        ? expf(scores[base + token] * scale - maximum)
        : 0.0f;
    scores[base + token] = weight;
    local_sum += weight;
  }
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) {
    local_sum += __shfl_down(local_sum, offset, kWave);
  }
  const float denominator = __shfl(local_sum, 0, kWave);
  const float inverse = 1.0f / fmaxf(denominator, 1.0e-20f);
  for (int64_t token = lane; token < visible; token += kWave) {
    scores[base + token] *= inverse;
  }
}

__launch_bounds__(32, 8)
__global__ void
laguna_dense_initial_causal_softmax_tile_wave_rows_f32_kernel(
    float* __restrict__ scores,
    float* __restrict__ row_max,
    float* __restrict__ row_sum,
    float* __restrict__ merge_scales,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    int64_t rows,
    int64_t tile_start,
    int64_t tile_count,
    int64_t context,
    int64_t num_q_heads,
    int64_t start_position,
    float scale) {
  constexpr int kWave = 32;
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t group = static_cast<int64_t>(blockIdx.x);
  const int64_t row = group % rows;
  const int64_t q_head = group / rows;
  if (q_head >= num_q_heads || live_counts[0] < context ||
      row_positions[0] != start_position ||
      token_positions[tile_start] != tile_start ||
      token_positions[tile_start + tile_count - 1] !=
          tile_start + tile_count - 1 ||
      evict_mask[tile_start] != 0 ||
      evict_mask[tile_start + tile_count - 1] != 0) {
    return;
  }
  const int64_t visible_end = start_position + row + 1 < context
      ? start_position + row + 1
      : context;
  int64_t visible = visible_end - tile_start;
  visible = visible < 0 ? 0 : (visible < tile_count ? visible : tile_count);
  const int64_t base = group * tile_count;
  const float lowest = -3.4028234663852886e38f;
  float local_max = lowest;
  for (int64_t token = lane; token < visible; token += kWave) {
    local_max = fmaxf(local_max, scores[base + token] * scale);
  }
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) {
    local_max = fmaxf(
        local_max,
        __shfl_down(local_max, offset, kWave));
  }
  const float tile_maximum = __shfl(local_max, 0, kWave);
  float local_sum = 0.0f;
  for (int64_t token = lane; token < tile_count; token += kWave) {
    const float weight = token < visible
        ? expf(scores[base + token] * scale - tile_maximum)
        : 0.0f;
    scores[base + token] = weight;
    local_sum += weight;
  }
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) {
    local_sum += __shfl_down(local_sum, offset, kWave);
  }
  const float tile_sum = __shfl(local_sum, 0, kWave);
  if (lane == 0) {
    const bool first_tile = tile_start == 0;
    const float old_maximum = first_tile ? lowest : row_max[group];
    const float old_sum = first_tile ? 0.0f : row_sum[group];
    const float maximum = fmaxf(old_maximum, tile_maximum);
    const float old_scale =
        old_sum > 0.0f ? expf(old_maximum - maximum) : 0.0f;
    const float tile_scale =
        tile_sum > 0.0f ? expf(tile_maximum - maximum) : 0.0f;
    row_max[group] = maximum;
    row_sum[group] = old_sum * old_scale + tile_sum * tile_scale;
    merge_scales[group * 2] = old_scale;
    merge_scales[group * 2 + 1] = tile_scale;
  }
}

__launch_bounds__(32, 8)
__global__ void laguna_swa_union_softmax_wave_rows_f32_kernel(
    float* __restrict__ scores,
    int64_t rows,
    int64_t union_context,
    int64_t sliding_window,
    int64_t num_q_heads,
    float scale) {
  constexpr int kWave = 32;
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t group = static_cast<int64_t>(blockIdx.x);
  const int64_t row = group % rows;
  const int64_t q_head = group / rows;
  if (q_head >= num_q_heads) return;
  const int64_t visible_begin = row;
  const int64_t visible_end = row + sliding_window;
  const int64_t base = group * union_context;
  const float lowest = -3.4028234663852886e38f;
  float local_max = lowest;
  for (int64_t token = visible_begin + lane;
       token < visible_end;
       token += kWave) {
    local_max = fmaxf(local_max, scores[base + token] * scale);
  }
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down(local_max, offset, kWave));
  }
  const float maximum = __shfl(local_max, 0, kWave);
  float local_sum = 0.0f;
  for (int64_t token = lane; token < union_context; token += kWave) {
    const bool visible = token >= visible_begin && token < visible_end;
    const float weight = visible
        ? expf(scores[base + token] * scale - maximum)
        : 0.0f;
    scores[base + token] = weight;
    local_sum += weight;
  }
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) {
    local_sum += __shfl_down(local_sum, offset, kWave);
  }
  const float denominator = __shfl(local_sum, 0, kWave);
  const float inverse = 1.0f / fmaxf(denominator, 1.0e-20f);
  for (int64_t token = visible_begin + lane;
       token < visible_end;
       token += kWave) {
    scores[base + token] *= inverse;
  }
}

__launch_bounds__(256, 2)
__global__ void laguna_dense_initial_attention_tile_merge_f32_kernel(
    float* __restrict__ accumulator,
    const float* __restrict__ tile_output,
    float* __restrict__ final_output,
    const float* __restrict__ row_sum,
    const float* __restrict__ merge_scales,
    int64_t rows,
    int64_t num_q_heads,
    int64_t head_dim,
    int first_tile,
    int final_tile) {
  const int64_t total = rows * num_q_heads * head_dim;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t group = index / head_dim;
    const float old_scale = first_tile != 0
        ? 0.0f
        : merge_scales[group * 2];
    const float tile_scale = merge_scales[group * 2 + 1];
    const float value =
        (first_tile != 0 ? 0.0f : accumulator[index]) * old_scale +
        tile_output[index] * tile_scale;
    if (final_tile != 0) {
      final_output[index] =
          value / fmaxf(row_sum[group], 1.0e-20f);
    } else {
      accumulator[index] = value;
    }
  }
}

}  // namespace

extern "C" int hipengine_laguna_dense_initial_cache_bf16_to_f32_spans(
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* key_f32,
    float* value_f32,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    int64_t context,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    int global_layout,
    hipStream_t stream) {
  if (key_cache == nullptr || value_cache == nullptr || key_f32 == nullptr ||
      value_f32 == nullptr || base_offsets == nullptr ||
      live_counts == nullptr || token_positions == nullptr ||
      evict_mask == nullptr || context <= 0 || context > capacity ||
      capacity <= 0 || num_kv_heads != 8 || head_dim != 128 ||
      (global_layout != 0 && global_layout != 1) ||
      (global_layout != 0 &&
       (block_size != 256 || block_table_len <= 0 ||
        (capacity + block_size - 1) / block_size > block_table_len)) ||
      (global_layout == 0 && block_table_len < capacity)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = context * num_kv_heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  if (global_layout != 0) {
    hipLaunchKernelGGL(
        (laguna_dense_initial_cache_bf16_to_f32_kernel<true>),
        dim3(static_cast<unsigned int>(blocks)),
        dim3(threads),
        0,
        stream,
        key_cache,
        value_cache,
        key_f32,
        value_f32,
        base_offsets,
        live_counts,
        token_positions,
        evict_mask,
        context,
        capacity,
        block_size,
        num_kv_heads,
        head_dim);
  } else {
    hipLaunchKernelGGL(
        (laguna_dense_initial_cache_bf16_to_f32_kernel<false>),
        dim3(static_cast<unsigned int>(blocks)),
        dim3(threads),
        0,
        stream,
        key_cache,
        value_cache,
        key_f32,
        value_f32,
        base_offsets,
        live_counts,
        token_positions,
        evict_mask,
        context,
        capacity,
        block_size,
        num_kv_heads,
        head_dim);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_dense_initial_cache_block_bf16_to_f32_spans(
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* key_f32,
    float* value_f32,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    int64_t logical_start,
    int64_t count,
    int64_t context,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    int global_layout,
    hipStream_t stream) {
  if (key_cache == nullptr || value_cache == nullptr || key_f32 == nullptr ||
      value_f32 == nullptr || base_offsets == nullptr ||
      live_counts == nullptr || token_positions == nullptr ||
      evict_mask == nullptr || logical_start < 0 || count <= 0 ||
      context <= 0 || logical_start + count > context ||
      context > capacity || capacity <= 0 || num_kv_heads != 8 ||
      head_dim != 128 || (global_layout != 0 && global_layout != 1) ||
      (global_layout != 0 &&
       (block_size != 256 || block_table_len <= 0 ||
        (capacity + block_size - 1) / block_size > block_table_len)) ||
      (global_layout == 0 && block_table_len < capacity)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = count * num_kv_heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  if (global_layout != 0) {
    hipLaunchKernelGGL(
        (laguna_dense_initial_cache_block_bf16_to_f32_kernel<true>),
        dim3(static_cast<unsigned int>(blocks)),
        dim3(threads),
        0,
        stream,
        key_cache,
        value_cache,
        key_f32,
        value_f32,
        base_offsets,
        live_counts,
        token_positions,
        evict_mask,
        logical_start,
        count,
        context,
        capacity,
        block_size,
        num_kv_heads,
        head_dim);
  } else {
    hipLaunchKernelGGL(
        (laguna_dense_initial_cache_block_bf16_to_f32_kernel<false>),
        dim3(static_cast<unsigned int>(blocks)),
        dim3(threads),
        0,
        stream,
        key_cache,
        value_cache,
        key_f32,
        value_f32,
        base_offsets,
        live_counts,
        token_positions,
        evict_mask,
        logical_start,
        count,
        context,
        capacity,
        block_size,
        num_kv_heads,
        head_dim);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_dense_initial_contiguous_cache_block_bf16_to_f32_spans(
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* key_f32,
    float* value_f32,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    int64_t logical_start,
    int64_t count,
    int64_t context,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    int global_layout,
    hipStream_t stream) {
  if (key_cache == nullptr || value_cache == nullptr || key_f32 == nullptr ||
      value_f32 == nullptr || base_offsets == nullptr ||
      live_counts == nullptr || token_positions == nullptr ||
      evict_mask == nullptr || logical_start < 0 || count <= 0 ||
      context <= 0 || logical_start + count > context ||
      context > capacity || capacity <= 0 || num_kv_heads != 8 ||
      head_dim != 128 || global_layout != 1 || block_size != 256 ||
      block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = count * num_kv_heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(
      laguna_dense_initial_contiguous_cache_block_bf16_to_f32_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      key_cache,
      value_cache,
      key_f32,
      value_f32,
      live_counts,
      logical_start,
      count,
      context,
      num_kv_heads,
      head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_swa_union_bf16_to_f32_spans(
    const float* current_key,
    const float* current_value,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* key_f32,
    float* value_f32,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t capacity,
    int64_t sliding_window,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t start_position,
    hipStream_t stream) {
  if (current_key == nullptr || current_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr ||
      key_f32 == nullptr || value_f32 == nullptr ||
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || rows != 128 || capacity != 512 ||
      sliding_window != 512 || num_kv_heads != 8 || head_dim != 128 ||
      start_position < sliding_window) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t union_context = sliding_window + rows - 1;
  const int64_t total = union_context * num_kv_heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(
      laguna_swa_union_bf16_to_f32_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      current_key,
      current_value,
      key_cache,
      value_cache,
      key_f32,
      value_f32,

      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      rows,
      capacity,
      sliding_window,
      num_kv_heads,
      head_dim,
      start_position);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_dense_initial_query_head_transpose_f32(
    const float* input,
    float* output,
    int64_t rows,
    int64_t num_q_heads,
    int64_t head_dim,
    int to_head_major,
    hipStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 || rows > 2048 ||
      (num_q_heads != 48 && num_q_heads != 72) || head_dim != 128 ||
      (to_head_major != 0 && to_head_major != 1)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * num_q_heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  if (to_head_major != 0) {
    hipLaunchKernelGGL(
        (laguna_dense_initial_query_head_transpose_f32_kernel<true>),
        dim3(static_cast<unsigned int>(blocks)),
        dim3(threads),
        0,
        stream,
        input,
        output,
        rows,
        num_q_heads,
        head_dim);
  } else {
    hipLaunchKernelGGL(
        (laguna_dense_initial_query_head_transpose_f32_kernel<false>),
        dim3(static_cast<unsigned int>(blocks)),
        dim3(threads),
        0,
        stream,
        input,
        output,
        rows,
        num_q_heads,
        head_dim);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_dense_initial_causal_softmax_f32_spans(
    float* scores,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t context,
    int64_t num_q_heads,
    int64_t start_position,
    float scale,
    hipStream_t stream) {
  if (scores == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || rows <= 0 || rows > 2048 ||
      context <= 0 || context > 512 || num_q_heads <= 0 ||
      start_position < 0 || start_position + rows != context) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_dense_initial_causal_softmax_f32_kernel,
      dim3(static_cast<unsigned int>(rows * num_q_heads)),
      dim3(256),
      0,
      stream,
      scores,

      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      rows,
      context,
      num_q_heads,
      start_position,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_dense_initial_causal_softmax_wave_rows_f32_spans(
    float* scores,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t context,
    int64_t num_q_heads,
    int64_t start_position,
    float scale,
    hipStream_t stream) {
  if (scores == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || rows <= 0 || rows > 2048 ||
      context <= 0 || num_q_heads <= 0 ||
      start_position < 0 || start_position + rows != context) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_dense_initial_causal_softmax_wave_rows_f32_kernel,
      dim3(static_cast<unsigned int>(rows * num_q_heads)),
      dim3(32),
      0,
      stream,
      scores,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      rows,
      context,
      num_q_heads,
      start_position,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_dense_initial_causal_softmax_tile_wave_rows_f32_spans(
    float* scores,
    float* row_max,
    float* row_sum,
    float* merge_scales,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t rows,
    int64_t tile_start,
    int64_t tile_count,
    int64_t context,
    int64_t num_q_heads,
    int64_t start_position,
    float scale,
    hipStream_t stream) {
  if (scores == nullptr || row_max == nullptr || row_sum == nullptr ||
      merge_scales == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || rows <= 0 || rows > 2048 ||
      tile_start < 0 || tile_count <= 0 ||
      tile_start + tile_count > context || context <= 0 ||
      num_q_heads <= 0 || start_position < 0 ||
      start_position + rows != context) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_dense_initial_causal_softmax_tile_wave_rows_f32_kernel,
      dim3(static_cast<unsigned int>(rows * num_q_heads)),
      dim3(32),
      0,
      stream,
      scores,
      row_max,
      row_sum,
      merge_scales,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      rows,
      tile_start,
      tile_count,
      context,
      num_q_heads,
      start_position,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_swa_union_softmax_wave_rows_f32(
    float* scores,
    int64_t rows,
    int64_t union_context,
    int64_t sliding_window,
    int64_t num_q_heads,
    float scale,
    hipStream_t stream) {
  if (scores == nullptr || rows != 128 || union_context != 639 ||
      sliding_window != 512 || num_q_heads != 72) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_swa_union_softmax_wave_rows_f32_kernel,
      dim3(static_cast<unsigned int>(rows * num_q_heads)),
      dim3(32),
      0,
      stream,
      scores,
      rows,
      union_context,
      sliding_window,
      num_q_heads,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_dense_initial_attention_tile_merge_f32(
    float* accumulator,
    const float* tile_output,
    float* final_output,
    const float* row_sum,
    const float* merge_scales,
    int64_t rows,
    int64_t num_q_heads,
    int64_t head_dim,
    int first_tile,
    int final_tile,
    hipStream_t stream) {
  if (accumulator == nullptr || tile_output == nullptr ||
      final_output == nullptr || row_sum == nullptr ||
      merge_scales == nullptr || rows <= 0 || rows > 2048 ||
      num_q_heads <= 0 || head_dim != 128 ||
      (first_tile != 0 && first_tile != 1) ||
      (final_tile != 0 && final_tile != 1)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * num_q_heads * head_dim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(
      laguna_dense_initial_attention_tile_merge_f32_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      accumulator,
      tile_output,
      final_output,
      row_sum,
      merge_scales,
      rows,
      num_q_heads,
      head_dim,
      first_tile,
      final_tile);
  return static_cast<int>(hipGetLastError());

}

extern "C" int hipengine_laguna_global_head_rmsnorm_rope_write_kv_f32_bf16_spans(
    const float* query,
    const float* key,
    const float* value,
    const float* q_weight,
    const float* k_weight,
    const float* cos_table,
    const float* sin_table,
    float* query_out,
    float* key_out,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    int64_t* live_counts,
    int64_t* token_positions,
    uint8_t* evict_mask,
    const int64_t* row_positions,
    float eps,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    hipStream_t stream) {
  if (capacity <= 0 || block_size <= 0 || block_table_len <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || rotary_dim > head_dim || max_positions <= 0) {
    return hipErrorInvalidValue;
  }
  constexpr int threads = 256;
  laguna_head_rmsnorm_partial_rotary_write_kv_f32_bf16_kernel<true><<<
      static_cast<unsigned int>(num_q_heads + num_kv_heads),
      threads,
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(
      query,
      key,
      value,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      query_out,
      key_out,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      eps,
      capacity,
      block_size,
      block_table_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions);
  return hipGetLastError();
}

extern "C" int hipengine_laguna_global_head_rmsnorm_rope_write_kv_wave0_tree_f32_bf16_spans(
    const float* query,
    const float* key,
    const float* value,
    const float* q_weight,
    const float* k_weight,
    const float* cos_table,
    const float* sin_table,
    float* query_out,
    float* key_out,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    int64_t* live_counts,
    int64_t* token_positions,
    uint8_t* evict_mask,
    const int64_t* row_positions,
    float eps,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    hipStream_t stream) {
  if (capacity <= 0 || block_size != 256 || block_table_len <= 0 ||
      num_q_heads != 48 || num_kv_heads != 8 || head_dim != 128 ||
      rotary_dim <= 0 || rotary_dim > head_dim || max_positions <= 0) {
    return hipErrorInvalidValue;
  }
  constexpr int threads = 256;
  laguna_global_head_rmsnorm_partial_rotary_write_kv_wave0_tree_f32_bf16_kernel<<<
      static_cast<unsigned int>(num_q_heads + num_kv_heads),
      threads,
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(
      query,
      key,
      value,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      query_out,
      key_out,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      eps,
      capacity,
      block_size,
      block_table_len,
      num_q_heads,
      num_kv_heads,
      rotary_dim,
      max_positions);
  return hipGetLastError();
}

extern "C" int hipengine_laguna_swa_head_rmsnorm_rope_write_kv_f32_bf16_spans(
    const float* query,
    const float* key,
    const float* value,
    const float* q_weight,
    const float* k_weight,
    const float* cos_table,
    const float* sin_table,
    float* query_out,
    float* key_out,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    int64_t* live_counts,
    int64_t* token_positions,
    uint8_t* evict_mask,
    const int64_t* row_positions,
    float eps,
    int64_t capacity,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    hipStream_t stream) {
  if (capacity <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || rotary_dim > head_dim || max_positions <= 0) {
    return hipErrorInvalidValue;
  }
  constexpr int threads = 256;
  laguna_head_rmsnorm_partial_rotary_write_kv_f32_bf16_kernel<false><<<
      static_cast<unsigned int>(num_q_heads + num_kv_heads),
      threads,
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(
      query,
      key,
      value,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      query_out,
      key_out,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      eps,
      capacity,
      0,
      0,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions);
  return hipGetLastError();
}

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

extern "C" int hipengine_laguna_global_attention_decode_single_page_bf16_spans(
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
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      out == nullptr || base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity <= 0 || block_size != 256 ||
      block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len ||
      num_q_heads != 48 || num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int warps = threads / 32;
  const size_t shared_bytes =
      static_cast<size_t>(capacity + warps + head_dim) * sizeof(float);
  hipLaunchKernelGGL(
      laguna_global_attention_decode_single_page_bf16_kernel,
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

extern "C" int hipengine_laguna_global_attention_decode_single_page_softplus_gate_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const float* gate,
    uint16_t* gated_out,
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
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      out == nullptr || gate == nullptr || gated_out == nullptr ||
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity <= 0 || block_size != 256 ||
      block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len ||
      num_q_heads != 48 || num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int warps = threads / 32;
  const size_t shared_bytes =
      static_cast<size_t>(capacity + warps + head_dim) * sizeof(float);
  hipLaunchKernelGGL(
      laguna_global_attention_decode_single_page_softplus_gate_bf16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
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

extern "C" int hipengine_laguna_global_attention_decode_split_exact_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      out == nullptr || score_scratch == nullptr || physical_scratch == nullptr ||
      capacity <= 0 || block_size != 256 || block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len ||
      scratch_stride < scan_slots || scan_slots <= 0 || scan_slots > capacity ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_global_attention_split_exact_scores_kernel,
      dim3(static_cast<unsigned int>(num_q_heads),
           static_cast<unsigned int>(scan_slots)),
      dim3(32),
      0,
      stream,
      query,
      key_cache,
      score_scratch,
      physical_scratch,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      block_size,
      scratch_stride,
      scan_slots,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  int error = static_cast<int>(hipGetLastError());
  if (error != static_cast<int>(hipSuccess)) return error;
  hipLaunchKernelGGL(
      laguna_global_attention_split_exact_reduce_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(256),
      0,
      stream,
      value_cache,
      score_scratch,
      physical_scratch,
      out,
      live_counts,
      row_positions,
      capacity,
      scratch_stride,
      scan_slots,
      num_q_heads,
      num_kv_heads,
      head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_global_attention_decode_split_exact_gated_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t block_size,
    int64_t block_table_len,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      out == nullptr || gate == nullptr || gated_out == nullptr ||
      score_scratch == nullptr || physical_scratch == nullptr ||
      capacity <= 0 || block_size != 256 || block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len ||
      scratch_stride < scan_slots || scan_slots <= 0 || scan_slots > capacity ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_global_attention_split_exact_scores_kernel,
      dim3(static_cast<unsigned int>(num_q_heads),
           static_cast<unsigned int>(scan_slots)),
      dim3(32),
      0,
      stream,
      query,
      key_cache,
      score_scratch,
      physical_scratch,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      block_size,
      scratch_stride,
      scan_slots,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  int error = static_cast<int>(hipGetLastError());
  if (error != static_cast<int>(hipSuccess)) return error;
  hipLaunchKernelGGL(
      laguna_global_attention_split_exact_gated_reduce_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(256),
      0,
      stream,
      value_cache,
      score_scratch,
      physical_scratch,
      out,
      gate,
      gated_out,
      live_counts,
      row_positions,
      capacity,
      scratch_stride,
      scan_slots,
      num_q_heads,
      num_kv_heads,
      head_dim);
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

extern "C" int hipengine_laguna_global_attention_prefill_cached_exact_bf16_spans(
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
    int64_t start_position,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || current_key == nullptr || current_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr || out == nullptr ||
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || rows != 128 || capacity != 4096 ||
      block_size != 256 || block_table_len < 16 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128 ||
      (start_position != 0 && start_position != 128 &&
       start_position != 256 && start_position != 384)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int warps = threads / 32;
  const size_t shared_bytes =
      static_cast<size_t>(capacity + warps + head_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (laguna_global_attention_prefill_cached_exact_bf16_kernel<false>),
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)),
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
      rows,
      capacity,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      start_position,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_global_attention_prefill_dense_initial_cached_exact_bf16_spans(
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
    int64_t start_position,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || current_key == nullptr || current_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr || out == nullptr ||
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || rows != 128 || capacity != 4096 ||
      block_size != 256 || block_table_len < 16 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128 ||
      (start_position != 0 && start_position != 128 &&
       start_position != 256 && start_position != 384)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int warps = threads / 32;
  const size_t shared_bytes =
      static_cast<size_t>(capacity + warps + head_dim) * sizeof(float);
  hipLaunchKernelGGL(
      (laguna_global_attention_prefill_cached_exact_bf16_kernel<true>),
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)),
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
      rows,
      capacity,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      start_position,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_global_attention_prefill_dense_initial_fixed512_cached_exact_bf16_spans(
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
    int64_t start_position,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || current_key == nullptr || current_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr || out == nullptr ||
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || rows != 128 || capacity != 4096 ||
      block_size != 256 || block_table_len < 16 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128 ||
      (start_position != 0 && start_position != 128 &&
       start_position != 256 && start_position != 384)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int64_t kScoreCapacity = 512;
  const int threads = 256;
  const int warps = threads / 32;
  const size_t shared_bytes =
      static_cast<size_t>(kScoreCapacity + warps + head_dim) * sizeof(float);
  hipLaunchKernelGGL(
      laguna_global_attention_prefill_dense_initial_fixed512_cached_exact_bf16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)),
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
      rows,
      capacity,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      start_position,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_global_attention_prefill_qrow4_dense_initial_global_score_weight_replay_exact_bf16_spans(
    const float* query,
    const float* current_key,
    const float* current_value,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    float4* score_records,
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
    int64_t start_position,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || current_key == nullptr || current_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr || out == nullptr ||
      score_records == nullptr ||
      (reinterpret_cast<unsigned long long>(score_records) & 15ULL) != 0 ||
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || rows != 128 || capacity != 4096 ||
      block_size != 256 || block_table_len < 16 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128 ||
      (start_position != 256 && start_position != 384)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_global_attention_prefill_qrow4_dense_initial_global_score_weight_replay_exact_bf16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), 32),
      dim3(32),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      score_records,
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
      start_position,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_global_attention_prefill_qrow2_online_bf16_spans(
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
      block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int row_groups = static_cast<unsigned int>((rows + 1) / 2);
  hipLaunchKernelGGL(
      (laguna_global_attention_prefill_qrows_online_bf16_kernel<2>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_global_attention_prefill_qrow4_online_bf16_spans(
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
      block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int row_groups = static_cast<unsigned int>((rows + 3) / 4);
  hipLaunchKernelGGL(
      (laguna_global_attention_prefill_qrows_online_bf16_kernel<4>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_global_attention_prefill_qrow4_cached_online_bf16_spans(
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
      block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int row_groups = static_cast<unsigned int>((rows + 3) / 4);
  hipLaunchKernelGGL(
      (laguna_global_attention_prefill_qrows_online_bf16_kernel<4, true>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans(
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
      block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int row_groups = static_cast<unsigned int>((rows + 3) / 4);
  hipLaunchKernelGGL(
      (laguna_global_attention_prefill_qrows_online_bf16_kernel<4, true, true>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_global_attention_prefill_qrow6_cached_meta_online_bf16_spans(
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
      block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int row_groups = static_cast<unsigned int>((rows + 5) / 6);
  hipLaunchKernelGGL(
      (laguna_global_attention_prefill_qrows_online_bf16_kernel<6, true, true>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_global_attention_prefill_qrow4_dense_initial_online_bf16_spans(
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
      block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int row_groups = static_cast<unsigned int>((rows + 3) / 4);
  hipLaunchKernelGGL(
      (laguna_global_attention_prefill_qrows_online_bf16_kernel<
          4, true, true, true>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_global_attention_prefill_qrow6_dense_initial_online_bf16_spans(
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
      block_table_len <= 0 ||
      (capacity + block_size - 1) / block_size > block_table_len ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const unsigned int row_groups = static_cast<unsigned int>((rows + 5) / 6);
  hipLaunchKernelGGL(
      (laguna_global_attention_prefill_qrows_online_bf16_kernel<
          6, true, true, true>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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

extern "C" int hipengine_laguna_swa_attention_decode_split_exact_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t sliding_window,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      out == nullptr || score_scratch == nullptr || physical_scratch == nullptr ||
      capacity <= 0 || sliding_window <= 0 || sliding_window > capacity ||
      scratch_stride < scan_slots || scan_slots <= 0 || scan_slots > capacity ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_swa_attention_split_exact_scores_kernel,
      dim3(static_cast<unsigned int>(num_q_heads),
           static_cast<unsigned int>(scan_slots)),
      dim3(32),
      0,
      stream,
      query,
      key_cache,
      score_scratch,
      physical_scratch,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      sliding_window,
      scratch_stride,
      scan_slots,
      num_q_heads,
      num_kv_heads,
      head_dim);
  int error = static_cast<int>(hipGetLastError());
  if (error != static_cast<int>(hipSuccess)) return error;
  hipLaunchKernelGGL(
      laguna_swa_attention_split_exact_reduce_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(128),
      0,
      stream,
      value_cache,
      score_scratch,
      physical_scratch,
      out,
      live_counts,
      row_positions,
      capacity,
      scratch_stride,
      scan_slots,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_swa_attention_decode_split_exact_gated_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t sliding_window,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      out == nullptr || gate == nullptr || gated_out == nullptr ||
      score_scratch == nullptr || physical_scratch == nullptr ||
      capacity <= 0 || sliding_window <= 0 || sliding_window > capacity ||
      scratch_stride < scan_slots || scan_slots <= 0 || scan_slots > capacity ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_swa_attention_split_exact_scores_kernel,
      dim3(static_cast<unsigned int>(num_q_heads),
           static_cast<unsigned int>(scan_slots)),
      dim3(32),
      0,
      stream,
      query,
      key_cache,
      score_scratch,
      physical_scratch,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      sliding_window,
      scratch_stride,
      scan_slots,
      num_q_heads,
      num_kv_heads,
      head_dim);
  int error = static_cast<int>(hipGetLastError());
  if (error != static_cast<int>(hipSuccess)) return error;
  hipLaunchKernelGGL(
      laguna_swa_attention_split_exact_gated_reduce_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(128),
      0,
      stream,
      value_cache,
      score_scratch,
      physical_scratch,
      out,
      gate,
      gated_out,
      live_counts,
      row_positions,
      capacity,
      scratch_stride,
      scan_slots,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_split_tile16_exact_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t sliding_window,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      out == nullptr || score_scratch == nullptr || physical_scratch == nullptr ||
      capacity <= 0 || sliding_window <= 0 || sliding_window > capacity ||
      scratch_stride < scan_slots || scan_slots <= 0 || scan_slots > capacity ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t score_tiles =
      (scan_slots + LAGUNA_SPLIT_SCORE_TILE - 1) / LAGUNA_SPLIT_SCORE_TILE;
  hipLaunchKernelGGL(
      laguna_swa_attention_split_tile16_exact_scores_kernel,
      dim3(static_cast<unsigned int>(num_q_heads),
           static_cast<unsigned int>(score_tiles)),
      dim3(256),
      0,
      stream,
      query,
      key_cache,
      score_scratch,
      physical_scratch,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      sliding_window,
      scratch_stride,
      scan_slots,
      num_q_heads,
      num_kv_heads,
      head_dim);
  int error = static_cast<int>(hipGetLastError());
  if (error != static_cast<int>(hipSuccess)) return error;
  hipLaunchKernelGGL(
      laguna_swa_attention_split_exact_reduce_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(128),
      0,
      stream,
      value_cache,
      score_scratch,
      physical_scratch,
      out,
      live_counts,
      row_positions,
      capacity,
      scratch_stride,
      scan_slots,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_split_tile16_exact_gated_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t sliding_window,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      out == nullptr || gate == nullptr || gated_out == nullptr ||
      score_scratch == nullptr || physical_scratch == nullptr ||
      capacity <= 0 || sliding_window <= 0 || sliding_window > capacity ||
      scratch_stride < scan_slots || scan_slots <= 0 || scan_slots > capacity ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t score_tiles =
      (scan_slots + LAGUNA_SPLIT_SCORE_TILE - 1) / LAGUNA_SPLIT_SCORE_TILE;
  hipLaunchKernelGGL(
      laguna_swa_attention_split_tile16_exact_scores_kernel,
      dim3(static_cast<unsigned int>(num_q_heads),
           static_cast<unsigned int>(score_tiles)),
      dim3(256),
      0,
      stream,
      query,
      key_cache,
      score_scratch,
      physical_scratch,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      capacity,
      sliding_window,
      scratch_stride,
      scan_slots,
      num_q_heads,
      num_kv_heads,
      head_dim);
  int error = static_cast<int>(hipGetLastError());
  if (error != static_cast<int>(hipSuccess)) return error;
  hipLaunchKernelGGL(
      laguna_swa_attention_split_exact_gated_reduce_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(128),
      0,
      stream,
      value_cache,
      score_scratch,
      physical_scratch,
      out,
      gate,
      gated_out,
      live_counts,
      row_positions,
      capacity,
      scratch_stride,
      scan_slots,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

template <bool TILE16_SCORES, bool DIM2>
static int laguna_launch_swa_attention_split_exact_gated_wave_local(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t sliding_window,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      out == nullptr || gate == nullptr || gated_out == nullptr ||
      score_scratch == nullptr || physical_scratch == nullptr ||
      capacity <= 0 || sliding_window <= 0 || sliding_window > capacity ||
      scratch_stride < scan_slots || scan_slots <= 0 || scan_slots > capacity ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if constexpr (TILE16_SCORES) {
    const int64_t score_tiles =
        (scan_slots + LAGUNA_SPLIT_SCORE_TILE - 1) / LAGUNA_SPLIT_SCORE_TILE;
    hipLaunchKernelGGL(
        laguna_swa_attention_split_tile16_exact_scores_kernel,
        dim3(static_cast<unsigned int>(num_q_heads),
             static_cast<unsigned int>(score_tiles)),
        dim3(256),
        0,
        stream,
        query,
        key_cache,
        score_scratch,
        physical_scratch,
        base_offsets,
        live_counts,
        token_positions,
        evict_mask,
        row_positions,
        capacity,
        sliding_window,
        scratch_stride,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim);
  } else {
    hipLaunchKernelGGL(
        laguna_swa_attention_split_exact_scores_kernel,
        dim3(static_cast<unsigned int>(num_q_heads),
             static_cast<unsigned int>(scan_slots)),
        dim3(32),
        0,
        stream,
        query,
        key_cache,
        score_scratch,
        physical_scratch,
        base_offsets,
        live_counts,
        token_positions,
        evict_mask,
        row_positions,
        capacity,
        sliding_window,
        scratch_stride,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim);
  }
  int error = static_cast<int>(hipGetLastError());
  if (error != static_cast<int>(hipSuccess)) return error;
  if constexpr (DIM2) {
    hipLaunchKernelGGL(
        laguna_swa_attention_split_exact_gated_wave_local_dim2_reduce_kernel,
        dim3(static_cast<unsigned int>(num_q_heads)),
        dim3(64),
        0,
        stream,
        value_cache,
        score_scratch,
        physical_scratch,
        out,
        gate,
        gated_out,
        live_counts,
        row_positions,
        capacity,
        scratch_stride,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale);
  } else {
    hipLaunchKernelGGL(
        laguna_swa_attention_split_exact_gated_wave_local_reduce_kernel,
        dim3(static_cast<unsigned int>(num_q_heads)),
        dim3(128),
        0,
        stream,
        value_cache,
        score_scratch,
        physical_scratch,
        out,
        gate,
        gated_out,
        live_counts,
        row_positions,
        capacity,
        scratch_stride,
        scan_slots,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale);
  }
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_split_exact_gated_wave_local_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t sliding_window,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  return laguna_launch_swa_attention_split_exact_gated_wave_local<false, false>(
      query, key_cache, value_cache, out, gate, gated_out, score_scratch,
      physical_scratch, base_offsets, live_counts, token_positions, evict_mask,
      row_positions, capacity, sliding_window, scratch_stride, scan_slots,
      num_q_heads, num_kv_heads, head_dim, scale, stream);
}

extern "C" int
hipengine_laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t sliding_window,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  return laguna_launch_swa_attention_split_exact_gated_wave_local<true, false>(
      query, key_cache, value_cache, out, gate, gated_out, score_scratch,
      physical_scratch, base_offsets, live_counts, token_positions, evict_mask,
      row_positions, capacity, sliding_window, scratch_stride, scan_slots,
      num_q_heads, num_kv_heads, head_dim, scale, stream);
}

extern "C" int
hipengine_laguna_swa_attention_decode_split_exact_gated_wave_local_dim2_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t sliding_window,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  return laguna_launch_swa_attention_split_exact_gated_wave_local<false, true>(
      query, key_cache, value_cache, out, gate, gated_out, score_scratch,
      physical_scratch, base_offsets, live_counts, token_positions, evict_mask,
      row_positions, capacity, sliding_window, scratch_stride, scan_slots,
      num_q_heads, num_kv_heads, head_dim, scale, stream);
}

extern "C" int
hipengine_laguna_swa_attention_decode_split_tile16_exact_gated_wave_local_dim2_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    float* score_scratch,
    int32_t* physical_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t capacity,
    int64_t sliding_window,
    int64_t scratch_stride,
    int64_t scan_slots,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  return laguna_launch_swa_attention_split_exact_gated_wave_local<true, true>(
      query, key_cache, value_cache, out, gate, gated_out, score_scratch,
      physical_scratch, base_offsets, live_counts, token_positions, evict_mask,
      row_positions, capacity, sliding_window, scratch_stride, scan_slots,
      num_q_heads, num_kv_heads, head_dim, scale, stream);
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

extern "C" int hipengine_laguna_swa_attention_prefill_qrow2_exact_bf16_spans(
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
  const unsigned int row_groups = static_cast<unsigned int>((rows + 1) / 2);
  hipLaunchKernelGGL(
      (laguna_swa_attention_prefill_qrows_exact_bf16_kernel<2>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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

extern "C" int hipengine_laguna_swa_attention_prefill_qrow4_exact_bf16_spans(
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
  const unsigned int row_groups = static_cast<unsigned int>((rows + 3) / 4);
  hipLaunchKernelGGL(
      (laguna_swa_attention_prefill_qrows_exact_bf16_kernel<4>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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

extern "C" int hipengine_laguna_swa_attention_prefill_qrow4_sourcequal_exact_bf16_spans(
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
  const unsigned int row_groups = static_cast<unsigned int>((rows + 3) / 4);
  hipLaunchKernelGGL(
      (laguna_swa_attention_prefill_qrows_exact_bf16_kernel<4, true>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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

extern "C" int hipengine_laguna_swa_attention_prefill_qrow4_dense_first_fill_exact_bf16_spans(
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
  if (rows != 128 || capacity != 512 || sliding_window != 512 ||
      num_q_heads != 72 || num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_prefill_qrows_exact_bf16_kernel<4, true, true>),
      dim3(static_cast<unsigned int>(num_q_heads), 32),
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

extern "C" int hipengine_laguna_swa_attention_prefill_qrow4_cached_exact_bf16_spans(
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
    int64_t start_position,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || current_key == nullptr || current_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr || out == nullptr ||
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || rows != 128 || capacity != 512 ||
      sliding_window != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128 ||
      (start_position != 0 && start_position != 128 &&
       start_position != 256 && start_position != 384)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_attention_prefill_qrow4_cached_exact_bf16_kernel<false, false>),
      dim3(static_cast<unsigned int>(num_q_heads), 32),
      dim3(32),
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
      rows,
      capacity,
      sliding_window,
      num_q_heads,
      num_kv_heads,
      head_dim,
      start_position,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_swa_attention_prefill_qrow4_dense_initial_cached_exact_bf16_spans(
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
    int64_t start_position,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || current_key == nullptr || current_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr || out == nullptr ||
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || rows != 128 || capacity != 512 ||
      sliding_window != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128 ||
      (start_position != 0 && start_position != 128 &&
       start_position != 256 && start_position != 384)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_attention_prefill_qrow4_cached_exact_bf16_kernel<false, true>),
      dim3(static_cast<unsigned int>(num_q_heads), 32),
      dim3(32),
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
      rows,
      capacity,
      sliding_window,
      num_q_heads,
      num_kv_heads,
      head_dim,
      start_position,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_swa_attention_prefill_qrow4_dense_initial_global_score_replay_exact_bf16_spans(
    const float* query,
    const float* current_key,
    const float* current_value,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    float4* score_records,
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
    int64_t start_position,
    float scale,
    hipStream_t stream) {
  if (query == nullptr || current_key == nullptr || current_value == nullptr ||
      key_cache == nullptr || value_cache == nullptr || out == nullptr ||
      score_records == nullptr ||
      (reinterpret_cast<unsigned long long>(score_records) & 15ULL) != 0 ||
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || rows != 128 || capacity != 512 ||
      sliding_window != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128 ||
      (start_position != 256 && start_position != 384)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_swa_attention_prefill_qrow4_dense_initial_global_score_replay_exact_bf16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), 32),
      dim3(32),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      score_records,
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
      start_position,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_laguna_swa_attention_prefill_qrow2_online_bf16_spans(
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
  const unsigned int row_groups = static_cast<unsigned int>((rows + 1) / 2);
  hipLaunchKernelGGL(
      (laguna_swa_attention_prefill_qrows_online_bf16_kernel<2>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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

extern "C" int hipengine_laguna_swa_attention_prefill_qrow4_online_bf16_spans(
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
  const unsigned int row_groups = static_cast<unsigned int>((rows + 3) / 4);
  hipLaunchKernelGGL(
      (laguna_swa_attention_prefill_qrows_online_bf16_kernel<4>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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

extern "C" int hipengine_laguna_swa_attention_prefill_qrow4_sourcequal_online_bf16_spans(
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
  const unsigned int row_groups = static_cast<unsigned int>((rows + 3) / 4);
  hipLaunchKernelGGL(
      (laguna_swa_attention_prefill_qrows_online_bf16_kernel<4, true>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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

extern "C" int hipengine_laguna_swa_attention_prefill_qrow4_cached_online_bf16_spans(
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
  const unsigned int row_groups = static_cast<unsigned int>((rows + 3) / 4);
  hipLaunchKernelGGL(
      (laguna_swa_attention_prefill_qrows_online_bf16_kernel<4, true, true>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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

extern "C" int hipengine_laguna_swa_attention_prefill_qrow4_cached_meta_online_bf16_spans(
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
  const unsigned int row_groups = static_cast<unsigned int>((rows + 3) / 4);
  hipLaunchKernelGGL(
      (laguna_swa_attention_prefill_qrows_online_bf16_kernel<4, true, true, true>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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

extern "C" int hipengine_laguna_swa_attention_prefill_qrow4_dense_initial_online_bf16_spans(
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
  const unsigned int row_groups = static_cast<unsigned int>((rows + 3) / 4);
  hipLaunchKernelGGL(
      (laguna_swa_attention_prefill_qrows_online_bf16_kernel<
          4, true, true, true, true>),
      dim3(static_cast<unsigned int>(num_q_heads), row_groups),
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
