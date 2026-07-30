// hipEngine-native Laguna global/SWA KV and attention.
// Every body consumes the complete KVLiveSpans ABI: physical offsets, live
// count, absolute slot positions, eviction mask, and absolute query position.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

typedef uint16_t laguna_bf16_t;
typedef laguna_bf16_t __attribute__((ext_vector_type(16))) laguna_bf16x16_t;
typedef float __attribute__((ext_vector_type(8))) laguna_float8_t;

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

__device__ inline laguna_float8_t laguna_wmma_zero_acc() {
  laguna_float8_t acc;
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    acc[item] = 0.0f;
  }
  return acc;
}

__device__ inline laguna_bf16x16_t laguna_wmma_zero_bf16x16() {
  laguna_bf16x16_t values;
#pragma unroll
  for (int item = 0; item < 16; ++item) {
    values[item] = 0;
  }
  return values;
}

__device__ inline void laguna_three_term_bf16(
    float value,
    uint16_t& term0,
    uint16_t& term1,
    uint16_t& term2) {
  term0 = laguna_float_to_bf16_bits(value);
  const float residual0 = value - laguna_bf16_bits_to_float(term0);
  term1 = laguna_float_to_bf16_bits(residual0);
  const float residual1 =
      residual0 - laguna_bf16_bits_to_float(term1);
  term2 = laguna_float_to_bf16_bits(residual1);
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

// Natural Laguna global-attention reducer specialization. The live length
// remains dynamic, while the production 48Q/8KV/D128/capacity-4096 shape and
// scratch stride become compile-time constants. Grid, reduction tree, scalar
// slot order, and per-dimension FMA order match the generic exact owner.
__global__ __launch_bounds__(256)
void laguna_global_attention_split_exact_gated_fixedshape_reduce_kernel(
    const uint16_t* value_cache,
    float* score_scratch,
    const int32_t* physical_scratch,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    const int64_t* live_counts,
    const int64_t* row_positions,
    int scan_slots) {
  constexpr int kCapacity = 4096;
  constexpr int kScratchStride = 4096;
  constexpr int kNumKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kKvGroup = 6;
  constexpr int kNumWarps = 8;
  const int q_head = static_cast<int>(blockIdx.x);
  int context_len = static_cast<int>(
      live_counts[0] < kCapacity ? live_counts[0] : kCapacity);
  context_len = context_len < scan_slots ? context_len : scan_slots;
  if (context_len <= 0 || row_positions[0] < 0) return;

  __shared__ float warp_buf[kNumWarps];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int scratch_base = q_head * kScratchStride;

  float local_max = -3.4028234663852886e38f;
  for (int token = warp; token < context_len; token += kNumWarps) {
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
    for (int item = 1; item < kNumWarps; ++item) {
      max_score = fmaxf(max_score, warp_buf[item]);
    }
    warp_buf[0] = max_score;
  }
  __syncthreads();
  const float max_score = warp_buf[0];

  float local_sum = 0.0f;
  for (int token = warp; token < context_len; token += kNumWarps) {
    if (lane == 0) {
      const int idx = scratch_base + token;
      const bool visible = physical_scratch[idx] >= 0;
      const float weight =
          visible ? expf(score_scratch[idx] - max_score) : 0.0f;
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
    for (int item = 1; item < kNumWarps; ++item) total_sum += warp_buf[item];
    warp_buf[0] = 1.0f / fmaxf(total_sum, 1.0e-20f);
  }
  __syncthreads();
  const float inv_denom = warp_buf[0];
  const int kv_head = q_head / kKvGroup;
  const int q_offset = q_head * kHeadDim;
  for (int dim = static_cast<int>(threadIdx.x);
       dim < kHeadDim;
       dim += static_cast<int>(blockDim.x)) {
    float acc = 0.0f;
    for (int token = 0; token < context_len; ++token) {
      const int idx = scratch_base + token;
      const float weight = score_scratch[idx] * inv_denom;
      if (!(weight > 0.0f)) continue;
      const int physical_slot = physical_scratch[idx];
      const int value_offset =
          (physical_slot * kNumKvHeads + kv_head) * kHeadDim + dim;
      acc += weight * laguna_bf16_bits_to_float(value_cache[value_offset]);
    }
    const int output_idx = q_offset + dim;
    out[output_idx] = acc;
    gated_out[output_idx] = laguna_float_to_bf16_bits(
        acc * laguna_softplus_f32(gate[q_head]));
  }
}

// Natural global-attention fused owner. The retained one-head specialization
// keeps all 48 local256 workgroups and the fixed-shape reducer's eight-wave
// token partition/softmax merge order, then uses four dimension waves.
// Dynamic LDS scales with the live scan rather than the 4096-slot capacity.
__device__ inline float laguna_permlanex16_f32(float value) {
  union FloatBits {
    float value;
    uint32_t bits;
  } data;
  data.value = value;
  data.bits = __builtin_amdgcn_permlanex16(
      data.bits, data.bits, 0x76543210U, 0xFEDCBA98U, false, false);
  return data.value;
}

template <int kDppControl>
__device__ inline float laguna_dpp_move_f32(float value) {
  union FloatBits {
    float value;
    int bits;
  } data;
  data.value = value;
  data.bits = __builtin_amdgcn_mov_dpp(
      data.bits, kDppControl, 0xF, 0xF, true);
  return data.value;
}

__device__ inline float laguna_reduce_wave32_dpp_exact_tree(float value) {
  value += laguna_permlanex16_f32(value);
  value += laguna_dpp_move_f32<0x108>(value);
  value += laguna_dpp_move_f32<0x104>(value);
  value += laguna_dpp_move_f32<0x102>(value);
  value += laguna_dpp_move_f32<0x101>(value);
  return value;
}

template <
    int kOwnedQueries,
    int kValueStage,
    bool kVectorValueStage,
    bool kDirectVectorStore = false,
    bool kAssumeSoftmaxDomain = false,
    bool kSoftmaxExp32 = false,
    bool kMixed32Owners = false,
    bool kProducerMax = false,
    bool kWmmaThreeTermQk = false,
    bool kDppQkReduction = false,
    bool kVectorProbabilityReplay = false,
    bool kPreNormalizeProbability = false,
    int kLocalThreads = 256,
    bool kMixed40Owners = false>
__global__ __launch_bounds__(kLocalThreads)
void laguna_global_attention_fused_exact_gated_fixedshape_kernel(
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
    int scan_slots,
    float scale) {
  constexpr int kCapacity = 4096;
  constexpr int kBlockSize = 256;
  constexpr int kNumKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kQueryGroup = 6;
  static_assert(
      !kWmmaThreeTermQk || kProducerMax,
      "three-term WMMA QK requires producer-owned maxima");
  static_assert(
      !kPreNormalizeProbability || kVectorProbabilityReplay,
      "probability pre-normalization requires the vector probability plane");
  static_assert(
      kLocalThreads == 256 || kLocalThreads == 512,
      "global fixed-shape attention supports local256 or local512");
  static_assert(
      !(kMixed32Owners && kMixed40Owners),
      "global mixed32 and mixed40 ownership are mutually exclusive");
  constexpr int kOwnersPerKv =
      kMixed40Owners ? 5 : (kMixed32Owners ? 4 : kQueryGroup / kOwnedQueries);
  constexpr int kNumWarps = kLocalThreads / 32;
  // local512 may widen independent QK and value transport, but the softmax
  // denominator must retain the local256 eight-wave summation tree.
  constexpr int kSoftmaxWarps = 8;
  extern __shared__ float dynamic_shared[];
  float* shared_scores = dynamic_shared;
  const int score_stride = kVectorProbabilityReplay
      ? ((scan_slots + 3) & ~3)
      : scan_slots;
  int32_t* shared_physical = reinterpret_cast<int32_t*>(
      shared_scores + kOwnedQueries * score_stride);
  const int value_prefix_bytes =
      (kOwnedQueries * static_cast<int>(sizeof(float)) +
       static_cast<int>(sizeof(int32_t))) * scan_slots +
      kOwnedQueries * (score_stride - scan_slots) *
          static_cast<int>(sizeof(float));
  const int value_padding_bytes = kVectorValueStage
      ? ((16 - (value_prefix_bytes & 15)) & 15)
      : 0;
  uint16_t* shared_value = reinterpret_cast<uint16_t*>(
      reinterpret_cast<uint8_t*>(shared_physical + scan_slots) +
      value_padding_bytes);
  __shared__ float warp_buf[kOwnedQueries][kNumWarps];

  const int owner = static_cast<int>(blockIdx.x);
  const int kv_head = owner / kOwnersPerKv;
  const int query_owner = owner % kOwnersPerKv;
  const int q_head_base = kv_head * kQueryGroup +
      (kMixed40Owners
           ? (query_owner == 0 ? 0 : query_owner + 1)
           : kMixed32Owners
           ? (query_owner < 2 ? query_owner * 2 : 4 + query_owner - 2)
           : query_owner * kOwnedQueries);
  const int owned_queries =
      (kMixed40Owners && query_owner >= 1) ||
              (kMixed32Owners && query_owner >= 2)
          ? 1
          : kOwnedQueries;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int wave = tid >> 5;
  int context_len = static_cast<int>(
      live_counts[0] < kCapacity ? live_counts[0] : kCapacity);
  context_len = context_len < scan_slots ? context_len : scan_slots;
  if (context_len <= 0 || row_positions[0] < 0) return;

  if constexpr (kWmmaThreeTermQk) {
    for (int token = tid; token < context_len;
         token += static_cast<int>(blockDim.x)) {
      const int64_t key_position = token_positions[token];
      const bool visible = evict_mask[token] == 0 &&
                           key_position >= 0 &&
                           key_position <= row_positions[0];
      shared_physical[token] = visible
          ? static_cast<int>(
                laguna_global_physical_slot(
                    base_offsets, token, kBlockSize))
          : -1;
    }
    __syncthreads();

    const int lane_lo = lane & 15;
    const int lane_hi = lane >> 4;
    const int query_local = lane_lo;
    const bool query_valid =
        wave < 4 && query_local < owned_queries;
    const int query_head = q_head_base + query_local;
    float wave_max = -3.4028234663852886e38f;
    for (int stage_base = 0; stage_base < context_len; stage_base += 64) {
      if (wave >= 4) continue;
      const int token_local = wave * 16 + lane_lo;
      const int token = stage_base + token_local;
      const int physical_slot =
          token < context_len ? shared_physical[token] : -1;
      laguna_bf16x16_t key_values = laguna_wmma_zero_bf16x16();
      laguna_float8_t score0 = laguna_wmma_zero_acc();
      laguna_float8_t score1 = laguna_wmma_zero_acc();
      laguna_float8_t score2 = laguna_wmma_zero_acc();
#pragma unroll
      for (int dim_base = 0; dim_base < kHeadDim; dim_base += 16) {
        if (physical_slot >= 0) {
          const uint16_t* key_ptr =
              key_cache +
              (physical_slot * kNumKvHeads + kv_head) * kHeadDim +
              dim_base;
          key_values =
              *reinterpret_cast<const laguna_bf16x16_t*>(key_ptr);
        } else {
          key_values = laguna_wmma_zero_bf16x16();
        }
        laguna_bf16x16_t query0 = laguna_wmma_zero_bf16x16();
        laguna_bf16x16_t query1 = laguna_wmma_zero_bf16x16();
        laguna_bf16x16_t query2 = laguna_wmma_zero_bf16x16();
        if (query_valid) {
          const float* query_ptr =
              query + query_head * kHeadDim + dim_base;
#pragma unroll
          for (int item = 0; item < 16; ++item) {
            uint16_t term0;
            uint16_t term1;
            uint16_t term2;
            laguna_three_term_bf16(
                query_ptr[item],
                term0,
                term1,
                term2);
            query0[item] = term0;
            query1[item] = term1;
            query2[item] = term2;
          }
        }
        score0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            key_values, query0, score0);
        score1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            key_values, query1, score1);
        score2 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            key_values, query2, score2);
      }
      if (query_valid) {
#pragma unroll
        for (int item = 0; item < 8; ++item) {
          const int output_token =
              stage_base + wave * 16 + 2 * item + lane_hi;
          if (output_token < context_len) {
            const float score =
                shared_physical[output_token] >= 0
                    ? (score0[item] + score1[item] + score2[item]) * scale
                    : -3.4028234663852886e38f;
            shared_scores[query_local * score_stride + output_token] = score;
            wave_max = fmaxf(wave_max, score);
          }
        }
      }
    }
    if (wave < 4 && query_valid) {
      wave_max = fmaxf(wave_max, __shfl_xor(wave_max, 16, 32));
      if (lane_hi == 0) {
        warp_buf[query_local][wave] = wave_max;
      }
    } else if (wave >= 4 && lane < owned_queries) {
      warp_buf[lane][wave] = -3.4028234663852886e38f;
    }
  } else {
    float query_values[kOwnedQueries][4] = {};
    float producer_max[kOwnedQueries];
#pragma unroll
    for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
      producer_max[q_local] = -3.4028234663852886e38f;
      if (q_local >= owned_queries) continue;
      const int q_offset = (q_head_base + q_local) * kHeadDim + lane * 4;
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        query_values[q_local][item] = query[q_offset + item];
      }
    }

    for (int token = wave; token < context_len; token += kNumWarps) {
      const int64_t key_position = token_positions[token];
      const bool visible = evict_mask[token] == 0 &&
                           key_position >= 0 &&
                           key_position <= row_positions[0];
      int physical_slot = -1;
      if (visible) {
        physical_slot = static_cast<int>(
            laguna_global_physical_slot(base_offsets, token, kBlockSize));
      }
      float key_values[4] = {};
      if (physical_slot >= 0) {
        const int key_offset =
            (physical_slot * kNumKvHeads + kv_head) * kHeadDim + lane * 4;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
          key_values[item] =
              laguna_bf16_bits_to_float(key_cache[key_offset + item]);
        }
      }
#pragma unroll
      for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
        if (q_local >= owned_queries) continue;
        float acc = 0.0f;
        if (physical_slot >= 0) {
#pragma unroll
          for (int item = 0; item < 4; ++item) {
            acc += query_values[q_local][item] * key_values[item];
          }
        }
        if constexpr (kDppQkReduction) {
          acc = laguna_reduce_wave32_dpp_exact_tree(acc);
        } else {
          for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down(acc, offset);
          }
        }
        if (lane == 0) {
          const float score =
              physical_slot >= 0 ? acc * scale : -3.4028234663852886e38f;
          shared_scores[q_local * score_stride + token] = score;
          if constexpr (kProducerMax) {
            producer_max[q_local] = fmaxf(producer_max[q_local], score);
          }
        }
      }
      if (lane == 0) shared_physical[token] = physical_slot;
    }
    if constexpr (kProducerMax) {
      if (lane == 0) {
#pragma unroll
        for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
          if (q_local < owned_queries) {
            warp_buf[q_local][wave] = producer_max[q_local];
          }
        }
      }
    }
  }
  __syncthreads();

  if constexpr (!kProducerMax) {
    float local_max[kOwnedQueries];
#pragma unroll
    for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
      local_max[q_local] = -3.4028234663852886e38f;
    }
    for (int token = wave; token < context_len; token += kNumWarps) {
      if (lane == 0) {
#pragma unroll
        for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
          if (q_local >= owned_queries) continue;
          local_max[q_local] = fmaxf(
              local_max[q_local],
              shared_scores[q_local * score_stride + token]);
        }
      }
    }
#pragma unroll
    for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
      if (q_local >= owned_queries) continue;
      for (int offset = 16; offset > 0; offset >>= 1) {
        local_max[q_local] =
            fmaxf(local_max[q_local], __shfl_down(local_max[q_local], offset));
      }
      if (lane == 0) warp_buf[q_local][wave] = local_max[q_local];
    }
    __syncthreads();
  }
  if (tid == 0) {
#pragma unroll
    for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
      if (q_local >= owned_queries) continue;
      float max_score = warp_buf[q_local][0];
#pragma unroll
      for (int item = 1; item < kNumWarps; ++item) {
        max_score = fmaxf(max_score, warp_buf[q_local][item]);
      }
      warp_buf[q_local][0] = max_score;
    }
  }
  __syncthreads();

  float local_sum[kOwnedQueries] = {};
  if constexpr (kSoftmaxExp32) {
    constexpr int kSoftmaxBatch = 32;
    if (wave < kSoftmaxWarps) {
      for (int token_base = wave; token_base < context_len;
           token_base += kSoftmaxWarps * kSoftmaxBatch) {
        const int token = token_base + lane * kSoftmaxWarps;
        const bool in_range = token < context_len;
        const bool visible =
            in_range && shared_physical[token] >= 0;
#pragma unroll
        for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
          if (q_local >= owned_queries) continue;
          float lane_weight = 0.0f;
          if (in_range) {
            const int score_idx = q_local * score_stride + token;
            const float shifted_score =
                shared_scores[score_idx] - warp_buf[q_local][0];
            if constexpr (kAssumeSoftmaxDomain) {
              __builtin_assume(shifted_score <= 0.0f);
            }
            lane_weight = visible ? expf(shifted_score) : 0.0f;
            shared_scores[score_idx] = lane_weight;
          }
#pragma unroll
          for (int item = 0; item < kSoftmaxBatch; ++item) {
            const float weight = __shfl(lane_weight, item, 32);
            const int ordered_token =
                token_base + item * kSoftmaxWarps;
            if (lane == 0 && ordered_token < context_len) {
              local_sum[q_local] += weight;
            }
          }
        }
      }
    }
  } else {
    if (wave < kSoftmaxWarps) {
      for (int token = wave; token < context_len;
           token += kSoftmaxWarps) {
        if (lane != 0) continue;
        const bool visible = shared_physical[token] >= 0;
#pragma unroll
        for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
          if (q_local >= owned_queries) continue;
          const int score_idx = q_local * score_stride + token;
          const float shifted_score =
              shared_scores[score_idx] - warp_buf[q_local][0];
          if constexpr (kAssumeSoftmaxDomain) {
            // Every visible score participates in the maximum, so score-max is
            // finite and non-positive. Preserve compiler `expf` exactly while
            // exposing the bounded softmax domain.
            __builtin_assume(shifted_score <= 0.0f);
          }
          const float weight = visible
              ? expf(shifted_score)
              : 0.0f;
          shared_scores[score_idx] = weight;
          local_sum[q_local] += weight;
        }
      }
    }
  }
#pragma unroll
  for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
    if (q_local >= owned_queries) continue;
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_sum[q_local] += __shfl_down(local_sum[q_local], offset);
    }
    if (lane == 0) warp_buf[q_local][wave] = local_sum[q_local];
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
      if (q_local >= owned_queries) continue;
      float total_sum = warp_buf[q_local][0];
#pragma unroll
      for (int item = 1; item < kNumWarps; ++item) {
        total_sum += warp_buf[q_local][item];
      }
      warp_buf[q_local][0] = 1.0f / fmaxf(total_sum, 1.0e-20f);
    }
  }
  __syncthreads();

  if constexpr (kPreNormalizeProbability) {
#pragma unroll
    for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
      if (q_local >= owned_queries) continue;
      const float inverse_sum = warp_buf[q_local][0];
      for (int token = tid; token < context_len;
           token += static_cast<int>(blockDim.x)) {
        const int score_idx = q_local * score_stride + token;
        shared_scores[score_idx] *= inverse_sum;
      }
    }
    __syncthreads();
  }

  const int q_local = wave >> 2;
  const bool active_output = q_local < owned_queries;
  const int dim = (wave & 3) * 32 + lane;
  const int q_head = q_head_base + q_local;
  float acc = 0.0f;
  if constexpr (kValueStage > 0) {
    for (int stage_base = 0; stage_base < context_len;
         stage_base += kValueStage) {
      const int stage_count =
          context_len - stage_base < kValueStage
              ? context_len - stage_base
              : kValueStage;
      if constexpr (kVectorValueStage) {
        constexpr int kValuesPerVector =
            static_cast<int>(sizeof(uint4) / sizeof(uint16_t));
        constexpr int kVectorsPerSlot = kHeadDim / kValuesPerVector;
        auto* shared_vectors = reinterpret_cast<uint4*>(shared_value);
        for (int vector_idx = tid;
             vector_idx < stage_count * kVectorsPerSlot;
             vector_idx += static_cast<int>(blockDim.x)) {
          const int stage_token = vector_idx / kVectorsPerSlot;
          const int vector_in_slot =
              vector_idx - stage_token * kVectorsPerSlot;
          const int physical_slot =
              shared_physical[stage_base + stage_token];
          if constexpr (kDirectVectorStore) {
            if (physical_slot >= 0) {
              const int value_offset =
                  (physical_slot * kNumKvHeads + kv_head) * kHeadDim +
                  vector_in_slot * kValuesPerVector;
              shared_vectors[vector_idx] = *reinterpret_cast<const uint4*>(
                  value_cache + value_offset);
            } else {
              auto* shared_words =
                  reinterpret_cast<uint32_t*>(shared_vectors + vector_idx);
#pragma unroll
              for (int word = 0; word < 4; ++word) {
                shared_words[word] = 0;
              }
            }
          } else {
            uint4 value = {};
            if (physical_slot >= 0) {
              const int value_offset =
                  (physical_slot * kNumKvHeads + kv_head) * kHeadDim +
                  vector_in_slot * kValuesPerVector;
              value = *reinterpret_cast<const uint4*>(
                  value_cache + value_offset);
            }
            shared_vectors[vector_idx] = value;
          }
        }
      } else {
        for (int idx = tid; idx < stage_count * kHeadDim;
             idx += static_cast<int>(blockDim.x)) {
          const int stage_token = idx / kHeadDim;
          const int value_dim = idx - stage_token * kHeadDim;
          const int physical_slot =
              shared_physical[stage_base + stage_token];
          uint16_t value = 0;
          if (physical_slot >= 0) {
            const int value_offset =
                (physical_slot * kNumKvHeads + kv_head) * kHeadDim +
                value_dim;
            value = value_cache[value_offset];
          }
          shared_value[idx] = value;
        }
      }
      __syncthreads();
      if (active_output) {
        if constexpr (kVectorProbabilityReplay) {
          const auto* probability_vectors =
              reinterpret_cast<const float4*>(
                  shared_scores + q_local * score_stride);
          int stage_token = 0;
          for (; stage_token + 3 < stage_count; stage_token += 4) {
            const float4 probabilities =
                probability_vectors[(stage_base + stage_token) >> 2];
            const float weights[4] = {
                kPreNormalizeProbability
                    ? probabilities.x
                    : probabilities.x * warp_buf[q_local][0],
                kPreNormalizeProbability
                    ? probabilities.y
                    : probabilities.y * warp_buf[q_local][0],
                kPreNormalizeProbability
                    ? probabilities.z
                    : probabilities.z * warp_buf[q_local][0],
                kPreNormalizeProbability
                    ? probabilities.w
                    : probabilities.w * warp_buf[q_local][0],
            };
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              const float weight = weights[item];
              if (!(weight > 0.0f)) continue;
              acc += weight * laguna_bf16_bits_to_float(
                  shared_value[
                      (stage_token + item) * kHeadDim + dim]);
            }
          }
          for (; stage_token < stage_count; ++stage_token) {
            const int token = stage_base + stage_token;
            const float weight =
                kPreNormalizeProbability
                    ? shared_scores[q_local * score_stride + token]
                    : shared_scores[q_local * score_stride + token] *
                          warp_buf[q_local][0];
            if (!(weight > 0.0f)) continue;
            acc += weight * laguna_bf16_bits_to_float(
                shared_value[stage_token * kHeadDim + dim]);
          }
        } else {
          for (int stage_token = 0; stage_token < stage_count; ++stage_token) {
            const int token = stage_base + stage_token;
            const float weight =
                shared_scores[q_local * score_stride + token] *
                warp_buf[q_local][0];
            if (!(weight > 0.0f)) continue;
            acc += weight * laguna_bf16_bits_to_float(
                shared_value[stage_token * kHeadDim + dim]);
          }
        }
      }
      __syncthreads();
    }
  } else {
    if (active_output) {
      for (int token = 0; token < context_len; ++token) {
        const float weight =
            shared_scores[q_local * score_stride + token] *
            warp_buf[q_local][0];
        if (!(weight > 0.0f)) continue;
        const int physical_slot = shared_physical[token];
        const int value_offset =
            (physical_slot * kNumKvHeads + kv_head) * kHeadDim + dim;
        acc += weight * laguna_bf16_bits_to_float(value_cache[value_offset]);
      }
    }
  }
  if (active_output) {
    const int output_idx = q_head * kHeadDim + dim;
    out[output_idx] = acc;
    gated_out[output_idx] = laguna_float_to_bf16_bits(
        acc * laguna_softplus_f32(gate[q_head]));
  }
}

// Vulkan-informed global decode screen. One local256 workgroup owns one
// (KV-head, K64 split): four waves form the six-query QK tile and eight waves
// form 16 output dimensions each. Three non-overlapping BF16 terms preserve
// substantially more of the F32 query/probability operands than a direct
// conversion. Split maxima/denominators and raw numerators are merged by a
// separate query-head-parallel kernel.
__global__ __launch_bounds__(256)
void laguna_global_attention_wmma_gqa6_k64_three_term_raw_numerator_partial_kernel(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* numerator_scratch,
    float* state_scratch,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int scan_slots,
    int split_count,
    float scale) {
  constexpr int kCapacity = 4096;
  constexpr int kBlockSize = 256;
  constexpr int kNumQueryHeads = 48;
  constexpr int kNumKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kQueryGroup = 6;
  constexpr int kSplitTokens = 64;
  __shared__ float shared_probability[16][kSplitTokens];
  __shared__ int32_t shared_physical[kSplitTokens];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;
  const int wave = tid >> 5;
  const int split = static_cast<int>(blockIdx.x) % split_count;
  const int kv_head = static_cast<int>(blockIdx.x) / split_count;
  const int split_base = split * kSplitTokens;
  int context_len = static_cast<int>(
      live_counts[0] < kCapacity ? live_counts[0] : kCapacity);
  context_len = context_len < scan_slots ? context_len : scan_slots;
  const int64_t query_position = row_positions[0];

  if (tid < kSplitTokens) {
    const int token = split_base + tid;
    int physical_slot = -1;
    if (token < context_len && query_position >= 0) {
      const int64_t key_position = token_positions[token];
      if (evict_mask[token] == 0 &&
          key_position >= 0 &&
          key_position <= query_position) {
        physical_slot = static_cast<int>(
            laguna_global_physical_slot(
                base_offsets, token, kBlockSize));
      }
    }
    shared_physical[tid] = physical_slot;
  }
  __syncthreads();

  if (wave < 4) {
    const int token_local = wave * 16 + lane_lo;
    const int physical_slot = shared_physical[token_local];
    laguna_bf16x16_t key_values = laguna_wmma_zero_bf16x16();
    if (physical_slot >= 0) {
      const uint16_t* key_ptr =
          key_cache +
          (physical_slot * kNumKvHeads + kv_head) * kHeadDim;
      key_values =
          *reinterpret_cast<const laguna_bf16x16_t*>(key_ptr);
    }

    laguna_float8_t score0 = laguna_wmma_zero_acc();
    laguna_float8_t score1 = laguna_wmma_zero_acc();
    laguna_float8_t score2 = laguna_wmma_zero_acc();
    const int query_local = lane_lo;
    const bool query_valid = query_local < kQueryGroup;
    const int query_head = kv_head * kQueryGroup + query_local;
#pragma unroll
    for (int dim_base = 0; dim_base < kHeadDim; dim_base += 16) {
      if (dim_base != 0 && physical_slot >= 0) {
        const uint16_t* key_ptr =
            key_cache +
            (physical_slot * kNumKvHeads + kv_head) * kHeadDim +
            dim_base;
        key_values =
            *reinterpret_cast<const laguna_bf16x16_t*>(key_ptr);
      }
      laguna_bf16x16_t query0 = laguna_wmma_zero_bf16x16();
      laguna_bf16x16_t query1 = laguna_wmma_zero_bf16x16();
      laguna_bf16x16_t query2 = laguna_wmma_zero_bf16x16();
      if (query_valid) {
        const float* query_ptr =
            query + query_head * kHeadDim + dim_base;
#pragma unroll
        for (int item = 0; item < 16; ++item) {
          uint16_t term0;
          uint16_t term1;
          uint16_t term2;
          laguna_three_term_bf16(
              query_ptr[item],
              term0,
              term1,
              term2);
          query0[item] = term0;
          query1[item] = term1;
          query2[item] = term2;
        }
      }
      score0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
          key_values, query0, score0);
      score1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
          key_values, query1, score1);
      score2 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
          key_values, query2, score2);
    }
    if (query_valid) {
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        const int output_token =
            wave * 16 + 2 * item + lane_hi;
        const float score =
            (score0[item] + score1[item] + score2[item]) * scale;
        shared_probability[query_local][output_token] =
            shared_physical[output_token] >= 0
            ? score
            : -3.4028234663852886e38f;
      }
    }
  }
  __syncthreads();

  if (tid < kQueryGroup) {
    float max_score = -3.4028234663852886e38f;
#pragma unroll
    for (int token_local = 0;
         token_local < kSplitTokens;
         ++token_local) {
      if (shared_physical[token_local] >= 0) {
        max_score = fmaxf(
            max_score,
            shared_probability[tid][token_local]);
      }
    }
    float denominator = 0.0f;
#pragma unroll
    for (int token_local = 0;
         token_local < kSplitTokens;
         ++token_local) {
      float probability = 0.0f;
      if (shared_physical[token_local] >= 0) {
        probability = expf(
            shared_probability[tid][token_local] - max_score);
        denominator += probability;
      }
      shared_probability[tid][token_local] = probability;
    }
    const int query_head = kv_head * kQueryGroup + tid;
    const int state_idx =
        (split * kNumQueryHeads + query_head) * 2;
    state_scratch[state_idx] = max_score;
    state_scratch[state_idx + 1] = denominator;
  }
  __syncthreads();

  const int dim_base = wave * 16;
  laguna_float8_t numerator0 = laguna_wmma_zero_acc();
  laguna_float8_t numerator1 = laguna_wmma_zero_acc();
  laguna_float8_t numerator2 = laguna_wmma_zero_acc();
#pragma unroll
  for (int token_base = 0;
       token_base < kSplitTokens;
       token_base += 16) {
    const int query_local = lane_lo;
    laguna_bf16x16_t probability0 = laguna_wmma_zero_bf16x16();
    laguna_bf16x16_t probability1 = laguna_wmma_zero_bf16x16();
    laguna_bf16x16_t probability2 = laguna_wmma_zero_bf16x16();
    if (query_local < kQueryGroup) {
#pragma unroll
      for (int item = 0; item < 16; ++item) {
        uint16_t term0;
        uint16_t term1;
        uint16_t term2;
        laguna_three_term_bf16(
            shared_probability[query_local][token_base + item],
            term0,
            term1,
            term2);
        probability0[item] = term0;
        probability1[item] = term1;
        probability2[item] = term2;
      }
    }

    laguna_bf16x16_t value_values = laguna_wmma_zero_bf16x16();
    const int dim = dim_base + lane_lo;
#pragma unroll
    for (int item = 0; item < 16; ++item) {
      const int physical_slot =
          shared_physical[token_base + item];
      if (physical_slot >= 0) {
        value_values[item] = value_cache[
            (physical_slot * kNumKvHeads + kv_head) * kHeadDim +
            dim];
      }
    }
    numerator0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
        probability0, value_values, numerator0);
    numerator1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
        probability1, value_values, numerator1);
    numerator2 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
        probability2, value_values, numerator2);
  }
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    const int query_local = 2 * item + lane_hi;
    if (query_local < kQueryGroup) {
      const int query_head = kv_head * kQueryGroup + query_local;
      const int dim = dim_base + lane_lo;
      numerator_scratch[
          (split * kNumQueryHeads + query_head) * kHeadDim + dim] =
          numerator0[item] + numerator1[item] + numerator2[item];
    }
  }
}

__global__ __launch_bounds__(128)
void laguna_global_attention_wmma_gqa6_k64_three_term_raw_numerator_merge_kernel(
    const float* numerator_scratch,
    const float* state_scratch,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    int split_count) {
  constexpr int kNumQueryHeads = 48;
  constexpr int kHeadDim = 128;
  const int query_head = static_cast<int>(blockIdx.x);
  const int dim = static_cast<int>(threadIdx.x);
  if (query_head >= kNumQueryHeads || dim >= kHeadDim) return;

  float global_max = -3.4028234663852886e38f;
  for (int split = 0; split < split_count; ++split) {
    const float denominator =
        state_scratch[(split * kNumQueryHeads + query_head) * 2 + 1];
    if (denominator > 0.0f) {
      global_max = fmaxf(
          global_max,
          state_scratch[(split * kNumQueryHeads + query_head) * 2]);
    }
  }
  double denominator = 0.0;
  double numerator = 0.0;
  for (int split = 0; split < split_count; ++split) {
    const int state_idx =
        (split * kNumQueryHeads + query_head) * 2;
    const float split_denominator = state_scratch[state_idx + 1];
    if (!(split_denominator > 0.0f)) continue;
    const double factor = exp(
        static_cast<double>(state_scratch[state_idx] - global_max));
    denominator +=
        static_cast<double>(split_denominator) * factor;
    numerator +=
        static_cast<double>(
            numerator_scratch[
                (split * kNumQueryHeads + query_head) * kHeadDim + dim]) *
        factor;
  }
  const float context_value =
      denominator > 0.0
      ? static_cast<float>(numerator / denominator)
      : 0.0f;
  const int output_idx = query_head * kHeadDim + dim;
  out[output_idx] = context_value;
  gated_out[output_idx] = laguna_float_to_bf16_bits(
      context_value * laguna_softplus_f32(gate[query_head]));
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

__device__ __forceinline__ float laguna_wave32_sum_128_exact_dpp(
    float product_0,
    float product_32,
    float product_64,
    float product_96) {
  // Preserve the baseline 128-thread tree exactly, replacing only the
  // wave32 lane transport used by its +16,+8,+4,+2,+1 suffix.
  float sum = laguna_ordered_add_f32(
      laguna_ordered_add_f32(product_0, product_64),
      laguna_ordered_add_f32(product_32, product_96));
  sum = laguna_ordered_add_f32(sum, laguna_permlanex16_f32(sum));
  sum = laguna_ordered_add_f32(sum, laguna_dpp_move_f32<0x108>(sum));
  sum = laguna_ordered_add_f32(sum, laguna_dpp_move_f32<0x104>(sum));
  sum = laguna_ordered_add_f32(sum, laguna_dpp_move_f32<0x102>(sum));
  sum = laguna_ordered_add_f32(sum, laguna_dpp_move_f32<0x101>(sum));
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

// Exact three-query GQA score owner. One wave loads a shared SWA key once,
// then emits three independent query-head scores with the retained ordered
// products and reconstructed 128-thread reduction tree. The 24 owners keep
// the token axis grid-parallel, so unlike grouped value reduction this does
// not collapse the 512-token launch below the device's CU count.
template <int kOwnedQueries>
__global__ __launch_bounds__(32)
void laguna_swa_attention_split_exact_gqa3_scores_kernel(
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
  constexpr int kQueryGroup = 9;
  constexpr int kOwnersPerKv = kQueryGroup / kOwnedQueries;
  const int64_t owner = static_cast<int64_t>(blockIdx.x);
  const int64_t logical_slot = static_cast<int64_t>(blockIdx.y);
  const int64_t kv_head = owner / kOwnersPerKv;
  const int64_t query_owner = owner % kOwnersPerKv;
  if (kv_head >= num_kv_heads || logical_slot >= scan_slots) return;

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

  const int lane = static_cast<int>(threadIdx.x);
  const int64_t q_head_base =
      kv_head * kQueryGroup + query_owner * kOwnedQueries;
  float key_values[4] = {};
  if (physical_slot >= 0) {
    const int64_t key_base =
        (physical_slot * num_kv_heads + kv_head) * head_dim;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      key_values[part] =
          laguna_bf16_bits_to_float(key_cache[key_base + lane + part * 32]);
    }
  }
#pragma unroll
  for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
    const int64_t q_head = q_head_base + q_local;
    const int64_t q_offset = q_head * head_dim;
    float products[4] = {};
    if (physical_slot >= 0) {
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        products[part] = laguna_ordered_mul_f32(
            query[q_offset + lane + part * 32], key_values[part]);
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

// Tile16 sibling of the exact GQA3 score owner. Eight waves keep sixteen
// logical slots in flight while three query heads reuse each loaded key.
template <int kOwnedQueries>
__global__ __launch_bounds__(256)
void laguna_swa_attention_split_tile16_exact_gqa3_scores_kernel(
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
  constexpr int kQueryGroup = 9;
  constexpr int kOwnersPerKv = kQueryGroup / kOwnedQueries;
  const int64_t owner = static_cast<int64_t>(blockIdx.x);
  const int64_t kv_head = owner / kOwnersPerKv;
  const int64_t query_owner = owner % kOwnersPerKv;
  const int64_t tile_base =
      static_cast<int64_t>(blockIdx.y) * LAGUNA_SPLIT_SCORE_TILE;
  if (kv_head >= num_kv_heads || tile_base >= scan_slots) return;

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int64_t live_count = live_counts[0];
  const int64_t context_len =
      live_count < capacity ? live_count : capacity;
  const int64_t query_position = row_positions[0];
  const int64_t q_head_base =
      kv_head * kQueryGroup + query_owner * kOwnedQueries;
  float q_values[kOwnedQueries][4];
#pragma unroll
  for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
    const int64_t q_offset = (q_head_base + q_local) * head_dim;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      q_values[q_local][part] =
          query[q_offset + lane + part * 32];
    }
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

    float key_values[4] = {};
    if (physical_slot >= 0) {
      const int64_t key_base =
          (physical_slot * num_kv_heads + kv_head) * head_dim;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        key_values[part] = laguna_bf16_bits_to_float(
            key_cache[key_base + lane + part * 32]);
      }
    }
#pragma unroll
    for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
      float products[4] = {};
      if (physical_slot >= 0) {
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          products[part] = laguna_ordered_mul_f32(
              q_values[q_local][part], key_values[part]);
        }
      }
      const float dot = laguna_wave32_sum_128_exact(
          products[0], products[1], products[2], products[3]);
      if (lane == 0 && logical_slot < scan_slots) {
        const int64_t q_head = q_head_base + q_local;
        const int64_t idx = q_head * scratch_stride + logical_slot;
        score_scratch[idx] =
            physical_slot >= 0 ? dot : -3.4028234663852886e38f;
        physical_scratch[idx] = static_cast<int32_t>(physical_slot);
      }
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

// Saturated Laguna SWA specialization. The runtime admits this body only for
// the natural 72Q/8KV/D128, capacity/window/scan 512 shape. It preserves the
// retained one-query-head block, four independent waves, scalar slot order,
// and per-dimension FMA sequence while removing dynamic shape arithmetic and
// loop bounds from the hot reducer.
__global__ __launch_bounds__(128)
void laguna_swa_attention_split_exact_gated_wave_local_fixed512_reduce_kernel(
    const uint16_t* value_cache,
    const float* score_scratch,
    const int32_t* physical_scratch,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    const int64_t* row_positions,
    float scale) {
  constexpr int kScanSlots = 512;
  constexpr int kNumKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kQueryGroup = 9;
  const int q_head = static_cast<int>(blockIdx.x);
  const int dim = static_cast<int>(threadIdx.x);
  const int lane = dim & 31;
  const int q_offset = q_head * kHeadDim;
  if (row_positions[0] < 0) {
    out[q_offset + dim] = 0.0f;
    gated_out[q_offset + dim] = 0;
    return;
  }

  const int scratch_base = q_head * kScanSlots;
  float wave_max = -3.4028234663852886e38f;
  if (lane == 0) {
#pragma unroll 4
    for (int slot_base = 0; slot_base < kScanSlots; slot_base += 4) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int idx = scratch_base + slot_base + item;
        if (physical_scratch[idx] >= 0) {
          wave_max = fmaxf(wave_max, score_scratch[idx] * scale);
        }
      }
    }
  }
  wave_max = __shfl(wave_max, 0, 32);

  const int kv_head = q_head / kQueryGroup;
  float wave_denom = 0.0f;
  float output_acc = 0.0f;
#pragma unroll 4
  for (int slot_base = 0; slot_base < kScanSlots; slot_base += 4) {
    float weights[4] = {};
    if (lane == 0) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int idx = scratch_base + slot_base + item;
        const bool visible = physical_scratch[idx] >= 0;
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
      const int physical_slot =
          physical_scratch[scratch_base + slot_base + item];
      if (physical_slot >= 0) {
        const int value_offset =
            (physical_slot * kNumKvHeads + kv_head) * kHeadDim + dim;
        output_acc = fmaf(
            weight,
            laguna_bf16_bits_to_float(value_cache[value_offset]),
            output_acc);
      }
    }
  }
  wave_denom = __shfl(wave_denom, 0, 32);
  const float context_value = output_acc / fmaxf(wave_denom, 1.0e-20f);
  const int output_idx = q_offset + dim;
  out[output_idx] = context_value;
  gated_out[output_idx] = laguna_float_to_bf16_bits(
      context_value * laguna_softplus_f32(gate[q_head]));
}

// GQA2 sibling of the fused exact fixed512 path. Forty ordinary local256
// workgroups own pairs of adjacent query heads (the fifth owner in each KV
// group carries the ninth head alone). Eight waves reuse each K vector across
// the owned exact QK trees, then retain one wave per 32 output dimensions.
__global__ __launch_bounds__(256)
void laguna_swa_attention_fused_exact_gated_gqa2_fixed512_kernel(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    const int32_t* base_offsets,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    float scale) {
  constexpr int kScanSlots = 512;
  constexpr int kNumKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kQueryGroup = 9;
  constexpr int kOwnedQueries = 2;
  constexpr int kOwnersPerKv = 5;
  constexpr int kScoreWaves = 8;
  __shared__ float shared_scores[kOwnedQueries][kScanSlots];
  __shared__ int32_t shared_physical[kScanSlots];

  const int owner = static_cast<int>(blockIdx.x);
  const int kv_head = owner / kOwnersPerKv;
  const int query_owner = owner % kOwnersPerKv;
  const int q_head_base =
      kv_head * kQueryGroup + query_owner * kOwnedQueries;
  const int owned_queries =
      q_head_base + kOwnedQueries <= (kv_head + 1) * kQueryGroup
          ? kOwnedQueries
          : 1;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int64_t query_position = row_positions[0];

  float query_values[kOwnedQueries][4] = {};
#pragma unroll
  for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
    if (q_local < owned_queries) {
      const int q_offset = (q_head_base + q_local) * kHeadDim;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        query_values[q_local][part] =
            query[q_offset + lane + part * 32];
      }
    }
  }
  for (int logical_slot = wave; logical_slot < kScanSlots;
       logical_slot += kScoreWaves) {
    int physical_slot = -1;
    if (lane == 0) {
      const int64_t key_position = token_positions[logical_slot];
      const bool visible = query_position >= 0 &&
                           evict_mask[logical_slot] == 0 &&
                           key_position >= 0 &&
                           key_position <= query_position &&
                           key_position > query_position - kScanSlots;
      if (visible) {
        physical_slot = base_offsets[logical_slot];
        if (physical_slot < 0 || physical_slot >= kScanSlots) {
          physical_slot = -1;
        }
      }
    }
    physical_slot = __shfl(physical_slot, 0, 32);

    float key_values[4] = {};
    if (physical_slot >= 0) {
      const int key_base =
          (physical_slot * kNumKvHeads + kv_head) * kHeadDim;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        key_values[part] = laguna_bf16_bits_to_float(
            key_cache[key_base + lane + part * 32]);
      }
    }
#pragma unroll
    for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
      if (q_local < owned_queries) {
        float products[4] = {};
        if (physical_slot >= 0) {
#pragma unroll
          for (int part = 0; part < 4; ++part) {
            products[part] = laguna_ordered_mul_f32(
                query_values[q_local][part], key_values[part]);
          }
        }
        const float dot = laguna_wave32_sum_128_exact(
            products[0], products[1], products[2], products[3]);
        if (lane == 0) {
          shared_scores[q_local][logical_slot] =
              physical_slot >= 0 ? dot : -3.4028234663852886e38f;
        }
      }
    }
    if (lane == 0) {
      shared_physical[logical_slot] = physical_slot;
    }
  }
  __syncthreads();

  const int q_local = wave / 4;
  if (q_local >= owned_queries) return;
  const int dim = (wave & 3) * 32 + lane;
  const int q_head = q_head_base + q_local;
  const int q_offset = q_head * kHeadDim;
  if (query_position < 0) {
    out[q_offset + dim] = 0.0f;
    gated_out[q_offset + dim] = 0;
    return;
  }

  float wave_max = -3.4028234663852886e38f;
  if (lane == 0) {
#pragma unroll 4
    for (int slot_base = 0; slot_base < kScanSlots; slot_base += 4) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int idx = slot_base + item;
        if (shared_physical[idx] >= 0) {
          wave_max = fmaxf(
              wave_max, shared_scores[q_local][idx] * scale);
        }
      }
    }
  }
  wave_max = __shfl(wave_max, 0, 32);

  float wave_denom = 0.0f;
  float output_acc = 0.0f;
#pragma unroll 4
  for (int slot_base = 0; slot_base < kScanSlots; slot_base += 4) {
    float weights[4] = {};
    if (lane == 0) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int idx = slot_base + item;
        const bool visible = shared_physical[idx] >= 0;
        const float weight = visible
            ? expf(shared_scores[q_local][idx] * scale - wave_max)
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
      const int physical_slot = shared_physical[slot_base + item];
      if (physical_slot >= 0) {
        const int value_offset =
            (physical_slot * kNumKvHeads + kv_head) * kHeadDim + dim;
        output_acc = fmaf(
            weight,
            laguna_bf16_bits_to_float(value_cache[value_offset]),
            output_acc);
      }
    }
  }
  wave_denom = __shfl(wave_denom, 0, 32);
  const float context_value = output_acc / fmaxf(wave_denom, 1.0e-20f);
  const int output_idx = q_offset + dim;
  out[output_idx] = context_value;
  gated_out[output_idx] = laguna_float_to_bf16_bits(
      context_value * laguna_softplus_f32(gate[q_head]));
}

// Exact fused-GQA3 family. The retained base maps 24 ordinary local384
// workgroups to three queries each. The gfx1151 mixed specialization maps
// four owners per KV head as 2+2+2+3 (32 blocks). The mixed40 diagnostic maps
// five owners per KV head as 2+2+2+2+1 (40 blocks). Idle waves still
// participate in every staged-V barrier. Active heads preserve the same QK,
// softmax, PV, gate, and store association in all forms.
template <bool kDenseRing>
__device__ __forceinline__ int laguna_swa_fixed512_physical_slot(
    const int32_t* shared_physical,
    int logical_slot) {
  if constexpr (kDenseRing) {
    return logical_slot;
  }
  return shared_physical[logical_slot];
}

template <
    int kValueStage,
    bool kVectorValueStage,
    bool kDirectVectorStore = false,
    bool kAssumeSoftmaxDomain = false,
    bool kMixed32Owners = false,
    bool kSoftmaxExp4 = false,
    bool kSoftmaxExp8 = false,
    bool kSoftmaxExp16 = false,
    bool kSoftmaxExp32 = false,
    bool kProducerMax = false,
    bool kProducerGate = false,
    bool kStoreContext = true,
    bool kDppQkReduction = false,
    bool kStageProbabilityCache = false,
    bool kIdleProbabilityProducer = false,
    bool kVectorDenomReplay = false,
    bool kVectorProbabilityReplay = false,
    bool kIdleVectorDenomReplay = false,
    bool kMixed40Owners = false,
    bool kTailProbabilityProducer = false,
    bool kDenomPrefetch4 = false,
    bool kProducerValueTail = false,
    int kLocalThreads = 384,
    bool kDualTailProbabilityProducer = false,
    bool kOutputShardedProbability = false,
    bool kDenseRing = false>
__global__ __launch_bounds__(kLocalThreads)
void laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const float* gate,
    uint16_t* gated_out,
    const int32_t* base_offsets,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    float scale) {
  constexpr int kScanSlots = 512;
  constexpr int kNumKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kQueryGroup = 9;
  constexpr int kOwnedQueries = 3;
  constexpr int kOwnersPerKv =
      kMixed40Owners ? 5 : (kMixed32Owners ? 4 : 3);
  constexpr int kScoreWaves = kLocalThreads / 32;
  static_assert(
      kLocalThreads == 384 || kLocalThreads == 512,
      "mixed-owner SWA supports local384/local512 specializations");
  static_assert(
      !kStageProbabilityCache ||
          ((kValueStage == 64 || kValueStage == 128) &&
           kSoftmaxExp32 && kProducerMax),
      "stage probability cache requires V-stage64/128, exp32, and producer "
      "max");
  static_assert(
      !kVectorDenomReplay || kStageProbabilityCache,
      "vector denominator replay requires the stage probability cache");
  static_assert(
      !kVectorProbabilityReplay ||
          (kStageProbabilityCache && kVectorDenomReplay),
      "vector probability replay requires vector denominator stage cache");
  static_assert(
      !kIdleVectorDenomReplay ||
          (kStageProbabilityCache && kVectorDenomReplay &&
           !kIdleProbabilityProducer),
      "idle vector denominator replay requires the active-producer vector "
      "denominator stage cache");
  static_assert(
      !(kMixed32Owners && kMixed40Owners),
      "mixed32 and mixed40 ownership are mutually exclusive");
  static_assert(
      !kTailProbabilityProducer ||
          (kMixed40Owners && kStageProbabilityCache &&
           kIdleVectorDenomReplay && !kIdleProbabilityProducer),
      "tail probability producers require mixed40 stage cache with separate "
      "idle-wave denominator replay");
  static_assert(
      !kDenomPrefetch4 ||
          (kStageProbabilityCache && kVectorDenomReplay),
      "four-vector denominator prefetch requires vector denominator replay");
  static_assert(
      !kProducerValueTail ||
          ((kValueStage == 64 || kValueStage == 128) && kVectorValueStage &&
           kDirectVectorStore && kMixed40Owners &&
           kTailProbabilityProducer),
      "producer value tail requires the direct mixed40 tail-producer "
      "V-stage64/128 path");
  static_assert(
      !kDualTailProbabilityProducer ||
          (kValueStage == 128 && kProducerValueTail &&
           kTailProbabilityProducer && kLocalThreads == 512),
      "dual tail probability producers require the local512 direct mixed40 "
      "V-stage128 producer-value-tail path");
  static_assert(
      !kOutputShardedProbability ||
          (kValueStage == 128 && kProducerValueTail &&
           kTailProbabilityProducer && kLocalThreads == 512 &&
           !kDualTailProbabilityProducer),
      "output-sharded probability requires the local512 direct mixed40 "
      "V-stage128 producer-value-tail path");
  static_assert(
      !kDenseRing || (kOutputShardedProbability && kDppQkReduction),
      "dense-ring specialization requires the current output-sharded "
      "DPP-QK path");
  __shared__ float shared_scores[kOwnedQueries][kScanSlots];
  __shared__ int32_t shared_physical[kDenseRing ? 1 : kScanSlots];
  __shared__ float shared_wave_max[kOwnedQueries][kScoreWaves];
  __shared__ float shared_gate[kOwnedQueries];
  alignas(16) __shared__ float shared_probability[
      kOwnedQueries][kStageProbabilityCache ? kValueStage : 1];
  __shared__ float shared_denom[
      kStageProbabilityCache ? kOwnedQueries : 1];
  alignas(16) __shared__ uint16_t shared_value[
      (kValueStage > 0 ? kValueStage : 1) * kHeadDim];

  const int owner = static_cast<int>(blockIdx.x);
  const int kv_head = owner / kOwnersPerKv;
  const int query_owner = owner % kOwnersPerKv;
  const int q_head_base = kv_head * kQueryGroup +
      (kMixed40Owners
           ? query_owner * 2
           : kMixed32Owners
           ? (query_owner < 3 ? query_owner * 2 : 6)
           : query_owner * kOwnedQueries);
  const int owned_queries =
      kMixed40Owners
          ? (query_owner < 4 ? 2 : 1)
          : (kMixed32Owners && query_owner < 3 ? 2 : kOwnedQueries);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int64_t query_position = row_positions[0];

  if constexpr (kProducerGate) {
    if (tid < owned_queries) {
      shared_gate[tid] =
          laguna_softplus_f32(gate[q_head_base + tid]);
    }
  }
  float producer_max[kOwnedQueries];
#pragma unroll
  for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
    producer_max[q_local] = -3.4028234663852886e38f;
  }
  float query_values[kOwnedQueries][4] = {};
#pragma unroll
  for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
    if (q_local >= owned_queries) continue;
    const int q_offset =
        (q_head_base + q_local) * kHeadDim;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      query_values[q_local][part] =
          query[q_offset + lane + part * 32];
    }
  }
  for (int logical_slot = wave; logical_slot < kScanSlots;
       logical_slot += kScoreWaves) {
    int physical_slot = logical_slot;
    if constexpr (!kDenseRing) {
      physical_slot = -1;
      if (lane == 0) {
        const int64_t key_position = token_positions[logical_slot];
        const bool visible = query_position >= 0 &&
                             evict_mask[logical_slot] == 0 &&
                             key_position >= 0 &&
                             key_position <= query_position &&
                             key_position > query_position - kScanSlots;
        if (visible) {
          physical_slot = base_offsets[logical_slot];
          if (physical_slot < 0 || physical_slot >= kScanSlots) {
            physical_slot = -1;
          }
        }
      }
      physical_slot = __shfl(physical_slot, 0, 32);
    }

    float key_values[4] = {};
    if (physical_slot >= 0) {
      const int key_base =
          (physical_slot * kNumKvHeads + kv_head) * kHeadDim;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        key_values[part] = laguna_bf16_bits_to_float(
            key_cache[key_base + lane + part * 32]);
      }
    }
#pragma unroll
    for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
      if (q_local >= owned_queries) continue;
      float products[4] = {};
      if (physical_slot >= 0) {
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          products[part] = laguna_ordered_mul_f32(
              query_values[q_local][part], key_values[part]);
        }
      }
      const float dot = kDppQkReduction
          ? laguna_wave32_sum_128_exact_dpp(
                products[0], products[1], products[2], products[3])
          : laguna_wave32_sum_128_exact(
                products[0], products[1], products[2], products[3]);
      if (lane == 0) {
        shared_scores[q_local][logical_slot] =
            physical_slot >= 0 ? dot : -3.4028234663852886e38f;
        if constexpr (kProducerMax) {
          if (physical_slot >= 0) {
            producer_max[q_local] = fmaxf(
                producer_max[q_local], dot * scale);
          }
        }
      }
    }
    if constexpr (!kDenseRing) {
      if (lane == 0) {
        shared_physical[logical_slot] = physical_slot;
      }
    }
  }
  if constexpr (kProducerMax) {
    if (lane == 0) {
#pragma unroll
      for (int q_local = 0; q_local < kOwnedQueries; ++q_local) {
        if (q_local < owned_queries) {
          shared_wave_max[q_local][wave] = producer_max[q_local];
        }
      }
    }
  }
  __syncthreads();

  const int q_local = wave / 4;
  const bool active_output = q_local < owned_queries;
  const int dim = (wave & 3) * 32 + lane;
  const int q_head = q_head_base + q_local;
  const int q_offset = q_head * kHeadDim;
  if (query_position < 0) {
    if (active_output) {
      out[q_offset + dim] = 0.0f;
      gated_out[q_offset + dim] = 0;
    }
    return;
  }

  float wave_max = -3.4028234663852886e38f;
  if constexpr (kProducerMax) {
    if (active_output && lane == 0) {
#pragma unroll
      for (int source_wave = 0; source_wave < kScoreWaves; ++source_wave) {
        wave_max = fmaxf(
            wave_max, shared_wave_max[q_local][source_wave]);
      }
    }
  } else if (active_output && lane == 0) {
#pragma unroll 4
    for (int slot_base = 0; slot_base < kScanSlots; slot_base += 4) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int idx = slot_base + item;
        if (laguna_swa_fixed512_physical_slot<kDenseRing>(
                shared_physical, idx) >= 0) {
          wave_max = fmaxf(
              wave_max, shared_scores[q_local][idx] * scale);
        }
      }
    }
  }
  wave_max = __shfl(wave_max, 0, 32);
  const bool idle_probability_producer =
      kStageProbabilityCache && kIdleProbabilityProducer &&
      owned_queries < kOwnedQueries &&
      wave >= 8 && wave < 8 + owned_queries;
  const int tail_probability_producer_count =
      (kDualTailProbabilityProducer ? 2 : 1) * owned_queries;
  const bool tail_probability_producer =
      kTailProbabilityProducer && !kOutputShardedProbability &&
      wave >= kScoreWaves - tail_probability_producer_count;
  const int tail_probability_producer_index =
      tail_probability_producer
          ? wave - (kScoreWaves - tail_probability_producer_count)
          : -1;
  const int probability_q_local =
      tail_probability_producer
          ? (kDualTailProbabilityProducer
                 ? tail_probability_producer_index / 2
                 : tail_probability_producer_index)
          : (idle_probability_producer ? wave - 8 : q_local);
  const int tail_probability_producer_part =
      kOutputShardedProbability && active_output
          ? wave & 3
          : kDualTailProbabilityProducer && tail_probability_producer
          ? tail_probability_producer_index & 1
          : 0;
  const bool idle_vector_denom_replay =
      kIdleVectorDenomReplay && owned_queries < kOwnedQueries &&
      wave >= 8 && wave < 8 + owned_queries;
  const int denominator_q_local =
      idle_vector_denom_replay ? wave - 8 : probability_q_local;
  float probability_max = wave_max;
  if (idle_probability_producer || tail_probability_producer) {
    probability_max = -3.4028234663852886e38f;
    if (lane == 0) {
#pragma unroll
      for (int source_wave = 0; source_wave < kScoreWaves; ++source_wave) {
        probability_max = fmaxf(
            probability_max,
            shared_wave_max[probability_q_local][source_wave]);
      }
    }
    probability_max = __shfl(probability_max, 0, 32);
  }

  float wave_denom = 0.0f;
  float output_acc = 0.0f;
  if constexpr (kValueStage > 0) {
    for (int stage_base = 0; stage_base < kScanSlots;
         stage_base += kValueStage) {
      const bool stage_probability_producer =
          kStageProbabilityCache &&
          (kOutputShardedProbability
               ? active_output
               : kTailProbabilityProducer
               ? tail_probability_producer
               : kIdleProbabilityProducer && owned_queries < kOwnedQueries
               ? idle_probability_producer
               : active_output && (wave & 3) == 0);
      const bool stage_denom_producer =
          kIdleVectorDenomReplay && owned_queries < kOwnedQueries
              ? idle_vector_denom_replay
              : stage_probability_producer;
      if constexpr (kStageProbabilityCache) {
        if (stage_probability_producer) {
#pragma unroll
          for (int local_base =
                   (kDualTailProbabilityProducer ||
                    kOutputShardedProbability)
                       ? tail_probability_producer_part * 32
                       : 0;
               local_base < kValueStage;
               local_base +=
                   kOutputShardedProbability
                       ? kValueStage
                       : kDualTailProbabilityProducer ? 64 : 32) {
            const int idx = stage_base + local_base + lane;
            const bool visible =
                laguna_swa_fixed512_physical_slot<kDenseRing>(
                    shared_physical, idx) >= 0;
            const float shifted_score =
                shared_scores[probability_q_local][idx] * scale -
                probability_max;
            if constexpr (kAssumeSoftmaxDomain) {
              __builtin_assume(shifted_score <= 0.0f);
            }
            const float lane_weight =
                visible ? expf(shifted_score) : 0.0f;
            shared_probability[probability_q_local][local_base + lane] =
                lane_weight;
            if constexpr (!kVectorDenomReplay) {
#pragma unroll
              for (int item = 0; item < 32; ++item) {
                const float weight = __shfl(lane_weight, item, 32);
                if (lane == 0 &&
                    laguna_swa_fixed512_physical_slot<kDenseRing>(
                        shared_physical,
                        stage_base + local_base + item) >= 0) {
                  wave_denom =
                      laguna_ordered_add_f32(wave_denom, weight);
                }
              }
            }
          }
          if constexpr (!kVectorDenomReplay) {
            if (lane == 0) {
              shared_denom[probability_q_local] = wave_denom;
            }
          }
        }
      }
      if constexpr (kVectorValueStage) {
        constexpr int kValuesPerVector =
            static_cast<int>(sizeof(uint4) / sizeof(uint16_t));
        constexpr int kVectorsPerSlot = kHeadDim / kValuesPerVector;
        auto* shared_vectors = reinterpret_cast<uint4*>(shared_value);
        int value_loader_tid = tid;
        int value_loader_threads = static_cast<int>(blockDim.x);
        int value_loader_limit = kValueStage * kVectorsPerSlot;
        if constexpr (
            kStageProbabilityCache && !kOutputShardedProbability) {
          if (stage_probability_producer) {
            if constexpr (kProducerValueTail) {
              value_loader_tid =
                  kValueStage * kVectorsPerSlot -
                  tail_probability_producer_count * 32 +
                  tail_probability_producer_index * 32 + lane;
              value_loader_threads = kValueStage * kVectorsPerSlot;
            } else {
              value_loader_tid = -1;
            }
          } else {
            int producers_before = 0;
#pragma unroll
            for (int query_idx = 0;
                 query_idx < kOwnedQueries;
                 ++query_idx) {
              const int producer_wave =
                  kTailProbabilityProducer
                      ? kScoreWaves - owned_queries + query_idx
                      : kIdleProbabilityProducer &&
                          owned_queries < kOwnedQueries
                      ? 8 + query_idx
                      : query_idx * 4;
              if (query_idx < owned_queries && producer_wave < wave) {
                ++producers_before;
              }
            }
            value_loader_tid =
                (wave - producers_before) * 32 + lane;
            value_loader_threads =
                (kScoreWaves - tail_probability_producer_count) * 32;
            if constexpr (kProducerValueTail) {
              value_loader_limit -=
                  tail_probability_producer_count * 32;
            }
          }
        }
        for (int vector_idx = value_loader_tid;
             vector_idx >= 0 &&
             vector_idx < value_loader_limit;
             vector_idx += value_loader_threads) {
          const int stage_slot = vector_idx / kVectorsPerSlot;
          const int vector_in_slot =
              vector_idx - stage_slot * kVectorsPerSlot;
          const int physical_slot =
              laguna_swa_fixed512_physical_slot<kDenseRing>(
                  shared_physical, stage_base + stage_slot);
          if constexpr (kDirectVectorStore) {
            if (physical_slot >= 0) {
              const int value_offset =
                  (physical_slot * kNumKvHeads + kv_head) * kHeadDim +
                  vector_in_slot * kValuesPerVector;
              shared_vectors[vector_idx] = *reinterpret_cast<const uint4*>(
                  value_cache + value_offset);
            } else {
              auto* shared_words =
                  reinterpret_cast<uint32_t*>(shared_vectors + vector_idx);
#pragma unroll
              for (int word = 0; word < 4; ++word) {
                shared_words[word] = 0;
              }
            }
          } else {
            uint4 value = {};
            if (physical_slot >= 0) {
              const int value_offset =
                  (physical_slot * kNumKvHeads + kv_head) * kHeadDim +
                  vector_in_slot * kValuesPerVector;
              value = *reinterpret_cast<const uint4*>(
                  value_cache + value_offset);
            }
            shared_vectors[vector_idx] = value;
          }
        }
      } else {
        for (int idx = tid; idx < kValueStage * kHeadDim;
             idx += static_cast<int>(blockDim.x)) {
          const int stage_slot = idx / kHeadDim;
          const int value_dim = idx - stage_slot * kHeadDim;
          const int physical_slot =
              laguna_swa_fixed512_physical_slot<kDenseRing>(
                  shared_physical, stage_base + stage_slot);
          uint16_t value = 0;
          if (physical_slot >= 0) {
            const int value_offset =
                (physical_slot * kNumKvHeads + kv_head) * kHeadDim +
                value_dim;
            value = value_cache[value_offset];
          }
          shared_value[idx] = value;
        }
      }
      __syncthreads();
      if constexpr (kStageProbabilityCache && kVectorDenomReplay) {
        if (stage_denom_producer && lane == 0) {
          const auto* probability_vectors =
              reinterpret_cast<const float4*>(
                  shared_probability[denominator_q_local]);
          if constexpr (kDenomPrefetch4) {
#pragma unroll
            for (int item = 0; item < kValueStage / 4; item += 4) {
              const float4 weights0 = probability_vectors[item + 0];
              const float4 weights1 = probability_vectors[item + 1];
              const float4 weights2 = probability_vectors[item + 2];
              const float4 weights3 = probability_vectors[item + 3];
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights0.x);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights0.y);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights0.z);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights0.w);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights1.x);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights1.y);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights1.z);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights1.w);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights2.x);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights2.y);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights2.z);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights2.w);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights3.x);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights3.y);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights3.z);
              wave_denom =
                  laguna_ordered_add_f32(wave_denom, weights3.w);
            }
          } else {
#pragma unroll
            for (int item = 0; item < kValueStage / 4; ++item) {
              const float4 weights = probability_vectors[item];
              wave_denom = laguna_ordered_add_f32(wave_denom, weights.x);
              wave_denom = laguna_ordered_add_f32(wave_denom, weights.y);
              wave_denom = laguna_ordered_add_f32(wave_denom, weights.z);
              wave_denom = laguna_ordered_add_f32(wave_denom, weights.w);
            }
          }
          shared_denom[denominator_q_local] = wave_denom;
        }
      }
      if constexpr (kStageProbabilityCache) {
        if (active_output) {
          if constexpr (kVectorProbabilityReplay) {
            const auto* probability_vectors =
                reinterpret_cast<const float4*>(
                    shared_probability[q_local]);
#pragma unroll
            for (int item = 0; item < kValueStage / 4; ++item) {
              const float4 weights = probability_vectors[item];
              const int value_base = item * 4;
              output_acc = fmaf(
                  weights.x,
                  laguna_bf16_bits_to_float(
                      shared_value[
                          (value_base + 0) * kHeadDim + dim]),
                  output_acc);
              output_acc = fmaf(
                  weights.y,
                  laguna_bf16_bits_to_float(
                      shared_value[
                          (value_base + 1) * kHeadDim + dim]),
                  output_acc);
              output_acc = fmaf(
                  weights.z,
                  laguna_bf16_bits_to_float(
                      shared_value[
                          (value_base + 2) * kHeadDim + dim]),
                  output_acc);
              output_acc = fmaf(
                  weights.w,
                  laguna_bf16_bits_to_float(
                      shared_value[
                          (value_base + 3) * kHeadDim + dim]),
                  output_acc);
            }
          } else {
#pragma unroll
            for (int local_base = 0; local_base < kValueStage;
                 local_base += 32) {
#pragma unroll
              for (int item = 0; item < 32; ++item) {
                const int value_idx =
                    (local_base + item) * kHeadDim + dim;
                output_acc = fmaf(
                    shared_probability[q_local][local_base + item],
                    laguna_bf16_bits_to_float(shared_value[value_idx]),
                    output_acc);
              }
            }
          }
          if constexpr (!kVectorDenomReplay) {
            wave_denom = shared_denom[q_local];
          }
        }
      } else {
        constexpr int kSoftmaxBatch =
            kSoftmaxExp32 ? 32 : (kSoftmaxExp16 ? 16 : (kSoftmaxExp8 ? 8 : 4));
#pragma unroll
        for (int local_base = 0; local_base < kValueStage;
             local_base += kSoftmaxBatch) {
          float weights[kSoftmaxBatch] = {};
          if constexpr (
              kSoftmaxExp4 || kSoftmaxExp8 || kSoftmaxExp16 || kSoftmaxExp32) {
            float lane_weight = 0.0f;
            if (active_output && lane < kSoftmaxBatch) {
              const int idx = stage_base + local_base + lane;
              const bool visible =
                  laguna_swa_fixed512_physical_slot<kDenseRing>(
                      shared_physical, idx) >= 0;
              const float shifted_score =
                  shared_scores[q_local][idx] * scale - wave_max;
              if constexpr (kAssumeSoftmaxDomain) {
                __builtin_assume(shifted_score <= 0.0f);
              }
              lane_weight = visible ? expf(shifted_score) : 0.0f;
            }
#pragma unroll
            for (int item = 0; item < kSoftmaxBatch; ++item) {
              weights[item] = __shfl(lane_weight, item, 32);
              if (active_output && lane == 0 &&
                  laguna_swa_fixed512_physical_slot<kDenseRing>(
                      shared_physical,
                      stage_base + local_base + item) >= 0) {
                wave_denom =
                    laguna_ordered_add_f32(wave_denom, weights[item]);
              }
            }
          } else if (active_output && lane == 0) {
#pragma unroll
            for (int item = 0; item < kSoftmaxBatch; ++item) {
              const int idx = stage_base + local_base + item;
              const bool visible =
                  laguna_swa_fixed512_physical_slot<kDenseRing>(
                      shared_physical, idx) >= 0;
              const float shifted_score =
                  shared_scores[q_local][idx] * scale - wave_max;
              if constexpr (kAssumeSoftmaxDomain) {
                // The visible score's own maximum participates in `wave_max`,
                // so score-max is finite and non-positive. Keep the exact
                // compiler `expf` expression while exposing that domain.
                __builtin_assume(shifted_score <= 0.0f);
              }
              const float weight =
                  visible ? expf(shifted_score) : 0.0f;
              weights[item] = weight;
              if (visible) {
                wave_denom = laguna_ordered_add_f32(wave_denom, weight);
              }
            }
          }
          if (active_output) {
#pragma unroll
            for (int item = 0; item < kSoftmaxBatch; ++item) {
              const float weight =
                  (kSoftmaxExp4 || kSoftmaxExp8 || kSoftmaxExp16 ||
                   kSoftmaxExp32)
                  ? weights[item]
                  : __shfl(weights[item], 0, 32);
              const int value_idx =
                  (local_base + item) * kHeadDim + dim;
              output_acc = fmaf(
                  weight,
                  laguna_bf16_bits_to_float(shared_value[value_idx]),
                  output_acc);
            }
          }
        }
      }
      __syncthreads();
    }
    if constexpr (kStageProbabilityCache && kVectorDenomReplay) {
      if (active_output) {
        wave_denom = shared_denom[q_local];
      }
    }
  } else {
#pragma unroll 4
    for (int slot_base = 0; slot_base < kScanSlots; slot_base += 4) {
      float weights[4] = {};
      if (active_output && lane == 0) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
          const int idx = slot_base + item;
          const bool visible =
              laguna_swa_fixed512_physical_slot<kDenseRing>(
                  shared_physical, idx) >= 0;
          const float weight = visible
              ? expf(shared_scores[q_local][idx] * scale - wave_max)
              : 0.0f;
          weights[item] = weight;
          if (visible) {
            wave_denom = laguna_ordered_add_f32(wave_denom, weight);
          }
        }
      }
      if (active_output) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
          const float weight = __shfl(weights[item], 0, 32);
          const int physical_slot =
              laguna_swa_fixed512_physical_slot<kDenseRing>(
                  shared_physical, slot_base + item);
          if (physical_slot >= 0) {
            const int value_offset =
                (physical_slot * kNumKvHeads + kv_head) * kHeadDim + dim;
            output_acc = fmaf(
                weight,
                laguna_bf16_bits_to_float(value_cache[value_offset]),
                output_acc);
          }
        }
      }
    }
  }
  wave_denom = __shfl(wave_denom, 0, 32);
  if (active_output) {
    const float context_value =
        output_acc / fmaxf(wave_denom, 1.0e-20f);
    const int output_idx = q_offset + dim;
    if constexpr (kStoreContext) {
      out[output_idx] = context_value;
    }
    gated_out[output_idx] = laguna_float_to_bf16_bits(
        context_value *
        (kProducerGate
             ? shared_gate[q_local]
             : laguna_softplus_f32(gate[q_head])));
  }
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

__global__ __launch_bounds__(32) void laguna_swa_attention_prefill_qrow2_exact_bf16_kernel(
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
  constexpr int kQueryRows = 2;
  const int64_t query_row_base =
      static_cast<int64_t>(blockIdx.y) * kQueryRows;
  const int64_t q_head = static_cast<int64_t>(blockIdx.x);
  const int64_t lane = static_cast<int64_t>(threadIdx.x);
  if (query_row_base >= rows || q_head >= num_q_heads || lane >= 32) return;

  const int64_t start_position = row_positions[0];
  const int64_t prior_live_count = live_counts[0];
  const int64_t q_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / q_group;
  const int64_t start_slot = start_position % capacity;
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
    if (current_row < 0) current_row += capacity;
    const bool has_current = current_row >= 0 && current_row < rows;
    const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    const bool cache_valid = physical_slot >= 0 && physical_slot < capacity;
    const int64_t current_base =
        (current_row * num_kv_heads + kv_head) * head_dim;
    const int64_t cache_base =
        (physical_slot * num_kv_heads + kv_head) * head_dim;
    float current_keys[4];
    float cached_keys[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int64_t dim = lane + part * 32;
      current_keys[part] = has_current
          ? laguna_bf16_bits_to_float(
                laguna_float_to_bf16_bits(current_key[current_base + dim]))
          : 0.0f;
      cached_keys[part] = cache_valid
          ? laguna_bf16_bits_to_float(key_cache[cache_base + dim])
          : 0.0f;
    }
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      const int64_t query_row = query_row_base + row_index;
      if (query_row >= rows) continue;
      const int64_t query_position = start_position + query_row;
      const bool from_current = has_current && current_row <= query_row;
      const int64_t key_position = from_current
          ? start_position + current_row
          : token_positions[logical_slot];
      const bool source_valid = from_current || cache_valid;
      const bool visible = (from_current || evict_mask[logical_slot] == 0) &&
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
    if (current_row < 0) current_row += capacity;
    const bool has_current = current_row >= 0 && current_row < rows;
    const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    const bool cache_valid = physical_slot >= 0 && physical_slot < capacity;
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
      current_keys[part] = has_current
          ? laguna_bf16_bits_to_float(
                laguna_float_to_bf16_bits(current_key[current_base + dim]))
          : 0.0f;
      cached_keys[part] = cache_valid
          ? laguna_bf16_bits_to_float(key_cache[cache_base + dim])
          : 0.0f;
      current_values[part] = has_current
          ? laguna_bf16_bits_to_float(
                laguna_float_to_bf16_bits(current_value[current_base + dim]))
          : 0.0f;
      cached_values[part] = cache_valid
          ? laguna_bf16_bits_to_float(value_cache[cache_base + dim])
          : 0.0f;
    }
#pragma unroll
    for (int row_index = 0; row_index < kQueryRows; ++row_index) {
      const int64_t query_row = query_row_base + row_index;
      if (query_row >= rows) continue;
      const int64_t query_position = start_position + query_row;
      const bool from_current = has_current && current_row <= query_row;
      const int64_t key_position = from_current
          ? start_position + current_row
          : token_positions[logical_slot];
      const bool source_valid = from_current || cache_valid;
      const bool visible = (from_current || evict_mask[logical_slot] == 0) &&
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

extern "C" int
hipengine_laguna_global_attention_decode_split_exact_gated_fixedshape_bf16_spans(
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
      capacity != 4096 || block_size != 256 || block_table_len != 16 ||
      scratch_stride != 4096 || scan_slots <= 0 || scan_slots > 4096 ||
      num_q_heads != 48 || num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_global_attention_split_exact_scores_kernel,
      dim3(48, static_cast<unsigned int>(scan_slots)),
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
      laguna_global_attention_split_exact_gated_fixedshape_reduce_kernel,
      dim3(48),
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
      static_cast<int>(scan_slots));
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_fused_exact_gated_gqa1_fixedshape_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4096 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const size_t shared_bytes = static_cast<size_t>(scan_slots) *
      (sizeof(float) + sizeof(int32_t));
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          1,
          0,
          false>),
      dim3(48),
      dim3(256),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_fixedshape_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4000 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kOwnedQueries = 2;
  constexpr int kValueStage = 64;
  constexpr int kHeadDim = 128;
  const size_t shared_bytes =
      static_cast<size_t>(kOwnedQueries * scan_slots) * sizeof(float) +
      static_cast<size_t>(scan_slots) * sizeof(int32_t) +
      static_cast<size_t>(kValueStage * kHeadDim) * sizeof(uint16_t);
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          kOwnedQueries,
          kValueStage,
          false>),
      dim3(24),
      dim3(256),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_fixedshape_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4000 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kOwnedQueries = 2;
  constexpr int kValueStage = 64;
  constexpr int kHeadDim = 128;
  const size_t value_prefix_bytes =
      static_cast<size_t>(kOwnedQueries * scan_slots) * sizeof(float) +
      static_cast<size_t>(scan_slots) * sizeof(int32_t);
  const size_t value_padding_bytes =
      (16 - (value_prefix_bytes & 15)) & 15;
  const size_t shared_bytes =
      value_prefix_bytes + value_padding_bytes +
      static_cast<size_t>(kValueStage * kHeadDim) * sizeof(uint16_t);
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          kOwnedQueries,
          kValueStage,
          true>),
      dim3(24),
      dim3(256),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_fixedshape_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4000 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kOwnedQueries = 2;
  constexpr int kValueStage = 64;
  constexpr int kHeadDim = 128;
  const size_t value_prefix_bytes =
      static_cast<size_t>(kOwnedQueries * scan_slots) * sizeof(float) +
      static_cast<size_t>(scan_slots) * sizeof(int32_t);
  const size_t value_padding_bytes =
      (16 - (value_prefix_bytes & 15)) & 15;
  const size_t shared_bytes =
      value_prefix_bytes + value_padding_bytes +
      static_cast<size_t>(kValueStage * kHeadDim) * sizeof(uint16_t);
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          kOwnedQueries,
          kValueStage,
          true,
          true>),
      dim3(24),
      dim3(256),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4000 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kOwnedQueries = 2;
  constexpr int kValueStage = 64;
  constexpr int kHeadDim = 128;
  const size_t value_prefix_bytes =
      static_cast<size_t>(kOwnedQueries * scan_slots) * sizeof(float) +
      static_cast<size_t>(scan_slots) * sizeof(int32_t);
  const size_t value_padding_bytes =
      (16 - (value_prefix_bytes & 15)) & 15;
  const size_t shared_bytes =
      value_prefix_bytes + value_padding_bytes +
      static_cast<size_t>(kValueStage * kHeadDim) * sizeof(uint16_t);
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          kOwnedQueries,
          kValueStage,
          true,
          true,
          true>),
      dim3(24),
      dim3(256),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_fused_exact_gated_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4000 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kOwnedQueries = 2;
  constexpr int kValueStage = 64;
  constexpr int kHeadDim = 128;
  const size_t value_prefix_bytes =
      static_cast<size_t>(kOwnedQueries * scan_slots) * sizeof(float) +
      static_cast<size_t>(scan_slots) * sizeof(int32_t);
  const size_t value_padding_bytes =
      (16 - (value_prefix_bytes & 15)) & 15;
  const size_t shared_bytes =
      value_prefix_bytes + value_padding_bytes +
      static_cast<size_t>(kValueStage * kHeadDim) * sizeof(uint16_t);
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          kOwnedQueries,
          kValueStage,
          true,
          true,
          true,
          true>),
      dim3(24),
      dim3(256),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4000 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kOwnedQueries = 2;
  constexpr int kValueStage = 64;
  constexpr int kHeadDim = 128;
  const size_t value_prefix_bytes =
      static_cast<size_t>(kOwnedQueries * scan_slots) * sizeof(float) +
      static_cast<size_t>(scan_slots) * sizeof(int32_t);
  const size_t value_padding_bytes =
      (16 - (value_prefix_bytes & 15)) & 15;
  const size_t shared_bytes =
      value_prefix_bytes + value_padding_bytes +
      static_cast<size_t>(kValueStage * kHeadDim) * sizeof(uint16_t);
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          kOwnedQueries,
          kValueStage,
          true,
          true,
          true,
          true,
          true>),
      dim3(32),
      dim3(256),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4000 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kOwnedQueries = 2;
  constexpr int kValueStage = 64;
  constexpr int kHeadDim = 128;
  const size_t value_prefix_bytes =
      static_cast<size_t>(kOwnedQueries * scan_slots) * sizeof(float) +
      static_cast<size_t>(scan_slots) * sizeof(int32_t);
  const size_t value_padding_bytes =
      (16 - (value_prefix_bytes & 15)) & 15;
  const size_t shared_bytes =
      value_prefix_bytes + value_padding_bytes +
      static_cast<size_t>(kValueStage * kHeadDim) * sizeof(uint16_t);
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          kOwnedQueries,
          kValueStage,
          true,
          true,
          true,
          true,
          true,
          true>),
      dim3(32),
      dim3(256),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4000 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kOwnedQueries = 2;
  constexpr int kValueStage = 64;
  constexpr int kHeadDim = 128;
  const size_t value_prefix_bytes =
      static_cast<size_t>(kOwnedQueries * scan_slots) * sizeof(float) +
      static_cast<size_t>(scan_slots) * sizeof(int32_t);
  const size_t value_padding_bytes =
      (16 - (value_prefix_bytes & 15)) & 15;
  const size_t shared_bytes =
      value_prefix_bytes + value_padding_bytes +
      static_cast<size_t>(kValueStage * kHeadDim) * sizeof(uint16_t);
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          kOwnedQueries,
          kValueStage,
          true,
          true,
          true,
          true,
          true,
          true,
          false,
          true>),
      dim3(32),
      dim3(256),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4000 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kOwnedQueries = 2;
  constexpr int kValueStage = 64;
  constexpr int kHeadDim = 128;
  const int64_t score_stride = (scan_slots + 3) & ~int64_t{3};
  const size_t value_prefix_bytes =
      static_cast<size_t>(kOwnedQueries * score_stride) * sizeof(float) +
      static_cast<size_t>(scan_slots) * sizeof(int32_t);
  const size_t value_padding_bytes =
      (16 - (value_prefix_bytes & 15)) & 15;
  const size_t shared_bytes =
      value_prefix_bytes + value_padding_bytes +
      static_cast<size_t>(kValueStage * kHeadDim) * sizeof(uint16_t);
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          kOwnedQueries,
          kValueStage,
          true,
          true,
          true,
          true,
          true,
          true,
          false,
          true,
          true>),
      dim3(32),
      dim3(256),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4000 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kOwnedQueries = 2;
  constexpr int kValueStage = 64;
  constexpr int kHeadDim = 128;
  const int64_t score_stride = (scan_slots + 3) & ~int64_t{3};
  const size_t value_prefix_bytes =
      static_cast<size_t>(kOwnedQueries * score_stride) * sizeof(float) +
      static_cast<size_t>(scan_slots) * sizeof(int32_t);
  const size_t value_padding_bytes =
      (16 - (value_prefix_bytes & 15)) & 15;
  const size_t shared_bytes =
      value_prefix_bytes + value_padding_bytes +
      static_cast<size_t>(kValueStage * kHeadDim) * sizeof(uint16_t);
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          kOwnedQueries,
          kValueStage,
          true,
          true,
          true,
          true,
          true,
          true,
          false,
          true,
          true,
          true>),
      dim3(32),
      dim3(256),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_fused_exact_gated_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4000 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kOwnedQueries = 2;
  constexpr int kValueStage = 64;
  constexpr int kHeadDim = 128;
  constexpr int kLocalThreads = 512;
  const int64_t score_stride = (scan_slots + 3) & ~int64_t{3};
  const size_t value_prefix_bytes =
      static_cast<size_t>(kOwnedQueries * score_stride) * sizeof(float) +
      static_cast<size_t>(scan_slots) * sizeof(int32_t);
  const size_t value_padding_bytes =
      (16 - (value_prefix_bytes & 15)) & 15;
  const size_t shared_bytes =
      value_prefix_bytes + value_padding_bytes +
      static_cast<size_t>(kValueStage * kHeadDim) * sizeof(uint16_t);
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          kOwnedQueries,
          kValueStage,
          true,
          true,
          true,
          true,
          true,
          true,
          false,
          true,
          true,
          true,
          kLocalThreads>),
      dim3(32),
      dim3(kLocalThreads),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4000 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kOwnedQueries = 2;
  constexpr int kValueStage = 64;
  constexpr int kHeadDim = 128;
  constexpr int kLocalThreads = 512;
  const int64_t score_stride = (scan_slots + 3) & ~int64_t{3};
  const size_t value_prefix_bytes =
      static_cast<size_t>(kOwnedQueries * score_stride) * sizeof(float) +
      static_cast<size_t>(scan_slots) * sizeof(int32_t);
  const size_t value_padding_bytes =
      (16 - (value_prefix_bytes & 15)) & 15;
  const size_t shared_bytes =
      value_prefix_bytes + value_padding_bytes +
      static_cast<size_t>(kValueStage * kHeadDim) * sizeof(uint16_t);
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          kOwnedQueries,
          kValueStage,
          true,
          true,
          true,
          true,
          false,
          true,
          false,
          true,
          true,
          true,
          kLocalThreads,
          true>),
      dim3(40),
      dim3(kLocalThreads),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_wmma_qk_three_term_mixed32_exp32_producer_max_exact_pv_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 4000 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kOwnedQueries = 2;
  constexpr int kValueStage = 64;
  constexpr int kHeadDim = 128;
  const size_t value_prefix_bytes =
      static_cast<size_t>(kOwnedQueries * scan_slots) * sizeof(float) +
      static_cast<size_t>(scan_slots) * sizeof(int32_t);
  const size_t value_padding_bytes =
      (16 - (value_prefix_bytes & 15)) & 15;
  const size_t shared_bytes =
      value_prefix_bytes + value_padding_bytes +
      static_cast<size_t>(kValueStage * kHeadDim) * sizeof(uint16_t);
  hipLaunchKernelGGL(
      (laguna_global_attention_fused_exact_gated_fixedshape_kernel<
          kOwnedQueries,
          kValueStage,
          true,
          true,
          true,
          true,
          true,
          true,
          true>),
      dim3(32),
      dim3(256),
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
      static_cast<int>(scan_slots),
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_global_attention_decode_wmma_gqa6_k64_three_term_raw_numerator_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 4096 || block_size != 256 ||
      block_table_len != 16 || scratch_stride != 4096 ||
      scan_slots <= 0 || scan_slots > 1024 || num_q_heads != 48 ||
      num_kv_heads != 8 || head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int split_count =
      (static_cast<int>(scan_slots) + 63) / 64;
  hipLaunchKernelGGL(
      laguna_global_attention_wmma_gqa6_k64_three_term_raw_numerator_partial_kernel,
      dim3(static_cast<unsigned int>(8 * split_count)),
      dim3(256),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      score_scratch,
      reinterpret_cast<float*>(physical_scratch),
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      static_cast<int>(scan_slots),
      split_count,
      scale);
  int error = static_cast<int>(hipGetLastError());
  if (error != static_cast<int>(hipSuccess)) return error;
  hipLaunchKernelGGL(
      laguna_global_attention_wmma_gqa6_k64_three_term_raw_numerator_merge_kernel,
      dim3(48),
      dim3(128),
      0,
      stream,
      score_scratch,
      reinterpret_cast<const float*>(physical_scratch),
      out,
      gate,
      gated_out,
      split_count);
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

template <
    bool TILE16_SCORES,
    bool DIM2,
    bool GQA = false,
    bool FIXED512 = false>
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
  if constexpr (GQA) {
    if (num_q_heads != 72 || num_kv_heads != 8) {
      return static_cast<int>(hipErrorInvalidValue);
    }
  }
  if constexpr (FIXED512) {
    if (capacity != 512 || sliding_window != 512 ||
        scratch_stride != 512 || scan_slots != 512 ||
        num_q_heads != 72 || num_kv_heads != 8 || head_dim != 128) {
      return static_cast<int>(hipErrorInvalidValue);
    }
  }
  if constexpr (TILE16_SCORES) {
    const int64_t score_tiles =
        (scan_slots + LAGUNA_SPLIT_SCORE_TILE - 1) / LAGUNA_SPLIT_SCORE_TILE;
    if constexpr (GQA) {
      hipLaunchKernelGGL(
          (laguna_swa_attention_split_tile16_exact_gqa3_scores_kernel<3>),
          dim3(static_cast<unsigned int>(num_kv_heads * 3),
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
    }
  } else {
    if constexpr (GQA) {
      hipLaunchKernelGGL(
          (laguna_swa_attention_split_exact_gqa3_scores_kernel<3>),
          dim3(static_cast<unsigned int>(num_kv_heads * 3),
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
  }
  int error = static_cast<int>(hipGetLastError());
  if (error != static_cast<int>(hipSuccess)) return error;
  if constexpr (FIXED512) {
    hipLaunchKernelGGL(
        laguna_swa_attention_split_exact_gated_wave_local_fixed512_reduce_kernel,
        dim3(72),
        dim3(128),
        0,
        stream,
        value_cache,
        score_scratch,
        physical_scratch,
        out,
        gate,
        gated_out,
        row_positions,
        scale);
  } else if constexpr (DIM2) {
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
hipengine_laguna_swa_attention_decode_split_exact_gated_gqa3_scores_bf16_spans(
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
  return laguna_launch_swa_attention_split_exact_gated_wave_local<
      false, false, true>(
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
hipengine_laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_bf16_spans(
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
  return laguna_launch_swa_attention_split_exact_gated_wave_local<
      true, false, true>(
      query, key_cache, value_cache, out, gate, gated_out, score_scratch,
      physical_scratch, base_offsets, live_counts, token_positions, evict_mask,
      row_positions, capacity, sliding_window, scratch_stride, scan_slots,
      num_q_heads, num_kv_heads, head_dim, scale, stream);
}

extern "C" int
hipengine_laguna_swa_attention_decode_split_tile16_exact_gated_gqa3_scores_fixed512_bf16_spans(
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
  return laguna_launch_swa_attention_split_exact_gated_wave_local<
      true, false, true, true>(
      query, key_cache, value_cache, out, gate, gated_out, score_scratch,
      physical_scratch, base_offsets, live_counts, token_positions, evict_mask,
      row_positions, capacity, sliding_window, scratch_stride, scan_slots,
      num_q_heads, num_kv_heads, head_dim, scale, stream);
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_gqa2_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      laguna_swa_attention_fused_exact_gated_gqa2_fixed512_kernel,
      dim3(40),
      dim3(256),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_gqa3_local384_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          0,
          false>),
      dim3(24),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          false>),
      dim3(24),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true>),
      dim3(24),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true>),
      dim3(24),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true>),
      dim3(24),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          false,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          false,
          false,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          false,
          false,
          false,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          false,
          false,
          false,
          true,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          false,
          false,
          false,
          true,
          true,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_producer_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true,
          false,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true,
          false,
          true,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true,
          false,
          true,
          true,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          false,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true,
          false,
          true,
          true,
          true,
          true>),
      dim3(40),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          false,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true,
          false,
          true,
          true,
          true,
          true,
          true>),
      dim3(40),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          false,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true,
          false,
          true,
          true,
          true,
          true,
          true,
          false,
          false,
          512>),
      dim3(40),
      dim3(512),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          false,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true,
          false,
          true,
          true,
          true,
          true,
          true,
          false,
          true,
          512>),
      dim3(40),
      dim3(512),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          128,
          true,
          true,
          true,
          false,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true,
          false,
          true,
          true,
          true,
          true,
          true,
          false,
          true,
          512>),
      dim3(40),
      dim3(512),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_dual_tail_producer_value_tail_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          128,
          true,
          true,
          true,
          false,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true,
          false,
          true,
          true,
          true,
          true,
          true,
          false,
          true,
          512,
          true>),
      dim3(40),
      dim3(512),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          128,
          true,
          true,
          true,
          false,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true,
          false,
          true,
          true,
          true,
          true,
          true,
          false,
          true,
          512,
          false,
          true>),
      dim3(40),
      dim3(512),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          128,
          true,
          true,
          true,
          false,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          true,
          true,
          false,
          true,
          true,
          true,
          true,
          true,
          false,
          true,
          512,
          false,
          true>),
      dim3(40),
      dim3(512),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_local512_exp32_producer_max_gate_stage_pcache_output_sharded_probability_dpp_qk_dense_ring_allwave_value_idle_vec4_denom_probability_vstage128_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          128,
          true,
          true,
          true,
          false,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          true,
          true,
          false,
          true,
          true,
          true,
          true,
          true,
          false,
          true,
          512,
          false,
          true,
          true>),
      dim3(40),
      dim3(512),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_value_tail_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          false,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true,
          false,
          true,
          true,
          true,
          true,
          true,
          false,
          true>),
      dim3(40),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_tail_producer_denom_prefetch4_idle_vec4_probability_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          false,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          false,
          true,
          false,
          true,
          true,
          true,
          true,
          true,
          true>),
      dim3(40),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          true,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_dpp_qk_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          false,
          false,
          false,
          true,
          true,
          true,
          true,
          true>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_gated_only_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
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
      base_offsets == nullptr || live_counts == nullptr ||
      token_positions == nullptr || evict_mask == nullptr ||
      row_positions == nullptr || capacity != 512 ||
      sliding_window != 512 || scratch_stride != 512 ||
      scan_slots != 512 || num_q_heads != 72 || num_kv_heads != 8 ||
      head_dim != 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (laguna_swa_attention_fused_exact_gated_gqa3_local384_fixed512_kernel<
          64,
          true,
          true,
          true,
          true,
          false,
          false,
          false,
          true,
          true,
          true,
          false>),
      dim3(32),
      dim3(384),
      0,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      gate,
      gated_out,
      base_offsets,
      token_positions,
      evict_mask,
      row_positions,
      scale);
  return static_cast<int>(hipGetLastError());
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
      laguna_swa_attention_prefill_qrow2_exact_bf16_kernel,
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
