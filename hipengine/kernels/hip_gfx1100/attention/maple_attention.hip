// Net-new Maple QK/RoPE/KV/attention kernels from the public DeepGrove math.
// All cache accesses consume the hipEngine KVLiveSpans pointer ABI.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {

__device__ inline float bf16_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__device__ inline int64_t positive_mod(int64_t value, int64_t modulus) {
  const int64_t rem = value % modulus;
  return rem < 0 ? rem + modulus : rem;
}

__global__ void maple_kv_span_update_kernel(
    int64_t* __restrict__ live_counts,
    int64_t* __restrict__ token_positions,
    uint8_t* __restrict__ evict_mask,
    int64_t* __restrict__ row_positions,
    int64_t position,
    int64_t capacity) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int64_t slot = positive_mod(position, capacity);
  token_positions[slot] = position;
  evict_mask[slot] = 0;
  live_counts[0] = position + 1 < capacity ? position + 1 : capacity;
  row_positions[0] = position;
}

// Batched span publish (P4 prefill): marks positions [start, start+rows) valid
// in the ring.  One block per position row; block 0 also updates live_counts
// and row_positions.
__global__ void maple_kv_span_update_batched_kernel(
    int64_t* __restrict__ live_counts,
    int64_t* __restrict__ token_positions,
    uint8_t* __restrict__ evict_mask,
    int64_t* __restrict__ row_positions,
    int64_t start,
    int64_t rows,
    int64_t capacity) {
  const int64_t local = blockIdx.x;
  if (local >= rows) {
    return;
  }
  const int64_t position = start + local;
  const int64_t slot = positive_mod(position, capacity);
  if (threadIdx.x == 0) {
    token_positions[slot] = position;
    evict_mask[slot] = 0;
  }
  if (local == 0 && threadIdx.x == 0) {
    const int64_t last = start + rows - 1;
    live_counts[0] = last + 1 < capacity ? last + 1 : capacity;
    row_positions[0] = last;
  }
}

__global__ void maple_qknorm_rope_kv_write_kernel(
    uint16_t* __restrict__ qkv,
    const uint16_t* __restrict__ q_norm_weight,
    const uint16_t* __restrict__ k_norm_weight,
    uint16_t* __restrict__ key_cache,
    uint16_t* __restrict__ value_cache,
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t rope_dim,
    float eps,
    float rope_theta,
    int64_t capacity) {
  const int64_t head = blockIdx.x;
  const int64_t total_heads = q_heads + kv_heads;
  if (head >= total_heads) {
    return;
  }
  const bool is_q = head < q_heads;
  const int64_t local_head = is_q ? head : head - q_heads;
  const int64_t q_size = q_heads * head_dim;
  const int64_t kv_size = kv_heads * head_dim;
  const int64_t source_base = is_q ? head * head_dim : q_size + local_head * head_dim;
  const uint16_t* norm_weight = is_q ? q_norm_weight : k_norm_weight;
  const int64_t tid = threadIdx.x;

  extern __shared__ uint8_t shared_raw[];
  float* sums = reinterpret_cast<float*>(shared_raw);
  uint16_t* normalized = reinterpret_cast<uint16_t*>(sums + blockDim.x);
  float value = 0.0f;
  if (tid < head_dim) {
    value = bf16_to_float(qkv[source_base + tid]);
  }
  sums[tid] = value * value;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    __syncthreads();
  }
  if (tid < head_dim) {
    const float inv_rms = rsqrtf(sums[0] / static_cast<float>(head_dim) + eps);
    normalized[tid] = float_to_bf16(
        value * inv_rms * bf16_to_float(norm_weight[tid]));
  }
  __syncthreads();

  if (tid < head_dim) {
    float out_value = bf16_to_float(normalized[tid]);
    if (rope_dim > 0 && tid < rope_dim) {
      const int64_t half = rope_dim / 2;
      const int64_t pair = tid < half ? tid + half : tid - half;
      const int64_t frequency = tid < half ? tid : tid - half;
      const float exponent = -static_cast<float>(frequency) / static_cast<float>(half);
      const float theta = static_cast<float>(row_positions[0]) * powf(rope_theta, exponent);
      const float paired = bf16_to_float(normalized[pair]);
      const float cosine = cosf(theta);
      const float sine = sinf(theta);
      out_value = tid < half
          ? out_value * cosine - paired * sine
          : out_value * cosine + paired * sine;
    }
    const uint16_t out_bits = float_to_bf16(out_value);
    qkv[source_base + tid] = out_bits;

    if (!is_q) {
      const int64_t position = row_positions[0];
      const int64_t logical_slot = positive_mod(position, capacity);
      const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
      const bool valid = live_counts[0] > 0
          && token_positions[logical_slot] == position
          && evict_mask[logical_slot] == 0;
      if (valid && physical_slot >= 0 && physical_slot < capacity) {
        const int64_t cache_offset = physical_slot * kv_size + local_head * head_dim + tid;
        key_cache[cache_offset] = out_bits;
        value_cache[cache_offset] = qkv[q_size + kv_size + local_head * head_dim + tid];
      }
    }
  }
}

// Batched qknorm+RoPE+KV write (P2 prefill): processes T prompt rows at
// absolute positions [start, start+T).  Grid is (total_heads x T); block
// (head, row) qknorm-normalizes that head's vector for row, applies RoPE at
// position = start+row, writes the normalized q back into the q-region of
// qkv[row], and writes k/v into the shared KVLiveSpans ring physical slots
// (same ABI as the decode qknorm write, so prefill and decode share one
// cache).  qkv layout is [T, q_size + 2*kv_size].
__global__ void maple_qknorm_rope_kv_write_batched_kernel(
    uint16_t* __restrict__ qkv,
    const uint16_t* __restrict__ q_norm_weight,
    const uint16_t* __restrict__ k_norm_weight,
    uint16_t* __restrict__ key_cache,
    uint16_t* __restrict__ value_cache,
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t rope_dim,
    float eps,
    float rope_theta,
    int64_t start,
    int64_t capacity) {
  const int64_t head = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int64_t total_heads = q_heads + kv_heads;
  if (head >= total_heads) {
    return;
  }
  const bool is_q = head < q_heads;
  const int64_t local_head = is_q ? head : head - q_heads;
  const int64_t q_size = q_heads * head_dim;
  const int64_t kv_size = kv_heads * head_dim;
  const int64_t row_base = row * (q_size + 2 * kv_size);
  const int64_t source_base = is_q ? head * head_dim : q_size + local_head * head_dim;
  const uint16_t* norm_weight = is_q ? q_norm_weight : k_norm_weight;
  const int64_t tid = threadIdx.x;
  const int64_t position = start + row;

  extern __shared__ uint8_t shared_raw[];
  float* sums = reinterpret_cast<float*>(shared_raw);
  uint16_t* normalized = reinterpret_cast<uint16_t*>(sums + blockDim.x);
  float value = 0.0f;
  if (tid < head_dim) {
    value = bf16_to_float(qkv[row_base + source_base + tid]);
  }
  sums[tid] = value * value;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    __syncthreads();
  }
  if (tid < head_dim) {
    const float inv_rms = rsqrtf(sums[0] / static_cast<float>(head_dim) + eps);
    normalized[tid] = float_to_bf16(
        value * inv_rms * bf16_to_float(norm_weight[tid]));
  }
  __syncthreads();

  if (tid < head_dim) {
    float out_value = bf16_to_float(normalized[tid]);
    if (rope_dim > 0 && tid < rope_dim) {
      const int64_t half = rope_dim / 2;
      const int64_t pair = tid < half ? tid + half : tid - half;
      const int64_t frequency = tid < half ? tid : tid - half;
      const float exponent = -static_cast<float>(frequency) / static_cast<float>(half);
      const float theta = static_cast<float>(position) * powf(rope_theta, exponent);
      const float paired = bf16_to_float(normalized[pair]);
      const float cosine = cosf(theta);
      const float sine = sinf(theta);
      out_value = tid < half
          ? out_value * cosine - paired * sine
          : out_value * cosine + paired * sine;
    }
    const uint16_t out_bits = float_to_bf16(out_value);
    qkv[row_base + source_base + tid] = out_bits;

    if (!is_q) {
      const int64_t logical_slot = positive_mod(position, capacity);
      const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
      const bool valid = live_counts[0] > 0
          && token_positions[logical_slot] == position
          && evict_mask[logical_slot] == 0;
      if (valid && physical_slot >= 0 && physical_slot < capacity) {
        const int64_t cache_offset = physical_slot * kv_size + local_head * head_dim + tid;
        key_cache[cache_offset] = out_bits;
        value_cache[cache_offset] = qkv[row_base + q_size + kv_size + local_head * head_dim + tid];
      }
    }
  }
}

// Batched per-request QK-norm+RoPE+KV write (D5 batch decode).
// Grid is (total_heads x rows); block (head, row) processes one request row at
// its own absolute position row_positions[row], writing that request's current
// K/V into the shared KV arena (identity-base_offsets position-offset arena so
// independent requests occupy disjoint position ranges). Bit-equivalent to the
// c1 qknorm_rope_kv_write kernel per row.
__global__ void maple_qknorm_rope_kv_write_batched_decode_kernel(
    uint16_t* __restrict__ qkv,
    const uint16_t* __restrict__ q_norm_weight,
    const uint16_t* __restrict__ k_norm_weight,
    uint16_t* __restrict__ key_cache,
    uint16_t* __restrict__ value_cache,
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    const int64_t* __restrict__ row_base_offsets,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t rope_dim,
    float eps,
    float rope_theta,
    int64_t rows,
    int64_t capacity) {
  const int64_t head = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int64_t total_heads = q_heads + kv_heads;
  if (head >= total_heads || row >= rows) {
    return;
  }
  const bool is_q = head < q_heads;
  const int64_t local_head = is_q ? head : head - q_heads;
  const int64_t q_size = q_heads * head_dim;
  const int64_t kv_size = kv_heads * head_dim;
  const int64_t row_base = row * (q_size + 2 * kv_size);
  const int64_t source_base = is_q ? head * head_dim : q_size + local_head * head_dim;
  const uint16_t* norm_weight = is_q ? q_norm_weight : k_norm_weight;
  const int64_t tid = threadIdx.x;
  const int64_t position = row_positions[row];
  const int64_t request_capacity = capacity / rows;
  const int64_t slot = row_base_offsets[row] + positive_mod(position, request_capacity);

  extern __shared__ uint8_t shared_raw[];
  float* sums = reinterpret_cast<float*>(shared_raw);
  uint16_t* normalized = reinterpret_cast<uint16_t*>(sums + blockDim.x);
  float value = 0.0f;
  if (tid < head_dim) {
    value = bf16_to_float(qkv[row_base + source_base + tid]);
  }
  sums[tid] = value * value;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    __syncthreads();
  }
  if (tid < head_dim) {
    const float inv_rms = rsqrtf(sums[0] / static_cast<float>(head_dim) + eps);
    normalized[tid] = float_to_bf16(
        value * inv_rms * bf16_to_float(norm_weight[tid]));
  }
  __syncthreads();

  if (tid < head_dim) {
    float out_value = bf16_to_float(normalized[tid]);
    if (rope_dim > 0 && tid < rope_dim) {
      const int64_t half = rope_dim / 2;
      const int64_t pair = tid < half ? tid + half : tid - half;
      const int64_t frequency = tid < half ? tid : tid - half;
      const float exponent = -static_cast<float>(frequency) / static_cast<float>(half);
      const float theta = static_cast<float>(position) * powf(rope_theta, exponent);
      const float paired = bf16_to_float(normalized[pair]);
      const float cosine = cosf(theta);
      const float sine = sinf(theta);
      out_value = tid < half
          ? out_value * cosine - paired * sine
          : out_value * cosine + paired * sine;
    }
    const uint16_t out_bits = float_to_bf16(out_value);
    qkv[row_base + source_base + tid] = out_bits;

    if (!is_q) {
      const int64_t logical_slot = positive_mod(slot, capacity);
      const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
      const bool valid = live_counts[row] > 0
          && token_positions[logical_slot] == position
          && evict_mask[logical_slot] == 0;
      if (valid && physical_slot >= 0 && physical_slot < capacity) {
        const int64_t cache_offset = physical_slot * kv_size + local_head * head_dim + tid;
        key_cache[cache_offset] = out_bits;
        value_cache[cache_offset] =
            qkv[row_base + q_size + kv_size + local_head * head_dim + tid];
      }
    }
  }
}

// Batched per-request attention decode (D5). Grid is (q_heads x rows); block
// (q_head, row) runs the same online-softmax scan as the c1 decode kernel but
// over request row's own live span [row_positions[row]-live_counts[row]+1,
// row_positions[row]], reading per-row live_counts/row_positions from the
// KVLiveSpans ABI. Bit-equivalent to c independent c1 decode runs.
__global__ void maple_attention_decode_batched_kernel(
    const uint16_t* __restrict__ qkv,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    uint16_t* __restrict__ out,
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    const int64_t* __restrict__ row_base_offsets,
    int64_t rows,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    int64_t capacity) {
  const int64_t q_head = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (q_head >= q_heads || row >= rows) {
    return;
  }
  const int64_t tid = threadIdx.x;
  const int64_t group_size = q_heads / kv_heads;
  const int64_t kv_head = q_head / group_size;
  const int64_t kv_stride = kv_heads * head_dim;
  const int64_t q_size = q_heads * head_dim;
  const int64_t qkv_stride = q_size + 2 * kv_stride;
  const int64_t request_capacity = capacity / rows;
  const int64_t live = live_counts[row] < request_capacity
      ? live_counts[row]
      : request_capacity;
  const int64_t query_position = row_positions[row];
  const int64_t arena_base = row_base_offsets[row];
  const int64_t start_position = query_position - live + 1;
  const float q = tid < head_dim
      ? bf16_to_float(qkv[row * qkv_stride + q_head * head_dim + tid])
      : 0.0f;
  float accumulator = 0.0f;
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  extern __shared__ float dot_scratch[];
  for (int64_t token = 0; token < live; ++token) {
    const int64_t absolute_position = start_position + token;
    const int64_t logical_slot =
        arena_base + positive_mod(absolute_position, request_capacity);
    const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    const bool valid = physical_slot >= 0 && physical_slot < capacity
        && token_positions[logical_slot] == absolute_position
        && evict_mask[logical_slot] == 0;
    if (!valid) {
      continue;
    }
    float dot = 0.0f;
    if (tid < head_dim) {
      const int64_t cache_offset = physical_slot * kv_stride + kv_head * head_dim + tid;
      dot = q * bf16_to_float(key_cache[cache_offset]);
    }
    dot_scratch[tid] = dot;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        dot_scratch[tid] += dot_scratch[tid + stride];
      }
      __syncthreads();
    }
    const float score = dot_scratch[0] * scale;
    const float next_max = fmaxf(running_max, score);
    const float old_scale = isinf(running_max) ? 0.0f : expf(running_max - next_max);
    const float new_scale = expf(score - next_max);
    if (tid < head_dim) {
      const int64_t cache_offset = physical_slot * kv_stride + kv_head * head_dim + tid;
      accumulator = accumulator * old_scale
          + bf16_to_float(value_cache[cache_offset]) * new_scale;
    }
    running_sum = running_sum * old_scale + new_scale;
    running_max = next_max;
    __syncthreads();
  }
  if (tid < head_dim) {
    out[row * q_size + q_head * head_dim + tid] = float_to_bf16(
        running_sum > 0.0f ? accumulator / running_sum : 0.0f);
  }
}

__global__ void maple_attention_decode_kernel(
    const uint16_t* __restrict__ qkv,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    uint16_t* __restrict__ out,
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    int64_t capacity) {
  const int64_t q_head = blockIdx.x;
  if (q_head >= q_heads) {
    return;
  }
  const int64_t tid = threadIdx.x;
  const int64_t group_size = q_heads / kv_heads;
  const int64_t kv_head = q_head / group_size;
  const int64_t live = live_counts[0] < capacity ? live_counts[0] : capacity;
  const int64_t query_position = row_positions[0];
  const int64_t start_position = query_position - live + 1;
  const int64_t kv_stride = kv_heads * head_dim;
  const float q = tid < head_dim
      ? bf16_to_float(qkv[q_head * head_dim + tid])
      : 0.0f;
  float accumulator = 0.0f;
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  extern __shared__ float dot_scratch[];
  for (int64_t token = 0; token < live; ++token) {
    const int64_t absolute_position = start_position + token;
    const int64_t logical_slot = positive_mod(absolute_position, capacity);
    const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    const bool valid = physical_slot >= 0 && physical_slot < capacity
        && token_positions[logical_slot] == absolute_position
        && evict_mask[logical_slot] == 0;
    if (!valid) {
      continue;
    }
    float dot = 0.0f;
    if (tid < head_dim) {
      const int64_t cache_offset = physical_slot * kv_stride + kv_head * head_dim + tid;
      dot = q * bf16_to_float(key_cache[cache_offset]);
    }
    dot_scratch[tid] = dot;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        dot_scratch[tid] += dot_scratch[tid + stride];
      }
      __syncthreads();
    }
    const float score = dot_scratch[0] * scale;
    const float next_max = fmaxf(running_max, score);
    const float old_scale = isinf(running_max) ? 0.0f : expf(running_max - next_max);
    const float new_scale = expf(score - next_max);
    if (tid < head_dim) {
      const int64_t cache_offset = physical_slot * kv_stride + kv_head * head_dim + tid;
      accumulator = accumulator * old_scale
          + bf16_to_float(value_cache[cache_offset]) * new_scale;
    }
    running_sum = running_sum * old_scale + new_scale;
    running_max = next_max;
    __syncthreads();
  }
  if (tid < head_dim) {
    out[q_head * head_dim + tid] = float_to_bf16(
        running_sum > 0.0f ? accumulator / running_sum : 0.0f);
  }
}

// Fused QK-norm+RoPE+KV-write + online-softmax attention decode (D2).
// One kernel per q-head block: (1) QK-norm+RoPE the current token's Q, (2)
// QK-norm+RoPE+KV-write the current token's K/V for this head's kv_head (done
// redundantly by every q-head in the group so each block is self-contained with
// no cross-block race), (3) run the same online-softmax scan as
// maple_attention_decode_kernel over the live history (including the just-
// written current slot). Bit-exact with the unfused qknorm_rope_kv_write +
// attention_decode chain (same reduction/rounding order). Grid is (q_heads).
__global__ void maple_attention_fused_qknorm_decode_kernel(
    uint16_t* __restrict__ qkv,
    const uint16_t* __restrict__ q_norm_weight,
    const uint16_t* __restrict__ k_norm_weight,
    uint16_t* __restrict__ key_cache,
    uint16_t* __restrict__ value_cache,
    uint16_t* __restrict__ out,
    const int32_t* __restrict__ base_offsets,
    const int64_t* __restrict__ live_counts,
    const int64_t* __restrict__ token_positions,
    const uint8_t* __restrict__ evict_mask,
    const int64_t* __restrict__ row_positions,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t rope_dim,
    float eps,
    float rope_theta,
    float scale,
    int64_t capacity) {
  const int64_t q_head = blockIdx.x;
  if (q_head >= q_heads) {
    return;
  }
  const int64_t group_size = q_heads / kv_heads;
  const int64_t kv_head = q_head / group_size;
  const int64_t kv_stride = kv_heads * head_dim;
  const int64_t q_size = q_heads * head_dim;
  const int64_t kv_size = kv_heads * head_dim;
  const int64_t tid = threadIdx.x;
  const int64_t live = live_counts[0] < capacity ? live_counts[0] : capacity;
  const int64_t query_position = row_positions[0];
  const int64_t start_position = query_position - live + 1;

  extern __shared__ uint8_t shared_raw[];
  float* sums = reinterpret_cast<float*>(shared_raw);
  uint16_t* normalized = reinterpret_cast<uint16_t*>(sums + blockDim.x);
  float* dot_scratch =
      reinterpret_cast<float*>(reinterpret_cast<uint16_t*>(normalized + head_dim));

  // (1) Q norm + RoPE.
  const uint16_t* qsrc = qkv + q_head * head_dim;
  float q = tid < head_dim ? bf16_to_float(qsrc[tid]) : 0.0f;
  sums[tid] = q * q;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sums[tid] += sums[tid + stride];
    }
    __syncthreads();
  }
  if (tid < head_dim) {
    const float inv_rms = rsqrtf(sums[0] / static_cast<float>(head_dim) + eps);
    normalized[tid] = float_to_bf16(
        q * inv_rms * bf16_to_float(q_norm_weight[tid]));
  }
  __syncthreads();
  if (tid < head_dim) {
    float qv = bf16_to_float(normalized[tid]);
    if (rope_dim > 0 && tid < rope_dim) {
      const int64_t half = rope_dim / 2;
      const int64_t pair = tid < half ? tid + half : tid - half;
      const int64_t frequency = tid < half ? tid : tid - half;
      const float exponent =
          -static_cast<float>(frequency) / static_cast<float>(half);
      const float theta = static_cast<float>(query_position)
          * powf(rope_theta, exponent);
      const float paired = bf16_to_float(normalized[pair]);
      const float cosine = cosf(theta);
      const float sine = sinf(theta);
      qv = tid < half ? qv * cosine - paired * sine
                      : qv * cosine + paired * sine;
    }
    // q must match the bf16-rounded Q that the unfused path stores in qkv
    // and the attention kernel reads back.
    q = tid < head_dim ? bf16_to_float(float_to_bf16(qv)) : 0.0f;
  }
  __syncthreads();

  // (2) K norm + RoPE + V write for this head's kv_head (redundant in-group).
  {
    const uint16_t* ksrc = qkv + q_size + kv_head * head_dim;
    const uint16_t* vsrc = qkv + q_size + kv_size + kv_head * head_dim;
    float k = tid < head_dim ? bf16_to_float(ksrc[tid]) : 0.0f;
    sums[tid] = k * k;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        sums[tid] += sums[tid + stride];
      }
      __syncthreads();
    }
    if (tid < head_dim) {
      const float inv_rms = rsqrtf(sums[0] / static_cast<float>(head_dim) + eps);
      normalized[tid] = float_to_bf16(
          k * inv_rms * bf16_to_float(k_norm_weight[tid]));
    }
    __syncthreads();
    const int64_t logical_slot = positive_mod(query_position, capacity);
    const int64_t physical_slot =
        static_cast<int64_t>(base_offsets[logical_slot]);
    const bool valid = live > 0
        && token_positions[logical_slot] == query_position
        && evict_mask[logical_slot] == 0;
    if (tid < head_dim && valid && physical_slot >= 0 && physical_slot < capacity) {
      float kv = bf16_to_float(normalized[tid]);
      if (rope_dim > 0 && tid < rope_dim) {
        const int64_t half = rope_dim / 2;
        const int64_t pair = tid < half ? tid + half : tid - half;
        const int64_t frequency = tid < half ? tid : tid - half;
        const float exponent =
            -static_cast<float>(frequency) / static_cast<float>(half);
        const float theta = static_cast<float>(query_position)
            * powf(rope_theta, exponent);
        const float paired = bf16_to_float(normalized[pair]);
        const float cosine = cosf(theta);
        const float sine = sinf(theta);
        kv = tid < half ? kv * cosine - paired * sine
                        : kv * cosine + paired * sine;
      }
      const uint16_t k_bits = float_to_bf16(kv);
      const int64_t cache_offset =
          physical_slot * kv_stride + kv_head * head_dim + tid;
      key_cache[cache_offset] = k_bits;
      value_cache[cache_offset] = vsrc[tid];
    }
  }
  __syncthreads();

  // (3) Online-softmax attention over the live history.
  float accumulator = 0.0f;
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  for (int64_t token = 0; token < live; ++token) {
    const int64_t absolute_position = start_position + token;
    const int64_t logical_slot = positive_mod(absolute_position, capacity);
    const int64_t physical_slot =
        static_cast<int64_t>(base_offsets[logical_slot]);
    const bool valid = physical_slot >= 0 && physical_slot < capacity
        && token_positions[logical_slot] == absolute_position
        && evict_mask[logical_slot] == 0;
    if (!valid) {
      continue;
    }
    float dot = 0.0f;
    if (tid < head_dim) {
      const int64_t cache_offset =
          physical_slot * kv_stride + kv_head * head_dim + tid;
      dot = q * bf16_to_float(key_cache[cache_offset]);
    }
    dot_scratch[tid] = dot;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        dot_scratch[tid] += dot_scratch[tid + stride];
      }
      __syncthreads();
    }
    const float score = dot_scratch[0] * scale;
    const float next_max = fmaxf(running_max, score);
    const float old_scale = isinf(running_max) ? 0.0f : expf(running_max - next_max);
    const float new_scale = expf(score - next_max);
    if (tid < head_dim) {
      const int64_t cache_offset =
          physical_slot * kv_stride + kv_head * head_dim + tid;
      accumulator = accumulator * old_scale
          + bf16_to_float(value_cache[cache_offset]) * new_scale;
    }
    running_sum = running_sum * old_scale + new_scale;
    running_max = next_max;
    __syncthreads();
  }
  if (tid < head_dim) {
    out[q_head * head_dim + tid] = float_to_bf16(
        running_sum > 0.0f ? accumulator / running_sum : 0.0f);
  }
}

// Batched prefill attention (P2): processes T query rows causally against a
// dense prefix KV cache.  Grid is (q_heads x T) blocks, one block per
// (head, row); each block runs the same online-softmax scan as the decode
// kernel over the causal prefix [0, row] (inclusive), producing out[row].
// The KV cache here is the raw [seq, kv_heads, head_dim] prefill buffer
// (positions already qknorm+RoPE'd), not the KVLiveSpans ring.
__global__ void maple_attention_prefill_kernel(
    const uint16_t* __restrict__ qkv,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t q_head = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (q_head >= q_heads || row >= rows) {
    return;
  }
  const int64_t tid = threadIdx.x;
  const int64_t group_size = q_heads / kv_heads;
  const int64_t kv_head = q_head / group_size;
  const int64_t kv_stride = kv_heads * head_dim;
  const float q = tid < head_dim
      ? bf16_to_float(qkv[row * q_heads * head_dim + q_head * head_dim + tid])
      : 0.0f;
  float accumulator = 0.0f;
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  extern __shared__ float dot_scratch[];
  for (int64_t token = 0; token <= row; ++token) {
    float dot = 0.0f;
    if (tid < head_dim) {
      const int64_t cache_offset = token * kv_stride + kv_head * head_dim + tid;
      dot = q * bf16_to_float(key_cache[cache_offset]);
    }
    dot_scratch[tid] = dot;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        dot_scratch[tid] += dot_scratch[tid + stride];
      }
      __syncthreads();
    }
    const float score = dot_scratch[0] * scale;
    const float next_max = fmaxf(running_max, score);
    const float old_scale = isinf(running_max) ? 0.0f : expf(running_max - next_max);
    const float new_scale = expf(score - next_max);
    if (tid < head_dim) {
      const int64_t cache_offset = token * kv_stride + kv_head * head_dim + tid;
      accumulator = accumulator * old_scale
          + bf16_to_float(value_cache[cache_offset]) * new_scale;
    }
    running_sum = running_sum * old_scale + new_scale;
    running_max = next_max;
    __syncthreads();
  }
  if (tid < head_dim) {
    out[row * q_heads * head_dim + q_head * head_dim + tid] = float_to_bf16(
        running_sum > 0.0f ? accumulator / running_sum : 0.0f);
  }
}

// Ring-aware batched causal prefill attention (P2, shared-cache variant): same
// online-softmax scan as maple_attention_prefill_kernel, but reads the KV cache
// through the KVLiveSpans ring (logical slot -> base_offsets[logical] physical
// slot), so it can consume the output of
// maple_qknorm_rope_kv_write_batched_kernel and share one cache with decode.
// Grid is (q_heads x T); block (qh, row) attends to the complete live causal
// prefix ending at start+row, including positions written by earlier chunks.
// qkv is the full [T, q_size + 2*kv_size] buffer and the query is read from its
// q-region. Callers must not append a chunk that overwrites still-live prefix
// slots before this kernel runs.
__global__ void maple_attention_prefill_ring_kernel(
    const uint16_t* __restrict__ qkv,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    uint16_t* __restrict__ out,
    const int32_t* __restrict__ base_offsets,
    int64_t rows,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    int64_t start,
    int64_t capacity) {
  const int64_t q_head = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (q_head >= q_heads || row >= rows) {
    return;
  }
  const int64_t tid = threadIdx.x;
  const int64_t group_size = q_heads / kv_heads;
  const int64_t kv_head = q_head / group_size;
  const int64_t kv_stride = kv_heads * head_dim;
  const int64_t q_size = q_heads * head_dim;
  const int64_t qkv_stride = q_size + 2 * kv_stride;
  const float q = tid < head_dim
      ? bf16_to_float(qkv[row * qkv_stride + q_head * head_dim + tid])
      : 0.0f;
  float accumulator = 0.0f;
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  extern __shared__ float dot_scratch[];
  const int64_t query_position = start + row;
  const int64_t first_position = query_position + 1 > capacity
      ? query_position + 1 - capacity
      : 0;
  for (int64_t absolute_position = first_position;
       absolute_position <= query_position;
       ++absolute_position) {
    const int64_t logical_slot = positive_mod(absolute_position, capacity);
    const int64_t physical_slot = static_cast<int64_t>(base_offsets[logical_slot]);
    float dot = 0.0f;
    if (tid < head_dim) {
      const int64_t cache_offset = physical_slot * kv_stride + kv_head * head_dim + tid;
      dot = q * bf16_to_float(key_cache[cache_offset]);
    }
    dot_scratch[tid] = dot;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        dot_scratch[tid] += dot_scratch[tid + stride];
      }
      __syncthreads();
    }
    const float score = dot_scratch[0] * scale;
    const float next_max = fmaxf(running_max, score);
    const float old_scale = isinf(running_max) ? 0.0f : expf(running_max - next_max);
    const float new_scale = expf(score - next_max);
    if (tid < head_dim) {
      const int64_t cache_offset = physical_slot * kv_stride + kv_head * head_dim + tid;
      accumulator = accumulator * old_scale
          + bf16_to_float(value_cache[cache_offset]) * new_scale;
    }
    running_sum = running_sum * old_scale + new_scale;
    running_max = next_max;
    __syncthreads();
  }
  if (tid < head_dim) {
    out[row * q_heads * head_dim + q_head * head_dim + tid] = float_to_bf16(
        running_sum > 0.0f ? accumulator / running_sum : 0.0f);
  }
}

inline int launch_threads_for_dim(int64_t head_dim) {
  if (head_dim > 64) {
    return 128;
  }
  if (head_dim > 32) {
    return 64;
  }
  return 32;
}

}  // namespace

extern "C" int hipengine_maple_kv_span_update(
    int64_t* live_counts,
    int64_t* token_positions,
    uint8_t* evict_mask,
    int64_t* row_positions,
    int64_t position,
    int64_t capacity,
    hipStream_t stream) {
  if (position < 0 || capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      maple_kv_span_update_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      position,
      capacity);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_kv_span_update_batched(
    int64_t* live_counts,
    int64_t* token_positions,
    uint8_t* evict_mask,
    int64_t* row_positions,
    int64_t start,
    int64_t rows,
    int64_t capacity,
    hipStream_t stream) {
  if (start < 0 || rows <= 0 || capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      maple_kv_span_update_batched_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(1),
      0,
      stream,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      start,
      rows,
      capacity);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_qknorm_rope_kv_write_bf16(
    uint16_t* qkv,
    const uint16_t* q_norm_weight,
    const uint16_t* k_norm_weight,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t rope_dim,
    float eps,
    float rope_theta,
    int64_t capacity,
    hipStream_t stream) {
  if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0
      || head_dim <= 0 || head_dim > 128 || rope_dim < 0 || rope_dim > head_dim
      || rope_dim % 2 != 0 || capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_dim(head_dim);
  const int64_t blocks = q_heads + kv_heads;
  const int64_t shared_bytes = threads * sizeof(float) + head_dim * sizeof(uint16_t);
  hipLaunchKernelGGL(
      maple_qknorm_rope_kv_write_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      static_cast<unsigned int>(shared_bytes),
      stream,
      qkv,
      q_norm_weight,
      k_norm_weight,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      q_heads,
      kv_heads,
      head_dim,
      rope_dim,
      eps,
      rope_theta,
      capacity);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_qknorm_rope_kv_write_batched_bf16(
    uint16_t* qkv,
    const uint16_t* q_norm_weight,
    const uint16_t* k_norm_weight,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t rope_dim,
    float eps,
    float rope_theta,
    int64_t start,
    int64_t rows,
    int64_t capacity,
    hipStream_t stream) {
  if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0
      || head_dim <= 0 || head_dim > 128 || rope_dim < 0 || rope_dim > head_dim
      || rope_dim % 2 != 0 || rows <= 0 || capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_dim(head_dim);
  const int64_t blocks = q_heads + kv_heads;
  const int64_t shared_bytes = threads * sizeof(float) + head_dim * sizeof(uint16_t);
  hipLaunchKernelGGL(
      maple_qknorm_rope_kv_write_batched_kernel,
      dim3(static_cast<unsigned int>(blocks),
           static_cast<unsigned int>(rows)),
      dim3(threads),
      static_cast<unsigned int>(shared_bytes),
      stream,
      qkv,
      q_norm_weight,
      k_norm_weight,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      q_heads,
      kv_heads,
      head_dim,
      rope_dim,
      eps,
      rope_theta,
      start,
      capacity);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_qknorm_rope_kv_write_batched_decode_bf16(
    uint16_t* qkv,
    const uint16_t* q_norm_weight,
    const uint16_t* k_norm_weight,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    const int64_t* row_base_offsets,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t rope_dim,
    float eps,
    float rope_theta,
    int64_t rows,
    int64_t capacity,
    hipStream_t stream) {
  if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0
      || head_dim <= 0 || head_dim > 128 || rope_dim < 0 || rope_dim > head_dim
      || rope_dim % 2 != 0 || rows <= 0 || capacity <= 0 || capacity % rows != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_dim(head_dim);
  const int64_t blocks = q_heads + kv_heads;
  const int64_t shared_bytes = threads * sizeof(float) + head_dim * sizeof(uint16_t);
  hipLaunchKernelGGL(
      maple_qknorm_rope_kv_write_batched_decode_kernel,
      dim3(static_cast<unsigned int>(blocks),
           static_cast<unsigned int>(rows)),
      dim3(threads),
      static_cast<unsigned int>(shared_bytes),
      stream,
      qkv,
      q_norm_weight,
      k_norm_weight,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      row_base_offsets,
      q_heads,
      kv_heads,
      head_dim,
      rope_dim,
      eps,
      rope_theta,
      rows,
      capacity);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_attention_decode_bf16(
    const uint16_t* qkv,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    uint16_t* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    int64_t capacity,
    hipStream_t stream) {
  if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0
      || head_dim <= 0 || head_dim > 128 || capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_dim(head_dim);
  hipLaunchKernelGGL(
      maple_attention_decode_kernel,
      dim3(static_cast<unsigned int>(q_heads)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      qkv,
      key_cache,
      value_cache,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      q_heads,
      kv_heads,
      head_dim,
      scale,
      capacity);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_attention_fused_qknorm_decode_bf16(
    uint16_t* qkv,
    const uint16_t* q_norm_weight,
    const uint16_t* k_norm_weight,
    uint16_t* key_cache,
    uint16_t* value_cache,
    uint16_t* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t rope_dim,
    float eps,
    float rope_theta,
    float scale,
    int64_t capacity,
    hipStream_t stream) {
  if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0
      || head_dim <= 0 || head_dim > 128 || rope_dim < 0 || rope_dim > head_dim
      || rope_dim % 2 != 0 || capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_dim(head_dim);
  const int64_t shared_bytes =
      2 * threads * sizeof(float) + head_dim * sizeof(uint16_t);
  hipLaunchKernelGGL(
      maple_attention_fused_qknorm_decode_kernel,
      dim3(static_cast<unsigned int>(q_heads)),
      dim3(threads),
      static_cast<unsigned int>(shared_bytes),
      stream,
      qkv,
      q_norm_weight,
      k_norm_weight,
      key_cache,
      value_cache,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      q_heads,
      kv_heads,
      head_dim,
      rope_dim,
      eps,
      rope_theta,
      scale,
      capacity);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_attention_decode_batched_bf16(
    const uint16_t* qkv,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    uint16_t* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    const int64_t* token_positions,
    const uint8_t* evict_mask,
    const int64_t* row_positions,
    const int64_t* row_base_offsets,
    int64_t rows,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    int64_t capacity,
    hipStream_t stream) {
  if (rows <= 0 || q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0
      || head_dim <= 0 || head_dim > 128 || capacity <= 0 || capacity % rows != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_dim(head_dim);
  hipLaunchKernelGGL(
      maple_attention_decode_batched_kernel,
      dim3(static_cast<unsigned int>(q_heads),
           static_cast<unsigned int>(rows)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      qkv,
      key_cache,
      value_cache,
      out,
      base_offsets,
      live_counts,
      token_positions,
      evict_mask,
      row_positions,
      row_base_offsets,
      rows,
      q_heads,
      kv_heads,
      head_dim,
      scale,
      capacity);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_attention_prefill_bf16(
    const uint16_t* qkv,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    uint16_t* out,
    int64_t rows,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 || q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0
      || head_dim <= 0 || head_dim > 128) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_dim(head_dim);
  hipLaunchKernelGGL(
      maple_attention_prefill_kernel,
      dim3(static_cast<unsigned int>(q_heads),
           static_cast<unsigned int>(rows)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      qkv,
      key_cache,
      value_cache,
      out,
      rows,
      q_heads,
      kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_maple_attention_prefill_ring_bf16(
    const uint16_t* qkv,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    uint16_t* out,
    const int32_t* base_offsets,
    int64_t rows,
    int64_t q_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    int64_t start,
    int64_t capacity,
    hipStream_t stream) {
  if (rows <= 0 || q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0
      || head_dim <= 0 || head_dim > 128 || capacity <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = launch_threads_for_dim(head_dim);
  hipLaunchKernelGGL(
      maple_attention_prefill_ring_kernel,
      dim3(static_cast<unsigned int>(q_heads),
           static_cast<unsigned int>(rows)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      qkv,
      key_cache,
      value_cache,
      out,
      base_offsets,
      rows,
      q_heads,
      kv_heads,
      head_dim,
      scale,
      start,
      capacity);
  return static_cast<int>(hipGetLastError());
}
