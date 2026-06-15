// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip @ 59195ed.
// Qwen3.5 paged full-attention context-tensor decode body is preserved; public wrapper uses hipEngine KVLiveSpans.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>

typedef uint32_t uint32x4 __attribute__((ext_vector_type(4)));
using half_t = _Float16;

namespace {

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__device__ inline float sigmoid_f32(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ inline float softplus_f32(float value) {
  return value > 20.0f ? value : log1pf(expf(value));
}

template <typename scalar_t>
__device__ inline float scalar_to_float_qwen35(scalar_t value) {
  return static_cast<float>(value);
}

__device__ __forceinline__ void qwen35_full_attn_decode_impl(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    int64_t context_len,
    int64_t score_capacity,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    float* shared) {
  if (context_len <= 0 || context_len > score_capacity) {
    return;
  }
  float* scores = shared;
  float* partial = scores + score_capacity;
  const int64_t q_head = blockIdx.x;
  if (q_head >= num_q_heads) {
    return;
  }
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const int64_t q_offset = q_head * head_dim;

  float local_max = -3.4028234663852886e38f;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    float acc = 0.0f;
    const int64_t k_offset = (token * num_kv_heads + kv_head) * head_dim;
    for (int64_t dim = 0; dim < head_dim; ++dim) {
      acc += query[q_offset + dim] * bf16_bits_to_float(key_cache[k_offset + dim]);
    }
    const float score = acc * scale;
    scores[token] = score;
    local_max = fmaxf(local_max, score);
  }
  partial[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_score = partial[0];

  float local_sum = 0.0f;
  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    local_sum += expf(scores[token] - max_score);
  }
  partial[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_denom = 1.0f / fmaxf(partial[0], 1.0e-20f);

  for (int64_t token = threadIdx.x; token < context_len; token += blockDim.x) {
    scores[token] = expf(scores[token] - max_score) * inv_denom;
  }
  __syncthreads();

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < context_len; ++token) {
      const int64_t v_offset = (token * num_kv_heads + kv_head) * head_dim + dim;
      acc += scores[token] * bf16_bits_to_float(value_cache[v_offset]);
    }
    out[q_offset + dim] = acc;
  }
}

__global__ void qwen35_full_attn_decode_context_tensor_kernel(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int64_t* context_len_ptr,
    int64_t max_context_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  extern __shared__ float shared[];
  qwen35_full_attn_decode_impl(
      query,
      key_cache,
      value_cache,
      out,
      context_len_ptr[0],
      max_context_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale,
      shared);
}

// Packed dot product for paged attention key cache.
// Loads 8 bf16 values at a time via 128-bit vector loads (v_load_dwordx4).
// Requires head_dim divisible by 8 and k_offset aligned to 16 bytes.
__device__ __forceinline__ float paged_key_dot_vec8(
    const float* __restrict__ q_shared,
    const uint16_t* __restrict__ key_cache,
    int64_t k_offset,
    int64_t head_dim) {
  float acc = 0.0f;
  const uint32x4* k_vec = reinterpret_cast<const uint32x4*>(key_cache + k_offset);
  const int64_t vec_iters = head_dim >> 3;  // head_dim / 8
  for (int64_t i = 0; i < vec_iters; ++i) {
    const uint32x4 packed = k_vec[i];
    const int64_t dim_base = i << 3;  // i * 8
    // Each uint32 holds 2 bf16 values (lo + hi).
    acc += q_shared[dim_base + 0] * bf16_bits_to_float(static_cast<uint16_t>(packed[0] & 0xFFFFu));
    acc += q_shared[dim_base + 1] * bf16_bits_to_float(static_cast<uint16_t>(packed[0] >> 16));
    acc += q_shared[dim_base + 2] * bf16_bits_to_float(static_cast<uint16_t>(packed[1] & 0xFFFFu));
    acc += q_shared[dim_base + 3] * bf16_bits_to_float(static_cast<uint16_t>(packed[1] >> 16));
    acc += q_shared[dim_base + 4] * bf16_bits_to_float(static_cast<uint16_t>(packed[2] & 0xFFFFu));
    acc += q_shared[dim_base + 5] * bf16_bits_to_float(static_cast<uint16_t>(packed[2] >> 16));
    acc += q_shared[dim_base + 6] * bf16_bits_to_float(static_cast<uint16_t>(packed[3] & 0xFFFFu));
    acc += q_shared[dim_base + 7] * bf16_bits_to_float(static_cast<uint16_t>(packed[3] >> 16));
  }
  return acc;
}

__global__ void qwen35_paged_full_attn_decode_context_tensor_kernel(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    const int32_t* block_table,
    const int64_t* context_len_ptr,
    float* out,
    int64_t max_context_len,
    int64_t block_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t context_len = context_len_ptr[0];
  if (context_len <= 0 || context_len > max_context_len) {
    return;
  }
  extern __shared__ float shared[];
  float* scores = shared;
  float* warp_buf = scores + context_len;
  const int num_warps = blockDim.x / warpSize;
  float* q_shared = warp_buf + num_warps;
  const int64_t q_head = blockIdx.x;
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

  // Warp-per-token cooperative dot product. Derive the number of logical
  // warps from the compiled warpSize instead of assuming a fixed wave width.
  float local_max = -3.4028234663852886e38f;
  for (int64_t token = warp_id; token < context_len; token += num_warps) {
    const int64_t logical_block = token >> 8;
    const int64_t block_offset = token & 255;
    const int64_t physical_block = static_cast<int64_t>(block_table[logical_block]);
    const int64_t token_offset = (physical_block << 8) + block_offset;
    const int64_t k_offset = (token_offset * num_kv_heads + kv_head) * head_dim;
    float acc = 0.0f;
    for (int64_t dim = lane_id * 4; dim < head_dim; dim += warpSize * 4) {
      acc += q_shared[dim + 0] * bf16_bits_to_float(key_cache[k_offset + dim + 0]);
      acc += q_shared[dim + 1] * bf16_bits_to_float(key_cache[k_offset + dim + 1]);
      acc += q_shared[dim + 2] * bf16_bits_to_float(key_cache[k_offset + dim + 2]);
      acc += q_shared[dim + 3] * bf16_bits_to_float(key_cache[k_offset + dim + 3]);
    }
    // Warp shuffle reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      acc += __shfl_down(acc, offset);
    }
    if (lane_id == 0) {
      const float score = acc * scale;
      scores[token] = score;
      local_max = fmaxf(local_max, score);
    }
  }
  // Warp-level max reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    float other = __shfl_down(local_max, offset);
    local_max = fmaxf(local_max, other);
  }
  if (lane_id == 0) {
    warp_buf[warp_id] = local_max;
  }
  __syncthreads();
  // Cross-warp max (serial, negligible). Publish through shared memory so every
  // lane-0 softmax writer uses the same global max, not its private warp_buf[0].
  if (threadIdx.x == 0) {
    float max_score = warp_buf[0];
    for (int i = 1; i < num_warps; ++i) {
      max_score = fmaxf(max_score, warp_buf[i]);
    }
    warp_buf[0] = max_score;
  }
  __syncthreads();
  const float max_score = warp_buf[0];

  // Softmax sum - same warp-per-token pattern
  float local_sum = 0.0f;
  for (int64_t token = warp_id; token < context_len; token += num_warps) {
    if (lane_id == 0) {
      const float val = expf(scores[token] - max_score);
      scores[token] = val;
      local_sum += val;
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
    for (int i = 1; i < num_warps; ++i) {
      total_sum += warp_buf[i];
    }
    warp_buf[0] = 1.0f / fmaxf(total_sum, 1.0e-20f);
  }
  __syncthreads();
  const float inv_denom = warp_buf[0];

  for (int64_t token = warp_id; token < context_len; token += num_warps) {
    if (lane_id == 0) {
      scores[token] *= inv_denom;
    }
  }
  __syncthreads();

  // Value cache weighted sum - keep existing pattern (already coalesced)
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < context_len; ++token) {
      const int64_t logical_block = token >> 8;
      const int64_t block_offset = token & 255;
      const int64_t physical_block = static_cast<int64_t>(block_table[logical_block]);
      const int64_t v_offset = (((physical_block << 8) + block_offset) * num_kv_heads + kv_head) * head_dim + dim;
      acc += scores[token] * bf16_bits_to_float(value_cache[v_offset]);
    }
    out[q_offset + dim] = acc;
  }
}


__global__ void qwen35_full_attn_gate_mul_bf16_kernel(
    const float* attn,
    const uint16_t* gate,
    uint16_t* out,
    int64_t total) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float gated = attn[idx] * sigmoid_f32(bf16_bits_to_float(gate[idx]));
    out[idx] = float_to_bf16_bits(gated);
  }
}

__global__ void qwen35_full_attn_gate_mul_fp16_kernel(
    const float* attn,
    const half_t* gate,
    half_t* out,
    int64_t total) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float gated = attn[idx] * sigmoid_f32(static_cast<float>(gate[idx]));
    out[idx] = static_cast<half_t>(gated);
  }
}

__global__ void qwen35_paged_full_attn_decode_context_tensor_batch_kernel(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    const int32_t* block_table,
    const int64_t* context_len_ptr,
    float* out,
    int64_t rows,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= rows) {
    return;
  }
  const int64_t context_len = context_len_ptr[row];
  if (context_len <= 0 || context_len > max_context_len) {
    return;
  }
  extern __shared__ float shared[];
  float* scores = shared;
  float* warp_buf = scores + context_len;
  const int num_warps = blockDim.x / warpSize;
  float* q_shared = warp_buf + num_warps;
  // For c=2 long-context Qwen decode, the 1024-thread launch would otherwise
  // leave most threads idle while one thread per dim scans all values.
  // Keep tiny primitive fixtures on the exact c1-style accumulation order.
  const int raw_value_groups = (head_dim >= 128 && context_len >= 128) ? static_cast<int>(blockDim.x / static_cast<unsigned int>(head_dim)) : 0;
  const int value_groups = (blockDim.x >= 512 && raw_value_groups > 1) ? raw_value_groups : 1;
  float* value_partials = q_shared + head_dim;
  const int64_t q_head = blockIdx.x;
  if (q_head >= num_q_heads) {
    return;
  }
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const int64_t q_offset = (row * num_q_heads + q_head) * head_dim;
  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    q_shared[dim] = query[q_offset + dim];
  }
  __syncthreads();

  float local_max = -3.4028234663852886e38f;
  for (int64_t token = warp_id; token < context_len; token += num_warps) {
    const int64_t logical_block = token / block_size;
    const int64_t block_offset = token - logical_block * block_size;
    const int64_t physical_block = static_cast<int64_t>(block_table[row * block_table_len + logical_block]);
    const int64_t token_offset = ((row * block_table_len + physical_block) * block_size + block_offset) * num_kv_heads * head_dim;
    const int64_t k_offset = token_offset + kv_head * head_dim;
    float acc = 0.0f;
    for (int64_t dim = lane_id * 4; dim < head_dim; dim += warpSize * 4) {
      acc += q_shared[dim + 0] * bf16_bits_to_float(key_cache[k_offset + dim + 0]);
      acc += q_shared[dim + 1] * bf16_bits_to_float(key_cache[k_offset + dim + 1]);
      acc += q_shared[dim + 2] * bf16_bits_to_float(key_cache[k_offset + dim + 2]);
      acc += q_shared[dim + 3] * bf16_bits_to_float(key_cache[k_offset + dim + 3]);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      acc += __shfl_down(acc, offset);
    }
    if (lane_id == 0) {
      const float score = acc * scale;
      scores[token] = score;
      local_max = fmaxf(local_max, score);
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    float other = __shfl_down(local_max, offset);
    local_max = fmaxf(local_max, other);
  }
  if (lane_id == 0) {
    warp_buf[warp_id] = local_max;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float max_score = warp_buf[0];
    for (int i = 1; i < num_warps; ++i) {
      max_score = fmaxf(max_score, warp_buf[i]);
    }
    warp_buf[0] = max_score;
  }
  __syncthreads();
  const float max_score = warp_buf[0];

  float local_sum = 0.0f;
  for (int64_t token = warp_id; token < context_len; token += num_warps) {
    if (lane_id == 0) {
      const float val = expf(scores[token] - max_score);
      scores[token] = val;
      local_sum += val;
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
    for (int i = 1; i < num_warps; ++i) {
      total_sum += warp_buf[i];
    }
    warp_buf[0] = 1.0f / fmaxf(total_sum, 1.0e-20f);
  }
  __syncthreads();
  const float inv_denom = warp_buf[0];

  if (value_groups == 1) {
    for (int64_t token = warp_id; token < context_len; token += num_warps) {
      if (lane_id == 0) {
        scores[token] *= inv_denom;
      }
    }
    __syncthreads();
  }

  if (value_groups > 1) {
    const int64_t value_thread_count = static_cast<int64_t>(value_groups) * head_dim;
    if (static_cast<int64_t>(threadIdx.x) < value_thread_count) {
      const int64_t dim = static_cast<int64_t>(threadIdx.x) % head_dim;
      const int group = static_cast<int>(static_cast<int64_t>(threadIdx.x) / head_dim);
      float acc = 0.0f;
      for (int64_t token = group; token < context_len; token += value_groups) {
        const int64_t logical_block = token / block_size;
        const int64_t block_offset = token - logical_block * block_size;
        const int64_t physical_block = static_cast<int64_t>(block_table[row * block_table_len + logical_block]);
        const int64_t token_offset = ((row * block_table_len + physical_block) * block_size + block_offset) * num_kv_heads * head_dim;
        const int64_t v_offset = token_offset + kv_head * head_dim + dim;
        acc += (scores[token] * inv_denom) * bf16_bits_to_float(value_cache[v_offset]);
      }
      value_partials[static_cast<int64_t>(group) * head_dim + dim] = acc;
    }
    __syncthreads();
    if (threadIdx.x < head_dim) {
      float acc = 0.0f;
      for (int group = 0; group < value_groups; ++group) {
        acc += value_partials[static_cast<int64_t>(group) * head_dim + threadIdx.x];
      }
      out[q_offset + threadIdx.x] = acc;
    }
  } else {
    for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      float acc = 0.0f;
      for (int64_t token = 0; token < context_len; ++token) {
        const int64_t logical_block = token / block_size;
        const int64_t block_offset = token - logical_block * block_size;
        const int64_t physical_block = static_cast<int64_t>(block_table[row * block_table_len + logical_block]);
        const int64_t token_offset = ((row * block_table_len + physical_block) * block_size + block_offset) * num_kv_heads * head_dim;
        const int64_t v_offset = token_offset + kv_head * head_dim + dim;
        acc += scores[token] * bf16_bits_to_float(value_cache[v_offset]);
      }
      out[q_offset + dim] = acc;
    }
  }
}


__global__ void qwen35_paged_full_attn_decode_split_k_ctx_tensor_kernel(
    const float* __restrict__ query,         // [1, num_q_heads, head_dim]
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    const int32_t* __restrict__ block_table,
    const int64_t* __restrict__ context_len_ptr,  // device scalar
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int64_t chunk_size,
    int64_t num_splits,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t block_size,
    float scale) {
  const int64_t q_head = blockIdx.x;
  const int64_t split_idx = blockIdx.y;
  if (q_head >= num_q_heads || split_idx >= num_splits) {
    return;
  }

  const int64_t context_len = *context_len_ptr;
  const int64_t token_start = split_idx * chunk_size;
  const int64_t kv_head = (q_head * num_kv_heads) / num_q_heads;
  const int64_t q_offset = q_head * head_dim;
  const int64_t partial_idx = q_head * num_splits + split_idx;

  if (token_start >= context_len) {
    if (threadIdx.x == 0) {
      partial_m[partial_idx] = -3.4028234663852886e38f;
      partial_l[partial_idx] = 0.0f;
    }
    if (static_cast<int64_t>(threadIdx.x) < head_dim) {
      partial_out[partial_idx * head_dim + threadIdx.x] = 0.0f;
    }
    return;
  }

  const int64_t token_end_raw = token_start + chunk_size;
  const int64_t token_end = (token_end_raw < context_len) ? token_end_raw : context_len;
  const int64_t chunk_len = token_end - token_start;

  extern __shared__ float shared[];
  float* scores = shared;
  float* partial = scores + chunk_size;
  float* q_shared = partial + blockDim.x;

  for (int64_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    q_shared[i] = query[q_offset + i];
  }
  __syncthreads();

  float local_max = -3.4028234663852886e38f;
  for (int64_t lt = threadIdx.x; lt < chunk_len; lt += blockDim.x) {
    const int64_t token = token_start + lt;
    const int64_t logical_block = token / block_size;
    const int64_t block_offset = token - logical_block * block_size;
    const int64_t physical_block = static_cast<int64_t>(block_table[logical_block]);
    const int64_t token_offset = physical_block * block_size + block_offset;
    const int64_t k_offset = (token_offset * num_kv_heads + kv_head) * head_dim;
    const float acc = paged_key_dot_vec8(q_shared, key_cache, k_offset, head_dim);
    const float score = acc * scale;
    scores[lt] = score;
    local_max = fmaxf(local_max, score);
  }
  partial[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float chunk_max = partial[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int64_t lt = threadIdx.x; lt < chunk_len; lt += blockDim.x) {
    const float e = expf(scores[lt] - chunk_max);
    scores[lt] = e;
    local_sum += e;
  }
  partial[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float chunk_sum = partial[0];
  __syncthreads();

  if (static_cast<int64_t>(threadIdx.x) < head_dim) {
    const int64_t dim = threadIdx.x;
    float acc = 0.0f;
    for (int64_t lt = 0; lt < chunk_len; ++lt) {
      const int64_t token = token_start + lt;
      const int64_t logical_block = token / block_size;
      const int64_t block_offset = token - logical_block * block_size;
      const int64_t physical_block = static_cast<int64_t>(block_table[logical_block]);
      const int64_t v_offset = ((physical_block * block_size + block_offset) * num_kv_heads + kv_head) * head_dim + dim;
      acc += scores[lt] * bf16_bits_to_float(value_cache[v_offset]);
    }
    partial_out[partial_idx * head_dim + dim] = acc;
  }

  if (threadIdx.x == 0) {
    partial_m[partial_idx] = chunk_max;
    partial_l[partial_idx] = chunk_sum;
  }
}


__global__ void qwen35_paged_full_attn_decode_split_k_ctx_tensor_warp_kernel(
    const float* __restrict__ query,         // [1, 16, 256]
    const uint16_t* __restrict__ key_cache,  // [num_blocks, 256, 2, 256]
    const uint16_t* __restrict__ value_cache,
    const int32_t* __restrict__ block_table,
    const int64_t* __restrict__ context_len_ptr,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int64_t chunk_size,
    int64_t num_splits,
    float scale) {
  constexpr int64_t head_dim = 256;
  constexpr int64_t num_q_heads = 16;
  constexpr int64_t num_kv_heads = 2;
  constexpr int64_t block_size = 256;
  const int64_t q_head = blockIdx.x;
  const int64_t split_idx = blockIdx.y;
  if (q_head >= num_q_heads || split_idx >= num_splits) {
    return;
  }

  const int64_t context_len = *context_len_ptr;
  const int64_t token_start = split_idx * chunk_size;
  const int64_t kv_head = q_head >> 3;
  const int64_t q_offset = q_head * head_dim;
  const int64_t partial_idx = q_head * num_splits + split_idx;

  if (token_start >= context_len) {
    if (threadIdx.x == 0) {
      partial_m[partial_idx] = -3.4028234663852886e38f;
      partial_l[partial_idx] = 0.0f;
    }
    if (static_cast<int64_t>(threadIdx.x) < head_dim) {
      partial_out[partial_idx * head_dim + threadIdx.x] = 0.0f;
    }
    return;
  }

  const int64_t token_end_raw = token_start + chunk_size;
  const int64_t token_end = (token_end_raw < context_len) ? token_end_raw : context_len;
  const int64_t chunk_len = token_end - token_start;

  extern __shared__ float shared[];
  float* scores = shared;
  float* partial = scores + chunk_size;
  float* q_shared = partial + blockDim.x;
  int32_t* token_offsets = reinterpret_cast<int32_t*>(q_shared + head_dim);
  int32_t* split_meta = token_offsets + chunk_size;

  q_shared[threadIdx.x] = query[q_offset + threadIdx.x];
  if (threadIdx.x == 0) {
    const int64_t start_block = token_start >> 8;
    const int64_t blocks_in_chunk = (chunk_len + block_size - 1) >> 8;
    const int32_t base_physical_block = block_table[start_block];
    bool contiguous_blocks = true;
    for (int64_t b = 1; b < blocks_in_chunk; ++b) {
      if (block_table[start_block + b] != base_physical_block + static_cast<int32_t>(b)) {
        contiguous_blocks = false;
        break;
      }
    }
    split_meta[0] = contiguous_blocks ? 1 : 0;
    split_meta[1] = base_physical_block;
  }
  __syncthreads();

  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const int num_warps = blockDim.x / warpSize;
  const bool contiguous_blocks = split_meta[0] != 0;
  const int64_t base_physical_block = static_cast<int64_t>(split_meta[1]);

  float local_max = -3.4028234663852886e38f;
  for (int64_t lt = warp_id; lt < chunk_len; lt += num_warps) {
    const int64_t token = token_start + lt;
    int64_t token_offset;
    if (contiguous_blocks) {
      token_offset = (base_physical_block << 8) + lt;
    } else {
      const int64_t logical_block = token >> 8;
      const int64_t block_offset = token & 255;
      const int64_t physical_block = static_cast<int64_t>(block_table[logical_block]);
      token_offset = (physical_block << 8) + block_offset;
    }
    const int64_t k_offset = (token_offset * num_kv_heads + kv_head) * head_dim;
    float acc = 0.0f;
    for (int64_t dim = lane_id * 4; dim < head_dim; dim += warpSize * 4) {
      acc += q_shared[dim + 0] * bf16_bits_to_float(key_cache[k_offset + dim + 0]);
      acc += q_shared[dim + 1] * bf16_bits_to_float(key_cache[k_offset + dim + 1]);
      acc += q_shared[dim + 2] * bf16_bits_to_float(key_cache[k_offset + dim + 2]);
      acc += q_shared[dim + 3] * bf16_bits_to_float(key_cache[k_offset + dim + 3]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      acc += __shfl_down(acc, offset);
    }
    if (lane_id == 0) {
      const float score = acc * scale;
      scores[lt] = score;
      token_offsets[lt] = static_cast<int32_t>(token_offset);
      local_max = fmaxf(local_max, score);
    }
  }
  if (lane_id == 0) {
    partial[warp_id] = local_max;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    float warp_max = (threadIdx.x < num_warps) ? partial[threadIdx.x] : -3.4028234663852886e38f;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      warp_max = fmaxf(warp_max, __shfl_down(warp_max, offset));
    }
    if (threadIdx.x == 0) {
      partial[0] = warp_max;
    }
  }
  __syncthreads();
  const float chunk_max = partial[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int64_t lt = threadIdx.x; lt < chunk_len; lt += blockDim.x) {
    const float e = expf(scores[lt] - chunk_max);
    scores[lt] = e;
    local_sum += e;
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_sum += __shfl_down(local_sum, offset);
  }
  if (lane_id == 0) {
    partial[warp_id] = local_sum;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    float warp_sum = (threadIdx.x < num_warps) ? partial[threadIdx.x] : 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      warp_sum += __shfl_down(warp_sum, offset);
    }
    if (threadIdx.x == 0) {
      partial[0] = warp_sum;
    }
  }
  __syncthreads();
  const float chunk_sum = partial[0];
  __syncthreads();

  if (static_cast<int64_t>(threadIdx.x) < head_dim) {
    const int64_t dim = threadIdx.x;
    float acc = 0.0f;
    for (int64_t lt = 0; lt < chunk_len; ++lt) {
      const int64_t v_offset = (static_cast<int64_t>(token_offsets[lt]) * num_kv_heads + kv_head) * head_dim + dim;
      acc += scores[lt] * bf16_bits_to_float(value_cache[v_offset]);
    }
    partial_out[partial_idx * head_dim + dim] = acc;
  }

  if (threadIdx.x == 0) {
    partial_m[partial_idx] = chunk_max;
    partial_l[partial_idx] = chunk_sum;
  }
}

template <int64_t q_per_kv, int64_t num_q_heads, int64_t num_kv_heads>
__global__ void qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_kernel(
    const float* __restrict__ query,         // [1, num_q_heads, 256]
    const uint16_t* __restrict__ key_cache,  // [num_blocks, 256, num_kv_heads, 256]
    const uint16_t* __restrict__ value_cache,
    const int32_t* __restrict__ block_table,
    const int64_t* __restrict__ context_len_ptr,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int64_t chunk_size,
    int64_t num_splits,
    float scale) {
  constexpr int64_t head_dim = 256;
  constexpr int64_t block_size = 256;
  const int64_t kv_head = blockIdx.x;
  const int64_t split_idx = blockIdx.y;
  if (kv_head >= num_kv_heads || split_idx >= num_splits) {
    return;
  }

  const int64_t context_len = *context_len_ptr;
  const int64_t token_start = split_idx * chunk_size;
  const int64_t q_base = kv_head * q_per_kv;

  if (token_start >= context_len) {
    if (threadIdx.x == 0) {
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const int64_t partial_idx = (q_base + h) * num_splits + split_idx;
        partial_m[partial_idx] = -3.4028234663852886e38f;
        partial_l[partial_idx] = 0.0f;
      }
    }
    if (static_cast<int64_t>(threadIdx.x) < head_dim) {
      const int64_t dim = threadIdx.x;
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const int64_t partial_idx = (q_base + h) * num_splits + split_idx;
        partial_out[partial_idx * head_dim + dim] = 0.0f;
      }
    }
    return;
  }

  const int64_t token_end_raw = token_start + chunk_size;
  const int64_t token_end = (token_end_raw < context_len) ? token_end_raw : context_len;
  const int64_t chunk_len = token_end - token_start;

  extern __shared__ float shared[];
  float* scores = shared;                         // [8, chunk_size]
  const int num_warps = blockDim.x / warpSize;
  float* partial = scores + q_per_kv * chunk_size; // [8, num_warps]
  float* q_shared = partial + q_per_kv * num_warps; // [8, 256]
  int32_t* token_offsets = reinterpret_cast<int32_t*>(q_shared + q_per_kv * head_dim);
  int32_t* split_meta = token_offsets + chunk_size;

  for (int64_t h = 0; h < q_per_kv; ++h) {
    q_shared[h * head_dim + threadIdx.x] = query[(q_base + h) * head_dim + threadIdx.x];
  }
  if (threadIdx.x == 0) {
    const int64_t start_block = token_start >> 8;
    const int64_t blocks_in_chunk = (chunk_len + block_size - 1) >> 8;
    const int32_t base_physical_block = block_table[start_block];
    bool contiguous_blocks = true;
    for (int64_t b = 1; b < blocks_in_chunk; ++b) {
      if (block_table[start_block + b] != base_physical_block + static_cast<int32_t>(b)) {
        contiguous_blocks = false;
        break;
      }
    }
    split_meta[0] = contiguous_blocks ? 1 : 0;
    split_meta[1] = base_physical_block;
  }
  __syncthreads();

  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const bool contiguous_blocks = split_meta[0] != 0;
  const int64_t base_physical_block = static_cast<int64_t>(split_meta[1]);

  float local_max[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    local_max[h] = -3.4028234663852886e38f;
  }

  for (int64_t lt = warp_id; lt < chunk_len; lt += num_warps) {
    const int64_t token = token_start + lt;
    int64_t token_offset;
    if (contiguous_blocks) {
      token_offset = (base_physical_block << 8) + lt;
    } else {
      const int64_t logical_block = token >> 8;
      const int64_t block_offset = token & 255;
      const int64_t physical_block = static_cast<int64_t>(block_table[logical_block]);
      token_offset = (physical_block << 8) + block_offset;
    }
    const int64_t k_offset = (token_offset * num_kv_heads + kv_head) * head_dim;
    float acc[q_per_kv];
    for (int64_t h = 0; h < q_per_kv; ++h) {
      acc[h] = 0.0f;
    }
    for (int64_t dim = lane_id * 4; dim < head_dim; dim += warpSize * 4) {
      const float k0 = bf16_bits_to_float(key_cache[k_offset + dim + 0]);
      const float k1 = bf16_bits_to_float(key_cache[k_offset + dim + 1]);
      const float k2 = bf16_bits_to_float(key_cache[k_offset + dim + 2]);
      const float k3 = bf16_bits_to_float(key_cache[k_offset + dim + 3]);
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const float* q = q_shared + h * head_dim + dim;
        acc[h] += q[0] * k0;
        acc[h] += q[1] * k1;
        acc[h] += q[2] * k2;
        acc[h] += q[3] * k3;
      }
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      for (int64_t h = 0; h < q_per_kv; ++h) {
        acc[h] += __shfl_down(acc[h], offset);
      }
    }
    if (lane_id == 0) {
      token_offsets[lt] = static_cast<int32_t>(token_offset);
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const float score = acc[h] * scale;
        scores[h * chunk_size + lt] = score;
        local_max[h] = fmaxf(local_max[h], score);
      }
    }
  }
  if (lane_id == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      partial[h * num_warps + warp_id] = local_max[h];
    }
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      float warp_max = (threadIdx.x < num_warps) ? partial[h * num_warps + threadIdx.x]
                                                 : -3.4028234663852886e38f;
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_down(warp_max, offset));
      }
      if (threadIdx.x == 0) {
        partial[h * num_warps] = warp_max;
      }
    }
  }
  __syncthreads();

  float chunk_max[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    chunk_max[h] = partial[h * num_warps];
  }
  __syncthreads();

  float local_sum[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    local_sum[h] = 0.0f;
    for (int64_t lt = threadIdx.x; lt < chunk_len; lt += blockDim.x) {
      const float e = expf(scores[h * chunk_size + lt] - chunk_max[h]);
      scores[h * chunk_size + lt] = e;
      local_sum[h] += e;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      local_sum[h] += __shfl_down(local_sum[h], offset);
    }
  }
  if (lane_id == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      partial[h * num_warps + warp_id] = local_sum[h];
    }
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      float warp_sum = (threadIdx.x < num_warps) ? partial[h * num_warps + threadIdx.x] : 0.0f;
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        warp_sum += __shfl_down(warp_sum, offset);
      }
      if (threadIdx.x == 0) {
        partial[h * num_warps] = warp_sum;
      }
    }
  }
  __syncthreads();

  float chunk_sum[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    chunk_sum[h] = partial[h * num_warps];
  }
  __syncthreads();

  if (static_cast<int64_t>(threadIdx.x) < head_dim) {
    const int64_t dim = threadIdx.x;
    float acc[q_per_kv];
    for (int64_t h = 0; h < q_per_kv; ++h) {
      acc[h] = 0.0f;
    }
    if (contiguous_blocks) {
      const int64_t v_base = ((base_physical_block << 8) * num_kv_heads + kv_head) * head_dim + dim;
      constexpr int64_t v_stride = num_kv_heads * head_dim;
      for (int64_t lt = 0; lt < chunk_len; ++lt) {
        const float v = bf16_bits_to_float(value_cache[v_base + lt * v_stride]);
        for (int64_t h = 0; h < q_per_kv; ++h) {
          acc[h] += scores[h * chunk_size + lt] * v;
        }
      }
    } else {
      for (int64_t lt = 0; lt < chunk_len; ++lt) {
        const int64_t v_offset =
            (static_cast<int64_t>(token_offsets[lt]) * num_kv_heads + kv_head) * head_dim + dim;
        const float v = bf16_bits_to_float(value_cache[v_offset]);
        for (int64_t h = 0; h < q_per_kv; ++h) {
          acc[h] += scores[h * chunk_size + lt] * v;
        }
      }
    }
    for (int64_t h = 0; h < q_per_kv; ++h) {
      const int64_t partial_idx = (q_base + h) * num_splits + split_idx;
      partial_out[partial_idx * head_dim + dim] = acc[h];
    }
  }

  if (threadIdx.x == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      const int64_t partial_idx = (q_base + h) * num_splits + split_idx;
      partial_m[partial_idx] = chunk_max[h];
      partial_l[partial_idx] = chunk_sum[h];
    }
  }
}


template <int64_t q_per_kv, int64_t num_q_heads, int64_t num_kv_heads>
__global__ void qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_batch_kernel(
    const float* __restrict__ query,         // [rows, num_q_heads, 256]
    const uint16_t* __restrict__ key_cache,  // [num_blocks, 256, num_kv_heads, 256]
    const uint16_t* __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,
    const int64_t* __restrict__ context_len_ptr,
    float* __restrict__ partial_out,         // [rows, num_q_heads, num_splits, 256]
    float* __restrict__ partial_m,           // [rows, num_q_heads, num_splits]
    float* __restrict__ partial_l,           // [rows, num_q_heads, num_splits]
    int64_t rows,
    int64_t chunk_size,
    int64_t num_splits,
    int64_t block_table_len,
    float scale) {
  constexpr int64_t head_dim = 256;
  constexpr int64_t block_size = 256;
  const int64_t kv_head = blockIdx.x;
  const int64_t split_idx = blockIdx.y;
  const int64_t row = blockIdx.z;
  if (kv_head >= num_kv_heads || split_idx >= num_splits || row >= rows) {
    return;
  }

  const int64_t context_len = context_len_ptr[row];
  const int64_t token_start = split_idx * chunk_size;
  const int64_t q_base = kv_head * q_per_kv;
  const int32_t* __restrict__ block_table = block_tables + row * block_table_len;

  if (token_start >= context_len) {
    if (threadIdx.x == 0) {
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const int64_t partial_idx = ((row * num_q_heads + q_base + h) * num_splits) + split_idx;
        partial_m[partial_idx] = -3.4028234663852886e38f;
        partial_l[partial_idx] = 0.0f;
      }
    }
    if (static_cast<int64_t>(threadIdx.x) < head_dim) {
      const int64_t dim = threadIdx.x;
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const int64_t partial_idx = ((row * num_q_heads + q_base + h) * num_splits) + split_idx;
        partial_out[partial_idx * head_dim + dim] = 0.0f;
      }
    }
    return;
  }

  const int64_t token_end_raw = token_start + chunk_size;
  const int64_t token_end = (token_end_raw < context_len) ? token_end_raw : context_len;
  const int64_t chunk_len = token_end - token_start;

  extern __shared__ float shared[];
  float* scores = shared;                          // [8, chunk_size]
  const int num_warps = blockDim.x / warpSize;
  float* partial = scores + q_per_kv * chunk_size;  // [8, num_warps]
  float* q_shared = partial + q_per_kv * num_warps; // [8, 256]
  int32_t* token_offsets = reinterpret_cast<int32_t*>(q_shared + q_per_kv * head_dim);
  int32_t* split_meta = token_offsets + chunk_size;

  for (int64_t h = 0; h < q_per_kv; ++h) {
    q_shared[h * head_dim + threadIdx.x] = query[(row * num_q_heads + q_base + h) * head_dim + threadIdx.x];
  }
  if (threadIdx.x == 0) {
    const int64_t start_block = token_start >> 8;
    const int64_t blocks_in_chunk = (chunk_len + block_size - 1) >> 8;
    const int32_t base_physical_block = block_table[start_block];
    bool contiguous_blocks = true;
    for (int64_t b = 1; b < blocks_in_chunk; ++b) {
      if (block_table[start_block + b] != base_physical_block + static_cast<int32_t>(b)) {
        contiguous_blocks = false;
        break;
      }
    }
    split_meta[0] = contiguous_blocks ? 1 : 0;
    split_meta[1] = base_physical_block;
  }
  __syncthreads();

  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const bool contiguous_blocks = split_meta[0] != 0;
  const int64_t base_physical_block = static_cast<int64_t>(split_meta[1]);

  float local_max[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    local_max[h] = -3.4028234663852886e38f;
  }

  for (int64_t lt = warp_id; lt < chunk_len; lt += num_warps) {
    const int64_t token = token_start + lt;
    int64_t token_offset;
    if (contiguous_blocks) {
      token_offset = (base_physical_block << 8) + lt;
    } else {
      const int64_t logical_block = token >> 8;
      const int64_t block_offset = token & 255;
      const int64_t physical_block = static_cast<int64_t>(block_table[logical_block]);
      token_offset = (physical_block << 8) + block_offset;
    }
    const int64_t k_offset = (token_offset * num_kv_heads + kv_head) * head_dim;
    float acc[q_per_kv];
    for (int64_t h = 0; h < q_per_kv; ++h) {
      acc[h] = 0.0f;
    }
    for (int64_t dim = lane_id * 4; dim < head_dim; dim += warpSize * 4) {
      const float k0 = bf16_bits_to_float(key_cache[k_offset + dim + 0]);
      const float k1 = bf16_bits_to_float(key_cache[k_offset + dim + 1]);
      const float k2 = bf16_bits_to_float(key_cache[k_offset + dim + 2]);
      const float k3 = bf16_bits_to_float(key_cache[k_offset + dim + 3]);
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const float* q = q_shared + h * head_dim + dim;
        acc[h] += q[0] * k0;
        acc[h] += q[1] * k1;
        acc[h] += q[2] * k2;
        acc[h] += q[3] * k3;
      }
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      for (int64_t h = 0; h < q_per_kv; ++h) {
        acc[h] += __shfl_down(acc[h], offset);
      }
    }
    if (lane_id == 0) {
      token_offsets[lt] = static_cast<int32_t>(token_offset);
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const float score = acc[h] * scale;
        scores[h * chunk_size + lt] = score;
        local_max[h] = fmaxf(local_max[h], score);
      }
    }
  }
  if (lane_id == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      partial[h * num_warps + warp_id] = local_max[h];
    }
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      float warp_max = (threadIdx.x < num_warps) ? partial[h * num_warps + threadIdx.x]
                                                 : -3.4028234663852886e38f;
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_down(warp_max, offset));
      }
      if (threadIdx.x == 0) {
        partial[h * num_warps] = warp_max;
      }
    }
  }
  __syncthreads();

  float chunk_max[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    chunk_max[h] = partial[h * num_warps];
  }
  __syncthreads();

  float local_sum[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    local_sum[h] = 0.0f;
    for (int64_t lt = threadIdx.x; lt < chunk_len; lt += blockDim.x) {
      const float e = expf(scores[h * chunk_size + lt] - chunk_max[h]);
      scores[h * chunk_size + lt] = e;
      local_sum[h] += e;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      local_sum[h] += __shfl_down(local_sum[h], offset);
    }
  }
  if (lane_id == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      partial[h * num_warps + warp_id] = local_sum[h];
    }
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      float warp_sum = (threadIdx.x < num_warps) ? partial[h * num_warps + threadIdx.x] : 0.0f;
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        warp_sum += __shfl_down(warp_sum, offset);
      }
      if (threadIdx.x == 0) {
        partial[h * num_warps] = warp_sum;
      }
    }
  }
  __syncthreads();

  float chunk_sum[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    chunk_sum[h] = partial[h * num_warps];
  }
  __syncthreads();

  if (static_cast<int64_t>(threadIdx.x) < head_dim) {
    const int64_t dim = threadIdx.x;
    float acc[q_per_kv];
    for (int64_t h = 0; h < q_per_kv; ++h) {
      acc[h] = 0.0f;
    }
    if (contiguous_blocks) {
      const int64_t v_base = ((base_physical_block << 8) * num_kv_heads + kv_head) * head_dim + dim;
      constexpr int64_t v_stride = num_kv_heads * head_dim;
      for (int64_t lt = 0; lt < chunk_len; ++lt) {
        const float v = bf16_bits_to_float(value_cache[v_base + lt * v_stride]);
        for (int64_t h = 0; h < q_per_kv; ++h) {
          acc[h] += scores[h * chunk_size + lt] * v;
        }
      }
    } else {
      for (int64_t lt = 0; lt < chunk_len; ++lt) {
        const int64_t v_offset =
            (static_cast<int64_t>(token_offsets[lt]) * num_kv_heads + kv_head) * head_dim + dim;
        const float v = bf16_bits_to_float(value_cache[v_offset]);
        for (int64_t h = 0; h < q_per_kv; ++h) {
          acc[h] += scores[h * chunk_size + lt] * v;
        }
      }
    }
    for (int64_t h = 0; h < q_per_kv; ++h) {
      const int64_t partial_idx = ((row * num_q_heads + q_base + h) * num_splits) + split_idx;
      partial_out[partial_idx * head_dim + dim] = acc[h];
    }
  }

  if (threadIdx.x == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      const int64_t partial_idx = ((row * num_q_heads + q_base + h) * num_splits) + split_idx;
      partial_m[partial_idx] = chunk_max[h];
      partial_l[partial_idx] = chunk_sum[h];
    }
  }
}

template <int64_t q_per_kv, int64_t num_q_heads, int64_t num_kv_heads>
__global__ void qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_kernel(
    const float* __restrict__ query,         // [rows, num_q_heads, 256]
    const uint16_t* __restrict__ key_cache,  // [num_blocks, 256, num_kv_heads, 256]
    const uint16_t* __restrict__ value_cache,
    const half_t* __restrict__ gate,         // [rows, num_q_heads, 256] view-compatible
    half_t* __restrict__ out,                // [rows, num_q_heads * 256]
    const int32_t* __restrict__ block_tables,
    const int64_t* __restrict__ context_len_ptr,
    int64_t rows,
    int64_t chunk_size,
    int64_t block_table_len,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale) {
  constexpr int64_t head_dim = 256;
  constexpr int64_t block_size = 256;
  const int64_t kv_head = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (kv_head >= num_kv_heads || row >= rows) {
    return;
  }

  const int64_t context_len = context_len_ptr[row];
  const int64_t q_base = kv_head * q_per_kv;
  const int32_t* __restrict__ block_table = block_tables + row * block_table_len;

  if (context_len <= 0) {
    if (static_cast<int64_t>(threadIdx.x) < head_dim) {
      const int64_t dim = threadIdx.x;
      for (int64_t h = 0; h < q_per_kv; ++h) {
        out[(row * num_q_heads + q_base + h) * head_dim + dim] = static_cast<half_t>(0.0f);
      }
    }
    return;
  }

  const int64_t chunk_len = (context_len < chunk_size) ? context_len : chunk_size;

  extern __shared__ float shared[];
  float* scores = shared;                          // [8, chunk_size]
  const int num_warps = blockDim.x / warpSize;
  float* partial = scores + q_per_kv * chunk_size;  // [8, num_warps]
  float* q_shared = partial + q_per_kv * num_warps; // [8, 256]
  int32_t* token_offsets = reinterpret_cast<int32_t*>(q_shared + q_per_kv * head_dim);
  int32_t* split_meta = token_offsets + chunk_size;

  for (int64_t h = 0; h < q_per_kv; ++h) {
    q_shared[h * head_dim + threadIdx.x] = query[(row * num_q_heads + q_base + h) * head_dim + threadIdx.x];
  }
  if (threadIdx.x == 0) {
    const int64_t blocks_in_chunk = (chunk_len + block_size - 1) >> 8;
    const int32_t base_physical_block = block_table[0];
    bool contiguous_blocks = true;
    for (int64_t b = 1; b < blocks_in_chunk; ++b) {
      if (block_table[b] != base_physical_block + static_cast<int32_t>(b)) {
        contiguous_blocks = false;
        break;
      }
    }
    split_meta[0] = contiguous_blocks ? 1 : 0;
    split_meta[1] = base_physical_block;
  }
  __syncthreads();

  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const bool contiguous_blocks = split_meta[0] != 0;
  const int64_t base_physical_block = static_cast<int64_t>(split_meta[1]);

  float local_max[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    local_max[h] = -3.4028234663852886e38f;
  }

  for (int64_t lt = warp_id; lt < chunk_len; lt += num_warps) {
    int64_t token_offset;
    if (contiguous_blocks) {
      token_offset = (base_physical_block << 8) + lt;
    } else {
      const int64_t logical_block = lt >> 8;
      const int64_t block_offset = lt & 255;
      const int64_t physical_block = static_cast<int64_t>(block_table[logical_block]);
      token_offset = (physical_block << 8) + block_offset;
    }
    const int64_t k_offset = (token_offset * num_kv_heads + kv_head) * head_dim;
    float acc[q_per_kv];
    for (int64_t h = 0; h < q_per_kv; ++h) {
      acc[h] = 0.0f;
    }
    for (int64_t dim = lane_id * 4; dim < head_dim; dim += warpSize * 4) {
      const float k0 = bf16_bits_to_float(key_cache[k_offset + dim + 0]);
      const float k1 = bf16_bits_to_float(key_cache[k_offset + dim + 1]);
      const float k2 = bf16_bits_to_float(key_cache[k_offset + dim + 2]);
      const float k3 = bf16_bits_to_float(key_cache[k_offset + dim + 3]);
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const float* q = q_shared + h * head_dim + dim;
        acc[h] += q[0] * k0;
        acc[h] += q[1] * k1;
        acc[h] += q[2] * k2;
        acc[h] += q[3] * k3;
      }
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      for (int64_t h = 0; h < q_per_kv; ++h) {
        acc[h] += __shfl_down(acc[h], offset);
      }
    }
    if (lane_id == 0) {
      token_offsets[lt] = static_cast<int32_t>(token_offset);
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const float score = acc[h] * scale;
        scores[h * chunk_size + lt] = score;
        local_max[h] = fmaxf(local_max[h], score);
      }
    }
  }
  if (lane_id == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      partial[h * num_warps + warp_id] = local_max[h];
    }
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      float warp_max = (threadIdx.x < num_warps) ? partial[h * num_warps + threadIdx.x]
                                                 : -3.4028234663852886e38f;
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_down(warp_max, offset));
      }
      if (threadIdx.x == 0) {
        partial[h * num_warps] = warp_max;
      }
    }
  }
  __syncthreads();

  float chunk_max[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    chunk_max[h] = partial[h * num_warps];
  }
  __syncthreads();

  float local_sum[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    local_sum[h] = 0.0f;
    for (int64_t lt = threadIdx.x; lt < chunk_len; lt += blockDim.x) {
      const float e = expf(scores[h * chunk_size + lt] - chunk_max[h]);
      scores[h * chunk_size + lt] = e;
      local_sum[h] += e;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      local_sum[h] += __shfl_down(local_sum[h], offset);
    }
  }
  if (lane_id == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      partial[h * num_warps + warp_id] = local_sum[h];
    }
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      float warp_sum = (threadIdx.x < num_warps) ? partial[h * num_warps + threadIdx.x] : 0.0f;
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        warp_sum += __shfl_down(warp_sum, offset);
      }
      if (threadIdx.x == 0) {
        partial[h * num_warps] = warp_sum;
      }
    }
  }
  __syncthreads();

  float chunk_sum[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    chunk_sum[h] = partial[h * num_warps];
  }
  __syncthreads();

  if (static_cast<int64_t>(threadIdx.x) < head_dim) {
    const int64_t dim = threadIdx.x;
    float acc[q_per_kv];
    for (int64_t h = 0; h < q_per_kv; ++h) {
      acc[h] = 0.0f;
    }
    if (contiguous_blocks) {
      const int64_t v_base = ((base_physical_block << 8) * num_kv_heads + kv_head) * head_dim + dim;
      constexpr int64_t v_stride = num_kv_heads * head_dim;
      for (int64_t lt = 0; lt < chunk_len; ++lt) {
        const float v = bf16_bits_to_float(value_cache[v_base + lt * v_stride]);
        for (int64_t h = 0; h < q_per_kv; ++h) {
          acc[h] += scores[h * chunk_size + lt] * v;
        }
      }
    } else {
      for (int64_t lt = 0; lt < chunk_len; ++lt) {
        const int64_t v_offset =
            (static_cast<int64_t>(token_offsets[lt]) * num_kv_heads + kv_head) * head_dim + dim;
        const float v = bf16_bits_to_float(value_cache[v_offset]);
        for (int64_t h = 0; h < q_per_kv; ++h) {
          acc[h] += scores[h * chunk_size + lt] * v;
        }
      }
    }
    for (int64_t h = 0; h < q_per_kv; ++h) {
      const int64_t q_head = q_base + h;
      const int64_t gate_idx = (row * num_q_heads + q_head) * gate_stride1 + dim * gate_stride2;
      const float gate_v = scalar_to_float_qwen35(gate[gate_idx]);
      const float inv_l = 1.0f / fmaxf(chunk_sum[h], 1.0e-20f);
      out[(row * num_q_heads + q_head) * head_dim + dim] =
          static_cast<half_t>((acc[h] * inv_l) * sigmoid_f32(gate_v));
    }
  }
}


__global__ void qwen35_paged_full_attn_decode_split_k_reduce_kernel(
    const float* __restrict__ partial_out,   // [num_q_heads, num_splits, head_dim]
    const float* __restrict__ partial_m,     // [num_q_heads, num_splits]
    const float* __restrict__ partial_l,     // [num_q_heads, num_splits]
    float* __restrict__ out,                 // [1, num_q_heads, head_dim]
    int64_t num_splits,
    int64_t head_dim) {
  const int64_t q_head = blockIdx.x;

  extern __shared__ float sred[];
  float* sm = sred;                    // [num_splits]
  float* sl = sred + num_splits;       // [num_splits]
  float* sw = sl + num_splits;         // [num_splits], normalized split weights

  // Cooperatively load m and l partials into LDS.
  for (int64_t i = threadIdx.x; i < num_splits; i += blockDim.x) {
    sm[i] = partial_m[q_head * num_splits + i];
    sl[i] = partial_l[q_head * num_splits + i];
  }
  __syncthreads();

  // Each thread handles one output dim.
  if (static_cast<int64_t>(threadIdx.x) >= head_dim) {
    return;
  }
  const int64_t dim = threadIdx.x;

  // Compute the split rescale weights once per head. The previous version
  // recomputed exp(m_i - m_global) for every output dimension.
  if (threadIdx.x == 0) {
    float m_global = -3.4028234663852886e38f;
    for (int64_t s = 0; s < num_splits; ++s) {
      m_global = fmaxf(m_global, sm[s]);
    }
    float l_global = 0.0f;
    for (int64_t s = 0; s < num_splits; ++s) {
      const float li = sl[s];
      const float weight = (li > 0.0f) ? expf(sm[s] - m_global) : 0.0f;
      sw[s] = weight;
      l_global += li * weight;
    }
    const float inv_l = 1.0f / fmaxf(l_global, 1.0e-20f);
    for (int64_t s = 0; s < num_splits; ++s) {
      sw[s] *= inv_l;
    }
  }
  __syncthreads();

  // o_global = (sum_i o_i * exp(m_i - m_global)) / l_global
  float acc = 0.0f;
  for (int64_t s = 0; s < num_splits; ++s) {
    acc += partial_out[(q_head * num_splits + s) * head_dim + dim] * sw[s];
  }
  out[q_head * head_dim + dim] = acc;
}


template <bool SHORT_BLOCK256>
__global__ void qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel(
    const float* __restrict__ query,       // [rows, num_q_heads, head_dim]
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    const half_t* __restrict__ gate,       // [rows, num_q_heads, head_dim] view-compatible
    half_t* __restrict__ out,              // [rows, num_q_heads * head_dim]
    const int32_t* __restrict__ block_tables,
    const int64_t* __restrict__ context_counts,
    const int64_t* __restrict__ row_positions,
    int64_t rows,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale) {
  const int64_t q_head = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (q_head >= num_q_heads || row >= rows) {
    return;
  }
  int64_t context_len = context_counts[row];
  if (context_len <= 0) {
    return;
  }
  context_len = context_len > max_context_len ? max_context_len : context_len;
  const int64_t causal_position = row_positions ? row_positions[row] : (context_counts[row] - 1);
  if (causal_position < 0) {
    return;
  }
  const int64_t causal_len = causal_position + 1;
  const int64_t visible_len = context_len < causal_len ? context_len : causal_len;
  if (visible_len <= 0) {
    return;
  }

  extern __shared__ float shared[];
  float* scores = shared;
  float* partial = scores + max_context_len;
  float* q_shared = partial + blockDim.x;
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const float* q = query + (row * num_q_heads + q_head) * head_dim;
  const int32_t* row_table = block_tables + row * block_table_len;
  const bool block_size_256 = block_size == 256;
  int64_t short_physical_block0 = -1;
  int64_t short_physical_block1 = -1;
  if constexpr (SHORT_BLOCK256) {
    short_physical_block0 = block_table_len > 0 ? static_cast<int64_t>(row_table[0]) : -1;
    short_physical_block1 = block_table_len > 1 ? static_cast<int64_t>(row_table[1]) : -1;
  }

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    q_shared[dim] = bf16_bits_to_float(float_to_bf16_bits(q[dim]));
  }
  __syncthreads();

  float local_max = -3.4028234663852886e38f;
  for (int64_t token = threadIdx.x; token < visible_len; token += blockDim.x) {
    int64_t block_offset;
    int64_t physical_block;
    if constexpr (SHORT_BLOCK256) {
      const int64_t logical_block = token >> 8;
      block_offset = token & 255;
      physical_block = (logical_block == 0) ? short_physical_block0 : short_physical_block1;
    } else {
      const int64_t logical_block = block_size_256 ? (token >> 8) : (token / block_size);
      if (logical_block >= block_table_len) {
        scores[token] = -3.4028234663852886e38f;
        continue;
      }
      block_offset = block_size_256 ? (token & 255) : (token - logical_block * block_size);
      physical_block = static_cast<int64_t>(row_table[logical_block]);
    }
    if (physical_block < 0) {
      scores[token] = -3.4028234663852886e38f;
      continue;
    }
    const int64_t k_offset = ((physical_block * block_size + block_offset) * num_kv_heads + kv_head) * head_dim;
    float acc = 0.0f;
    if ((head_dim & 7) == 0) {
      acc = paged_key_dot_vec8(q_shared, key_cache, k_offset, head_dim);
    } else {
      for (int64_t dim = 0; dim < head_dim; ++dim) {
        acc += q_shared[dim] * bf16_bits_to_float(key_cache[k_offset + dim]);
      }
    }
    const float score = acc * scale;
    scores[token] = score;
    local_max = fmaxf(local_max, score);
  }
  partial[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_score = partial[0];

  float local_sum = 0.0f;
  for (int64_t token = threadIdx.x; token < visible_len; token += blockDim.x) {
    const float value = expf(scores[token] - max_score);
    scores[token] = value;
    local_sum += value;
  }
  partial[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_denom = 1.0f / fmaxf(partial[0], 1.0e-20f);

  for (int64_t token = threadIdx.x; token < visible_len; token += blockDim.x) {
    scores[token] *= inv_denom;
  }
  __syncthreads();

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < visible_len; ++token) {
      int64_t block_offset;
      int64_t physical_block;
      if constexpr (SHORT_BLOCK256) {
        const int64_t logical_block = token >> 8;
        block_offset = token & 255;
        physical_block = (logical_block == 0) ? short_physical_block0 : short_physical_block1;
      } else {
        const int64_t logical_block = block_size_256 ? (token >> 8) : (token / block_size);
        if (logical_block >= block_table_len) {
          continue;
        }
        block_offset = block_size_256 ? (token & 255) : (token - logical_block * block_size);
        physical_block = static_cast<int64_t>(row_table[logical_block]);
      }
      if (physical_block < 0) {
        continue;
      }
      const int64_t v_offset = ((physical_block * block_size + block_offset) * num_kv_heads + kv_head) * head_dim + dim;
      acc += scores[token] * bf16_bits_to_float(value_cache[v_offset]);
    }
    const int64_t gate_idx = (row * num_q_heads + q_head) * gate_stride1 + dim * gate_stride2;
    const float gate_v = static_cast<float>(gate[gate_idx]);
    const float attn_bf16 = bf16_bits_to_float(float_to_bf16_bits(acc));
    out[(row * num_q_heads + q_head) * head_dim + dim] = static_cast<half_t>(attn_bf16 * sigmoid_f32(gate_v));
  }
}

template <typename scale_t>
__global__ void qwen35_paged_full_attn_prefill_gqa_gate_int8_kernel(
    const float* __restrict__ query,       // [rows, num_q_heads, head_dim]
    const int8_t* __restrict__ key_cache,
    const int8_t* __restrict__ value_cache,
    const scale_t* __restrict__ k_scale,   // [num_blocks, block_size, num_kv_heads]
    const scale_t* __restrict__ v_scale,
    const half_t* __restrict__ gate,       // [rows, num_q_heads, head_dim] view-compatible
    half_t* __restrict__ out,              // [rows, num_q_heads * head_dim]
    const int32_t* __restrict__ block_tables,
    const int64_t* __restrict__ context_counts,
    const int64_t* __restrict__ row_positions,
    int64_t rows,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale) {
  const int64_t q_head = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (q_head >= num_q_heads || row >= rows || head_dim > static_cast<int64_t>(blockDim.x)) {
    return;
  }
  int64_t context_len = context_counts[row];
  if (context_len <= 0) {
    return;
  }
  context_len = context_len > max_context_len ? max_context_len : context_len;
  const int64_t causal_position = row_positions ? row_positions[row] : (context_counts[row] - 1);
  if (causal_position < 0) {
    return;
  }
  const int64_t causal_len = causal_position + 1;
  const int64_t visible_len = context_len < causal_len ? context_len : causal_len;
  if (visible_len <= 0) {
    return;
  }

  extern __shared__ float shared[];
  float* partial = shared;                  // [threads]
  float* q_shared = partial + blockDim.x;    // [head_dim <= threads]
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const float* q = query + (row * num_q_heads + q_head) * head_dim;
  const int32_t* row_table = block_tables + row * block_table_len;
  const int64_t tid = static_cast<int64_t>(threadIdx.x);

  if (tid < head_dim) {
    q_shared[tid] = bf16_bits_to_float(float_to_bf16_bits(q[tid]));
  }
  __syncthreads();

  float m = -3.4028234663852886e38f;
  float l = 0.0f;
  float out_acc = 0.0f;

  for (int64_t token = 0; token < visible_len; ++token) {
    const int64_t logical_block = token / block_size;
    int64_t block_offset = 0;
    int64_t physical_block = -1;
    bool valid_token = logical_block < block_table_len;
    if (valid_token) {
      block_offset = token - logical_block * block_size;
      physical_block = static_cast<int64_t>(row_table[logical_block]);
      valid_token = physical_block >= 0;
    }
    const int64_t token_offset = (physical_block * block_size + block_offset);
    const int64_t scale_offset = (token_offset * num_kv_heads + kv_head);
    const float key_s = valid_token ? scalar_to_float_qwen35(k_scale[scale_offset]) : 0.0f;
    const int64_t k_offset = (token_offset * num_kv_heads + kv_head) * head_dim;

    float local_dot = 0.0f;
    if (valid_token) {
      for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
        local_dot += q_shared[dim] * (static_cast<float>(key_cache[k_offset + dim]) * key_s);
      }
    }
    partial[tid] = local_dot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }

    const float score = valid_token ? (partial[0] * scale) : -3.4028234663852886e38f;
    const float new_m = fmaxf(m, score);
    const float old_weight = (m <= -3.0e38f) ? 0.0f : expf(m - new_m);
    const float token_weight = (score <= -3.0e38f) ? 0.0f : expf(score - new_m);
    if (tid < head_dim) {
      const int64_t v_offset = k_offset + tid;
      const float value_s = valid_token ? scalar_to_float_qwen35(v_scale[scale_offset]) : 0.0f;
      const float value = valid_token ? (static_cast<float>(value_cache[v_offset]) * value_s) : 0.0f;
      out_acc = out_acc * old_weight + token_weight * value;
    }
    l = l * old_weight + token_weight;
    m = new_m;
    __syncthreads();
  }

  if (tid < head_dim) {
    const float inv_l = 1.0f / fmaxf(l, 1.0e-20f);
    const int64_t gate_idx = (row * num_q_heads + q_head) * gate_stride1 + tid * gate_stride2;
    const float gate_v = static_cast<float>(gate[gate_idx]);
    const float attn_bf16 = bf16_bits_to_float(float_to_bf16_bits(out_acc * inv_l));
    out[(row * num_q_heads + q_head) * head_dim + tid] = static_cast<half_t>(attn_bf16 * sigmoid_f32(gate_v));
  }
}

// Tree-aware variant of qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel.
// Identical to the chain kernel except for the per-row ancestor mask applied
// to verifier rows whose K/V live at global positions [tree_committed_count,
// tree_committed_count + rows).  Positions strictly below tree_committed_count
// are committed context shared by all verifier rows; the mask only filters
// the verifier K/V block.  The mask layout is [rows, rows] uint8 with
// ancestor_mask[row * rows + j] == 1 iff verifier row `j` is an ancestor of
// verifier row `row` in the tree (a row is its own ancestor).
template <bool SHORT_BLOCK256>
__global__ void qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_kernel(
    const float* __restrict__ query,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    const half_t* __restrict__ gate,
    half_t* __restrict__ out,
    const int32_t* __restrict__ block_tables,
    const int64_t* __restrict__ context_counts,
    const int64_t* __restrict__ row_positions,
    const uint8_t* __restrict__ ancestor_mask,
    const int64_t* __restrict__ tree_committed_count_ptr,
    int64_t rows,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale) {
  const int64_t q_head = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (q_head >= num_q_heads || row >= rows) {
    return;
  }
  // Device-resident so HIP-graph replays see the per-cycle committed count
  // (#107: a by-value count is frozen at capture).
  const int64_t tree_committed_count = tree_committed_count_ptr[0];
  int64_t context_len = context_counts[row];
  if (context_len <= 0) {
    return;
  }
  context_len = context_len > max_context_len ? max_context_len : context_len;
  const int64_t causal_position = row_positions ? row_positions[row] : (context_counts[row] - 1);
  if (causal_position < 0) {
    return;
  }
  const int64_t causal_len = causal_position + 1;
  const int64_t visible_len = context_len < causal_len ? context_len : causal_len;
  if (visible_len <= 0) {
    return;
  }

  extern __shared__ float shared[];
  float* scores = shared;
  float* partial = scores + max_context_len;
  float* q_shared = partial + blockDim.x;
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const float* q = query + (row * num_q_heads + q_head) * head_dim;
  const int32_t* row_table = block_tables + row * block_table_len;
  const uint8_t* row_mask = ancestor_mask + row * rows;
  const bool block_size_256 = block_size == 256;
  int64_t short_physical_block0 = -1;
  int64_t short_physical_block1 = -1;
  if constexpr (SHORT_BLOCK256) {
    short_physical_block0 = block_table_len > 0 ? static_cast<int64_t>(row_table[0]) : -1;
    short_physical_block1 = block_table_len > 1 ? static_cast<int64_t>(row_table[1]) : -1;
  }

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    q_shared[dim] = bf16_bits_to_float(float_to_bf16_bits(q[dim]));
  }
  __syncthreads();

  float local_max = -3.4028234663852886e38f;
  for (int64_t token = threadIdx.x; token < visible_len; token += blockDim.x) {
    // Tree mask: only the verifier-row segment [tree_committed_count,
    // tree_committed_count + rows) is filtered.  Committed context positions
    // [0, tree_committed_count) are visible to every row.
    if (token >= tree_committed_count) {
      const int64_t row_offset = token - tree_committed_count;
      if (row_offset >= rows || row_mask[row_offset] == 0) {
        scores[token] = -3.4028234663852886e38f;
        continue;
      }
    }
    int64_t block_offset;
    int64_t physical_block;
    if constexpr (SHORT_BLOCK256) {
      const int64_t logical_block = token >> 8;
      block_offset = token & 255;
      physical_block = (logical_block == 0) ? short_physical_block0 : short_physical_block1;
    } else {
      const int64_t logical_block = block_size_256 ? (token >> 8) : (token / block_size);
      if (logical_block >= block_table_len) {
        scores[token] = -3.4028234663852886e38f;
        continue;
      }
      block_offset = block_size_256 ? (token & 255) : (token - logical_block * block_size);
      physical_block = static_cast<int64_t>(row_table[logical_block]);
    }
    if (physical_block < 0) {
      scores[token] = -3.4028234663852886e38f;
      continue;
    }
    const int64_t k_offset = ((physical_block * block_size + block_offset) * num_kv_heads + kv_head) * head_dim;
    float acc = 0.0f;
    if ((head_dim & 7) == 0) {
      acc = paged_key_dot_vec8(q_shared, key_cache, k_offset, head_dim);
    } else {
      for (int64_t dim = 0; dim < head_dim; ++dim) {
        acc += q_shared[dim] * bf16_bits_to_float(key_cache[k_offset + dim]);
      }
    }
    const float score = acc * scale;
    scores[token] = score;
    local_max = fmaxf(local_max, score);
  }
  partial[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_score = partial[0];

  float local_sum = 0.0f;
  for (int64_t token = threadIdx.x; token < visible_len; token += blockDim.x) {
    const float value = expf(scores[token] - max_score);
    scores[token] = value;
    local_sum += value;
  }
  partial[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_denom = 1.0f / fmaxf(partial[0], 1.0e-20f);

  for (int64_t token = threadIdx.x; token < visible_len; token += blockDim.x) {
    scores[token] *= inv_denom;
  }
  __syncthreads();

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < visible_len; ++token) {
      // Masked positions already have scores[token] == 0 after the softmax
      // (exp(-INF) == 0), so they contribute nothing to the V sum.  Skip the
      // V load when the mask is zero to avoid touching cache lines that may
      // belong to sibling-tree rows.
      if (token >= tree_committed_count) {
        const int64_t row_offset = token - tree_committed_count;
        if (row_offset >= rows || row_mask[row_offset] == 0) {
          continue;
        }
      }
      int64_t block_offset;
      int64_t physical_block;
      if constexpr (SHORT_BLOCK256) {
        const int64_t logical_block = token >> 8;
        block_offset = token & 255;
        physical_block = (logical_block == 0) ? short_physical_block0 : short_physical_block1;
      } else {
        const int64_t logical_block = block_size_256 ? (token >> 8) : (token / block_size);
        if (logical_block >= block_table_len) {
          continue;
        }
        block_offset = block_size_256 ? (token & 255) : (token - logical_block * block_size);
        physical_block = static_cast<int64_t>(row_table[logical_block]);
      }
      if (physical_block < 0) {
        continue;
      }
      const int64_t v_offset = ((physical_block * block_size + block_offset) * num_kv_heads + kv_head) * head_dim + dim;
      acc += scores[token] * bf16_bits_to_float(value_cache[v_offset]);
    }
    const int64_t gate_idx = (row * num_q_heads + q_head) * gate_stride1 + dim * gate_stride2;
    const float gate_v = static_cast<float>(gate[gate_idx]);
    const float attn_bf16 = bf16_bits_to_float(float_to_bf16_bits(acc));
    out[(row * num_q_heads + q_head) * head_dim + dim] = static_cast<half_t>(attn_bf16 * sigmoid_f32(gate_v));
  }
}

__global__ void qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_kernel(
    const float* __restrict__ query,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    const half_t* __restrict__ gate,
    half_t* __restrict__ out,
    const int32_t* __restrict__ block_tables,
    const int64_t* __restrict__ context_counts,
    const int64_t* __restrict__ row_positions,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ cu_seqlens_k,
    int64_t rows,
    int64_t segments,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale) {
  const int64_t q_head = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (q_head >= num_q_heads || row >= rows) {
    return;
  }

  int64_t segment = -1;
  for (int64_t idx = 0; idx < segments; ++idx) {
    const int64_t start = static_cast<int64_t>(cu_seqlens_q[idx]);
    const int64_t end = static_cast<int64_t>(cu_seqlens_q[idx + 1]);
    if (row >= start && row < end) {
      segment = idx;
      break;
    }
  }
  if (segment < 0) {
    return;
  }
  const int64_t q_start = static_cast<int64_t>(cu_seqlens_q[segment]);
  const int64_t k_prev = static_cast<int64_t>(cu_seqlens_k[segment]);
  const int64_t k_end = static_cast<int64_t>(cu_seqlens_k[segment + 1]);
  const int64_t k_len = k_end - k_prev;
  if (k_len <= 0) {
    return;
  }

  int64_t context_len = context_counts[row];
  if (context_len <= 0) {
    return;
  }
  context_len = context_len > max_context_len ? max_context_len : context_len;
  const int64_t causal_position = row_positions ? row_positions[row] : (context_counts[row] - 1);
  if (causal_position < 0) {
    return;
  }
  const int64_t segment_position_start = row_positions ? row_positions[q_start] : 0;
  const int64_t segment_visible_limit = segment_position_start + k_len;
  int64_t visible_len = context_len < (causal_position + 1) ? context_len : (causal_position + 1);
  visible_len = visible_len < segment_visible_limit ? visible_len : segment_visible_limit;
  if (visible_len <= 0) {
    return;
  }

  extern __shared__ float shared[];
  float* scores = shared;
  float* partial = scores + max_context_len;
  float* q_shared = partial + blockDim.x;
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const float* q = query + (row * num_q_heads + q_head) * head_dim;
  const int32_t* row_table = block_tables + row * block_table_len;

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    q_shared[dim] = bf16_bits_to_float(float_to_bf16_bits(q[dim]));
  }
  __syncthreads();

  float local_max = -3.4028234663852886e38f;
  for (int64_t token = threadIdx.x; token < visible_len; token += blockDim.x) {
    const int64_t logical_block = token / block_size;
    if (logical_block >= block_table_len) {
      scores[token] = -3.4028234663852886e38f;
      continue;
    }
    const int64_t block_offset = token - logical_block * block_size;
    const int64_t physical_block = static_cast<int64_t>(row_table[logical_block]);
    if (physical_block < 0) {
      scores[token] = -3.4028234663852886e38f;
      continue;
    }
    const int64_t k_offset = ((physical_block * block_size + block_offset) * num_kv_heads + kv_head) * head_dim;
    float acc = 0.0f;
    if ((head_dim & 7) == 0) {
      acc = paged_key_dot_vec8(q_shared, key_cache, k_offset, head_dim);
    } else {
      for (int64_t dim = 0; dim < head_dim; ++dim) {
        acc += q_shared[dim] * bf16_bits_to_float(key_cache[k_offset + dim]);
      }
    }
    const float score = acc * scale;
    scores[token] = score;
    local_max = fmaxf(local_max, score);
  }
  partial[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_score = partial[0];

  float local_sum = 0.0f;
  for (int64_t token = threadIdx.x; token < visible_len; token += blockDim.x) {
    const float value = expf(scores[token] - max_score);
    scores[token] = value;
    local_sum += value;
  }
  partial[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_denom = 1.0f / fmaxf(partial[0], 1.0e-20f);

  for (int64_t token = threadIdx.x; token < visible_len; token += blockDim.x) {
    scores[token] *= inv_denom;
  }
  __syncthreads();

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < visible_len; ++token) {
      const int64_t logical_block = token / block_size;
      if (logical_block >= block_table_len) {
        continue;
      }
      const int64_t block_offset = token - logical_block * block_size;
      const int64_t physical_block = static_cast<int64_t>(row_table[logical_block]);
      if (physical_block < 0) {
        continue;
      }
      const int64_t v_offset = ((physical_block * block_size + block_offset) * num_kv_heads + kv_head) * head_dim + dim;
      acc += scores[token] * bf16_bits_to_float(value_cache[v_offset]);
    }
    const int64_t gate_idx = (row * num_q_heads + q_head) * gate_stride1 + dim * gate_stride2;
    const float gate_v = static_cast<float>(gate[gate_idx]);
    const float attn_bf16 = bf16_bits_to_float(float_to_bf16_bits(acc));
    out[(row * num_q_heads + q_head) * head_dim + dim] = static_cast<half_t>(attn_bf16 * sigmoid_f32(gate_v));
  }
}


template <typename scalar_t>
__global__ void qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel(
    const float* __restrict__ partial_out,   // [num_q_heads, num_splits, head_dim]
    const float* __restrict__ partial_m,     // [num_q_heads, num_splits]
    const float* __restrict__ partial_l,     // [num_q_heads, num_splits]
    const scalar_t* __restrict__ gate,       // [1, num_q_heads, head_dim] view-compatible
    scalar_t* __restrict__ out,              // [1, num_q_heads * head_dim]
    int64_t num_splits,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2) {
  const int64_t q_head = blockIdx.x;

  extern __shared__ float sred[];
  float* sm = sred;
  float* sl = sred + num_splits;
  float* sw = sl + num_splits;

  for (int64_t i = threadIdx.x; i < num_splits; i += blockDim.x) {
    sm[i] = partial_m[q_head * num_splits + i];
    sl[i] = partial_l[q_head * num_splits + i];
  }
  __syncthreads();

  if (static_cast<int64_t>(threadIdx.x) >= head_dim) {
    return;
  }
  const int64_t dim = threadIdx.x;

  if (threadIdx.x == 0) {
    float m_global = -3.4028234663852886e38f;
    for (int64_t s = 0; s < num_splits; ++s) {
      m_global = fmaxf(m_global, sm[s]);
    }
    float l_global = 0.0f;
    for (int64_t s = 0; s < num_splits; ++s) {
      const float li = sl[s];
      const float weight = (li > 0.0f) ? expf(sm[s] - m_global) : 0.0f;
      sw[s] = weight;
      l_global += li * weight;
    }
    const float inv_l = 1.0f / fmaxf(l_global, 1.0e-20f);
    for (int64_t s = 0; s < num_splits; ++s) {
      sw[s] *= inv_l;
    }
  }
  __syncthreads();

  float acc = 0.0f;
  for (int64_t s = 0; s < num_splits; ++s) {
    acc += partial_out[(q_head * num_splits + s) * head_dim + dim] * sw[s];
  }

  const int64_t gate_idx = q_head * gate_stride1 + dim * gate_stride2;
  const float gate_v = scalar_to_float_qwen35(gate[gate_idx]);
  out[q_head * head_dim + dim] = static_cast<scalar_t>(acc * sigmoid_f32(gate_v));
}


template <typename scalar_t>
__global__ void qwen35_paged_full_attn_decode_split_k_reduce_gate_batch_kernel(
    const float* __restrict__ partial_out,   // [rows, num_q_heads, num_splits, head_dim]
    const float* __restrict__ partial_m,     // [rows, num_q_heads, num_splits]
    const float* __restrict__ partial_l,     // [rows, num_q_heads, num_splits]
    const scalar_t* __restrict__ gate,       // [rows, num_q_heads, head_dim] view-compatible
    scalar_t* __restrict__ out,              // [rows, num_q_heads * head_dim]
    int64_t rows,
    int64_t num_splits,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2) {
  const int64_t q_head = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (row >= rows) {
    return;
  }

  extern __shared__ float sred[];
  float* sm = sred;
  float* sl = sred + num_splits;
  float* sw = sl + num_splits;
  const int64_t partial_base = (row * gridDim.x + q_head) * num_splits;

  for (int64_t i = threadIdx.x; i < num_splits; i += blockDim.x) {
    sm[i] = partial_m[partial_base + i];
    sl[i] = partial_l[partial_base + i];
  }
  __syncthreads();

  if (static_cast<int64_t>(threadIdx.x) >= head_dim) {
    return;
  }
  const int64_t dim = threadIdx.x;

  if (threadIdx.x == 0) {
    float m_global = -3.4028234663852886e38f;
    for (int64_t s = 0; s < num_splits; ++s) {
      m_global = fmaxf(m_global, sm[s]);
    }
    float l_global = 0.0f;
    for (int64_t s = 0; s < num_splits; ++s) {
      const float li = sl[s];
      const float weight = (li > 0.0f) ? expf(sm[s] - m_global) : 0.0f;
      sw[s] = weight;
      l_global += li * weight;
    }
    const float inv_l = 1.0f / fmaxf(l_global, 1.0e-20f);
    for (int64_t s = 0; s < num_splits; ++s) {
      sw[s] *= inv_l;
    }
  }
  __syncthreads();

  float acc = 0.0f;
  for (int64_t s = 0; s < num_splits; ++s) {
    acc += partial_out[(partial_base + s) * head_dim + dim] * sw[s];
  }

  const int64_t gate_idx = (row * gridDim.x + q_head) * gate_stride1 + dim * gate_stride2;
  const float gate_v = scalar_to_float_qwen35(gate[gate_idx]);
  out[(row * gridDim.x + q_head) * head_dim + dim] = static_cast<scalar_t>(acc * sigmoid_f32(gate_v));
}


bool valid_dense_attn_shape(
    int64_t max_context_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  return max_context_len > 0 && num_q_heads > 0 && num_kv_heads > 0 &&
      (num_q_heads % num_kv_heads) == 0 && head_dim > 0 && head_dim <= 256;
}

bool valid_paged_attn_shape(
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  return max_context_len > 0 && block_size == 256 && block_table_len > 0 &&
      num_q_heads > 0 && num_kv_heads > 0 && (num_q_heads % num_kv_heads) == 0 &&
      head_dim > 0 && head_dim <= 256 && ((max_context_len + block_size - 1) / block_size) <= block_table_len;
}


__global__ void qwen35_full_attn_gate_mul_bf16_to_bf16_kernel(
    const uint16_t* attn,
    const uint16_t* gate,
    uint16_t* out,
    int64_t total) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float gated = bf16_bits_to_float(attn[idx]) * sigmoid_f32(bf16_bits_to_float(gate[idx]));
    out[idx] = float_to_bf16_bits(gated);
  }
}

template <typename scale_t, int64_t q_per_kv, int64_t num_q_heads, int64_t num_kv_heads>
__global__ void qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_kernel(
    const float* __restrict__ query,         // [1, num_q_heads, 256]
    const int8_t* __restrict__ key_cache,    // [num_blocks, 256, num_kv_heads, 256]
    const int8_t* __restrict__ value_cache,
    const scale_t* __restrict__ k_scale,     // [num_blocks, 256, num_kv_heads]
    const scale_t* __restrict__ v_scale,
    const int32_t* __restrict__ block_table,
    const int64_t* __restrict__ context_len_ptr,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int64_t chunk_size,
    int64_t num_splits,
    float scale) {
  constexpr int64_t head_dim = 256;
  constexpr int64_t block_size = 256;
  const int64_t kv_head = blockIdx.x;
  const int64_t split_idx = blockIdx.y;
  if (kv_head >= num_kv_heads || split_idx >= num_splits) {
    return;
  }

  const int64_t context_len = *context_len_ptr;
  const int64_t token_start = split_idx * chunk_size;
  const int64_t q_base = kv_head * q_per_kv;

  if (token_start >= context_len) {
    if (threadIdx.x == 0) {
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const int64_t partial_idx = (q_base + h) * num_splits + split_idx;
        partial_m[partial_idx] = -3.4028234663852886e38f;
        partial_l[partial_idx] = 0.0f;
      }
    }
    if (static_cast<int64_t>(threadIdx.x) < head_dim) {
      const int64_t dim = threadIdx.x;
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const int64_t partial_idx = (q_base + h) * num_splits + split_idx;
        partial_out[partial_idx * head_dim + dim] = 0.0f;
      }
    }
    return;
  }

  const int64_t token_end_raw = token_start + chunk_size;
  const int64_t token_end = (token_end_raw < context_len) ? token_end_raw : context_len;
  const int64_t chunk_len = token_end - token_start;

  extern __shared__ float shared[];
  float* scores = shared;                          // [q_per_kv, chunk_size]
  const int num_warps = blockDim.x / warpSize;
  float* partial = scores + q_per_kv * chunk_size;  // [q_per_kv, num_warps]
  float* q_shared = partial + q_per_kv * num_warps; // [q_per_kv, 256]
  int32_t* token_offsets = reinterpret_cast<int32_t*>(q_shared + q_per_kv * head_dim);
  int32_t* split_meta = token_offsets + chunk_size;

  for (int64_t h = 0; h < q_per_kv; ++h) {
    q_shared[h * head_dim + threadIdx.x] = query[(q_base + h) * head_dim + threadIdx.x];
  }
  if (threadIdx.x == 0) {
    const int64_t start_block = token_start >> 8;
    const int64_t start_block_offset = token_start & 255;
    const int64_t blocks_in_chunk = (start_block_offset + chunk_len + block_size - 1) >> 8;
    const int32_t base_physical_block = block_table[start_block];
    bool contiguous_blocks = true;
    for (int64_t b = 1; b < blocks_in_chunk; ++b) {
      if (block_table[start_block + b] != base_physical_block + static_cast<int32_t>(b)) {
        contiguous_blocks = false;
        break;
      }
    }
    split_meta[0] = contiguous_blocks ? 1 : 0;
    split_meta[1] = base_physical_block;
  }
  __syncthreads();

  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  const bool contiguous_blocks = split_meta[0] != 0;
  const int64_t base_physical_block = static_cast<int64_t>(split_meta[1]);
  const int64_t contiguous_token_base = (token_start >> 8) << 8;

  float local_max[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    local_max[h] = -3.4028234663852886e38f;
  }

  for (int64_t lt = warp_id; lt < chunk_len; lt += num_warps) {
    const int64_t token = token_start + lt;
    int64_t token_offset;
    if (contiguous_blocks) {
      token_offset = (base_physical_block << 8) + (token - contiguous_token_base);
    } else {
      const int64_t logical_block = token >> 8;
      const int64_t block_offset = token & 255;
      const int64_t physical_block = static_cast<int64_t>(block_table[logical_block]);
      token_offset = (physical_block << 8) + block_offset;
    }
    const int64_t scale_offset = token_offset * num_kv_heads + kv_head;
    const float key_s = scalar_to_float_qwen35(k_scale[scale_offset]);
    const int64_t k_offset = (token_offset * num_kv_heads + kv_head) * head_dim;
    float acc[q_per_kv];
    for (int64_t h = 0; h < q_per_kv; ++h) {
      acc[h] = 0.0f;
    }
    for (int64_t dim = lane_id * 4; dim < head_dim; dim += warpSize * 4) {
      const float k0 = static_cast<float>(key_cache[k_offset + dim + 0]) * key_s;
      const float k1 = static_cast<float>(key_cache[k_offset + dim + 1]) * key_s;
      const float k2 = static_cast<float>(key_cache[k_offset + dim + 2]) * key_s;
      const float k3 = static_cast<float>(key_cache[k_offset + dim + 3]) * key_s;
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const float* q = q_shared + h * head_dim + dim;
        acc[h] += q[0] * k0;
        acc[h] += q[1] * k1;
        acc[h] += q[2] * k2;
        acc[h] += q[3] * k3;
      }
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      for (int64_t h = 0; h < q_per_kv; ++h) {
        acc[h] += __shfl_down(acc[h], offset);
      }
    }
    if (lane_id == 0) {
      token_offsets[lt] = static_cast<int32_t>(token_offset);
      for (int64_t h = 0; h < q_per_kv; ++h) {
        const float score = acc[h] * scale;
        scores[h * chunk_size + lt] = score;
        local_max[h] = fmaxf(local_max[h], score);
      }
    }
  }
  if (lane_id == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      partial[h * num_warps + warp_id] = local_max[h];
    }
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      float warp_max = (threadIdx.x < num_warps) ? partial[h * num_warps + threadIdx.x]
                                                 : -3.4028234663852886e38f;
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_down(warp_max, offset));
      }
      if (threadIdx.x == 0) {
        partial[h * num_warps] = warp_max;
      }
    }
  }
  __syncthreads();

  float chunk_max[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    chunk_max[h] = partial[h * num_warps];
  }
  __syncthreads();

  float local_sum[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    local_sum[h] = 0.0f;
    for (int64_t lt = threadIdx.x; lt < chunk_len; lt += blockDim.x) {
      const float e = expf(scores[h * chunk_size + lt] - chunk_max[h]);
      scores[h * chunk_size + lt] = e;
      local_sum[h] += e;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      local_sum[h] += __shfl_down(local_sum[h], offset);
    }
  }
  if (lane_id == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      partial[h * num_warps + warp_id] = local_sum[h];
    }
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      float warp_sum = (threadIdx.x < num_warps) ? partial[h * num_warps + threadIdx.x] : 0.0f;
      for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        warp_sum += __shfl_down(warp_sum, offset);
      }
      if (threadIdx.x == 0) {
        partial[h * num_warps] = warp_sum;
      }
    }
  }
  __syncthreads();

  float chunk_sum[q_per_kv];
  for (int64_t h = 0; h < q_per_kv; ++h) {
    chunk_sum[h] = partial[h * num_warps];
  }
  __syncthreads();

  if (static_cast<int64_t>(threadIdx.x) < head_dim) {
    const int64_t dim = threadIdx.x;
    float acc[q_per_kv];
    for (int64_t h = 0; h < q_per_kv; ++h) {
      acc[h] = 0.0f;
    }
    if (contiguous_blocks) {
      const int64_t v_base = ((base_physical_block << 8) * num_kv_heads + kv_head) * head_dim + dim;
      constexpr int64_t v_stride = num_kv_heads * head_dim;
      for (int64_t lt = 0; lt < chunk_len; ++lt) {
        const int64_t local_token_offset = token_start + lt - contiguous_token_base;
        const int64_t token_offset = (base_physical_block << 8) + local_token_offset;
        const int64_t scale_offset = token_offset * num_kv_heads + kv_head;
        const float v = static_cast<float>(value_cache[v_base + local_token_offset * v_stride]) * scalar_to_float_qwen35(v_scale[scale_offset]);
        for (int64_t h = 0; h < q_per_kv; ++h) {
          acc[h] += scores[h * chunk_size + lt] * v;
        }
      }
    } else {
      for (int64_t lt = 0; lt < chunk_len; ++lt) {
        const int64_t token_offset = static_cast<int64_t>(token_offsets[lt]);
        const int64_t v_offset = (token_offset * num_kv_heads + kv_head) * head_dim + dim;
        const int64_t scale_offset = token_offset * num_kv_heads + kv_head;
        const float v = static_cast<float>(value_cache[v_offset]) * scalar_to_float_qwen35(v_scale[scale_offset]);
        for (int64_t h = 0; h < q_per_kv; ++h) {
          acc[h] += scores[h * chunk_size + lt] * v;
        }
      }
    }
    for (int64_t h = 0; h < q_per_kv; ++h) {
      const int64_t partial_idx = (q_base + h) * num_splits + split_idx;
      partial_out[partial_idx * head_dim + dim] = acc[h];
    }
  }

  if (threadIdx.x == 0) {
    for (int64_t h = 0; h < q_per_kv; ++h) {
      const int64_t partial_idx = (q_base + h) * num_splits + split_idx;
      partial_m[partial_idx] = chunk_max[h];
      partial_l[partial_idx] = chunk_sum[h];
    }
  }
}

template <bool SHORT_BLOCK256>
__global__ void qwen35_paged_full_attn_prefill_gqa_gate_bf16_kernel(
    const float* __restrict__ query,       // [rows, num_q_heads, head_dim]
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    const uint16_t* __restrict__ gate,     // [rows, num_q_heads, head_dim] BF16 view-compatible
    uint16_t* __restrict__ out,            // [rows, num_q_heads * head_dim] BF16
    const int32_t* __restrict__ block_tables,
    const int64_t* __restrict__ context_counts,
    const int64_t* __restrict__ row_positions,
    int64_t rows,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale) {
  const int64_t q_head = blockIdx.x;
  const int64_t row = blockIdx.y;
  if (q_head >= num_q_heads || row >= rows) {
    return;
  }
  int64_t context_len = context_counts[row];
  if (context_len <= 0) {
    return;
  }
  context_len = context_len > max_context_len ? max_context_len : context_len;
  const int64_t causal_position = row_positions ? row_positions[row] : (context_counts[row] - 1);
  if (causal_position < 0) {
    return;
  }
  const int64_t causal_len = causal_position + 1;
  const int64_t visible_len = context_len < causal_len ? context_len : causal_len;
  if (visible_len <= 0) {
    return;
  }

  extern __shared__ float shared[];
  float* scores = shared;
  float* warp_buf = scores + visible_len;
  const int num_warps = blockDim.x / warpSize;
  float* q_shared = warp_buf + num_warps;
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const float* q = query + (row * num_q_heads + q_head) * head_dim;
  const int32_t* row_table = block_tables + row * block_table_len;
  const int lane_id = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    q_shared[dim] = q[dim];
  }
  __syncthreads();

  float local_max = -3.4028234663852886e38f;
  for (int64_t token = warp_id; token < visible_len; token += num_warps) {
    int64_t block_offset;
    int64_t physical_block;
    if constexpr (SHORT_BLOCK256) {
      const int64_t logical_block = token >> 8;
      block_offset = token & 255;
      physical_block = (logical_block == 0)
          ? (block_table_len > 0 ? static_cast<int64_t>(row_table[0]) : -1)
          : (block_table_len > 1 ? static_cast<int64_t>(row_table[1]) : -1);
    } else {
      const int64_t logical_block = (block_size == 256) ? (token >> 8) : (token / block_size);
      if (logical_block >= block_table_len) {
        if (lane_id == 0) {
          scores[token] = -3.4028234663852886e38f;
        }
        continue;
      }
      block_offset = (block_size == 256) ? (token & 255) : (token - logical_block * block_size);
      physical_block = static_cast<int64_t>(row_table[logical_block]);
    }
    if (physical_block < 0) {
      if (lane_id == 0) {
        scores[token] = -3.4028234663852886e38f;
      }
      continue;
    }
    const int64_t k_offset = ((physical_block * block_size + block_offset) * num_kv_heads + kv_head) * head_dim;
    float acc = 0.0f;
    for (int64_t dim = lane_id * 4; dim < head_dim; dim += warpSize * 4) {
      acc += q_shared[dim + 0] * bf16_bits_to_float(key_cache[k_offset + dim + 0]);
      acc += q_shared[dim + 1] * bf16_bits_to_float(key_cache[k_offset + dim + 1]);
      acc += q_shared[dim + 2] * bf16_bits_to_float(key_cache[k_offset + dim + 2]);
      acc += q_shared[dim + 3] * bf16_bits_to_float(key_cache[k_offset + dim + 3]);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      acc += __shfl_down(acc, offset);
    }
    if (lane_id == 0) {
      const float score = acc * scale;
      scores[token] = score;
      local_max = fmaxf(local_max, score);
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    float other = __shfl_down(local_max, offset);
    local_max = fmaxf(local_max, other);
  }
  if (lane_id == 0) {
    warp_buf[warp_id] = local_max;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float max_score = warp_buf[0];
    for (int i = 1; i < num_warps; ++i) {
      max_score = fmaxf(max_score, warp_buf[i]);
    }
    warp_buf[0] = max_score;
  }
  __syncthreads();
  const float max_score = warp_buf[0];

  float local_sum = 0.0f;
  for (int64_t token = warp_id; token < visible_len; token += num_warps) {
    if (lane_id == 0) {
      const float val = expf(scores[token] - max_score);
      scores[token] = val;
      local_sum += val;
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
    for (int i = 1; i < num_warps; ++i) {
      total_sum += warp_buf[i];
    }
    warp_buf[0] = 1.0f / fmaxf(total_sum, 1.0e-20f);
  }
  __syncthreads();
  const float inv_denom = warp_buf[0];

  for (int64_t token = warp_id; token < visible_len; token += num_warps) {
    if (lane_id == 0) {
      scores[token] *= inv_denom;
    }
  }
  __syncthreads();

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < visible_len; ++token) {
      int64_t block_offset;
      int64_t physical_block;
      if constexpr (SHORT_BLOCK256) {
        const int64_t logical_block = token >> 8;
        block_offset = token & 255;
        physical_block = (logical_block == 0)
            ? (block_table_len > 0 ? static_cast<int64_t>(row_table[0]) : -1)
            : (block_table_len > 1 ? static_cast<int64_t>(row_table[1]) : -1);
      } else {
        const int64_t logical_block = (block_size == 256) ? (token >> 8) : (token / block_size);
        if (logical_block >= block_table_len) {
          continue;
        }
        block_offset = (block_size == 256) ? (token & 255) : (token - logical_block * block_size);
        physical_block = static_cast<int64_t>(row_table[logical_block]);
      }
      if (physical_block < 0) {
        continue;
      }
      const int64_t v_offset = ((physical_block * block_size + block_offset) * num_kv_heads + kv_head) * head_dim + dim;
      acc += scores[token] * bf16_bits_to_float(value_cache[v_offset]);
    }
    const int64_t gate_idx = (row * num_q_heads + q_head) * gate_stride1 + dim * gate_stride2;
    const float gated = acc * sigmoid_f32(bf16_bits_to_float(gate[gate_idx]));
    out[(row * num_q_heads + q_head) * head_dim + dim] = float_to_bf16_bits(gated);
  }
}

bool valid_qwen35_gqa_split_shape(
    int64_t chunk_size,
    int64_t num_splits,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  return valid_paged_attn_shape(chunk_size * num_splits, block_size, block_table_len, num_q_heads, num_kv_heads, head_dim) &&
      chunk_size > 0 && num_splits > 0 && block_size == 256 && num_q_heads == 16 && num_kv_heads == 2 && head_dim == 256;
}

template <typename scale_t>
int launch_qwen35_paged_full_attn_decode_split_k_gqa_context_int8(
    const float* query,
    const int8_t* key_cache,
    const int8_t* value_cache,
    const scale_t* k_scale,
    const scale_t* v_scale,
    float* partial_out,
    float* partial_m,
    float* partial_l,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t chunk_size,
    int64_t num_splits,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (!valid_qwen35_gqa_split_shape(chunk_size, num_splits, block_size, block_table_len, num_q_heads, num_kv_heads, head_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t q_per_kv = 8;
  const size_t shared_bytes =
      static_cast<size_t>(q_per_kv * chunk_size + q_per_kv * threads + q_per_kv * head_dim) * sizeof(float) +
      static_cast<size_t>(chunk_size + 2) * sizeof(int32_t);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_kernel<scale_t, 8, 16, 2>),
      dim3(static_cast<unsigned int>(num_kv_heads), static_cast<unsigned int>(num_splits)),
      dim3(threads),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      k_scale,
      v_scale,
      base_offsets,
      live_counts,
      partial_out,
      partial_m,
      partial_l,
      chunk_size,
      num_splits,
      scale);
  return static_cast<int>(hipGetLastError());
}

}  // namespace


extern "C" int hipengine_qwen35_full_attn_gate_mul_bf16(
    const float* attn,
    const uint16_t* gate,
    uint16_t* out,
    int64_t total,
    hipStream_t stream) {
  if (total <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(
      qwen35_full_attn_gate_mul_bf16_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      attn,
      gate,
      out,
      total);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_full_attn_gate_mul_fp16(
    const float* attn,
    const half_t* gate,
    half_t* out,
    int64_t total,
    hipStream_t stream) {
  if (total <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(
      qwen35_full_attn_gate_mul_fp16_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      attn,
      gate,
      out,
      total);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_full_attn_decode_context_bf16(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int64_t* context_len,
    int64_t max_context_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (!valid_dense_attn_shape(max_context_len, num_q_heads, num_kv_heads, head_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const size_t shared_bytes = static_cast<size_t>(max_context_len + threads) * sizeof(float);
  hipLaunchKernelGGL(
      qwen35_full_attn_decode_context_tensor_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      out,
      context_len,
      max_context_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_paged_full_attn_decode_split_k_context_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* partial_out,
    float* partial_m,
    float* partial_l,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t chunk_size,
    int64_t num_splits,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (!valid_paged_attn_shape(chunk_size * num_splits, block_size, block_table_len, num_q_heads, num_kv_heads, head_dim) ||
      chunk_size <= 0 || num_splits <= 0 || (head_dim % 8) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const size_t shared_bytes = static_cast<size_t>(chunk_size + threads + head_dim) * sizeof(float);
  hipLaunchKernelGGL(
      qwen35_paged_full_attn_decode_split_k_ctx_tensor_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(num_splits)),
      dim3(threads),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      partial_out,
      partial_m,
      partial_l,
      chunk_size,
      num_splits,
      num_q_heads,
      num_kv_heads,
      head_dim,
      block_size,
      scale);
  return static_cast<int>(hipGetLastError());
}


extern "C" int hipengine_qwen35_paged_full_attn_decode_split_k_warp_context_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* partial_out,
    float* partial_m,
    float* partial_l,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t chunk_size,
    int64_t num_splits,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (!valid_paged_attn_shape(chunk_size * num_splits, block_size, block_table_len, num_q_heads, num_kv_heads, head_dim) ||
      chunk_size <= 0 || num_splits <= 0 || block_size != 256 || num_q_heads != 16 || num_kv_heads != 2 || head_dim != 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const size_t shared_bytes = static_cast<size_t>(chunk_size + threads + head_dim) * sizeof(float) +
      static_cast<size_t>(chunk_size + 2) * sizeof(int32_t);
  hipLaunchKernelGGL(
      qwen35_paged_full_attn_decode_split_k_ctx_tensor_warp_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(num_splits)),
      dim3(threads),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      partial_out,
      partial_m,
      partial_l,
      chunk_size,
      num_splits,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_paged_full_attn_decode_split_k_gqa_context_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* partial_out,
    float* partial_m,
    float* partial_l,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t chunk_size,
    int64_t num_splits,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (!valid_paged_attn_shape(chunk_size * num_splits, block_size, block_table_len, num_q_heads, num_kv_heads, head_dim) ||
      chunk_size <= 0 || num_splits <= 0 || block_size != 256 || num_q_heads != 16 || num_kv_heads != 2 || head_dim != 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t q_per_kv = 8;
  const size_t shared_bytes =
      static_cast<size_t>(q_per_kv * chunk_size + q_per_kv * (threads / 64) + q_per_kv * head_dim) * sizeof(float) +
      static_cast<size_t>(chunk_size + 2) * sizeof(int32_t);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_kernel<8, 16, 2>),
      dim3(static_cast<unsigned int>(num_kv_heads), static_cast<unsigned int>(num_splits)),
      dim3(threads),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      partial_out,
      partial_m,
      partial_l,
      chunk_size,
      num_splits,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_paged_full_attn_decode_split_k_gqa_context_batch_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* partial_out,
    float* partial_m,
    float* partial_l,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t rows,
    int64_t chunk_size,
    int64_t num_splits,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 ||
      !valid_paged_attn_shape(chunk_size * num_splits, block_size, block_table_len, num_q_heads, num_kv_heads, head_dim) ||
      chunk_size <= 0 || num_splits <= 0 || block_size != 256 || num_q_heads != 16 || num_kv_heads != 2 || head_dim != 256) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t q_per_kv = 8;
  const size_t shared_bytes =
      static_cast<size_t>(q_per_kv * chunk_size + q_per_kv * (threads / 64) + q_per_kv * head_dim) * sizeof(float) +
      static_cast<size_t>(chunk_size + 2) * sizeof(int32_t);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_batch_kernel<8, 16, 2>),
      dim3(static_cast<unsigned int>(num_kv_heads), static_cast<unsigned int>(num_splits), static_cast<unsigned int>(rows)),
      dim3(threads),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      partial_out,
      partial_m,
      partial_l,
      rows,
      chunk_size,
      num_splits,
      block_table_len,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    const half_t* gate,
    half_t* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t rows,
    int64_t chunk_size,
    int64_t num_splits,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 ||
      !valid_paged_attn_shape(chunk_size * num_splits, block_size, block_table_len, num_q_heads, num_kv_heads, head_dim) ||
      chunk_size <= 0 || num_splits != 1 || block_size != 256 || num_q_heads != 16 ||
      num_kv_heads != 2 || head_dim != 256 || gate_stride1 <= 0 || gate_stride2 <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t q_per_kv = 8;
  const size_t shared_bytes =
      static_cast<size_t>(q_per_kv * chunk_size + q_per_kv * (threads / 64) + q_per_kv * head_dim) * sizeof(float) +
      static_cast<size_t>(chunk_size + 2) * sizeof(int32_t);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_direct_kernel<8, 16, 2>),
      dim3(static_cast<unsigned int>(num_kv_heads), static_cast<unsigned int>(rows)),
      dim3(threads),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      gate,
      out,
      base_offsets,
      live_counts,
      rows,
      chunk_size,
      block_table_len,
      gate_stride1,
      gate_stride2,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_paged_full_attn_decode_split_k_reduce_f32(
    const float* partial_out,
    const float* partial_m,
    const float* partial_l,
    float* out,
    int64_t num_q_heads,
    int64_t num_splits,
    int64_t head_dim,
    hipStream_t stream) {
  if (num_q_heads <= 0 || num_splits <= 0 || head_dim <= 0 || head_dim > 1024) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int reduce_threads = static_cast<int>(head_dim);
  const size_t shared_bytes = static_cast<size_t>(num_splits) * 3 * sizeof(float);
  hipLaunchKernelGGL(
      qwen35_paged_full_attn_decode_split_k_reduce_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(reduce_threads),
      shared_bytes,
      stream,
      partial_out,
      partial_m,
      partial_l,
      out,
      num_splits,
      head_dim);
  return static_cast<int>(hipGetLastError());
}


extern "C" int hipengine_qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    const half_t* gate,
    half_t* out,
    const int32_t* block_tables,
    const int64_t* context_counts,
    const int64_t* row_positions,
    int64_t rows,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 || max_context_len <= 0 || block_size <= 0 || block_table_len <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || gate_stride1 <= 0 || gate_stride2 <= 0 ||
      (num_q_heads % num_kv_heads) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  // Keep short prefill rows at one RDNA3 wave; 4K rows need the retained
  // 64-thread path for value-loop throughput. The earlier 256-thread launch was
  // repeat-nondeterministic on gfx1100 for identical inputs. Shared scratch must
  // reserve score slots, one reduction slot per thread, and one cached query head.
  const int threads = max_context_len <= 1024 ? 32 : 64;
  const size_t shared = static_cast<size_t>(max_context_len + threads + head_dim) * sizeof(float);
  const bool use_short_block256 = block_size == 256 && max_context_len <= 512;
#define HIPENGINE_LAUNCH_PREFILL_GQA(SHORT) \
  hipLaunchKernelGGL( \
      (qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel<SHORT>), \
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)), \
      dim3(threads), \
      shared, \
      stream, \
      query, \
      key_cache, \
      value_cache, \
      gate, \
      out, \
      block_tables, \
      context_counts, \
      row_positions, \
      rows, \
      max_context_len, \
      block_size, \
      block_table_len, \
      num_q_heads, \
      num_kv_heads, \
      head_dim, \
      gate_stride1, \
      gate_stride2, \
      scale)
  if (use_short_block256) {
    HIPENGINE_LAUNCH_PREFILL_GQA(true);
  } else {
    HIPENGINE_LAUNCH_PREFILL_GQA(false);
  }
#undef HIPENGINE_LAUNCH_PREFILL_GQA
  return static_cast<int>(hipGetLastError());
}

template <typename scale_t>
int launch_qwen35_paged_full_attn_prefill_gqa_gate_int8(
    const float* query,
    const int8_t* key_cache,
    const int8_t* value_cache,
    const scale_t* k_scale,
    const scale_t* v_scale,
    const half_t* gate,
    half_t* out,
    const int32_t* block_tables,
    const int64_t* context_counts,
    const int64_t* row_positions,
    int64_t rows,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 || max_context_len <= 0 || block_size <= 0 || block_table_len <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || head_dim > 256 ||
      gate_stride1 <= 0 || gate_stride2 <= 0 || (num_q_heads % num_kv_heads) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const size_t shared = static_cast<size_t>(threads + head_dim) * sizeof(float);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qwen35_paged_full_attn_prefill_gqa_gate_int8_kernel<scale_t>),
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)),
      dim3(threads),
      shared,
      stream,
      query,
      key_cache,
      value_cache,
      k_scale,
      v_scale,
      gate,
      out,
      block_tables,
      context_counts,
      row_positions,
      rows,
      max_context_len,
      block_size,
      block_table_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      gate_stride1,
      gate_stride2,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_paged_full_attn_prefill_gqa_gate_int8_scale_f32_spans(
    const float* query,
    const int8_t* key_cache,
    const int8_t* value_cache,
    const float* k_scale,
    const float* v_scale,
    const half_t* gate,
    half_t* out,
    const int32_t* block_tables,
    const int64_t* context_counts,
    const int64_t* row_positions,
    int64_t rows,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale,
    hipStream_t stream) {
  return launch_qwen35_paged_full_attn_prefill_gqa_gate_int8<float>(
      query,
      key_cache,
      value_cache,
      k_scale,
      v_scale,
      gate,
      out,
      block_tables,
      context_counts,
      row_positions,
      rows,
      max_context_len,
      block_size,
      block_table_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      gate_stride1,
      gate_stride2,
      scale,
      stream);
}

extern "C" int hipengine_qwen35_paged_full_attn_prefill_gqa_gate_int8_scale_fp16_spans(
    const float* query,
    const int8_t* key_cache,
    const int8_t* value_cache,
    const half_t* k_scale,
    const half_t* v_scale,
    const half_t* gate,
    half_t* out,
    const int32_t* block_tables,
    const int64_t* context_counts,
    const int64_t* row_positions,
    int64_t rows,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale,
    hipStream_t stream) {
  return launch_qwen35_paged_full_attn_prefill_gqa_gate_int8<half_t>(
      query,
      key_cache,
      value_cache,
      k_scale,
      v_scale,
      gate,
      out,
      block_tables,
      context_counts,
      row_positions,
      rows,
      max_context_len,
      block_size,
      block_table_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      gate_stride1,
      gate_stride2,
      scale,
      stream);
}

extern "C" int hipengine_qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    const half_t* gate,
    half_t* out,
    const int32_t* block_tables,
    const int64_t* context_counts,
    const int64_t* row_positions,
    const uint8_t* ancestor_mask,
    const int64_t* tree_committed_count_ptr,
    int64_t rows,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 || max_context_len <= 0 || block_size <= 0 || block_table_len <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || gate_stride1 <= 0 || gate_stride2 <= 0 ||
      (num_q_heads % num_kv_heads) != 0 || tree_committed_count_ptr == nullptr || ancestor_mask == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  // Same thread/shared sizing as the chain GQA gate kernel.
  const int threads = max_context_len <= 1024 ? 32 : 64;
  const size_t shared = static_cast<size_t>(max_context_len + threads + head_dim) * sizeof(float);
  const bool use_short_block256 = block_size == 256 && max_context_len <= 512;
#define HIPENGINE_LAUNCH_PREFILL_GQA_TREE(SHORT) \
  hipLaunchKernelGGL( \
      (qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_kernel<SHORT>), \
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)), \
      dim3(threads), \
      shared, \
      stream, \
      query, \
      key_cache, \
      value_cache, \
      gate, \
      out, \
      block_tables, \
      context_counts, \
      row_positions, \
      ancestor_mask, \
      tree_committed_count_ptr, \
      rows, \
      max_context_len, \
      block_size, \
      block_table_len, \
      num_q_heads, \
      num_kv_heads, \
      head_dim, \
      gate_stride1, \
      gate_stride2, \
      scale)
  if (use_short_block256) {
    HIPENGINE_LAUNCH_PREFILL_GQA_TREE(true);
  } else {
    HIPENGINE_LAUNCH_PREFILL_GQA_TREE(false);
  }
#undef HIPENGINE_LAUNCH_PREFILL_GQA_TREE
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    const half_t* gate,
    half_t* out,
    const int32_t* block_tables,
    const int64_t* context_counts,
    const int64_t* row_positions,
    const int32_t* cu_seqlens_q,
    const int32_t* cu_seqlens_k,
    int64_t rows,
    int64_t segments,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 || segments <= 0 || max_context_len <= 0 || block_size <= 0 || block_table_len <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || gate_stride1 <= 0 || gate_stride2 <= 0 ||
      (num_q_heads % num_kv_heads) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  // Match the single-request short-row one-wave policy and keep enough scratch
  // for reduction slots plus the cached query head on short varlen rows.
  const int threads = max_context_len <= 1024 ? 32 : 64;
  const size_t shared = static_cast<size_t>(max_context_len + threads + head_dim) * sizeof(float);
  hipLaunchKernelGGL(
      qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)),
      dim3(threads),
      shared,
      stream,
      query,
      key_cache,
      value_cache,
      gate,
      out,
      block_tables,
      context_counts,
      row_positions,
      cu_seqlens_q,
      cu_seqlens_k,
      rows,
      segments,
      max_context_len,
      block_size,
      block_table_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      gate_stride1,
      gate_stride2,
      scale);
  return static_cast<int>(hipGetLastError());
}


extern "C" int hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_bf16(
    const float* partial_out,
    const float* partial_m,
    const float* partial_l,
    const uint16_t* gate,
    uint16_t* out,
    int64_t num_q_heads,
    int64_t num_splits,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    hipStream_t stream) {
  if (num_q_heads <= 0 || num_splits <= 0 || head_dim <= 0 || head_dim > 1024 ||
      gate_stride1 <= 0 || gate_stride2 <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int reduce_threads = static_cast<int>(head_dim);
  const size_t shared_bytes = static_cast<size_t>(num_splits) * 3 * sizeof(float);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<hip_bfloat16>),
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(reduce_threads),
      shared_bytes,
      stream,
      partial_out,
      partial_m,
      partial_l,
      reinterpret_cast<const hip_bfloat16*>(gate),
      reinterpret_cast<hip_bfloat16*>(out),
      num_splits,
      head_dim,
      gate_stride1,
      gate_stride2);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_fp16(
    const float* partial_out,
    const float* partial_m,
    const float* partial_l,
    const half_t* gate,
    half_t* out,
    int64_t num_q_heads,
    int64_t num_splits,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    hipStream_t stream) {
  if (num_q_heads <= 0 || num_splits <= 0 || head_dim <= 0 || head_dim > 1024 ||
      gate_stride1 <= 0 || gate_stride2 <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int reduce_threads = static_cast<int>(head_dim);
  const size_t shared_bytes = static_cast<size_t>(num_splits) * 3 * sizeof(float);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<half_t>),
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(reduce_threads),
      shared_bytes,
      stream,
      partial_out,
      partial_m,
      partial_l,
      gate,
      out,
      num_splits,
      head_dim,
      gate_stride1,
      gate_stride2);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_fp16_batch(
    const float* partial_out,
    const float* partial_m,
    const float* partial_l,
    const half_t* gate,
    half_t* out,
    int64_t rows,
    int64_t num_q_heads,
    int64_t num_splits,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    hipStream_t stream) {
  if (rows <= 0 || num_q_heads <= 0 || num_splits <= 0 || head_dim <= 0 || head_dim > 1024 ||
      gate_stride1 <= 0 || gate_stride2 <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int reduce_threads = static_cast<int>(head_dim);
  const size_t shared_bytes = static_cast<size_t>(num_splits) * 3 * sizeof(float);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qwen35_paged_full_attn_decode_split_k_reduce_gate_batch_kernel<half_t>),
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)),
      dim3(reduce_threads),
      shared_bytes,
      stream,
      partial_out,
      partial_m,
      partial_l,
      gate,
      out,
      rows,
      num_splits,
      head_dim,
      gate_stride1,
      gate_stride2);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_f32(
    const float* partial_out,
    const float* partial_m,
    const float* partial_l,
    const float* gate,
    float* out,
    int64_t num_q_heads,
    int64_t num_splits,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    hipStream_t stream) {
  if (num_q_heads <= 0 || num_splits <= 0 || head_dim <= 0 || head_dim > 1024 ||
      gate_stride1 <= 0 || gate_stride2 <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int reduce_threads = static_cast<int>(head_dim);
  const size_t shared_bytes = static_cast<size_t>(num_splits) * 3 * sizeof(float);
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<float>),
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(reduce_threads),
      shared_bytes,
      stream,
      partial_out,
      partial_m,
      partial_l,
      gate,
      out,
      num_splits,
      head_dim,
      gate_stride1,
      gate_stride2);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_paged_full_attn_decode_context_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (!valid_paged_attn_shape(max_context_len, block_size, block_table_len, num_q_heads, num_kv_heads, head_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const size_t shared_bytes = static_cast<size_t>(max_context_len + 8 + head_dim) * sizeof(float);
  hipLaunchKernelGGL(
      qwen35_paged_full_attn_decode_context_tensor_kernel,
      dim3(static_cast<unsigned int>(num_q_heads)),
      dim3(threads),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      out,
      max_context_len,
      block_size,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_paged_full_attn_decode_context_bf16_batch_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    float* out,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t rows,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 || !valid_paged_attn_shape(max_context_len, block_size, block_table_len, num_q_heads, num_kv_heads, head_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  // c=2 decode is dominated by fixed launch/context overhead. Give the
  // small-row, sub-1K context batch path more warps so each head processes
  // more context tokens concurrently while larger retained rows keep the
  // existing 256-thread shape.
  const int threads = (rows <= 2 && max_context_len < 1024) ? 1024 : 256;
  const int max_warps = threads / 32;
  const int raw_value_groups = (head_dim >= 128 && max_context_len >= 128) ? static_cast<int>(threads / head_dim) : 0;
  const int value_groups = (threads >= 512 && raw_value_groups > 1) ? raw_value_groups : 1;
  const size_t shared_bytes = static_cast<size_t>(max_context_len + max_warps + head_dim + value_groups * head_dim) * sizeof(float);
  hipLaunchKernelGGL(
      qwen35_paged_full_attn_decode_context_tensor_batch_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)),
      dim3(threads),
      shared_bytes,
      stream,
      query,
      key_cache,
      value_cache,
      base_offsets,
      live_counts,
      out,
      rows,
      max_context_len,
      block_size,
      block_table_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_full_attn_gate_mul_bf16_to_bf16(
    const uint16_t* attn,
    const uint16_t* gate,
    uint16_t* out,
    int64_t total,
    hipStream_t stream) {
  if (total <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(
      qwen35_full_attn_gate_mul_bf16_to_bf16_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      attn,
      gate,
      out,
      total);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_paged_full_attn_decode_split_k_gqa_context_int8_scale_f32_spans(
    const float* query,
    const int8_t* key_cache,
    const int8_t* value_cache,
    const float* k_scale,
    const float* v_scale,
    float* partial_out,
    float* partial_m,
    float* partial_l,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t chunk_size,
    int64_t num_splits,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  return launch_qwen35_paged_full_attn_decode_split_k_gqa_context_int8<float>(
      query,
      key_cache,
      value_cache,
      k_scale,
      v_scale,
      partial_out,
      partial_m,
      partial_l,
      base_offsets,
      live_counts,
      chunk_size,
      num_splits,
      block_size,
      block_table_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale,
      stream);
}

extern "C" int hipengine_qwen35_paged_full_attn_decode_split_k_gqa_context_int8_scale_fp16_spans(
    const float* query,
    const int8_t* key_cache,
    const int8_t* value_cache,
    const half_t* k_scale,
    const half_t* v_scale,
    float* partial_out,
    float* partial_m,
    float* partial_l,
    const int32_t* base_offsets,
    const int64_t* live_counts,
    int64_t chunk_size,
    int64_t num_splits,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream) {
  return launch_qwen35_paged_full_attn_decode_split_k_gqa_context_int8<half_t>(
      query,
      key_cache,
      value_cache,
      k_scale,
      v_scale,
      partial_out,
      partial_m,
      partial_l,
      base_offsets,
      live_counts,
      chunk_size,
      num_splits,
      block_size,
      block_table_len,
      num_q_heads,
      num_kv_heads,
      head_dim,
      scale,
      stream);
}

extern "C" int hipengine_qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans(
    const float* query,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    const uint16_t* gate,
    uint16_t* out,
    const int32_t* block_tables,
    const int64_t* context_counts,
    const int64_t* row_positions,
    int64_t rows,
    int64_t max_context_len,
    int64_t block_size,
    int64_t block_table_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t gate_stride1,
    int64_t gate_stride2,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 || max_context_len <= 0 || block_size <= 0 || block_table_len <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || gate_stride1 <= 0 || gate_stride2 <= 0 ||
      (num_q_heads % num_kv_heads) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = max_context_len <= 1024 ? 32 : 64;
  const size_t shared = static_cast<size_t>(max_context_len + threads + head_dim) * sizeof(float);
  const bool use_short_block256 = block_size == 256 && max_context_len <= 512;
#define HIPENGINE_LAUNCH_PREFILL_GQA_BF16(SHORT) \
  hipLaunchKernelGGL( \
      (qwen35_paged_full_attn_prefill_gqa_gate_bf16_kernel<SHORT>), \
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)), \
      dim3(threads), \
      shared, \
      stream, \
      query, \
      key_cache, \
      value_cache, \
      gate, \
      out, \
      block_tables, \
      context_counts, \
      row_positions, \
      rows, \
      max_context_len, \
      block_size, \
      block_table_len, \
      num_q_heads, \
      num_kv_heads, \
      head_dim, \
      gate_stride1, \
      gate_stride2, \
      scale)
  if (use_short_block256) {
    HIPENGINE_LAUNCH_PREFILL_GQA_BF16(true);
  } else {
    HIPENGINE_LAUNCH_PREFILL_GQA_BF16(false);
  }
#undef HIPENGINE_LAUNCH_PREFILL_GQA_BF16
  return static_cast<int>(hipGetLastError());
}
