// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip @ 59195ed.
// Qwen3.5 paged full-attention context-tensor decode body is preserved; public wrapper uses hipENGINE KVLiveSpans.

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

  for (int64_t token = warp_id; token < context_len; token += num_warps) {
    if (lane_id == 0) {
      scores[token] *= inv_denom;
    }
  }
  __syncthreads();

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
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const float* q = query + (row * num_q_heads + q_head) * head_dim;
  const int32_t* row_table = block_tables + row * block_table_len;

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
    for (int64_t dim = 0; dim < head_dim; ++dim) {
      acc += q[dim] * bf16_bits_to_float(key_cache[k_offset + dim]);
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
    out[(row * num_q_heads + q_head) * head_dim + dim] = static_cast<half_t>(acc * sigmoid_f32(gate_v));
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
  const int threads = 256;
  const size_t shared = static_cast<size_t>(max_context_len * 2) * sizeof(float);
  hipLaunchKernelGGL(
      qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel,
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
  const int threads = 256;
  const size_t shared_bytes = static_cast<size_t>(max_context_len + 8 + head_dim) * sizeof(float);
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
