// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip @ 59195ed.
// Qwen3.5 paged full-attention context-tensor decode body is preserved; public wrapper uses HIPENGINE KVLiveSpans.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef uint32_t uint32x4 __attribute__((ext_vector_type(4)));

namespace {

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
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
