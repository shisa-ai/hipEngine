// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip @ 59195ed.
// Qwen3.5 paged full-attention context-tensor decode body is preserved; public wrapper uses HIPENGINE KVLiveSpans.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
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
