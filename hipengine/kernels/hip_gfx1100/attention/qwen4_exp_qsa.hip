// hipEngine-native Qwen3.8-Flash-Next QSA strict control primitives.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <float.h>

namespace {

__global__ void qsa_split_norm_rope_f32_kernel(
    const float* __restrict__ q_projected,
    const float* __restrict__ key,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const int64_t* __restrict__ position,
    float* __restrict__ query_out,
    float* __restrict__ key_out,
    float* __restrict__ gate_out,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    float theta,
    float eps) {
  extern __shared__ float shared[];
  float* values = shared;
  float* partial = values + head_dim;
  const int64_t head = blockIdx.x;
  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const int64_t source_stride = is_query ? 2 * head_dim : head_dim;
  const float* source = is_query
      ? q_projected + local_head * source_stride
      : key + local_head * source_stride;
  const float* weight = is_query ? q_weight : k_weight;
  float sumsq = 0.0f;
  for (int64_t dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
    const float value = source[dimension];
    values[dimension] = value;
    sumsq += value * value;
    if (is_query) {
      gate_out[local_head * head_dim + dimension] = source[head_dim + dimension];
    }
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  float* output = is_query
      ? query_out + local_head * head_dim
      : key_out + local_head * head_dim;
  for (int64_t dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
    float result = values[dimension] * inv_rms * weight[dimension];
    if (dimension < rotary_dim) {
      const int64_t even = dimension & ~1LL;
      const int64_t pair = even / 2;
      const float x = values[even] * inv_rms * weight[even];
      const float y = values[even + 1] * inv_rms * weight[even + 1];
      const float exponent = -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim);
      const float angle = static_cast<float>(position[0]) * powf(theta, exponent);
      const float c = cosf(angle);
      const float s = sinf(angle);
      result = (dimension & 1) ? x * s + y * c : x * c - y * s;
    }
    output[dimension] = result;
  }
}

__global__ void qsa_gate_context_f32_kernel(
    const float* __restrict__ context,
    const float* __restrict__ gate,
    float* __restrict__ output,
    int64_t elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = context[index] / (1.0f + expf(-gate[index]));
  }
}

__global__ void qsa_pool_norm_rope_f32_kernel(
    const float* __restrict__ raw_keys,
    const int32_t* __restrict__ member_indices,
    const int64_t* __restrict__ block_starts,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t ratio,
    int64_t index_dim,
    int64_t rotary_dim,
    float theta,
    float eps) {
  extern __shared__ float shared[];
  float* pooled = shared;
  float* partial = shared + index_dim;
  const int64_t block = blockIdx.x;
  for (int64_t dimension = threadIdx.x; dimension < index_dim; dimension += blockDim.x) {
    float value = 0.0f;
    for (int64_t member = 0; member < ratio; ++member) {
      const int32_t physical = member_indices[block * ratio + member];
      value += raw_keys[static_cast<int64_t>(physical) * index_dim + dimension];
    }
    pooled[dimension] = value / static_cast<float>(ratio);
  }
  __syncthreads();
  float sumsq = 0.0f;
  for (int64_t dimension = threadIdx.x; dimension < index_dim; dimension += blockDim.x) {
    sumsq += pooled[dimension] * pooled[dimension];
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(index_dim) + eps);
  for (int64_t dimension = threadIdx.x; dimension < index_dim; dimension += blockDim.x) {
    float value = pooled[dimension] * inv_rms * weight[dimension];
    if (dimension < rotary_dim) {
      const int64_t pair = dimension / 2;
      const int64_t even = pair * 2;
      const float x = pooled[even] * inv_rms * weight[even];
      const float y = pooled[even + 1] * inv_rms * weight[even + 1];
      const float exponent = -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim);
      const float angle = static_cast<float>(block_starts[block]) * powf(theta, exponent);
      const float cosine = cosf(angle);
      const float sine = sinf(angle);
      value = (dimension & 1) ? x * sine + y * cosine : x * cosine - y * sine;
    }
    output[block * index_dim + dimension] = value;
  }
}

__global__ void qsa_score_f32_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ pooled_keys,
    float* __restrict__ scores,
    int64_t blocks,
    int64_t heads,
    int64_t index_dim) {
  extern __shared__ float partial[];
  const int64_t pair = blockIdx.x;
  const int64_t query = pair / blocks;
  const int64_t block = pair % blocks;
  float value = 0.0f;
  for (int64_t head = threadIdx.x; head < heads; head += blockDim.x) {
    float dot = 0.0f;
    const int64_t query_offset = (query * heads + head) * index_dim;
    const int64_t key_offset = block * index_dim;
    for (int64_t dimension = 0; dimension < index_dim; ++dimension) {
      dot += queries[query_offset + dimension] * pooled_keys[key_offset + dimension];
    }
    value += fmaxf(dot, 0.0f);
  }
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    scores[pair] = partial[0] * rsqrtf(static_cast<float>(index_dim));
  }
}

__global__ void qsa_select_blocks_f32_i64_kernel(
    const float* __restrict__ scores,
    const int64_t* __restrict__ block_starts,
    const int64_t* __restrict__ query_positions,
    int64_t* __restrict__ selected_starts,
    int32_t* __restrict__ selected_counts,
    int64_t blocks,
    int64_t ratio,
    int64_t budget) {
  if (threadIdx.x != 0) {
    return;
  }
  const int64_t query = blockIdx.x;
  int64_t count = 0;
  for (int64_t slot = 0; slot < budget; ++slot) {
    selected_starts[query * budget + slot] = -1;
  }
  for (int64_t slot = 0; slot < budget; ++slot) {
    float best_score = -FLT_MAX;
    int64_t best_start = INT64_MAX;
    for (int64_t block = 0; block < blocks; ++block) {
      const int64_t start = block_starts[block];
      if (start + ratio - 1 > query_positions[query]) {
        continue;
      }
      bool used = false;
      for (int64_t prior = 0; prior < count; ++prior) {
        used = used || selected_starts[query * budget + prior] == start;
      }
      if (used) {
        continue;
      }
      const float score = scores[query * blocks + block];
      if (score > best_score || (score == best_score && start < best_start)) {
        best_score = score;
        best_start = start;
      }
    }
    if (best_start == INT64_MAX) {
      break;
    }
    selected_starts[query * budget + count] = best_start;
    ++count;
  }
  for (int64_t outer = 1; outer < count; ++outer) {
    const int64_t current = selected_starts[query * budget + outer];
    int64_t inner = outer - 1;
    while (inner >= 0 && selected_starts[query * budget + inner] > current) {
      selected_starts[query * budget + inner + 1] = selected_starts[query * budget + inner];
      --inner;
    }
    selected_starts[query * budget + inner + 1] = current;
  }
  selected_counts[query] = static_cast<int32_t>(count);
}

}  // namespace

extern "C" int hipengine_qwen4_exp_qsa_split_norm_rope_f32(
    const void* q_projected,
    const void* key,
    const void* q_weight,
    const void* k_weight,
    const void* position,
    void* query_out,
    void* key_out,
    void* gate_out,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    float theta,
    float eps,
    hipStream_t stream) {
  if (!q_projected || !key || !q_weight || !k_weight || !position || !query_out ||
      !key_out || !gate_out || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim & 1) || theta <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const size_t shared = static_cast<size_t>(head_dim + threads) * sizeof(float);
  hipLaunchKernelGGL(
      qsa_split_norm_rope_f32_kernel,
      dim3(static_cast<uint32_t>(num_q_heads + num_kv_heads)), dim3(threads), shared, stream,
      static_cast<const float*>(q_projected), static_cast<const float*>(key),
      static_cast<const float*>(q_weight), static_cast<const float*>(k_weight),
      static_cast<const int64_t*>(position), static_cast<float*>(query_out),
      static_cast<float*>(key_out), static_cast<float*>(gate_out),
      num_q_heads, num_kv_heads, head_dim, rotary_dim, theta, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_gate_context_f32(
    const void* context,
    const void* gate,
    void* output,
    int64_t elements,
    hipStream_t stream) {
  if (!context || !gate || !output || elements <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const uint32_t blocks = static_cast<uint32_t>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      qsa_gate_context_f32_kernel, dim3(blocks), dim3(threads), 0, stream,
      static_cast<const float*>(context), static_cast<const float*>(gate),
      static_cast<float*>(output), elements);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_pool_norm_rope_f32(
    const void* raw_keys,
    const void* member_indices,
    const void* block_starts,
    const void* weight,
    void* output,
    int64_t blocks,
    int64_t ratio,
    int64_t index_dim,
    int64_t rotary_dim,
    float theta,
    float eps,
    hipStream_t stream) {
  if (!raw_keys || !member_indices || !block_starts || !weight || !output ||
      blocks <= 0 || ratio <= 0 || index_dim <= 0 || rotary_dim <= 0 ||
      rotary_dim > index_dim || (rotary_dim & 1) || theta <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const size_t shared = static_cast<size_t>(index_dim + threads) * sizeof(float);
  hipLaunchKernelGGL(
      qsa_pool_norm_rope_f32_kernel,
      dim3(static_cast<uint32_t>(blocks)), dim3(threads), shared, stream,
      static_cast<const float*>(raw_keys), static_cast<const int32_t*>(member_indices),
      static_cast<const int64_t*>(block_starts), static_cast<const float*>(weight),
      static_cast<float*>(output), ratio, index_dim, rotary_dim, theta, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_score_f32(
    const void* queries,
    const void* pooled_keys,
    void* scores,
    int64_t query_count,
    int64_t blocks,
    int64_t heads,
    int64_t index_dim,
    hipStream_t stream) {
  if (!queries || !pooled_keys || !scores || query_count <= 0 || blocks <= 0 ||
      heads <= 0 || index_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 32;
  hipLaunchKernelGGL(
      qsa_score_f32_kernel,
      dim3(static_cast<uint32_t>(query_count * blocks)), dim3(threads), threads * sizeof(float), stream,
      static_cast<const float*>(queries), static_cast<const float*>(pooled_keys),
      static_cast<float*>(scores), blocks, heads, index_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_qsa_select_blocks_f32_i64(
    const void* scores,
    const void* block_starts,
    const void* query_positions,
    void* selected_starts,
    void* selected_counts,
    int64_t query_count,
    int64_t blocks,
    int64_t ratio,
    int64_t budget,
    hipStream_t stream) {
  if (!scores || !block_starts || !query_positions || !selected_starts || !selected_counts ||
      query_count <= 0 || blocks <= 0 || ratio <= 0 || budget <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qsa_select_blocks_f32_i64_kernel,
      dim3(static_cast<uint32_t>(query_count)), dim3(1), 0, stream,
      static_cast<const float*>(scores), static_cast<const int64_t*>(block_starts),
      static_cast<const int64_t*>(query_positions), static_cast<int64_t*>(selected_starts),
      static_cast<int32_t*>(selected_counts), blocks, ratio, budget);
  return static_cast<int>(hipGetLastError());
}
