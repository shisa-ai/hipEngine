// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip @ 59195ed.
// Qwen3.5 partial rotary and fused head-RMSNorm+rotary kernel bodies are preserved; wrappers are HIPENGINE C ABI.

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

__global__ void qwen35_partial_rotary_kernel(
    const float* query,
    const float* key,
    const float* cos,
    const float* sin,
    float* query_out,
    float* key_out,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim) {
  const int64_t total_heads = num_q_heads + num_kv_heads;
  const int64_t total = total_heads * head_dim;
  const int64_t query_total = num_q_heads * head_dim;
  const int64_t half_rotary = rotary_dim / 2;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const bool is_query = idx < query_total;
    const int64_t local = is_query ? idx : idx - query_total;
    const int64_t dim = local % head_dim;
    const int64_t base = local - dim;
    const float* src = is_query ? query : key;
    float* dst = is_query ? query_out : key_out;
    float value = src[local];
    if (dim < rotary_dim) {
      const int64_t pair_dim = dim < half_rotary ? dim + half_rotary : dim - half_rotary;
      const float paired = src[base + pair_dim];
      const float rotated = dim < half_rotary ? -paired : paired;
      value = value * cos[dim] + rotated * sin[dim];
    }
    dst[local] = value;
  }
}

__global__ void qwen35_head_rmsnorm_partial_rotary_kernel(
    const float* query,
    const float* key,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const float* cos,
    const float* sin,
    float* query_out,
    float* key_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim) {
  extern __shared__ float partial[];
  const int64_t head = blockIdx.x;
  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const float* src = is_query ? query : key;
  const uint16_t* weight = is_query ? q_weight : k_weight;
  float* dst = is_query ? query_out : key_out;
  const int64_t offset = local_head * head_dim;
  float sumsq = 0.0f;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const float value = src[offset + dim];
    sumsq += value * value;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf((partial[0] / static_cast<float>(head_dim)) + eps);
  const int64_t half_rotary = rotary_dim / 2;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const float value = src[offset + dim] * inv_rms * (1.0f + bf16_bits_to_float(weight[dim]));
    float out_value = value;
    if (dim < rotary_dim) {
      const int64_t pair_dim = dim < half_rotary ? dim + half_rotary : dim - half_rotary;
      const float paired =
          src[offset + pair_dim] * inv_rms * (1.0f + bf16_bits_to_float(weight[pair_dim]));
      const float rotated = dim < half_rotary ? -paired : paired;
      out_value = value * cos[dim] + rotated * sin[dim];
    }
    dst[offset + dim] = out_value;
  }
}

__global__ void qwen35_head_rmsnorm_partial_rotary_position_kernel(
    const float* query,
    const float* key,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const float* cos_table,
    const float* sin_table,
    const int64_t* position_ptr,
    float* query_out,
    float* key_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions) {
  const int64_t position = position_ptr[0];
  if (position < 0 || position >= max_positions) {
    return;
  }
  const float* cos = cos_table + position * rotary_dim;
  const float* sin = sin_table + position * rotary_dim;
  extern __shared__ float partial[];
  const int64_t head = blockIdx.x;
  const bool is_query = head < num_q_heads;
  const int64_t local_head = is_query ? head : head - num_q_heads;
  const float* src = is_query ? query : key;
  const uint16_t* weight = is_query ? q_weight : k_weight;
  float* dst = is_query ? query_out : key_out;
  const int64_t offset = local_head * head_dim;
  float sumsq = 0.0f;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const float value = src[offset + dim];
    sumsq += value * value;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf((partial[0] / static_cast<float>(head_dim)) + eps);
  const int64_t half_rotary = rotary_dim / 2;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const float value = src[offset + dim] * inv_rms * (1.0f + bf16_bits_to_float(weight[dim]));
    float out_value = value;
    if (dim < rotary_dim) {
      const int64_t pair_dim = dim < half_rotary ? dim + half_rotary : dim - half_rotary;
      const float paired =
          src[offset + pair_dim] * inv_rms * (1.0f + bf16_bits_to_float(weight[pair_dim]));
      const float rotated = dim < half_rotary ? -paired : paired;
      out_value = value * cos[dim] + rotated * sin[dim];
    }
    dst[offset + dim] = out_value;
  }
}


bool valid_common_shape(int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim, int64_t rotary_dim) {
  return num_q_heads > 0 && num_kv_heads > 0 && head_dim > 0 && rotary_dim > 0 &&
      rotary_dim <= head_dim && (rotary_dim % 2) == 0;
}

}  // namespace

extern "C" int hipengine_qwen35_partial_rotary_f32(
    const float* query,
    const float* key,
    const float* cos,
    const float* sin,
    float* query_out,
    float* key_out,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    hipStream_t stream) {
  if (!valid_common_shape(num_q_heads, num_kv_heads, head_dim, rotary_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t total = (num_q_heads + num_kv_heads) * head_dim;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(
      qwen35_partial_rotary_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      query,
      key,
      cos,
      sin,
      query_out,
      key_out,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_head_rmsnorm_partial_rotary_f32_bf16(
    const float* query,
    const float* key,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const float* cos,
    const float* sin,
    float* query_out,
    float* key_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    hipStream_t stream) {
  if (!valid_common_shape(num_q_heads, num_kv_heads, head_dim, rotary_dim)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  hipLaunchKernelGGL(
      qwen35_head_rmsnorm_partial_rotary_kernel,
      dim3(static_cast<unsigned int>(num_q_heads + num_kv_heads)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      query,
      key,
      q_weight,
      k_weight,
      cos,
      sin,
      query_out,
      key_out,
      eps,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_head_rmsnorm_partial_rotary_position_f32_bf16(
    const float* query,
    const float* key,
    const uint16_t* q_weight,
    const uint16_t* k_weight,
    const float* cos_table,
    const float* sin_table,
    const int64_t* position,
    float* query_out,
    float* key_out,
    float eps,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t max_positions,
    hipStream_t stream) {
  if (!valid_common_shape(num_q_heads, num_kv_heads, head_dim, rotary_dim) || max_positions <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  hipLaunchKernelGGL(
      qwen35_head_rmsnorm_partial_rotary_position_kernel,
      dim3(static_cast<unsigned int>(num_q_heads + num_kv_heads)),
      dim3(threads),
      threads * sizeof(float),
      stream,
      query,
      key,
      q_weight,
      k_weight,
      cos_table,
      sin_table,
      position,
      query_out,
      key_out,
      eps,
      num_q_heads,
      num_kv_heads,
      head_dim,
      rotary_dim,
      max_positions);
  return static_cast<int>(hipGetLastError());
}
