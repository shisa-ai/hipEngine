// Small BF16 elementwise helpers for GGUF Qwen3.5 bring-up.

#include <hip/hip_runtime.h>

#include <cstdint>

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

__device__ inline float sigmoidf_stable(float value) {
  if (value >= 0.0f) {
    const float z = expf(-value);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(value);
  return z / (1.0f + z);
}

__global__ void gguf_rmsnorm_bf16_f32_weight_kernel(
    const uint16_t* __restrict__ x,
    const float* __restrict__ weight,
    uint16_t* __restrict__ out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t tid = threadIdx.x;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]);
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
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]) * inv_rms * weight[idx];
    out[row_base + idx] = float_to_bf16_bits(value);
  }
}

__global__ void gguf_add_rmsnorm_bf16_f32_weight_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ add,
    const float* __restrict__ weight,
    uint16_t* __restrict__ norm_out,
    uint16_t* __restrict__ residual_out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t tid = threadIdx.x;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * hidden_size;
  float sum = 0.0f;
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]) + bf16_bits_to_float(add[row_base + idx]);
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
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t idx = tid; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(x[row_base + idx]) + bf16_bits_to_float(add[row_base + idx]);
    residual_out[row_base + idx] = float_to_bf16_bits(value);
    norm_out[row_base + idx] = float_to_bf16_bits(value * inv_rms * weight[idx]);
  }
}

__global__ void gguf_bf16_add_kernel(
    const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ b,
    uint16_t* __restrict__ out,
    int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    out[idx] = float_to_bf16_bits(bf16_bits_to_float(a[idx]) + bf16_bits_to_float(b[idx]));
  }
}

__global__ void gguf_gate_repeat_value_bf16_kernel(
    const uint16_t* __restrict__ q_gate,
    const uint16_t* __restrict__ value,
    uint16_t* __restrict__ out,
    int64_t head_count,
    int64_t head_count_kv,
    int64_t head_dim) {
  const int64_t n = head_count * head_dim;
  const int64_t ratio = head_count / head_count_kv;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    const int64_t head = idx / head_dim;
    const int64_t dim = idx - head * head_dim;
    const int64_t kv_head = head / ratio;
    const float gate = sigmoidf_stable(bf16_bits_to_float(q_gate[idx]));
    const float v = bf16_bits_to_float(value[kv_head * head_dim + dim]);
    out[idx] = float_to_bf16_bits(gate * v);
  }
}

extern "C" int hipengine_gguf_rmsnorm_bf16_f32_weight(
    const uint16_t* x,
    const float* weight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_rmsnorm_bf16_f32_weight_kernel<<<
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(x, weight, out, eps, hidden_size);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_add_rmsnorm_bf16_f32_weight(
    const uint16_t* x,
    const uint16_t* add,
    const float* weight,
    uint16_t* norm_out,
    uint16_t* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  gguf_add_rmsnorm_bf16_f32_weight_kernel<<<
      static_cast<unsigned int>(rows),
      static_cast<unsigned int>(threads),
      static_cast<size_t>(threads) * sizeof(float),
      stream>>>(x, add, weight, norm_out, residual_out, eps, hidden_size);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_bf16_add(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* out,
    int64_t n,
    int64_t threads,
    hipStream_t stream) {
  if (n <= 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const int64_t blocks = (n + threads - 1) / threads;
  gguf_bf16_add_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads), 0, stream>>>(
      a, b, out, n);
  return hipGetLastError();
}

extern "C" int hipengine_gguf_gate_repeat_value_bf16(
    const uint16_t* q_gate,
    const uint16_t* value,
    uint16_t* out,
    int64_t head_count,
    int64_t head_count_kv,
    int64_t head_dim,
    int64_t threads,
    hipStream_t stream) {
  if (head_count <= 0 || head_count_kv <= 0 || head_dim <= 0 || head_count % head_count_kv != 0) {
    return hipErrorInvalidValue;
  }
  if (threads <= 0) {
    threads = 256;
  }
  const int64_t n = head_count * head_dim;
  const int64_t blocks = (n + threads - 1) / threads;
  gguf_gate_repeat_value_bf16_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads), 0, stream>>>(
      q_gate, value, out, head_count, head_count_kv, head_dim);
  return hipGetLastError();
}
