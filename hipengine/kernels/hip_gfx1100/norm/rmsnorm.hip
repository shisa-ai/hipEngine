// Ported from nano-vllm-amd @ 59195ed.
// Qwen kernels come from csrc/amd/qwen35_expert.hip.
// PARO out-kernels come from nanovllm/native/qwen35/paroquant_kernels.py.
// Kernel bodies are preserved from the source families; wrappers are hipEngine C ABI.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

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

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ inline uint16_t float_to_scalar<uint16_t>(float value) {
  return float_to_bf16_bits(value);
}

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

__global__ void qwen35_rmsnorm_kernel(
    const uint16_t* hidden_states,
    const uint16_t* weight,
    uint16_t* out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t token = blockIdx.x;
  const int64_t row_offset = token * hidden_size;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(hidden_states[row_offset + idx]);
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

  const float inv_rms = rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(hidden_states[row_offset + idx]);
    const float scale = 1.0f + bf16_bits_to_float(weight[idx]);
    out[row_offset + idx] = float_to_bf16_bits(value * inv_rms * scale);
  }
}

__global__ void qwen35_add_rmsnorm_kernel(
    const uint16_t* hidden_states,
    const uint16_t* residual,
    const uint16_t* weight,
    uint16_t* out,
    uint16_t* residual_out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t token = blockIdx.x;
  const int64_t row_offset = token * hidden_size;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(hidden_states[row_offset + idx])
        + bf16_bits_to_float(residual[row_offset + idx]);
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

  const float inv_rms = rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_bits_to_float(hidden_states[row_offset + idx])
        + bf16_bits_to_float(residual[row_offset + idx]);
    const float scale = 1.0f + bf16_bits_to_float(weight[idx]);
    residual_out[row_offset + idx] = float_to_bf16_bits(value);
    out[row_offset + idx] = float_to_bf16_bits(value * inv_rms * scale);
  }
}

__global__ void qwen35_add_rmsnorm_f32_kernel(
    const float* hidden_states,
    const uint16_t* residual,
    const uint16_t* weight,
    uint16_t* out,
    uint16_t* residual_out,
    float eps,
    int64_t hidden_size) {
  extern __shared__ float partial[];
  const int64_t token = blockIdx.x;
  const int64_t row_offset = token * hidden_size;
  float sumsq = 0.0f;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t idx = threadIdx.x * 8; idx + 7 < hidden_size; idx += vec_stride) {
    const int64_t off = row_offset + idx;
    const float v0 = hidden_states[off + 0] + bf16_bits_to_float(residual[off + 0]);
    const float v1 = hidden_states[off + 1] + bf16_bits_to_float(residual[off + 1]);
    const float v2 = hidden_states[off + 2] + bf16_bits_to_float(residual[off + 2]);
    const float v3 = hidden_states[off + 3] + bf16_bits_to_float(residual[off + 3]);
    const float v4 = hidden_states[off + 4] + bf16_bits_to_float(residual[off + 4]);
    const float v5 = hidden_states[off + 5] + bf16_bits_to_float(residual[off + 5]);
    const float v6 = hidden_states[off + 6] + bf16_bits_to_float(residual[off + 6]);
    const float v7 = hidden_states[off + 7] + bf16_bits_to_float(residual[off + 7]);
    sumsq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4 + v5 * v5 + v6 * v6 + v7 * v7;
  }
  for (int64_t idx = (hidden_size & ~7) + threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = hidden_states[row_offset + idx] + bf16_bits_to_float(residual[row_offset + idx]);
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

  const float inv_rms = rsqrtf((partial[0] / static_cast<float>(hidden_size)) + eps);
  for (int64_t idx = threadIdx.x * 8; idx + 7 < hidden_size; idx += vec_stride) {
    const int64_t off = row_offset + idx;
    const float w0 = 1.0f + bf16_bits_to_float(weight[idx + 0]);
    const float w1 = 1.0f + bf16_bits_to_float(weight[idx + 1]);
    const float w2 = 1.0f + bf16_bits_to_float(weight[idx + 2]);
    const float w3 = 1.0f + bf16_bits_to_float(weight[idx + 3]);
    const float w4 = 1.0f + bf16_bits_to_float(weight[idx + 4]);
    const float w5 = 1.0f + bf16_bits_to_float(weight[idx + 5]);
    const float w6 = 1.0f + bf16_bits_to_float(weight[idx + 6]);
    const float w7 = 1.0f + bf16_bits_to_float(weight[idx + 7]);
    const float v0 = hidden_states[off + 0] + bf16_bits_to_float(residual[off + 0]);
    const float v1 = hidden_states[off + 1] + bf16_bits_to_float(residual[off + 1]);
    const float v2 = hidden_states[off + 2] + bf16_bits_to_float(residual[off + 2]);
    const float v3 = hidden_states[off + 3] + bf16_bits_to_float(residual[off + 3]);
    const float v4 = hidden_states[off + 4] + bf16_bits_to_float(residual[off + 4]);
    const float v5 = hidden_states[off + 5] + bf16_bits_to_float(residual[off + 5]);
    const float v6 = hidden_states[off + 6] + bf16_bits_to_float(residual[off + 6]);
    const float v7 = hidden_states[off + 7] + bf16_bits_to_float(residual[off + 7]);
    residual_out[off + 0] = float_to_bf16_bits(v0);
    residual_out[off + 1] = float_to_bf16_bits(v1);
    residual_out[off + 2] = float_to_bf16_bits(v2);
    residual_out[off + 3] = float_to_bf16_bits(v3);
    residual_out[off + 4] = float_to_bf16_bits(v4);
    residual_out[off + 5] = float_to_bf16_bits(v5);
    residual_out[off + 6] = float_to_bf16_bits(v6);
    residual_out[off + 7] = float_to_bf16_bits(v7);
    out[off + 0] = float_to_bf16_bits(v0 * inv_rms * w0);
    out[off + 1] = float_to_bf16_bits(v1 * inv_rms * w1);
    out[off + 2] = float_to_bf16_bits(v2 * inv_rms * w2);
    out[off + 3] = float_to_bf16_bits(v3 * inv_rms * w3);
    out[off + 4] = float_to_bf16_bits(v4 * inv_rms * w4);
    out[off + 5] = float_to_bf16_bits(v5 * inv_rms * w5);
    out[off + 6] = float_to_bf16_bits(v6 * inv_rms * w6);
    out[off + 7] = float_to_bf16_bits(v7 * inv_rms * w7);
  }
  for (int64_t idx = (hidden_size & ~7) + threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = hidden_states[row_offset + idx] + bf16_bits_to_float(residual[row_offset + idx]);
    const float scale = 1.0f + bf16_bits_to_float(weight[idx]);
    residual_out[row_offset + idx] = float_to_bf16_bits(value);
    out[row_offset + idx] = float_to_bf16_bits(value * inv_rms * scale);
  }
}

__global__ void qwen35_head_rmsnorm_kernel(
    const float* hidden_states,
    const uint16_t* weight,
    float* out,
    float eps,
    int64_t head_dim) {
  extern __shared__ float partial[];
  const int64_t head = blockIdx.x;
  const int64_t offset = head * head_dim;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < head_dim; idx += blockDim.x) {
    const float value = hidden_states[offset + idx];
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
  for (int64_t idx = threadIdx.x; idx < head_dim; idx += blockDim.x) {
    const float scale = 1.0f + bf16_bits_to_float(weight[idx]);
    out[offset + idx] = hidden_states[offset + idx] * inv_rms * scale;
  }
}

template <typename scalar_t>
__global__ void paro_rmsnorm_out_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    float eps,
    int64_t rows,
    int64_t hidden) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t row_base = row * hidden;

  float sumsq = 0.0f;
  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (hidden / vec_stride) * vec_stride;
  for (int64_t idx = static_cast<int64_t>(tid) * 8; idx < main_end; idx += vec_stride) {
    const int64_t off = row_base + idx;
    const float v0 = scalar_to_float(x[off + 0]);
    const float v1 = scalar_to_float(x[off + 1]);
    const float v2 = scalar_to_float(x[off + 2]);
    const float v3 = scalar_to_float(x[off + 3]);
    const float v4 = scalar_to_float(x[off + 4]);
    const float v5 = scalar_to_float(x[off + 5]);
    const float v6 = scalar_to_float(x[off + 6]);
    const float v7 = scalar_to_float(x[off + 7]);
    sumsq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4 + v5 * v5 + v6 * v6 + v7 * v7;
  }
  for (int64_t idx = main_end + tid; idx < hidden; idx += blockDim.x) {
    const float value = scalar_to_float(x[row_base + idx]);
    sumsq += value * value;
  }

  __shared__ float partial[256];
  partial[tid] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
  for (int64_t idx = static_cast<int64_t>(tid) * 8; idx < main_end; idx += vec_stride) {
    const int64_t off = row_base + idx;
    out[off + 0] = float_to_scalar<scalar_t>(scalar_to_float(x[off + 0]) * inv_rms * scalar_to_float(weight[idx + 0]));
    out[off + 1] = float_to_scalar<scalar_t>(scalar_to_float(x[off + 1]) * inv_rms * scalar_to_float(weight[idx + 1]));
    out[off + 2] = float_to_scalar<scalar_t>(scalar_to_float(x[off + 2]) * inv_rms * scalar_to_float(weight[idx + 2]));
    out[off + 3] = float_to_scalar<scalar_t>(scalar_to_float(x[off + 3]) * inv_rms * scalar_to_float(weight[idx + 3]));
    out[off + 4] = float_to_scalar<scalar_t>(scalar_to_float(x[off + 4]) * inv_rms * scalar_to_float(weight[idx + 4]));
    out[off + 5] = float_to_scalar<scalar_t>(scalar_to_float(x[off + 5]) * inv_rms * scalar_to_float(weight[idx + 5]));
    out[off + 6] = float_to_scalar<scalar_t>(scalar_to_float(x[off + 6]) * inv_rms * scalar_to_float(weight[idx + 6]));
    out[off + 7] = float_to_scalar<scalar_t>(scalar_to_float(x[off + 7]) * inv_rms * scalar_to_float(weight[idx + 7]));
  }
  for (int64_t idx = main_end + tid; idx < hidden; idx += blockDim.x) {
    out[row_base + idx] = float_to_scalar<scalar_t>(
        scalar_to_float(x[row_base + idx]) * inv_rms * scalar_to_float(weight[idx]));
  }
}

template <typename scalar_t>
__device__ inline float rounded_residual_sum(scalar_t x, scalar_t add) {
  return scalar_to_float(float_to_scalar<scalar_t>(scalar_to_float(x) + scalar_to_float(add)));
}

template <typename scalar_t>
__global__ void paro_add_rmsnorm_out_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ add,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ norm_out,
    scalar_t* __restrict__ residual_out,
    float eps,
    int64_t rows,
    int64_t hidden) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t row_base = row * hidden;

  float sumsq = 0.0f;
  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (hidden / vec_stride) * vec_stride;
  for (int64_t idx = static_cast<int64_t>(tid) * 8; idx < main_end; idx += vec_stride) {
    const int64_t off = row_base + idx;
    const float v0 = rounded_residual_sum(x[off + 0], add[off + 0]);
    const float v1 = rounded_residual_sum(x[off + 1], add[off + 1]);
    const float v2 = rounded_residual_sum(x[off + 2], add[off + 2]);
    const float v3 = rounded_residual_sum(x[off + 3], add[off + 3]);
    const float v4 = rounded_residual_sum(x[off + 4], add[off + 4]);
    const float v5 = rounded_residual_sum(x[off + 5], add[off + 5]);
    const float v6 = rounded_residual_sum(x[off + 6], add[off + 6]);
    const float v7 = rounded_residual_sum(x[off + 7], add[off + 7]);
    sumsq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4 + v5 * v5 + v6 * v6 + v7 * v7;
  }
  for (int64_t idx = main_end + tid; idx < hidden; idx += blockDim.x) {
    const float value = rounded_residual_sum(x[row_base + idx], add[row_base + idx]);
    sumsq += value * value;
  }

  __shared__ float partial[256];
  partial[tid] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
  for (int64_t idx = static_cast<int64_t>(tid) * 8; idx < main_end; idx += vec_stride) {
    const int64_t off = row_base + idx;
    const float v0 = rounded_residual_sum(x[off + 0], add[off + 0]);
    const float v1 = rounded_residual_sum(x[off + 1], add[off + 1]);
    const float v2 = rounded_residual_sum(x[off + 2], add[off + 2]);
    const float v3 = rounded_residual_sum(x[off + 3], add[off + 3]);
    const float v4 = rounded_residual_sum(x[off + 4], add[off + 4]);
    const float v5 = rounded_residual_sum(x[off + 5], add[off + 5]);
    const float v6 = rounded_residual_sum(x[off + 6], add[off + 6]);
    const float v7 = rounded_residual_sum(x[off + 7], add[off + 7]);
    residual_out[off + 0] = float_to_scalar<scalar_t>(v0);
    residual_out[off + 1] = float_to_scalar<scalar_t>(v1);
    residual_out[off + 2] = float_to_scalar<scalar_t>(v2);
    residual_out[off + 3] = float_to_scalar<scalar_t>(v3);
    residual_out[off + 4] = float_to_scalar<scalar_t>(v4);
    residual_out[off + 5] = float_to_scalar<scalar_t>(v5);
    residual_out[off + 6] = float_to_scalar<scalar_t>(v6);
    residual_out[off + 7] = float_to_scalar<scalar_t>(v7);
    norm_out[off + 0] = float_to_scalar<scalar_t>(v0 * inv_rms * scalar_to_float(weight[idx + 0]));
    norm_out[off + 1] = float_to_scalar<scalar_t>(v1 * inv_rms * scalar_to_float(weight[idx + 1]));
    norm_out[off + 2] = float_to_scalar<scalar_t>(v2 * inv_rms * scalar_to_float(weight[idx + 2]));
    norm_out[off + 3] = float_to_scalar<scalar_t>(v3 * inv_rms * scalar_to_float(weight[idx + 3]));
    norm_out[off + 4] = float_to_scalar<scalar_t>(v4 * inv_rms * scalar_to_float(weight[idx + 4]));
    norm_out[off + 5] = float_to_scalar<scalar_t>(v5 * inv_rms * scalar_to_float(weight[idx + 5]));
    norm_out[off + 6] = float_to_scalar<scalar_t>(v6 * inv_rms * scalar_to_float(weight[idx + 6]));
    norm_out[off + 7] = float_to_scalar<scalar_t>(v7 * inv_rms * scalar_to_float(weight[idx + 7]));
  }
  for (int64_t idx = main_end + tid; idx < hidden; idx += blockDim.x) {
    const int64_t off = row_base + idx;
    const float value = rounded_residual_sum(x[off], add[off]);
    residual_out[off] = float_to_scalar<scalar_t>(value);
    norm_out[off] = float_to_scalar<scalar_t>(value * inv_rms * scalar_to_float(weight[idx]));
  }
}

constexpr int kRmsNormThreads = 256;

}  // namespace

extern "C" int hipengine_qwen35_rmsnorm_bf16(
    const uint16_t* hidden_states,
    const uint16_t* weight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_rmsnorm_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kRmsNormThreads),
      kRmsNormThreads * sizeof(float),
      stream,
      hidden_states,
      weight,
      out,
      eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_add_rmsnorm_bf16(
    const uint16_t* hidden_states,
    const uint16_t* residual,
    const uint16_t* weight,
    uint16_t* out,
    uint16_t* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_add_rmsnorm_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kRmsNormThreads),
      kRmsNormThreads * sizeof(float),
      stream,
      hidden_states,
      residual,
      weight,
      out,
      residual_out,
      eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_add_rmsnorm_f32_bf16(
    const float* hidden_states,
    const uint16_t* residual,
    const uint16_t* weight,
    uint16_t* out,
    uint16_t* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_add_rmsnorm_f32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kRmsNormThreads),
      kRmsNormThreads * sizeof(float),
      stream,
      hidden_states,
      residual,
      weight,
      out,
      residual_out,
      eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_head_rmsnorm_f32_bf16(
    const float* hidden_states,
    const uint16_t* weight,
    float* out,
    int64_t heads,
    int64_t head_dim,
    float eps,
    hipStream_t stream) {
  if (heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      qwen35_head_rmsnorm_kernel,
      dim3(static_cast<unsigned int>(heads)),
      dim3(kRmsNormThreads),
      kRmsNormThreads * sizeof(float),
      stream,
      hidden_states,
      weight,
      out,
      eps,
      head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_paro_rmsnorm_out_bf16(
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      paro_rmsnorm_out_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kRmsNormThreads),
      0,
      stream,
      x,
      weight,
      out,
      eps,
      rows,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_paro_add_rmsnorm_out_bf16(
    const uint16_t* x,
    const uint16_t* add,
    const uint16_t* weight,
    uint16_t* norm_out,
    uint16_t* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      paro_add_rmsnorm_out_kernel<uint16_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kRmsNormThreads),
      0,
      stream,
      x,
      add,
      weight,
      norm_out,
      residual_out,
      eps,
      rows,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_paro_rmsnorm_out_fp16(
    const half_t* x,
    const half_t* weight,
    half_t* out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      paro_rmsnorm_out_kernel<half_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kRmsNormThreads),
      0,
      stream,
      x,
      weight,
      out,
      eps,
      rows,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_paro_add_rmsnorm_out_fp16(
    const half_t* x,
    const half_t* add,
    const half_t* weight,
    half_t* norm_out,
    half_t* residual_out,
    int64_t rows,
    int64_t hidden_size,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden_size <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      paro_add_rmsnorm_out_kernel<half_t>,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kRmsNormThreads),
      0,
      stream,
      x,
      add,
      weight,
      norm_out,
      residual_out,
      eps,
      rows,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}
