// hipEngine-native Qwen3.8-Flash-Next gated-residual strict primitives.
// No external kernel body is ported; formulas follow the frozen qwen4exp oracle.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union { uint32_t u32; float f32; } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16_bits(float value) {
  union { float f32; uint32_t u32; } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__global__ void repeat_bf16_branches_kernel(
    const uint16_t* __restrict__ input,
    uint16_t* __restrict__ output,
    int64_t rows,
    int64_t branches,
    int64_t hidden) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = rows * branches * hidden;
  if (index < elements) {
    const int64_t row = index / (branches * hidden);
    output[index] = input[row * hidden + index % hidden];
  }
}

__global__ void grouped_rmsnorm_bf16_f32_kernel(
    const uint16_t* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t branches,
    int64_t hidden,
    float eps) {
  extern __shared__ float partial[];
  const int64_t group = blockIdx.x;
  const int64_t offset = group * hidden;
  float sumsq = 0.0f;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float value = bf16_bits_to_float(residual[offset + column]);
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
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
  const int64_t branch = group % branches;
  const int64_t weight_offset = branch * hidden;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    output[offset + column] =
        bf16_bits_to_float(residual[offset + column]) * inv_rms * weight[weight_offset + column];
  }
}

__global__ void grouped_rmsnorm_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t branches,
    int64_t hidden,
    float eps) {
  extern __shared__ float partial[];
  const int64_t group = blockIdx.x;
  const int64_t offset = group * hidden;
  float sumsq = 0.0f;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float value = input[offset + column];
    sumsq += value * value;
  }
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
  const int64_t branch = group % branches;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    output[offset + column] = input[offset + column] * inv_rms
        * weight[branch * hidden + column];
  }
}

__global__ void gated_mean_f32_kernel(
    const float* __restrict__ normalized,
    const float* __restrict__ gate,
    float* __restrict__ output,
    int64_t branches,
    int64_t hidden) {
  const int64_t row = blockIdx.x;
  const int64_t residual_width = branches * hidden;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    float value = 0.0f;
    for (int64_t branch = 0; branch < branches; ++branch) {
      const int64_t index = row * residual_width + branch * hidden + column;
      value += normalized[index] * gate[index];
    }
    output[row * hidden + column] = value / static_cast<float>(branches);
  }
}

__global__ void gated_mean_sigmoid_f32_kernel(
    const float* __restrict__ normalized,
    float* __restrict__ gate_logits,
    float* __restrict__ output,
    int64_t branches,
    int64_t hidden) {
  const int64_t column_tiles = (hidden + blockDim.x - 1) / blockDim.x;
  const int64_t row = blockIdx.x / column_tiles;
  const int64_t column = (blockIdx.x % column_tiles) * blockDim.x + threadIdx.x;
  if (column < hidden) {
    const int64_t residual_width = branches * hidden;
    float value = 0.0f;
    for (int64_t branch = 0; branch < branches; ++branch) {
      const int64_t index = row * residual_width + branch * hidden + column;
      const float gate = 1.0f / (1.0f + expf(-gate_logits[index]));
      gate_logits[index] = gate;
      value += normalized[index] * gate;
    }
    output[row * hidden + column] = value / static_cast<float>(branches);
  }
}

__global__ void gr_write_bf16_f32_kernel(
    const uint16_t* __restrict__ residual,
    const float* __restrict__ block_output,
    const float* __restrict__ inject_logits,
    uint16_t* __restrict__ output,
    int64_t branches,
    int64_t hidden) {
  const int64_t group = blockIdx.x;
  const int64_t row = group / branches;
  const int64_t branch = group % branches;
  const float inject = inject_logits[row * branches + branch] / static_cast<float>(branches);
  const float scatter = 2.0f / (1.0f + expf(-inject));
  const int64_t offset = group * hidden;
  const int64_t block_offset = row * hidden;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float value = bf16_bits_to_float(residual[offset + column])
        + scatter * block_output[block_offset + column];
    output[offset + column] = float_to_bf16_bits(value);
  }
}

__global__ void scaled_silu_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t elements,
    float scale) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float value = input[index] * scale;
    output[index] = value / (1.0f + expf(-value));
  }
}

__global__ void silu_mul_f32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int64_t elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float value = gate[index];
    output[index] = value / (1.0f + expf(-value)) * up[index];
  }
}

__global__ void sigmoid_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = 1.0f / (1.0f + expf(-input[index]));
  }
}

__global__ void sigmoid_gated_rmsnorm_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ gate,
    float* __restrict__ output,
    int64_t head_dim,
    float eps) {
  extern __shared__ float partial[];
  const int64_t head = blockIdx.x;
  const int64_t offset = head * head_dim;
  float sumsq = 0.0f;
  for (int64_t column = threadIdx.x; column < head_dim; column += blockDim.x) {
    const float value = input[offset + column];
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
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  for (int64_t column = threadIdx.x; column < head_dim; column += blockDim.x) {
    const int64_t index = offset + column;
    const float sigmoid_gate = 1.0f / (1.0f + expf(-gate[index]));
    output[index] = input[index] * inv_rms * weight[column] * sigmoid_gate;
  }
}

}  // namespace

extern "C" int hipengine_qwen4_exp_repeat_bf16_branches(
    const void* input,
    void* output,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    hipStream_t stream) {
  if (!input || !output || rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const int64_t elements = rows * branches * hidden;
  hipLaunchKernelGGL(
      repeat_bf16_branches_kernel,
      dim3(static_cast<uint32_t>((elements + threads - 1) / threads)), dim3(threads), 0, stream,
      static_cast<const uint16_t*>(input), static_cast<uint16_t*>(output), rows, branches, hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_grouped_rmsnorm_bf16_f32(
    const void* residual,
    const void* weight,
    void* output,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    float eps,
    hipStream_t stream) {
  if (!residual || !weight || !output || rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      grouped_rmsnorm_bf16_f32_kernel,
      dim3(static_cast<uint32_t>(rows * branches)), dim3(threads), threads * sizeof(float), stream,
      static_cast<const uint16_t*>(residual), static_cast<const float*>(weight),
      static_cast<float*>(output), branches, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_grouped_rmsnorm_f32(
    const void* input,
    const void* weight,
    void* output,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    float eps,
    hipStream_t stream) {
  if (!input || !weight || !output || rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      grouped_rmsnorm_f32_kernel,
      dim3(static_cast<uint32_t>(rows * branches)), dim3(threads), threads * sizeof(float), stream,
      static_cast<const float*>(input), static_cast<const float*>(weight),
      static_cast<float*>(output), branches, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_gated_mean_f32(
    const void* normalized,
    const void* gate,
    void* output,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    hipStream_t stream) {
  if (!normalized || !gate || !output || rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      gated_mean_f32_kernel,
      dim3(static_cast<uint32_t>(rows)), dim3(threads), 0, stream,
      static_cast<const float*>(normalized), static_cast<const float*>(gate),
      static_cast<float*>(output), branches, hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_gated_mean_sigmoid_f32(
    const void* normalized,
    void* gate_logits,
    void* output,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    hipStream_t stream) {
  if (!normalized || !gate_logits || !output ||
      rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const uint32_t column_tiles = static_cast<uint32_t>((hidden + threads - 1) / threads);
  hipLaunchKernelGGL(
      gated_mean_sigmoid_f32_kernel,
      dim3(static_cast<uint32_t>(rows) * column_tiles), dim3(threads), 0, stream,
      static_cast<const float*>(normalized), static_cast<float*>(gate_logits),
      static_cast<float*>(output), branches, hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_gr_write_bf16_f32(
    const void* residual,
    const void* block_output,
    const void* inject_logits,
    void* output,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    hipStream_t stream) {
  if (!residual || !block_output || !inject_logits || !output ||
      rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      gr_write_bf16_f32_kernel,
      dim3(static_cast<uint32_t>(rows * branches)), dim3(threads), 0, stream,
      static_cast<const uint16_t*>(residual), static_cast<const float*>(block_output),
      static_cast<const float*>(inject_logits), static_cast<uint16_t*>(output),
      branches, hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_scaled_silu_f32(
    const void* input,
    void* output,
    int64_t elements,
    float scale,
    hipStream_t stream) {
  if (!input || !output || elements <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const uint32_t blocks = static_cast<uint32_t>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      scaled_silu_f32_kernel, dim3(blocks), dim3(threads), 0, stream,
      static_cast<const float*>(input), static_cast<float*>(output), elements, scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_silu_mul_f32(
    const void* gate,
    const void* up,
    void* output,
    int64_t elements,
    hipStream_t stream) {
  if (!gate || !up || !output || elements <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const uint32_t blocks = static_cast<uint32_t>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      silu_mul_f32_kernel, dim3(blocks), dim3(threads), 0, stream,
      static_cast<const float*>(gate), static_cast<const float*>(up),
      static_cast<float*>(output), elements);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_sigmoid_f32(
    const void* input,
    void* output,
    int64_t elements,
    hipStream_t stream) {
  if (!input || !output || elements <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const uint32_t blocks = static_cast<uint32_t>((elements + threads - 1) / threads);
  hipLaunchKernelGGL(
      sigmoid_f32_kernel, dim3(blocks), dim3(threads), 0, stream,
      static_cast<const float*>(input), static_cast<float*>(output), elements);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_sigmoid_gated_rmsnorm_f32(
    const void* input,
    const void* weight,
    const void* gate,
    void* output,
    int64_t heads,
    int64_t head_dim,
    float eps,
    hipStream_t stream) {
  if (!input || !weight || !gate || !output || heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      sigmoid_gated_rmsnorm_f32_kernel,
      dim3(static_cast<uint32_t>(heads)), dim3(threads), threads * sizeof(float), stream,
      static_cast<const float*>(input), static_cast<const float*>(weight),
      static_cast<const float*>(gate), static_cast<float*>(output), head_dim, eps);
  return static_cast<int>(hipGetLastError());
}
