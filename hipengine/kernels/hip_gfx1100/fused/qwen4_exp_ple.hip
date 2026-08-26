// hipEngine-native Qwen3.8-Flash-Next PLE strict primitives.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__global__ void ple_signed_sqrt_gate_f32_kernel(
    const float* __restrict__ key,
    const float* __restrict__ query,
    float* __restrict__ gate,
    int64_t hidden) {
  extern __shared__ float partial[];
  const int64_t group = blockIdx.x;
  const int64_t offset = group * hidden;
  float dot = 0.0f;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    dot += key[offset + column] * query[offset + column];
  }
  partial[threadIdx.x] = dot;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float score = partial[0] * rsqrtf(static_cast<float>(hidden));
    const float transformed = copysignf(sqrtf(fmaxf(fabsf(score), 1.0e-6f)), score);
    gate[group] = 1.0f / (1.0f + expf(-transformed));
  }
}

__global__ void ple_repeat_gated_value_f32_kernel(
    const float* __restrict__ value,
    const float* __restrict__ gate,
    float* __restrict__ output,
    int64_t branches,
    int64_t hidden) {
  const int64_t group = blockIdx.x;
  const int64_t row = group / branches;
  const int64_t offset = group * hidden;
  const float scale = gate[group];
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    output[offset + column] = value[row * hidden + column] * scale;
  }
}

__global__ void ple_dilated_depthwise_conv_f32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ history,
    float* __restrict__ output,
    int64_t rows,
    int64_t channels,
    int64_t kernel_size,
    int64_t dilation,
    int64_t history_rows) {
  const int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }
  for (int64_t row = 0; row < rows; ++row) {
    float value = 0.0f;
    for (int64_t tap = 0; tap < kernel_size; ++tap) {
      const int64_t logical = row - (kernel_size - 1 - tap) * dilation;
      const float input_value = logical >= 0
          ? input[logical * channels + channel]
          : history[(history_rows + logical) * channels + channel];
      value += input_value * weights[channel * kernel_size + tap];
    }
    output[row * channels + channel] = value;
  }
  for (int64_t state_row = 0; state_row < history_rows; ++state_row) {
    const int64_t logical = rows - history_rows + state_row;
    const float value = logical >= 0
        ? input[logical * channels + channel]
        : history[(history_rows + logical) * channels + channel];
    history[state_row * channels + channel] = value;
  }
}

}  // namespace

extern "C" int hipengine_qwen4_exp_ple_signed_sqrt_gate_f32(
    const void* key,
    const void* query,
    void* gate,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    hipStream_t stream) {
  if (!key || !query || !gate || rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      ple_signed_sqrt_gate_f32_kernel,
      dim3(static_cast<uint32_t>(rows * branches)), dim3(threads), threads * sizeof(float), stream,
      static_cast<const float*>(key), static_cast<const float*>(query),
      static_cast<float*>(gate), hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_ple_repeat_gated_value_f32(
    const void* value,
    const void* gate,
    void* output,
    int64_t rows,
    int64_t branches,
    int64_t hidden,
    hipStream_t stream) {
  if (!value || !gate || !output || rows <= 0 || branches <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  hipLaunchKernelGGL(
      ple_repeat_gated_value_f32_kernel,
      dim3(static_cast<uint32_t>(rows * branches)), dim3(threads), 0, stream,
      static_cast<const float*>(value), static_cast<const float*>(gate),
      static_cast<float*>(output), branches, hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_ple_dilated_depthwise_conv_f32(
    const void* input,
    const void* weights,
    void* history,
    void* output,
    int64_t rows,
    int64_t channels,
    int64_t kernel_size,
    int64_t dilation,
    hipStream_t stream) {
  if (!input || !weights || !history || !output || rows <= 0 || channels <= 0 ||
      kernel_size <= 0 || dilation <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const int64_t history_rows = (kernel_size - 1) * dilation;
  const uint32_t blocks = static_cast<uint32_t>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      ple_dilated_depthwise_conv_f32_kernel,
      dim3(blocks), dim3(threads), 0, stream,
      static_cast<const float*>(input), static_cast<const float*>(weights),
      static_cast<float*>(history), static_cast<float*>(output),
      rows, channels, kernel_size, dilation, history_rows);
  return static_cast<int>(hipGetLastError());
}
