// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip @ 59195ed.
// Qwen3.5 linear-attention decode convolution kernel bodies are preserved; wrappers are HIPENGINE C ABI.

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

__device__ inline float silu_f32(float value) {
  return value / (1.0f + expf(-value));
}

template <typename scalar_t>
__device__ inline float scalar_to_float_qwen35(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float_qwen35<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

__global__ void qwen35_linear_attn_conv_decode_kernel(
    const float* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }

  const int64_t offset = channel * kernel_size;
  float acc = 0.0f;
  for (int64_t idx = 0; idx < kernel_size - 1; ++idx) {
    const float value = conv_state[offset + idx + 1];
    acc += value * conv_weight[offset + idx];
    conv_state[offset + idx] = value;
  }
  const float newest = hidden_states[channel];
  acc += newest * conv_weight[offset + kernel_size - 1];
  conv_state[offset + kernel_size - 1] = newest;
  out[channel] = silu_f32(acc);
}

template <typename scalar_t>
__global__ void qwen35_linear_attn_conv_decode_lowp_kernel(
    const scalar_t* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }

  const int64_t offset = channel * kernel_size;
  float acc = 0.0f;
  for (int64_t idx = 0; idx < kernel_size - 1; ++idx) {
    const float value = conv_state[offset + idx + 1];
    acc += value * conv_weight[offset + idx];
    conv_state[offset + idx] = value;
  }
  const float newest = scalar_to_float_qwen35(hidden_states[channel]);
  acc += newest * conv_weight[offset + kernel_size - 1];
  conv_state[offset + kernel_size - 1] = newest;
  out[channel] = silu_f32(acc);
}


bool valid_conv_shape(int64_t channels, int64_t kernel_size) {
  return channels > 0 && kernel_size > 0;
}

}  // namespace

extern "C" int hipengine_qwen35_linear_attn_conv_decode_f32(
    const float* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (!valid_conv_shape(channels, kernel_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      qwen35_linear_attn_conv_decode_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      conv_weight,
      out,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen35_linear_attn_conv_decode_bf16(
    const uint16_t* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (!valid_conv_shape(channels, kernel_size)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((channels + threads - 1) / threads);
  hipLaunchKernelGGL(
      (qwen35_linear_attn_conv_decode_lowp_kernel<uint16_t>),
      dim3(static_cast<unsigned int>(blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      conv_weight,
      out,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}
