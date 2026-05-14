// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip @ 59195ed.
// Qwen3.5 linear-attention decode convolution kernel bodies are preserved; wrappers are hipENGINE C ABI.

#include <hip/hip_runtime.h>
#include <stdint.h>

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


__global__ void qwen35_linear_attn_conv_prefill_kernel(
    const float* __restrict__ hidden_states,
    const float* __restrict__ conv_state,
    const float* __restrict__ conv_weight,
    float* __restrict__ out,
    int64_t tokens,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t size = tokens * channels;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < size;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t token = idx / channels;
    const int64_t channel = idx - token * channels;
    const int64_t offset = channel * kernel_size;
    float acc = 0.0f;
    if (kernel_size == 4) {
      const int64_t padded0 = token;
      const int64_t padded1 = token + 1;
      const int64_t padded2 = token + 2;
      const int64_t padded3 = token + 3;
      const float value0 = (
          padded0 < 3 ? conv_state[offset + padded0 + 1] : hidden_states[(padded0 - 3) * channels + channel]);
      const float value1 = (
          padded1 < 3 ? conv_state[offset + padded1 + 1] : hidden_states[(padded1 - 3) * channels + channel]);
      const float value2 = (
          padded2 < 3 ? conv_state[offset + padded2 + 1] : hidden_states[(padded2 - 3) * channels + channel]);
      const float value3 = hidden_states[(padded3 - 3) * channels + channel];
      volatile float product0 = value0 * conv_weight[offset];
      volatile float product1 = value1 * conv_weight[offset + 1];
      volatile float product2 = value2 * conv_weight[offset + 2];
      volatile float product3 = value3 * conv_weight[offset + 3];
      acc = (product0 + product1) + (product2 + product3);
    } else {
      for (int64_t k = 0; k < kernel_size; ++k) {
        const int64_t padded = token + k;
        const float value = (
            padded < kernel_size - 1
                ? conv_state[offset + padded + 1]
                : hidden_states[(padded - (kernel_size - 1)) * channels + channel]);
        volatile float product = value * conv_weight[offset + k];
        acc += product;
      }
    }
    out[idx] = silu_f32(acc);
  }
}

__global__ void qwen35_linear_attn_conv_prefill_state_kernel(
    const float* __restrict__ hidden_states,
    float* __restrict__ conv_state,
    int64_t tokens,
    int64_t channels,
    int64_t kernel_size) {
  const int64_t size = channels * kernel_size;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < size;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t channel = idx / kernel_size;
    const int64_t k = idx - channel * kernel_size;
    const int64_t offset = channel * kernel_size;
    if (tokens >= kernel_size) {
      conv_state[offset + k] = hidden_states[(tokens - kernel_size + k) * channels + channel];
    } else {
      const int64_t kept = kernel_size - tokens;
      conv_state[offset + k] = (
          k < kept
              ? conv_state[offset + k + tokens]
              : hidden_states[(k - kept) * channels + channel]);
    }
  }
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

extern "C" int hipengine_qwen35_linear_attn_conv_decode_fp16(
    const half_t* hidden_states,
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
      (qwen35_linear_attn_conv_decode_lowp_kernel<half_t>),
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

extern "C" int hipengine_qwen35_linear_attn_conv_prefill_f32(
    const float* hidden_states,
    float* conv_state,
    const float* conv_weight,
    float* out,
    int64_t tokens,
    int64_t channels,
    int64_t kernel_size,
    hipStream_t stream) {
  if (tokens <= 0 || !valid_conv_shape(channels, kernel_size) || tokens < kernel_size) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int64_t output_size = tokens * channels;
  const int64_t output_blocks64 = (output_size + threads - 1) / threads;
  const int output_blocks = static_cast<int>(output_blocks64 < 65535 ? output_blocks64 : 65535);
  hipLaunchKernelGGL(
      qwen35_linear_attn_conv_prefill_kernel,
      dim3(static_cast<unsigned int>(output_blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      conv_weight,
      out,
      tokens,
      channels,
      kernel_size);
  int err = static_cast<int>(hipGetLastError());
  if (err != static_cast<int>(hipSuccess)) {
    return err;
  }
  const int64_t state_size = channels * kernel_size;
  const int64_t state_blocks64 = (state_size + threads - 1) / threads;
  const int state_blocks = static_cast<int>(state_blocks64 < 65535 ? state_blocks64 : 65535);
  hipLaunchKernelGGL(
      qwen35_linear_attn_conv_prefill_state_kernel,
      dim3(static_cast<unsigned int>(state_blocks)),
      dim3(threads),
      0,
      stream,
      hidden_states,
      conv_state,
      tokens,
      channels,
      kernel_size);
  return static_cast<int>(hipGetLastError());
}
