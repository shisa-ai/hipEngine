// Ported from nano-vllm-amd/csrc/amd/qwen35_expert.hip @ 59195ed.
// W8A16 linear kernel bodies are preserved; wrappers are hipENGINE C ABI.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
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
__device__ inline float scalar_to_float_qwen35(scalar_t value) {
  return static_cast<float>(value);
}

__global__ void w8a16_linear_kernel(
    const uint16_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    float* out,
    int64_t hidden_size,
    int64_t out_features) {
  extern __shared__ float partial[];
  const int64_t row = blockIdx.x;
  const int64_t token = blockIdx.y;
  if (row >= out_features) {
    return;
  }
  float acc = 0.0f;
  const int64_t hidden_row = token * hidden_size;
  const int64_t weight_row = row * hidden_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
    const int64_t h_off = hidden_row + k;
    const int64_t w_off = weight_row + k;
    acc += bf16_bits_to_float(hidden_states[h_off + 0]) * static_cast<float>(static_cast<int>(weight[w_off + 0]));
    acc += bf16_bits_to_float(hidden_states[h_off + 1]) * static_cast<float>(static_cast<int>(weight[w_off + 1]));
    acc += bf16_bits_to_float(hidden_states[h_off + 2]) * static_cast<float>(static_cast<int>(weight[w_off + 2]));
    acc += bf16_bits_to_float(hidden_states[h_off + 3]) * static_cast<float>(static_cast<int>(weight[w_off + 3]));
    acc += bf16_bits_to_float(hidden_states[h_off + 4]) * static_cast<float>(static_cast<int>(weight[w_off + 4]));
    acc += bf16_bits_to_float(hidden_states[h_off + 5]) * static_cast<float>(static_cast<int>(weight[w_off + 5]));
    acc += bf16_bits_to_float(hidden_states[h_off + 6]) * static_cast<float>(static_cast<int>(weight[w_off + 6]));
    acc += bf16_bits_to_float(hidden_states[h_off + 7]) * static_cast<float>(static_cast<int>(weight[w_off + 7]));
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
    acc += bf16_bits_to_float(hidden_states[hidden_row + k]) *
        static_cast<float>(static_cast<int>(weight[weight_row + k]));
  }
  partial[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[token * out_features + row] = partial[0] * weight_scale[row];
  }
}


template <typename scalar_t>
__global__ void w8a16_linear_lowp_out_kernel(
    const scalar_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    scalar_t* out,
    int64_t hidden_size,
    int64_t out_features) {
  extern __shared__ float partial[];
  const int64_t row = blockIdx.x;
  const int64_t token = blockIdx.y;
  if (row >= out_features) {
    return;
  }
  float acc = 0.0f;
  const int64_t hidden_row = token * hidden_size;
  const int64_t weight_row = row * hidden_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
    const int64_t h_off = hidden_row + k;
    const int64_t w_off = weight_row + k;
    acc += scalar_to_float_qwen35(hidden_states[h_off + 0]) * static_cast<float>(static_cast<int>(weight[w_off + 0]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 1]) * static_cast<float>(static_cast<int>(weight[w_off + 1]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 2]) * static_cast<float>(static_cast<int>(weight[w_off + 2]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 3]) * static_cast<float>(static_cast<int>(weight[w_off + 3]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 4]) * static_cast<float>(static_cast<int>(weight[w_off + 4]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 5]) * static_cast<float>(static_cast<int>(weight[w_off + 5]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 6]) * static_cast<float>(static_cast<int>(weight[w_off + 6]));
    acc += scalar_to_float_qwen35(hidden_states[h_off + 7]) * static_cast<float>(static_cast<int>(weight[w_off + 7]));
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
    acc += scalar_to_float_qwen35(hidden_states[hidden_row + k]) *
        static_cast<float>(static_cast<int>(weight[weight_row + k]));
  }
  partial[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[token * out_features + row] = static_cast<scalar_t>(partial[0] * weight_scale[row]);
  }
}


__global__ void w8a16_linear_f32_kernel(
    const float* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    float* out,
    int64_t hidden_size,
    int64_t out_features) {
  extern __shared__ float partial[];
  const int64_t row = blockIdx.x;
  const int64_t token = blockIdx.y;
  if (row >= out_features) {
    return;
  }
  float acc = 0.0f;
  const int64_t hidden_row = token * hidden_size;
  const int64_t weight_row = row * hidden_size;
  const int64_t vec_stride = blockDim.x * 8;
  for (int64_t k = threadIdx.x * 8; k + 7 < hidden_size; k += vec_stride) {
    const int64_t h_off = hidden_row + k;
    const int64_t w_off = weight_row + k;
    acc += hidden_states[h_off + 0] * static_cast<float>(static_cast<int>(weight[w_off + 0]));
    acc += hidden_states[h_off + 1] * static_cast<float>(static_cast<int>(weight[w_off + 1]));
    acc += hidden_states[h_off + 2] * static_cast<float>(static_cast<int>(weight[w_off + 2]));
    acc += hidden_states[h_off + 3] * static_cast<float>(static_cast<int>(weight[w_off + 3]));
    acc += hidden_states[h_off + 4] * static_cast<float>(static_cast<int>(weight[w_off + 4]));
    acc += hidden_states[h_off + 5] * static_cast<float>(static_cast<int>(weight[w_off + 5]));
    acc += hidden_states[h_off + 6] * static_cast<float>(static_cast<int>(weight[w_off + 6]));
    acc += hidden_states[h_off + 7] * static_cast<float>(static_cast<int>(weight[w_off + 7]));
  }
  for (int64_t k = (hidden_size & ~7) + threadIdx.x; k < hidden_size; k += blockDim.x) {
    acc += hidden_states[hidden_row + k] * static_cast<float>(static_cast<int>(weight[weight_row + k]));
  }
  partial[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[token * out_features + row] = partial[0] * weight_scale[row];
  }
}


bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256 || threads == 512;
}

}  // namespace

extern "C" int hipengine_w8a16_linear_bf16_f32_out(
    const uint16_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    float* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      w8a16_linear_kernel,
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      hidden_states,
      weight,
      weight_scale,
      out,
      hidden_size,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_linear_bf16_lowp_out(
    const uint16_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    uint16_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (w8a16_linear_lowp_out_kernel<hip_bfloat16>),
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      reinterpret_cast<const hip_bfloat16*>(hidden_states),
      weight,
      weight_scale,
      reinterpret_cast<hip_bfloat16*>(out),
      hidden_size,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_linear_fp16_lowp_out(
    const half_t* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    half_t* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (w8a16_linear_lowp_out_kernel<half_t>),
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      hidden_states,
      weight,
      weight_scale,
      out,
      hidden_size,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_w8a16_linear_f32_f32_out(
    const float* hidden_states,
    const int8_t* weight,
    const float* weight_scale,
    float* out,
    int64_t tokens,
    int64_t hidden_size,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (tokens <= 0 || hidden_size <= 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      w8a16_linear_f32_kernel,
      dim3(static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<size_t>(threads) * sizeof(float),
      stream,
      hidden_states,
      weight,
      weight_scale,
      out,
      hidden_size,
      out_features);
  return static_cast<int>(hipGetLastError());
}
