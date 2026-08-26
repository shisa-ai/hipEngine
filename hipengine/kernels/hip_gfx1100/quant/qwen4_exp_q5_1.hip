// hipEngine-native raw GGUF Q5_1 selected-expert GEMV for Qwen4Exp.

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

__device__ inline float load_fp16(const uint8_t* pointer) {
  return static_cast<float>(*reinterpret_cast<const _Float16*>(pointer));
}

__device__ inline float dequant_q5_1(const uint8_t* block, int index) {
  const float d = load_fp16(block);
  const float m = load_fp16(block + 2);
  const uint32_t qh = *reinterpret_cast<const uint32_t*>(block + 4);
  const uint8_t packed = block[8 + (index & 15)];
  const uint8_t low = index < 16 ? (packed & 0x0fU) : (packed >> 4);
  const uint8_t high = static_cast<uint8_t>((qh >> index) & 1U);
  return d * static_cast<float>(low | (high << 4)) + m;
}

__global__ void q5_1_selected_gemv_bf16_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int32_t* __restrict__ selected,
    const uint8_t* __restrict__ weights,
    uint16_t* __restrict__ output,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t row_bytes) {
  extern __shared__ float partial[];
  const int64_t pair = blockIdx.x;
  const int64_t row = pair / out_features;
  const int64_t out = pair % out_features;
  const int32_t expert = selected[row];
  if (expert < 0 || expert >= num_experts) {
    return;
  }
  const uint8_t* weight_row = weights
      + (static_cast<int64_t>(expert) * out_features + out) * row_bytes;
  float value = 0.0f;
  for (int64_t column = threadIdx.x; column < in_features; column += blockDim.x) {
    const int64_t block_index = column / 32;
    const int within = static_cast<int>(column & 31);
    const float weight = dequant_q5_1(weight_row + block_index * 24, within);
    value += bf16_bits_to_float(input[row * in_features + column]) * weight;
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
    output[pair] = float_to_bf16_bits(partial[0]);
  }
}

}  // namespace

extern "C" int hipengine_qwen4_exp_q5_1_selected_gemv_bf16_bf16_out(
    const void* input,
    const void* selected,
    const void* weights,
    void* output,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !selected || !weights || !output || rows <= 0 || num_experts <= 0 ||
      in_features <= 0 || (in_features % 32) || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const int64_t row_bytes = (in_features / 32) * 24;
  hipLaunchKernelGGL(
      q5_1_selected_gemv_bf16_bf16_kernel,
      dim3(static_cast<uint32_t>(rows * out_features)), dim3(threads),
      threads * sizeof(float), stream,
      static_cast<const uint16_t*>(input), static_cast<const int32_t*>(selected),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}
