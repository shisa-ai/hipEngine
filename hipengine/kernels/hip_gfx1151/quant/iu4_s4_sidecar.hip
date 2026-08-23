#include <hip/hip_runtime.h>

#include <cstdint>
#include <cmath>

namespace {

using i32x2_t = int __attribute__((__vector_size__(2 * sizeof(int))));
using i32x8_t = int __attribute__((__vector_size__(8 * sizeof(int))));

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

__device__ inline uint8_t quantize_u4(float value, float scale, int zero) {
  int q = __float2int_rn(value / scale) + zero;
  q = q < 0 ? 0 : (q > 15 ? 15 : q);
  return static_cast<uint8_t>(q);
}

__launch_bounds__(256, 1)
__global__ void iu4_u4_quantize_bf16_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ packed,
    float* __restrict__ scales,
    int32_t* __restrict__ zero_points,
    int64_t rows,
    int64_t hidden) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (row >= rows) return;

  __shared__ float reductions[512];
  __shared__ float row_scale;
  __shared__ int32_t row_zero;
  float local_min = INFINITY;
  float local_max = -INFINITY;
  const uint16_t* row_input = input + row * hidden;
  for (int64_t k = tid; k < hidden; k += blockDim.x) {
    const float value = bf16_bits_to_float(row_input[k]);
    local_min = fminf(local_min, value);
    local_max = fmaxf(local_max, value);
  }
  reductions[tid] = local_min;
  reductions[256 + tid] = local_max;
  __syncthreads();

  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reductions[tid] = fminf(reductions[tid], reductions[tid + stride]);
      reductions[256 + tid] =
          fmaxf(reductions[256 + tid], reductions[256 + tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    const float lo = fminf(reductions[0], 0.0f);
    const float hi = fmaxf(reductions[256], 0.0f);
    float scale = (hi - lo) / 15.0f;
    if (!(scale > 0.0f) || !isfinite(scale)) scale = 1.0f;
    int zero = __float2int_rn(-lo / scale);
    zero = zero < 0 ? 0 : (zero > 15 ? 15 : zero);
    row_scale = scale;
    row_zero = zero;
    scales[row] = scale;
    zero_points[row] = zero;
  }
  __syncthreads();

  const int64_t row_tile = row / 16;
  const int64_t row_in_tile = row % 16;
  const int64_t k_tiles = hidden / 16;
  for (int64_t byte = tid; byte < hidden / 2; byte += blockDim.x) {
    const uint8_t low = quantize_u4(
        bf16_bits_to_float(row_input[2 * byte]), row_scale, row_zero);
    const uint8_t high = quantize_u4(
        bf16_bits_to_float(row_input[2 * byte + 1]), row_scale, row_zero);
    const int64_t k_tile = byte / 8;
    const int64_t byte_in_tile = byte % 8;
    const int64_t packed_index =
        ((row_tile * k_tiles + k_tile) * 16 + row_in_tile) * 8 + byte_in_tile;
    packed[packed_index] = static_cast<uint8_t>(low | (high << 4));
  }
}

__launch_bounds__(32, 1)
__global__ void iu4_s4_matmul_i32_probe_kernel(
    const uint8_t* __restrict__ activations,
    const uint8_t* __restrict__ weights,
    int32_t* __restrict__ output,
    int64_t rows,
    int64_t hidden,
    int64_t out_features) {
  const int lane = static_cast<int>(threadIdx.x);
  const int lane16 = lane & 15;
  const int half = lane >> 4;
  const int64_t row_base = static_cast<int64_t>(blockIdx.y) * 16;
  const int64_t activation_row = row_base + lane16;
  const int64_t output_col = static_cast<int64_t>(blockIdx.x) * 16 + lane16;
  i32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

  for (int64_t k = 0; k < hidden; k += 16) {
    i32x2_t a = {0, 0};
    i32x2_t b = {0, 0};
    const int64_t k_tile = k / 16;
    if (activation_row < rows) {
      a = *reinterpret_cast<const i32x2_t*>(
          activations + ((blockIdx.y * (hidden / 16) + k_tile) * 16 + lane16) * 8);
    }
    if (output_col < out_features) {
      b = *reinterpret_cast<const i32x2_t*>(
          weights + ((blockIdx.x * (hidden / 16) + k_tile) * 16 + lane16) * 8);
    }
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
        false, a, true, b, acc, false);
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t row = row_base + half + 2 * j;
    if (row < rows && output_col < out_features) {
      output[row * out_features + output_col] = acc[j];
    }
  }
}

__launch_bounds__(64, 1)
__global__ void iu4_s4_dual_silu_bf16_kernel(
    const uint8_t* __restrict__ activations,
    const float* __restrict__ activation_scales,
    const int32_t* __restrict__ activation_zero_points,
    const uint8_t* __restrict__ gate_weights,
    const float* __restrict__ gate_scales,
    const int32_t* __restrict__ gate_sums,
    const uint8_t* __restrict__ up_weights,
    const float* __restrict__ up_scales,
    const int32_t* __restrict__ up_sums,
    uint16_t* __restrict__ output,
    int64_t rows,
    int64_t hidden,
    int64_t out_features) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int lane16 = lane & 15;
  const int half = lane >> 4;
  const int64_t row_base = static_cast<int64_t>(blockIdx.y) * 16;
  const int64_t activation_row = row_base + lane16;
  const int64_t output_col = static_cast<int64_t>(blockIdx.x) * 16 + lane16;
  const uint8_t* selected_weights = wave == 0 ? gate_weights : up_weights;
  const float* selected_scales = wave == 0 ? gate_scales : up_scales;
  const int32_t* selected_sums = wave == 0 ? gate_sums : up_sums;
  i32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

  for (int64_t k = 0; k < hidden; k += 16) {
    i32x2_t a = {0, 0};
    i32x2_t b = {0, 0};
    const int64_t k_tile = k / 16;
    if (activation_row < rows) {
      a = *reinterpret_cast<const i32x2_t*>(
          activations + ((blockIdx.y * (hidden / 16) + k_tile) * 16 + lane16) * 8);
    }
    if (output_col < out_features) {
      b = *reinterpret_cast<const i32x2_t*>(
          selected_weights + ((blockIdx.x * (hidden / 16) + k_tile) * 16 + lane16) * 8);
    }
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
        false, a, true, b, acc, false);
  }

  __shared__ uint16_t projection_bits[2 * 16 * 16];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int local_row = half + 2 * j;
    const int64_t row = row_base + local_row;
    uint16_t value_bits = 0;
    if (row < rows && output_col < out_features) {
      const int32_t corrected =
          acc[j] - activation_zero_points[row] * selected_sums[output_col];
      const float value = static_cast<float>(corrected) *
                          activation_scales[row] * selected_scales[output_col];
      value_bits = float_to_bf16_bits(value);
    }
    projection_bits[wave * 256 + local_row * 16 + lane16] = value_bits;
  }
  __syncthreads();

  const int64_t remaining_rows = rows - row_base;
  const int valid_rows = static_cast<int>(remaining_rows < 16 ? remaining_rows : 16);
  const int valid_values = valid_rows * 16;
  for (int index = tid; index < valid_values; index += blockDim.x) {
    const int local_row = index / 16;
    const int local_col = index - local_row * 16;
    const int64_t row = row_base + local_row;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * 16 + local_col;
    if (col < out_features) {
      const float gate = bf16_bits_to_float(projection_bits[index]);
      const float up = bf16_bits_to_float(projection_bits[256 + index]);
      const float sigmoid = 1.0f / (1.0f + expf(-gate));
      output[row * out_features + col] = float_to_bf16_bits(gate * sigmoid * up);
    }
  }
}

}  // namespace

extern "C" int hipengine_iu4_u4_quantize_bf16(
    const void* input,
    void* packed,
    void* scales,
    void* zero_points,
    int64_t rows,
    int64_t hidden,
    void* stream) {
  if (input == nullptr || packed == nullptr || scales == nullptr ||
      zero_points == nullptr || rows <= 0 || rows > 128 || hidden <= 0 ||
      hidden % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      iu4_u4_quantize_bf16_kernel,
      dim3(static_cast<uint32_t>(rows)),
      dim3(256),
      0,
      reinterpret_cast<hipStream_t>(stream),
      static_cast<const uint16_t*>(input),
      static_cast<uint8_t*>(packed),
      static_cast<float*>(scales),
      static_cast<int32_t*>(zero_points),
      rows,
      hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_iu4_s4_matmul_i32_probe(
    const void* activations,
    const void* weights,
    void* output,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    void* stream) {
  if (activations == nullptr || weights == nullptr || output == nullptr ||
      rows <= 0 || rows > 128 || hidden <= 0 || hidden % 16 != 0 ||
      out_features <= 0 || out_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      iu4_s4_matmul_i32_probe_kernel,
      dim3(static_cast<uint32_t>(out_features / 16),
           static_cast<uint32_t>((rows + 15) / 16)),
      dim3(32),
      0,
      reinterpret_cast<hipStream_t>(stream),
      static_cast<const uint8_t*>(activations),
      static_cast<const uint8_t*>(weights),
      static_cast<int32_t*>(output),
      rows,
      hidden,
      out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_iu4_s4_dual_silu_bf16_out(
    const void* activations,
    const void* activation_scales,
    const void* activation_zero_points,
    const void* gate_weights,
    const void* gate_scales,
    const void* gate_sums,
    const void* up_weights,
    const void* up_scales,
    const void* up_sums,
    void* output,
    int64_t rows,
    int64_t hidden,
    int64_t out_features,
    void* stream) {
  if (activations == nullptr || activation_scales == nullptr ||
      activation_zero_points == nullptr || gate_weights == nullptr ||
      gate_scales == nullptr || gate_sums == nullptr || up_weights == nullptr ||
      up_scales == nullptr || up_sums == nullptr || output == nullptr ||
      rows < 2 || rows > 128 || hidden <= 0 || hidden % 16 != 0 ||
      out_features <= 0 || out_features % 16 != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      iu4_s4_dual_silu_bf16_kernel,
      dim3(static_cast<uint32_t>(out_features / 16),
           static_cast<uint32_t>((rows + 15) / 16)),
      dim3(64),
      0,
      reinterpret_cast<hipStream_t>(stream),
      static_cast<const uint8_t*>(activations),
      static_cast<const float*>(activation_scales),
      static_cast<const int32_t*>(activation_zero_points),
      static_cast<const uint8_t*>(gate_weights),
      static_cast<const float*>(gate_scales),
      static_cast<const int32_t*>(gate_sums),
      static_cast<const uint8_t*>(up_weights),
      static_cast<const float*>(up_scales),
      static_cast<const int32_t*>(up_sums),
      static_cast<uint16_t*>(output),
      rows,
      hidden,
      out_features);
  return static_cast<int>(hipGetLastError());
}
