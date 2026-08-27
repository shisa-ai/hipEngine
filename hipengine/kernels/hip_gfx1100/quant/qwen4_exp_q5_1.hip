// hipEngine-native raw GGUF Q5_1 selected-expert GEMV for Qwen4Exp.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

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

template <int ROW_BATCH>
__global__ void q5_1_selected_grouped_prefill_bf16_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ expert_start,
    const uint8_t* __restrict__ weights,
    uint16_t* __restrict__ output,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t row_bytes) {
  extern __shared__ float partial[];
  const int64_t out = blockIdx.x;
  const int64_t expert = blockIdx.y;
  if (out >= out_features || expert >= num_experts) return;
  const int64_t start = expert_start[expert];
  const int64_t end = expert_start[expert + 1];
  if (start < 0 || end < start || end > compact_rows || start == end) return;
  const uint8_t* weight_row = weights
      + (expert * out_features + out) * row_bytes;
  for (int64_t base = start; base < end; base += ROW_BATCH) {
    float value[ROW_BATCH] = {};
    for (int64_t column = threadIdx.x; column < in_features; column += blockDim.x) {
      const int64_t block_index = column / 32;
      const int within = static_cast<int>(column & 31);
      const float weight = dequant_q5_1(weight_row + block_index * 24, within);
#pragma unroll
      for (int row = 0; row < ROW_BATCH; ++row) {
        const int64_t compact_row = base + row;
        if (compact_row < end) {
          value[row] += bf16_bits_to_float(
              input[compact_row * in_features + column]) * weight;
        }
      }
    }
#pragma unroll
    for (int row = 0; row < ROW_BATCH; ++row) {
      partial[row * blockDim.x + threadIdx.x] = value[row];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
#pragma unroll
        for (int row = 0; row < ROW_BATCH; ++row) {
          partial[row * blockDim.x + threadIdx.x] +=
              partial[row * blockDim.x + threadIdx.x + stride];
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
#pragma unroll
      for (int row = 0; row < ROW_BATCH; ++row) {
        const int64_t compact_row = base + row;
        if (compact_row < end) {
          output[compact_row * out_features + out] =
              float_to_bf16_bits(partial[row * blockDim.x]);
        }
      }
    }
    __syncthreads();
  }
}

template <int TILE_M, int TILE_N>
__launch_bounds__(32, 8)
__global__ void q5_1_selected_grouped_wmma_prefill_bf16_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ weights,
    uint16_t* __restrict__ output,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t row_bytes) {
  constexpr int TM = TILE_M / 16;
  constexpr int TN = TILE_N / 16;
  const int tid = threadIdx.x;
  const int out_start = blockIdx.x * TILE_M;
  const int64_t row_tile = blockIdx.y;
  if (out_start >= out_features) return;
  const int64_t expert = tile_expert[row_tile];
  if (expert < 0 || expert >= num_experts) return;
  const int64_t compact_start = expert_start_compact[expert];
  const int64_t compact_end = expert_start_compact[expert + 1];
  const int64_t expert_rows = compact_end - compact_start;
  const int64_t local_start = row_tile * 16 - expert_start_wmma[expert];
  if (expert_rows <= 0 || compact_start < 0 || compact_end > compact_rows) return;

  int my_out[TM];
  int safe_out[TM];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    my_out[tm] = out_start + tm * 16 + (tid & 15);
    safe_out[tm] = my_out[tm] < out_features ? my_out[tm] : out_features - 1;
  }
  int64_t local_row[TN];
  int64_t safe_row[TN];
#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    local_row[tn] = local_start + tn * 16 + (tid & 15);
    const int64_t bounded = local_row[tn] < expert_rows
        ? local_row[tn] : expert_rows - 1;
    safe_row[tn] = compact_start + bounded;
  }
  float8_t acc[TM][TN];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      acc[tm][tn] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
  }
  const int blocks_per_row = in_features / 32;
  for (int blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* block_ptr[TM];
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      block_ptr[tm] = weights
          + (expert * out_features + safe_out[tm]) * row_bytes + blk * 24;
    }
#pragma unroll
    for (int kt = 0; kt < 2; ++kt) {
      const int k_base = blk * 32 + kt * 16;
      half16_t activation[TN];
#pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          activation[tn][kk] = static_cast<half_t>(bf16_bits_to_float(
              input[safe_row[tn] * in_features + k_base + kk]));
        }
      }
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        half16_t weight;
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          weight[kk] = static_cast<half_t>(
              dequant_q5_1(block_ptr[tm], kt * 16 + kk));
        }
#pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              weight, activation[tn], acc[tm][tn]);
        }
      }
    }
  }
#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    if (local_row[tn] >= expert_rows) continue;
    const int64_t out_row = compact_start + local_row[tn];
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = out_start + tm * 16 + 2 * j + (tid >> 4);
        if (col < out_features) {
          output[out_row * out_features + col] =
              float_to_bf16_bits(static_cast<float>(acc[tm][tn][j]));
        }
      }
    }
  }
}

__global__ void q5_1_selected_gemv_bf16_bf16_kernel(
    const uint16_t* __restrict__ input,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ weights,
    uint16_t* __restrict__ output,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t row_bytes) {
  extern __shared__ float partial[];
  const int64_t pair = blockIdx.x;
  const int64_t row = pair / out_features;
  const int64_t out = pair % out_features;
  const int64_t input_row = row / (rows / x_rows);
  const int64_t expert = selected[row];
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
    value += bf16_bits_to_float(input[input_row * in_features + column]) * weight;
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

extern "C" int hipengine_qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out(
    const void* input,
    const void* expert_start,
    const void* weights,
    void* output,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !expert_start || !weights || !output || compact_rows <= 0 ||
      num_experts <= 0 || in_features <= 0 || (in_features % 32) ||
      out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  constexpr int row_batch = 8;
  const int64_t row_bytes = (in_features / 32) * 24;
  hipLaunchKernelGGL(
      (q5_1_selected_grouped_prefill_bf16_bf16_kernel<row_batch>),
      dim3(static_cast<uint32_t>(out_features), static_cast<uint32_t>(num_experts)),
      dim3(threads), row_batch * threads * sizeof(float), stream,
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      compact_rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out(
    const void* input,
    const void* expert_start_compact,
    const void* expert_start_wmma,
    const void* tile_expert,
    const void* weights,
    void* output,
    int64_t compact_rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!input || !expert_start_compact || !expert_start_wmma || !tile_expert ||
      !weights || !output || compact_rows <= 0 || num_experts <= 0 ||
      in_features <= 0 || (in_features % 32) || out_features <= 0 ||
      wmma_total_rows <= 0 || (wmma_total_rows % 16)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int tile_m = 32;
  constexpr int tile_n = 16;
  const int64_t row_bytes = (in_features / 32) * 24;
  hipLaunchKernelGGL(
      (q5_1_selected_grouped_wmma_prefill_bf16_bf16_kernel<tile_m, tile_n>),
      dim3(static_cast<uint32_t>((out_features + tile_m - 1) / tile_m),
           static_cast<uint32_t>(wmma_total_rows / tile_n)),
      dim3(32), 0, stream,
      static_cast<const uint16_t*>(input),
      static_cast<const int64_t*>(expert_start_compact),
      static_cast<const int64_t*>(expert_start_wmma),
      static_cast<const int64_t*>(tile_expert),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      compact_rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out(
    const void* input,
    const void* selected,
    const void* weights,
    void* output,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !selected || !weights || !output || x_rows <= 0 || rows <= 0 ||
      rows % x_rows || num_experts <= 0 || in_features <= 0 ||
      (in_features % 32) || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 64;
  const int64_t row_bytes = (in_features / 32) * 24;
  hipLaunchKernelGGL(
      q5_1_selected_gemv_bf16_bf16_kernel,
      dim3(static_cast<uint32_t>(rows * out_features)), dim3(threads),
      threads * sizeof(float), stream,
      static_cast<const uint16_t*>(input), static_cast<const int64_t*>(selected),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      x_rows, rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_qwen4_exp_q5_1_selected_gemv_bf16_bf16_out(
    const void* input,
    const void* selected,
    const void* weights,
    void* output,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!input || !selected || !weights || !output || x_rows <= 0 || rows <= 0 ||
      rows % x_rows || num_experts <= 0 || in_features <= 0 ||
      (in_features % 32) || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int threads = 256;
  const int64_t row_bytes = (in_features / 32) * 24;
  hipLaunchKernelGGL(
      q5_1_selected_gemv_bf16_bf16_kernel,
      dim3(static_cast<uint32_t>(rows * out_features)), dim3(threads),
      threads * sizeof(float), stream,
      static_cast<const uint16_t*>(input), static_cast<const int64_t*>(selected),
      static_cast<const uint8_t*>(weights), static_cast<uint16_t*>(output),
      x_rows, rows, num_experts, in_features, out_features, row_bytes);
  return static_cast<int>(hipGetLastError());
}
