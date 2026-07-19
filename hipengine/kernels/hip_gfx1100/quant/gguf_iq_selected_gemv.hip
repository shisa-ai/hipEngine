// Raw GGUF IQ3_XXS/IQ4_XS selected-expert decode kernels for Qwen3.6 Q3.
//
// ABI (matching the existing qwen35 GGUF selected path):
//   x        [x_rows, in_features] BF16
//   selected [rows]                int64 device expert ids
//   weights  [experts, out_features, raw_row_bytes]
//   out      [rows, out_features]  BF16
//
// Gate/up is fused through SiLU and computes four adjacent output rows per
// workgroup.  Each lane retains eight BF16 activations while streaming those
// four rows, cutting activation traffic and exposing independent dequant chains
// without introducing a resident repack.  The gate/up accumulator is rounded
// through BF16 before SiLU to preserve the engine's existing split-kernel
// numerical boundary.  IQ4_XS down uses the same four-row activation reuse.
//
// Launches contain no allocation, host read, or synchronization and are safe to
// record in the resident decode graph.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

#include "gguf_iq_tables.h"

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int IQ3_XXS_BLOCK_BYTES = 98;
constexpr int IQ4_XS_BLOCK_BYTES = 136;
constexpr int ROW_TILE = 4;
constexpr int THREADS = 256;
constexpr int WAVE = 32;
constexpr int WAVES = THREADS / WAVE;

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

__device__ inline float round_trip_bf16(float value) {
  return bf16_bits_to_float(float_to_bf16_bits(value));
}

__device__ inline uint16_t load_u16_le(const uint8_t* ptr) {
  return static_cast<uint16_t>(ptr[0]) |
         (static_cast<uint16_t>(ptr[1]) << 8);
}

__device__ inline uint32_t load_u32_le(const uint8_t* ptr) {
  return static_cast<uint32_t>(ptr[0]) |
         (static_cast<uint32_t>(ptr[1]) << 8) |
         (static_cast<uint32_t>(ptr[2]) << 16) |
         (static_cast<uint32_t>(ptr[3]) << 24);
}

__device__ inline float load_fp16_le(const uint8_t* ptr) {
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = load_u16_le(ptr);
  return static_cast<float>(in.f16);
}

__device__ inline uint32_t iq_sign_mask(uint32_t sign_index) {
  return sign_index | ((__popc(sign_index) & 1U) << 7);
}

__device__ inline float iq3_xxs_group8_dot(
    const uint8_t* __restrict__ block,
    int group8,
    const float (&xv)[8]) {
  const int group32 = group8 >> 2;
  const int lane8 = group8 & 3;
  const uint32_t aux = load_u32_le(block + 2 + 64 + group32 * 4);
  const float db = load_fp16_le(block) *
                   (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
  const uint32_t signs = iq_sign_mask((aux >> (7 * lane8)) & 0x7FU);
  const int qoff = 2 + group32 * 8 + lane8 * 2;
  const uint32_t grid0 = kIq3XxsGrid[block[qoff]];
  const uint32_t grid1 = kIq3XxsGrid[block[qoff + 1]];
  float dot = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float mag0 = static_cast<float>((grid0 >> (8 * j)) & 0xFFU);
    const float mag1 = static_cast<float>((grid1 >> (8 * j)) & 0xFFU);
    dot = fmaf((signs & (1U << j)) ? -mag0 : mag0, xv[j], dot);
    dot = fmaf((signs & (1U << (j + 4))) ? -mag1 : mag1, xv[j + 4], dot);
  }
  return db * dot;
}

__device__ inline float iq4_xs_group8_dot(
    const uint8_t* __restrict__ block,
    int group8,
    const float (&xv)[8]) {
  const int subblock = group8 >> 2;
  const int chunk8 = group8 & 3;
  const uint16_t scales_h = load_u16_le(block + 2);
  const uint8_t scales_l = block[4 + (subblock >> 1)];
  const int encoded_scale =
      ((scales_l >> (4 * (subblock & 1))) & 0x0F) |
      (((scales_h >> (2 * subblock)) & 0x03) << 4);
  const float dl = load_fp16_le(block) * static_cast<float>(encoded_scale - 32);
  const int qoff = 8 + subblock * 16 + (chunk8 & 1) * 8;
  const int shift = (chunk8 >= 2) ? 4 : 0;
  float dot = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint8_t q = (block[qoff + j] >> shift) & 0x0F;
    dot = fmaf(static_cast<float>(kIq4NlValues[q]), xv[j], dot);
  }
  return dl * dot;
}

template <int qtype>
__device__ inline float iq_group8_dot(
    const uint8_t* __restrict__ block,
    int group8,
    const float (&xv)[8]) {
  if constexpr (qtype == 3) {
    return iq3_xxs_group8_dot(block, group8, xv);
  } else {
    return iq4_xs_group8_dot(block, group8, xv);
  }
}

template <int qtype>
__launch_bounds__(THREADS, 1) __global__ void iq_selected_gate_up_silu_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features) {
  const int64_t row = blockIdx.y;
  const int64_t out_base = static_cast<int64_t>(blockIdx.x) * ROW_TILE;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t expert = selected[row];
  if (expert < 0 || expert >= num_experts) return;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row = (x_rows == 1) ? 0 : row / lanes_per_x_row;
  const uint16_t* __restrict__ x_ptr = x + x_row * in_features;

  static_assert(qtype == 3 || qtype == 4, "IQ qtype must be 3 or 4");
  constexpr int raw_block_bytes =
      qtype == 3 ? IQ3_XXS_BLOCK_BYTES : IQ4_XS_BLOCK_BYTES;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * raw_block_bytes;
  const int64_t expert_base = expert * out_features * row_bytes;
  const int64_t groups8 = in_features >> 3;

  float gate_acc[ROW_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
  float up_acc[ROW_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int64_t group = tid; group < groups8; group += blockDim.x) {
    const int64_t block_index = group >> 5;
    const int group8 = static_cast<int>(group & 31);
    float xv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      xv[j] = bf16_bits_to_float(x_ptr[(group << 3) + j]);
    }
#pragma unroll
    for (int rr = 0; rr < ROW_TILE; ++rr) {
      const int64_t out_col = out_base + rr;
      if (out_col < out_features) {
        const int64_t block_offset =
            expert_base + out_col * row_bytes + block_index * raw_block_bytes;
        gate_acc[rr] = fmaf(
            1.0f,
            iq_group8_dot<qtype>(gate_weight + block_offset, group8, xv),
            gate_acc[rr]);
        up_acc[rr] = fmaf(
            1.0f,
            iq_group8_dot<qtype>(up_weight + block_offset, group8, xv),
            up_acc[rr]);
      }
    }
  }

#pragma unroll
  for (int rr = 0; rr < ROW_TILE; ++rr) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      gate_acc[rr] += __shfl_down(gate_acc[rr], offset);
      up_acc[rr] += __shfl_down(up_acc[rr], offset);
    }
  }

  __shared__ float gate_partial[ROW_TILE * WAVES];
  __shared__ float up_partial[ROW_TILE * WAVES];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int rr = 0; rr < ROW_TILE; ++rr) {
      gate_partial[rr * WAVES + wave] = gate_acc[rr];
      up_partial[rr * WAVES + wave] = up_acc[rr];
    }
  }
  __syncthreads();

  if (wave == 0 && lane < ROW_TILE) {
    const int rr = lane;
    const int64_t out_col = out_base + rr;
    if (out_col < out_features) {
      float gate = 0.0f;
      float up = 0.0f;
#pragma unroll
      for (int w = 0; w < WAVES; ++w) {
        gate += gate_partial[rr * WAVES + w];
        up += up_partial[rr * WAVES + w];
      }
      gate = round_trip_bf16(gate);
      up = round_trip_bf16(up);
      const float silu = gate / (1.0f + expf(-gate));
      out[row * out_features + out_col] = float_to_bf16_bits(silu * up);
    }
  }
}

__launch_bounds__(THREADS, 1) __global__ void iq4_xs_selected_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ weight,
    uint16_t* __restrict__ out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features) {
  const int64_t row = blockIdx.y;
  const int64_t out_base = static_cast<int64_t>(blockIdx.x) * ROW_TILE;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t expert = selected[row];
  if (expert < 0 || expert >= num_experts) return;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row = (x_rows == 1) ? 0 : row / lanes_per_x_row;
  const uint16_t* __restrict__ x_ptr = x + x_row * in_features;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * IQ4_XS_BLOCK_BYTES;
  const int64_t expert_base = expert * out_features * row_bytes;
  const int64_t groups8 = in_features >> 3;
  float acc[ROW_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int64_t group = tid; group < groups8; group += blockDim.x) {
    const int64_t block_index = group >> 5;
    const int group8 = static_cast<int>(group & 31);
    float xv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      xv[j] = bf16_bits_to_float(x_ptr[(group << 3) + j]);
    }
#pragma unroll
    for (int rr = 0; rr < ROW_TILE; ++rr) {
      const int64_t out_col = out_base + rr;
      if (out_col < out_features) {
        const int64_t block_offset =
            expert_base + out_col * row_bytes + block_index * IQ4_XS_BLOCK_BYTES;
        acc[rr] = fmaf(
            1.0f,
            iq4_xs_group8_dot(weight + block_offset, group8, xv),
            acc[rr]);
      }
    }
  }

#pragma unroll
  for (int rr = 0; rr < ROW_TILE; ++rr) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[rr] += __shfl_down(acc[rr], offset);
    }
  }

  __shared__ float partial[ROW_TILE * WAVES];
  const int wave = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
#pragma unroll
    for (int rr = 0; rr < ROW_TILE; ++rr) {
      partial[rr * WAVES + wave] = acc[rr];
    }
  }
  __syncthreads();

  if (wave == 0 && lane < ROW_TILE) {
    const int rr = lane;
    const int64_t out_col = out_base + rr;
    if (out_col < out_features) {
      float total = 0.0f;
#pragma unroll
      for (int w = 0; w < WAVES; ++w) {
        total += partial[rr * WAVES + w];
      }
      out[row * out_features + out_col] = float_to_bf16_bits(total);
    }
  }
}

bool valid_common(
    const void* x,
    const void* selected,
    const void* weight,
    const void* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features) {
  return x != nullptr && selected != nullptr && weight != nullptr && out != nullptr &&
         x_rows > 0 && rows > 0 && (rows % x_rows) == 0 && num_experts > 0 &&
         in_features > 0 && (in_features % QK_K) == 0 && out_features > 0;
}

template <int qtype>
int launch_gate_up_silu(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!valid_common(
          x, selected, gate_weight, out, x_rows, rows, num_experts,
          in_features, out_features) || up_weight == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(
      static_cast<unsigned int>((out_features + ROW_TILE - 1) / ROW_TILE),
      static_cast<unsigned int>(rows));
  hipLaunchKernelGGL(
      (iq_selected_gate_up_silu_kernel<qtype>),
      grid,
      dim3(THREADS),
      0,
      stream,
      x,
      selected,
      gate_weight,
      up_weight,
      out,
      x_rows,
      rows,
      num_experts,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

int launch_iq4_xs_selected(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* weight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!valid_common(
          x, selected, weight, out, x_rows, rows, num_experts,
          in_features, out_features)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(
      static_cast<unsigned int>((out_features + ROW_TILE - 1) / ROW_TILE),
      static_cast<unsigned int>(rows));
  hipLaunchKernelGGL(
      iq4_xs_selected_gemv_kernel,
      grid,
      dim3(THREADS),
      0,
      stream,
      x,
      selected,
      weight,
      out,
      x_rows,
      rows,
      num_experts,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_iq3_xxs_selected_fused_gate_up_silu_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_gate_up_silu<3>(
      x, selected, gate_weight, up_weight, out, x_rows, rows, num_experts,
      in_features, out_features, stream);
}

extern "C" int hipengine_gguf_iq4_xs_selected_fused_gate_up_silu_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* gate_weight,
    const uint8_t* up_weight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_gate_up_silu<4>(
      x, selected, gate_weight, up_weight, out, x_rows, rows, num_experts,
      in_features, out_features, stream);
}

extern "C" int hipengine_gguf_iq4_xs_selected_gemv_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* weight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_iq4_xs_selected(
      x, selected, weight, out, x_rows, rows, num_experts, in_features,
      out_features, stream);
}
