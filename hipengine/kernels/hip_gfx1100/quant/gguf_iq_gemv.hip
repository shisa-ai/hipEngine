// Raw selected-MoE GEMV kernels for GGUF IQ3_XXS and IQ4_XS experts.
//
// Quant decoding follows llama.cpp@1ebf790cd.  Decode scheduling starts from
// qwen-kernel@52e240f: IQ3 maps one 8-value group to each work item and fuses
// gate+up+SiLU; IQ4 weighted down maps (selected slot, 32-value subblock)
// tasks into one all-route output reduction.  hipEngine keeps single selected
// primitives as the required unfused fallback.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int IQ3_XXS_BLOCK_BYTES = 98;
constexpr int IQ4_XS_BLOCK_BYTES = 136;

// Generated from llama.cpp ggml/src/ggml-common.h (MIT).  Each u32 packs four
// unsigned IQ3 grid magnitudes in little-endian byte order.
__device__ __constant__ uint32_t IQ3_XXS_GRID[256] = {
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu, 0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
    0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu, 0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
    0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u, 0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
    0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u, 0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
    0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu, 0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
    0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u, 0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
    0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu, 0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
    0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu, 0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
    0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu, 0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
    0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu, 0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
    0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u, 0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
    0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u, 0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
    0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu, 0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
    0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu, 0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
    0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u, 0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
    0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u, 0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
    0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu, 0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
    0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu, 0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
    0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u, 0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
    0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu, 0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
    0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu, 0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
    0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu, 0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
    0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu, 0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
    0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu, 0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
    0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u, 0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
    0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u, 0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
    0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu, 0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
    0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
    0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u, 0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
    0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu, 0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
    0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u, 0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
    0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu, 0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u,
};

__device__ __constant__ int8_t IQ4_NL_VALUES[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113,
};

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

__device__ inline float fp16_bytes_to_float(const uint8_t* ptr) {
  const uint16_t bits = static_cast<uint16_t>(ptr[0]) |
                        (static_cast<uint16_t>(ptr[1]) << 8);
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
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

__device__ inline uint32_t iq3_sign_byte(uint32_t selector) {
  return selector | ((static_cast<uint32_t>(__popc(selector)) & 1U) << 7);
}

__device__ inline float iq3_xxs_group_dot(
    const uint8_t* __restrict__ block,
    int group8,
    const float* __restrict__ x_values) {
  const uint8_t* __restrict__ qs = block + 2;
  const int group32 = group8 >> 2;
  const int local8 = group8 & 3;
  const uint32_t aux = load_u32_le(qs + 64 + 4 * group32);
  const float d = fp16_bytes_to_float(block);
  const float db = d * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
  const uint32_t signs = iq3_sign_byte((aux >> (7 * local8)) & 127U);
  const uint32_t grid1 = IQ3_XXS_GRID[qs[group32 * 8 + 2 * local8]];
  const uint32_t grid2 = IQ3_XXS_GRID[qs[group32 * 8 + 2 * local8 + 1]];

  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float magnitude1 = static_cast<float>((grid1 >> (8 * j)) & 255U);
    const float magnitude2 = static_cast<float>((grid2 >> (8 * j)) & 255U);
    sum += ((signs & (1U << j)) != 0U ? -magnitude1 : magnitude1) * x_values[j];
    sum += ((signs & (1U << (j + 4))) != 0U ? -magnitude2 : magnitude2) *
           x_values[j + 4];
  }
  return db * sum;
}

__device__ inline float iq4_xs_subblock_dot(
    const uint8_t* __restrict__ block,
    int subblock,
    const uint16_t* __restrict__ x) {
  const float d = fp16_bytes_to_float(block);
  const uint16_t scales_h = load_u16_le(block + 2);
  const uint8_t scale_l_byte = block[4 + (subblock >> 1)];
  const uint32_t scale_l =
      (static_cast<uint32_t>(scale_l_byte) >> (4 * (subblock & 1))) & 0x0FU;
  const uint32_t scale_h =
      (static_cast<uint32_t>(scales_h) >> (2 * subblock)) & 0x03U;
  const int scale = static_cast<int>(scale_l | (scale_h << 4)) - 32;
  const float dl = d * static_cast<float>(scale);
  const uint8_t* __restrict__ qs = block + 8 + subblock * 16;

  float sum_lo = 0.0f;
  float sum_hi = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint8_t packed = qs[j];
    sum_lo += static_cast<float>(IQ4_NL_VALUES[packed & 0x0FU]) *
              bf16_bits_to_float(x[j]);
    sum_hi += static_cast<float>(IQ4_NL_VALUES[packed >> 4]) *
              bf16_bits_to_float(x[j + 16]);
  }
  return dl * (sum_lo + sum_hi);
}

__device__ inline float reduce_block_sum(float value) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  __shared__ float wave_sums[8];
  if (lane == 0) {
    wave_sums[wave] = value;
  }
  __syncthreads();
  float total = 0.0f;
  if (tid == 0) {
    const int waves = (blockDim.x + 31) >> 5;
    for (int i = 0; i < waves; ++i) {
      total += wave_sums[i];
    }
  }
  return total;
}

__device__ inline void reduce_block_pair(float& value_a, float& value_b) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value_a += __shfl_down(value_a, offset);
    value_b += __shfl_down(value_b, offset);
  }
  __shared__ float wave_sums_a[8];
  __shared__ float wave_sums_b[8];
  if (lane == 0) {
    wave_sums_a[wave] = value_a;
    wave_sums_b[wave] = value_b;
  }
  __syncthreads();
  if (tid == 0) {
    value_a = 0.0f;
    value_b = 0.0f;
    const int waves = (blockDim.x + 31) >> 5;
    for (int i = 0; i < waves; ++i) {
      value_a += wave_sums_a[i];
      value_b += wave_sums_b[i];
    }
  }
}

__device__ inline int selected_x_row(int row, int x_rows, int rows) {
  const int lanes_per_x_row = rows / x_rows;
  return x_rows == 1 ? 0 : row / lanes_per_x_row;
}

__launch_bounds__(256, 2)
__global__ void gguf_iq3_xxs_selected_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  const int row_bytes = (in_features / QK_K) * IQ3_XXS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const uint8_t* __restrict__ weight_row = valid_expert
      ? qweight + expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes
      : qweight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * in_features;

  float acc = 0.0f;
  if (valid_expert) {
    const int groups8 = in_features / 8;
    for (int group8_linear = threadIdx.x; group8_linear < groups8;
         group8_linear += blockDim.x) {
      const int block_idx = group8_linear >> 5;
      const int group8 = group8_linear & 31;
      const int x_offset = block_idx * QK_K + group8 * 8;
      float x_values[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        x_values[j] = bf16_bits_to_float(x_row_ptr[x_offset + j]);
      }
      const uint8_t* __restrict__ block =
          weight_row + static_cast<int64_t>(block_idx) * IQ3_XXS_BLOCK_BYTES;
      acc += iq3_xxs_group_dot(block, group8, x_values);
    }
  }
  const float total = reduce_block_sum(acc);
  if (threadIdx.x == 0) {
    out[static_cast<int64_t>(row) * out_features + out_col] =
        float_to_bf16_bits(total);
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_iq3_xxs_selected_dual_silu_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  const int row_bytes = (in_features / QK_K) * IQ3_XXS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int64_t weight_offset =
      expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes;
  const uint8_t* __restrict__ gate_row =
      valid_expert ? gate_weight + weight_offset : gate_weight;
  const uint8_t* __restrict__ up_row =
      valid_expert ? up_weight + weight_offset : up_weight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * in_features;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  if (valid_expert) {
    const int groups8 = in_features / 8;
    for (int group8_linear = threadIdx.x; group8_linear < groups8;
         group8_linear += blockDim.x) {
      const int block_idx = group8_linear >> 5;
      const int group8 = group8_linear & 31;
      const int x_offset = block_idx * QK_K + group8 * 8;
      float x_values[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        x_values[j] = bf16_bits_to_float(x_row_ptr[x_offset + j]);
      }
      const int64_t block_offset =
          static_cast<int64_t>(block_idx) * IQ3_XXS_BLOCK_BYTES;
      gate_acc += iq3_xxs_group_dot(gate_row + block_offset, group8, x_values);
      up_acc += iq3_xxs_group_dot(up_row + block_offset, group8, x_values);
    }
  }
  reduce_block_pair(gate_acc, up_acc);
  if (threadIdx.x == 0) {
    // Preserve the registered unfused chain's two projection rounding
    // boundaries before applying the existing BF16 SiLU contract.
    const float gate = bf16_bits_to_float(float_to_bf16_bits(gate_acc));
    const float up = bf16_bits_to_float(float_to_bf16_bits(up_acc));
    const float sig = 1.0f / (1.0f + expf(-gate));
    out[static_cast<int64_t>(row) * out_features + out_col] =
        float_to_bf16_bits(gate * sig * up);
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_iq4_xs_selected_gemv_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int x_rows,
    int rows,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int64_t expert_id = selected[row];
  const bool valid_expert = expert_id >= 0 && expert_id < num_experts;
  const int row_bytes = (in_features / QK_K) * IQ4_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const uint8_t* __restrict__ weight_row = valid_expert
      ? qweight + expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes
      : qweight;
  const int x_row = selected_x_row(row, x_rows, rows);
  const uint16_t* __restrict__ x_row_ptr =
      x + static_cast<int64_t>(x_row) * in_features;

  float acc = 0.0f;
  if (valid_expert) {
    const int subblocks = in_features / 32;
    for (int unit = threadIdx.x; unit < subblocks; unit += blockDim.x) {
      const int block_idx = unit >> 3;
      const int subblock = unit & 7;
      const uint8_t* __restrict__ block =
          weight_row + static_cast<int64_t>(block_idx) * IQ4_XS_BLOCK_BYTES;
      const uint16_t* __restrict__ x_subblock =
          x_row_ptr + block_idx * QK_K + subblock * 32;
      acc += iq4_xs_subblock_dot(block, subblock, x_subblock);
    }
  }
  const float total = reduce_block_sum(acc);
  if (threadIdx.x == 0) {
    out[static_cast<int64_t>(row) * out_features + out_col] =
        float_to_bf16_bits(total);
  }
}

__launch_bounds__(256, 2)
__global__ void gguf_iq4_xs_weighted_selected_down_kernel(
    const uint16_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const float* __restrict__ routing_weights,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int tokens,
    int top_k,
    int num_experts,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int token = blockIdx.y;
  if (out_col >= out_features || token >= tokens) return;

  const int row_bytes = (in_features / QK_K) * IQ4_XS_BLOCK_BYTES;
  const int64_t expert_bytes = static_cast<int64_t>(out_features) * row_bytes;
  const int subblocks = in_features / 32;
  const int tasks = top_k * subblocks;
  const int64_t route_base = static_cast<int64_t>(token) * top_k;

  float acc = 0.0f;
  for (int task = threadIdx.x; task < tasks; task += blockDim.x) {
    const int slot = task / subblocks;
    const int unit = task - slot * subblocks;
    const int64_t expert_id = selected[route_base + slot];
    if (expert_id < 0 || expert_id >= num_experts) continue;
    const int block_idx = unit >> 3;
    const int subblock = unit & 7;
    const uint8_t* __restrict__ weight_row =
        qweight + expert_id * expert_bytes + static_cast<int64_t>(out_col) * row_bytes;
    const uint8_t* __restrict__ block =
        weight_row + static_cast<int64_t>(block_idx) * IQ4_XS_BLOCK_BYTES;
    const uint16_t* __restrict__ x_subblock =
        x + (route_base + slot) * in_features + block_idx * QK_K + subblock * 32;
    acc += routing_weights[route_base + slot] *
           iq4_xs_subblock_dot(block, subblock, x_subblock);
  }
  const float total = reduce_block_sum(acc);
  if (threadIdx.x == 0) {
    out[static_cast<int64_t>(token) * out_features + out_col] =
        float_to_bf16_bits(total);
  }
}

inline bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128 || threads == 256;
}

template <typename Kernel>
int launch_selected(
    Kernel kernel,
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || qweight == nullptr || out == nullptr ||
      x_rows <= 0 || rows <= 0 || rows % x_rows != 0 || num_experts <= 0 ||
      in_features <= 0 || in_features % QK_K != 0 || out_features <= 0 ||
      !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(
      static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows));
  kernel<<<grid, block, 0, stream>>>(
      x,
      selected,
      qweight,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_iq3_xxs_selected_gemv_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected(
      gguf_iq3_xxs_selected_gemv_kernel,
      x,
      selected,
      qweight,
      out,
      x_rows,
      rows,
      num_experts,
      in_features,
      out_features,
      threads,
      stream);
}

extern "C" int hipengine_gguf_iq4_xs_selected_gemv_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected(
      gguf_iq4_xs_selected_gemv_kernel,
      x,
      selected,
      qweight,
      out,
      x_rows,
      rows,
      num_experts,
      in_features,
      out_features,
      threads,
      stream);
}

extern "C" int hipengine_gguf_iq3_xxs_selected_dual_silu_gemv_bf16_bf16_out(
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
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || gate_weight == nullptr ||
      up_weight == nullptr || out == nullptr || x_rows <= 0 || rows <= 0 ||
      rows % x_rows != 0 || num_experts <= 0 || in_features <= 0 ||
      in_features % QK_K != 0 || out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(
      static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows));
  gguf_iq3_xxs_selected_dual_silu_gemv_kernel<<<grid, block, 0, stream>>>(
      x,
      selected,
      gate_weight,
      up_weight,
      out,
      static_cast<int>(x_rows),
      static_cast<int>(rows),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_iq4_xs_weighted_selected_down_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* selected,
    const float* routing_weights,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t tokens,
    int64_t top_k,
    int64_t num_experts,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || selected == nullptr || routing_weights == nullptr ||
      qweight == nullptr || out == nullptr || tokens <= 0 || top_k <= 0 ||
      num_experts <= 0 || in_features <= 0 || in_features % QK_K != 0 ||
      out_features <= 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(
      static_cast<unsigned int>(out_features), static_cast<unsigned int>(tokens));
  gguf_iq4_xs_weighted_selected_down_kernel<<<grid, block, 0, stream>>>(
      x,
      selected,
      routing_weights,
      qweight,
      out,
      static_cast<int>(tokens),
      static_cast<int>(top_k),
      static_cast<int>(num_experts),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}
