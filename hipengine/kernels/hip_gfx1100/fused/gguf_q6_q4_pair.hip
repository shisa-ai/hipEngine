// Exact rows1 block-parallel GGUF projection-pair launches.
//
// The recurrent heterogeneous grid preserves standard-Q6 local128/16-column
// arithmetic and four independent Q4 local32/8-column owners. The narrow K/V
// grids preserve the qualified Q4 col4 and planar-qmicro-Q6 arithmetic trees.
// Block classes share only one launch and never serialize either projection.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int T16_COLS = 16;
constexpr int Q4_TILE_COLS = 8;
constexpr int Q4_SUBBLOCK = 32;
constexpr int Q4_SUBBLOCKS = 8;
constexpr int Q6_GROUPS16 = 16;

constexpr int Q6_T16_D_OFFSET = 0;
constexpr int Q6_T16_SCALE_OFFSET = Q6_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q6_T16_QL_OFFSET =
    Q6_T16_SCALE_OFFSET + Q6_GROUPS16 * T16_COLS;
constexpr int Q6_T16_QH_OFFSET =
    Q6_T16_QL_OFFSET + QK_K * (T16_COLS / 2);
constexpr int Q6_T16_BLOCK_BYTES =
    Q6_T16_QH_OFFSET + QK_K * (T16_COLS / 4);
constexpr int Q6_T16_QMICRO_OFFSET = Q6_T16_QL_OFFSET;
constexpr int Q6_T16_QMICRO_RECORD_BYTES = 12;

constexpr int Q4_T16_D_OFFSET = 0;
constexpr int Q4_T16_DMIN_OFFSET = Q4_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_SCALE_OFFSET =
    Q4_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_MIN_OFFSET =
    Q4_T16_SCALE_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_Q_OFFSET =
    Q4_T16_MIN_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_BLOCK_BYTES =
    Q4_T16_Q_OFFSET +
    Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);

__device__ __forceinline__ float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__device__ __forceinline__ float fp16_bytes_to_float(
    const uint8_t* ptr) {
  const uint16_t bits = static_cast<uint16_t>(ptr[0]) |
                        (static_cast<uint16_t>(ptr[1]) << 8);
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

__device__ __forceinline__ float fp16_bits_to_float(uint16_t bits) {
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

#define GGUF_T16_WAVE_REDUCE(acc) \
  do {                            \
    acc += __shfl_down(acc, 16);  \
    acc += __shfl_down(acc, 8);   \
    acc += __shfl_down(acc, 4);   \
    acc += __shfl_down(acc, 2);   \
    acc += __shfl_down(acc, 1);   \
  } while (0)

__device__ __forceinline__ int t16_q6_quant(
    const uint8_t* __restrict__ tile,
    int k_in_block,
    int col) {
  const uint8_t low_packed =
      tile[Q6_T16_QL_OFFSET +
           k_in_block * (T16_COLS / 2) + (col >> 1)];
  const uint8_t low =
      (col & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed =
      tile[Q6_T16_QH_OFFSET +
           k_in_block * (T16_COLS / 4) + (col >> 2)];
  const uint8_t high =
      (high_packed >> (2 * (col & 3))) & 0x03u;
  return static_cast<int>(low | (high << 4)) - 32;
}

__device__ __forceinline__ void run_q4_wave(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    uint16_t* __restrict__ out,
    int64_t out_group,
    int64_t in_features) {
  const int lane = threadIdx.x & 31;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tile = out_group >> 1;
  const int col_base = static_cast<int>(out_group & 1) * Q4_TILE_COLS;

  float acc[Q4_TILE_COLS];
#pragma unroll
  for (int local_col = 0; local_col < Q4_TILE_COLS; ++local_col) {
    acc[local_col] = 0.0f;
  }

  const int k_lane_base = lane * 8;
  const int sb = k_lane_base >> 5;
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles +
        (out_tile * blocks_per_row + blk) * Q4_T16_BLOCK_BYTES;

    float scale[Q4_TILE_COLS];
    float min_value[Q4_TILE_COLS];
#pragma unroll
    for (int local_pair = 0;
         local_pair < Q4_TILE_COLS / 2;
         ++local_pair) {
      const int local_col0 = local_pair * 2;
      const int local_col1 = local_col0 + 1;
      const int col0 = col_base + local_col0;
      const uint32_t d = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_D_OFFSET + col0 * 2);
      const uint32_t dmin = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_DMIN_OFFSET + col0 * 2);
      const uint16_t coeff_scale =
          *reinterpret_cast<const uint16_t*>(
              tile + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
      const uint16_t coeff_min =
          *reinterpret_cast<const uint16_t*>(
              tile + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
      scale[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(d)) *
          static_cast<float>(coeff_scale & 0xFFu);
      scale[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(d >> 16)) *
          static_cast<float>(coeff_scale >> 8);
      min_value[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin)) *
          static_cast<float>(coeff_min & 0xFFu);
      min_value[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin >> 16)) *
          static_cast<float>(coeff_min >> 8);
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int k_in_block = k_lane_base + j;
      const int lane32 = k_in_block & 31;
      const float xv = bf16_bits_to_float(
          x[blk * QK_K + k_in_block]);
#pragma unroll
      for (int local_pair = 0;
           local_pair < Q4_TILE_COLS / 2;
           ++local_pair) {
        const int local_col0 = local_pair * 2;
        const int local_col1 = local_col0 + 1;
        const int col0 = col_base + local_col0;
        const int q_index =
            Q4_T16_Q_OFFSET +
            (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) +
            (col0 >> 1);
        const uint8_t packed = tile[q_index];
        acc[local_col0] = fmaf(
            xv,
            static_cast<float>(packed & 0x0Fu) *
                    scale[local_col0] -
                min_value[local_col0],
            acc[local_col0]);
        acc[local_col1] = fmaf(
            xv,
            static_cast<float>(packed >> 4) *
                    scale[local_col1] -
                min_value[local_col1],
            acc[local_col1]);
      }
    }
  }

#pragma unroll
  for (int local_col = 0; local_col < Q4_TILE_COLS; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc[local_col]);
  }
  if (lane == 0) {
    const int64_t out_base = out_group * Q4_TILE_COLS;
#pragma unroll
    for (int local_col = 0; local_col < Q4_TILE_COLS; ++local_col) {
      out[out_base + local_col] =
          float_to_bf16_bits(0.0f + acc[local_col]);
    }
  }
}

__device__ __forceinline__ void run_q4_col4_wave(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    uint16_t* __restrict__ out,
    int64_t out_group,
    int64_t in_features) {
  constexpr int kTileCols = 4;
  constexpr int kGroupsPerTile = T16_COLS / kTileCols;
  const int lane = threadIdx.x & 31;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tile = out_group / kGroupsPerTile;
  const int col_base =
      static_cast<int>(out_group % kGroupsPerTile) * kTileCols;
  float acc[kTileCols];
#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    acc[local_col] = 0.0f;
  }

  const int k_lane_base = lane * 8;
  const int sb = k_lane_base >> 5;
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q4_T16_BLOCK_BYTES;
    float scale[kTileCols];
    float min_value[kTileCols];
#pragma unroll
    for (int local_pair = 0; local_pair < kTileCols / 2; ++local_pair) {
      const int local_col0 = local_pair * 2;
      const int local_col1 = local_col0 + 1;
      const int col0 = col_base + local_col0;
      const uint32_t d = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_D_OFFSET + col0 * 2);
      const uint32_t dmin = *reinterpret_cast<const uint32_t*>(
          tile + Q4_T16_DMIN_OFFSET + col0 * 2);
      const uint16_t coeff_scale = *reinterpret_cast<const uint16_t*>(
          tile + Q4_T16_SCALE_OFFSET + sb * T16_COLS + col0);
      const uint16_t coeff_min = *reinterpret_cast<const uint16_t*>(
          tile + Q4_T16_MIN_OFFSET + sb * T16_COLS + col0);
      scale[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(d)) *
          static_cast<float>(coeff_scale & 0xFFu);
      scale[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(d >> 16)) *
          static_cast<float>(coeff_scale >> 8);
      min_value[local_col0] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin)) *
          static_cast<float>(coeff_min & 0xFFu);
      min_value[local_col1] =
          fp16_bits_to_float(static_cast<uint16_t>(dmin >> 16)) *
          static_cast<float>(coeff_min >> 8);
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int k_in_block = k_lane_base + j;
      const int lane32 = k_in_block & 31;
      float weight[kTileCols];
#pragma unroll
      for (int local_pair = 0; local_pair < kTileCols / 2; ++local_pair) {
        const int local_col0 = local_pair * 2;
        const int local_col1 = local_col0 + 1;
        const int col0 = col_base + local_col0;
        const int q_index =
            Q4_T16_Q_OFFSET +
            (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) +
            (col0 >> 1);
        const uint8_t packed = tile[q_index];
        weight[local_col0] =
            static_cast<float>(packed & 0x0Fu) * scale[local_col0] -
            min_value[local_col0];
        weight[local_col1] =
            static_cast<float>(packed >> 4) * scale[local_col1] -
            min_value[local_col1];
      }
      const float xv = bf16_bits_to_float(x[blk * QK_K + k_in_block]);
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        acc[local_col] = fmaf(xv, weight[local_col], acc[local_col]);
      }
    }
  }

#pragma unroll
  for (int local_col = 0; local_col < kTileCols; ++local_col) {
    GGUF_T16_WAVE_REDUCE(acc[local_col]);
  }
  if (lane == 0) {
    const int64_t out_base = out_group * kTileCols;
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      out[out_base + local_col] =
          float_to_bf16_bits(0.0f + acc[local_col]);
    }
  }
}

__global__ __launch_bounds__(128, 2)
void q6_q4_t16_mixed_grid_pair_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ q6_tiles,
    const uint8_t* __restrict__ q4_tiles,
    uint16_t* __restrict__ q6_out,
    uint16_t* __restrict__ q4_out,
    int64_t in_features,
    int64_t q6_out_features,
    int64_t q4_out_features) {
  const int64_t q6_grid_blocks = q6_out_features / T16_COLS;
  if (static_cast<int64_t>(blockIdx.x) >= q6_grid_blocks) {
    const int64_t q4_block =
        static_cast<int64_t>(blockIdx.x) - q6_grid_blocks;
    const int64_t out_group =
        q4_block * 4 + static_cast<int64_t>(threadIdx.x >> 5);
    if (out_group < q4_out_features / Q4_TILE_COLS) {
      run_q4_wave(x, q4_tiles, q4_out, out_group, in_features);
    }
    return;
  }

  const int64_t out_tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;
  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc[col] = 0.0f;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        q6_tiles +
        (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid;
         k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const int g16 = k_in_block >> 4;
      const int8_t* __restrict__ scales =
          reinterpret_cast<const int8_t*>(
              tile + Q6_T16_SCALE_OFFSET);
      const float xv =
          bf16_bits_to_float(x[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float d = fp16_bytes_to_float(
            tile + Q6_T16_D_OFFSET + col * 2);
        const int scale_i =
            static_cast<int>(scales[g16 * T16_COLS + col]);
        const int q_i = t16_q6_quant(tile, k_in_block, col);
        acc[col] +=
            xv * (d * static_cast<float>(scale_i * q_i));
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    GGUF_T16_WAVE_REDUCE(acc[col]);
  }

  __shared__ float xchg[4 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg[sub_id * T16_COLS + col] = acc[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    uint16_t* __restrict__ out_row =
        q6_out + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < 4; ++g) {
        total += xchg[g * T16_COLS + col];
      }
      out_row[col] = float_to_bf16_bits(total);
    }
  }
}

__global__ __launch_bounds__(128, 2)
void q4_q4_t16_narrow_col4_pair_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    uint16_t* __restrict__ out_a,
    uint16_t* __restrict__ out_b,
    int64_t in_features,
    int64_t out_features_a) {
  const int64_t blocks_a = out_features_a / T16_COLS;
  const bool use_a = static_cast<int64_t>(blockIdx.x) < blocks_a;
  const int64_t pair_block = use_a
      ? static_cast<int64_t>(blockIdx.x)
      : static_cast<int64_t>(blockIdx.x) - blocks_a;
  const int64_t out_group =
      pair_block * 4 + static_cast<int64_t>(threadIdx.x >> 5);
  run_q4_col4_wave(
      x,
      use_a ? tiles_a : tiles_b,
      use_a ? out_a : out_b,
      out_group,
      in_features);
}

__global__ __launch_bounds__(128, 2)
void q4_q6_t16_narrow_col4_planar_pair_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ q4_tiles,
    const uint8_t* __restrict__ q6_tiles,
    uint16_t* __restrict__ q4_out,
    uint16_t* __restrict__ q6_out,
    int64_t in_features,
    int64_t q4_out_features) {
  const int64_t q4_blocks = q4_out_features / T16_COLS;
  if (static_cast<int64_t>(blockIdx.x) < q4_blocks) {
    const int64_t out_group =
        static_cast<int64_t>(blockIdx.x) * 4 +
        static_cast<int64_t>(threadIdx.x >> 5);
    run_q4_col4_wave(x, q4_tiles, q4_out, out_group, in_features);
    return;
  }

  const int64_t out_tile = static_cast<int64_t>(blockIdx.x) - q4_blocks;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;
  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc[col] = 0.0f;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        q6_tiles +
        (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    const int8_t* __restrict__ scales =
        reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
    for (int k_in_block = tid; k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const int g16 = k_in_block >> 4;
      const int subblock = k_in_block >> 5;
      const int pack = (k_in_block & 31) >> 2;
      const int k4 = k_in_block & 3;
      const float xv = bf16_bits_to_float(x[k_base + k_in_block]);
#pragma unroll
      for (int col_quartet = 0; col_quartet < T16_COLS / 4;
           ++col_quartet) {
        const int record_index =
            (subblock * (T16_COLS / 4) + col_quartet) * 8 + pack;
        const uint8_t* __restrict__ record =
            tile + Q6_T16_QMICRO_OFFSET +
            record_index * Q6_T16_QMICRO_RECORD_BYTES;
        const uint16_t low_pair =
            static_cast<uint16_t>(record[k4]) |
            (static_cast<uint16_t>(record[4 + k4]) << 8);
        const uint8_t high_packed = record[8 + k4];
#pragma unroll
        for (int col4 = 0; col4 < 4; ++col4) {
          const int col = col_quartet * 4 + col4;
          const uint8_t low_byte =
              static_cast<uint8_t>(low_pair >> (8 * (col4 >> 1)));
          const uint8_t low =
              (col4 & 1) ? (low_byte >> 4) : (low_byte & 0x0Fu);
          const uint8_t high =
              (high_packed >> (2 * col4)) & 0x03u;
          const int q_i = static_cast<int>(low | (high << 4)) - 32;
          const float d = fp16_bytes_to_float(
              tile + Q6_T16_D_OFFSET + col * 2);
          const int scale_i =
              static_cast<int>(scales[g16 * T16_COLS + col]);
          acc[col] +=
              xv * (d * static_cast<float>(scale_i * q_i));
        }
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    GGUF_T16_WAVE_REDUCE(acc[col]);
  }
  __shared__ float xchg[4 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg[sub_id * T16_COLS + col] = acc[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    uint16_t* __restrict__ out_row = q6_out + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        total += xchg[g * T16_COLS + col];
      }
      out_row[col] = float_to_bf16_bits(total);
    }
  }
}

}  // namespace

extern "C" int
hipengine_gguf_q6_q4_t16_mixed_grid_pair_bf16_bf16_out(
    const void* x,
    const void* q6_tiles,
    const void* q4_tiles,
    void* q6_out,
    void* q4_out,
    int64_t rows,
    int64_t in_features,
    int64_t q6_out_features,
    int64_t q4_out_features,
    hipStream_t stream) {
  if (x == nullptr || q6_tiles == nullptr || q4_tiles == nullptr ||
      q6_out == nullptr || q4_out == nullptr || rows != 1 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      q6_out_features <= 0 || (q6_out_features % T16_COLS) != 0 ||
      q4_out_features <= 0 || (q4_out_features % 32) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t q6_blocks = q6_out_features / T16_COLS;
  const int64_t q4_blocks = q4_out_features / 32;
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(q6_blocks + q4_blocks));
  hipLaunchKernelGGL(
      q6_q4_t16_mixed_grid_pair_bf16_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const uint8_t*>(q6_tiles),
      reinterpret_cast<const uint8_t*>(q4_tiles),
      reinterpret_cast<uint16_t*>(q6_out),
      reinterpret_cast<uint16_t*>(q4_out),
      in_features,
      q6_out_features,
      q4_out_features);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_gguf_q4_q4_t16_narrow_col4_pair_bf16_bf16_out(
    const void* x,
    const void* tiles_a,
    const void* tiles_b,
    void* out_a,
    void* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    hipStream_t stream) {
  if (x == nullptr || tiles_a == nullptr || tiles_b == nullptr ||
      out_a == nullptr || out_b == nullptr || rows != 1 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features_a <= 0 || (out_features_a % T16_COLS) != 0 ||
      out_features_b <= 0 || (out_features_b % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(
      (out_features_a + out_features_b) / T16_COLS));
  hipLaunchKernelGGL(
      q4_q4_t16_narrow_col4_pair_bf16_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles_a),
      reinterpret_cast<const uint8_t*>(tiles_b),
      reinterpret_cast<uint16_t*>(out_a),
      reinterpret_cast<uint16_t*>(out_b),
      in_features,
      out_features_a);
  return static_cast<int>(hipGetLastError());
}

extern "C" int
hipengine_gguf_q4_q6_t16_narrow_col4_planar_pair_bf16_bf16_out(
    const void* x,
    const void* q4_tiles,
    const void* q6_tiles,
    void* q4_out,
    void* q6_out,
    int64_t rows,
    int64_t in_features,
    int64_t q4_out_features,
    int64_t q6_out_features,
    hipStream_t stream) {
  if (x == nullptr || q4_tiles == nullptr || q6_tiles == nullptr ||
      q4_out == nullptr || q6_out == nullptr || rows != 1 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      q4_out_features <= 0 || (q4_out_features % T16_COLS) != 0 ||
      q6_out_features <= 0 || (q6_out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(
      (q4_out_features + q6_out_features) / T16_COLS));
  hipLaunchKernelGGL(
      q4_q6_t16_narrow_col4_planar_pair_bf16_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const uint8_t*>(q4_tiles),
      reinterpret_cast<const uint8_t*>(q6_tiles),
      reinterpret_cast<uint16_t*>(q4_out),
      reinterpret_cast<uint16_t*>(q6_out),
      in_features,
      q4_out_features);
  return static_cast<int>(hipGetLastError());
}
