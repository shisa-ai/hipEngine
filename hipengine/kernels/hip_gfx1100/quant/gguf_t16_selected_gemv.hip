// Selected MoE decode-shaped GEMV for GGUF K-family T16 replacement layouts.
//
// P9.H3 slice: consumes resident T16 tiles instead of raw row-major GGUF
// expert bytes.  Q4_K uses the dual gate/up selected ABI and can also use the
// single-output down-projection ABI for Q4_K_S; Q5_K/Q6_K use the same
// single-output selected down-projection ABI.  The compact-MoE scheduler ABI
// is the same as the P9.B raw pack8 kernels:
//
//   x                    [compact_rows, in_features]
//   expert_start_compact [num_experts + 1]
//   tiles                [E, out_tiles16, blocks_per_row, tile_bytes]
//   out                  row-major compact output
//
// One block computes one 16-column output tile for one compact row.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int T16_COLS = 16;
constexpr int Q4_SUBBLOCK = 32;
constexpr int Q4_SUBBLOCKS = 8;
constexpr int Q6_GROUPS16 = 16;

constexpr int Q4_T16_D_OFFSET = 0;
constexpr int Q4_T16_DMIN_OFFSET = Q4_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_SCALE_OFFSET = Q4_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_MIN_OFFSET = Q4_T16_SCALE_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_Q_OFFSET = Q4_T16_MIN_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_BLOCK_BYTES = Q4_T16_Q_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);

constexpr int Q5_T16_D_OFFSET = 0;
constexpr int Q5_T16_DMIN_OFFSET = Q5_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q5_T16_SCALE_OFFSET = Q5_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q5_T16_MIN_OFFSET = Q5_T16_SCALE_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q5_T16_QL_OFFSET = Q5_T16_MIN_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q5_T16_QH_OFFSET = Q5_T16_QL_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);
constexpr int Q5_T16_BLOCK_BYTES = Q5_T16_QH_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 8);

constexpr int Q6_T16_D_OFFSET = 0;
constexpr int Q6_T16_SCALE_OFFSET = Q6_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q6_T16_QL_OFFSET = Q6_T16_SCALE_OFFSET + Q6_GROUPS16 * T16_COLS;
constexpr int Q6_T16_QH_OFFSET = Q6_T16_QL_OFFSET + QK_K * (T16_COLS / 2);
constexpr int Q6_T16_BLOCK_BYTES = Q6_T16_QH_OFFSET + QK_K * (T16_COLS / 4);

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

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t v) {
  return bf16_bits_to_float(v);
}

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

template <>
__device__ inline uint16_t float_to_scalar<uint16_t>(float v) {
  return float_to_bf16_bits(v);
}

template <typename scalar_t>
__device__ inline float round_trip_scalar_to_float(float v) {
  return scalar_to_float<scalar_t>(float_to_scalar<scalar_t>(v));
}

#define GGUF_T16_WAVE_REDUCE(acc) \
  do {                            \
    acc += __shfl_down(acc, 16);  \
    acc += __shfl_down(acc, 8);   \
    acc += __shfl_down(acc, 4);   \
    acc += __shfl_down(acc, 2);   \
    acc += __shfl_down(acc, 1);   \
  } while (0)

__device__ inline int64_t find_expert_for_compact_row(
    const int64_t* __restrict__ expert_start_compact,
    int64_t compact_row,
    int64_t num_experts) {
  for (int e = 0; e < static_cast<int>(num_experts); ++e) {
    if (compact_row >= expert_start_compact[e] && compact_row < expert_start_compact[e + 1]) {
      return static_cast<int64_t>(e);
    }
  }
  return -1;
}

template <int tile_bytes>
__device__ inline const uint8_t* expert_tile_ptr(
    const uint8_t* __restrict__ tiles,
    int64_t expert_id,
    int64_t out_tile,
    int64_t block_idx,
    int64_t out_tiles,
    int64_t blocks_per_row) {
  const int64_t tile_index = (expert_id * out_tiles + out_tile) * blocks_per_row + block_idx;
  return tiles + tile_index * tile_bytes;
}

__device__ inline uint8_t t16_q4_quant(const uint8_t* __restrict__ tile, int sb, int lane32, int col) {
  const uint8_t packed = tile[Q4_T16_Q_OFFSET + (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) + (col >> 1)];
  return (col & 1) ? (packed >> 4) : (packed & 0x0Fu);
}

__device__ inline uint8_t t16_q5_quant(const uint8_t* __restrict__ tile, int sb, int lane32, int col) {
  const uint8_t low_packed = tile[Q5_T16_QL_OFFSET + (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) + (col >> 1)];
  const uint8_t low = (col & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed = tile[Q5_T16_QH_OFFSET + (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 8) + (col >> 3)];
  const uint8_t high = (high_packed >> (col & 7)) & 0x01u;
  return low | (high << 4);
}

__device__ inline float t16_q4_weight(const uint8_t* __restrict__ tile, int k_in_block, int col) {
  const int sb = k_in_block >> 5;
  const int lane32 = k_in_block & 31;
  const float d = fp16_bytes_to_float(tile + Q4_T16_D_OFFSET + col * 2);
  const float dmin = fp16_bytes_to_float(tile + Q4_T16_DMIN_OFFSET + col * 2);
  const float scale = static_cast<float>(tile[Q4_T16_SCALE_OFFSET + sb * T16_COLS + col]);
  const float minv = static_cast<float>(tile[Q4_T16_MIN_OFFSET + sb * T16_COLS + col]);
  const float q = static_cast<float>(t16_q4_quant(tile, sb, lane32, col));
  return d * scale * q - dmin * minv;
}

__device__ inline int t16_q6_quant(const uint8_t* __restrict__ tile, int k_in_block, int col) {
  const uint8_t low_packed = tile[Q6_T16_QL_OFFSET + k_in_block * (T16_COLS / 2) + (col >> 1)];
  const uint8_t low = (col & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed = tile[Q6_T16_QH_OFFSET + k_in_block * (T16_COLS / 4) + (col >> 2)];
  const uint8_t high = (high_packed >> (2 * (col & 3))) & 0x03u;
  return static_cast<int>(low | (high << 4)) - 32;
}

template <typename scalar_t>
__global__ __launch_bounds__(256, 2) void q4_k_t16_selected_dual_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts) {
  const int64_t global_tile = blockIdx.x;
  const int64_t compact_row = blockIdx.y;
  const int tid = threadIdx.x;
  if (compact_row >= compact_rows) return;

  const int64_t out_tiles_a = out_features_a / T16_COLS;
  const int64_t out_tiles_b = out_features_b / T16_COLS;
  const bool use_b = global_tile >= out_tiles_a;
  const int64_t local_tile = use_b ? (global_tile - out_tiles_a) : global_tile;
  if ((!use_b && local_tile >= out_tiles_a) || (use_b && local_tile >= out_tiles_b)) return;

  const int64_t expert_id = find_expert_for_compact_row(expert_start_compact, compact_row, num_experts);
  if (expert_id < 0) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_stride = out_features_a + out_features_b;
  const int64_t out_base = use_b ? out_features_a : 0;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const int64_t out_tiles = use_b ? out_tiles_b : out_tiles_a;
  const uint8_t* __restrict__ tiles = use_b ? tiles_b : tiles_a;
  const scalar_t* __restrict__ x_row = x + compact_row * in_features;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles, expert_id, local_tile, blk, out_tiles, blocks_per_row);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int sb = k_in_block >> 5;
      const int lane32 = k_in_block & 31;
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float wv = t16_q4_weight(tile, k_in_block, col);
        acc[col] += xv * wv;
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc[col]);

  __shared__ float xchg[8 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) xchg[sub_id * T16_COLS + col] = acc[col];
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row = out + compact_row * out_stride + out_base + local_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) total += xchg[g * T16_COLS + col];
      if (local_tile * T16_COLS + col < local_out_features) {
        out_row[col] = float_to_scalar<scalar_t>(total);
      }
    }
  }
}

template <typename scalar_t, int qtype>
__global__ __launch_bounds__(128, 4) void qk_t16_selected_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  static_assert(qtype == 4 || qtype == 5 || qtype == 6, "qtype must be 4, 5, or 6");
  const int64_t out_tile = blockIdx.x;
  const int64_t compact_row = blockIdx.y;
  const int tid = threadIdx.x;
  if (compact_row >= compact_rows) return;

  const int64_t expert_id = find_expert_for_compact_row(expert_start_compact, compact_row, num_experts);
  if (expert_id < 0) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x_row = x + compact_row * in_features;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile;
    if constexpr (qtype == 4) {
      tile = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    } else if constexpr (qtype == 5) {
      tile = expert_tile_ptr<Q5_T16_BLOCK_BYTES>(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    } else {
      tile = expert_tile_ptr<Q6_T16_BLOCK_BYTES>(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    }
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
      if constexpr (qtype == 4) {
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float wv = t16_q4_weight(tile, k_in_block, col);
          acc[col] += xv * wv;
        }
      } else if constexpr (qtype == 5) {
        const int sb = k_in_block >> 5;
        const int lane32 = k_in_block & 31;
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float d = fp16_bytes_to_float(tile + Q5_T16_D_OFFSET + col * 2);
          const float dmin = fp16_bytes_to_float(tile + Q5_T16_DMIN_OFFSET + col * 2);
          const float scale = d * static_cast<float>(tile[Q5_T16_SCALE_OFFSET + sb * T16_COLS + col]);
          const float minv = dmin * static_cast<float>(tile[Q5_T16_MIN_OFFSET + sb * T16_COLS + col]);
          const float q = static_cast<float>(t16_q5_quant(tile, sb, lane32, col));
          acc[col] += xv * (scale * q - minv);
        }
      } else {
        const int g16 = k_in_block >> 4;
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float d = fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
          const int8_t* __restrict__ scales = reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
          const float scale = d * static_cast<float>(scales[g16 * T16_COLS + col]);
          const float q = static_cast<float>(t16_q6_quant(tile, k_in_block, col));
          acc[col] += xv * (scale * q);
        }
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc[col]);

  __shared__ float xchg[4 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) xchg[sub_id * T16_COLS + col] = acc[col];
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row = out + compact_row * out_features + out_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) total += xchg[g * T16_COLS + col];
      out_row[col] = float_to_scalar<scalar_t>(total);
    }
  }
}


template <typename scalar_t>
__global__ __launch_bounds__(256, 2) void q4_k_t16_selected_dual_direct_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out_a,
    scalar_t* __restrict__ out_b,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x_row = x + x_row_idx * in_features;

  float acc_a[T16_COLS];
  float acc_b[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc_a[col] = 0.0f;
    acc_b[col] = 0.0f;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile_a = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles_a, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const uint8_t* __restrict__ tile_b = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles_b, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int sb = k_in_block >> 5;
      const int lane32 = k_in_block & 31;
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float wv_a = t16_q4_weight(tile_a, k_in_block, col);
        const float wv_b = t16_q4_weight(tile_b, k_in_block, col);
        acc_a[col] += xv * wv_a;
        acc_b[col] += xv * wv_b;
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc_a[col]);
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc_b[col]);

  __shared__ float xchg_a[8 * T16_COLS];
  __shared__ float xchg_b[8 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg_a[sub_id * T16_COLS + col] = acc_a[col];
      xchg_b[sub_id * T16_COLS + col] = acc_b[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row_a = out_a + row * out_features + out_tile * T16_COLS;
    scalar_t* __restrict__ out_row_b = out_b + row * out_features + out_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total_a = 0.0f;
      float total_b = 0.0f;
      for (int g = 0; g < sub_groups; ++g) {
        total_a += xchg_a[g * T16_COLS + col];
        total_b += xchg_b[g * T16_COLS + col];
      }
      out_row_a[col] = float_to_scalar<scalar_t>(total_a);
      out_row_b[col] = float_to_scalar<scalar_t>(total_b);
    }
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(256, 1) void q4_k_t16_selected_dual_silu_direct_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x_row = x + x_row_idx * in_features;

  float acc_a[T16_COLS];
  float acc_b[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc_a[col] = 0.0f;
    acc_b[col] = 0.0f;
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile_a = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles_a, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const uint8_t* __restrict__ tile_b = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(
        tiles_b, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float wv_a = t16_q4_weight(tile_a, k_in_block, col);
        const float wv_b = t16_q4_weight(tile_b, k_in_block, col);
        acc_a[col] += xv * wv_a;
        acc_b[col] += xv * wv_b;
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc_a[col]);
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc_b[col]);

  __shared__ float xchg_a[8 * T16_COLS];
  __shared__ float xchg_b[8 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      xchg_a[sub_id * T16_COLS + col] = acc_a[col];
      xchg_b[sub_id * T16_COLS + col] = acc_b[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row = out + row * out_features + out_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total_a = 0.0f;
      float total_b = 0.0f;
      for (int g = 0; g < sub_groups; ++g) {
        total_a += xchg_a[g * T16_COLS + col];
        total_b += xchg_b[g * T16_COLS + col];
      }
      // Preserve the split-kernel reduction/rounding contract: the legacy
      // direct path writes BF16 gate/up first, then the SiLU kernel reads those
      // rounded values.  Round-trip through scalar_t before applying SiLU so
      // P9.E2 full-model logits remain aligned with the existing reference.
      const float gate = round_trip_scalar_to_float<scalar_t>(total_a);
      const float up = round_trip_scalar_to_float<scalar_t>(total_b);
      const float sig = 1.0f / (1.0f + expf(-gate));
      out_row[col] = float_to_scalar<scalar_t>(gate * sig * up);
    }
  }
}

template <typename scalar_t, int qtype>
__global__ __launch_bounds__(128, 2) void qk_t16_selected_direct_gemv_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  static_assert(qtype == 4 || qtype == 5 || qtype == 6, "qtype must be 4, 5, or 6");
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row_idx = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  const scalar_t* __restrict__ x_row = x + x_row_idx * in_features;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile;
    if constexpr (qtype == 4) {
      tile = expert_tile_ptr<Q4_T16_BLOCK_BYTES>(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    } else if constexpr (qtype == 5) {
      tile = expert_tile_ptr<Q5_T16_BLOCK_BYTES>(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    } else {
      tile = expert_tile_ptr<Q6_T16_BLOCK_BYTES>(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row);
    }
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
      if constexpr (qtype == 4) {
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float wv = t16_q4_weight(tile, k_in_block, col);
          acc[col] += xv * wv;
        }
      } else if constexpr (qtype == 5) {
        const int sb = k_in_block >> 5;
        const int lane32 = k_in_block & 31;
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float d = fp16_bytes_to_float(tile + Q5_T16_D_OFFSET + col * 2);
          const float dmin = fp16_bytes_to_float(tile + Q5_T16_DMIN_OFFSET + col * 2);
          const float scale = d * static_cast<float>(tile[Q5_T16_SCALE_OFFSET + sb * T16_COLS + col]);
          const float minv = dmin * static_cast<float>(tile[Q5_T16_MIN_OFFSET + sb * T16_COLS + col]);
          const float q = static_cast<float>(t16_q5_quant(tile, sb, lane32, col));
          acc[col] += xv * (scale * q - minv);
        }
      } else {
        const int g16 = k_in_block >> 4;
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          const float d = fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
          const int8_t* __restrict__ scales = reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
          const float scale = d * static_cast<float>(scales[g16 * T16_COLS + col]);
          const float q = static_cast<float>(t16_q6_quant(tile, k_in_block, col));
          acc[col] += xv * (scale * q);
        }
      }
    }
  }

#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc[col]);

  __shared__ float xchg[4 * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) xchg[sub_id * T16_COLS + col] = acc[col];
  }
  __syncthreads();

  if (tid == 0) {
    scalar_t* __restrict__ out_row = out + row * out_features + out_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) total += xchg[g * T16_COLS + col];
      out_row[col] = float_to_scalar<scalar_t>(total);
    }
  }
}

template <typename scalar_t>
int launch_q4_dual(const void* x,
                   const void* expert_start_compact,
                   const void* tiles_a,
                   const void* tiles_b,
                   void* out,
                   int64_t compact_rows,
                   int64_t in_features,
                   int64_t out_features_a,
                   int64_t out_features_b,
                   int64_t num_experts,
                   hipStream_t stream) {
  if (!x || !expert_start_compact || !tiles_a || !tiles_b || !out || compact_rows <= 0 ||
      in_features <= 0 || out_features_a <= 0 || out_features_b <= 0 || num_experts <= 0 ||
      (in_features % QK_K) != 0 || (out_features_a % T16_COLS) != 0 ||
      (out_features_b % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>((out_features_a + out_features_b) / T16_COLS),
                  static_cast<unsigned int>(compact_rows));
  hipLaunchKernelGGL((q4_k_t16_selected_dual_gemv_kernel<scalar_t>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x),
                     reinterpret_cast<const int64_t*>(expert_start_compact),
                     reinterpret_cast<const uint8_t*>(tiles_a),
                     reinterpret_cast<const uint8_t*>(tiles_b),
                     reinterpret_cast<scalar_t*>(out), compact_rows, in_features,
                     out_features_a, out_features_b, num_experts);
  return hipGetLastError();
}

template <typename scalar_t, int qtype>
int launch_qk_single(const void* x,
                     const void* expert_start_compact,
                     const void* tiles,
                     void* out,
                     int64_t compact_rows,
                     int64_t in_features,
                     int64_t out_features,
                     int64_t num_experts,
                     hipStream_t stream) {
  if (!x || !expert_start_compact || !tiles || !out || compact_rows <= 0 || in_features <= 0 ||
      out_features <= 0 || num_experts <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS),
                  static_cast<unsigned int>(compact_rows));
  hipLaunchKernelGGL((qk_t16_selected_gemv_kernel<scalar_t, qtype>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x),
                     reinterpret_cast<const int64_t*>(expert_start_compact),
                     reinterpret_cast<const uint8_t*>(tiles), reinterpret_cast<scalar_t*>(out),
                     compact_rows, in_features, out_features, num_experts);
  return hipGetLastError();
}


template <typename scalar_t>
int launch_q4_dual_direct(const void* x,
                          const void* selected,
                          const void* tiles_a,
                          const void* tiles_b,
                          void* out_a,
                          void* out_b,
                          int64_t x_rows,
                          int64_t rows,
                          int64_t in_features,
                          int64_t out_features,
                          int64_t num_experts,
                          hipStream_t stream) {
  if (!x || !selected || !tiles_a || !tiles_b || !out_a || !out_b || x_rows <= 0 || rows <= 0 ||
      num_experts <= 0 || in_features <= 0 || out_features <= 0 || (rows % x_rows) != 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  // Match the legacy raw-GGUF selected dual GEMV reduction topology (128
  // threads / 4 waves) so replacement-layout Q4 gate/up decode remains
  // bitwise-aligned with the P9.E2 row-GEMV reference path.
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q4_k_t16_selected_dual_direct_gemv_kernel<scalar_t>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x), reinterpret_cast<const int64_t*>(selected),
                     reinterpret_cast<const uint8_t*>(tiles_a), reinterpret_cast<const uint8_t*>(tiles_b),
                     reinterpret_cast<scalar_t*>(out_a), reinterpret_cast<scalar_t*>(out_b),
                     x_rows, rows, in_features, out_features, num_experts);
  return hipGetLastError();
}

template <typename scalar_t>
int launch_q4_dual_silu_direct(const void* x,
                                const void* selected,
                                const void* tiles_a,
                                const void* tiles_b,
                                void* out,
                                int64_t x_rows,
                                int64_t rows,
                                int64_t in_features,
                                int64_t out_features,
                                int64_t num_experts,
                                hipStream_t stream) {
  if (!x || !selected || !tiles_a || !tiles_b || !out || x_rows <= 0 || rows <= 0 ||
      rows % x_rows != 0 || in_features <= 0 || out_features <= 0 || num_experts <= 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q4_k_t16_selected_dual_silu_direct_gemv_kernel<scalar_t>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x), reinterpret_cast<const int64_t*>(selected),
                     reinterpret_cast<const uint8_t*>(tiles_a), reinterpret_cast<const uint8_t*>(tiles_b),
                     reinterpret_cast<scalar_t*>(out), x_rows, rows, in_features, out_features,
                     num_experts);
  return static_cast<int>(hipGetLastError());
}


template <typename scalar_t, int qtype>
int launch_qk_direct(const void* x,
                     const void* selected,
                     const void* tiles,
                     void* out,
                     int64_t x_rows,
                     int64_t rows,
                     int64_t in_features,
                     int64_t out_features,
                     int64_t num_experts,
                     hipStream_t stream) {
  if (!x || !selected || !tiles || !out || x_rows <= 0 || rows <= 0 || num_experts <= 0 ||
      in_features <= 0 || out_features <= 0 || (rows % x_rows) != 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((qk_t16_selected_direct_gemv_kernel<scalar_t, qtype>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x), reinterpret_cast<const int64_t*>(selected),
                     reinterpret_cast<const uint8_t*>(tiles), reinterpret_cast<scalar_t*>(out),
                     x_rows, rows, in_features, out_features, num_experts);
  return hipGetLastError();
}

}  // namespace


extern "C" int hipengine_gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_a, const void* tiles_b,
    void* out_a, void* out_b, int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q4_dual_direct<uint16_t>(x, selected, tiles_a, tiles_b, out_a, out_b,
                                         x_rows, rows, in_features, out_features,
                                         num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_gemv_fp16_fp16_out(
    const void* x, const void* selected, const void* tiles_a, const void* tiles_b,
    void* out_a, void* out_b, int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q4_dual_direct<half_t>(x, selected, tiles_a, tiles_b, out_a, out_b,
                                       x_rows, rows, in_features, out_features,
                                       num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles_a, const void* tiles_b,
    void* out, int64_t x_rows, int64_t rows, int64_t num_experts,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q4_dual_silu_direct<uint16_t>(x, selected, tiles_a, tiles_b, out,
                                               x_rows, rows, in_features,
                                               out_features, num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<uint16_t, 4>(x, selected, tiles, out, x_rows, rows,
                                      in_features, out_features, num_experts,
                                      stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_gemv_fp16_fp16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<half_t, 4>(x, selected, tiles, out, x_rows, rows,
                                    in_features, out_features, num_experts,
                                    stream);
}

extern "C" int hipengine_gguf_q5_k_t16_selected_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<uint16_t, 5>(x, selected, tiles, out, x_rows, rows,
                                      in_features, out_features, num_experts,
                                      stream);
}

extern "C" int hipengine_gguf_q5_k_t16_selected_gemv_fp16_fp16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<half_t, 5>(x, selected, tiles, out, x_rows, rows,
                                    in_features, out_features, num_experts,
                                    stream);
}

extern "C" int hipengine_gguf_q6_k_t16_selected_gemv_bf16_bf16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<uint16_t, 6>(x, selected, tiles, out, x_rows, rows,
                                      in_features, out_features, num_experts,
                                      stream);
}

extern "C" int hipengine_gguf_q6_k_t16_selected_gemv_fp16_fp16_out(
    const void* x, const void* selected, const void* tiles, void* out,
    int64_t x_rows, int64_t rows, int64_t num_experts, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_qk_direct<half_t, 6>(x, selected, tiles, out, x_rows, rows,
                                    in_features, out_features, num_experts,
                                    stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_gemv_decode_compact_bf16_bf16_out(
    const void* x, const void* expert_start_compact, const void* tiles_a, const void* tiles_b,
    void* out, int64_t compact_rows, int64_t in_features, int64_t out_features_a,
    int64_t out_features_b, int64_t num_experts, hipStream_t stream) {
  return launch_q4_dual<uint16_t>(x, expert_start_compact, tiles_a, tiles_b, out,
                                  compact_rows, in_features, out_features_a,
                                  out_features_b, num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_gemv_decode_compact_fp16_fp16_out(
    const void* x, const void* expert_start_compact, const void* tiles_a, const void* tiles_b,
    void* out, int64_t compact_rows, int64_t in_features, int64_t out_features_a,
    int64_t out_features_b, int64_t num_experts, hipStream_t stream) {
  return launch_q4_dual<half_t>(x, expert_start_compact, tiles_a, tiles_b, out,
                                compact_rows, in_features, out_features_a,
                                out_features_b, num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_gemv_decode_compact_bf16_bf16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_single<uint16_t, 4>(x, expert_start_compact, tiles, out,
                                      compact_rows, in_features, out_features,
                                      num_experts, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_gemv_decode_compact_fp16_fp16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_single<half_t, 4>(x, expert_start_compact, tiles, out,
                                    compact_rows, in_features, out_features,
                                    num_experts, stream);
}

extern "C" int hipengine_gguf_q5_k_t16_selected_gemv_decode_compact_bf16_bf16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_single<uint16_t, 5>(x, expert_start_compact, tiles, out,
                                      compact_rows, in_features, out_features,
                                      num_experts, stream);
}

extern "C" int hipengine_gguf_q5_k_t16_selected_gemv_decode_compact_fp16_fp16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_single<half_t, 5>(x, expert_start_compact, tiles, out,
                                    compact_rows, in_features, out_features,
                                    num_experts, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_selected_gemv_decode_compact_bf16_bf16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_single<uint16_t, 6>(x, expert_start_compact, tiles, out,
                                      compact_rows, in_features, out_features,
                                      num_experts, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_selected_gemv_decode_compact_fp16_fp16_out(
    const void* x, const void* expert_start_compact, const void* tiles, void* out,
    int64_t compact_rows, int64_t in_features, int64_t out_features,
    int64_t num_experts, hipStream_t stream) {
  return launch_qk_single<half_t, 6>(x, expert_start_compact, tiles, out,
                                    compact_rows, in_features, out_features,
                                    num_experts, stream);
}
