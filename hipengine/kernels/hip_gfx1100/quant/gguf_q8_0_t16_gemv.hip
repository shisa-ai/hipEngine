// Dense decode-shaped GEMV for the GGUF Q8_0 T16 replacement layout.
//
// P9.H3 slice: consumes the byte-neutral Q8T16 layout from
// ``hipengine.quant.gguf_t16`` instead of raw row-major GGUF Q8_0 blocks.
// One tile stores 16 adjacent output columns for one 32-wide K block:
//
//   d[16] fp16 bytes followed by q8[32 K lanes, 16 columns] int8 bytes
//
// Grid: one block computes one (row, out_tile16). Accumulation is FP32 and the
// output scalar type matches the existing Q8_0 pack8 decode wrappers.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int Q8_0_BLOCK = 32;
constexpr int Q8_1_BLOCK = 32;
constexpr int T16_COLS = 16;
constexpr int T16_D_BYTES = T16_COLS * 2;
constexpr int T16_BLOCK_BYTES = T16_D_BYTES + Q8_0_BLOCK * T16_COLS;

struct gguf_q8_1_block {
  half_t d;
  half_t s;
  int8_t qs[Q8_1_BLOCK];
};

static_assert(sizeof(gguf_q8_1_block) == 36, "GGUF q8_1 block layout must match GGML");

bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128;
}

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

__device__ inline int q8_0_dp4a(uint32_t w, uint32_t x, int c) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(true, static_cast<int>(w), true, static_cast<int>(x), c, false);
#else
  int acc = c;
  for (int i = 0; i < 4; ++i) {
    acc += static_cast<int>(static_cast<int8_t>((w >> (8 * i)) & 0xFFu)) *
           static_cast<int>(static_cast<int8_t>((x >> (8 * i)) & 0xFFu));
  }
  return acc;
#endif
}

__device__ inline uint32_t pack_t16_col4(const int8_t* __restrict__ q, int lane0, int col) {
  const int base = lane0 * T16_COLS + col;
  return static_cast<uint32_t>(static_cast<uint8_t>(q[base + 0 * T16_COLS])) |
         (static_cast<uint32_t>(static_cast<uint8_t>(q[base + 1 * T16_COLS])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(q[base + 2 * T16_COLS])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(q[base + 3 * T16_COLS])) << 24);
}

#define GGUF_T16_WAVE_REDUCE(acc) \
  do {                            \
    acc += __shfl_down(acc, 16);  \
    acc += __shfl_down(acc, 8);   \
    acc += __shfl_down(acc, 4);   \
    acc += __shfl_down(acc, 2);   \
    acc += __shfl_down(acc, 1);   \
  } while (0)

template <typename scalar_t, typename out_t>
__global__ __launch_bounds__(128, 4) void q8_0_t16_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int out_tile = static_cast<int>(blockIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / Q8_0_BLOCK;
  const scalar_t* __restrict__ x_row = x + static_cast<int64_t>(row) * in_features;
  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc[col] = 0.0f;
  }

  for (int64_t k = tid; k < in_features; k += blockDim.x) {
    const int64_t block_idx = k / Q8_0_BLOCK;
    const int lane = static_cast<int>(k & (Q8_0_BLOCK - 1));
    const uint8_t* __restrict__ tile =
        tiles + (static_cast<int64_t>(out_tile) * blocks_per_row + block_idx) * T16_BLOCK_BYTES;
    const int8_t* __restrict__ q = reinterpret_cast<const int8_t*>(tile + T16_D_BYTES + lane * T16_COLS);
    const float xv = scalar_to_float<scalar_t>(x_row[k]);
    float d[T16_COLS];
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      d[col] = fp16_bytes_to_float(tile + col * 2);
    }
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      acc[col] += xv * (d[col] * static_cast<float>(q[col]));
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
    const int sub_groups = blockDim.x >> 5;
    out_t* __restrict__ out_row = out + static_cast<int64_t>(row) * out_features + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) {
        total += xchg[g * T16_COLS + col];
      }
      out_row[col] = float_to_scalar<out_t>(total);
    }
  }
}

template <typename scalar_t, typename out_t>
__global__ __launch_bounds__(128, 4) void q8_0_t16_dual_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b) {
  const int total_tiles_a = static_cast<int>(out_features_a / T16_COLS);
  const int out_tile_global = static_cast<int>(blockIdx.x);
  const bool use_b = out_tile_global >= total_tiles_a;
  const int local_tile = use_b ? (out_tile_global - total_tiles_a) : out_tile_global;
  const int row = static_cast<int>(blockIdx.y);
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / Q8_0_BLOCK;
  const uint8_t* __restrict__ tiles = use_b ? tiles_b : tiles_a;
  const int64_t out_stride = out_features_a + out_features_b;
  const int64_t out_base = use_b ? out_features_a : 0;
  const scalar_t* __restrict__ x_row = x + static_cast<int64_t>(row) * in_features;
  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc[col] = 0.0f;
  }

  for (int64_t k = tid; k < in_features; k += blockDim.x) {
    const int64_t block_idx = k / Q8_0_BLOCK;
    const int lane = static_cast<int>(k & (Q8_0_BLOCK - 1));
    const uint8_t* __restrict__ tile =
        tiles + (static_cast<int64_t>(local_tile) * blocks_per_row + block_idx) * T16_BLOCK_BYTES;
    const int8_t* __restrict__ q = reinterpret_cast<const int8_t*>(tile + T16_D_BYTES + lane * T16_COLS);
    const float xv = scalar_to_float<scalar_t>(x_row[k]);
    float d[T16_COLS];
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      d[col] = fp16_bytes_to_float(tile + col * 2);
    }
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      acc[col] += xv * (d[col] * static_cast<float>(q[col]));
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
    const int sub_groups = blockDim.x >> 5;
    out_t* __restrict__ out_row = out + static_cast<int64_t>(row) * out_stride + out_base + local_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) {
        total += xchg[g * T16_COLS + col];
      }
      out_row[col] = float_to_scalar<out_t>(total);
    }
  }
}

template <typename scalar_t, typename out_t>
__global__ __launch_bounds__(128, 4) void q8_0_t16_dual_split_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b) {
  const int total_tiles_a = static_cast<int>(out_features_a / T16_COLS);
  const int out_tile_global = static_cast<int>(blockIdx.x);
  const bool use_b = out_tile_global >= total_tiles_a;
  const int local_tile = use_b ? (out_tile_global - total_tiles_a) : out_tile_global;
  const int row = static_cast<int>(blockIdx.y);
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / Q8_0_BLOCK;
  const uint8_t* __restrict__ tiles = use_b ? tiles_b : tiles_a;
  out_t* __restrict__ out = use_b ? out_b : out_a;
  const int64_t out_features = use_b ? out_features_b : out_features_a;
  const scalar_t* __restrict__ x_row = x + static_cast<int64_t>(row) * in_features;
  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc[col] = 0.0f;
  }

  for (int64_t k = tid; k < in_features; k += blockDim.x) {
    const int64_t block_idx = k / Q8_0_BLOCK;
    const int lane = static_cast<int>(k & (Q8_0_BLOCK - 1));
    const uint8_t* __restrict__ tile =
        tiles + (static_cast<int64_t>(local_tile) * blocks_per_row + block_idx) * T16_BLOCK_BYTES;
    const int8_t* __restrict__ q = reinterpret_cast<const int8_t*>(tile + T16_D_BYTES + lane * T16_COLS);
    const float xv = scalar_to_float<scalar_t>(x_row[k]);
    float d[T16_COLS];
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      d[col] = fp16_bytes_to_float(tile + col * 2);
    }
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      acc[col] += xv * (d[col] * static_cast<float>(q[col]));
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
    const int sub_groups = blockDim.x >> 5;
    out_t* __restrict__ out_row = out + static_cast<int64_t>(row) * out_features + local_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) {
        total += xchg[g * T16_COLS + col];
      }
      out_row[col] = float_to_scalar<out_t>(total);
    }
  }
}

template <typename scalar_t, typename out_t, int ROW_TILE>
__global__ __launch_bounds__(128, 2) void q8_0_t16_dual_split_rowtile_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b) {
  const int total_tiles_a = static_cast<int>(out_features_a / T16_COLS);
  const int out_tile_global = static_cast<int>(blockIdx.x);
  const bool use_b = out_tile_global >= total_tiles_a;
  const int local_tile = use_b ? (out_tile_global - total_tiles_a) : out_tile_global;
  const int row0 = static_cast<int>(blockIdx.y) * ROW_TILE;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / Q8_0_BLOCK;
  const uint8_t* __restrict__ tiles = use_b ? tiles_b : tiles_a;
  out_t* __restrict__ out = use_b ? out_b : out_a;
  const int64_t out_features = use_b ? out_features_b : out_features_a;
  float acc[ROW_TILE][T16_COLS];
#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      acc[r][col] = 0.0f;
    }
  }

  for (int64_t k = tid; k < in_features; k += blockDim.x) {
    const int64_t block_idx = k / Q8_0_BLOCK;
    const int lane = static_cast<int>(k & (Q8_0_BLOCK - 1));
    const uint8_t* __restrict__ tile =
        tiles + (static_cast<int64_t>(local_tile) * blocks_per_row + block_idx) * T16_BLOCK_BYTES;
    const int8_t* __restrict__ q = reinterpret_cast<const int8_t*>(tile + T16_D_BYTES + lane * T16_COLS);
    float d[T16_COLS];
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      d[col] = fp16_bytes_to_float(tile + col * 2);
    }
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      const int row = row0 + r;
      if (row < rows) {
        const scalar_t* __restrict__ x_row = x + static_cast<int64_t>(row) * in_features;
        const float xv = scalar_to_float<scalar_t>(x_row[k]);
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          acc[r][col] += xv * (d[col] * static_cast<float>(q[col]));
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r) {
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      GGUF_T16_WAVE_REDUCE(acc[r][col]);
    }
  }

  __shared__ float xchg[4 * ROW_TILE * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        xchg[(sub_id * ROW_TILE + r) * T16_COLS + col] = acc[r][col];
      }
    }
  }
  __syncthreads();

  if (tid == 0) {
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      const int row = row0 + r;
      if (row < rows) {
        out_t* __restrict__ out_row = out + static_cast<int64_t>(row) * out_features + local_tile * T16_COLS;
#pragma unroll
        for (int col = 0; col < T16_COLS; ++col) {
          float total = 0.0f;
          for (int g = 0; g < sub_groups; ++g) {
            total += xchg[(g * ROW_TILE + r) * T16_COLS + col];
          }
          out_row[col] = float_to_scalar<out_t>(total);
        }
      }
    }
  }
}

template <typename out_t>
__global__ __launch_bounds__(128, 4) void q8_0_t16_dual_split_q8_1_dp4a_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b) {
  const int total_tiles_a = static_cast<int>(out_features_a / T16_COLS);
  const int out_tile_global = static_cast<int>(blockIdx.x);
  const bool use_b = out_tile_global >= total_tiles_a;
  const int local_tile = use_b ? (out_tile_global - total_tiles_a) : out_tile_global;
  const int row = static_cast<int>(blockIdx.y);
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / Q8_0_BLOCK;
  const uint8_t* __restrict__ tiles = use_b ? tiles_b : tiles_a;
  out_t* __restrict__ out = use_b ? out_b : out_a;
  const int64_t out_features = use_b ? out_features_b : out_features_a;
  const gguf_q8_1_block* __restrict__ xq_row = xq + static_cast<int64_t>(row) * blocks_per_row;
  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc[col] = 0.0f;
  }

  for (int64_t block_idx = tid; block_idx < blocks_per_row; block_idx += blockDim.x) {
    const uint8_t* __restrict__ tile =
        tiles + (static_cast<int64_t>(local_tile) * blocks_per_row + block_idx) * T16_BLOCK_BYTES;
    const int8_t* __restrict__ q = reinterpret_cast<const int8_t*>(tile + T16_D_BYTES);
    const gguf_q8_1_block* __restrict__ xb = xq_row + block_idx;
    const float xd = static_cast<float>(xb->d);
    float d[T16_COLS];
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      d[col] = fp16_bytes_to_float(tile + col * 2);
    }
#pragma unroll
    for (int g = 0; g < Q8_0_BLOCK / 4; ++g) {
      const uint32_t x4 = *reinterpret_cast<const uint32_t*>(xb->qs + g * 4);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const uint32_t w4 = pack_t16_col4(q, g * 4, col);
        const int dot = q8_0_dp4a(w4, x4, 0);
        acc[col] += d[col] * xd * static_cast<float>(dot);
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
    const int sub_groups = blockDim.x >> 5;
    out_t* __restrict__ out_row = out + static_cast<int64_t>(row) * out_features + local_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) {
        total += xchg[g * T16_COLS + col];
      }
      out_row[col] = float_to_scalar<out_t>(total);
    }
  }
}

template <typename scalar_t, typename out_t>
__global__ __launch_bounds__(128, 4) void q8_0_t16_triple_split_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    const uint8_t* __restrict__ tiles_c,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    out_t* __restrict__ out_c,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t out_features_c) {
  const int total_tiles_a = static_cast<int>(out_features_a / T16_COLS);
  const int total_tiles_b = static_cast<int>(out_features_b / T16_COLS);
  const int out_tile_global = static_cast<int>(blockIdx.x);
  const bool use_b = out_tile_global >= total_tiles_a && out_tile_global < (total_tiles_a + total_tiles_b);
  const bool use_c = out_tile_global >= (total_tiles_a + total_tiles_b);
  const int local_tile = use_c ? (out_tile_global - total_tiles_a - total_tiles_b)
                               : (use_b ? (out_tile_global - total_tiles_a) : out_tile_global);
  const int row = static_cast<int>(blockIdx.y);
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / Q8_0_BLOCK;
  const uint8_t* __restrict__ tiles = use_c ? tiles_c : (use_b ? tiles_b : tiles_a);
  out_t* __restrict__ out = use_c ? out_c : (use_b ? out_b : out_a);
  const int64_t out_features = use_c ? out_features_c : (use_b ? out_features_b : out_features_a);
  const scalar_t* __restrict__ x_row = x + static_cast<int64_t>(row) * in_features;
  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) {
    acc[col] = 0.0f;
  }

  for (int64_t k = tid; k < in_features; k += blockDim.x) {
    const int64_t block_idx = k / Q8_0_BLOCK;
    const int lane = static_cast<int>(k & (Q8_0_BLOCK - 1));
    const uint8_t* __restrict__ tile =
        tiles + (static_cast<int64_t>(local_tile) * blocks_per_row + block_idx) * T16_BLOCK_BYTES;
    const int8_t* __restrict__ q = reinterpret_cast<const int8_t*>(tile + T16_D_BYTES + lane * T16_COLS);
    const float xv = scalar_to_float<scalar_t>(x_row[k]);
    float d[T16_COLS];
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      d[col] = fp16_bytes_to_float(tile + col * 2);
    }
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      acc[col] += xv * (d[col] * static_cast<float>(q[col]));
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
    const int sub_groups = blockDim.x >> 5;
    out_t* __restrict__ out_row = out + static_cast<int64_t>(row) * out_features + local_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) {
        total += xchg[g * T16_COLS + col];
      }
      out_row[col] = float_to_scalar<out_t>(total);
    }
  }
}

template <typename scalar_t, typename out_t>
int launch_single(const void* x,
                  const void* tiles,
                  void* out,
                  int64_t rows,
                  int64_t in_features,
                  int64_t out_features,
                  int64_t threads,
                  hipStream_t stream) {
  if (!x || !tiles || !out || rows <= 0 || in_features <= 0 || out_features <= 0 ||
      (in_features % Q8_0_BLOCK) != 0 || (out_features % T16_COLS) != 0 ||
      !valid_threads(threads)) {
    return hipErrorInvalidValue;
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q8_0_t16_gemv_kernel<scalar_t, out_t>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x), reinterpret_cast<const uint8_t*>(tiles),
                     reinterpret_cast<out_t*>(out), rows, in_features, out_features);
  return hipGetLastError();
}

template <typename scalar_t, typename out_t>
int launch_dual(const void* x,
                const void* tiles_a,
                const void* tiles_b,
                void* out,
                int64_t rows,
                int64_t in_features,
                int64_t out_features_a,
                int64_t out_features_b,
                int64_t threads,
                hipStream_t stream) {
  if (!x || !tiles_a || !tiles_b || !out || rows <= 0 || in_features <= 0 ||
      out_features_a <= 0 || out_features_b <= 0 || (in_features % Q8_0_BLOCK) != 0 ||
      (out_features_a % T16_COLS) != 0 || (out_features_b % T16_COLS) != 0 ||
      !valid_threads(threads)) {
    return hipErrorInvalidValue;
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>((out_features_a + out_features_b) / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q8_0_t16_dual_gemv_kernel<scalar_t, out_t>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x), reinterpret_cast<const uint8_t*>(tiles_a),
                     reinterpret_cast<const uint8_t*>(tiles_b), reinterpret_cast<out_t*>(out),
                     rows, in_features, out_features_a, out_features_b);
  return hipGetLastError();
}

template <typename scalar_t, typename out_t>
int launch_dual_split(const void* x,
                      const void* tiles_a,
                      const void* tiles_b,
                      void* out_a,
                      void* out_b,
                      int64_t rows,
                      int64_t in_features,
                      int64_t out_features_a,
                      int64_t out_features_b,
                      int64_t threads,
                      hipStream_t stream) {
  if (!x || !tiles_a || !tiles_b || !out_a || !out_b || rows <= 0 || in_features <= 0 ||
      out_features_a <= 0 || out_features_b <= 0 || (in_features % Q8_0_BLOCK) != 0 ||
      (out_features_a % T16_COLS) != 0 || (out_features_b % T16_COLS) != 0 ||
      !valid_threads(threads)) {
    return hipErrorInvalidValue;
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>((out_features_a + out_features_b) / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q8_0_t16_dual_split_gemv_kernel<scalar_t, out_t>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x), reinterpret_cast<const uint8_t*>(tiles_a),
                     reinterpret_cast<const uint8_t*>(tiles_b), reinterpret_cast<out_t*>(out_a),
                     reinterpret_cast<out_t*>(out_b), rows, in_features, out_features_a,
                     out_features_b);
  return hipGetLastError();
}

template <typename scalar_t, typename out_t, int ROW_TILE>
int launch_dual_split_rowtile(const void* x,
                              const void* tiles_a,
                              const void* tiles_b,
                              void* out_a,
                              void* out_b,
                              int64_t rows,
                              int64_t in_features,
                              int64_t out_features_a,
                              int64_t out_features_b,
                              int64_t threads,
                              hipStream_t stream) {
  if (!x || !tiles_a || !tiles_b || !out_a || !out_b || rows <= 0 || in_features <= 0 ||
      out_features_a <= 0 || out_features_b <= 0 || (in_features % Q8_0_BLOCK) != 0 ||
      (out_features_a % T16_COLS) != 0 || (out_features_b % T16_COLS) != 0 ||
      !valid_threads(threads)) {
    return hipErrorInvalidValue;
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>((out_features_a + out_features_b) / T16_COLS),
                  static_cast<unsigned int>((rows + ROW_TILE - 1) / ROW_TILE));
  hipLaunchKernelGGL((q8_0_t16_dual_split_rowtile_gemv_kernel<scalar_t, out_t, ROW_TILE>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x), reinterpret_cast<const uint8_t*>(tiles_a),
                     reinterpret_cast<const uint8_t*>(tiles_b), reinterpret_cast<out_t*>(out_a),
                     reinterpret_cast<out_t*>(out_b), rows, in_features, out_features_a,
                     out_features_b);
  return hipGetLastError();
}

template <typename out_t>
int launch_dual_split_q8_1_dp4a(const void* xq,
                                const void* tiles_a,
                                const void* tiles_b,
                                void* out_a,
                                void* out_b,
                                int64_t rows,
                                int64_t in_features,
                                int64_t out_features_a,
                                int64_t out_features_b,
                                int64_t threads,
                                hipStream_t stream) {
  if (!xq || !tiles_a || !tiles_b || !out_a || !out_b || rows <= 0 || in_features <= 0 ||
      out_features_a <= 0 || out_features_b <= 0 || (in_features % Q8_0_BLOCK) != 0 ||
      (out_features_a % T16_COLS) != 0 || (out_features_b % T16_COLS) != 0 ||
      !valid_threads(threads)) {
    return hipErrorInvalidValue;
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>((out_features_a + out_features_b) / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q8_0_t16_dual_split_q8_1_dp4a_kernel<out_t>), grid, block, 0, stream,
                     reinterpret_cast<const gguf_q8_1_block*>(xq),
                     reinterpret_cast<const uint8_t*>(tiles_a),
                     reinterpret_cast<const uint8_t*>(tiles_b),
                     reinterpret_cast<out_t*>(out_a),
                     reinterpret_cast<out_t*>(out_b), rows, in_features, out_features_a,
                     out_features_b);
  return hipGetLastError();
}

template <typename scalar_t, typename out_t>
int launch_triple_split(const void* x,
                        const void* tiles_a,
                        const void* tiles_b,
                        const void* tiles_c,
                        void* out_a,
                        void* out_b,
                        void* out_c,
                        int64_t rows,
                        int64_t in_features,
                        int64_t out_features_a,
                        int64_t out_features_b,
                        int64_t out_features_c,
                        int64_t threads,
                        hipStream_t stream) {
  if (!x || !tiles_a || !tiles_b || !tiles_c || !out_a || !out_b || !out_c || rows <= 0 || in_features <= 0 ||
      out_features_a <= 0 || out_features_b <= 0 || out_features_c <= 0 || (in_features % Q8_0_BLOCK) != 0 ||
      (out_features_a % T16_COLS) != 0 || (out_features_b % T16_COLS) != 0 ||
      (out_features_c % T16_COLS) != 0 || !valid_threads(threads)) {
    return hipErrorInvalidValue;
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>((out_features_a + out_features_b + out_features_c) / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q8_0_t16_triple_split_gemv_kernel<scalar_t, out_t>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x), reinterpret_cast<const uint8_t*>(tiles_a),
                     reinterpret_cast<const uint8_t*>(tiles_b), reinterpret_cast<const uint8_t*>(tiles_c),
                     reinterpret_cast<out_t*>(out_a), reinterpret_cast<out_t*>(out_b),
                     reinterpret_cast<out_t*>(out_c), rows, in_features, out_features_a,
                     out_features_b, out_features_c);
  return hipGetLastError();
}

}  // namespace

extern "C" int hipengine_gguf_q8_0_t16_gemv_decode_bf16_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows, int64_t in_features,
    int64_t out_features, int64_t threads, hipStream_t stream) {
  return launch_single<uint16_t, uint16_t>(x, tiles, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q8_0_t16_gemv_decode_fp16_fp16_out(
    const void* x, const void* tiles, void* out, int64_t rows, int64_t in_features,
    int64_t out_features, int64_t threads, hipStream_t stream) {
  return launch_single<half_t, half_t>(x, tiles, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q8_0_t16_gemv_decode_f32_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows, int64_t in_features,
    int64_t out_features, int64_t threads, hipStream_t stream) {
  return launch_single<float, uint16_t>(x, tiles, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out(
    const void* x, const void* tiles_a, const void* tiles_b, void* out, int64_t rows,
    int64_t in_features, int64_t out_features_a, int64_t out_features_b, int64_t threads,
    hipStream_t stream) {
  return launch_dual<uint16_t, uint16_t>(x, tiles_a, tiles_b, out, rows, in_features,
                                        out_features_a, out_features_b, threads, stream);
}

extern "C" int hipengine_gguf_q8_0_t16_dual_gate_up_gemv_decode_fp16_fp16_out(
    const void* x, const void* tiles_a, const void* tiles_b, void* out, int64_t rows,
    int64_t in_features, int64_t out_features_a, int64_t out_features_b, int64_t threads,
    hipStream_t stream) {
  return launch_dual<half_t, half_t>(x, tiles_a, tiles_b, out, rows, in_features,
                                    out_features_a, out_features_b, threads, stream);
}

extern "C" int hipengine_gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out(
    const void* x, const void* tiles_a, const void* tiles_b, void* out_a, void* out_b,
    int64_t rows, int64_t in_features, int64_t out_features_a, int64_t out_features_b,
    int64_t threads, hipStream_t stream) {
  return launch_dual_split<uint16_t, uint16_t>(x, tiles_a, tiles_b, out_a, out_b, rows,
                                              in_features, out_features_a, out_features_b,
                                              threads, stream);
}

extern "C" int hipengine_gguf_q8_0_t16_dual_gemv_decode_fp16_fp16_out(
    const void* x, const void* tiles_a, const void* tiles_b, void* out_a, void* out_b,
    int64_t rows, int64_t in_features, int64_t out_features_a, int64_t out_features_b,
    int64_t threads, hipStream_t stream) {
  return launch_dual_split<half_t, half_t>(x, tiles_a, tiles_b, out_a, out_b, rows,
                                          in_features, out_features_a, out_features_b,
                                          threads, stream);
}

extern "C" int hipengine_gguf_q8_0_t16_dual_gemv_decode_rowtile2_bf16_bf16_out(
    const void* x, const void* tiles_a, const void* tiles_b, void* out_a, void* out_b,
    int64_t rows, int64_t in_features, int64_t out_features_a, int64_t out_features_b,
    int64_t threads, hipStream_t stream) {
  return launch_dual_split_rowtile<uint16_t, uint16_t, 2>(x, tiles_a, tiles_b, out_a, out_b,
                                                         rows, in_features, out_features_a,
                                                         out_features_b, threads, stream);
}

extern "C" int hipengine_gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out(
    const void* x, const void* tiles_a, const void* tiles_b, void* out_a, void* out_b,
    int64_t rows, int64_t in_features, int64_t out_features_a, int64_t out_features_b,
    int64_t threads, hipStream_t stream) {
  return launch_dual_split_rowtile<uint16_t, uint16_t, 4>(x, tiles_a, tiles_b, out_a, out_b,
                                                         rows, in_features, out_features_a,
                                                         out_features_b, threads, stream);
}

extern "C" int hipengine_gguf_q8_0_t16_dual_gemv_decode_q8_1_dp4a_bf16_bf16_out(
    const void* xq, const void* tiles_a, const void* tiles_b, void* out_a, void* out_b,
    int64_t rows, int64_t in_features, int64_t out_features_a, int64_t out_features_b,
    int64_t threads, hipStream_t stream) {
  return launch_dual_split_q8_1_dp4a<uint16_t>(xq, tiles_a, tiles_b, out_a, out_b, rows,
                                              in_features, out_features_a, out_features_b,
                                              threads, stream);
}

extern "C" int hipengine_gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out(
    const void* x, const void* tiles_a, const void* tiles_b, const void* tiles_c,
    void* out_a, void* out_b, void* out_c, int64_t rows, int64_t in_features,
    int64_t out_features_a, int64_t out_features_b, int64_t out_features_c,
    int64_t threads, hipStream_t stream) {
  return launch_triple_split<uint16_t, uint16_t>(x, tiles_a, tiles_b, tiles_c,
                                                out_a, out_b, out_c, rows,
                                                in_features, out_features_a,
                                                out_features_b, out_features_c,
                                                threads, stream);
}

extern "C" int hipengine_gguf_q8_0_t16_triple_gemv_decode_fp16_fp16_out(
    const void* x, const void* tiles_a, const void* tiles_b, const void* tiles_c,
    void* out_a, void* out_b, void* out_c, int64_t rows, int64_t in_features,
    int64_t out_features_a, int64_t out_features_b, int64_t out_features_c,
    int64_t threads, hipStream_t stream) {
  return launch_triple_split<half_t, half_t>(x, tiles_a, tiles_b, tiles_c,
                                            out_a, out_b, out_c, rows,
                                            in_features, out_features_a,
                                            out_features_b, out_features_c,
                                            threads, stream);
}
