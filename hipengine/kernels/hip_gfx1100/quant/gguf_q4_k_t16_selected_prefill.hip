// Experimental compact selected-MoE WMMA prefill for GGUF Q4_K T16 tiles.
//
// P9.C14 prototype: consumes the P9.C13/P9.H2 replacement layout
//   tiles[E, out_tiles16, blocks_per_row, 2368]
// for Q4_K gate/up expert tensors while preserving the existing compact MoE
// scheduler/output ABI used by the raw GGUF Q4_K selected WMMA prefill path.
//
// One wave32 block computes a 16-row x 32-column bundle (two 16-column WMMA
// tiles) for one compact selected-MoE row tile.  Accumulation stays FP32, input
// WMMA operands are half, and output is BF16/FP16 to match the raw-Q4_K kernel.

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS
#define HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS 1
#endif

namespace {

using half_t = _Float16;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

constexpr int QK_K = 256;
constexpr int T16_COLS = 16;
constexpr int Q4_SUBBLOCK = 32;
constexpr int Q4_SUBBLOCKS = 8;
constexpr int K_TILE = 16;
constexpr int K_TILES_PER_SUBBLOCK = Q4_SUBBLOCK / K_TILE;  // == 2
constexpr int Q8_1_BLOCK = 32;
constexpr int Q8_1_MMQ_BLOCK = 4 * Q8_1_BLOCK;  // == 128

struct block_q8_1_mmq_ds4 {
  uint16_t ds4[8];
  int8_t qs[Q8_1_MMQ_BLOCK];
};
static_assert(sizeof(block_q8_1_mmq_ds4) == Q8_1_MMQ_BLOCK + 8 * sizeof(uint16_t),
              "unexpected DS4 Q8_1 MMQ block size");

constexpr int Q4_T16_D_OFFSET = 0;
constexpr int Q4_T16_DMIN_OFFSET = Q4_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_SCALE_OFFSET = Q4_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_MIN_OFFSET = Q4_T16_SCALE_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_Q_OFFSET = Q4_T16_MIN_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_BLOCK_BYTES = Q4_T16_Q_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);
constexpr int Q4_QMICRO_T16_D_OFFSET = 0;
constexpr int Q4_QMICRO_T16_DMIN_OFFSET = Q4_QMICRO_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q4_QMICRO_T16_META_OFFSET = Q4_QMICRO_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q4_QMICRO_T16_Q_OFFSET = Q4_QMICRO_T16_META_OFFSET + 2 * Q4_SUBBLOCKS * (T16_COLS / 4) * 3;
constexpr int Q4_QMICRO_T16_BLOCK_BYTES = Q4_QMICRO_T16_Q_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);

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

__device__ inline float fp16_bits_to_float(uint16_t bits) {
  union {
    uint16_t u16;
    half_t f16;
  } in;
  in.u16 = bits;
  return static_cast<float>(in.f16);
}

__device__ inline float fp16_bytes_to_float(const uint8_t* ptr) {
  const uint16_t bits = static_cast<uint16_t>(ptr[0]) |
                        (static_cast<uint16_t>(ptr[1]) << 8);
  return fp16_bits_to_float(bits);
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
__device__ inline half16_t load_compact_row_half16(
    const scalar_t* __restrict__ x,
    int64_t compact_row,
    int64_t in_features,
    int64_t k_base,
    bool valid_row) {
  half16_t out;
  if (valid_row) {
    const scalar_t* __restrict__ row = x + compact_row * in_features + k_base;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      out[i] = static_cast<half_t>(scalar_to_float<scalar_t>(row[i]));
    }
  } else {
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = static_cast<half_t>(0.0f);
  }
  return out;
}

template <>
__device__ inline half16_t load_compact_row_half16<half_t>(
    const half_t* __restrict__ x,
    int64_t compact_row,
    int64_t in_features,
    int64_t k_base,
    bool valid_row) {
  if (!valid_row) {
    half16_t out;
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = static_cast<half_t>(0.0f);
    return out;
  }
  return *reinterpret_cast<const half16_t*>(x + compact_row * in_features + k_base);
}

template <bool qmicro = false>
__device__ inline const uint8_t* q4_t16_tile_ptr(
    const uint8_t* __restrict__ tiles,
    int64_t expert_id,
    int64_t out_tile,
    int64_t block_idx,
    int64_t out_tiles,
    int64_t blocks_per_row) {
  const int64_t tile_index = (expert_id * out_tiles + out_tile) * blocks_per_row + block_idx;
  constexpr int tile_bytes = qmicro ? Q4_QMICRO_T16_BLOCK_BYTES : Q4_T16_BLOCK_BYTES;
  return tiles + tile_index * tile_bytes;
}

__device__ inline uint8_t qmicro_coefficient(
    const uint8_t* __restrict__ tile,
    int plane,
    int subblock,
    int col) {
  const int record = ((plane * Q4_SUBBLOCKS + subblock) * (T16_COLS / 4)) + (col >> 2);
  const uint8_t* __restrict__ ptr = tile + Q4_QMICRO_T16_META_OFFSET + record * 3;
  const uint32_t word = static_cast<uint32_t>(ptr[0]) |
                        (static_cast<uint32_t>(ptr[1]) << 8) |
                        (static_cast<uint32_t>(ptr[2]) << 16);
  return static_cast<uint8_t>((word >> (6 * (col & 3))) & 0x3Fu);
}

template <bool qmicro = false>
__device__ inline uint8_t q4_t16_quant(
    const uint8_t* __restrict__ tile,
    int subblock,
    int lane32,
    int col) {
  constexpr int q_offset = qmicro ? Q4_QMICRO_T16_Q_OFFSET : Q4_T16_Q_OFFSET;
  const uint8_t packed = tile[q_offset +
                              (subblock * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) +
                              (col >> 1)];
  return (col & 1) ? (packed >> 4) : (packed & 0x0Fu);
}

__device__ inline int pack_i8x4(const int8_t* ptr) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |= static_cast<uint32_t>(static_cast<uint8_t>(ptr[i])) << (8 * i);
  }
  return static_cast<int>(out);
}

__device__ inline int pack_q4_t16_u8x4(
    const uint8_t* __restrict__ tile,
    int subblock,
    int lane32_base,
    int col) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |= static_cast<uint32_t>(q4_t16_quant(tile, subblock, lane32_base + i, col)) << (8 * i);
  }
  return static_cast<int>(out);
}

template <bool qmicro = false>
__device__ inline void load_q4_t16_b_reg(
    const uint8_t* __restrict__ tile,
    int subblock,
    int kt,
    int col,
    bool valid_col,
    half16_t& b_reg) {
  if (valid_col) {
    constexpr int d_offset = qmicro ? Q4_QMICRO_T16_D_OFFSET : Q4_T16_D_OFFSET;
    constexpr int dmin_offset = qmicro ? Q4_QMICRO_T16_DMIN_OFFSET : Q4_T16_DMIN_OFFSET;
    const float d_f = fp16_bytes_to_float(tile + d_offset + col * 2);
    const float dmin_f = fp16_bytes_to_float(tile + dmin_offset + col * 2);
    const uint8_t scale = qmicro
        ? qmicro_coefficient(tile, 0, subblock, col)
        : tile[Q4_T16_SCALE_OFFSET + subblock * T16_COLS + col];
    const uint8_t minv = qmicro
        ? qmicro_coefficient(tile, 1, subblock, col)
        : tile[Q4_T16_MIN_OFFSET + subblock * T16_COLS + col];
    const float scale_f = d_f * static_cast<float>(scale);
    const float min_f = dmin_f * static_cast<float>(minv);
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
      const int lane32 = kt * K_TILE + kk;
      const uint8_t q = q4_t16_quant<qmicro>(tile, subblock, lane32, col);
      b_reg[kk] = static_cast<half_t>(scale_f * static_cast<float>(q) - min_f);
    }
  } else {
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) b_reg[kk] = static_cast<half_t>(0.0f);
  }
}

bool valid_selected_dual_t16_common(
    const void* x,
    const void* expert_start_compact,
    const void* expert_start_wmma,
    const void* tile_expert,
    const void* tiles_a,
    const void* tiles_b,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  return x != nullptr && expert_start_compact != nullptr && expert_start_wmma != nullptr &&
      tile_expert != nullptr && tiles_a != nullptr && tiles_b != nullptr && out != nullptr &&
      compact_rows > 0 && in_features > 0 && out_features_a > 0 && out_features_b > 0 &&
      (out_features_a % T16_COLS) == 0 && (out_features_b % T16_COLS) == 0 &&
      num_experts > 0 && wmma_total_rows > 0 && (wmma_total_rows % 16) == 0 &&
      (in_features % QK_K) == 0;
}

template <typename scalar_t, bool qmicro = false>
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane16 = tid & 15;
  const int half_id = tid >> 4;

  const int64_t col_start = static_cast<int64_t>(col_tile) * 32;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (col_start >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const int64_t blocks_per_row = in_features / QK_K;

#pragma unroll
  for (int tm = 0; tm < 2; ++tm) {
    const int64_t global_col_base = col_start + tm * 16;
    if (global_col_base >= out_features_total) continue;

    const bool use_b = global_col_base >= out_features_a;
    const int64_t local_col_base = use_b ? (global_col_base - out_features_a) : global_col_base;
    const int64_t local_out_features = use_b ? out_features_b : out_features_a;
    const int64_t local_col = local_col_base + lane16;
    const bool valid_col = (global_col_base + lane16) < out_features_total &&
        local_col < local_out_features;
    const int64_t out_tiles = local_out_features / T16_COLS;
    const int64_t out_tile = local_col_base / T16_COLS;
    const uint8_t* __restrict__ tiles = use_b ? tiles_b : tiles_a;

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
      const int64_t g_k_base = blk * QK_K;
#pragma unroll
      for (int sb = 0; sb < Q4_SUBBLOCKS; ++sb) {
#pragma unroll
        for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
          const int k_off = sb * Q4_SUBBLOCK + kt * K_TILE;
          const int64_t k_base = g_k_base + k_off;
          const int64_t local_row = tile_row_in_expert + lane16;
          const bool valid_row = local_row < expert_count;
          const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;
          const half16_t a_reg = load_compact_row_half16<scalar_t>(
              x, compact_row, in_features, k_base, valid_row);
          const uint8_t* __restrict__ tile = valid_col
              ? q4_t16_tile_ptr<qmicro>(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row)
              : tiles;
          half16_t b_reg;
          load_q4_t16_b_reg<qmicro>(tile, sb, kt, lane16, valid_col, b_reg);
          acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
      }
    }

    const int64_t global_out_col = global_col_base + lane16;
    if (global_out_col >= out_features_total) continue;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
      if (local_out_row < expert_count) {
        const int64_t compact_out_row = compact_base + local_out_row;
        if (compact_out_row < compact_rows) {
          out[compact_out_row * out_features_total + global_out_col] =
              float_to_scalar<scalar_t>(acc[j]);
        }
      }
    }
  }
}

// GPF-3A diagnostic: keep the two 16-column WMMA accumulators live together
// so both halves of the existing 32-column output tile consume one activation
// load.  The baseline kernel places `tm` outside the K loop and therefore
// reloads the identical compact activation fragment twice.  Each accumulator
// below sees the same K iteration and WMMA order as baseline; only the shared
// activation load and independent instruction scheduling change.
template <typename scalar_t>
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_shared_x_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane16 = tid & 15;
  const int half_id = tid >> 4;

  const int64_t col_start = static_cast<int64_t>(col_tile) * 32;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (col_start >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;
  const int64_t local_row = tile_row_in_expert + lane16;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;
  const int64_t blocks_per_row = in_features / QK_K;

  float8_t acc[2] = {
      {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
      {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
  };

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const int64_t g_k_base = blk * QK_K;
#pragma unroll
    for (int sb = 0; sb < Q4_SUBBLOCKS; ++sb) {
#pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_SUBBLOCK + kt * K_TILE;
        const int64_t k_base = g_k_base + k_off;
        const half16_t a_reg = load_compact_row_half16<scalar_t>(
            x, compact_row, in_features, k_base, valid_row);

#pragma unroll
        for (int tm = 0; tm < 2; ++tm) {
          const int64_t global_col_base = col_start + tm * 16;
          if (global_col_base >= out_features_total) continue;

          const bool use_b = global_col_base >= out_features_a;
          const int64_t local_col_base =
              use_b ? (global_col_base - out_features_a) : global_col_base;
          const int64_t local_out_features = use_b ? out_features_b : out_features_a;
          const int64_t local_col = local_col_base + lane16;
          const bool valid_col =
              (global_col_base + lane16) < out_features_total &&
              local_col < local_out_features;
          const int64_t out_tiles = local_out_features / T16_COLS;
          const int64_t out_tile = local_col_base / T16_COLS;
          const uint8_t* __restrict__ tiles = use_b ? tiles_b : tiles_a;
          const uint8_t* __restrict__ tile = valid_col
              ? q4_t16_tile_ptr(
                    tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row)
              : tiles;
          half16_t b_reg;
          load_q4_t16_b_reg(tile, sb, kt, lane16, valid_col, b_reg);
          acc[tm] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a_reg, b_reg, acc[tm]);
        }
      }
    }
  }

#pragma unroll
  for (int tm = 0; tm < 2; ++tm) {
    const int64_t global_col_base = col_start + tm * 16;
    const int64_t global_out_col = global_col_base + lane16;
    if (global_out_col >= out_features_total) continue;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
      if (local_out_row < expert_count) {
        const int64_t compact_out_row = compact_base + local_out_row;
        if (compact_out_row < compact_rows) {
          out[compact_out_row * out_features_total + global_out_col] =
              float_to_scalar<scalar_t>(acc[tm][j]);
        }
      }
    }
  }
}

// R12 ladder step 1: multi-wave block + LDS-staged activation reuse + 32-row
// pairing for the selected-expert gate+up leg.
//
// Geometry.  The shipping ``compact32`` kernel launches one wave32 per
// (16-row, 32-column) output tile, reads the compact activation fragment from
// global memory once per column half, and dequantizes every Q4_K column tile
// once per 16-row tile.  Measured on gfx1151 at the real shape this leaves the
// gate+up leg at 9.03 TFLOP/s against 22.22 TFLOP/s for the dense ``nwave``
// kernel, which differs in exactly two ways: a NUM_WAVES-wave block that shares
// one LDS-staged activation tile, and TM=2 row tiles per wave so each
// dequantized weight fragment feeds two WMMA accumulators.
//
// This kernel mirrors that design for the compact selected-MoE layout.  A wave
// owns TM=2 sixteen-column slabs and TN=2 sixteen-row tiles (32 rows).  The
// block's NUM_WAVES waves all consume the same 32-row activation tile staged in
// LDS, and each dequantized weight fragment is reused across both row tiles.
//
// The two 16-row tiles of a wave need not belong to the same expert: the
// compact layout pads every expert's row count to a multiple of 16, so adjacent
// tiles straddle an expert boundary roughly half the time.  The weight fragment
// is shared only when both tiles resolve to the same expert id; otherwise each
// tile is dequantized against its own expert's tile.  The branch is uniform
// across the wave.
//
// Arithmetic contract: bit-identical to
// ``gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_kernel``.  Every output
// accumulator keeps the baseline K traversal order (blk, subblock, k-tile) and
// the baseline operand order (activation first, weight second).  The activation
// staged through LDS is converted with the same bf16/fp16 -> half path, so the
// WMMA operand bits are unchanged.
template <typename scalar_t, bool qmicro = false, int NUM_WAVES = 4>
__launch_bounds__(NUM_WAVES * 32, 8 / NUM_WAVES)
__global__ void gguf_q4_k_t16_selected_dual_wmma_prefill_pair32_nwave_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  constexpr int TN = 2;  // 16-row tiles per wave (32 rows)
  constexpr int TM = 1;  // 16-column slabs per wave
  constexpr int BLOCK_COLS = NUM_WAVES * (TM * 16);

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int wave_lane = tid & 31;
  const int lane16 = wave_lane & 15;
  const int half_id = wave_lane >> 4;

  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t block_col_start = static_cast<int64_t>(blockIdx.x) * BLOCK_COLS;
  const int64_t row_tile0 = static_cast<int64_t>(blockIdx.y) * TN;
  const int64_t row_tiles_total = wmma_total_rows / 16;
  // Block-uniform guards: every thread in the block must reach __syncthreads().
  if (block_col_start >= out_features_total) return;
  if (row_tile0 >= row_tiles_total) return;

  const int64_t col_start = block_col_start + static_cast<int64_t>(wave) * (TM * 16);

  const int64_t row_tile_of[TN] = {row_tile0, row_tile0 + 1};
  // int32 bookkeeping keeps the live per-tile state small enough for a second
  // resident block per CU; the layout sizes that reach this kernel (compact
  // rows, tile counts) all fit comfortably in 31 bits.
  int32_t expert_id[TN];
  int32_t compact_base[TN];
  int32_t expert_count[TN];
  int32_t tile_row_in_expert[TN];
#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    const int64_t e = (row_tile_of[tn] < row_tiles_total) ? tile_expert[row_tile_of[tn]] : -1;
    expert_id[tn] = static_cast<int32_t>(e);
    if (e < 0 || e >= num_experts) {
      compact_base[tn] = 0;
      expert_count[tn] = 0;
      tile_row_in_expert[tn] = 0;
    } else {
      compact_base[tn] = static_cast<int32_t>(expert_start_compact[e]);
      expert_count[tn] =
          static_cast<int32_t>(expert_start_compact[e + 1] - expert_start_compact[e]);
      tile_row_in_expert[tn] =
          static_cast<int32_t>(row_tile_of[tn] * 16 - expert_start_wmma[e]);
    }
  }
  const bool expert0_valid = expert_id[0] >= 0 && expert_id[0] < num_experts;
  const bool expert1_valid = expert_id[1] >= 0 && expert_id[1] < num_experts;
  const bool same_expert = expert0_valid && expert1_valid && (expert_id[0] == expert_id[1]);

  // Per-column-slab bookkeeping, identical to the compact32 baseline.
  int32_t out_tile[TM];
  int32_t out_tiles[TM];
  bool valid_col[TM];
  const uint8_t* tile_base0[TM];
  const uint8_t* tile_base1[TM];
  constexpr int32_t TILE_BYTES = qmicro ? Q4_QMICRO_T16_BLOCK_BYTES : Q4_T16_BLOCK_BYTES;
  const int32_t blocks_per_row32 = static_cast<int32_t>(in_features / QK_K);
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    const int64_t global_col_base = col_start + tm * 16;
    const bool use_b = global_col_base >= out_features_a;
    const int64_t local_col_base = use_b ? (global_col_base - out_features_a) : global_col_base;
    const int64_t local_out_features = use_b ? out_features_b : out_features_a;
    const int64_t global_col = global_col_base + lane16;
    const int64_t local_col = local_col_base + lane16;
    out_tiles[tm] = static_cast<int32_t>(local_out_features / T16_COLS);
    out_tile[tm] = static_cast<int32_t>(local_col_base / T16_COLS);
    valid_col[tm] = (global_col < out_features_total) && (local_col < local_out_features);
    const uint8_t* __restrict__ sel = use_b ? tiles_b : tiles_a;
    // Hoist the (expert, out_tile, block 0) base pointer: the K loop then only
    // adds blk * TILE_BYTES.
    tile_base0[tm] = (valid_col[tm] && expert0_valid)
        ? sel + (static_cast<int64_t>(expert_id[0]) * out_tiles[tm] + out_tile[tm]) *
                    blocks_per_row32 * TILE_BYTES
        : sel;
    tile_base1[tm] = (valid_col[tm] && expert1_valid)
        ? sel + (static_cast<int64_t>(expert_id[1]) * out_tiles[tm] + out_tile[tm]) *
                    blocks_per_row32 * TILE_BYTES
        : sel;
  }

  const int32_t blocks_per_row = blocks_per_row32;

  __shared__ half16_t activation_tile[TN * 16];

  float8_t acc[TM][TN];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      acc[tm][tn] = float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
  }

  for (int32_t blk = 0; blk < blocks_per_row; ++blk) {
    const int32_t g_k_base = blk * QK_K;
#pragma unroll
    for (int sb = 0; sb < Q4_SUBBLOCKS; ++sb) {
#pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_SUBBLOCK + kt * K_TILE;
        const int32_t k_base = g_k_base + k_off;

        // One wave stages the whole 32-row activation tile for the block.
        half16_t a_reg[TN];
        {
          if (wave == 0) {
            const int tn_s = wave_lane >> 4;
            const int lr = wave_lane & 15;
            const bool ok = (tile_row_in_expert[tn_s] + lr) < expert_count[tn_s];
            const int32_t crow =
                ok ? (compact_base[tn_s] + tile_row_in_expert[tn_s] + lr) : 0;
            activation_tile[wave_lane] =
                load_compact_row_half16<scalar_t>(x, crow, in_features, k_base, ok);
          }
          __syncthreads();
#pragma unroll
          for (int tn = 0; tn < TN; ++tn) {
            a_reg[tn] = activation_tile[tn * 16 + lane16];
          }
          // Every wave has copied the shared tile before wave 0 may overwrite it.
          __syncthreads();
        }

#pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
          half16_t b_reg;
          load_q4_t16_b_reg<qmicro>(tile_base0[tm] + blk * TILE_BYTES, sb, kt, lane16, valid_col[tm] && expert0_valid, b_reg);
          acc[tm][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg[0], b_reg, acc[tm][0]);
          if (same_expert) {
            acc[tm][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg[1], b_reg, acc[tm][1]);
          } else {
            half16_t b_reg1;
            load_q4_t16_b_reg<qmicro>(tile_base1[tm] + blk * TILE_BYTES, sb, kt, lane16, valid_col[tm] && expert1_valid, b_reg1);
            acc[tm][1] =
                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg[1], b_reg1, acc[tm][1]);
          }
        }
      }
    }
  }

#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      const int64_t global_out_col = col_start + tm * 16 + lane16;
      if (global_out_col >= out_features_total) continue;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t local_out_row = tile_row_in_expert[tn] + half_id + j * 2;
        if (local_out_row < expert_count[tn]) {
          const int64_t compact_out_row = compact_base[tn] + local_out_row;
          if (compact_out_row < compact_rows) {
            out[compact_out_row * out_features_total + global_out_col] =
                float_to_scalar<scalar_t>(acc[tm][tn][j]);
          }
        }
      }
    }
  }
}

__launch_bounds__(64, 1)
__global__ void gguf_q4_k_t16_selected_dual_q8_1_ds4_wmma32_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ tiles_a,
    const uint8_t* __restrict__ tiles_b,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int wave_tid = tid & 31;
  const int lane16 = wave_tid & 15;
  const int half_id = wave_tid >> 4;

  const int64_t col_base = static_cast<int64_t>(col_tile) * 32 + wave_id * 16;
  const int64_t global_col = col_base + lane16;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (global_col >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const bool valid_col = local_col < local_out_features;
  const int64_t out_tiles = local_out_features / T16_COLS;
  const int64_t out_tile = local_col / T16_COLS;
  const int col_in_tile = static_cast<int>(local_col & (T16_COLS - 1));
  const uint8_t* __restrict__ tiles = use_b ? tiles_b : tiles_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t local_a_row = tile_row_in_expert + lane16;
  const bool valid_a_row = local_a_row < expert_count;
  const int64_t compact_a_row = valid_a_row ? (compact_base + local_a_row) : 0;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
#pragma unroll
    for (int sb = 0; sb < Q4_SUBBLOCKS; ++sb) {
      const uint8_t* __restrict__ tile = valid_col
          ? q4_t16_tile_ptr(tiles, expert_id, out_tile, blk, out_tiles, blocks_per_row)
          : tiles;
      float scale_f = 0.0f;
      float min_f = 0.0f;
      if (valid_col) {
        const float d_f = fp16_bytes_to_float(tile + Q4_T16_D_OFFSET + col_in_tile * 2);
        const float dmin_f = fp16_bytes_to_float(tile + Q4_T16_DMIN_OFFSET + col_in_tile * 2);
        scale_f = d_f * static_cast<float>(tile[Q4_T16_SCALE_OFFSET + sb * T16_COLS + col_in_tile]);
        min_f = dmin_f * static_cast<float>(tile[Q4_T16_MIN_OFFSET + sb * T16_COLS + col_in_tile]);
      }

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a_row && compact_a_row < compact_rows) {
          const int64_t q8_block = compact_a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
          const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK + kt * K_TILE;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        int32x4_t b_vec = {0, 0, 0, 0};
        if (valid_col) {
          const int lane32_base = kt * K_TILE;
          b_vec = int32x4_t{
              pack_q4_t16_u8x4(tile, sb, lane32_base + 0, col_in_tile),
              pack_q4_t16_u8x4(tile, sb, lane32_base + 4, col_in_tile),
              pack_q4_t16_u8x4(tile, sb, lane32_base + 8, col_in_tile),
              pack_q4_t16_u8x4(tile, sb, lane32_base + 12, col_in_tile),
          };
        }
        dot_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, a_vec, true, b_vec, dot_acc, false);
      }

#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
        if (valid_col && local_out_row < expert_count) {
          const int64_t compact_out_row = compact_base + local_out_row;
          if (compact_out_row < compact_rows) {
            const int64_t q8_block = compact_out_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
            const int ds4_idx = sb & 3;
            const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
            const float xd = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 0]);
            const float xsum = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 1]);
            acc_f[j] += scale_f * xd * static_cast<float>(dot_acc[j]) - min_f * xsum;
          }
        }
      }
    }
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
    if (valid_col && local_out_row < expert_count) {
      const int64_t compact_out_row = compact_base + local_out_row;
      if (compact_out_row < compact_rows) {
        out[compact_out_row * out_features_total + global_col] = float_to_bf16_bits(acc_f[j]);
      }
    }
  }
}

template <typename scalar_t, bool qmicro = false>
int launch_selected_dual_t16_compact32(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles_a,
    const uint8_t* tiles_b,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_selected_dual_t16_common(
          x,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          tiles_a,
          tiles_b,
          out,
          compact_rows,
          in_features,
          out_features_a,
          out_features_b,
          num_experts,
          wmma_total_rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 31) / 32),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      (gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_kernel<scalar_t, qmicro>),
      grid,
      block,
      0,
      stream,
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      tiles_a,
      tiles_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t>
int launch_selected_dual_t16_compact32_shared_x(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles_a,
    const uint8_t* tiles_b,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_selected_dual_t16_common(
          x,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          tiles_a,
          tiles_b,
          out,
          compact_rows,
          in_features,
          out_features_a,
          out_features_b,
          num_experts,
          wmma_total_rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 31) / 32),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      (gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_shared_x_kernel<scalar_t>),
      grid,
      block,
      0,
      stream,
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      tiles_a,
      tiles_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, bool qmicro = false, int NUM_WAVES = 4>
int launch_selected_dual_t16_pair32_nwave(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles_a,
    const uint8_t* tiles_b,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_selected_dual_t16_common(
          x,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          tiles_a,
          tiles_b,
          out,
          compact_rows,
          in_features,
          out_features_a,
          out_features_b,
          num_experts,
          wmma_total_rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  constexpr int BLOCK_COLS = NUM_WAVES * 16;
  const dim3 block(NUM_WAVES * 32);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + BLOCK_COLS - 1) / BLOCK_COLS),
      static_cast<unsigned int>((row_tiles + 1) / 2));
  hipLaunchKernelGGL(
      (gguf_q4_k_t16_selected_dual_wmma_prefill_pair32_nwave_kernel<
          scalar_t, qmicro, NUM_WAVES>),
      grid,
      block,
      0,
      stream,
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      tiles_a,
      tiles_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_t16_q8_1_ds4_wmma32_compact32(
    const block_q8_1_mmq_ds4* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles_a,
    const uint8_t* tiles_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_selected_dual_t16_common(
          x_q8,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          tiles_a,
          tiles_b,
          out,
          compact_rows,
          in_features,
          out_features_a,
          out_features_b,
          num_experts,
          wmma_total_rows) || (in_features % Q8_1_MMQ_BLOCK) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(64);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 31) / 32),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_t16_selected_dual_q8_1_ds4_wmma32_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      tiles_a,
      tiles_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q4_k_qmicro_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles_a,
    const uint8_t* tiles_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_t16_compact32<uint16_t, true>(
      x, expert_start_compact, expert_start_wmma, tile_expert, tiles_a,
      tiles_b, out, compact_rows, in_features, out_features_a,
      out_features_b, num_experts, wmma_total_rows, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles_a,
    const uint8_t* tiles_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_t16_compact32<uint16_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      tiles_a,
      tiles_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_shared_x_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles_a,
    const uint8_t* tiles_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_t16_compact32_shared_x<uint16_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      tiles_a,
      tiles_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

// R12 ladder step 1.  NUM_WAVES is a schedule knob only; the BF16 owner below
// is the measured one (NUM_WAVES = 2, 88 VGPR, 1 KiB LDS, 64-thread block) and
// the FP16 sibling carries the same value so the two exports differ only in the
// activation staging type.
extern "C" int hipengine_gguf_q4_k_t16_selected_dual_wmma_prefill_pair32_nwave_bf16_bf16_out(
    const uint16_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles_a,
    const uint8_t* tiles_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_t16_pair32_nwave<uint16_t, false, 2>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      tiles_a,
      tiles_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_wmma_prefill_pair32_nwave_fp16_fp16_out(
    const half_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles_a,
    const uint8_t* tiles_b,
    half_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_t16_pair32_nwave<half_t, false, 2>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      tiles_a,
      tiles_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles_a,
    const uint8_t* tiles_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_t16_q8_1_ds4_wmma32_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      tiles_a,
      tiles_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_qmicro_t16_selected_dual_wmma_prefill_compact32_fp16_fp16_out(
    const half_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles_a,
    const uint8_t* tiles_b,
    half_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_t16_compact32<half_t, true>(
      x, expert_start_compact, expert_start_wmma, tile_expert, tiles_a,
      tiles_b, out, compact_rows, in_features, out_features_a,
      out_features_b, num_experts, wmma_total_rows, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_fp16_fp16_out(
    const half_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles_a,
    const uint8_t* tiles_b,
    half_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_t16_compact32<half_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      tiles_a,
      tiles_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_shared_x_fp16_fp16_out(
    const half_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles_a,
    const uint8_t* tiles_b,
    half_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_t16_compact32_shared_x<half_t>(
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      tiles_a,
      tiles_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}
