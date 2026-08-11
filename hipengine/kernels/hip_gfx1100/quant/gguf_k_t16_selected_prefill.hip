// Compact selected-MoE WMMA prefill for GGUF Q4_K/Q5_K/Q6_K T16 tile layouts.
//
// P10.B2 / P10.B3: ports the structure of gguf_k_selected_prefill.hip
// (raw Q5_K/Q6_K selected single-output WMMA prefill) to consume the
// resident T16 tile layout defined in hipengine/quant/gguf_t16.py:
//
//   Q4T16 tiles[E, out_tiles16, blocks_per_row, 2368]
//   Q5T16 tiles[E, out_tiles16, blocks_per_row, 2880]
//   Q6T16 tiles[E, out_tiles16, blocks_per_row, 3360]
//
// One wave32 block computes a 16x16 output tile for one compact selected-MoE
// row tile.  Accumulation stays FP32, input WMMA operands are half, output is
// BF16/FP16 to match the raw-Q5/Q6 kernel.  Tile dispatch shape matches the
// existing raw 16x16 legacy variant (no TM/TN expansion), keeping the
// compact MoE scheduler ABI shared with Q4T16 selected-dual.

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS
#define HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS 2
#endif

namespace {

using half_t = _Float16;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

constexpr int QK_K = 256;
constexpr int T16_COLS = 16;
constexpr int Q4_SUBBLOCK = 32;
constexpr int Q4_SUBBLOCKS = 8;
constexpr int Q6_GROUPS16 = 16;
constexpr int K_TILE = 16;
constexpr int K_TILES_PER_SUBBLOCK = Q4_SUBBLOCK / K_TILE;  // == 2
constexpr int K_TILES_PER_BLOCK = QK_K / K_TILE;            // == 16

// ---- Q4T16 layout (matches hipengine/quant/gguf_q4_k.py) ----------------
constexpr int Q4_T16_D_OFFSET = 0;
constexpr int Q4_T16_DMIN_OFFSET = Q4_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_SCALE_OFFSET = Q4_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q4_T16_MIN_OFFSET = Q4_T16_SCALE_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_Q_OFFSET = Q4_T16_MIN_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q4_T16_BLOCK_BYTES = Q4_T16_Q_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);

// ---- Q5T16 layout (matches hipengine/quant/gguf_t16.py) ----------------
constexpr int Q5_T16_D_OFFSET = 0;
constexpr int Q5_T16_DMIN_OFFSET = Q5_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q5_T16_SCALE_OFFSET = Q5_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q5_T16_MIN_OFFSET = Q5_T16_SCALE_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q5_T16_QL_OFFSET = Q5_T16_MIN_OFFSET + Q4_SUBBLOCKS * T16_COLS;
constexpr int Q5_T16_QH_OFFSET = Q5_T16_QL_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);
constexpr int Q5_T16_BLOCK_BYTES = Q5_T16_QH_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 8);
constexpr int Q5_QMICRO_T16_D_OFFSET = 0;
constexpr int Q5_QMICRO_T16_DMIN_OFFSET = Q5_QMICRO_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q5_QMICRO_T16_META_OFFSET = Q5_QMICRO_T16_DMIN_OFFSET + T16_COLS * 2;
constexpr int Q5_QMICRO_T16_QL_OFFSET = Q5_QMICRO_T16_META_OFFSET + 2 * Q4_SUBBLOCKS * (T16_COLS / 4) * 3;
constexpr int Q5_QMICRO_T16_QH_OFFSET = Q5_QMICRO_T16_QL_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 2);
constexpr int Q5_QMICRO_T16_BLOCK_BYTES = Q5_QMICRO_T16_QH_OFFSET + Q4_SUBBLOCKS * Q4_SUBBLOCK * (T16_COLS / 8);

// ---- Q6T16 layout (matches hipengine/quant/gguf_t16.py) ----------------
constexpr int Q6_T16_D_OFFSET = 0;
constexpr int Q6_T16_SCALE_OFFSET = Q6_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q6_T16_QL_OFFSET = Q6_T16_SCALE_OFFSET + Q6_GROUPS16 * T16_COLS;
constexpr int Q6_T16_QH_OFFSET = Q6_T16_QL_OFFSET + QK_K * (T16_COLS / 2);
constexpr int Q6_T16_BLOCK_BYTES = Q6_T16_QH_OFFSET + QK_K * (T16_COLS / 4);

// ---- scalar conversion helpers (identical to gguf_q4_k_t16_selected_prefill.hip) ----

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

__device__ inline float8_t zero_float8() {
  return float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

__device__ inline void compensated_wmma_accumulate(
    const half16_t a,
    const half16_t b,
    float8_t* sum,
    float8_t* compensation) {
  const float8_t partial = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
      a, b, zero_float8());
  const float8_t corrected = partial - *compensation;
  const float8_t next_sum = *sum + corrected;
  *compensation = (next_sum - *sum) - corrected;
  *sum = next_sum;
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

template <int qtype, bool qmicro = false>
__host__ __device__ constexpr int qtype_block_bytes() {
  if constexpr (qtype == 4) {
    return Q4_T16_BLOCK_BYTES;
  } else if constexpr (qtype == 5) {
    return qmicro ? Q5_QMICRO_T16_BLOCK_BYTES : Q5_T16_BLOCK_BYTES;
  } else {
    return Q6_T16_BLOCK_BYTES;
  }
}

__device__ inline const uint8_t* k_t16_tile_ptr(
    const uint8_t* __restrict__ tiles,
    int64_t expert_id,
    int64_t out_tile,
    int64_t block_idx,
    int64_t out_tiles,
    int64_t blocks_per_row,
    int tile_bytes) {
  const int64_t tile_index = (expert_id * out_tiles + out_tile) * blocks_per_row + block_idx;
  return tiles + tile_index * static_cast<int64_t>(tile_bytes);
}

// Q4T16 dequant: same 4-bit value and per-subblock scale/min math as GGUF
// block_q4_K, with columns transposed into the T16 replacement layout.
__device__ inline uint8_t q4_t16_quant(
    const uint8_t* __restrict__ tile,
    int subblock,
    int lane32,
    int col) {
  const uint8_t packed = tile[Q4_T16_Q_OFFSET +
                              (subblock * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) +
                              (col >> 1)];
  return (col & 1) ? (packed >> 4) : (packed & 0x0Fu);
}

// Q5T16 dequant: ``col`` is the column inside the 16-col tile slab; the kernel
// pre-loads d/dmin/scale/min once per (subblock, col) so the hot WMMA inner
// loop only does the K-tile-local low / high nibble assembly.
__device__ inline uint8_t qmicro_coefficient(
    const uint8_t* __restrict__ tile,
    int plane,
    int subblock,
    int col) {
  const int record = ((plane * Q4_SUBBLOCKS + subblock) * (T16_COLS / 4)) + (col >> 2);
  const uint8_t* __restrict__ ptr = tile + Q5_QMICRO_T16_META_OFFSET + record * 3;
  const uint32_t word = static_cast<uint32_t>(ptr[0]) |
                        (static_cast<uint32_t>(ptr[1]) << 8) |
                        (static_cast<uint32_t>(ptr[2]) << 16);
  return static_cast<uint8_t>((word >> (6 * (col & 3))) & 0x3Fu);
}

template <bool qmicro = false>
__device__ inline uint8_t q5_t16_quant(
    const uint8_t* __restrict__ tile,
    int subblock,
    int lane32,
    int col) {
  constexpr int ql_offset = qmicro ? Q5_QMICRO_T16_QL_OFFSET : Q5_T16_QL_OFFSET;
  constexpr int qh_offset = qmicro ? Q5_QMICRO_T16_QH_OFFSET : Q5_T16_QH_OFFSET;
  const uint8_t low_packed = tile[ql_offset +
                                  (subblock * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) +
                                  (col >> 1)];
  const uint8_t low = (col & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed = tile[qh_offset +
                                   (subblock * Q4_SUBBLOCK + lane32) * (T16_COLS / 8) +
                                   (col >> 3)];
  const uint8_t high = (high_packed >> (col & 7)) & 0x01u;
  return low | (high << 4);
}

// Q6T16 dequant: signed 6-bit values with per-group16 int8 scales (no min).
__device__ inline int q6_t16_quant(
    const uint8_t* __restrict__ tile,
    int k_in_block,
    int col) {
  const uint8_t low_packed = tile[Q6_T16_QL_OFFSET + k_in_block * (T16_COLS / 2) + (col >> 1)];
  const uint8_t low = (col & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed = tile[Q6_T16_QH_OFFSET + k_in_block * (T16_COLS / 4) + (col >> 2)];
  const uint8_t high = (high_packed >> (2 * (col & 3))) & 0x03u;
  return static_cast<int>(low | (high << 4)) - 32;
}

bool valid_selected_k_t16_common(
    const void* x,
    const void* expert_start_compact,
    const void* expert_start_wmma,
    const void* tile_expert,
    const void* tiles,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  return x != nullptr && expert_start_compact != nullptr && expert_start_wmma != nullptr &&
      tile_expert != nullptr && tiles != nullptr && out != nullptr &&
      compact_rows > 0 && in_features > 0 && out_features > 0 &&
      (out_features % T16_COLS) == 0 &&
      num_experts > 0 && wmma_total_rows > 0 && (wmma_total_rows % 16) == 0 &&
      (in_features % QK_K) == 0;
}

// One wave32 block computes a 16x16 tile of one compact selected-MoE down
// projection output.  Lane mapping mirrors the raw Q5/Q6 selected legacy
// kernel: lane16 indexes the column within the tile, half_id selects which
// row-half stores from the float8 accumulator on writeback.
template <typename scalar_t, int qtype, bool qmicro = false>
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_k_t16_selected_wmma_prefill_compact_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  static_assert(qtype == 4 || qtype == 5 || qtype == 6, "qtype must be 4, 5, or 6");
  const int col_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane16 = tid & 15;
  const int half_id = tid >> 4;

  const int64_t col_start = static_cast<int64_t>(col_tile) * 16;
  if (col_start >= out_features) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const int64_t out_col = col_start + lane16;
  const int64_t local_row = tile_row_in_expert + lane16;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;
  const bool valid_col = out_col < out_features;
  const int col_in_slab = static_cast<int>(out_col & (T16_COLS - 1));
  const int64_t out_tile_index = out_col / T16_COLS;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;
  constexpr int tile_bytes = qtype_block_bytes<qtype, qmicro>();

  float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile = valid_col
        ? k_t16_tile_ptr(tiles, expert_id, out_tile_index, blk, out_tiles, blocks_per_row, tile_bytes)
        : tiles;
    const int64_t g_k_base = blk * QK_K;

    // HIP 7.2 over-unrolls these gfx1100 outer quant-block loops and spills
    // the live WMMA fragments. Keep only the inner K16 work unrolled there;
    // gfx1151 retains the independently validated fully-unrolled schedule.
    if constexpr (qtype == 4) {
      const float d_f = valid_col ? fp16_bytes_to_float(tile + Q4_T16_D_OFFSET + col_in_slab * 2) : 0.0f;
      const float dmin_f = valid_col ? fp16_bytes_to_float(tile + Q4_T16_DMIN_OFFSET + col_in_slab * 2) : 0.0f;
#if defined(__gfx1100__)
#pragma unroll 1
#else
#pragma unroll
#endif
      for (int sb = 0; sb < Q4_SUBBLOCKS; ++sb) {
        const float scale_f = valid_col
            ? d_f * static_cast<float>(tile[Q4_T16_SCALE_OFFSET + sb * T16_COLS + col_in_slab])
            : 0.0f;
        const float min_f = valid_col
            ? dmin_f * static_cast<float>(tile[Q4_T16_MIN_OFFSET + sb * T16_COLS + col_in_slab])
            : 0.0f;
#pragma unroll
        for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
          const int k_off = sb * Q4_SUBBLOCK + kt * K_TILE;
          const int64_t k_base = g_k_base + k_off;
          const half16_t a_reg = load_compact_row_half16<scalar_t>(
              x, compact_row, in_features, k_base, valid_row);
          half16_t b_reg;
          if (valid_col) {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) {
              const int lane32 = kt * K_TILE + kk;
              const uint8_t q = q4_t16_quant(tile, sb, lane32, col_in_slab);
              b_reg[kk] = static_cast<half_t>(scale_f * static_cast<float>(q) - min_f);
            }
          } else {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) b_reg[kk] = static_cast<half_t>(0.0f);
          }
          acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
      }
    } else if constexpr (qtype == 5) {
      constexpr int d_offset = qmicro ? Q5_QMICRO_T16_D_OFFSET : Q5_T16_D_OFFSET;
      constexpr int dmin_offset = qmicro ? Q5_QMICRO_T16_DMIN_OFFSET : Q5_T16_DMIN_OFFSET;
      const float d_f = valid_col ? fp16_bytes_to_float(tile + d_offset + col_in_slab * 2) : 0.0f;
      const float dmin_f = valid_col ? fp16_bytes_to_float(tile + dmin_offset + col_in_slab * 2) : 0.0f;
#if defined(__gfx1100__)
#pragma unroll 1
#else
#pragma unroll
#endif
      for (int sb = 0; sb < Q4_SUBBLOCKS; ++sb) {
        const uint8_t scale = valid_col
            ? (qmicro
                   ? qmicro_coefficient(tile, 0, sb, col_in_slab)
                   : tile[Q5_T16_SCALE_OFFSET + sb * T16_COLS + col_in_slab])
            : 0;
        const uint8_t minv = valid_col
            ? (qmicro
                   ? qmicro_coefficient(tile, 1, sb, col_in_slab)
                   : tile[Q5_T16_MIN_OFFSET + sb * T16_COLS + col_in_slab])
            : 0;
        const float scale_f = d_f * static_cast<float>(scale);
        const float min_f = dmin_f * static_cast<float>(minv);
#pragma unroll
        for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
          const int k_off = sb * Q4_SUBBLOCK + kt * K_TILE;
          const int64_t k_base = g_k_base + k_off;
          const half16_t a_reg = load_compact_row_half16<scalar_t>(
              x, compact_row, in_features, k_base, valid_row);
          half16_t b_reg;
          if (valid_col) {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) {
              const int lane32 = kt * K_TILE + kk;
              const uint8_t q = q5_t16_quant<qmicro>(tile, sb, lane32, col_in_slab);
              b_reg[kk] = static_cast<half_t>(scale_f * static_cast<float>(q) - min_f);
            }
          } else {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) b_reg[kk] = static_cast<half_t>(0.0f);
          }
          acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
        }
      }
    } else {  // qtype == 6
      const float d_f = valid_col ? fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col_in_slab * 2) : 0.0f;
      const int8_t* __restrict__ scales =
          reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
#if defined(__gfx1100__)
#pragma unroll 1
#else
#pragma unroll
#endif
      for (int kt = 0; kt < K_TILES_PER_BLOCK; ++kt) {
        const int k_off = kt * K_TILE;
        const int64_t k_base = g_k_base + k_off;
        const half16_t a_reg = load_compact_row_half16<scalar_t>(
            x, compact_row, in_features, k_base, valid_row);
        half16_t b_reg;
        if (valid_col) {
          // 16 consecutive K-elements fall into one group16, sharing one scale.
          const int g16 = k_off >> 4;
          const float scale_f = d_f * static_cast<float>(scales[g16 * T16_COLS + col_in_slab]);
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            const int k_in_block = k_off + kk;
            const int q = q6_t16_quant(tile, k_in_block, col_in_slab);
            b_reg[kk] = static_cast<half_t>(scale_f * static_cast<float>(q));
          }
        } else {
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) b_reg[kk] = static_cast<half_t>(0.0f);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc);
      }
    }
  }

  if (out_col < out_features) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
      if (local_out_row < expert_count) {
        const int64_t compact_out_row = compact_base + local_out_row;
        if (compact_out_row < compact_rows) {
          out[compact_out_row * out_features + out_col] = float_to_scalar<scalar_t>(acc[j]);
        }
      }
    }
  }
}

// Dense one-expert Q4T16 prefill over the same bytes and arithmetic as the
// selected producer above. One wave owns 48 output columns x 64 input rows so
// each decoded weight fragment feeds four independent activation fragments;
// every output retains the original K16 WMMA association and BF16 store.
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_q4_t16_dense_wmma_prefill_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  constexpr int OUT_TILES_PER_BLOCK = 3;
  constexpr int ROW_TILES_PER_BLOCK = 4;
  const int64_t col_tile_base =
      static_cast<int64_t>(blockIdx.x) * OUT_TILES_PER_BLOCK;
  const int64_t row_base =
      static_cast<int64_t>(blockIdx.y) * ROW_TILES_PER_BLOCK * 16;
  const int lane16 = threadIdx.x & 15;
  const int half_id = threadIdx.x >> 4;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;

  float8_t acc[OUT_TILES_PER_BLOCK][ROW_TILES_PER_BLOCK];
#pragma unroll
  for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
#pragma unroll
    for (int tn = 0; tn < ROW_TILES_PER_BLOCK; ++tn) {
      acc[tm][tn] = zero_float8();
    }
  }
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* tile[OUT_TILES_PER_BLOCK];
    float d_f[OUT_TILES_PER_BLOCK];
    float dmin_f[OUT_TILES_PER_BLOCK];
#pragma unroll
    for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
      const int64_t col_tile = col_tile_base + tm;
      const int64_t safe_col_tile =
          col_tile < out_tiles ? col_tile : out_tiles - 1;
      tile[tm] = k_t16_tile_ptr(
          tiles,
          0,
          safe_col_tile,
          blk,
          out_tiles,
          blocks_per_row,
          Q4_T16_BLOCK_BYTES);
      d_f[tm] = fp16_bytes_to_float(
          tile[tm] + Q4_T16_D_OFFSET + lane16 * 2);
      dmin_f[tm] = fp16_bytes_to_float(
          tile[tm] + Q4_T16_DMIN_OFFSET + lane16 * 2);
    }
    const int64_t block_k_base = blk * QK_K;
#if defined(__gfx1100__)
#pragma unroll 1
#else
#pragma unroll
#endif
    for (int sb = 0; sb < Q4_SUBBLOCKS; ++sb) {
      float scale_f[OUT_TILES_PER_BLOCK];
      float min_f[OUT_TILES_PER_BLOCK];
#pragma unroll
      for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
        scale_f[tm] = d_f[tm] * static_cast<float>(
            tile[tm][Q4_T16_SCALE_OFFSET + sb * T16_COLS + lane16]);
        min_f[tm] = dmin_f[tm] * static_cast<float>(
            tile[tm][Q4_T16_MIN_OFFSET + sb * T16_COLS + lane16]);
      }
#pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_SUBBLOCK + kt * K_TILE;
        half16_t a_reg[ROW_TILES_PER_BLOCK];
#pragma unroll
        for (int tn = 0; tn < ROW_TILES_PER_BLOCK; ++tn) {
          const int64_t input_row = row_base + tn * 16 + lane16;
          const bool valid_row = input_row < rows;
          a_reg[tn] = load_compact_row_half16<uint16_t>(
              x,
              valid_row ? input_row : 0,
              in_features,
              block_k_base + k_off,
              valid_row);
        }
#pragma unroll
        for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
          half16_t b_reg;
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            const int lane32 = kt * K_TILE + kk;
            const uint8_t q =
                q4_t16_quant(tile[tm], sb, lane32, lane16);
            b_reg[kk] = static_cast<half_t>(
                scale_f[tm] * static_cast<float>(q) - min_f[tm]);
          }
#pragma unroll
          for (int tn = 0; tn < ROW_TILES_PER_BLOCK; ++tn) {
            acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                a_reg[tn], b_reg, acc[tm][tn]);
          }
        }
      }
    }
  }

#pragma unroll
  for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
    const int64_t out_col =
        (col_tile_base + tm) * T16_COLS + lane16;
    if (out_col >= out_features) {
      continue;
    }
#pragma unroll
    for (int tn = 0; tn < ROW_TILES_PER_BLOCK; ++tn) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t out_row = row_base + tn * 16 + half_id + j * 2;
        if (out_row < rows) {
          out[out_row * out_features + out_col] =
              float_to_scalar<uint16_t>(acc[tm][tn][j]);
        }
      }
    }
  }
}

// Cooperative dense Q4T16 producer. Four waves keep the same 48-column x
// 64-row arithmetic owner as the single-wave kernel, but share one decoded
// K256 weight slab through LDS across 256 input rows. The slab is output-major
// so one lane's K16 B fragment is contiguous and compiles to vector LDS loads.
// Every output retains the same FP16 operand, K16 WMMA sequence, FP32
// accumulator, and BF16 store.
__launch_bounds__(128, 1)
__global__ void gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  constexpr int OUT_TILES_PER_BLOCK = 3;
  constexpr int ROW_TILES_PER_WAVE = 4;
  constexpr int WAVES_PER_BLOCK = 4;
  constexpr int OUT_COLS_PER_BLOCK = OUT_TILES_PER_BLOCK * T16_COLS;
  constexpr int ROWS_PER_WAVE = ROW_TILES_PER_WAVE * 16;
  const int64_t col_tile_base =
      static_cast<int64_t>(blockIdx.x) * OUT_TILES_PER_BLOCK;
  const int wave = threadIdx.x >> 5;
  const int lane16 = threadIdx.x & 15;
  const int half_id = (threadIdx.x & 31) >> 4;
  const int64_t row_base =
      static_cast<int64_t>(blockIdx.y) * WAVES_PER_BLOCK * ROWS_PER_WAVE +
      wave * ROWS_PER_WAVE;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;

  __shared__ half_t weight_tile[QK_K * OUT_COLS_PER_BLOCK];
  float8_t acc[OUT_TILES_PER_BLOCK][ROW_TILES_PER_WAVE];
#pragma unroll
  for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
#pragma unroll
    for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
      acc[tm][tn] = zero_float8();
    }
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const int decode_pair = threadIdx.x >> 2;
    if (decode_pair < OUT_COLS_PER_BLOCK / 2) {
      const int tm = decode_pair / (T16_COLS / 2);
      const int tile_pair = decode_pair % (T16_COLS / 2);
      const int tile_col0 = tile_pair * 2;
      const int64_t col_tile = col_tile_base + tm;
      const int64_t safe_col_tile =
          col_tile < out_tiles ? col_tile : out_tiles - 1;
      const uint8_t* __restrict__ tile = k_t16_tile_ptr(
          tiles,
          0,
          safe_col_tile,
          blk,
          out_tiles,
          blocks_per_row,
          Q4_T16_BLOCK_BYTES);
      const float d0 = fp16_bytes_to_float(
          tile + Q4_T16_D_OFFSET + tile_col0 * 2);
      const float d1 = fp16_bytes_to_float(
          tile + Q4_T16_D_OFFSET + (tile_col0 + 1) * 2);
      const float dmin0 = fp16_bytes_to_float(
          tile + Q4_T16_DMIN_OFFSET + tile_col0 * 2);
      const float dmin1 = fp16_bytes_to_float(
          tile + Q4_T16_DMIN_OFFSET + (tile_col0 + 1) * 2);
      const int k_segment = (threadIdx.x & 3) * 8;
#pragma unroll
      for (int sb = 0; sb < Q4_SUBBLOCKS; ++sb) {
        const uint16_t scale_pair =
            *reinterpret_cast<const uint16_t*>(
                tile + Q4_T16_SCALE_OFFSET +
                sb * T16_COLS + tile_col0);
        const uint16_t min_pair =
            *reinterpret_cast<const uint16_t*>(
                tile + Q4_T16_MIN_OFFSET +
                sb * T16_COLS + tile_col0);
        const float scale0 =
            d0 * static_cast<float>(scale_pair & 0xFFu);
        const float scale1 =
            d1 * static_cast<float>(scale_pair >> 8);
        const float min0 =
            dmin0 * static_cast<float>(min_pair & 0xFFu);
        const float min1 =
            dmin1 * static_cast<float>(min_pair >> 8);
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          const int lane32 = k_segment + kk;
          const uint8_t packed = tile[
              Q4_T16_Q_OFFSET +
              (sb * Q4_SUBBLOCK + lane32) * (T16_COLS / 2) +
              tile_pair];
          const int k = sb * Q4_SUBBLOCK + lane32;
          const int col0 = decode_pair * 2;
          weight_tile[col0 * QK_K + k] =
              static_cast<half_t>(
                  scale0 * static_cast<float>(packed & 0x0Fu) - min0);
          weight_tile[(col0 + 1) * QK_K + k] =
              static_cast<half_t>(
                  scale1 * static_cast<float>(packed >> 4) - min1);
        }
      }
    }
    __syncthreads();

    const int64_t block_k_base = blk * QK_K;
#pragma unroll 1
    for (int sb = 0; sb < Q4_SUBBLOCKS; ++sb) {
#pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_SUBBLOCK + kt * K_TILE;
        half16_t a_reg[ROW_TILES_PER_WAVE];
#pragma unroll
        for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
          const int64_t input_row = row_base + tn * 16 + lane16;
          const bool valid_row = input_row < rows;
          a_reg[tn] = load_compact_row_half16<uint16_t>(
              x,
              valid_row ? input_row : 0,
              in_features,
              block_k_base + k_off,
              valid_row);
        }
#pragma unroll
        for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
          half16_t b_reg;
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            b_reg[kk] = weight_tile[
                (tm * T16_COLS + lane16) * QK_K + k_off + kk];
          }
#pragma unroll
          for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
            acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                a_reg[tn], b_reg, acc[tm][tn]);
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
    const int64_t out_col =
        (col_tile_base + tm) * T16_COLS + lane16;
    if (out_col >= out_features) {
      continue;
    }
#pragma unroll
    for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t out_row = row_base + tn * 16 + half_id + j * 2;
        if (out_row < rows) {
          out[out_row * out_features + out_col] =
              float_to_scalar<uint16_t>(acc[tm][tn][j]);
        }
      }
    }
  }
}

// Dense one-expert Q5T16 prefill over the same bytes and arithmetic as the
// selected producer above. One wave owns 32 output columns x 64 input rows,
// preserving each output's K16 WMMA association while amortizing activation
// and weight transport in both matrix dimensions.
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_q5_t16_dense_wmma_prefill_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  constexpr int OUT_TILES_PER_BLOCK = 2;
  constexpr int ROW_TILES_PER_BLOCK = 4;
  const int64_t col_tile_base =
      static_cast<int64_t>(blockIdx.x) * OUT_TILES_PER_BLOCK;
  const int64_t row_base =
      static_cast<int64_t>(blockIdx.y) * ROW_TILES_PER_BLOCK * 16;
  const int lane16 = threadIdx.x & 15;
  const int half_id = threadIdx.x >> 4;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;

  float8_t acc[OUT_TILES_PER_BLOCK][ROW_TILES_PER_BLOCK];
#pragma unroll
  for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
#pragma unroll
    for (int tn = 0; tn < ROW_TILES_PER_BLOCK; ++tn) {
      acc[tm][tn] = zero_float8();
    }
  }
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* tile[OUT_TILES_PER_BLOCK];
    float d_f[OUT_TILES_PER_BLOCK];
    float dmin_f[OUT_TILES_PER_BLOCK];
#pragma unroll
    for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
      const int64_t col_tile = col_tile_base + tm;
      const int64_t safe_col_tile =
          col_tile < out_tiles ? col_tile : out_tiles - 1;
      tile[tm] = k_t16_tile_ptr(
          tiles,
          0,
          safe_col_tile,
          blk,
          out_tiles,
          blocks_per_row,
          Q5_T16_BLOCK_BYTES);
      d_f[tm] = fp16_bytes_to_float(
          tile[tm] + Q5_T16_D_OFFSET + lane16 * 2);
      dmin_f[tm] = fp16_bytes_to_float(
          tile[tm] + Q5_T16_DMIN_OFFSET + lane16 * 2);
    }
    const int64_t block_k_base = blk * QK_K;
#if defined(__gfx1100__)
#pragma unroll 1
#else
#pragma unroll
#endif
    for (int sb = 0; sb < Q4_SUBBLOCKS; ++sb) {
      float scale_f[OUT_TILES_PER_BLOCK];
      float min_f[OUT_TILES_PER_BLOCK];
#pragma unroll
      for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
        scale_f[tm] = d_f[tm] * static_cast<float>(
            tile[tm][Q5_T16_SCALE_OFFSET + sb * T16_COLS + lane16]);
        min_f[tm] = dmin_f[tm] * static_cast<float>(
            tile[tm][Q5_T16_MIN_OFFSET + sb * T16_COLS + lane16]);
      }
#pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_SUBBLOCK + kt * K_TILE;
        half16_t a_reg[ROW_TILES_PER_BLOCK];
#pragma unroll
        for (int tn = 0; tn < ROW_TILES_PER_BLOCK; ++tn) {
          const int64_t input_row = row_base + tn * 16 + lane16;
          const bool valid_row = input_row < rows;
          a_reg[tn] = load_compact_row_half16<uint16_t>(
              x,
              valid_row ? input_row : 0,
              in_features,
              block_k_base + k_off,
              valid_row);
        }
#pragma unroll
        for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
          half16_t b_reg;
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            const int lane32 = kt * K_TILE + kk;
            const uint8_t q = q5_t16_quant(
                tile[tm], sb, lane32, lane16);
            b_reg[kk] = static_cast<half_t>(
                scale_f[tm] * static_cast<float>(q) - min_f[tm]);
          }
#pragma unroll
          for (int tn = 0; tn < ROW_TILES_PER_BLOCK; ++tn) {
            acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                a_reg[tn], b_reg, acc[tm][tn]);
          }
        }
      }
    }
  }

#pragma unroll
  for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
    const int64_t out_col =
        (col_tile_base + tm) * T16_COLS + lane16;
    if (out_col >= out_features) {
      continue;
    }
#pragma unroll
    for (int tn = 0; tn < ROW_TILES_PER_BLOCK; ++tn) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t out_row = row_base + tn * 16 + half_id + j * 2;
        if (out_row < rows) {
          out[out_row * out_features + out_col] =
              float_to_scalar<uint16_t>(acc[tm][tn][j]);
        }
      }
    }
  }
}

// Quality-safe Laguna candidate: each block owns one natural expert and one
// 16-column output tile, then walks that expert's compact rows in M16 tiles.
// This consumes the resident no-D2H active-expert metadata directly, avoids a
// padded-row prefix/table, and Kahan-accumulates independent K16 WMMA partials.
template <typename scalar_t, int qtype>
__launch_bounds__(32, HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS)
__global__ void gguf_k_t16_selected_expert_major_wmma_comp_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ active_experts,
    const int64_t* __restrict__ active_count_ptr,
    const uint8_t* __restrict__ tiles,
    scalar_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts) {
  static_assert(qtype == 4 || qtype == 6, "expert-major path supports Q4 or Q6");
  const int64_t out_tiles = out_features / T16_COLS;
  const int64_t active_idx = static_cast<int64_t>(blockIdx.x) / out_tiles;
  const int64_t col_tile = static_cast<int64_t>(blockIdx.x) - active_idx * out_tiles;
  const int64_t active_count = active_count_ptr[0];
  if (active_idx >= active_count || active_count <= 0 || active_count > num_experts) return;

  const int64_t expert_id = active_experts[active_idx];
  if (expert_id < 0 || expert_id >= num_experts) return;
  const int64_t begin = expert_start_compact[expert_id];
  const int64_t end = expert_start_compact[expert_id + 1];
  if (begin < 0 || begin >= end || end > compact_rows) return;

  const int lane16 = threadIdx.x & 15;
  const int half_id = threadIdx.x >> 4;
  const int64_t out_col = col_tile * T16_COLS + lane16;
  const int col_in_slab = lane16;
  const int64_t blocks_per_row = in_features / QK_K;
  constexpr int tile_bytes = qtype_block_bytes<qtype>();

  for (int64_t row_base = begin; row_base < end; row_base += 16) {
    const int64_t local_row = row_base + lane16;
    const bool valid_row = local_row < end;
    const int64_t compact_row = valid_row ? local_row : 0;
    float8_t sum = zero_float8();
    float8_t compensation = zero_float8();

    for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
      const uint8_t* __restrict__ tile = k_t16_tile_ptr(
          tiles, expert_id, col_tile, blk, out_tiles, blocks_per_row, tile_bytes);
      const int64_t g_k_base = blk * QK_K;

      if constexpr (qtype == 4) {
        const float d_f = fp16_bytes_to_float(
            tile + Q4_T16_D_OFFSET + col_in_slab * 2);
        const float dmin_f = fp16_bytes_to_float(
            tile + Q4_T16_DMIN_OFFSET + col_in_slab * 2);
#pragma unroll
        for (int sb = 0; sb < Q4_SUBBLOCKS; ++sb) {
          const float scale_f = d_f * static_cast<float>(
              tile[Q4_T16_SCALE_OFFSET + sb * T16_COLS + col_in_slab]);
          const float min_f = dmin_f * static_cast<float>(
              tile[Q4_T16_MIN_OFFSET + sb * T16_COLS + col_in_slab]);
#pragma unroll
          for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
            const int k_off = sb * Q4_SUBBLOCK + kt * K_TILE;
            const half16_t a_reg = load_compact_row_half16<scalar_t>(
                x, compact_row, in_features, g_k_base + k_off, valid_row);
            half16_t b_reg;
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) {
              const int lane32 = kt * K_TILE + kk;
              const uint8_t q = q4_t16_quant(
                  tile, sb, lane32, col_in_slab);
              b_reg[kk] = static_cast<half_t>(
                  scale_f * static_cast<float>(q) - min_f);
            }
            compensated_wmma_accumulate(
                a_reg, b_reg, &sum, &compensation);
          }
        }
      } else {
        const float d_f = fp16_bytes_to_float(
            tile + Q6_T16_D_OFFSET + col_in_slab * 2);
        const int8_t* __restrict__ scales =
            reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
#pragma unroll
        for (int kt = 0; kt < K_TILES_PER_BLOCK; ++kt) {
          const int k_off = kt * K_TILE;
          const half16_t a_reg = load_compact_row_half16<scalar_t>(
              x, compact_row, in_features, g_k_base + k_off, valid_row);
          const int g16 = k_off >> 4;
          const float scale_f = d_f * static_cast<float>(
              scales[g16 * T16_COLS + col_in_slab]);
          half16_t b_reg;
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            const int q = q6_t16_quant(tile, k_off + kk, col_in_slab);
            b_reg[kk] = static_cast<half_t>(
                scale_f * static_cast<float>(q));
          }
          compensated_wmma_accumulate(
              a_reg, b_reg, &sum, &compensation);
        }
      }
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t out_row = row_base + half_id + j * 2;
      if (out_row < end) {
        out[out_row * out_features + out_col] =
            float_to_scalar<scalar_t>(sum[j]);
      }
    }
  }
}

template <typename scalar_t, int qtype>
int launch_k_t16_selected_expert_major_comp(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* active_experts,
    const int64_t* active_count,
    const uint8_t* tiles,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    hipStream_t stream) {
  if (x == nullptr || expert_start_compact == nullptr ||
      active_experts == nullptr || active_count == nullptr || tiles == nullptr ||
      out == nullptr || compact_rows <= 0 || in_features <= 0 ||
      (in_features % QK_K) != 0 || out_features <= 0 ||
      (out_features % T16_COLS) != 0 || num_experts <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_tiles = out_features / T16_COLS;
  const dim3 block(32);
  const dim3 grid(static_cast<unsigned int>(num_experts * out_tiles));
  hipLaunchKernelGGL(
      (gguf_k_t16_selected_expert_major_wmma_comp_kernel<scalar_t, qtype>),
      grid, block, 0, stream, x, expert_start_compact, active_experts,
      active_count, tiles, out, compact_rows, in_features, out_features,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, int qtype, bool qmicro = false>
int launch_k_t16_selected(
    const scalar_t* x,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* tiles,
    scalar_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_selected_k_t16_common(
          x,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          tiles,
          out,
          compact_rows,
          in_features,
          out_features,
          num_experts,
          wmma_total_rows)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features + 15) / 16),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      (gguf_k_t16_selected_wmma_prefill_compact_kernel<scalar_t, qtype, qmicro>),
      grid,
      block,
      0,
      stream,
      x,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      tiles,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_q4_t16_dense_wmma_prefill_bf16(
    const uint16_t* x,
    const uint8_t* tiles,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (x == nullptr || tiles == nullptr || out == nullptr || rows <= 0 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int Q4_DENSE_TILE_COLS = 3 * T16_COLS;
  constexpr int Q4_DENSE_TILE_ROWS = 4 * 16;
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>(
          (out_features + Q4_DENSE_TILE_COLS - 1) / Q4_DENSE_TILE_COLS),
      static_cast<unsigned int>(
          (rows + Q4_DENSE_TILE_ROWS - 1) / Q4_DENSE_TILE_ROWS));
  hipLaunchKernelGGL(
      gguf_q4_t16_dense_wmma_prefill_bf16_kernel,
      grid,
      block,
      0,
      stream,
      x,
      tiles,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

int launch_q4_t16_dense_wmma_prefill_shared_b_bf16(
    const uint16_t* x,
    const uint8_t* tiles,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (x == nullptr || tiles == nullptr || out == nullptr || rows <= 0 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int Q4_DENSE_TILE_COLS = 3 * T16_COLS;
  constexpr int Q4_DENSE_TILE_ROWS = 4 * 4 * 16;
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(
          (out_features + Q4_DENSE_TILE_COLS - 1) / Q4_DENSE_TILE_COLS),
      static_cast<unsigned int>(
          (rows + Q4_DENSE_TILE_ROWS - 1) / Q4_DENSE_TILE_ROWS));
  hipLaunchKernelGGL(
      gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel,
      grid,
      block,
      0,
      stream,
      x,
      tiles,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

int launch_q5_t16_dense_wmma_prefill_bf16(
    const uint16_t* x,
    const uint8_t* tiles,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (x == nullptr || tiles == nullptr || out == nullptr || rows <= 0 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features <= 0 || (out_features % T16_COLS) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int Q5_DENSE_TILE_COLS = 2 * T16_COLS;
  constexpr int Q5_DENSE_TILE_ROWS = 4 * 16;
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>(
          (out_features + Q5_DENSE_TILE_COLS - 1) / Q5_DENSE_TILE_COLS),
      static_cast<unsigned int>(
          (rows + Q5_DENSE_TILE_ROWS - 1) / Q5_DENSE_TILE_ROWS));
  hipLaunchKernelGGL(
      gguf_q5_t16_dense_wmma_prefill_bf16_kernel,
      grid,
      block,
      0,
      stream,
      x,
      tiles,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

#define HIPENGINE_GGUF_K_T16_SELECTED_WMMA_EXPORT(NAME, QTYPE) \
extern "C" int hipengine_##NAME##_selected_wmma_prefill_compact_bf16_bf16_out( \
    const uint16_t* x, \
    const int64_t* expert_start_compact, \
    const int64_t* expert_start_wmma, \
    const int64_t* tile_expert, \
    const uint8_t* tiles, \
    uint16_t* out, \
    int64_t compact_rows, \
    int64_t in_features, \
    int64_t out_features, \
    int64_t num_experts, \
    int64_t wmma_total_rows, \
    hipStream_t stream) { \
  return launch_k_t16_selected<uint16_t, QTYPE>( \
      x, expert_start_compact, expert_start_wmma, tile_expert, tiles, out, \
      compact_rows, in_features, out_features, num_experts, wmma_total_rows, stream); \
} \
extern "C" int hipengine_##NAME##_selected_wmma_prefill_compact_fp16_fp16_out( \
    const half_t* x, \
    const int64_t* expert_start_compact, \
    const int64_t* expert_start_wmma, \
    const int64_t* tile_expert, \
    const uint8_t* tiles, \
    half_t* out, \
    int64_t compact_rows, \
    int64_t in_features, \
    int64_t out_features, \
    int64_t num_experts, \
    int64_t wmma_total_rows, \
    hipStream_t stream) { \
  return launch_k_t16_selected<half_t, QTYPE>( \
      x, expert_start_compact, expert_start_wmma, tile_expert, tiles, out, \
      compact_rows, in_features, out_features, num_experts, wmma_total_rows, stream); \
}

HIPENGINE_GGUF_K_T16_SELECTED_WMMA_EXPORT(gguf_q4_k_t16, 4)
HIPENGINE_GGUF_K_T16_SELECTED_WMMA_EXPORT(gguf_q5_k_t16, 5)
HIPENGINE_GGUF_K_T16_SELECTED_WMMA_EXPORT(gguf_q6_k_t16, 6)

extern "C" int hipengine_gguf_q4_k_t16_wmma_prefill_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* tiles,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q4_t16_dense_wmma_prefill_bf16(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* tiles,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q4_t16_dense_wmma_prefill_shared_b_bf16(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q5_k_t16_wmma_prefill_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* tiles,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q5_t16_dense_wmma_prefill_bf16(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q5_k_qmicro_t16_selected_wmma_prefill_compact_bf16_bf16_out(
    const uint16_t* x, const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma, const int64_t* tile_expert,
    const uint8_t* tiles, uint16_t* out, int64_t compact_rows,
    int64_t in_features, int64_t out_features, int64_t num_experts,
    int64_t wmma_total_rows, hipStream_t stream) {
  return launch_k_t16_selected<uint16_t, 5, true>(
      x, expert_start_compact, expert_start_wmma, tile_expert, tiles, out,
      compact_rows, in_features, out_features, num_experts, wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q5_k_qmicro_t16_selected_wmma_prefill_compact_fp16_fp16_out(
    const half_t* x, const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma, const int64_t* tile_expert,
    const uint8_t* tiles, half_t* out, int64_t compact_rows,
    int64_t in_features, int64_t out_features, int64_t num_experts,
    int64_t wmma_total_rows, hipStream_t stream) {
  return launch_k_t16_selected<half_t, 5, true>(
      x, expert_start_compact, expert_start_wmma, tile_expert, tiles, out,
      compact_rows, in_features, out_features, num_experts, wmma_total_rows,
      stream);
}

#define HIPENGINE_GGUF_K_T16_EXPERT_MAJOR_COMP_EXPORT(NAME, QTYPE) \
extern "C" int hipengine_##NAME##_selected_expert_major_wmma_comp_bf16_bf16_out( \
    const uint16_t* x, \
    const int64_t* expert_start_compact, \
    const int64_t* active_experts, \
    const int64_t* active_count, \
    const uint8_t* tiles, \
    uint16_t* out, \
    int64_t compact_rows, \
    int64_t in_features, \
    int64_t out_features, \
    int64_t num_experts, \
    hipStream_t stream) { \
  return launch_k_t16_selected_expert_major_comp<uint16_t, QTYPE>( \
      x, expert_start_compact, active_experts, active_count, tiles, out, \
      compact_rows, in_features, out_features, num_experts, stream); \
}

HIPENGINE_GGUF_K_T16_EXPERT_MAJOR_COMP_EXPORT(gguf_q4_k_t16, 4)
HIPENGINE_GGUF_K_T16_EXPERT_MAJOR_COMP_EXPORT(gguf_q6_k_t16, 6)
