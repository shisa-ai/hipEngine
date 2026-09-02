// Diagnostic selected-MoE Q4_K x prequantized-Q8_1 prefill prototype.
//
// This standalone kernel is a same-shape experiment for the llama.cpp MMQ
// prefill hypothesis.  It consumes raw GGUF Q4_K gate/up expert weights and a
// prequantized activation tensor with one Q8_1-style scale and sum per 32-K
// block.  It is intentionally not wired into model dispatch; use the matching
// Python wrapper/microbench to compare against the resident Q4T16 selected-WMMA
// prefill path.

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q8_1_BLOCK = 32;
constexpr int Q4_K_BLOCK_BYTES = 144;
constexpr int Q4_K_SCALE_BYTES = 12;
constexpr int Q4_K_QS_OFFSET = 2 + 2 + Q4_K_SCALE_BYTES;
constexpr int Q4_K_QS_BYTES = QK_K / 2;
constexpr int Q4_K_TILE16_COLS = 16;
constexpr int Q4_K_SUBBLOCK = 32;
constexpr int Q4_K_SUBBLOCKS = QK_K / Q4_K_SUBBLOCK;  // == 8
constexpr int Q8_1_MMQ_BLOCK = 4 * Q8_1_BLOCK;        // == 128
constexpr int Q4_K_T16_D_OFFSET = 0;
constexpr int Q4_K_T16_DMIN_OFFSET = Q4_K_T16_D_OFFSET + Q4_K_TILE16_COLS * 2;
constexpr int Q4_K_T16_SCALE_OFFSET =
    Q4_K_T16_DMIN_OFFSET + Q4_K_TILE16_COLS * 2;
constexpr int Q4_K_T16_MIN_OFFSET =
    Q4_K_T16_SCALE_OFFSET + Q4_K_SUBBLOCKS * Q4_K_TILE16_COLS;
constexpr int Q4_K_T16_Q_OFFSET =
    Q4_K_T16_MIN_OFFSET + Q4_K_SUBBLOCKS * Q4_K_TILE16_COLS;
constexpr int Q4_K_T16_BLOCK_BYTES =
    Q4_K_T16_Q_OFFSET +
    Q4_K_SUBBLOCKS * Q4_K_SUBBLOCK * (Q4_K_TILE16_COLS / 2);
constexpr int Q4_K_T16_DUAL_D_OFFSET = 0;
constexpr int Q4_K_T16_DUAL_DMIN_OFFSET =
    Q4_K_T16_DUAL_D_OFFSET + (Q4_K_TILE16_COLS / 2) * 8;
constexpr int Q4_K_T16_DUAL_SCALE_OFFSET =
    Q4_K_T16_DUAL_DMIN_OFFSET + (Q4_K_TILE16_COLS / 2) * 8;
constexpr int Q4_K_T16_DUAL_MIN_OFFSET =
    Q4_K_T16_DUAL_SCALE_OFFSET +
    Q4_K_SUBBLOCKS * (Q4_K_TILE16_COLS / 2) * 4;
constexpr int Q4_K_T16_DUAL_Q_OFFSET =
    Q4_K_T16_DUAL_MIN_OFFSET +
    Q4_K_SUBBLOCKS * (Q4_K_TILE16_COLS / 2) * 4;
constexpr int Q4_K_T16_DUAL_BLOCK_BYTES =
    Q4_K_T16_DUAL_Q_OFFSET +
    Q4_K_SUBBLOCKS * Q4_K_SUBBLOCK * 2 * 8;
constexpr int Q6_K_T16_D_OFFSET = 0;
constexpr int Q6_K_T16_SCALE_OFFSET =
    Q6_K_T16_D_OFFSET + Q4_K_TILE16_COLS * 2;
constexpr int Q6_K_T16_QL_OFFSET =
    Q6_K_T16_SCALE_OFFSET + 16 * Q4_K_TILE16_COLS;
constexpr int Q6_K_T16_QH_OFFSET =
    Q6_K_T16_QL_OFFSET + QK_K * (Q4_K_TILE16_COLS / 2);
constexpr int Q6_K_T16_BLOCK_BYTES =
    Q6_K_T16_QH_OFFSET + QK_K * (Q4_K_TILE16_COLS / 4);
constexpr int Q6_K_T16_QMICRO_OFFSET = Q6_K_T16_QL_OFFSET;
constexpr int Q6_K_T16_QMICRO_RECORD_BYTES = 12;
constexpr int Q4_K_LAYOUT_RAW = 0;
constexpr int Q4_K_LAYOUT_X8 = 1;
constexpr int Q4_K_LAYOUT_T16 = 2;
constexpr int SPARSE_EXACT_MAX_BLOCKS = 4096;

struct block_q8_1_mmq_ds4 {
  // Matches llama.cpp's DS4 block_q8_1_mmq layout for Q4_K/Q5_K MMQ:
  // four (scale, sum) half2 pairs followed by 128 int8 quants.
  uint16_t ds4[8];
  int8_t qs[Q8_1_MMQ_BLOCK];
};
static_assert(sizeof(block_q8_1_mmq_ds4) == 4 * Q8_1_BLOCK + 8 * sizeof(uint16_t),
              "unexpected DS4 Q8_1 MMQ block size");

struct block_q8_1_mmq_ds4_f32 {
  // Range-safe counterpart used for post-SiLU selected-down inputs.  The
  // activation bytes are negligible beside streamed expert weights, while
  // retaining scale/sum in FP32 avoids FP16 overflow and subnormal collapse.
  float ds4[8];
  int8_t qs[Q8_1_MMQ_BLOCK];
};
static_assert(
    sizeof(block_q8_1_mmq_ds4_f32) ==
        4 * Q8_1_BLOCK + 8 * sizeof(float),
    "unexpected FP32-DS Q8_1 MMQ block size");

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

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline float bf16_rounding_boundary_distance(float value) {
  if (!isfinite(value)) return INFINITY;
  const uint16_t center_bits = float_to_bf16_bits(value);
  uint16_t lower_bits;
  uint16_t upper_bits;
  if ((center_bits & 0x7FFFU) == 0) {
    lower_bits = 0x8001U;
    upper_bits = 0x0001U;
  } else if ((center_bits & 0x8000U) != 0) {
    lower_bits = static_cast<uint16_t>(center_bits + 1U);
    upper_bits = static_cast<uint16_t>(center_bits - 1U);
  } else {
    lower_bits = static_cast<uint16_t>(center_bits - 1U);
    upper_bits = static_cast<uint16_t>(center_bits + 1U);
  }
  const float center = bf16_bits_to_float(center_bits);
  const float lower_boundary =
      0.5f * (bf16_bits_to_float(lower_bits) + center);
  const float upper_boundary =
      0.5f * (center + bf16_bits_to_float(upper_bits));
  return fmaxf(
      0.0f,
      fminf(value - lower_boundary, upper_boundary - value));
}

__device__ inline uint16_t float_to_fp16_bits(float value) {
  union {
    half_t f16;
    uint16_t u16;
  } out;
  out.f16 = static_cast<half_t>(value);
  return out.u16;
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

__device__ inline uint8_t gguf_q4_k_scale(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] & 0x0Fu) | ((scales[idx] >> 2) & 0x30u);
}

__device__ inline uint8_t gguf_q4_k_min(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[4 + subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30u);
}

__device__ inline uint8_t gguf_q4_k_quant(
    const uint8_t* qs,
    int subblock,
    int lane32) {
  const int pair = subblock >> 1;
  const uint8_t packed = qs[pair * 32 + lane32];
  return (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
}

__device__ inline int pack_i8x4(const int8_t* ptr) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |= static_cast<uint32_t>(static_cast<uint8_t>(ptr[i])) << (8 * i);
  }
  return static_cast<int>(out);
}

__device__ inline int pack_u8x4(const uint8_t* ptr) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |= static_cast<uint32_t>(ptr[i]) << (8 * i);
  }
  return static_cast<int>(out);
}

__device__ inline int pack_q4_k_u8x4(
    const uint8_t* qs,
    int subblock,
    int lane32_base) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |= static_cast<uint32_t>(gguf_q4_k_quant(qs, subblock, lane32_base + i)) << (8 * i);
  }
  return static_cast<int>(out);
}

__device__ inline uint8_t gguf_q4_k_t16_quant(
    const uint8_t* __restrict__ tile,
    int subblock,
    int lane32,
    int col) {
  const uint8_t packed =
      tile[Q4_K_T16_Q_OFFSET +
           (subblock * Q4_K_SUBBLOCK + lane32) *
               (Q4_K_TILE16_COLS / 2) +
           (col >> 1)];
  return (col & 1) ? (packed >> 4) : (packed & 0x0Fu);
}

__device__ inline float gguf_q4_k_t16_weight(
    const uint8_t* __restrict__ tile,
    int k_in_block,
    int col) {
  const int subblock = k_in_block >> 5;
  const int lane32 = k_in_block & 31;
  const float d =
      fp16_bytes_to_float(
          tile + Q4_K_T16_D_OFFSET + col * 2);
  const float dmin =
      fp16_bytes_to_float(
          tile + Q4_K_T16_DMIN_OFFSET + col * 2);
  const float scale =
      d * static_cast<float>(
              tile[Q4_K_T16_SCALE_OFFSET +
                   subblock * Q4_K_TILE16_COLS + col]);
  const float min_value =
      dmin * static_cast<float>(
                 tile[Q4_K_T16_MIN_OFFSET +
                      subblock * Q4_K_TILE16_COLS + col]);
  const float quant = static_cast<float>(
      gguf_q4_k_t16_quant(tile, subblock, lane32, col));
  return scale * quant - min_value;
}

__device__ inline int pack_q4_k_t16_u8x4(
    const uint8_t* __restrict__ tile,
    int subblock,
    int lane32_base,
    int col) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |=
        static_cast<uint32_t>(
            gguf_q4_k_t16_quant(tile, subblock, lane32_base + i, col))
        << (8 * i);
  }
  return static_cast<int>(out);
}

__device__ inline uint32_t pack_q4_k_t16_raw_u8x4(
    const uint8_t* __restrict__ tile,
    int subblock,
    int lane32_base,
    int col) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |=
        static_cast<uint32_t>(
            tile[Q4_K_T16_Q_OFFSET +
                 (subblock * Q4_K_SUBBLOCK + lane32_base + i) *
                     (Q4_K_TILE16_COLS / 2) +
                 (col >> 1)])
        << (8 * i);
  }
  return out;
}

__device__ inline uint32_t decode_q4_k_t16_raw_u8x4(
    uint32_t raw,
    int col) {
  return (col & 1) ? ((raw >> 4) & 0x0F0F0F0Fu)
                   : (raw & 0x0F0F0F0Fu);
}

__device__ inline uint32_t pack_q4_k_t16_dual_raw_u8x4(
    const uint8_t* __restrict__ tile,
    int subblock,
    int lane32_base,
    int col,
    bool use_b) {
  uint32_t out = 0;
  const int half = col >> 3;
  const int col_pair = (col & 7) >> 1;
  const int matrix_offset = use_b ? 4 : 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |=
        static_cast<uint32_t>(
            tile[Q4_K_T16_DUAL_Q_OFFSET +
                 ((subblock * Q4_K_SUBBLOCK + lane32_base + i) * 2 +
                  half) *
                     8 +
                 matrix_offset + col_pair])
        << (8 * i);
  }
  return out;
}

__device__ inline uint32_t pack_q4_k_t16_dual_u8x4(
    const uint8_t* __restrict__ tile,
    int subblock,
    int lane32_base,
    int col,
    bool use_b) {
  return decode_q4_k_t16_raw_u8x4(
      pack_q4_k_t16_dual_raw_u8x4(
          tile, subblock, lane32_base, col, use_b),
      col);
}

__device__ inline void pack_q4_k_t16_u8x4_pair(
    const uint8_t* __restrict__ tile,
    int subblock,
    int lane32_base,
    int col_pair,
    uint32_t& low_cols,
    uint32_t& high_cols) {
  uint32_t low = 0;
  uint32_t high = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint8_t packed =
        tile[Q4_K_T16_Q_OFFSET +
             (subblock * Q4_K_SUBBLOCK + lane32_base + i) *
                 (Q4_K_TILE16_COLS / 2) +
             col_pair];
    low |= static_cast<uint32_t>(packed & 0x0Fu) << (8 * i);
    high |= static_cast<uint32_t>(packed >> 4) << (8 * i);
  }
  low_cols = low;
  high_cols = high;
}

__device__ inline uint8_t gguf_q6_k_t16_quant_unsigned(
    const uint8_t* __restrict__ tile,
    int k_in_block,
    int col) {
  const uint8_t low_packed =
      tile[Q6_K_T16_QL_OFFSET +
           k_in_block * (Q4_K_TILE16_COLS / 2) + (col >> 1)];
  const uint8_t low =
      (col & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed =
      tile[Q6_K_T16_QH_OFFSET +
           k_in_block * (Q4_K_TILE16_COLS / 4) + (col >> 2)];
  const uint8_t high =
      (high_packed >> (2 * (col & 3))) & 0x03u;
  return low | (high << 4);
}

__device__ inline uint32_t pack_q6_k_t16_u8x4(
    const uint8_t* __restrict__ tile,
    int subblock,
    int lane32_base,
    int col) {
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out |= static_cast<uint32_t>(
               gguf_q6_k_t16_quant_unsigned(
                   tile, subblock * Q8_1_BLOCK + lane32_base + i, col))
           << (8 * i);
  }
  return out;
}

__device__ inline void pack_q6_k_t16_u8x4_quartet(
    const uint8_t* __restrict__ tile,
    int subblock,
    int lane32_base,
    int col_quartet,
    uint32_t (&cols)[4]) {
  cols[0] = 0;
  cols[1] = 0;
  cols[2] = 0;
  cols[3] = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int k_in_block = subblock * Q8_1_BLOCK + lane32_base + i;
    const int ql_base =
        Q6_K_T16_QL_OFFSET +
        k_in_block * (Q4_K_TILE16_COLS / 2) + col_quartet * 2;
    const uint8_t ql01 = tile[ql_base];
    const uint8_t ql23 = tile[ql_base + 1];
    const uint8_t qh =
        tile[Q6_K_T16_QH_OFFSET +
             k_in_block * (Q4_K_TILE16_COLS / 4) + col_quartet];
    cols[0] |=
        static_cast<uint32_t>((ql01 & 0x0Fu) | ((qh & 0x03u) << 4))
        << (8 * i);
    cols[1] |=
        static_cast<uint32_t>((ql01 >> 4) | (((qh >> 2) & 0x03u) << 4))
        << (8 * i);
    cols[2] |=
        static_cast<uint32_t>((ql23 & 0x0Fu) | (((qh >> 4) & 0x03u) << 4))
        << (8 * i);
    cols[3] |=
        static_cast<uint32_t>((ql23 >> 4) | (((qh >> 6) & 0x03u) << 4))
        << (8 * i);
  }
}

__device__ inline void pack_q6_k_t16_qmicro_u8x4_quartet(
    const uint8_t* __restrict__ tile,
    int subblock,
    int pack,
    int col_quartet,
    uint32_t (&cols)[4]) {
  const int record_index =
      (subblock * (Q4_K_TILE16_COLS / 4) + col_quartet) * 8 + pack;
  const uint8_t* __restrict__ record =
      tile + Q6_K_T16_QMICRO_OFFSET +
      record_index * Q6_K_T16_QMICRO_RECORD_BYTES;
  const uint32_t ql_lo =
      reinterpret_cast<const uint32_t*>(record)[0];
  const uint32_t ql_hi =
      reinterpret_cast<const uint32_t*>(record)[1];
  const uint32_t qh =
      reinterpret_cast<const uint32_t*>(record)[2];
  cols[0] = 0;
  cols[1] = 0;
  cols[2] = 0;
  cols[3] = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t ql_word = i < 2 ? ql_lo : ql_hi;
    const int ql_shift = (i & 1) * 16;
    const uint8_t ql01 =
        static_cast<uint8_t>((ql_word >> ql_shift) & 0xFFu);
    const uint8_t ql23 =
        static_cast<uint8_t>((ql_word >> (ql_shift + 8)) & 0xFFu);
    const uint8_t qh_byte =
        static_cast<uint8_t>((qh >> (8 * i)) & 0xFFu);
    cols[0] |=
        static_cast<uint32_t>(
            (ql01 & 0x0Fu) | ((qh_byte & 0x03u) << 4))
        << (8 * i);
    cols[1] |=
        static_cast<uint32_t>(
            (ql01 >> 4) | (((qh_byte >> 2) & 0x03u) << 4))
        << (8 * i);
    cols[2] |=
        static_cast<uint32_t>(
            (ql23 & 0x0Fu) | (((qh_byte >> 4) & 0x03u) << 4))
        << (8 * i);
    cols[3] |=
        static_cast<uint32_t>(
            (ql23 >> 4) | (((qh_byte >> 6) & 0x03u) << 4))
        << (8 * i);
  }
}

__device__ inline int gguf_q4_k_mmq_dp4a(uint32_t q4, uint32_t q8, int acc) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(
      true,
      static_cast<int>(q4),
      true,
      static_cast<int>(q8),
      acc,
      false);
#else
  const int8_t x0 = static_cast<int8_t>(q8 & 0xFFu);
  const int8_t x1 = static_cast<int8_t>((q8 >> 8) & 0xFFu);
  const int8_t x2 = static_cast<int8_t>((q8 >> 16) & 0xFFu);
  const int8_t x3 = static_cast<int8_t>((q8 >> 24) & 0xFFu);
  return acc + static_cast<int>(q4 & 0xFFu) * static_cast<int>(x0) +
      static_cast<int>((q4 >> 8) & 0xFFu) * static_cast<int>(x1) +
      static_cast<int>((q4 >> 16) & 0xFFu) * static_cast<int>(x2) +
      static_cast<int>((q4 >> 24) & 0xFFu) * static_cast<int>(x3);
#endif
}

struct alignas(4) q4_k_mmq32_cache {
  // Matches the Vulkan Q4_K MMQ shared/register cache: four words hold two
  // nibble planes each, followed by FP16-rounded (d*scale, dmin*min).
  uint32_t qs[4];
  uint16_t dm[2];
};
static_assert(sizeof(q4_k_mmq32_cache) == 20, "unexpected Q4_K MMQ cache size");

struct alignas(4) q8_1_mmq32_cache {
  int32_t qs[8];
  uint16_t ds[2];
};
static_assert(sizeof(q8_1_mmq32_cache) == 36, "unexpected Q8_1 MMQ cache size");

__device__ inline float wave32_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  return value;
}

__device__ inline float wave32_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down(value, offset));
  }
  return value;
}

__device__ inline float subgroup16_max(float value) {
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down(value, offset, 16));
  }
  return value;
}

__device__ inline int subgroup16_sum(int value) {
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset, 16);
  }
  return value;
}

template <int residual_passes>
__launch_bounds__(128, 1)
__global__ void gguf_q8_1_mmq_ds4_pack_bf16_kernel(
    const uint16_t* __restrict__ x,
    block_q8_1_mmq_ds4* __restrict__ out,
    int64_t rows,
    int64_t hidden) {
  const int64_t q8_block = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int ds4_idx = tid >> 5;
  const int lane = tid & 31;
  const int64_t blocks_per_row = hidden / Q8_1_MMQ_BLOCK;
  if (row >= rows || q8_block >= blocks_per_row) return;

  const int64_t in_idx = row * hidden + q8_block * Q8_1_MMQ_BLOCK + tid;
  float residual = bf16_bits_to_float(x[in_idx]);

  __shared__ float scale_shared[4];
  __shared__ float sum_shared[4];
#pragma unroll
  for (int pass = 0; pass < residual_passes; ++pass) {
    const float sum = wave32_sum(residual);
    const float max_abs = wave32_max(fabsf(residual));
    if (lane == 0) {
      scale_shared[ds4_idx] = max_abs / 127.0f;
      sum_shared[ds4_idx] = sum;
    }
    __syncthreads();

    const float scale = scale_shared[ds4_idx];
    int q = 0;
    if (scale > 0.0f) {
      const float qf =
          fminf(127.0f, fmaxf(-127.0f, rintf(residual / scale)));
      q = static_cast<int>(qf);
    }
    const int64_t plane_blocks = rows * blocks_per_row;
    block_q8_1_mmq_ds4* __restrict__ out_block =
        out + static_cast<int64_t>(pass) * plane_blocks +
        row * blocks_per_row + q8_block;
    out_block->qs[tid] = static_cast<int8_t>(q);
    const uint16_t scale_bits = float_to_fp16_bits(scale);
    if (lane == 0) {
      out_block->ds4[2 * ds4_idx + 0] = scale_bits;
      out_block->ds4[2 * ds4_idx + 1] =
          float_to_fp16_bits(sum_shared[ds4_idx]);
    }
    residual = fmaf(
        -static_cast<float>(q), fp16_bits_to_float(scale_bits), residual);
    __syncthreads();
  }
}

template <
    int residual_passes,
    bool Q6_HALF_SUMS = false>
__launch_bounds__(128, 1)
__global__ void gguf_q8_1_mmq_ds4_f32_pack_bf16_kernel(
    const uint16_t* __restrict__ x,
    block_q8_1_mmq_ds4_f32* __restrict__ out,
    int64_t rows,
    int64_t hidden) {
  const int64_t q8_block = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int ds4_idx = tid >> 5;
  const int lane = tid & 31;
  const int64_t blocks_per_row = hidden / Q8_1_MMQ_BLOCK;
  if (row >= rows || q8_block >= blocks_per_row) return;
  static_assert(
      !Q6_HALF_SUMS || residual_passes == 1,
      "Q6 half-sum packing supports the one-plane D4 path");

  const int64_t in_idx =
      row * hidden + q8_block * Q8_1_MMQ_BLOCK + tid;
  float residual = bf16_bits_to_float(x[in_idx]);

  __shared__ float scale_shared[4];
  __shared__ float sum_shared[4];
#pragma unroll
  for (int pass = 0; pass < residual_passes; ++pass) {
    const float sum =
        Q6_HALF_SUMS ? 0.0f : wave32_sum(residual);
    const float max_abs = wave32_max(fabsf(residual));
    if (lane == 0) {
      scale_shared[ds4_idx] = max_abs / 127.0f;
      sum_shared[ds4_idx] = sum;
    }
    __syncthreads();

    const float scale = scale_shared[ds4_idx];
    int q = 0;
    if (scale > 0.0f) {
      const float qf =
          fminf(127.0f, fmaxf(-127.0f, rintf(residual / scale)));
      q = static_cast<int>(qf);
    }
    const int half_sum =
        Q6_HALF_SUMS ? subgroup16_sum(q) : 0;
    const int upper_half_sum =
        Q6_HALF_SUMS ? __shfl(half_sum, 16, 32) : 0;

    const int64_t plane_blocks = rows * blocks_per_row;
    block_q8_1_mmq_ds4_f32* __restrict__ out_block =
        out + static_cast<int64_t>(pass) * plane_blocks +
        row * blocks_per_row + q8_block;
    out_block->qs[tid] = static_cast<int8_t>(q);
    if (lane == 0) {
      out_block->ds4[2 * ds4_idx + 0] = scale;
      if constexpr (Q6_HALF_SUMS) {
        const uint32_t packed_sums =
            static_cast<uint16_t>(half_sum) |
            (static_cast<uint32_t>(
                 static_cast<uint16_t>(upper_half_sum))
             << 16);
        reinterpret_cast<uint32_t*>(out_block->ds4)
            [2 * ds4_idx + 1] = packed_sums;
      } else {
        out_block->ds4[2 * ds4_idx + 1] = sum_shared[ds4_idx];
      }
    }
    residual = fmaf(-static_cast<float>(q), scale, residual);
    __syncthreads();
  }
}

template <
    int residual_passes,
    bool Q6_HALF_SUMS = false>
__launch_bounds__(128, 1)
__global__ void gguf_q8_1_mmq_ds4_f32_pack_dual_silu_bf16_kernel(
    const uint16_t* __restrict__ gate_up,
    block_q8_1_mmq_ds4_f32* __restrict__ out,
    int64_t rows,
    int64_t hidden) {
  const int64_t q8_block = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int ds4_idx = tid >> 5;
  const int lane = tid & 31;
  const int64_t blocks_per_row = hidden / Q8_1_MMQ_BLOCK;
  if (row >= rows || q8_block >= blocks_per_row) return;
  static_assert(
      !Q6_HALF_SUMS || residual_passes == 1,
      "Q6 half-sum packing supports the one-plane D4 path");

  const int64_t col = q8_block * Q8_1_MMQ_BLOCK + tid;
  const int64_t row_base = row * hidden * 2;
  const float gate = bf16_bits_to_float(gate_up[row_base + col]);
  const float up = bf16_bits_to_float(gate_up[row_base + hidden + col]);
  const float sig = 1.0f / (1.0f + expf(-gate));
  // Preserve the standalone SiLU kernel's BF16 output boundary before
  // quantization. This makes the fused body byte-identical to
  // silu_mul_dual_out_bf16 followed by the ordinary FP32-metadata packer.
  float residual =
      bf16_bits_to_float(float_to_bf16_bits(gate * sig * up));

  __shared__ float scale_shared[4];
  __shared__ float sum_shared[4];
#pragma unroll
  for (int pass = 0; pass < residual_passes; ++pass) {
    const float sum =
        Q6_HALF_SUMS ? 0.0f : wave32_sum(residual);
    const float max_abs = wave32_max(fabsf(residual));
    if (lane == 0) {
      scale_shared[ds4_idx] = max_abs / 127.0f;
      sum_shared[ds4_idx] = sum;
    }
    __syncthreads();

    const float scale = scale_shared[ds4_idx];
    int q = 0;
    if (scale > 0.0f) {
      const float qf =
          fminf(127.0f, fmaxf(-127.0f, rintf(residual / scale)));
      q = static_cast<int>(qf);
    }
    const int half_sum =
        Q6_HALF_SUMS ? subgroup16_sum(q) : 0;
    const int upper_half_sum =
        Q6_HALF_SUMS ? __shfl(half_sum, 16, 32) : 0;

    const int64_t plane_blocks = rows * blocks_per_row;
    block_q8_1_mmq_ds4_f32* __restrict__ out_block =
        out + static_cast<int64_t>(pass) * plane_blocks +
        row * blocks_per_row + q8_block;
    out_block->qs[tid] = static_cast<int8_t>(q);
    if (lane == 0) {
      out_block->ds4[2 * ds4_idx + 0] = scale;
      if constexpr (Q6_HALF_SUMS) {
        const uint32_t packed_sums =
            static_cast<uint16_t>(half_sum) |
            (static_cast<uint32_t>(
                 static_cast<uint16_t>(upper_half_sum))
             << 16);
        reinterpret_cast<uint32_t*>(out_block->ds4)
            [2 * ds4_idx + 1] = packed_sums;
      } else {
        out_block->ds4[2 * ds4_idx + 1] = sum_shared[ds4_idx];
      }
    }
    residual = fmaf(-static_cast<float>(q), scale, residual);
    __syncthreads();
  }
}

template <bool PRECOMPUTE_HALF_SUMS = false>
__launch_bounds__(128, 1)
__global__ void gguf_q8_1_mmq_ds8_f32_pack_bf16_kernel(
    const uint16_t* __restrict__ x,
    block_q8_1_mmq_ds4_f32* __restrict__ out,
    int16_t* __restrict__ half_sums,
    int64_t rows,
    int64_t hidden) {
  const int64_t q8_block = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int ds8_idx = tid >> 4;
  const int lane16 = tid & 15;
  const int64_t blocks_per_row = hidden / Q8_1_MMQ_BLOCK;
  if (row >= rows || q8_block >= blocks_per_row) return;

  const int64_t in_idx =
      row * hidden + q8_block * Q8_1_MMQ_BLOCK + tid;
  const float value = bf16_bits_to_float(x[in_idx]);
  const float max_abs = subgroup16_max(fabsf(value));

  __shared__ float scale_shared[8];
  if (lane16 == 0) {
    scale_shared[ds8_idx] = max_abs / 127.0f;
  }
  __syncthreads();

  const float scale = scale_shared[ds8_idx];
  int q = 0;
  if (scale > 0.0f) {
    const float qf =
        fminf(127.0f, fmaxf(-127.0f, rintf(value / scale)));
    q = static_cast<int>(qf);
  }
  const int half_sum =
      PRECOMPUTE_HALF_SUMS ? subgroup16_sum(q) : 0;
  block_q8_1_mmq_ds4_f32* __restrict__ out_block =
      out + row * blocks_per_row + q8_block;
  out_block->qs[tid] = static_cast<int8_t>(q);
  if (lane16 == 0) {
    out_block->ds4[ds8_idx] = scale;
    if constexpr (PRECOMPUTE_HALF_SUMS) {
      const int64_t half_blocks_per_row = hidden / 16;
      half_sums[
          row * half_blocks_per_row + q8_block * 8 + ds8_idx] =
          static_cast<int16_t>(half_sum);
    }
  }
}

bool valid_q8_1_mmq_ds4_pack(
    const void* x,
    const void* out,
    int64_t rows,
    int64_t hidden) {
  return x != nullptr && out != nullptr && rows > 0 && hidden > 0 &&
      (hidden % Q8_1_MMQ_BLOCK) == 0;
}

bool valid_q8_1_selected_dual_common(
    const void* x_qs,
    const void* x_d,
    const void* x_sum,
    const void* expert_start_compact,
    const void* expert_start_wmma,
    const void* tile_expert,
    const void* qweight_a,
    const void* qweight_b,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows) {
  return x_qs != nullptr && x_d != nullptr && x_sum != nullptr &&
      expert_start_compact != nullptr && expert_start_wmma != nullptr &&
      tile_expert != nullptr && qweight_a != nullptr && qweight_b != nullptr &&
      out != nullptr && compact_rows > 0 && in_features > 0 &&
      out_features_a > 0 && out_features_b > 0 && num_experts > 0 &&
      wmma_total_rows > 0 && (wmma_total_rows % 16) == 0 &&
      (in_features % QK_K) == 0 && (out_features_a % 16) == 0 &&
      (out_features_b % 16) == 0;
}

bool valid_q8_1_selected_dual_mmq32(
    const void* x_q8,
    const void* compact_to_source,
    const void* expert_start_compact,
    const void* expert_start_mmq32,
    const void* tile_expert,
    const void* qweight_a,
    const void* qweight_b,
    const void* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows) {
  return compact_to_source != nullptr && valid_q8_1_selected_dual_common(
             x_q8,
             x_q8,
             x_q8,
             expert_start_compact,
             expert_start_mmq32,
             tile_expert,
             qweight_a,
             qweight_b,
             out,
             compact_rows,
             in_features,
             out_features_a,
             out_features_b,
             num_experts,
             mmq_total_rows) &&
      (out_features_a % 32) == 0 && (out_features_b % 32) == 0 &&
      (mmq_total_rows % 32) == 0;
}

// One 256-thread block computes a scalar 16-row x 16-column output tile.  This
// deliberately simple mapping is a correctness/performance baseline for the
// integer-dot direction; it is not yet a llama.cpp-style tiled MMQ kernel.
__launch_bounds__(256, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_prefill_compact32_kernel(
    const int8_t* __restrict__ x_qs,
    const float* __restrict__ x_d,
    const float* __restrict__ x_sum,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
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
  const int row_lane = tid >> 4;
  const int col_lane = tid & 15;

  const int64_t global_col = static_cast<int64_t>(col_tile) * 16 + col_lane;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (global_col >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;
  const int64_t local_row = tile_row_in_expert + row_lane;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;

  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const bool valid_col = local_col < local_out_features;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_blocks_per_row = in_features / Q8_1_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row =
      valid_col ? (expert_base + local_col * row_bytes) : expert_base;

  float acc = 0.0f;
  if (valid_row && valid_col && compact_row < compact_rows) {
    for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
      const uint8_t* __restrict__ block_ptr = weight_row + blk * Q4_K_BLOCK_BYTES;
      const float d_f = fp16_bytes_to_float(block_ptr);
      const float dmin_f = fp16_bytes_to_float(block_ptr + 2);
      const uint8_t* __restrict__ scales = block_ptr + 4;
      const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
#pragma unroll
      for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
        const int64_t q8_block = compact_row * q8_blocks_per_row + blk * Q4_K_SUBBLOCKS + sb;
        const int8_t* __restrict__ xq = x_qs + q8_block * Q8_1_BLOCK;
        const float xd = x_d[q8_block];
        const float xsum = x_sum[q8_block];
        int dot = 0;
#pragma unroll
        for (int lane32 = 0; lane32 < Q8_1_BLOCK; ++lane32) {
          const int q4 = static_cast<int>(gguf_q4_k_quant(qs, sb, lane32));
          dot += q4 * static_cast<int>(xq[lane32]);
        }
        const float scale_f = d_f * static_cast<float>(gguf_q4_k_scale(scales, sb));
        const float min_f = dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb));
        acc += scale_f * xd * static_cast<float>(dot) - min_f * xsum;
      }
    }
    out[compact_row * out_features_total + global_col] = float_to_bf16_bits(acc);
  }
}

__launch_bounds__(256, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
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
  const int row_lane = tid >> 4;
  const int col_lane = tid & 15;

  const int64_t global_col = static_cast<int64_t>(col_tile) * 16 + col_lane;
  const int64_t out_features_total = out_features_a + out_features_b;
  if (global_col >= out_features_total) return;

  const int64_t expert_id = tile_expert[row_tile];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count = expert_start_compact[expert_id + 1] - compact_base;
  const int64_t wmma_base = expert_start_wmma[expert_id];
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;
  const int64_t local_row = tile_row_in_expert + row_lane;
  const bool valid_row = local_row < expert_count;
  const int64_t compact_row = valid_row ? (compact_base + local_row) : 0;

  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const bool valid_col = local_col < local_out_features;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row =
      valid_col ? (expert_base + local_col * row_bytes) : expert_base;

  float acc = 0.0f;
  if (valid_row && valid_col && compact_row < compact_rows) {
    for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
      const uint8_t* __restrict__ block_ptr = weight_row + blk * Q4_K_BLOCK_BYTES;
      const float d_f = fp16_bytes_to_float(block_ptr);
      const float dmin_f = fp16_bytes_to_float(block_ptr + 2);
      const uint8_t* __restrict__ scales = block_ptr + 4;
      const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
#pragma unroll
      for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
        const int64_t q8_block = compact_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
        const int ds4_idx = sb & 3;
        const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
        const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK;
        const float xd = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 0]);
        const float xsum = fp16_bits_to_float(xb.ds4[2 * ds4_idx + 1]);
        int dot = 0;
#pragma unroll
        for (int lane32 = 0; lane32 < Q8_1_BLOCK; ++lane32) {
          const int q4 = static_cast<int>(gguf_q4_k_quant(qs, sb, lane32));
          dot += q4 * static_cast<int>(xq[lane32]);
        }
        const float scale_f = d_f * static_cast<float>(gguf_q4_k_scale(scales, sb));
        const float min_f = dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb));
        acc += scale_f * xd * static_cast<float>(dot) - min_f * xsum;
      }
    }
    out[compact_row * out_features_total + global_col] = float_to_bf16_bits(acc);
  }
}

// Source-faithful translation of llama.cpp Vulkan's small K-quant MMQ tile.
//
// Vulkan uses one 64-lane subgroup for a 32-output-column x 32-routed-row
// result tile. gfx1151 executes wave32, so this block uses four waves: all four
// reuse the same 32 decoded Q4_K rows and 32 Q8_1 rows, while each wave owns one
// routed-row residue. Each lane therefore keeps eight FP32 accumulators instead
// of the 16 required by a literal two-wave mapping. WMITER remains one,
// matching Vulkan's K-quant pressure control.
template <int WEIGHT_LAYOUT, int ACTIVATION_PASSES, bool COLLECT_RISKS>
__device__ __forceinline__ void
gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_body(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ compact_to_source,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_mmq32,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    int* __restrict__ risk_count,
    int* __restrict__ risk_tiles,
    int64_t max_risks,
    float risk_threshold,
    q4_k_mmq32_cache* __restrict__ weight_tile,
    q8_1_mmq32_cache* __restrict__ activation_tile) {
  constexpr int MMQ_M = 32;
  constexpr int MMQ_N = 32;
  constexpr int MMQ_K = 32;
  constexpr int THREADS = 128;
  constexpr int TM = 1;
  constexpr int WNITER = 8;

  const int tid = static_cast<int>(threadIdx.x);
  const int tile_col = static_cast<int>(blockIdx.x);
  const int tile_row = static_cast<int>(blockIdx.y);
  const int64_t expert_id = tile_expert[tile_row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t global_col_base = static_cast<int64_t>(tile_col) * MMQ_M;
  if (global_col_base >= out_features_total) return;

  const bool use_b = global_col_base >= out_features_a;
  const int64_t local_col_base =
      use_b ? (global_col_base - out_features_a) : global_col_base;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count =
      expert_start_compact[expert_id + 1] - compact_base;
  const int64_t padded_base = expert_start_mmq32[expert_id];
  const int64_t tile_row_in_expert =
      static_cast<int64_t>(tile_row) * MMQ_N - padded_base;
  const int64_t remaining_rows = expert_count - tile_row_in_expert;
  const int valid_rows =
      remaining_rows <= 0
          ? 0
          : (remaining_rows >= MMQ_N ? MMQ_N
                                     : static_cast<int>(remaining_rows));

  const int64_t blocks_per_weight_row = in_features / QK_K;
  const int64_t q8_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t weight_row_bytes =
      blocks_per_weight_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * weight_row_bytes;
  const uint8_t* __restrict__ expert_base =
      qweight + expert_id * expert_bytes;

  float acc[WNITER * TM];
#pragma unroll
  for (int i = 0; i < WNITER * TM; ++i) acc[i] = 0.0f;

  const int tiwr = tid & 31;
  const int tiwc = tid >> 5;
  const int load_weight_part = tid & 3;
  const int load_weight_col = tid >> 2;
  const int load_activation_part = tid & 3;
  const int load_activation_row = tid >> 2;
  const int k32_iters = static_cast<int>(in_features / MMQ_K);

  for (int k32 = 0; k32 < k32_iters; ++k32) {
    const int weight_block = k32 >> 3;
    const int subblock = k32 & 7;

#pragma unroll
    for (int col_offset = load_weight_col;
         col_offset < MMQ_M;
         col_offset += THREADS * 8 / MMQ_K) {
      const int64_t local_col = local_col_base + col_offset;
      const bool valid_col = local_col < local_out_features;
      uint32_t packed_q4 = 0;
      uint16_t packed_scale = 0;
      uint16_t packed_min = 0;
      if (valid_col) {
        const uint8_t* __restrict__ block = nullptr;
        const uint8_t* __restrict__ t16_tile = nullptr;
        int t16_col = 0;
        if constexpr (WEIGHT_LAYOUT == Q4_K_LAYOUT_X8) {
          constexpr int X8_COLS = 8;
          const int64_t out_packed = local_out_features / X8_COLS;
          const int64_t out_pack = local_col / X8_COLS;
          const int64_t col_in_pack = local_col % X8_COLS;
          const int64_t x8_block =
              ((expert_id * out_packed + out_pack) *
                   blocks_per_weight_row +
               static_cast<int64_t>(weight_block)) *
                  X8_COLS +
              col_in_pack;
          block = qweight + x8_block * Q4_K_BLOCK_BYTES;
        } else if constexpr (WEIGHT_LAYOUT == Q4_K_LAYOUT_T16) {
          const int64_t out_tiles =
              local_out_features / Q4_K_TILE16_COLS;
          const int64_t out_tile = local_col / Q4_K_TILE16_COLS;
          t16_col =
              static_cast<int>(local_col % Q4_K_TILE16_COLS);
          const int64_t tile_index =
              (expert_id * out_tiles + out_tile) *
                  blocks_per_weight_row +
              static_cast<int64_t>(weight_block);
          t16_tile =
              qweight + tile_index * Q4_K_T16_BLOCK_BYTES;
        } else {
          block =
              expert_base + local_col * weight_row_bytes +
              static_cast<int64_t>(weight_block) * Q4_K_BLOCK_BYTES;
        }
        const int lane8 = load_weight_part * 8;
        uint32_t low = 0;
        uint32_t high = 0;
        if constexpr (WEIGHT_LAYOUT == Q4_K_LAYOUT_T16) {
          low = static_cast<uint32_t>(
              pack_q4_k_t16_u8x4(
                  t16_tile, subblock, lane8, t16_col));
          high = static_cast<uint32_t>(
              pack_q4_k_t16_u8x4(
                  t16_tile, subblock, lane8 + 4, t16_col));
        } else {
          const uint8_t* __restrict__ qs = block + Q4_K_QS_OFFSET;
          low = static_cast<uint32_t>(
              pack_q4_k_u8x4(qs, subblock, lane8));
          high = static_cast<uint32_t>(
              pack_q4_k_u8x4(qs, subblock, lane8 + 4));
        }
        packed_q4 = low | (high << 4);
        if (load_weight_part == 0) {
          float d = 0.0f;
          float dmin = 0.0f;
          uint8_t scale = 0;
          uint8_t min = 0;
          if constexpr (WEIGHT_LAYOUT == Q4_K_LAYOUT_T16) {
            d = fp16_bytes_to_float(
                t16_tile + Q4_K_T16_D_OFFSET + t16_col * 2);
            dmin = fp16_bytes_to_float(
                t16_tile + Q4_K_T16_DMIN_OFFSET + t16_col * 2);
            scale =
                t16_tile[Q4_K_T16_SCALE_OFFSET +
                         subblock * Q4_K_TILE16_COLS + t16_col];
            min =
                t16_tile[Q4_K_T16_MIN_OFFSET +
                         subblock * Q4_K_TILE16_COLS + t16_col];
          } else {
            const uint8_t* __restrict__ scales = block + 4;
            d = fp16_bytes_to_float(block);
            dmin = fp16_bytes_to_float(block + 2);
            scale = gguf_q4_k_scale(scales, subblock);
            min = gguf_q4_k_min(scales, subblock);
          }
          packed_scale = float_to_fp16_bits(
              d * static_cast<float>(scale));
          packed_min = float_to_fp16_bits(
              dmin * static_cast<float>(min));
        }
      }
      weight_tile[col_offset].qs[load_weight_part] = packed_q4;
      if (load_weight_part == 0) {
        weight_tile[col_offset].dm[0] = packed_scale;
        weight_tile[col_offset].dm[1] = packed_min;
      }
    }

    const int64_t local_route =
        tile_row_in_expert + load_activation_row;
    const bool valid_route =
        local_route >= 0 && local_route < expert_count;
#pragma unroll
    for (int activation_pass = 0;
         activation_pass < ACTIVATION_PASSES;
         ++activation_pass) {
      if (valid_route) {
        const int64_t compact_row = compact_base + local_route;
        const int64_t source_row = compact_to_source[compact_row];
        const int64_t plane_blocks =
            q8_blocks_per_row * source_rows;
        const block_q8_1_mmq_ds4& xb =
            x_q8[static_cast<int64_t>(activation_pass) * plane_blocks +
                 source_row * q8_blocks_per_row + (k32 >> 2)];
        const int ds4_idx = k32 & 3;
        const int32_t* __restrict__ x_words =
            reinterpret_cast<const int32_t*>(
                xb.qs + ds4_idx * Q8_1_BLOCK +
                load_activation_part * 8);
#pragma unroll
        for (int word = 0; word < 2; ++word) {
          activation_tile[load_activation_row]
              .qs[load_activation_part * 2 + word] = x_words[word];
        }
        if (load_activation_part == 0) {
          activation_tile[load_activation_row].ds[0] =
              xb.ds4[2 * ds4_idx + 0];
          activation_tile[load_activation_row].ds[1] =
              xb.ds4[2 * ds4_idx + 1];
        }
      } else {
#pragma unroll
        for (int word = 0; word < 2; ++word) {
          activation_tile[load_activation_row]
              .qs[load_activation_part * 2 + word] = 0;
        }
        if (load_activation_part == 0) {
          activation_tile[load_activation_row].ds[0] = 0;
          activation_tile[load_activation_row].ds[1] = 0;
        }
      }
      __syncthreads();

      q4_k_mmq32_cache weight_cache[TM];
#pragma unroll
      for (int cr = 0; cr < TM; ++cr) {
        weight_cache[cr] = weight_tile[tiwr * TM + cr];
      }

#pragma unroll
      for (int wsic = 0; wsic < WNITER; ++wsic) {
        if (wsic * 4 + tiwc >= valid_rows) continue;
        const q8_1_mmq32_cache activation_cache =
            activation_tile[wsic * 4 + tiwc];
        const float xd =
            fp16_bits_to_float(activation_cache.ds[0]);
        const float xsum =
            fp16_bits_to_float(activation_cache.ds[1]);
#pragma unroll
        for (int cr = 0; cr < TM; ++cr) {
          int dot = 0;
#pragma unroll
          for (int qword = 0; qword < 8; ++qword) {
            const uint32_t q4_word =
                (weight_cache[cr].qs[qword >> 1] >>
                 ((qword & 1) * 4)) &
                0x0F0F0F0Fu;
            dot = gguf_q4_k_mmq_dp4a(
                q4_word,
                static_cast<uint32_t>(
                    activation_cache.qs[qword]),
                dot);
          }
          const float dscale =
              fp16_bits_to_float(weight_cache[cr].dm[0]);
          const float dmin =
              fp16_bits_to_float(weight_cache[cr].dm[1]);
          acc[wsic * TM + cr] +=
              xd * dscale * static_cast<float>(dot) -
              dmin * xsum;
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int wsic = 0; wsic < WNITER; ++wsic) {
    const int64_t local_route = tile_row_in_expert + wsic * 4 + tiwc;
    if (local_route < 0 || local_route >= expert_count) continue;
    const int64_t compact_row = compact_base + local_route;
#pragma unroll
    for (int cr = 0; cr < TM; ++cr) {
      const int64_t global_col = global_col_base + tiwr * TM + cr;
      if (global_col < out_features_total) {
        const float value = acc[wsic * TM + cr];
        out[compact_row * out_features_total + global_col] =
            float_to_bf16_bits(value);
        if constexpr (COLLECT_RISKS) {
          const bool at_risk =
              bf16_rounding_boundary_distance(value) <=
              risk_threshold;
          const unsigned long long risk_mask =
              __ballot(at_risk);
          const int half = tiwr >> 4;
          const unsigned long long half_mask =
              half == 0
                  ? (risk_mask & 0xFFFFULL)
                  : ((risk_mask >> 16) & 0xFFFFULL);
          if ((tiwr & 15) == 0 && half_mask != 0) {
            const int slot = atomicAdd(risk_count, 1);
            if (slot < max_risks) {
              const int64_t tiles_per_row =
                  out_features_total / Q4_K_TILE16_COLS;
              risk_tiles[slot] = static_cast<int>(
                  compact_row * tiles_per_row +
                  global_col / Q4_K_TILE16_COLS);
            }
          }
        }
      }
    }
  }
}

__launch_bounds__(128, 2)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ compact_to_source,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_mmq32,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows) {
  __shared__ q4_k_mmq32_cache weight_tile[32];
  __shared__ q8_1_mmq32_cache activation_tile[32];
  gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_body<
      Q4_K_LAYOUT_RAW, 1, false>(
      x_q8,
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      nullptr,
      nullptr,
      0,
      0.0f,
      weight_tile,
      activation_tile);
}

__launch_bounds__(128, 2)
__global__ void gguf_q4_k_x8_selected_dual_q8_1_ds4_mmq32_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ compact_to_source,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_mmq32,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows) {
  __shared__ q4_k_mmq32_cache weight_tile[32];
  __shared__ q8_1_mmq32_cache activation_tile[32];
  gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_body<
      Q4_K_LAYOUT_X8, 1, false>(
      x_q8,
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      nullptr,
      nullptr,
      0,
      0.0f,
      weight_tile,
      activation_tile);
}

template <int ACTIVATION_PASSES, bool COLLECT_RISKS>
__launch_bounds__(128, 2)
__global__ void gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ compact_to_source,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_mmq32,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    int* __restrict__ risk_count,
    int* __restrict__ risk_tiles,
    int64_t max_risks,
    float risk_threshold) {
  __shared__ q4_k_mmq32_cache weight_tile[32];
  __shared__ q8_1_mmq32_cache activation_tile[32];
  gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_body<
      Q4_K_LAYOUT_T16, ACTIVATION_PASSES, COLLECT_RISKS>(
      x_q8,
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      risk_count,
      risk_tiles,
      max_risks,
      risk_threshold,
      weight_tile,
      activation_tile);
}

__launch_bounds__(256, 2)
__global__ void gguf_q4_k_t16_selected_dual_sparse_exact_correct_bf16_kernel(
    const uint16_t* __restrict__ source_x,
    const int64_t* __restrict__ compact_to_source,
    const int64_t* __restrict__ expert_start_compact,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
    const int* __restrict__ risk_count,
    const int* __restrict__ risk_tiles,
    int64_t max_risks,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts) {
  const int64_t out_features_total =
      out_features_a + out_features_b;
  const int64_t total_tiles_per_row =
      out_features_total / Q4_K_TILE16_COLS;
  const int64_t total_tiles =
      compact_rows * total_tiles_per_row;
  const int queued = max(*risk_count, 0);
  const bool overflow = queued > max_risks;
  const int64_t bounded_queued =
      queued < max_risks ? queued : max_risks;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  __shared__ float wave_sums[8 * Q4_K_TILE16_COLS];
  const int64_t work_items =
      overflow ? total_tiles : bounded_queued;

  for (int64_t slot = static_cast<int64_t>(blockIdx.x);
       slot < work_items;
       slot += static_cast<int64_t>(gridDim.x)) {
    const int64_t tile_key =
        overflow ? slot : static_cast<int64_t>(risk_tiles[slot]);
    if (tile_key < 0 || tile_key >= total_tiles) continue;
    const int64_t compact_row =
        tile_key / total_tiles_per_row;
    const int64_t global_tile =
        tile_key - compact_row * total_tiles_per_row;
    const int64_t source_row = compact_to_source[compact_row];
    if (source_row < 0 || source_row >= source_rows) continue;

    int64_t expert_id = -1;
    for (int64_t expert = 0; expert < num_experts; ++expert) {
      if (compact_row >= expert_start_compact[expert] &&
          compact_row < expert_start_compact[expert + 1]) {
        expert_id = expert;
        break;
      }
    }
    if (expert_id < 0) continue;

    const int64_t out_tiles_a =
        out_features_a / Q4_K_TILE16_COLS;
    const bool use_b = global_tile >= out_tiles_a;
    const int64_t local_tile =
        use_b ? global_tile - out_tiles_a : global_tile;
    const int64_t local_out_features =
        use_b ? out_features_b : out_features_a;
    const int64_t out_tiles =
        local_out_features / Q4_K_TILE16_COLS;
    const uint8_t* __restrict__ qweight =
        use_b ? qweight_b : qweight_a;
    const int64_t blocks_per_row = in_features / QK_K;
    const uint16_t* __restrict__ x_row =
        source_x + source_row * in_features;

    float acc[Q4_K_TILE16_COLS];
#pragma unroll
    for (int col = 0; col < Q4_K_TILE16_COLS; ++col) {
      acc[col] = 0.0f;
    }
    for (int64_t block = 0; block < blocks_per_row; ++block) {
      const int64_t tile_index =
          (expert_id * out_tiles + local_tile) *
              blocks_per_row +
          block;
      const uint8_t* __restrict__ tile =
          qweight + tile_index * Q4_K_T16_BLOCK_BYTES;
      const int64_t k_base = block * QK_K;
      for (int k_in_block = tid;
           k_in_block < QK_K;
           k_in_block += static_cast<int>(blockDim.x)) {
        const float xv =
            bf16_bits_to_float(x_row[k_base + k_in_block]);
#pragma unroll
        for (int col = 0; col < Q4_K_TILE16_COLS; ++col) {
          acc[col] +=
              xv * gguf_q4_k_t16_weight(
                       tile, k_in_block, col);
        }
      }
    }

#pragma unroll
    for (int col = 0; col < Q4_K_TILE16_COLS; ++col) {
      acc[col] = wave32_sum(acc[col]);
    }
    if (lane == 0) {
#pragma unroll
      for (int col = 0; col < Q4_K_TILE16_COLS; ++col) {
        wave_sums[wave * Q4_K_TILE16_COLS + col] =
            acc[col];
      }
    }
    __syncthreads();
    if (tid == 0) {
      const int64_t global_col_base =
          global_tile * Q4_K_TILE16_COLS;
#pragma unroll
      for (int col = 0; col < Q4_K_TILE16_COLS; ++col) {
        float total = 0.0f;
#pragma unroll
        for (int wave_idx = 0; wave_idx < 8; ++wave_idx) {
          total +=
              wave_sums[wave_idx * Q4_K_TILE16_COLS + col];
        }
        out[compact_row * out_features_total +
            global_col_base + col] =
            float_to_bf16_bits(total);
      }
    }
    __syncthreads();
  }
}

__launch_bounds__(32, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
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
  const int lane16 = tid & 15;
  const int half_id = tid >> 4;

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  const int64_t col_base = static_cast<int64_t>(col_tile) * 16;
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
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row = valid_col
      ? (expert_base + local_col * row_bytes)
      : expert_base;

  const int64_t local_a_row = tile_row_in_expert + lane16;
  const bool valid_a_row = local_a_row < expert_count;
  const int64_t compact_a_row = valid_a_row ? (compact_base + local_a_row) : 0;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const uint8_t* __restrict__ qs = nullptr;
      float scale_f = 0.0f;
      float min_f = 0.0f;
      if (valid_col) {
        const uint8_t* __restrict__ block_ptr = weight_row + blk * Q4_K_BLOCK_BYTES;
        const float d_f = fp16_bytes_to_float(block_ptr);
        const float dmin_f = fp16_bytes_to_float(block_ptr + 2);
        const uint8_t* __restrict__ scales = block_ptr + 4;
        qs = block_ptr + Q4_K_QS_OFFSET;
        scale_f = d_f * static_cast<float>(gguf_q4_k_scale(scales, sb));
        min_f = dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb));
      }

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a_row && compact_a_row < compact_rows) {
          const int64_t q8_block = compact_a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
          const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK + kt * 16;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        int32x4_t b_vec = {0, 0, 0, 0};
        if (valid_col) {
          const int lane32_base = kt * 16;
          b_vec = int32x4_t{
              pack_q4_k_u8x4(qs, sb, lane32_base + 0),
              pack_q4_k_u8x4(qs, sb, lane32_base + 4),
              pack_q4_k_u8x4(qs, sb, lane32_base + 8),
              pack_q4_k_u8x4(qs, sb, lane32_base + 12),
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

__launch_bounds__(64, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
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

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

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
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row = valid_col
      ? (expert_base + local_col * row_bytes)
      : expert_base;

  const int64_t local_a_row = tile_row_in_expert + lane16;
  const bool valid_a_row = local_a_row < expert_count;
  const int64_t compact_a_row = valid_a_row ? (compact_base + local_a_row) : 0;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const uint8_t* __restrict__ qs = nullptr;
      float scale_f = 0.0f;
      float min_f = 0.0f;
      if (valid_col) {
        const uint8_t* __restrict__ block_ptr = weight_row + blk * Q4_K_BLOCK_BYTES;
        const float d_f = fp16_bytes_to_float(block_ptr);
        const float dmin_f = fp16_bytes_to_float(block_ptr + 2);
        const uint8_t* __restrict__ scales = block_ptr + 4;
        qs = block_ptr + Q4_K_QS_OFFSET;
        scale_f = d_f * static_cast<float>(gguf_q4_k_scale(scales, sb));
        min_f = dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb));
      }

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a_row && compact_a_row < compact_rows) {
          const int64_t q8_block = compact_a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
          const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK + kt * 16;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        int32x4_t b_vec = {0, 0, 0, 0};
        if (valid_col) {
          const int lane32_base = kt * 16;
          b_vec = int32x4_t{
              pack_q4_k_u8x4(qs, sb, lane32_base + 0),
              pack_q4_k_u8x4(qs, sb, lane32_base + 4),
              pack_q4_k_u8x4(qs, sb, lane32_base + 8),
              pack_q4_k_u8x4(qs, sb, lane32_base + 12),
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

__launch_bounds__(128, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
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

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  const int64_t col_base = static_cast<int64_t>(col_tile) * 64 + wave_id * 16;
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
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row = valid_col
      ? (expert_base + local_col * row_bytes)
      : expert_base;

  const int64_t local_a_row = tile_row_in_expert + lane16;
  const bool valid_a_row = local_a_row < expert_count;
  const int64_t compact_a_row = valid_a_row ? (compact_base + local_a_row) : 0;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const uint8_t* __restrict__ qs = nullptr;
      float scale_f = 0.0f;
      float min_f = 0.0f;
      if (valid_col) {
        const uint8_t* __restrict__ block_ptr = weight_row + blk * Q4_K_BLOCK_BYTES;
        const float d_f = fp16_bytes_to_float(block_ptr);
        const float dmin_f = fp16_bytes_to_float(block_ptr + 2);
        const uint8_t* __restrict__ scales = block_ptr + 4;
        qs = block_ptr + Q4_K_QS_OFFSET;
        scale_f = d_f * static_cast<float>(gguf_q4_k_scale(scales, sb));
        min_f = dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb));
      }

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a_row && compact_a_row < compact_rows) {
          const int64_t q8_block = compact_a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
          const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK + kt * 16;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        int32x4_t b_vec = {0, 0, 0, 0};
        if (valid_col) {
          const int lane32_base = kt * 16;
          b_vec = int32x4_t{
              pack_q4_k_u8x4(qs, sb, lane32_base + 0),
              pack_q4_k_u8x4(qs, sb, lane32_base + 4),
              pack_q4_k_u8x4(qs, sb, lane32_base + 8),
              pack_q4_k_u8x4(qs, sb, lane32_base + 12),
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

__launch_bounds__(64, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ q4_a,
    const float* __restrict__ scale_a,
    const float* __restrict__ min_a,
    const uint8_t* __restrict__ q4_b,
    const float* __restrict__ scale_b,
    const float* __restrict__ min_b,
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

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

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
  if (local_col >= local_out_features) return;
  const uint8_t* __restrict__ q4 = use_b ? q4_b : q4_a;
  const float* __restrict__ scale = use_b ? scale_b : scale_a;
  const float* __restrict__ minv = use_b ? min_b : min_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t out_tiles = local_out_features / Q4_K_TILE16_COLS;
  const int64_t local_tile = local_col / Q4_K_TILE16_COLS;
  const int64_t col_in_tile = local_col & (Q4_K_TILE16_COLS - 1);

  const int64_t local_a_row = tile_row_in_expert + lane16;
  const bool valid_a_row = local_a_row < expert_count;
  const int64_t compact_a_row = valid_a_row ? (compact_base + local_a_row) : 0;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const int64_t preview_index =
          (((((expert_id * out_tiles + local_tile) * Q4_K_TILE16_COLS + col_in_tile) *
             blocks_per_row + blk) * Q4_K_SUBBLOCKS + sb));
      const uint8_t* __restrict__ q4_vec = q4 + preview_index * Q4_K_SUBBLOCK;
      const float scale_f = scale[preview_index];
      const float min_f = minv[preview_index];

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a_row && compact_a_row < compact_rows) {
          const int64_t q8_block = compact_a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
          const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK + kt * 16;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        const int lane32_base = kt * 16;
        const int32x4_t b_vec = int32x4_t{
            pack_u8x4(q4_vec + lane32_base + 0),
            pack_u8x4(q4_vec + lane32_base + 4),
            pack_u8x4(q4_vec + lane32_base + 8),
            pack_u8x4(q4_vec + lane32_base + 12),
        };
        dot_acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, a_vec, true, b_vec, dot_acc, false);
      }

#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int64_t local_out_row = tile_row_in_expert + half_id + j * 2;
        if (local_out_row < expert_count) {
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
    if (local_out_row < expert_count) {
      const int64_t compact_out_row = compact_base + local_out_row;
      if (compact_out_row < compact_rows) {
        out[compact_out_row * out_features_total + global_col] = float_to_bf16_bits(acc_f[j]);
      }
    }
  }
}

__launch_bounds__(64, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
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

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  __shared__ uint8_t sh_qs[2][16][Q4_K_QS_BYTES];
  __shared__ float sh_scale[2][16][Q4_K_SUBBLOCKS];
  __shared__ float sh_min[2][16][Q4_K_SUBBLOCKS];

  const int64_t col_base = static_cast<int64_t>(col_tile) * 32 + wave_id * 16;
  const int64_t global_col = col_base + lane16;
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t expert_id_raw = tile_expert[row_tile];
  const bool valid_expert = expert_id_raw >= 0 && expert_id_raw < num_experts;
  const int64_t expert_id = valid_expert ? expert_id_raw : 0;

  const int64_t compact_base = valid_expert ? expert_start_compact[expert_id] : 0;
  const int64_t expert_count = valid_expert
      ? (expert_start_compact[expert_id + 1] - compact_base)
      : 0;
  const int64_t wmma_base = valid_expert ? expert_start_wmma[expert_id] : 0;
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const bool in_output_range = global_col < out_features_total;
  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const bool valid_col = valid_expert && in_output_range && local_col < local_out_features;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row = valid_col
      ? (expert_base + local_col * row_bytes)
      : expert_base;

  const int64_t local_a_row = tile_row_in_expert + lane16;
  const bool valid_a_row = valid_expert && local_a_row < expert_count;
  const int64_t compact_a_row = valid_a_row ? (compact_base + local_a_row) : 0;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    if (half_id == 0) {
      const uint8_t* __restrict__ block_ptr = valid_col
          ? (weight_row + blk * Q4_K_BLOCK_BYTES)
          : weight_row;
      const float d_f = valid_col ? fp16_bytes_to_float(block_ptr) : 0.0f;
      const float dmin_f = valid_col ? fp16_bytes_to_float(block_ptr + 2) : 0.0f;
      const uint8_t* __restrict__ scales = block_ptr + 4;
      const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
#pragma unroll
      for (int i = 0; i < Q4_K_QS_BYTES; ++i) {
        sh_qs[wave_id][lane16][i] = valid_col ? qs[i] : 0;
      }
#pragma unroll
      for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
        sh_scale[wave_id][lane16][sb] = valid_col
            ? d_f * static_cast<float>(gguf_q4_k_scale(scales, sb))
            : 0.0f;
        sh_min[wave_id][lane16][sb] = valid_col
            ? dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb))
            : 0.0f;
      }
    }
    __syncthreads();

#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const float scale_f = sh_scale[wave_id][lane16][sb];
      const float min_f = sh_min[wave_id][lane16][sb];
      const uint8_t* __restrict__ qs_tile = &sh_qs[wave_id][lane16][0];

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a_row && compact_a_row < compact_rows) {
          const int64_t q8_block = compact_a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
          const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK + kt * 16;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        const int lane32_base = kt * 16;
        const int32x4_t b_vec = int32x4_t{
            pack_q4_k_u8x4(qs_tile, sb, lane32_base + 0),
            pack_q4_k_u8x4(qs_tile, sb, lane32_base + 4),
            pack_q4_k_u8x4(qs_tile, sb, lane32_base + 8),
            pack_q4_k_u8x4(qs_tile, sb, lane32_base + 12),
        };
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
    __syncthreads();
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

__launch_bounds__(64, 1)
__global__ void gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_wmma,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
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

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  __shared__ uint8_t sh_q4[2][16][Q4_K_SUBBLOCKS][Q4_K_SUBBLOCK];
  __shared__ float sh_scale[2][16][Q4_K_SUBBLOCKS];
  __shared__ float sh_min[2][16][Q4_K_SUBBLOCKS];

  const int64_t col_base = static_cast<int64_t>(col_tile) * 32 + wave_id * 16;
  const int64_t global_col = col_base + lane16;
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t expert_id_raw = tile_expert[row_tile];
  const bool valid_expert = expert_id_raw >= 0 && expert_id_raw < num_experts;
  const int64_t expert_id = valid_expert ? expert_id_raw : 0;

  const int64_t compact_base = valid_expert ? expert_start_compact[expert_id] : 0;
  const int64_t expert_count = valid_expert
      ? (expert_start_compact[expert_id + 1] - compact_base)
      : 0;
  const int64_t wmma_base = valid_expert ? expert_start_wmma[expert_id] : 0;
  const int64_t row_start_in_wmma = static_cast<int64_t>(row_tile) * 16;
  const int64_t tile_row_in_expert = row_start_in_wmma - wmma_base;

  const bool in_output_range = global_col < out_features_total;
  const bool use_b = global_col >= out_features_a;
  const int64_t local_col = use_b ? (global_col - out_features_a) : global_col;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const bool valid_col = valid_expert && in_output_range && local_col < local_out_features;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t q8_mmq_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const int64_t expert_bytes = local_out_features * row_bytes;
  const uint8_t* __restrict__ expert_base = qweight + expert_id * expert_bytes;
  const uint8_t* __restrict__ weight_row = valid_col
      ? (expert_base + local_col * row_bytes)
      : expert_base;

  const int64_t local_a_row = tile_row_in_expert + lane16;
  const bool valid_a_row = valid_expert && local_a_row < expert_count;
  const int64_t compact_a_row = valid_a_row ? (compact_base + local_a_row) : 0;

  float acc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    if (half_id == 0) {
      const uint8_t* __restrict__ block_ptr = valid_col
          ? (weight_row + blk * Q4_K_BLOCK_BYTES)
          : weight_row;
      const float d_f = valid_col ? fp16_bytes_to_float(block_ptr) : 0.0f;
      const float dmin_f = valid_col ? fp16_bytes_to_float(block_ptr + 2) : 0.0f;
      const uint8_t* __restrict__ scales = block_ptr + 4;
      const uint8_t* __restrict__ qs = block_ptr + Q4_K_QS_OFFSET;
#pragma unroll
      for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
        sh_scale[wave_id][lane16][sb] = valid_col
            ? d_f * static_cast<float>(gguf_q4_k_scale(scales, sb))
            : 0.0f;
        sh_min[wave_id][lane16][sb] = valid_col
            ? dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb))
            : 0.0f;
#pragma unroll
        for (int lane32 = 0; lane32 < Q4_K_SUBBLOCK; ++lane32) {
          sh_q4[wave_id][lane16][sb][lane32] = valid_col
              ? gguf_q4_k_quant(qs, sb, lane32)
              : 0;
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const float scale_f = sh_scale[wave_id][lane16][sb];
      const float min_f = sh_min[wave_id][lane16][sb];
      const uint8_t* __restrict__ q4_tile = &sh_q4[wave_id][lane16][sb][0];

      int32x8_t dot_acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        int32x4_t a_vec = {0, 0, 0, 0};
        if (valid_a_row && compact_a_row < compact_rows) {
          const int64_t q8_block = compact_a_row * q8_mmq_blocks_per_row + blk * 2 + (sb >> 2);
          const int ds4_idx = sb & 3;
          const block_q8_1_mmq_ds4& xb = x_q8[q8_block];
          const int8_t* __restrict__ xq = xb.qs + ds4_idx * Q8_1_BLOCK + kt * 16;
          a_vec = int32x4_t{
              pack_i8x4(xq + 0),
              pack_i8x4(xq + 4),
              pack_i8x4(xq + 8),
              pack_i8x4(xq + 12),
          };
        }
        const int lane32_base = kt * 16;
        const int32x4_t b_vec = int32x4_t{
            pack_u8x4(q4_tile + lane32_base + 0),
            pack_u8x4(q4_tile + lane32_base + 4),
            pack_u8x4(q4_tile + lane32_base + 8),
            pack_u8x4(q4_tile + lane32_base + 12),
        };
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
    __syncthreads();
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

__launch_bounds__(32, 1)
__global__ void gguf_q4_k_q8_1_wmma_i8_probe_16x16_kernel(
    const int8_t* __restrict__ a_rows,
    const uint8_t* __restrict__ b_cols,
    int32_t* __restrict__ out) {
  const int tid = threadIdx.x;
  const int lane16 = tid & 15;
  const int half_id = tid >> 4;

  typedef int int32x4_t __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int int32x8_t __attribute__((__vector_size__(8 * sizeof(int))));

  int32x4_t a_vec = {
      pack_i8x4(a_rows + lane16 * 16 + 0),
      pack_i8x4(a_rows + lane16 * 16 + 4),
      pack_i8x4(a_rows + lane16 * 16 + 8),
      pack_i8x4(a_rows + lane16 * 16 + 12),
  };
  int32x4_t b_vec = {
      pack_u8x4(b_cols + lane16 * 16 + 0),
      pack_u8x4(b_cols + lane16 * 16 + 4),
      pack_u8x4(b_cols + lane16 * 16 + 8),
      pack_u8x4(b_cols + lane16 * 16 + 12),
  };
  int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
  acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
      true, a_vec, true, b_vec, acc, false);

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int row = half_id + j * 2;
    out[row * 16 + lane16] = acc[j];
  }
}

int launch_wmma_i8_probe_16x16(
    const int8_t* a_rows,
    const uint8_t* b_cols,
    int32_t* out,
    hipStream_t stream) {
  if (a_rows == nullptr || b_cols == nullptr || out == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gguf_q4_k_q8_1_wmma_i8_probe_16x16_kernel,
      dim3(1),
      dim3(32),
      0,
      stream,
      a_rows,
      b_cols,
      out);
  return static_cast<int>(hipGetLastError());
}

template <int residual_passes>
int launch_q8_1_mmq_ds4_pack_bf16(
    const uint16_t* x,
    block_q8_1_mmq_ds4* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (!valid_q8_1_mmq_ds4_pack(x, out, rows, hidden)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks_per_row = hidden / Q8_1_MMQ_BLOCK;
  hipLaunchKernelGGL(
      (gguf_q8_1_mmq_ds4_pack_bf16_kernel<residual_passes>),
      dim3(static_cast<unsigned int>(blocks_per_row), static_cast<unsigned int>(rows)),
      dim3(128),
      0,
      stream,
      x,
      out,
      rows,
      hidden);
  return static_cast<int>(hipGetLastError());
}

template <
    int residual_passes,
    bool Q6_HALF_SUMS = false>
int launch_q8_1_mmq_ds4_f32_pack_bf16(
    const uint16_t* x,
    block_q8_1_mmq_ds4_f32* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (!valid_q8_1_mmq_ds4_pack(x, out, rows, hidden)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks_per_row = hidden / Q8_1_MMQ_BLOCK;
  hipLaunchKernelGGL(
      (gguf_q8_1_mmq_ds4_f32_pack_bf16_kernel<
          residual_passes,
          Q6_HALF_SUMS>),
      dim3(
          static_cast<unsigned int>(blocks_per_row),
          static_cast<unsigned int>(rows)),
      dim3(128),
      0,
      stream,
      x,
      out,
      rows,
      hidden);
  return static_cast<int>(hipGetLastError());
}

template <
    int residual_passes,
    bool Q6_HALF_SUMS = false>
int launch_q8_1_mmq_ds4_f32_pack_dual_silu_bf16(
    const uint16_t* gate_up,
    block_q8_1_mmq_ds4_f32* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (!valid_q8_1_mmq_ds4_pack(gate_up, out, rows, hidden)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks_per_row = hidden / Q8_1_MMQ_BLOCK;
  hipLaunchKernelGGL(
      (gguf_q8_1_mmq_ds4_f32_pack_dual_silu_bf16_kernel<
          residual_passes,
          Q6_HALF_SUMS>),
      dim3(
          static_cast<unsigned int>(blocks_per_row),
          static_cast<unsigned int>(rows)),
      dim3(128),
      0,
      stream,
      gate_up,
      out,
      rows,
      hidden);
  return static_cast<int>(hipGetLastError());
}

template <bool PRECOMPUTE_HALF_SUMS = false>
int launch_q8_1_mmq_ds8_f32_pack_bf16(
    const uint16_t* x,
    block_q8_1_mmq_ds4_f32* out,
    int16_t* half_sums,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (!valid_q8_1_mmq_ds4_pack(x, out, rows, hidden) ||
      (PRECOMPUTE_HALF_SUMS && half_sums == nullptr)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks_per_row = hidden / Q8_1_MMQ_BLOCK;
  hipLaunchKernelGGL(
      (gguf_q8_1_mmq_ds8_f32_pack_bf16_kernel<
          PRECOMPUTE_HALF_SUMS>),
      dim3(
          static_cast<unsigned int>(blocks_per_row),
          static_cast<unsigned int>(rows)),
      dim3(128),
      0,
      stream,
      x,
      out,
      half_sums,
      rows,
      hidden);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_compact32(
    const int8_t* x_qs,
    const float* x_d,
    const float* x_sum,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_q8_1_selected_dual_common(
          x_qs,
          x_d,
          x_sum,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          qweight_a,
          qweight_b,
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
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 15) / 16),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_qs,
      x_d,
      x_sum,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_wmma_compact32(
    const block_q8_1_mmq_ds4* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          qweight_a,
          qweight_b,
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
      static_cast<unsigned int>((out_features_total + 15) / 16),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_wmma32_compact32(
    const block_q8_1_mmq_ds4* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          qweight_a,
          qweight_b,
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
  const dim3 block(64);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 31) / 32),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_wmma64_compact32(
    const block_q8_1_mmq_ds4* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          qweight_a,
          qweight_b,
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
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 63) / 64),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_preview_wmma32_compact32(
    const block_q8_1_mmq_ds4* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* q4_a,
    const float* scale_a,
    const float* min_a,
    const uint8_t* q4_b,
    const float* scale_b,
    const float* min_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          q4_a,
          q4_b,
          out,
          compact_rows,
          in_features,
          out_features_a,
          out_features_b,
          num_experts,
          wmma_total_rows) || scale_a == nullptr || min_a == nullptr ||
      scale_b == nullptr || min_b == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = wmma_total_rows / 16;
  const dim3 block(64);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 31) / 32),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      q4_a,
      scale_a,
      min_a,
      q4_b,
      scale_b,
      min_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_wmma32_ldspack_compact32(
    const block_q8_1_mmq_ds4* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          qweight_a,
          qweight_b,
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
  const dim3 block(64);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 31) / 32),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_wmma32_lds_compact32(
    const block_q8_1_mmq_ds4* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          qweight_a,
          qweight_b,
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
  const dim3 block(64);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 31) / 32),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows);
  return static_cast<int>(hipGetLastError());
}

struct alignas(4) q6_k_mmq64_cache {
  uint32_t qs[8];
  float scales[2];
};
static_assert(
    sizeof(q6_k_mmq64_cache) == 40,
    "unexpected Q6_K MMQ cache size");

struct alignas(4) q8_1_mmq32_f32_cache {
  int32_t qs[8];
  int32_t sums[2];
  float d;
  float s;
};
static_assert(
    sizeof(q8_1_mmq32_f32_cache) == 48,
    "unexpected FP32-DS Q8_1 cache size");

struct alignas(4) q8_1_q6_mmq32_f32_compact_cache {
  int32_t qs[8];
  int16_t sums[2];
  float d;
};
static_assert(
    sizeof(q8_1_q6_mmq32_f32_compact_cache) == 40,
    "unexpected compact Q6 FP32-DS Q8_1 cache size");

template <bool COMPACT>
struct q6_activation_cache_selector {
  using type = q8_1_mmq32_f32_cache;
};

template <>
struct q6_activation_cache_selector<true> {
  using type = q8_1_q6_mmq32_f32_compact_cache;
};

struct alignas(4) q4_k_mmq64_f32_cache {
  uint32_t qs[8];
  float dm[2];
};
static_assert(
    sizeof(q4_k_mmq64_f32_cache) == 40,
    "unexpected FP32 Q4_K MMQ cache size");

template <
    int ACTIVATION_PASSES,
    bool IDENTITY_SOURCE = false,
    bool SPLIT16 = false,
    int MMQ_M = 64,
    bool ROWVEC_ACTIVATION_STAGE = false,
    bool WAVE_COLUMNS = false,
    int THREADS = 128,
    bool DIRECT_WAVE_DECODE = false,
    bool DOUBLE_BUFFER_ACTIVATION = false,
    int RAW_WEIGHT_PREFETCH_PACKS = 0,
    bool PRECOMPUTED_ACTIVATION_SUMS = false,
    bool DUAL_INTERLEAVED_WEIGHT = false>
__launch_bounds__(THREADS, 1)
__global__ void
gguf_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4_f32* __restrict__ x_q8,
    const int64_t* __restrict__ compact_to_source,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_mmq32,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    const int16_t* __restrict__ precomputed_activation_sums) {
  constexpr int MMQ_N = 32;
  constexpr int MMQ_K = 32;
  constexpr int WNITER = 8;
  constexpr int COL_ITERS = WAVE_COLUMNS ? 1 : MMQ_M / 32;
  constexpr int ROW_ITERS = WAVE_COLUMNS ? MMQ_N : WNITER;
  typedef int mmq_i32x4_t
      __attribute__((__vector_size__(4 * sizeof(int))));
  static_assert(
      !DIRECT_WAVE_DECODE || WAVE_COLUMNS,
      "direct Q4 decode requires wave-column mapping");
  static_assert(
      !DOUBLE_BUFFER_ACTIVATION ||
          (WAVE_COLUMNS && DIRECT_WAVE_DECODE),
      "activation double buffer requires direct wave-column decode");
  static_assert(
      RAW_WEIGHT_PREFETCH_PACKS == 0 ||
          RAW_WEIGHT_PREFETCH_PACKS == 8,
      "raw Q4 prefetch supports 0 or 8 packs");
  static_assert(
      RAW_WEIGHT_PREFETCH_PACKS == 0 ||
          (ACTIVATION_PASSES == 1 &&
           ROWVEC_ACTIVATION_STAGE && WAVE_COLUMNS &&
           DIRECT_WAVE_DECODE &&
           ((SPLIT16 && MMQ_M == 128 && THREADS == 128 &&
             DOUBLE_BUFFER_ACTIVATION) ||
            (!SPLIT16 && MMQ_M == 64 && THREADS == 64 &&
             !DOUBLE_BUFFER_ACTIVATION))),
      "raw Q4 prefetch requires admitted gate/up or down geometry");
  static_assert(
      !PRECOMPUTED_ACTIVATION_SUMS ||
          (ACTIVATION_PASSES == 1 && SPLIT16 &&
           MMQ_M == 128 && ROWVEC_ACTIVATION_STAGE &&
           WAVE_COLUMNS && THREADS == 128 &&
           DIRECT_WAVE_DECODE && DOUBLE_BUFFER_ACTIVATION &&
           RAW_WEIGHT_PREFETCH_PACKS == 8),
      "precomputed sums require production D8 gate/up geometry");
  static_assert(
      !DUAL_INTERLEAVED_WEIGHT ||
          (ACTIVATION_PASSES == 1 && SPLIT16 &&
           MMQ_M == 128 && ROWVEC_ACTIVATION_STAGE &&
           WAVE_COLUMNS && THREADS == 128 &&
           DIRECT_WAVE_DECODE && DOUBLE_BUFFER_ACTIVATION &&
           RAW_WEIGHT_PREFETCH_PACKS == 8),
      "dual-interleaved weights require production D8 gate/up geometry");

  const int tid = static_cast<int>(threadIdx.x);
  const int tile_col = static_cast<int>(blockIdx.x);
  const int tile_row = static_cast<int>(blockIdx.y);
  const int64_t expert_id = tile_expert[tile_row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t global_col_base = static_cast<int64_t>(tile_col) * MMQ_M;
  if (global_col_base >= out_features_total) return;
  const bool use_b = global_col_base >= out_features_a;
  const int64_t local_col_base =
      use_b ? global_col_base - out_features_a : global_col_base;
  const int64_t local_out_features =
      use_b ? out_features_b : out_features_a;
  const uint8_t* __restrict__ qweight =
      DUAL_INTERLEAVED_WEIGHT
          ? qweight_a
          : (use_b ? qweight_b : qweight_a);

  const int64_t compact_base = expert_start_compact[expert_id];
  const int64_t expert_count =
      expert_start_compact[expert_id + 1] - compact_base;
  const int64_t padded_base = expert_start_mmq32[expert_id];
  const int64_t tile_row_in_expert =
      static_cast<int64_t>(tile_row) * MMQ_N - padded_base;
  const int64_t remaining_rows = expert_count - tile_row_in_expert;
  const int valid_rows =
      remaining_rows <= 0
          ? 0
          : (remaining_rows >= MMQ_N ? MMQ_N
                                     : static_cast<int>(remaining_rows));

  const int64_t blocks_per_weight_row = in_features / QK_K;
  const int64_t q8_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t out_tiles =
      local_out_features / Q4_K_TILE16_COLS;
  const int64_t plane_blocks = source_rows * q8_blocks_per_row;
  const int64_t half_blocks_per_row = in_features / 16;
  const int wave = tid >> 5;
  const int lane = tid & 31;

  float acc[COL_ITERS][ROW_ITERS];
#pragma unroll
  for (int col_iter = 0; col_iter < COL_ITERS; ++col_iter) {
#pragma unroll
    for (int row_iter = 0; row_iter < ROW_ITERS; ++row_iter) {
      acc[col_iter][row_iter] = 0.0f;
    }
  }

  __shared__ q4_k_mmq64_f32_cache
      weight_tile[WAVE_COLUMNS ? 1 : MMQ_M];
  __shared__ q8_1_mmq32_f32_cache
      activation_tile[DOUBLE_BUFFER_ACTIVATION ? 2 : 1]
                     [ACTIVATION_PASSES][MMQ_N];

  const int k32_iters = static_cast<int>(in_features / MMQ_K);
  uint32_t prefetched_raw[
      RAW_WEIGHT_PREFETCH_PACKS > 0 ? RAW_WEIGHT_PREFETCH_PACKS : 1];
  for (int k32 = 0; k32 < k32_iters; ++k32) {
    const int activation_slot =
        DOUBLE_BUFFER_ACTIVATION ? (k32 & 1) : 0;
    const int weight_block = k32 >> 3;
    const int subblock = k32 & 7;

    q4_k_mmq64_f32_cache wave_weight{};
    if constexpr (WAVE_COLUMNS) {
      const int col_offset = wave * 32 + lane;
      const int64_t local_col = local_col_base + col_offset;
      if (local_col < local_out_features) {
        const int64_t out_tile = local_col / Q4_K_TILE16_COLS;
        const int col = static_cast<int>(
            local_col % Q4_K_TILE16_COLS);
        const int64_t tile_index =
            (expert_id * out_tiles + out_tile) *
                blocks_per_weight_row +
            weight_block;
        const uint8_t* __restrict__ tile =
            qweight +
            tile_index *
                (DUAL_INTERLEAVED_WEIGHT
                     ? Q4_K_T16_DUAL_BLOCK_BYTES
                     : Q4_K_T16_BLOCK_BYTES);
#pragma unroll
        for (int pack = 0; pack < 8; ++pack) {
          if constexpr (DIRECT_WAVE_DECODE) {
            if constexpr (RAW_WEIGHT_PREFETCH_PACKS > 0) {
              if (pack < RAW_WEIGHT_PREFETCH_PACKS && k32 > 0) {
                wave_weight.qs[pack] =
                    decode_q4_k_t16_raw_u8x4(
                        prefetched_raw[pack], col);
              } else {
                if constexpr (DUAL_INTERLEAVED_WEIGHT) {
                  wave_weight.qs[pack] =
                      pack_q4_k_t16_dual_u8x4(
                          tile, subblock, pack * 4, col, use_b);
                } else {
                  wave_weight.qs[pack] =
                      pack_q4_k_t16_u8x4(
                          tile, subblock, pack * 4, col);
                }
              }
            } else {
              if constexpr (DUAL_INTERLEAVED_WEIGHT) {
                wave_weight.qs[pack] =
                    pack_q4_k_t16_dual_u8x4(
                        tile, subblock, pack * 4, col, use_b);
              } else {
                wave_weight.qs[pack] =
                    pack_q4_k_t16_u8x4(
                        tile, subblock, pack * 4, col);
              }
            }
          } else {
            uint32_t packed0 = 0;
            uint32_t packed1 = 0;
            if ((lane & 1) == 0) {
              pack_q4_k_t16_u8x4_pair(
                  tile,
                  subblock,
                  pack * 4,
                  col >> 1,
                  packed0,
                  packed1);
            }
            const uint32_t high_col = __shfl(
                packed1, lane & ~1, 32);
            wave_weight.qs[pack] =
                (lane & 1) ? high_col : packed0;
          }
        }
        if constexpr (DUAL_INTERLEAVED_WEIGHT) {
          const int col_pair = col >> 1;
          const int matrix_offset = use_b ? 4 : 0;
          wave_weight.dm[0] =
              fp16_bytes_to_float(
                  tile + Q4_K_T16_DUAL_D_OFFSET +
                  col_pair * 8 + matrix_offset + (col & 1) * 2) *
              static_cast<float>(
                  tile[Q4_K_T16_DUAL_SCALE_OFFSET +
                       (subblock * (Q4_K_TILE16_COLS / 2) + col_pair) *
                           4 +
                       (use_b ? 2 : 0) + (col & 1)]);
          wave_weight.dm[1] =
              fp16_bytes_to_float(
                  tile + Q4_K_T16_DUAL_DMIN_OFFSET +
                  col_pair * 8 + matrix_offset + (col & 1) * 2) *
              static_cast<float>(
                  tile[Q4_K_T16_DUAL_MIN_OFFSET +
                       (subblock * (Q4_K_TILE16_COLS / 2) + col_pair) *
                           4 +
                       (use_b ? 2 : 0) + (col & 1)]);
        } else {
          wave_weight.dm[0] =
              fp16_bytes_to_float(
                  tile + Q4_K_T16_D_OFFSET + col * 2) *
              static_cast<float>(
                  tile[Q4_K_T16_SCALE_OFFSET +
                       subblock * Q4_K_TILE16_COLS + col]);
          wave_weight.dm[1] =
              fp16_bytes_to_float(
                  tile + Q4_K_T16_DMIN_OFFSET + col * 2) *
              static_cast<float>(
                  tile[Q4_K_T16_MIN_OFFSET +
                       subblock * Q4_K_TILE16_COLS + col]);
        }
        if constexpr (RAW_WEIGHT_PREFETCH_PACKS > 0) {
          if (k32 + 1 < k32_iters) {
            const int next_k32 = k32 + 1;
            const int next_weight_block = next_k32 >> 3;
            const int next_subblock = next_k32 & 7;
            const int64_t next_tile_index =
                (expert_id * out_tiles + out_tile) *
                    blocks_per_weight_row +
                next_weight_block;
            const uint8_t* __restrict__ next_tile =
                qweight +
                next_tile_index *
                    (DUAL_INTERLEAVED_WEIGHT
                         ? Q4_K_T16_DUAL_BLOCK_BYTES
                         : Q4_K_T16_BLOCK_BYTES);
#pragma unroll
            for (int pack = 0;
                 pack < RAW_WEIGHT_PREFETCH_PACKS;
                 ++pack) {
              if constexpr (DUAL_INTERLEAVED_WEIGHT) {
                prefetched_raw[pack] =
                    pack_q4_k_t16_dual_raw_u8x4(
                        next_tile,
                        next_subblock,
                        pack * 4,
                        col,
                        use_b);
              } else {
                prefetched_raw[pack] =
                    pack_q4_k_t16_raw_u8x4(
                        next_tile, next_subblock, pack * 4, col);
              }
            }
          }
        }
      }
    } else {
      for (int item = tid; item < (MMQ_M / 2) * 8; item += THREADS) {
        const int col_pair = item >> 3;
        const int pack = item & 7;
        const int col_offset = col_pair * 2;
        const int64_t local_col = local_col_base + col_offset;
        uint32_t packed0 = 0;
        uint32_t packed1 = 0;
        if (local_col + 1 < local_out_features) {
          const int64_t out_tile = local_col / Q4_K_TILE16_COLS;
          const int col = static_cast<int>(
              local_col % Q4_K_TILE16_COLS);
          const int64_t tile_index =
              (expert_id * out_tiles + out_tile) *
                  blocks_per_weight_row +
              weight_block;
          const uint8_t* __restrict__ tile =
              qweight + tile_index * Q4_K_T16_BLOCK_BYTES;
          pack_q4_k_t16_u8x4_pair(
              tile,
              subblock,
              pack * 4,
              col >> 1,
              packed0,
              packed1);
        }
        weight_tile[col_offset].qs[pack] = packed0;
        weight_tile[col_offset + 1].qs[pack] = packed1;
      }
      if (tid < MMQ_M) {
        const int col_offset = tid;
        const int64_t local_col = local_col_base + col_offset;
        float dscale = 0.0f;
        float dmin = 0.0f;
        if (local_col < local_out_features) {
          const int64_t out_tile = local_col / Q4_K_TILE16_COLS;
          const int col = static_cast<int>(
              local_col % Q4_K_TILE16_COLS);
          const int64_t tile_index =
              (expert_id * out_tiles + out_tile) *
                  blocks_per_weight_row +
              weight_block;
          const uint8_t* __restrict__ tile =
              qweight + tile_index * Q4_K_T16_BLOCK_BYTES;
          dscale =
              fp16_bytes_to_float(
                  tile + Q4_K_T16_D_OFFSET + col * 2) *
              static_cast<float>(
                  tile[Q4_K_T16_SCALE_OFFSET +
                       subblock * Q4_K_TILE16_COLS + col]);
          dmin =
              fp16_bytes_to_float(
                  tile + Q4_K_T16_DMIN_OFFSET + col * 2) *
              static_cast<float>(
                  tile[Q4_K_T16_MIN_OFFSET +
                       subblock * Q4_K_TILE16_COLS + col]);
        }
        weight_tile[col_offset].dm[0] = dscale;
        weight_tile[col_offset].dm[1] = dmin;
      }
    }

    if constexpr (ROWVEC_ACTIVATION_STAGE) {
      if (tid < MMQ_N) {
        const int row_offset = tid;
        const bool valid_row = row_offset < valid_rows;
        const int64_t compact_row =
            valid_row
                ? compact_base + tile_row_in_expert + row_offset
                : 0;
        const int64_t source_row =
            valid_row && compact_row < compact_rows
                ? (IDENTITY_SOURCE ? compact_row
                                   : compact_to_source[compact_row])
                : 0;
#pragma unroll
        for (int pass = 0; pass < ACTIVATION_PASSES; ++pass) {
          float d = 0.0f;
          float s = 0.0f;
          int sum0 = 0;
          int sum1 = 0;
          mmq_i32x4_t packed0 = {0, 0, 0, 0};
          mmq_i32x4_t packed1 = {0, 0, 0, 0};
          if (valid_row && source_row >= 0 &&
              source_row < source_rows) {
            const int64_t q8_index =
                static_cast<int64_t>(pass) * plane_blocks +
                source_row * q8_blocks_per_row + (k32 >> 2);
            const block_q8_1_mmq_ds4_f32& xb = x_q8[q8_index];
            d = xb.ds4[2 * (k32 & 3)];
            s = xb.ds4[2 * (k32 & 3) + 1];
            const int8_t* __restrict__ qs =
                xb.qs + (k32 & 3) * Q8_1_BLOCK;
            packed0 =
                *reinterpret_cast<const mmq_i32x4_t*>(qs);
            packed1 =
                *reinterpret_cast<const mmq_i32x4_t*>(qs + 16);
            if constexpr (PRECOMPUTED_ACTIVATION_SUMS) {
              const int64_t sum_base =
                  source_row * half_blocks_per_row +
                  static_cast<int64_t>(k32) * 2;
              sum0 = static_cast<int>(
                  precomputed_activation_sums[sum_base]);
              sum1 = static_cast<int>(
                  precomputed_activation_sums[sum_base + 1]);
            }
          }
#pragma unroll
          for (int pack = 0; pack < 4; ++pack) {
            const int32_t packed = packed0[pack];
            activation_tile[activation_slot][pass][row_offset].qs[pack] =
                packed;
            if constexpr (
                SPLIT16 && !PRECOMPUTED_ACTIVATION_SUMS) {
              sum0 = gguf_q4_k_mmq_dp4a(
                  0x01010101u,
                  static_cast<uint32_t>(packed),
                  sum0);
            }
          }
#pragma unroll
          for (int pack = 0; pack < 4; ++pack) {
            const int32_t packed = packed1[pack];
            activation_tile[activation_slot][pass][row_offset]
                           .qs[pack + 4] = packed;
            if constexpr (
                SPLIT16 && !PRECOMPUTED_ACTIVATION_SUMS) {
              sum1 = gguf_q4_k_mmq_dp4a(
                  0x01010101u,
                  static_cast<uint32_t>(packed),
                  sum1);
            }
          }
          activation_tile[activation_slot][pass][row_offset].d = d;
          activation_tile[activation_slot][pass][row_offset].s = s;
          activation_tile[activation_slot][pass][row_offset].sums[0] =
              sum0;
          activation_tile[activation_slot][pass][row_offset].sums[1] =
              sum1;
        }
      }
    } else {
      for (int item = tid; item < MMQ_N * 8; item += THREADS) {
        const int row_offset = item >> 3;
        const int pack = item & 7;
        const bool valid_row = row_offset < valid_rows;
        const int64_t compact_row =
            valid_row
                ? compact_base + tile_row_in_expert + row_offset
                : 0;
        const int64_t source_row =
            valid_row && compact_row < compact_rows
                ? (IDENTITY_SOURCE ? compact_row
                                   : compact_to_source[compact_row])
                : 0;
#pragma unroll
        for (int pass = 0; pass < ACTIVATION_PASSES; ++pass) {
          int32_t packed = 0;
          if (valid_row && source_row >= 0 &&
              source_row < source_rows) {
            const int64_t q8_index =
                static_cast<int64_t>(pass) * plane_blocks +
                source_row * q8_blocks_per_row + (k32 >> 2);
            const block_q8_1_mmq_ds4_f32& xb = x_q8[q8_index];
            const int8_t* __restrict__ qs =
                xb.qs + (k32 & 3) * Q8_1_BLOCK + pack * 4;
            packed = pack_i8x4(qs);
          }
          activation_tile[activation_slot][pass][row_offset].qs[pack] =
              packed;
        }
      }
      if (tid < MMQ_N) {
        const int row_offset = tid;
        const bool valid_row = row_offset < valid_rows;
        const int64_t compact_row =
            valid_row
                ? compact_base + tile_row_in_expert + row_offset
                : 0;
        const int64_t source_row =
            valid_row && compact_row < compact_rows
                ? (IDENTITY_SOURCE ? compact_row
                                   : compact_to_source[compact_row])
                : 0;
#pragma unroll
        for (int pass = 0; pass < ACTIVATION_PASSES; ++pass) {
          float d = 0.0f;
          float s = 0.0f;
          int sum0 = 0;
          int sum1 = 0;
          if (valid_row && source_row >= 0 &&
              source_row < source_rows) {
            const int64_t q8_index =
                static_cast<int64_t>(pass) * plane_blocks +
                source_row * q8_blocks_per_row + (k32 >> 2);
            const block_q8_1_mmq_ds4_f32& xb = x_q8[q8_index];
            d = xb.ds4[2 * (k32 & 3)];
            s = xb.ds4[2 * (k32 & 3) + 1];
            if constexpr (SPLIT16) {
              const int8_t* __restrict__ qs =
                  xb.qs + (k32 & 3) * Q8_1_BLOCK;
#pragma unroll
              for (int pack = 0; pack < 4; ++pack) {
                sum0 = gguf_q4_k_mmq_dp4a(
                    0x01010101u,
                    static_cast<uint32_t>(
                        pack_i8x4(qs + pack * 4)),
                    sum0);
              }
#pragma unroll
              for (int pack = 4; pack < 8; ++pack) {
                sum1 = gguf_q4_k_mmq_dp4a(
                    0x01010101u,
                    static_cast<uint32_t>(
                        pack_i8x4(qs + pack * 4)),
                    sum1);
              }
            }
          }
          activation_tile[activation_slot][pass][row_offset].d = d;
          activation_tile[activation_slot][pass][row_offset].s = s;
          activation_tile[activation_slot][pass][row_offset].sums[0] =
              sum0;
          activation_tile[activation_slot][pass][row_offset].sums[1] =
              sum1;
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int col_iter = 0; col_iter < COL_ITERS; ++col_iter) {
      const int col_offset =
          WAVE_COLUMNS ? wave * 32 + lane : lane + col_iter * 32;
      const q4_k_mmq64_f32_cache* wb_ptr = nullptr;
      if constexpr (WAVE_COLUMNS) {
        wb_ptr = &wave_weight;
      } else {
        wb_ptr = &weight_tile[col_offset];
      }
      const q4_k_mmq64_f32_cache& wb = *wb_ptr;
#pragma unroll
      for (int row_iter = 0; row_iter < ROW_ITERS; ++row_iter) {
        const int row_offset =
            WAVE_COLUMNS ? row_iter : wave * WNITER + row_iter;
        if (row_offset >= valid_rows) continue;
#pragma unroll
        for (int pass = 0; pass < ACTIVATION_PASSES; ++pass) {
          const q8_1_mmq32_f32_cache& xb =
              activation_tile[activation_slot][pass][row_offset];
          if constexpr (SPLIT16) {
            int dot0 = 0;
            int dot1 = 0;
#pragma unroll
            for (int pack = 0; pack < 4; ++pack) {
              dot0 = gguf_q4_k_mmq_dp4a(
                  wb.qs[pack],
                  static_cast<uint32_t>(xb.qs[pack]),
                  dot0);
            }
#pragma unroll
            for (int pack = 4; pack < 8; ++pack) {
              dot1 = gguf_q4_k_mmq_dp4a(
                  wb.qs[pack],
                  static_cast<uint32_t>(xb.qs[pack]),
                  dot1);
            }
            acc[col_iter][row_iter] +=
                wb.dm[0] *
                    (xb.d * static_cast<float>(dot0) +
                     xb.s * static_cast<float>(dot1)) -
                wb.dm[1] *
                    (xb.d * static_cast<float>(xb.sums[0]) +
                     xb.s * static_cast<float>(xb.sums[1]));
          } else {
            int dot = 0;
#pragma unroll
            for (int pack = 0; pack < 8; ++pack) {
              dot = gguf_q4_k_mmq_dp4a(
                  wb.qs[pack],
                  static_cast<uint32_t>(xb.qs[pack]),
                  dot);
            }
            acc[col_iter][row_iter] +=
                xb.d * wb.dm[0] * static_cast<float>(dot) -
                wb.dm[1] * xb.s;
          }
        }
      }
    }
    if constexpr (!DOUBLE_BUFFER_ACTIVATION) {
      __syncthreads();
    }
  }

#pragma unroll
  for (int col_iter = 0; col_iter < COL_ITERS; ++col_iter) {
    const int64_t global_col =
        global_col_base +
        (WAVE_COLUMNS ? wave * 32 + lane
                      : lane + col_iter * 32);
    if (global_col >= out_features_total) continue;
#pragma unroll
    for (int row_iter = 0; row_iter < ROW_ITERS; ++row_iter) {
      const int row_offset =
          WAVE_COLUMNS ? row_iter : wave * WNITER + row_iter;
      if (row_offset < valid_rows) {
        const int64_t compact_row =
            compact_base + tile_row_in_expert + row_offset;
        if (compact_row < compact_rows) {
          out[compact_row * out_features_total + global_col] =
              float_to_bf16_bits(acc[col_iter][row_iter]);
        }
      }
    }
  }
}

__device__ inline void pack_q6_k_t16_qmicro_u8x4_quartet_permute(
    const uint8_t* __restrict__ tile,
    int subblock,
    int pack,
    int col_quartet,
    uint32_t (&cols)[4]) {
  const int record_index =
      (subblock * (Q4_K_TILE16_COLS / 4) + col_quartet) * 8 + pack;
  const uint8_t* __restrict__ record =
      tile + Q6_K_T16_QMICRO_OFFSET +
      record_index * Q6_K_T16_QMICRO_RECORD_BYTES;
  const uint32_t ql_lo =
      reinterpret_cast<const uint32_t*>(record)[0];
  const uint32_t ql_hi =
      reinterpret_cast<const uint32_t*>(record)[1];
  const uint32_t qh =
      reinterpret_cast<const uint32_t*>(record)[2];
  const uint32_t ql01 = static_cast<uint32_t>(
      __builtin_amdgcn_perm(
          static_cast<int>(ql_hi),
          static_cast<int>(ql_lo),
          0x06040200));
  const uint32_t ql23 = static_cast<uint32_t>(
      __builtin_amdgcn_perm(
          static_cast<int>(ql_hi),
          static_cast<int>(ql_lo),
          0x07050301));
  constexpr uint32_t LOW_MASK = 0x0F0F0F0Fu;
  cols[0] =
      (ql01 & LOW_MASK) | ((qh & 0x03030303u) << 4);
  cols[1] =
      ((ql01 >> 4) & LOW_MASK) | ((qh & 0x0C0C0C0Cu) << 2);
  cols[2] =
      (ql23 & LOW_MASK) | (qh & 0x30303030u);
  cols[3] =
      ((ql23 >> 4) & LOW_MASK) | ((qh & 0xC0C0C0C0u) >> 2);
}

__device__ inline void pack_q6_k_t16_qmicro_planar_u8x4_quartet(
    const uint8_t* __restrict__ tile,
    int subblock,
    int pack,
    int col_quartet,
    uint32_t (&cols)[4]) {
  const int record_index =
      (subblock * (Q4_K_TILE16_COLS / 4) + col_quartet) * 8 + pack;
  const uint8_t* __restrict__ record =
      tile + Q6_K_T16_QMICRO_OFFSET +
      record_index * Q6_K_T16_QMICRO_RECORD_BYTES;
  const uint32_t ql01 =
      reinterpret_cast<const uint32_t*>(record)[0];
  const uint32_t ql23 =
      reinterpret_cast<const uint32_t*>(record)[1];
  const uint32_t qh =
      reinterpret_cast<const uint32_t*>(record)[2];
  constexpr uint32_t LOW_MASK = 0x0F0F0F0Fu;
  cols[0] =
      (ql01 & LOW_MASK) | ((qh & 0x03030303u) << 4);
  cols[1] =
      ((ql01 >> 4) & LOW_MASK) | ((qh & 0x0C0C0C0Cu) << 2);
  cols[2] =
      (ql23 & LOW_MASK) | (qh & 0x30303030u);
  cols[3] =
      ((ql23 >> 4) & LOW_MASK) | ((qh & 0xC0C0C0C0u) >> 2);
}

__device__ inline void pack_q6_k_t16_qmicro_planar_words_u8x4_quartet(
    uint32_t ql01,
    uint32_t ql23,
    uint32_t qh,
    uint32_t (&cols)[4]) {
  constexpr uint32_t LOW_MASK = 0x0F0F0F0Fu;
  cols[0] =
      (ql01 & LOW_MASK) | ((qh & 0x03030303u) << 4);
  cols[1] =
      ((ql01 >> 4) & LOW_MASK) | ((qh & 0x0C0C0C0Cu) << 2);
  cols[2] =
      (ql23 & LOW_MASK) | (qh & 0x30303030u);
  cols[3] =
      ((ql23 >> 4) & LOW_MASK) | ((qh & 0xC0C0C0C0u) >> 2);
}

template <
    int ACTIVATION_PASSES,
    bool IDENTITY_SOURCE = false,
    bool SPLIT16 = false,
    int MMQ_M = 64,
    bool ROWVEC_ACTIVATION_STAGE = false,
    bool WAVE_COLUMNS = false,
    int THREADS = 128,
    bool DIRECT_WAVE_DECODE = false,
    bool DOUBLE_BUFFER_ACTIVATION = false,
    int RAW_WEIGHT_PREFETCH_PACKS = 0,
    bool PRECOMPUTED_ACTIVATION_SUMS = false,
    bool DUAL_INTERLEAVED_WEIGHT = false>
int launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32(
    const block_q8_1_mmq_ds4_f32* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream,
    const int16_t* precomputed_activation_sums = nullptr) {
  if (x_q8 == nullptr ||
      (!IDENTITY_SOURCE && compact_to_source == nullptr) ||
      expert_start_compact == nullptr ||
      expert_start_mmq32 == nullptr || tile_expert == nullptr ||
      qweight_a == nullptr ||
      (!DUAL_INTERLEAVED_WEIGHT && qweight_b == nullptr) ||
      out == nullptr ||
      (PRECOMPUTED_ACTIVATION_SUMS &&
       precomputed_activation_sums == nullptr) ||
      compact_rows <= 0 || source_rows <= 0 || in_features <= 0 ||
      out_features_a <= 0 || out_features_b < 0 || num_experts <= 0 ||
      mmq_total_rows <= 0 || (in_features % QK_K) != 0 ||
      (out_features_a % MMQ_M) != 0 ||
      (out_features_b % MMQ_M) != 0 ||
      (mmq_total_rows % 32) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(THREADS);
  const dim3 grid(
      static_cast<unsigned int>(
          (out_features_a + out_features_b) / MMQ_M),
      static_cast<unsigned int>(mmq_total_rows / 32));
  hipLaunchKernelGGL(
      (gguf_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32_prefill_compact32_kernel<
          ACTIVATION_PASSES,
          IDENTITY_SOURCE,
          SPLIT16,
          MMQ_M,
          ROWVEC_ACTIVATION_STAGE,
          WAVE_COLUMNS,
          THREADS,
          DIRECT_WAVE_DECODE,
          DOUBLE_BUFFER_ACTIVATION,
          RAW_WEIGHT_PREFETCH_PACKS,
          PRECOMPUTED_ACTIVATION_SUMS,
          DUAL_INTERLEAVED_WEIGHT>),
      grid,
      block,
      0,
      stream,
      x_q8,
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      precomputed_activation_sums);
  return static_cast<int>(hipGetLastError());
}

template <
    int ACTIVATION_PASSES,
    bool ROWVEC_ACTIVATION_STAGE = false,
    bool WAVE_COLUMNS = false,
    int THREADS = 128,
    int MMQ_N = 32,
    bool QMICRO = false,
    bool COMPACT_ACTIVATION_CACHE = false,
    bool HALF_ROW_ACTIVATION_STAGE = false,
    bool SKIP_PADDED_ACTIVATION_STAGE = false,
    bool QMICRO_PERMUTE = false,
    bool QMICRO_PLANAR = false,
    bool INTEGER_WMMA = false,
    bool WMMA_HOIST_ACTIVATION = false,
    bool WMMA_PREFETCH_WEIGHT = false,
    bool WMMA_PREFETCH_ACTIVATION = false,
    bool PRECOMPUTED_ACTIVATION_SUMS = false,
    bool DENSE_LINEAR = false>
__launch_bounds__(THREADS, 1)
__global__ void
gguf_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32_prefill_compact32_kernel(
    const block_q8_1_mmq_ds4_f32* __restrict__ x_q8,
    const int64_t* __restrict__ expert_start_compact,
    const int64_t* __restrict__ expert_start_mmq32,
    const int64_t* __restrict__ tile_expert,
    const uint8_t* __restrict__ qweight,
    uint16_t* __restrict__ out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows) {
  constexpr int MMQ_M = 64;
  constexpr int MMQ_K = 32;
  constexpr int WNITER = MMQ_N / (THREADS / 32);
  constexpr int COL_ITERS = WAVE_COLUMNS ? 1 : 2;
  constexpr int ROW_ITERS = WAVE_COLUMNS ? MMQ_N : WNITER;
  typedef int mmq_i32x4_t
      __attribute__((__vector_size__(4 * sizeof(int))));
  typedef int mmq_i32x8_t
      __attribute__((__vector_size__(8 * sizeof(int))));
  using q6_activation_cache_t =
      typename q6_activation_cache_selector<
          COMPACT_ACTIVATION_CACHE>::type;
  static_assert(ACTIVATION_PASSES > 0, "at least one activation plane required");
  static_assert(MMQ_N == 32 || MMQ_N == 64, "unsupported row tile");
  static_assert(THREADS % 32 == 0, "thread count must contain whole waves");
  static_assert(MMQ_N % (THREADS / 32) == 0, "row tile must divide across waves");
  static_assert(
      !HALF_ROW_ACTIVATION_STAGE ||
          (ROWVEC_ACTIVATION_STAGE && COMPACT_ACTIVATION_CACHE &&
           QMICRO && MMQ_N == 64 && THREADS == 128),
      "half-row activation staging requires production Q6 qmicro geometry");
  static_assert(
      !SKIP_PADDED_ACTIVATION_STAGE || HALF_ROW_ACTIVATION_STAGE,
      "skipping padded activation rows requires half-row staging");
  static_assert(
      !QMICRO_PERMUTE || QMICRO,
      "qmicro permute decode requires qmicro layout");
  static_assert(
      !QMICRO_PLANAR || QMICRO,
      "qmicro planar decode requires qmicro layout");
  static_assert(
      !(QMICRO_PLANAR && QMICRO_PERMUTE),
      "qmicro planar and permute decode are mutually exclusive");
  static_assert(
      !INTEGER_WMMA ||
          (ACTIVATION_PASSES == 1 && ROWVEC_ACTIVATION_STAGE &&
           !WAVE_COLUMNS && THREADS == 128 && MMQ_N == 64 &&
           QMICRO && COMPACT_ACTIVATION_CACHE &&
           HALF_ROW_ACTIVATION_STAGE &&
           SKIP_PADDED_ACTIVATION_STAGE && !QMICRO_PERMUTE &&
           QMICRO_PLANAR),
      "integer WMMA requires production Q6 planar row64 geometry");
  static_assert(
      !WMMA_HOIST_ACTIVATION || INTEGER_WMMA,
      "WMMA activation hoisting requires integer WMMA");
  static_assert(
      !WMMA_PREFETCH_WEIGHT ||
          (INTEGER_WMMA && WMMA_HOIST_ACTIVATION &&
           QMICRO_PLANAR && THREADS == 128 && MMQ_N == 64),
      "WMMA weight prefetch requires production planar-Q6 geometry");
  static_assert(
      !WMMA_PREFETCH_ACTIVATION ||
          (WMMA_PREFETCH_WEIGHT && HALF_ROW_ACTIVATION_STAGE &&
           ACTIVATION_PASSES == 1),
      "WMMA activation prefetch requires the production weight-prefetch path");
  static_assert(
      !PRECOMPUTED_ACTIVATION_SUMS ||
          (COMPACT_ACTIVATION_CACHE && HALF_ROW_ACTIVATION_STAGE &&
           ACTIVATION_PASSES == 1),
      "precomputed activation sums require the compact half-row D4 path");

  const int tid = static_cast<int>(threadIdx.x);
  const int tile_col = static_cast<int>(blockIdx.x);
  const int tile_row = static_cast<int>(blockIdx.y);
  const int64_t expert_id = DENSE_LINEAR ? 0 : tile_expert[tile_row];
  if (!DENSE_LINEAR && (expert_id < 0 || expert_id >= num_experts)) return;

  const int64_t global_col_base = static_cast<int64_t>(tile_col) * MMQ_M;
  if (global_col_base >= out_features) return;
  const int64_t compact_base =
      DENSE_LINEAR ? 0 : expert_start_compact[expert_id];
  const int64_t expert_count =
      DENSE_LINEAR
          ? compact_rows
          : expert_start_compact[expert_id + 1] - compact_base;
  const int64_t padded_base =
      DENSE_LINEAR ? 0 : expert_start_mmq32[expert_id];
  const int64_t tile_row_in_expert =
      static_cast<int64_t>(tile_row) * MMQ_N - padded_base;
  const int64_t remaining_rows = expert_count - tile_row_in_expert;
  const int valid_rows =
      remaining_rows <= 0
          ? 0
          : (remaining_rows >= MMQ_N ? MMQ_N
                                     : static_cast<int>(remaining_rows));

  const int64_t blocks_per_weight_row = in_features / QK_K;
  const int64_t q8_blocks_per_row = in_features / Q8_1_MMQ_BLOCK;
  const int64_t out_tiles = out_features / Q4_K_TILE16_COLS;
  const int64_t plane_blocks = compact_rows * q8_blocks_per_row;
  const int wave = tid >> 5;
  const int lane = tid & 31;

  float acc[COL_ITERS][ROW_ITERS];
#pragma unroll
  for (int col_iter = 0; col_iter < COL_ITERS; ++col_iter) {
#pragma unroll
    for (int row_iter = 0; row_iter < ROW_ITERS; ++row_iter) {
      acc[col_iter][row_iter] = 0.0f;
    }
  }

  __shared__ q6_k_mmq64_cache
      weight_tile[WAVE_COLUMNS ? 1 : MMQ_M];
  __shared__ q6_activation_cache_t
      activation_tile[ACTIVATION_PASSES][MMQ_N];

  uint32_t prefetched_qmicro[3] = {0, 0, 0};
  uint16_t prefetched_d_bits = 0;
  int8_t prefetched_scale = 0;
  mmq_i32x4_t prefetched_activation_qs = {0, 0, 0, 0};
  float prefetched_activation_d = 0.0f;
  uint32_t prefetched_activation_sums = 0;

  const int k32_iters = static_cast<int>(in_features / MMQ_K);
  for (int k32 = 0; k32 < k32_iters; ++k32) {
    const int weight_block = k32 >> 3;
    const int subblock = k32 & 7;

    q6_k_mmq64_cache wave_weight{};
    if constexpr (WAVE_COLUMNS) {
      const int col_offset = wave * 32 + lane;
      const int64_t local_col = global_col_base + col_offset;
      if (local_col < out_features) {
        const int64_t out_tile = local_col / Q4_K_TILE16_COLS;
        const int col = static_cast<int>(
            local_col % Q4_K_TILE16_COLS);
        const int64_t tile_index =
            (expert_id * out_tiles + out_tile) *
                blocks_per_weight_row +
            weight_block;
        const uint8_t* __restrict__ tile =
            qweight + tile_index * Q6_K_T16_BLOCK_BYTES;
        const int quartet_leader = lane & ~3;
#pragma unroll
        for (int pack = 0; pack < 8; ++pack) {
          uint32_t packed[4] = {0, 0, 0, 0};
          if ((lane & 3) == 0) {
            if constexpr (QMICRO) {
              pack_q6_k_t16_qmicro_u8x4_quartet(
                  tile,
                  subblock,
                  pack,
                  col >> 2,
                  packed);
            } else {
              pack_q6_k_t16_u8x4_quartet(
                  tile,
                  subblock,
                  pack * 4,
                  col >> 2,
                  packed);
            }
          }
          const uint32_t packed0 =
              __shfl(packed[0], quartet_leader, 32);
          const uint32_t packed1 =
              __shfl(packed[1], quartet_leader, 32);
          const uint32_t packed2 =
              __shfl(packed[2], quartet_leader, 32);
          const uint32_t packed3 =
              __shfl(packed[3], quartet_leader, 32);
          switch (lane & 3) {
            case 0:
              wave_weight.qs[pack] = packed0;
              break;
            case 1:
              wave_weight.qs[pack] = packed1;
              break;
            case 2:
              wave_weight.qs[pack] = packed2;
              break;
            default:
              wave_weight.qs[pack] = packed3;
              break;
          }
        }
        const float d = fp16_bytes_to_float(
            tile + Q6_K_T16_D_OFFSET + col * 2);
        const int8_t* __restrict__ scales =
            reinterpret_cast<const int8_t*>(
                tile + Q6_K_T16_SCALE_OFFSET);
        wave_weight.scales[0] =
            d * static_cast<float>(
                    scales[(subblock * 2) *
                               Q4_K_TILE16_COLS +
                           col]);
        wave_weight.scales[1] =
            d * static_cast<float>(
                    scales[(subblock * 2 + 1) *
                               Q4_K_TILE16_COLS +
                           col]);
      }
    } else {
      for (int item = tid; item < (MMQ_M / 4) * 8; item += THREADS) {
        const int col_quartet = item >> 3;
        const int pack = item & 7;
        const int col_offset = col_quartet * 4;
        const int64_t local_col = global_col_base + col_offset;
        uint32_t packed[4] = {0, 0, 0, 0};
        if (local_col + 3 < out_features) {
          const int64_t out_tile = local_col / Q4_K_TILE16_COLS;
          const int col = static_cast<int>(
              local_col % Q4_K_TILE16_COLS);
          const int64_t tile_index =
              (expert_id * out_tiles + out_tile) *
                  blocks_per_weight_row +
              weight_block;
          const uint8_t* __restrict__ tile =
              qweight + tile_index * Q6_K_T16_BLOCK_BYTES;
          if constexpr (WMMA_PREFETCH_WEIGHT) {
            if (k32 > 0) {
              pack_q6_k_t16_qmicro_planar_words_u8x4_quartet(
                  prefetched_qmicro[0],
                  prefetched_qmicro[1],
                  prefetched_qmicro[2],
                  packed);
            } else {
              pack_q6_k_t16_qmicro_planar_u8x4_quartet(
                  tile,
                  subblock,
                  pack,
                  col >> 2,
                  packed);
            }
          } else if constexpr (QMICRO_PLANAR) {
            pack_q6_k_t16_qmicro_planar_u8x4_quartet(
                tile,
                subblock,
                pack,
                col >> 2,
                packed);
          } else if constexpr (QMICRO_PERMUTE) {
            pack_q6_k_t16_qmicro_u8x4_quartet_permute(
                tile,
                subblock,
                pack,
                col >> 2,
                packed);
          } else if constexpr (QMICRO) {
            pack_q6_k_t16_qmicro_u8x4_quartet(
                tile,
                subblock,
                pack,
                col >> 2,
                packed);
          } else {
            pack_q6_k_t16_u8x4_quartet(
                tile,
                subblock,
                pack * 4,
                col >> 2,
                packed);
          }
        }
        weight_tile[col_offset].qs[pack] = packed[0];
        weight_tile[col_offset + 1].qs[pack] = packed[1];
        weight_tile[col_offset + 2].qs[pack] = packed[2];
        weight_tile[col_offset + 3].qs[pack] = packed[3];
      }
      if (tid < MMQ_M * 2) {
        const int col_offset = tid >> 1;
        const int scale_half = tid & 1;
        const int64_t local_col = global_col_base + col_offset;
        float scale = 0.0f;
        if (local_col < out_features) {
          const int64_t out_tile = local_col / Q4_K_TILE16_COLS;
          const int col = static_cast<int>(
              local_col % Q4_K_TILE16_COLS);
          const int64_t tile_index =
              (expert_id * out_tiles + out_tile) *
                  blocks_per_weight_row +
              weight_block;
          const uint8_t* __restrict__ tile =
              qweight + tile_index * Q6_K_T16_BLOCK_BYTES;
          if constexpr (WMMA_PREFETCH_WEIGHT) {
            if (k32 > 0) {
              scale =
                  fp16_bits_to_float(prefetched_d_bits) *
                  static_cast<float>(prefetched_scale);
            } else {
              const float d = fp16_bytes_to_float(
                  tile + Q6_K_T16_D_OFFSET + col * 2);
              const int8_t* __restrict__ scales =
                  reinterpret_cast<const int8_t*>(
                      tile + Q6_K_T16_SCALE_OFFSET);
              scale =
                  d * static_cast<float>(
                          scales[(subblock * 2 + scale_half) *
                                     Q4_K_TILE16_COLS +
                                 col]);
            }
          } else {
            const float d = fp16_bytes_to_float(
                tile + Q6_K_T16_D_OFFSET + col * 2);
            const int8_t* __restrict__ scales =
                reinterpret_cast<const int8_t*>(
                    tile + Q6_K_T16_SCALE_OFFSET);
            scale =
                d * static_cast<float>(
                        scales[(subblock * 2 + scale_half) *
                                   Q4_K_TILE16_COLS +
                               col]);
          }
        }
        weight_tile[col_offset].scales[scale_half] = scale;
      }
    }

    if constexpr (HALF_ROW_ACTIVATION_STAGE) {
      const int row_offset = tid >> 1;
      const int half = tid & 1;
      const bool valid_row = row_offset < valid_rows;
      const int64_t compact_row =
          valid_row
              ? compact_base + tile_row_in_expert + row_offset
              : 0;
      const bool stage_row =
          !SKIP_PADDED_ACTIVATION_STAGE ||
          (valid_row && compact_row < compact_rows);
      if (stage_row) {
#pragma unroll
        for (int pass = 0; pass < ACTIVATION_PASSES; ++pass) {
          float d = 0.0f;
          int sum = 0;
          uint32_t packed_sums = 0;
          mmq_i32x4_t packed = {0, 0, 0, 0};
          if (valid_row && compact_row < compact_rows) {
            if constexpr (WMMA_PREFETCH_ACTIVATION) {
              if (k32 > 0) {
                packed = prefetched_activation_qs;
                if (half == 0) {
                  d = prefetched_activation_d;
                }
                if constexpr (PRECOMPUTED_ACTIVATION_SUMS) {
                  packed_sums = prefetched_activation_sums;
                }
              } else {
                const int64_t q8_index =
                    static_cast<int64_t>(pass) * plane_blocks +
                    compact_row * q8_blocks_per_row + (k32 >> 2);
                const block_q8_1_mmq_ds4_f32& xb = x_q8[q8_index];
                if (half == 0) {
                  d = xb.ds4[2 * (k32 & 3)];
                }
                if constexpr (PRECOMPUTED_ACTIVATION_SUMS) {
                  packed_sums =
                      reinterpret_cast<const uint32_t*>(xb.ds4)
                          [2 * (k32 & 3) + 1];
                }
                const int8_t* __restrict__ qs =
                    xb.qs + (k32 & 3) * Q8_1_BLOCK + half * 16;
                packed = *reinterpret_cast<const mmq_i32x4_t*>(qs);
              }
            } else {
              const int64_t q8_index =
                  static_cast<int64_t>(pass) * plane_blocks +
                  compact_row * q8_blocks_per_row + (k32 >> 2);
              const block_q8_1_mmq_ds4_f32& xb = x_q8[q8_index];
              if (half == 0) {
                d = xb.ds4[2 * (k32 & 3)];
              }
              if constexpr (PRECOMPUTED_ACTIVATION_SUMS) {
                packed_sums =
                    reinterpret_cast<const uint32_t*>(xb.ds4)
                        [2 * (k32 & 3) + 1];
              }
              const int8_t* __restrict__ qs =
                  xb.qs + (k32 & 3) * Q8_1_BLOCK + half * 16;
              packed = *reinterpret_cast<const mmq_i32x4_t*>(qs);
            }
          }
#pragma unroll
          for (int pack = 0; pack < 4; ++pack) {
            const int32_t packed_qs = packed[pack];
            activation_tile[pass][row_offset]
                           .qs[half * 4 + pack] = packed_qs;
            if constexpr (!PRECOMPUTED_ACTIVATION_SUMS) {
              sum = gguf_q4_k_mmq_dp4a(
                  0x01010101u,
                  static_cast<uint32_t>(packed_qs),
                  sum);
            }
          }
          if constexpr (PRECOMPUTED_ACTIVATION_SUMS) {
            sum = static_cast<int16_t>(
                packed_sums >> (half * 16));
          }
          activation_tile[pass][row_offset].sums[half] =
              static_cast<int16_t>(sum);
          if (half == 0) {
            activation_tile[pass][row_offset].d = d;
          }
        }
      }
    } else if constexpr (ROWVEC_ACTIVATION_STAGE) {
      if (tid < MMQ_N) {
        const int row_offset = tid;
        const bool valid_row = row_offset < valid_rows;
        const int64_t compact_row =
            valid_row
                ? compact_base + tile_row_in_expert + row_offset
                : 0;
#pragma unroll
        for (int pass = 0; pass < ACTIVATION_PASSES; ++pass) {
          float d = 0.0f;
          float s = 0.0f;
          int sum0 = 0;
          int sum1 = 0;
          mmq_i32x4_t packed0 = {0, 0, 0, 0};
          mmq_i32x4_t packed1 = {0, 0, 0, 0};
          if (valid_row && compact_row < compact_rows) {
            const int64_t q8_index =
                static_cast<int64_t>(pass) * plane_blocks +
                compact_row * q8_blocks_per_row + (k32 >> 2);
            const block_q8_1_mmq_ds4_f32& xb = x_q8[q8_index];
            d = xb.ds4[2 * (k32 & 3)];
            if constexpr (!COMPACT_ACTIVATION_CACHE) {
              s = xb.ds4[2 * (k32 & 3) + 1];
            }
            const int8_t* __restrict__ qs =
                xb.qs + (k32 & 3) * Q8_1_BLOCK;
            packed0 =
                *reinterpret_cast<const mmq_i32x4_t*>(qs);
            packed1 =
                *reinterpret_cast<const mmq_i32x4_t*>(qs + 16);
          }
#pragma unroll
          for (int pack = 0; pack < 4; ++pack) {
            const int32_t packed = packed0[pack];
            activation_tile[pass][row_offset].qs[pack] = packed;
            sum0 = gguf_q4_k_mmq_dp4a(
                0x01010101u,
                static_cast<uint32_t>(packed),
                sum0);
          }
#pragma unroll
          for (int pack = 0; pack < 4; ++pack) {
            const int32_t packed = packed1[pack];
            activation_tile[pass][row_offset].qs[pack + 4] = packed;
            sum1 = gguf_q4_k_mmq_dp4a(
                0x01010101u,
                static_cast<uint32_t>(packed),
                sum1);
          }
          activation_tile[pass][row_offset].d = d;
          if constexpr (COMPACT_ACTIVATION_CACHE) {
            activation_tile[pass][row_offset].sums[0] =
                static_cast<int16_t>(sum0);
            activation_tile[pass][row_offset].sums[1] =
                static_cast<int16_t>(sum1);
          } else {
            activation_tile[pass][row_offset].s = s;
            activation_tile[pass][row_offset].sums[0] = sum0;
            activation_tile[pass][row_offset].sums[1] = sum1;
          }
        }
      }
    } else {
      for (int item = tid; item < MMQ_N * 8; item += THREADS) {
        const int row_offset = item >> 3;
        const int pack = item & 7;
        const bool valid_row = row_offset < valid_rows;
        const int64_t compact_row =
            valid_row
                ? compact_base + tile_row_in_expert + row_offset
                : 0;
#pragma unroll
        for (int pass = 0; pass < ACTIVATION_PASSES; ++pass) {
          int32_t packed = 0;
          if (valid_row && compact_row < compact_rows) {
            const int64_t q8_index =
                static_cast<int64_t>(pass) * plane_blocks +
                compact_row * q8_blocks_per_row + (k32 >> 2);
            const block_q8_1_mmq_ds4_f32& xb = x_q8[q8_index];
            const int8_t* __restrict__ qs =
                xb.qs + (k32 & 3) * Q8_1_BLOCK + pack * 4;
            packed = pack_i8x4(qs);
          }
          activation_tile[pass][row_offset].qs[pack] = packed;
        }
      }
      if (tid < MMQ_N) {
        const int row_offset = tid;
        const bool valid_row = row_offset < valid_rows;
        const int64_t compact_row =
            valid_row
                ? compact_base + tile_row_in_expert + row_offset
                : 0;
#pragma unroll
        for (int pass = 0; pass < ACTIVATION_PASSES; ++pass) {
          float d = 0.0f;
          float s = 0.0f;
          int sum0 = 0;
          int sum1 = 0;
          if (valid_row && compact_row < compact_rows) {
            const int64_t q8_index =
                static_cast<int64_t>(pass) * plane_blocks +
                compact_row * q8_blocks_per_row + (k32 >> 2);
            const block_q8_1_mmq_ds4_f32& xb = x_q8[q8_index];
            const int8_t* __restrict__ qs =
                xb.qs + (k32 & 3) * Q8_1_BLOCK;
            d = xb.ds4[2 * (k32 & 3)];
            if constexpr (!COMPACT_ACTIVATION_CACHE) {
              s = xb.ds4[2 * (k32 & 3) + 1];
            }
#pragma unroll
            for (int pack = 0; pack < 4; ++pack) {
              sum0 = gguf_q4_k_mmq_dp4a(
                  0x01010101u,
                  static_cast<uint32_t>(
                      pack_i8x4(qs + pack * 4)),
                  sum0);
            }
#pragma unroll
            for (int pack = 4; pack < 8; ++pack) {
              sum1 = gguf_q4_k_mmq_dp4a(
                  0x01010101u,
                  static_cast<uint32_t>(
                      pack_i8x4(qs + pack * 4)),
                  sum1);
            }
          }
          activation_tile[pass][row_offset].d = d;
          if constexpr (COMPACT_ACTIVATION_CACHE) {
            activation_tile[pass][row_offset].sums[0] =
                static_cast<int16_t>(sum0);
            activation_tile[pass][row_offset].sums[1] =
                static_cast<int16_t>(sum1);
          } else {
            activation_tile[pass][row_offset].s = s;
            activation_tile[pass][row_offset].sums[0] = sum0;
            activation_tile[pass][row_offset].sums[1] = sum1;
          }
        }
      }
    }
    __syncthreads();

    if constexpr (WMMA_PREFETCH_WEIGHT) {
      if (k32 + 1 < k32_iters) {
        const int next_k32 = k32 + 1;
        const int next_weight_block = next_k32 >> 3;
        const int next_subblock = next_k32 & 7;
        const int col_quartet = tid >> 3;
        const int pack = tid & 7;
        const int col_offset = col_quartet * 4;
        const int64_t local_col = global_col_base + col_offset;
        const int64_t out_tile = local_col / Q4_K_TILE16_COLS;
        const int col = static_cast<int>(
            local_col % Q4_K_TILE16_COLS);
        const int64_t tile_index =
            (expert_id * out_tiles + out_tile) *
                blocks_per_weight_row +
            next_weight_block;
        const uint8_t* __restrict__ tile =
            qweight + tile_index * Q6_K_T16_BLOCK_BYTES;
        const int record_index =
            (next_subblock * (Q4_K_TILE16_COLS / 4) +
             (col >> 2)) *
                8 +
            pack;
        const uint32_t* __restrict__ record =
            reinterpret_cast<const uint32_t*>(
                tile + Q6_K_T16_QMICRO_OFFSET +
                record_index * Q6_K_T16_QMICRO_RECORD_BYTES);
        prefetched_qmicro[0] = record[0];
        prefetched_qmicro[1] = record[1];
        prefetched_qmicro[2] = record[2];

        const int scale_col_offset = tid >> 1;
        const int scale_half = tid & 1;
        const int64_t scale_local_col =
            global_col_base + scale_col_offset;
        const int64_t scale_out_tile =
            scale_local_col / Q4_K_TILE16_COLS;
        const int scale_col = static_cast<int>(
            scale_local_col % Q4_K_TILE16_COLS);
        const int64_t scale_tile_index =
            (expert_id * out_tiles + scale_out_tile) *
                blocks_per_weight_row +
            next_weight_block;
        const uint8_t* __restrict__ scale_tile =
            qweight + scale_tile_index * Q6_K_T16_BLOCK_BYTES;
        prefetched_d_bits =
            *reinterpret_cast<const uint16_t*>(
                scale_tile + Q6_K_T16_D_OFFSET +
                scale_col * 2);
        prefetched_scale =
            reinterpret_cast<const int8_t*>(
                scale_tile + Q6_K_T16_SCALE_OFFSET)[
                (next_subblock * 2 + scale_half) *
                    Q4_K_TILE16_COLS +
                scale_col];
      }
    }

    if constexpr (WMMA_PREFETCH_ACTIVATION) {
      if (k32 + 1 < k32_iters) {
        const int next_k32 = k32 + 1;
        const int row_offset = tid >> 1;
        const int half = tid & 1;
        const bool valid_row = row_offset < valid_rows;
        const int64_t compact_row =
            valid_row
                ? compact_base + tile_row_in_expert + row_offset
                : 0;
        prefetched_activation_qs = mmq_i32x4_t{0, 0, 0, 0};
        prefetched_activation_d = 0.0f;
        prefetched_activation_sums = 0;
        if (valid_row && compact_row < compact_rows) {
          const int64_t q8_index =
              compact_row * q8_blocks_per_row + (next_k32 >> 2);
          const block_q8_1_mmq_ds4_f32& xb = x_q8[q8_index];
          if (half == 0) {
            prefetched_activation_d =
                xb.ds4[2 * (next_k32 & 3)];
          }
          if constexpr (PRECOMPUTED_ACTIVATION_SUMS) {
            prefetched_activation_sums =
                reinterpret_cast<const uint32_t*>(xb.ds4)
                    [2 * (next_k32 & 3) + 1];
          }
          const int8_t* __restrict__ qs =
              xb.qs + (next_k32 & 3) * Q8_1_BLOCK + half * 16;
          prefetched_activation_qs =
              *reinterpret_cast<const mmq_i32x4_t*>(qs);
        }
      }
    }

    if constexpr (INTEGER_WMMA) {
      const int lane16 = lane & 15;
      const int half_id = lane >> 4;
      const int a_row_offset = wave * 16 + lane16;
      if constexpr (WMMA_HOIST_ACTIVATION) {
        mmq_i32x4_t a_vec0 = {0, 0, 0, 0};
        mmq_i32x4_t a_vec1 = {0, 0, 0, 0};
        if (a_row_offset < valid_rows) {
          const q6_activation_cache_t& a =
              activation_tile[0][a_row_offset];
          a_vec0 = mmq_i32x4_t{
              a.qs[0], a.qs[1], a.qs[2], a.qs[3]};
          a_vec1 = mmq_i32x4_t{
              a.qs[4], a.qs[5], a.qs[6], a.qs[7]};
        }
#pragma unroll
        for (int col_fragment = 0; col_fragment < 4; ++col_fragment) {
          const int col_offset = col_fragment * 16 + lane16;
          const q6_k_mmq64_cache& wb = weight_tile[col_offset];
          const mmq_i32x4_t b_vec0 = {
              static_cast<int>(wb.qs[0]),
              static_cast<int>(wb.qs[1]),
              static_cast<int>(wb.qs[2]),
              static_cast<int>(wb.qs[3])};
          const mmq_i32x4_t b_vec1 = {
              static_cast<int>(wb.qs[4]),
              static_cast<int>(wb.qs[5]),
              static_cast<int>(wb.qs[6]),
              static_cast<int>(wb.qs[7])};
          mmq_i32x8_t dot0 = {0, 0, 0, 0, 0, 0, 0, 0};
          mmq_i32x8_t dot1 = {0, 0, 0, 0, 0, 0, 0, 0};
          dot0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, a_vec0, true, b_vec0, dot0, false);
          dot1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, a_vec1, true, b_vec1, dot1, false);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int row_offset = wave * 16 + half_id + j * 2;
            if (row_offset >= valid_rows) continue;
            const q6_activation_cache_t& xb =
                activation_tile[0][row_offset];
            const int corrected0 =
                dot0[j] - 32 * xb.sums[0];
            const int corrected1 =
                dot1[j] - 32 * xb.sums[1];
            const int acc_index = col_fragment * 8 + j;
            acc[acc_index >> 4][acc_index & 15] +=
                xb.d *
                (wb.scales[0] * static_cast<float>(corrected0) +
                 wb.scales[1] * static_cast<float>(corrected1));
          }
        }
      } else {
#pragma unroll
        for (int col_fragment = 0; col_fragment < 4; ++col_fragment) {
          const int col_offset = col_fragment * 16 + lane16;
          const q6_k_mmq64_cache& wb = weight_tile[col_offset];
#pragma unroll
          for (int pass = 0; pass < ACTIVATION_PASSES; ++pass) {
            mmq_i32x4_t a_vec0 = {0, 0, 0, 0};
            mmq_i32x4_t a_vec1 = {0, 0, 0, 0};
            if (a_row_offset < valid_rows) {
              const q6_activation_cache_t& a =
                  activation_tile[pass][a_row_offset];
              a_vec0 = mmq_i32x4_t{
                  a.qs[0], a.qs[1], a.qs[2], a.qs[3]};
              a_vec1 = mmq_i32x4_t{
                  a.qs[4], a.qs[5], a.qs[6], a.qs[7]};
            }
            const mmq_i32x4_t b_vec0 = {
                static_cast<int>(wb.qs[0]),
                static_cast<int>(wb.qs[1]),
                static_cast<int>(wb.qs[2]),
                static_cast<int>(wb.qs[3])};
            const mmq_i32x4_t b_vec1 = {
                static_cast<int>(wb.qs[4]),
                static_cast<int>(wb.qs[5]),
                static_cast<int>(wb.qs[6]),
                static_cast<int>(wb.qs[7])};
            mmq_i32x8_t dot0 = {0, 0, 0, 0, 0, 0, 0, 0};
            mmq_i32x8_t dot1 = {0, 0, 0, 0, 0, 0, 0, 0};
            dot0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, a_vec0, true, b_vec0, dot0, false);
            dot1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, a_vec1, true, b_vec1, dot1, false);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const int row_offset = wave * 16 + half_id + j * 2;
              if (row_offset >= valid_rows) continue;
              const q6_activation_cache_t& xb =
                  activation_tile[pass][row_offset];
              const int corrected0 =
                  dot0[j] - 32 * xb.sums[0];
              const int corrected1 =
                  dot1[j] - 32 * xb.sums[1];
              const int acc_index = col_fragment * 8 + j;
              acc[acc_index >> 4][acc_index & 15] +=
                  xb.d *
                  (wb.scales[0] * static_cast<float>(corrected0) +
                   wb.scales[1] * static_cast<float>(corrected1));
            }
          }
        }
      }
    } else {
#pragma unroll
      for (int col_iter = 0; col_iter < COL_ITERS; ++col_iter) {
        const int col_offset =
            WAVE_COLUMNS ? wave * 32 + lane : lane + col_iter * 32;
        const q6_k_mmq64_cache* wb_ptr = nullptr;
        if constexpr (WAVE_COLUMNS) {
          wb_ptr = &wave_weight;
        } else {
          wb_ptr = &weight_tile[col_offset];
        }
        const q6_k_mmq64_cache& wb = *wb_ptr;
#pragma unroll
        for (int row_iter = 0; row_iter < ROW_ITERS; ++row_iter) {
          const int row_offset =
              WAVE_COLUMNS ? row_iter : wave * WNITER + row_iter;
          if (row_offset >= valid_rows) continue;
#pragma unroll
          for (int pass = 0; pass < ACTIVATION_PASSES; ++pass) {
            const q6_activation_cache_t& xb =
                activation_tile[pass][row_offset];
            int dot0 = 0;
            int dot1 = 0;
#pragma unroll
            for (int pack = 0; pack < 4; ++pack) {
              const uint32_t xpack =
                  static_cast<uint32_t>(xb.qs[pack]);
              dot0 = gguf_q4_k_mmq_dp4a(
                  wb.qs[pack], xpack, dot0);
            }
#pragma unroll
            for (int pack = 4; pack < 8; ++pack) {
              const uint32_t xpack =
                  static_cast<uint32_t>(xb.qs[pack]);
              dot1 = gguf_q4_k_mmq_dp4a(
                  wb.qs[pack], xpack, dot1);
            }
            dot0 -= 32 * xb.sums[0];
            dot1 -= 32 * xb.sums[1];
            acc[col_iter][row_iter] +=
                xb.d *
                (wb.scales[0] * static_cast<float>(dot0) +
                 wb.scales[1] * static_cast<float>(dot1));
          }
        }
      }
    }
    __syncthreads();
  }

  if constexpr (INTEGER_WMMA) {
    const int lane16 = lane & 15;
    const int half_id = lane >> 4;
#pragma unroll
    for (int col_fragment = 0; col_fragment < 4; ++col_fragment) {
      const int64_t global_col =
          global_col_base + col_fragment * 16 + lane16;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int row_offset = wave * 16 + half_id + j * 2;
        if (row_offset < valid_rows) {
          const int64_t compact_row =
              compact_base + tile_row_in_expert + row_offset;
          if (compact_row < compact_rows) {
            const int acc_index = col_fragment * 8 + j;
            out[compact_row * out_features + global_col] =
                float_to_bf16_bits(
                    acc[acc_index >> 4][acc_index & 15]);
          }
        }
      }
    }
  } else {
#pragma unroll
    for (int col_iter = 0; col_iter < COL_ITERS; ++col_iter) {
      const int64_t global_col =
          global_col_base +
          (WAVE_COLUMNS ? wave * 32 + lane
                        : lane + col_iter * 32);
      if (global_col >= out_features) continue;
#pragma unroll
      for (int row_iter = 0; row_iter < ROW_ITERS; ++row_iter) {
        const int row_offset =
            WAVE_COLUMNS ? row_iter : wave * WNITER + row_iter;
        if (row_offset < valid_rows) {
          const int64_t compact_row =
              compact_base + tile_row_in_expert + row_offset;
          if (compact_row < compact_rows) {
            out[compact_row * out_features + global_col] =
                float_to_bf16_bits(acc[col_iter][row_iter]);
          }
        }
      }
    }
  }
}

template <
    int ACTIVATION_PASSES,
    bool ROWVEC_ACTIVATION_STAGE = false,
    bool WAVE_COLUMNS = false,
    int THREADS = 128,
    int MMQ_N = 32,
    bool QMICRO = false,
    bool COMPACT_ACTIVATION_CACHE = false,
    bool HALF_ROW_ACTIVATION_STAGE = false,
    bool SKIP_PADDED_ACTIVATION_STAGE = false,
    bool QMICRO_PERMUTE = false,
    bool QMICRO_PLANAR = false,
    bool INTEGER_WMMA = false,
    bool WMMA_HOIST_ACTIVATION = false,
    bool WMMA_PREFETCH_WEIGHT = false,
    bool WMMA_PREFETCH_ACTIVATION = false,
    bool PRECOMPUTED_ACTIVATION_SUMS = false,
    bool DENSE_LINEAR = false>
int launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32(
    const block_q8_1_mmq_ds4_f32* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  if (x_q8 == nullptr ||
      (!DENSE_LINEAR &&
       (expert_start_compact == nullptr || expert_start_mmq32 == nullptr ||
        tile_expert == nullptr)) ||
      qweight == nullptr || out == nullptr || compact_rows <= 0 ||
      in_features <= 0 || out_features <= 0 || num_experts <= 0 ||
      mmq_total_rows <= 0 || (in_features % QK_K) != 0 ||
      (out_features % 64) != 0 || (mmq_total_rows % MMQ_N) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(THREADS);
  const dim3 grid(
      static_cast<unsigned int>(out_features / 64),
      static_cast<unsigned int>(mmq_total_rows / MMQ_N));
  hipLaunchKernelGGL(
      (gguf_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32_prefill_compact32_kernel<
          ACTIVATION_PASSES,
          ROWVEC_ACTIVATION_STAGE,
          WAVE_COLUMNS,
          THREADS,
          MMQ_N,
          QMICRO,
          COMPACT_ACTIVATION_CACHE,
          HALF_ROW_ACTIVATION_STAGE,
          SKIP_PADDED_ACTIVATION_STAGE,
          QMICRO_PERMUTE,
          QMICRO_PLANAR,
          INTEGER_WMMA,
          WMMA_HOIST_ACTIVATION,
          WMMA_PREFETCH_WEIGHT,
          WMMA_PREFETCH_ACTIVATION,
          PRECOMPUTED_ACTIVATION_SUMS,
          DENSE_LINEAR>),
      grid,
      block,
      0,
      stream,
      x_q8,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows);
  return static_cast<int>(hipGetLastError());
}

template <
    int WEIGHT_LAYOUT,
    int ACTIVATION_PASSES = 1,
    bool COLLECT_RISKS = false>
int launch_selected_dual_q8_1_ds4_mmq32_compact32(
    const block_q8_1_mmq_ds4* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream,
    int* risk_count = nullptr,
    int* risk_tiles = nullptr,
    int64_t max_risks = 0,
    float risk_threshold = 0.0f) {
  if (!valid_q8_1_selected_dual_mmq32(
          x_q8,
          compact_to_source,
          expert_start_compact,
          expert_start_mmq32,
          tile_expert,
          qweight_a,
          qweight_b,
          out,
          compact_rows,
          in_features,
          out_features_a,
          out_features_b,
          num_experts,
          mmq_total_rows) ||
      (COLLECT_RISKS &&
       (risk_count == nullptr || risk_tiles == nullptr ||
        max_risks <= 0 || max_risks > INT32_MAX ||
        risk_threshold < 0.0f))) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t row_tiles = mmq_total_rows / 32;
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(out_features_total / 32),
      static_cast<unsigned int>(row_tiles));
  if constexpr (WEIGHT_LAYOUT == Q4_K_LAYOUT_X8) {
    hipLaunchKernelGGL(
        gguf_q4_k_x8_selected_dual_q8_1_ds4_mmq32_prefill_compact32_kernel,
        grid,
        block,
        0,
        stream,
        x_q8,
        compact_to_source,
        expert_start_compact,
        expert_start_mmq32,
        tile_expert,
        qweight_a,
        qweight_b,
        out,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        mmq_total_rows);
  } else if constexpr (WEIGHT_LAYOUT == Q4_K_LAYOUT_T16) {
    hipLaunchKernelGGL(
        (gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_prefill_compact32_kernel<
            ACTIVATION_PASSES, COLLECT_RISKS>),
        grid,
        block,
        0,
        stream,
        x_q8,
        compact_to_source,
        expert_start_compact,
        expert_start_mmq32,
        tile_expert,
        qweight_a,
        qweight_b,
        out,
        compact_rows,
        source_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        mmq_total_rows,
        risk_count,
        risk_tiles,
        max_risks,
        risk_threshold);
  } else {
    hipLaunchKernelGGL(
        gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_kernel,
        grid,
        block,
        0,
        stream,
        x_q8,
        compact_to_source,
        expert_start_compact,
        expert_start_mmq32,
        tile_expert,
        qweight_a,
        qweight_b,
        out,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        mmq_total_rows);
  }
  return static_cast<int>(hipGetLastError());
}

int launch_selected_dual_q8_1_ds4_compact32(
    const block_q8_1_mmq_ds4* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  if (!valid_q8_1_selected_dual_common(
          x_q8,
          x_q8,
          x_q8,
          expert_start_compact,
          expert_start_wmma,
          tile_expert,
          qweight_a,
          qweight_b,
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
  const dim3 block(256);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + 15) / 16),
      static_cast<unsigned int>(row_tiles));
  hipLaunchKernelGGL(
      gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_kernel,
      grid,
      block,
      0,
      stream,
      x_q8,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
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

extern "C" int hipengine_gguf_q4_k_q8_1_wmma_i8_probe_16x16(
    const int8_t* a_rows,
    const uint8_t* b_cols,
    int32_t* out,
    hipStream_t stream) {
  return launch_wmma_i8_probe_16x16(a_rows, b_cols, out, stream);
}

extern "C" int hipengine_gguf_q8_1_mmq_ds4_pack_bf16(
    const uint16_t* x,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q8_1_mmq_ds4_pack_bf16<1>(
      x,
      reinterpret_cast<block_q8_1_mmq_ds4*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int hipengine_gguf_q8_1_mmq_ds4_pack_bf16_d4x3(
    const uint16_t* x,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q8_1_mmq_ds4_pack_bf16<3>(
      x,
      reinterpret_cast<block_q8_1_mmq_ds4*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int hipengine_gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x3(
    const uint16_t* x,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q8_1_mmq_ds4_f32_pack_bf16<3>(
      x,
      reinterpret_cast<block_q8_1_mmq_ds4_f32*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int hipengine_gguf_q8_1_mmq_ds4_f32_pack_dual_silu_bf16_d4x3(
    const uint16_t* gate_up,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q8_1_mmq_ds4_f32_pack_dual_silu_bf16<3>(
      gate_up,
      reinterpret_cast<block_q8_1_mmq_ds4_f32*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int hipengine_gguf_q8_1_mmq_ds4_f32_pack_dual_silu_bf16_d4(
    const uint16_t* gate_up,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q8_1_mmq_ds4_f32_pack_dual_silu_bf16<1>(
      gate_up,
      reinterpret_cast<block_q8_1_mmq_ds4_f32*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int
hipengine_gguf_q8_1_mmq_ds4_f32_pack_dual_silu_bf16_d4_q6_half_sums(
    const uint16_t* gate_up,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q8_1_mmq_ds4_f32_pack_dual_silu_bf16<1, true>(
      gate_up,
      reinterpret_cast<block_q8_1_mmq_ds4_f32*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int hipengine_gguf_q8_1_mmq_ds4_f32_pack_dual_silu_bf16_d4x2(
    const uint16_t* gate_up,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q8_1_mmq_ds4_f32_pack_dual_silu_bf16<2>(
      gate_up,
      reinterpret_cast<block_q8_1_mmq_ds4_f32*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int hipengine_gguf_q8_1_mmq_ds4_f32_pack_bf16_d4(
    const uint16_t* x,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q8_1_mmq_ds4_f32_pack_bf16<1>(
      x,
      reinterpret_cast<block_q8_1_mmq_ds4_f32*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int
hipengine_gguf_q8_1_mmq_ds4_f32_pack_bf16_d4_q6_half_sums(
    const uint16_t* x,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q8_1_mmq_ds4_f32_pack_bf16<1, true>(
      x,
      reinterpret_cast<block_q8_1_mmq_ds4_f32*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int hipengine_gguf_q8_1_mmq_ds8_f32_pack_bf16(
    const uint16_t* x,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q8_1_mmq_ds8_f32_pack_bf16(
      x,
      reinterpret_cast<block_q8_1_mmq_ds4_f32*>(out),
      nullptr,
      rows,
      hidden,
      stream);
}

extern "C" int
hipengine_gguf_q8_1_mmq_ds8_f32_pack_bf16_q4_half_sums(
    const uint16_t* x,
    void* out,
    int16_t* half_sums,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q8_1_mmq_ds8_f32_pack_bf16<true>(
      x,
      reinterpret_cast<block_q8_1_mmq_ds4_f32*>(out),
      half_sums,
      rows,
      hidden,
      stream);
}

extern "C" int hipengine_gguf_q8_1_mmq_ds4_f32_pack_bf16_d4x2(
    const uint16_t* x,
    void* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  return launch_q8_1_mmq_ds4_f32_pack_bf16<2>(
      x,
      reinterpret_cast<block_q8_1_mmq_ds4_f32*>(out),
      rows,
      hidden,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<1>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32_rowvec_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<
      1,
      false,
      false,
      64,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds8_f32_mmq128x32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<
      1,
      false,
      true,
      128>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds8_f32_mmq128x32_rowvec_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<
      1,
      false,
      true,
      128,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds8_f32_mmq128x32_wavecols_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<
      1,
      false,
      true,
      128,
      true,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds8_f32_mmq128x32_wavecols_direct_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<
      1,
      false,
      true,
      128,
      true,
      true,
      128,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds8_f32_mmq128x32_wavecols_direct_doublebuf_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<
      1,
      false,
      true,
      128,
      true,
      true,
      128,
      true,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

template <
    int RAW_WEIGHT_PREFETCH_PACKS,
    bool PRECOMPUTED_ACTIVATION_SUMS = false>
int launch_q4_k_t16_raw_prefetch_production(
    const void* x_q8,
    const int16_t* precomputed_activation_sums,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<
      1,
      false,
      true,
      128,
      true,
      true,
      128,
      true,
      true,
      RAW_WEIGHT_PREFETCH_PACKS,
      PRECOMPUTED_ACTIVATION_SUMS>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream,
      precomputed_activation_sums);
}

extern "C" int
hipengine_gguf_q4_k_t16_raw_prefetch_p8_selected_dual_q8_1_ds8_f32_mmq128x32_wavecols_direct_doublebuf_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_raw_prefetch_production<8>(
      x_q8,
      nullptr,
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_dual_interleaved_raw_prefetch_p8_selected_dual_q8_1_ds8_f32_mmq128x32_wavecols_direct_doublebuf_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_dual,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  if (out_features_a != out_features_b) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<
      1,
      false,
      true,
      128,
      true,
      true,
      128,
      true,
      true,
      8,
      false,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_dual,
      nullptr,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_raw_prefetch_p8_precomputed_sums_selected_dual_q8_1_ds8_f32_mmq128x32_wavecols_direct_doublebuf_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int16_t* precomputed_activation_sums,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_raw_prefetch_production<8, true>(
      x_q8,
      precomputed_activation_sums,
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds4x2_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<2>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<3>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_q8_1_ds4_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<1, true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      nullptr,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      qweight,
      out,
      compact_rows,
      compact_rows,
      in_features,
      out_features,
      0,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_q8_1_ds4_f32_mmq64x32_rowvec_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      64,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      nullptr,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      qweight,
      out,
      compact_rows,
      compact_rows,
      in_features,
      out_features,
      0,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_q8_1_ds4_f32_mmq64x32_wavecols_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      64,
      true,
      true,
      64>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      nullptr,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      qweight,
      out,
      compact_rows,
      compact_rows,
      in_features,
      out_features,
      0,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_selected_q8_1_ds4_f32_mmq64x32_wavecols_direct_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      64,
      true,
      true,
      64,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      nullptr,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      qweight,
      out,
      compact_rows,
      compact_rows,
      in_features,
      out_features,
      0,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q4_k_t16_raw_prefetch_p8_selected_q8_1_ds4_f32_mmq64x32_wavecols_direct_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q4_k_t16_selected_dual_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      64,
      true,
      true,
      64,
      true,
      false,
      8>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      nullptr,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      qweight,
      out,
      compact_rows,
      compact_rows,
      in_features,
      out_features,
      0,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_selected_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<3>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<1>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32_rowvec_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<1, true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq64,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      128,
      64>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq64,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq64,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      128,
      64,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq64,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_compact_activation_selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq64,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      128,
      64,
      true,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq64,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_half_row_activation_selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq64,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      128,
      64,
      true,
      true,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq64,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_skip_padded_activation_selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq64,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      128,
      64,
      true,
      true,
      true,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq64,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_permute_skip_padded_activation_selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq64,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      128,
      64,
      true,
      true,
      true,
      true,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq64,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_skip_padded_activation_selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq64,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      128,
      64,
      true,
      true,
      true,
      true,
      false,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq64,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_integer_wmma_skip_padded_activation_selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq64,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      128,
      64,
      true,
      true,
      true,
      true,
      false,
      true,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq64,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_integer_wmma_hoist_activation_skip_padded_activation_selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq64,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      128,
      64,
      true,
      true,
      true,
      true,
      false,
      true,
      true,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq64,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_integer_wmma_hoist_activation_prefetch_weight_skip_padded_activation_selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq64,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      128,
      64,
      true,
      true,
      true,
      true,
      false,
      true,
      true,
      true,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq64,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_integer_wmma_hoist_activation_prefetch_weight_activation_skip_padded_activation_selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq64,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      128,
      64,
      true,
      true,
      true,
      true,
      false,
      true,
      true,
      true,
      true,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq64,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_integer_wmma_hoist_activation_prefetch_weight_activation_precomputed_sums_skip_padded_activation_selected_q8_1_ds4_f32_mmq64x64_rowvec_prefill_compact64_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq64,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      128,
      64,
      true,
      true,
      true,
      true,
      false,
      true,
      true,
      true,
      true,
      true,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq64,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_integer_mmq64x64_dense_bf16_bf16_out(
    const void* x_q8,
    const void* qweight,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1,
      true,
      false,
      128,
      64,
      true,
      true,
      true,
      true,
      false,
      true,
      true,
      true,
      true,
      true,
      true,
      true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      nullptr,
      nullptr,
      nullptr,
      reinterpret_cast<const uint8_t*>(qweight),
      reinterpret_cast<uint16_t*>(out),
      rows,
      in_features,
      out_features,
      1,
      64,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_selected_q8_1_ds4_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1, false, false, 128, 32, true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_selected_q8_1_ds4x2_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      2, false, false, 128, 32, true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_selected_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      3, false, false, 128, 32, true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_selected_q8_1_ds4_f32_mmq64x32_rowvec_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<
      1, true, false, 128, 32, true>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_selected_q8_1_ds4x2_f32_mmq64x32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_q6_k_t16_selected_q8_1_ds4_f32_mmq64x32<2>(
      reinterpret_cast<const block_q8_1_mmq_ds4_f32*>(x_q8),
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight,
      out,
      compact_rows,
      in_features,
      out_features,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out(
    const int8_t* x_qs,
    const float* x_d,
    const float* x_sum,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_compact32(
      x_qs,
      x_d,
      x_sum,
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_ds4_mmq32_compact32<Q4_K_LAYOUT_RAW>(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_x8_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_ds4_mmq32_compact32<Q4_K_LAYOUT_X8>(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_ds4_mmq32_compact32<Q4_K_LAYOUT_T16>(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds4x3_mmq32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_ds4_mmq32_compact32<
      Q4_K_LAYOUT_T16, 3>(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_q8_1_ds4x3_guarded_mmq32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_mmq32,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int* risk_count,
    int* risk_tiles,
    int64_t max_risks,
    float risk_threshold,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t mmq_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_ds4_mmq32_compact32<
      Q4_K_LAYOUT_T16, 3, true>(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      compact_to_source,
      expert_start_compact,
      expert_start_mmq32,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      mmq_total_rows,
      stream,
      risk_count,
      risk_tiles,
      max_risks,
      risk_threshold);
}

extern "C" int hipengine_gguf_q4_k_t16_selected_dual_sparse_exact_correct_bf16(
    const uint16_t* source_x,
    const int64_t* compact_to_source,
    const int64_t* expert_start_compact,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    const int* risk_count,
    const int* risk_tiles,
    int64_t max_risks,
    int64_t compact_rows,
    int64_t source_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    hipStream_t stream) {
  const int64_t out_features_total =
      out_features_a + out_features_b;
  const int64_t total_tiles =
      compact_rows * out_features_total /
      Q4_K_TILE16_COLS;
  if (source_x == nullptr || compact_to_source == nullptr ||
      expert_start_compact == nullptr || qweight_a == nullptr ||
      qweight_b == nullptr || out == nullptr ||
      risk_count == nullptr || risk_tiles == nullptr ||
      max_risks <= 0 || max_risks > INT32_MAX ||
      compact_rows <= 0 || source_rows <= 0 ||
      in_features <= 0 || (in_features % QK_K) != 0 ||
      out_features_a <= 0 ||
      (out_features_a % Q4_K_TILE16_COLS) != 0 ||
      out_features_b <= 0 ||
      (out_features_b % Q4_K_TILE16_COLS) != 0 ||
      num_experts <= 0 || total_tiles <= 0 ||
      total_tiles > INT32_MAX) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  int64_t grid_blocks =
      max_risks < total_tiles ? max_risks : total_tiles;
  if (grid_blocks > SPARSE_EXACT_MAX_BLOCKS) {
    grid_blocks = SPARSE_EXACT_MAX_BLOCKS;
  }
  hipLaunchKernelGGL(
      gguf_q4_k_t16_selected_dual_sparse_exact_correct_bf16_kernel,
      dim3(static_cast<unsigned int>(grid_blocks)),
      dim3(256),
      0,
      stream,
      source_x,
      compact_to_source,
      expert_start_compact,
      qweight_a,
      qweight_b,
      out,
      risk_count,
      risk_tiles,
      max_risks,
      compact_rows,
      source_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_ds4_wmma_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_ds4_wmma32_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_ds4_wmma64_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* q4_a,
    const float* scale_a,
    const float* min_a,
    const uint8_t* q4_b,
    const float* scale_b,
    const float* min_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_ds4_preview_wmma32_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      q4_a,
      scale_a,
      min_a,
      q4_b,
      scale_b,
      min_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_ds4_wmma32_ldspack_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_ds4_wmma32_lds_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}

extern "C" int hipengine_gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out(
    const void* x_q8,
    const int64_t* expert_start_compact,
    const int64_t* expert_start_wmma,
    const int64_t* tile_expert,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t compact_rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t num_experts,
    int64_t wmma_total_rows,
    hipStream_t stream) {
  return launch_selected_dual_q8_1_ds4_compact32(
      reinterpret_cast<const block_q8_1_mmq_ds4*>(x_q8),
      expert_start_compact,
      expert_start_wmma,
      tile_expert,
      qweight_a,
      qweight_b,
      out,
      compact_rows,
      in_features,
      out_features_a,
      out_features_b,
      num_experts,
      wmma_total_rows,
      stream);
}
