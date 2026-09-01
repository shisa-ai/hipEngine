// Dense decode-shaped GEMV for GGUF Q6_K T16 replacement layout.
//
// P9.H3 extension for the qwen35moe Q6_K lm-head fallback: consumes the
// byte-neutral Q6T16 tile layout produced by ``hipengine.quant.gguf_t16``.
// One block computes one 16-column output tile for one input row.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

constexpr int QK_K = 256;
constexpr int T16_COLS = 16;
constexpr int Q6_GROUPS16 = 16;
constexpr int Q6_T16_D_OFFSET = 0;
constexpr int Q6_T16_SCALE_OFFSET = Q6_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q6_T16_QL_OFFSET = Q6_T16_SCALE_OFFSET + Q6_GROUPS16 * T16_COLS;
constexpr int Q6_T16_QH_OFFSET = Q6_T16_QL_OFFSET + QK_K * (T16_COLS / 2);
constexpr int Q6_T16_BLOCK_BYTES = Q6_T16_QH_OFFSET + QK_K * (T16_COLS / 4);
constexpr int Q6_T16_QMICRO_OFFSET = Q6_T16_QL_OFFSET;
constexpr int Q6_T16_QMICRO_RECORD_BYTES = 12;
constexpr int Q8_1_BLOCK = 32;

struct gguf_q8_1_block {
  half_t d;
  half_t s;
  int8_t qs[Q8_1_BLOCK];
};

static_assert(
    sizeof(gguf_q8_1_block) == 36,
    "GGUF q8_1 block layout must match GGML");

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

__device__ inline bool better_pair(
    float value,
    int32_t index,
    float best_value,
    int32_t best_index) {
  return value > best_value ||
         (value == best_value && index < best_index);
}

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t v) {
  return bf16_bits_to_float(v);
}

__device__ inline float8_t zero_float8() {
  return float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

__device__ inline half16_t load_bf16_row_half16(
    const uint16_t* __restrict__ x,
    int64_t row,
    int64_t in_features,
    int64_t k_base,
    bool valid_row) {
  half16_t out;
  if (valid_row) {
    const uint16_t* __restrict__ values = x + row * in_features + k_base;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      out[i] = static_cast<half_t>(bf16_bits_to_float(values[i]));
    }
  } else {
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = static_cast<half_t>(0.0f);
  }
  return out;
}

template <typename out_t>
__device__ inline out_t float_to_output(float v) {
  return static_cast<out_t>(v);
}

template <>
__device__ inline uint16_t float_to_output<uint16_t>(float v) {
  return float_to_bf16_bits(v);
}

#define GGUF_T16_WAVE_REDUCE(acc) \
  do {                            \
    acc += __shfl_down(acc, 16);  \
    acc += __shfl_down(acc, 8);   \
    acc += __shfl_down(acc, 4);   \
    acc += __shfl_down(acc, 2);   \
    acc += __shfl_down(acc, 1);   \
  } while (0)

__device__ inline float q6_t16_permlanex16_f32(float value) {
  union FloatBits {
    float value;
    uint32_t bits;
  } data;
  data.value = value;
  data.bits = __builtin_amdgcn_permlanex16(
      data.bits, data.bits, 0x76543210U, 0xFEDCBA98U, false, false);
  return data.value;
}

template <int DPP_CTRL>
__device__ inline float q6_t16_dpp_move_f32(float value) {
  union FloatBits {
    float value;
    int bits;
  } data;
  data.value = value;
  data.bits = __builtin_amdgcn_mov_dpp(
      data.bits, DPP_CTRL, 0xF, 0xF, true);
  return data.value;
}

__device__ inline float q6_t16_dpp_add_row_shl1_f32(float value) {
  float result;
  asm volatile(
      "v_add_f32_dpp %0, %1, %1 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1"
      : "=v"(result)
      : "v"(value));
  return result;
}

__device__ inline void q6_t16_wave_reduce_dpp(float& value) {
  value += q6_t16_permlanex16_f32(value);
  value += q6_t16_dpp_move_f32<0x108>(value);
  value += q6_t16_dpp_move_f32<0x104>(value);
  value += q6_t16_dpp_move_f32<0x102>(value);
  value = q6_t16_dpp_add_row_shl1_f32(value);
}

__device__ inline int q6_planar_dp4a_u8_s8(
    uint32_t a,
    uint32_t b,
    int c) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(
      true,
      static_cast<int>(a),
      true,
      static_cast<int>(b),
      c,
      false);
#else
  const int8_t b0 = static_cast<int8_t>(b & 0xFFu);
  const int8_t b1 = static_cast<int8_t>((b >> 8) & 0xFFu);
  const int8_t b2 = static_cast<int8_t>((b >> 16) & 0xFFu);
  const int8_t b3 = static_cast<int8_t>((b >> 24) & 0xFFu);
  return c + static_cast<int>(a & 0xFFu) * static_cast<int>(b0) +
         static_cast<int>((a >> 8) & 0xFFu) * static_cast<int>(b1) +
         static_cast<int>((a >> 16) & 0xFFu) * static_cast<int>(b2) +
         static_cast<int>((a >> 24) & 0xFFu) * static_cast<int>(b3);
#endif
}

__device__ inline uint32_t q6_planar_pack4_unsigned(
    uint32_t low01,
    uint32_t low23,
    uint32_t high,
    int col4) {
  const uint32_t low_pair = (col4 < 2) ? low01 : low23;
  const uint32_t low =
      ((col4 & 1) ? (low_pair >> 4) : low_pair) & 0x0F0F0F0Fu;
  const uint32_t high2 =
      ((high >> (2 * col4)) & 0x03030303u) << 4;
  return low | high2;
}

__device__ inline void q6_planar_reduce8(
    float& a0,
    float& a1,
    float& a2,
    float& a3,
    float& a4,
    float& a5,
    float& a6,
    float& a7,
    float* xchg,
    int tid,
    float& r0,
    float& r1,
    float& r2,
    float& r3,
    float& r4,
    float& r5,
    float& r6,
    float& r7) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    a0 += __shfl_down(a0, offset);
    a1 += __shfl_down(a1, offset);
    a2 += __shfl_down(a2, offset);
    a3 += __shfl_down(a3, offset);
    a4 += __shfl_down(a4, offset);
    a5 += __shfl_down(a5, offset);
    a6 += __shfl_down(a6, offset);
    a7 += __shfl_down(a7, offset);
  }
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
    xchg[sub_id * 8 + 0] = a0;
    xchg[sub_id * 8 + 1] = a1;
    xchg[sub_id * 8 + 2] = a2;
    xchg[sub_id * 8 + 3] = a3;
    xchg[sub_id * 8 + 4] = a4;
    xchg[sub_id * 8 + 5] = a5;
    xchg[sub_id * 8 + 6] = a6;
    xchg[sub_id * 8 + 7] = a7;
  }
  __syncthreads();
  r0 = r1 = r2 = r3 = r4 = r5 = r6 = r7 = 0.0f;
  if (tid == 0) {
    const int sub_groups = blockDim.x >> 5;
    for (int g = 0; g < sub_groups; ++g) {
      r0 += xchg[g * 8 + 0];
      r1 += xchg[g * 8 + 1];
      r2 += xchg[g * 8 + 2];
      r3 += xchg[g * 8 + 3];
      r4 += xchg[g * 8 + 4];
      r5 += xchg[g * 8 + 5];
      r6 += xchg[g * 8 + 6];
      r7 += xchg[g * 8 + 7];
    }
  }
}

__device__ inline int t16_q6_quant(const uint8_t* __restrict__ tile, int k_in_block, int col) {
  const uint8_t low_packed = tile[Q6_T16_QL_OFFSET + k_in_block * (T16_COLS / 2) + (col >> 1)];
  const uint8_t low = (col & 1) ? (low_packed >> 4) : (low_packed & 0x0Fu);
  const uint8_t high_packed = tile[Q6_T16_QH_OFFSET + k_in_block * (T16_COLS / 4) + (col >> 2)];
  const uint8_t high = (high_packed >> (2 * (col & 3))) & 0x03u;
  return static_cast<int>(low | (high << 4)) - 32;
}

template <typename scalar_t, typename out_t>
__global__ __launch_bounds__(128, 4) void q6_k_t16_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const scalar_t* __restrict__ x_row = x + row * in_features;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int g16 = k_in_block >> 4;
      const int8_t* __restrict__ scales = reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float d = fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
        const int scale_i = static_cast<int>(scales[g16 * T16_COLS + col]);
        const int q_i = t16_q6_quant(tile, k_in_block, col);
        acc[col] += xv * (d * static_cast<float>(scale_i * q_i));
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
    out_t* __restrict__ out_row = out + row * out_features + out_tile * T16_COLS;
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) total += xchg[g * T16_COLS + col];
      out_row[col] = float_to_output<out_t>(total);
    }
  }
}

// Exact planar-qmicro c1 sibling. One local128 block retains the legacy
// 16-column output ownership, K partitions, FMA stream, reduction tree, and
// BF16 boundary while decoding one byte-neutral K4/column-quartet record once.
template <typename out_t>
__global__ __launch_bounds__(128, 4)
void q6_k_t16_qmicro_planar_gemv_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_tile = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const uint16_t* __restrict__ x_row = x + row * in_features;
  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    const int8_t* __restrict__ scales =
        reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
    for (int k_in_block = tid; k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const int g16 = k_in_block >> 4;
      const int subblock = k_in_block >> 5;
      const int pack = (k_in_block & 31) >> 2;
      const int k4 = k_in_block & 3;
      const float xv = bf16_bits_to_float(x_row[k_base + k_in_block]);
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
          const float d =
              fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
          const int scale_i =
              static_cast<int>(scales[g16 * T16_COLS + col]);
          acc[col] +=
              xv * (d * static_cast<float>(scale_i * q_i));
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
    for (int col = 0; col < T16_COLS; ++col) {
      xchg[sub_id * T16_COLS + col] = acc[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    out_t* __restrict__ out_row =
        out + row * out_features + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        total += xchg[g * T16_COLS + col];
      }
      out_row[col] = float_to_output<out_t>(total);
    }
  }
}

// Exact serial-c1 down+residual sibling. Keep a separate copy of the direct
// planar projection so its fallback code generation remains frozen. The store
// first rounds the dot to BF16, then reproduces the standalone BF16 add and
// final BF16 rounding.
__global__ __launch_bounds__(128, 4)
void q6_k_t16_qmicro_planar_gemv_bf16_residual_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    const uint16_t* __restrict__ residual,
    uint16_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;
  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
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
          const float d =
              fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
          const int scale_i =
              static_cast<int>(scales[g16 * T16_COLS + col]);
          acc[col] +=
              xv * (d * static_cast<float>(scale_i * q_i));
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
    for (int col = 0; col < T16_COLS; ++col) {
      xchg[sub_id * T16_COLS + col] = acc[col];
    }
  }
  __syncthreads();

  if (tid == 0) {
    const int64_t out_base = out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        total += xchg[g * T16_COLS + col];
      }
      const int64_t idx = out_base + col;
      const uint16_t projected = float_to_bf16_bits(total);
      out[idx] = float_to_bf16_bits(
          bf16_bits_to_float(residual[idx]) +
          bf16_bits_to_float(projected));
    }
  }
}

// Dense prefill owner over the same Q6T16 bytes. One wave32 block computes a
// 16x16 output tile. Its arithmetic is the one-expert specialization of the
// registered selected-Q6T16 direct WMMA producer; c=1 and small-B retain the
// GEMV/rowtile fallbacks above.
__global__ __launch_bounds__(32, 2)
void q6_k_t16_wmma_prefill_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t col_tile = blockIdx.x;
  const int64_t row_base = static_cast<int64_t>(blockIdx.y) * 16;
  const int lane16 = threadIdx.x & 15;
  const int half_id = threadIdx.x >> 4;
  const int64_t input_row = row_base + lane16;
  const bool valid_row = input_row < rows;
  const int64_t out_col = col_tile * T16_COLS + lane16;
  const int64_t blocks_per_row = in_features / QK_K;

  float8_t acc = zero_float8();
  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (col_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
    const float d = fp16_bytes_to_float(
        tile + Q6_T16_D_OFFSET + lane16 * 2);
    const int8_t* __restrict__ scales =
        reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
    const int64_t block_k_base = blk * QK_K;
#if defined(__gfx1100__)
#pragma unroll 1
#else
#pragma unroll
#endif
    for (int kt = 0; kt < Q6_GROUPS16; ++kt) {
      const int k_in_block = kt * 16;
      const half16_t a = load_bf16_row_half16(
          x,
          valid_row ? input_row : 0,
          in_features,
          block_k_base + k_in_block,
          valid_row);
      const float scale = d * static_cast<float>(
          scales[kt * T16_COLS + lane16]);
      half16_t b;
#pragma unroll
      for (int kk = 0; kk < 16; ++kk) {
        const int q = t16_q6_quant(tile, k_in_block + kk, lane16);
        b[kk] = static_cast<half_t>(scale * static_cast<float>(q));
      }
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }
  }

#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int64_t out_row = row_base + half_id + j * 2;
    if (out_row < rows) {
      out[out_row * out_features + out_col] = float_to_bf16_bits(acc[j]);
    }
  }
}

// Four-wave shared-weight standard-T16 companion. Each wave preserves the
// exact 48-column x 64-row sequence screened by the tiled producer; the
// workgroup decodes each 48 x 256 Q6 slab once into LDS for four independent
// row ranges instead of repeating the standard-T16 decode in four blocks.
__global__ __launch_bounds__(128, 1)
void q6_k_t16_wmma_prefill_shared4_bf16_kernel(
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
  constexpr int ROWS_PER_BLOCK = WAVES_PER_BLOCK * ROWS_PER_WAVE;
  __shared__ half_t weight_tile[OUT_COLS_PER_BLOCK * QK_K];

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane16 = tid & 15;
  const int half_id = (tid & 31) >> 4;
  const int64_t col_tile_base =
      static_cast<int64_t>(blockIdx.x) * OUT_TILES_PER_BLOCK;
  const int64_t block_row_base =
      static_cast<int64_t>(blockIdx.y) * ROWS_PER_BLOCK;
  const int64_t row_base = block_row_base + wave * ROWS_PER_WAVE;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;

  float8_t acc[OUT_TILES_PER_BLOCK][ROW_TILES_PER_WAVE];
#pragma unroll
  for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
#pragma unroll
    for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
      acc[tm][tn] = zero_float8();
    }
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    constexpr int DECODE_UNITS = OUT_COLS_PER_BLOCK * Q6_GROUPS16;
    for (int decode_unit = tid; decode_unit < DECODE_UNITS;
         decode_unit += blockDim.x) {
      const int local_col = decode_unit / Q6_GROUPS16;
      const int kt = decode_unit % Q6_GROUPS16;
      const int tm = local_col / T16_COLS;
      const int col = local_col % T16_COLS;
      const int64_t col_tile = col_tile_base + tm;
      const int64_t safe_col_tile =
          col_tile < out_tiles ? col_tile : out_tiles - 1;
      const uint8_t* __restrict__ tile =
          tiles + (safe_col_tile * blocks_per_row + blk) *
                      Q6_T16_BLOCK_BYTES;
      const float d = fp16_bytes_to_float(
          tile + Q6_T16_D_OFFSET + col * 2);
      const int8_t* __restrict__ scales =
          reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
      const float scale =
          d * static_cast<float>(scales[kt * T16_COLS + col]);
      const int k_off = kt * 16;
#pragma unroll
      for (int kk = 0; kk < 16; ++kk) {
        const int q = t16_q6_quant(tile, k_off + kk, col);
        weight_tile[local_col * QK_K + k_off + kk] =
            static_cast<half_t>(scale * static_cast<float>(q));
      }
    }
    __syncthreads();

    const int64_t block_k_base = blk * QK_K;
#if defined(__gfx1100__)
#pragma unroll 1
#else
#pragma unroll
#endif
    for (int kt = 0; kt < Q6_GROUPS16; ++kt) {
      const int k_off = kt * 16;
      half16_t a[ROW_TILES_PER_WAVE];
#pragma unroll
      for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
        const int64_t input_row = row_base + tn * 16 + lane16;
        const bool valid_row = input_row < rows;
        a[tn] = load_bf16_row_half16(
            x,
            valid_row ? input_row : 0,
            in_features,
            block_k_base + k_off,
            valid_row);
      }
#pragma unroll
      for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
        half16_t b;
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          b[kk] = weight_tile[
              (tm * T16_COLS + lane16) * QK_K + k_off + kk];
        }
#pragma unroll
        for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a[tn], b, acc[tm][tn]);
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
              float_to_bf16_bits(acc[tm][tn][j]);
        }
      }
    }
  }
}

// Planar-qmicro replacement of the dense WMMA producer. One wave owns a
// 48-column x 64-row tile, reusing each decoded weight fragment across four
// activation fragments while preserving every per-output K16 WMMA/BF16
// boundary from the legacy-T16 producer.
__global__ __launch_bounds__(32, 2)
void q6_k_t16_qmicro_planar_wmma_prefill_bf16_kernel(
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
    const int8_t* scales[OUT_TILES_PER_BLOCK];
    float d[OUT_TILES_PER_BLOCK];
#pragma unroll
    for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
      const int64_t col_tile = col_tile_base + tm;
      const int64_t safe_col_tile =
          col_tile < out_tiles ? col_tile : out_tiles - 1;
      tile[tm] =
          tiles + (safe_col_tile * blocks_per_row + blk) *
                      Q6_T16_BLOCK_BYTES;
      d[tm] = fp16_bytes_to_float(
          tile[tm] + Q6_T16_D_OFFSET + lane16 * 2);
      scales[tm] = reinterpret_cast<const int8_t*>(
          tile[tm] + Q6_T16_SCALE_OFFSET);
    }
    const int64_t block_k_base = blk * QK_K;
#if defined(__gfx1100__)
#pragma unroll 1
#else
#pragma unroll
#endif
    for (int kt = 0; kt < Q6_GROUPS16; ++kt) {
      const int k_in_block = kt * 16;
      half16_t a[ROW_TILES_PER_BLOCK];
#pragma unroll
      for (int tn = 0; tn < ROW_TILES_PER_BLOCK; ++tn) {
        const int64_t input_row = row_base + tn * 16 + lane16;
        const bool valid_row = input_row < rows;
        a[tn] = load_bf16_row_half16(
            x,
            valid_row ? input_row : 0,
            in_features,
            block_k_base + k_in_block,
            valid_row);
      }
      const int subblock = k_in_block >> 5;
      const int pack_base = (k_in_block & 31) >> 2;
      const int col_quartet = lane16 >> 2;
      const int col4 = lane16 & 3;
#pragma unroll
      for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
        const float scale = d[tm] * static_cast<float>(
            scales[tm][kt * T16_COLS + lane16]);
        half16_t b;
#pragma unroll
        for (int pack4 = 0; pack4 < 4; ++pack4) {
          const int record_index =
              (subblock * (T16_COLS / 4) + col_quartet) * 8 +
              pack_base + pack4;
          const uint8_t* __restrict__ record =
              tile[tm] + Q6_T16_QMICRO_OFFSET +
              record_index * Q6_T16_QMICRO_RECORD_BYTES;
          const uint32_t low_word =
              reinterpret_cast<const uint32_t*>(record)[col4 >> 1];
          const uint32_t high_word =
              *reinterpret_cast<const uint32_t*>(record + 8);
#pragma unroll
          for (int k4 = 0; k4 < 4; ++k4) {
            const uint8_t low_byte =
                static_cast<uint8_t>(low_word >> (8 * k4));
            const uint8_t low =
                (col4 & 1) ? (low_byte >> 4) : (low_byte & 0x0Fu);
            const uint8_t high_byte =
                static_cast<uint8_t>(high_word >> (8 * k4));
            const uint8_t high =
                (high_byte >> (2 * col4)) & 0x03u;
            const int q = static_cast<int>(low | (high << 4)) - 32;
            b[pack4 * 4 + k4] =
                static_cast<half_t>(scale * static_cast<float>(q));
          }
        }
#pragma unroll
        for (int tn = 0; tn < ROW_TILES_PER_BLOCK; ++tn) {
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a[tn], b, acc[tm][tn]);
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
              float_to_bf16_bits(acc[tm][tn][j]);
        }
      }
    }
  }
}

// Four-wave shared-weight companion to the retained planar-qmicro producer.
// ROW_TILES_PER_WAVE controls only row ownership and padding: four is the
// existing 256-row bulk owner, while one is the 64-row small-prefill candidate.
// Every wave preserves the parent's K16 WMMA/F32 accumulator/BF16 store order.
template <int ROW_TILES_PER_WAVE>
__device__ __forceinline__ void
q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_body(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    half_t* __restrict__ weight_tile) {
  static_assert(ROW_TILES_PER_WAVE > 0, "row tiles per wave must be positive");
  constexpr int OUT_TILES_PER_BLOCK = 3;
  constexpr int WAVES_PER_BLOCK = 4;
  constexpr int OUT_COLS_PER_BLOCK = OUT_TILES_PER_BLOCK * T16_COLS;
  constexpr int ROWS_PER_WAVE = ROW_TILES_PER_WAVE * 16;
  constexpr int ROWS_PER_BLOCK = WAVES_PER_BLOCK * ROWS_PER_WAVE;

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane16 = tid & 15;
  const int half_id = (tid & 31) >> 4;
  const int64_t col_tile_base =
      static_cast<int64_t>(blockIdx.x) * OUT_TILES_PER_BLOCK;
  const int64_t block_row_base =
      static_cast<int64_t>(blockIdx.y) * ROWS_PER_BLOCK;
  const int64_t row_base = block_row_base + wave * ROWS_PER_WAVE;
  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t out_tiles = out_features / T16_COLS;

  float8_t acc[OUT_TILES_PER_BLOCK][ROW_TILES_PER_WAVE];
#pragma unroll
  for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
#pragma unroll
    for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
      acc[tm][tn] = zero_float8();
    }
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    constexpr int DECODE_UNITS = OUT_COLS_PER_BLOCK * Q6_GROUPS16;
    for (int decode_unit = tid; decode_unit < DECODE_UNITS;
         decode_unit += blockDim.x) {
      const int local_col = decode_unit / Q6_GROUPS16;
      const int kt = decode_unit % Q6_GROUPS16;
      const int tm = local_col / T16_COLS;
      const int col = local_col % T16_COLS;
      const int64_t col_tile = col_tile_base + tm;
      const int64_t safe_col_tile =
          col_tile < out_tiles ? col_tile : out_tiles - 1;
      const uint8_t* __restrict__ tile =
          tiles + (safe_col_tile * blocks_per_row + blk) *
                      Q6_T16_BLOCK_BYTES;
      const float d = fp16_bytes_to_float(
          tile + Q6_T16_D_OFFSET + col * 2);
      const int8_t* __restrict__ scales =
          reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
      const float scale =
          d * static_cast<float>(scales[kt * T16_COLS + col]);
      const int k_off = kt * 16;
#pragma unroll
      for (int kk = 0; kk < 16; ++kk) {
        const int k = k_off + kk;
        const int subblock = k >> 5;
        const int pack = (k & 31) >> 2;
        const int k4 = k & 3;
        const int col_quartet = col >> 2;
        const int col4 = col & 3;
        const int record_index =
            (subblock * (T16_COLS / 4) + col_quartet) * 8 + pack;
        const uint8_t* __restrict__ record =
            tile + Q6_T16_QMICRO_OFFSET +
            record_index * Q6_T16_QMICRO_RECORD_BYTES;
        const uint8_t low_byte = record[(col4 >> 1) * 4 + k4];
        const uint8_t low =
            (col4 & 1) ? (low_byte >> 4) : (low_byte & 0x0Fu);
        const uint8_t high = (record[8 + k4] >> (2 * col4)) & 0x03u;
        const int q = static_cast<int>(low | (high << 4)) - 32;
        weight_tile[local_col * QK_K + k] =
            static_cast<half_t>(scale * static_cast<float>(q));
      }
    }
    __syncthreads();

    const int64_t block_k_base = blk * QK_K;
#if defined(__gfx1100__)
#pragma unroll 1
#else
#pragma unroll
#endif
    for (int kt = 0; kt < Q6_GROUPS16; ++kt) {
      const int k_off = kt * 16;
      half16_t a[ROW_TILES_PER_WAVE];
#pragma unroll
      for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
        const int64_t input_row = row_base + tn * 16 + lane16;
        const bool valid_row = input_row < rows;
        a[tn] = load_bf16_row_half16(
            x,
            valid_row ? input_row : 0,
            in_features,
            block_k_base + k_off,
            valid_row);
      }
#pragma unroll
      for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
        half16_t b;
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          b[kk] = weight_tile[
              (tm * T16_COLS + lane16) * QK_K + k_off + kk];
        }
#pragma unroll
        for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a[tn], b, acc[tm][tn]);
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
              float_to_bf16_bits(acc[tm][tn][j]);
        }
      }
    }
  }
}

__global__ __launch_bounds__(128, 1)
void q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  __shared__ half_t weight_tile[3 * T16_COLS * QK_K];
  q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_body<4>(
      x, tiles, out, rows, in_features, out_features, weight_tile);
}

__global__ __launch_bounds__(128, 1)
void q6_k_t16_qmicro_planar_wmma_prefill_shared4_row64_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  __shared__ half_t weight_tile[3 * T16_COLS * QK_K];
  q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_body<1>(
      x, tiles, out, rows, in_features, out_features, weight_tile);
}

// Exact c=1 LM-head producer plus top-1 stage 1. Each existing 16-logit
// producer writes its unchanged FP32 logits and one tile maximum. A separate
// tiny reducer publishes the autoregressive control scalars without rereading
// the complete logits surface.
// The projection ownership and arithmetic above are intentionally duplicated
// here so this route cannot perturb the registered generic fallback. The six
// reserved pointer slots preserve the independently gated gfx1151 kernarg/
// codegen shape; they do not restore the rejected grid-completion owner.
__global__ __launch_bounds__(128, 4)
void q6_k_t16_gemv_top1_stage1_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    float* __restrict__ out,
    float* __restrict__ tile_values,
    int32_t* __restrict__ tile_indices,
    int64_t*,
    float*,
    int64_t*,
    int64_t*,
    int64_t*,
    int32_t*,
    int64_t in_features,
    int64_t out_features,
    int64_t) {
  const int64_t out_tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int g16 = k_in_block >> 4;
      const int8_t* __restrict__ scales =
          reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
      const float xv = bf16_bits_to_float(x[k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float d =
            fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
        const int scale_i =
            static_cast<int>(scales[g16 * T16_COLS + col]);
        const int q_i = t16_q6_quant(tile, k_in_block, col);
        acc[col] += xv * (d * static_cast<float>(scale_i * q_i));
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
    float best_value = -3.4028234663852886e+38F;
    int32_t best_index = INT32_MAX;
    const int sub_groups = blockDim.x >> 5;
    float* __restrict__ out_tile_row = out + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
      for (int g = 0; g < sub_groups; ++g) {
        total += xchg[g * T16_COLS + col];
      }
      out_tile_row[col] = total;
      const int32_t index =
          static_cast<int32_t>(out_tile * T16_COLS + col);
      if (better_pair(total, index, best_value, best_index)) {
        best_value = total;
        best_index = index;
      }
    }
    tile_values[out_tile] = best_value;
    tile_indices[out_tile] = best_index;
  }
}

// Planar-qmicro companion to the exact root-head producer/top-1 stage. It
// changes only record addressing; logits, tile maxima, tie order, and the
// ordered four-wave sum remain bit-identical to the legacy-T16 producer.
__global__ __launch_bounds__(128, 4)
void q6_k_t16_qmicro_planar_gemv_top1_stage1_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    float* __restrict__ out,
    float* __restrict__ tile_values,
    int32_t* __restrict__ tile_indices,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;

  float acc[T16_COLS];
#pragma unroll
  for (int col = 0; col < T16_COLS; ++col) acc[col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
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
          const float d =
              fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
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
    float best_value = -3.4028234663852886e+38F;
    int32_t best_index = INT32_MAX;
    float* __restrict__ out_tile_row = out + out_tile * T16_COLS;
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) {
      float total = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        total += xchg[g * T16_COLS + col];
      }
      out_tile_row[col] = total;
      const int32_t index =
          static_cast<int32_t>(out_tile * T16_COLS + col);
      if (better_pair(total, index, best_value, best_index)) {
        best_value = total;
        best_index = index;
      }
    }
    tile_values[out_tile] = best_value;
    tile_indices[out_tile] = best_index;
  }
}

template <typename scalar_t, typename out_t>
int launch_q6_t16(const void* x,
                  const void* tiles,
                  void* out,
                  int64_t rows,
                  int64_t in_features,
                  int64_t out_features,
                  hipStream_t stream) {
  if (!x || !tiles || !out || rows <= 0 || in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS), static_cast<unsigned int>(rows));
  hipLaunchKernelGGL((q6_k_t16_gemv_kernel<scalar_t, out_t>), grid, block, 0, stream,
                     reinterpret_cast<const scalar_t*>(x), reinterpret_cast<const uint8_t*>(tiles),
                     reinterpret_cast<out_t*>(out), rows, in_features, out_features);
  return hipGetLastError();
}

template <typename out_t>
int launch_q6_t16_qmicro_planar(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows <= 0 || in_features <= 0 ||
      out_features <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(out_features / T16_COLS),
      static_cast<unsigned int>(rows));
  hipLaunchKernelGGL(
      (q6_k_t16_qmicro_planar_gemv_bf16_kernel<out_t>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<out_t*>(out),
      rows,
      in_features,
      out_features);
  return hipGetLastError();
}

int launch_q6_t16_qmicro_planar_top1_stage1(
    const void* x,
    const void* tiles,
    void* out,
    void* tile_values,
    void* tile_indices,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || !tile_values || !tile_indices ||
      in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0 ||
      out_features > static_cast<int64_t>(INT32_MAX)) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS));
  hipLaunchKernelGGL(
      q6_k_t16_qmicro_planar_gemv_top1_stage1_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<float*>(out),
      reinterpret_cast<float*>(tile_values),
      reinterpret_cast<int32_t*>(tile_indices),
      in_features,
      out_features);
  return hipGetLastError();
}

int launch_q6_t16_qmicro_planar_residual(
    const void* x,
    const void* tiles,
    const void* residual,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !residual || !out || rows != 1 ||
      in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS));
  hipLaunchKernelGGL(
      q6_k_t16_qmicro_planar_gemv_bf16_residual_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<const uint16_t*>(residual),
      reinterpret_cast<uint16_t*>(out),
      in_features,
      out_features);
  return hipGetLastError();
}

int launch_q6_t16_wmma_prefill(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows <= 0 || in_features <= 0 ||
      out_features <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>(out_features / T16_COLS),
      static_cast<unsigned int>((rows + 15) / 16));
  hipLaunchKernelGGL(
      q6_k_t16_wmma_prefill_bf16_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<uint16_t*>(out),
      rows,
      in_features,
      out_features);
  return hipGetLastError();
}

int launch_q6_t16_wmma_prefill_shared4(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows <= 0 || in_features <= 0 ||
      out_features <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  constexpr int Q6_DENSE_TILE_COLS = 3 * T16_COLS;
  constexpr int Q6_DENSE_TILE_ROWS = 4 * 4 * 16;
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(
          (out_features + Q6_DENSE_TILE_COLS - 1) / Q6_DENSE_TILE_COLS),
      static_cast<unsigned int>(
          (rows + Q6_DENSE_TILE_ROWS - 1) / Q6_DENSE_TILE_ROWS));
  hipLaunchKernelGGL(
      q6_k_t16_wmma_prefill_shared4_bf16_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<uint16_t*>(out),
      rows,
      in_features,
      out_features);
  return hipGetLastError();
}

int launch_q6_t16_qmicro_planar_wmma_prefill(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows <= 0 || in_features <= 0 ||
      out_features <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(32);
  constexpr int Q6_DENSE_TILE_COLS = 3 * T16_COLS;
  constexpr int Q6_DENSE_TILE_ROWS = 4 * 16;
  const dim3 grid(
      static_cast<unsigned int>(
          (out_features + Q6_DENSE_TILE_COLS - 1) / Q6_DENSE_TILE_COLS),
      static_cast<unsigned int>(
          (rows + Q6_DENSE_TILE_ROWS - 1) / Q6_DENSE_TILE_ROWS));
  hipLaunchKernelGGL(
      q6_k_t16_qmicro_planar_wmma_prefill_bf16_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<uint16_t*>(out),
      rows,
      in_features,
      out_features);
  return hipGetLastError();
}

template <int ROW_TILES_PER_WAVE>
int launch_q6_t16_qmicro_planar_wmma_prefill_shared4_impl(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows <= 0 || in_features <= 0 ||
      out_features <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  constexpr int Q6_DENSE_TILE_COLS = 3 * T16_COLS;
  constexpr int Q6_DENSE_TILE_ROWS = 4 * ROW_TILES_PER_WAVE * 16;
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(
          (out_features + Q6_DENSE_TILE_COLS - 1) / Q6_DENSE_TILE_COLS),
      static_cast<unsigned int>(
          (rows + Q6_DENSE_TILE_ROWS - 1) / Q6_DENSE_TILE_ROWS));
  if constexpr (ROW_TILES_PER_WAVE == 1) {
    hipLaunchKernelGGL(
        q6_k_t16_qmicro_planar_wmma_prefill_shared4_row64_bf16_kernel,
        grid,
        block,
        0,
        stream,
        reinterpret_cast<const uint16_t*>(x),
        reinterpret_cast<const uint8_t*>(tiles),
        reinterpret_cast<uint16_t*>(out),
        rows,
        in_features,
        out_features);
  } else {
    static_assert(ROW_TILES_PER_WAVE == 4, "unsupported shared4 row tiling");
    hipLaunchKernelGGL(
        q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_kernel,
        grid,
        block,
        0,
        stream,
        reinterpret_cast<const uint16_t*>(x),
        reinterpret_cast<const uint8_t*>(tiles),
        reinterpret_cast<uint16_t*>(out),
        rows,
        in_features,
        out_features);
  }
  return hipGetLastError();
}

int launch_q6_t16_qmicro_planar_wmma_prefill_shared4(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q6_t16_qmicro_planar_wmma_prefill_shared4_impl<4>(
      x, tiles, out, rows, in_features, out_features, stream);
}

int launch_q6_t16_qmicro_planar_wmma_prefill_shared4_row64(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q6_t16_qmicro_planar_wmma_prefill_shared4_impl<1>(
      x, tiles, out, rows, in_features, out_features, stream);
}

int launch_q6_t16_top1_stage1(
    const void* x,
    const void* tiles,
    void* out,
    void* tile_values,
    void* tile_indices,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || !tile_values || !tile_indices ||
      in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0 ||
      out_features > static_cast<int64_t>(INT32_MAX)) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS));
  hipLaunchKernelGGL(
      q6_k_t16_gemv_top1_stage1_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<float*>(out),
      reinterpret_cast<float*>(tile_values),
      reinterpret_cast<int32_t*>(tile_indices),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      in_features,
      out_features,
      0);
  return hipGetLastError();
}

// Small-B weight-load-amortized rowtile variant: one block computes one 16-col
// output tile for ROW_TILE rows, reading each weight tile (and its dequant) ONCE
// and reusing across all rows. Bit-identical per (row,col) to
// q6_k_t16_gemv_kernel (same k order, same dequant, same wave reduce). For the
// small-B verifier path where the 417MB Q6_K lm-head would otherwise be re-read
// per row.
template <typename scalar_t, typename out_t, int ROW_TILE>
__global__ __launch_bounds__(128, 2) void q6_k_t16_gemv_rowtile_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;

  float acc[ROW_TILE][T16_COLS];
#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) acc[r][col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    const int8_t* __restrict__ scales = reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int g16 = k_in_block >> 4;
      float xr[ROW_TILE];
#pragma unroll
      for (int r = 0; r < ROW_TILE; ++r)
        xr[r] = scalar_to_float<scalar_t>(x[r * in_features + k_base + k_in_block]);
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        const float d = fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
        const int scale_i = static_cast<int>(scales[g16 * T16_COLS + col]);
        const int q_i = t16_q6_quant(tile, k_in_block, col);
        const float w = d * static_cast<float>(scale_i * q_i);
#pragma unroll
        for (int r = 0; r < ROW_TILE; ++r) acc[r][col] += xr[r] * w;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) GGUF_T16_WAVE_REDUCE(acc[r][col]);

  __shared__ float xchg[4 * ROW_TILE * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col)
        xchg[(sub_id * ROW_TILE + r) * T16_COLS + col] = acc[r][col];
  }
  __syncthreads();

  if (tid == 0) {
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      out_t* __restrict__ out_row = out + r * out_features + out_tile * T16_COLS;
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        float total = 0.0f;
        for (int g = 0; g < sub_groups; ++g)
          total += xchg[(g * ROW_TILE + r) * T16_COLS + col];
        out_row[col] = float_to_output<out_t>(total);
      }
    }
  }
}

// Exact column-split sibling of the small-B rowtile. Each local128 block keeps
// the production K partitions/reduction tree but owns eight columns rather than
// sixteen, reducing per-thread accumulator pressure while preserving every
// output's arithmetic order.
template <typename scalar_t, typename out_t, int ROW_TILE>
__global__ __launch_bounds__(128, 4) void q6_k_t16_gemv_rowtile_col8_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  const int64_t out_group = blockIdx.x;
  const int64_t out_tile = out_group >> 1;
  const int col_base = static_cast<int>(out_group & 1) * kTileCols;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;

  float acc[ROW_TILE][kTileCols];
#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col)
      acc[r][local_col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    const int8_t* __restrict__ scales =
        reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
    for (int k_in_block = tid; k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const int g16 = k_in_block >> 4;
      float xr[ROW_TILE];
#pragma unroll
      for (int r = 0; r < ROW_TILE; ++r) {
        xr[r] = scalar_to_float<scalar_t>(
            x[r * in_features + k_base + k_in_block]);
      }
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        const int col = col_base + local_col;
        const float d =
            fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
        const int scale_i =
            static_cast<int>(scales[g16 * T16_COLS + col]);
        const int q_i = t16_q6_quant(tile, k_in_block, col);
        const float w = d * static_cast<float>(scale_i * q_i);
#pragma unroll
        for (int r = 0; r < ROW_TILE; ++r) {
          acc[r][local_col] += xr[r] * w;
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col)
      GGUF_T16_WAVE_REDUCE(acc[r][local_col]);

  __shared__ float xchg[4 * ROW_TILE * kTileCols];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        xchg[(sub_id * ROW_TILE + r) * kTileCols + local_col] =
            acc[r][local_col];
      }
  }
  __syncthreads();

  if (tid == 0) {
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      out_t* __restrict__ out_row =
          out + r * out_features + out_group * kTileCols;
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        float total = 0.0f;
        for (int g = 0; g < sub_groups; ++g) {
          total += xchg[(g * ROW_TILE + r) * kTileCols + local_col];
        }
        out_row[local_col] = float_to_output<out_t>(total);
      }
    }
  }
}

// Exact planar-qmicro sibling of the generic 16-column rows=2 F32 rowtile.
// It retains the standard local128 K partitions, FMA stream, wave tree, and
// block reduction while decoding the byte-neutral K4/quartet records.
template <typename out_t, int ROW_TILE>
__global__ __launch_bounds__(128, 4)
void q6_k_t16_qmicro_planar_gemv_rowtile_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_tile = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;

  float acc[ROW_TILE][T16_COLS];
#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col) acc[r][col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    const int8_t* __restrict__ scales =
        reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
    for (int k_in_block = tid; k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const int g16 = k_in_block >> 4;
      const int subblock = k_in_block >> 5;
      const int pack = (k_in_block & 31) >> 2;
      const int k4 = k_in_block & 3;
      float xr[ROW_TILE];
#pragma unroll
      for (int r = 0; r < ROW_TILE; ++r) {
        xr[r] = bf16_bits_to_float(
            x[r * in_features + k_base + k_in_block]);
      }
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
          const float d =
              fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
          const int scale_i =
              static_cast<int>(scales[g16 * T16_COLS + col]);
          const float w = d * static_cast<float>(scale_i * q_i);
#pragma unroll
          for (int r = 0; r < ROW_TILE; ++r) {
            acc[r][col] += xr[r] * w;
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
    for (int col = 0; col < T16_COLS; ++col)
      GGUF_T16_WAVE_REDUCE(acc[r][col]);

  __shared__ float xchg[4 * ROW_TILE * T16_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        xchg[(sub_id * ROW_TILE + r) * T16_COLS + col] = acc[r][col];
      }
  }
  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      out_t* __restrict__ out_row =
          out + r * out_features + out_tile * T16_COLS;
#pragma unroll
      for (int col = 0; col < T16_COLS; ++col) {
        float total = 0.0f;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          total += xchg[(g * ROW_TILE + r) * T16_COLS + col];
        }
        out_row[col] = float_to_output<out_t>(total);
      }
    }
  }
}

// Exact planar-qmicro sibling of the col8 verifier rowtile. The byte-neutral
// payload groups each K4/column-quartet as four QL01 bytes, four QL23 bytes,
// then four QH bytes. Every row/output retains the legacy col8 K traversal,
// integer coefficient product, FP32 FMA stream, wave tree, and BF16 boundary.
template <
    typename out_t,
    int ROW_TILE,
    bool USE_DPP_REDUCTION = false,
    bool PARALLEL_EPILOGUE = false>
__global__ __launch_bounds__(128, 4)
void q6_k_t16_qmicro_planar_gemv_rowtile_col8_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  constexpr int kQuartetsPerBlock = kTileCols / 4;
  const int64_t out_group = blockIdx.x;
  const int64_t out_tile = out_group >> 1;
  const int col_base = static_cast<int>(out_group & 1) * kTileCols;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;

  float acc[ROW_TILE][kTileCols];
#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col)
      acc[r][local_col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    const int8_t* __restrict__ scales =
        reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
    for (int k_in_block = tid; k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const int g16 = k_in_block >> 4;
      const int subblock = k_in_block >> 5;
      const int pack = (k_in_block & 31) >> 2;
      const int k4 = k_in_block & 3;
      float xr[ROW_TILE];
#pragma unroll
      for (int r = 0; r < ROW_TILE; ++r) {
        xr[r] = bf16_bits_to_float(
            x[r * in_features + k_base + k_in_block]);
      }
#pragma unroll
      for (int local_quartet = 0; local_quartet < kQuartetsPerBlock;
           ++local_quartet) {
        const int col_quartet = (col_base >> 2) + local_quartet;
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
          const int local_col = local_quartet * 4 + col4;
          const int col = col_base + local_col;
          const uint8_t low_byte =
              static_cast<uint8_t>(low_pair >> (8 * (col4 >> 1)));
          const uint8_t low =
              (col4 & 1) ? (low_byte >> 4) : (low_byte & 0x0Fu);
          const uint8_t high =
              (high_packed >> (2 * col4)) & 0x03u;
          const int q_i = static_cast<int>(low | (high << 4)) - 32;
          const float d =
              fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
          const int scale_i =
              static_cast<int>(scales[g16 * T16_COLS + col]);
          const float w = d * static_cast<float>(scale_i * q_i);
#pragma unroll
          for (int r = 0; r < ROW_TILE; ++r) {
            acc[r][local_col] += xr[r] * w;
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      if constexpr (USE_DPP_REDUCTION) {
        q6_t16_wave_reduce_dpp(acc[r][local_col]);
      } else {
        GGUF_T16_WAVE_REDUCE(acc[r][local_col]);
      }
    }

  __shared__ float xchg[4 * ROW_TILE * kTileCols];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        xchg[(sub_id * ROW_TILE + r) * kTileCols + local_col] =
            acc[r][local_col];
      }
  }
  __syncthreads();

  if constexpr (PARALLEL_EPILOGUE) {
    if (tid < ROW_TILE * kTileCols) {
      const int r = tid / kTileCols;
      const int local_col = tid % kTileCols;
      float total = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        total += xchg[(g * ROW_TILE + r) * kTileCols + local_col];
      }
      out[r * out_features + out_group * kTileCols + local_col] =
          float_to_output<out_t>(total);
    }
  } else if (tid == 0) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      out_t* __restrict__ out_row =
          out + r * out_features + out_group * kTileCols;
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        float total = 0.0f;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          total += xchg[(g * ROW_TILE + r) * kTileCols + local_col];
        }
        out_row[local_col] = float_to_output<out_t>(total);
      }
    }
  }
}

// Approximate dense Q6 sibling using GGML-compatible Q8_1 activations and the
// RDNA3 scalar integer-dot primitive. It consumes the existing byte-neutral
// planar-qmicro resident bytes: each aligned 12-byte K4/column-quartet record
// yields four directly loadable low-plane bytes plus four high-plane bytes.
// One thread policy is shared by c1 and verifier rows for each measured shape,
// so every row reproduces the same X8 arithmetic and reduction association.
template <int ROW_TILE, bool ADD_RESIDUAL>
__launch_bounds__(256, 2)
__global__ void q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const uint8_t* __restrict__ tiles,
    const uint16_t* __restrict__ residual,
    uint16_t* __restrict__ out,
    int in_features,
    int out_features,
    int q8_blocks_per_row) {
  constexpr int kTileCols = 8;
  const int out_group = blockIdx.x;
  const int out_tile = out_group >> 1;
  const int col_base = (out_group & 1) * kTileCols;
  const int tid = threadIdx.x;
  const int blocks_per_row = in_features / QK_K;
  const int groups4 = in_features >> 2;

  float acc[ROW_TILE][kTileCols] = {};
  for (int group = tid; group < groups4; group += blockDim.x) {
    const int block_idx = group >> 6;
    const int within4 = group & 63;
    const int subblock = within4 >> 3;
    const int pack = within4 & 7;
    const int lane4 = pack << 2;
    const int scale_index = within4 >> 2;
    const int q8_index = block_idx * 8 + subblock;
    const uint8_t* __restrict__ tile =
        tiles +
        (static_cast<int64_t>(out_tile) * blocks_per_row + block_idx) *
            Q6_T16_BLOCK_BYTES;
    const int8_t* __restrict__ scales =
        reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);

    uint32_t qpack[kTileCols];
#pragma unroll
    for (int local_quartet = 0; local_quartet < 2; ++local_quartet) {
      const int col_quartet = (col_base >> 2) + local_quartet;
      const int record_index =
          (subblock * (T16_COLS / 4) + col_quartet) * 8 + pack;
      const uint8_t* __restrict__ record =
          tile + Q6_T16_QMICRO_OFFSET +
          record_index * Q6_T16_QMICRO_RECORD_BYTES;
      const uint32_t low01 =
          *reinterpret_cast<const uint32_t*>(record + 0);
      const uint32_t low23 =
          *reinterpret_cast<const uint32_t*>(record + 4);
      const uint32_t high =
          *reinterpret_cast<const uint32_t*>(record + 8);
#pragma unroll
      for (int col4 = 0; col4 < 4; ++col4) {
        qpack[local_quartet * 4 + col4] =
            q6_planar_pack4_unsigned(low01, low23, high, col4);
      }
    }

    float d[kTileCols];
    float scale[kTileCols];
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col) {
      const int col = col_base + local_col;
      d[local_col] =
          fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
      scale[local_col] = static_cast<float>(
          scales[scale_index * T16_COLS + col]);
    }

#pragma unroll
    for (int row = 0; row < ROW_TILE; ++row) {
      const gguf_q8_1_block* __restrict__ xb =
          xq + static_cast<int64_t>(row) * q8_blocks_per_row + q8_index;
      const uint32_t x_pack =
          *reinterpret_cast<const uint32_t*>(xb->qs + lane4);
      const int q8_sum =
          q6_planar_dp4a_u8_s8(0x01010101u, x_pack, 0);
      const float xd = static_cast<float>(xb->d);
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        const int dot =
            q6_planar_dp4a_u8_s8(qpack[local_col], x_pack, 0) -
            32 * q8_sum;
        acc[row][local_col] +=
            xd * d[local_col] * scale[local_col] *
            static_cast<float>(dot);
      }
    }
  }

  __shared__ float xchg[8 * 8];
#pragma unroll
  for (int row = 0; row < ROW_TILE; ++row) {
    float r0, r1, r2, r3, r4, r5, r6, r7;
    q6_planar_reduce8(
        acc[row][0],
        acc[row][1],
        acc[row][2],
        acc[row][3],
        acc[row][4],
        acc[row][5],
        acc[row][6],
        acc[row][7],
        xchg,
        tid,
        r0,
        r1,
        r2,
        r3,
        r4,
        r5,
        r6,
        r7);
    if (tid == 0) {
      const int64_t base =
          static_cast<int64_t>(row) * out_features +
          static_cast<int64_t>(out_group) * kTileCols;
      const float values[kTileCols] = {r0, r1, r2, r3, r4, r5, r6, r7};
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        uint16_t projected = float_to_bf16_bits(values[local_col]);
        if constexpr (ADD_RESIDUAL) {
          projected = float_to_bf16_bits(
              bf16_bits_to_float(residual[base + local_col]) +
              bf16_bits_to_float(projected));
        }
        out[base + local_col] = projected;
      }
    }
    __syncthreads();
  }
}

// Exact FFN-down plus rounded-BF16 residual sibling. The projection body is
// intentionally separate from the registered unfused rowtile so that fallback
// codegen remains frozen. Each output preserves the complete Q6 dot/reduction,
// rounds that projection to BF16, then performs the same BF16 add and final
// BF16 rounding as the standalone residual primitive.
template <int ROW_TILE>
__global__ __launch_bounds__(128, 4)
void q6_k_t16_qmicro_planar_gemv_rowtile_col8_residual_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    const uint16_t* __restrict__ residual,
    uint16_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features) {
  constexpr int kTileCols = 8;
  constexpr int kQuartetsPerBlock = kTileCols / 4;
  const int64_t out_group = blockIdx.x;
  const int64_t out_tile = out_group >> 1;
  const int col_base = static_cast<int>(out_group & 1) * kTileCols;
  const int tid = threadIdx.x;
  const int64_t blocks_per_row = in_features / QK_K;

  float acc[ROW_TILE][kTileCols];
#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col)
      acc[r][local_col] = 0.0f;

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* __restrict__ tile =
        tiles + (out_tile * blocks_per_row + blk) * Q6_T16_BLOCK_BYTES;
    const int64_t k_base = blk * QK_K;
    const int8_t* __restrict__ scales =
        reinterpret_cast<const int8_t*>(tile + Q6_T16_SCALE_OFFSET);
    for (int k_in_block = tid; k_in_block < QK_K;
         k_in_block += blockDim.x) {
      const int g16 = k_in_block >> 4;
      const int subblock = k_in_block >> 5;
      const int pack = (k_in_block & 31) >> 2;
      const int k4 = k_in_block & 3;
      float xr[ROW_TILE];
#pragma unroll
      for (int r = 0; r < ROW_TILE; ++r) {
        xr[r] = bf16_bits_to_float(
            x[r * in_features + k_base + k_in_block]);
      }
#pragma unroll
      for (int local_quartet = 0; local_quartet < kQuartetsPerBlock;
           ++local_quartet) {
        const int col_quartet = (col_base >> 2) + local_quartet;
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
          const int local_col = local_quartet * 4 + col4;
          const int col = col_base + local_col;
          const uint8_t low_byte =
              static_cast<uint8_t>(low_pair >> (8 * (col4 >> 1)));
          const uint8_t low =
              (col4 & 1) ? (low_byte >> 4) : (low_byte & 0x0Fu);
          const uint8_t high =
              (high_packed >> (2 * col4)) & 0x03u;
          const int q_i = static_cast<int>(low | (high << 4)) - 32;
          const float d =
              fp16_bytes_to_float(tile + Q6_T16_D_OFFSET + col * 2);
          const int scale_i =
              static_cast<int>(scales[g16 * T16_COLS + col]);
          const float w = d * static_cast<float>(scale_i * q_i);
#pragma unroll
          for (int r = 0; r < ROW_TILE; ++r) {
            acc[r][local_col] += xr[r] * w;
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
    for (int local_col = 0; local_col < kTileCols; ++local_col)
      GGUF_T16_WAVE_REDUCE(acc[r][local_col]);

  __shared__ float xchg[4 * ROW_TILE * kTileCols];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r)
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        xchg[(sub_id * ROW_TILE + r) * kTileCols + local_col] =
            acc[r][local_col];
      }
  }
  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      const int64_t row_base =
          static_cast<int64_t>(r) * out_features + out_group * kTileCols;
#pragma unroll
      for (int local_col = 0; local_col < kTileCols; ++local_col) {
        float total = 0.0f;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          total += xchg[(g * ROW_TILE + r) * kTileCols + local_col];
        }
        const int64_t idx = row_base + local_col;
        const uint16_t projected = float_to_bf16_bits(total);
        out[idx] = float_to_bf16_bits(
            bf16_bits_to_float(residual[idx]) +
            bf16_bits_to_float(projected));
      }
    }
  }
}

template <typename scalar_t, typename out_t>
int launch_q6_t16_rowtile(const void* x,
                          const void* tiles,
                          void* out,
                          int64_t rows,
                          int64_t in_features,
                          int64_t out_features,
                          hipStream_t stream) {
  if (!x || !tiles || !out || rows < 2 || rows > 8 || in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS));
  const scalar_t* xp = reinterpret_cast<const scalar_t*>(x);
  const uint8_t* tp = reinterpret_cast<const uint8_t*>(tiles);
  out_t* op = reinterpret_cast<out_t*>(out);
#define Q6_T16_ROWTILE_CASE(N)                                                 \
  case N:                                                                      \
    hipLaunchKernelGGL((q6_k_t16_gemv_rowtile_kernel<scalar_t, out_t, N>),     \
                       grid, block, 0, stream, xp, tp, op, rows, in_features,  \
                       out_features);                                          \
    break;
  switch (static_cast<int>(rows)) {
    Q6_T16_ROWTILE_CASE(2)
    Q6_T16_ROWTILE_CASE(3)
    Q6_T16_ROWTILE_CASE(4)
    Q6_T16_ROWTILE_CASE(5)
    Q6_T16_ROWTILE_CASE(6)
    Q6_T16_ROWTILE_CASE(7)
    Q6_T16_ROWTILE_CASE(8)
    default:
      return hipErrorInvalidValue;
  }
#undef Q6_T16_ROWTILE_CASE
  return hipGetLastError();
}

template <typename scalar_t, typename out_t>
int launch_q6_t16_rowtile_col8(const void* x,
                               const void* tiles,
                               void* out,
                               int64_t rows,
                               int64_t in_features,
                               int64_t out_features,
                               hipStream_t stream) {
  if (!x || !tiles || !out || rows < 2 || rows > 8 || in_features <= 0 ||
      out_features <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
  const scalar_t* xp = reinterpret_cast<const scalar_t*>(x);
  const uint8_t* tp = reinterpret_cast<const uint8_t*>(tiles);
  out_t* op = reinterpret_cast<out_t*>(out);
#define Q6_T16_ROWTILE_COL8_CASE(N)                                           \
  case N:                                                                      \
    hipLaunchKernelGGL(                                                        \
        (q6_k_t16_gemv_rowtile_col8_kernel<scalar_t, out_t, N>),              \
        grid, block, 0, stream, xp, tp, op, rows, in_features, out_features);  \
    break;
  switch (static_cast<int>(rows)) {
    Q6_T16_ROWTILE_COL8_CASE(2)
    Q6_T16_ROWTILE_COL8_CASE(3)
    Q6_T16_ROWTILE_COL8_CASE(4)
    Q6_T16_ROWTILE_COL8_CASE(5)
    Q6_T16_ROWTILE_COL8_CASE(6)
    Q6_T16_ROWTILE_COL8_CASE(7)
    Q6_T16_ROWTILE_COL8_CASE(8)
    default:
      return hipErrorInvalidValue;
  }
#undef Q6_T16_ROWTILE_COL8_CASE
  return hipGetLastError();
}

int launch_q6_t16_qmicro_planar_rowtile_col8_residual(
    const void* x,
    const void* tiles,
    const void* residual,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !residual || !out || rows < 2 || rows > 4 ||
      in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 || (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
#define Q6_T16_QMICRO_PLANAR_RESIDUAL_CASE(N)                                \
  case N:                                                                     \
    hipLaunchKernelGGL(                                                       \
        (q6_k_t16_qmicro_planar_gemv_rowtile_col8_residual_kernel<N>),       \
        grid, block, 0, stream,                                               \
        reinterpret_cast<const uint16_t*>(x),                                \
        reinterpret_cast<const uint8_t*>(tiles),                             \
        reinterpret_cast<const uint16_t*>(residual),                         \
        reinterpret_cast<uint16_t*>(out), in_features, out_features);        \
    break
  switch (static_cast<int>(rows)) {
    Q6_T16_QMICRO_PLANAR_RESIDUAL_CASE(2);
    Q6_T16_QMICRO_PLANAR_RESIDUAL_CASE(3);
    Q6_T16_QMICRO_PLANAR_RESIDUAL_CASE(4);
    default:
      return hipErrorInvalidValue;
  }
#undef Q6_T16_QMICRO_PLANAR_RESIDUAL_CASE
  return hipGetLastError();
}

template <typename out_t>
int launch_q6_t16_qmicro_planar_rowtile(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows != 2 || in_features <= 0 ||
      out_features <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / T16_COLS));
  hipLaunchKernelGGL(
      (q6_k_t16_qmicro_planar_gemv_rowtile_kernel<out_t, 2>),
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<out_t*>(out),
      rows,
      in_features,
      out_features);
  return hipGetLastError();
}

template <typename out_t>
int launch_q6_t16_qmicro_planar_rowtile_col8(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows < 2 || rows > 8 || in_features <= 0 ||
      out_features <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x);
  const uint8_t* tp = reinterpret_cast<const uint8_t*>(tiles);
  out_t* op = reinterpret_cast<out_t*>(out);
#define Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_CASE(N)                            \
  case N:                                                                      \
    hipLaunchKernelGGL(                                                        \
        (q6_k_t16_qmicro_planar_gemv_rowtile_col8_kernel<out_t, N>),          \
        grid, block, 0, stream, xp, tp, op, rows, in_features, out_features); \
    break;
  switch (static_cast<int>(rows)) {
    Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_CASE(2)
    Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_CASE(3)
    Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_CASE(4)
    Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_CASE(5)
    Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_CASE(6)
    Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_CASE(7)
    case 8:
      hipLaunchKernelGGL(
          (q6_k_t16_qmicro_planar_gemv_rowtile_col8_kernel<out_t, 8, true>),
          grid, block, 0, stream, xp, tp, op, rows, in_features, out_features);
      break;
    default:
      return hipErrorInvalidValue;
  }
#undef Q6_T16_QMICRO_PLANAR_ROWTILE_COL8_CASE
  return hipGetLastError();
}

template <typename out_t>
int launch_q6_t16_qmicro_planar_rowtile_col8_parallel_epilogue_r6(
    const void* x,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!x || !tiles || !out || rows != 6 || in_features <= 0 ||
      out_features <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0) {
    return hipErrorInvalidValue;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
  hipLaunchKernelGGL(
      (q6_k_t16_qmicro_planar_gemv_rowtile_col8_kernel<
          out_t, 6, false, true>),
      grid, block, 0, stream,
      reinterpret_cast<const uint16_t*>(x),
      reinterpret_cast<const uint8_t*>(tiles),
      reinterpret_cast<out_t*>(out), rows, in_features, out_features);
  return hipGetLastError();
}

template <bool ADD_RESIDUAL>
int launch_q6_t16_qmicro_planar_q8_1_dp4a(
    const void* xq,
    const void* tiles,
    const void* residual,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  const int64_t min_rows = ADD_RESIDUAL ? 2 : 1;
  if (!xq || !tiles || !out || (ADD_RESIDUAL && !residual) ||
      rows < min_rows || rows > 4 || in_features <= 0 ||
      out_features <= 0 || (in_features % QK_K) != 0 ||
      (out_features % T16_COLS) != 0 ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return hipErrorInvalidValue;
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_features / 8));
  const int q8_blocks_per_row = static_cast<int>(in_features / Q8_1_BLOCK);
#define Q6_PLANAR_Q8_1_CASE(N)                                                \
  case N:                                                                     \
    hipLaunchKernelGGL(                                                       \
        (q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_kernel<N, ADD_RESIDUAL>),     \
        grid,                                                                 \
        block,                                                                \
        0,                                                                    \
        stream,                                                               \
        reinterpret_cast<const gguf_q8_1_block*>(xq),                        \
        reinterpret_cast<const uint8_t*>(tiles),                             \
        reinterpret_cast<const uint16_t*>(residual),                         \
        reinterpret_cast<uint16_t*>(out),                                    \
        static_cast<int>(in_features),                                       \
        static_cast<int>(out_features),                                      \
        q8_blocks_per_row);                                                   \
    break
  switch (static_cast<int>(rows)) {
    Q6_PLANAR_Q8_1_CASE(1);
    Q6_PLANAR_Q8_1_CASE(2);
    Q6_PLANAR_Q8_1_CASE(3);
    Q6_PLANAR_Q8_1_CASE(4);
    default:
      return hipErrorInvalidValue;
  }
#undef Q6_PLANAR_Q8_1_CASE
  return hipGetLastError();
}

}  // namespace

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q6_t16_qmicro_planar<uint16_t>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_residual_bf16_out(
    const void* x, const void* tiles, const void* residual, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q6_t16_qmicro_planar_residual(
      x, tiles, residual, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q6_t16_qmicro_planar<float>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_top1_stage1(
    const void* x,
    const void* tiles,
    void* out,
    void* tile_values,
    void* tile_indices,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q6_t16_qmicro_planar_top1_stage1(
      x,
      tiles,
      out,
      tile_values,
      tile_indices,
      in_features,
      out_features,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q6_t16_qmicro_planar_wmma_prefill(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q6_t16_qmicro_planar_wmma_prefill_shared4(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_row64_bf16_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q6_t16_qmicro_planar_wmma_prefill_shared4_row64(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_wmma_prefill_shared4_bf16_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q6_t16_wmma_prefill_shared4(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_wmma_prefill_bf16_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q6_t16_wmma_prefill(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_gemv_rowtile_bf16_f32_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  if (rows >= 3 && rows <= 4) {
    return launch_q6_t16_rowtile_col8<uint16_t, float>(
        x, tiles, out, rows, in_features, out_features, stream);
  }
  return launch_q6_t16_rowtile<uint16_t, float>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_gemv_rowtile_bf16_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  if (rows >= 2 && rows <= 5) {
    return launch_q6_t16_rowtile_col8<uint16_t, uint16_t>(
        x, tiles, out, rows, in_features, out_features, stream);
  }
  return launch_q6_t16_rowtile<uint16_t, uint16_t>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_gemv_rowtile_col8_bf16_f32_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q6_t16_rowtile_col8<uint16_t, float>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_gemv_rowtile_col8_bf16_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q6_t16_rowtile_col8<uint16_t, uint16_t>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  if (rows >= 2 && rows <= 8) {
    return launch_q6_t16_qmicro_planar_rowtile_col8<uint16_t>(
        x, tiles, out, rows, in_features, out_features, stream);
  }
  return launch_q6_t16_qmicro_planar<uint16_t>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_parallel_epilogue_r6_bf16_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  return launch_q6_t16_qmicro_planar_rowtile_col8_parallel_epilogue_r6<
      uint16_t>(x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_bf16_out(
    const void* xq,
    const void* tiles,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_q6_t16_qmicro_planar_q8_1_dp4a<false>(
      xq,
      tiles,
      nullptr,
      out,
      rows,
      in_features,
      out_features,
      threads,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_bf16_residual_bf16_out(
    const void* xq,
    const void* tiles,
    const void* residual,
    void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_q6_t16_qmicro_planar_q8_1_dp4a<true>(
      xq,
      tiles,
      residual,
      out,
      rows,
      in_features,
      out_features,
      threads,
      stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_residual_bf16_out(
    const void* x, const void* tiles, const void* residual, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    hipStream_t stream) {
  return launch_q6_t16_qmicro_planar_rowtile_col8_residual(
      x, tiles, residual, out, rows, in_features, out_features, stream);
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  if (rows == 2) {
    return launch_q6_t16_qmicro_planar_rowtile<float>(
        x, tiles, out, rows, in_features, out_features, stream);
  }
  if (rows >= 3 && rows <= 8) {
    return launch_q6_t16_qmicro_planar_rowtile_col8<float>(
        x, tiles, out, rows, in_features, out_features, stream);
  }
  return hipErrorInvalidValue;
}

extern "C" int
hipengine_gguf_q6_k_t16_qmicro_planar_gemv_rowtile_col8_bf16_f32_out(
    const void* x, const void* tiles, void* out, int64_t rows,
    int64_t in_features, int64_t out_features, hipStream_t stream) {
  if (rows >= 2 && rows <= 4) {
    return launch_q6_t16_qmicro_planar_rowtile_col8<float>(
        x, tiles, out, rows, in_features, out_features, stream);
  }
  return launch_q6_t16_qmicro_planar<float>(
      x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_gemv_decode_bf16_f32_out(
    const void* x, const void* tiles, void* out, int64_t rows, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_q6_t16<uint16_t, float>(x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_gemv_decode_bf16_bf16_out(
    const void* x, const void* tiles, void* out, int64_t rows, int64_t in_features,
    int64_t out_features, hipStream_t stream) {
  return launch_q6_t16<uint16_t, uint16_t>(x, tiles, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_t16_gemv_decode_bf16_f32_top1_stage1(
    const void* x,
    const void* tiles,
    void* out,
    void* tile_values,
    void* tile_indices,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q6_t16_top1_stage1(
      x,
      tiles,
      out,
      tile_values,
      tile_indices,
      in_features,
      out_features,
      stream);
}
