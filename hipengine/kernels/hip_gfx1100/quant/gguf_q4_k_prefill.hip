// Batched WMMA prefill GEMM for raw GGUF Q4_K weights on gfx1100.
//
// Mirrors the structure of awq_fusedw4_prefill_fp16_kernel and
// awq_fusedw4_prefill_dual_fp16_kernel (see
// hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip): 32 threads per
// block (one wave32), each block computes a (AWQ_TILE_M x AWQ_TILE_N)
// output tile via __builtin_amdgcn_wmma_f32_16x16x16_f16_w32. The only
// algorithmic difference from the AWQ template is the inner K-tile dequant
// block, which here reads raw GGUF block_q4_K bytes (144 bytes per 256 K:
// fp16 d + fp16 dmin + 12 packed scale/min bytes + 128 packed q4 bytes).
//
// See docs/GGUF.md "P8: real batched prefill GEMM" and the task #6
// WORKLOG audit for the raw block layout and dequant/addressing plan.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

constexpr int QK_K = 256;
constexpr int Q4_K_BLOCK_BYTES = 144;
constexpr int Q4_K_SCALE_BYTES = 12;
constexpr int Q4_K_QS_OFFSET = 2 + 2 + Q4_K_SCALE_BYTES;
constexpr int Q4_K_SUBBLOCK = 32;
constexpr int Q4_K_SUBBLOCKS = QK_K / Q4_K_SUBBLOCK;  // == 8
constexpr int Q6_K_BLOCK_BYTES = 210;
constexpr int K_TILE = 16;                            // WMMA inner-K size
constexpr int K_TILES_PER_SUBBLOCK = Q4_K_SUBBLOCK / K_TILE;  // == 2

// ---- scalar conversion helpers (same shape as gguf_q8_0_prefill.hip) ----

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

template <typename out_t>
__device__ inline void store_output(out_t* out, int64_t idx, float value) {
  out[idx] = static_cast<out_t>(value);
}

template <>
__device__ inline void store_output<uint16_t>(uint16_t* out, int64_t idx, float value) {
  out[idx] = float_to_bf16_bits(value);
}

template <typename scalar_t>
__device__ inline half16_t load_act_half16(const scalar_t* p) {
  half16_t out;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[i] = static_cast<half_t>(scalar_to_float<scalar_t>(p[i]));
  }
  return out;
}

template <>
__device__ inline half16_t load_act_half16<half_t>(const half_t* p) {
  return *reinterpret_cast<const half16_t*>(p);
}

// GGML/GGUF Q4_K packed 6-bit scale/min helpers. These must match
// gguf_q4_k_gemv.hip and hipengine.quant.gguf._dequant_q4_k exactly.
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

__device__ inline int gguf_q6_k_quant(
    const uint8_t* block,
    int within) {
  const int group32 = within >> 5;
  const int lane = within & 31;
  const uint8_t* ql = block;
  const uint8_t* qh = block + 128;
  const int base64 = group32 >= 4 ? 64 : 0;
  const bool low_nibble = (group32 & 2) == 0;
  const int ql_group = group32 & 1;
  const uint8_t ql_byte =
      ql[base64 + ql_group * 32 + lane];
  const uint8_t low =
      low_nibble ? (ql_byte & 0x0Fu) : (ql_byte >> 4);
  const int qh_base = group32 >= 4 ? 32 : 0;
  const uint8_t high =
      (qh[qh_base + lane] >> (2 * (group32 & 3))) & 0x03u;
  return static_cast<int>(low | (high << 4)) - 32;
}

__device__ inline int pack8_shift_for_lane(int lane) {
  const int packed_pos = (lane & 1) ? (4 + (lane >> 1)) : (lane >> 1);
  return packed_pos * 4;
}

// -------------------------------------------------------------------------
// gguf_q4_k_prefill_wmma_kernel
//
// One block (wave32) computes a TM x TN tile of `out`, where
//   AWQ_TILE_M = TM rows of out_features (output columns of GEMM),
//   AWQ_TILE_N = TN rows of `rows` (token rows / activation rows).
// `qweight` is raw block_q4_K byte stream: per output row `c`,
//   row_bytes = (in_features / 256) * 144
//   block b at offset c * row_bytes + b * 144
//     [0..1]     fp16 d
//     [2..3]     fp16 dmin
//     [4..15]    uint8 scales[12]
//     [16..143]  uint8 qs[128]
// -------------------------------------------------------------------------
template <typename scalar_t, typename out_t, int AWQ_TILE_M, int AWQ_TILE_N>
__launch_bounds__(32, 4)
__global__ void gguf_q4_k_prefill_wmma_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  constexpr int TM = AWQ_TILE_M / 16;
  constexpr int TN = AWQ_TILE_N / 16;
  const int out_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int row_start = row_tile * AWQ_TILE_N;
  const int out_start = out_tile * AWQ_TILE_M;
  if (row_start >= rows || out_start >= out_features) return;

  const int blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * Q4_K_BLOCK_BYTES;

  int my_out[TM];
  int safe_out[TM];
  #pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    my_out[tm] = out_start + tm * 16 + (tid & 15);
    safe_out[tm] = (my_out[tm] < out_features) ? my_out[tm] : (out_features - 1);
  }

  int my_token[TN];
  int safe_token[TN];
  #pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    my_token[tn] = row_start + tn * 16 + (tid & 15);
    safe_token[tn] = (my_token[tn] < rows) ? my_token[tn] : (rows - 1);
  }

  float8_t acc[TM][TN];
  #pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    #pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      acc[tm][tn] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
  }

  for (int blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* block_ptr[TM];
    float d_f[TM];
    float dmin_f[TM];
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      block_ptr[tm] = qweight + static_cast<int64_t>(safe_out[tm]) * row_bytes +
                      static_cast<int64_t>(blk) * Q4_K_BLOCK_BYTES;
      d_f[tm] = fp16_bytes_to_float(block_ptr[tm]);
      dmin_f[tm] = fp16_bytes_to_float(block_ptr[tm] + 2);
    }

    const int g_k_base = blk * QK_K;

    #pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      float scale_f[TM];
      float min_f[TM];
      #pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        const uint8_t* scales = block_ptr[tm] + 4;
        scale_f[tm] = d_f[tm] * static_cast<float>(gguf_q4_k_scale(scales, sb));
        min_f[tm] = dmin_f[tm] * static_cast<float>(gguf_q4_k_min(scales, sb));
      }

      #pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_K_SUBBLOCK + kt * K_TILE;

        half16_t b_reg[TN];
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
          const scalar_t* x_row =
              x + static_cast<int64_t>(safe_token[tn]) * in_features;
          b_reg[tn] = load_act_half16<scalar_t>(x_row + g_k_base + k_off);
        }

        half16_t a_reg;
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
          const uint8_t* qs = block_ptr[tm] + Q4_K_QS_OFFSET;
          #pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            const int lane32 = kt * K_TILE + kk;
            const uint8_t q = gguf_q4_k_quant(qs, sb, lane32);
            a_reg[kk] = static_cast<half_t>(
                scale_f[tm] * static_cast<float>(q) - min_f[tm]);
          }
          #pragma unroll
          for (int tn = 0; tn < TN; ++tn) {
            acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                a_reg, b_reg[tn], acc[tm][tn]);
          }
        }
      }
    }
  }

  #pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    const int out_row = row_start + tn * 16 + (tid & 15);
    if (out_row >= rows) continue;
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = out_start + tm * 16 + 2 * j + (tid >> 4);
        if (col < out_features) {
          const int64_t idx =
              static_cast<int64_t>(out_row) * out_features + col;
          store_output<out_t>(out, idx, static_cast<float>(acc[tm][tn][j]));
        }
      }
    }
  }
}

// Raw-Q6_K dense sibling. Q6_K has one signed scale per K16, so each WMMA K
// step reads that scale once per output column and expands the 6-bit values
// directly into the operand fragment. A 64x16 output tile amortizes every
// decoded weight fragment across sixteen activation rows while retaining the
// source GGUF byte stream as the sole resident representation.
template <typename scalar_t, typename out_t, int TILE_M, int TILE_N>
__launch_bounds__(32, 4)
__global__ void gguf_q6_k_prefill_wmma_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  constexpr int TM = TILE_M / 16;
  constexpr int TN = TILE_N / 16;
  const int tid = threadIdx.x;
  const int lane16 = tid & 15;
  const int out_start = static_cast<int>(blockIdx.x) * TILE_M;
  const int row_start = static_cast<int>(blockIdx.y) * TILE_N;
  if (out_start >= out_features || row_start >= rows) return;

  const int blocks_per_row = in_features / QK_K;
  const int64_t row_bytes =
      static_cast<int64_t>(blocks_per_row) * Q6_K_BLOCK_BYTES;

  int safe_out[TM];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    const int col = out_start + tm * 16 + lane16;
    safe_out[tm] = col < out_features ? col : out_features - 1;
  }

  int safe_token[TN];
#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    const int row = row_start + tn * 16 + lane16;
    safe_token[tn] = row < rows ? row : rows - 1;
  }

  float8_t acc[TM][TN];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      acc[tm][tn] = float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
  }

  for (int blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* block_ptr[TM];
    float d_f[TM];
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      block_ptr[tm] =
          qweight + static_cast<int64_t>(safe_out[tm]) * row_bytes +
          static_cast<int64_t>(blk) * Q6_K_BLOCK_BYTES;
      d_f[tm] = fp16_bytes_to_float(block_ptr[tm] + 208);
    }

    const int g_k_base = blk * QK_K;
#pragma unroll
    for (int kt = 0; kt < QK_K / K_TILE; ++kt) {
      const int k_off = kt * K_TILE;
      half16_t b_reg[TN];
#pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
        const scalar_t* x_row =
            x + static_cast<int64_t>(safe_token[tn]) * in_features;
        b_reg[tn] = load_act_half16<scalar_t>(x_row + g_k_base + k_off);
      }

      half16_t a_reg;
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        const int8_t scale =
            reinterpret_cast<const int8_t*>(block_ptr[tm] + 192)[kt];
        const float dscale = d_f[tm] * static_cast<float>(scale);
#pragma unroll
        for (int kk = 0; kk < K_TILE; ++kk) {
          a_reg[kk] = static_cast<half_t>(
              dscale * static_cast<float>(
                           gguf_q6_k_quant(block_ptr[tm], k_off + kk)));
        }
#pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a_reg, b_reg[tn], acc[tm][tn]);
        }
      }
    }
  }

#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    const int out_row = row_start + tn * 16 + lane16;
    if (out_row >= rows) continue;
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = out_start + tm * 16 + 2 * j + (tid >> 4);
        if (col < out_features) {
          store_output<out_t>(
              out,
              static_cast<int64_t>(out_row) * out_features + col,
              static_cast<float>(acc[tm][tn][j]));
        }
      }
    }
  }
}

// Resident pack8 sibling of the raw GGUF kernel above. The qweight plane is
// [out_features / 8, in_features] int32, with one nibble per output column.
// The effective d*scale and dmin*min terms are already expanded to F32 planes
// [in_features / 32, out_features]. Keeping the same wave32 WMMA mapping lets
// dense/shared Laguna projections use the fast wide tile without a raw
// sidecar or a cache rebuild.
template <typename scalar_t, typename out_t, int TILE_M, int TILE_N>
__launch_bounds__(32, 4)
__global__ void gguf_q4_k_pack8_prefill_wmma_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ qweight,
    const float* __restrict__ scales,
    const float* __restrict__ mins,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  constexpr int TM = TILE_M / 16;
  constexpr int TN = TILE_N / 16;
  const int tid = threadIdx.x;
  const int out_start = static_cast<int>(blockIdx.x) * TILE_M;
  const int row_start = static_cast<int>(blockIdx.y) * TILE_N;
  if (out_start >= out_features || row_start >= rows) return;

  int safe_out[TM];
  int out_pack[TM];
  int out_shift[TM];
  #pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    const int col = out_start + tm * 16 + (tid & 15);
    safe_out[tm] = col < out_features ? col : out_features - 1;
    out_pack[tm] = safe_out[tm] >> 3;
    out_shift[tm] = pack8_shift_for_lane(safe_out[tm] & 7);
  }

  int safe_token[TN];
  #pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    const int row = row_start + tn * 16 + (tid & 15);
    safe_token[tn] = row < rows ? row : rows - 1;
  }

  float8_t acc[TM][TN];
  #pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    #pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      acc[tm][tn] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
  }

  for (int k_base = 0; k_base < in_features; k_base += K_TILE) {
    half16_t b_reg[TN];
    #pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      const scalar_t* x_row =
          x + static_cast<int64_t>(safe_token[tn]) * in_features;
      b_reg[tn] = load_act_half16<scalar_t>(x_row + k_base);
    }

    const int group32 = k_base >> 5;
    half16_t a_reg;
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      const float scale =
          scales[static_cast<int64_t>(group32) * out_features + safe_out[tm]];
      const float minv =
          mins[static_cast<int64_t>(group32) * out_features + safe_out[tm]];
      const int64_t qbase =
          static_cast<int64_t>(out_pack[tm]) * in_features + k_base;
      #pragma unroll
      for (int kk = 0; kk < K_TILE; ++kk) {
        const uint32_t packed = static_cast<uint32_t>(qweight[qbase + kk]);
        const uint8_t q =
            static_cast<uint8_t>((packed >> out_shift[tm]) & 0x0Fu);
        a_reg[kk] = static_cast<half_t>(
            scale * static_cast<float>(q) - minv);
      }
      #pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
        acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            a_reg, b_reg[tn], acc[tm][tn]);
      }
    }
  }

  #pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    const int out_row = row_start + tn * 16 + (tid & 15);
    if (out_row >= rows) continue;
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = out_start + tm * 16 + 2 * j + (tid >> 4);
        if (col < out_features) {
          const int64_t idx =
              static_cast<int64_t>(out_row) * out_features + col;
          store_output<out_t>(out, idx, static_cast<float>(acc[tm][tn][j]));
        }
      }
    }
  }
}

// Operation-complete resident-pack8 dense gate/up prefill. This is the
// pack8-layout sibling of the retained Q4T16 dual-WMMA+SiLU owner: four waves
// own one 32-column x 256-row tile, decode both weight matrices into one 32-KiB
// LDS union, reuse every activation K16 fragment for gate and up, round both
// projection boundaries to BF16 in LDS, then evaluate the existing BF16 SiLU
// contract without global gate/up intermediates.
__launch_bounds__(128, 1)
__global__ void gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_kernel(
    const uint16_t* __restrict__ x,
    const int32_t* __restrict__ qweight_a,
    const float* __restrict__ scales_a,
    const float* __restrict__ mins_a,
    const int32_t* __restrict__ qweight_b,
    const float* __restrict__ scales_b,
    const float* __restrict__ mins_b,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  constexpr int MATRICES = 2;
  constexpr int OUT_COLS_PER_BLOCK = 32;
  constexpr int OUT_TILES_PER_BLOCK = OUT_COLS_PER_BLOCK / 16;
  constexpr int ROW_TILES_PER_WAVE = 4;
  constexpr int WAVES_PER_BLOCK = 4;
  constexpr int ROWS_PER_WAVE = ROW_TILES_PER_WAVE * 16;
  constexpr int ROWS_PER_BLOCK = WAVES_PER_BLOCK * ROWS_PER_WAVE;
  union SharedStorage {
    half_t weights[MATRICES][OUT_COLS_PER_BLOCK * QK_K];
    uint16_t rounded[MATRICES][ROWS_PER_BLOCK * OUT_COLS_PER_BLOCK];
  };
  __shared__ SharedStorage shared;

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane16 = tid & 15;
  const int half_id = (tid & 31) >> 4;
  const int64_t col_base =
      static_cast<int64_t>(blockIdx.x) * OUT_COLS_PER_BLOCK;
  const int64_t block_row_base =
      static_cast<int64_t>(blockIdx.y) * ROWS_PER_BLOCK;
  const int64_t row_base = block_row_base + wave * ROWS_PER_WAVE;
  const int64_t blocks_per_row = in_features / QK_K;

  float8_t acc_a[OUT_TILES_PER_BLOCK][ROW_TILES_PER_WAVE];
  float8_t acc_b[OUT_TILES_PER_BLOCK][ROW_TILES_PER_WAVE];
#pragma unroll
  for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
#pragma unroll
    for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
      acc_a[tm][tn] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
      acc_b[tm][tn] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
  }

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    const int decode_pair = tid >> 2;
    const int matrix = decode_pair / (OUT_COLS_PER_BLOCK / 2);
    const int local_pair = decode_pair % (OUT_COLS_PER_BLOCK / 2);
    const int local_col0 = local_pair * 2;
    const int64_t out_col0 = col_base + local_col0;
    const int out_pack = static_cast<int>(out_col0 >> 3);
    const int shift0 = pack8_shift_for_lane(static_cast<int>(out_col0 & 7));
    const int shift1 = pack8_shift_for_lane(static_cast<int>((out_col0 + 1) & 7));
    const int k_segment = (tid & 3) * 8;
    const int32_t* __restrict__ qweight = matrix == 0 ? qweight_a : qweight_b;
    const float* __restrict__ scales = matrix == 0 ? scales_a : scales_b;
    const float* __restrict__ mins = matrix == 0 ? mins_a : mins_b;
    const int64_t block_k_base = blk * QK_K;

#pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      const int64_t group32 = blk * Q4_K_SUBBLOCKS + sb;
      const float scale0 = scales[group32 * out_features + out_col0];
      const float scale1 = scales[group32 * out_features + out_col0 + 1];
      const float min0 = mins[group32 * out_features + out_col0];
      const float min1 = mins[group32 * out_features + out_col0 + 1];
      const int64_t qbase =
          static_cast<int64_t>(out_pack) * in_features + block_k_base +
          sb * Q4_K_SUBBLOCK + k_segment;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        const uint32_t packed = static_cast<uint32_t>(qweight[qbase + kk]);
        const float q0 = static_cast<float>((packed >> shift0) & 0x0Fu);
        const float q1 = static_cast<float>((packed >> shift1) & 0x0Fu);
        const int k = sb * Q4_K_SUBBLOCK + k_segment + kk;
        shared.weights[matrix][local_col0 * QK_K + k] =
            static_cast<half_t>(scale0 * q0 - min0);
        shared.weights[matrix][(local_col0 + 1) * QK_K + k] =
            static_cast<half_t>(scale1 * q1 - min1);
      }
    }
    __syncthreads();

#pragma unroll 1
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
#pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_K_SUBBLOCK + kt * K_TILE;
        half16_t activation[ROW_TILES_PER_WAVE];
#pragma unroll
        for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
          const int64_t input_row = row_base + tn * 16 + lane16;
          const int64_t safe_row = input_row < rows ? input_row : 0;
          activation[tn] = load_act_half16<uint16_t>(
              x + safe_row * in_features + block_k_base + k_off);
        }
#pragma unroll
        for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
          half16_t weight_a;
          half16_t weight_b;
#pragma unroll
          for (int kk = 0; kk < K_TILE; ++kk) {
            const int index =
                (tm * 16 + lane16) * QK_K + k_off + kk;
            weight_a[kk] = shared.weights[0][index];
            weight_b[kk] = shared.weights[1][index];
          }
#pragma unroll
          for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
            acc_a[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                activation[tn], weight_a, acc_a[tm][tn]);
            acc_b[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                activation[tn], weight_b, acc_b[tm][tn]);
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int tm = 0; tm < OUT_TILES_PER_BLOCK; ++tm) {
    const int local_col = tm * 16 + lane16;
#pragma unroll
    for (int tn = 0; tn < ROW_TILES_PER_WAVE; ++tn) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int local_row =
            wave * ROWS_PER_WAVE + tn * 16 + half_id + j * 2;
        const int64_t out_row = block_row_base + local_row;
        if (out_row < rows) {
          const int index = local_row * OUT_COLS_PER_BLOCK + local_col;
          shared.rounded[0][index] = float_to_bf16_bits(acc_a[tm][tn][j]);
          shared.rounded[1][index] = float_to_bf16_bits(acc_b[tm][tn][j]);
        }
      }
    }
  }
  __syncthreads();

  constexpr int RESULT_COUNT = ROWS_PER_BLOCK * OUT_COLS_PER_BLOCK;
  for (int index = tid; index < RESULT_COUNT; index += blockDim.x) {
    const int local_row = index / OUT_COLS_PER_BLOCK;
    const int local_col = index - local_row * OUT_COLS_PER_BLOCK;
    const int64_t out_row = block_row_base + local_row;
    if (out_row < rows) {
      const float gate = bf16_bits_to_float(shared.rounded[0][index]);
      const float up = bf16_bits_to_float(shared.rounded[1][index]);
      const float sig = 1.0f / (1.0f + expf(-gate));
      out[out_row * out_features + col_base + local_col] =
          float_to_bf16_bits(gate * sig * up);
    }
  }
}

// Same body as the single-output kernel, but blockIdx.x is split into the
// A/gate and B/up output-tile ranges. hipENGINE's GGUF dense pair ABI uses
// one shared activation matrix and matching A/B out_features, unlike the
// parent AWQ dual kernel that accepts separate x_a/x_b and packed widths.
template <typename scalar_t, typename out_t, int AWQ_TILE_M, int AWQ_TILE_N>
__launch_bounds__(32, 4)
__global__ void gguf_q4_k_prefill_dual_wmma_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    out_t* __restrict__ out_a,
    out_t* __restrict__ out_b,
    int rows,
    int in_features,
    int out_features,
    int out_tiles_a) {
  constexpr int TM = AWQ_TILE_M / 16;
  constexpr int TN = AWQ_TILE_N / 16;
  const bool use_b = blockIdx.x >= static_cast<unsigned int>(out_tiles_a);
  const int out_tile = use_b ? static_cast<int>(blockIdx.x) - out_tiles_a : static_cast<int>(blockIdx.x);
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;
  out_t* __restrict__ out = use_b ? out_b : out_a;
  const int row_start = row_tile * AWQ_TILE_N;
  const int out_start = out_tile * AWQ_TILE_M;
  if (row_start >= rows || out_start >= out_features) return;

  const int blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * Q4_K_BLOCK_BYTES;

  int my_out[TM];
  int safe_out[TM];
  #pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    my_out[tm] = out_start + tm * 16 + (tid & 15);
    safe_out[tm] = (my_out[tm] < out_features) ? my_out[tm] : (out_features - 1);
  }

  int my_token[TN];
  int safe_token[TN];
  #pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    my_token[tn] = row_start + tn * 16 + (tid & 15);
    safe_token[tn] = (my_token[tn] < rows) ? my_token[tn] : (rows - 1);
  }

  float8_t acc[TM][TN];
  #pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    #pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      acc[tm][tn] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
  }

  for (int blk = 0; blk < blocks_per_row; ++blk) {
    const uint8_t* block_ptr[TM];
    float d_f[TM];
    float dmin_f[TM];
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      block_ptr[tm] = qweight + static_cast<int64_t>(safe_out[tm]) * row_bytes +
                      static_cast<int64_t>(blk) * Q4_K_BLOCK_BYTES;
      d_f[tm] = fp16_bytes_to_float(block_ptr[tm]);
      dmin_f[tm] = fp16_bytes_to_float(block_ptr[tm] + 2);
    }

    const int g_k_base = blk * QK_K;

    #pragma unroll
    for (int sb = 0; sb < Q4_K_SUBBLOCKS; ++sb) {
      float scale_f[TM];
      float min_f[TM];
      #pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        const uint8_t* scales = block_ptr[tm] + 4;
        scale_f[tm] = d_f[tm] * static_cast<float>(gguf_q4_k_scale(scales, sb));
        min_f[tm] = dmin_f[tm] * static_cast<float>(gguf_q4_k_min(scales, sb));
      }

      #pragma unroll
      for (int kt = 0; kt < K_TILES_PER_SUBBLOCK; ++kt) {
        const int k_off = sb * Q4_K_SUBBLOCK + kt * K_TILE;

        half16_t b_reg[TN];
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
          const scalar_t* x_row =
              x + static_cast<int64_t>(safe_token[tn]) * in_features;
          b_reg[tn] = load_act_half16<scalar_t>(x_row + g_k_base + k_off);
        }

        half16_t a_reg;
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
          const uint8_t* qs = block_ptr[tm] + Q4_K_QS_OFFSET;
          #pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            const int lane32 = kt * K_TILE + kk;
            const uint8_t q = gguf_q4_k_quant(qs, sb, lane32);
            a_reg[kk] = static_cast<half_t>(
                scale_f[tm] * static_cast<float>(q) - min_f[tm]);
          }
          #pragma unroll
          for (int tn = 0; tn < TN; ++tn) {
            acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                a_reg, b_reg[tn], acc[tm][tn]);
          }
        }
      }
    }
  }

  #pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    const int out_row = row_start + tn * 16 + (tid & 15);
    if (out_row >= rows) continue;
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = out_start + tm * 16 + 2 * j + (tid >> 4);
        if (col < out_features) {
          const int64_t idx =
              static_cast<int64_t>(out_row) * out_features + col;
          store_output<out_t>(out, idx, static_cast<float>(acc[tm][tn][j]));
        }
      }
    }
  }
}

template <typename scalar_t, typename out_t>
int launch_gguf_q4_k_prefill_wmma(
    const scalar_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr ||
      rows <= 0 || in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 ||
      (tile_m != 16 && tile_m != 32 && tile_m != 64) ||
      (tile_n != 16 && tile_n != 32)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features + tile_m - 1) / tile_m),
      static_cast<unsigned int>((rows + tile_n - 1) / tile_n));

#define HIPENGINE_LAUNCH_Q4_K_WMMA(TM, TN) \
  hipLaunchKernelGGL( \
      (gguf_q4_k_prefill_wmma_kernel<scalar_t, out_t, TM, TN>), \
      grid, \
      block, \
      0, \
      stream, \
      x, \
      qweight, \
      out, \
      static_cast<int>(rows), \
      static_cast<int>(in_features), \
      static_cast<int>(out_features)); \
  return static_cast<int>(hipGetLastError())

  if (tile_m == 16 && tile_n == 16) { HIPENGINE_LAUNCH_Q4_K_WMMA(16, 16); }
  if (tile_m == 32 && tile_n == 16) { HIPENGINE_LAUNCH_Q4_K_WMMA(32, 16); }
  if (tile_m == 16 && tile_n == 32) { HIPENGINE_LAUNCH_Q4_K_WMMA(16, 32); }
  if (tile_m == 32 && tile_n == 32) { HIPENGINE_LAUNCH_Q4_K_WMMA(32, 32); }
  if (tile_m == 64 && tile_n == 16) { HIPENGINE_LAUNCH_Q4_K_WMMA(64, 16); }
  HIPENGINE_LAUNCH_Q4_K_WMMA(64, 32);
#undef HIPENGINE_LAUNCH_Q4_K_WMMA
}

template <typename scalar_t, typename out_t>
int launch_gguf_q4_k_pack8_prefill_wmma(
    const scalar_t* x,
    const int32_t* qweight,
    const float* scales,
    const float* mins,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || scales == nullptr || mins == nullptr ||
      out == nullptr || rows <= 0 || in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 || (out_features % 8) != 0 ||
      (tile_m != 16 && tile_m != 32 && tile_m != 64) ||
      (tile_n != 16 && tile_n != 32)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features + tile_m - 1) / tile_m),
      static_cast<unsigned int>((rows + tile_n - 1) / tile_n));

#define HIPENGINE_LAUNCH_Q4_K_PACK8_WMMA(TM, TN) \
  hipLaunchKernelGGL( \
      (gguf_q4_k_pack8_prefill_wmma_kernel<scalar_t, out_t, TM, TN>), \
      grid, \
      block, \
      0, \
      stream, \
      x, \
      qweight, \
      scales, \
      mins, \
      out, \
      static_cast<int>(rows), \
      static_cast<int>(in_features), \
      static_cast<int>(out_features)); \
  return static_cast<int>(hipGetLastError())

  if (tile_m == 16 && tile_n == 16) { HIPENGINE_LAUNCH_Q4_K_PACK8_WMMA(16, 16); }
  if (tile_m == 32 && tile_n == 16) { HIPENGINE_LAUNCH_Q4_K_PACK8_WMMA(32, 16); }
  if (tile_m == 16 && tile_n == 32) { HIPENGINE_LAUNCH_Q4_K_PACK8_WMMA(16, 32); }
  if (tile_m == 32 && tile_n == 32) { HIPENGINE_LAUNCH_Q4_K_PACK8_WMMA(32, 32); }
  if (tile_m == 64 && tile_n == 16) { HIPENGINE_LAUNCH_Q4_K_PACK8_WMMA(64, 16); }
  HIPENGINE_LAUNCH_Q4_K_PACK8_WMMA(64, 32);
#undef HIPENGINE_LAUNCH_Q4_K_PACK8_WMMA
}

int launch_gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16(
    const uint16_t* x,
    const int32_t* qweight_a,
    const float* scales_a,
    const float* mins_a,
    const int32_t* qweight_b,
    const float* scales_b,
    const float* mins_b,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  constexpr int OUT_COLS_PER_BLOCK = 32;
  constexpr int ROWS_PER_BLOCK = 256;
  if (x == nullptr || qweight_a == nullptr || scales_a == nullptr ||
      mins_a == nullptr || qweight_b == nullptr || scales_b == nullptr ||
      mins_b == nullptr || out == nullptr || rows <= 0 || in_features <= 0 ||
      out_features <= 0 || (in_features % QK_K) != 0 ||
      (out_features % OUT_COLS_PER_BLOCK) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(128);
  const dim3 grid(
      static_cast<unsigned int>(out_features / OUT_COLS_PER_BLOCK),
      static_cast<unsigned int>((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK));
  hipLaunchKernelGGL(
      gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_kernel,
      grid,
      block,
      0,
      stream,
      x,
      qweight_a,
      scales_a,
      mins_a,
      qweight_b,
      scales_b,
      mins_b,
      out,
      rows,
      in_features,
      out_features);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, typename out_t>
int launch_gguf_q6_k_prefill_wmma(
    const scalar_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr ||
      rows <= 0 || in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 ||
      (tile_m != 16 && tile_m != 32 && tile_m != 64) ||
      (tile_n != 16 && tile_n != 32)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features + tile_m - 1) / tile_m),
      static_cast<unsigned int>((rows + tile_n - 1) / tile_n));

#define HIPENGINE_LAUNCH_Q6_K_WMMA(TM, TN) \
  hipLaunchKernelGGL( \
      (gguf_q6_k_prefill_wmma_kernel<scalar_t, out_t, TM, TN>), \
      grid, \
      block, \
      0, \
      stream, \
      x, \
      qweight, \
      out, \
      static_cast<int>(rows), \
      static_cast<int>(in_features), \
      static_cast<int>(out_features)); \
  return static_cast<int>(hipGetLastError())

  if (tile_m == 16 && tile_n == 16) { HIPENGINE_LAUNCH_Q6_K_WMMA(16, 16); }
  if (tile_m == 32 && tile_n == 16) { HIPENGINE_LAUNCH_Q6_K_WMMA(32, 16); }
  if (tile_m == 16 && tile_n == 32) { HIPENGINE_LAUNCH_Q6_K_WMMA(16, 32); }
  if (tile_m == 32 && tile_n == 32) { HIPENGINE_LAUNCH_Q6_K_WMMA(32, 32); }
  if (tile_m == 64 && tile_n == 16) { HIPENGINE_LAUNCH_Q6_K_WMMA(64, 16); }
  HIPENGINE_LAUNCH_Q6_K_WMMA(64, 32);
#undef HIPENGINE_LAUNCH_Q6_K_WMMA
}

template <typename scalar_t, typename out_t>
int launch_gguf_q4_k_prefill_dual_wmma(
    const scalar_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    out_t* out_a,
    out_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  if (x == nullptr || qweight_a == nullptr || qweight_b == nullptr ||
      out_a == nullptr || out_b == nullptr ||
      rows <= 0 || in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 ||
      (tile_m != 16 && tile_m != 32 && tile_m != 64) ||
      (tile_n != 16 && tile_n != 32)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_tiles = (out_features + tile_m - 1) / tile_m;
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>(out_tiles * 2),
      static_cast<unsigned int>((rows + tile_n - 1) / tile_n));

#define HIPENGINE_LAUNCH_Q4_K_DUAL_WMMA(TM, TN) \
  hipLaunchKernelGGL( \
      (gguf_q4_k_prefill_dual_wmma_kernel<scalar_t, out_t, TM, TN>), \
      grid, \
      block, \
      0, \
      stream, \
      x, \
      qweight_a, \
      qweight_b, \
      out_a, \
      out_b, \
      static_cast<int>(rows), \
      static_cast<int>(in_features), \
      static_cast<int>(out_features), \
      static_cast<int>(out_tiles)); \
  return static_cast<int>(hipGetLastError())

  if (tile_m == 16 && tile_n == 16) { HIPENGINE_LAUNCH_Q4_K_DUAL_WMMA(16, 16); }
  if (tile_m == 32 && tile_n == 16) { HIPENGINE_LAUNCH_Q4_K_DUAL_WMMA(32, 16); }
  if (tile_m == 16 && tile_n == 32) { HIPENGINE_LAUNCH_Q4_K_DUAL_WMMA(16, 32); }
  if (tile_m == 32 && tile_n == 32) { HIPENGINE_LAUNCH_Q4_K_DUAL_WMMA(32, 32); }
  if (tile_m == 64 && tile_n == 16) { HIPENGINE_LAUNCH_Q4_K_DUAL_WMMA(64, 16); }
  HIPENGINE_LAUNCH_Q4_K_DUAL_WMMA(64, 32);
#undef HIPENGINE_LAUNCH_Q4_K_DUAL_WMMA
}

}  // namespace

// -------------------------------------------------------------------------
// C ABI exports. Single-output naming:
//   hipengine_gguf_q4_k_wmma_prefill_<in>_<out>_out
// Dual gate+up naming:
//   hipengine_gguf_q4_k_wmma_prefill_dual_bf16_bf16_out
// -------------------------------------------------------------------------

extern "C" int hipengine_gguf_q4_k_wmma_prefill_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q4_k_prefill_wmma<uint16_t, uint16_t>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q4_k_pack8_wmma_prefill_bf16_bf16_out(
    const uint16_t* x,
    const int32_t* qweight,
    const float* scales,
    const float* mins,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q4_k_pack8_prefill_wmma<uint16_t, uint16_t>(
      x,
      qweight,
      scales,
      mins,
      out,
      rows,
      in_features,
      out_features,
      tile_m,
      tile_n,
      stream);
}

extern "C" int hipengine_gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16_bf16_out(
    const uint16_t* x,
    const int32_t* qweight_a,
    const float* scales_a,
    const float* mins_a,
    const int32_t* qweight_b,
    const float* scales_b,
    const float* mins_b,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_gguf_q4_k_pack8_dual_wmma_prefill_silu_bf16(
      x,
      qweight_a,
      scales_a,
      mins_a,
      qweight_b,
      scales_b,
      mins_b,
      out,
      rows,
      in_features,
      out_features,
      stream);
}

extern "C" int hipengine_gguf_q6_k_wmma_prefill_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q6_k_prefill_wmma<uint16_t, uint16_t>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q4_k_wmma_prefill_bf16_fp16_out(
    const uint16_t* x,
    const uint8_t* qweight,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q4_k_prefill_wmma<uint16_t, half_t>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q4_k_wmma_prefill_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q4_k_prefill_wmma<uint16_t, float>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q4_k_wmma_prefill_fp16_bf16_out(
    const half_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q4_k_prefill_wmma<half_t, uint16_t>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q4_k_wmma_prefill_fp16_fp16_out(
    const half_t* x,
    const uint8_t* qweight,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q4_k_prefill_wmma<half_t, half_t>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q4_k_wmma_prefill_fp16_f32_out(
    const half_t* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q4_k_prefill_wmma<half_t, float>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q4_k_wmma_prefill_f32_bf16_out(
    const float* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q4_k_prefill_wmma<float, uint16_t>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q4_k_wmma_prefill_f32_fp16_out(
    const float* x,
    const uint8_t* qweight,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q4_k_prefill_wmma<float, half_t>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q4_k_wmma_prefill_f32_f32_out(
    const float* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q4_k_prefill_wmma<float, float>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q4_k_wmma_prefill_dual_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out_a,
    uint16_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q4_k_prefill_dual_wmma<uint16_t, uint16_t>(
      x, qweight_a, qweight_b, out_a, out_b, rows, in_features, out_features,
      tile_m, tile_n, stream);
}
