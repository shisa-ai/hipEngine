// W4A16 prefill GEMM for raw dense GGUF IQ weights on RDNA3.
//
// Structure mirrors gguf_q6_k_prefill_wmma_kernel in
// hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_prefill.hip: 32 threads (one
// wave32) per block, a TILE_M x TILE_N output tile built with
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32, raw GGUF bytes as the sole
// resident representation, and activations read straight from the caller's
// row-major buffer. Only the inner dequant differs, and it uses the same
// expressions as the bit-exact strict dense kernel (gguf_iq_dense.hip).
//
// Why this exists beside the integer MMQ route: that route quantizes
// activations to int8 with one scale per 32, which costs about 5.3e-03
// relative error against an f32 reference. Expanding both operands into fp16
// instead measures 2.1e-04 on the same tensors - 25x tighter - because fp16
// keeps an 11-bit mantissa on each side where int8 keeps seven bits on one.
// (bf16 would be the wrong choice at 1.6e-03: its 8-bit mantissa puts error
// back onto the weights, which the integer path keeps exact.)
//
// Codec math: llama.cpp@17252c769a63c1cb650ce98ae309cf4de0da7778
// ggml/src/ggml-quants.c; grid tables from ggml-common.h (MIT).

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {
#include "gguf_iq_dense_tables.h"
#include "gguf_iq2_xs_dense_table.h"
#include "gguf_iq3_s_grid.h"

using half_t = _Float16;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

constexpr int QK_K = 256;
constexpr int K_TILE = 16;  // WMMA inner-K size

__device__ inline float bf16_bits_to_float(uint16_t value) {
  union { uint32_t u32; float f32; } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t float_to_bf16_bits(float value) {
  union { float f32; uint32_t u32; } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

__device__ inline float fp16_bytes_to_float(const uint8_t* ptr) {
  union { uint16_t u16; half_t f16; } in;
  in.u16 = static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
  return static_cast<float>(in.f16);
}

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v) { return static_cast<float>(v); }
template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t v) { return bf16_bits_to_float(v); }

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
  for (int i = 0; i < 16; ++i)
    out[i] = static_cast<half_t>(scalar_to_float<scalar_t>(p[i]));
  return out;
}
template <>
__device__ inline half16_t load_act_half16<half_t>(const half_t* p) {
  return *reinterpret_cast<const half16_t*>(p);
}

// Same retained choice as the strict dense kernel: the 16-byte codebook stays
// a __constant__ table, because replacing it with packed immediates measured
// slower there (it trades an L1-resident byte load for ~7 VALU).
__device__ __constant__ int8_t iq4_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ inline int iq_sign_bits(int i) { return i | ((__popc(i) & 1) << 7); }
__device__ inline float iq2_xs_magnitude(int code) {
  return float((0x2B2B1908u >> (8 * code)) & 0xffu);
}

// Per-element decode, expression-for-expression the strict dense kernel's
// weight<Q>() so the two routes agree on the codec. QTYPE numbering matches
// gguf_iq_dense.hip: 0 IQ4_XS, 1 IQ4_NL, 2 IQ3_S, 3 Q3_K, 4 IQ3_XXS,
// 5 IQ2_S, 6 IQ2_XS.
template <int Q>
__device__ inline float iq_weight(const uint8_t* b, int k) {
  if constexpr (Q == 1) {
    int code = (b[2 + (k & 15)] >> (4 * (k >> 4))) & 15;
    return fp16_bytes_to_float(b) * float(iq4_values[code]);
  } else if constexpr (Q == 0) {
    int g = k >> 5, j = k & 31;
    int hi = int(b[2]) | (int(b[3]) << 8);
    int scale = (((b[4 + g / 2] >> (4 * (g & 1))) & 15) | (((hi >> (2 * g)) & 3) << 4)) - 32;
    int code = (b[8 + g * 16 + (j & 15)] >> (4 * (j >> 4))) & 15;
    return (fp16_bytes_to_float(b) * float(scale)) * float(iq4_values[code]);
  } else if constexpr (Q == 4) {
    int g = k >> 5, j = k & 31;
    const uint8_t* p = b + 66 + 4 * g;
    uint32_t aux = uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
                   (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
    int index = b[2 + g * 8 + j / 4];
    int signs = iq_sign_bits((aux >> (7 * (j / 8))) & 127);
    float d = (fp16_bytes_to_float(b) * (0.5f + float(aux >> 28))) * 0.5f;
    float mag = d * float(iq3_xxs_grid[index * 4 + (j & 3)]);
    return mag * ((signs & (1 << (j & 7))) ? -1.0f : 1.0f);
  } else if constexpr (Q == 6) {
    int g = k >> 5, j = k & 31, slot = j >> 3;
    const uint8_t* p = b + 2 + 2 * (g * 4 + slot);
    int word = int(p[0]) | (int(p[1]) << 8);
    int code = (IQ2_XS_GRID_PACKED[word & 511] >> (2 * (j & 7))) & 3;
    int scale = (b[66 + g] >> (4 * (j >> 4))) & 15;
    float d = (fp16_bytes_to_float(b) * (0.5f + float(scale))) * 0.25f;
    float mag = d * iq2_xs_magnitude(code);
    return mag * ((iq_sign_bits(word >> 9) & (1 << (j & 7))) ? -1.0f : 1.0f);
  } else if constexpr (Q == 5) {
    int g = k >> 5, j = k & 31, slot = j >> 3;
    int index = b[2 + g * 4 + slot] | (((b[66 + g] >> (2 * slot)) & 3) << 8);
    int scale = (b[74 + g] >> (4 * (j >> 4))) & 15;
    float d = (fp16_bytes_to_float(b) * (0.5f + float(scale))) * 0.25f;
    float mag = d * float(iq2_s_grid[index * 8 + (j & 7)]);
    return mag * ((b[34 + g * 4 + slot] & (1 << (j & 7))) ? -1.0f : 1.0f);
  } else if constexpr (Q == 3) {
    int group = k >> 4;
    int scale = ((b[96 + (group & 7)] >> (4 * (group >> 3))) & 15) |
                (((b[104 + (group & 3)] >> (2 * (group >> 2))) & 3) << 4);
    int q = (b[32 + (k >> 7) * 32 + (k & 31)] >> (2 * ((k & 127) >> 5))) & 3;
    q -= (b[k & 31] & (1 << (k >> 5))) ? 0 : 4;
    return (fp16_bytes_to_float(b + 108) * float(scale - 32)) * float(q);
  } else {
    int g = k >> 5, j = k & 31, grid_slot = j >> 2;
    int index = b[2 + g * 8 + grid_slot] | (((b[66 + g] >> grid_slot) & 1) << 8);
    int scale = 1 + 2 * ((b[106 + g / 2] >> (4 * (g & 1))) & 15);
    float mag = (fp16_bytes_to_float(b) * float(scale)) * float(iq3_grid[index * 4 + (j & 3)]);
    return mag * ((b[74 + g * 4 + j / 8] & (1 << (j & 7))) ? -1.0f : 1.0f);
  }
}

template <int Q>
constexpr int iq_block_bytes() {
  return Q == 0 ? 136 : (Q == 1 ? 18 : (Q == 2 ? 110 : (Q == 3 ? 110 :
         (Q == 4 ? 98 : (Q == 5 ? 82 : 74)))));
}
// IQ4_NL stores 32 elements per block; every other supported quant stores 256.
template <int Q>
constexpr int iq_block_elements() { return Q == 1 ? 32 : QK_K; }

// TILE_M output columns x TILE_N activation rows per wave32 block. Each
// decoded weight fragment is reused across TILE_N rows, which is the whole
// point: the strict GEMV reuses it across eight.
template <int Q, typename scalar_t, typename out_t, int TILE_M, int TILE_N>
__launch_bounds__(32, 4)
__global__ void gguf_iq_prefill_wmma_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  constexpr int TM = TILE_M / 16;
  constexpr int TN = TILE_N / 16;
  constexpr int BLOCK_BYTES = iq_block_bytes<Q>();
  constexpr int BLOCK_ELEMS = iq_block_elements<Q>();

  const int out_start = blockIdx.x * TILE_M;
  const int row_start = blockIdx.y * TILE_N;
  const int tid = threadIdx.x;
  if (row_start >= rows || out_start >= out_features) return;

  const int blocks_per_row = in_features / BLOCK_ELEMS;
  const int64_t row_bytes = int64_t(blocks_per_row) * BLOCK_BYTES;

  int safe_out[TM];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    const int col = out_start + tm * 16 + (tid & 15);
    safe_out[tm] = col < out_features ? col : out_features - 1;
  }
  int safe_token[TN];
#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    const int row = row_start + tn * 16 + (tid & 15);
    safe_token[tn] = row < rows ? row : rows - 1;
  }

  float8_t acc[TM][TN];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm)
#pragma unroll
    for (int tn = 0; tn < TN; ++tn)
      acc[tm][tn] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};

  // The hi+lo split path (Q3_K, IQ2_S, IQ2_XS). The WMMA products of fp16
  // operands are exact in the f32 accumulator - verified against an f64 dot
  // of fp16-rounded weights on a real tensor (2026-09-09, W7900: the kernel
  // and the simulation's bf16-ULP distance distributions match to the third
  // decimal, 10.4% of outputs >=1 ULP both) - so a second pass over the
  // fp16 rounding residual restores the weight to ~22 mantissa bits and the
  // route's intrinsic error drops under the bf16 output floor. Unsplit, the
  // route's entire noise is that rounding: it is what breaches the
  // production-referenced mean for Q3_K (2/3 probe contexts over 1e-3, one
  // absolute-ceiling row; the accumulation order contributes nothing). The
  // originally routed four quants keep the single-pass path: they pass the
  // gate as shipped and the split would double their WMMA cost.
  constexpr bool SPLIT_LO = (Q == 3 || Q == 5 || Q == 6);
  for (int k_base = 0; k_base < in_features; k_base += K_TILE) {
    half16_t b_reg[TN];
#pragma unroll
    for (int tn = 0; tn < TN; ++tn)
      b_reg[tn] = load_act_half16<scalar_t>(
          x + int64_t(safe_token[tn]) * in_features + k_base);

#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      const uint8_t* row_ptr = qweight + int64_t(safe_out[tm]) * row_bytes;
      half16_t a_hi;
      [[maybe_unused]] half16_t a_lo;
#pragma unroll
      for (int kk = 0; kk < K_TILE; ++kk) {
        const int k = k_base + kk;
        const uint8_t* block = row_ptr + int64_t(k / BLOCK_ELEMS) * BLOCK_BYTES;
        const float wv = iq_weight<Q>(block, k % BLOCK_ELEMS);
        const half_t hi = static_cast<half_t>(wv);
        a_hi[kk] = hi;
        if constexpr (SPLIT_LO)
          a_lo[kk] = static_cast<half_t>(wv - static_cast<float>(hi));
      }
#pragma unroll
      for (int tn = 0; tn < TN; ++tn)
        acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            a_hi, b_reg[tn], acc[tm][tn]);
      if constexpr (SPLIT_LO) {
#pragma unroll
        for (int tn = 0; tn < TN; ++tn)
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a_lo, b_reg[tn], acc[tm][tn]);
      }
    }
  }

#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    const int out_row = row_start + tn * 16 + (tid & 15);
    if (out_row >= rows) continue;
#pragma unroll
    for (int tm = 0; tm < TM; ++tm)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int col = out_start + tm * 16 + 2 * j + (tid >> 4);
        if (col < out_features)
          store_output<out_t>(out, int64_t(out_row) * out_features + col,
                              float(acc[tm][tn][j]));
      }
  }
}

// Swept 2026-09-09 on IQ4_XS 512x(5120->17408), gfx1151. Row reuse dominates:
// each decoded weight fragment is shared across TILE_N activation rows, so the
// cost falls steeply as TILE_N grows and rises with TILE_M.
//   64x16 33.3 ms   64x32 24.7   32x32 20.8   32x64 17.2
//   16x64 12.0      16x96 12.6   16x128 11.4  32x128 15.1   128x16 42.2
// 16x128 is the retained shape on gfx1151; past it the curve is flat.
// The Python build wrapper (gguf_iq_wmma_prefill.py _flags/_retained_tiles)
// always passes explicit tiles: gfx1100 builds use 16x64 through 64 rows
// and 32x64 above (re-swept 2026-09-09 on the 7900 XTX and W7900, 18-20%
// under 16x128 at 512 rows on both cards and 2.4x at 8 rows), gfx1151 keeps
// 16x128. These #defaults only apply to direct .hip compilation.
#ifndef IQ_WMMA_TILE_M
#define IQ_WMMA_TILE_M 16
#endif
#ifndef IQ_WMMA_TILE_N
#define IQ_WMMA_TILE_N 128
#endif

template <int Q>
int launch_iq_wmma_prefill(const void* x, const void* w, void* out,
                           int rows, int in_features, int out_features,
                           hipStream_t stream) {
  if (!x || !w || !out || rows <= 0 || in_features <= 0 || out_features <= 0)
    return int(hipErrorInvalidValue);
  if (in_features % iq_block_elements<Q>()) return int(hipErrorInvalidValue);
  dim3 grid((out_features + IQ_WMMA_TILE_M - 1) / IQ_WMMA_TILE_M,
            (rows + IQ_WMMA_TILE_N - 1) / IQ_WMMA_TILE_N);
  hipLaunchKernelGGL(
      (gguf_iq_prefill_wmma_kernel<Q, uint16_t, uint16_t,
                                   IQ_WMMA_TILE_M, IQ_WMMA_TILE_N>),
      grid, dim3(32), 0, stream,
      static_cast<const uint16_t*>(x), static_cast<const uint8_t*>(w),
      static_cast<uint16_t*>(out), rows, in_features, out_features);
  return int(hipGetLastError());
}

// Cooperative shared-LDS multi-wave W4A16 prefill (2026-09-10 review
// follow-up): one 256-thread block owns 32 output columns x 256 activation
// rows; all eight waves decode the 32x256-K weight slab into LDS once per
// K-block, then consume it with the one-wave owner's exact K16 WMMA
// association. The one-wave owner re-decodes the same column slab once per
// 64-row tile - at 512 rows that is 8x redundant decode, and the IQ element
// decode (codebook lookups, per-32 scales) is expensive enough to dominate.
// The per-element arithmetic is identical (the same iq_weight<Q> value, the
// same fp16 rounding incl. the hi+lo split path) and the k-tile sequence per
// output is unchanged, so the kernel is bit-exact with the one-wave owner.
template <int Q, typename scalar_t, typename out_t,
          int OUT_TILES_PER_BLOCK = 2, int ROW_TILES_PER_WAVE = 2,
          int WAVES_PER_BLOCK = 8>
__launch_bounds__(WAVES_PER_BLOCK * 32, 1)
__global__ void gguf_iq_prefill_wmma_coop_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  constexpr int TM = OUT_TILES_PER_BLOCK;
  constexpr int TN = ROW_TILES_PER_WAVE;
  constexpr int BLOCK_BYTES = iq_block_bytes<Q>();
  constexpr int BLOCK_ELEMS = iq_block_elements<Q>();
  constexpr int OUT_COLS = TM * 16;
  constexpr int ROWS_PER_WAVE = TN * 16;
  constexpr int ROWS_PER_BLOCK = WAVES_PER_BLOCK * ROWS_PER_WAVE;
  constexpr int THREADS = WAVES_PER_BLOCK * 32;
  // Column-owner decode mapping: OUT_COLS columns share THREADS threads,
  // each owning SEG consecutive K of one column (SEG must be a multiple of
  // 32 so a thread's elements stay on 32-element scale-group boundaries).
  static_assert(THREADS % OUT_COLS == 0, "columns must divide the threads");
  constexpr int THREADS_PER_COL = THREADS / OUT_COLS;
  constexpr int SEG = QK_K / THREADS_PER_COL;
  static_assert(SEG % 32 == 0, "segment must span whole scale groups");

  // +16 halves of padding makes the column stride 272 halves = 544 B:
  // 17 x 32 B, so every row stays 32 B aligned for the half16 vector
  // loads in the WMMA phase (one LDS vector per b fragment instead of
  // sixteen 2 B scalars).
  constexpr int SLAB_STRIDE = QK_K + 16;
  __shared__ __align__(1024) half_t slab_hi[OUT_COLS][SLAB_STRIDE];
  __shared__ __align__(1024) half_t slab_lo[OUT_COLS][SLAB_STRIDE];
  // IQ4_XS fused codebook LUT (the strict decode kernel's trick): the
  // per-element __constant__ iq4_values[code] load is a divergent scalar
  // access that costs ~24% of a decode-bound kernel. s_cb[byte] is the low
  // nibble's codebook value, s_cb[256 + byte] the high one, fusing the
  // nibble extract and the codebook into one LDS read. Values are
  // entry-for-entry iq4_values.
  __shared__ float s_cb[512];
  if constexpr (Q == 0) {
    for (int i = threadIdx.x; i < 512; i += THREADS) {
      const int byte = i & 255;
      const int nib = i >= 256 ? (byte >> 4) : (byte & 15);
      s_cb[i] = float(iq4_values[nib]);
    }
    __syncthreads();
  } else if constexpr (Q == 1) {
    // IQ4_NL: the same divergent __constant__ iq4_values[code] access;
    // a plain 16-entry LDS copy.
    for (int i = threadIdx.x; i < 16; i += THREADS)
      s_cb[i] = float(iq4_values[i]);
    __syncthreads();
  }

  const int out_start = static_cast<int>(blockIdx.x) * OUT_COLS;
  const int row_start = static_cast<int>(blockIdx.y) * ROWS_PER_BLOCK;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane16 = tid & 15;
  const int half_id = (tid & 31) >> 4;
  constexpr bool SPLIT_LO = (Q == 3 || Q == 5 || Q == 6);
  const int64_t blocks_per_row = in_features / BLOCK_ELEMS;
  const int64_t row_bytes = int64_t(blocks_per_row) * BLOCK_BYTES;

  int safe_out[TM];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    const int col = out_start + tm * 16 + lane16;
    safe_out[tm] = col < out_features ? col : out_features - 1;
  }
  int safe_token[TN];
#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    const int row = row_start + wave * ROWS_PER_WAVE + tn * 16 + lane16;
    safe_token[tn] = row < rows ? row : rows - 1;
  }

  float8_t acc[TM][TN];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm)
#pragma unroll
    for (int tn = 0; tn < TN; ++tn)
      acc[tm][tn] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};

  const int dec_col = tid / THREADS_PER_COL;
  const int dec_k0 = (tid % THREADS_PER_COL) * SEG;
  const uint8_t* dec_row =
      qweight + int64_t(out_start + dec_col < out_features
                             ? out_start + dec_col
                             : out_features - 1) * row_bytes;

  for (int64_t blk = 0; blk < blocks_per_row * BLOCK_ELEMS / QK_K; ++blk) {
    const int64_t k_blk = int64_t(blk) * QK_K;
    const bool wave_active = row_start + wave * ROWS_PER_WAVE < rows;
    if constexpr (Q == 0) {
      // IQ4_XS column-owner fast path: dec_k0 is 32-aligned and k_blk is
      // 256-aligned, so this thread's SEG elements span whole scale groups
      // of one 136-byte block (BLOCK_ELEMS == QK_K for IQ4_XS). The generic
      // iq_weight path cannot prove that invariant (dec_k0 is a runtime
      // value), so its per-element form re-loads d, the hi word and the
      // scale byte per element.
      static_assert(BLOCK_BYTES == 136 && BLOCK_ELEMS == 256);
      const uint8_t* block = dec_row + (k_blk >> 8) * BLOCK_BYTES;
      const float d = fp16_bytes_to_float(block);
      const int hi_word = int(block[2]) | (int(block[3]) << 8);
      const int g0 = dec_k0 >> 5;
#pragma unroll
      for (int s = 0; s < SEG / 32; ++s) {
        const int g = g0 + s;
        const int scale =
            (((block[4 + g / 2] >> (4 * (g & 1))) & 15) |
             (((hi_word >> (2 * g)) & 3) << 4)) - 32;
        const float ds = d * float(scale);
        const uint8_t* codes = block + 8 + g * 16;
#pragma unroll
        for (int kk = 0; kk < 32; ++kk) {
          slab_hi[dec_col][dec_k0 + s * 32 + kk] = static_cast<half_t>(
              ds * s_cb[((kk >> 4) << 8) + codes[kk & 15]]);
        }
      }
    } else if constexpr (Q == 1) {
      // IQ4_NL fast path: dec_k0 is 32-aligned and BLOCK_ELEMS == 32, so
      // each 32-element sub-segment spans exactly one 18-byte block with
      // one shared fp16 d. The generic form re-loads d per element.
      static_assert(BLOCK_BYTES == 18 && BLOCK_ELEMS == 32);
#pragma unroll
      for (int s = 0; s < SEG / 32; ++s) {
        const uint8_t* block =
            dec_row + int64_t((k_blk + dec_k0) >> 5) * BLOCK_BYTES +
            int64_t(s) * BLOCK_BYTES;
        const float d = fp16_bytes_to_float(block);
#pragma unroll
        for (int kk = 0; kk < 32; ++kk) {
          const int code = (block[2 + (kk & 15)] >> (4 * (kk >> 4))) & 15;
          slab_hi[dec_col][dec_k0 + s * 32 + kk] =
              static_cast<half_t>(d * s_cb[code]);
        }
      }
    } else {
#pragma unroll
      for (int kk = 0; kk < SEG; ++kk) {
        const int k = dec_k0 + kk;
        const uint8_t* block =
            dec_row + int64_t((k_blk + k) / BLOCK_ELEMS) * BLOCK_BYTES;
        const float wv = iq_weight<Q>(block, (k_blk + k) % BLOCK_ELEMS);
        const half_t hi = static_cast<half_t>(wv);
        slab_hi[dec_col][k] = hi;
        if constexpr (SPLIT_LO)
          slab_lo[dec_col][k] =
              static_cast<half_t>(wv - static_cast<float>(hi));
      }
    }
    __syncthreads();

    const int64_t block_k_base = k_blk;
    if (wave_active) {
#pragma unroll 1
    for (int kt = 0; kt < QK_K / K_TILE; ++kt) {
      const int k_off = kt * K_TILE;
      half16_t a_reg[TN];
#pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
        const int64_t input_row =
            row_start + wave * ROWS_PER_WAVE + tn * 16 + lane16;
        const bool valid_row = input_row < rows;
        a_reg[tn] = load_act_half16<scalar_t>(
            x + int64_t(valid_row ? input_row : 0) * in_features +
                block_k_base + k_off);
        if (!valid_row) a_reg[tn] = half16_t{};
      }
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        const half16_t b_reg =
            *reinterpret_cast<const half16_t*>(
                &slab_hi[tm * 16 + lane16][k_off]);
#pragma unroll
        for (int tn = 0; tn < TN; ++tn)
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a_reg[tn], b_reg, acc[tm][tn]);
        if constexpr (SPLIT_LO) {
          const half16_t b_lo =
              *reinterpret_cast<const half16_t*>(
                  &slab_lo[tm * 16 + lane16][k_off]);
#pragma unroll
          for (int tn = 0; tn < TN; ++tn)
            acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                a_reg[tn], b_lo, acc[tm][tn]);
        }
      }
    }
    }  // wave_active
    __syncthreads();
  }

#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    const int col = out_start + tm * 16 + lane16;
    if (col >= out_features) continue;
#pragma unroll
    for (int tn = 0; tn < TN; ++tn)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        // Q5-family fragment convention: this kernel passes
        // wmma(activations, weights), so the C fragment maps
        // col <- lane16 and row <- (half_id + 2j) - the mirror of the
        // one-wave owner's store, which pairs with its swapped operands.
        const int out_row =
            row_start + wave * ROWS_PER_WAVE + tn * 16 + half_id + j * 2;
        if (out_row < rows)
          store_output<out_t>(out, int64_t(out_row) * out_features + col,
                              float(acc[tm][tn][j]));
      }
  }
}

template <int Q, int OUT_TILES, int WAVES>
int launch_iq_wmma_prefill_coop(const void* x, const void* w, void* out,
                                 int rows, int in_features, int out_features,
                                 hipStream_t stream) {
  if (!x || !w || !out || rows <= 0 || in_features <= 0 || out_features <= 0)
    return int(hipErrorInvalidValue);
  if (in_features % iq_block_elements<Q>()) return int(hipErrorInvalidValue);
  if (in_features % QK_K) return int(hipErrorInvalidValue);
  dim3 grid((out_features + OUT_TILES * 16 - 1) / (OUT_TILES * 16),
            (rows + WAVES * 32 - 1) / (WAVES * 32));
  hipLaunchKernelGGL(
      (gguf_iq_prefill_wmma_coop_kernel<Q, uint16_t, uint16_t, OUT_TILES, 2,
                                        WAVES>),
      grid, dim3(WAVES * 32), 0, stream,
      static_cast<const uint16_t*>(x), static_cast<const uint8_t*>(w),
      static_cast<uint16_t*>(out), rows, in_features, out_features);
  return int(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_iq_wmma_prefill_bf16_bf16_out(
    const void* x, const void* w, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    int quant, void* stream) {
  const int r = int(rows), k = int(in_features), n = int(out_features);
  switch (quant) {
    case 0: return launch_iq_wmma_prefill<0>(x, w, out, r, k, n, (hipStream_t)stream);
    case 1: return launch_iq_wmma_prefill<1>(x, w, out, r, k, n, (hipStream_t)stream);
    case 2: return launch_iq_wmma_prefill<2>(x, w, out, r, k, n, (hipStream_t)stream);
    case 3: return launch_iq_wmma_prefill<3>(x, w, out, r, k, n, (hipStream_t)stream);
    case 4: return launch_iq_wmma_prefill<4>(x, w, out, r, k, n, (hipStream_t)stream);
    case 5: return launch_iq_wmma_prefill<5>(x, w, out, r, k, n, (hipStream_t)stream);
    case 6: return launch_iq_wmma_prefill<6>(x, w, out, r, k, n, (hipStream_t)stream);
    default: return int(hipErrorInvalidValue);
  }
}

extern "C" int hipengine_gguf_iq_wmma_prefill_coop_bf16_bf16_out(
    const void* x, const void* w, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    int quant, void* stream) {
  const int r = int(rows), k = int(in_features), n = int(out_features);
  switch (quant) {
    case 0: return launch_iq_wmma_prefill_coop<0, 2, 8>(x, w, out, r, k, n, (hipStream_t)stream);
    case 1: return launch_iq_wmma_prefill_coop<1, 2, 8>(x, w, out, r, k, n, (hipStream_t)stream);
    case 2: return launch_iq_wmma_prefill_coop<2, 2, 8>(x, w, out, r, k, n, (hipStream_t)stream);
    case 3: return launch_iq_wmma_prefill_coop<3, 2, 8>(x, w, out, r, k, n, (hipStream_t)stream);
    case 4: return launch_iq_wmma_prefill_coop<4, 2, 8>(x, w, out, r, k, n, (hipStream_t)stream);
    case 5: return launch_iq_wmma_prefill_coop<5, 2, 8>(x, w, out, r, k, n, (hipStream_t)stream);
    case 6: return launch_iq_wmma_prefill_coop<6, 2, 8>(x, w, out, r, k, n, (hipStream_t)stream);
    default: return int(hipErrorInvalidValue);
  }
}

// 64-column variant: only the single-slab quants (the hi+lo split path
// needs two 34.8 KB slabs, over the 64 KB LDS budget at this width).
extern "C" int hipengine_gguf_iq_wmma_prefill_coop64_bf16_bf16_out(
    const void* x, const void* w, void* out,
    int64_t rows, int64_t in_features, int64_t out_features,
    int quant, void* stream) {
  const int r = int(rows), k = int(in_features), n = int(out_features);
  switch (quant) {
    case 0: return launch_iq_wmma_prefill_coop<0, 4, 8>(x, w, out, r, k, n, (hipStream_t)stream);
    case 1: return launch_iq_wmma_prefill_coop<1, 4, 8>(x, w, out, r, k, n, (hipStream_t)stream);
    case 2: return launch_iq_wmma_prefill_coop<2, 4, 8>(x, w, out, r, k, n, (hipStream_t)stream);
    case 4: return launch_iq_wmma_prefill_coop<4, 4, 8>(x, w, out, r, k, n, (hipStream_t)stream);
    default: return int(hipErrorInvalidValue);
  }
}
