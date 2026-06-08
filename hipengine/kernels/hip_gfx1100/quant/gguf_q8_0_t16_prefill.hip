// Dense WMMA prefill GEMM for GGUF Q8_0 T16 tiles on gfx1100.
//
// P10.B4: ports gguf_q8_0_prefill_wmma_kernel to consume the resident
// T16 replacement layout (Q8_0_T16) defined in hipengine/quant/gguf_t16.py:
//
//   tiles[out_tiles16, blocks_per_row, 544]
//
// Each tile bundles 16 output columns / one 32-K Q8_0 block:
//   bytes[0..31]    : 16 fp16 d values (per column)
//   bytes[32..543]  : 32 K-rows x 16 cols of int8 quants (row-major within tile)
//
// One wave32 block computes a (AWQ_TILE_M x AWQ_TILE_N) output tile via
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32. Tile selection (TM in {16,32,64},
// TN in {16,32}) mirrors the raw Q8_0 prefill kernel so the shape-aware
// dispatch policy in hipengine/runtime/gguf_linear.py can reuse the same
// (tile_m, tile_n) decision tree.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

constexpr int Q8_0_QK = 32;
constexpr int T16_COLS = 16;
constexpr int K_TILE = 16;
constexpr int K_TILES_PER_BLOCK = Q8_0_QK / K_TILE;  // == 2
constexpr int Q8_0_T16_D_OFFSET = 0;
constexpr int Q8_0_T16_Q_OFFSET = Q8_0_T16_D_OFFSET + T16_COLS * 2;
constexpr int Q8_0_T16_BLOCK_BYTES = Q8_0_T16_Q_OFFSET + Q8_0_QK * T16_COLS;

// ---- scalar conversion helpers (identical to gguf_q8_0_prefill.hip) -----

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
  uint16_t bits = static_cast<uint16_t>(ptr[0]) |
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

__device__ inline const uint8_t* q8_t16_tile_ptr(
    const uint8_t* __restrict__ tiles,
    int64_t out_tile,
    int64_t block_idx,
    int64_t blocks_per_row) {
  const int64_t tile_index = out_tile * blocks_per_row + block_idx;
  return tiles + tile_index * Q8_0_T16_BLOCK_BYTES;
}

// -------------------------------------------------------------------------
// gguf_q8_0_t16_prefill_wmma_kernel
//
// One block (wave32) computes a TM x TN tile of `out`, with:
//   AWQ_TILE_M = TM rows of out_features (output columns of the GEMM),
//   AWQ_TILE_N = TN rows of `rows` (token rows / activation rows).
// `tiles` is the resident T16 byte-lossless replacement layout for Q8_0.
// Per (out_tile16, k_block) tile slab:
//   bytes[0..31]    : 16 fp16 d (one per output column inside the slab)
//   bytes[32..543]  : 32 K x 16 cols of int8 quant (k_in_block-major)
// -------------------------------------------------------------------------
template <typename scalar_t, typename out_t, int AWQ_TILE_M, int AWQ_TILE_N>
__launch_bounds__(32, 8)
__global__ void gguf_q8_0_t16_prefill_wmma_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ tiles,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  constexpr int TM = AWQ_TILE_M / 16;
  constexpr int TN = AWQ_TILE_N / 16;
  const int out_tile_base = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int row_start = row_tile * AWQ_TILE_N;
  const int out_start = out_tile_base * AWQ_TILE_M;
  if (row_start >= rows || out_start >= out_features) return;

  const int blocks_per_row = in_features / Q8_0_QK;
  const int lane16 = tid & 15;

  // Per-TM column information. Each TM step targets a different 16-col slab
  // inside the T16 layout, so we resolve the slab index and lane-relative
  // column once.
  int my_out[TM];
  int safe_out[TM];
  int slab_index[TM];
  int slab_lane[TM];
  bool slab_in_range[TM];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    my_out[tm] = out_start + tm * 16 + lane16;
    safe_out[tm] = (my_out[tm] < out_features) ? my_out[tm] : (out_features - 1);
    const int slab_base = out_start + tm * 16;
    slab_index[tm] = slab_base / T16_COLS;
    slab_lane[tm] = lane16;
    slab_in_range[tm] = slab_base < out_features;
  }

  int my_token[TN];
  int safe_token[TN];
#pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    my_token[tn] = row_start + tn * 16 + lane16;
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
    // Per-TM: pointer to this column slab's T16 tile in the resident layout,
    // plus the fp16 d for the lane's column inside the slab.
    const uint8_t* tile_ptr[TM];
    half_t d_h[TM];
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      tile_ptr[tm] = slab_in_range[tm]
          ? q8_t16_tile_ptr(tiles, slab_index[tm], blk, blocks_per_row)
          : tiles;
      const int col = slab_lane[tm];
      d_h[tm] = slab_in_range[tm]
          ? static_cast<half_t>(fp16_bytes_to_float(tile_ptr[tm] + Q8_0_T16_D_OFFSET + col * 2))
          : static_cast<half_t>(0.0f);
    }

    const int g_k_base = blk * Q8_0_QK;

#pragma unroll
    for (int kt = 0; kt < K_TILES_PER_BLOCK; ++kt) {
      const int k_off = kt * K_TILE;

      // Activation tile: each WMMA b operand spans 16 K-elements of one token
      // row.
      half16_t b_reg[TN];
#pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
        const scalar_t* x_row =
            x + static_cast<int64_t>(safe_token[tn]) * in_features;
        b_reg[tn] = load_act_half16<scalar_t>(x_row + g_k_base + k_off);
      }

      // Weight tile: dequantize 16 K-elements of this column slab.
      // The T16 layout stores 16 int8 quants per K-row (one per slab col),
      // so the byte offset for kk is q_off + (k_off+kk)*16 + col.
      half16_t a_reg;
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        const int col = slab_lane[tm];
        const uint8_t* qs_base = tile_ptr[tm] + Q8_0_T16_Q_OFFSET;
        if (slab_in_range[tm]) {
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            const int k_in_block = k_off + kk;
            const int8_t q = static_cast<int8_t>(qs_base[k_in_block * T16_COLS + col]);
            a_reg[kk] = d_h[tm] * static_cast<half_t>(static_cast<float>(q));
          }
        } else {
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) a_reg[kk] = static_cast<half_t>(0.0f);
        }
#pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a_reg, b_reg[tn], acc[tm][tn]);
        }
      }
    }
  }

  // PARO lane-to-output-slot mapping (8 floats per lane, 2 cols per j-step,
  // tid >> 4 selects the half-wave).
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
int launch_gguf_q8_0_t16_prefill_wmma(
    const scalar_t* x,
    const uint8_t* tiles,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  if (x == nullptr || tiles == nullptr || out == nullptr ||
      rows <= 0 || in_features <= 0 || out_features <= 0 ||
      (in_features % Q8_0_QK) != 0 ||
      (out_features % T16_COLS) != 0 ||
      (tile_m != 16 && tile_m != 32 && tile_m != 64) ||
      (tile_n != 16 && tile_n != 32)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features + tile_m - 1) / tile_m),
      static_cast<unsigned int>((rows + tile_n - 1) / tile_n));

#define HIPENGINE_LAUNCH_Q8_T16_WMMA(TM, TN) \
  hipLaunchKernelGGL( \
      (gguf_q8_0_t16_prefill_wmma_kernel<scalar_t, out_t, TM, TN>), \
      grid, \
      block, \
      0, \
      stream, \
      x, \
      tiles, \
      out, \
      static_cast<int>(rows), \
      static_cast<int>(in_features), \
      static_cast<int>(out_features)); \
  return static_cast<int>(hipGetLastError())

  if (tile_m == 16 && tile_n == 16) { HIPENGINE_LAUNCH_Q8_T16_WMMA(16, 16); }
  if (tile_m == 32 && tile_n == 16) { HIPENGINE_LAUNCH_Q8_T16_WMMA(32, 16); }
  if (tile_m == 16 && tile_n == 32) { HIPENGINE_LAUNCH_Q8_T16_WMMA(16, 32); }
  if (tile_m == 32 && tile_n == 32) { HIPENGINE_LAUNCH_Q8_T16_WMMA(32, 32); }
  if (tile_m == 64 && tile_n == 16) { HIPENGINE_LAUNCH_Q8_T16_WMMA(64, 16); }
  HIPENGINE_LAUNCH_Q8_T16_WMMA(64, 32);
#undef HIPENGINE_LAUNCH_Q8_T16_WMMA
}

}  // namespace

#define HIPENGINE_GGUF_Q8_T16_WMMA_EXPORT(IN_SCALAR_C, IN_SCALAR, OUT_SCALAR_C, OUT_SCALAR) \
extern "C" int hipengine_gguf_q8_0_t16_wmma_prefill_##IN_SCALAR_C##_##OUT_SCALAR_C##_out( \
    const IN_SCALAR* x, \
    const uint8_t* tiles, \
    OUT_SCALAR* out, \
    int64_t rows, \
    int64_t in_features, \
    int64_t out_features, \
    int64_t tile_m, \
    int64_t tile_n, \
    hipStream_t stream) { \
  return launch_gguf_q8_0_t16_prefill_wmma<IN_SCALAR, OUT_SCALAR>( \
      x, tiles, out, rows, in_features, out_features, tile_m, tile_n, stream); \
}

HIPENGINE_GGUF_Q8_T16_WMMA_EXPORT(bf16, uint16_t, bf16, uint16_t)
HIPENGINE_GGUF_Q8_T16_WMMA_EXPORT(bf16, uint16_t, fp16, half_t)
HIPENGINE_GGUF_Q8_T16_WMMA_EXPORT(bf16, uint16_t, f32, float)
HIPENGINE_GGUF_Q8_T16_WMMA_EXPORT(fp16, half_t, bf16, uint16_t)
HIPENGINE_GGUF_Q8_T16_WMMA_EXPORT(fp16, half_t, fp16, half_t)
HIPENGINE_GGUF_Q8_T16_WMMA_EXPORT(fp16, half_t, f32, float)
HIPENGINE_GGUF_Q8_T16_WMMA_EXPORT(f32, float, bf16, uint16_t)
HIPENGINE_GGUF_Q8_T16_WMMA_EXPORT(f32, float, fp16, half_t)
HIPENGINE_GGUF_Q8_T16_WMMA_EXPORT(f32, float, f32, float)
