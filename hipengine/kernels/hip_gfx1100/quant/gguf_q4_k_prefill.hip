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
