// Batched WMMA prefill GEMM for GGUF Q8_0 weights on gfx1100.
//
// Mirrors the structure of awq_fusedw4_prefill_fp16_kernel (see
// hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip): 32 threads per
// block (one wave32), each block computes a (AWQ_TILE_M x AWQ_TILE_N)
// output tile via __builtin_amdgcn_wmma_f32_16x16x16_f16_w32. The only
// algorithmic difference from the AWQ template is the inner K-tile dequant
// block, which here reads GGUF Q8_0 raw bytes (34 bytes per 32-K block:
// fp16 d + int8 qs[32]).
//
// See docs/GGUF.md "P8: real batched prefill GEMM" for the wider plan and
// the reference kernel catalog this file ports from.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

constexpr int Q8_0_BLOCK = 32;
constexpr int Q8_0_BLOCK_BYTES = 34;  // 2 byte fp16 d + 32 byte int8 qs
constexpr int K_TILE = 16;            // WMMA inner-K size
constexpr int K_TILES_PER_BLOCK = Q8_0_BLOCK / K_TILE;  // == 2

// ---- scalar conversion helpers (same shape as in gguf_k_gemv.hip) -------

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

// Load 16 K-elements from an activation row into a half16_t WMMA b operand.
// For fp16 input this is a single packed vector load. For bf16 / f32 input,
// each lane converts element-by-element. Per-K-tile lookups are 16 elements
// in row-contiguous memory; address arithmetic is identical across types.
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
  // Fast path: direct vector load. Caller guarantees 16-element alignment
  // along the K axis (in_features % 32 == 0 and row stride == in_features).
  return *reinterpret_cast<const half16_t*>(p);
}

// -------------------------------------------------------------------------
// gguf_q8_0_prefill_wmma_kernel
//
// One block (wave32) computes a TM x TN tile of `out`, where
//   AWQ_TILE_M = TM rows of out_features (output columns of the GEMM),
//   AWQ_TILE_N = TN rows of `rows` (token rows / activation rows).
// `qweight` is raw Q8_0 byte stream: per output row `c`,
//   row_bytes = (in_features / 32) * 34
//   block b at offset c * row_bytes + b * 34
//     [0..1]  fp16 d
//     [2..33] int8 qs[32]
// -------------------------------------------------------------------------
template <typename scalar_t, typename out_t, int AWQ_TILE_M, int AWQ_TILE_N>
__launch_bounds__(32, 8)
__global__ void gguf_q8_0_prefill_wmma_kernel(
    const scalar_t* __restrict__ x,        // [rows, in_features]
    const uint8_t* __restrict__ qweight,   // [out_features, in_features]_q8_0
    out_t* __restrict__ out,               // [rows, out_features]
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

  const int blocks_per_row = in_features / Q8_0_BLOCK;
  const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * Q8_0_BLOCK_BYTES;

  // Per-TM: which output column each WMMA lane writes / reads.
  // The PARO lane convention is reused unchanged: lanes 0..15 and 16..31
  // share the same 16 output columns but hold complementary acc slots.
  int my_out[TM];
  int safe_out[TM];
  #pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    my_out[tm] = out_start + tm * 16 + (tid & 15);
    safe_out[tm] = (my_out[tm] < out_features) ? my_out[tm] : (out_features - 1);
  }

  // Per-TN: which row each WMMA lane gathers activations from.
  int my_token[TN];
  int safe_token[TN];
  #pragma unroll
  for (int tn = 0; tn < TN; ++tn) {
    my_token[tn] = row_start + tn * 16 + (tid & 15);
    safe_token[tn] = (my_token[tn] < rows) ? my_token[tn] : (rows - 1);
  }

  // Output accumulators, one float8_t per (TM, TN) WMMA tile.
  float8_t acc[TM][TN];
  #pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    #pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      acc[tm][tn] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
  }

  // Iterate over Q8_0 blocks along the K axis. Each block carries
  // K_TILES_PER_BLOCK == 2 K-tiles of 16 elements.
  for (int blk = 0; blk < blocks_per_row; ++blk) {
    // Per-TM: pointer to this column's Q8_0 block in the raw byte stream
    // (one block has 1 fp16 d + 32 int8 qs).
    const uint8_t* block_ptr[TM];
    half_t d_h[TM];
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      block_ptr[tm] = qweight + static_cast<int64_t>(safe_out[tm]) * row_bytes +
                      static_cast<int64_t>(blk) * Q8_0_BLOCK_BYTES;
      d_h[tm] = static_cast<half_t>(fp16_bytes_to_float(block_ptr[tm]));
    }

    const int g_k_base = blk * Q8_0_BLOCK;

    #pragma unroll
    for (int kt = 0; kt < K_TILES_PER_BLOCK; ++kt) {
      const int k_off = kt * K_TILE;

      // Activation tile: each WMMA b operand spans 16 K-elements of one
      // row. Lanes within a wave each hold the b operand for a distinct
      // token row in the TN partition.
      half16_t b_reg[TN];
      #pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
        const scalar_t* x_row =
            x + static_cast<int64_t>(safe_token[tn]) * in_features;
        b_reg[tn] = load_act_half16<scalar_t>(x_row + g_k_base + k_off);
      }

      // Weight tile: a single half16_t per TM holding 16 dequantized K
      // weights of this column. Read 16 int8 quants from this column's
      // current Q8_0 block, multiply by the per-block fp16 scale.
      half16_t a_reg;
      #pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        const int8_t* qs =
            reinterpret_cast<const int8_t*>(block_ptr[tm] + 2);
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          a_reg[kk] =
              d_h[tm] * static_cast<half_t>(static_cast<float>(qs[k_off + kk]));
        }
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
              a_reg, b_reg[tn], acc[tm][tn]);
        }
      }
    }
  }

  // Store back into out[row, col]. PARO's lane-to-output-slot mapping
  // (8 floats per lane, 2 cols per j-step, tid >> 4 selects half-wave)
  // is reused verbatim.
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

// -------------------------------------------------------------------------
// Launcher. Picks the templated kernel by (tile_m, tile_n).
// Supported tiles match the PARO defaults so the dispatch heuristic is
// reusable: TM in {16, 32, 64}, TN in {16, 32}.
// -------------------------------------------------------------------------
template <typename scalar_t, typename out_t>
int launch_gguf_q8_0_prefill_wmma(
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
      (in_features % Q8_0_BLOCK) != 0 ||
      (tile_m != 16 && tile_m != 32 && tile_m != 64) ||
      (tile_n != 16 && tile_n != 32)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features + tile_m - 1) / tile_m),
      static_cast<unsigned int>((rows + tile_n - 1) / tile_n));

#define HIPENGINE_LAUNCH_Q8_0_WMMA(TM, TN) \
  hipLaunchKernelGGL( \
      (gguf_q8_0_prefill_wmma_kernel<scalar_t, out_t, TM, TN>), \
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

  if (tile_m == 16 && tile_n == 16) { HIPENGINE_LAUNCH_Q8_0_WMMA(16, 16); }
  if (tile_m == 32 && tile_n == 16) { HIPENGINE_LAUNCH_Q8_0_WMMA(32, 16); }
  if (tile_m == 16 && tile_n == 32) { HIPENGINE_LAUNCH_Q8_0_WMMA(16, 32); }
  if (tile_m == 32 && tile_n == 32) { HIPENGINE_LAUNCH_Q8_0_WMMA(32, 32); }
  if (tile_m == 64 && tile_n == 16) { HIPENGINE_LAUNCH_Q8_0_WMMA(64, 16); }
  HIPENGINE_LAUNCH_Q8_0_WMMA(64, 32);
#undef HIPENGINE_LAUNCH_Q8_0_WMMA
}

// -------------------------------------------------------------------------
// P9.C1 dual gate+up variant for the shared-expert prefill path.
//
// Same per-block compute as ``gguf_q8_0_prefill_wmma_kernel`` but with two
// weight tensors. The output is the row-major concatenated layout
// ``[rows, out_features_a + out_features_b]`` that ``silu_mul_dual_out_*``
// consumes directly (matching the P9.B3 decode-side dual). Each col_tile
// targets either the gate (``qweight_a``) or the up (``qweight_b``)
// weights based on ``out_tile * AWQ_TILE_M >= out_features_a``; callers
// must align ``out_features_a`` and ``out_features_b`` to ``AWQ_TILE_M``.
// -------------------------------------------------------------------------
template <typename scalar_t, typename out_t, int AWQ_TILE_M, int AWQ_TILE_N>
__launch_bounds__(32, 8)
__global__ void gguf_q8_0_prefill_dual_wmma_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features_a,
    int out_features_b) {
  constexpr int TM = AWQ_TILE_M / 16;
  constexpr int TN = AWQ_TILE_N / 16;
  const int out_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int row_start = row_tile * AWQ_TILE_N;
  const int out_start = out_tile * AWQ_TILE_M;
  const int out_features_total = out_features_a + out_features_b;
  if (row_start >= rows || out_start >= out_features_total) return;

  const bool use_b = out_start >= out_features_a;
  const int local_out_start = use_b ? (out_start - out_features_a) : out_start;
  const int local_out_features = use_b ? out_features_b : out_features_a;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;

  const int blocks_per_row = in_features / Q8_0_BLOCK;
  const int64_t row_bytes = static_cast<int64_t>(blocks_per_row) * Q8_0_BLOCK_BYTES;

  int my_out[TM];
  int safe_out[TM];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    my_out[tm] = local_out_start + tm * 16 + (tid & 15);
    safe_out[tm] = (my_out[tm] < local_out_features) ? my_out[tm] : (local_out_features - 1);
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
    half_t d_h[TM];
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      block_ptr[tm] = qweight + static_cast<int64_t>(safe_out[tm]) * row_bytes +
                      static_cast<int64_t>(blk) * Q8_0_BLOCK_BYTES;
      d_h[tm] = static_cast<half_t>(fp16_bytes_to_float(block_ptr[tm]));
    }

    const int g_k_base = blk * Q8_0_BLOCK;

#pragma unroll
    for (int kt = 0; kt < K_TILES_PER_BLOCK; ++kt) {
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
        const int8_t* qs =
            reinterpret_cast<const int8_t*>(block_ptr[tm] + 2);
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          a_reg[kk] =
              d_h[tm] * static_cast<half_t>(static_cast<float>(qs[k_off + kk]));
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
    const int out_row = row_start + tn * 16 + (tid & 15);
    if (out_row >= rows) continue;
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int local_col = local_out_start + tm * 16 + 2 * j + (tid >> 4);
        if (local_col < local_out_features) {
          const int concat_col = use_b ? (out_features_a + local_col) : local_col;
          const int64_t idx =
              static_cast<int64_t>(out_row) * out_features_total + concat_col;
          store_output<out_t>(out, idx, static_cast<float>(acc[tm][tn][j]));
        }
      }
    }
  }
}

template <typename scalar_t, typename out_t>
int launch_gguf_q8_0_prefill_dual_wmma(
    const scalar_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  if (x == nullptr || qweight_a == nullptr || qweight_b == nullptr || out == nullptr ||
      rows <= 0 || in_features <= 0 || out_features_a <= 0 || out_features_b <= 0 ||
      (in_features % Q8_0_BLOCK) != 0 ||
      (tile_m != 16 && tile_m != 32 && tile_m != 64) ||
      (tile_n != 16 && tile_n != 32) ||
      (out_features_a % tile_m) != 0 || (out_features_b % tile_m) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_total = out_features_a + out_features_b;
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features_total + tile_m - 1) / tile_m),
      static_cast<unsigned int>((rows + tile_n - 1) / tile_n));

#define HIPENGINE_LAUNCH_Q8_0_DUAL_WMMA(TM, TN) \
  hipLaunchKernelGGL( \
      (gguf_q8_0_prefill_dual_wmma_kernel<scalar_t, out_t, TM, TN>), \
      grid, \
      block, \
      0, \
      stream, \
      x, \
      qweight_a, \
      qweight_b, \
      out, \
      static_cast<int>(rows), \
      static_cast<int>(in_features), \
      static_cast<int>(out_features_a), \
      static_cast<int>(out_features_b)); \
  return static_cast<int>(hipGetLastError())

  if (tile_m == 16 && tile_n == 16) { HIPENGINE_LAUNCH_Q8_0_DUAL_WMMA(16, 16); }
  if (tile_m == 32 && tile_n == 16) { HIPENGINE_LAUNCH_Q8_0_DUAL_WMMA(32, 16); }
  if (tile_m == 16 && tile_n == 32) { HIPENGINE_LAUNCH_Q8_0_DUAL_WMMA(16, 32); }
  if (tile_m == 32 && tile_n == 32) { HIPENGINE_LAUNCH_Q8_0_DUAL_WMMA(32, 32); }
  if (tile_m == 64 && tile_n == 16) { HIPENGINE_LAUNCH_Q8_0_DUAL_WMMA(64, 16); }
  HIPENGINE_LAUNCH_Q8_0_DUAL_WMMA(64, 32);
#undef HIPENGINE_LAUNCH_Q8_0_DUAL_WMMA
}

}  // namespace

// -------------------------------------------------------------------------
// C ABI exports. Naming matches the existing gguf_k_gemv.hip pattern:
//
//   hipengine_gguf_q8_0_wmma_prefill_<in>_<out>_out
//
// The Python registry binds these under the kernel keys
//   ("hip_gfx1100", "linear", "gguf_q8_0", "wmma_prefill_<in>_<out>_out").
// -------------------------------------------------------------------------

extern "C" int hipengine_gguf_q8_0_wmma_prefill_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q8_0_prefill_wmma<uint16_t, uint16_t>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q8_0_wmma_prefill_bf16_fp16_out(
    const uint16_t* x,
    const uint8_t* qweight,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q8_0_prefill_wmma<uint16_t, half_t>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q8_0_wmma_prefill_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q8_0_prefill_wmma<uint16_t, float>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q8_0_wmma_prefill_fp16_bf16_out(
    const half_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q8_0_prefill_wmma<half_t, uint16_t>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q8_0_wmma_prefill_fp16_fp16_out(
    const half_t* x,
    const uint8_t* qweight,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q8_0_prefill_wmma<half_t, half_t>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q8_0_prefill_dual_wmma<uint16_t, uint16_t>(
      x, qweight_a, qweight_b, out, rows, in_features,
      out_features_a, out_features_b, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q8_0_wmma_prefill_dual_gate_up_fp16_fp16_out(
    const half_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q8_0_prefill_dual_wmma<half_t, half_t>(
      x, qweight_a, qweight_b, out, rows, in_features,
      out_features_a, out_features_b, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q8_0_wmma_prefill_fp16_f32_out(
    const half_t* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q8_0_prefill_wmma<half_t, float>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q8_0_wmma_prefill_f32_bf16_out(
    const float* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q8_0_prefill_wmma<float, uint16_t>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q8_0_wmma_prefill_f32_fp16_out(
    const float* x,
    const uint8_t* qweight,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q8_0_prefill_wmma<float, half_t>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}

extern "C" int hipengine_gguf_q8_0_wmma_prefill_f32_f32_out(
    const float* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_gguf_q8_0_prefill_wmma<float, float>(
      x, qweight, out, rows, in_features, out_features, tile_m, tile_n, stream);
}
