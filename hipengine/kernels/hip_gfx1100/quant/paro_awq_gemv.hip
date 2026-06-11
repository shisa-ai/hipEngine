// Ported from nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py @ 59195ed.
// Selected and generic pack8 GEMV kernel bodies are preserved; wrappers are hipEngine C ABI.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

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

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

template <>
__device__ inline uint16_t float_to_scalar<uint16_t>(float v) {
  return float_to_bf16_bits(v);
}

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t v) {
  return bf16_bits_to_float(v);
}

__device__ inline int awq_shift_for_pack_lane(int lane) {
  const int packed_pos = (lane & 1) ? (4 + (lane >> 1)) : (lane >> 1);
  return packed_pos * 4;
}

__device__ inline float awq_prefill_dequant_weight(half_t scale_h, int q, int z) {
  // Match the stock FP16 prefill WMMA path's dequantization rounding: scale and
  // zero-point products are half-rounded before the f32 accumulation consumes
  // the dequantized weight.
  const half_t zp_h = -static_cast<half_t>(static_cast<float>(z)) * scale_h;
  const half_t w_h = scale_h * static_cast<half_t>(static_cast<float>(q)) + zp_h;
  return static_cast<float>(w_h);
}

__device__ inline float awq_prefill_dequant_weight(uint16_t scale_bits, int q, int z) {
  // BF16 pack8 kernels do not use the FP16 prefill WMMA path; keep their prior
  // float dequantization semantics.
  return static_cast<float>(q - z) * bf16_bits_to_float(scale_bits);
}

template <typename scalar_t>
__device__ inline float awq_decode_dequant_weight(scalar_t scale, int q, int z) {
  // Match the stock row-wise pack8 GEMV path: dequantize in f32 as
  // ``float(q - z) * float(scale)``.  This differs from the FP16 prefill WMMA
  // compatibility path above, which intentionally half-rounds the scale/zp
  // products before f32 accumulation.
  return static_cast<float>(q - z) * scalar_to_float(scale);
}

// RDNA3 HIP defaults to wave32 for this profile; even wave64-oriented
// shuffle probes only exchanged within 32-lane halves. Reduce each 32-lane
// wave with __shfl_down, then explicitly sum however many waves the block has.
// The pack8 wrappers allow
// 64/128/256 threads; the previous fixed 4-half-wave sum read uninitialized
// xchg slots for 64-thread launches and could write past xchg for 256.
#define PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7) \
  for (int offset = 16; offset > 0; offset >>= 1) { \
    acc0 += __shfl_down(acc0, offset); \
    acc1 += __shfl_down(acc1, offset); \
    acc2 += __shfl_down(acc2, offset); \
    acc3 += __shfl_down(acc3, offset); \
    acc4 += __shfl_down(acc4, offset); \
    acc5 += __shfl_down(acc5, offset); \
    acc6 += __shfl_down(acc6, offset); \
    acc7 += __shfl_down(acc7, offset); \
  } \
  __shared__ float xchg[8 * 8]; \
  const int sub_id = tid >> 5; \
  const int sub_lane = tid & 31; \
  if (sub_lane == 0) { \
    xchg[sub_id * 8 + 0] = acc0; \
    xchg[sub_id * 8 + 1] = acc1; \
    xchg[sub_id * 8 + 2] = acc2; \
    xchg[sub_id * 8 + 3] = acc3; \
    xchg[sub_id * 8 + 4] = acc4; \
    xchg[sub_id * 8 + 5] = acc5; \
    xchg[sub_id * 8 + 6] = acc6; \
    xchg[sub_id * 8 + 7] = acc7; \
  } \
  __syncthreads(); \
  float r0 = 0.0f; \
  float r1 = 0.0f; \
  float r2 = 0.0f; \
  float r3 = 0.0f; \
  float r4 = 0.0f; \
  float r5 = 0.0f; \
  float r6 = 0.0f; \
  float r7 = 0.0f; \
  if (tid == 0) { \
    const int sub_groups = blockDim.x >> 5; \
    for (int g = 0; g < sub_groups; ++g) { \
      r0 += xchg[g * 8 + 0]; \
      r1 += xchg[g * 8 + 1]; \
      r2 += xchg[g * 8 + 2]; \
      r3 += xchg[g * 8 + 3]; \
      r4 += xchg[g * 8 + 4]; \
      r5 += xchg[g * 8 + 5]; \
      r6 += xchg[g * 8 + 6]; \
      r7 += xchg[g * 8 + 7]; \
    } \
  }


template <int AWQ_TILE_M, int AWQ_TILE_N, bool qweight_transposed>
__launch_bounds__(32, 8)
__global__ void awq_fusedw4_prefill_fp16_kernel(
    const int32_t* __restrict__ qweight_t,
    const int32_t* __restrict__ qzeros,
    const half_t* __restrict__ scales,
    const half_t* __restrict__ x,
    half_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features,
    int out_packed,
    int group_size) {
  constexpr int TM = AWQ_TILE_M / 16;
  constexpr int TN = AWQ_TILE_N / 16;
  const int out_tile = blockIdx.x;
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int row_start = row_tile * AWQ_TILE_N;
  const int out_start = out_tile * AWQ_TILE_M;
  if (row_start >= rows || out_start >= out_features) return;

  const int groups = in_features / group_size;
  const int k_tiles_per_group = group_size / 16;

  int my_out[TM];
  int safe_out[TM];
  int out_pack[TM];
  int shift[TM];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    my_out[tm] = out_start + tm * 16 + (tid & 15);
    safe_out[tm] = (my_out[tm] < out_features) ? my_out[tm] : (out_features - 1);
    out_pack[tm] = safe_out[tm] >> 3;
    shift[tm] = awq_shift_for_pack_lane(safe_out[tm] & 7);
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

  for (int g = 0; g < groups; ++g) {
    half_t scale_h[TM];
    half_t zp_h[TM];
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      scale_h[tm] = scales[static_cast<int64_t>(g) * out_features + safe_out[tm]];
      const uint32_t zp_packed = static_cast<uint32_t>(qzeros[static_cast<int64_t>(g) * out_packed + out_pack[tm]]);
      const int zp_q = static_cast<int>((zp_packed >> shift[tm]) & 0xFu);
      zp_h[tm] = -static_cast<half_t>(static_cast<float>(zp_q)) * scale_h[tm];
    }

    const int g_k_base = g * group_size;
    for (int kt = 0; kt < k_tiles_per_group; ++kt) {
      const int k_off = kt * 16;
      uint32_t pw[TM][16];
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        if constexpr (qweight_transposed) {
          const int32_t* __restrict__ qw_row = qweight_t + static_cast<int64_t>(out_pack[tm]) * in_features;
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            pw[tm][kk] = static_cast<uint32_t>(qw_row[g_k_base + k_off + kk]);
          }
        } else {
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            const int k = g_k_base + k_off + kk;
            pw[tm][kk] = static_cast<uint32_t>(qweight_t[static_cast<int64_t>(k) * out_packed + out_pack[tm]]);
          }
        }
      }

      half16_t b_reg[TN];
#pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
        const half_t* __restrict__ x_row = x + static_cast<int64_t>(safe_token[tn]) * in_features;
        b_reg[tn] = *reinterpret_cast<const half16_t*>(x_row + g_k_base + k_off);
      }

      half16_t a_reg;
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          const int q = static_cast<int>((pw[tm][kk] >> shift[tm]) & 0xFu);
          a_reg[kk] = scale_h[tm] * static_cast<half_t>(static_cast<float>(q)) + zp_h[tm];
        }
#pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg[tn], acc[tm][tn]);
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
          out[static_cast<int64_t>(out_row) * out_features + col] = static_cast<half_t>(static_cast<float>(acc[tm][tn][j]));
        }
      }
    }
  }
}

template <bool qweight_transposed>
int launch_awq_fusedw4_prefill_fp16(
    const half_t* x,
    const int32_t* qweight_t,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || (group_size % 16) != 0 ||
      (tile_m != 16 && tile_m != 32 && tile_m != 64) ||
      (tile_n != 16 && tile_n != 32)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features = out_packed * 8;
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>((out_features + tile_m - 1) / tile_m),
      static_cast<unsigned int>((rows + tile_n - 1) / tile_n));

#define HIPENGINE_LAUNCH_FUSEDW4(TM, TN) \
  hipLaunchKernelGGL( \
      (awq_fusedw4_prefill_fp16_kernel<TM, TN, qweight_transposed>), \
      grid, \
      block, \
      0, \
      stream, \
      qweight_t, \
      qzeros, \
      scales, \
      x, \
      out, \
      static_cast<int>(rows), \
      static_cast<int>(in_features), \
      static_cast<int>(out_features), \
      static_cast<int>(out_packed), \
      static_cast<int>(group_size)); \
  return static_cast<int>(hipGetLastError())

  if (tile_m == 16 && tile_n == 16) { HIPENGINE_LAUNCH_FUSEDW4(16, 16); }
  if (tile_m == 32 && tile_n == 16) { HIPENGINE_LAUNCH_FUSEDW4(32, 16); }
  if (tile_m == 16 && tile_n == 32) { HIPENGINE_LAUNCH_FUSEDW4(16, 32); }
  if (tile_m == 32 && tile_n == 32) { HIPENGINE_LAUNCH_FUSEDW4(32, 32); }
  if (tile_m == 64 && tile_n == 16) { HIPENGINE_LAUNCH_FUSEDW4(64, 16); }
  HIPENGINE_LAUNCH_FUSEDW4(64, 32);
#undef HIPENGINE_LAUNCH_FUSEDW4
}

template <int AWQ_TILE_M, int AWQ_TILE_N>
__launch_bounds__(32, 8)
__global__ void awq_fusedw4_prefill_dual_fp16_kernel(
    const half_t* __restrict__ x_a,
    const half_t* __restrict__ x_b,
    const int32_t* __restrict__ qweight_a_t,
    const int32_t* __restrict__ qzeros_a,
    const half_t* __restrict__ scales_a,
    const int32_t* __restrict__ qweight_b_t,
    const int32_t* __restrict__ qzeros_b,
    const half_t* __restrict__ scales_b,
    half_t* __restrict__ out_a,
    half_t* __restrict__ out_b,
    int rows,
    int in_features,
    int out_features_a,
    int out_packed_a,
    int out_features_b,
    int out_packed_b,
    int group_size,
    int out_tiles_a) {
  constexpr int TM = AWQ_TILE_M / 16;
  constexpr int TN = AWQ_TILE_N / 16;
  const bool use_b = blockIdx.x >= out_tiles_a;
  const int out_tile = use_b ? static_cast<int>(blockIdx.x) - out_tiles_a : static_cast<int>(blockIdx.x);
  const int row_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const half_t* __restrict__ x = use_b ? x_b : x_a;
  const int32_t* __restrict__ qweight_t = use_b ? qweight_b_t : qweight_a_t;
  const int32_t* __restrict__ qzeros = use_b ? qzeros_b : qzeros_a;
  const half_t* __restrict__ scales = use_b ? scales_b : scales_a;
  half_t* __restrict__ out = use_b ? out_b : out_a;
  const int out_features = use_b ? out_features_b : out_features_a;
  const int out_packed = use_b ? out_packed_b : out_packed_a;
  const int row_start = row_tile * AWQ_TILE_N;
  const int out_start = out_tile * AWQ_TILE_M;
  if (row_start >= rows || out_start >= out_features) return;

  const int groups = in_features / group_size;
  const int k_tiles_per_group = group_size / 16;

  int my_out[TM];
  int safe_out[TM];
  int out_pack[TM];
  int shift[TM];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    my_out[tm] = out_start + tm * 16 + (tid & 15);
    safe_out[tm] = (my_out[tm] < out_features) ? my_out[tm] : (out_features - 1);
    out_pack[tm] = safe_out[tm] >> 3;
    shift[tm] = awq_shift_for_pack_lane(safe_out[tm] & 7);
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

  for (int g = 0; g < groups; ++g) {
    half_t scale_h[TM];
    half_t zp_h[TM];
#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
      scale_h[tm] = scales[static_cast<int64_t>(g) * out_features + safe_out[tm]];
      const uint32_t zp_packed = static_cast<uint32_t>(qzeros[static_cast<int64_t>(g) * out_packed + out_pack[tm]]);
      const int zp_q = static_cast<int>((zp_packed >> shift[tm]) & 0xFu);
      zp_h[tm] = -static_cast<half_t>(static_cast<float>(zp_q)) * scale_h[tm];
    }

    const int g_k_base = g * group_size;
    for (int kt = 0; kt < k_tiles_per_group; ++kt) {
      const int k_off = kt * 16;
      uint32_t pw[TM][16];
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
        const int32_t* __restrict__ qw_row = qweight_t + static_cast<int64_t>(out_pack[tm]) * in_features;
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          pw[tm][kk] = static_cast<uint32_t>(qw_row[g_k_base + k_off + kk]);
        }
      }

      half16_t b_reg[TN];
#pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
        const half_t* __restrict__ x_row = x + static_cast<int64_t>(safe_token[tn]) * in_features;
        b_reg[tn] = *reinterpret_cast<const half16_t*>(x_row + g_k_base + k_off);
      }

      half16_t a_reg;
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          const int q = static_cast<int>((pw[tm][kk] >> shift[tm]) & 0xFu);
          a_reg[kk] = scale_h[tm] * static_cast<half_t>(static_cast<float>(q)) + zp_h[tm];
        }
#pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
          acc[tm][tn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg[tn], acc[tm][tn]);
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
          out[static_cast<int64_t>(out_row) * out_features + col] = static_cast<half_t>(static_cast<float>(acc[tm][tn][j]));
        }
      }
    }
  }
}

int launch_awq_fusedw4_prefill_dual_fp16(
    const half_t* x_a,
    const half_t* x_b,
    const int32_t* qweight_a_t,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b_t,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out_a,
    half_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || (group_size % 16) != 0 ||
      (tile_m != 16 && tile_m != 32 && tile_m != 64) ||
      (tile_n != 16 && tile_n != 32)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features_a = out_packed_a * 8;
  const int64_t out_features_b = out_packed_b * 8;
  const int64_t out_tiles_a = (out_features_a + tile_m - 1) / tile_m;
  const int64_t out_tiles_b = (out_features_b + tile_m - 1) / tile_m;
  const dim3 block(32);
  const dim3 grid(
      static_cast<unsigned int>(out_tiles_a + out_tiles_b),
      static_cast<unsigned int>((rows + tile_n - 1) / tile_n));

#define HIPENGINE_LAUNCH_DUAL_FUSEDW4(TM, TN) \
  hipLaunchKernelGGL( \
      (awq_fusedw4_prefill_dual_fp16_kernel<TM, TN>), \
      grid, \
      block, \
      0, \
      stream, \
      x_a, \
      x_b, \
      qweight_a_t, \
      qzeros_a, \
      scales_a, \
      qweight_b_t, \
      qzeros_b, \
      scales_b, \
      out_a, \
      out_b, \
      static_cast<int>(rows), \
      static_cast<int>(in_features), \
      static_cast<int>(out_features_a), \
      static_cast<int>(out_packed_a), \
      static_cast<int>(out_features_b), \
      static_cast<int>(out_packed_b), \
      static_cast<int>(group_size), \
      static_cast<int>(out_tiles_a)); \
  return static_cast<int>(hipGetLastError())

  if (tile_m == 16 && tile_n == 16) { HIPENGINE_LAUNCH_DUAL_FUSEDW4(16, 16); }
  if (tile_m == 32 && tile_n == 16) { HIPENGINE_LAUNCH_DUAL_FUSEDW4(32, 16); }
  if (tile_m == 16 && tile_n == 32) { HIPENGINE_LAUNCH_DUAL_FUSEDW4(16, 32); }
  if (tile_m == 32 && tile_n == 32) { HIPENGINE_LAUNCH_DUAL_FUSEDW4(32, 32); }
  if (tile_m == 64 && tile_n == 16) { HIPENGINE_LAUNCH_DUAL_FUSEDW4(64, 16); }
  HIPENGINE_LAUNCH_DUAL_FUSEDW4(64, 32);
#undef HIPENGINE_LAUNCH_DUAL_FUSEDW4
}

template <typename scalar_t, bool qweight_transposed>
__launch_bounds__(128, 4)
__global__ void gemv_awq_pack8_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const scalar_t* __restrict__ scales,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features,
    int64_t out_packed,
    int group_size,
    bool has_bias) {
  const int64_t out_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t out_col = out_pack << 3;
  const int64_t x_base = row * in_features;
  const int64_t qw_base = qweight_transposed ? out_pack * in_features : out_pack;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  float acc4 = 0.0f;
  float acc5 = 0.0f;
  float acc6 = 0.0f;
  float acc7 = 0.0f;

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * out_packed + out_pack]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales[group * out_features + out_col];
    const float s0 = scalar_to_float(scale_ptr[0]);
    const float s1 = scalar_to_float(scale_ptr[1]);
    const float s2 = scalar_to_float(scale_ptr[2]);
    const float s3 = scalar_to_float(scale_ptr[3]);
    const float s4 = scalar_to_float(scale_ptr[4]);
    const float s5 = scalar_to_float(scale_ptr[5]);
    const float s6 = scalar_to_float(scale_ptr[6]);
    const float s7 = scalar_to_float(scale_ptr[7]);

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xv = scalar_to_float(x[x_base + k_base + j]);
      const int64_t qw_idx = qweight_transposed
          ? (qw_base + k_base + j)
          : ((k_base + j) * out_packed + out_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
      acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - z0) * s0, acc0);
      acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - z1) * s1, acc1);
      acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - z2) * s2, acc2);
      acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - z3) * s3, acc3);
      acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - z4) * s4, acc4);
      acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - z5) * s5, acc5);
      acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - z6) * s6, acc6);
      acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - z7) * s7, acc7);
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * out_packed + out_pack]);
    const int64_t qw_idx = qweight_transposed ? (qw_base + in_col) : (in_col * out_packed + out_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales[group * out_features + out_col];
    const float xv = scalar_to_float(x[x_base + in_col]);
    acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF)) * scalar_to_float(scale_ptr[0]), acc0);
    acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF)) * scalar_to_float(scale_ptr[1]), acc1);
    acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF)) * scalar_to_float(scale_ptr[2]), acc2);
    acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF)) * scalar_to_float(scale_ptr[3]), acc3);
    acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF)) * scalar_to_float(scale_ptr[4]), acc4);
    acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF)) * scalar_to_float(scale_ptr[5]), acc5);
    acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF)) * scalar_to_float(scale_ptr[6]), acc6);
    acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF)) * scalar_to_float(scale_ptr[7]), acc7);
  }

  PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

  if (tid == 0) {
    scalar_t* out_ptr = &out[row * out_features + out_col];
    if (has_bias) {
      out_ptr[0] = float_to_scalar<scalar_t>(r0 + scalar_to_float(bias[out_col + 0]));
      out_ptr[1] = float_to_scalar<scalar_t>(r1 + scalar_to_float(bias[out_col + 1]));
      out_ptr[2] = float_to_scalar<scalar_t>(r2 + scalar_to_float(bias[out_col + 2]));
      out_ptr[3] = float_to_scalar<scalar_t>(r3 + scalar_to_float(bias[out_col + 3]));
      out_ptr[4] = float_to_scalar<scalar_t>(r4 + scalar_to_float(bias[out_col + 4]));
      out_ptr[5] = float_to_scalar<scalar_t>(r5 + scalar_to_float(bias[out_col + 5]));
      out_ptr[6] = float_to_scalar<scalar_t>(r6 + scalar_to_float(bias[out_col + 6]));
      out_ptr[7] = float_to_scalar<scalar_t>(r7 + scalar_to_float(bias[out_col + 7]));
    } else {
      out_ptr[0] = float_to_scalar<scalar_t>(r0);
      out_ptr[1] = float_to_scalar<scalar_t>(r1);
      out_ptr[2] = float_to_scalar<scalar_t>(r2);
      out_ptr[3] = float_to_scalar<scalar_t>(r3);
      out_ptr[4] = float_to_scalar<scalar_t>(r4);
      out_ptr[5] = float_to_scalar<scalar_t>(r5);
      out_ptr[6] = float_to_scalar<scalar_t>(r6);
      out_ptr[7] = float_to_scalar<scalar_t>(r7);
    }
  }
}

// M12.6: multi-row W4 pack8 GEMV.  The stock ``awq_fusedw4_prefill_fp16_kernel``
// runs with WMMA TILE_N=16 (smallest supported) even for our 4 verifier rows,
// wasting 12/16 of every WMMA tile and only filling ~17% of the W7900's wave
// capacity.  This variant collapses the per-(out_pack, row) grid into a
// single ``out_pack`` grid and loops over all ``rows`` per block, so each W4
// weight tile streams from HBM exactly once and feeds all verifier rows.  No
// WMMA: regular FMA gives full per-thread accumulation across all real rows
// without wasted slots.  ``MAX_ROWS`` templates the per-thread accumulator
// matrix; we instantiate the variant inside ``launch_gemv_awq_pack8_multi_row``.
template <typename scalar_t, bool qweight_transposed, int MAX_ROWS, bool decode_dequant>
__launch_bounds__(128, 4)
__global__ void gemv_awq_pack8_multi_row_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const scalar_t* __restrict__ scales,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t out_packed,
    int group_size,
    bool has_bias) {
  const int64_t out_pack = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t out_col = out_pack << 3;
  const int64_t qw_base = qweight_transposed ? out_pack * in_features : out_pack;

  float acc[MAX_ROWS][8];
#pragma unroll
  for (int r = 0; r < MAX_ROWS; ++r) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      acc[r][j] = 0.0f;
    }
  }

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * out_packed + out_pack]);
    int z[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      z[j] = static_cast<int>((pz >> awq_shift_for_pack_lane(j)) & 0xF);
    }
    const scalar_t* __restrict__ scale_ptr = &scales[group * out_features + out_col];
    scalar_t s[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j] = scale_ptr[j];
    }
    uint32_t pw[8];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int64_t qw_idx = qweight_transposed
          ? (qw_base + k_base + kk)
          : ((k_base + kk) * out_packed + out_pack);
      pw[kk] = static_cast<uint32_t>(qweight[qw_idx]);
    }
#pragma unroll
    for (int r = 0; r < MAX_ROWS; ++r) {
      if (r >= rows) break;
      const int64_t x_base = static_cast<int64_t>(r) * in_features;
      float xs[8];
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        xs[kk] = scalar_to_float(x[x_base + k_base + kk]);
      }
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        const float xv = xs[kk];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int q = static_cast<int>((pw[kk] >> awq_shift_for_pack_lane(j)) & 0xF);
          const float wq = decode_dequant
              ? awq_decode_dequant_weight(s[j], q, z[j])
              : awq_prefill_dequant_weight(s[j], q, z[j]);
          acc[r][j] = fmaf(xv, wq, acc[r][j]);
        }
      }
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * out_packed + out_pack]);
    const int64_t qw_idx = qweight_transposed ? (qw_base + in_col) : (in_col * out_packed + out_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* __restrict__ scale_ptr = &scales[group * out_features + out_col];
#pragma unroll
    for (int r = 0; r < MAX_ROWS; ++r) {
      if (r >= rows) break;
      const float xv = scalar_to_float(x[static_cast<int64_t>(r) * in_features + in_col]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int q = static_cast<int>((pw >> awq_shift_for_pack_lane(j)) & 0xF);
        const int z = static_cast<int>((pz >> awq_shift_for_pack_lane(j)) & 0xF);
        const float wq = decode_dequant
            ? awq_decode_dequant_weight(scale_ptr[j], q, z)
            : awq_prefill_dequant_weight(scale_ptr[j], q, z);
        acc[r][j] = fmaf(xv, wq, acc[r][j]);
      }
    }
  }

  // Per-row block-wide reduction.  Reuse the PARO_PACK8 reduction macro by
  // copying the row's accumulators into the standard acc0..7 names; the macro
  // uses a shared ``xchg`` slab so we must serialize the rows.
#pragma unroll
  for (int r = 0; r < MAX_ROWS; ++r) {
    if (r >= rows) {
      break;
    }
    float acc0 = acc[r][0];
    float acc1 = acc[r][1];
    float acc2 = acc[r][2];
    float acc3 = acc[r][3];
    float acc4 = acc[r][4];
    float acc5 = acc[r][5];
    float acc6 = acc[r][6];
    float acc7 = acc[r][7];
    PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);
    if (tid == 0) {
      scalar_t* __restrict__ out_ptr = &out[static_cast<int64_t>(r) * out_features + out_col];
      if (has_bias) {
        out_ptr[0] = float_to_scalar<scalar_t>(r0 + scalar_to_float(bias[out_col + 0]));
        out_ptr[1] = float_to_scalar<scalar_t>(r1 + scalar_to_float(bias[out_col + 1]));
        out_ptr[2] = float_to_scalar<scalar_t>(r2 + scalar_to_float(bias[out_col + 2]));
        out_ptr[3] = float_to_scalar<scalar_t>(r3 + scalar_to_float(bias[out_col + 3]));
        out_ptr[4] = float_to_scalar<scalar_t>(r4 + scalar_to_float(bias[out_col + 4]));
        out_ptr[5] = float_to_scalar<scalar_t>(r5 + scalar_to_float(bias[out_col + 5]));
        out_ptr[6] = float_to_scalar<scalar_t>(r6 + scalar_to_float(bias[out_col + 6]));
        out_ptr[7] = float_to_scalar<scalar_t>(r7 + scalar_to_float(bias[out_col + 7]));
      } else {
        out_ptr[0] = float_to_scalar<scalar_t>(r0);
        out_ptr[1] = float_to_scalar<scalar_t>(r1);
        out_ptr[2] = float_to_scalar<scalar_t>(r2);
        out_ptr[3] = float_to_scalar<scalar_t>(r3);
        out_ptr[4] = float_to_scalar<scalar_t>(r4);
        out_ptr[5] = float_to_scalar<scalar_t>(r5);
        out_ptr[6] = float_to_scalar<scalar_t>(r6);
        out_ptr[7] = float_to_scalar<scalar_t>(r7);
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t, bool qweight_transposed, bool separate_inputs>
__launch_bounds__(128, 4)
__global__ void gemv_awq_dual_pack8_kernel(
    const scalar_t* __restrict__ x_a,
    const scalar_t* __restrict__ x_b,
    const int32_t* __restrict__ qweight_a,
    const int32_t* __restrict__ qzeros_a,
    const scalar_t* __restrict__ scales_a,
    const int32_t* __restrict__ qweight_b,
    const int32_t* __restrict__ qzeros_b,
    const scalar_t* __restrict__ scales_b,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int group_size) {
  const int64_t global_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const bool use_b = global_pack >= out_packed_a;
  const int64_t local_pack = use_b ? (global_pack - out_packed_a) : global_pack;
  const int64_t local_out_packed = use_b ? out_packed_b : out_packed_a;
  const int64_t local_out_features = local_out_packed << 3;
  const int64_t out_features = (out_packed_a + out_packed_b) << 3;
  const int64_t out_col = global_pack << 3;
  const int64_t local_out_col = local_pack << 3;
  const int64_t x_base = row * in_features;
  const scalar_t* __restrict__ x = (separate_inputs && use_b) ? x_b : x_a;
  const int32_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;
  const int32_t* __restrict__ qzeros = use_b ? qzeros_b : qzeros_a;
  const scalar_t* __restrict__ scales = use_b ? scales_b : scales_a;
  const int64_t qweight_base = qweight_transposed ? (local_pack * in_features) : local_pack;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  float acc4 = 0.0f;
  float acc5 = 0.0f;
  float acc6 = 0.0f;
  float acc7 = 0.0f;

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
    const float s0 = scalar_to_float(scale_ptr[0]);
    const float s1 = scalar_to_float(scale_ptr[1]);
    const float s2 = scalar_to_float(scale_ptr[2]);
    const float s3 = scalar_to_float(scale_ptr[3]);
    const float s4 = scalar_to_float(scale_ptr[4]);
    const float s5 = scalar_to_float(scale_ptr[5]);
    const float s6 = scalar_to_float(scale_ptr[6]);
    const float s7 = scalar_to_float(scale_ptr[7]);

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xv = scalar_to_float(x[x_base + k_base + j]);
      const int64_t qw_idx = qweight_transposed
          ? (qweight_base + k_base + j)
          : ((k_base + j) * local_out_packed + local_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
      acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - z0) * s0, acc0);
      acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - z1) * s1, acc1);
      acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - z2) * s2, acc2);
      acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - z3) * s3, acc3);
      acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - z4) * s4, acc4);
      acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - z5) * s5, acc5);
      acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - z6) * s6, acc6);
      acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - z7) * s7, acc7);
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int64_t qw_idx = qweight_transposed
        ? (qweight_base + in_col)
        : (in_col * local_out_packed + local_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
    const float xv = scalar_to_float(x[x_base + in_col]);
    acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF)) * scalar_to_float(scale_ptr[0]), acc0);
    acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF)) * scalar_to_float(scale_ptr[1]), acc1);
    acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF)) * scalar_to_float(scale_ptr[2]), acc2);
    acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF)) * scalar_to_float(scale_ptr[3]), acc3);
    acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF)) * scalar_to_float(scale_ptr[4]), acc4);
    acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF)) * scalar_to_float(scale_ptr[5]), acc5);
    acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF)) * scalar_to_float(scale_ptr[6]), acc6);
    acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF)) * scalar_to_float(scale_ptr[7]), acc7);
  }

  PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

  if (tid == 0) {
    scalar_t* out_ptr = &out[row * out_features + out_col];
    out_ptr[0] = float_to_scalar<scalar_t>(r0);
    out_ptr[1] = float_to_scalar<scalar_t>(r1);
    out_ptr[2] = float_to_scalar<scalar_t>(r2);
    out_ptr[3] = float_to_scalar<scalar_t>(r3);
    out_ptr[4] = float_to_scalar<scalar_t>(r4);
    out_ptr[5] = float_to_scalar<scalar_t>(r5);
    out_ptr[6] = float_to_scalar<scalar_t>(r6);
    out_ptr[7] = float_to_scalar<scalar_t>(r7);
  }
}

// M12.6: multi-row dual pack8 W4 GEMV.  Same fusion idea as
// ``gemv_awq_pack8_multi_row_kernel`` but for the dual (gate+up) variant used
// by the linear-attention QKV+Z and full-attention QKV/O projections.  Each
// block reads its weight tile (either gate or up depending on out_pack range)
// exactly once and loops over all ``rows`` verifier rows.
// ``out_layout``: 0 = concatenated single output buffer (existing
// ``gemv_awq_dual_pack8_kernel`` ABI: ``out[row * (out_features_a + out_features_b) + out_col]``),
// 1 = separate ``out_a`` and ``out_b`` buffers (existing
// ``awq_fusedw4_prefill_dual_fp16_kernel`` ABI).
template <typename scalar_t, bool qweight_transposed, bool separate_inputs, int OUT_LAYOUT, int MAX_ROWS,
          bool decode_dequant = false>
__launch_bounds__(128, 4)
__global__ void gemv_awq_dual_pack8_multi_row_kernel(
    const scalar_t* __restrict__ x_a,
    const scalar_t* __restrict__ x_b,
    const int32_t* __restrict__ qweight_a,
    const int32_t* __restrict__ qzeros_a,
    const scalar_t* __restrict__ scales_a,
    const int32_t* __restrict__ qweight_b,
    const int32_t* __restrict__ qzeros_b,
    const scalar_t* __restrict__ scales_b,
    scalar_t* __restrict__ out_a,
    scalar_t* __restrict__ out_b,  // ignored when OUT_LAYOUT == 0
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int group_size) {
  const int64_t global_pack = blockIdx.x;
  const int tid = threadIdx.x;
  const bool use_b = global_pack >= out_packed_a;
  const int64_t local_pack = use_b ? (global_pack - out_packed_a) : global_pack;
  const int64_t local_out_packed = use_b ? out_packed_b : out_packed_a;
  const int64_t local_out_features = local_out_packed << 3;
  const int64_t out_features_a = out_packed_a << 3;
  const int64_t out_features_b = out_packed_b << 3;
  const int64_t out_features_concat = out_features_a + out_features_b;
  const int64_t out_col_concat = global_pack << 3;
  const int64_t local_out_col = local_pack << 3;
  const scalar_t* __restrict__ x = (separate_inputs && use_b) ? x_b : x_a;
  const int32_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;
  const int32_t* __restrict__ qzeros = use_b ? qzeros_b : qzeros_a;
  const scalar_t* __restrict__ scales = use_b ? scales_b : scales_a;
  scalar_t* __restrict__ out_buf = (OUT_LAYOUT == 0 || !use_b) ? out_a : out_b;
  const int64_t out_buf_stride = (OUT_LAYOUT == 0) ? out_features_concat : (use_b ? out_features_b : out_features_a);
  const int64_t out_col_local = (OUT_LAYOUT == 0) ? out_col_concat : local_out_col;
  const int64_t qweight_base = qweight_transposed ? (local_pack * in_features) : local_pack;

  float acc[MAX_ROWS][8];
#pragma unroll
  for (int r = 0; r < MAX_ROWS; ++r) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      acc[r][j] = 0.0f;
    }
  }

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    int z[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      z[j] = static_cast<int>((pz >> awq_shift_for_pack_lane(j)) & 0xF);
    }
    const scalar_t* __restrict__ scale_ptr = &scales[group * local_out_features + local_out_col];
    scalar_t s[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j] = scale_ptr[j];
    }
    uint32_t pw[8];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int64_t qw_idx = qweight_transposed
          ? (qweight_base + k_base + kk)
          : ((k_base + kk) * local_out_packed + local_pack);
      pw[kk] = static_cast<uint32_t>(qweight[qw_idx]);
    }
#pragma unroll
    for (int r = 0; r < MAX_ROWS; ++r) {
      if (r >= rows) break;
      const int64_t x_base = static_cast<int64_t>(r) * in_features;
      float xs[8];
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        xs[kk] = scalar_to_float(x[x_base + k_base + kk]);
      }
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        const float xv = xs[kk];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int q = static_cast<int>((pw[kk] >> awq_shift_for_pack_lane(j)) & 0xF);
          const float wq = decode_dequant
              ? awq_decode_dequant_weight(s[j], q, z[j])
              : awq_prefill_dequant_weight(s[j], q, z[j]);
          acc[r][j] = fmaf(xv, wq, acc[r][j]);
        }
      }
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int64_t qw_idx = qweight_transposed
        ? (qweight_base + in_col)
        : (in_col * local_out_packed + local_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* __restrict__ scale_ptr = &scales[group * local_out_features + local_out_col];
#pragma unroll
    for (int r = 0; r < MAX_ROWS; ++r) {
      if (r >= rows) break;
      const float xv = scalar_to_float(x[static_cast<int64_t>(r) * in_features + in_col]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int q = static_cast<int>((pw >> awq_shift_for_pack_lane(j)) & 0xF);
        const int z = static_cast<int>((pz >> awq_shift_for_pack_lane(j)) & 0xF);
        const float wq = decode_dequant
            ? awq_decode_dequant_weight(scale_ptr[j], q, z)
            : awq_prefill_dequant_weight(scale_ptr[j], q, z);
        acc[r][j] = fmaf(xv, wq, acc[r][j]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < MAX_ROWS; ++r) {
    if (r >= rows) break;
    float acc0 = acc[r][0];
    float acc1 = acc[r][1];
    float acc2 = acc[r][2];
    float acc3 = acc[r][3];
    float acc4 = acc[r][4];
    float acc5 = acc[r][5];
    float acc6 = acc[r][6];
    float acc7 = acc[r][7];
    PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);
    if (tid == 0) {
      scalar_t* __restrict__ out_ptr = &out_buf[static_cast<int64_t>(r) * out_buf_stride + out_col_local];
      out_ptr[0] = float_to_scalar<scalar_t>(r0);
      out_ptr[1] = float_to_scalar<scalar_t>(r1);
      out_ptr[2] = float_to_scalar<scalar_t>(r2);
      out_ptr[3] = float_to_scalar<scalar_t>(r3);
      out_ptr[4] = float_to_scalar<scalar_t>(r4);
      out_ptr[5] = float_to_scalar<scalar_t>(r5);
      out_ptr[6] = float_to_scalar<scalar_t>(r6);
      out_ptr[7] = float_to_scalar<scalar_t>(r7);
    }
    __syncthreads();
  }
}

template <typename scalar_t, bool qweight_transposed>
__launch_bounds__(128, 4)
__global__ void gemv_awq_dual_pack8_transposed_rotate_staged_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ rotated_a,
    scalar_t* __restrict__ rotated_b,
    const int16_t* __restrict__ pairs_a,
    const int16_t* __restrict__ pairs_b,
    const scalar_t* __restrict__ theta_a,
    const scalar_t* __restrict__ theta_b,
    const scalar_t* __restrict__ channel_scales_a,
    const scalar_t* __restrict__ channel_scales_b,
    const int32_t* __restrict__ qweight_a,
    const int32_t* __restrict__ qzeros_a,
    const scalar_t* __restrict__ scales_a,
    const int32_t* __restrict__ qweight_b,
    const int32_t* __restrict__ qzeros_b,
    const scalar_t* __restrict__ scales_b,
    scalar_t* __restrict__ out,
    int32_t* __restrict__ barrier,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int group_size,
    int krot,
    int barrier_count_target,
    int barrier_ready_value) {
  const int64_t global_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t groups = in_features / group_size;
  const int half_group = group_size / 2;
  const int64_t rotate_blocks = groups * 2;
  const int64_t row_base = row * in_features;

  if (global_pack < rotate_blocks) {
    const int rotation = static_cast<int>(global_pack / groups);
    const int group = static_cast<int>(global_pack - static_cast<int64_t>(rotation) * groups);
    const int base = group * group_size;
    const int col0 = base + tid;
    const int col1 = col0 + half_group;
    const int16_t* __restrict__ pairs = (rotation == 0) ? pairs_a : pairs_b;
    const scalar_t* __restrict__ theta = (rotation == 0) ? theta_a : theta_b;
    const scalar_t* __restrict__ channel_scales = (rotation == 0) ? channel_scales_a : channel_scales_b;
    scalar_t* __restrict__ rotated = (rotation == 0) ? rotated_a : rotated_b;
    extern __shared__ float rotate_buf[];

    if (tid < half_group) {
      rotate_buf[tid] = scalar_to_float(x[row_base + col0]) * scalar_to_float(channel_scales[col0]);
      rotate_buf[tid + half_group] = scalar_to_float(x[row_base + col1]) * scalar_to_float(channel_scales[col1]);
    }
    __syncthreads();

    for (int r = 0; r < krot; ++r) {
      if (tid < half_group) {
        const int pair_base = r * static_cast<int>(in_features) + base + 2 * tid;
        const int i = static_cast<int>(pairs[pair_base + 0]);
        const int j = static_cast<int>(pairs[pair_base + 1]);
        const float angle = scalar_to_float(theta[r * (in_features / 2) + group * half_group + tid]);
        float s, c;
        sincosf(angle, &s, &c);
        const float xi = rotate_buf[i];
        const float xj = rotate_buf[j];
        rotate_buf[i] = fmaf(xj, s, xi * c);
        rotate_buf[j] = fmaf(xi, -s, xj * c);
      }
      __syncthreads();
    }
    if (tid < half_group) {
      rotated[row_base + col0] = float_to_scalar<scalar_t>(rotate_buf[tid]);
      rotated[row_base + col1] = float_to_scalar<scalar_t>(rotate_buf[tid + half_group]);
    }
    __threadfence();
    if (tid == 0) {
      // M14.fuse.barrier: keyed barrier mode lets the host keep a cumulative
      // rotate-block counter in barrier[0] and a monotonically increasing
      // ready epoch in barrier[1].  The original memset-per-launch path passes
      // target=total_rotate and ready=1 after zeroing the barrier; keyed callers
      // pass cumulative target/epoch values and avoid the extra hipMemsetAsync.
      const int prior = atomicAdd(&barrier[0], 1);
      if (prior + 1 == barrier_count_target) {
        __threadfence();
        atomicExch(&barrier[1], barrier_ready_value);
      }
    }
  }

  while (atomicAdd(&barrier[1], 0) < barrier_ready_value) {
  }
  __threadfence();

  const bool use_b = global_pack >= out_packed_a;
  const int64_t local_pack = use_b ? (global_pack - out_packed_a) : global_pack;
  const int64_t local_out_packed = use_b ? out_packed_b : out_packed_a;
  const int64_t local_out_features = local_out_packed << 3;
  const int64_t out_features = (out_packed_a + out_packed_b) << 3;
  const int64_t out_col = global_pack << 3;
  const int64_t local_out_col = local_pack << 3;
  const scalar_t* __restrict__ x_rot = use_b ? rotated_b : rotated_a;
  const int32_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;
  const int32_t* __restrict__ qzeros = use_b ? qzeros_b : qzeros_a;
  const scalar_t* __restrict__ scales = use_b ? scales_b : scales_a;
  const int64_t qweight_base = qweight_transposed ? (local_pack * in_features) : local_pack;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  float acc4 = 0.0f;
  float acc5 = 0.0f;
  float acc6 = 0.0f;
  float acc7 = 0.0f;

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
    const float s0 = scalar_to_float(scale_ptr[0]);
    const float s1 = scalar_to_float(scale_ptr[1]);
    const float s2 = scalar_to_float(scale_ptr[2]);
    const float s3 = scalar_to_float(scale_ptr[3]);
    const float s4 = scalar_to_float(scale_ptr[4]);
    const float s5 = scalar_to_float(scale_ptr[5]);
    const float s6 = scalar_to_float(scale_ptr[6]);
    const float s7 = scalar_to_float(scale_ptr[7]);

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xv = scalar_to_float(x_rot[row_base + k_base + j]);
      const int64_t qw_idx = qweight_transposed
          ? (qweight_base + k_base + j)
          : ((k_base + j) * local_out_packed + local_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
      acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - z0) * s0, acc0);
      acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - z1) * s1, acc1);
      acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - z2) * s2, acc2);
      acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - z3) * s3, acc3);
      acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - z4) * s4, acc4);
      acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - z5) * s5, acc5);
      acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - z6) * s6, acc6);
      acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - z7) * s7, acc7);
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int64_t qw_idx = qweight_transposed
        ? (qweight_base + in_col)
        : (in_col * local_out_packed + local_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
    const float xv = scalar_to_float(x_rot[row_base + in_col]);
    acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF)) * scalar_to_float(scale_ptr[0]), acc0);
    acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF)) * scalar_to_float(scale_ptr[1]), acc1);
    acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF)) * scalar_to_float(scale_ptr[2]), acc2);
    acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF)) * scalar_to_float(scale_ptr[3]), acc3);
    acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF)) * scalar_to_float(scale_ptr[4]), acc4);
    acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF)) * scalar_to_float(scale_ptr[5]), acc5);
    acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF)) * scalar_to_float(scale_ptr[6]), acc6);
    acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF)) * scalar_to_float(scale_ptr[7]), acc7);
  }

  PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

  if (tid == 0) {
    scalar_t* out_ptr = &out[row * out_features + out_col];
    out_ptr[0] = float_to_scalar<scalar_t>(r0);
    out_ptr[1] = float_to_scalar<scalar_t>(r1);
    out_ptr[2] = float_to_scalar<scalar_t>(r2);
    out_ptr[3] = float_to_scalar<scalar_t>(r3);
    out_ptr[4] = float_to_scalar<scalar_t>(r4);
    out_ptr[5] = float_to_scalar<scalar_t>(r5);
    out_ptr[6] = float_to_scalar<scalar_t>(r6);
    out_ptr[7] = float_to_scalar<scalar_t>(r7);
  }
}

template <typename scalar_t, bool qweight_transposed>
__launch_bounds__(128, 4) __global__ void gemv_awq_selected_dual_pack8_strided_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const int32_t* __restrict__ qweight_a,
    const int32_t* __restrict__ qzeros_a,
    const scalar_t* __restrict__ scales_a,
    const int32_t* __restrict__ qweight_b,
    const int32_t* __restrict__ qzeros_b,
    const scalar_t* __restrict__ scales_b,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int group_size) {
  const int64_t global_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const bool use_b = global_pack >= out_packed_a;
  const int64_t local_pack = use_b ? (global_pack - out_packed_a) : global_pack;
  const int64_t local_out_packed = use_b ? out_packed_b : out_packed_a;
  const int64_t local_out_features = local_out_packed << 3;
  const int64_t out_features = (out_packed_a + out_packed_b) << 3;
  const int64_t out_col = global_pack << 3;
  const int64_t local_out_col = local_pack << 3;
  const int64_t groups = in_features / group_size;
  const int64_t lanes_per_x_row = static_cast<int64_t>(gridDim.y) / x_rows;
  const int64_t x_row = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t x_base = x_row * in_features;
  const int32_t* __restrict__ qweight_base = use_b ? qweight_b : qweight_a;
  const int32_t* __restrict__ qzeros_base = use_b ? qzeros_b : qzeros_a;
  const scalar_t* __restrict__ scales_base = use_b ? scales_b : scales_a;
  const int64_t expert_qweight_stride = qweight_transposed
      ? (local_out_packed * in_features)
      : (in_features * local_out_packed);
  const int32_t* __restrict__ qweight = qweight_base + expert_id * expert_qweight_stride
      + (qweight_transposed ? (local_pack * in_features) : 0);
  const int32_t* __restrict__ qzeros = qzeros_base + expert_id * groups * local_out_packed;
  const scalar_t* __restrict__ scales = scales_base + expert_id * groups * local_out_features;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  float acc4 = 0.0f;
  float acc5 = 0.0f;
  float acc6 = 0.0f;
  float acc7 = 0.0f;

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
    const float s0 = scalar_to_float(scale_ptr[0]);
    const float s1 = scalar_to_float(scale_ptr[1]);
    const float s2 = scalar_to_float(scale_ptr[2]);
    const float s3 = scalar_to_float(scale_ptr[3]);
    const float s4 = scalar_to_float(scale_ptr[4]);
    const float s5 = scalar_to_float(scale_ptr[5]);
    const float s6 = scalar_to_float(scale_ptr[6]);
    const float s7 = scalar_to_float(scale_ptr[7]);

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xv = scalar_to_float(x[x_base + k_base + j]);
      const int64_t qw_idx = qweight_transposed ? (k_base + j) : ((k_base + j) * local_out_packed + local_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
      acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - z0) * s0, acc0);
      acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - z1) * s1, acc1);
      acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - z2) * s2, acc2);
      acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - z3) * s3, acc3);
      acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - z4) * s4, acc4);
      acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - z5) * s5, acc5);
      acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - z6) * s6, acc6);
      acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - z7) * s7, acc7);
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int64_t qw_idx = qweight_transposed ? in_col : (in_col * local_out_packed + local_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
    const float xv = scalar_to_float(x[x_base + in_col]);
    acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF)) * scalar_to_float(scale_ptr[0]), acc0);
    acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF)) * scalar_to_float(scale_ptr[1]), acc1);
    acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF)) * scalar_to_float(scale_ptr[2]), acc2);
    acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF)) * scalar_to_float(scale_ptr[3]), acc3);
    acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF)) * scalar_to_float(scale_ptr[4]), acc4);
    acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF)) * scalar_to_float(scale_ptr[5]), acc5);
    acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF)) * scalar_to_float(scale_ptr[6]), acc6);
    acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF)) * scalar_to_float(scale_ptr[7]), acc7);
  }

  PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

  if (tid == 0) {
    scalar_t* out_ptr = &out[row * out_features + out_col];
    out_ptr[0] = float_to_scalar<scalar_t>(r0);
    out_ptr[1] = float_to_scalar<scalar_t>(r1);
    out_ptr[2] = float_to_scalar<scalar_t>(r2);
    out_ptr[3] = float_to_scalar<scalar_t>(r3);
    out_ptr[4] = float_to_scalar<scalar_t>(r4);
    out_ptr[5] = float_to_scalar<scalar_t>(r5);
    out_ptr[6] = float_to_scalar<scalar_t>(r6);
    out_ptr[7] = float_to_scalar<scalar_t>(r7);
  }
}

template <typename scalar_t, bool qweight_transposed>
__launch_bounds__(128, 4)
__global__ void gemv_awq_selected_dual_pack8_strided_rotate_staged_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ rotated,
    const int64_t* __restrict__ selected,
    const int16_t* __restrict__ pairs,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ channel_scales,
    const int32_t* __restrict__ qweight_a,
    const int32_t* __restrict__ qzeros_a,
    const scalar_t* __restrict__ scales_a,
    const int32_t* __restrict__ qweight_b,
    const int32_t* __restrict__ qzeros_b,
    const scalar_t* __restrict__ scales_b,
    scalar_t* __restrict__ out,
    int32_t* __restrict__ barrier,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int group_size,
    int krot,
    int barrier_count_target,
    int barrier_ready_value) {
  const int64_t bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t groups = in_features / group_size;
  const int64_t rotate_blocks = x_rows * groups;
  const int64_t out_packed_total = out_packed_a + out_packed_b;
  const int half_group = group_size / 2;

  if (bid < rotate_blocks) {
    const int64_t x_row = bid / groups;
    const int group = static_cast<int>(bid - x_row * groups);
    const int base = group * group_size;
    const int col0 = base + tid;
    const int col1 = col0 + half_group;
    const int64_t row_base = x_row * in_features;
    extern __shared__ float rotate_buf[];

    if (tid < half_group) {
      rotate_buf[tid] = scalar_to_float(x[row_base + col0]) * scalar_to_float(channel_scales[col0]);
      rotate_buf[tid + half_group] = scalar_to_float(x[row_base + col1]) * scalar_to_float(channel_scales[col1]);
    }
    __syncthreads();

    for (int r = 0; r < krot; ++r) {
      if (tid < half_group) {
        const int pair_base = r * static_cast<int>(in_features) + base + 2 * tid;
        const int i = static_cast<int>(pairs[pair_base + 0]);
        const int j = static_cast<int>(pairs[pair_base + 1]);
        const float angle = scalar_to_float(theta[r * (in_features / 2) + group * half_group + tid]);
        float s, c;
        sincosf(angle, &s, &c);
        const float xi = rotate_buf[i];
        const float xj = rotate_buf[j];
        rotate_buf[i] = fmaf(xj, s, xi * c);
        rotate_buf[j] = fmaf(xi, -s, xj * c);
      }
      __syncthreads();
    }
    if (tid < half_group) {
      rotated[row_base + col0] = float_to_scalar<scalar_t>(rotate_buf[tid]);
      rotated[row_base + col1] = float_to_scalar<scalar_t>(rotate_buf[tid + half_group]);
    }
    __threadfence();
    if (tid == 0) {
      const int prior = atomicAdd(&barrier[0], 1);
      if (prior + 1 == barrier_count_target) {
        __threadfence();
        atomicExch(&barrier[1], barrier_ready_value);
      }
    }
    return;
  }

  while (atomicAdd(&barrier[1], 0) < barrier_ready_value) {
  }
  __threadfence();

  const int64_t gemv_bid = bid - rotate_blocks;
  const int64_t row = gemv_bid / out_packed_total;
  const int64_t global_pack = gemv_bid - row * out_packed_total;
  if (row >= rows) return;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const bool use_b = global_pack >= out_packed_a;
  const int64_t local_pack = use_b ? (global_pack - out_packed_a) : global_pack;
  const int64_t local_out_packed = use_b ? out_packed_b : out_packed_a;
  const int64_t local_out_features = local_out_packed << 3;
  const int64_t out_features = out_packed_total << 3;
  const int64_t out_col = global_pack << 3;
  const int64_t local_out_col = local_pack << 3;
  const int64_t lanes_per_x_row = rows / x_rows;
  const int64_t x_row = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t x_base = x_row * in_features;
  const int32_t* __restrict__ qweight_base = use_b ? qweight_b : qweight_a;
  const int32_t* __restrict__ qzeros_base = use_b ? qzeros_b : qzeros_a;
  const scalar_t* __restrict__ scales_base = use_b ? scales_b : scales_a;
  const int64_t expert_qweight_stride = qweight_transposed
      ? (local_out_packed * in_features)
      : (in_features * local_out_packed);
  const int32_t* __restrict__ qweight = qweight_base + expert_id * expert_qweight_stride
      + (qweight_transposed ? (local_pack * in_features) : 0);
  const int32_t* __restrict__ qzeros = qzeros_base + expert_id * groups * local_out_packed;
  const scalar_t* __restrict__ scales = scales_base + expert_id * groups * local_out_features;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  float acc4 = 0.0f;
  float acc5 = 0.0f;
  float acc6 = 0.0f;
  float acc7 = 0.0f;

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
    const float s0 = scalar_to_float(scale_ptr[0]);
    const float s1 = scalar_to_float(scale_ptr[1]);
    const float s2 = scalar_to_float(scale_ptr[2]);
    const float s3 = scalar_to_float(scale_ptr[3]);
    const float s4 = scalar_to_float(scale_ptr[4]);
    const float s5 = scalar_to_float(scale_ptr[5]);
    const float s6 = scalar_to_float(scale_ptr[6]);
    const float s7 = scalar_to_float(scale_ptr[7]);

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xv = scalar_to_float(rotated[x_base + k_base + j]);
      const int64_t qw_idx = qweight_transposed ? (k_base + j) : ((k_base + j) * local_out_packed + local_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
      acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - z0) * s0, acc0);
      acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - z1) * s1, acc1);
      acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - z2) * s2, acc2);
      acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - z3) * s3, acc3);
      acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - z4) * s4, acc4);
      acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - z5) * s5, acc5);
      acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - z6) * s6, acc6);
      acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - z7) * s7, acc7);
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int64_t qw_idx = qweight_transposed ? in_col : (in_col * local_out_packed + local_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
    const float xv = scalar_to_float(rotated[x_base + in_col]);
    acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF)) * scalar_to_float(scale_ptr[0]), acc0);
    acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF)) * scalar_to_float(scale_ptr[1]), acc1);
    acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF)) * scalar_to_float(scale_ptr[2]), acc2);
    acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF)) * scalar_to_float(scale_ptr[3]), acc3);
    acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF)) * scalar_to_float(scale_ptr[4]), acc4);
    acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF)) * scalar_to_float(scale_ptr[5]), acc5);
    acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF)) * scalar_to_float(scale_ptr[6]), acc6);
    acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF)) * scalar_to_float(scale_ptr[7]), acc7);
  }

  PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

  if (tid == 0) {
    scalar_t* out_ptr = &out[row * out_features + out_col];
    out_ptr[0] = float_to_scalar<scalar_t>(r0);
    out_ptr[1] = float_to_scalar<scalar_t>(r1);
    out_ptr[2] = float_to_scalar<scalar_t>(r2);
    out_ptr[3] = float_to_scalar<scalar_t>(r3);
    out_ptr[4] = float_to_scalar<scalar_t>(r4);
    out_ptr[5] = float_to_scalar<scalar_t>(r5);
    out_ptr[6] = float_to_scalar<scalar_t>(r6);
    out_ptr[7] = float_to_scalar<scalar_t>(r7);
  }
}

template <typename scalar_t, bool qweight_transposed>
__launch_bounds__(128, 4)
__global__ void gemv_awq_selected_pack8_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const scalar_t* __restrict__ scales,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features,
    int64_t out_packed,
    int64_t num_experts,
    int group_size) {
  const int64_t out_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t out_col = out_pack << 3;
  const int64_t groups = in_features / group_size;
  const int64_t x_base = row * in_features;
  const int64_t expert_qweight_base = expert_id * in_features * out_packed;
  const int64_t expert_qweight_t_base = expert_id * out_packed * in_features;
  const int64_t qw_base = qweight_transposed ? (expert_qweight_t_base + out_pack * in_features) : (expert_qweight_base + out_pack);
  const int32_t* __restrict__ qzeros_e = qzeros + expert_id * groups * out_packed;
  const scalar_t* __restrict__ scales_e = scales + expert_id * groups * out_features;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  float acc4 = 0.0f;
  float acc5 = 0.0f;
  float acc6 = 0.0f;
  float acc7 = 0.0f;

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros_e[group * out_packed + out_pack]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales_e[group * out_features + out_col];
    const float s0 = scalar_to_float(scale_ptr[0]);
    const float s1 = scalar_to_float(scale_ptr[1]);
    const float s2 = scalar_to_float(scale_ptr[2]);
    const float s3 = scalar_to_float(scale_ptr[3]);
    const float s4 = scalar_to_float(scale_ptr[4]);
    const float s5 = scalar_to_float(scale_ptr[5]);
    const float s6 = scalar_to_float(scale_ptr[6]);
    const float s7 = scalar_to_float(scale_ptr[7]);

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xv = scalar_to_float(x[x_base + k_base + j]);
      const int64_t qw_idx = qweight_transposed
          ? (qw_base + k_base + j)
          : (expert_qweight_base + (k_base + j) * out_packed + out_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
      acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - z0) * s0, acc0);
      acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - z1) * s1, acc1);
      acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - z2) * s2, acc2);
      acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - z3) * s3, acc3);
      acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - z4) * s4, acc4);
      acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - z5) * s5, acc5);
      acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - z6) * s6, acc6);
      acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - z7) * s7, acc7);
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros_e[group * out_packed + out_pack]);
    const int64_t qw_idx = qweight_transposed ? (qw_base + in_col) : (expert_qweight_base + in_col * out_packed + out_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales_e[group * out_features + out_col];
    const float xv = scalar_to_float(x[x_base + in_col]);
    acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF)) * scalar_to_float(scale_ptr[0]), acc0);
    acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF)) * scalar_to_float(scale_ptr[1]), acc1);
    acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF)) * scalar_to_float(scale_ptr[2]), acc2);
    acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF)) * scalar_to_float(scale_ptr[3]), acc3);
    acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF)) * scalar_to_float(scale_ptr[4]), acc4);
    acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF)) * scalar_to_float(scale_ptr[5]), acc5);
    acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF)) * scalar_to_float(scale_ptr[6]), acc6);
    acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF)) * scalar_to_float(scale_ptr[7]), acc7);
  }

  PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

  if (tid == 0) {
    scalar_t* out_ptr = &out[row * out_features + out_col];
    out_ptr[0] = float_to_scalar<scalar_t>(r0);
    out_ptr[1] = float_to_scalar<scalar_t>(r1);
    out_ptr[2] = float_to_scalar<scalar_t>(r2);
    out_ptr[3] = float_to_scalar<scalar_t>(r3);
    out_ptr[4] = float_to_scalar<scalar_t>(r4);
    out_ptr[5] = float_to_scalar<scalar_t>(r5);
    out_ptr[6] = float_to_scalar<scalar_t>(r6);
    out_ptr[7] = float_to_scalar<scalar_t>(r7);
  }
}

template <typename scalar_t, bool qweight_transposed>
__launch_bounds__(128, 4)
__global__ void gemv_awq_selected_pack8_silu_rotate_staged_kernel(
    const scalar_t* __restrict__ gate_up,
    scalar_t* __restrict__ down_input,
    const int64_t* __restrict__ selected,
    const int16_t* __restrict__ pairs,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ channel_scales,
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const scalar_t* __restrict__ scales,
    scalar_t* __restrict__ out,
    int32_t* __restrict__ barrier,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int group_size,
    int krot,
    int barrier_count_target,
    int barrier_ready_value) {
  const int64_t row = blockIdx.x;
  const int64_t pack_or_group = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t groups = in_features / group_size;
  const int half_group = group_size / 2;

  if (row >= rows) return;

  if (pack_or_group < groups) {
    const int group = static_cast<int>(pack_or_group);
    const int base = group * group_size;
    const int col0 = base + tid;
    const int col1 = col0 + half_group;
    const int64_t row_base = row * in_features;
    const int64_t gate_up_base = row_base * 2;
    extern __shared__ float rotate_buf[];

    if (tid < half_group) {
      const float gate0 = scalar_to_float(gate_up[gate_up_base + col0]);
      const float gate1 = scalar_to_float(gate_up[gate_up_base + col1]);
      const float up0 = scalar_to_float(gate_up[gate_up_base + in_features + col0]);
      const float up1 = scalar_to_float(gate_up[gate_up_base + in_features + col1]);
      const scalar_t act0 = float_to_scalar<scalar_t>(gate0 * (1.0f / (1.0f + expf(-gate0))) * up0);
      const scalar_t act1 = float_to_scalar<scalar_t>(gate1 * (1.0f / (1.0f + expf(-gate1))) * up1);
      rotate_buf[tid] = scalar_to_float(act0) * scalar_to_float(channel_scales[col0]);
      rotate_buf[tid + half_group] = scalar_to_float(act1) * scalar_to_float(channel_scales[col1]);
    }
    __syncthreads();

    for (int r = 0; r < krot; ++r) {
      if (tid < half_group) {
        const int pair_base = r * static_cast<int>(in_features) + base + 2 * tid;
        const int i = static_cast<int>(pairs[pair_base + 0]);
        const int j = static_cast<int>(pairs[pair_base + 1]);
        const float angle = scalar_to_float(theta[r * (in_features / 2) + group * half_group + tid]);
        float s, c;
        sincosf(angle, &s, &c);
        const float xi = rotate_buf[i];
        const float xj = rotate_buf[j];
        rotate_buf[i] = fmaf(xj, s, xi * c);
        rotate_buf[j] = fmaf(xi, -s, xj * c);
      }
      __syncthreads();
    }

    if (tid < half_group) {
      down_input[row_base + col0] = float_to_scalar<scalar_t>(rotate_buf[tid]);
      down_input[row_base + col1] = float_to_scalar<scalar_t>(rotate_buf[tid + half_group]);
    }
    __threadfence();
    if (tid == 0) {
      const int prior = atomicAdd(&barrier[0], 1);
      if (prior + 1 == barrier_count_target) {
        __threadfence();
        atomicExch(&barrier[1], barrier_ready_value);
      }
    }
  }

  while (atomicAdd(&barrier[1], 0) < barrier_ready_value) {
  }
  __threadfence();

  if (pack_or_group >= out_packed) return;
  const int64_t out_pack = pack_or_group;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const int64_t out_col = out_pack << 3;
  const int64_t out_features = out_packed << 3;
  const int64_t x_base = row * in_features;
  const int64_t expert_qweight_base = expert_id * in_features * out_packed;
  const int64_t expert_qweight_t_base = expert_id * out_packed * in_features;
  const int64_t qw_base = qweight_transposed ? (expert_qweight_t_base + out_pack * in_features) : (expert_qweight_base + out_pack);
  const int32_t* __restrict__ qzeros_e = qzeros + expert_id * groups * out_packed;
  const scalar_t* __restrict__ scales_e = scales + expert_id * groups * out_features;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  float acc4 = 0.0f;
  float acc5 = 0.0f;
  float acc6 = 0.0f;
  float acc7 = 0.0f;

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros_e[group * out_packed + out_pack]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales_e[group * out_features + out_col];
    const float s0 = scalar_to_float(scale_ptr[0]);
    const float s1 = scalar_to_float(scale_ptr[1]);
    const float s2 = scalar_to_float(scale_ptr[2]);
    const float s3 = scalar_to_float(scale_ptr[3]);
    const float s4 = scalar_to_float(scale_ptr[4]);
    const float s5 = scalar_to_float(scale_ptr[5]);
    const float s6 = scalar_to_float(scale_ptr[6]);
    const float s7 = scalar_to_float(scale_ptr[7]);

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xv = scalar_to_float(down_input[x_base + k_base + j]);
      const int64_t qw_idx = qweight_transposed
          ? (qw_base + k_base + j)
          : (expert_qweight_base + (k_base + j) * out_packed + out_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
      acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - z0) * s0, acc0);
      acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - z1) * s1, acc1);
      acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - z2) * s2, acc2);
      acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - z3) * s3, acc3);
      acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - z4) * s4, acc4);
      acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - z5) * s5, acc5);
      acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - z6) * s6, acc6);
      acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - z7) * s7, acc7);
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros_e[group * out_packed + out_pack]);
    const int64_t qw_idx = qweight_transposed ? (qw_base + in_col) : (expert_qweight_base + in_col * out_packed + out_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales_e[group * out_features + out_col];
    const float xv = scalar_to_float(down_input[x_base + in_col]);
    acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF)) * scalar_to_float(scale_ptr[0]), acc0);
    acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF)) * scalar_to_float(scale_ptr[1]), acc1);
    acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF)) * scalar_to_float(scale_ptr[2]), acc2);
    acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF)) * scalar_to_float(scale_ptr[3]), acc3);
    acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF)) * scalar_to_float(scale_ptr[4]), acc4);
    acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF)) * scalar_to_float(scale_ptr[5]), acc5);
    acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF)) * scalar_to_float(scale_ptr[6]), acc6);
    acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF)) * scalar_to_float(scale_ptr[7]), acc7);
  }

  PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

  if (tid == 0) {
    scalar_t* out_ptr = &out[row * out_features + out_col];
    out_ptr[0] = float_to_scalar<scalar_t>(r0);
    out_ptr[1] = float_to_scalar<scalar_t>(r1);
    out_ptr[2] = float_to_scalar<scalar_t>(r2);
    out_ptr[3] = float_to_scalar<scalar_t>(r3);
    out_ptr[4] = float_to_scalar<scalar_t>(r4);
    out_ptr[5] = float_to_scalar<scalar_t>(r5);
    out_ptr[6] = float_to_scalar<scalar_t>(r6);
    out_ptr[7] = float_to_scalar<scalar_t>(r7);
  }
}


// Fused rotation + selected dual pack8 GEMV kernel
// Applies PARO pairwise rotation to input, then runs selected-expert dual W4 GEMV
// on the rotated result, all in one launch with no intermediate tensor.
template <typename scalar_t, bool qweight_transposed>
__launch_bounds__(128, 4)
__global__ void gemv_awq_selected_dual_pack8_strided_rotate_out_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ selected,
    const int16_t* __restrict__ pairs,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ channel_scales,
    const int32_t* __restrict__ qweight_a,
    const int32_t* __restrict__ qzeros_a,
    const scalar_t* __restrict__ scales_a,
    const int32_t* __restrict__ qweight_b,
    const int32_t* __restrict__ qzeros_b,
    const scalar_t* __restrict__ scales_b,
    scalar_t* __restrict__ out,
    int64_t x_rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int group_size,
    int krot) {
  const int64_t global_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t expert_id = selected[row];
  if (expert_id < 0 || expert_id >= num_experts) return;

  const bool use_b = global_pack >= out_packed_a;
  const int64_t local_pack = use_b ? (global_pack - out_packed_a) : global_pack;
  const int64_t local_out_packed = use_b ? out_packed_b : out_packed_a;
  const int64_t local_out_features = local_out_packed << 3;
  const int64_t out_features = (out_packed_a + out_packed_b) << 3;
  const int64_t out_col = global_pack << 3;
  const int64_t local_out_col = local_pack << 3;
  const int64_t groups = in_features / group_size;
  const int64_t lanes_per_x_row = static_cast<int64_t>(gridDim.y) / x_rows;
  const int64_t x_row = (x_rows == 1) ? 0 : (row / lanes_per_x_row);
  const int64_t x_base = x_row * in_features;
  const int32_t* __restrict__ qweight_base = use_b ? qweight_b : qweight_a;
  const int32_t* __restrict__ qzeros_base = use_b ? qzeros_b : qzeros_a;
  const scalar_t* __restrict__ scales_base = use_b ? scales_b : scales_a;
  const int64_t expert_qweight_stride = qweight_transposed
      ? (local_out_packed * in_features)
      : (in_features * local_out_packed);
  const int32_t* __restrict__ qweight = qweight_base + expert_id * expert_qweight_stride
      + (qweight_transposed ? (local_pack * in_features) : 0);
  const int32_t* __restrict__ qzeros = qzeros_base + expert_id * groups * local_out_packed;
  const scalar_t* __restrict__ scales = scales_base + expert_id * groups * local_out_features;

  // ---- Fused rotation: apply channel scales + pairwise rotation in shared memory ----
  // Stage the full rotated input row in dynamic shared memory.
  // For in_features=2048, this is 8 KiB; for in_features=512, 2 KiB.
  extern __shared__ float rotated_buf[];

  for (int64_t idx = tid; idx < in_features; idx += blockDim.x) {
    rotated_buf[idx] = scalar_to_float(x[x_base + idx]) * scalar_to_float(channel_scales[idx]);
  }
  __syncthreads();

  // Apply krot pairwise rotations
  const int half_group = group_size / 2;
  for (int r = 0; r < krot; ++r) {
    for (int g = 0; g < groups; ++g) {
      const int base = g * group_size;
      for (int lane = tid; lane < half_group; lane += blockDim.x) {
        const int pair_base = r * static_cast<int>(in_features) + base + 2 * lane;
        const int i = static_cast<int>(pairs[pair_base + 0]);
        const int j = static_cast<int>(pairs[pair_base + 1]);
        const float angle = scalar_to_float(theta[r * (in_features / 2) + g * half_group + lane]);
        float s, c;
        sincosf(angle, &s, &c);
        const float xi = rotated_buf[base + i];
        const float xj = rotated_buf[base + j];
        rotated_buf[base + i] = fmaf(xj, s, xi * c);
        rotated_buf[base + j] = fmaf(xi, -s, xj * c);
      }
      __syncthreads();
    }
  }

  // M13.B.1 (Option C): force the LDS-resident rotated buffer through a
  // scalar_t round-trip so the GEMV consumes the same fp16/bf16-quantized
  // values that the standalone paro_rotate1 kernel would have written to
  // HBM.  Cost: in_features / blockDim.x conversions per thread + 1 sync.
  // Benefit: bit-exact equivalence with the unfused (paro_rotate1 + GEMV)
  // chain, so the existing exact-AR gate holds without re-baselining.
  // Without this pass the fused kernel keeps higher-precision fp32 values
  // in LDS and produces numerically different (slightly more precise)
  // outputs than the unfused chain it replaces.
  for (int64_t idx = tid; idx < in_features; idx += blockDim.x) {
    rotated_buf[idx] = scalar_to_float(float_to_scalar<scalar_t>(rotated_buf[idx]));
  }
  __syncthreads();

  // ---- Selected dual pack8 GEMV on rotated result ----
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  float acc4 = 0.0f;
  float acc5 = 0.0f;
  float acc6 = 0.0f;
  float acc7 = 0.0f;

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int z0 = static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF);
    const int z1 = static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF);
    const int z2 = static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF);
    const int z3 = static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF);
    const int z4 = static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF);
    const int z5 = static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF);
    const int z6 = static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF);
    const int z7 = static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF);

    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
    const float s0 = scalar_to_float(scale_ptr[0]);
    const float s1 = scalar_to_float(scale_ptr[1]);
    const float s2 = scalar_to_float(scale_ptr[2]);
    const float s3 = scalar_to_float(scale_ptr[3]);
    const float s4 = scalar_to_float(scale_ptr[4]);
    const float s5 = scalar_to_float(scale_ptr[5]);
    const float s6 = scalar_to_float(scale_ptr[6]);
    const float s7 = scalar_to_float(scale_ptr[7]);

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xv = rotated_buf[k_base + j];
      const int64_t qw_idx = qweight_transposed ? (k_base + j) : ((k_base + j) * local_out_packed + local_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
      acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - z0) * s0, acc0);
      acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - z1) * s1, acc1);
      acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - z2) * s2, acc2);
      acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - z3) * s3, acc3);
      acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - z4) * s4, acc4);
      acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - z5) * s5, acc5);
      acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - z6) * s6, acc6);
      acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - z7) * s7, acc7);
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int64_t qw_idx = qweight_transposed ? in_col : (in_col * local_out_packed + local_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
    const float xv = rotated_buf[in_col];
    acc0 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(0)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(0)) & 0xF)) * scalar_to_float(scale_ptr[0]), acc0);
    acc1 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(1)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(1)) & 0xF)) * scalar_to_float(scale_ptr[1]), acc1);
    acc2 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(2)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(2)) & 0xF)) * scalar_to_float(scale_ptr[2]), acc2);
    acc3 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(3)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(3)) & 0xF)) * scalar_to_float(scale_ptr[3]), acc3);
    acc4 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(4)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(4)) & 0xF)) * scalar_to_float(scale_ptr[4]), acc4);
    acc5 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(5)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(5)) & 0xF)) * scalar_to_float(scale_ptr[5]), acc5);
    acc6 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(6)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(6)) & 0xF)) * scalar_to_float(scale_ptr[6]), acc6);
    acc7 = fmaf(xv, static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(7)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(7)) & 0xF)) * scalar_to_float(scale_ptr[7]), acc7);
  }

  PARO_PACK8_SHFL_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7);

  if (tid == 0) {
    scalar_t* out_ptr = &out[row * out_features + out_col];
    out_ptr[0] = float_to_scalar<scalar_t>(r0);
    out_ptr[1] = float_to_scalar<scalar_t>(r1);
    out_ptr[2] = float_to_scalar<scalar_t>(r2);
    out_ptr[3] = float_to_scalar<scalar_t>(r3);
    out_ptr[4] = float_to_scalar<scalar_t>(r4);
    out_ptr[5] = float_to_scalar<scalar_t>(r5);
    out_ptr[6] = float_to_scalar<scalar_t>(r6);
    out_ptr[7] = float_to_scalar<scalar_t>(r7);
  }
}


bool valid_threads(int64_t threads) {
  return threads == 64 || threads == 128;
}

template <typename scalar_t, bool qweight_transposed>
int launch_pack8(
    const scalar_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const scalar_t* scales,
    scalar_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_pack8_kernel<scalar_t, qweight_transposed>),
      dim3(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      qweight,
      qzeros,
      scales,
      nullptr,
      out,
      in_features,
      out_packed << 3,
      out_packed,
      static_cast<int>(group_size),
      false);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, bool qweight_transposed, bool separate_inputs>
int launch_dual_pack8(
    const scalar_t* x_a,
    const scalar_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const scalar_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const scalar_t* scales_b,
    scalar_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_dual_pack8_kernel<scalar_t, qweight_transposed, separate_inputs>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x_a,
      x_b,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      in_features,
      out_packed_a,
      out_packed_b,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, bool qweight_transposed>
int launch_dual_pack8_rotate_staged(
    const scalar_t* x,
    scalar_t* rotated_a,
    scalar_t* rotated_b,
    const int16_t* pairs_a,
    const int16_t* pairs_b,
    const scalar_t* theta_a,
    const scalar_t* theta_b,
    const scalar_t* channel_scales_a,
    const scalar_t* channel_scales_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const scalar_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const scalar_t* scales_b,
    scalar_t* out,
    int32_t* barrier,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    hipStream_t stream) {
  const int64_t groups = (group_size > 0) ? (in_features / group_size) : 0;
  const int64_t total_rotate = groups * 2 * rows;
  if (rows <= 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 || group_size <= 0 ||
      krot < 0 || (in_features % group_size) != 0 || (group_size % 8) != 0 || !valid_threads(threads) ||
      (out_packed_a + out_packed_b) < (groups * 2) || total_rotate <= 0 || total_rotate > 2147483647LL) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipError_t memset_err = hipMemsetAsync(barrier, 0, 2 * sizeof(int32_t), stream);
  if (memset_err != hipSuccess) {
    return static_cast<int>(memset_err);
  }
  hipLaunchKernelGGL(
      (gemv_awq_dual_pack8_transposed_rotate_staged_kernel<scalar_t, qweight_transposed>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<unsigned int>(group_size * sizeof(float)),
      stream,
      x,
      rotated_a,
      rotated_b,
      pairs_a,
      pairs_b,
      theta_a,
      theta_b,
      channel_scales_a,
      channel_scales_b,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      barrier,
      in_features,
      out_packed_a,
      out_packed_b,
      static_cast<int>(group_size),
      static_cast<int>(krot),
      static_cast<int>(total_rotate),
      1);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, bool qweight_transposed>
int launch_dual_pack8_rotate_staged_keyed(
    const scalar_t* x,
    scalar_t* rotated_a,
    scalar_t* rotated_b,
    const int16_t* pairs_a,
    const int16_t* pairs_b,
    const scalar_t* theta_a,
    const scalar_t* theta_b,
    const scalar_t* channel_scales_a,
    const scalar_t* channel_scales_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const scalar_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const scalar_t* scales_b,
    scalar_t* out,
    int32_t* barrier,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    int64_t barrier_count_target,
    int64_t barrier_ready_value,
    hipStream_t stream) {
  const int64_t groups = (group_size > 0) ? (in_features / group_size) : 0;
  const int64_t total_rotate = groups * 2 * rows;
  if (rows <= 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 || group_size <= 0 ||
      krot < 0 || (in_features % group_size) != 0 || (group_size % 8) != 0 || !valid_threads(threads) ||
      (out_packed_a + out_packed_b) < (groups * 2) || total_rotate <= 0 || total_rotate > 2147483647LL ||
      barrier_count_target <= 0 || barrier_count_target > 2147483647LL ||
      barrier_ready_value <= 0 || barrier_ready_value > 2147483647LL) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_dual_pack8_transposed_rotate_staged_kernel<scalar_t, qweight_transposed>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<unsigned int>(group_size * sizeof(float)),
      stream,
      x,
      rotated_a,
      rotated_b,
      pairs_a,
      pairs_b,
      theta_a,
      theta_b,
      channel_scales_a,
      channel_scales_b,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      barrier,
      in_features,
      out_packed_a,
      out_packed_b,
      static_cast<int>(group_size),
      static_cast<int>(krot),
      static_cast<int>(barrier_count_target),
      static_cast<int>(barrier_ready_value));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, bool qweight_transposed>
int launch_selected_dual_rotate(
    const scalar_t* x,
    const int64_t* selected,
    const int16_t* pairs,
    const scalar_t* theta,
    const scalar_t* channel_scales,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const scalar_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const scalar_t* scales_b,
    scalar_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    hipStream_t stream) {
  if (x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 ||
      num_experts <= 0 || group_size <= 0 || krot < 0 || (in_features % group_size) != 0 ||
      (group_size % 8) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_dual_pack8_strided_rotate_out_kernel<scalar_t, qweight_transposed>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<unsigned int>(in_features * sizeof(float)),
      stream,
      x,
      selected,
      pairs,
      theta,
      channel_scales,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      x_rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, bool qweight_transposed>
int launch_selected_dual_rotate_staged_keyed(
    const scalar_t* x,
    scalar_t* rotated,
    const int64_t* selected,
    const int16_t* pairs,
    const scalar_t* theta,
    const scalar_t* channel_scales,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const scalar_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const scalar_t* scales_b,
    scalar_t* out,
    int32_t* barrier,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    int64_t barrier_count_target,
    int64_t barrier_ready_value,
    hipStream_t stream) {
  const int64_t groups = (group_size > 0) ? (in_features / group_size) : 0;
  const int64_t rotate_blocks = x_rows * groups;
  const int64_t gemv_blocks = rows * (out_packed_a + out_packed_b);
  if (x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 ||
      num_experts <= 0 || group_size <= 0 || krot < 0 || (in_features % group_size) != 0 ||
      (group_size % 8) != 0 || !valid_threads(threads) || rotate_blocks <= 0 || gemv_blocks <= 0 ||
      rotate_blocks > 2147483647LL || gemv_blocks > 2147483647LL ||
      barrier_count_target <= 0 || barrier_count_target > 2147483647LL ||
      barrier_ready_value <= 0 || barrier_ready_value > 2147483647LL) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_dual_pack8_strided_rotate_staged_kernel<scalar_t, qweight_transposed>),
      dim3(static_cast<unsigned int>(rotate_blocks + gemv_blocks)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<unsigned int>(group_size * sizeof(float)),
      stream,
      x,
      rotated,
      selected,
      pairs,
      theta,
      channel_scales,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      barrier,
      x_rows,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      static_cast<int>(group_size),
      static_cast<int>(krot),
      static_cast<int>(barrier_count_target),
      static_cast<int>(barrier_ready_value));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, bool qweight_transposed>
int launch_selected_dual_rotate_staged(
    const scalar_t* x,
    scalar_t* rotated,
    const int64_t* selected,
    const int16_t* pairs,
    const scalar_t* theta,
    const scalar_t* channel_scales,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const scalar_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const scalar_t* scales_b,
    scalar_t* out,
    int32_t* barrier,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    hipStream_t stream) {
  if (barrier == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t groups = (group_size > 0) ? (in_features / group_size) : 0;
  const int64_t rotate_blocks = x_rows * groups;
  hipError_t memset_err = hipMemsetAsync(barrier, 0, 2 * sizeof(int32_t), stream);
  if (memset_err != hipSuccess) {
    return static_cast<int>(memset_err);
  }
  return launch_selected_dual_rotate_staged_keyed<scalar_t, qweight_transposed>(
      x,
      rotated,
      selected,
      pairs,
      theta,
      channel_scales,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      barrier,
      x_rows,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      group_size,
      krot,
      threads,
      rotate_blocks,
      1,
      stream);
}

template <typename scalar_t, bool qweight_transposed>
int launch_selected_dual_pack8(
    const scalar_t* x,
    const int64_t* selected,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const scalar_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const scalar_t* scales_b,
    scalar_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 ||
      num_experts <= 0 || group_size <= 0 || (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_dual_pack8_strided_kernel<scalar_t, qweight_transposed>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      selected,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      x_rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, bool qweight_transposed>
int launch_selected_pack8(
    const scalar_t* x,
    const int64_t* selected,
    const int32_t* qweight,
    const int32_t* qzeros,
    const scalar_t* scales,
    scalar_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed <= 0 || num_experts <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_pack8_kernel<scalar_t, qweight_transposed>),
      dim3(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      selected,
      qweight,
      qzeros,
      scales,
      out,
      in_features,
      out_packed << 3,
      out_packed,
      num_experts,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, bool qweight_transposed>
int launch_selected_pack8_silu_rotate_staged_keyed(
    const scalar_t* gate_up,
    scalar_t* down_input,
    const int64_t* selected,
    const int16_t* pairs,
    const scalar_t* theta,
    const scalar_t* channel_scales,
    const int32_t* qweight,
    const int32_t* qzeros,
    const scalar_t* scales,
    scalar_t* out,
    int32_t* barrier,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    int64_t barrier_count_target,
    int64_t barrier_ready_value,
    hipStream_t stream) {
  const int64_t groups = (group_size > 0) ? (in_features / group_size) : 0;
  const int64_t rotate_blocks = rows * groups;
  const int64_t grid_y = (out_packed > groups) ? out_packed : groups;
  if (rows <= 0 || in_features <= 0 || out_packed <= 0 || num_experts <= 0 || group_size <= 0 || krot <= 0 ||
      (in_features % group_size) != 0 || (group_size % 2) != 0 || !valid_threads(threads) ||
      rotate_blocks <= 0 || rows > 2147483647LL || grid_y <= 0 || grid_y > 2147483647LL ||
      barrier_count_target <= 0 || barrier_count_target > 2147483647LL ||
      barrier_ready_value <= 0 || barrier_ready_value > 2147483647LL) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_pack8_silu_rotate_staged_kernel<scalar_t, qweight_transposed>),
      dim3(static_cast<unsigned int>(rows), static_cast<unsigned int>(grid_y)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<unsigned int>(group_size * sizeof(float)),
      stream,
      gate_up,
      down_input,
      selected,
      pairs,
      theta,
      channel_scales,
      qweight,
      qzeros,
      scales,
      out,
      barrier,
      rows,
      in_features,
      out_packed,
      num_experts,
      static_cast<int>(group_size),
      static_cast<int>(krot),
      static_cast<int>(barrier_count_target),
      static_cast<int>(barrier_ready_value));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, bool qweight_transposed>
int launch_selected_pack8_silu_rotate_staged(
    const scalar_t* gate_up,
    scalar_t* down_input,
    const int64_t* selected,
    const int16_t* pairs,
    const scalar_t* theta,
    const scalar_t* channel_scales,
    const int32_t* qweight,
    const int32_t* qzeros,
    const scalar_t* scales,
    scalar_t* out,
    int32_t* barrier,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    hipStream_t stream) {
  if (barrier == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t groups = (group_size > 0) ? (in_features / group_size) : 0;
  const int64_t rotate_blocks = rows * groups;
  hipError_t memset_err = hipMemsetAsync(barrier, 0, 2 * sizeof(int32_t), stream);
  if (memset_err != hipSuccess) {
    return static_cast<int>(memset_err);
  }
  return launch_selected_pack8_silu_rotate_staged_keyed<scalar_t, qweight_transposed>(
      gate_up,
      down_input,
      selected,
      pairs,
      theta,
      channel_scales,
      qweight,
      qzeros,
      scales,
      out,
      barrier,
      rows,
      in_features,
      out_packed,
      num_experts,
      group_size,
      krot,
      threads,
      rotate_blocks,
      1,
      stream);
}


// Output-column-tiled c>1 decode GEMV (C3.0c).  One block per output pack loads
// each weight pack ONCE and accumulates all C active columns (rows of x), so
// weight bytes/token stop scaling with C.  The per-output accumulation order
// over k and across warps is identical to gemv_awq_pack8_kernel, so results are
// bit-exact vs the per-row strided kernel.
template <typename scalar_t, bool qweight_transposed, int C>
__launch_bounds__(128, 2) __global__ void gemv_awq_pack8_output_tiled_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const scalar_t* __restrict__ scales,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features,
    int64_t out_packed,
    int group_size) {
  const int64_t out_pack = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t out_col = out_pack << 3;
  const int64_t qw_base = qweight_transposed ? out_pack * in_features : out_pack;

  float acc[C][8];
#pragma unroll
  for (int r = 0; r < C; ++r) {
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      acc[r][o] = 0.0f;
    }
  }

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * out_packed + out_pack]);
    int z[8];
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      z[o] = static_cast<int>((pz >> awq_shift_for_pack_lane(o)) & 0xF);
    }
    const scalar_t* scale_ptr = &scales[group * out_features + out_col];
    float s[8];
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      s[o] = scalar_to_float(scale_ptr[o]);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t qw_idx = qweight_transposed ? (qw_base + k_base + j) : ((k_base + j) * out_packed + out_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
      float w[8];
#pragma unroll
      for (int o = 0; o < 8; ++o) {
        w[o] = static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(o)) & 0xF) - z[o]) * s[o];
      }
#pragma unroll
      for (int r = 0; r < C; ++r) {
        const float xv = scalar_to_float(x[static_cast<int64_t>(r) * in_features + k_base + j]);
#pragma unroll
        for (int o = 0; o < 8; ++o) {
          acc[r][o] = fmaf(xv, w[o], acc[r][o]);
        }
      }
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * out_packed + out_pack]);
    const int64_t qw_idx = qweight_transposed ? (qw_base + in_col) : (in_col * out_packed + out_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales[group * out_features + out_col];
    float w[8];
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      w[o] = static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(o)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(o)) & 0xF)) * scalar_to_float(scale_ptr[o]);
    }
#pragma unroll
    for (int r = 0; r < C; ++r) {
      const float xv = scalar_to_float(x[static_cast<int64_t>(r) * in_features + in_col]);
#pragma unroll
      for (int o = 0; o < 8; ++o) {
        acc[r][o] = fmaf(xv, w[o], acc[r][o]);
      }
    }
  }

  // Within-warp shuffle reduce (offsets 16..1, matches gemv_awq_pack8_kernel),
  // then cross-warp sum in ascending warp order through shared memory.
#pragma unroll
  for (int r = 0; r < C; ++r) {
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      float a = acc[r][o];
      for (int offset = 16; offset > 0; offset >>= 1) {
        a += __shfl_down(a, offset);
      }
      acc[r][o] = a;
    }
  }
  __shared__ float xchg[4 * C * 8];
  const int warp_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int r = 0; r < C; ++r) {
#pragma unroll
      for (int o = 0; o < 8; ++o) {
        xchg[(warp_id * C + r) * 8 + o] = acc[r][o];
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int r = 0; r < C; ++r) {
      scalar_t* out_ptr = &out[static_cast<int64_t>(r) * out_features + out_col];
#pragma unroll
      for (int o = 0; o < 8; ++o) {
        float sum = 0.0f;
        for (int g = 0; g < sub_groups; ++g) {
          sum += xchg[(g * C + r) * 8 + o];
        }
        out_ptr[o] = float_to_scalar<scalar_t>(sum);
      }
    }
  }
}

template <typename scalar_t, bool qweight_transposed, int C>
__launch_bounds__(128, 2) __global__ void gemv_awq_pack8_output_tiled_combine_residual_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ qweight,
    const int32_t* __restrict__ qzeros,
    const scalar_t* __restrict__ scales,
    const scalar_t* __restrict__ selected,
    const float* __restrict__ routing_weights,
    const float* __restrict__ shared_gate_logits,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ out,
    int64_t in_features,
    int64_t out_features,
    int64_t out_packed,
    int group_size,
    int64_t rows_per_token,
    int64_t gate_stride) {
  const int64_t out_pack = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t out_col = out_pack << 3;
  const int64_t qw_base = qweight_transposed ? out_pack * in_features : out_pack;

  float acc[C][8];
#pragma unroll
  for (int r = 0; r < C; ++r) {
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      acc[r][o] = 0.0f;
    }
  }

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * out_packed + out_pack]);
    int z[8];
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      z[o] = static_cast<int>((pz >> awq_shift_for_pack_lane(o)) & 0xF);
    }
    const scalar_t* scale_ptr = &scales[group * out_features + out_col];
    float s[8];
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      s[o] = scalar_to_float(scale_ptr[o]);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t qw_idx = qweight_transposed ? (qw_base + k_base + j) : ((k_base + j) * out_packed + out_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
      float w[8];
#pragma unroll
      for (int o = 0; o < 8; ++o) {
        w[o] = static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(o)) & 0xF) - z[o]) * s[o];
      }
#pragma unroll
      for (int r = 0; r < C; ++r) {
        const float xv = scalar_to_float(x[static_cast<int64_t>(r) * in_features + k_base + j]);
#pragma unroll
        for (int o = 0; o < 8; ++o) {
          acc[r][o] = fmaf(xv, w[o], acc[r][o]);
        }
      }
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * out_packed + out_pack]);
    const int64_t qw_idx = qweight_transposed ? (qw_base + in_col) : (in_col * out_packed + out_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales[group * out_features + out_col];
    float w[8];
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      w[o] = static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(o)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(o)) & 0xF)) * scalar_to_float(scale_ptr[o]);
    }
#pragma unroll
    for (int r = 0; r < C; ++r) {
      const float xv = scalar_to_float(x[static_cast<int64_t>(r) * in_features + in_col]);
#pragma unroll
      for (int o = 0; o < 8; ++o) {
        acc[r][o] = fmaf(xv, w[o], acc[r][o]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < C; ++r) {
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      float a = acc[r][o];
      for (int offset = 16; offset > 0; offset >>= 1) {
        a += __shfl_down(a, offset);
      }
      acc[r][o] = a;
    }
  }
  __shared__ float xchg[4 * C * 8];
  const int warp_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int r = 0; r < C; ++r) {
#pragma unroll
      for (int o = 0; o < 8; ++o) {
        xchg[(warp_id * C + r) * 8 + o] = acc[r][o];
      }
    }
  }
  __syncthreads();

  const int sub_groups = blockDim.x >> 5;
  for (int elem = tid; elem < C * 8; elem += blockDim.x) {
    const int r = elem >> 3;
    const int o = elem & 7;
    const int64_t col = out_col + o;
    if (col >= out_features) {
      continue;
    }
    float sum = 0.0f;
    for (int g = 0; g < sub_groups; ++g) {
      sum += xchg[(g * C + r) * 8 + o];
    }
    const scalar_t shared_lowp = float_to_scalar<scalar_t>(sum);
    const int64_t token_offset = static_cast<int64_t>(r) * out_features;
    float selected_acc = 0.0f;
    for (int64_t row = 0; row < rows_per_token; ++row) {
      const int64_t lane = static_cast<int64_t>(r) * rows_per_token + row;
      selected_acc = fmaf(
          scalar_to_float(selected[lane * out_features + col]),
          routing_weights[lane],
          selected_acc);
    }
    const scalar_t selected_lowp = float_to_scalar<scalar_t>(selected_acc);
    const float gate = 1.0f / (1.0f + expf(-shared_gate_logits[static_cast<int64_t>(r) * gate_stride]));
    const int64_t idx = token_offset + col;
    const float value =
        scalar_to_float(residual[idx]) +
        scalar_to_float(selected_lowp) +
        gate * scalar_to_float(shared_lowp);
    out[idx] = float_to_scalar<scalar_t>(value);
  }
}

template <typename scalar_t, bool qweight_transposed>
int launch_pack8_output_tiled(
    const scalar_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const scalar_t* scales,
    scalar_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (in_features <= 0 || out_packed <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(static_cast<unsigned int>(out_packed));
  const dim3 block(static_cast<unsigned int>(threads));
#define PARO_LAUNCH_PACK8_OT(CVAL) \
  hipLaunchKernelGGL((gemv_awq_pack8_output_tiled_kernel<scalar_t, qweight_transposed, CVAL>), \
      grid, block, 0, stream, x, qweight, qzeros, scales, out, in_features, \
      out_packed << 3, out_packed, static_cast<int>(group_size))
  switch (rows) {
    case 2: PARO_LAUNCH_PACK8_OT(2); break;
    case 4: PARO_LAUNCH_PACK8_OT(4); break;
    case 8: PARO_LAUNCH_PACK8_OT(8); break;
    default: return static_cast<int>(hipErrorInvalidValue);
  }
#undef PARO_LAUNCH_PACK8_OT
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, bool qweight_transposed>
int launch_pack8_output_tiled_combine_residual(
    const scalar_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const scalar_t* scales,
    const scalar_t* selected,
    const float* routing_weights,
    const float* shared_gate_logits,
    const scalar_t* residual,
    scalar_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t rows_per_token,
    int64_t gate_stride,
    int64_t threads,
    hipStream_t stream) {
  if (in_features <= 0 || out_packed <= 0 || group_size <= 0 ||
      rows_per_token <= 0 || gate_stride <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(static_cast<unsigned int>(out_packed));
  const dim3 block(static_cast<unsigned int>(threads));
#define PARO_LAUNCH_PACK8_OT_COMBINE(CVAL) \
  hipLaunchKernelGGL((gemv_awq_pack8_output_tiled_combine_residual_kernel<scalar_t, qweight_transposed, CVAL>), \
      grid, block, 0, stream, x, qweight, qzeros, scales, selected, routing_weights, \
      shared_gate_logits, residual, out, in_features, out_packed << 3, out_packed, \
      static_cast<int>(group_size), rows_per_token, gate_stride)
  switch (rows) {
    case 2: PARO_LAUNCH_PACK8_OT_COMBINE(2); break;
    case 4: PARO_LAUNCH_PACK8_OT_COMBINE(4); break;
    case 8: PARO_LAUNCH_PACK8_OT_COMBINE(8); break;
    default: return static_cast<int>(hipErrorInvalidValue);
  }
#undef PARO_LAUNCH_PACK8_OT_COMBINE
  return static_cast<int>(hipGetLastError());
}


// Output-column-tiled c>1 decode dual-projection GEMV (C3.0c).  Same structure
// as gemv_awq_pack8_output_tiled_kernel plus the a/b output-group prologue from
// gemv_awq_dual_pack8_kernel: one block per global output pack loads its weight
// pack once and accumulates all C columns.  Bit-exact vs gemv_awq_dual_pack8_kernel.
// OUT_LAYOUT: 0 = packed gate||up output, 1 = split output buffers.
template <typename scalar_t, bool qweight_transposed, bool separate_inputs, int OUT_LAYOUT, int C>
__launch_bounds__(128, 2) __global__ void gemv_awq_dual_pack8_output_tiled_kernel(
    const scalar_t* __restrict__ x_a,
    const scalar_t* __restrict__ x_b,
    const int32_t* __restrict__ qweight_a,
    const int32_t* __restrict__ qzeros_a,
    const scalar_t* __restrict__ scales_a,
    const int32_t* __restrict__ qweight_b,
    const int32_t* __restrict__ qzeros_b,
    const scalar_t* __restrict__ scales_b,
    scalar_t* __restrict__ out_a,
    scalar_t* __restrict__ out_b,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int group_size) {
  const int64_t global_pack = blockIdx.x;
  const int tid = threadIdx.x;
  const bool use_b = global_pack >= out_packed_a;
  const int64_t local_pack = use_b ? (global_pack - out_packed_a) : global_pack;
  const int64_t local_out_packed = use_b ? out_packed_b : out_packed_a;
  const int64_t local_out_features = local_out_packed << 3;
  const int64_t local_out_col = local_pack << 3;
  const scalar_t* __restrict__ x = (separate_inputs && use_b) ? x_b : x_a;
  const int32_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;
  const int32_t* __restrict__ qzeros = use_b ? qzeros_b : qzeros_a;
  const scalar_t* __restrict__ scales = use_b ? scales_b : scales_a;
  scalar_t* __restrict__ out = (OUT_LAYOUT == 0 || !use_b) ? out_a : out_b;
  const int64_t out_features = (OUT_LAYOUT == 0) ? ((out_packed_a + out_packed_b) << 3) : local_out_features;
  const int64_t out_col = (OUT_LAYOUT == 0) ? (global_pack << 3) : local_out_col;
  const int64_t qweight_base = qweight_transposed ? (local_pack * in_features) : local_pack;

  float acc[C][8];
#pragma unroll
  for (int r = 0; r < C; ++r) {
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      acc[r][o] = 0.0f;
    }
  }

  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end; k_base += vec_stride) {
    const int64_t group = k_base / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    int z[8];
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      z[o] = static_cast<int>((pz >> awq_shift_for_pack_lane(o)) & 0xF);
    }
    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
    float s[8];
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      s[o] = scalar_to_float(scale_ptr[o]);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t qw_idx = qweight_transposed ? (qweight_base + k_base + j) : ((k_base + j) * local_out_packed + local_pack);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
      float w[8];
#pragma unroll
      for (int o = 0; o < 8; ++o) {
        w[o] = static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(o)) & 0xF) - z[o]) * s[o];
      }
#pragma unroll
      for (int r = 0; r < C; ++r) {
        const float xv = scalar_to_float(x[static_cast<int64_t>(r) * in_features + k_base + j]);
#pragma unroll
        for (int o = 0; o < 8; ++o) {
          acc[r][o] = fmaf(xv, w[o], acc[r][o]);
        }
      }
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group = in_col / group_size;
    const uint32_t pz = static_cast<uint32_t>(qzeros[group * local_out_packed + local_pack]);
    const int64_t qw_idx = qweight_transposed ? (qweight_base + in_col) : (in_col * local_out_packed + local_pack);
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_idx]);
    const scalar_t* scale_ptr = &scales[group * local_out_features + local_out_col];
    float w[8];
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      w[o] = static_cast<float>(static_cast<int>((pw >> awq_shift_for_pack_lane(o)) & 0xF) - static_cast<int>((pz >> awq_shift_for_pack_lane(o)) & 0xF)) * scalar_to_float(scale_ptr[o]);
    }
#pragma unroll
    for (int r = 0; r < C; ++r) {
      const float xv = scalar_to_float(x[static_cast<int64_t>(r) * in_features + in_col]);
#pragma unroll
      for (int o = 0; o < 8; ++o) {
        acc[r][o] = fmaf(xv, w[o], acc[r][o]);
      }
    }
  }

  // Within-warp shuffle reduce (offsets 16..1) then ascending-warp cross-warp
  // sum via shared memory: identical per-output order to gemv_awq_dual_pack8_kernel.
#pragma unroll
  for (int r = 0; r < C; ++r) {
#pragma unroll
    for (int o = 0; o < 8; ++o) {
      float a = acc[r][o];
      for (int offset = 16; offset > 0; offset >>= 1) {
        a += __shfl_down(a, offset);
      }
      acc[r][o] = a;
    }
  }
  __shared__ float xchg[4 * C * 8];
  const int warp_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int r = 0; r < C; ++r) {
#pragma unroll
      for (int o = 0; o < 8; ++o) {
        xchg[(warp_id * C + r) * 8 + o] = acc[r][o];
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    const int sub_groups = blockDim.x >> 5;
#pragma unroll
    for (int r = 0; r < C; ++r) {
      scalar_t* out_ptr = &out[static_cast<int64_t>(r) * out_features + out_col];
#pragma unroll
      for (int o = 0; o < 8; ++o) {
        float sum = 0.0f;
        for (int g = 0; g < sub_groups; ++g) {
          sum += xchg[(g * C + r) * 8 + o];
        }
        out_ptr[o] = float_to_scalar<scalar_t>(sum);
      }
    }
  }
}

template <typename scalar_t, bool qweight_transposed, bool separate_inputs, int OUT_LAYOUT>
int launch_dual_pack8_output_tiled(
    const scalar_t* x_a,
    const scalar_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const scalar_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const scalar_t* scales_b,
    scalar_t* out_a,
    scalar_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 grid(static_cast<unsigned int>(out_packed_a + out_packed_b));
  const dim3 block(static_cast<unsigned int>(threads));
#define PARO_LAUNCH_DUAL_PACK8_OT(CVAL) \
  hipLaunchKernelGGL((gemv_awq_dual_pack8_output_tiled_kernel<scalar_t, qweight_transposed, separate_inputs, OUT_LAYOUT, CVAL>), \
      grid, block, 0, stream, x_a, x_b, qweight_a, qzeros_a, scales_a, qweight_b, qzeros_b, scales_b, \
      out_a, out_b, in_features, out_packed_a, out_packed_b, static_cast<int>(group_size))
  switch (rows) {
    case 2: PARO_LAUNCH_DUAL_PACK8_OT(2); break;
    case 4: PARO_LAUNCH_DUAL_PACK8_OT(4); break;
    case 8: PARO_LAUNCH_DUAL_PACK8_OT(8); break;
    default: return static_cast<int>(hipErrorInvalidValue);
  }
#undef PARO_LAUNCH_DUAL_PACK8_OT
  return static_cast<int>(hipGetLastError());
}

}  // namespace


extern "C" int hipengine_gemv_awq_pack8_strided_bf16(
    const uint16_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_pack8_kernel<uint16_t, false>),
      dim3(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      qweight,
      qzeros,
      scales,
      nullptr,
      out,
      in_features,
      out_packed << 3,
      out_packed,
      static_cast<int>(group_size),
      false);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_pack8_transposed_bf16(
    const uint16_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_pack8_kernel<uint16_t, true>),
      dim3(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      qweight,
      qzeros,
      scales,
      nullptr,
      out,
      in_features,
      out_packed << 3,
      out_packed,
      static_cast<int>(group_size),
      false);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_dual_pack8_strided_bf16(
    const uint16_t* x,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_dual_pack8_kernel<uint16_t, false, false>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      nullptr,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      in_features,
      out_packed_a,
      out_packed_b,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_dual_pack8_transposed_bf16(
    const uint16_t* x_a,
    const uint16_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_dual_pack8_kernel<uint16_t, true, true>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x_a,
      x_b,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      in_features,
      out_packed_a,
      out_packed_b,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_dual_pack8_transposed_rotate_staged_bf16(
    const uint16_t* x,
    uint16_t* rotated_a,
    uint16_t* rotated_b,
    const int16_t* pairs_a,
    const int16_t* pairs_b,
    const uint16_t* theta_a,
    const uint16_t* theta_b,
    const uint16_t* channel_scales_a,
    const uint16_t* channel_scales_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int32_t* barrier,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    hipStream_t stream) {
  return launch_dual_pack8_rotate_staged<uint16_t, true>(
      x,
      rotated_a,
      rotated_b,
      pairs_a,
      pairs_b,
      theta_a,
      theta_b,
      channel_scales_a,
      channel_scales_b,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      barrier,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      group_size,
      krot,
      threads,
      stream);
}

extern "C" int hipengine_gemv_awq_dual_pack8_transposed_rotate_staged_keyed_bf16(
    const uint16_t* x,
    uint16_t* rotated_a,
    uint16_t* rotated_b,
    const int16_t* pairs_a,
    const int16_t* pairs_b,
    const uint16_t* theta_a,
    const uint16_t* theta_b,
    const uint16_t* channel_scales_a,
    const uint16_t* channel_scales_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int32_t* barrier,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    int64_t barrier_count_target,
    int64_t barrier_ready_value,
    hipStream_t stream) {
  return launch_dual_pack8_rotate_staged_keyed<uint16_t, true>(
      x,
      rotated_a,
      rotated_b,
      pairs_a,
      pairs_b,
      theta_a,
      theta_b,
      channel_scales_a,
      channel_scales_b,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      barrier,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      group_size,
      krot,
      threads,
      barrier_count_target,
      barrier_ready_value,
      stream);
}

extern "C" int hipengine_gemv_awq_pack8_strided_fp16(
    const half_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_pack8<half_t, false>(
      x, qweight, qzeros, scales, out, rows, in_features, out_packed, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_pack8_transposed_fp16(
    const half_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_pack8<half_t, true>(
      x, qweight, qzeros, scales, out, rows, in_features, out_packed, group_size, threads, stream);
}

// M12.6: multi-row pack8 launcher.  Picks the smallest template instantiation
// that fits ``rows`` so each block stores no more accumulator slots than
// needed.  Only rows 1..8 are supported; callers above that fall back to the
// stock prefill (WMMA) kernel.
template <typename scalar_t, bool qweight_transposed, bool decode_dequant = false>
static int launch_pack8_multi_row(
    const scalar_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const scalar_t* scales,
    scalar_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || rows > 8 || in_features <= 0 || out_packed <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_features = out_packed << 3;
#define HIPENGINE_LAUNCH_PACK8_MULTI_ROW(MAX_ROWS) \
  hipLaunchKernelGGL( \
      (gemv_awq_pack8_multi_row_kernel<scalar_t, qweight_transposed, MAX_ROWS, decode_dequant>), \
      dim3(static_cast<unsigned int>(out_packed)), \
      dim3(static_cast<unsigned int>(threads)), \
      0, \
      stream, \
      x, \
      qweight, \
      qzeros, \
      scales, \
      nullptr, \
      out, \
      rows, \
      in_features, \
      out_features, \
      out_packed, \
      static_cast<int>(group_size), \
      false); \
  return static_cast<int>(hipGetLastError())

  if (rows <= 2) { HIPENGINE_LAUNCH_PACK8_MULTI_ROW(2); }
  if (rows <= 4) { HIPENGINE_LAUNCH_PACK8_MULTI_ROW(4); }
  HIPENGINE_LAUNCH_PACK8_MULTI_ROW(8);
#undef HIPENGINE_LAUNCH_PACK8_MULTI_ROW
}

extern "C" int hipengine_gemv_awq_pack8_multi_row_strided_fp16(
    const half_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_pack8_multi_row<half_t, false>(
      x, qweight, qzeros, scales, out, rows, in_features, out_packed, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_pack8_multi_row_transposed_fp16(
    const half_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_pack8_multi_row<half_t, true>(
      x, qweight, qzeros, scales, out, rows, in_features, out_packed, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_pack8_multi_row_decode_strided_fp16(
    const half_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_pack8_multi_row<half_t, false, true>(
      x, qweight, qzeros, scales, out, rows, in_features, out_packed, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_pack8_multi_row_decode_transposed_fp16(
    const half_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_pack8_multi_row<half_t, true, true>(
      x, qweight, qzeros, scales, out, rows, in_features, out_packed, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_pack8_multi_row_strided_bf16(
    const uint16_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_pack8_multi_row<uint16_t, false>(
      x, qweight, qzeros, scales, out, rows, in_features, out_packed, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_pack8_multi_row_transposed_bf16(
    const uint16_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_pack8_multi_row<uint16_t, true>(
      x, qweight, qzeros, scales, out, rows, in_features, out_packed, group_size, threads, stream);
}


extern "C" int hipengine_gemv_awq_dual_pack8_transposed_rotate_staged_fp16(
    const half_t* x,
    half_t* rotated_a,
    half_t* rotated_b,
    const int16_t* pairs_a,
    const int16_t* pairs_b,
    const half_t* theta_a,
    const half_t* theta_b,
    const half_t* channel_scales_a,
    const half_t* channel_scales_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int32_t* barrier,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    hipStream_t stream) {
  return launch_dual_pack8_rotate_staged<half_t, true>(
      x,
      rotated_a,
      rotated_b,
      pairs_a,
      pairs_b,
      theta_a,
      theta_b,
      channel_scales_a,
      channel_scales_b,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      barrier,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      group_size,
      krot,
      threads,
      stream);
}

extern "C" int hipengine_gemv_awq_dual_pack8_transposed_rotate_staged_keyed_fp16(
    const half_t* x,
    half_t* rotated_a,
    half_t* rotated_b,
    const int16_t* pairs_a,
    const int16_t* pairs_b,
    const half_t* theta_a,
    const half_t* theta_b,
    const half_t* channel_scales_a,
    const half_t* channel_scales_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int32_t* barrier,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    int64_t barrier_count_target,
    int64_t barrier_ready_value,
    hipStream_t stream) {
  return launch_dual_pack8_rotate_staged_keyed<half_t, true>(
      x,
      rotated_a,
      rotated_b,
      pairs_a,
      pairs_b,
      theta_a,
      theta_b,
      channel_scales_a,
      channel_scales_b,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      barrier,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      group_size,
      krot,
      threads,
      barrier_count_target,
      barrier_ready_value,
      stream);
}

extern "C" int hipengine_awq_fusedw4_prefill_fp16(
    const half_t* x,
    const int32_t* qweight_t,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_awq_fusedw4_prefill_fp16<true>(
      x, qweight_t, qzeros, scales, out, rows, in_features, out_packed, group_size, tile_m, tile_n, stream);
}

extern "C" int hipengine_awq_fusedw4_prefill_dual_fp16(
    const half_t* x_a,
    const half_t* x_b,
    const int32_t* qweight_a_t,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b_t,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out_a,
    half_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_awq_fusedw4_prefill_dual_fp16(
      x_a,
      x_b,
      qweight_a_t,
      qzeros_a,
      scales_a,
      qweight_b_t,
      qzeros_b,
      scales_b,
      out_a,
      out_b,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      group_size,
      tile_m,
      tile_n,
      stream);
}

extern "C" int hipengine_awq_fusedw4_prefill_strided_fp16(
    const half_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t tile_m,
    int64_t tile_n,
    hipStream_t stream) {
  return launch_awq_fusedw4_prefill_fp16<false>(
      x, qweight, qzeros, scales, out, rows, in_features, out_packed, group_size, tile_m, tile_n, stream);
}

extern "C" int hipengine_gemv_awq_dual_pack8_strided_fp16(
    const half_t* x,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_dual_pack8<half_t, false, false>(
      x,
      nullptr,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      group_size,
      threads,
      stream);
}

extern "C" int hipengine_gemv_awq_dual_pack8_transposed_fp16(
    const half_t* x_a,
    const half_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_dual_pack8<half_t, true, true>(
      x_a,
      x_b,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      group_size,
      threads,
      stream);
}

// M12.6: multi-row dual pack8 launcher.  Same pattern as
// ``launch_pack8_multi_row`` but for two-projection (gate+up) GEMVs.
// ``OUT_LAYOUT`` picks between concatenated single-buffer output (matches
// the existing single-row ``gemv_awq_dual_pack8_*`` ABI) and split output
// buffers (matches the existing ``awq_fusedw4_prefill_dual_fp16`` ABI).
template <typename scalar_t, bool qweight_transposed, bool separate_inputs, int OUT_LAYOUT,
          bool decode_dequant = false>
static int launch_dual_pack8_multi_row(
    const scalar_t* x_a,
    const scalar_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const scalar_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const scalar_t* scales_b,
    scalar_t* out_a,
    scalar_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || rows > 8 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 ||
      group_size <= 0 || (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
#define HIPENGINE_LAUNCH_DUAL_PACK8_MULTI_ROW(MAX_ROWS) \
  hipLaunchKernelGGL( \
      (gemv_awq_dual_pack8_multi_row_kernel<scalar_t, qweight_transposed, separate_inputs, OUT_LAYOUT, MAX_ROWS, decode_dequant>), \
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b)), \
      dim3(static_cast<unsigned int>(threads)), \
      0, \
      stream, \
      x_a, \
      x_b, \
      qweight_a, \
      qzeros_a, \
      scales_a, \
      qweight_b, \
      qzeros_b, \
      scales_b, \
      out_a, \
      out_b, \
      rows, \
      in_features, \
      out_packed_a, \
      out_packed_b, \
      static_cast<int>(group_size)); \
  return static_cast<int>(hipGetLastError())

  if (rows <= 2) { HIPENGINE_LAUNCH_DUAL_PACK8_MULTI_ROW(2); }
  if (rows <= 4) { HIPENGINE_LAUNCH_DUAL_PACK8_MULTI_ROW(4); }
  HIPENGINE_LAUNCH_DUAL_PACK8_MULTI_ROW(8);
#undef HIPENGINE_LAUNCH_DUAL_PACK8_MULTI_ROW
}

extern "C" int hipengine_gemv_awq_dual_pack8_multi_row_strided_fp16(
    const half_t* x_a,
    const half_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_dual_pack8_multi_row<half_t, false, false, 0>(
      x_a, nullptr, qweight_a, qzeros_a, scales_a, qweight_b, qzeros_b, scales_b,
      out, nullptr, rows, in_features, out_packed_a, out_packed_b, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_dual_pack8_multi_row_transposed_fp16(
    const half_t* x_a,
    const half_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_dual_pack8_multi_row<half_t, true, true, 0>(
      x_a, x_b, qweight_a, qzeros_a, scales_a, qweight_b, qzeros_b, scales_b,
      out, nullptr, rows, in_features, out_packed_a, out_packed_b, group_size, threads, stream);
}

// Split-output variants matching the ``awq_fusedw4_prefill_dual_fp16`` ABI.
extern "C" int hipengine_gemv_awq_dual_pack8_multi_row_split_transposed_fp16(
    const half_t* x_a,
    const half_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out_a,
    half_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_dual_pack8_multi_row<half_t, true, true, 1>(
      x_a, x_b, qweight_a, qzeros_a, scales_a, qweight_b, qzeros_b, scales_b,
      out_a, out_b, rows, in_features, out_packed_a, out_packed_b, group_size, threads, stream);
}

// M15.3: decode-dequant split-output dual.  Bit-identical to two
// ``gemv_awq_pack8_multi_row_decode_transposed_fp16`` single GEMVs (same f32
// dequant, k-order, and PARO_PACK8 reduction per output) but fuses both
// projections into one launch to cut the verifier launch count.
extern "C" int hipengine_gemv_awq_dual_pack8_multi_row_decode_split_transposed_fp16(
    const half_t* x_a,
    const half_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out_a,
    half_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_dual_pack8_multi_row<half_t, true, true, 1, true>(
      x_a, x_b, qweight_a, qzeros_a, scales_a, qweight_b, qzeros_b, scales_b,
      out_a, out_b, rows, in_features, out_packed_a, out_packed_b, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_selected_dual_pack8_strided_rotate_out_fp16(
    const half_t* x,
    const int64_t* selected,
    const int16_t* pairs,
    const half_t* theta,
    const half_t* channel_scales,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected_dual_rotate<half_t, false>(
      x,
      selected,
      pairs,
      theta,
      channel_scales,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      x_rows,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      group_size,
      krot,
      threads,
      stream);
}

extern "C" int hipengine_gemv_awq_selected_dual_pack8_strided_fp16(
    const half_t* x,
    const int64_t* selected,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected_dual_pack8<half_t, false>(
      x,
      selected,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      x_rows,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      group_size,
      threads,
      stream);
}

extern "C" int hipengine_gemv_awq_selected_dual_pack8_transposed_fp16(
    const half_t* x,
    const int64_t* selected,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected_dual_pack8<half_t, true>(
      x,
      selected,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      x_rows,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      group_size,
      threads,
      stream);
}

// M13.B.1: transposed-layout fused rotate+selected_dual_pack8 GEMV.  The
// templated kernel body already handles `qweight_transposed=true`; this just
// exposes the missing extern-C wrapper so the runtime MoE selected gate_up
// path can replace its (paro_rotate1 + selected_dual_GEMV) pair with one
// fused launch.  Numerics are bit-exact with the unfused chain via the
// Option-C LDS scalar_t round-trip inside the kernel body.
extern "C" int hipengine_gemv_awq_selected_dual_pack8_transposed_rotate_out_fp16(
    const half_t* x,
    const int64_t* selected,
    const int16_t* pairs,
    const half_t* theta,
    const half_t* channel_scales,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected_dual_rotate<half_t, true>(
      x,
      selected,
      pairs,
      theta,
      channel_scales,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      x_rows,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      group_size,
      krot,
      threads,
      stream);
}

extern "C" int hipengine_gemv_awq_selected_dual_pack8_transposed_rotate_staged_fp16(
    const half_t* x,
    half_t* rotated,
    const int64_t* selected,
    const int16_t* pairs,
    const half_t* theta,
    const half_t* channel_scales,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int32_t* barrier,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected_dual_rotate_staged<half_t, true>(
      x,
      rotated,
      selected,
      pairs,
      theta,
      channel_scales,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      barrier,
      x_rows,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      group_size,
      krot,
      threads,
      stream);
}

extern "C" int hipengine_gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16(
    const half_t* x,
    half_t* rotated,
    const int64_t* selected,
    const int16_t* pairs,
    const half_t* theta,
    const half_t* channel_scales,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int32_t* barrier,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    int64_t barrier_count_target,
    int64_t barrier_ready_value,
    hipStream_t stream) {
  return launch_selected_dual_rotate_staged_keyed<half_t, true>(
      x,
      rotated,
      selected,
      pairs,
      theta,
      channel_scales,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      barrier,
      x_rows,
      rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      group_size,
      krot,
      threads,
      barrier_count_target,
      barrier_ready_value,
      stream);
}

extern "C" int hipengine_gemv_awq_selected_pack8_strided_fp16(
    const half_t* x,
    const int64_t* selected,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected_pack8<half_t, false>(
      x, selected, qweight, qzeros, scales, out, rows, in_features, out_packed, num_experts, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_selected_pack8_transposed_fp16(
    const half_t* x,
    const int64_t* selected,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected_pack8<half_t, true>(
      x, selected, qweight, qzeros, scales, out, rows, in_features, out_packed, num_experts, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_selected_pack8_transposed_silu_rotate_staged_fp16(
    const half_t* gate_up,
    half_t* down_input,
    const int64_t* selected,
    const int16_t* pairs,
    const half_t* theta,
    const half_t* channel_scales,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int32_t* barrier,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    hipStream_t stream) {
  return launch_selected_pack8_silu_rotate_staged<half_t, true>(
      gate_up,
      down_input,
      selected,
      pairs,
      theta,
      channel_scales,
      qweight,
      qzeros,
      scales,
      out,
      barrier,
      rows,
      in_features,
      out_packed,
      num_experts,
      group_size,
      krot,
      threads,
      stream);
}

extern "C" int hipengine_gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16(
    const half_t* gate_up,
    half_t* down_input,
    const int64_t* selected,
    const int16_t* pairs,
    const half_t* theta,
    const half_t* channel_scales,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int32_t* barrier,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    int64_t barrier_count_target,
    int64_t barrier_ready_value,
    hipStream_t stream) {
  return launch_selected_pack8_silu_rotate_staged_keyed<half_t, true>(
      gate_up,
      down_input,
      selected,
      pairs,
      theta,
      channel_scales,
      qweight,
      qzeros,
      scales,
      out,
      barrier,
      rows,
      in_features,
      out_packed,
      num_experts,
      group_size,
      krot,
      threads,
      barrier_count_target,
      barrier_ready_value,
      stream);
}

extern "C" int hipengine_gemv_awq_selected_dual_pack8_strided_rotate_out_bf16(
    const uint16_t* x,
    const int64_t* selected,
    const int16_t* pairs,
    const uint16_t* theta,
    const uint16_t* channel_scales,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t krot,
    int64_t threads,
    hipStream_t stream) {
  if (x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 ||
      num_experts <= 0 || group_size <= 0 || krot < 0 || (in_features % group_size) != 0 ||
      (group_size % 8) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_dual_pack8_strided_rotate_out_kernel<uint16_t, false>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      static_cast<unsigned int>(in_features * sizeof(float)),
      stream,
      x,
      selected,
      pairs,
      theta,
      channel_scales,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      x_rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      static_cast<int>(group_size),
      static_cast<int>(krot));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_selected_dual_pack8_strided_bf16(
    const uint16_t* x,
    const int64_t* selected,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 ||
      num_experts <= 0 || group_size <= 0 || (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_dual_pack8_strided_kernel<uint16_t, false>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      selected,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      x_rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_selected_dual_pack8_transposed_bf16(
    const uint16_t* x,
    const int64_t* selected,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int64_t x_rows,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (x_rows <= 0 || rows <= 0 || (rows % x_rows) != 0 || in_features <= 0 || out_packed_a <= 0 || out_packed_b <= 0 ||
      num_experts <= 0 || group_size <= 0 || (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_dual_pack8_strided_kernel<uint16_t, true>),
      dim3(static_cast<unsigned int>(out_packed_a + out_packed_b), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      selected,
      qweight_a,
      qzeros_a,
      scales_a,
      qweight_b,
      qzeros_b,
      scales_b,
      out,
      x_rows,
      in_features,
      out_packed_a,
      out_packed_b,
      num_experts,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_selected_pack8_strided_bf16(
    const uint16_t* x,
    const int64_t* selected,
    const int32_t* qweight,
    const int32_t* qzeros,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed <= 0 || num_experts <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_pack8_kernel<uint16_t, false>),
      dim3(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      selected,
      qweight,
      qzeros,
      scales,
      out,
      in_features,
      out_packed << 3,
      out_packed,
      num_experts,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_gemv_awq_selected_pack8_transposed_bf16(
    const uint16_t* x,
    const int64_t* selected,
    const int32_t* qweight,
    const int32_t* qzeros,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t num_experts,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  if (rows <= 0 || in_features <= 0 || out_packed <= 0 || num_experts <= 0 || group_size <= 0 ||
      (in_features % group_size) != 0 || !valid_threads(threads)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      (gemv_awq_selected_pack8_kernel<uint16_t, true>),
      dim3(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(threads)),
      0,
      stream,
      x,
      selected,
      qweight,
      qzeros,
      scales,
      out,
      in_features,
      out_packed << 3,
      out_packed,
      num_experts,
      static_cast<int>(group_size));
  return static_cast<int>(hipGetLastError());
}


extern "C" int hipengine_gemv_awq_pack8_output_tiled_bf16(
    const uint16_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_pack8_output_tiled<uint16_t, false>(
      x, qweight, qzeros, scales, out, rows, in_features, out_packed, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_pack8_output_tiled_transposed_bf16(
    const uint16_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const uint16_t* scales,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_pack8_output_tiled<uint16_t, true>(
      x, qweight, qzeros, scales, out, rows, in_features, out_packed, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_pack8_output_tiled_fp16(
    const half_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_pack8_output_tiled<half_t, false>(
      x, qweight, qzeros, scales, out, rows, in_features, out_packed, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_pack8_output_tiled_transposed_fp16(
    const half_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_pack8_output_tiled<half_t, true>(
      x, qweight, qzeros, scales, out, rows, in_features, out_packed, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16(
    const half_t* x,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half_t* scales,
    const half_t* selected,
    const float* routing_weights,
    const float* shared_gate_logits,
    const half_t* residual,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed,
    int64_t group_size,
    int64_t rows_per_token,
    int64_t gate_stride,
    int64_t threads,
    hipStream_t stream) {
  return launch_pack8_output_tiled_combine_residual<half_t, true>(
      x, qweight, qzeros, scales, selected, routing_weights, shared_gate_logits,
      residual, out, rows, in_features, out_packed, group_size, rows_per_token,
      gate_stride, threads, stream);
}

extern "C" int hipengine_gemv_awq_dual_pack8_output_tiled_transposed_fp16(
    const half_t* x_a,
    const half_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_dual_pack8_output_tiled<half_t, true, true, 0>(
      x_a, x_b, qweight_a, qzeros_a, scales_a, qweight_b, qzeros_b, scales_b,
      out, nullptr, rows, in_features, out_packed_a, out_packed_b, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_dual_pack8_output_tiled_split_transposed_fp16(
    const half_t* x_a,
    const half_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out_a,
    half_t* out_b,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_dual_pack8_output_tiled<half_t, true, true, 1>(
      x_a, x_b, qweight_a, qzeros_a, scales_a, qweight_b, qzeros_b, scales_b,
      out_a, out_b, rows, in_features, out_packed_a, out_packed_b, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_dual_pack8_output_tiled_transposed_bf16(
    const uint16_t* x_a,
    const uint16_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_dual_pack8_output_tiled<uint16_t, true, true, 0>(
      x_a, x_b, qweight_a, qzeros_a, scales_a, qweight_b, qzeros_b, scales_b,
      out, nullptr, rows, in_features, out_packed_a, out_packed_b, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_dual_pack8_output_tiled_strided_fp16(
    const half_t* x_a,
    const half_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const half_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const half_t* scales_b,
    half_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_dual_pack8_output_tiled<half_t, false, false, 0>(
      x_a, x_b, qweight_a, qzeros_a, scales_a, qweight_b, qzeros_b, scales_b,
      out, nullptr, rows, in_features, out_packed_a, out_packed_b, group_size, threads, stream);
}

extern "C" int hipengine_gemv_awq_dual_pack8_output_tiled_strided_bf16(
    const uint16_t* x_a,
    const uint16_t* x_b,
    const int32_t* qweight_a,
    const int32_t* qzeros_a,
    const uint16_t* scales_a,
    const int32_t* qweight_b,
    const int32_t* qzeros_b,
    const uint16_t* scales_b,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_packed_a,
    int64_t out_packed_b,
    int64_t group_size,
    int64_t threads,
    hipStream_t stream) {
  return launch_dual_pack8_output_tiled<uint16_t, false, false, 0>(
      x_a, x_b, qweight_a, qzeros_a, scales_a, qweight_b, qzeros_b, scales_b,
      out, nullptr, rows, in_features, out_packed_a, out_packed_b, group_size, threads, stream);
}
