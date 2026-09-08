// Dense single-row/small-row q8_1-DP4A Q4_K GEMV: load-reuse A/B leaf screen.
//
// Port of the nasone32/llama.cpp "rdna-boosts block 10: k-quant-boosts"
// adjacent-chunk load amortization (source: nasone32/llama.cpp-RDNA3-
// 7900xtx-opt commit efa4e86410c07723deaa458bdadd8c08f1029928,
// ggml/src/ggml-cuda/vecdotq.cuh: VDR_Q4_K_Q8_1_MMVQ 2 -> 4 — one thread
// step covers a full 32-element subblock so the shared q8_1 data loads,
// the block-scale loads and the weight sub-scales/mins are amortized
// across all of the covered elements and output columns) into the
// hipEngine raw-GGUF q8_1-dp4a decode idiom.
//
// Two kernels, SAME subblock-strided thread mapping and SAME per-pack f32
// expression evaluated in the SAME order:
//
//   * control: re-decodes the Q4_K block header (d, dmin), the 6-bit
//     scale/min pair and the q8_1 block scale for EVERY 4-k pack — the
//     unamortized idiom.
//   * vdr: hoists d*scale, dmin*min (per column) and the q8_1 block scale
//     once per 32-element subblock and reuses them for all 8 packs; the
//     8 activation u32 packs are loaded once and reused across the 8
//     output columns.
//
// The RED contract in tests/test_gguf_q4_k_q8_1_dp4a_vdr_gemv.py is
// BIT-EXACT equality between the two kernels: the only admissible
// difference is load scheduling, never values. This is a leaf screen —
// the production dense Q4_K decode owner (float pack8 GEMV) is untouched
// and remains the registered exact path; the q8_1-activation arithmetic
// class is a separate, production-profile-gated question.
//
// Grid: (out_features / 8, rows). One block of 128 threads computes one
// (8-output-column pack, row); threads walk whole 32-element subblocks.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int QK_K = 256;
constexpr int Q4_K_BLOCK_BYTES = 144;
constexpr int Q4_K_QS_OFFSET = 2 + 2 + 12;  // d + dmin + 12 scale bytes
constexpr int Q4_K_SUBBLOCK = 32;
constexpr int Q4_K_SUBBLOCKS = QK_K / Q4_K_SUBBLOCK;  // == 8
constexpr int Q8_1_BLOCK = 32;
constexpr int PACKS_PER_SUBBLOCK = Q4_K_SUBBLOCK / 4;  // == 8
constexpr int OUT_COLS = 8;

using half_t = _Float16;

struct gguf_q8_1_block {
  half_t d;
  half_t s;
  int8_t qs[Q8_1_BLOCK];
};
static_assert(sizeof(gguf_q8_1_block) == 36, "GGUF q8_1 block layout must match GGML");

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

__device__ inline int q4_k_dp4a(uint32_t a, uint32_t b, int c) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(
      true, static_cast<int>(a), true, static_cast<int>(b), c, false);
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

__device__ inline uint8_t q4_k_scale(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] & 0x0Fu) | ((scales[idx] >> 2) & 0x30u);
}

__device__ inline uint8_t q4_k_min(const uint8_t* scales, int subblock) {
  if (subblock < 4) {
    return scales[4 + subblock] & 0x3Fu;
  }
  const int idx = subblock - 4;
  return (scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30u);
}

__device__ inline uint32_t q4_k_pack4(
    const uint8_t* qs, int subblock, int lane4) {
  const int pair = subblock >> 1;
  const uint32_t packed =
      *reinterpret_cast<const uint32_t*>(qs + pair * Q4_K_SUBBLOCK + lane4);
  return (subblock & 1) ? ((packed >> 4) & 0x0F0F0F0Fu)
                        : (packed & 0x0F0F0F0Fu);
}

template <typename out_t>
__device__ inline void store_output(out_t* out, int64_t index, float value) {
  out[index] = static_cast<out_t>(value);
}

template <>
__device__ inline void store_output<uint16_t>(
    uint16_t* out, int64_t index, float value) {
  out[index] = float_to_bf16_bits(value);
}

// Wave-level reduce then cross-wave exchange, the proven raw-GGUF idiom:
// lane 0 of each wave writes its reduced partials and thread 0 sums the
// four wave slots and stores the eight outputs.
template <typename out_t>
__device__ inline void gemv_finalize(
    float* acc,
    out_t* out,
    int64_t row,
    int64_t out_col,
    int64_t out_features,
    int tid) {
#pragma unroll
  for (int c = 0; c < OUT_COLS; ++c) {
    float v = acc[c];
    v += __shfl_down(v, 16);
    v += __shfl_down(v, 8);
    v += __shfl_down(v, 4);
    v += __shfl_down(v, 2);
    v += __shfl_down(v, 1);
    acc[c] = v;
  }
  __shared__ float xchg[4 * OUT_COLS];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
#pragma unroll
    for (int c = 0; c < OUT_COLS; ++c) {
      xchg[sub_id * OUT_COLS + c] = acc[c];
    }
  }
  __syncthreads();
  if (tid == 0) {
    out_t* __restrict__ out_ptr =
        out + row * out_features + out_col;
#pragma unroll
    for (int c = 0; c < OUT_COLS; ++c) {
      float v = 0.0f;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        v += xchg[g * OUT_COLS + c];
      }
      store_output<out_t>(out_ptr, c, v);
    }
  }
}

template <typename out_t>
__launch_bounds__(128, 2) __global__ void q4_k_q8_1_dp4a_ctl_gemv_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_col = static_cast<int64_t>(blockIdx.x) << 3;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows || out_col >= out_features) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const uint8_t* __restrict__ row_ptr[OUT_COLS];
#pragma unroll
  for (int ch = 0; ch < OUT_COLS; ++ch) {
    row_ptr[ch] = qweight + (out_col + ch) * row_bytes;
  }
  const gguf_q8_1_block* __restrict__ xq_row =
      xq + row * (in_features / Q8_1_BLOCK);

  float acc[OUT_COLS];
#pragma unroll
  for (int c = 0; c < OUT_COLS; ++c) acc[c] = 0.0f;

  // Subblock-strided walk; every pack re-decodes the block header, the
  // scale/min pair and the q8_1 block scale (unamortized control).
  const int groups32 = static_cast<int>(in_features >> 5);
  for (int g32 = tid; g32 < groups32; g32 += blockDim.x) {
    const int block_idx = g32 >> 3;
    const int subblock = g32 & 7;
    const gguf_q8_1_block* __restrict__ xb =
        xq_row + static_cast<int64_t>(block_idx) * Q4_K_SUBBLOCKS + subblock;
    for (int pack = 0; pack < PACKS_PER_SUBBLOCK; ++pack) {
      const int lane4 = pack << 2;
      const uint32_t x_pack =
          *reinterpret_cast<const uint32_t*>(xb->qs + lane4);
      const int x_sum4 = q4_k_dp4a(0x01010101u, x_pack, 0);
      const float xd = static_cast<float>(xb->d);
#pragma unroll
      for (int ch = 0; ch < OUT_COLS; ++ch) {
        const uint8_t* __restrict__ block =
            row_ptr[ch] + static_cast<int64_t>(block_idx) * Q4_K_BLOCK_BYTES;
        const float d = fp16_bytes_to_float(block);
        const float dmin = fp16_bytes_to_float(block + 2);
        const uint8_t* __restrict__ scales = block + 4;
        const uint8_t* __restrict__ qs = block + Q4_K_QS_OFFSET;
        const uint32_t q4 = q4_k_pack4(qs, subblock, lane4);
        const int dot = q4_k_dp4a(q4, x_pack, 0);
        acc[ch] += xd * (d * static_cast<float>(q4_k_scale(scales, subblock)) *
                             static_cast<float>(dot) -
                         dmin * static_cast<float>(q4_k_min(scales, subblock)) *
                             static_cast<float>(x_sum4));
      }
    }
  }

  gemv_finalize<out_t>(acc, out, row, out_col, out_features, tid);
}

template <typename out_t>
__launch_bounds__(128, 2) __global__ void q4_k_q8_1_dp4a_vdr_gemv_kernel(
    const gguf_q8_1_block* __restrict__ xq,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_col = static_cast<int64_t>(blockIdx.x) << 3;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows || out_col >= out_features) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
  const uint8_t* __restrict__ row_ptr[OUT_COLS];
#pragma unroll
  for (int ch = 0; ch < OUT_COLS; ++ch) {
    row_ptr[ch] = qweight + (out_col + ch) * row_bytes;
  }
  const gguf_q8_1_block* __restrict__ xq_row =
      xq + row * (in_features / Q8_1_BLOCK);

  float acc[OUT_COLS];
#pragma unroll
  for (int c = 0; c < OUT_COLS; ++c) acc[c] = 0.0f;

  // Same subblock-strided walk and per-pack f32 expression; the shared
  // loads are hoisted once per 32-element subblock (nasone32 efa4e864
  // adjacent-chunk amortization).
  const int groups32 = static_cast<int>(in_features >> 5);
  for (int g32 = tid; g32 < groups32; g32 += blockDim.x) {
    const int block_idx = g32 >> 3;
    const int subblock = g32 & 7;
    const gguf_q8_1_block* __restrict__ xb =
        xq_row + static_cast<int64_t>(block_idx) * Q4_K_SUBBLOCKS + subblock;

    // Hoisted activation loads: 8 u32 packs + the block scale, reused by
    // all 8 output columns.
    uint32_t x_pack[PACKS_PER_SUBBLOCK];
    int x_sum4[PACKS_PER_SUBBLOCK];
#pragma unroll
    for (int pack = 0; pack < PACKS_PER_SUBBLOCK; ++pack) {
      x_pack[pack] =
          *reinterpret_cast<const uint32_t*>(xb->qs + (pack << 2));
      x_sum4[pack] = q4_k_dp4a(0x01010101u, x_pack[pack], 0);
    }
    const float xd = static_cast<float>(xb->d);

    // Hoisted per-column metadata: d*scale and dmin*min decoded once per
    // subblock instead of once per pack.
    const uint8_t* __restrict__ qs_ptr[OUT_COLS];
    float d_scale[OUT_COLS];
    float dmin_min[OUT_COLS];
#pragma unroll
    for (int ch = 0; ch < OUT_COLS; ++ch) {
      const uint8_t* __restrict__ block =
          row_ptr[ch] + static_cast<int64_t>(block_idx) * Q4_K_BLOCK_BYTES;
      const float d = fp16_bytes_to_float(block);
      const float dmin = fp16_bytes_to_float(block + 2);
      const uint8_t* __restrict__ scales = block + 4;
      qs_ptr[ch] = block + Q4_K_QS_OFFSET;
      d_scale[ch] = d * static_cast<float>(q4_k_scale(scales, subblock));
      dmin_min[ch] = dmin * static_cast<float>(q4_k_min(scales, subblock));
    }

    for (int pack = 0; pack < PACKS_PER_SUBBLOCK; ++pack) {
      const int lane4 = pack << 2;
#pragma unroll
      for (int ch = 0; ch < OUT_COLS; ++ch) {
        const uint32_t q4 = q4_k_pack4(qs_ptr[ch], subblock, lane4);
        const int dot = q4_k_dp4a(q4, x_pack[pack], 0);
        acc[ch] += xd * (d_scale[ch] * static_cast<float>(dot) -
                         dmin_min[ch] * static_cast<float>(x_sum4[pack]));
      }
    }
  }

  gemv_finalize<out_t>(acc, out, row, out_col, out_features, tid);
}

}  // namespace

#define DEFINE_LAUNCH(SYMBOL, KERNEL, OUT_T)                               \
  extern "C" int SYMBOL(                                                   \
      const void* xq,                                                      \
      const uint8_t* qweight,                                              \
      void* out,                                                           \
      int64_t rows,                                                        \
      int64_t in_features,                                                 \
      int64_t out_features,                                                \
      hipStream_t stream) {                                                \
    if (xq == nullptr || qweight == nullptr || out == nullptr || rows <= 0 || \
        in_features <= 0 || (in_features % QK_K) != 0 ||                   \
        out_features <= 0 || (out_features % 8) != 0) {                    \
      return static_cast<int>(hipErrorInvalidValue);                       \
    }                                                                      \
    const dim3 block(128);                                                 \
    const dim3 grid(                                                       \
        static_cast<unsigned int>(out_features / 8),                       \
        static_cast<unsigned int>(rows));                                  \
    hipLaunchKernelGGL(                                                    \
        (KERNEL), grid, block, 0, stream,                                  \
        reinterpret_cast<const gguf_q8_1_block*>(xq),                      \
        qweight,                                                           \
        reinterpret_cast<OUT_T*>(out),                                     \
        rows,                                                              \
        in_features,                                                      \
        out_features);                                                     \
    return static_cast<int>(hipGetLastError());                            \
  }

DEFINE_LAUNCH(
    hipengine_gguf_q4_k_q8_1_dp4a_ctl_bf16_f32_out,
    q4_k_q8_1_dp4a_ctl_gemv_kernel<float>,
    float)
DEFINE_LAUNCH(
    hipengine_gguf_q4_k_q8_1_dp4a_vdr_bf16_f32_out,
    q4_k_q8_1_dp4a_vdr_gemv_kernel<float>,
    float)
DEFINE_LAUNCH(
    hipengine_gguf_q4_k_q8_1_dp4a_ctl_bf16_bf16_out,
    q4_k_q8_1_dp4a_ctl_gemv_kernel<uint16_t>,
    uint16_t)
DEFINE_LAUNCH(
    hipengine_gguf_q4_k_q8_1_dp4a_vdr_bf16_bf16_out,
    q4_k_q8_1_dp4a_vdr_gemv_kernel<uint16_t>,
    uint16_t)
