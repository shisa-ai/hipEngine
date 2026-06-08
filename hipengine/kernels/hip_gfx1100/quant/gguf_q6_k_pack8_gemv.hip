// Dense decode-shaped pack8 GEMV for raw GGUF Q6_K weights.
//
// P9.B4b implementation note (kernel added during task #24 to enable the
// Q6_K F32 lm-head correctness fixture): mirrors the structure of
// ``gguf_q4_k_pack8_gemv.hip`` (single-output PARO-style pack8 GEMV) with
// the inner k loop swapped for raw GGUF Q6_K block dequant (int8 per-16-K
// scales + fp16 super-scale + 2 high bits per element).
//
//   x          [rows, in_features]
//   qweight    [out_features, row_bytes]   raw block_q6_K bytes
//   out        [rows, out_features]
//
// Four ``(scalar_in_t, scalar_out_t)`` combinations are supported: BF16/BF16,
// FP16/FP16, BF16/F32, FP16/F32. The F32 output variants are the lm-head
// logits projection case for qwen35moe Qwen3.6-35B-A3B-UD-Q4_K_M (the tied
// output weight is stored as Q6_K).
//
// No new ABI and no resident weight sidecar/repack: raw GGUF Q6_K bytes
// stay on device.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q6_K_BLOCK_BYTES = 210;
constexpr int Q6_K_GROUPS16 = QK_K / 16;  // == 16
constexpr int PACK8 = 8;

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

template <>
__device__ inline float float_to_scalar<float>(float v) {
  return v;
}

#define GGUF_Q6K_PACK8_WAVE_REDUCE(acc)               \
  do {                                                 \
    acc += __shfl_down(acc, 16);                       \
    acc += __shfl_down(acc, 8);                        \
    acc += __shfl_down(acc, 4);                        \
    acc += __shfl_down(acc, 2);                        \
    acc += __shfl_down(acc, 1);                        \
  } while (0)

// Dense Q6_K pack8 GEMV decode.
//
// Grid: (out_features / 8, rows). One block per (output pack of 8, row).
// Block: 128 threads (4 wave32 waves; __launch_bounds__(128, 4)).
template <typename scalar_in_t, typename scalar_out_t>
__launch_bounds__(128, 4) __global__ void gguf_q6_k_pack8_gemv_kernel(
    const scalar_in_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    scalar_out_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t out_col = out_pack << 3;
  if (out_col >= out_features) return;

  const int64_t blocks_per_row = in_features / QK_K;
  const int64_t row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;

  const uint8_t* __restrict__ row_ptr[PACK8];
#pragma unroll
  for (int ch = 0; ch < PACK8; ++ch) {
    row_ptr[ch] = qweight + (out_col + ch) * row_bytes;
  }

  const scalar_in_t* __restrict__ x_row = x + row * in_features;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

  // Per-block hoist of (d * scales[g16]) for all 8 output channels x 16
  // groups of 16. 128 floats, cooperatively loaded by the first 128 threads.
  __shared__ float s_scale[PACK8 * Q6_K_GROUPS16];

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    if (tid < PACK8 * Q6_K_GROUPS16) {
      const int ch = tid >> 4;
      const int g16 = tid & 15;
      const uint8_t* __restrict__ block = row_ptr[ch] + blk * Q6_K_BLOCK_BYTES;
      const float d_f = fp16_bytes_to_float(block + 208);
      const int8_t* __restrict__ scales = reinterpret_cast<const int8_t*>(block + 192);
      s_scale[ch * Q6_K_GROUPS16 + g16] = d_f * static_cast<float>(scales[g16]);
    }
    __syncthreads();

    const int64_t k_block_base = blk * QK_K;
    // Per-channel ql (128 bytes) and qh (64 bytes) pointers.
    const uint8_t* __restrict__ ql0 = row_ptr[0] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql1 = row_ptr[1] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql2 = row_ptr[2] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql3 = row_ptr[3] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql4 = row_ptr[4] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql5 = row_ptr[5] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql6 = row_ptr[6] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ ql7 = row_ptr[7] + blk * Q6_K_BLOCK_BYTES;
    const uint8_t* __restrict__ qhp0 = ql0 + 128;
    const uint8_t* __restrict__ qhp1 = ql1 + 128;
    const uint8_t* __restrict__ qhp2 = ql2 + 128;
    const uint8_t* __restrict__ qhp3 = ql3 + 128;
    const uint8_t* __restrict__ qhp4 = ql4 + 128;
    const uint8_t* __restrict__ qhp5 = ql5 + 128;
    const uint8_t* __restrict__ qhp6 = ql6 + 128;
    const uint8_t* __restrict__ qhp7 = ql7 + 128;

    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int group32 = k_in_block >> 5;
      const int lane = k_in_block & 31;
      const int g16 = k_in_block >> 4;
      const int base64 = (group32 >= 4) ? 64 : 0;
      const bool low_nibble = (group32 & 2) == 0;
      const int ql_group = group32 & 1;
      const int ql_idx = base64 + ql_group * 32 + lane;
      const int qh_base = (group32 >= 4) ? 32 : 0;
      const int qh_idx = qh_base + lane;
      const int qh_shift = 2 * (group32 & 3);

      const float xv = scalar_to_float<scalar_in_t>(x_row[k_block_base + k_in_block]);

      const uint8_t lb0 = ql0[ql_idx];
      const uint8_t lb1 = ql1[ql_idx];
      const uint8_t lb2 = ql2[ql_idx];
      const uint8_t lb3 = ql3[ql_idx];
      const uint8_t lb4 = ql4[ql_idx];
      const uint8_t lb5 = ql5[ql_idx];
      const uint8_t lb6 = ql6[ql_idx];
      const uint8_t lb7 = ql7[ql_idx];
      const uint8_t l0 = low_nibble ? (lb0 & 0x0Fu) : (lb0 >> 4);
      const uint8_t l1 = low_nibble ? (lb1 & 0x0Fu) : (lb1 >> 4);
      const uint8_t l2 = low_nibble ? (lb2 & 0x0Fu) : (lb2 >> 4);
      const uint8_t l3 = low_nibble ? (lb3 & 0x0Fu) : (lb3 >> 4);
      const uint8_t l4 = low_nibble ? (lb4 & 0x0Fu) : (lb4 >> 4);
      const uint8_t l5 = low_nibble ? (lb5 & 0x0Fu) : (lb5 >> 4);
      const uint8_t l6 = low_nibble ? (lb6 & 0x0Fu) : (lb6 >> 4);
      const uint8_t l7 = low_nibble ? (lb7 & 0x0Fu) : (lb7 >> 4);
      const uint8_t h0 = (qhp0[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h1 = (qhp1[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h2 = (qhp2[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h3 = (qhp3[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h4 = (qhp4[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h5 = (qhp5[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h6 = (qhp6[qh_idx] >> qh_shift) & 0x03u;
      const uint8_t h7 = (qhp7[qh_idx] >> qh_shift) & 0x03u;
      const float q0 = static_cast<float>(static_cast<int>(l0 | (h0 << 4)) - 32);
      const float q1 = static_cast<float>(static_cast<int>(l1 | (h1 << 4)) - 32);
      const float q2 = static_cast<float>(static_cast<int>(l2 | (h2 << 4)) - 32);
      const float q3 = static_cast<float>(static_cast<int>(l3 | (h3 << 4)) - 32);
      const float q4 = static_cast<float>(static_cast<int>(l4 | (h4 << 4)) - 32);
      const float q5 = static_cast<float>(static_cast<int>(l5 | (h5 << 4)) - 32);
      const float q6 = static_cast<float>(static_cast<int>(l6 | (h6 << 4)) - 32);
      const float q7 = static_cast<float>(static_cast<int>(l7 | (h7 << 4)) - 32);

      acc0 = fmaf(xv, s_scale[0 * Q6_K_GROUPS16 + g16] * q0, acc0);
      acc1 = fmaf(xv, s_scale[1 * Q6_K_GROUPS16 + g16] * q1, acc1);
      acc2 = fmaf(xv, s_scale[2 * Q6_K_GROUPS16 + g16] * q2, acc2);
      acc3 = fmaf(xv, s_scale[3 * Q6_K_GROUPS16 + g16] * q3, acc3);
      acc4 = fmaf(xv, s_scale[4 * Q6_K_GROUPS16 + g16] * q4, acc4);
      acc5 = fmaf(xv, s_scale[5 * Q6_K_GROUPS16 + g16] * q5, acc5);
      acc6 = fmaf(xv, s_scale[6 * Q6_K_GROUPS16 + g16] * q6, acc6);
      acc7 = fmaf(xv, s_scale[7 * Q6_K_GROUPS16 + g16] * q7, acc7);
    }
    __syncthreads();
  }

  GGUF_Q6K_PACK8_WAVE_REDUCE(acc0);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc1);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc2);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc3);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc4);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc5);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc6);
  GGUF_Q6K_PACK8_WAVE_REDUCE(acc7);

  __shared__ float xchg[4 * PACK8];
  const int sub_id = tid >> 5;
  const int sub_lane = tid & 31;
  if (sub_lane == 0) {
    xchg[sub_id * PACK8 + 0] = acc0;
    xchg[sub_id * PACK8 + 1] = acc1;
    xchg[sub_id * PACK8 + 2] = acc2;
    xchg[sub_id * PACK8 + 3] = acc3;
    xchg[sub_id * PACK8 + 4] = acc4;
    xchg[sub_id * PACK8 + 5] = acc5;
    xchg[sub_id * PACK8 + 6] = acc6;
    xchg[sub_id * PACK8 + 7] = acc7;
  }
  __syncthreads();

  if (tid == 0) {
    float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
    float r4 = 0.0f, r5 = 0.0f, r6 = 0.0f, r7 = 0.0f;
    const int sub_groups = blockDim.x >> 5;
    for (int g = 0; g < sub_groups; ++g) {
      r0 += xchg[g * PACK8 + 0];
      r1 += xchg[g * PACK8 + 1];
      r2 += xchg[g * PACK8 + 2];
      r3 += xchg[g * PACK8 + 3];
      r4 += xchg[g * PACK8 + 4];
      r5 += xchg[g * PACK8 + 5];
      r6 += xchg[g * PACK8 + 6];
      r7 += xchg[g * PACK8 + 7];
    }
    scalar_out_t* __restrict__ out_ptr = out + row * out_features + out_col;
    out_ptr[0] = float_to_scalar<scalar_out_t>(r0);
    out_ptr[1] = float_to_scalar<scalar_out_t>(r1);
    out_ptr[2] = float_to_scalar<scalar_out_t>(r2);
    out_ptr[3] = float_to_scalar<scalar_out_t>(r3);
    out_ptr[4] = float_to_scalar<scalar_out_t>(r4);
    out_ptr[5] = float_to_scalar<scalar_out_t>(r5);
    out_ptr[6] = float_to_scalar<scalar_out_t>(r6);
    out_ptr[7] = float_to_scalar<scalar_out_t>(r7);
  }
}

template <typename scalar_in_t, typename scalar_out_t>
int launch_q6_k_pack8_gemv(
    const scalar_in_t* x,
    const uint8_t* qweight,
    scalar_out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr ||
      rows <= 0 || in_features <= 0 || out_features <= 0 ||
      (out_features & 7) != 0 || (in_features % QK_K) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed = out_features >> 3;
  hipLaunchKernelGGL(
      (gguf_q6_k_pack8_gemv_kernel<scalar_in_t, scalar_out_t>),
      dim3(static_cast<unsigned int>(out_packed),
           static_cast<unsigned int>(rows)),
      dim3(128),
      0,
      stream,
      x, qweight, out, rows, in_features, out_features);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv<uint16_t, uint16_t>(
      x, qweight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_fp16_fp16_out(
    const _Float16* x,
    const uint8_t* qweight,
    _Float16* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv<_Float16, _Float16>(
      x, qweight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv<uint16_t, float>(
      x, qweight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q6_k_pack8_gemv_decode_fp16_f32_out(
    const _Float16* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q6_k_pack8_gemv<_Float16, float>(
      x, qweight, out, rows, in_features, out_features, stream);
}
