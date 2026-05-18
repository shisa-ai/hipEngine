// Dense decode-shaped pack8 GEMV for raw GGUF Q4_K weights.
//
// P9.B4 implementation note: mirrors the structure of
// ``paro_awq_gemv.hip::gemv_awq_pack8_kernel`` (single output) for the AWQ
// PARO path, with the inner k loop swapped for raw GGUF Q4_K block dequant
// (per-block ``d``/``dmin`` plus 8 subblocks each carrying a 6-bit scale and
// a 6-bit min). Q4_K has 256-K blocks; per-block scale/min for all 8 output
// channels of the pack are hoisted into shared memory once per block so the
// inner k loop stays in registers.
//
//   x          [rows, in_features]
//   qweight    [out_features, row_bytes]   raw block_q4_K bytes
//   out        [rows, out_features]
//
// Three output dtypes are supported (``scalar_in_t`` and ``scalar_out_t``
// chosen independently): BF16/BF16, FP16/FP16, BF16/F32, FP16/F32. The
// last two are for the qwen35moe lm-head logits projection when the tied
// output weight is stored as Q4_K.
//
// No new ABI and no resident weight sidecar/repack: raw GGUF Q4_K bytes
// stay on device.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int QK_K = 256;
constexpr int Q4_K_BLOCK_BYTES = 144;
constexpr int Q4_K_QS_OFFSET = 2 + 2 + 12;  // d + dmin + 12 scale bytes
constexpr int Q4_K_SUBBLOCK = 32;
constexpr int Q4_K_SUBBLOCKS = QK_K / Q4_K_SUBBLOCK;  // == 8
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

__device__ inline uint8_t gguf_q4_k_quant(const uint8_t* qs, int subblock, int lane32) {
  const int pair = subblock >> 1;
  const uint8_t packed = qs[pair * Q4_K_SUBBLOCK + lane32];
  return (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
}

#define GGUF_Q4K_PACK8_WAVE_REDUCE(acc)              \
  do {                                                \
    acc += __shfl_down(acc, 16);                      \
    acc += __shfl_down(acc, 8);                       \
    acc += __shfl_down(acc, 4);                       \
    acc += __shfl_down(acc, 2);                       \
    acc += __shfl_down(acc, 1);                       \
  } while (0)

// Dense Q4_K pack8 GEMV decode.
//
// Grid: (out_features / 8, rows). One block per (output pack of 8, row).
// Block: 128 threads (4 wave32 waves; __launch_bounds__(128, 4)).
template <typename scalar_in_t, typename scalar_out_t>
__launch_bounds__(128, 4) __global__ void gguf_q4_k_pack8_gemv_kernel(
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
  const int64_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;

  const uint8_t* __restrict__ row_ptr[PACK8];
#pragma unroll
  for (int ch = 0; ch < PACK8; ++ch) {
    row_ptr[ch] = qweight + (out_col + ch) * row_bytes;
  }

  const scalar_in_t* __restrict__ x_row = x + row * in_features;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

  // Per-block hoist of (d * scales[sb], dmin * mins[sb]) for all 8 output
  // channels x 8 subblocks. 64 floats each, cooperatively loaded by the
  // first 64 threads of the block.
  __shared__ float s_scale[PACK8 * Q4_K_SUBBLOCKS];   // [ch, sb] = d * scales(sb)
  __shared__ float s_min[PACK8 * Q4_K_SUBBLOCKS];     // [ch, sb] = dmin * mins(sb)

  for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
    if (tid < PACK8 * Q4_K_SUBBLOCKS) {
      const int ch = tid >> 3;
      const int sb = tid & 7;
      const uint8_t* __restrict__ block = row_ptr[ch] + blk * Q4_K_BLOCK_BYTES;
      const float d_f = fp16_bytes_to_float(block);
      const float dmin_f = fp16_bytes_to_float(block + 2);
      const uint8_t* __restrict__ scales = block + 4;
      s_scale[ch * Q4_K_SUBBLOCKS + sb] =
          d_f * static_cast<float>(gguf_q4_k_scale(scales, sb));
      s_min[ch * Q4_K_SUBBLOCKS + sb] =
          dmin_f * static_cast<float>(gguf_q4_k_min(scales, sb));
    }
    __syncthreads();

    const int64_t k_block_base = blk * QK_K;
    const uint8_t* __restrict__ qs0 = row_ptr[0] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs1 = row_ptr[1] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs2 = row_ptr[2] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs3 = row_ptr[3] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs4 = row_ptr[4] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs5 = row_ptr[5] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs6 = row_ptr[6] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    const uint8_t* __restrict__ qs7 = row_ptr[7] + blk * Q4_K_BLOCK_BYTES + Q4_K_QS_OFFSET;
    for (int k_in_block = tid; k_in_block < QK_K; k_in_block += blockDim.x) {
      const int sb = k_in_block >> 5;
      const int lane32 = k_in_block & 31;
      const float xv = scalar_to_float<scalar_in_t>(x_row[k_block_base + k_in_block]);
      const float q0 = static_cast<float>(gguf_q4_k_quant(qs0, sb, lane32));
      const float q1 = static_cast<float>(gguf_q4_k_quant(qs1, sb, lane32));
      const float q2 = static_cast<float>(gguf_q4_k_quant(qs2, sb, lane32));
      const float q3 = static_cast<float>(gguf_q4_k_quant(qs3, sb, lane32));
      const float q4 = static_cast<float>(gguf_q4_k_quant(qs4, sb, lane32));
      const float q5 = static_cast<float>(gguf_q4_k_quant(qs5, sb, lane32));
      const float q6 = static_cast<float>(gguf_q4_k_quant(qs6, sb, lane32));
      const float q7 = static_cast<float>(gguf_q4_k_quant(qs7, sb, lane32));
      acc0 = fmaf(xv, s_scale[0 * Q4_K_SUBBLOCKS + sb] * q0 - s_min[0 * Q4_K_SUBBLOCKS + sb], acc0);
      acc1 = fmaf(xv, s_scale[1 * Q4_K_SUBBLOCKS + sb] * q1 - s_min[1 * Q4_K_SUBBLOCKS + sb], acc1);
      acc2 = fmaf(xv, s_scale[2 * Q4_K_SUBBLOCKS + sb] * q2 - s_min[2 * Q4_K_SUBBLOCKS + sb], acc2);
      acc3 = fmaf(xv, s_scale[3 * Q4_K_SUBBLOCKS + sb] * q3 - s_min[3 * Q4_K_SUBBLOCKS + sb], acc3);
      acc4 = fmaf(xv, s_scale[4 * Q4_K_SUBBLOCKS + sb] * q4 - s_min[4 * Q4_K_SUBBLOCKS + sb], acc4);
      acc5 = fmaf(xv, s_scale[5 * Q4_K_SUBBLOCKS + sb] * q5 - s_min[5 * Q4_K_SUBBLOCKS + sb], acc5);
      acc6 = fmaf(xv, s_scale[6 * Q4_K_SUBBLOCKS + sb] * q6 - s_min[6 * Q4_K_SUBBLOCKS + sb], acc6);
      acc7 = fmaf(xv, s_scale[7 * Q4_K_SUBBLOCKS + sb] * q7 - s_min[7 * Q4_K_SUBBLOCKS + sb], acc7);
    }
    __syncthreads();  // protect s_scale/s_min before the next block reloads them
  }

  GGUF_Q4K_PACK8_WAVE_REDUCE(acc0);
  GGUF_Q4K_PACK8_WAVE_REDUCE(acc1);
  GGUF_Q4K_PACK8_WAVE_REDUCE(acc2);
  GGUF_Q4K_PACK8_WAVE_REDUCE(acc3);
  GGUF_Q4K_PACK8_WAVE_REDUCE(acc4);
  GGUF_Q4K_PACK8_WAVE_REDUCE(acc5);
  GGUF_Q4K_PACK8_WAVE_REDUCE(acc6);
  GGUF_Q4K_PACK8_WAVE_REDUCE(acc7);

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
int launch_q4_k_pack8_gemv(
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
      (gguf_q4_k_pack8_gemv_kernel<scalar_in_t, scalar_out_t>),
      dim3(static_cast<unsigned int>(out_packed),
           static_cast<unsigned int>(rows)),
      dim3(128),
      0,
      stream,
      x, qweight, out, rows, in_features, out_features);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q4_k_pack8_gemv_decode_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q4_k_pack8_gemv<uint16_t, uint16_t>(
      x, qweight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q4_k_pack8_gemv_decode_fp16_fp16_out(
    const _Float16* x,
    const uint8_t* qweight,
    _Float16* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q4_k_pack8_gemv<_Float16, _Float16>(
      x, qweight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q4_k_pack8_gemv_decode_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q4_k_pack8_gemv<uint16_t, float>(
      x, qweight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q4_k_pack8_gemv_decode_fp16_f32_out(
    const _Float16* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q4_k_pack8_gemv<_Float16, float>(
      x, qweight, out, rows, in_features, out_features, stream);
}
