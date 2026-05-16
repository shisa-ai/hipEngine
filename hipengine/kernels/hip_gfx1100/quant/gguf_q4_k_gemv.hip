// GGUF block_q4_K GEMV spike.  This kernel consumes raw GGUF Q4_K tensor
// bytes directly and preserves GGML Q4_K scale/min math; it is intentionally
// separate from PARO/AWQ kernels.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;
constexpr int QK_K = 256;
constexpr int Q4_K_BLOCK_BYTES = 144;

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
__device__ inline float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float scalar_to_float<uint16_t>(uint16_t value) {
  return bf16_bits_to_float(value);
}

template <typename out_t>
__device__ inline void store_output(out_t* out, int64_t idx, float value) {
  out[idx] = static_cast<out_t>(value);
}

template <>
__device__ inline void store_output<uint16_t>(uint16_t* out, int64_t idx, float value) {
  out[idx] = float_to_bf16_bits(value);
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

__device__ inline float gguf_q4_k_weight(const uint8_t* row, int k) {
  const int block_idx = k / QK_K;
  const int within = k - block_idx * QK_K;
  const int subblock = within >> 5;
  const int lane = within & 31;
  const uint8_t* block = row + static_cast<int64_t>(block_idx) * Q4_K_BLOCK_BYTES;
  const float d = fp16_bytes_to_float(block);
  const float dmin = fp16_bytes_to_float(block + 2);
  const uint8_t* scales = block + 4;
  const uint8_t* qs = block + 16;
  const int pair = subblock >> 1;
  const uint8_t packed = qs[pair * 32 + lane];
  const uint8_t q = (subblock & 1) ? (packed >> 4) : (packed & 0x0Fu);
  const float scale = static_cast<float>(gguf_q4_k_scale(scales, subblock));
  const float minv = static_cast<float>(gguf_q4_k_min(scales, subblock));
  return d * scale * static_cast<float>(q) - dmin * minv;
}

__device__ inline float reduce_block_sum(float acc) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down(acc, offset);
  }
  __shared__ float wave_sums[8];
  if (lane == 0) {
    wave_sums[wave] = acc;
  }
  __syncthreads();
  float total = 0.0f;
  if (tid == 0) {
    const int waves = (blockDim.x + 31) >> 5;
    for (int i = 0; i < waves; ++i) {
      total += wave_sums[i];
    }
  }
  return total;
}

__device__ inline int pack8_shift_for_lane(int lane) {
  const int packed_pos = (lane & 1) ? (4 + (lane >> 1)) : (lane >> 1);
  return packed_pos * 4;
}

#define GGUF_Q4K_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, r0, r1, r2, r3, r4, r5, r6, r7) \
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

template <typename scalar_t, typename out_t>
__launch_bounds__(256, 2)
__global__ void gguf_q4_k_gemv_out_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features) {
  const int out_col = blockIdx.x;
  const int row = blockIdx.y;
  if (out_col >= out_features || row >= rows) return;

  const int row_bytes = (in_features / QK_K) * Q4_K_BLOCK_BYTES;
  const uint8_t* __restrict__ weight_row =
      qweight + static_cast<int64_t>(out_col) * row_bytes;
  const scalar_t* __restrict__ x_row =
      x + static_cast<int64_t>(row) * in_features;

  float acc = 0.0f;
  for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
    const float xv = scalar_to_float<scalar_t>(x_row[k]);
    const float wv = gguf_q4_k_weight(weight_row, k);
    acc += xv * wv;
  }
  const float total = reduce_block_sum(acc);
  if (threadIdx.x == 0) {
    store_output<out_t>(out, static_cast<int64_t>(row) * out_features + out_col, total);
  }
}

template <typename scalar_t, typename out_t>
int launch_gguf_q4_k_gemv_out(
    const scalar_t* x,
    const uint8_t* qweight,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || out == nullptr || rows <= 0 ||
      in_features <= 0 || out_features <= 0 || (in_features % QK_K) != 0 ||
      (threads != 64 && threads != 128 && threads != 256)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_features), static_cast<unsigned int>(rows));
  gguf_q4_k_gemv_out_kernel<scalar_t, out_t><<<grid, block, 0, stream>>>(
      x,
      qweight,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features));
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, typename out_t>
__launch_bounds__(128, 4)
__global__ void gguf_q4_k_pack8_gemv_out_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ qweight,
    const float* __restrict__ scales,
    const float* __restrict__ mins,
    out_t* __restrict__ out,
    int rows,
    int in_features,
    int out_features,
    int out_packed) {
  const int64_t out_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (out_pack >= out_packed || row >= rows) return;

  const int64_t out_col = out_pack << 3;
  const int64_t x_base = row * static_cast<int64_t>(in_features);
  const int64_t qw_base = out_pack * static_cast<int64_t>(in_features);

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
  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end;
       k_base += vec_stride) {
    const int64_t group32 = k_base >> 5;
    const float* __restrict__ scale_ptr = scales + group32 * out_features + out_col;
    const float* __restrict__ min_ptr = mins + group32 * out_features + out_col;
    const float s0 = scale_ptr[0];
    const float s1 = scale_ptr[1];
    const float s2 = scale_ptr[2];
    const float s3 = scale_ptr[3];
    const float s4 = scale_ptr[4];
    const float s5 = scale_ptr[5];
    const float s6 = scale_ptr[6];
    const float s7 = scale_ptr[7];
    const float m0 = min_ptr[0];
    const float m1 = min_ptr[1];
    const float m2 = min_ptr[2];
    const float m3 = min_ptr[3];
    const float m4 = min_ptr[4];
    const float m5 = min_ptr[5];
    const float m6 = min_ptr[6];
    const float m7 = min_ptr[7];

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xv = scalar_to_float<scalar_t>(x[x_base + k_base + j]);
      const uint32_t pw = static_cast<uint32_t>(qweight[qw_base + k_base + j]);
      const float q0 = static_cast<float>((pw >> pack8_shift_for_lane(0)) & 0x0Fu);
      const float q1 = static_cast<float>((pw >> pack8_shift_for_lane(1)) & 0x0Fu);
      const float q2 = static_cast<float>((pw >> pack8_shift_for_lane(2)) & 0x0Fu);
      const float q3 = static_cast<float>((pw >> pack8_shift_for_lane(3)) & 0x0Fu);
      const float q4 = static_cast<float>((pw >> pack8_shift_for_lane(4)) & 0x0Fu);
      const float q5 = static_cast<float>((pw >> pack8_shift_for_lane(5)) & 0x0Fu);
      const float q6 = static_cast<float>((pw >> pack8_shift_for_lane(6)) & 0x0Fu);
      const float q7 = static_cast<float>((pw >> pack8_shift_for_lane(7)) & 0x0Fu);
      acc0 = fmaf(xv, q0 * s0 - m0, acc0);
      acc1 = fmaf(xv, q1 * s1 - m1, acc1);
      acc2 = fmaf(xv, q2 * s2 - m2, acc2);
      acc3 = fmaf(xv, q3 * s3 - m3, acc3);
      acc4 = fmaf(xv, q4 * s4 - m4, acc4);
      acc5 = fmaf(xv, q5 * s5 - m5, acc5);
      acc6 = fmaf(xv, q6 * s6 - m6, acc6);
      acc7 = fmaf(xv, q7 * s7 - m7, acc7);
    }
  }

  for (int64_t in_col = main_end + tid; in_col < in_features; in_col += blockDim.x) {
    const int64_t group32 = in_col >> 5;
    const uint32_t pw = static_cast<uint32_t>(qweight[qw_base + in_col]);
    const float* __restrict__ scale_ptr = scales + group32 * out_features + out_col;
    const float* __restrict__ min_ptr = mins + group32 * out_features + out_col;
    const float xv = scalar_to_float<scalar_t>(x[x_base + in_col]);
    acc0 = fmaf(xv, static_cast<float>((pw >> pack8_shift_for_lane(0)) & 0x0Fu) * scale_ptr[0] - min_ptr[0], acc0);
    acc1 = fmaf(xv, static_cast<float>((pw >> pack8_shift_for_lane(1)) & 0x0Fu) * scale_ptr[1] - min_ptr[1], acc1);
    acc2 = fmaf(xv, static_cast<float>((pw >> pack8_shift_for_lane(2)) & 0x0Fu) * scale_ptr[2] - min_ptr[2], acc2);
    acc3 = fmaf(xv, static_cast<float>((pw >> pack8_shift_for_lane(3)) & 0x0Fu) * scale_ptr[3] - min_ptr[3], acc3);
    acc4 = fmaf(xv, static_cast<float>((pw >> pack8_shift_for_lane(4)) & 0x0Fu) * scale_ptr[4] - min_ptr[4], acc4);
    acc5 = fmaf(xv, static_cast<float>((pw >> pack8_shift_for_lane(5)) & 0x0Fu) * scale_ptr[5] - min_ptr[5], acc5);
    acc6 = fmaf(xv, static_cast<float>((pw >> pack8_shift_for_lane(6)) & 0x0Fu) * scale_ptr[6] - min_ptr[6], acc6);
    acc7 = fmaf(xv, static_cast<float>((pw >> pack8_shift_for_lane(7)) & 0x0Fu) * scale_ptr[7] - min_ptr[7], acc7);
  }

  GGUF_Q4K_PACK8_REDUCE(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                        r0, r1, r2, r3, r4, r5, r6, r7);
  if (tid == 0) {
    const int64_t out_base = row * static_cast<int64_t>(out_features) + out_col;
    store_output<out_t>(out, out_base + 0, r0);
    store_output<out_t>(out, out_base + 1, r1);
    store_output<out_t>(out, out_base + 2, r2);
    store_output<out_t>(out, out_base + 3, r3);
    store_output<out_t>(out, out_base + 4, r4);
    store_output<out_t>(out, out_base + 5, r5);
    store_output<out_t>(out, out_base + 6, r6);
    store_output<out_t>(out, out_base + 7, r7);
  }
}

template <typename scalar_t, typename out_t>
int launch_gguf_q4_k_pack8_gemv_out(
    const scalar_t* x,
    const int32_t* qweight,
    const float* scales,
    const float* mins,
    out_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  if (x == nullptr || qweight == nullptr || scales == nullptr || mins == nullptr ||
      out == nullptr || rows <= 0 || in_features <= 0 || out_features <= 0 ||
      (in_features % QK_K) != 0 || (out_features % 8) != 0 ||
      (threads != 64 && threads != 128)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed = out_features >> 3;
  const dim3 block(static_cast<unsigned int>(threads));
  const dim3 grid(static_cast<unsigned int>(out_packed), static_cast<unsigned int>(rows));
  gguf_q4_k_pack8_gemv_out_kernel<scalar_t, out_t><<<grid, block, 0, stream>>>(
      x,
      qweight,
      scales,
      mins,
      out,
      static_cast<int>(rows),
      static_cast<int>(in_features),
      static_cast<int>(out_features),
      static_cast<int>(out_packed));
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q4_k_gemv_f32_f32_out(
    const float* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q4_k_gemv_out<float, float>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q4_k_gemv_fp16_f32_out(
    const half_t* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q4_k_gemv_out<half_t, float>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q4_k_gemv_bf16_f32_out(
    const uint16_t* x,
    const uint8_t* qweight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q4_k_gemv_out<uint16_t, float>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q4_k_gemv_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q4_k_gemv_out<uint16_t, uint16_t>(
      x, qweight, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q4_k_pack8_gemv_f32_f32_out(
    const float* x,
    const int32_t* qweight,
    const float* scales,
    const float* mins,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q4_k_pack8_gemv_out<float, float>(
      x, qweight, scales, mins, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q4_k_pack8_gemv_fp16_f32_out(
    const half_t* x,
    const int32_t* qweight,
    const float* scales,
    const float* mins,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q4_k_pack8_gemv_out<half_t, float>(
      x, qweight, scales, mins, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q4_k_pack8_gemv_bf16_f32_out(
    const uint16_t* x,
    const int32_t* qweight,
    const float* scales,
    const float* mins,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q4_k_pack8_gemv_out<uint16_t, float>(
      x, qweight, scales, mins, out, rows, in_features, out_features, threads, stream);
}

extern "C" int hipengine_gguf_q4_k_pack8_gemv_bf16_bf16_out(
    const uint16_t* x,
    const int32_t* qweight,
    const float* scales,
    const float* mins,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t threads,
    hipStream_t stream) {
  return launch_gguf_q4_k_pack8_gemv_out<uint16_t, uint16_t>(
      x, qweight, scales, mins, out, rows, in_features, out_features, threads, stream);
}
