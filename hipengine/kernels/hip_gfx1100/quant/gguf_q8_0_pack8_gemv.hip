// Dense decode-shaped pack8 GEMV for raw GGUF Q8_0 weights.
//
// P9.B3 implementation note: mirrors the structure of
// ``paro_awq_gemv.hip::gemv_awq_pack8_kernel`` (single output) and
// ``gemv_awq_dual_pack8_kernel`` (concatenated gate+up dual) for the AWQ
// PARO path, but with the inner k loop swapped for raw GGUF Q8_0 dequant
// (``d * int8``). Q8_0 has a tiny block size (32 K's), so each
// ``tid * 8 + j`` octet lands in a single block; the block's ``d`` is loaded
// once per iteration and amortised across the 8 inner ``j`` lanes.
//
//   x          [rows, in_features]
//   qweight    [out_features, row_bytes]   raw block_q8_0 bytes
//   out        [rows, out_features]
//
// The dual variant takes ``qweight_a`` (gate) and ``qweight_b`` (up) and
// emits a row-major concatenated output ``[rows, out_features_a + out_features_b]``
// (gate occupies columns ``[0, out_features_a)``; up occupies the rest), so
// the runtime can feed ``silu_mul_dual_out_*`` without any extra copy.
//
// No new ABI and no resident weight sidecar/repack: raw GGUF Q8_0 bytes
// stay on device.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

using half_t = _Float16;

constexpr int Q8_0_BLOCK = 32;
constexpr int Q8_0_BLOCK_BYTES = 34;  // fp16 d + 32 int8 q
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

#define GGUF_Q8_PACK8_WAVE_REDUCE(acc)                       \
  do {                                                       \
    acc += __shfl_down(acc, 16);                             \
    acc += __shfl_down(acc, 8);                              \
    acc += __shfl_down(acc, 4);                              \
    acc += __shfl_down(acc, 2);                              \
    acc += __shfl_down(acc, 1);                              \
  } while (0)

// Accumulate ``q8_0`` weighted dot-product for 8 output channels of one pack
// over the entire ``in_features`` axis. The 8 ``row_ptr`` entries point at
// the start of each output channel's raw Q8_0 row bytes.
template <typename scalar_t>
__device__ inline void q8_0_pack8_accumulate(
    const scalar_t* __restrict__ x_row,
    const uint8_t* __restrict__ row_ptr0,
    const uint8_t* __restrict__ row_ptr1,
    const uint8_t* __restrict__ row_ptr2,
    const uint8_t* __restrict__ row_ptr3,
    const uint8_t* __restrict__ row_ptr4,
    const uint8_t* __restrict__ row_ptr5,
    const uint8_t* __restrict__ row_ptr6,
    const uint8_t* __restrict__ row_ptr7,
    int64_t in_features,
    int tid,
    int block_dim,
    float& acc0,
    float& acc1,
    float& acc2,
    float& acc3,
    float& acc4,
    float& acc5,
    float& acc6,
    float& acc7) {
  const int64_t vec_stride = static_cast<int64_t>(block_dim) * 8;
  const int64_t main_end = (in_features / vec_stride) * vec_stride;

  for (int64_t k_base = static_cast<int64_t>(tid) * 8; k_base < main_end;
       k_base += vec_stride) {
    // All 8 ``j`` lanes share one Q8_0 block; hoist ``d`` once per pack.
    const int64_t block_idx = k_base / Q8_0_BLOCK;
    const int lane_base = static_cast<int>(k_base & (Q8_0_BLOCK - 1));

    const uint8_t* __restrict__ b0 = row_ptr0 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b1 = row_ptr1 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b2 = row_ptr2 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b3 = row_ptr3 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b4 = row_ptr4 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b5 = row_ptr5 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b6 = row_ptr6 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b7 = row_ptr7 + block_idx * Q8_0_BLOCK_BYTES;

    const float d0 = fp16_bytes_to_float(b0);
    const float d1 = fp16_bytes_to_float(b1);
    const float d2 = fp16_bytes_to_float(b2);
    const float d3 = fp16_bytes_to_float(b3);
    const float d4 = fp16_bytes_to_float(b4);
    const float d5 = fp16_bytes_to_float(b5);
    const float d6 = fp16_bytes_to_float(b6);
    const float d7 = fp16_bytes_to_float(b7);

    const int8_t* __restrict__ qs0 = reinterpret_cast<const int8_t*>(b0 + 2);
    const int8_t* __restrict__ qs1 = reinterpret_cast<const int8_t*>(b1 + 2);
    const int8_t* __restrict__ qs2 = reinterpret_cast<const int8_t*>(b2 + 2);
    const int8_t* __restrict__ qs3 = reinterpret_cast<const int8_t*>(b3 + 2);
    const int8_t* __restrict__ qs4 = reinterpret_cast<const int8_t*>(b4 + 2);
    const int8_t* __restrict__ qs5 = reinterpret_cast<const int8_t*>(b5 + 2);
    const int8_t* __restrict__ qs6 = reinterpret_cast<const int8_t*>(b6 + 2);
    const int8_t* __restrict__ qs7 = reinterpret_cast<const int8_t*>(b7 + 2);

#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int lane = lane_base + j;
      const float xv = scalar_to_float<scalar_t>(x_row[k_base + j]);
      acc0 = fmaf(xv, d0 * static_cast<float>(qs0[lane]), acc0);
      acc1 = fmaf(xv, d1 * static_cast<float>(qs1[lane]), acc1);
      acc2 = fmaf(xv, d2 * static_cast<float>(qs2[lane]), acc2);
      acc3 = fmaf(xv, d3 * static_cast<float>(qs3[lane]), acc3);
      acc4 = fmaf(xv, d4 * static_cast<float>(qs4[lane]), acc4);
      acc5 = fmaf(xv, d5 * static_cast<float>(qs5[lane]), acc5);
      acc6 = fmaf(xv, d6 * static_cast<float>(qs6[lane]), acc6);
      acc7 = fmaf(xv, d7 * static_cast<float>(qs7[lane]), acc7);
    }
  }

  // Tail (in_features < vec_stride * iters): 1 element per thread.
  for (int64_t k = main_end + tid; k < in_features; k += block_dim) {
    const int64_t block_idx = k / Q8_0_BLOCK;
    const int lane = static_cast<int>(k & (Q8_0_BLOCK - 1));
    const uint8_t* __restrict__ b0 = row_ptr0 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b1 = row_ptr1 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b2 = row_ptr2 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b3 = row_ptr3 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b4 = row_ptr4 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b5 = row_ptr5 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b6 = row_ptr6 + block_idx * Q8_0_BLOCK_BYTES;
    const uint8_t* __restrict__ b7 = row_ptr7 + block_idx * Q8_0_BLOCK_BYTES;
    const float xv = scalar_to_float<scalar_t>(x_row[k]);
    acc0 = fmaf(xv,
                fp16_bytes_to_float(b0) *
                    static_cast<float>(reinterpret_cast<const int8_t*>(b0 + 2)[lane]),
                acc0);
    acc1 = fmaf(xv,
                fp16_bytes_to_float(b1) *
                    static_cast<float>(reinterpret_cast<const int8_t*>(b1 + 2)[lane]),
                acc1);
    acc2 = fmaf(xv,
                fp16_bytes_to_float(b2) *
                    static_cast<float>(reinterpret_cast<const int8_t*>(b2 + 2)[lane]),
                acc2);
    acc3 = fmaf(xv,
                fp16_bytes_to_float(b3) *
                    static_cast<float>(reinterpret_cast<const int8_t*>(b3 + 2)[lane]),
                acc3);
    acc4 = fmaf(xv,
                fp16_bytes_to_float(b4) *
                    static_cast<float>(reinterpret_cast<const int8_t*>(b4 + 2)[lane]),
                acc4);
    acc5 = fmaf(xv,
                fp16_bytes_to_float(b5) *
                    static_cast<float>(reinterpret_cast<const int8_t*>(b5 + 2)[lane]),
                acc5);
    acc6 = fmaf(xv,
                fp16_bytes_to_float(b6) *
                    static_cast<float>(reinterpret_cast<const int8_t*>(b6 + 2)[lane]),
                acc6);
    acc7 = fmaf(xv,
                fp16_bytes_to_float(b7) *
                    static_cast<float>(reinterpret_cast<const int8_t*>(b7 + 2)[lane]),
                acc7);
  }
}

template <typename scalar_t>
__device__ inline void q8_0_pack8_reduce_and_store(
    scalar_t* __restrict__ out_ptr,
    int tid,
    int block_dim,
    float acc0,
    float acc1,
    float acc2,
    float acc3,
    float acc4,
    float acc5,
    float acc6,
    float acc7) {
  GGUF_Q8_PACK8_WAVE_REDUCE(acc0);
  GGUF_Q8_PACK8_WAVE_REDUCE(acc1);
  GGUF_Q8_PACK8_WAVE_REDUCE(acc2);
  GGUF_Q8_PACK8_WAVE_REDUCE(acc3);
  GGUF_Q8_PACK8_WAVE_REDUCE(acc4);
  GGUF_Q8_PACK8_WAVE_REDUCE(acc5);
  GGUF_Q8_PACK8_WAVE_REDUCE(acc6);
  GGUF_Q8_PACK8_WAVE_REDUCE(acc7);

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
    const int sub_groups = block_dim >> 5;
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

// Dense Q8_0 pack8 GEMV decode.
//
// Grid: (out_features / 8, rows). One block per (output pack of 8, row).
template <typename scalar_t>
__launch_bounds__(128, 4) __global__ void gguf_q8_0_pack8_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  const int64_t out_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t out_col = out_pack << 3;
  if (out_col >= out_features) return;

  const int64_t row_bytes = (in_features / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
  const uint8_t* __restrict__ row_ptr0 = qweight + (out_col + 0) * row_bytes;
  const uint8_t* __restrict__ row_ptr1 = qweight + (out_col + 1) * row_bytes;
  const uint8_t* __restrict__ row_ptr2 = qweight + (out_col + 2) * row_bytes;
  const uint8_t* __restrict__ row_ptr3 = qweight + (out_col + 3) * row_bytes;
  const uint8_t* __restrict__ row_ptr4 = qweight + (out_col + 4) * row_bytes;
  const uint8_t* __restrict__ row_ptr5 = qweight + (out_col + 5) * row_bytes;
  const uint8_t* __restrict__ row_ptr6 = qweight + (out_col + 6) * row_bytes;
  const uint8_t* __restrict__ row_ptr7 = qweight + (out_col + 7) * row_bytes;

  const scalar_t* __restrict__ x_row = x + row * in_features;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

  q8_0_pack8_accumulate<scalar_t>(
      x_row,
      row_ptr0, row_ptr1, row_ptr2, row_ptr3,
      row_ptr4, row_ptr5, row_ptr6, row_ptr7,
      in_features, tid, blockDim.x,
      acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  scalar_t* __restrict__ out_ptr = out + row * out_features + out_col;
  q8_0_pack8_reduce_and_store<scalar_t>(
      out_ptr, tid, blockDim.x,
      acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
}

// Dense Q8_0 dual gate+up pack8 GEMV decode. ``out_features_a`` columns of
// gate followed by ``out_features_b`` columns of up, in row-major order.
template <typename scalar_t>
__launch_bounds__(128, 4) __global__ void gguf_q8_0_pack8_dual_gate_up_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight_a,
    const uint8_t* __restrict__ qweight_b,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b) {
  const int64_t global_pack = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= rows) return;

  const int64_t out_packed_a = out_features_a >> 3;
  const int64_t out_packed_b = out_features_b >> 3;
  const bool use_b = global_pack >= out_packed_a;
  const int64_t local_pack = use_b ? (global_pack - out_packed_a) : global_pack;
  const int64_t local_out_features = use_b ? out_features_b : out_features_a;
  const int64_t out_features_total = out_features_a + out_features_b;
  const int64_t local_out_col = local_pack << 3;
  const int64_t out_col = global_pack << 3;
  if (out_col >= out_features_total) return;

  const int64_t row_bytes = (in_features / Q8_0_BLOCK) * Q8_0_BLOCK_BYTES;
  const uint8_t* __restrict__ qweight = use_b ? qweight_b : qweight_a;
  const uint8_t* __restrict__ row_ptr0 = qweight + (local_out_col + 0) * row_bytes;
  const uint8_t* __restrict__ row_ptr1 = qweight + (local_out_col + 1) * row_bytes;
  const uint8_t* __restrict__ row_ptr2 = qweight + (local_out_col + 2) * row_bytes;
  const uint8_t* __restrict__ row_ptr3 = qweight + (local_out_col + 3) * row_bytes;
  const uint8_t* __restrict__ row_ptr4 = qweight + (local_out_col + 4) * row_bytes;
  const uint8_t* __restrict__ row_ptr5 = qweight + (local_out_col + 5) * row_bytes;
  const uint8_t* __restrict__ row_ptr6 = qweight + (local_out_col + 6) * row_bytes;
  const uint8_t* __restrict__ row_ptr7 = qweight + (local_out_col + 7) * row_bytes;

  const scalar_t* __restrict__ x_row = x + row * in_features;

  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

  q8_0_pack8_accumulate<scalar_t>(
      x_row,
      row_ptr0, row_ptr1, row_ptr2, row_ptr3,
      row_ptr4, row_ptr5, row_ptr6, row_ptr7,
      in_features, tid, blockDim.x,
      acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

  scalar_t* __restrict__ out_ptr = out + row * out_features_total + out_col;
  q8_0_pack8_reduce_and_store<scalar_t>(
      out_ptr, tid, blockDim.x,
      acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
}

bool valid_q8_0_pack8_single(
    const void* x,
    const void* qweight,
    const void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features) {
  return x != nullptr && qweight != nullptr && out != nullptr &&
      rows > 0 && in_features > 0 && out_features > 0 &&
      (out_features & 7) == 0 && (in_features % Q8_0_BLOCK) == 0;
}

bool valid_q8_0_pack8_dual(
    const void* x,
    const void* qweight_a,
    const void* qweight_b,
    const void* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b) {
  return x != nullptr && qweight_a != nullptr && qweight_b != nullptr &&
      out != nullptr && rows > 0 && in_features > 0 &&
      out_features_a > 0 && out_features_b > 0 &&
      (out_features_a & 7) == 0 && (out_features_b & 7) == 0 &&
      (in_features % Q8_0_BLOCK) == 0;
}

template <typename scalar_t>
int launch_q8_0_pack8_single(
    const scalar_t* x,
    const uint8_t* qweight,
    scalar_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (!valid_q8_0_pack8_single(x, qweight, out, rows, in_features, out_features)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed = out_features >> 3;
  hipLaunchKernelGGL(
      (gguf_q8_0_pack8_gemv_kernel<scalar_t>),
      dim3(static_cast<unsigned int>(out_packed),
           static_cast<unsigned int>(rows)),
      dim3(128),
      0,
      stream,
      x, qweight, out, rows, in_features, out_features);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t>
int launch_q8_0_pack8_dual(
    const scalar_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    scalar_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    hipStream_t stream) {
  if (!valid_q8_0_pack8_dual(
          x, qweight_a, qweight_b, out, rows, in_features,
          out_features_a, out_features_b)) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t out_packed_total = (out_features_a + out_features_b) >> 3;
  hipLaunchKernelGGL(
      (gguf_q8_0_pack8_dual_gate_up_gemv_kernel<scalar_t>),
      dim3(static_cast<unsigned int>(out_packed_total),
           static_cast<unsigned int>(rows)),
      dim3(128),
      0,
      stream,
      x, qweight_a, qweight_b, out, rows, in_features,
      out_features_a, out_features_b);
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int hipengine_gguf_q8_0_pack8_gemv_decode_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q8_0_pack8_single<uint16_t>(x, qweight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q8_0_pack8_gemv_decode_fp16_fp16_out(
    const _Float16* x,
    const uint8_t* qweight,
    _Float16* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  return launch_q8_0_pack8_single<_Float16>(x, qweight, out, rows, in_features, out_features, stream);
}

extern "C" int hipengine_gguf_q8_0_pack8_dual_gate_up_gemv_decode_bf16_bf16_out(
    const uint16_t* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    uint16_t* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    hipStream_t stream) {
  return launch_q8_0_pack8_dual<uint16_t>(
      x, qweight_a, qweight_b, out, rows, in_features, out_features_a, out_features_b, stream);
}

extern "C" int hipengine_gguf_q8_0_pack8_dual_gate_up_gemv_decode_fp16_fp16_out(
    const _Float16* x,
    const uint8_t* qweight_a,
    const uint8_t* qweight_b,
    _Float16* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features_a,
    int64_t out_features_b,
    hipStream_t stream) {
  return launch_q8_0_pack8_dual<_Float16>(
      x, qweight_a, qweight_b, out, rows, in_features, out_features_a, out_features_b, stream);
}
