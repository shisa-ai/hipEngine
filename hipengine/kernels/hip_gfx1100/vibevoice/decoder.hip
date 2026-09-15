// VibeVoice-TTS acoustic-tokenizer decoder kernels (bf16 storage, fp32
// accumulation). Row-major activations: (rows, channels).
//
// The causal streaming transposed convolution (SConvTranspose1d) with
// kernel K = 2*stride has exactly two contributing taps per output row:
//   out[m] = W[k0]   @ x[j0]     + W[k0+r] @ (j0 == 0 ? prefix : x[j0-1]),
// with j0 = m / r, k0 = m % r; the j0 = 0 tap reads the LAST prefix row
// (global row prefix_rows - 1 of [prefix | x]). Left context rows live in
// a fixed-length zero-padded prefix buffer of K - 1 rows so per-frame
// carries equal a single-pass forward exactly (zeros are no-ops in the
// accumulation). Weights are uploaded tap-major [K][C_in][C_out] so lanes
// over C_out read coalesced.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

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

constexpr int VV_CONVTR_THREADS = 256;

__global__ void vv_convtr_gemm_kernel(
    const uint16_t* __restrict__ prefix,
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ w_t,
    const uint16_t* __restrict__ b,
    uint16_t* __restrict__ out,
    int64_t prefix_rows,
    int64_t rows,
    int64_t rows_out,
    int64_t c_in,
    int64_t c_out,
    int64_t k_len,
    int64_t stride) {
  const int64_t total = rows_out * c_out;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int64_t m = idx / c_out;
  const int64_t n = idx - m * c_out;
  const int64_t j0 = m / stride;
  const int64_t k0 = m - j0 * stride;
  float acc = bf16_bits_to_float(b[n]);
  {
    // tap 1: W[k0] @ x[j0] (j0 is always a local row: 0 <= j0 < rows)
    const int64_t base = j0 * c_in;
    const uint16_t* wbase = w_t + k0 * c_in * c_out + n;
    for (int64_t ci = 0; ci < c_in; ++ci) {
      acc = fmaf(bf16_bits_to_float(x[base + ci]), bf16_bits_to_float(wbase[ci * c_out]), acc);
    }
  }
  {
    // tap 2: W[k0 + stride] @ (j0 == 0 ? prefix[prefix_rows - 1] : x[j0 - 1])
    const bool from_prefix = j0 == 0;
    const uint16_t* src = from_prefix ? prefix : x;
    const int64_t base = (from_prefix ? prefix_rows - 1 : j0 - 1) * c_in;
    const uint16_t* wbase = w_t + (k0 + stride) * c_in * c_out + n;
    for (int64_t ci = 0; ci < c_in; ++ci) {
      acc = fmaf(bf16_bits_to_float(src[base + ci]), bf16_bits_to_float(wbase[ci * c_out]), acc);
    }
  }
  out[m * c_out + n] = float_to_bf16_bits(acc);
}

}  // namespace

extern "C" int hipengine_vv_convtr_gemm_bf16(
    const uint16_t* prefix,
    const uint16_t* x,
    const uint16_t* w_t,
    const uint16_t* b,
    uint16_t* out,
    int64_t prefix_rows,
    int64_t rows,
    int64_t rows_out,
    int64_t c_in,
    int64_t c_out,
    int64_t k_len,
    int64_t stride,
    hipStream_t stream) {
  if (prefix_rows < 0 || rows <= 0 || c_in <= 0 || c_out <= 0 || k_len <= 0 || stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (rows_out != rows * stride) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows_out * c_out;
  const int64_t blocks = (total + VV_CONVTR_THREADS - 1) / VV_CONVTR_THREADS;
  hipLaunchKernelGGL(
      vv_convtr_gemm_kernel, dim3(static_cast<unsigned int>(blocks)), dim3(VV_CONVTR_THREADS),
      0, stream, prefix, x, w_t, b, out, prefix_rows, rows, rows_out, c_in, c_out, k_len, stride);
  return static_cast<int>(hipGetLastError());
}
