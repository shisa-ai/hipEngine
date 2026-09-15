// Roofline probe for gfx11 wave32: practical FP32 FMA rate, practical stream
// bandwidth, and the achieved rate of a known instruction mix.
//
// The dense GGUF projection this project is optimizing runs at a small fraction
// of the datasheet FP32 rate. Deciding whether that fraction is headroom or a
// hardware limit needs the hardware's actual ceiling measured on the same part,
// not read off a spec sheet: an APU sharing a power budget with its CPU can sit
// well under the theoretical rate.
//
// Three kernels, each measuring one bound:
//
//   fma_peak   register-resident FMA only. No memory traffic in the loop, so it
//              measures the FP32 issue rate. This is the denominator that
//              "percent of peak" should use.
//   mixed      a fixed ratio of FMA to integer instructions per iteration, so
//              the issue-mix ceiling predicted from a kernel's disassembly can
//              be checked against a measured kernel with that same mix.
//   stream     a float4 copy, measuring achievable memory bandwidth.
//
// All three take their iteration count at runtime so launch overhead is
// amortized rather than measured.

#include <hip/hip_runtime.h>
#include <cstdint>

#define ROOFLINE_ACC 16

// Pure FP32 FMA. ROOFLINE_ACC independent dependency chains, which is enough to
// cover the FMA latency on wave32.
extern "C" __global__ void roofline_fma_peak_kernel(
    float* __restrict__ out, int64_t iters) {
  float a[ROOFLINE_ACC];
#pragma unroll
  for (int i = 0; i < ROOFLINE_ACC; ++i) {
    a[i] = 1.0f + 1.0e-7f * static_cast<float>(threadIdx.x + i);
  }
  const float m = 0.9999999f;
  const float c = 1.0e-9f;
  for (int64_t it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < ROOFLINE_ACC; ++i) {
      a[i] = fmaf(a[i], m, c);
    }
  }
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < ROOFLINE_ACC; ++i) {
    s += a[i];
  }
  out[static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x] = s;
}

// FMA plus integer work at a fixed ratio. The integer chain feeds a value that is
// only stored when the caller passes a non-null side pointer, so it cannot be
// eliminated but also never runs in the measured configuration.
extern "C" __global__ void roofline_mixed_fma40_int105_kernel(
    float* __restrict__ out, int* __restrict__ side, int64_t iters) {
  float a[ROOFLINE_ACC];
#pragma unroll
  for (int i = 0; i < ROOFLINE_ACC; ++i) {
    a[i] = 1.0f + 1.0e-7f * static_cast<float>(threadIdx.x + i);
  }
  const float m = 0.9999999f;
  const float c = 1.0e-9f;
  uint32_t q0 = static_cast<uint32_t>(threadIdx.x) * 2654435761u + 1u;
  uint32_t q1 = q0 ^ 0x9e3779b9u;
  uint32_t q2 = q1 + 0x85ebca6bu;
  uint32_t q3 = q2 ^ 0xc2b2ae35u;
  for (int64_t it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < ROOFLINE_ACC * 2; ++i) {
      a[i & (ROOFLINE_ACC - 1)] = fmaf(a[i & (ROOFLINE_ACC - 1)], m, c);
    }
    // 105 integer ops per 32 FMAs, matching the measured issue mix of the dense
    // GGUF projection's inner loop.
#pragma unroll
    for (int i = 0; i < 105; ++i) {
      switch (i & 3) {
        case 0: q0 = q0 * 1664525u + 1013904223u; break;
        case 1: q1 = q1 * 1664525u + 1013904223u; break;
        case 2: q2 = q2 * 1664525u + 1013904223u; break;
        default: q3 = q3 * 1664525u + 1013904223u; break;
      }
    }
  }
  if (side != nullptr) {
    side[static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x] =
        static_cast<int>(q0 ^ q1 ^ q2 ^ q3);
  }
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < ROOFLINE_ACC; ++i) {
    s += a[i];
  }
  out[static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x] = s;
}

// Streaming copy, one float4 per thread per iteration.
extern "C" __global__ void roofline_stream_kernel(
    float4* __restrict__ dst, const float4* __restrict__ src, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    dst[i] = src[i];
  }
}

// ---------------------------------------------------------------------------
// Host-side launch wrappers. Same C ABI shape as the other gfx1100 kernels: raw
// pointers plus a stream, returning a HIP error code.
// ---------------------------------------------------------------------------

extern "C" int hipengine_roofline_fma_peak(
    float* out, int64_t iters, int64_t blocks, int64_t threads,
    hipStream_t stream) {
  if (out == nullptr || iters <= 0 || blocks <= 0 || threads <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      roofline_fma_peak_kernel, dim3(static_cast<unsigned int>(blocks)),
      dim3(static_cast<unsigned int>(threads)), 0, stream, out, iters);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_roofline_mixed_fma40_int105(
    float* out, int* side, int64_t iters, int64_t blocks, int64_t threads,
    hipStream_t stream) {
  if (out == nullptr || iters <= 0 || blocks <= 0 || threads <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      roofline_mixed_fma40_int105_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(static_cast<unsigned int>(threads)), 0, stream, out, side, iters);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_roofline_stream(
    float* dst, const float* src, int64_t n_float4, int64_t blocks,
    int64_t threads, hipStream_t stream) {
  if (dst == nullptr || src == nullptr || n_float4 <= 0 || blocks <= 0 ||
      threads <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      roofline_stream_kernel, dim3(static_cast<unsigned int>(blocks)),
      dim3(static_cast<unsigned int>(threads)), 0, stream,
      reinterpret_cast<float4*>(dst), reinterpret_cast<const float4*>(src),
      n_float4);
  return static_cast<int>(hipGetLastError());
}
