// VibeVoice-TTS diffusion head elementwise kernels (bf16 storage).
//
// The head is an MLP-DiT: every linear routes through dense_gemv; these
// kernels cover the elementwise ops with the fork's exact eager rounding --
// one bf16 round per binary op, in the same order.
//
// Kernels:
//   rmsnorm         two-rounding RMSNorm: normalized -> bf16, then the
//                   * weight product -> bf16 (w == nullptr skips the second
//                   rounding: the final layer's no-affine norm)
//   silu            r(g * sigmoid(g))
//   add             r(a + b)
//   modulate        r(r(x * r(1 + s)) + sh)   (adaLN; three roundings)
//   gated_residual  r(h + r(gate * y))        (HeadLayer out)
//   cfg_combine     r(u + r(cfg * (c - u)))  (two-branch CFG; cfg fp32 host
//                                              scalar, never pre-rounded, and
//                                              the (c - u) difference stays in
//                                              fp32 -- only the product and the
//                                              final sum round)
//
// The adaLN linear emits (rows, 3*H) / (rows, 2*H) chunks; shift/scale/gate
// live at column offsets inside each source row, so those kernels take an
// explicit source-row stride (in elements). x/y/h/out are contiguous
// (rows, H).

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

__device__ inline uint16_t rbf(float value) { return float_to_bf16_bits(value); }

constexpr int VV_DIFF_WARPS = 8;  // 256 threads, warp-per-row

__global__ void vv_diff_rmsnorm_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    uint16_t* __restrict__ out, int64_t rows, int64_t hidden, float eps) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = (int)blockDim.x >> 5;
  for (int64_t row = blockIdx.x * warps_per_block + warp;
       row < rows;
       row += (int64_t)gridDim.x * warps_per_block) {
    const int64_t base = row * hidden;
    float sumsq = 0.0f;
    for (int64_t i = lane; i < hidden; i += 32) {
      const float v = bf16_bits_to_float(x[base + i]);
      sumsq = fmaf(v, v, sumsq);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sumsq += __shfl_xor(sumsq, offset);
    }
    const float inv = rsqrtf(sumsq / static_cast<float>(hidden) + eps);
    for (int64_t i = lane; i < hidden; i += 32) {
      const uint16_t normed = rbf(bf16_bits_to_float(x[base + i]) * inv);
      if (w == nullptr) {
        out[base + i] = normed;
      } else {
        out[base + i] = rbf(bf16_bits_to_float(normed) * bf16_bits_to_float(w[i]));
      }
    }
  }
}

__global__ void vv_diff_rmsnorm_modulate_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ chunk, uint16_t* __restrict__ out,
    int64_t rows, int64_t hidden, int64_t src_stride, float eps) {
  // One launch = one rmsnorm + one modulate, with the identical rounding
  // sequence as the unfused pair: rbf(normed) (optionally rbf(normed*w))
  // feeds rbf(x*(1+cv)) + sv on the bf16 grid.
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t warps_per_block = blockDim.x >> 5;
  for (int64_t row = blockIdx.x * warps_per_block + warp; row < rows;
       row += (int64_t)gridDim.x * warps_per_block) {
    const int64_t base = row * hidden;
    float sumsq = 0.0f;
    for (int64_t i = lane; i < hidden; i += 32) {
      const float v = bf16_bits_to_float(x[base + i]);
      sumsq = fmaf(v, v, sumsq);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sumsq += __shfl_xor(sumsq, offset);
    }
    const float inv = rsqrtf(sumsq / static_cast<float>(hidden) + eps);
    const int64_t chunk_base = row * src_stride;
    for (int64_t i = lane; i < hidden; i += 32) {
      const uint16_t normed_bits = rbf(bf16_bits_to_float(x[base + i]) * inv);
      const float normed = bf16_bits_to_float(normed_bits);
      const uint16_t normed_out_bits =
          (w == nullptr) ? normed_bits
                         : rbf(normed * bf16_bits_to_float(w[i]));
      const float xv = bf16_bits_to_float(normed_out_bits);
      const float sv = bf16_bits_to_float(chunk[chunk_base + i]);
      const float cv = bf16_bits_to_float(chunk[chunk_base + hidden + i]);
      const uint16_t one_plus = rbf(1.0f + cv);
      const uint16_t scaled = rbf(xv * bf16_bits_to_float(one_plus));
      out[base + i] = rbf(bf16_bits_to_float(scaled) + sv);
    }
  }
}

__global__ void vv_diff_silu_kernel(
    const uint16_t* __restrict__ g, uint16_t* __restrict__ out, int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float v = bf16_bits_to_float(g[idx]);
    const float sig = 1.0f / (1.0f + expf(-v));
    out[idx] = rbf(v * sig);
  }
}

__global__ void vv_diff_add_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ b,
    uint16_t* __restrict__ out, int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = rbf(bf16_bits_to_float(a[idx]) + bf16_bits_to_float(b[idx]));
  }
}

__global__ void vv_diff_modulate_kernel_impl(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ chunk,
    uint16_t* __restrict__ out, int64_t rows, int64_t width, int64_t src_stride) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < rows * width) {
    const int64_t row = idx / width;
    const int64_t col = idx - row * width;
    const int64_t src = row * src_stride + col;
    const float xv = bf16_bits_to_float(x[idx]);
    const float sv = bf16_bits_to_float(chunk[src]);
    const float cv = bf16_bits_to_float(chunk[src + width]);
    // rbf() returns bf16 BITS: round-trip through uint16 and reinterpret.
    const uint16_t one_plus = rbf(1.0f + cv);
    const uint16_t scaled = rbf(xv * bf16_bits_to_float(one_plus));
    out[idx] = rbf(bf16_bits_to_float(scaled) + sv);
  }
}

__global__ void vv_diff_gated_residual_kernel(
    const uint16_t* __restrict__ h, const uint16_t* __restrict__ gate_chunk,
    const uint16_t* __restrict__ y, uint16_t* __restrict__ out,
    int64_t rows, int64_t width, int64_t gate_stride) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < rows * width) {
    const int64_t row = idx / width;
    const int64_t col = idx - row * width;
    const uint16_t gated = rbf(
        bf16_bits_to_float(gate_chunk[row * gate_stride + col]) * bf16_bits_to_float(y[idx]));
    out[idx] = rbf(bf16_bits_to_float(h[idx]) + bf16_bits_to_float(gated));
  }
}

__global__ void vv_diff_cfg_combine_kernel(
    const uint16_t* __restrict__ c, const uint16_t* __restrict__ u, float cfg,
    uint16_t* __restrict__ out, int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float c32 = bf16_bits_to_float(c[idx]);
    const float u32 = bf16_bits_to_float(u[idx]);
    // The difference is deliberately left in fp32: the sampling oracle is
    // bf16_round(u + bf16_round(cfg * (c - u))), two roundings, not three.
    const uint16_t scaled = rbf(cfg * (c32 - u32));
    out[idx] = rbf(u32 + bf16_bits_to_float(scaled));
  }
}

__global__ void vv_diff_dpm_step_kernel(
    const uint16_t* __restrict__ sample, const uint16_t* __restrict__ eps,
    const uint16_t* __restrict__ x0_prev, uint16_t* __restrict__ sample_out,
    uint16_t* __restrict__ x0_out, float alpha_b, float sigma_b, float scale,
    float coef_b, float half_coef_b, float inv_r0_b, int64_t n, int order) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const float s = bf16_bits_to_float(sample[idx]);
  const float e = bf16_bits_to_float(eps[idx]);
  // x0 = r(r(alpha_b * sample) - r(sigma_b * eps))
  const float t1 = bf16_bits_to_float(rbf(alpha_b * s));
  const float t2 = bf16_bits_to_float(rbf(sigma_b * e));
  const uint16_t x0 = rbf(t1 - t2);
  const float m0 = bf16_bits_to_float(x0);
  // term0 is a true FP32 product (scale is not bf16-rounded); term1/term2 round.
  float prev = scale * s - bf16_bits_to_float(rbf(coef_b * m0));
  // order == 1 (first order) uses only coef_b; order == 2 adds the midpoint term.
  if (order > 1) {
    const float m1 = bf16_bits_to_float(x0_prev[idx]);
    const float d1 = bf16_bits_to_float(rbf(inv_r0_b * bf16_bits_to_float(rbf(m0 - m1))));
    prev -= bf16_bits_to_float(rbf(half_coef_b * d1));
  }
  sample_out[idx] = rbf(prev);
  x0_out[idx] = x0;
}

}  // namespace

extern "C" int hipengine_vv_diff_dpm_step_bf16(
    const uint16_t* sample, const uint16_t* eps, const uint16_t* x0_prev,
    uint16_t* sample_out, uint16_t* x0_out, float alpha_b, float sigma_b,
    float scale, float coef_b, float half_coef_b, float inv_r0_b, int64_t n,
    int order, hipStream_t stream) {
  if (n <= 0) return static_cast<int>(hipErrorInvalidValue);
  const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
  hipLaunchKernelGGL(vv_diff_dpm_step_kernel, dim3(blocks), dim3(256), 0, stream,
                     sample, eps, x0_prev, sample_out, x0_out, alpha_b, sigma_b,
                     scale, coef_b, half_coef_b, inv_r0_b, n, order);
  return static_cast<int>(hipGetLastError());
}


extern "C" int hipengine_vv_diff_rmsnorm_bf16(
    const uint16_t* x, const uint16_t* w, uint16_t* out, int64_t rows,
    int64_t hidden, float eps, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) return static_cast<int>(hipErrorInvalidValue);
  const unsigned blocks = static_cast<unsigned>((rows + VV_DIFF_WARPS - 1) / VV_DIFF_WARPS);
  hipLaunchKernelGGL(vv_diff_rmsnorm_kernel, dim3(blocks), dim3(VV_DIFF_WARPS * 32),
                     0, stream, x, w, out, rows, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_diff_rmsnorm_modulate_bf16(
    const uint16_t* x, const uint16_t* w, const uint16_t* chunk,
    uint16_t* out, int64_t rows, int64_t hidden, int64_t src_stride,
    float eps, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0 || src_stride < 2 * hidden)
    return static_cast<int>(hipErrorInvalidValue);
  const unsigned blocks = static_cast<unsigned>((rows + VV_DIFF_WARPS - 1) / VV_DIFF_WARPS);
  hipLaunchKernelGGL(vv_diff_rmsnorm_modulate_kernel,
                     dim3(blocks), dim3(VV_DIFF_WARPS * 32), 0, stream,
                     x, w, chunk, out, rows, hidden, src_stride, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_diff_silu_bf16(
    const uint16_t* g, uint16_t* out, int64_t n, hipStream_t stream) {
  if (n <= 0) return static_cast<int>(hipErrorInvalidValue);
  hipLaunchKernelGGL(vv_diff_silu_kernel, dim3(static_cast<unsigned>((n + 255) / 256)),
                     dim3(256), 0, stream, g, out, n);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_diff_add_bf16(
    const uint16_t* a, const uint16_t* b, uint16_t* out, int64_t n, hipStream_t stream) {
  if (n <= 0) return static_cast<int>(hipErrorInvalidValue);
  hipLaunchKernelGGL(vv_diff_add_kernel, dim3(static_cast<unsigned>((n + 255) / 256)),
                     dim3(256), 0, stream, a, b, out, n);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_diff_modulate_bf16(
    const uint16_t* x, const uint16_t* chunk, uint16_t* out, int64_t rows,
    int64_t width, int64_t src_stride, hipStream_t stream) {
  if (rows <= 0 || width <= 0 || src_stride < 2 * width)
    return static_cast<int>(hipErrorInvalidValue);
  const int64_t n = rows * width;
  hipLaunchKernelGGL(vv_diff_modulate_kernel_impl,
                     dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256),
                     0, stream, x, chunk, out, rows, width, src_stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_diff_gated_residual_bf16(
    const uint16_t* h, const uint16_t* gate_chunk, const uint16_t* y,
    uint16_t* out, int64_t rows, int64_t width, int64_t gate_stride,
    hipStream_t stream) {
  if (rows <= 0 || width <= 0 || gate_stride < width)
    return static_cast<int>(hipErrorInvalidValue);
  const int64_t n = rows * width;
  hipLaunchKernelGGL(vv_diff_gated_residual_kernel,
                     dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256),
                     0, stream, h, gate_chunk, y, out, rows, width, gate_stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_diff_cfg_combine_bf16(
    const uint16_t* c, const uint16_t* u, float cfg, uint16_t* out,
    int64_t n, hipStream_t stream) {
  if (n <= 0) return static_cast<int>(hipErrorInvalidValue);
  hipLaunchKernelGGL(vv_diff_cfg_combine_kernel,
                     dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256),
                     0, stream, c, u, cfg, out, n);
  return static_cast<int>(hipGetLastError());
}
