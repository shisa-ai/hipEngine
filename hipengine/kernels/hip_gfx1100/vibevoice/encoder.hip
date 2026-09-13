// VibeVoice-ASR audio front-end kernels (bf16 storage, fp32 accumulation).
//
// Layouts: activations are row-major (rows = positions/tokens, cols =
// channels) bf16. Causal conv kernels take an explicit prefix buffer holding
// the previous chunk's tail rows so chunk-carry equals single-pass exactly.
// Strided pointwise conv weights are transposed at upload to [K][C_in][C_out]
// so gemm-style lanes over C_out read coalesced.

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

// Plain-weight RMSNorm (T5-style: no +1 offset), fp32 statistics.
__global__ void vv_rmsnorm_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ w,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t hidden,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * hidden;
  float sumsq = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = bf16_bits_to_float(x[base + i]);
    sumsq = fmaf(v, v, sumsq);
  }
  __shared__ float partial[256];
  partial[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = bf16_bits_to_float(x[base + i]);
    out[base + i] = float_to_bf16_bits(v * inv * bf16_bits_to_float(w[i]));
  }
}

__global__ void vv_gelu_kernel(
    const uint16_t* __restrict__ x,
    uint16_t* __restrict__ out,
    int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float v = bf16_bits_to_float(x[idx]);
    out[idx] = float_to_bf16_bits(0.5f * v * (1.0f + erff(v * 0.7071067811865476f)));
  }
}

// out = x + y * gamma[col]   (ConvNeXt FFN epilogue; row-major rows x width)
__global__ void vv_scale_residual_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ y,
    const uint16_t* __restrict__ gamma,
    uint16_t* __restrict__ out,
    int64_t total,
    int64_t width) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    const int64_t col = idx % width;
    const float xv = bf16_bits_to_float(x[idx]);
    const float yv = bf16_bits_to_float(y[idx]);
    out[idx] = float_to_bf16_bits(fmaf(yv, bf16_bits_to_float(gamma[col]), xv));
  }
}

// Depthwise causal conv K, stride 1, groups = C, with fused residual +
// gamma epilogue. Input rows are the *normed* activation; negative row
// indices read the prefix buffer (prefix_rows == K - 1).
//   out[p, c] = resid[p, c] + gamma[c] * (sum_k w[c, k] * A(p - K + 1 + k, c) + b[c])
__global__ void vv_depthwise_conv_kernel(
    const uint16_t* __restrict__ prefix,
    const uint16_t* __restrict__ normed,
    const uint16_t* __restrict__ resid,
    const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ b,
    const uint16_t* __restrict__ gamma,
    uint16_t* __restrict__ out,
    int64_t prefix_rows,
    int64_t rows,
    int64_t c,
    int64_t k_len) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * c;
  if (idx >= total) {
    return;
  }
  const int64_t p = idx / c;
  const int64_t ch = idx - p * c;
  const uint16_t* src = nullptr;
  int64_t base = 0;
  float acc = bf16_bits_to_float(b[ch]);
  const int64_t left = k_len - 1;
  const uint16_t* wrow = w + ch * k_len;
  for (int64_t k = 0; k < k_len; ++k) {
    const int64_t i = p - left + k;
    if (i < 0) {
      src = prefix;
      base = (i + prefix_rows) * c + ch;
    } else {
      src = normed;
      base = i * c + ch;
    }
    acc = fmaf(bf16_bits_to_float(src[base]), bf16_bits_to_float(wrow[k]), acc);
  }
  const float scaled = fmaf(acc, bf16_bits_to_float(gamma[ch]), bf16_bits_to_float(resid[idx]));
  out[idx] = float_to_bf16_bits(scaled);
}

// Strided causal pointwise conv as a per-thread dot product.
//   out[p, n] = b[n] + sum_{k, ci} w_t[k, ci, n] * A(p * stride + k, ci)
// A(ii, ci): ii < prefix_rows reads prefix, else the current chunk x.
// w_t is [K][C_in][C_out] (transposed at upload from nn.Conv1d [C_out, C_in, K]).
#define VV_CONV_TM 4
#define VV_CONV_TN 64
__global__ void vv_conv_gemm_kernel(
    const uint16_t* __restrict__ prefix,
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ w_t,
    const uint16_t* __restrict__ b,
    uint16_t* __restrict__ out,
    int64_t prefix_rows,
    int64_t rows_out,
    int64_t c_in,
    int64_t c_out,
    int64_t k_len,
    int64_t stride) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * VV_CONV_TN + threadIdx.x;
  const int64_t p = static_cast<int64_t>(blockIdx.y) * VV_CONV_TM + threadIdx.y;
  if (p >= rows_out || n >= c_out) {
    return;
  }
  const int64_t row0 = p * stride;
  float acc = bf16_bits_to_float(b[n]);
  for (int64_t k = 0; k < k_len; ++k) {
    const int64_t ii = row0 + k;
    const uint16_t* src = (ii < prefix_rows) ? prefix : x;
    const int64_t base = (ii < prefix_rows ? ii : ii - prefix_rows) * c_in;
    const uint16_t* wbase = w_t + k * c_in * c_out + n;
    for (int64_t ci = 0; ci < c_in; ++ci) {
      acc = fmaf(bf16_bits_to_float(src[base + ci]), bf16_bits_to_float(wbase[ci * c_out]), acc);
    }
  }
  out[p * c_out + n] = float_to_bf16_bits(acc);
}

// latents += scale[batch] * noise (recorded-noise acoustic sampling)
__global__ void vv_add_scaled_noise_kernel(
    const uint16_t* __restrict__ lat,
    const float* __restrict__ scale,
    const uint16_t* __restrict__ noise,
    uint16_t* __restrict__ out,
    int64_t total,
    int64_t frames_x_hidden) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    const int64_t bcol = idx / frames_x_hidden;
    out[idx] = float_to_bf16_bits(
        bf16_bits_to_float(lat[idx]) + scale[bcol] * bf16_bits_to_float(noise[idx]));
  }
}

// out[r, c] = x[r, c] + b[c]  (bias add for the bias-less dense prefill GEMM)
__global__ void vv_add_bias_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ b,
    uint16_t* __restrict__ out,
    int64_t total,
    int64_t width) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    const int64_t col = idx % width;
    out[idx] = float_to_bf16_bits(bf16_bits_to_float(x[idx]) + bf16_bits_to_float(b[col]));
  }
}

// out[i] = x[i] + b[i % width]  (fp32 bias add for GEMV chains)
__global__ void vv_add_bias_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t total,
    int64_t width) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    const int64_t col = idx % width;
    out[idx] = x[idx] + b[col];
  }
}

}  // namespace

extern "C" int hipengine_vv_add_bias_f32(
    const float* x,
    const float* b,
    float* out,
    int64_t total,
    int64_t width,
    hipStream_t stream) {
  if (total <= 0 || width <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (total + 255) / 256;
  hipLaunchKernelGGL(
      vv_add_bias_f32_kernel, dim3(static_cast<unsigned int>(blocks)), dim3(256),
      0, stream, x, b, out, total, width);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_add_bias_bf16(
    const uint16_t* x,
    const uint16_t* b,
    uint16_t* out,
    int64_t total,
    int64_t width,
    hipStream_t stream) {
  if (total <= 0 || width <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (total + 255) / 256;
  hipLaunchKernelGGL(
      vv_add_bias_kernel, dim3(static_cast<unsigned int>(blocks)), dim3(256),
      0, stream, x, b, out, total, width);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_rmsnorm_bf16(
    const uint16_t* x,
    const uint16_t* w,
    uint16_t* out,
    int64_t rows,
    int64_t hidden,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      vv_rmsnorm_kernel, dim3(static_cast<unsigned int>(rows)), dim3(256),
      0, stream, x, w, out, rows, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_gelu_bf16(
    const uint16_t* x,
    uint16_t* out,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (n + 255) / 256;
  hipLaunchKernelGGL(
      vv_gelu_kernel, dim3(static_cast<unsigned int>(blocks)), dim3(256),
      0, stream, x, out, n);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_scale_residual_bf16(
    const uint16_t* x,
    const uint16_t* y,
    const uint16_t* gamma,
    uint16_t* out,
    int64_t total,
    int64_t width,
    hipStream_t stream) {
  if (total <= 0 || width <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (total + 255) / 256;
  hipLaunchKernelGGL(
      vv_scale_residual_kernel, dim3(static_cast<unsigned int>(blocks)), dim3(256),
      0, stream, x, y, gamma, out, total, width);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_depthwise_conv_bf16(
    const uint16_t* prefix,
    const uint16_t* normed,
    const uint16_t* resid,
    const uint16_t* w,
    const uint16_t* b,
    const uint16_t* gamma,
    uint16_t* out,
    int64_t prefix_rows,
    int64_t rows,
    int64_t c,
    int64_t k_len,
    hipStream_t stream) {
  if (rows <= 0 || c <= 0 || k_len <= 0 || prefix_rows != k_len - 1) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * c;
  const int64_t blocks = (total + 255) / 256;
  hipLaunchKernelGGL(
      vv_depthwise_conv_kernel, dim3(static_cast<unsigned int>(blocks)), dim3(256),
      0, stream, prefix, normed, resid, w, b, gamma, out, prefix_rows, rows, c, k_len);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_conv_gemm_bf16(
    const uint16_t* prefix,
    const uint16_t* x,
    const uint16_t* w_t,
    const uint16_t* b,
    uint16_t* out,
    int64_t prefix_rows,
    int64_t rows_out,
    int64_t c_in,
    int64_t c_out,
    int64_t k_len,
    int64_t stride,
    hipStream_t stream) {
  if (rows_out <= 0 || c_in <= 0 || c_out <= 0 || k_len <= 0 || stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const dim3 block(VV_CONV_TN, VV_CONV_TM);
  const dim3 grid(
      static_cast<unsigned int>((c_out + VV_CONV_TN - 1) / VV_CONV_TN),
      static_cast<unsigned int>((rows_out + VV_CONV_TM - 1) / VV_CONV_TM));
  hipLaunchKernelGGL(
      vv_conv_gemm_kernel, grid, block,
      0, stream, prefix, x, w_t, b, out, prefix_rows, rows_out, c_in, c_out, k_len, stride);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_add_scaled_noise_bf16(
    const uint16_t* lat,
    const float* scale,
    const uint16_t* noise,
    uint16_t* out,
    int64_t total,
    int64_t frames_x_hidden,
    hipStream_t stream) {
  if (total <= 0 || frames_x_hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (total + 255) / 256;
  hipLaunchKernelGGL(
      vv_add_scaled_noise_kernel, dim3(static_cast<unsigned int>(blocks)), dim3(256),
      0, stream, lat, scale, noise, out, total, frames_x_hidden);
  return static_cast<int>(hipGetLastError());
}
