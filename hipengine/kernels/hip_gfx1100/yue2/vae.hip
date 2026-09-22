// YuE2 Oobleck VAE decoder kernels (FP32).
//
// The released decoder is FP32 end to end and its arithmetic is not
// reassociable into a different class: the reference evaluates
// weight-normalized Conv1d/ConvTranspose1d and SnakeBeta in FP32, so these
// kernels accumulate in FP32 with the reference's own loop order (kernel tap
// outside, input channel inside) and apply the same `1e-9` SnakeBeta
// denominator. Weight normalization is folded on the host once, at load time.
//
// Layouts are PyTorch's: activations `[C, T]`, conv weights `[C_out, C_in, K]`,
// transposed weights `[C_in, C_out, K]`. One thread per output element keeps the
// index arithmetic trivially auditable; the tiled decode bounds residency, not
// this kernel's shape.

#include <hip/hip_runtime.h>
#include <cstdint>

// out[co, t] = bias[co] + sum_{ci, k} x[ci, t * stride - padding + k * dilation] * w[co, ci, k]
__global__ void yue2_vae_conv1d_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int64_t channels_in,
    int64_t channels_out,
    int64_t length,
    int64_t out_length,
    int64_t kernel,
    int64_t stride,
    int64_t dilation,
    int64_t padding) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= channels_out * out_length) {
    return;
  }
  const int64_t co = idx / out_length;
  const int64_t t = idx - co * out_length;
  float acc = bias != nullptr ? bias[co] : 0.0f;
  const int64_t base = t * stride - padding;
  const float* wrow = weight + co * channels_in * kernel;
  for (int64_t k = 0; k < kernel; ++k) {
    const int64_t pos = base + k * dilation;
    if (pos < 0 || pos >= length) {
      continue;
    }
    const float* wc = wrow + k;
    for (int64_t ci = 0; ci < channels_in; ++ci) {
      acc += x[ci * length + pos] * wc[ci * kernel];
    }
  }
  out[idx] = acc;
}

// out[co, t] = bias[co] + sum_{ci, k : t = i * stride - padding + k} x[ci, i] * w[ci, co, k]
__global__ void yue2_vae_conv_transpose1d_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int64_t channels_in,
    int64_t channels_out,
    int64_t length,
    int64_t out_length,
    int64_t kernel,
    int64_t stride,
    int64_t padding) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= channels_out * out_length) {
    return;
  }
  const int64_t co = idx / out_length;
  const int64_t t = idx - co * out_length;
  float acc = bias != nullptr ? bias[co] : 0.0f;
  for (int64_t k = 0; k < kernel; ++k) {
    const int64_t numerator = t + padding - k;
    if (numerator < 0 || numerator % stride != 0) {
      continue;
    }
    const int64_t i = numerator / stride;
    if (i >= length) {
      continue;
    }
    const float* wc = weight + co * kernel + k;
    for (int64_t ci = 0; ci < channels_in; ++ci) {
      acc += x[ci * length + i] * wc[ci * channels_out * kernel];
    }
  }
  out[idx] = acc;
}

// out[c, t] = x + (1 / (exp(beta[c]) + 1e-9)) * sin(x * exp(alpha[c]))^2
//
// alpha and beta are stored in log scale by the released decoder, so both are
// exponentiated per call exactly as the reference does; the denominator epsilon
// is the reference's `no_div_by_zero`.
__global__ void yue2_vae_snake_beta_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int64_t channels,
    int64_t length) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= channels * length) {
    return;
  }
  const int64_t c = idx / length;
  const float value = x[idx];
  const float gain = expf(alpha[c]);
  const float bias = expf(beta[c]);
  const float wave = sinf(value * gain);
  out[idx] = value + (1.0f / (bias + 1.0e-9f)) * wave * wave;
}

// out[i] = x[i] + y[i], used to add a residual unit's input back to its output.
__global__ void yue2_vae_add_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int64_t total) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  out[idx] = x[idx] + y[idx];
}

extern "C" int hipengine_yue2_vae_conv1d_f32(
    const float* x, const float* weight, const float* bias, float* out,
    int64_t channels_in, int64_t channels_out, int64_t length, int64_t out_length,
    int64_t kernel, int64_t stride, int64_t dilation, int64_t padding,
    hipStream_t stream) {
  const int64_t total = channels_out * out_length;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(yue2_vae_conv1d_f32_kernel, dim3(blocks), dim3(threads), 0, stream,
                     x, weight, bias, out, channels_in, channels_out, length, out_length,
                     kernel, stride, dilation, padding);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_yue2_vae_conv_transpose1d_f32(
    const float* x, const float* weight, const float* bias, float* out,
    int64_t channels_in, int64_t channels_out, int64_t length, int64_t out_length,
    int64_t kernel, int64_t stride, int64_t padding,
    hipStream_t stream) {
  const int64_t total = channels_out * out_length;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(yue2_vae_conv_transpose1d_f32_kernel, dim3(blocks), dim3(threads), 0, stream,
                     x, weight, bias, out, channels_in, channels_out, length, out_length,
                     kernel, stride, padding);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_yue2_vae_snake_beta_f32(
    const float* x, const float* alpha, const float* beta, float* out,
    int64_t channels, int64_t length,
    hipStream_t stream) {
  const int64_t total = channels * length;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(yue2_vae_snake_beta_f32_kernel, dim3(blocks), dim3(threads), 0, stream,
                     x, alpha, beta, out, channels, length);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_yue2_vae_add_f32(
    const float* x, const float* y, float* out, int64_t total,
    hipStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(yue2_vae_add_f32_kernel, dim3(blocks), dim3(threads), 0, stream,
                     x, y, out, total);
  return static_cast<int>(hipGetLastError());
}
