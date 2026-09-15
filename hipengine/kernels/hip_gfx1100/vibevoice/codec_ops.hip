// VibeVoice acoustic-tokenizer decoder primitives (gfx1100/gfx1151 lineage).
//
// All kernels operate on raw device pointers with fp32 storage and mirror the
// CPU reference (hipengine/kernels/cpu_reference/vibevoice_codec.py):
//   - conv1d_valid_dense_f32: valid dense causal conv, x [in_c, l_in],
//     weight [out_c, in_c, k], output [out_c, l_in - k + 1]; the caller
//     pre-pends the zero/context columns so streaming and full decodes share
//     one primitive.
//   - conv1d_valid_depthwise_f32: groups == channels, weight [c, 1, k].
//   - conv_transpose1d_causal_f32: torch ConvTranspose1d weight layout
//     [in_c, out_c, k], stride s, causal right trim of (k - s) samples so the
//     output length is exactly l_in * s.
//   - rmsnorm_channels_f32: per-time-step RMSNorm over the channel axis
//     (the fork's ConvRMSNorm), out = x / sqrt(mean(x^2) + eps) * w.
//   - gelu_erf_f32: exact erf GELU (ACT2FN['gelu']).
//   - linear_f32: out = w [out, in] @ x [in, l] + b.
//   - add_f32 / channel_scale_f32: residual add and layer-scale multiply.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

// ---------------------------------------------------------------- conv1d

// One thread per output element; accumulation over (in_c, k) in tap-major
// order to match the reference's per-tap matmul reduction.
__global__ void hipengine_vv_conv1d_valid_dense_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t in_c,
    int64_t out_c,
    int64_t k,
    int64_t l_out) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= out_c * l_out) {
    return;
  }
  const int64_t o = index / l_out;
  const int64_t j = index % l_out;
  const float* w_o = w + o * in_c * k;
  float acc = 0.0f;
  for (int64_t tap = 0; tap < k; ++tap) {
    const float* w_tap = w_o + tap;
    const float* x_win = x + tap * 1 + j;  // x[ic, j + tap] = x + ic*l_in + j + tap
    // in_c reduction; l_in == l_out + k - 1.
    const int64_t l_in = l_out + k - 1;
    float part = 0.0f;
    for (int64_t ic = 0; ic < in_c; ++ic) {
      part += w_tap[ic * k] * x_win[ic * l_in];
    }
    acc += part;
  }
  out[index] = acc + b[o];
}

// Depthwise causal conv: one thread per (channel, position).
__global__ void hipengine_vv_conv1d_valid_depthwise_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t channels,
    int64_t k,
    int64_t l_out) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= channels * l_out) {
    return;
  }
  const int64_t c = index / l_out;
  const int64_t j = index % l_out;
  const int64_t l_in = l_out + k - 1;
  const float* x_c = x + c * l_in;
  const float* w_c = w + c * k;
  float acc = 0.0f;
  for (int64_t tap = 0; tap < k; ++tap) {
    acc += w_c[tap] * x_c[j + tap];
  }
  out[index] = acc + b[c];
}

// Causal transposed conv: one thread per (out_channel, output position).
// Output j collects input positions i = (j - tap) / s for taps with
// tap == j mod s; taps feeding only the trimmed right padding are skipped.
// Only positions j >= keep_from are written, compacted densely to the first
// kept_len columns of the output, so streaming callers can read the kept
// tail as a dense [out_c, kept_len] tensor without a host-side view.
__global__ void hipengine_vv_conv_transpose1d_causal_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t in_c,
    int64_t out_c,
    int64_t k,
    int64_t stride,
    int64_t l_out,
    int64_t keep_from,
    int64_t kept_len) {
  const int64_t total = out_c * kept_len;
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int64_t o = index / kept_len;
  const int64_t j = keep_from + (index % kept_len);
  const int64_t l_in = (l_out + stride - 1) / stride;  // ceil: input positions
  const int64_t tap0 = j % stride;
  float acc = 0.0f;
  for (int64_t tap = tap0; tap < k; tap += stride) {
    if (tap > j) {
      continue;  // j - tap would be negative: no contributing input position.
    }
    const int64_t i = (j - tap) / stride;
    if (i >= l_in) {
      continue;
    }
    float part = 0.0f;
    for (int64_t ic = 0; ic < in_c; ++ic) {
      part += x[ic * l_in + i] * w[(ic * out_c + o) * k + tap];
    }
    acc += part;
  }
  out[index] = acc + b[o];
}

// ----------------------------------------------------------- norms / acts

// Per-time-step RMSNorm over the channel axis of [c, l].
__global__ void hipengine_vv_rmsnorm_channels_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int64_t channels,
    int64_t length,
    float eps) {
  const int64_t j = static_cast<int64_t>(blockIdx.x);
  if (j >= length) {
    return;
  }
  __shared__ float partial[256];
  float acc = 0.0f;
  for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
    const float v = x[c * length + j];
    acc += v * v;
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(partial[0] / static_cast<float>(channels) + eps);
  for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
    out[c * length + j] = x[c * length + j] * inv * w[c];
  }
}

__global__ void hipengine_vv_gelu_erf_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float v = x[i];
  out[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
}

__global__ void hipengine_vv_add_f32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = x[i] + y[i];
}

// out[i] = x[i] * scale[i % channels]: layer scales broadcast over time.
__global__ void hipengine_vv_channel_scale_f32(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int64_t channels,
    int64_t length) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= channels * length) {
    return;
  }
  out[index] = x[index] * scale[index / length];
}

// ------------------------------------------------- conv window assembly

// Assemble a conv window buffer from a device-resident cache tail and a new
// contiguous input: dst[c, 0:keep] = last `keep` columns of cache_src,
// dst[c, keep:keep+new_len] = x_src. One thread per element; keep == 0 (an
// empty convtr cache on its first chunk) skips the cache half entirely.
__global__ void hipengine_vv_concat_gather_f32(
    const float* __restrict__ cache_src,
    const float* __restrict__ x_src,
    float* __restrict__ dst,
    int64_t channels,
    int64_t keep,
    int64_t cache_len,
    int64_t new_len) {
  const int64_t row_len = keep + new_len;
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= channels * row_len) {
    return;
  }
  const int64_t c = index / row_len;
  const int64_t j = index % row_len;
  if (j < keep) {
    dst[index] = cache_src[c * cache_len + (cache_len - keep + j)];
  } else {
    dst[index] = x_src[c * new_len + (j - keep)];
  }
}

// ------------------------------------------------------------- linear

// out[o * l + j] = sum_i w[o * in + i] * x[i * l + j] + b[o].
__global__ void hipengine_vv_linear_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t in_features,
    int64_t out_features,
    int64_t length) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= out_features * length) {
    return;
  }
  const int64_t o = index / length;
  const int64_t j = index % length;
  const float* w_o = w + o * in_features;
  float acc = 0.0f;
  for (int64_t i = 0; i < in_features; ++i) {
    acc += w_o[i] * x[i * length + j];
  }
  out[index] = acc + b[o];
}

extern "C" {

hipError_t hipengine_vv_conv1d_valid_dense(
    const float* x, const float* w, const float* b, float* out,
    int64_t in_c, int64_t out_c, int64_t k, int64_t l_out, hipStream_t stream) {
  const int64_t total = out_c * l_out;
  if (total <= 0) return hipSuccess;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(hipengine_vv_conv1d_valid_dense_f32, dim3(blocks), dim3(threads),
                     0, stream, x, w, b, out, in_c, out_c, k, l_out);
  return hipGetLastError();
}

hipError_t hipengine_vv_conv1d_valid_depthwise(
    const float* x, const float* w, const float* b, float* out,
    int64_t channels, int64_t k, int64_t l_out, hipStream_t stream) {
  const int64_t total = channels * l_out;
  if (total <= 0) return hipSuccess;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(hipengine_vv_conv1d_valid_depthwise_f32, dim3(blocks), dim3(threads),
                     0, stream, x, w, b, out, channels, k, l_out);
  return hipGetLastError();
}

hipError_t hipengine_vv_conv_transpose1d_causal(
    const float* x, const float* w, const float* b, float* out,
    int64_t in_c, int64_t out_c, int64_t k, int64_t stride, int64_t l_out,
    int64_t keep_from, int64_t kept_len, hipStream_t stream) {
  if (kept_len <= 0) return hipSuccess;
  const int threads = 256;
  const int blocks = static_cast<int>((out_c * kept_len + threads - 1) / threads);
  hipLaunchKernelGGL(hipengine_vv_conv_transpose1d_causal_f32, dim3(blocks), dim3(threads),
                     0, stream, x, w, b, out, in_c, out_c, k, stride, l_out,
                     keep_from, kept_len);
  return hipGetLastError();
}

hipError_t hipengine_vv_rmsnorm_channels(
    const float* x, const float* w, float* out,
    int64_t channels, int64_t length, float eps, hipStream_t stream) {
  if (length <= 0) return hipSuccess;
  hipLaunchKernelGGL(hipengine_vv_rmsnorm_channels_f32, dim3(static_cast<unsigned>(length)),
                     dim3(256), 0, stream, x, w, out, channels, length, eps);
  return hipGetLastError();
}

hipError_t hipengine_vv_gelu_erf(
    const float* x, float* out, int64_t n, hipStream_t stream) {
  if (n <= 0) return hipSuccess;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(hipengine_vv_gelu_erf_f32, dim3(blocks), dim3(threads),
                     0, stream, x, out, n);
  return hipGetLastError();
}

hipError_t hipengine_vv_add(
    const float* x, const float* y, float* out, int64_t n, hipStream_t stream) {
  if (n <= 0) return hipSuccess;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(hipengine_vv_add_f32, dim3(blocks), dim3(threads), 0, stream,
                     x, y, out, n);
  return hipGetLastError();
}

hipError_t hipengine_vv_channel_scale(
    const float* x, const float* scale, float* out,
    int64_t channels, int64_t length, hipStream_t stream) {
  const int64_t total = channels * length;
  if (total <= 0) return hipSuccess;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(hipengine_vv_channel_scale_f32, dim3(blocks), dim3(threads),
                     0, stream, x, scale, out, channels, length);
  return hipGetLastError();
}

hipError_t hipengine_vv_linear(
    const float* x, const float* w, const float* b, float* out,
    int64_t in_features, int64_t out_features, int64_t length, hipStream_t stream) {
  const int64_t total = out_features * length;
  if (total <= 0) return hipSuccess;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(hipengine_vv_linear_f32, dim3(blocks), dim3(threads), 0, stream,
                     x, w, b, out, in_features, out_features, length);
  return hipGetLastError();
}

hipError_t hipengine_vv_concat_gather(
    const float* cache_src, const float* x_src, float* dst,
    int64_t channels, int64_t keep, int64_t cache_len, int64_t new_len,
    hipStream_t stream) {
  const int64_t total = channels * (keep + new_len);
  if (total <= 0) return hipSuccess;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(hipengine_vv_concat_gather_f32, dim3(blocks), dim3(threads),
                     0, stream, cache_src, x_src, dst, channels, keep,
                     cache_len, new_len);
  return hipGetLastError();
}

}  // extern "C"
