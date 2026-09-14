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
// Warp-per-row: lane-strided channel loop + shuffle reduction so small
// hidden sizes do not waste three quarters of each block.
__global__ void vv_rmsnorm_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ w,
    uint16_t* __restrict__ out,
    int64_t rows,
    int64_t hidden,
    float eps) {
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
      const float v = bf16_bits_to_float(x[base + i]);
      out[base + i] = float_to_bf16_bits(v * inv * bf16_bits_to_float(w[i]));
    }
  }
}

__device__ __forceinline__ float vv_fast_erf(float z) {
  // Abramowitz & Stegun 7.1.26 rational fit: |err| <= 1.5e-7, far below
  // bf16 storage precision; avoids the slow device erff call.
  const float sign = z < 0.0f ? -1.0f : 1.0f;
  const float t = 1.0f / (1.0f + 0.3275911f * fabsf(z));
  const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * expf(-z * z);
  return sign * y;
}

__global__ void vv_gelu_kernel(
    const uint16_t* __restrict__ x,
    uint16_t* __restrict__ out,
    int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float v = bf16_bits_to_float(x[idx]);
    out[idx] = float_to_bf16_bits(0.5f * v * (1.0f + vv_fast_erf(v * 0.7071067811865476f)));
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

// Batched rotate-half rope over rows with explicit positions.
//   q/k: (rows, heads, head_dim) fp32; cos/sin tables (max_pos, head_dim).
//   out[t, h, d] for d < half: x[d]*cos - x[d+half]*sin;
//   for d >= half: x[d]*cos + x[d-half]*sin (cos/sin duplicated halves).
__global__ void vv_rope_positions_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int64_t* __restrict__ positions,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int64_t rows,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  const int64_t total_q = rows * num_q_heads * head_dim;
  const int64_t total_k = rows * num_kv_heads * head_dim;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total_q + total_k;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const bool is_q = idx < total_q;
    const int64_t local = is_q ? idx : idx - total_q;
    const int64_t heads = is_q ? num_q_heads : num_kv_heads;
    const int64_t row = local / (heads * head_dim);
    const int64_t within = local - row * heads * head_dim;
    const int64_t dim = within % head_dim;
    const float* src = is_q ? q : k;
    float* dst = is_q ? q_out : k_out;
    const int64_t pos = positions[row];
    const float* cos_p = cos_table + pos * head_dim;
    const float* sin_p = sin_table + pos * head_dim;
    const int64_t half = head_dim / 2;
    const int64_t base = local - dim;
    const int64_t pair_dim = dim < half ? dim + half : dim - half;
    const float paired = src[base + pair_dim];
    const float rotated = dim < half ? -paired : paired;
    dst[local] = src[local] * cos_p[dim] + rotated * sin_p[dim];
  }
}

// Causal prefill attention over a dense contiguous KV cache.
//   q: (rows, num_q_heads, head_dim) fp32; k/v caches: (capacity, kv, hd) bf16;
//   out: (rows, num_q_heads, head_dim) fp32. Row r attends positions [0, pos[r]].
__global__ void vv_prefill_attention_kernel(
    const float* __restrict__ q,
    const uint16_t* __restrict__ key_cache,
    const uint16_t* __restrict__ value_cache,
    const int64_t* __restrict__ positions,
    float* __restrict__ out,
    int64_t rows,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t max_ctx,
    float scale) {
  const int64_t row = blockIdx.y;
  const int64_t q_head = blockIdx.x;
  if (row >= rows || q_head >= num_q_heads) {
    return;
  }
  const int64_t ctx = positions[row] + 1;
  const int64_t kv_group = num_q_heads / num_kv_heads;
  const int64_t kv_head = q_head / kv_group;
  const float* q_row = q + (row * num_q_heads + q_head) * head_dim;
  extern __shared__ float shared[];
  float* scores = shared;              // capacity max_ctx
  float* partial = scores + max_ctx;   // blockDim floats at a fixed offset
  float local_max = -3.4028234663852886e38f;
  for (int64_t token = threadIdx.x; token < ctx; token += blockDim.x) {
    const int64_t k_off = (token * num_kv_heads + kv_head) * head_dim;
    float acc = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      acc = fmaf(q_row[d], bf16_bits_to_float(key_cache[k_off + d]), acc);
    }
    const float score = acc * scale;
    scores[token] = score;
    local_max = fmaxf(local_max, score);
  }
  partial[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_score = partial[0];
  __syncthreads();
  float local_sum = 0.0f;
  for (int64_t token = threadIdx.x; token < ctx; token += blockDim.x) {
    local_sum += expf(scores[token] - max_score);
  }
  partial[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / partial[0];
  // normalized attention weights live in shared so the value pass reads
  // them instead of recomputing expf per (dim, token) pair
  for (int64_t token = threadIdx.x; token < ctx; token += blockDim.x) {
    scores[token] = expf(scores[token] - max_score) * inv_sum;
  }
  __syncthreads();
  float* out_row = out + (row * num_q_heads + q_head) * head_dim;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t token = 0; token < ctx; ++token) {
      const int64_t v_off = (token * num_kv_heads + kv_head) * head_dim + d;
      acc = fmaf(scores[token], bf16_bits_to_float(value_cache[v_off]), acc);
    }
    out_row[d] = acc;
  }
}

// im2col for a strided causal conv: A[p, ci*K + k] = X(p*stride + k
// - prefix_rows, ci), with zero for pre-history taps. Matches the raw
// nn.Conv1d weight layout [C_out, C_in, K] flattened to [C_out, C_in*K].
__global__ void vv_im2col_kernel(
    const uint16_t* __restrict__ x,
    uint16_t* __restrict__ out,
    int64_t rows_out,
    int64_t c_in,
    int64_t k_len,
    int64_t stride,
    int64_t prefix_rows) {
  const int64_t total = rows_out * c_in * k_len;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t p = idx / (c_in * k_len);
    const int64_t rest = idx - p * c_in * k_len;
    const int64_t ci = rest / k_len;
    const int64_t k = rest - ci * k_len;
    const int64_t row = p * stride + k - prefix_rows;
    out[idx] = (row < 0) ? 0 : x[row * c_in + ci];
  }
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
  // warp-per-row: 256 threads = 8 warps process 8 rows per block pass
  const int64_t blocks = (rows + 7) / 8;
  hipLaunchKernelGGL(
      vv_rmsnorm_kernel, dim3(static_cast<unsigned int>(blocks)), dim3(256),
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

extern "C" int hipengine_vv_rope_positions_f32(
    const float* q,
    const float* k,
    const float* cos_table,
    const float* sin_table,
    const int64_t* positions,
    float* q_out,
    float* k_out,
    int64_t rows,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    hipStream_t stream) {
  if (rows <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * (num_q_heads + num_kv_heads) * head_dim;
  const int64_t blocks = (total + 255) / 256;
  hipLaunchKernelGGL(
      vv_rope_positions_kernel, dim3(static_cast<unsigned int>(blocks)), dim3(256),
      0, stream, q, k, cos_table, sin_table, positions, q_out, k_out,
      rows, num_q_heads, num_kv_heads, head_dim);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_prefill_attention_f32(
    const float* q,
    const uint16_t* key_cache,
    const uint16_t* value_cache,
    const int64_t* positions,
    float* out,
    int64_t rows,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t max_ctx,
    float scale,
    hipStream_t stream) {
  if (rows <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || scale <= 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t shared_bytes = (max_ctx + 256) * sizeof(float);
  if (shared_bytes > 64 * 1024) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      vv_prefill_attention_kernel,
      dim3(static_cast<unsigned int>(num_q_heads), static_cast<unsigned int>(rows)),
      dim3(256),
      static_cast<unsigned int>(shared_bytes),
      stream, q, key_cache, value_cache, positions, out,
      rows, num_q_heads, num_kv_heads, head_dim, max_ctx, scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_vv_im2col_bf16(
    const uint16_t* x,
    uint16_t* out,
    int64_t rows_out,
    int64_t c_in,
    int64_t k_len,
    int64_t stride,
    int64_t prefix_rows,
    hipStream_t stream) {
  if (rows_out <= 0 || c_in <= 0 || k_len <= 0 || stride <= 0 || prefix_rows < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows_out * c_in * k_len;
  const int64_t blocks = (total + 255) / 256;
  hipLaunchKernelGGL(
      vv_im2col_kernel, dim3(static_cast<unsigned int>(blocks)), dim3(256),
      0, stream, x, out, rows_out, c_in, k_len, stride, prefix_rows);
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
