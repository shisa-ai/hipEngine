// Native MTP NextN draft-head kernels (GGUF path).
//
// Correctness-first, size-agnostic F32 kernels for the Qwen3.6 GGUF MTP NextN
// draft layer.  These mirror the ``cpu_reference`` numpy oracle math exactly so
// the M3 fixture gate (KL<=0.05 / top-1>=90%) can run on a real GPU backend
// instead of the registry's cpu_reference fallback.  M6 swaps the hot inner
// GEMVs for WMMA / K-quant tuned kernels on real shapes; these kernels remain
// the correctness baseline.
//
// eh_proj = enorm(embed) + hnorm(hidden) -> concat -> eh_proj linear.
//   out[row, j] = sum_k e_norm[row,k]*weight[j, k] + h_norm[row,k]*weight[j, k+hidden]
//   where weight is row-major [hidden, 2*hidden] (cpu_reference: fused @ weight.T).

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float warp_reduce_sum(float value) {
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset);
  }
  return value;
}

__device__ inline float block_reduce_sum(float value, float* shared) {
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int warps = (blockDim.x + warpSize - 1) / warpSize;
    float total = 0.0f;
    for (int i = 0; i < warps; ++i) {
      total += shared[i];
    }
    shared[0] = total;
  }
  __syncthreads();
  return shared[0];
}

}  // namespace

// RMSNorm F32, one block per row.  out[row,:] = x[row,:] * rsqrt(mean(x^2)+eps) * weight.
extern "C" __global__ void hipengine_mtp_rmsnorm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int64_t rows,
    int64_t hidden,
    float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  extern __shared__ float shared[];
  const float* xr = x + row * hidden;
  float* outr = out + row * hidden;
  float sumsq = 0.0f;
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < hidden;
       i += static_cast<int64_t>(blockDim.x)) {
    const float v = xr[i];
    sumsq += v * v;
  }
  sumsq = block_reduce_sum(sumsq, shared);
  const float inv = rsqrtf(sumsq / static_cast<float>(hidden) + eps);
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < hidden;
       i += static_cast<int64_t>(blockDim.x)) {
    outr[i] = xr[i] * inv * weight[i];
  }
}

extern "C" int hipengine_mtp_rmsnorm_f32(
    const float* x,
    const float* weight,
    float* out,
    int64_t rows,
    int64_t hidden,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int kWarpSize = 64;
  int block = 256;
  if (hidden < block) {
    block = static_cast<int>(hidden);
  }
  const int warps = (block + kWarpSize - 1) / kWarpSize;
  const int shbytes = (warps + 1) * sizeof(float);
  hipLaunchKernelGGL(
      hipengine_mtp_rmsnorm_f32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(block)),
      shbytes,
      stream,
      x, weight, out, rows, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

// eh_proj F32 GEMV, one block per row, threads stride over output features j.
// out[row, j] = sum_{k=0}^{hidden-1} e_norm[row,k]*weight[j*2hidden + k]
//             + sum_{k=0}^{hidden-1} h_norm[row,k]*weight[j*2hidden + hidden + k]
extern "C" __global__ void hipengine_mtp_eh_proj_f32_kernel(
    const float* __restrict__ e_norm,
    const float* __restrict__ h_norm,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int64_t rows,
    int64_t hidden) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const float* er = e_norm + row * hidden;
  const float* hr = h_norm + row * hidden;
  float* outr = out + row * hidden;
  const int64_t two_h = hidden * 2;
  for (int64_t j = static_cast<int64_t>(threadIdx.x); j < hidden;
       j += static_cast<int64_t>(blockDim.x)) {
    const float* w = weight + j * two_h;
    float acc = 0.0f;
    for (int64_t k = 0; k < hidden; ++k) {
      acc += er[k] * w[k];
    }
    for (int64_t k = 0; k < hidden; ++k) {
      acc += hr[k] * w[hidden + k];
    }
    outr[j] = acc;
  }
}

extern "C" int hipengine_mtp_eh_proj_f32(
    const float* e_norm,
    const float* h_norm,
    const float* weight,
    float* out,
    int64_t rows,
    int64_t hidden,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int block = 256;
  hipLaunchKernelGGL(
      hipengine_mtp_eh_proj_f32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(block),
      0,
      stream,
      e_norm, h_norm, weight, out, rows, hidden);
  return static_cast<int>(hipGetLastError());
}

// ---- MTP NextN attention sub-kernels (F32, correctness-first) ----

// Generic F32 linear: out[t, j] = sum_k x[t, k] * weight[j, k]  (weight row-major [out, in]).
// One block per (token, out_feature_chunk); here one block per token, threads stride over j.
extern "C" __global__ void hipengine_mtp_linear_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int64_t tokens,
    int64_t in_features,
    int64_t out_features) {
  const int64_t t = static_cast<int64_t>(blockIdx.x);
  if (t >= tokens) {
    return;
  }
  const float* xr = x + t * in_features;
  float* outr = out + t * out_features;
  for (int64_t j = static_cast<int64_t>(threadIdx.x); j < out_features;
       j += static_cast<int64_t>(blockDim.x)) {
    const float* w = weight + j * in_features;
    float acc = 0.0f;
    for (int64_t k = 0; k < in_features; ++k) {
      acc += xr[k] * w[k];
    }
    outr[j] = acc;
  }
}

extern "C" int hipengine_mtp_linear_f32(
    const float* x,
    const float* weight,
    float* out,
    int64_t tokens,
    int64_t in_features,
    int64_t out_features,
    hipStream_t stream) {
  if (tokens <= 0 || in_features <= 0 || out_features <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int block = 256;
  hipLaunchKernelGGL(
      hipengine_mtp_linear_f32_kernel,
      dim3(static_cast<unsigned int>(tokens)),
      dim3(block),
      0,
      stream,
      x, weight, out, tokens, in_features, out_features);
  return static_cast<int>(hipGetLastError());
}

// Sigmoid gate multiply: out[t, h, d] = attn[t, h, d] * sigmoid(gate[t, h, d]).
// Flattened over (tokens*heads) rows of head_dim.
extern "C" __global__ void hipengine_mtp_sigmoid_gate_mul_f32_kernel(
    const float* __restrict__ attn,
    const float* __restrict__ gate,
    float* __restrict__ out,
    int64_t rows,
    int64_t head_dim) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int64_t base = row * head_dim;
  for (int64_t d = static_cast<int64_t>(threadIdx.x); d < head_dim;
       d += static_cast<int64_t>(blockDim.x)) {
    const float g = 1.0f / (1.0f + __expf(-gate[base + d]));
    out[base + d] = attn[base + d] * g;
  }
}

extern "C" int hipengine_mtp_sigmoid_gate_mul_f32(
    const float* attn,
    const float* gate,
    float* out,
    int64_t rows,
    int64_t head_dim,
    hipStream_t stream) {
  if (rows <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  int block = 256;
  if (head_dim < block) {
    block = static_cast<int>(head_dim);
  }
  if (block < 1) {
    block = 1;
  }
  hipLaunchKernelGGL(
      hipengine_mtp_sigmoid_gate_mul_f32_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(static_cast<unsigned int>(block)),
      0,
      stream,
      attn, gate, out, rows, head_dim);
  return static_cast<int>(hipGetLastError());
}

// Elementwise add: out = a + b, flattened.
extern "C" __global__ void hipengine_mtp_add_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t n) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = a[i] + b[i];
  }
}

extern "C" int hipengine_mtp_add_f32(
    const float* a,
    const float* b,
    float* out,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  hipLaunchKernelGGL(
      hipengine_mtp_add_f32_kernel,
      dim3(static_cast<unsigned int>(grid)),
      dim3(block),
      0,
      stream,
      a, b, out, n);
  return static_cast<int>(hipGetLastError());
}

// Dense causal GQA attention (F32, correctness-first).
// query: [tokens, heads, qk_head_dim]
// key_cache / value_cache: [cache_tokens, kv_heads, head_dim]
// positions / context_counts: [tokens] int64
// out: [tokens, heads, value_head_dim]
// One block per (token, q_head). Thread 0 serially scores visible positions
// into shared mem; all threads softmax-reduce; threads stride over value_head_dim
// for the weighted sum.  Correctness-first -- M6 rewrites for real shapes.
extern "C" __global__ void hipengine_mtp_dense_attn_f32_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const int64_t* __restrict__ positions,
    const int64_t* __restrict__ context_counts,
    float* __restrict__ out,
    int64_t tokens,
    int64_t heads,
    int64_t kv_heads,
    int64_t qk_head_dim,
    int64_t value_head_dim,
    int64_t cache_tokens,
    float scale) {
  const int64_t t = static_cast<int64_t>(blockIdx.x);
  const int64_t qh = static_cast<int64_t>(blockIdx.y);
  if (t >= tokens || qh >= heads) {
    return;
  }
  const int64_t kv_group = heads / kv_heads;
  const int64_t kh = qh / kv_group;
  const int64_t position = positions[t];
  const int64_t context = context_counts[t];
  // Visible (causal) cache positions are 0..vcount-1 where vcount = min(position, context-1)+1.
  int64_t vcount = position;
  if (context - 1 < vcount) {
    vcount = context - 1;
  }
  vcount += 1;
  if (vcount <= 0) {
    vcount = 0;
  }
  extern __shared__ float shared[];
  float* scores = shared;          // [vcount]
  float* out_acc = shared + cache_tokens;  // [value_head_dim]

  const float* qr = query + (t * heads + qh) * qk_head_dim;

  if (threadIdx.x == 0) {
    float max_score = -INFINITY;
    for (int64_t v = 0; v < vcount; ++v) {
      const float* kr = key_cache + (v * kv_heads + kh) * qk_head_dim;
      float dot = 0.0f;
      for (int64_t d = 0; d < qk_head_dim; ++d) {
        dot += kr[d] * qr[d];
      }
      float s = dot * scale;
      scores[v] = s;
      if (s > max_score) {
        max_score = s;
      }
    }
    float exp_sum = 0.0f;
    for (int64_t v = 0; v < vcount; ++v) {
      float e = __expf(scores[v] - max_score);
      scores[v] = e;
      exp_sum += e;
    }
    float inv = (exp_sum > 0.0f) ? (1.0f / exp_sum) : 0.0f;
    for (int64_t d = 0; d < value_head_dim; ++d) {
      float acc = 0.0f;
      for (int64_t v = 0; v < vcount; ++v) {
        const float* vr = value_cache + (v * kv_heads + kh) * value_head_dim;
        acc += scores[v] * inv * vr[d];
      }
      out_acc[d] = acc;
    }
  }
  __syncthreads();
  float* outr = out + (t * heads + qh) * value_head_dim;
  for (int64_t d = static_cast<int64_t>(threadIdx.x); d < value_head_dim;
       d += static_cast<int64_t>(blockDim.x)) {
    outr[d] = out_acc[d];
  }
}

extern "C" int hipengine_mtp_dense_attn_f32(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int64_t* positions,
    const int64_t* context_counts,
    float* out,
    int64_t tokens,
    int64_t heads,
    int64_t kv_heads,
    int64_t qk_head_dim,
    int64_t value_head_dim,
    int64_t cache_tokens,
    float scale,
    hipStream_t stream) {
  if (tokens <= 0 || heads <= 0 || kv_heads <= 0 || qk_head_dim <= 0 ||
      value_head_dim <= 0 || cache_tokens <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int shbytes = static_cast<int>((cache_tokens + value_head_dim) * sizeof(float));
  int block = 256;
  if (value_head_dim < block) {
    block = static_cast<int>(value_head_dim);
  }
  if (block < 1) {
    block = 1;
  }
  dim3 grid(static_cast<unsigned int>(tokens), static_cast<unsigned int>(heads));
  hipLaunchKernelGGL(
      hipengine_mtp_dense_attn_f32_kernel,
      grid,
      dim3(static_cast<unsigned int>(block)),
      shbytes,
      stream,
      query, key_cache, value_cache, positions, context_counts, out,
      tokens, heads, kv_heads, qk_head_dim, value_head_dim, cache_tokens, scale);
  return static_cast<int>(hipGetLastError());
}

// ---- MTP NextN FFN sub-kernels (F32, correctness-first) ----

// SiLU-mul fused: out[i] = silu(gate[i]) * up[i], where silu(x) = x/(1+exp(-x)).
extern "C" __global__ void hipengine_mtp_silu_mul_f32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int64_t n) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float g = gate[i];
    out[i] = g / (1.0f + __expf(-g)) * up[i];
  }
}

extern "C" int hipengine_mtp_silu_mul_f32(
    const float* gate,
    const float* up,
    float* out,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  hipLaunchKernelGGL(
      hipengine_mtp_silu_mul_f32_kernel,
      dim3(static_cast<unsigned int>(grid)),
      dim3(block),
      0,
      stream,
      gate, up, out, n);
  return static_cast<int>(hipGetLastError());
}

// Elementwise mul: out[i] = a[i] * b[i].
extern "C" __global__ void hipengine_mtp_mul_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t n) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = a[i] * b[i];
  }
}

extern "C" int hipengine_mtp_mul_f32(
    const float* a,
    const float* b,
    float* out,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  hipLaunchKernelGGL(
      hipengine_mtp_mul_f32_kernel,
      dim3(static_cast<unsigned int>(grid)),
      dim3(block),
      0,
      stream,
      a, b, out, n);
  return static_cast<int>(hipGetLastError());
}

// Scalar scale: out[i] = scalar * x[i].
extern "C" __global__ void hipengine_mtp_scale_f32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    float scalar,
    int64_t n) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = scalar * x[i];
  }
}

extern "C" int hipengine_mtp_scale_f32(
    const float* x,
    float* out,
    float scalar,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  constexpr int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  hipLaunchKernelGGL(
      hipengine_mtp_scale_f32_kernel,
      dim3(static_cast<unsigned int>(grid)),
      dim3(block),
      0,
      stream,
      x, out, scalar, n);
  return static_cast<int>(hipGetLastError());
}

// Row scale (broadcast): out[t, d] = scale[t] * x[t, d].  One block per row.
extern "C" __global__ void hipengine_mtp_row_scale_f32_kernel(
    const float* __restrict__ scale,
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t tokens,
    int64_t hidden) {
  const int64_t t = static_cast<int64_t>(blockIdx.x);
  if (t >= tokens) {
    return;
  }
  const float s = scale[t];
  const int64_t base = t * hidden;
  for (int64_t d = static_cast<int64_t>(threadIdx.x); d < hidden;
       d += static_cast<int64_t>(blockDim.x)) {
    out[base + d] = s * x[base + d];
  }
}

extern "C" int hipengine_mtp_row_scale_f32(
    const float* scale,
    const float* x,
    float* out,
    int64_t tokens,
    int64_t hidden,
    hipStream_t stream) {
  if (tokens <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  int block = 256;
  if (hidden < block) {
    block = static_cast<int>(hidden);
  }
  if (block < 1) {
    block = 1;
  }
  hipLaunchKernelGGL(
      hipengine_mtp_row_scale_f32_kernel,
      dim3(static_cast<unsigned int>(tokens)),
      dim3(static_cast<unsigned int>(block)),
      0,
      stream,
      scale, x, out, tokens, hidden);
  return static_cast<int>(hipGetLastError());
}

// ---- MTP NextN RoPE kernel (F32, split-half rotary embedding) ----
// Apply split-half RoPE to x[..., :rotary_dim], pass-through x[..., rotary_dim:].
// [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
// cos/sin have shape [tokens, half] where half = rotary_dim/2.
// One block per (token, head) row, threads stride over half-dim pairs.
extern "C" __global__ void hipengine_mtp_rope_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ out,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t cos_stride) {  // cos_stride = half (elements per token in cos table)
  const int64_t t = static_cast<int64_t>(blockIdx.x);
  const int64_t h = static_cast<int64_t>(blockIdx.y);
  if (t >= tokens || h >= heads) {
    return;
  }
  const int64_t row_base = (t * heads + h) * head_dim;
  const int64_t half = rotary_dim / 2;
  const int64_t cos_base = t * cos_stride;

  // Rotate the rotary_dim portion
  for (int64_t i = static_cast<int64_t>(threadIdx.x); i < half;
       i += static_cast<int64_t>(blockDim.x)) {
    const float x1 = x[row_base + i];
    const float x2 = x[row_base + half + i];
    const float c = cos[cos_base + i];
    const float s = sin[cos_base + i];
    out[row_base + i] = x1 * c - x2 * s;
    out[row_base + half + i] = x1 * s + x2 * c;
  }
  // Pass-through the non-rotary portion
  for (int64_t i = static_cast<int64_t>(half) * 2 + static_cast<int64_t>(threadIdx.x);
       i < head_dim;
       i += static_cast<int64_t>(blockDim.x)) {
    out[row_base + i] = x[row_base + i];
  }
}

extern "C" int hipengine_mtp_rope_f32(
    const float* x,
    const float* cos,
    const float* sin,
    float* out,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t cos_stride,
    hipStream_t stream) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0 || rotary_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  int block = 256;
  if (rotary_dim / 2 < block) {
    block = static_cast<int>(rotary_dim / 2);
  }
  if (block < 1) {
    block = 1;
  }
  dim3 grid(static_cast<unsigned int>(tokens), static_cast<unsigned int>(heads));
  hipLaunchKernelGGL(
      hipengine_mtp_rope_f32_kernel,
      grid,
      dim3(static_cast<unsigned int>(block)),
      0,
      stream,
      x, cos, sin, out, tokens, heads, head_dim, rotary_dim, cos_stride);
  return static_cast<int>(hipGetLastError());
}
