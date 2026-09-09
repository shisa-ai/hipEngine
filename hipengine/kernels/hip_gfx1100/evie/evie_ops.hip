// EVIE-4.5B fp32 elementwise/norm/rope kernels (gfx1100/gfx1151 lineage).
//
// All kernels operate on raw device pointers with fp32 storage. They mirror
// the CPU reference (hipengine/kernels/cpu_reference/evie.py) exactly:
//   - rmsnorm uses the Qwen delta form out = (1 + w) * x / sqrt(mean(x^2)+eps)
//   - layernorm is the vision-tower affine form (w, b)
//   - rope applies rotate_half semantics on the leading rotary dims with
//     per-token cos/sin rows (interleaved mRoPE tables are precomputed on
//     the host; pair->axis assignment is baked into those tables)
//   - gdn gates compute beta = sigmoid(b) and decay = exp(-exp(A_log) *
//     softplus(a + dt_bias)) per (token, v_head)

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>
#include <hip/amd_detail/amd_hip_fp16.h>

namespace {

__device__ __forceinline__ float evie_silu_f32(float value) {
  return value / (1.0f + expf(-value));
}

__device__ __forceinline__ float evie_softplus_f32(float value) {
  if (value > 20.0f) {
    return value;
  }
  return log1pf(expf(value));
}

// One block per row, 256 threads. Classic shared-memory tree reduction
// (wave-size agnostic, mirrors the gdn.hip pattern).
__device__ __forceinline__ float evie_row_sum_squares(
    const float* __restrict__ row, int64_t hidden) {
  __shared__ float partial[256];
  float acc = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    acc += row[i] * row[i];
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return partial[0];
}

}  // namespace

// out = (1 + w) * x / sqrt(mean(x^2) + eps), row-wise over hidden dim.
__global__ void evie_rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int64_t rows,
    int64_t hidden,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float* xr = x + row * hidden;
  const float total = evie_row_sum_squares(xr, hidden);
  if (threadIdx.x == 0) {
    const float inv = rsqrtf(total / static_cast<float>(hidden) + eps);
    float* orow = out + row * hidden;
    for (int64_t i = 0; i < hidden; ++i) {
      orow[i] = (1.0f + w[i]) * xr[i] * inv;
    }
  }
}

extern "C" int hipengine_evie_rmsnorm_f32(
    const float* x,
    const float* w,
    float* out,
    int64_t rows,
    int64_t hidden,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      evie_rmsnorm_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(256),
      0,
      stream,
      x, w, out, rows, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

// out = x + y (residual add).
__global__ void evie_add_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int64_t n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = x[idx] + y[idx];
  }
}

extern "C" int hipengine_evie_add_f32(
    const float* x,
    const float* y,
    float* out,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks =
      static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_add_kernel, dim3(blocks), dim3(threads), 0, stream, x, y, out, n);
  return static_cast<int>(hipGetLastError());
}

// Vision LayerNorm: (x - mean) / sqrt(var + eps) * w + b, row-wise.
__global__ void evie_layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t rows,
    int64_t hidden,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float* xr = x + row * hidden;
  float acc = 0.0f;
  float acc2 = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    acc += xr[i];
    acc2 += xr[i] * xr[i];
  }
  __shared__ float p1[256];
  __shared__ float p2[256];
  p1[threadIdx.x] = acc;
  p2[threadIdx.x] = acc2;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) {
      p1[threadIdx.x] += p1[threadIdx.x + stride];
      p2[threadIdx.x] += p2[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float s1 = p1[0];
    float s2 = p2[0];
    const float mean = s1 / static_cast<float>(hidden);
    const float var = s2 / static_cast<float>(hidden) - mean * mean;
    const float inv = rsqrtf(var + eps);
    float* orow = out + row * hidden;
    for (int64_t i = 0; i < hidden; ++i) {
      orow[i] = (xr[i] - mean) * inv * w[i] + b[i];
    }
  }
}

extern "C" int hipengine_evie_layernorm_f32(
    const float* x,
    const float* w,
    const float* b,
    float* out,
    int64_t rows,
    int64_t hidden,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      evie_layernorm_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(256),
      0,
      stream,
      x, w, b, out, rows, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

// SwiGLU: gate = silu(gate) * up (in place on gate).
__global__ void evie_silu_mul_kernel(
    float* __restrict__ gate,
    const float* __restrict__ up,
    int64_t n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    gate[idx] = evie_silu_f32(gate[idx]) * up[idx];
  }
}

extern "C" int hipengine_evie_silu_mul_f32(
    float* gate,
    const float* up,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_silu_mul_kernel, dim3(blocks), dim3(threads), 0, stream,
      gate, up, n);
  return static_cast<int>(hipGetLastError());
}

// out = gelu_tanh(x) (vision MLP).
__global__ void evie_gelu_tanh_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float v = x[idx];
    const float inner =
        0.7978845608028654f * (v + 0.044715f * v * v * v);
    out[idx] = 0.5f * v * (1.0f + tanhf(inner));
  }
}

extern "C" int hipengine_evie_gelu_tanh_f32(
    const float* x,
    float* out,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_gelu_tanh_kernel, dim3(blocks), dim3(threads), 0, stream,
      x, out, n);
  return static_cast<int>(hipGetLastError());
}

// out = gelu_erf(x) (merger MLP).
__global__ void evie_gelu_erf_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] =
        0.5f * x[idx] * (1.0f + erff(x[idx] * 0.7071067811865476f));
  }
}

extern "C" int hipengine_evie_gelu_erf_f32(
    const float* x,
    float* out,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_gelu_erf_kernel, dim3(blocks), dim3(threads), 0, stream,
      x, out, n);
  return static_cast<int>(hipGetLastError());
}

// Gated attention output: out = out * sigmoid(gate).
__global__ void evie_sigmoid_mul_kernel(
    const float* __restrict__ gate,
    float* __restrict__ out,
    int64_t n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = out[idx] / (1.0f + expf(-gate[idx]));
  }
}

extern "C" int hipengine_evie_sigmoid_mul_f32(
    const float* gate,
    float* out,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_sigmoid_mul_kernel, dim3(blocks), dim3(threads), 0, stream,
      gate, out, n);
  return static_cast<int>(hipGetLastError());
}

// Per-row L2 normalize (MRL head): out = x / max(||x||, 1e-12).
__global__ void evie_l2norm_rows_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t rows,
    int64_t dim) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float* xr = x + row * dim;
  const float total = evie_row_sum_squares(xr, dim);
  if (threadIdx.x == 0) {
    const float norm = fmaxf(sqrtf(total), 1e-12f);
    float* orow = out + row * dim;
    for (int64_t i = 0; i < dim; ++i) {
      orow[i] = xr[i] / norm;
    }
  }
}

extern "C" int hipengine_evie_l2norm_rows_f32(
    const float* x,
    float* out,
    int64_t rows,
    int64_t dim,
    hipStream_t stream) {
  if (rows <= 0 || dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      evie_l2norm_rows_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(256),
      0,
      stream,
      x, out, rows, dim);
  return static_cast<int>(hipGetLastError());
}

// GDN gate prep: beta = sigmoid(b); decay = exp(-A * softplus(a + dt)),
// A = exp(A_log). Layouts: b/a (tokens, heads); A_log/dt (heads).
__global__ void evie_gdn_gates_kernel(
    const float* __restrict__ b,
    const float* __restrict__ a,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ beta,
    float* __restrict__ decay,
    int64_t tokens,
    int64_t heads) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = tokens * heads;
  if (idx >= n) {
    return;
  }
  const int64_t head = idx % heads;
  const float rate = expf(a_log[head]);
  beta[idx] = 1.0f / (1.0f + expf(-b[idx]));
  decay[idx] = expf(-rate * evie_softplus_f32(a[idx] + dt_bias[head]));
}

extern "C" int hipengine_evie_gdn_gates_f32(
    const float* b,
    const float* a,
    const float* a_log,
    const float* dt_bias,
    float* beta,
    float* decay,
    int64_t tokens,
    int64_t heads,
    hipStream_t stream) {
  if (tokens <= 0 || heads <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t n = tokens * heads;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_gdn_gates_kernel, dim3(blocks), dim3(threads), 0, stream,
      b, a, a_log, dt_bias, beta, decay, tokens, heads);
  return static_cast<int>(hipGetLastError());
}

// GDN q/k normalization: qn = l2norm(q) * q_scale, kn = l2norm(k).
// Layouts: q/k (tokens, heads, head_k_dim); eps inside the sqrt like FLA.
__global__ void evie_gdn_l2norm_scale_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float q_scale,
    int64_t rows,
    int64_t head_dim) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const float* qr = q + row * head_dim;
  const float* kr = k + row * head_dim;
  float sq = 0.0f;
  float sk = 0.0f;
  for (int64_t i = 0; i < head_dim; ++i) {
    sq += qr[i] * qr[i];
    sk += kr[i] * kr[i];
  }
  const float iq = rsqrtf(sq + 1e-6f) * q_scale;
  const float ik = rsqrtf(sk + 1e-6f);
  float* qo = q_out + row * head_dim;
  float* ko = k_out + row * head_dim;
  for (int64_t i = 0; i < head_dim; ++i) {
    qo[i] = qr[i] * iq;
    ko[i] = kr[i] * ik;
  }
}

extern "C" int hipengine_evie_gdn_l2norm_scale_f32(
    const float* q,
    const float* k,
    float* q_out,
    float* k_out,
    float q_scale,
    int64_t rows,
    int64_t head_dim,
    hipStream_t stream) {
  if (rows <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 64;
  const int blocks = static_cast<int>((rows + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_gdn_l2norm_scale_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      stream,
      q, k, q_out, k_out, q_scale, rows, head_dim);
  return static_cast<int>(hipGetLastError());
}

// mRoPE apply (rotate_half) on leading rotary dims.
// x: per-token rows of (heads, head_dim) with row stride row_stride floats;
// cos/sin: (tokens, rotary_dim) doubled form.
__global__ void evie_rope_kernel(
    float* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t row_stride) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = tokens * heads;
  if (idx >= n) {
    return;
  }
  const int64_t token = idx / heads;
  const int64_t row = token * row_stride + (idx % heads) * head_dim;
  const int64_t half = rotary_dim / 2;
  const float* crow = cos + token * rotary_dim;
  const float* srow = sin + token * rotary_dim;
  for (int64_t d = 0; d < half; ++d) {
    const float x1 = x[row + d];
    const float x2 = x[row + d + half];
    // rotate_half: first half x1*cos - x2*sin, second half x2*cos + x1*sin
    // (tables are doubled, so crow[d] == crow[d + half]).
    x[row + d] = x1 * crow[d] - x2 * srow[d];
    x[row + d + half] = x2 * crow[d + half] + x1 * srow[d + half];
  }
}

extern "C" int hipengine_evie_rope_f32(
    float* x,
    const float* cos,
    const float* sin,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t row_stride,
    hipStream_t stream) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0 || rotary_dim <= 0 ||
      rotary_dim > head_dim || (rotary_dim & 1) != 0 || row_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t n = tokens * heads;
  const int threads = 64;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_rope_kernel, dim3(blocks), dim3(threads), 0, stream,
      x, cos, sin, tokens, heads, head_dim, rotary_dim, row_stride);
  return static_cast<int>(hipGetLastError());
}

// Split the fused q_proj output (tokens, 16, 512) into contiguous
// q (tokens, 16, 256) and gate (tokens, 16, 256) buffers.
__global__ void evie_split_qgate_kernel(
    const float* __restrict__ src,
    float* __restrict__ q_dst,
    float* __restrict__ gate_dst,
    int64_t tokens) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = tokens * 16 * 256;
  if (idx >= n) {
    return;
  }
  const int64_t i = idx % 256;
  const int64_t head = (idx / 256) % 16;
  const int64_t token = idx / (256 * 16);
  const int64_t src_base = token * 8192 + head * 512;
  q_dst[idx] = src[src_base + i];
  gate_dst[idx] = src[src_base + 256 + i];
}

extern "C" int hipengine_evie_split_qgate_f32(
    const float* src,
    float* q_dst,
    float* gate_dst,
    int64_t tokens,
    hipStream_t stream) {
  if (tokens <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t n = tokens * 16 * 256;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_split_qgate_kernel, dim3(blocks), dim3(threads), 0, stream,
      src, q_dst, gate_dst, tokens);
  return static_cast<int>(hipGetLastError());
}

// Expand head planes into a contiguous (tokens, dst_heads, dim) layout,
// optionally repeating each source head `reps` times (GQA).
// src row: (src_stride,) floats; plane for head h at src_offset + h*dim.
__global__ void evie_expand_heads_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t tokens,
    int64_t src_offset,
    int64_t src_stride,
    int64_t src_heads,
    int64_t dim,
    int64_t reps) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t dst_heads = src_heads * reps;
  const int64_t n = tokens * dst_heads * dim;
  if (idx >= n) {
    return;
  }
  const int64_t i = idx % dim;
  const int64_t head = (idx / dim) % dst_heads;
  const int64_t token = idx / (dim * dst_heads);
  const int64_t src_head = head / reps;
  dst[idx] = src[token * src_stride + src_offset + src_head * dim + i];
}

extern "C" int hipengine_evie_expand_heads_f32(
    const float* src,
    float* dst,
    int64_t tokens,
    int64_t src_offset,
    int64_t src_stride,
    int64_t src_heads,
    int64_t dim,
    int64_t reps,
    hipStream_t stream) {
  if (tokens <= 0 || src_heads <= 0 || dim <= 0 || reps <= 0 ||
      src_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t n = tokens * src_heads * reps * dim;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_expand_heads_kernel, dim3(blocks), dim3(threads), 0, stream,
      src, dst, tokens, src_offset, src_stride, src_heads, dim, reps);
  return static_cast<int>(hipGetLastError());
}

// Gather one head plane into a transposed (dim, tokens) layout so the
// attention output GEMM can consume it as W (out_features, in_features).
__global__ void evie_gather_head_tp_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t tokens,
    int64_t src_offset,
    int64_t src_stride,
    int64_t dim) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = tokens * dim;
  if (idx >= n) {
    return;
  }
  const int64_t i = idx % dim;
  const int64_t token = idx / dim;
  dst[i * tokens + token] = src[token * src_stride + src_offset + i];
}

extern "C" int hipengine_evie_gather_head_tp_f32(
    const float* src,
    float* dst,
    int64_t tokens,
    int64_t src_offset,
    int64_t src_stride,
    int64_t dim,
    hipStream_t stream) {
  if (tokens <= 0 || dim <= 0 || src_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t n = tokens * dim;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_gather_head_tp_kernel, dim3(blocks), dim3(threads), 0, stream,
      src, dst, tokens, src_offset, src_stride, dim);
  return static_cast<int>(hipGetLastError());
}

// Scatter one contiguous (tokens, dim) plane back into a per-token row of
// stride dst_stride at plane offset head*dim.
__global__ void evie_scatter_head_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t tokens,
    int64_t dst_stride,
    int64_t head,
    int64_t dim) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = tokens * dim;
  if (idx >= n) {
    return;
  }
  const int64_t i = idx % dim;
  const int64_t token = idx / dim;
  dst[token * dst_stride + head * dim + i] = src[idx];
}

extern "C" int hipengine_evie_scatter_head_f32(
    const float* src,
    float* dst,
    int64_t tokens,
    int64_t dst_stride,
    int64_t head,
    int64_t dim,
    hipStream_t stream) {
  if (tokens <= 0 || dst_stride <= 0 || head < 0 || dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t n = tokens * dim;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_scatter_head_kernel, dim3(blocks), dim3(threads), 0, stream,
      src, dst, tokens, dst_stride, head, dim);
  return static_cast<int>(hipGetLastError());
}

// Row softmax over scores: (rows, cols) fp32, in place.
__global__ void evie_softmax_rows_kernel(
    float* __restrict__ scores,
    int64_t rows,
    int64_t cols,
    int64_t tokens,
    int64_t head_stride) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  float* sr =
      scores + (row / tokens) * head_stride + (row % tokens) * cols;
  // pass 1: parallel max
  float m = -INFINITY;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    m = fmaxf(m, sr[i]);
  }
  __shared__ float sm[256];
  sm[threadIdx.x] = m;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) {
      sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float mx = sm[0];
  // pass 2: parallel sum of exponentials (write normalized exponentials)
  float denom = 0.0f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float e = expf(sr[i] - mx);
    sr[i] = e;
    denom += e;
  }
  sm[threadIdx.x] = denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) {
      sm[threadIdx.x] += sm[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv = 1.0f / sm[0];
  // pass 3: parallel normalize
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    sr[i] *= inv;
  }
}

extern "C" int hipengine_evie_softmax_rows_f32(
    float* scores,
    int64_t rows,
    int64_t cols,
    int64_t tokens,
    int64_t head_stride,
    hipStream_t stream) {
  if (rows <= 0 || cols <= 0 || tokens <= 0 || head_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      evie_softmax_rows_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(256),
      0,
      stream,
      scores, rows, cols, tokens, head_stride);
  return static_cast<int>(hipGetLastError());
}

// Embedding lookup with visual-feature injection: for tokens whose id equals
// image_token_id, read the next row (in token order) from visual_features.
__global__ void evie_embed_lookup_kernel(
    const int64_t* __restrict__ ids,
    const float* __restrict__ table,
    const float* __restrict__ visual_features,
    float* __restrict__ out,
    int64_t tokens,
    int64_t hidden,
    int64_t image_token_id) {
  const int64_t token = blockIdx.x;
  if (token >= tokens) {
    return;
  }
  const int64_t id = ids[token];
  const float* src;
  if (id == image_token_id) {
    int64_t vi = 0;
    for (int64_t t = 0; t < token; ++t) {
      if (ids[t] == image_token_id) {
        ++vi;
      }
    }
    src = visual_features + vi * hidden;
  } else {
    src = table + id * hidden;
  }
  float* dst = out + token * hidden;
  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dst[i] = src[i];
  }
}

extern "C" int hipengine_evie_embed_lookup_f32(
    const int64_t* ids,
    const float* table,
    const float* visual_features,
    float* out,
    int64_t tokens,
    int64_t hidden,
    int64_t image_token_id,
    hipStream_t stream) {
  if (tokens <= 0 || hidden <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      evie_embed_lookup_kernel,
      dim3(static_cast<unsigned int>(tokens)),
      dim3(256),
      0,
      stream,
      ids, table, visual_features, out, tokens, hidden, image_token_id);
  return static_cast<int>(hipGetLastError());
}

// Attention-output residual for batched attention into a packed buffer:
// out[t, :] += attn[t, :] (no-op helper kept for API symmetry).
__global__ void evie_scaled_add_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    float scale,
    int64_t n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] += scale * x[idx];
  }
}

// out[t, :] += bias[:] broadcast over rows of width width.
__global__ void evie_add_bias_kernel(
    float* __restrict__ out,
    const float* __restrict__ bias,
    int64_t n,
    int64_t width) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] += bias[idx % width];
  }
}

extern "C" int hipengine_evie_add_bias_f32(
    float* out,
    const float* bias,
    int64_t n,
    int64_t width,
    hipStream_t stream) {
  if (n <= 0 || width <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_add_bias_kernel, dim3(blocks), dim3(threads), 0, stream,
      out, bias, n, width);
  return static_cast<int>(hipGetLastError());
}

// out = x * scale (elementwise).
__global__ void evie_scale_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    float scale,
    int64_t n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = x[idx] * scale;
  }
}

extern "C" int hipengine_evie_scale_f32(
    const float* x,
    float* out,
    float scale,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_scale_kernel, dim3(blocks), dim3(threads), 0, stream,
      x, out, scale, n);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_evie_scaled_add_f32(
    const float* x,
    float* out,
    float scale,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_scaled_add_kernel, dim3(blocks), dim3(threads), 0, stream,
      x, out, scale, n);
  return static_cast<int>(hipGetLastError());
}


// GDN q/k normalization with head repeat: reads q/k (tokens, kv_heads, dim)
// and writes q_out/k_out (tokens, v_heads, dim) where v_heads = 2*kv_heads
// (each kv head serves two consecutive v heads), with FLA l2norm and the
// query scale applied.
__global__ void evie_gdn_l2norm_scale_repeat_kernel(
    const float* __restrict__ src,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float q_scale,
    int64_t tokens,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t src_stride,
    int64_t k_offset) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t rows = tokens * kv_heads;
  if (idx >= rows) {
    return;
  }
  const int64_t token = idx / kv_heads;
  const int64_t head = idx % kv_heads;
  const float* qr = src + token * src_stride + head * head_dim;
  const float* kr = src + token * src_stride + k_offset + head * head_dim;
  float sq = 0.0f;
  float sk = 0.0f;
  for (int64_t i = 0; i < head_dim; ++i) {
    sq += qr[i] * qr[i];
    sk += kr[i] * kr[i];
  }
  const float iq = rsqrtf(sq + 1e-6f) * q_scale;
  const float ik = rsqrtf(sk + 1e-6f);
  const int64_t out_base = (token * kv_heads + head) * 2 * head_dim;
  for (int rep = 0; rep < 2; ++rep) {
    float* qo = q_out + out_base + rep * head_dim;
    float* ko = k_out + out_base + rep * head_dim;
    for (int64_t i = 0; i < head_dim; ++i) {
      qo[i] = qr[i] * iq;
      ko[i] = kr[i] * ik;
    }
  }
}

extern "C" int hipengine_evie_gdn_l2norm_scale_repeat_f32(
    const float* src,
    float* q_out,
    float* k_out,
    float q_scale,
    int64_t tokens,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t src_stride,
    int64_t k_offset,
    hipStream_t stream) {
  if (tokens <= 0 || kv_heads <= 0 || head_dim <= 0 || src_stride <= 0 ||
      k_offset < 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t rows = tokens * kv_heads;
  const int threads = 64;
  const int blocks = static_cast<int>((rows + threads - 1) / threads);
  hipLaunchKernelGGL(
      evie_gdn_l2norm_scale_repeat_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      stream,
      src, q_out, k_out, q_scale, tokens, kv_heads, head_dim, src_stride,
      k_offset);
  return static_cast<int>(hipGetLastError());
}

// GDN RMSNormGated: out = rms(x) * w * silu(z), per v-head over head_v_dim.
// x/z: (tokens, v_heads * head_v_dim); w: (head_v_dim,).
__global__ void evie_gdn_rmsnorm_gate_kernel(
    const float* __restrict__ x,
    const float* __restrict__ z,
    const float* __restrict__ w,
    float* __restrict__ out,
    int64_t rows,
    int64_t head_dim,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float* xr = x + row * head_dim;
  float acc = 0.0f;
  for (int64_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
    acc += xr[i] * xr[i];
  }
  __shared__ float partial[256];
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float inv = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
    const float* zr = z + row * head_dim;
    float* orow = out + row * head_dim;
    for (int64_t i = 0; i < head_dim; ++i) {
      const float g = zr[i] / (1.0f + expf(-zr[i]));
      orow[i] = xr[i] * inv * w[i] * g;
    }
  }
}

extern "C" int hipengine_evie_gdn_rmsnorm_gate_f32(
    const float* x,
    const float* z,
    const float* w,
    float* out,
    int64_t rows,
    int64_t head_dim,
    float eps,
    hipStream_t stream) {
  if (rows <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      evie_gdn_rmsnorm_gate_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(256),
      0,
      stream,
      x, z, w, out, rows, head_dim, eps);
  return static_cast<int>(hipGetLastError());
}

// ---- fp16 production-path helpers -------------------------------------------

__global__ void evie_cast_f32_to_f16_kernel(
    const float* __restrict__ x,
    __half* __restrict__ out,
    int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = __float2half(x[i]);
  }
}

extern "C" int hipengine_evie_cast_f32_to_f16(
    const float* x,
    void* out,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (n + 255) / 256;
  hipLaunchKernelGGL(
      evie_cast_f32_to_f16_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(256),
      0,
      stream,
      x, static_cast<__half*>(out), n);
  return static_cast<int>(hipGetLastError());
}

__global__ void evie_cast_f16_to_f32_kernel(
    const __half* __restrict__ x,
    float* __restrict__ out,
    int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = __half2float(x[i]);
  }
}

extern "C" int hipengine_evie_cast_f16_to_f32(
    const void* x,
    float* out,
    int64_t n,
    hipStream_t stream) {
  if (n <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t blocks = (n + 255) / 256;
  hipLaunchKernelGGL(
      evie_cast_f16_to_f32_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(256),
      0,
      stream,
      static_cast<const __half*>(x), out, n);
  return static_cast<int>(hipGetLastError());
}

// ---- fp16 attention softmax (production path) ---------------------------------

__global__ void evie_scale_softmax_rows_f16_kernel(
    __half* __restrict__ scores,
    float scale,
    int64_t rows,
    int64_t cols,
    int64_t tokens,
    int64_t head_stride) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  __half* sr =
      scores + (row / tokens) * head_stride + (row % tokens) * cols;
  float m = -INFINITY;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    m = fmaxf(m, __half2float(sr[i]) * scale);
  }
  __shared__ float sm[256];
  sm[threadIdx.x] = m;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) {
      sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float mx = sm[0];
  float denom = 0.0f;
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float e = expf(__half2float(sr[i]) * scale - mx);
    sr[i] = __float2half(e);
    denom += e;
  }
  sm[threadIdx.x] = denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) {
      sm[threadIdx.x] += sm[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv = 1.0f / sm[0];
  for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    sr[i] = __float2half(__half2float(sr[i]) * inv);
  }
}

extern "C" int hipengine_evie_scale_softmax_rows_f16(
    void* scores,
    float scale,
    int64_t rows,
    int64_t cols,
    int64_t tokens,
    int64_t head_stride,
    hipStream_t stream) {
  if (rows <= 0 || cols <= 0 || tokens <= 0 || head_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      evie_scale_softmax_rows_f16_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(256),
      0,
      stream,
      static_cast<__half*>(scores), scale, rows, cols, tokens, head_stride);
  return static_cast<int>(hipGetLastError());
}

// ---- fp16 attention layout kernels -------------------------------------------

__global__ void evie_gather_qkv_f16_kernel(
    const float* __restrict__ x,
    __half* __restrict__ out,
    int64_t tokens,
    int64_t row_stride,
    int64_t heads,
    int64_t head_dim,
    int64_t head_stride) {
  const int64_t plane_total = heads * head_stride;
  const int64_t total = 3 * plane_total;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    const int64_t plane = idx / plane_total;
    int64_t rem = idx % plane_total;
    const int64_t h = rem / head_stride;
    rem %= head_stride;
    const int64_t t = rem / head_dim;
    const int64_t i = rem % head_dim;
    if (t < tokens) {
      out[idx] = __float2half(
          x[t * row_stride + plane * heads * head_dim + h * head_dim + i]);
    }
  }
}

extern "C" int hipengine_evie_gather_qkv_f16(
    const float* x,
    void* out,
    int64_t tokens,
    int64_t row_stride,
    int64_t heads,
    int64_t head_dim,
    int64_t head_stride,
    hipStream_t stream) {
  if (tokens <= 0 || row_stride <= 0 || heads <= 0 || head_dim <= 0
      || head_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = 3 * heads * head_stride;
  const int64_t blocks = (total + 255) / 256;
  hipLaunchKernelGGL(
      evie_gather_qkv_f16_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(256),
      0,
      stream,
      x, static_cast<__half*>(out), tokens, row_stride, heads, head_dim,
      head_stride);
  return static_cast<int>(hipGetLastError());
}

__global__ void evie_scatter_heads_f16_kernel(
    const __half* __restrict__ in,
    float* __restrict__ out,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    int64_t head_stride) {
  const int64_t total = heads * tokens * head_dim;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    int64_t rem = idx;
    const int64_t h = rem / (tokens * head_dim);
    rem %= tokens * head_dim;
    const int64_t t = rem / head_dim;
    const int64_t i = rem % head_dim;
    out[t * heads * head_dim + h * head_dim + i] =
        __half2float(in[h * head_stride + t * head_dim + i]);
  }
}

extern "C" int hipengine_evie_scatter_heads_f16(
    const void* in,
    float* out,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    int64_t head_stride,
    hipStream_t stream) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0 || head_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = heads * tokens * head_dim;
  const int64_t blocks = (total + 255) / 256;
  hipLaunchKernelGGL(
      evie_scatter_heads_f16_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(256),
      0,
      stream,
      static_cast<const __half*>(in), out, tokens, heads, head_dim,
      head_stride);
  return static_cast<int>(hipGetLastError());
}

__global__ void evie_cast_f16_to_f32_strided_kernel(
    const __half* __restrict__ x,
    float* __restrict__ out,
    int64_t rows,
    int64_t cols,
    int64_t out_stride) {
  const int64_t total = rows * cols;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    const int64_t r = idx / cols;
    const int64_t c = idx % cols;
    out[r * out_stride + c] = __half2float(x[idx]);
  }
}

extern "C" int hipengine_evie_cast_f16_to_f32_strided(
    const void* x,
    float* out,
    int64_t rows,
    int64_t cols,
    int64_t out_stride,
    hipStream_t stream) {
  if (rows <= 0 || cols <= 0 || out_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * cols;
  const int64_t blocks = (total + 255) / 256;
  hipLaunchKernelGGL(
      evie_cast_f16_to_f32_strided_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(256),
      0,
      stream,
      static_cast<const __half*>(x), out, rows, cols, out_stride);
  return static_cast<int>(hipGetLastError());
}

__global__ void evie_copy_f32_strided_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t rows,
    int64_t cols,
    int64_t out_stride) {
  const int64_t total = rows * cols;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += (int64_t)gridDim.x * blockDim.x) {
    const int64_t r = idx / cols;
    const int64_t c = idx % cols;
    out[r * out_stride + c] = x[idx];
  }
}

extern "C" int hipengine_evie_cast_f32_to_f32_strided(
    const float* x,
    float* out,
    int64_t rows,
    int64_t cols,
    int64_t out_stride,
    hipStream_t stream) {
  if (rows <= 0 || cols <= 0 || out_stride <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  const int64_t total = rows * cols;
  const int64_t blocks = (total + 255) / 256;
  hipLaunchKernelGGL(
      evie_copy_f32_strided_kernel,
      dim3(static_cast<unsigned int>(blocks)),
      dim3(256),
      0,
      stream,
      x, out, rows, cols, out_stride);
  return static_cast<int>(hipGetLastError());
}
