// TimesFM 2.5 200M fused elementwise / norm / RoPE / attention kernels.
//
// Storage dtype is FP16 for activations and KV caches (GEMM-adjacent), FP32
// for the small norm/scale vectors; all internal math is FP32.  Semantics
// mirror google-research/timesfm src/timesfm/torch/{transformer,
// normalization, util}.py: RMSNorm uses a multiplicative `scale` (no +1) with
// epsilon inside the rsqrt; RoPE is non-interleaved halves; attention is
// unscaled dot-product with the make_attn_mask causal+front-mask semantics.
//
// The QKV projection output stays packed [rows, 3*model_dims]; the q/k/v
// pieces are addressed through (patch_stride, base) so no repacking copy is
// needed between the GEMM and the fused kernels.
//
// Wrappers follow the hipEngine C ABI (return hipGetLastError).

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float softplus_(float x) {
  return log1pf(expf(-fabsf(x))) + fmaxf(x, 0.0f);
}

using half_t = _Float16;

template <typename T>
__device__ __forceinline__ float ld(const T* p) { return (float)(*p); }
template <typename T>
__device__ __forceinline__ void st(T* p, float v) { *p = (T)v; }
template <typename T>
__device__ __forceinline__ T to_half(float v) { return (T)v; }

// out[rows, features] = rmsnorm(x) * scale; one block per row.
template <typename T>
__global__ void rmsnorm_kernel(
    const T* __restrict__ x,
    const float* __restrict__ scale,
    T* __restrict__ out,
    int64_t rows,
    int features,
    float eps) {
  __shared__ float partial[kThreads];
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  const T* xr = x + row * features;
  T* orow = out + row * features;
  float sum = 0.0f;
  for (int j = threadIdx.x; j < features; j += blockDim.x) {
    float v = ld(xr + j);
    sum += v * v;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = rsqrtf(partial[0] / (float)features + eps);
  for (int j = threadIdx.x; j < features; j += blockDim.x) {
    orow[j] = to_half<T>(ld(xr + j) * inv * scale[j]);
  }
}

// out = rmsnorm(x) * scale + other; one block per row.
template <typename T>
__global__ void norm_add_kernel(
    const T* __restrict__ x,
    const T* __restrict__ other,
    const float* __restrict__ scale,
    T* __restrict__ out,
    int64_t rows,
    int features,
    float eps) {
  __shared__ float partial[kThreads];
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  const T* xr = x + row * features;
  const T* orow = other + row * features;
  T* dst = out + row * features;
  float sum = 0.0f;
  for (int j = threadIdx.x; j < features; j += blockDim.x) {
    float v = ld(xr + j);
    sum += v * v;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = rsqrtf(partial[0] / (float)features + eps);
  for (int j = threadIdx.x; j < features; j += blockDim.x) {
    dst[j] = to_half<T>(ld(xr + j) * inv * scale[j] + ld(orow + j));
  }
}

// x = x + bias, in place; one block per row.
template <typename T>
__global__ void bias_kernel(
    T* __restrict__ x,
    const float* __restrict__ bias,
    int64_t rows,
    int features) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  T* xr = x + row * features;
  for (int j = threadIdx.x; j < features; j += blockDim.x) {
    xr[j] = to_half<T>(ld(xr + j) + bias[j]);
  }
}

// x = swish(x + bias), in place; one block per row.
template <typename T>
__global__ void bias_swish_kernel(
    T* __restrict__ x,
    const float* __restrict__ bias,
    int64_t rows,
    int features) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;
  T* xr = x + row * features;
  for (int j = threadIdx.x; j < features; j += blockDim.x) {
    float v = ld(xr + j) + bias[j];
    xr[j] = to_half<T>(v / (1.0f + expf(-v)));
  }
}

// x = swish(x), in place.
template <typename T>
__global__ void swish_kernel(T* __restrict__ x, int64_t count) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= count) return;
  float v = ld(x + i);
  x[i] = to_half<T>(v / (1.0f + expf(-v)));
}

// out = a + b, elementwise.
template <typename T>
__global__ void add_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out,
    int64_t count) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= count) return;
  out[i] = to_half<T>(ld(a + i) + ld(b + i));
}

// RoPE (non-interleaved halves) on a packed-heads view:
// x[b, n, base + h*D + j], patch stride `patch_stride`.
template <typename T>
__global__ void rope_kernel(
    T* __restrict__ x,
    const float* __restrict__ pos,
    int B,
    int N,
    int H,
    int D,
    int patch_stride,
    int base) {
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t total = (int64_t)B * N * H * (D / 2);
  if (idx >= total) return;
  int half = D / 2;
  int j = idx % half;
  int h = (idx / half) % H;
  int n = (idx / ((int64_t)half * H)) % N;
  int b = idx / ((int64_t)half * H * N);
  float p = pos[(int64_t)b * N + n];
  float fraction = 2.0f * (float)j / (float)D;
  float timescale = powf(10000.0f, fraction);
  float angle = p / timescale;
  float sin_a = sinf(angle);
  float cos_a = cosf(angle);
  int64_t off = (int64_t)b * N * patch_stride + (int64_t)n * patch_stride + base + h * D;
  float x1 = ld(x + off + j);
  float x2 = ld(x + off + j + half);
  st(x + off + j, x1 * cos_a - x2 * sin_a);
  st(x + off + j + half, x2 * cos_a + x1 * sin_a);
}

// Per-head RMSNorm on a packed-heads view (q or k inside the QKV buffer).
template <typename T>
__global__ void head_rmsnorm_kernel(
    T* __restrict__ x,
    const float* __restrict__ scale,
    int B,
    int N,
    int H,
    int D,
    int patch_stride,
    int base,
    float eps) {
  __shared__ float partial[kThreads];
  int64_t idx = blockIdx.x;
  int64_t total = (int64_t)B * N * H;
  if (idx >= total) return;
  int h = idx % H;
  int n = (idx / H) % N;
  int b = idx / ((int64_t)H * N);
  T* xr = x + (int64_t)b * N * patch_stride + (int64_t)n * patch_stride + base + h * D;
  float sum = 0.0f;
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    float v = ld(xr + j);
    sum += v * v;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = rsqrtf(partial[0] / (float)D + eps);
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    xr[j] = to_half<T>(ld(xr + j) * inv * scale[j]);
  }
}

// Per-dim query scaling (softplus factors) on a packed-heads view.
template <typename T>
__global__ void head_perdim_scale_kernel(
    T* __restrict__ x,
    const float* __restrict__ scale,
    int B,
    int N,
    int H,
    int D,
    int patch_stride,
    int base) {
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t total = (int64_t)B * N * H;
  if (idx >= total) return;
  int h = idx % H;
  int n = (idx / H) % N;
  int b = idx / ((int64_t)H * N);
  T* xr = x + (int64_t)b * N * patch_stride + (int64_t)n * patch_stride + base + h * D;
  float factor = 1.442695041f / sqrtf((float)D);
  for (int j = 0; j < D; ++j) {
    xr[j] = to_half<T>(ld(xr + j) * factor * softplus_(scale[j]));
  }
}

// Scatter packed k/v into the contiguous FP16 decode cache at `start`.
template <typename T>
__global__ void scatter_kv_kernel(
    const T* __restrict__ qkv,
    T* __restrict__ cache_k,
    T* __restrict__ cache_v,
    int B,
    int N,
    int S,
    int H,
    int D,
    int patch_stride,
    int start) {
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t total = (int64_t)B * N * H * D;
  if (idx >= total) return;
  int d = idx % D;
  int h = (idx / D) % H;
  int n = (idx / ((int64_t)D * H)) % N;
  int b = idx / ((int64_t)D * H * N);
  int64_t src = (int64_t)b * N * patch_stride + (int64_t)n * patch_stride + h * D + d;
  int64_t dst = (((int64_t)b * S + (start + n)) * H + h) * D + d;
  cache_k[dst] = qkv[src + H * D];
  cache_v[dst] = qkv[src + 2 * H * D];
}

// Unscaled masked dot-product attention; one block per (b, h, q) row.
// q is read from the packed QKV GEMM output via (q_patch_stride, 0); k/v are
// the contiguous per-layer FP16 caches.  mask:
// (qi + q_offset[b] >= ki) && (ki >= num_masked[b]).
template <typename T>
__global__ void attention_kernel(
    const T* __restrict__ q_packed,
    const T* __restrict__ k,
    const T* __restrict__ v,
    const int* __restrict__ num_masked,
    const int* __restrict__ q_offset,
    T* __restrict__ out,
    int B,
    int Q,
    int S,
    int H,
    int D,
    int q_patch_stride) {
  extern __shared__ float smem[];
  float* scores = smem;          // S floats
  float* q_row = smem + S;       // D floats

  int q_idx = blockIdx.x % Q;
  int h = (blockIdx.x / Q) % H;
  int b = blockIdx.x / (Q * H);
  if (b >= B) return;

  const T* qv = q_packed + (int64_t)b * Q * q_patch_stride
      + (int64_t)q_idx * q_patch_stride + h * D;
  for (int j = threadIdx.x; j < D; j += blockDim.x) q_row[j] = ld(qv + j);
  __syncthreads();

  int nm = num_masked[b];
  int off = q_offset[b];

  for (int kk = threadIdx.x; kk < S; kk += blockDim.x) {
    float s = 0.0f;
    bool allowed = (q_idx + off >= kk) && (kk >= nm);
    if (allowed) {
      const T* kv = k + (((int64_t)b * S + kk) * H + h) * D;
      for (int j = 0; j < D; ++j) s += q_row[j] * ld(kv + j);
    } else {
      s = -3.402823466e38f / 2.0f;
    }
    scores[kk] = s;
  }
  __syncthreads();

  float m = -INFINITY;
  for (int kk = threadIdx.x; kk < S; kk += blockDim.x) m = fmaxf(m, scores[kk]);
  for (int w = 16; w > 0; w >>= 1) {
    m = fmaxf(m, __shfl_xor(m, w));
  }
  __shared__ float warp_max[kThreads / 32];
  if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    float mm = warp_max[0];
    for (int w = 1; w < kThreads / 32; ++w) mm = fmaxf(mm, warp_max[w]);
    warp_max[0] = mm;
  }
  __syncthreads();
  float row_max = warp_max[0];

  float sum = 0.0f;
  for (int kk = threadIdx.x; kk < S; kk += blockDim.x) {
    float e = expf(scores[kk] - row_max);
    scores[kk] = e;
    sum += e;
  }
  for (int w = 16; w > 0; w >>= 1) sum += __shfl_xor(sum, w);
  __shared__ float warp_sum[kThreads / 32];
  if ((threadIdx.x & 31) == 0) warp_sum[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float ss = warp_sum[0];
    for (int w = 1; w < kThreads / 32; ++w) ss += warp_sum[w];
    warp_sum[0] = ss;
  }
  __syncthreads();
  float inv_sum = 1.0f / warp_sum[0];

  if (threadIdx.x < D) {
    int d = threadIdx.x;
    float acc = 0.0f;
    const T* vb = v + (int64_t)b * S * H * D + (int64_t)h * D + d;
    for (int kk = 0; kk < S; ++kk) {
      acc += scores[kk] * ld(vb + (int64_t)kk * H * D);
    }
    out[(((int64_t)b * Q + q_idx) * H + h) * D + d] = to_half<T>(acc * inv_sum);
  }
}


// ---- Batched-GEMM attention path (FP16 production) ----

// Read packed q (post-RoPE), apply query_ln RMSNorm + per-dim softplus
// scaling, and write the transposed [B, H, Q, D] layout for the QK^T batched
// GEMM.
template <typename T>
__global__ void q_norm_transpose_kernel(
    const T* __restrict__ qkv,
    const float* __restrict__ q_ln,
    const float* __restrict__ per_dim,
    int B,
    int Q,
    int H,
    int D,
    int patch_stride,
    T* __restrict__ qt) {
  __shared__ float partial[kThreads];
  int64_t idx = blockIdx.x;  // b * H * Q + h * Q + q
  int64_t total = (int64_t)B * H * Q;
  if (idx >= total) return;
  int q = idx % Q;
  int h = (idx / Q) % H;
  int b = idx / ((int64_t)Q * H);
  const T* src = qkv + (int64_t)b * Q * patch_stride + (int64_t)q * patch_stride + h * D;
  float sum = 0.0f;
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    float v = ld(src + j);
    sum += v * v;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = rsqrtf(partial[0] / (float)D + 1.0e-6f);
  float factor = 1.442695041f / sqrtf((float)D);
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float v = ld(src + d) * inv * q_ln[d];
    v *= factor * softplus_(per_dim[d]);
    qt[(((int64_t)b * H + h) * Q + q) * D + d] = to_half<T>(v);
  }
}

// Read packed k (post-RoPE), apply key_ln RMSNorm, and write the head-major
// cache [B, H, S, D] at `start` (for the QK^T batched GEMM).
template <typename T>
__global__ void k_norm_scatter_kernel(
    const T* __restrict__ qkv,
    const float* __restrict__ k_ln,
    int B,
    int N,
    int S,
    int H,
    int D,
    int patch_stride,
    int start,
    T* __restrict__ cache_k) {
  __shared__ float partial[kThreads];
  int64_t idx = blockIdx.x;  // b * H * N + h * N + n
  int64_t total = (int64_t)B * H * N;
  if (idx >= total) return;
  int n = idx % N;
  int h = (idx / N) % H;
  int b = idx / ((int64_t)N * H);
  const T* src = qkv + (int64_t)b * N * patch_stride + (int64_t)n * patch_stride
      + (int64_t)H * D + h * D;
  float sum = 0.0f;
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    float v = ld(src + j);
    sum += v * v;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = rsqrtf(partial[0] / (float)D + 1.0e-6f);
  int64_t dst_base = (((int64_t)b * H + h) * S + (start + n)) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    cache_k[dst_base + d] = to_half<T>(ld(src + d) * inv * k_ln[d]);
  }
}

// Scatter packed v into the head-major cache [B, H, S, D] at `start`.
template <typename T>
__global__ void v_scatter_kernel(
    const T* __restrict__ qkv,
    int B,
    int N,
    int S,
    int H,
    int D,
    int patch_stride,
    int start,
    T* __restrict__ cache_v) {
  int64_t idx = blockIdx.x;  // b * H * N + h * N + n
  int64_t total = (int64_t)B * H * N;
  if (idx >= total) return;
  int n = idx % N;
  int h = (idx / N) % H;
  int b = idx / ((int64_t)N * H);
  const T* src = qkv + (int64_t)b * N * patch_stride + (int64_t)n * patch_stride
      + 2 * (int64_t)H * D + h * D;
  int64_t dst_base = (((int64_t)b * H + h) * S + (start + n)) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    cache_v[dst_base + d] = src[d];
  }
}

// In-place masked softmax over each row of scores [B, H, Q, S].
// mask: (q + q_offset[b] >= kv) && (kv >= num_masked[b]).
template <typename T>
__global__ void mask_softmax_kernel(
    T* __restrict__ scores,
    const int* __restrict__ num_masked,
    const int* __restrict__ q_offset,
    int B,
    int Q,
    int S,
    int H) {
  __shared__ float partial[kThreads];
  int64_t row = blockIdx.x;
  int64_t total = (int64_t)B * H * Q;
  if (row >= total) return;
  int q = row % Q;
  int b = row / ((int64_t)Q * H);
  int nm = num_masked[b];
  int off = q_offset[b];
  T* sr = scores + row * S;
  float m = -INFINITY;
  for (int kk = threadIdx.x; kk < S; kk += blockDim.x) {
    if ((q + off >= kk) && (kk >= nm)) m = fmaxf(m, ld(sr + kk));
  }
  partial[threadIdx.x] = m;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float row_max = partial[0];
  if (row_max == -INFINITY) row_max = 0.0f;
  float sum = 0.0f;
  for (int kk = threadIdx.x; kk < S; kk += blockDim.x) {
    float e = ((q + off >= kk) && (kk >= nm)) ? expf(ld(sr + kk) - row_max) : 0.0f;
    sr[kk] = to_half<T>(e);
    sum += e;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = 1.0f / partial[0];
  for (int kk = threadIdx.x; kk < S; kk += blockDim.x) {
    sr[kk] = to_half<T>(ld(sr + kk) * inv);
  }
}

// Transpose [B, H, Q, D] -> [B, Q, H * D] rows for the out-proj GEMM.
template <typename T>
__global__ void transpose_heads_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int B,
    int Q,
    int H,
    int D) {
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t total = (int64_t)B * H * Q * D;
  if (idx >= total) return;
  int d = idx % D;
  int q = (idx / D) % Q;
  int h = (idx / ((int64_t)D * Q)) % H;
  int b = idx / ((int64_t)D * Q * H);
  int64_t s = (((int64_t)b * H + h) * Q + q) * D + d;
  int64_t dd = ((int64_t)b * Q + q) * (H * D) + h * D + d;
  dst[dd] = src[s];
}

inline unsigned grid_for(int64_t count, int threads) {
  return (unsigned)((count + threads - 1) / threads);
}

}  // namespace

extern "C" int timesfm_rmsnorm_f16(
    const half_t* x, const float* scale, half_t* out,
    int64_t rows, int features, float eps, hipStream_t stream) {
  if (rows <= 0 || features <= 0) return (int)hipErrorInvalidValue;
  hipLaunchKernelGGL(rmsnorm_kernel<half_t>, dim3((unsigned)rows),
                     dim3(kThreads), 0, stream,
                     x, scale, out, rows, features, eps);
  return (int)hipGetLastError();
}

extern "C" int timesfm_norm_add_f16(
    const half_t* x, const half_t* other, const float* scale, half_t* out,
    int64_t rows, int features, float eps, hipStream_t stream) {
  if (rows <= 0 || features <= 0) return (int)hipErrorInvalidValue;
  hipLaunchKernelGGL(norm_add_kernel<half_t>, dim3((unsigned)rows),
                     dim3(kThreads), 0, stream,
                     x, other, scale, out, rows, features, eps);
  return (int)hipGetLastError();
}

extern "C" int timesfm_bias_f16(
    half_t* x, const float* bias, int64_t rows, int features,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0) return (int)hipErrorInvalidValue;
  hipLaunchKernelGGL(bias_kernel<half_t>, dim3((unsigned)rows),
                     dim3(kThreads), 0, stream, x, bias, rows, features);
  return (int)hipGetLastError();
}

extern "C" int timesfm_bias_swish_f16(
    half_t* x, const float* bias, int64_t rows, int features,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0) return (int)hipErrorInvalidValue;
  hipLaunchKernelGGL(bias_swish_kernel<half_t>, dim3((unsigned)rows),
                     dim3(kThreads), 0, stream,
                     x, bias, rows, features);
  return (int)hipGetLastError();
}

extern "C" int timesfm_swish_f16(half_t* x, int64_t count, hipStream_t stream) {
  if (count <= 0) return (int)hipErrorInvalidValue;
  hipLaunchKernelGGL(swish_kernel<half_t>, dim3(grid_for(count, kThreads)),
                     dim3(kThreads), 0, stream, x, count);
  return (int)hipGetLastError();
}

extern "C" int timesfm_add_f16(
    const half_t* a, const half_t* b, half_t* out, int64_t count,
    hipStream_t stream) {
  if (count <= 0) return (int)hipErrorInvalidValue;
  hipLaunchKernelGGL(add_kernel<half_t>, dim3(grid_for(count, kThreads)),
                     dim3(kThreads), 0, stream, a, b, out, count);
  return (int)hipGetLastError();
}

extern "C" int timesfm_rope_f16(
    half_t* x, const float* pos,
    int B, int N, int H, int D, int patch_stride, int base,
    hipStream_t stream) {
  if (B <= 0 || N <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * N * H * (D / 2);
  hipLaunchKernelGGL(rope_kernel<half_t>, dim3(grid_for(total, kThreads)),
                     dim3(kThreads), 0, stream,
                     x, pos, B, N, H, D, patch_stride, base);
  return (int)hipGetLastError();
}

extern "C" int timesfm_head_rmsnorm_f16(
    half_t* x, const float* scale,
    int B, int N, int H, int D, int patch_stride, int base, float eps,
    hipStream_t stream) {
  if (B <= 0 || N <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * N * H;
  hipLaunchKernelGGL(head_rmsnorm_kernel<half_t>, dim3((unsigned)total),
                     dim3(kThreads), 0, stream,
                     x, scale, B, N, H, D, patch_stride, base, eps);
  return (int)hipGetLastError();
}

extern "C" int timesfm_head_perdim_scale_f16(
    half_t* x, const float* scale,
    int B, int N, int H, int D, int patch_stride, int base,
    hipStream_t stream) {
  if (B <= 0 || N <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * N * H;
  hipLaunchKernelGGL(head_perdim_scale_kernel<half_t>, dim3(grid_for(total, kThreads)),
                     dim3(kThreads), 0, stream,
                     x, scale, B, N, H, D, patch_stride, base);
  return (int)hipGetLastError();
}

extern "C" int timesfm_scatter_kv_f16(
    const half_t* qkv, half_t* cache_k, half_t* cache_v,
    int B, int N, int S, int H, int D, int patch_stride, int start,
    hipStream_t stream) {
  if (B <= 0 || N <= 0 || S <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * N * H * D;
  hipLaunchKernelGGL(scatter_kv_kernel<half_t>, dim3(grid_for(total, kThreads)),
                     dim3(kThreads), 0, stream,
                     qkv, cache_k, cache_v, B, N, S, H, D, patch_stride, start);
  return (int)hipGetLastError();
}

extern "C" int timesfm_attention_f16(
    const half_t* q_packed, const half_t* k, const half_t* v,
    const int* num_masked, const int* q_offset, half_t* out,
    int B, int Q, int S, int H, int D, int q_patch_stride,
    hipStream_t stream) {
  if (B <= 0 || Q <= 0 || S <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  if (S > 8192) return (int)hipErrorInvalidValue;
  unsigned grid = (unsigned)((int64_t)B * Q * H);
  size_t shared = (size_t)(S + D) * sizeof(float);
  hipLaunchKernelGGL(attention_kernel<half_t>, dim3(grid), dim3(kThreads),
                     shared, stream,
                     q_packed, k, v, num_masked, q_offset, out,
                     B, Q, S, H, D, q_patch_stride);
  return (int)hipGetLastError();
}

extern "C" int timesfm_rmsnorm_f32(
    const float* x, const float* scale, float* out,
    int64_t rows, int features, float eps, hipStream_t stream) {
  if (rows <= 0 || features <= 0) return (int)hipErrorInvalidValue;
  hipLaunchKernelGGL(rmsnorm_kernel<float>, dim3((unsigned)rows),
                     dim3(kThreads), 0, stream,
                     x, scale, out, rows, features, eps);
  return (int)hipGetLastError();
}

extern "C" int timesfm_norm_add_f32(
    const float* x, const float* other, const float* scale, float* out,
    int64_t rows, int features, float eps, hipStream_t stream) {
  if (rows <= 0 || features <= 0) return (int)hipErrorInvalidValue;
  hipLaunchKernelGGL(norm_add_kernel<float>, dim3((unsigned)rows),
                     dim3(kThreads), 0, stream,
                     x, other, scale, out, rows, features, eps);
  return (int)hipGetLastError();
}

extern "C" int timesfm_bias_f32(
    float* x, const float* bias, int64_t rows, int features,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0) return (int)hipErrorInvalidValue;
  hipLaunchKernelGGL(bias_kernel<float>, dim3((unsigned)rows),
                     dim3(kThreads), 0, stream, x, bias, rows, features);
  return (int)hipGetLastError();
}

extern "C" int timesfm_bias_swish_f32(
    float* x, const float* bias, int64_t rows, int features,
    hipStream_t stream) {
  if (rows <= 0 || features <= 0) return (int)hipErrorInvalidValue;
  hipLaunchKernelGGL(bias_swish_kernel<float>, dim3((unsigned)rows),
                     dim3(kThreads), 0, stream,
                     x, bias, rows, features);
  return (int)hipGetLastError();
}

extern "C" int timesfm_swish_f32(float* x, int64_t count, hipStream_t stream) {
  if (count <= 0) return (int)hipErrorInvalidValue;
  hipLaunchKernelGGL(swish_kernel<float>, dim3(grid_for(count, kThreads)),
                     dim3(kThreads), 0, stream, x, count);
  return (int)hipGetLastError();
}

extern "C" int timesfm_add_f32(
    const float* a, const float* b, float* out, int64_t count,
    hipStream_t stream) {
  if (count <= 0) return (int)hipErrorInvalidValue;
  hipLaunchKernelGGL(add_kernel<float>, dim3(grid_for(count, kThreads)),
                     dim3(kThreads), 0, stream, a, b, out, count);
  return (int)hipGetLastError();
}

extern "C" int timesfm_rope_f32(
    float* x, const float* pos,
    int B, int N, int H, int D, int patch_stride, int base,
    hipStream_t stream) {
  if (B <= 0 || N <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * N * H * (D / 2);
  hipLaunchKernelGGL(rope_kernel<float>, dim3(grid_for(total, kThreads)),
                     dim3(kThreads), 0, stream,
                     x, pos, B, N, H, D, patch_stride, base);
  return (int)hipGetLastError();
}

extern "C" int timesfm_head_rmsnorm_f32(
    float* x, const float* scale,
    int B, int N, int H, int D, int patch_stride, int base, float eps,
    hipStream_t stream) {
  if (B <= 0 || N <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * N * H;
  hipLaunchKernelGGL(head_rmsnorm_kernel<float>, dim3((unsigned)total),
                     dim3(kThreads), 0, stream,
                     x, scale, B, N, H, D, patch_stride, base, eps);
  return (int)hipGetLastError();
}

extern "C" int timesfm_head_perdim_scale_f32(
    float* x, const float* scale,
    int B, int N, int H, int D, int patch_stride, int base,
    hipStream_t stream) {
  if (B <= 0 || N <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * N * H;
  hipLaunchKernelGGL(head_perdim_scale_kernel<float>, dim3(grid_for(total, kThreads)),
                     dim3(kThreads), 0, stream,
                     x, scale, B, N, H, D, patch_stride, base);
  return (int)hipGetLastError();
}

extern "C" int timesfm_scatter_kv_f32(
    const float* qkv, float* cache_k, float* cache_v,
    int B, int N, int S, int H, int D, int patch_stride, int start,
    hipStream_t stream) {
  if (B <= 0 || N <= 0 || S <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * N * H * D;
  hipLaunchKernelGGL(scatter_kv_kernel<float>, dim3(grid_for(total, kThreads)),
                     dim3(kThreads), 0, stream,
                     qkv, cache_k, cache_v, B, N, S, H, D, patch_stride, start);
  return (int)hipGetLastError();
}

extern "C" int timesfm_attention_f32(
    const float* q_packed, const float* k, const float* v,
    const int* num_masked, const int* q_offset, float* out,
    int B, int Q, int S, int H, int D, int q_patch_stride,
    hipStream_t stream) {
  if (B <= 0 || Q <= 0 || S <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  if (S > 8192) return (int)hipErrorInvalidValue;
  unsigned grid = (unsigned)((int64_t)B * Q * H);
  size_t shared = (size_t)(S + D) * sizeof(float);
  hipLaunchKernelGGL(attention_kernel<float>, dim3(grid), dim3(kThreads),
                     shared, stream,
                     q_packed, k, v, num_masked, q_offset, out,
                     B, Q, S, H, D, q_patch_stride);
  return (int)hipGetLastError();
}

extern "C" int timesfm_q_norm_transpose_f16(
    const half_t* qkv, const float* q_ln, const float* per_dim,
    int B, int Q, int H, int D, int patch_stride, half_t* qt,
    hipStream_t stream) {
  if (B <= 0 || Q <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * H * Q;
  hipLaunchKernelGGL(q_norm_transpose_kernel<half_t>, dim3((unsigned)total),
                     dim3(kThreads), 0, stream,
                     qkv, q_ln, per_dim, B, Q, H, D, patch_stride, qt);
  return (int)hipGetLastError();
}

extern "C" int timesfm_k_norm_scatter_f16(
    const half_t* qkv, const float* k_ln,
    int B, int N, int S, int H, int D, int patch_stride, int start,
    half_t* cache_k, hipStream_t stream) {
  if (B <= 0 || N <= 0 || S <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * H * N;
  hipLaunchKernelGGL(k_norm_scatter_kernel<half_t>, dim3((unsigned)total),
                     dim3(kThreads), 0, stream,
                     qkv, k_ln, B, N, S, H, D, patch_stride, start, cache_k);
  return (int)hipGetLastError();
}

extern "C" int timesfm_v_scatter_f16(
    const half_t* qkv,
    int B, int N, int S, int H, int D, int patch_stride, int start,
    half_t* cache_v, hipStream_t stream) {
  if (B <= 0 || N <= 0 || S <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * H * N;
  hipLaunchKernelGGL(v_scatter_kernel<half_t>, dim3((unsigned)total),
                     dim3(kThreads), 0, stream,
                     qkv, B, N, S, H, D, patch_stride, start, cache_v);
  return (int)hipGetLastError();
}

extern "C" int timesfm_mask_softmax_f16(
    half_t* scores, const int* num_masked, const int* q_offset,
    int B, int Q, int S, int H, hipStream_t stream) {
  if (B <= 0 || Q <= 0 || S <= 0 || H <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * H * Q;
  hipLaunchKernelGGL(mask_softmax_kernel<half_t>, dim3((unsigned)total),
                     dim3(kThreads), 0, stream,
                     scores, num_masked, q_offset, B, Q, S, H);
  return (int)hipGetLastError();
}

extern "C" int timesfm_transpose_heads_f16(
    const half_t* src, half_t* dst, int B, int Q, int H, int D,
    hipStream_t stream) {
  if (B <= 0 || Q <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * H * Q * D;
  hipLaunchKernelGGL(transpose_heads_kernel<half_t>, dim3(grid_for(total, kThreads)),
                     dim3(kThreads), 0, stream, src, dst, B, Q, H, D);
  return (int)hipGetLastError();
}
