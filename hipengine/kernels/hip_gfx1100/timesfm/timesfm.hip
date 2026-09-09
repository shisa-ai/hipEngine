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
constexpr int kThreadsHead = 128;  // D=80 head vectors waste 256-thread blocks

__device__ __forceinline__ float softplus_(float x) {
  return log1pf(expf(-fabsf(x))) + fmaxf(x, 0.0f);
}

using half_t = _Float16;
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

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
    const float* __restrict__ timescale,
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
  float angle = p / timescale[j];
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
  // One block per row; each thread keeps its slice in registers so the row
  // is read once and written once (S <= 8 * kThreads).
  __shared__ float partial[kThreads];
  constexpr int kMaxPerThread = 8;
  int64_t row = blockIdx.x;
  int64_t total = (int64_t)B * H * Q;
  if (row >= total) return;
  int q = row % Q;
  int b = row / ((int64_t)Q * H);
  int nm = num_masked[b];
  int off = q_offset[b];
  T* sr = scores + row * S;

  float vals[kMaxPerThread];
  int count = 0;
  float m = -INFINITY;
  for (int kk = threadIdx.x; kk < S; kk += blockDim.x) {
    // Masked entries are -INFINITY so expf(v - row_max) is exactly zero.
    float v = -INFINITY;
    if ((q + off >= kk) && (kk >= nm)) {
      v = ld(sr + kk);
      m = fmaxf(m, v);
    }
    vals[count++] = v;
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
  count = 0;
  for (int kk = threadIdx.x; kk < S; kk += blockDim.x) {
    float e = expf(vals[count++] - row_max);
    vals[count - 1] = e;
    sum += e;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  float sum_total = partial[0];
  // Rows with zero allowed keys (front-masked queries) mirror the reference
  // softmax over all-masked scores: uniform 1/S weights, never NaN.
  float inv = (sum_total > 0.0f) ? 1.0f / sum_total : 1.0f / (float)S;
  count = 0;
  for (int kk = threadIdx.x; kk < S; kk += blockDim.x) {
    float v = vals[count++];
    sr[kk] = to_half<T>((sum_total > 0.0f) ? v * inv : inv);
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
  // One block per output row (b, q); D/4 half4 columns per row.
  // Reads/writes are unit-stride in d within each head (80-element runs).
  const int row = blockIdx.y;
  if (row >= B * Q) return;
  const int col = threadIdx.x;  // half4 index within the row: H * (D/4)
  const int h = col / (D / 4);
  const int d4 = col % (D / 4);
  const int q = row % Q;
  const int b = row / Q;
  const int64_t s_off =
      (((int64_t)b * H + h) * Q + q) * D + d4 * 4;
  const int64_t d_off = (int64_t)row * (H * D) + h * D + d4 * 4;
  uint2 v = *reinterpret_cast<const uint2*>(src + s_off);
  *reinterpret_cast<uint2*>(dst + d_off) = v;
}


// Merged q-norm-transpose + k-norm-scatter + v-scatter (post-RoPE), one
// block of 128 threads per (b, n, h) head-vector: q gets query_ln RMSNorm +
// per-dim softplus scaling into the [B, H, Q, D] transpose, k gets key_ln
// RMSNorm into the head-major cache, v is copied straight through.
template <typename T>
__global__ void qkv_norm_scatter_kernel(
    const T* __restrict__ qkv,
    const float* __restrict__ pos,
    const float* __restrict__ timescale,
    const float* __restrict__ qscale,   // q_ln * log2(e)/sqrt(D) * softplus(per_dim)
    const float* __restrict__ k_ln,
    int B,
    int N,
    int S,
    int H,
    int D,
    int patch_stride,
    int start,
    T* __restrict__ qt,
    T* __restrict__ cache_k,
    T* __restrict__ cache_v) {
  // One warp per (b, n, h) head vector: warp-shuffle reductions, no block
  // syncs.  Lanes cover the D/2 RoPE pairs as j = lane and j = lane + 32.
  constexpr int WARPS = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t idx = (int64_t)blockIdx.x * WARPS + warp;
  if (idx >= (int64_t)B * H * N) return;
  const int n = (int)(idx % N);
  const int h = (int)((idx / N) % H);
  const int b = (int)(idx / ((int64_t)N * H));
  const T* row = qkv + (int64_t)b * N * patch_stride + (int64_t)n * patch_stride + h * D;
  const T* q = row;
  const T* k = row + (int64_t)H * D;
  const T* v = row + 2 * (int64_t)H * D;

  const float p = pos[(int64_t)b * N + n];
  const int half = D / 2;

  float rope[2][4];  // per rep: qv, qv2, kv, kv2
  float qsq = 0.0f, ksq = 0.0f;
#pragma unroll
  for (int rep = 0; rep < 2; ++rep) {
    const int j = lane + rep * 32;
    rope[rep][0] = rope[rep][1] = rope[rep][2] = rope[rep][3] = 0.0f;
    if (j < half) {
      float angle = p / timescale[j];
      float sin_a, cos_a;
      __sincosf(angle, &sin_a, &cos_a);
      const float q1 = ld(q + j), q2 = ld(q + j + half);
      const float k1 = ld(k + j), k2 = ld(k + j + half);
      rope[rep][0] = q1 * cos_a - q2 * sin_a;
      rope[rep][1] = q2 * cos_a + q1 * sin_a;
      rope[rep][2] = k1 * cos_a - k2 * sin_a;
      rope[rep][3] = k2 * cos_a + k1 * sin_a;
      qsq += rope[rep][0] * rope[rep][0] + rope[rep][1] * rope[rep][1];
      ksq += rope[rep][2] * rope[rep][2] + rope[rep][3] * rope[rep][3];
    }
  }
#pragma unroll
  for (int w = 16; w > 0; w >>= 1) {
    qsq += __shfl_xor(qsq, w);
    ksq += __shfl_xor(ksq, w);
  }
  const float qinv = rsqrtf(qsq / (float)D + 1.0e-6f);
  const float kinv = rsqrtf(ksq / (float)D + 1.0e-6f);

  const int64_t qt_dst = (((int64_t)b * H + h) * N + n) * D;
  const int64_t kv_dst = (((int64_t)b * H + h) * S + (start + n)) * D;
#pragma unroll
  for (int rep = 0; rep < 2; ++rep) {
    const int j = lane + rep * 32;
    if (j < half) {
      qt[qt_dst + j] = to_half<T>(rope[rep][0] * qinv * qscale[j]);
      qt[qt_dst + j + half] = to_half<T>(rope[rep][1] * qinv * qscale[j + half]);
      cache_k[kv_dst + j] = to_half<T>(rope[rep][2] * kinv * k_ln[j]);
      cache_k[kv_dst + j + half] = to_half<T>(rope[rep][3] * kinv * k_ln[j + half]);
      cache_v[kv_dst + j] = v[j];
      cache_v[kv_dst + j + half] = v[j + half];
    }
  }
}




// Short-query (AR) flash attention: one 128-thread block per (batch*head,
// q tile); the four wave32s each process a quarter of the kv tiles and the
// block merges the partial online-softmax accumulators through LDS.  The
// merge weight is (m == -inf) ? 1 : exp(m - M), which handles causally-empty
// chunks (l = 0) and uniform-fallback rows (all chunks m = -inf, weight 1,
// preserving the 1/S mean-of-V reference behaviour).
template <typename T>
__global__ void flash_short_kernel(
    const T* __restrict__ qt,        // [B, H, Q, D]
    const T* __restrict__ cache_k,   // [B, H, S, D]
    const T* __restrict__ cache_v,   // [B, H, S, D]
    const int* __restrict__ num_masked,
    const int* __restrict__ q_offset,
    T* __restrict__ out,            // [B, H, Q, D]
    int B,
    int Q,
    int S,
    int H,
    int D) {
  const int q0 = blockIdx.x * 16;
  const int bh = blockIdx.y;
  const int b = bh / H;
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;   // kv chunk id (0..3)
  const int col = lane & 15;
  const int half_id = lane >> 4;
  const int d_tiles = D / 16;
  const int nm = num_masked[b];
  const int off = q_offset[b];

  if (q0 >= Q) return;

  const T* q_base = qt + ((int64_t)bh * Q) * D;
  const T* k_base = cache_k + ((int64_t)bh * S) * D;
  const T* v_base = cache_v + ((int64_t)bh * S) * D;

  half16_t q_frag[5];
  const int q_row = q0 + (lane & 15);
  const bool q_valid = q_row < Q;
#pragma unroll
  for (int dt = 0; dt < 5; ++dt) {
    if (dt < d_tiles) {
      const T* src = q_base + (int64_t)(q_valid ? q_row : 0) * D + dt * 16;
#pragma unroll
      for (int i = 0; i < 16; ++i) q_frag[dt][i] = q_valid ? src[i] : (T)0;
    }
  }

  float m_run[8], l_run[8];
  float8_t acc_o[5];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    m_run[j] = -INFINITY;
    l_run[j] = 0.0f;
  }
#pragma unroll
  for (int dt = 0; dt < 5; ++dt) {
    acc_o[dt] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (dt >= d_tiles) break;
  }

  __shared__ half_t p_stage[4][2][16][17];
  extern __shared__ float merge_dyn[];  // [4][16][D+2]: O then (m, l) per row

  const int q_hi = min(q0 + 15, Q - 1);
  const int s_limit = min(S - 1, q_hi + off);
  const bool has_unif = (q0 == 0) && (off < nm);
  const int s_end = has_unif ? (S - 1) : s_limit;
  bool unif[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) unif[j] = (q0 + half_id + 2 * j + off) < nm;

  // Eight tiles per iteration (two per wave); the loop bound is uniform so
  // the __syncthreads stay convergent, and waves past s_end skip only their
  // own compute.  Each wave's two tiles share one reduction/staging round.
  for (int base = 0; base <= s_end; base += 128) {
    const int s0 = base + wave * 32;
    const bool mine = s0 <= s_end;
    const int s1 = s0 + 16;
    const bool mine1 = s1 <= s_end;
    float8_t sc0 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float8_t sc1 = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (mine) {
      const int k_row = s0 + col;
      for (int dt = 0; dt < d_tiles; ++dt) {
        half16_t b_frag;
        if (k_row < S) {
          const T* src = k_base + (int64_t)k_row * D + dt * 16;
#pragma unroll
          for (int i = 0; i < 16; ++i) b_frag[i] = src[i];
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i) b_frag[i] = (T)0;
        }
        sc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_frag[dt], b_frag, sc0);
      }
    }
    if (mine1) {
      const int k_row = s1 + col;
      for (int dt = 0; dt < d_tiles; ++dt) {
        half16_t b_frag;
        if (k_row < S) {
          const T* src = k_base + (int64_t)k_row * D + dt * 16;
#pragma unroll
          for (int i = 0; i < 16; ++i) b_frag[i] = src[i];
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i) b_frag[i] = (T)0;
        }
        sc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_frag[dt], b_frag, sc1);
      }
    } else if (mine) {
#pragma unroll
      for (int j = 0; j < 8; ++j) sc1[j] = -INFINITY;
    }
    float tile_max[8], alpha[8], m_new[8], tile_sum[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) tile_max[j] = -INFINITY;
#pragma unroll
    for (int j = 0; j < 8; ++j) tile_sum[j] = 0.0f;
    if (mine) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int r = half_id + 2 * j;
        const bool row_ok = (q0 + r < Q);
        const bool allowed0 = row_ok && (q0 + r + off >= s0 + col) && (s0 + col >= nm) && (s0 + col < S);
        if (!allowed0) sc0[j] = -INFINITY;
        tile_max[j] = fmaxf(tile_max[j], allowed0 ? sc0[j] : -INFINITY);
        const bool allowed1 = row_ok && mine1 && (q0 + r + off >= s1 + col) && (s1 + col >= nm) && (s1 + col < S);
        if (!allowed1) sc1[j] = -INFINITY;
        tile_max[j] = fmaxf(tile_max[j], allowed1 ? sc1[j] : -INFINITY);
      }
#pragma unroll
      for (int w = 8; w > 0; w >>= 1) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          tile_max[j] = fmaxf(tile_max[j], __shfl_xor(tile_max[j], w, 16));
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        m_new[j] = fmaxf(m_run[j], tile_max[j]);
        alpha[j] = (m_run[j] == -INFINITY) ? 1.0f : __expf(m_run[j] - m_new[j]);
        m_run[j] = m_new[j];
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const bool unif_now = unif[j] && (s0 + col < S);
        float e = unif_now ? 1.0f
                           : (sc0[j] == -INFINITY ? 0.0f : __expf(sc0[j] - m_new[j]));
        sc0[j] = e;
        tile_sum[j] += e;
        const bool unif_now1 = unif[j] && (s1 + col < S);
        float e1 = unif_now1 ? 1.0f
                             : (sc1[j] == -INFINITY ? 0.0f : __expf(sc1[j] - m_new[j]));
        sc1[j] = e1;
        tile_sum[j] += e1;
      }
#pragma unroll
      for (int w = 8; w > 0; w >>= 1) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          tile_sum[j] += __shfl_xor(tile_sum[j], w, 16);
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) l_run[j] = l_run[j] * alpha[j] + tile_sum[j];
#pragma unroll
      for (int dt = 0; dt < d_tiles; ++dt) {
#pragma unroll
        for (int j = 0; j < 8; ++j) acc_o[dt][j] *= alpha[j];
      }
    }
    __syncthreads();
    if (mine) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        p_stage[wave][0][half_id + 2 * j][col] = (half_t)sc0[j];
        p_stage[wave][1][half_id + 2 * j][col] = (half_t)sc1[j];
      }
    }
    __syncthreads();
    if (mine) {
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        if (t == 1 && !mine1) break;
        const int st = s0 + t * 16;
        half16_t p_frag;
#pragma unroll
        for (int i = 0; i < 16; ++i) p_frag[i] = p_stage[wave][t][lane & 15][i];
        for (int dt = 0; dt < d_tiles; ++dt) {
          half16_t b_frag;
          const T* vcol = v_base + (int64_t)st * D + dt * 16 + col;
#pragma unroll
          for (int i = 0; i < 16; ++i) b_frag[i] = (st + i < S) ? vcol[(int64_t)i * D] : (T)0;
          acc_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p_frag, b_frag, acc_o[dt]);
        }
      }
    }
  }

  // Stage this wave's partial accumulators and (m, l) through LDS.
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int r = half_id + 2 * j;
    if (q0 + r < Q) {
      for (int dt = 0; dt < d_tiles; ++dt) {
#pragma unroll
        for (int c2 = 0; c2 < 16; ++c2) {}  // placeholder, replaced below
        merge_dyn[(wave * 16 + r) * (D + 2) + dt * 16 + col] = acc_o[dt][j];
      }
      if (col == 0) {
        merge_dyn[(wave * 16 + r) * (D + 2) + D + 0] = m_run[j];
        merge_dyn[(wave * 16 + r) * (D + 2) + D + 1] = l_run[j];
      }
    }
  }
  __syncthreads();

  // Merge: rows assigned across the 128 threads (4 rows in typical AR use).
  T* out_base = out + ((int64_t)bh * Q) * D;
  for (int r = 0; r < 16; ++r) {
    if (q0 + r >= Q) break;
    // 128 threads stride over D.
    if (threadIdx.x < D) {
      const int d = threadIdx.x;
      float m_max = -INFINITY;
#pragma unroll
      for (int w = 0; w < 4; ++w)
        m_max = fmaxf(m_max, merge_dyn[(w * 16 + r) * (D + 2) + D]);
      if (m_max == -INFINITY) m_max = 0.0f;
      float acc = 0.0f, l = 0.0f;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const float m_w = merge_dyn[(w * 16 + r) * (D + 2) + D];
        const float wgt = (m_w == -INFINITY) ? 1.0f : expf(m_w - m_max);
        acc += wgt * merge_dyn[(w * 16 + r) * (D + 2) + d];
        l += wgt * merge_dyn[(w * 16 + r) * (D + 2) + D + 1];
      }
      out_base[(int64_t)(q0 + r) * D + d] = (T)(l > 0.0f ? acc / l : 0.0f);
    }
  }
}

// Flash-attention-style fused causal attention (FP16 production path).
//
// One wave32 block per (batch*head, 16-row q tile).  Uses the RDNA
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 intrinsic with the same
// fragment conventions as the PARO WMMA kernels:
//   - A fragment: lane loads 16 consecutive k elements of row (lane & 15)
//   - B fragment: lane loads 16 consecutive k elements of column (lane & 15)
//   - C fragment: lane holds rows (lane >> 4) + 2*j, column (lane & 15)
//
// Per-row softmax statistics live in the two 16-lane halves (a row's 16
// columns are exactly one half-group), so reductions use width-16 shuffles.
// The probability tile is staged through LDS between the QK and AV WMMA
// calls (C layout -> A layout).  Causality skips whole upper-triangle kv
// tiles; front-masked queries (no allowed keys) emit zeros, matching the
// uniform-1/S reference behaviour is NOT needed here because such rows'
// outputs are always discarded downstream; the reference-visible behaviour
// is preserved by the strict FP32 path.
//
// D must be a multiple of 16 (TimesFM: 80 = 5 tiles).
template <typename T, int TPB>
__global__ void flash_attention_kernel(
    const T* __restrict__ qt,        // [B, H, Q, D] normed+scaled queries
    const T* __restrict__ cache_k,   // [B, H, S, D]
    const T* __restrict__ cache_v,  // [B, H, S, D]
    const int* __restrict__ num_masked,
    const int* __restrict__ q_offset,
    T* __restrict__ out,            // [B, H, Q, D]
    int B,
    int Q,
    int S,
    int H,
    int D) {
  constexpr int WAVES = TPB / 32;
  const int q0 = (blockIdx.x * WAVES + (threadIdx.x >> 5)) * 16;
  const int bh = blockIdx.y;
  const int b = bh / H;
  const int h = bh % H;
  const int lane = threadIdx.x & 31;
  const int col = lane & 15;
  const int half_id = lane >> 4;
  const int d_tiles = D / 16;
  const int nm = num_masked[b];
  const int off = q_offset[b];

  const T* q_base = qt + ((int64_t)bh * Q) * D;
  const T* k_base = cache_k + ((int64_t)bh * S) * D;
  const T* v_base = cache_v + ((int64_t)bh * S) * D;

  // A fragments: full q row (lane & 15) across all d tiles.
  half16_t q_frag[5];
  const int q_row = q0 + (lane & 15);
  const bool q_valid = q_row < Q;
#pragma unroll
  for (int dt = 0; dt < 5; ++dt) {
    if (dt < d_tiles) {
      const T* src = q_base + (int64_t)(q_valid ? q_row : 0) * D + dt * 16;
#pragma unroll
      for (int i = 0; i < 16; ++i) q_frag[dt][i] = q_valid ? src[i] : (T)0;
    }
  }

  // Per-thread rows of the C fragment: rows half_id + 2*j (j = 0..7).
  float m_run[8], l_run[8];
  float8_t acc_o[5];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    m_run[j] = -INFINITY;
    l_run[j] = 0.0f;
  }
#pragma unroll
  for (int dt = 0; dt < 5; ++dt) {
    acc_o[dt] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (dt >= d_tiles) break;
  }

  __shared__ half_t p_stage[WAVES][4][16][17];

  // Highest query row in this tile (global) for the causal bound.
  const int q_hi = min(q0 + 15, Q - 1);
  const int s_limit = min(S - 1, q_hi + off);
  // Front-masked (all-masked) query rows exist only when off < num_masked,
  // i.e. only in the first q tile of the prefill.  The reference softmax
  // over an all-masked scores row is uniform 1/S, so those rows must average
  // V over the *whole* cache — extend this tile's kv loop to S accordingly.
  const bool has_unif = (q0 == 0) && (off < nm);
  const int s_end = has_unif ? (S - 1) : s_limit;
  if (q0 >= Q) return;
  // Per C-fragment row: uniform fallback (all keys masked) rows.
  bool unif[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) unif[j] = (q0 + half_id + 2 * j + off) < nm;

  // Four kv tiles per round: the max/sum reductions and the LDS staging
  // round are amortized over four tiles (two __syncthreads per 64 keys).
  for (int s0 = 0; s0 <= s_end; s0 += 64) {
    float8_t sc[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int st = s0 + t * 16;
      sc[t] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      if (st <= s_end) {
        const int k_row = st + col;
        for (int dt = 0; dt < d_tiles; ++dt) {
          half16_t b_frag;
          if (k_row < S) {
            const T* src = k_base + (int64_t)k_row * D + dt * 16;
#pragma unroll
            for (int i = 0; i < 16; ++i) b_frag[i] = src[i];
          } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) b_frag[i] = (T)0;
          }
          sc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_frag[dt], b_frag, sc[t]);
        }
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) sc[t][j] = -INFINITY;
      }
    }
    // Mask + row max over all four tiles.
    float tile_max[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) tile_max[j] = -INFINITY;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int r = half_id + 2 * j;
      const bool row_ok = (q0 + r < Q);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int st = s0 + t * 16;
        const int sv = st + col;
        const bool allowed = row_ok && (st <= s_end) && (q0 + r + off >= sv) && (sv >= nm) && (sv < S);
        if (!allowed) sc[t][j] = -INFINITY;
        tile_max[j] = fmaxf(tile_max[j], allowed ? sc[t][j] : -INFINITY);
      }
    }
    // Row max across the 16-lane half (the row's columns).
#pragma unroll
    for (int w = 8; w > 0; w >>= 1) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        tile_max[j] = fmaxf(tile_max[j], __shfl_xor(tile_max[j], w, 16));
      }
    }
    // Update running max / scale.
    float alpha[8], m_new[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      m_new[j] = fmaxf(m_run[j], tile_max[j]);
      alpha[j] = (m_run[j] == -INFINITY) ? 1.0f : __expf(m_run[j] - m_new[j]);
      m_run[j] = m_new[j];
    }
    // Exponentiate and row-sum over all four tiles.
    float tile_sum[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) tile_sum[j] = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int st = s0 + t * 16;
        const bool unif_now = unif[j] && (st + col < S);
        float e = unif_now ? 1.0f
                           : (sc[t][j] == -INFINITY ? 0.0f : __expf(sc[t][j] - m_new[j]));
        sc[t][j] = e;
        tile_sum[j] += e;
      }
    }
#pragma unroll
    for (int w = 8; w > 0; w >>= 1) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        tile_sum[j] += __shfl_xor(tile_sum[j], w, 16);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) l_run[j] = l_run[j] * alpha[j] + tile_sum[j];
#pragma unroll
    for (int dt = 0; dt < d_tiles; ++dt) {
#pragma unroll
      for (int j = 0; j < 8; ++j) acc_o[dt][j] *= alpha[j];
    }
    // Stage all four P tiles through LDS (C layout -> A layout).
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        p_stage[threadIdx.x >> 5][t][half_id + 2 * j][col] = (half_t)sc[t][j];
      }
    }
    __syncthreads();
    // AV WMMA per tile.
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int st = s0 + t * 16;
      if (st > s_end) break;
      half16_t p_frag;
#pragma unroll
      for (int i = 0; i < 16; ++i) p_frag[i] = p_stage[threadIdx.x >> 5][t][lane & 15][i];
      for (int dt = 0; dt < d_tiles; ++dt) {
        half16_t b_frag;
        const T* vcol = v_base + (int64_t)st * D + dt * 16 + col;
#pragma unroll
        for (int i = 0; i < 16; ++i) b_frag[i] = (st + i < S) ? vcol[(int64_t)i * D] : (T)0;
        acc_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p_frag, b_frag, acc_o[dt]);
      }
    }
  }

  // Epilogue: normalize and store.  All-masked rows (l == 0) emit zeros.
  T* out_base = out + ((int64_t)bh * Q) * D;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int r = half_id + 2 * j;
    const float inv = (q0 + r < Q && l_run[j] > 0.0f) ? 1.0f / l_run[j] : 0.0f;
    for (int dt = 0; dt < d_tiles; ++dt) {
      if (q0 + r < Q) {
        out_base[(int64_t)(q0 + r) * D + dt * 16 + col] = (half_t)(acc_o[dt][j] * inv);
      }
    }
  }
  (void)q_frag;
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
    half_t* x, const float* pos, const float* timescale,
    int B, int N, int H, int D, int patch_stride, int base,
    hipStream_t stream) {
  if (B <= 0 || N <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * N * H * (D / 2);
  hipLaunchKernelGGL(rope_kernel<half_t>, dim3(grid_for(total, kThreads)),
                     dim3(kThreads), 0, stream,
                     x, pos, timescale, B, N, H, D, patch_stride, base);
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


extern "C" int timesfm_flash_attention_f16(
    const half_t* qt, const half_t* k, const half_t* v,
    const int* num_masked, const int* q_offset, half_t* out,
    int B, int Q, int S, int H, int D, hipStream_t stream) {
  if (B <= 0 || Q <= 0 || S <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  if (D % 16 != 0) return (int)hipErrorInvalidValue;
  // Four q tiles per block (128 threads) for enough resident waves per CU on
  // long-prefill shapes; a single tile per 32-thread block for short (AR)
  // queries where four-way packing would idle three of four waves.
  if (Q >= 64) {
    dim3 grid((unsigned)((Q + 63) / 64), (unsigned)((int64_t)B * H));
    hipLaunchKernelGGL((flash_attention_kernel<half_t, 128>), grid, dim3(128), 0,
                       stream, qt, k, v, num_masked, q_offset, out,
                       B, Q, S, H, D);
  } else {
    // Short-query path: four wave32s per block split the kv tiles.
    dim3 grid((unsigned)((Q + 15) / 16), (unsigned)((int64_t)B * H));
    size_t shared = (size_t)4 * 16 * (D + 2) * sizeof(float);
    hipLaunchKernelGGL(flash_short_kernel<half_t>, grid, dim3(128), shared,
                       stream, qt, k, v, num_masked, q_offset, out,
                       B, Q, S, H, D);
  }
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
    float* x, const float* pos, const float* timescale,
    int B, int N, int H, int D, int patch_stride, int base,
    hipStream_t stream) {
  if (B <= 0 || N <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * N * H * (D / 2);
  hipLaunchKernelGGL(rope_kernel<float>, dim3(grid_for(total, kThreads)),
                     dim3(kThreads), 0, stream,
                     x, pos, timescale, B, N, H, D, patch_stride, base);
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
                     dim3(kThreadsHead), 0, stream,
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
                     dim3(kThreadsHead), 0, stream,
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
                     dim3(kThreadsHead), 0, stream,
                     qkv, B, N, S, H, D, patch_stride, start, cache_v);
  return (int)hipGetLastError();
}

extern "C" int timesfm_mask_softmax_f16(
    half_t* scores, const int* num_masked, const int* q_offset,
    int B, int Q, int S, int H, hipStream_t stream) {
  if (B <= 0 || Q <= 0 || S <= 0 || H <= 0) return (int)hipErrorInvalidValue;
  if (S > 8 * kThreads) return (int)hipErrorInvalidValue;
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
  if (D % 4 != 0) return (int)hipErrorInvalidValue;
  // One block per output row; H*(D/4) threads, each moving one half4.
  unsigned threads = (unsigned)(H * (D / 4));
  if (threads > 1024) return (int)hipErrorInvalidValue;
  dim3 grid(1, (unsigned)(B * Q));
  hipLaunchKernelGGL(transpose_heads_kernel<half_t>, grid, dim3(threads), 0,
                     stream, src, dst, B, Q, H, D);
  return (int)hipGetLastError();
}


extern "C" int timesfm_qkv_norm_scatter_f16(
    const half_t* qkv, const float* pos, const float* timescale,
    const float* qscale, const float* k_ln,
    int B, int N, int S, int H, int D, int patch_stride, int start,
    half_t* qt, half_t* cache_k, half_t* cache_v,
    hipStream_t stream) {
  if (B <= 0 || N <= 0 || S <= 0 || H <= 0 || D <= 0 || D > 128) return (int)hipErrorInvalidValue;
  int64_t total = (int64_t)B * H * N;
  hipLaunchKernelGGL(qkv_norm_scatter_kernel<half_t>,
                     dim3((unsigned)((total + 7) / 8)),
                     dim3(256), 0, stream,
                     qkv, pos, timescale, qscale, k_ln,
                     B, N, S, H, D, patch_stride, start,
                     qt, cache_k, cache_v);
  return (int)hipGetLastError();
}
