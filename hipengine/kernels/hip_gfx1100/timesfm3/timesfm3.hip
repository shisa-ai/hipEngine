// TimesFM 3.0 500M HIP kernels.
//
// Storage dtype is FP16 for activations (GEMM-adjacent), FP32 for the small
// norm/scale vectors; all internal math is FP32.  Semantics mirror
// google-research/timesfm src/timesfm3/torch/{model,transformer}.py: RMSNorm
// follows torch nn.RMSNorm (eps = finfo(float32).eps, NOT 2.5's 1e-6);
// attention scores are q.k * sqrt(head_dim) (SDPA with scale=sqrt(D)).
//
// This file adds the pieces that differ from the 2.5 family in
// kernels/hip_gfx1100/timesfm/timesfm.hip:
//   - qkv_norm_scatter with a runtime epsilon (2.5's fuses a hardcoded 1e-6)
//   - non-causal variate attention across the variate axis, with per-(batch,
//     variate) leading-mask counts; fully-masked query rows produce zeros
//     (CPU SDPA semantics)
//   - ReLU elementwise
// The sequence-attention path (rope, norms, scatter, flash) reuses the 2.5
// kernels directly; the fully-masked leading rows they compute with the 2.5
// uniform fallback are always outside the decoded horizon slice.
//
// Wrappers follow the hipEngine C ABI (return hipGetLastError).

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kVarBlock = 128;   // 4 warps x 32 lanes
constexpr int kVarMaxV = 32;     // spec.max_variates

using half_t = _Float16;

template <typename T>
__device__ __forceinline__ float ld(const T* p) { return (float)(*p); }
template <typename T>
__device__ __forceinline__ void st(T* p, float v) { *p = (T)v; }

// ---------------------------------------------------------------------------
// ReLU elementwise.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void relu_kernel(T* __restrict__ x, int64_t count) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const float v = ld(x + i);
    st(x + i, v > 0.0f ? v : 0.0f);
  }
}

// ---------------------------------------------------------------------------
// Fused rope + QK norm + per-dim scale + scatter (2.5's kernel with a runtime
// epsilon for the torch nn.RMSNorm of TimesFM 3.0).
// ---------------------------------------------------------------------------

__global__ void qkv_norm_scatter_eps_kernel(
    const half_t* __restrict__ qkv,
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
    float eps,
    half_t* __restrict__ qt,
    half_t* __restrict__ cache_k,
    half_t* __restrict__ cache_v) {
  // One warp per (b, n, h) head vector: warp-shuffle reductions.
  constexpr int WARPS = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t idx = (int64_t)blockIdx.x * WARPS + warp;
  if (idx >= (int64_t)B * H * N) return;
  const int n = (int)(idx % N);
  const int h = (int)((idx / N) % H);
  const int b = (int)(idx / ((int64_t)N * H));
  const half_t* row = qkv + (int64_t)b * N * patch_stride + (int64_t)n * patch_stride + h * D;
  const half_t* q = row;
  const half_t* k = row + (int64_t)H * D;
  const half_t* v = row + 2 * (int64_t)H * D;

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
  const float qinv = rsqrtf(qsq / (float)D + eps);
  const float kinv = rsqrtf(ksq / (float)D + eps);

  const int64_t qt_dst = (((int64_t)b * H + h) * N + n) * D;
  const int64_t kv_dst = (((int64_t)b * H + h) * S + (start + n)) * D;
#pragma unroll
  for (int rep = 0; rep < 2; ++rep) {
    const int j = lane + rep * 32;
    if (j < half) {
      qt[qt_dst + j] = (half_t)(rope[rep][0] * qinv * qscale[j]);
      qt[qt_dst + j + half] = (half_t)(rope[rep][1] * qinv * qscale[j + half]);
      cache_k[kv_dst + j] = (half_t)(rope[rep][2] * kinv * k_ln[j]);
      cache_k[kv_dst + j + half] = (half_t)(rope[rep][3] * kinv * k_ln[j + half]);
      cache_v[kv_dst + j] = v[j];
      cache_v[kv_dst + j + half] = v[j + half];
    }
  }
}

// ---------------------------------------------------------------------------
// Variate attention (non-causal, no RoPE).
//
// q/k/v are [B*V, N, H*D] row-major with row index ((b*V + v)*N + n); the
// per-(b,v) leading-mask counts front_masked[b*V + v] exclude a variate as a
// key at patch n when n < front_masked.  Scores are q.k * sqrt(D); query rows
// whose keys are all masked produce zeros (CPU SDPA semantics).  Output is
// written back into the same [B*V, N, H*D] layout.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void var_attention_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    const int* __restrict__ front_masked,
    T* __restrict__ out,
    int B,          // batch
    int V,          // variates (<= kVarMaxV)
    int N,          // patches
    int H,          // heads
    int D) {        // head dim
  // NOTE: D is a runtime parameter; stage via dynamically sized LDS.
  extern __shared__ float lds[];
  // Layout of lds: q [V][D], k [V][D], v [V][D], w [4 warps][V].
  float* q_s = lds;
  float* k_s = lds + (size_t)V * D;
  float* v_s = lds + 2 * (size_t)V * D;
  float* w_s = lds + 3 * (size_t)V * D;  // per-warp rows: w_s[warp*V + vk]

  const int b = blockIdx.x / N;
  const int n = blockIdx.x % N;
  const int h = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  // Cooperative load of q/k/v for this (b, n, h) into LDS.
  const size_t elems = (size_t)V * D;
  const T* base_q = q + (((size_t)b * V) * N + n) * (size_t)(H * D) + (size_t)h * D;
  const T* base_k = k + (((size_t)b * V) * N + n) * (size_t)(H * D) + (size_t)h * D;
  const T* base_v = v + (((size_t)b * V) * N + n) * (size_t)(H * D) + (size_t)h * D;
  for (size_t i = threadIdx.x; i < 3 * elems; i += blockDim.x) {
    const size_t r = i % elems;
    const int which = (int)(i / elems);
    const int vi = (int)(r / D);
    const int d = (int)(r % D);
    const T* src = which == 0 ? base_q + (size_t)vi * N * (H * D)
                : which == 1 ? base_k + (size_t)vi * N * (H * D)
                             : base_v + (size_t)vi * N * (H * D);
    const float val = ld(src + d);
    if (which == 0) q_s[vi * D + d] = val;
    else if (which == 1) k_s[vi * D + d] = val;
    else v_s[vi * D + d] = val;
  }
  __syncthreads();

  const float score_scale = sqrtf((float)D);

  // Each warp owns one query variate row per pass (4 warps per block).
  for (int vq = warp; vq < V; vq += (blockDim.x >> 5)) {
    // Lane l computes the score against key variate l (l < V).
    float score = -INFINITY;
    if (lane < V) {
      const int fm = front_masked[(size_t)b * V + lane];
      if (n >= fm) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
          dot += q_s[vq * D + d] * k_s[lane * D + d];
        }
        score = dot * score_scale;
      }
    }
    // Softmax over the row (warp reduction).
    float m = score;
#pragma unroll
    for (int w = 16; w > 0; w >>= 1) m = fmaxf(m, __shfl_xor(m, w));
    float e = (score == -INFINITY) ? 0.0f : expf(score - m);
    float sum = e;
#pragma unroll
    for (int w = 16; w > 0; w >>= 1) sum += __shfl_xor(sum, w);
    const bool all_masked = (sum <= 0.0f);
    const float wgt = all_masked ? 0.0f : e / sum;
    // Weights for the whole row live in lanes 0..V-1; publish to this
    // warp's private slice (warps run concurrently over query rows).
    if (lane < V) w_s[warp * V + lane] = wgt;
    __syncwarp();
    // Output: lane l owns dims l, l+32, ... (D = 80 -> up to 3 dims).
    for (int d = lane; d < D; d += 32) {
      float acc = 0.0f;
      if (!all_masked) {
        for (int vk = 0; vk < V; ++vk) {
          acc += w_s[warp * V + vk] * v_s[vk * D + d];
        }
      }
      T* dst = out + (((size_t)b * V + vq) * N + n) * (size_t)(H * D) + (size_t)h * D + d;
      st(dst, acc);
    }
    __syncwarp();
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// C ABI wrappers.
// ---------------------------------------------------------------------------

extern "C" int timesfm3_relu_f16(half_t* x, int64_t count, hipStream_t stream) {
  if (count <= 0) return (int)hipErrorInvalidValue;
  const int64_t blocks = (count + 255) / 256;
  hipLaunchKernelGGL(relu_kernel<half_t>, dim3((unsigned)blocks), dim3(256), 0, stream,
                     x, count);
  return (int)hipGetLastError();
}

extern "C" int timesfm3_relu_f32(float* x, int64_t count, hipStream_t stream) {
  if (count <= 0) return (int)hipErrorInvalidValue;
  const int64_t blocks = (count + 255) / 256;
  hipLaunchKernelGGL(relu_kernel<float>, dim3((unsigned)blocks), dim3(256), 0, stream,
                     x, count);
  return (int)hipGetLastError();
}

extern "C" int timesfm3_qkv_norm_scatter_f16(
    const half_t* qkv,
    const float* pos,
    const float* timescale,
    const float* qscale,
    const float* k_ln,
    int B, int N, int S, int H, int D, int patch_stride, int start,
    float eps,
    half_t* qt,
    half_t* cache_k,
    half_t* cache_v,
    hipStream_t stream) {
  if (B <= 0 || N <= 0 || S <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  const int64_t warps_needed = (int64_t)B * H * N;
  const unsigned blocks = (unsigned)((warps_needed + 7) / 8);
  hipLaunchKernelGGL(qkv_norm_scatter_eps_kernel, dim3(blocks), dim3(256), 0, stream,
                     qkv, pos, timescale, qscale, k_ln,
                     B, N, S, H, D, patch_stride, start, eps,
                     qt, cache_k, cache_v);
  return (int)hipGetLastError();
}

extern "C" int timesfm3_var_attention_f16(
    const half_t* q,
    const half_t* k,
    const half_t* v,
    const int* front_masked,
    half_t* out,
    int B, int V, int N, int H, int D,
    hipStream_t stream) {
  if (B <= 0 || V <= 0 || N <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  if (V > kVarMaxV) return (int)hipErrorInvalidValue;
  const size_t lds_bytes =
      3 * (size_t)V * D * sizeof(float) + 4 * (size_t)V * sizeof(float);
  dim3 grid((unsigned)((int64_t)B * N), (unsigned)H);
  hipLaunchKernelGGL(var_attention_kernel<half_t>, grid, dim3(kVarBlock),
                     lds_bytes, stream,
                     q, k, v, front_masked, out, B, V, N, H, D);
  return (int)hipGetLastError();
}

extern "C" int timesfm3_var_attention_f32(
    const float* q,
    const float* k,
    const float* v,
    const int* front_masked,
    float* out,
    int B, int V, int N, int H, int D,
    hipStream_t stream) {
  if (B <= 0 || V <= 0 || N <= 0 || H <= 0 || D <= 0) return (int)hipErrorInvalidValue;
  if (V > kVarMaxV) return (int)hipErrorInvalidValue;
  const size_t lds_bytes =
      3 * (size_t)V * D * sizeof(float) + 4 * (size_t)V * sizeof(float);
  dim3 grid((unsigned)((int64_t)B * N), (unsigned)H);
  hipLaunchKernelGGL(var_attention_kernel<float>, grid, dim3(kVarBlock),
                     lds_bytes, stream,
                     q, k, v, front_masked, out, B, V, N, H, D);
  return (int)hipGetLastError();
}
