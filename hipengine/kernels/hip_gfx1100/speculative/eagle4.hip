// EAGLE-4 drafter kernels (clean-room): RoPE without q/k-norm and top-K retrieve.
//
// The EAGLE-4 single-layer drafter uses standard Llama MHA (no q/k normalization),
// so the DFlash rmsnorm+rotary kernels do not apply; eagle4_rope rotates q/k only.
// The efficient LM head retrieves top-K candidates from the low-rank score estimate
// (eagle4_topk_rows); the full-rank shortlist scoring reuses the DFlash dense kernel.
// Backbone fusion/norms/MLP/attention reuse the DFlash primitives.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

__device__ inline float bf16_to_f32(uint16_t v) {
  return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

// Split-half RoPE on q or k, no normalization. x: [rows, n_heads, head_dim] f32,
// rotated in place. positions: [rows] i32. One workgroup per (row, head); lanes
// stride over head_dim/2 pairs.
__global__ void eagle4_rope_kernel(
    float* __restrict__ x, const int* __restrict__ positions,
    int64_t rows, int n_heads, int head_dim, float theta) {
  const int64_t row = blockIdx.x;
  const int head = blockIdx.y;
  if (row >= rows || head >= n_heads) return;
  const int half = head_dim / 2;
  float* v = x + (row * n_heads + head) * head_dim;
  float pos = static_cast<float>(positions[row]);
  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    float inv_freq = 1.0f / __powf(theta, static_cast<float>(d) / static_cast<float>(half));
    float angle = pos * inv_freq;
    float c = __cosf(angle), s = __sinf(angle);
    float x1 = v[d], x2 = v[d + half];
    v[d] = x1 * c - x2 * s;
    v[d + half] = x1 * s + x2 * c;
  }
}

// Per-row top-K ids over f32 scores [rows, vocab] (descending). One workgroup per
// row; iterative block-max extraction over a WRITABLE working copy of the scores
// (the selected element is set to -inf each round). K <= 128. out_ids: [rows, K]
// i32. scores_work is modified in place; pass a scratch copy of the estimates.
// Two-stage parallel top-K. The naive single-block repeated-argmax serialized K
// full-vocab passes in one workgroup (~14 ms at V=248320, K=128). Stage 1 splits
// the vocab into disjoint chunks, one workgroup per chunk, each finding its local
// top-K (disjoint chunks make the -inf marking race-free); stage 2 selects the
// global top-K from the compact G*K candidates. The selected SET equals repeated
// global argmax (the global top-K lies in the union of chunk top-Ks); only
// boundary tie order can differ, which the downstream full-lm_head re-score +
// argmax makes irrelevant.
__global__ void eagle4_topk_stage1_kernel(
    float* __restrict__ scores, float* __restrict__ out_vals,
    int* __restrict__ out_idxs, int64_t rows, int64_t vocab, int K,
    int64_t chunk) {
  const int64_t row = blockIdx.y;
  const int64_t block = blockIdx.x;
  if (row >= rows) return;
  const int64_t begin = block * chunk;
  const int64_t end = min(begin + chunk, vocab);
  float* s = scores + row * vocab;
  __shared__ float red_val[32];
  __shared__ int red_idx[32];
  const int nwarps = (blockDim.x + 31) / 32;
  float* ov = out_vals + (row * gridDim.x + block) * K;
  int* oi = out_idxs + (row * gridDim.x + block) * K;
  for (int i = 0; i < K; i++) {
    float best = -3.402823466e38f;
    int besti = -1;
    for (int64_t idx = begin + threadIdx.x; idx < end; idx += blockDim.x) {
      float v = s[idx];
      if (v > best) { best = v; besti = (int)idx; }
    }
    for (int o = 16; o > 0; o >>= 1) {
      float ov2 = __shfl_down(best, o);
      int oi2 = __shfl_down(besti, o);
      if (ov2 > best) { best = ov2; besti = oi2; }
    }
    if ((threadIdx.x & 31) == 0) { red_val[threadIdx.x >> 5] = best; red_idx[threadIdx.x >> 5] = besti; }
    __syncthreads();
    if (threadIdx.x == 0) {
      float bv = -3.402823466e38f; int bi = -1;
      for (int w = 0; w < nwarps; w++) if (red_val[w] > bv) { bv = red_val[w]; bi = red_idx[w]; }
      ov[i] = bv; oi[i] = bi;
      if (bi >= 0) s[bi] = -3.402823466e38f;
    }
    __syncthreads();
  }
}

__global__ void eagle4_topk_stage2_kernel(
    float* __restrict__ in_vals, const int* __restrict__ in_idxs,
    int* __restrict__ out_ids, int64_t rows, int64_t ncand, int K) {
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  float* v = in_vals + row * ncand;
  const int* idx = in_idxs + row * ncand;
  __shared__ float red_val[32];
  __shared__ int red_idx[32];
  const int nwarps = (blockDim.x + 31) / 32;
  for (int i = 0; i < K; i++) {
    float best = -3.402823466e38f;
    int besti = -1;
    for (int64_t j = threadIdx.x; j < ncand; j += blockDim.x) {
      float val = v[j];
      if (val > best) { best = val; besti = (int)j; }
    }
    for (int o = 16; o > 0; o >>= 1) {
      float ov2 = __shfl_down(best, o);
      int oi2 = __shfl_down(besti, o);
      if (ov2 > best) { best = ov2; besti = oi2; }
    }
    if ((threadIdx.x & 31) == 0) { red_val[threadIdx.x >> 5] = best; red_idx[threadIdx.x >> 5] = besti; }
    __syncthreads();
    if (threadIdx.x == 0) {
      float bv = -3.402823466e38f; int bi = -1;
      for (int w = 0; w < nwarps; w++) if (red_val[w] > bv) { bv = red_val[w]; bi = red_idx[w]; }
      out_ids[row * K + i] = (bi >= 0) ? idx[bi] : -1;
      if (bi >= 0) v[bi] = -3.402823466e38f;
    }
    __syncthreads();
  }
}

// Concatenate two [rows, H] bf16 tensors along the feature dim -> [rows, 2H].
__global__ void eagle4_concat_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ b,
    uint16_t* __restrict__ out, int64_t rows, int64_t H) {
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const uint16_t* ar = a + row * H;
  const uint16_t* br = b + row * H;
  uint16_t* o = out + row * (2 * H);
  for (int64_t d = threadIdx.x; d < H; d += blockDim.x) {
    o[d] = ar[d];
    o[d + H] = br[d];
  }
}

__global__ void eagle4_concat3_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ b,
    const uint16_t* __restrict__ c, uint16_t* __restrict__ out,
    int64_t rows, int64_t H) {
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const uint16_t* ar = a + row * H;
  const uint16_t* br = b + row * H;
  const uint16_t* cr = c + row * H;
  uint16_t* o = out + row * (3 * H);
  for (int64_t d = threadIdx.x; d < H; d += blockDim.x) {
    o[d] = ar[d];
    o[d + H] = br[d];
    o[d + 2 * H] = cr[d];
  }
}

}  // namespace

extern "C" int hipengine_eagle4_concat(
    const void* a, const void* b, void* out, int64_t rows, int64_t H,
    hipStream_t stream) {
  if (rows <= 0 || H <= 0) return hipErrorInvalidValue;
  hipLaunchKernelGGL(eagle4_concat_kernel, dim3((unsigned)rows), dim3(256), 0,
                     stream, (const uint16_t*)a, (const uint16_t*)b,
                     (uint16_t*)out, rows, H);
  return hipGetLastError();
}

extern "C" int hipengine_eagle4_concat3(
    const void* a, const void* b, const void* c, void* out,
    int64_t rows, int64_t H, hipStream_t stream) {
  if (rows <= 0 || H <= 0) return hipErrorInvalidValue;
  hipLaunchKernelGGL(eagle4_concat3_kernel, dim3((unsigned)rows), dim3(256), 0,
                     stream, (const uint16_t*)a, (const uint16_t*)b,
                     (const uint16_t*)c, (uint16_t*)out, rows, H);
  return hipGetLastError();
}

extern "C" int hipengine_eagle4_rope(
    void* x, const void* positions, int64_t rows, int n_heads, int head_dim,
    float theta, hipStream_t stream) {
  if (rows <= 0) return hipErrorInvalidValue;
  dim3 grid((unsigned)rows, (unsigned)n_heads);
  hipLaunchKernelGGL(eagle4_rope_kernel, grid, dim3(128), 0, stream,
                     (float*)x, (const int*)positions, rows, n_heads, head_dim, theta);
  return hipGetLastError();
}

extern "C" int hipengine_eagle4_topk(
    void* scores_work, void* out_ids, int64_t rows, int64_t vocab, int K,
    hipStream_t stream) {
  if (rows <= 0 || K <= 0 || K > 128) return hipErrorInvalidValue;
  const int64_t chunk = 2048;
  const int64_t nblocks = (vocab + chunk - 1) / chunk;
  // Persistent single-stream scratch (the draft head runs sequentially on one
  // stream); grown on demand, reused across calls to avoid per-launch hipMalloc.
  static float* s_vals = nullptr;
  static int* s_idxs = nullptr;
  static int64_t s_cap = 0;
  const int64_t need = rows * nblocks * K;
  if (need > s_cap) {
    float* nv = nullptr; int* ni = nullptr;
    hipError_t e1 = hipMalloc(&nv, need * sizeof(float));
    hipError_t e2 = hipMalloc(&ni, need * sizeof(int));
    if (e1 != hipSuccess || e2 != hipSuccess) {
      if (nv) hipFree(nv); if (ni) hipFree(ni);
      return hipErrorOutOfMemory;
    }
    if (s_vals) hipFree(s_vals);
    if (s_idxs) hipFree(s_idxs);
    s_vals = nv; s_idxs = ni; s_cap = need;
  }
  hipLaunchKernelGGL(eagle4_topk_stage1_kernel,
                     dim3((unsigned)nblocks, (unsigned)rows), dim3(256), 0, stream,
                     (float*)scores_work, s_vals, s_idxs, rows, vocab, K, chunk);
  hipLaunchKernelGGL(eagle4_topk_stage2_kernel, dim3((unsigned)rows), dim3(256), 0,
                     stream, s_vals, s_idxs, (int*)out_ids, rows, nblocks * K, K);
  return hipGetLastError();
}