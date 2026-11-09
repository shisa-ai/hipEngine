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
__global__ void eagle4_topk_kernel(
    float* __restrict__ scores_work, int* __restrict__ out_ids,
    int64_t rows, int64_t vocab, int K) {
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  float* s = scores_work + row * vocab;
  __shared__ float red_val[32];
  __shared__ int red_idx[32];
  const int nwarps = (blockDim.x + 31) / 32;
  for (int i = 0; i < K; i++) {
    float best = -3.402823466e38f;
    int besti = -1;
    for (int64_t idx = threadIdx.x; idx < vocab; idx += blockDim.x) {
      float v = s[idx];
      if (v > best) { best = v; besti = (int)idx; }
    }
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_down(best, o);
      int oi = __shfl_down(besti, o);
      if (ov > best) { best = ov; besti = oi; }
    }
    if ((threadIdx.x & 31) == 0) { red_val[threadIdx.x >> 5] = best; red_idx[threadIdx.x >> 5] = besti; }
    __syncthreads();
    if (threadIdx.x == 0) {
      float bv = -3.402823466e38f; int bi = -1;
      for (int w = 0; w < nwarps; w++) if (red_val[w] > bv) { bv = red_val[w]; bi = red_idx[w]; }
      out_ids[row * K + i] = bi;
      if (bi >= 0) s[bi] = -3.402823466e38f;
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
  hipLaunchKernelGGL(eagle4_topk_kernel, dim3((unsigned)rows), dim3(256), 0,
                     stream, (float*)scores_work, (int*)out_ids, rows, vocab, K);
  return hipGetLastError();
}