// Muse Glimmer flash-decoding attention: key-split, GQA-packed, BF16 KV.
//
// Geometry: 32 Q heads / 2 KV heads (16:1), head_dim 128, contiguous per-layer
// KV [max_context, n_kv_heads, 128] BF16. One decode token per call. The grid
// splits the live key range across blocks so all 40 CUs stay busy at long
// context (the gemma4 strict decode kernel walks keys serially per head and
// collapses past ~1K keys — measured 134 ms/step at 4K vs llama.cpp's 0.45).
//
// Block layout: 128 threads = 4 waves; each wave owns one Q head (4 Q heads
// per block, all sharing one KV head). grid = (n_splits, n_q_heads/4). Each
// wave walks its key slice serially: QK dot via lane-parallel 128 dims (4 per
// lane) + wave reduction, online softmax, PV accumulation (4 per lane).
// Partials (m, l, acc) go to global scratch; a combine kernel merges splits.
//
// SWA: keys <= pos - window are skipped (slice bounds clamped host-side).
// NoPE full layers attend the whole live range. The arithmetic is online
// softmax in F32 — a different association than the strict kernel, gated by
// the CPU-reference outer floor (KL <= 0.05, top-1 >= 90%).
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

#define GLIMMER_HEAD_DIM 128
#define GLIMMER_DIMS_PER_LANE 4
#define GLIMMER_Q_PER_BLOCK 4

__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  return __uint_as_float(u);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  uint32_t lsb = (u >> 16) & 1;
  u += 0x7FFF + lsb;
  return static_cast<uint16_t>(u >> 16);
}

__device__ __forceinline__ float wave_sum(float v) {
  for (int off = 16; off; off >>= 1) v += __shfl_down(v, off);
  return __shfl(v, 0);
}

// Partials layout: [n_splits, n_q_heads, 128 + 2] f32 (acc, m, l).
// Multi-row (prefill) mode: grid.z = tokens, each block owns one query row;
// key range is causal+SWA clamped per row. With n_splits == 1 the split
// kernel writes the normalized BF16 output directly (no combine pass).
__global__ __launch_bounds__(128, 4) void muse_glimmer_decode_attn_split_kernel(
    const uint16_t* __restrict__ q,        // [tokens, n_q_heads, 128] bf16
    const uint16_t* __restrict__ k_cache,  // [max_ctx, n_kv_heads, 128] bf16
    const uint16_t* __restrict__ v_cache,
    float* __restrict__ partials,          // [n_splits, tokens, n_q_heads, 130]
    uint16_t* __restrict__ out,            // [tokens, n_q_heads, 128] bf16
    int64_t n_q_heads, int64_t n_kv_heads, int64_t key_begin, int64_t key_end,
    int64_t n_splits, int64_t tokens, int64_t window, float scale,
    const int64_t* __restrict__ key_end_dev) {
  const int split = blockIdx.x;
  const int q_group = blockIdx.y;  // 4 Q heads per block
  const int token = blockIdx.z;
  const int wave = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int qh = q_group * GLIMMER_Q_PER_BLOCK + wave;
  if (qh >= n_q_heads || token >= tokens) return;
  const int kvh = qh / (n_q_heads / n_kv_heads);
  if (key_end_dev != nullptr) key_end = *key_end_dev;

  // Per-row causal + SWA clamp (key_end is the live extent; row position is
  // key_end - tokens + token for a contiguous append).
  const int64_t pos = key_end - tokens + token;
  int64_t row_end = min(key_end, pos + 1);
  int64_t row_begin = key_begin;
  if (window > 0) row_begin = max(row_begin, pos - window + 1);
  const int64_t span = row_end - row_begin;
  const int64_t slice = (span + n_splits - 1) / n_splits;
  const int64_t beg = row_begin + split * slice;
  const int64_t end = min(beg + slice, row_end);

  // Load this wave's Q row into registers (4 dims per lane).
  float qreg[GLIMMER_DIMS_PER_LANE];
#pragma unroll
  for (int i = 0; i < GLIMMER_DIMS_PER_LANE; i++)
    qreg[i] = bf16_to_f32(q[(token * n_q_heads + qh) * GLIMMER_HEAD_DIM + lane * GLIMMER_DIMS_PER_LANE + i]);

  float m = -INFINITY;
  float l = 0.0f;
  float acc[GLIMMER_DIMS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int64_t kv_stride = n_kv_heads * GLIMMER_HEAD_DIM;
  for (int64_t key = beg; key < end; key++) {
    const uint16_t* krow = k_cache + key * kv_stride + kvh * GLIMMER_HEAD_DIM;
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < GLIMMER_DIMS_PER_LANE; i++)
      dot += qreg[i] * bf16_to_f32(krow[lane * GLIMMER_DIMS_PER_LANE + i]);
    dot = wave_sum(dot) * scale;

    float m_new = fmaxf(m, dot);
    float alpha = __expf(m - m_new);
    float p = __expf(dot - m_new);
    l = l * alpha + p;
    const uint16_t* vrow = v_cache + key * kv_stride + kvh * GLIMMER_HEAD_DIM;
#pragma unroll
    for (int i = 0; i < GLIMMER_DIMS_PER_LANE; i++)
      acc[i] = acc[i] * alpha + p * bf16_to_f32(vrow[lane * GLIMMER_DIMS_PER_LANE + i]);
    m = m_new;
  }

  if (n_splits == 1) {
    uint16_t* orow = out + (token * n_q_heads + qh) * GLIMMER_HEAD_DIM;
    const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
#pragma unroll
    for (int i = 0; i < GLIMMER_DIMS_PER_LANE; i++)
      orow[lane * GLIMMER_DIMS_PER_LANE + i] = f32_to_bf16(acc[i] * inv_l);
    return;
  }
  float* outp = partials + ((int64_t)split * tokens * n_q_heads + token * n_q_heads + qh) * (GLIMMER_HEAD_DIM + 2);
#pragma unroll
  for (int i = 0; i < GLIMMER_DIMS_PER_LANE; i++)
    outp[lane * GLIMMER_DIMS_PER_LANE + i] = acc[i];
  if (lane == 0) {
    outp[GLIMMER_HEAD_DIM] = m;
    outp[GLIMMER_HEAD_DIM + 1] = l;
  }
}

__global__ void muse_glimmer_decode_attn_combine_kernel(
    const float* __restrict__ partials, uint16_t* __restrict__ out,
    int64_t n_q_heads, int64_t n_splits, int64_t tokens) {
  const int qh = blockIdx.x;
  const int token = blockIdx.y;
  const int lane = threadIdx.x;  // 128 threads, one per dim
  if (qh >= n_q_heads || token >= tokens) return;
  float m = -INFINITY;
  const int64_t stride = GLIMMER_HEAD_DIM + 2;
  const int64_t base = (int64_t)token * n_q_heads + qh;
  const int64_t tstride = tokens * n_q_heads * stride;
  for (int64_t s = 0; s < n_splits; s++) {
    float ms = partials[s * tstride + base * stride + GLIMMER_HEAD_DIM];
    m = fmaxf(m, ms);
  }
  float l = 0.0f;
  float acc = 0.0f;
  for (int64_t s = 0; s < n_splits; s++) {
    const float* p = partials + s * tstride + base * stride;
    float ms = p[GLIMMER_HEAD_DIM];
    float ls = p[GLIMMER_HEAD_DIM + 1];
    float w = __expf(ms - m);
    l += w * ls;
    acc += w * p[lane];
  }
  out[(token * n_q_heads + qh) * GLIMMER_HEAD_DIM + lane] = f32_to_bf16(acc / l);
}

// Graph-safe KV append: destination row read from a device position buffer.
__global__ void muse_glimmer_kv_append_kernel(
    uint16_t* __restrict__ dst, const uint16_t* __restrict__ src,
    const int64_t* __restrict__ pos, int64_t row_elems, int64_t n) {
  const int64_t off = (*pos) * row_elems;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x)
    dst[off + i] = src[i];
}

extern "C" int hipengine_muse_glimmer_kv_append(
    uint16_t* dst, const uint16_t* src, const int64_t* pos, int64_t row_elems,
    int64_t n, hipStream_t stream) {
  if (!dst || !src || !pos || row_elems <= 0 || n <= 0) return hipErrorInvalidValue;
  muse_glimmer_kv_append_kernel<<<1, 256, 0, stream>>>(dst, src, pos, row_elems, n);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_muse_glimmer_decode_attn(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out,
    float* partials, int64_t n_q_heads, int64_t n_kv_heads, int64_t key_begin,
    int64_t key_end, int64_t n_splits, int64_t tokens, int64_t window,
    float scale, const int64_t* key_end_dev, hipStream_t stream) {
  if (!q || !k || !v || !out || !partials || n_q_heads <= 0 || n_kv_heads <= 0 ||
      n_q_heads % n_kv_heads || n_q_heads % GLIMMER_Q_PER_BLOCK ||
      key_end <= key_begin || n_splits <= 0 || tokens <= 0)
    return hipErrorInvalidValue;
  dim3 grid(static_cast<unsigned>(n_splits),
            static_cast<unsigned>(n_q_heads / GLIMMER_Q_PER_BLOCK),
            static_cast<unsigned>(tokens));
  muse_glimmer_decode_attn_split_kernel<<<grid, 128, 0, stream>>>(
      q, k, v, partials, out, n_q_heads, n_kv_heads, key_begin, key_end,
      n_splits, tokens, window, scale, key_end_dev);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) return static_cast<int>(err);
  if (n_splits > 1) {
    dim3 cgrid(static_cast<unsigned>(n_q_heads), static_cast<unsigned>(tokens));
    muse_glimmer_decode_attn_combine_kernel<<<cgrid, 128, 0, stream>>>(
        partials, out, n_q_heads, n_splits, tokens);
    err = hipGetLastError();
  }
  return static_cast<int>(err);
}