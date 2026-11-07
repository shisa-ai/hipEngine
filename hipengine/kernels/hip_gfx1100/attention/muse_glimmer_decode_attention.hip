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
#define GLIMMER_KEY_UNROLL 8

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
__global__ __launch_bounds__(32 * GLIMMER_Q_PER_BLOCK) void muse_glimmer_decode_attn_split_kernel(
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
  // Keys run in groups of GLIMMER_KEY_UNROLL: the K/V loads, dots and wave
  // reductions of a group are independent and overlap, then the online
  // softmax updates apply in ascending key order. Every per-key operation and
  // the update order equal the one-key-at-a-time loop, so results are
  // bit-identical to it; the unroll only removes the serial load latency.
  const uint16_t* kbase = k_cache + kvh * GLIMMER_HEAD_DIM + lane * GLIMMER_DIMS_PER_LANE;
  const uint16_t* vbase = v_cache + kvh * GLIMMER_HEAD_DIM + lane * GLIMMER_DIMS_PER_LANE;
  for (int64_t key0 = beg; key0 < end; key0 += GLIMMER_KEY_UNROLL) {
    float dot[GLIMMER_KEY_UNROLL];
    uint2 vraw[GLIMMER_KEY_UNROLL];
#pragma unroll
    for (int u = 0; u < GLIMMER_KEY_UNROLL; u++) {
      const int64_t key = key0 + u < end ? key0 + u : end - 1;
      const uint2 kraw = *reinterpret_cast<const uint2*>(kbase + key * kv_stride);
      vraw[u] = *reinterpret_cast<const uint2*>(vbase + key * kv_stride);
      const uint16_t* kb = reinterpret_cast<const uint16_t*>(&kraw);
      float d = 0.0f;
#pragma unroll
      for (int i = 0; i < GLIMMER_DIMS_PER_LANE; i++) d += qreg[i] * bf16_to_f32(kb[i]);
      dot[u] = d;
    }
#pragma unroll
    for (int u = 0; u < GLIMMER_KEY_UNROLL; u++) dot[u] = wave_sum(dot[u]) * scale;
#pragma unroll
    for (int u = 0; u < GLIMMER_KEY_UNROLL; u++) {
      if (key0 + u >= end) break;
      const float m_new = fmaxf(m, dot[u]);
      const float alpha = __expf(m - m_new);
      const float p = __expf(dot[u] - m_new);
      l = l * alpha + p;
      const uint16_t* vb = reinterpret_cast<const uint16_t*>(&vraw[u]);
#pragma unroll
      for (int i = 0; i < GLIMMER_DIMS_PER_LANE; i++)
        acc[i] = acc[i] * alpha + p * bf16_to_f32(vb[i]);
      m = m_new;
    }
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
    if (ms == -INFINITY) continue;  // empty split: contributes exactly zero
    float ls = p[GLIMMER_HEAD_DIM + 1];
    float w = __expf(ms - m);
    l += w * ls;
    acc += w * p[lane];
  }
  out[(token * n_q_heads + qh) * GLIMMER_HEAD_DIM + lane] = f32_to_bf16(acc / l);
}

// Combine + Glimmer sigmoid attention gate in one launch. Same split merge as
// muse_glimmer_decode_attn_combine_kernel; the BF16-rounded attention value
// is then gated exactly as gguf_gate_mul_bf16 does (stable sigmoid of the
// BF16 gate, product rounded to BF16), so the output is bit-identical to
// combine -> gate_mul.
__device__ __forceinline__ float glimmer_sigmoidf_stable(float value) {
  if (value >= 0.0f) {
    const float z = expf(-value);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(value);
  return z / (1.0f + z);
}

__global__ void muse_glimmer_decode_attn_combine_gated_kernel(
    const float* __restrict__ partials, const uint16_t* __restrict__ gate,
    uint16_t* __restrict__ out, int64_t n_q_heads, int64_t n_splits, int64_t tokens) {
  const int qh = blockIdx.x;
  const int token = blockIdx.y;
  const int lane = threadIdx.x;
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
    if (ms == -INFINITY) continue;  // empty split: contributes exactly zero
    float ls = p[GLIMMER_HEAD_DIM + 1];
    float w = __expf(ms - m);
    l += w * ls;
    acc += w * p[lane];
  }
  const int64_t idx = (token * n_q_heads + qh) * GLIMMER_HEAD_DIM + lane;
  const float a = bf16_to_f32(f32_to_bf16(acc / l));
  const float g = glimmer_sigmoidf_stable(bf16_to_f32(gate[idx]));
  out[idx] = f32_to_bf16(a * g);
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
  muse_glimmer_decode_attn_split_kernel<<<grid, 32 * GLIMMER_Q_PER_BLOCK, 0, stream>>>(
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
// Fused rows-1 decode glue: per-head Q/K RMSNorm + rotary on BF16 inputs,
// BF16 Q out, BF16 K written straight into the cache row at *pos, plus the V
// row append. One block per head (Q heads, then KV heads) and one trailing
// block for V. Each head block reproduces the chain bf16_to_f32 ->
// gguf_head_rmsnorm_partial_rotary_positions_f32_weight (256-thread tree
// reduction, same expressions) -> f32_to_bf16 exactly, so the outputs and
// the cache rows are bit-identical to the seven-launch chain it replaces.
#define GLIMMER_QK_ROPE_THREADS 256

__global__ __launch_bounds__(GLIMMER_QK_ROPE_THREADS)
void muse_glimmer_qk_norm_rope_kv_append_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v, const float* __restrict__ q_weight,
    const float* __restrict__ k_weight, const float* __restrict__ cos_table,
    const float* __restrict__ sin_table, const int64_t* __restrict__ pos,
    uint16_t* __restrict__ q_out, uint16_t* __restrict__ k_cache,
    uint16_t* __restrict__ v_cache, float eps, int64_t n_q_heads,
    int64_t n_kv_heads, int64_t head_dim, int64_t max_positions) {
  __shared__ float partial[GLIMMER_QK_ROPE_THREADS];
  const int64_t head = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t position = *pos;
  if (position < 0 || position >= max_positions) return;
  const int64_t kv_dim = n_kv_heads * head_dim;
  if (head == n_q_heads + n_kv_heads) {
    uint16_t* dst = v_cache + position * kv_dim;
    for (int64_t i = tid; i < kv_dim; i += blockDim.x) dst[i] = v[i];
    return;
  }
  const bool is_query = head < n_q_heads;
  const int64_t local_head = is_query ? head : head - n_q_heads;
  const uint16_t* src = (is_query ? q : k) + local_head * head_dim;
  const float* weight = is_query ? q_weight : k_weight;
  uint16_t* dst = is_query ? q_out + local_head * head_dim
                           : k_cache + position * kv_dim + local_head * head_dim;
  const float* cos = cos_table + position * head_dim;
  const float* sin = sin_table + position * head_dim;

  float sum = 0.0f;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float value = bf16_to_f32(src[dim]);
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
  const int64_t half_rotary = head_dim / 2;
  for (int64_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const float value = bf16_to_f32(src[dim]) * inv_rms * weight[dim];
    const int64_t pair_dim = dim < half_rotary ? dim + half_rotary : dim - half_rotary;
    const float paired = bf16_to_f32(src[pair_dim]) * inv_rms * weight[pair_dim];
    const float rotated = dim < half_rotary ? -paired : paired;
    dst[dim] = f32_to_bf16(value * cos[dim] + rotated * sin[dim]);
  }
}

extern "C" int hipengine_muse_glimmer_qk_norm_rope_kv_append(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, const float* q_weight,
    const float* k_weight, const float* cos_table, const float* sin_table,
    const int64_t* pos, uint16_t* q_out, uint16_t* k_cache, uint16_t* v_cache,
    float eps, int64_t n_q_heads, int64_t n_kv_heads, int64_t head_dim,
    int64_t max_positions, hipStream_t stream) {
  if (!q || !k || !v || !q_weight || !k_weight || !cos_table || !sin_table || !pos ||
      !q_out || !k_cache || !v_cache || n_q_heads <= 0 || n_kv_heads <= 0 ||
      head_dim <= 0 || (head_dim % 2) != 0 || max_positions <= 0) {
    return hipErrorInvalidValue;
  }
  const unsigned int blocks = static_cast<unsigned int>(n_q_heads + n_kv_heads + 1);
  muse_glimmer_qk_norm_rope_kv_append_kernel<<<blocks, GLIMMER_QK_ROPE_THREADS, 0, stream>>>(
      q, k, v, q_weight, k_weight, cos_table, sin_table, pos, q_out, k_cache, v_cache, eps,
      n_q_heads, n_kv_heads, head_dim, max_positions);
  return static_cast<int>(hipGetLastError());
}

// Fused rows-1 residual boundary: gemma4_add_rmsnorm_scale (post-norm of the
// branch, plain F32 weight, + residual, layer scalar 1) followed by
// gemma4_rmsnorm (the next pre-norm) over the BF16-rounded residual. One
// 256-thread block per row, the gemma4 tree reduction and expressions, and
// stage 2 reads back the rounded residual this same thread stored, so both
// outputs are bit-identical to the two-launch chain.
#define GLIMMER_NORM_THREADS 256

__device__ __forceinline__ void glimmer_block_sum(float* partial, float value) {
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
}

__global__ __launch_bounds__(GLIMMER_NORM_THREADS)
void muse_glimmer_add_rmsnorm_rmsnorm_kernel(
    const uint16_t* __restrict__ branch, const uint16_t* __restrict__ residual,
    const float* __restrict__ post_weight, const float* __restrict__ pre_weight,
    uint16_t* __restrict__ residual_out, uint16_t* __restrict__ normed_out,
    float post_eps, float pre_eps, int64_t hidden_size) {
  __shared__ float partial[GLIMMER_NORM_THREADS];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
  const float scalar = 1.0f;
  float sumsq = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_to_f32(branch[row_offset + idx]);
    sumsq += value * value;
  }
  glimmer_block_sum(partial, sumsq);
  const float inv_rms1 = rsqrtf((partial[0] / static_cast<float>(hidden_size)) + post_eps);
  __syncthreads();
  float sumsq2 = 0.0f;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_to_f32(branch[row_offset + idx]);
    const float normed = value * inv_rms1 * post_weight[idx];
    const float base = bf16_to_f32(residual[row_offset + idx]);
    const uint16_t rounded = f32_to_bf16((base + normed) * scalar);
    residual_out[row_offset + idx] = rounded;
    const float r = bf16_to_f32(rounded);
    sumsq2 += r * r;
  }
  glimmer_block_sum(partial, sumsq2);
  const float inv_rms2 = rsqrtf((partial[0] / static_cast<float>(hidden_size)) + pre_eps);
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float value = bf16_to_f32(residual_out[row_offset + idx]);
    normed_out[row_offset + idx] = f32_to_bf16(value * inv_rms2 * pre_weight[idx]);
  }
}

// Register-resident sibling for rows up to N * 256 elements: the same
// thread->element ownership (idx = tid + e * 256, ascending e), the same
// expressions and the same block tree, with the loop fully unrolled so the
// loads issue together and stage 2/3 reuse registers instead of re-reading.
// Bit-identical to the strided kernel above.
template <int N>
__global__ __launch_bounds__(GLIMMER_NORM_THREADS)
void muse_glimmer_add_rmsnorm_rmsnorm_reg_kernel(
    const uint16_t* __restrict__ branch, const uint16_t* __restrict__ residual,
    const float* __restrict__ post_weight, const float* __restrict__ pre_weight,
    uint16_t* __restrict__ residual_out, uint16_t* __restrict__ normed_out,
    float post_eps, float pre_eps, int64_t hidden_size) {
  __shared__ float partial[GLIMMER_NORM_THREADS];
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * hidden_size;
  const int tid = threadIdx.x;
  // Every global read is issued up front (one DRAM round trip instead of
  // three dependent ones); the arithmetic below is unchanged.
  float bv[N];
  float rv[N];
  float w1[N];
  float w2[N];
#pragma unroll
  for (int e = 0; e < N; ++e) {
    const int64_t idx = tid + static_cast<int64_t>(e) * GLIMMER_NORM_THREADS;
    const bool live = idx < hidden_size;
    bv[e] = live ? bf16_to_f32(branch[row_offset + idx]) : 0.0f;
    rv[e] = live ? bf16_to_f32(residual[row_offset + idx]) : 0.0f;
    w1[e] = live ? post_weight[idx] : 0.0f;
    w2[e] = live ? pre_weight[idx] : 0.0f;
  }
  float sumsq = 0.0f;
#pragma unroll
  for (int e = 0; e < N; ++e) {
    const int64_t idx = tid + static_cast<int64_t>(e) * GLIMMER_NORM_THREADS;
    if (idx < hidden_size) sumsq += bv[e] * bv[e];
  }
  glimmer_block_sum(partial, sumsq);
  const float inv_rms1 = rsqrtf((partial[0] / static_cast<float>(hidden_size)) + post_eps);
  __syncthreads();
  const float scalar = 1.0f;
  float sumsq2 = 0.0f;
#pragma unroll
  for (int e = 0; e < N; ++e) {
    const int64_t idx = tid + static_cast<int64_t>(e) * GLIMMER_NORM_THREADS;
    if (idx < hidden_size) {
      const float normed = bv[e] * inv_rms1 * w1[e];
      const uint16_t rounded = f32_to_bf16((rv[e] + normed) * scalar);
      residual_out[row_offset + idx] = rounded;
      rv[e] = bf16_to_f32(rounded);
      sumsq2 += rv[e] * rv[e];
    }
  }
  glimmer_block_sum(partial, sumsq2);
  const float inv_rms2 = rsqrtf((partial[0] / static_cast<float>(hidden_size)) + pre_eps);
#pragma unroll
  for (int e = 0; e < N; ++e) {
    const int64_t idx = tid + static_cast<int64_t>(e) * GLIMMER_NORM_THREADS;
    if (idx < hidden_size) {
      normed_out[row_offset + idx] = f32_to_bf16(rv[e] * inv_rms2 * w2[e]);
    }
  }
}

extern "C" int hipengine_muse_glimmer_add_rmsnorm_rmsnorm(
    const uint16_t* branch, const uint16_t* residual, const float* post_weight,
    const float* pre_weight, uint16_t* residual_out, uint16_t* normed_out,
    int64_t rows, int64_t hidden_size, float post_eps, float pre_eps, hipStream_t stream) {
  if (!branch || !residual || !post_weight || !pre_weight || !residual_out || !normed_out ||
      rows <= 0 || hidden_size <= 0 || residual_out == residual || residual_out == branch) {
    return hipErrorInvalidValue;
  }
  const unsigned int grid = static_cast<unsigned int>(rows);
  const int64_t per_thread = (hidden_size + GLIMMER_NORM_THREADS - 1) / GLIMMER_NORM_THREADS;
#define GLIMMER_NORM_REG_CASE(N)                                                         \
  if (per_thread <= N) {                                                                 \
    muse_glimmer_add_rmsnorm_rmsnorm_reg_kernel<N><<<grid, GLIMMER_NORM_THREADS, 0, stream>>>( \
        branch, residual, post_weight, pre_weight, residual_out, normed_out, post_eps,     \
        pre_eps, hidden_size);                                                           \
    return static_cast<int>(hipGetLastError());                                          \
  }
  GLIMMER_NORM_REG_CASE(8)
  GLIMMER_NORM_REG_CASE(16)
  GLIMMER_NORM_REG_CASE(26)
  GLIMMER_NORM_REG_CASE(32)
#undef GLIMMER_NORM_REG_CASE
  muse_glimmer_add_rmsnorm_rmsnorm_kernel<<<grid, GLIMMER_NORM_THREADS, 0, stream>>>(
      branch, residual, post_weight, pre_weight, residual_out, normed_out, post_eps, pre_eps,
      hidden_size);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hipengine_muse_glimmer_decode_attn_gated(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, const uint16_t* gate,
    uint16_t* out, float* partials, int64_t n_q_heads, int64_t n_kv_heads,
    int64_t key_begin, int64_t key_end, int64_t n_splits, int64_t tokens, int64_t window,
    float scale, const int64_t* key_end_dev, hipStream_t stream) {
  if (!q || !k || !v || !gate || !out || !partials || n_q_heads <= 0 || n_kv_heads <= 0 ||
      n_q_heads % n_kv_heads || n_q_heads % GLIMMER_Q_PER_BLOCK ||
      key_end <= key_begin || n_splits <= 1 || tokens <= 0)
    return hipErrorInvalidValue;
  dim3 grid(static_cast<unsigned>(n_splits),
            static_cast<unsigned>(n_q_heads / GLIMMER_Q_PER_BLOCK),
            static_cast<unsigned>(tokens));
  muse_glimmer_decode_attn_split_kernel<<<grid, 32 * GLIMMER_Q_PER_BLOCK, 0, stream>>>(
      q, k, v, partials, out, n_q_heads, n_kv_heads, key_begin, key_end,
      n_splits, tokens, window, scale, key_end_dev);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) return static_cast<int>(err);
  dim3 cgrid(static_cast<unsigned>(n_q_heads), static_cast<unsigned>(tokens));
  muse_glimmer_decode_attn_combine_gated_kernel<<<cgrid, 128, 0, stream>>>(
      partials, gate, out, n_q_heads, n_splits, tokens);
  return static_cast<int>(hipGetLastError());
}
