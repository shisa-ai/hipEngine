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

// Shared-row WMMA decode/verify attention (absolute key splits).
//
// One kernel serves AR decode (tokens == 1) and the exact speculative block
// verify (tokens > 1), and a verify row is bit-identical to the decode step at
// its position. Layout: grid = (n_blocks, n_kv_heads, ceil(tokens / 8)),
// block = one wave per query token (<= 8). The wave packs the KV head's 16 Q
// heads into the WMMA row dimension, so one staged K/V tile serves all 16 heads
// of every token in the block (the scalar split kernel above re-reads it per 4
// heads and per token).
//
// Exactness: keys are cut into absolute splits of `split_keys` keys
// ([s*split_keys, (s+1)*split_keys)), and block b owns splits b, b+n_blocks,
// ... in ascending order, walking each in absolute 16-key tiles. A token's
// partial therefore depends only on its own position and window, never on the
// other tokens in the block or on the live extent: a wave skips tiles it
// cannot see, and zeroes the entries it cannot see in tiles it sees partly, so
// every tile contributes exactly what it contributes when that token runs
// alone. split_keys and n_blocks must be the
// same for decode and verify. The combine kernels above merge the n_blocks
// partials in ascending block order.
//
// Arithmetic: QK^T and PV on the 16x16x16 f16 WMMA (BF16 -> F16 is exact for
// in-range values) with F32 online softmax per 16-key tile, as in
// muse_glimmer_flash_prefill_wmma_mt_kernel, except for the per-wave tile
// classes in the tile loop.
namespace glimmer_shared {

constexpr int HEAD_DIM = 128;
constexpr int GQA = 16;
constexpr int K_TILE = 16;
constexpr int MAX_TOKENS = 8;

typedef _Float16 half_t;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__device__ inline float8_t zero8() {
  return float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

}  // namespace glimmer_shared

__global__ __launch_bounds__(32 * glimmer_shared::MAX_TOKENS)
void muse_glimmer_decode_attn_shared_kernel(
    const uint16_t* __restrict__ q,        // [tokens, n_q_heads, 128] bf16
    const uint16_t* __restrict__ k_cache,  // [max_ctx, n_kv_heads, 128] bf16
    const uint16_t* __restrict__ v_cache,
    float* __restrict__ partials,          // [n_blocks, tokens, n_q_heads, 130]
    int64_t n_q_heads, int64_t n_kv_heads, int64_t key_begin, int64_t key_end,
    int64_t split_keys, int64_t tokens, int64_t window, float scale,
    const int64_t* __restrict__ key_end_dev) {
  using namespace glimmer_shared;
  const int64_t n_blocks = gridDim.x;
  const int64_t block = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lane16 = lane & 15;
  const int half_id = lane >> 4;
  const int nthreads = blockDim.x;
  if (key_end_dev != nullptr) key_end = *key_end_dev;

  const int64_t first_token = static_cast<int64_t>(blockIdx.z) * MAX_TOKENS;
  const int64_t last_token = min<int64_t>(tokens, first_token + nthreads / 32) - 1;
  const int64_t token = first_token + wave;
  const bool active = token <= last_token;
  const int64_t own = active ? token : last_token;
  const int64_t position = key_end - tokens + own;
  int64_t row_begin = key_begin;
  if (window > 0) row_begin = max(row_begin, position - window + 1);
  const int64_t row_end = position + 1;

  // Union of the block's visible ranges (rows are consecutive positions).
  const int64_t first_pos = key_end - tokens + first_token;
  const int64_t last_pos = key_end - tokens + last_token;
  int64_t union_begin = key_begin;
  if (window > 0) union_begin = max(union_begin, first_pos - window + 1);
  const int64_t union_end = last_pos + 1;

  const int q_head_base = kv_head * GQA;
  half16_t q_frag[HEAD_DIM / 16];
#pragma unroll
  for (int dt = 0; dt < HEAD_DIM / 16; ++dt) {
    const uint16_t* qrow = q + (own * n_q_heads + q_head_base + lane16) * HEAD_DIM + dt * 16;
#pragma unroll
    for (int k = 0; k < 16; ++k) q_frag[dt][k] = static_cast<half_t>(bf16_to_f32(qrow[k]));
  }

  float running_max = -INFINITY;
  float running_sum = 0.f;
  float8_t out_acc[HEAD_DIM / 16];
#pragma unroll
  for (int dt = 0; dt < HEAD_DIM / 16; ++dt) out_acc[dt] = zero8();

  __shared__ __align__(1024) half_t k_tile[K_TILE * HEAD_DIM];
  __shared__ __align__(1024) half_t v_tile[K_TILE * HEAD_DIM];

  const int64_t kv_stride = n_kv_heads * HEAD_DIM;
  const int64_t first_split = union_begin / split_keys;
  const int64_t last_split = (union_end - 1) / split_keys;
  // First split this block owns at or after first_split.
  int64_t split = first_split + ((block - first_split % n_blocks) + n_blocks) % n_blocks;
  for (; split <= last_split; split += n_blocks) {
    const int64_t split_begin = split * split_keys;
    const int64_t split_end = min(split_begin + split_keys, union_end);
    // Absolute 16-key tiles; skip those wholly before the union.
    int64_t k_base = max(split_begin, (union_begin / K_TILE) * K_TILE);
    for (; k_base < split_end; k_base += K_TILE) {
      const int keys_this = static_cast<int>(min<int64_t>(K_TILE, split_end - k_base));
      for (int slot = threadIdx.x; slot < K_TILE * HEAD_DIM / 8; slot += nthreads) {
        const int key_local = slot / (HEAD_DIM / 8);
        const int dim = (slot % (HEAD_DIM / 8)) * 8;
        uint4 kraw = make_uint4(0, 0, 0, 0);
        uint4 vraw = make_uint4(0, 0, 0, 0);
        if (key_local < keys_this) {
          const int64_t off = (k_base + key_local) * kv_stride + kv_head * HEAD_DIM + dim;
          kraw = *reinterpret_cast<const uint4*>(k_cache + off);
          vraw = *reinterpret_cast<const uint4*>(v_cache + off);
        }
        const uint16_t* kb = reinterpret_cast<const uint16_t*>(&kraw);
        const uint16_t* vb = reinterpret_cast<const uint16_t*>(&vraw);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          k_tile[key_local * HEAD_DIM + dim + e] = static_cast<half_t>(bf16_to_f32(kb[e]));
          v_tile[key_local * HEAD_DIM + dim + e] = static_cast<half_t>(bf16_to_f32(vb[e]));
        }
      }
      __syncthreads();

      // Per-wave tile class. A tile with no key this row can see is skipped
      // outright, leaving the row's state untouched, so tiles that exist only
      // because other rows share the block cannot affect it. A partially
      // visible tile zeroes the K/V entries this row cannot see before the
      // WMMAs: the f16 WMMA is not exactly invariant to entries multiplied by
      // a zero probability (measured on gfx1151: a masked key's V row moved a
      // visible output by one BF16 ulp), and a one-row call stages zeros past
      // its position. Fully visible tiles need neither. The branch is uniform
      // within the wave.
      const int64_t vis_lo = max(k_base, row_begin);
      const int64_t vis_hi = min(k_base + K_TILE, row_end);
      if (active && vis_lo < vis_hi) {
        const bool partial = vis_lo != k_base || vis_hi != k_base + K_TILE;
        const int64_t lane_key = k_base + lane16;
        const bool lane_key_visible = lane_key >= row_begin && lane_key < row_end;
        float8_t scores = zero8();
#pragma unroll
        for (int dt = 0; dt < HEAD_DIM / 16; ++dt) {
          half16_t k_frag;
#pragma unroll
          for (int k = 0; k < 16; ++k) k_frag[k] = k_tile[lane16 * HEAD_DIM + dt * 16 + k];
          if (partial && !lane_key_visible) k_frag = half16_t{};
          scores = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k_frag, q_frag[dt], scores);
        }

        float local_max = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int64_t key = k_base + 2 * j + half_id;
          const bool visible = key >= row_begin && key < row_end;
          const float sc = visible ? scores[j] * scale : -INFINITY;
          scores[j] = sc;
          local_max = fmaxf(local_max, sc);
        }
        const float batch_max = fmaxf(local_max, __shfl_xor(local_max, 16));
        const float next_max = fmaxf(running_max, batch_max);
        const float prev_scale = isfinite(running_max) ? expf(running_max - next_max) : 0.f;
#pragma unroll
        for (int dt = 0; dt < HEAD_DIM / 16; ++dt) {
#pragma unroll
          for (int j = 0; j < 8; ++j) out_acc[dt][j] *= prev_scale;
        }
        float local_sum = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float pr = isfinite(scores[j]) ? expf(scores[j] - next_max) : 0.f;
          scores[j] = pr;
          local_sum += pr;
        }
        const float batch_sum = local_sum + __shfl_xor(local_sum, 16);
        running_sum = prev_scale * running_sum + batch_sum;
        running_max = next_max;

        half16_t p_frag;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float mine = scores[j];
          const float partner = __shfl_xor(mine, 16);
          p_frag[2 * j + half_id] = static_cast<half_t>(mine);
          p_frag[2 * j + (half_id ^ 1)] = static_cast<half_t>(partner);
        }
        unsigned visible_keys = 0xFFFFu;
        if (partial) {
          visible_keys = 0;
#pragma unroll
          for (int k = 0; k < 16; ++k) {
            const int64_t key = k_base + k;
            if (key >= row_begin && key < row_end) visible_keys |= 1u << k;
          }
        }
#pragma unroll
        for (int dt = 0; dt < HEAD_DIM / 16; ++dt) {
          half16_t v_frag;
#pragma unroll
          for (int k = 0; k < 16; ++k) v_frag[k] = v_tile[k * HEAD_DIM + dt * 16 + lane16];
          if (partial) {
#pragma unroll
            for (int k = 0; k < 16; ++k)
              if (!((visible_keys >> k) & 1u)) v_frag[k] = static_cast<half_t>(0.f);
          }
          out_acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(v_frag, p_frag, out_acc[dt]);
        }
      }
      __syncthreads();
    }
  }

  if (!active) return;
  // Unnormalized partial (acc, m, l) for the combine kernels.
  float* outp = partials +
      ((block * tokens + token) * n_q_heads + q_head_base + lane16) * (HEAD_DIM + 2);
#pragma unroll
  for (int dt = 0; dt < HEAD_DIM / 16; ++dt) {
#pragma unroll
    for (int j = 0; j < 8; ++j) outp[dt * 16 + 2 * j + half_id] = out_acc[dt][j];
  }
  if (half_id == 0) {
    outp[HEAD_DIM] = running_max;
    outp[HEAD_DIM + 1] = running_sum;
  }
}

extern "C" int hipengine_muse_glimmer_decode_attn_shared(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, const uint16_t* gate,
    uint16_t* out, float* partials, int64_t n_q_heads, int64_t n_kv_heads,
    int64_t key_begin, int64_t key_end, int64_t split_keys, int64_t n_blocks,
    int64_t tokens, int64_t window, float scale, const int64_t* key_end_dev,
    hipStream_t stream) {
  using namespace glimmer_shared;
  if (!q || !k || !v || !out || !partials || n_kv_heads <= 0 ||
      n_q_heads != n_kv_heads * GQA || key_end <= key_begin || tokens <= 0 ||
      split_keys <= 0 || split_keys % K_TILE || n_blocks <= 0)
    return hipErrorInvalidValue;
  const int64_t groups = (tokens + MAX_TOKENS - 1) / MAX_TOKENS;
  const int64_t waves = tokens < MAX_TOKENS ? tokens : MAX_TOKENS;
  dim3 grid(static_cast<unsigned>(n_blocks), static_cast<unsigned>(n_kv_heads),
            static_cast<unsigned>(groups));
  muse_glimmer_decode_attn_shared_kernel<<<grid, static_cast<unsigned>(32 * waves), 0, stream>>>(
      q, k, v, partials, n_q_heads, n_kv_heads, key_begin, key_end, split_keys, tokens,
      window, scale, key_end_dev);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) return static_cast<int>(err);
  dim3 cgrid(static_cast<unsigned>(n_q_heads), static_cast<unsigned>(tokens));
  if (gate != nullptr) {
    muse_glimmer_decode_attn_combine_gated_kernel<<<cgrid, 128, 0, stream>>>(
        partials, gate, out, n_q_heads, n_blocks, tokens);
  } else {
    muse_glimmer_decode_attn_combine_kernel<<<cgrid, 128, 0, stream>>>(
        partials, out, n_q_heads, n_blocks, tokens);
  }
  return static_cast<int>(hipGetLastError());
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
  // blockIdx.y selects the block row; tokens==1 keeps blockIdx.y==0 so the
  // rows-1 decode launch is unchanged. Each row runs the identical per-head
  // chain at its own position, so multi-row output equals the rows-1 kernel
  // applied per row bit-for-bit.
  const int64_t row = blockIdx.y;
  const int tid = threadIdx.x;
  const int64_t position = pos[row];
  if (position < 0 || position >= max_positions) return;
  const int64_t kv_dim = n_kv_heads * head_dim;
  const int64_t q_dim = n_q_heads * head_dim;
  if (head == n_q_heads + n_kv_heads) {
    uint16_t* dst = v_cache + position * kv_dim;
    const uint16_t* vsrc = v + row * kv_dim;
    for (int64_t i = tid; i < kv_dim; i += blockDim.x) dst[i] = vsrc[i];
    return;
  }
  const bool is_query = head < n_q_heads;
  const int64_t local_head = is_query ? head : head - n_q_heads;
  const uint16_t* src = (is_query ? q + row * q_dim : k + row * kv_dim)
                      + local_head * head_dim;
  const float* weight = is_query ? q_weight : k_weight;
  uint16_t* dst = is_query ? q_out + row * q_dim + local_head * head_dim
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
    int64_t max_positions, int64_t tokens, hipStream_t stream) {
  if (!q || !k || !v || !q_weight || !k_weight || !cos_table || !sin_table || !pos ||
      !q_out || !k_cache || !v_cache || n_q_heads <= 0 || n_kv_heads <= 0 ||
      head_dim <= 0 || (head_dim % 2) != 0 || max_positions <= 0 || tokens <= 0) {
    return hipErrorInvalidValue;
  }
  const dim3 grid(static_cast<unsigned int>(n_q_heads + n_kv_heads + 1),
                  static_cast<unsigned int>(tokens));
  muse_glimmer_qk_norm_rope_kv_append_kernel<<<grid, GLIMMER_QK_ROPE_THREADS, 0, stream>>>(
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
