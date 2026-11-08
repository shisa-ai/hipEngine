// Muse Glimmer DFlash draft window attention: non-causal WMMA block-flash.
//
// The drafter attends its block rows (anchor + masks) over a sliding window of
// the injected context PLUS the whole block bidirectionally (DFlash non-causal
// token mode). The mask is `key > position - window` with NO causal upper bound
// (block rows see later block rows). This adapts the muse_glimmer flash-prefill
// multi-token WMMA pattern to the draft's 4:1 GQA (32 Q / 8 KV heads, d128):
// one block owns TOKENS_PER_BLOCK adjacent query rows of one KV head, every
// staged 16-key K/V tile is shared by all waves, and the WMMA row dimension
// packs the GQA Q heads (lane16 < GQA active; the rest zeroed). The draft only
// proposes, so this is a production flash route (online softmax reassociation),
// not bit-exact to the scalar Gemma4 block kernel; the exact target verify keeps
// DFlash IDs == true AR.
//
// KV layout: [max_ctx, n_kv_heads, 128] bf16 contiguous; the caller offsets the
// K/V pointers to the window start and passes keys=span, so key index 0 is the
// first kept key. position = key_end - tokens + token reproduces the caller's
// row_offset + token frame.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int HEAD_DIM = 128;
constexpr int K_TILE = 16;
constexpr int TOKENS_PER_BLOCK = 1;
constexpr int MT_THREADS = TOKENS_PER_BLOCK * 32;

typedef _Float16 half_t;
typedef half_t __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__device__ inline float bf16_to_f32(uint16_t v) {
  return __uint_as_float(static_cast<uint32_t>(v) << 16);
}
__device__ inline uint16_t f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7FFFU + ((u >> 16) & 1U);
  return static_cast<uint16_t>(u >> 16);
}
__device__ inline float8_t zero8() {
  return float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}
__device__ inline float lane_xor16(float v) { return __shfl_xor(v, 16); }

__global__ __launch_bounds__(MT_THREADS)
void muse_glimmer_draft_flash_wmma_kernel(
    const uint16_t* __restrict__ q,        // [tokens, n_q_heads, 128] bf16
    const uint16_t* __restrict__ k_cache,  // [span, n_kv_heads, 128] bf16
    const uint16_t* __restrict__ v_cache,
    uint16_t* __restrict__ out,            // [tokens, n_q_heads, 128] bf16
    int64_t n_q_heads, int64_t n_kv_heads, int64_t key_end,
    int64_t tokens, int64_t window, int64_t gqa, float scale) {
  const int kv_head = blockIdx.y;
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lane16 = lane & 15;
  const int half_id = lane >> 4;
  const int64_t first_token = static_cast<int64_t>(blockIdx.x) * TOKENS_PER_BLOCK;
  const int64_t last_token = min<int64_t>(tokens, first_token + TOKENS_PER_BLOCK) - 1;
  const int64_t token = first_token + wave;
  const bool active = token <= last_token;
  const int64_t position = key_end - tokens + (active ? token : last_token);
  const int64_t q_head_base = kv_head * gqa;
  const bool head_active = lane16 < gqa;

  // Non-causal union range: every row walks up to key_end (the whole block is
  // visible to every row); the window only trims the leading context keys.
  const int64_t first_pos = key_end - tokens + first_token;
  int64_t union_begin = 0;
  if (window > 0) union_begin = max<int64_t>(0, first_pos - window + 1);
  union_begin = (union_begin / K_TILE) * K_TILE;
  const int64_t union_end = key_end;

  half16_t q_frag[HEAD_DIM / 16];
#pragma unroll
  for (int dt = 0; dt < HEAD_DIM / 16; ++dt) {
    const int64_t qh = q_head_base + lane16;
    const uint16_t* qrow =
        q + ((active ? token : last_token) * n_q_heads + qh) * HEAD_DIM + dt * 16;
#pragma unroll
    for (int k = 0; k < 16; ++k)
      q_frag[dt][k] = (head_active && active)
                          ? static_cast<half_t>(bf16_to_f32(qrow[k]))
                          : static_cast<half_t>(0.f);
  }

  float running_max = -INFINITY;
  float running_sum = 0.f;
  float8_t out_acc[HEAD_DIM / 16];
#pragma unroll
  for (int dt = 0; dt < HEAD_DIM / 16; ++dt) out_acc[dt] = zero8();

  __shared__ __align__(1024) half_t k_tile[K_TILE * HEAD_DIM];
  __shared__ __align__(1024) half_t v_tile[K_TILE * HEAD_DIM];

  const int64_t kv_stride = n_kv_heads * HEAD_DIM;
  for (int64_t k_base = union_begin; k_base < union_end; k_base += K_TILE) {
    const int keys_this = static_cast<int>(min<int64_t>(K_TILE, union_end - k_base));
    for (int slot = threadIdx.x; slot < K_TILE * HEAD_DIM / 8; slot += MT_THREADS) {
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

    float8_t scores = zero8();
#pragma unroll
    for (int dt = 0; dt < HEAD_DIM / 16; ++dt) {
      half16_t k_frag;
#pragma unroll
      for (int k = 0; k < 16; ++k) k_frag[k] = k_tile[lane16 * HEAD_DIM + dt * 16 + k];
      scores = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k_frag, q_frag[dt], scores);
    }

    // Non-causal window mask: no `key <= position` upper bound (block rows see
    // later block rows); only the sliding-window lower bound trims keys.
    float local_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int key_local = 2 * j + half_id;
      const int64_t key = k_base + key_local;
      bool visible = active && head_active && key_local < keys_this;
      if (window > 0) visible = visible && (key > position - window);
      const float sc = visible ? scores[j] * scale : -INFINITY;
      scores[j] = sc;
      local_max = fmaxf(local_max, sc);
    }
    const float batch_max = fmaxf(local_max, lane_xor16(local_max));
    const bool has_data = isfinite(batch_max);
    const float next_max = has_data ? fmaxf(running_max, batch_max) : running_max;
    const float prev_scale =
        !has_data ? 1.f : (isfinite(running_max) ? expf(running_max - next_max) : 0.f);
#pragma unroll
    for (int dt = 0; dt < HEAD_DIM / 16; ++dt) {
#pragma unroll
      for (int j = 0; j < 8; ++j) out_acc[dt][j] *= prev_scale;
    }
    float local_sum = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float pr = (has_data && isfinite(scores[j])) ? expf(scores[j] - next_max) : 0.f;
      scores[j] = pr;
      local_sum += pr;
    }
    const float batch_sum = local_sum + lane_xor16(local_sum);
    running_sum = prev_scale * running_sum + batch_sum;
    running_max = has_data ? next_max : running_max;

    half16_t p_frag;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float own = scores[j];
      const float partner = lane_xor16(own);
      p_frag[2 * j + half_id] = static_cast<half_t>(own);
      p_frag[2 * j + (half_id ^ 1)] = static_cast<half_t>(partner);
    }
#pragma unroll
    for (int dt = 0; dt < HEAD_DIM / 16; ++dt) {
      half16_t v_frag;
#pragma unroll
      for (int k = 0; k < 16; ++k) v_frag[k] = v_tile[k * HEAD_DIM + dt * 16 + lane16];
      out_acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(v_frag, p_frag, out_acc[dt]);
    }
    __syncthreads();
  }

  if (!active || !head_active) return;
  const float inv_l = running_sum > 0.f ? 1.f / running_sum : 0.f;
  const int64_t qh = q_head_base + lane16;
#pragma unroll
  for (int dt = 0; dt < HEAD_DIM / 16; ++dt) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int dim = dt * 16 + 2 * j + half_id;
      out[(token * n_q_heads + qh) * HEAD_DIM + dim] =
          f32_to_bf16(out_acc[dt][j] * inv_l);
    }
  }
}

}  // namespace

extern "C" int hipengine_muse_glimmer_draft_flash_wmma(
    const void* q, const void* k, const void* v, void* out,
    int64_t n_q_heads, int64_t n_kv_heads, int64_t key_end,
    int64_t tokens, int64_t window, float scale, hipStream_t stream) {
  if (!q || !k || !v || !out || n_q_heads <= 0 || n_kv_heads <= 0 ||
      n_q_heads % n_kv_heads != 0 || key_end <= 0 || tokens <= 0) {
    return hipErrorInvalidValue;
  }
  const int64_t gqa = n_q_heads / n_kv_heads;
  if (gqa > 16) return hipErrorInvalidValue;
  const dim3 grid(static_cast<unsigned int>((tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK),
                  static_cast<unsigned int>(n_kv_heads));
  hipLaunchKernelGGL(muse_glimmer_draft_flash_wmma_kernel, grid, dim3(MT_THREADS), 0, stream,
                     reinterpret_cast<const uint16_t*>(q),
                     reinterpret_cast<const uint16_t*>(k),
                     reinterpret_cast<const uint16_t*>(v),
                     reinterpret_cast<uint16_t*>(out),
                     n_q_heads, n_kv_heads, key_end, tokens, window, gqa, scale);
  return hipGetLastError();
}