// Muse Glimmer WMMA flash prefill attention (GQA-packed, contiguous KV).
//
// One block owns one query token and one KV head's 16 Q heads (GQA 16:1),
// walking the causal/SWA key range in 16-key tiles. QK^T and PV use the
// 16x16x16 f16 WMMA (the 16 Q heads pack the WMMA row dimension), with FP32
// online softmax between tiles. K/V for each 16-key slice are staged in LDS
// once and reused across all 16 Q heads. This replaces the scalar lane-dot
// prefill path (muse_glimmer_decode_attention tokens>1) whose QK/PV dots run
// one key at a time per wave.
//
// KV layout: [max_ctx, n_kv_heads, 128] bf16 contiguous. Causal frame:
// position = key_end - tokens + token; row attends keys <= position. SWA
// (window>0) additionally drops keys <= position - window. NoPE full layers
// pass window=0 and attend the whole causal prefix.
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int HEAD_DIM = 128;
constexpr int GQA = 16;          // Q heads per KV head (32 / 2)
constexpr int K_TILE = 16;       // keys per WMMA tile
constexpr int WAVES = 1;
constexpr int THREADS = WAVES * 32;

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
__device__ inline float lane_xor16(float v) {
  return __shfl_xor(v, 16);
}

// Each wave owns 4 of the 16 Q heads (GQA/ WAVES). The WMMA row dimension
// packs the 16 Q heads; lane&15 selects the Q head, lane>>4 the score pair.
__global__ __launch_bounds__(THREADS, 1)
void muse_glimmer_flash_prefill_wmma_kernel(
    const uint16_t* __restrict__ q,        // [tokens, n_q_heads, 128] bf16
    const uint16_t* __restrict__ k_cache,  // [max_ctx, n_kv_heads, 128] bf16
    const uint16_t* __restrict__ v_cache,
    uint16_t* __restrict__ out,            // [tokens, n_q_heads, 128] bf16
    int64_t n_q_heads, int64_t n_kv_heads, int64_t key_begin, int64_t key_end,
    int64_t tokens, int64_t window, float scale) {
  const int token = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lane16 = lane & 15;
  const int half_id = lane >> 4;
  if (token >= tokens || kv_head >= n_kv_heads) return;
  const int64_t position = key_end - tokens + token;
  int64_t row_begin = key_begin;
  if (window > 0) row_begin = max(row_begin, position - window + 1);
  const int64_t row_end = min(key_end, position + 1);
  const int q_head_base = kv_head * GQA;

  // Load this wave's 4 Q heads' rows? The WMMA packs all 16 Q heads, so every
  // wave cooperatively holds the full 16-head Q tile fragments. Each lane
  // loads Q for head lane16 (16 heads), 16 dims per dim-tile.
  half16_t q_frag[HEAD_DIM / 16];
#pragma unroll
  for (int dt = 0; dt < HEAD_DIM / 16; ++dt) {
    const int64_t qh = q_head_base + lane16;
    const uint16_t* qrow = q + (token * n_q_heads + qh) * HEAD_DIM + dt * 16;
#pragma unroll
    for (int k = 0; k < 16; ++k)
      q_frag[dt][k] = static_cast<half_t>(bf16_to_f32(qrow[k]));
  }

  // Online-softmax state per Q head (lane16), 8 output-dim pairs.
  float running_max = -INFINITY;
  float running_sum = 0.f;
  float8_t out_acc[HEAD_DIM / 16];
#pragma unroll
  for (int dt = 0; dt < HEAD_DIM / 16; ++dt) out_acc[dt] = zero8();

  __shared__ __align__(1024) half_t k_tile[K_TILE * HEAD_DIM];
  __shared__ __align__(1024) half_t v_tile[K_TILE * HEAD_DIM];

  const int64_t kv_stride = n_kv_heads * HEAD_DIM;
  for (int64_t k_base = row_begin; k_base < row_end; k_base += K_TILE) {
    const int keys_this = static_cast<int>(min<int64_t>(K_TILE, row_end - k_base));
    // Stage K and V for the 16-key slice (all 128 dims) cooperatively.
    for (int idx = threadIdx.x; idx < K_TILE * HEAD_DIM; idx += THREADS) {
      const int key_local = idx / HEAD_DIM;
      const int dim = idx % HEAD_DIM;
      const int64_t key = k_base + key_local;
      half_t kv = static_cast<half_t>(0.f);
      half_t vv = static_cast<half_t>(0.f);
      if (key_local < keys_this) {
        const int64_t off = key * kv_stride + kv_head * HEAD_DIM + dim;
        kv = static_cast<half_t>(bf16_to_f32(k_cache[off]));
        vv = static_cast<half_t>(bf16_to_f32(v_cache[off]));
      }
      k_tile[key_local * HEAD_DIM + dim] = kv;
      v_tile[key_local * HEAD_DIM + dim] = vv;
    }
    __syncthreads();

    // QK^T: scores[16 Q heads x 16 keys]. WMMA A=key(16keys x16dim),
    // B=query(16heads x16dim) -> C[16keys x16heads]; lane16=head, pair=key.
    float8_t scores = zero8();
#pragma unroll
    for (int dt = 0; dt < HEAD_DIM / 16; ++dt) {
      half16_t k_frag;
      const int key_local = lane16;  // 16 keys
#pragma unroll
      for (int k = 0; k < 16; ++k)
        k_frag[k] = k_tile[key_local * HEAD_DIM + dt * 16 + k];
      scores = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
          k_frag, q_frag[dt], scores);
    }
    __syncthreads();

    // Mask + scale + running max. scores element j corresponds to
    // key = 2*j + half_id (lane16 = Q head).
    float local_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int key_local = 2 * j + half_id;
      const int64_t key = k_base + key_local;
      bool visible = key_local < keys_this && key <= position;
      if (window > 0) visible = visible && (key > position - window);
      float s = visible ? scores[j] * scale : -INFINITY;
      scores[j] = s;
      local_max = fmaxf(local_max, s);
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
      const float p = (has_data && isfinite(scores[j])) ? expf(scores[j] - next_max) : 0.f;
      scores[j] = p;
      local_sum += p;
    }
    const float batch_sum = local_sum + lane_xor16(local_sum);
    running_sum = prev_scale * running_sum + batch_sum;
    running_max = has_data ? next_max : running_max;

    // PV: out[dim, head] += sum_key V[key, dim] * P[head, key]. WMMA A=V^T
    // (row=dim=lane16, k=key), B=P^T (k=key, col=head=lane16) -> C[row=dim=
    // 2*j+half_id, col=head=lane16]. The P fragment gathers each key's
    // probability from the half_id lane that owns score key=2*j+half_id.
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
      for (int k = 0; k < 16; ++k)
        v_frag[k] = v_tile[k * HEAD_DIM + dt * 16 + lane16];  // V[key=k, dim]
      out_acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
          v_frag, p_frag, out_acc[dt]);
    }
    __syncthreads();
  }

  // Normalize and publish: out[token, q_head_base+lane16, dim].
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

extern "C" int hipengine_muse_glimmer_flash_prefill_wmma(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out,
    int64_t n_q_heads, int64_t n_kv_heads, int64_t key_begin, int64_t key_end,
    int64_t tokens, int64_t window, float scale, hipStream_t stream) {
  if (!q || !k || !v || !out || n_q_heads <= 0 || n_kv_heads <= 0 ||
      n_q_heads != n_kv_heads * GQA || key_end <= key_begin || tokens <= 0)
    return static_cast<int>(hipErrorInvalidValue);
  const dim3 grid(static_cast<unsigned>(tokens),
                  static_cast<unsigned>(n_kv_heads));
  muse_glimmer_flash_prefill_wmma_kernel<<<grid, THREADS, 0, stream>>>(
      q, k, v, out, n_q_heads, n_kv_heads, key_begin, key_end, tokens, window,
      scale);
  return static_cast<int>(hipGetLastError());
}