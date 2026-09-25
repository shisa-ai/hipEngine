// Gemma 4 prefill attention on gfx1100.
//
// Every Qwen3.5 prefill attention variant is *gate-shaped*: it reads an
// attention gate and multiplies the result by `sigmoid(gate)`. Gemma 4 has no
// such gate. Laguna's ungated prefill kernel hard-codes Laguna's head geometry,
// so neither can be reused and this family exists.
//
// Three details of the Gemma 4 attention contract are easy to get wrong:
//
//   * `scale` is always 1.0, never `head_dim**-0.5`. Gemma 4 folds the softmax
//     scaling into the query norm weight, so the kernel multiplies by whatever
//     scale it is handed rather than assuming the usual reciprocal square root.
//   * The mask is a *keep* mask supplied by the caller, not an implied causal
//     test. Sliding-window layers need `key > query - window` in addition to
//     `key <= query`, and which layers those are is a config decision, so the
//     kernel applies the mask and does not re-derive it.
//   * GQA maps query head `h` to key/value head `h / (num_heads / num_kv_heads)`.
//
// This kernel is a correctness-first block kernel: one block per (token, head)
// pair, three passes over the key block (max, denominator, weighted sum). It is
// not the fastest possible shape — a flash-style single-pass kernel with an
// online softmax would cut the key traffic threefold — but it is the one that is
// obviously right, and prefill speed is not the current constraint.
//
// Decode (tokens == 1) does not use this kernel. It launches
// `gemma4_attention_decode_kernel` below, which computes the *same arithmetic*
// — same products, same stride-tree order, same per-lane logit partitions —
// with two workgroup barriers per block instead of roughly 2 * keys. The two
// entry points are compared bit-for-bit by the decode parity tests.

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int kGemma4AttentionMaxThreads = 256;

__device__ inline float gemma4_attn_bf16_bits_to_float(uint16_t value) {
  union {
    uint32_t u32;
    float f32;
  } out;
  out.u32 = static_cast<uint32_t>(value) << 16;
  return out.f32;
}

__device__ inline uint16_t gemma4_attn_float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1U;
  in.u32 += 0x7FFFU + lsb;
  return static_cast<uint16_t>(in.u32 >> 16);
}

template <typename scalar_t>
__device__ inline scalar_t gemma4_attn_to_scalar(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ inline uint16_t gemma4_attn_to_scalar<uint16_t>(float value) {
  return gemma4_attn_float_to_bf16_bits(value);
}

template <typename scalar_t>
__device__ inline float gemma4_attn_to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float gemma4_attn_to_float<uint16_t>(uint16_t value) {
  return gemma4_attn_bf16_bits_to_float(value);
}

__device__ inline float gemma4_attn_block_max(float* partial, float value) {
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  return partial[0];
}

__device__ inline float gemma4_attn_block_sum(float* partial, float value) {
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return partial[0];
}

// One block per (token, head). Shared layout:
//   q_s      [head_dim]     the query row, loaded once
//   logits_s [tokens]       per-key logits, written in pass 1 and reused
//   partial  [blockDim]     reduction scratch
template <typename scalar_t>
__global__ void gemma4_attention_prefill_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const uint8_t* __restrict__ keep_mask,
    scalar_t* __restrict__ out,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int64_t tokens,
    int64_t keys) {
  extern __shared__ float shared[];
  float* q_s = shared;
  float* logits_s = q_s + head_dim;
  float* partial = logits_s + keys;

  const int64_t token = static_cast<int64_t>(blockIdx.x) / num_heads;
  const int64_t head = static_cast<int64_t>(blockIdx.x) - token * num_heads;
  const int64_t kv_groups = num_heads / num_kv_heads;
  const int64_t kv_head = head / kv_groups;

  const scalar_t* q_row = query + (token * num_heads + head) * head_dim;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    q_s[d] = gemma4_attn_to_float(q_row[d]);
  }
  __syncthreads();

  // The key count is separate from the query count so this same kernel serves
  // both a dense prefill block (keys == tokens) and a decode step reading a
  // KV cache (tokens == 1, keys == live context).
  const uint8_t* mask_row = keep_mask + token * keys;

  // Pass 1: logits and their running maximum. A masked-out key is -inf so it
  // contributes nothing to either the denominator or the numerator.
  float running_max = -INFINITY;
  for (int64_t j = 0; j < keys; ++j) {
    float logit = -INFINITY;
    if (mask_row[j] != 0) {
      const scalar_t* k_row = key + (j * num_kv_heads + kv_head) * head_dim;
      float dot = 0.0f;
      for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dot += q_s[d] * gemma4_attn_to_float(k_row[d]);
      }
      logit = gemma4_attn_block_sum(partial, dot) * scale;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      logits_s[j] = logit;
    }
    running_max = fmaxf(running_max, logit);
    __syncthreads();
  }
  const float row_max = gemma4_attn_block_max(partial, running_max);
  __syncthreads();

  // Pass 2: denominator.
  float local_sum = 0.0f;
  for (int64_t j = threadIdx.x; j < keys; j += blockDim.x) {
    local_sum += expf(logits_s[j] - row_max);
  }
  const float denominator = gemma4_attn_block_sum(partial, local_sum);
  __syncthreads();

  // Pass 3: weighted sum of V, then the normalise.
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t j = 0; j < keys; ++j) {
      const float weight = expf(logits_s[j] - row_max);
      if (weight == 0.0f) {
        continue;
      }
      const scalar_t* v_row = value + (j * num_kv_heads + kv_head) * head_dim;
      acc += weight * gemma4_attn_to_float(v_row[d]);
    }
    out[(token * num_heads + head) * head_dim + d] =
        gemma4_attn_to_scalar<scalar_t>(acc / denominator);
  }
}

template <typename scalar_t>
int launch_gemma4_attention_prefill(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    const uint8_t* keep_mask,
    scalar_t* out,
    int64_t tokens,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream,
    int64_t keys) {
  if (tokens <= 0 || keys <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (num_heads % num_kv_heads != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  // The tree reductions require a power-of-two block width. Rounding down
  // would leave dimensions unrepresented; launching head_dim threads directly
  // drops lanes whenever head_dim is not a power of two.
  int threads = 1;
  while (threads < head_dim && threads < kGemma4AttentionMaxThreads) {
    threads <<= 1;
  }
  const size_t shared_bytes =
      static_cast<size_t>(head_dim + keys + threads) * sizeof(float);
  if (shared_bytes > 64 * 1024) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  hipLaunchKernelGGL(
      gemma4_attention_prefill_kernel<scalar_t>,
      dim3(static_cast<unsigned int>(tokens * num_heads)),
      dim3(static_cast<unsigned int>(threads)),
      shared_bytes,
      stream,
      query,
      key,
      value,
      keep_mask,
      out,
      num_heads,
      num_kv_heads,
      head_dim,
      scale,
      tokens,
      keys);
  return static_cast<int>(hipGetLastError());
}

// Decode twin of the block kernel above, bit-exact by construction.
//
// Decode keeps the block kernel's parallel shape — one block per (token,
// head), blockDim = next_pow2(head_dim) capped at kGemma4AttentionMaxThreads —
// because that shape hides load latency across waves. What it changes is the
// barrier schedule, where the measured cost lives (a workgroup barrier costs
// about 200 cycles; the block kernel executes roughly 10 per key):
//
//   * pass 1 processes kTile keys per barrier round. Each key's reduction
//     occupies its own partial[kb] row, so kTile keys share one
//     write/3-stride/broadcast barrier sequence: 5 barriers per kTile keys
//     instead of about 10 per key.
//   * within a round, strides >= 32 cross waves through shared partials and
//     strides < 32 stay inside one wave (__shfl_down, no barrier) — the same
//     (t, t + stride) pairs in the same descending order, so every key's sum
//     is bit-identical to the block kernel's block_sum.
//   * pass 2 stores exp(logit - row_max) back over its logit, so pass 3 reads
//     the weight instead of recomputing expf per (element, key). The stored
//     value is expf of the same arguments on the same device: the same bits
//     the block kernel recomputes.
//
// The shuffles stay outside their lane guards: on gfx1100 a __shfl_down whose
// source lane sits in an inactive branch returns 0, which silently drops
// terms. All lanes of a wave always execute the shuffle; only lanes below the
// stride accumulate.
//
// kTile is a template parameter so the slot arrays live in registers; the
// launcher picks the largest tile that fits the LDS budget.

// Eight-value tree reproducing the block kernel's 256-lane reduction exactly:
// strides 128/64/32/16/8 are lane distances 16/8/4/2/1 and strides 4/2/1 are
// in-lane adds on lane 0. Returns the element-0 result on every lane.
__device__ __forceinline__ float gemma4_warp_tree(float (&v)[8], unsigned int lane) {
#pragma unroll
  for (int distance = 16; distance >= 1; distance >>= 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float other = __shfl_down(v[i], distance);
      if (lane < static_cast<unsigned int>(distance)) {
        v[i] += other;
      }
    }
  }
  if (lane == 0) {
    v[0] += v[4];
    v[1] += v[5];
    v[2] += v[6];
    v[3] += v[7];
    v[0] += v[2];
    v[1] += v[3];
    v[0] += v[1];
  }
  return v[0];
}

// ---------------------------------------------------------------------------
// Warp-per-(token, head) decode kernel.
//
// The block kernel reduces 256 tree lanes through LDS: a barrier round per
// stride >= 32, a shared partial row per key, and a publish/broadcast pair per
// tile. Every one of those is a cross-wave round trip, and measurement puts the
// cost there rather than in the loads or the arithmetic (ablation study,
// worklog entry 20260925T194545.594823Z). This kernel keeps the same 256-lane
// tree, the same ascending-j publish order, the same pass-2 lane partition and
// the same pass-3 per-dimension chains, but maps eight tree lanes onto each
// warp lane so the whole reduction is five shuffles plus three in-lane adds
// with no LDS traffic and no barriers inside the key walk.
//
// Tree-lane layout: warp lane w owns tree lanes [8w, 8w+8). The block kernel's
// strides 128/64/32/16/8 become lane distances 16/8/4/2/1 and its strides
// 4/2/1 become in-lane adds on lane 0 - the same pairs in the same order, so
// every logit and the denominator are bit-identical to the block kernel's.
//
// The shuffles stay outside their lane guards for the same reason the block
// kernel's do: on gfx1100 a __shfl_down whose source lane sits in an inactive
// branch returns 0 and silently drops terms.
template <typename scalar_t, int kHalves, int kKeysPerTile, int kWarpsPerBlock>
__global__ void gemma4_attention_decode_warp_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const uint8_t* __restrict__ keep_mask,
    scalar_t* __restrict__ out,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int64_t tokens,
    int64_t keys) {
  constexpr int kWarpLanes = 32;
  constexpr int kOwn = 8;                       // tree lanes per warp lane
  constexpr int kTreeLanes = kWarpLanes * kOwn; // 256, the block kernel's lane count

  extern __shared__ float shared[];
  const unsigned int lane = threadIdx.x & (kWarpLanes - 1);
  const unsigned int warp = threadIdx.x / kWarpLanes;
  const int64_t head_index =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + static_cast<int64_t>(warp);
  if (head_index >= tokens * num_heads) {
    return;
  }
  float* logits_s = shared + static_cast<size_t>(warp) * static_cast<size_t>(keys);

  const int64_t token = head_index / num_heads;
  const int64_t head = head_index - token * num_heads;
  const int64_t kv_head = head / (num_heads / num_kv_heads);
  const int base = static_cast<int>(lane) * kOwn;

  const scalar_t* q_row = query + head_index * head_dim;
  float q_reg[kHalves][kOwn];
#pragma unroll
  for (int h = 0; h < kHalves; ++h) {
#pragma unroll
    for (int i = 0; i < kOwn; ++i) {
      q_reg[h][i] = gemma4_attn_to_float(q_row[h * kTreeLanes + base + i]);
    }
  }

  const uint8_t* mask_row = keep_mask + token * keys;

  // Pass 1: per-key logits. A whole tile of keys is in flight at once - mask
  // bytes first, then every key's K row unconditionally (masked and
  // out-of-range keys read the clamped row 0 and are discarded at publish), so
  // the tile's memory-level parallelism is exposed instead of one key per
  // memory round trip. running_max walks the same ascending-j fmaxf chain.
  float running_max = -INFINITY;
  for (int64_t j0 = 0; j0 < keys; j0 += kKeysPerTile) {
    float dot[kKeysPerTile][kOwn];
    bool active[kKeysPerTile];
#pragma unroll
    for (int t = 0; t < kKeysPerTile; ++t) {
      const int64_t j = j0 + t;
      const bool in_range = j < keys;
      active[t] = in_range && mask_row[in_range ? j : 0] != 0;
    }
#pragma unroll
    for (int t = 0; t < kKeysPerTile; ++t) {
      const int64_t j = j0 + t;
      const int64_t j_load = j < keys ? j : 0;
      const scalar_t* k_row = key + (j_load * num_kv_heads + kv_head) * head_dim;
#pragma unroll
      for (int i = 0; i < kOwn; ++i) {
        dot[t][i] = 0.0f;
      }
#pragma unroll
      for (int h = 0; h < kHalves; ++h) {
#pragma unroll
        for (int i = 0; i < kOwn; ++i) {
          dot[t][i] += q_reg[h][i] * gemma4_attn_to_float(k_row[h * kTreeLanes + base + i]);
        }
      }
    }
#pragma unroll
    for (int t = 0; t < kKeysPerTile; ++t) {
      const int64_t j = j0 + t;
      float v[kOwn];
#pragma unroll
      for (int i = 0; i < kOwn; ++i) {
        v[i] = dot[t][i];
      }
      const float reduced = gemma4_warp_tree(v, lane);
      if (lane == 0) {
        const float logit = active[t] ? reduced * scale : -INFINITY;
        if (j < keys) {
          logits_s[j] = logit;
        }
        running_max = fmaxf(running_max, logit);
      }
    }
  }
  const float row_max = __shfl(running_max, 0);

  // Pass 2: the block kernel's per-lane denominator partition - tree lane L
  // accumulates keys j == L (mod 256) in ascending order - with the weights
  // stored back over their logits for pass 3 to reuse.
  float acc[kOwn];
#pragma unroll
  for (int i = 0; i < kOwn; ++i) {
    acc[i] = 0.0f;
  }
  for (int64_t j0 = base; j0 < keys; j0 += kTreeLanes) {
#pragma unroll
    for (int i = 0; i < kOwn; ++i) {
      const int64_t j = j0 + i;
      if (j < keys) {
        const float weight = expf(logits_s[j] - row_max);
        acc[i] += weight;
        logits_s[j] = weight;
      }
    }
  }
  const float denominator = __shfl(gemma4_warp_tree(acc, lane), 0);

  // Pass 3: the block kernel's weighted sums with the precomputed weight, one
  // ascending-j chain per dimension.
  float out_reg[kHalves][kOwn];
#pragma unroll
  for (int h = 0; h < kHalves; ++h) {
#pragma unroll
    for (int i = 0; i < kOwn; ++i) {
      out_reg[h][i] = 0.0f;
    }
  }
  for (int64_t j = 0; j < keys; ++j) {
    const float weight = logits_s[j];
    if (weight == 0.0f) {
      continue;
    }
    const scalar_t* v_row = value + (j * num_kv_heads + kv_head) * head_dim;
#pragma unroll
    for (int h = 0; h < kHalves; ++h) {
#pragma unroll
      for (int i = 0; i < kOwn; ++i) {
        out_reg[h][i] += weight * gemma4_attn_to_float(v_row[h * kTreeLanes + base + i]);
      }
    }
  }
  scalar_t* out_row = out + head_index * head_dim;
#pragma unroll
  for (int h = 0; h < kHalves; ++h) {
#pragma unroll
    for (int i = 0; i < kOwn; ++i) {
      out_row[h * kTreeLanes + base + i] =
          gemma4_attn_to_scalar<scalar_t>(out_reg[h][i] / denominator);
    }
  }
}

// Single-value tree for pass 2: shared partials across waves, shuffles within
// one wave, returns the element-0 result to every lane.
__device__ inline float gemma4_decode_tree1(
    float* partial,
    float value,
    unsigned int threads,
    unsigned int tid) {
  partial[tid] = value;
  __syncthreads();
  for (unsigned int stride = threads >> 1; stride >= 32; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  float v = partial[tid];
  unsigned int start = threads >> 1;
  if (start > 16) {
    start = 16;
  }
  for (unsigned int stride = start; stride >= 1; stride >>= 1) {
    const float other = __shfl_down(v, stride);
    if (tid < stride) {
      v += other;
    }
  }
  if (tid == 0) {
    partial[0] = v;
  }
  __syncthreads();
  return partial[0];
}

template <typename scalar_t, int kTile>
__global__ void gemma4_attention_decode_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const uint8_t* __restrict__ keep_mask,
    scalar_t* __restrict__ out,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int64_t tokens,
    int64_t keys) {
  extern __shared__ float shared[];
  float* q_s = shared;
  float* logits_s = q_s + head_dim;
  float* partial = logits_s + keys;
  const unsigned int threads = blockDim.x;
  const unsigned int tid = threadIdx.x;

  const int64_t token = static_cast<int64_t>(blockIdx.x) / num_heads;
  const int64_t head = static_cast<int64_t>(blockIdx.x) - token * num_heads;
  const int64_t kv_head = head / (num_heads / num_kv_heads);

  const scalar_t* q_row = query + (token * num_heads + head) * head_dim;
  for (int64_t d = tid; d < head_dim; d += threads) {
    q_s[d] = gemma4_attn_to_float(q_row[d]);
  }
  __syncthreads();

  const uint8_t* mask_row = keep_mask + token * keys;

  // Pass 1: batched per-key logits. Phase A computes a tile of keys' dots
  // (one bundled barrier round follows), phase B reduces all slots together,
  // phase C publishes each logit from lane 0 and broadcasts it through
  // partial[0..kTile). running_max walks the same ascending-j fmaxf chain the
  // block kernel walks.
  float running_max = -INFINITY;
  for (int64_t j0 = 0; j0 < keys; j0 += kTile) {
    float slot[kTile];
    bool active[kTile];
    for (int kb = 0; kb < kTile; ++kb) {
      const int64_t j = j0 + kb;
      const bool keep = j < keys && mask_row[j] != 0;
      float dot = 0.0f;
      if (keep) {
        const scalar_t* k_row = key + (j * num_kv_heads + kv_head) * head_dim;
        for (int64_t d = tid; d < head_dim; d += threads) {
          dot += q_s[d] * gemma4_attn_to_float(k_row[d]);
        }
      }
      active[kb] = keep;
      partial[kb * threads + tid] = dot;
    }
    __syncthreads();
    for (unsigned int stride = threads >> 1; stride >= 32; stride >>= 1) {
      if (tid < stride) {
        for (int kb = 0; kb < kTile; ++kb) {
          partial[kb * threads + tid] += partial[kb * threads + tid + stride];
        }
      }
      __syncthreads();
    }
    for (int kb = 0; kb < kTile; ++kb) {
      slot[kb] = partial[kb * threads + tid];
    }
    unsigned int start = threads >> 1;
    if (start > 16) {
      start = 16;
    }
    for (unsigned int stride = start; stride >= 1; stride >>= 1) {
      for (int kb = 0; kb < kTile; ++kb) {
        const float other = __shfl_down(slot[kb], stride);
        if (tid < stride) {
          slot[kb] += other;
        }
      }
    }
    if (tid == 0) {
      for (int kb = 0; kb < kTile; ++kb) {
        const int64_t j = j0 + kb;
        const float logit = active[kb] ? slot[kb] * scale : -INFINITY;
        if (j < keys) {
          logits_s[j] = logit;
        }
        partial[kb] = logit;
      }
    }
    __syncthreads();
    for (int kb = 0; kb < kTile; ++kb) {
      running_max = fmaxf(running_max, partial[kb]);
    }
  }
  const float row_max = running_max;

  // Pass 2: the block kernel's per-lane denominator partition (j == lane, mod
  // blockDim), accumulating the same expf terms in the same order — and
  // storing each weight over its logit for pass 3 to reuse. Every logit is
  // visible: the last tile's broadcast sync ends pass 1.
  float local_sum = 0.0f;
  for (int64_t j = tid; j < keys; j += threads) {
    const float weight = expf(logits_s[j] - row_max);
    local_sum += weight;
    logits_s[j] = weight;
  }
  const float denominator = gemma4_decode_tree1(partial, local_sum, threads, tid);

  // Pass 3: the block kernel's weighted sums with the precomputed weight.
  for (int64_t d = tid; d < head_dim; d += threads) {
    float acc = 0.0f;
    for (int64_t j = 0; j < keys; ++j) {
      const float weight = logits_s[j];
      if (weight == 0.0f) {
        continue;
      }
      const scalar_t* v_row = value + (j * num_kv_heads + kv_head) * head_dim;
      acc += weight * gemma4_attn_to_float(v_row[d]);
    }
    out[(token * num_heads + head) * head_dim + d] =
        gemma4_attn_to_scalar<scalar_t>(acc / denominator);
  }
}

template <typename scalar_t, int kTile>
void gemma4_decode_launch(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    const uint8_t* keep_mask,
    scalar_t* out,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int64_t tokens,
    int64_t keys,
    hipStream_t stream,
    unsigned int grid,
    unsigned int threads,
    size_t shared_bytes) {
  gemma4_attention_decode_kernel<scalar_t, kTile><<<dim3(grid), dim3(threads), shared_bytes, stream>>>(
      query,
      key,
      value,
      keep_mask,
      out,
      num_heads,
      num_kv_heads,
      head_dim,
      scale,
      tokens,
      keys);
}

// Warp-per-head decode launcher. head_dim must be a multiple of the block
// kernel's 256 tree lanes (1 or 2 halves here); the caller falls back to the
// block kernel for every other geometry. Shared memory is one logit row per
// warp, which is smaller than the block kernel's requirement for the same
// shape, so the caller's shared-bytes check stays a valid upper bound.
template <typename scalar_t, int kHalves, int kWarpsPerBlock>
int gemma4_decode_warp_launch(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    const uint8_t* keep_mask,
    scalar_t* out,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int64_t tokens,
    int64_t keys,
    hipStream_t stream) {
  constexpr int kKeysPerTile = 4;
  const int64_t heads = tokens * num_heads;
  const unsigned int grid =
      static_cast<unsigned int>((heads + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const unsigned int threads = static_cast<unsigned int>(kWarpsPerBlock) * 32u;
  const size_t shared_bytes =
      static_cast<size_t>(kWarpsPerBlock) * static_cast<size_t>(keys) * sizeof(float);
  gemma4_attention_decode_warp_kernel<scalar_t, kHalves, kKeysPerTile, kWarpsPerBlock>
      <<<dim3(grid), dim3(threads), shared_bytes, stream>>>(
          query,
          key,
          value,
          keep_mask,
          out,
          num_heads,
          num_kv_heads,
          head_dim,
          scale,
          tokens,
          keys);
  return static_cast<int>(hipGetLastError());
}

template <typename scalar_t, int kHalves>
int gemma4_decode_warp_dispatch(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    const uint8_t* keep_mask,
    scalar_t* out,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int64_t tokens,
    int64_t keys,
    hipStream_t stream) {
  // Pack as many (token, head) pairs per block as the logit rows allow, then
  // fall back to the block kernel if even one row does not fit.
  const size_t row_bytes = static_cast<size_t>(keys) * sizeof(float);
  if (row_bytes * 4 <= 48 * 1024) {
    return gemma4_decode_warp_launch<scalar_t, kHalves, 4>(
        query, key, value, keep_mask, out, num_heads, num_kv_heads, head_dim, scale, tokens,
        keys, stream);
  }
  if (row_bytes * 2 <= 48 * 1024) {
    return gemma4_decode_warp_launch<scalar_t, kHalves, 2>(
        query, key, value, keep_mask, out, num_heads, num_kv_heads, head_dim, scale, tokens,
        keys, stream);
  }
  if (row_bytes <= 48 * 1024) {
    return gemma4_decode_warp_launch<scalar_t, kHalves, 1>(
        query, key, value, keep_mask, out, num_heads, num_kv_heads, head_dim, scale, tokens,
        keys, stream);
  }
  return -1;  // no warp-kernel shape fits: caller uses the block kernel
}

// Which decode kernel the launcher last selected: 1 = warp-per-head (head_dim
// == 512), 0 = block kernel. Introspection only - it changes no behaviour, and
// it is what the routing test asserts instead of inferring the path from
// timings.
inline int& gemma4_decode_selection() {
  static int selection = 0;
  return selection;
}

template <typename scalar_t>
int launch_gemma4_attention_decode(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    const uint8_t* keep_mask,
    scalar_t* out,
    int64_t tokens,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream,
    int64_t keys) {
  if (tokens <= 0 || keys <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (num_heads % num_kv_heads != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  int threads_i = 1;
  while (threads_i < head_dim && threads_i < kGemma4AttentionMaxThreads) {
    threads_i <<= 1;
  }
  const unsigned int threads = static_cast<unsigned int>(threads_i);
  // Warp-per-head path: head_dim == 512 (two of the block kernel's 256 tree
  // lanes per row), where the shuffle-only tree measured 28% faster than the
  // block kernel (851 vs 1186 us at keys=1024). At head_dim == 256 the same
  // kernel loses (822 vs 700 us) because one warp per (token, head) leaves 16
  // warps on the machine and there is not enough concurrency to hide the key
  // walk, so that geometry keeps the block kernel.
  if (head_dim == 512) {
    const int rc = gemma4_decode_warp_dispatch<scalar_t, 2>(query, key, value, keep_mask, out,
        num_heads, num_kv_heads, head_dim, scale, tokens, keys, stream);
    if (rc >= 0) {
      gemma4_decode_selection() = 1;  // warp-per-head
      return rc;
    }
  }
  gemma4_decode_selection() = 0;  // block kernel
  // Largest barrier batch whose partial rows still fit the LDS budget. kTile
  // 1 always fits whenever the caller's shared-bytes check passed.
  int tile = 1;
  for (int candidate = 8; candidate > 1; candidate >>= 1) {
    const size_t candidate_bytes =
        static_cast<size_t>(head_dim + keys + candidate * threads_i) * sizeof(float);
    if (candidate_bytes <= 64 * 1024) {
      tile = candidate;
      break;
    }
  }
  const size_t shared_bytes =
      static_cast<size_t>(head_dim + keys + tile * threads_i) * sizeof(float);
  const unsigned int grid = static_cast<unsigned int>(tokens * num_heads);
  switch (tile) {
    case 16:
      gemma4_decode_launch<scalar_t, 16>(query, key, value, keep_mask, out, num_heads,
          num_kv_heads, head_dim, scale, tokens, keys, stream, grid, threads, shared_bytes);
      break;
    case 8:
      gemma4_decode_launch<scalar_t, 8>(query, key, value, keep_mask, out, num_heads,
          num_kv_heads, head_dim, scale, tokens, keys, stream, grid, threads, shared_bytes);
      break;
    case 4:
      gemma4_decode_launch<scalar_t, 4>(query, key, value, keep_mask, out, num_heads,
          num_kv_heads, head_dim, scale, tokens, keys, stream, grid, threads, shared_bytes);
      break;
    case 2:
      gemma4_decode_launch<scalar_t, 2>(query, key, value, keep_mask, out, num_heads,
          num_kv_heads, head_dim, scale, tokens, keys, stream, grid, threads, shared_bytes);
      break;
    default:
      gemma4_decode_launch<scalar_t, 1>(query, key, value, keep_mask, out, num_heads,
          num_kv_heads, head_dim, scale, tokens, keys, stream, grid, threads, shared_bytes);
      break;
  }
  return static_cast<int>(hipGetLastError());
}


}  // namespace

// Returns 1 when the last decode launch used the warp-per-head kernel, 0 when it
// used the block kernel. Test/introspection surface for the routing contract.
extern "C" int hipengine_gemma4_attention_decode_variant(void) {
  return gemma4_decode_selection();
}

extern "C" int hipengine_gemma4_attention_prefill_bf16(
    const uint16_t* query,
    const uint16_t* key,
    const uint16_t* value,
    const uint8_t* keep_mask,
    uint16_t* out,
    int64_t tokens,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream,
    int64_t keys) {
  return launch_gemma4_attention_prefill<uint16_t>(
      query, key, value, keep_mask, out, tokens, num_heads, num_kv_heads, head_dim, scale,
      stream, keys);
}

// F32 entry point for validation against the CPU reference: the reference works
// in f32, so comparing against it through a bf16 round trip would measure the
// rounding rather than the kernel.
extern "C" int hipengine_gemma4_attention_prefill_f32(
    const float* query,
    const float* key,
    const float* value,
    const uint8_t* keep_mask,
    float* out,
    int64_t tokens,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream,
    int64_t keys) {
  return launch_gemma4_attention_prefill<float>(
      query, key, value, keep_mask, out, tokens, num_heads, num_kv_heads, head_dim, scale,
      stream, keys);
}

// Decode entry points: tokens == 1 routes here from the Python wrapper. Same
// arithmetic as the prefill kernel above, two barriers per block instead of
// ~2 * keys — see gemma4_attention_decode_kernel.
extern "C" int hipengine_gemma4_attention_decode_bf16(
    const uint16_t* query,
    const uint16_t* key,
    const uint16_t* value,
    const uint8_t* keep_mask,
    uint16_t* out,
    int64_t tokens,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream,
    int64_t keys) {
  return launch_gemma4_attention_decode<uint16_t>(
      query, key, value, keep_mask, out, tokens, num_heads, num_kv_heads, head_dim, scale,
      stream, keys);
}

extern "C" int hipengine_gemma4_attention_decode_f32(
    const float* query,
    const float* key,
    const float* value,
    const uint8_t* keep_mask,
    float* out,
    int64_t tokens,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    hipStream_t stream,
    int64_t keys) {
  return launch_gemma4_attention_decode<float>(
      query, key, value, keep_mask, out, tokens, num_heads, num_kv_heads, head_dim, scale,
      stream, keys);
}
