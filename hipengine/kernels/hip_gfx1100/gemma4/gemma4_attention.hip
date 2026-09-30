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

// Defined below; the prefill launcher routes multi-token blocks into it.
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
    int64_t keys,
    float* split_workspace,
    int split_slices);

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
  // Multi-token blocks use the same key-class / barrier-batched family the
  // decode step does. Its kernels are bit-identical to the block kernel below
  // (same logit trees, same pass-2 and pass-3 partitions) and were measured at
  // 421 us against the block kernel's 652-700 at keys=1024, head_dim 256, so
  // prefill has no reason to keep the ~10 barriers per key the block kernel
  // spends. The fallback below still covers every other geometry.
  if (tokens > 1) {
    const int batched_rc = launch_gemma4_attention_decode<scalar_t>(
        query, key, value, keep_mask, out, tokens, num_heads, num_kv_heads,
        head_dim, scale, stream, keys, nullptr, 0);
    if (batched_rc >= 0) {
      return batched_rc;
    }
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

// The denominator reduction is pinned to this lane count whatever the block
// size is: gemma4_decode_tree1's addition order depends on it, and the block
// kernel (the bit-exactness reference) uses 256 lanes.
constexpr unsigned int kGemma4ReduceThreads = 256;

// Two adjacent head dimensions in one load: the class kernel's pass 3 assigns
// lane L the dimensions 2L and 2L+1, so a 4-byte (16-bit KV) or 8-byte (32-bit
// KV) load replaces two 2-byte ones. The alignas is what tells the compiler the
// pointer is aligned enough to emit a single vector load; it holds because
// head_dim is even for every geometry the launcher routes here and the KV base
// is a device allocation.
template <typename scalar_t>
struct alignas(2 * sizeof(scalar_t)) gemma4_pair2 {
  scalar_t lo;
  scalar_t hi;
};

// Scalar lanes must not form an aligned pair lvalue: odd scalar offsets are
// not pair-aligned, and the final scalar has no second element to read.
template <int kDimsPerLane, typename scalar_t>
__device__ __forceinline__ gemma4_pair2<scalar_t> gemma4_load_v(const scalar_t* ptr) {
  static_assert(kDimsPerLane == 1 || kDimsPerLane == 2, "invalid load width");
  if constexpr (kDimsPerLane == 2) {
    return *reinterpret_cast<const gemma4_pair2<scalar_t>*>(ptr);
  } else {
    return {ptr[0], static_cast<scalar_t>(0)};
  }
}

__device__ inline float gemma4_decode_tree1(
    float* partial, float value, unsigned int threads, unsigned int tid);

// Pass 3 of the key-class decode kernel, factored out so the load width is a
// compile-time choice: kDimsPerLane == 2 issues one gemma4_pair2 load for the
// dimensions 2L and 2L+1 (only used when head_dim >= 2 * threads, so no lane
// goes idle), kDimsPerLane == 1 issues the original 2-byte load. Both walk the
// keys ascending per dimension, which is what keeps the result bit-identical to
// the block kernel.
//
// kPrefetch is the pipeline depth: the V value of key j + kPrefetch is loaded
// while key j is accumulated. Each lane walks every key with one load per key
// inside a dependent accumulate chain, and the `weight == 0` skip stops the
// compiler from hoisting those loads, so without the pipeline the walk
// serialises at memory latency (iteration 22 measured 34 GB/s unique against
// 493 GB/s for the same loop with the loads removed). The pipeline only changes
// *when* a load is issued; the accumulation order is untouched.
//
// ``keys_begin`` is the first key that can carry a non-zero weight, read out of
// the caller's mask. Everything below it is masked, so its weight is the 0 the
// ``weight == 0`` skip already dropped -- which is why starting there removes V
// loads without removing an accumulated term, and why the surviving keys are
// still visited ascending over the same values. The pipe originates at
// ``keys_begin`` for the same reason: it changes which loads are issued and
// nothing else. Callers with no mask-derived bound pass 0 and get the original
// loop back.
template <int kDimsPerLane, int kPrefetch, typename scalar_t>
__device__ __forceinline__ void gemma4_decode_pass3(
    const float* __restrict__ logits_s,
    const scalar_t* __restrict__ v_base,
    scalar_t* __restrict__ out_row,
    int64_t v_row_stride,
    int64_t head_dim,
    int64_t keys_begin,
    int64_t keys,
    float denominator,
    unsigned int threads,
    unsigned int tid) {
  const int64_t step = static_cast<int64_t>(threads) * kDimsPerLane;
  for (int64_t d0 = static_cast<int64_t>(tid) * kDimsPerLane; d0 < head_dim; d0 += step) {
    gemma4_pair2<scalar_t> pipe[kPrefetch];
#pragma unroll
    for (int n = 0; n < kPrefetch; ++n) {
      const int64_t jn = keys_begin + n;
      pipe[n] = jn < keys
          ? gemma4_load_v<kDimsPerLane>(v_base + jn * v_row_stride + d0)
          : gemma4_pair2<scalar_t>{static_cast<scalar_t>(0), static_cast<scalar_t>(0)};
    }
    float acc_lo = 0.0f;
    float acc_hi = 0.0f;
    for (int64_t j = keys_begin; j < keys; ++j) {
      const gemma4_pair2<scalar_t> v_cur = pipe[0];
#pragma unroll
      for (int n = 0; n < kPrefetch - 1; ++n) {
        pipe[n] = pipe[n + 1];
      }
      const int64_t jn = j + kPrefetch;
      pipe[kPrefetch - 1] = jn < keys
          ? gemma4_load_v<kDimsPerLane>(v_base + jn * v_row_stride + d0)
          : gemma4_pair2<scalar_t>{static_cast<scalar_t>(0), static_cast<scalar_t>(0)};
      const float weight = logits_s[j];
      if (weight == 0.0f) {
        continue;
      }
      acc_lo += weight * gemma4_attn_to_float(v_cur.lo);
      if (kDimsPerLane == 2) {
        acc_hi += weight * gemma4_attn_to_float(v_cur.hi);
      }
    }
    out_row[d0] = gemma4_attn_to_scalar<scalar_t>(acc_lo / denominator);
    if (kDimsPerLane == 2 && d0 + 1 < head_dim) {
      out_row[d0 + 1] = gemma4_attn_to_scalar<scalar_t>(acc_hi / denominator);
    }
  }
}

// ---------------------------------------------------------------------------
// Key-class decode kernel: the block kernel's 128-warp concurrency with the
// intra-warp per-key reduction.
//
// Eight warps share one (token, head); warp w owns the keys w, w+8, w+16, ...
// and computes each of their logits with the shuffle-exact 256-lane tree above,
// so pass 1 has no LDS traffic and no barriers at all. The two cross-warp
// reductions move out of the key walk: each warp publishes its own running
// maximum and the row maximum is the fmaxf of the eight (order-independent, so
// bit-exact). Pass 2 and pass 3 are the block kernel's own partitions verbatim -
// tree lane L accumulates keys j == L (mod 256) ascending, and each dimension
// keeps its ascending-j chain - so only pass 1 changed shape.
template <typename scalar_t, int kHalves, int kKeysPerTile>
__global__ void gemma4_attention_decode_class_kernel(
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
    int64_t keys,
    // Split phase 1 (see gemma4_attention_decode_dimension_kernel): when
    // ``split_weights`` is non-null this kernel stops after pass 2 and writes the
    // precomputed weights plus the (denominator, row max) header instead of
    // running pass 3. The incumbent single-kernel path passes nullptr and is
    // byte-for-byte the kernel it was before this parameter existed.
    float* __restrict__ split_weights = nullptr,
    float* __restrict__ split_header = nullptr) {
  constexpr int kWarpLanes = 32;
  constexpr int kOwn = 8;                       // tree lanes per warp lane
  constexpr int kTreeLanes = kWarpLanes * kOwn; // 256, the block kernel's lane count

  extern __shared__ float shared[];
  float* logits_s = shared;
  // One kTreeLanes-float denominator region per 256-thread group, so every
  // group can run the pinned 256-lane tree without colliding; only group 0's
  // result is used. See kGemma4ReduceThreads.
  float* partial = logits_s + keys;
  float* max_s = partial + kTreeLanes * (blockDim.x / kGemma4ReduceThreads);

  const unsigned int threads = blockDim.x;
  const unsigned int tid = threadIdx.x;
  const unsigned int lane = tid & (kWarpLanes - 1);
  const unsigned int warp = tid / kWarpLanes;
  const unsigned int warps = threads / kWarpLanes;

  const int64_t head_index = static_cast<int64_t>(blockIdx.x);
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

  // Pass 1: this warp's key class, kKeysPerTile keys in flight, each logit
  // reduced by the intra-warp tree.
  float running_max = -INFINITY;
  for (int64_t j0 = static_cast<int64_t>(warp); j0 < keys;
       j0 += static_cast<int64_t>(warps) * kKeysPerTile) {
    float dot[kKeysPerTile][kOwn];
    bool active[kKeysPerTile];
#pragma unroll
    for (int t = 0; t < kKeysPerTile; ++t) {
      const int64_t j = j0 + static_cast<int64_t>(t) * warps;
      const bool in_range = j < keys;
      active[t] = in_range && mask_row[in_range ? j : 0] != 0;
    }
    // Skip a tile whose keys are all masked out for this row. A masked key's
    // logit is -INFINITY whether or not its dot is computed, so dropping the
    // dot, the warp tree and the K loads moves no surviving term: pass 2 still
    // reads -INFINITY at these positions and adds expf(-INFINITY - row_max) = 0
    // at the same ascending-j step. This is the multi-row counterpart of
    // ``key_begin`` in gemma4_layer, which can only shorten the key range for a
    // one-row block -- and it is what makes a causal prefill block pay for the
    // lower triangle instead of the whole square. A fully causal mask over
    // tokens == keys leaves half the tiles of the average row dead.
    bool any_active = false;
#pragma unroll
    for (int t = 0; t < kKeysPerTile; ++t) {
      any_active = any_active || active[t];
    }
    if (!any_active) {
      if (lane == 0) {
#pragma unroll
        for (int t = 0; t < kKeysPerTile; ++t) {
          const int64_t j = j0 + static_cast<int64_t>(t) * warps;
          if (j < keys) {
            logits_s[j] = -INFINITY;
          }
        }
      }
      continue;
    }
#pragma unroll
    for (int t = 0; t < kKeysPerTile; ++t) {
      const int64_t j = j0 + static_cast<int64_t>(t) * warps;
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
      const int64_t j = j0 + static_cast<int64_t>(t) * warps;
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
  if (lane == 0) {
    max_s[warp] = running_max;
  }
  __syncthreads();
  float row_max = -INFINITY;
  for (unsigned int w = 0; w < warps; ++w) {
    row_max = fmaxf(row_max, max_s[w]);
  }

  // Pass 2: the block kernel's per-lane denominator partition. The reduction is
  // deliberately pinned to the block kernel's 256 lanes whatever the block size
  // is: gemma4_decode_tree1's order depends on its lane count, so a 512-thread
  // block would otherwise round the denominator differently and stop being
  // bit-exact with the reference. Every 256-thread group runs the identical
  // arithmetic in its own 256-float region of `partial`, so group 0's result is
  // exactly the 256-lane one; the extra groups are redundant work in a pass that
  // costs nothing (measured: skipping it entirely saves no time).
  const unsigned int kReduceThreads = kGemma4ReduceThreads;
  const unsigned int reduce_tid = tid & (kReduceThreads - 1);
  float local_sum = 0.0f;
  if (tid < kReduceThreads) {
    for (int64_t j = tid; j < keys; j += kReduceThreads) {
      const float weight = expf(logits_s[j] - row_max);
      local_sum += weight;
      logits_s[j] = weight;
    }
  }
  gemma4_decode_tree1(partial + (tid / kReduceThreads) * kReduceThreads, local_sum,
                      kReduceThreads, reduce_tid);
  // Every thread reads group 0's result: the extra groups exist only so that
  // all threads reach gemma4_decode_tree1's barriers, and their sums are empty.
  const float denominator = partial[0];

  if (split_weights != nullptr) {
    // Export the single-kernel weights and denominator. Phase 2 partitions
    // dimensions, preserving each output's full ascending-key sum.
    for (int64_t j = tid; j < keys; j += threads) {
      split_weights[head_index * keys + j] = logits_s[j];
    }
    if (tid == 0) {
      split_header[head_index * 2] = denominator;
      split_header[head_index * 2 + 1] = row_max;
    }
    return;
  }

  // Pass 3: the block kernel's weighted sums with the precomputed weight, but
  // with the V loads issued kPrefetch keys ahead of the accumulation that
  // consumes them. Each lane walks all `keys` with one 2-byte load per key
  // inside a dependent accumulate chain, and the `weight == 0` skip below stops
  // the compiler from hoisting those loads, so without the pipeline the walk
  // serialises at memory latency (iteration 22 measured 34 GB/s unique against
  // 493 GB/s for the same loop with the loads removed). The prefetch only moves
  // *when* a load is issued: the accumulation stays the ascending-j chain the
  // block kernel uses, so the result is unchanged bit for bit.
  // Pass 3: the block kernel's weighted sums with the precomputed weight, but
  // pipelined (see gemma4_decode_pass3) and, when the row is wide enough to
  // keep every lane busy, loaded two dimensions at a time.
  const int64_t v_row_stride = num_kv_heads * head_dim;
  const scalar_t* v_base = value + kv_head * head_dim;
  scalar_t* out_row = out + head_index * head_dim;
  // Pass 3 walks this row's keys once per dimension, and *both* ends of that
  // walk are dead weight. The tail past the last key the mask keeps
  // contributes weight 0 to every dimension -- the same argument key_begin's
  // docstring makes for shortening the range of a one-row block -- and so does
  // the head before the first key the mask keeps: a masked key's weight is
  // expf(-INFINITY - row_max) == 0, which the walk's own ``weight == 0`` skip
  // already drops. Cutting the head removes V loads without removing a single
  // accumulated term, and the survivors are still visited ascending over the
  // same values, so the output is bit-for-bit what it was. That is the same
  // claim ``last_active`` already makes, extended to the other end, which is
  // why this is legal on the default path rather than merely fast.
  //
  // This is the whole remaining win for sliding rows: a causal row's first kept
  // key is key 0, so only the windowed layers gain, and at 4096 tokens a row at
  // the end of its window walks 1024 keys instead of 4096. Pass 1 already
  // skips fully masked tiles and pass 2 measures as free (its own comment:
  // skipping it entirely saves no time), so this is the only bound left that
  // costs anything to leave in place.
  //
  // Storage: `max_s` holds exactly one slot per warp and no more, so the
  // first-key bound goes in `partial`, whose 512 slots are free once
  // `denominator` is in a register. Both bounds are written under one barrier
  // -- the first `__syncthreads()` below, which is new, and which retires the
  // ``denominator = partial[0]`` reads before this bookkeeping reuses the
  // region pass 2's tree wrote -- and read under the second, the one that has
  // always published `last_active`.
  int local_first = -1;
  int local_last = -1;
  for (int64_t j = static_cast<int64_t>(tid); j < keys; j += threads) {
    if (mask_row[j] != 0) {
      // j ascends within a lane, so the first hit is this lane's minimum.
      if (local_first < 0) {
        local_first = static_cast<int>(j);
      }
      local_last = static_cast<int>(j);
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const int other = __shfl_down(local_last, offset);
    local_last = other > local_last ? other : local_last;
    const int other_first = __shfl_down(local_first, offset);
    if (other_first >= 0 && (local_first < 0 || other_first < local_first)) {
      local_first = other_first;
    }
  }
  __syncthreads();
  if ((tid & 31u) == 0u) {
    max_s[tid >> 5] = static_cast<float>(local_last);
    partial[tid >> 5] = static_cast<float>(local_first);
  }
  __syncthreads();
  int last_active = -1;
  // -1 means this warp saw no live key, so the block minimum ignores it and
  // `first_active` stays -1 only when the whole row is dead.
  int first_active = -1;
  for (unsigned int w = 0; w < (threads >> 5); ++w) {
    const int per_warp = static_cast<int>(max_s[w]);
    last_active = per_warp > last_active ? per_warp : last_active;
    const int per_first = static_cast<int>(partial[w]);
    if (per_first >= 0 && (first_active < 0 || per_first < first_active)) {
      first_active = per_first;
    }
  }
  const int64_t pass3_keys =
      static_cast<int64_t>(last_active) + 1 < keys ? static_cast<int64_t>(last_active) + 1 : keys;
  // A dead row keeps the empty range it had before this bound existed.
  const int64_t pass3_begin =
      first_active >= 0 && static_cast<int64_t>(first_active) < pass3_keys
      ? static_cast<int64_t>(first_active)
      : 0;
  // Pass 3's lower bound has to reach it as a literal on the path that does not
  // use it. Measured: with `keys_begin` a runtime value this loop costs about
  // +50% per iteration -- isolated by a variant carrying a runtime bound and
  // none of the bookkeeping above, so the bookkeeping is not responsible, and
  // identical in the binary to the fixed form (same VGPR and SGPR counts, zero
  // stack frame, same shared bytes and grid, actually *fewer* instructions).
  // Inferred cause, not measured: a compile-time-0 start lets the compiler fold
  // the loop start and the prefetched V addresses into the same addressing it
  // would have used without a lower bound at all, where a runtime start makes
  // them depend on a value that only exists in a register.
  //
  // Every causal row has first_active == 0, so the literal spelling covers
  // exactly the rows that would otherwise pay for a trim they never get, and
  // the runtime branch is left to windowed rows, where the span they cut is far
  // larger than that penalty. first_active is block-wide, so the branch is
  // never divergent.
  if (head_dim >= 2 * static_cast<int64_t>(threads)) {
    if (pass3_begin == 0) {
      gemma4_decode_pass3<2, 4>(logits_s, v_base, out_row, v_row_stride, head_dim, 0,
                                pass3_keys, denominator, threads, tid);
    } else {
      gemma4_decode_pass3<2, 4>(logits_s, v_base, out_row, v_row_stride, head_dim, pass3_begin,
                                pass3_keys, denominator, threads, tid);
    }
  } else {
    if (pass3_begin == 0) {
      gemma4_decode_pass3<1, 4>(logits_s, v_base, out_row, v_row_stride, head_dim, 0,
                                pass3_keys, denominator, threads, tid);
    } else {
      gemma4_decode_pass3<1, 4>(logits_s, v_base, out_row, v_row_stride, head_dim, pass3_begin,
                                pass3_keys, denominator, threads, tid);
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

// Each block owns one wave of output dimensions. Keys remain in strict
// ascending order, while disjoint dimensions execute in independent blocks.
template <typename scalar_t>
__global__ void gemma4_attention_decode_dimension_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ header,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ out,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t keys, int64_t dimension_blocks) {
  const int64_t head_index = blockIdx.x / dimension_blocks;
  const int64_t partition = blockIdx.x % dimension_blocks;
  const int64_t head = head_index % num_heads;
  const int64_t kv_head = head / (num_heads / num_kv_heads);
  extern __shared__ float weights_s[];
  for (int64_t j = threadIdx.x; j < keys; j += blockDim.x) {
    weights_s[j] = weights[head_index * keys + j];
  }
  __syncthreads();
  gemma4_decode_pass3<1, 4>(
      weights_s, value + kv_head * head_dim, out + head_index * head_dim,
      num_kv_heads * head_dim, head_dim, /*keys_begin=*/0, keys,
      header[head_index * 2],
      blockDim.x * dimension_blocks, partition * blockDim.x + threadIdx.x);
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

// Key-class decode launcher: eight warps per (token, head), one warp per key
// class, the block kernel's pass 2 and pass 3.
template <typename scalar_t, int kHalves>
int gemma4_decode_class_launch(
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
  constexpr int kKeysPerTile = 2;
  constexpr unsigned int kThreads = 512;
  const size_t shared_bytes =
      (static_cast<size_t>(keys) + 256 * (kThreads / kGemma4ReduceThreads) +
       kThreads / 32) * sizeof(float);
  if (shared_bytes > 64 * 1024) {
    return -1;
  }
  const unsigned int grid = static_cast<unsigned int>(tokens * num_heads);
  gemma4_attention_decode_class_kernel<scalar_t, kHalves, kKeysPerTile>
      <<<dim3(grid), dim3(kThreads), shared_bytes, stream>>>(
          query, key, value, keep_mask, out, num_heads, num_kv_heads, head_dim, scale, tokens,
          keys);
  return static_cast<int>(hipGetLastError());
}

// Last decode selection: 0 = block, 1 = single key-class, 2 = two-phase
// dimension partition, 3 = flash-decoding. Introspection only; this does not
// change dispatch.
inline int& gemma4_decode_selection() {
  static int selection = 0;
  return selection;
}

// Workspace contains phase-one weights and a (denominator, row max) header.
// head_dim and slices remain in the sizing ABI for existing callers.
size_t gemma4_decode_split_workspace_bytes(
    int64_t tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t keys,
    int slices) {
  const size_t weights = static_cast<size_t>(tokens) * num_heads * keys;
  const size_t header = static_cast<size_t>(tokens) * num_heads * 2;
  return (weights + header) * sizeof(float);
}

template <typename scalar_t, int kHalves>
int gemma4_decode_split_launch(
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
    float* workspace,
    int slices) {
  constexpr int kKeysPerTile = 2;
  constexpr unsigned int kThreads = 512;
  const size_t shared_bytes =
      (static_cast<size_t>(keys) + 256 * (kThreads / kGemma4ReduceThreads) +
       kThreads / 32) * sizeof(float);
  if (shared_bytes > 64 * 1024 || workspace == nullptr || slices < 2) {
    return -1;
  }
  float* weights = workspace;
  float* header = weights + static_cast<size_t>(tokens) * num_heads * keys;

  const unsigned int head_grid = static_cast<unsigned int>(tokens * num_heads);
  gemma4_attention_decode_class_kernel<scalar_t, kHalves, kKeysPerTile>
      <<<dim3(head_grid), dim3(kThreads), shared_bytes, stream>>>(
          query, key, value, keep_mask, out, num_heads, num_kv_heads, head_dim, scale, tokens,
          keys, weights, header);
  int rc = static_cast<int>(hipGetLastError());
  if (rc != 0) {
    return rc;
  }

  const int dimension_blocks = static_cast<int>((head_dim + 31) / 32);
  gemma4_attention_decode_dimension_kernel<scalar_t>
      <<<dim3(head_grid * dimension_blocks), dim3(32), keys * sizeof(float), stream>>>(
          weights, header, value, out, num_heads, num_kv_heads, head_dim,
          keys, dimension_blocks);
  return static_cast<int>(hipGetLastError());
}

// ---------------------------------------------------------------------------
// Flash-decoding (D2): KV-sliced partial attention with a combine pass.
//
// The two-phase split partitions output dimensions; this partitions KEYS.
// Each block owns one KV head's q-pair (the sliding layers' GQA ratio of 2
// packs exactly the two query heads that read the same K and V rows) and one
// contiguous key slice: phase A computes the slice's logits with the class
// kernel's warp tree and the slice's per-row maximum, phase B exponentiates
// against that maximum and accumulates the slice's weighted V sum over f32
// partials, and the combine kernel merges slices in ascending order with the
// standard max-rescale. K loads are shared by both q rows of the pair, and V
// once per key feeds both rows' accumulators.
//
// This reassociates: a slice softmax against its own maximum differs from the
// row softmax by an exact rescale factor -- exact in reals, f32-rounding-close
// in practice. Arithmetic the production route allows
// (docs/EXECUTION-PROFILES.md), and the strict chain stays reachable: a
// rejected request falls through to the split/class/block path untouched.
// Contract: agreement to float precision vs the single kernel in the unit
// battery, production KL/top-1 bars at the campaign gate.
//
// Partial layout, f32, per (pair, slice) where pair = token*num_kv_heads+kv:
//   base = (pair * slices + slice) * kFlashQPerKV * (2 + head_dim)
//   [q*(2 + head_dim) + 0]      = slice row max
//   [q*(2 + head_dim) + 1]      = slice denominator
//   [q*(2 + head_dim) + 2 + d]  = weighted V sum for dimension d
// hipengine_gemma4_decode_flash_workspace_bytes must match this exactly:
// tokens * num_heads * slices * (2 + head_dim) floats, since
// num_kv_heads * kFlashQPerKV == num_heads under the admission.

constexpr int kFlashThreads = 128;   // 4 warps
constexpr int kFlashWarps = 4;
constexpr int kFlashQPerKV = 2;      // the packing the admission admits

template <typename scalar_t>
__global__ void gemma4_attention_decode_flash_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const uint8_t* __restrict__ keep_mask,
    float* __restrict__ partials,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale,
    int64_t tokens,
    int64_t keys,
    int slices) {
  constexpr int kWarpLanes = 32;
  constexpr int kOwn = 8;  // tree lanes per warp lane
  const int64_t slice = static_cast<int64_t>(blockIdx.x) % slices;
  const int64_t kv = (static_cast<int64_t>(blockIdx.x) / slices) % num_kv_heads;
  const int64_t token = static_cast<int64_t>(blockIdx.x) / (slices * num_kv_heads);
  const int64_t j_begin = keys * slice / slices;
  const int64_t j_end = keys * (slice + 1) / slices;
  const int64_t slice_keys = j_end - j_begin;

  extern __shared__ float smem[];
  float* warp_max = smem;                              // [kFlashQPerKV][kFlashWarps]
  float* q_s = warp_max + kFlashQPerKV * kFlashWarps;  // [kFlashQPerKV][head_dim]
  // [kFlashQPerKV][slice_keys]: phase A writes logits; the weight pass below
  // overwrites them in place with exp(logit - row_max) * mask.
  float* logits_s = q_s + kFlashQPerKV * head_dim;

  const unsigned int tid = threadIdx.x;
  const unsigned int warp = tid / kWarpLanes;
  const unsigned int lane = tid & (kWarpLanes - 1);

  for (int64_t i = tid; i < kFlashQPerKV * head_dim; i += kFlashThreads) {
    const int64_t row = token * num_heads + kv * kFlashQPerKV + i / head_dim;
    q_s[i] = gemma4_attn_to_float(query[row * head_dim + (i % head_dim)]);
  }
  __syncthreads();

  const uint8_t* mask_row = keep_mask + token * keys;

  // Phase A: slice logits through the class kernel's warp tree. Each warp
  // owns every kFlashWarps-th key of the slice; K is loaded once per key and
  // dotted against both rows of the pair. The tree folds the eight per-lane
  // partials into element 0 on lane 0 only (the class kernel's own contract),
  // so lane 0 alone publishes the logit -- a shared store or running max from
  // any other lane would race a partial reduction into the same address.
  float running_max[kFlashQPerKV];
#pragma unroll
  for (int q = 0; q < kFlashQPerKV; ++q) {
    running_max[q] = -INFINITY;
  }
  for (int64_t j = j_begin + warp; j < j_end; j += kFlashWarps) {
    const bool keep = mask_row[j] != 0;
    const scalar_t* k_row = key + (j * num_kv_heads + kv) * head_dim;
    float k_reg[kOwn];
#pragma unroll
    for (int i = 0; i < kOwn; ++i) {
      k_reg[i] = gemma4_attn_to_float(k_row[lane * kOwn + i]);
    }
    for (int q = 0; q < kFlashQPerKV; ++q) {
      float v[kOwn];
#pragma unroll
      for (int i = 0; i < kOwn; ++i) {
        v[i] = q_s[q * head_dim + lane * kOwn + i] * k_reg[i];
      }
      const float reduced = gemma4_warp_tree(v, lane);
      if (lane == 0) {
        const float logit = keep ? reduced * scale : -INFINITY;
        logits_s[q * slice_keys + (j - j_begin)] = logit;
        running_max[q] = fmaxf(running_max[q], logit);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int q = 0; q < kFlashQPerKV; ++q) {
      warp_max[q * kFlashWarps + warp] = running_max[q];
    }
  }
  __syncthreads();

  float row_max[kFlashQPerKV];
#pragma unroll
  for (int q = 0; q < kFlashQPerKV; ++q) {
    float m = -INFINITY;
    for (unsigned int w = 0; w < kFlashWarps; ++w) {
      m = fmaxf(m, warp_max[q * kFlashWarps + w]);
    }
    row_max[q] = m;
  }

  // Weight pass: overwrite the logits in place with exp(logit - row_max),
  // zero for masked keys. Each (q, key) exponent is computed once here rather
  // than once per dimension in phase B; the stride covers any slice width.
  const int64_t weight_count = kFlashQPerKV * slice_keys;
  for (int64_t idx = tid; idx < weight_count; idx += kFlashThreads) {
    const int64_t q = idx / slice_keys;
    const int64_t jj = idx - q * slice_keys;
    const bool keep = mask_row[j_begin + jj] != 0;
    logits_s[idx] = keep ? expf(logits_s[idx] - row_max[q]) : 0.0f;
  }
  __syncthreads();

  // Phase B: every thread owns dimensions (2*tid, 2*tid+1) of both pair rows
  // and walks the whole slice itself (gemma4_decode_pass3's partition), so
  // each accumulator is complete as written -- no cross-warp publish, no
  // shared staging. The pipe keeps one key's load in flight while the
  // previous one accumulates.
  const int64_t d0 = static_cast<int64_t>(tid) * 2;
  const int64_t v_stride = num_kv_heads * head_dim;
  const scalar_t* v_base = value + kv * head_dim + j_begin * v_stride;
  constexpr gemma4_pair2<scalar_t> kZeroPair = {
      static_cast<scalar_t>(0), static_cast<scalar_t>(0)};
  gemma4_pair2<scalar_t> pipe[2];
  pipe[0] = 0 < slice_keys
      ? gemma4_load_v<2>(v_base + 0 * v_stride + d0)
      : kZeroPair;
  pipe[1] = 1 < slice_keys
      ? gemma4_load_v<2>(v_base + 1 * v_stride + d0)
      : kZeroPair;
  float acc[kFlashQPerKV][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  float denom[kFlashQPerKV] = {0.0f, 0.0f};
  for (int64_t jj = 0; jj < slice_keys; ++jj) {
    const gemma4_pair2<scalar_t> v_cur = pipe[0];
    pipe[0] = pipe[1];
    const int64_t jn = jj + 2;
    pipe[1] = jn < slice_keys
        ? gemma4_load_v<2>(v_base + jn * v_stride + d0)
        : kZeroPair;
    const float w0 = logits_s[0 * slice_keys + jj];
    const float w1 = logits_s[1 * slice_keys + jj];
    denom[0] += w0;
    denom[1] += w1;
    const float vlo = gemma4_attn_to_float(v_cur.lo);
    const float vhi = gemma4_attn_to_float(v_cur.hi);
    acc[0][0] += w0 * vlo;
    acc[0][1] += w0 * vhi;
    acc[1][0] += w1 * vlo;
    acc[1][1] += w1 * vhi;
  }

  const int64_t pair = token * num_kv_heads + kv;
  const int64_t slot = (pair * slices + slice) * kFlashQPerKV * (2 + head_dim);
  if (tid == 0) {
#pragma unroll
    for (int q = 0; q < kFlashQPerKV; ++q) {
      float* p = partials + slot + q * (2 + head_dim);
      p[0] = row_max[q];
      p[1] = denom[q];
    }
  }
#pragma unroll
  for (int q = 0; q < kFlashQPerKV; ++q) {
    float* p = partials + slot + q * (2 + head_dim);
    p[2 + d0] = acc[q][0];
    p[3 + d0] = acc[q][1];
  }
}

// Combine pass: merge slices in ascending order with the flash rescale. Dead
// slices (l == 0) carry no mass and are skipped before any exponent, so a
// -inf slice max can never reach expf(-inf - -inf). An all-masked row leaves
// L == 0 and O == 0, so O / L is NaN -- the incumbent's own NaN propagation
// for that row, matching at the bf16 round trip.
template <typename scalar_t>
__global__ void gemma4_attention_decode_flash_combine_kernel(
    const float* __restrict__ partials,
    scalar_t* __restrict__ out,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t tokens,
    int slices) {
  const int64_t head = static_cast<int64_t>(blockIdx.x);
  const int64_t token = head / num_heads;
  const int64_t h = head - token * num_heads;
  const int64_t ratio = num_heads / num_kv_heads;
  const int64_t kv = h / ratio;
  const int64_t q = h - kv * ratio;
  const int64_t pair = token * num_kv_heads + kv;
  const int64_t slot_stride = ratio * (2 + head_dim);
  const int64_t d0 = static_cast<int64_t>(threadIdx.x) * 2;
  float M = -INFINITY;
  float L = 0.0f;
  float O0 = 0.0f;
  float O1 = 0.0f;
  for (int s = 0; s < slices; ++s) {
    const float* p = partials + (pair * slices + s) * slot_stride + q * (2 + head_dim);
    const float l_s = p[1];
    if (!(l_s > 0.0f)) {
      continue;
    }
    const float m_s = p[0];
    const float o0 = p[2 + d0];
    const float o1 = p[3 + d0];
    if (!(L > 0.0f)) {
      M = m_s;
      L = l_s;
      O0 = o0;
      O1 = o1;
      continue;
    }
    const float M_new = fmaxf(M, m_s);
    const float alpha = expf(M - M_new);
    const float beta = expf(m_s - M_new);
    L = alpha * L + beta * l_s;
    O0 = alpha * O0 + beta * o0;
    O1 = alpha * O1 + beta * o1;
    M = M_new;
  }
  scalar_t* out_row = out + head * head_dim;
  out_row[d0] = gemma4_attn_to_scalar<scalar_t>(O0 / L);
  if (d0 + 1 < head_dim) {
    out_row[d0 + 1] = gemma4_attn_to_scalar<scalar_t>(O1 / L);
  }
}

// Flash launcher: KV slices over one q-pair, then the combine. Returns -1
// when the shape is outside what this template implements, so the caller
// falls through to the incumbent chain with the same workspace (the Python
// side sizes it as max(split, flash)).
template <typename scalar_t>
int gemma4_decode_flash_launch(
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
    float* workspace,
    int slices) {
  if (head_dim != 256 || tokens < 1 || slices < 2 || slices > keys ||
      num_kv_heads <= 0 || num_heads != num_kv_heads * kFlashQPerKV ||
      workspace == nullptr) {
    return -1;
  }
  const int64_t slice_max = (keys + slices - 1) / slices;
  const size_t shared_bytes =
      static_cast<size_t>(
          kFlashQPerKV * kFlashWarps + kFlashQPerKV * head_dim +
          kFlashQPerKV * slice_max) *
      sizeof(float);
  if (shared_bytes > 64 * 1024) {
    return -1;
  }
  const unsigned int grid =
      static_cast<unsigned int>(tokens * num_kv_heads * slices);
  gemma4_attention_decode_flash_kernel<scalar_t>
      <<<dim3(grid), dim3(kFlashThreads), shared_bytes, stream>>>(
          query, key, value, keep_mask, workspace, num_heads, num_kv_heads,
          head_dim, scale, tokens, keys, slices);
  int rc = static_cast<int>(hipGetLastError());
  if (rc != 0) {
    return rc;
  }
  gemma4_attention_decode_flash_combine_kernel<scalar_t>
      <<<dim3(tokens * num_heads), dim3(kFlashThreads), 0, stream>>>(
          workspace, out, num_heads, num_kv_heads, head_dim, tokens, slices);
  return static_cast<int>(hipGetLastError());
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
    int64_t keys,
    float* split_workspace,
    int split_slices) {
  if (tokens <= 0 || keys <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (num_heads % num_kv_heads != 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  int effective_slices = split_slices;
  if (split_slices < 0 && split_workspace != nullptr) {
    // Flash request: Python's admission already checked the capability
    // (head_dim 256, GQA ratio 2, decode token). Any rejection here falls
    // through to the incumbent chain on the same workspace, which the
    // caller sized as max(split, flash).
    const int flash_rc = gemma4_decode_flash_launch<scalar_t>(
        query, key, value, keep_mask, out, num_heads, num_kv_heads, head_dim,
        scale, tokens, keys, stream, split_workspace, -split_slices);
    if (flash_rc >= 0) {
      gemma4_decode_selection() = 3;  // flash-decoding
      return flash_rc;
    }
    effective_slices = -split_slices;
  }
  int threads_i = 1;
  while (threads_i < head_dim && threads_i < kGemma4AttentionMaxThreads) {
    threads_i <<= 1;
  }
  const unsigned int threads = static_cast<unsigned int>(threads_i);
  // Key-class path: head_dim is exactly one or two of the block kernel's 256
  // tree lanes, so each key's logit can be reduced by the intra-warp shuffle
  // tree instead of a shared partial row plus a barrier sequence. Measured
  // 421 us (head_dim 256) and 665 us (head_dim 512) against the block kernel's
  // 652-700 and 1186 us at keys=1024. Any other geometry uses the block kernel.
  if (head_dim == 256 || head_dim == 512) {
    if (effective_slices > 1 && split_workspace != nullptr) {
      const int split_rc = head_dim == 256
          ? gemma4_decode_split_launch<scalar_t, 1>(query, key, value, keep_mask, out, num_heads,
                num_kv_heads, head_dim, scale, tokens, keys, stream, split_workspace,
                effective_slices)
          : gemma4_decode_split_launch<scalar_t, 2>(query, key, value, keep_mask, out, num_heads,
                num_kv_heads, head_dim, scale, tokens, keys, stream, split_workspace,
                effective_slices);
      if (split_rc >= 0) {
        gemma4_decode_selection() = 2;  // key-class, two-phase split
        return split_rc;
      }
    }
    const int rc = head_dim == 256
        ? gemma4_decode_class_launch<scalar_t, 1>(query, key, value, keep_mask, out, num_heads,
              num_kv_heads, head_dim, scale, tokens, keys, stream)
        : gemma4_decode_class_launch<scalar_t, 2>(query, key, value, keep_mask, out, num_heads,
              num_kv_heads, head_dim, scale, tokens, keys, stream);
    if (rc >= 0) {
      gemma4_decode_selection() = 1;  // key-class
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

// Returns 0 for block, 1 for single key-class, 2 for dimension partition,
// or 3 for flash-decoding. Test/introspection surface for the routing
// contract.
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
//
// ``split_workspace`` is the two-phase split's scratch (see
// gemma4_decode_split_workspace_bytes) and ``split_slices`` its slice count;
// passing ``split_slices <= 1`` or a null workspace selects the single-kernel
// path. The caller owns the allocation because it is reused across steps and
// must outlive the in-flight stream.
// Which decode kernel the launcher last selected, exported so a caller can
// confirm the intended path actually ran rather than inferring it from
// timings: 0 = block kernel, 1 = key-class single kernel, 2 = key-class
// two-phase split, 3 = flash-decoding.
extern "C" int hipengine_gemma4_decode_selection(void) {
  return gemma4_decode_selection();
}

// Workspace sizing for the two-phase split, exported so the Python launcher and
// the kernel share one definition of the layout: a mismatch between the two
// would be memory corruption, not a slow path.
extern "C" size_t hipengine_gemma4_decode_split_workspace_bytes(
    int64_t tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t keys,
    int slices) {
  return gemma4_decode_split_workspace_bytes(tokens, num_heads, head_dim, keys, slices);
}

// Workspace sizing for flash-decoding, same single-definition rule. The
// partial layout comment on the flash kernel is the normative statement;
// this expression must match it: one (m, l, o[head_dim]) slot per head per
// slice, f32.
extern "C" size_t hipengine_gemma4_decode_flash_workspace_bytes(
    int64_t tokens,
    int64_t num_heads,
    int64_t head_dim,
    int slices) {
  return static_cast<size_t>(tokens) * static_cast<size_t>(num_heads) *
         static_cast<size_t>(slices) * (2 + static_cast<size_t>(head_dim)) *
         sizeof(float);
}

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
    int64_t keys,
    float* split_workspace,
    int split_slices) {
  return launch_gemma4_attention_decode<uint16_t>(
      query, key, value, keep_mask, out, tokens, num_heads, num_kv_heads, head_dim, scale,
      stream, keys, split_workspace, split_slices);
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
    int64_t keys,
    float* split_workspace,
    int split_slices) {
  return launch_gemma4_attention_decode<float>(
      query, key, value, keep_mask, out, tokens, num_heads, num_kv_heads, head_dim, scale,
      stream, keys, split_workspace, split_slices);
}
